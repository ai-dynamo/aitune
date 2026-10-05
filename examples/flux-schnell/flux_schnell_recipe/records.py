# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run provenance and immutable output directories for this example."""

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import uuid


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_identity(args, cfg):
    if args.model_id:
        return {"kind": "hub", "id": args.model_id, "revision": cfg["model"]["revision"]}
    root = Path(args.checkpoint).resolve()
    if not (root / "model_index.json").is_file():
        raise ValueError("--checkpoint must be a complete Diffusers directory with model_index.json")
    files = {str(p.relative_to(root)): file_sha(p) for p in sorted(root.rglob("*"))
             if p.is_file() and ".cache" not in p.relative_to(root).parts}
    return {"kind": "directory", "path": str(root), "files_sha256": digest(files)}


def contract(cfg, source):
    return {"source": source, **{k: cfg[k] for k in ("model", "workload", "precision", "execution")}}


def new_run(cfg, phase):
    run = cfg.output / phase / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    run.mkdir(parents=True)
    write_json(run / "resolved-config.json", cfg.data)
    return run


def configure_cache(run):
    # Set before importing AITune/Torch. All compiler caches persist with the run.
    for key, subdir in (("AITUNE_CACHE_DIR", "aitune-cache"), ("TORCHINDUCTOR_CACHE_DIR", "inductor-cache"),
                        ("TRITON_CACHE_DIR", "triton-cache"), ("CUDA_CACHE_PATH", "cuda-cache")):
        os.environ[key] = str(run / subdir)
    os.environ["AITUNE_TUNING_DATA_PATH"] = str(run / "aitune-tuning.json")


def environment():
    import torch
    import pynvml as nv
    packages = {}
    for name in ("aitune", "torch", "diffusers", "transformers", "torch-tensorrt", "tensorrt",
                 "torchao", "ai-dynamo", "ai-dynamo-runtime", "aiperf", "nvidia-ml-py", "scikit-image"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpu = torch.cuda.get_device_properties(0)
    nv.nvmlInit()
    try:
        driver = nv.nvmlSystemGetDriverVersion()
    finally:
        nv.nvmlShutdown()
    return {"python": sys.version, "platform": platform.platform(), "packages": packages,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "driver": driver,
            "gpu": {"name": gpu.name, "uuid": str(gpu.uuid), "compute_capability": [gpu.major, gpu.minor],
                    "memory_gib": gpu.total_memory / 2**30},
            "container_digest": os.environ.get("RECIPE_IMAGE_DIGEST"),
            "container_digest_missing_reason": None if os.environ.get("RECIPE_IMAGE_DIGEST") else "Set RECIPE_IMAGE_DIGEST to the evaluated image RepoDigest"}


def artifact_metadata(cfg):
    metadata = read_json(str(cfg.artifact) + ".json")
    if file_sha(cfg.artifact) != metadata["sha256"]:
        raise ValueError("Artifact hash differs from its tuning record")
    return metadata
