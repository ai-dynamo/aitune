# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small, real, two-stage LTX 2.3 outpaint with FlexTensor and optional AITune.

Run prompt encoding and each benchmark mode in separate processes. No JIT reset,
checkpoint conversion, parameter replacement, or quantization is performed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import statistics
import threading
import time
import urllib.request
from contextlib import contextmanager
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from typing import Any

BASE_MODEL = "dg845/LTX-2.3-Diffusers"
BASE_REVISION = "01fc4e67e5c88bf8c44f5217b9669b7e496fb985"
UPSAMPLER_MODEL = "dg845/LTX-2.3-Spatial-Upsampler-Diffusers"
UPSAMPLER_REVISION = "302e91c54f2b8ecc3ce85d7e4dd308c8e91b112e"
LORA_REPO = "oumoumad/LTX-2.3-22b-IC-LoRA-Outpaint"
LORA_REVISION = "b747e10700e7460f90d0cb1ceb061f69c719431a"
LORA_FILE = "ltx-2.3-22b-ic-lora-outpaint.safetensors"
DIFFUSERS_REVISION = "9159a587b1f148a2cc911deb72b6d755508ef38b"
SOURCE_URL = "https://upload.wikimedia.org/wikipedia/commons/6/6d/HelenHillInterview.theora.ogv"
PROMPT = "extend the scene naturally, consistent with the original footage"
DEFAULT_WIDTH, DEFAULT_HEIGHT = 256, 160
FRAMES, FPS, BLOCKS, SEED = 17, 24, 48, 171198
MANAGER_NAME = "ltx23-outpaint-mini"
DEFAULT_WORK_DIR = Path("tuning/ltx23_outpaint")
GUIDANCE = {
    "guidance_scale": 1.0,
    "stg_scale": 0.0,
    "modality_scale": 1.0,
    "guidance_rescale": 0.0,
    "audio_guidance_scale": 1.0,
    "audio_stg_scale": 0.0,
    "audio_modality_scale": 1.0,
    "audio_guidance_rescale": 0.0,
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_video(frames: Any, path: Path) -> None:
    """Use PyAV's bundled libx264; no system ffmpeg executable is needed."""
    import av
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=Fraction(FPS, 1))
        first = np.asarray(frames[0])
        stream.width, stream.height = first.shape[1], first.shape[0]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "fast"}
        for pixels in frames:
            frame = av.VideoFrame.from_ndarray(np.asarray(pixels).astype(np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def decode_video(path: Path) -> list[Any]:
    import av

    with av.open(str(path)) as container:
        return [frame.to_image().convert("RGB") for frame in container.decode(video=0)]


def source_size(args: argparse.Namespace) -> tuple[int, int]:
    """Keep source placement ratios fixed while validating the caller's stage-1 canvas."""
    if min(args.width, args.height) <= 0:
        raise ValueError("stage-1 width and height must be positive")
    if args.width % 32 or args.height % 32:
        raise ValueError("stage-1 width and height must be divisible by 32")
    width, height = args.width * 5 // 8, args.height * 3 // 5
    if min(width, height) < 2 or width % 2 or height % 2:
        raise ValueError("derived source width and height must be positive even dimensions for H.264")
    return width, height


def prepare_source(args: argparse.Namespace) -> None:
    """Prepare a short real public-domain source, not a synthetic model input."""
    source_dims = source_size(args)
    work_dir = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    source = args.input_video or work_dir / "HelenHillInterview.theora.ogv"
    if not source.exists():
        if args.input_video:
            raise FileNotFoundError(source)
        request = urllib.request.Request(
            SOURCE_URL, headers={"User-Agent": "AITune-example/1.0 (https://github.com/ai-dynamo/aitune)"}
        )
        with urllib.request.urlopen(request, timeout=60) as response, source.open("wb") as destination:
            shutil.copyfileobj(response, destination)
    frames = sample_source(source, args)
    if len(frames) != FRAMES:
        raise ValueError(f"source yielded only {len(frames)} frames; need {FRAMES}")
    save_video(frames, work_dir / "source.mp4")
    reference = prepare_reference(work_dir / "source.mp4", args)
    reference[0].save(work_dir / "reference_first_frame.png")
    write_json(
        work_dir / "source.json",
        {
            "url": SOURCE_URL if not args.input_video else None,
            "local_source": str(source.resolve()),
            "sha256": sha256(source),
            "start_seconds": args.start_seconds,
            "frames": FRAMES,
            "fps": FPS,
            "source_size": list(source_dims),
            "stage1_size": [args.width, args.height],
            "stage2_size": [2 * args.width, 2 * args.height],
            "padding": "exact RGB zero created in memory AFTER decoding source.mp4",
            "license_page": "https://commons.wikimedia.org/wiki/File:HelenHillInterview.theora.ogv",
        },
    )
    print(work_dir / "source.mp4")


def source_frames(source: Path):
    import av

    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "NONE"
        stream.codec_context.thread_count = 1
        for packet in container.demux(stream):
            # Zero-byte Theora packets repeat the previous picture. Passing them
            # to avcodec_send_packet is rejected in recent FFmpeg/PyAV wheels.
            if packet.size:
                yield from packet.decode()


def sample_source(source: Path, args: argparse.Namespace) -> list[Any]:
    """Resample display timestamps, retaining real held pictures between updates."""
    from PIL import ImageOps

    frames = []
    previous = None
    start_seconds = args.start_seconds
    source_dims = source_size(args)
    next_time = start_seconds
    for frame in source_frames(source):
        image = ImageOps.fit(frame.to_image().convert("RGB"), source_dims)
        while previous is not None and next_time < frame.time - 1e-6:
            frames.append(previous)
            next_time = start_seconds + len(frames) / FPS
            if len(frames) == FRAMES:
                return frames
        if frame.time + 1e-6 >= next_time:
            frames.append(image)
            next_time = start_seconds + len(frames) / FPS
        if len(frames) == FRAMES:
            break
        previous = image
    return frames


def prepare_reference(path: Path, args: argparse.Namespace) -> list[Any]:
    """Do not encode/decode padded canvases: the LoRA uses exact zeros as a sentinel."""
    from PIL import Image

    source_width, source_height = source_size(args)
    source = decode_video(path)
    if len(source) != FRAMES or any(image.size != (source_width, source_height) for image in source):
        raise ValueError(f"expected {FRAMES} source frames, each {source_width}x{source_height}")
    reference = []
    for image in source:
        canvas = Image.new("RGB", (args.width, args.height), (0, 0, 0))
        canvas.paste(image, ((args.width - source_width) // 2, (args.height - source_height) // 2))
        reference.append(canvas)
    return reference


def encode_prompt(args: argparse.Namespace) -> None:
    """Load only the pinned real Gemma encoder and exit before loading the denoiser."""
    import torch
    from diffusers import LTX2InContextPipeline
    from transformers import Gemma3ForConditionalGeneration, GemmaTokenizerFast

    started = time.perf_counter()
    tokenizer = GemmaTokenizerFast.from_pretrained(
        BASE_MODEL,
        subfolder="tokenizer",
        revision=BASE_REVISION,
        cache_dir=args.model_cache,
    )
    # The upstream text checkpoint is FP32. Convert while loading directly onto GPU,
    # rather than retaining its ~49 GB FP32 host copy on a small-RAM workstation.
    encoder = Gemma3ForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        subfolder="text_encoder",
        revision=BASE_REVISION,
        cache_dir=args.model_cache,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": 0},
    ).eval()
    pipe = LTX2InContextPipeline(
        scheduler=None,
        vae=None,
        audio_vae=None,
        text_encoder=encoder,
        tokenizer=tokenizer,
        connectors=None,
        transformer=None,
        vocoder=None,
    )
    with torch.no_grad():
        embeds, mask, _, _ = pipe.encode_prompt(
            prompt=PROMPT,
            negative_prompt=None,
            do_classifier_free_guidance=False,
            num_videos_per_prompt=1,
            max_sequence_length=1024,
            device=torch.device("cuda:0"),
            dtype=torch.bfloat16,
        )
    destination = args.work_dir / "prompt.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    provenance = {
        "prompt": PROMPT,
        "model": BASE_MODEL,
        "revision": BASE_REVISION,
        "text_encoder": "Gemma3ForConditionalGeneration",
        "dtype": "bfloat16",
        "max_sequence_length": 1024,
        "embedding_shape": list(embeds.shape),
        "seconds": time.perf_counter() - started,
    }
    torch.save(
        {
            "prompt_embeds": embeds.detach().cpu(),
            "prompt_attention_mask": mask.detach().cpu(),
            "provenance": provenance,
        },
        destination,
    )
    write_json(args.work_dir / "prompt.json", {**provenance, "sha256": sha256(destination)})
    print(destination)


def configure_aitune(args: argparse.Namespace) -> Any:
    import torch

    from aitune.torch.backend import TorchInductorJitBackend, TorchInductorJitBackendConfig
    from aitune.torch.config import config as global_config
    from aitune.torch.jit.config import JITMode, config
    from aitune.torch.jit.tune import deferred
    from aitune.torch.task.profiling import (
        AllSamplesProfilingStopStrategy,
        NumStepsMeasuringStopStrategy,
        ProfilingConfig,
    )
    from aitune.torch.tune_strategy import OneBackendStrategy

    config.mode = JITMode.TUNE_DEFERRED
    config.device = torch.device("cuda:0")
    config.cache_dir = args.work_dir / "combined" / "aitune-cache" / "jit"
    config.min_samples = 1
    config.batch_axis_required = False
    profiling = ProfilingConfig(
        batch_sizes=[1],
        measurement_stop_strategy=NumStepsMeasuringStopStrategy(num_steps=args.profile_steps, warmup_samples=1),
        profiling_stop_strategy=AllSamplesProfilingStopStrategy(),
    )
    backend = TorchInductorJitBackend(TorchInductorJitBackendConfig(fullgraph=True, dynamic=None, mode="default"))
    # Diagnostic is NOT disabled: normal AITune eager/candidate microbenchmarks run.
    # It does not reject Inductor solely on isolated per-block microbenchmark speed.
    config.strategy = OneBackendStrategy(backend, profiling_config=profiling).enable_correctness_check(True)
    config.strategy.enable_performance_validation(args.performance_validation)
    global_config.max_num_samples_stored = 1
    return deferred


def load_pipelines(args: argparse.Namespace) -> tuple[Any, Any, list[Any]]:
    import torch
    from accelerate import cpu_offload_with_hook
    from diffusers import LTX2InContextPipeline, LTX2LatentUpsamplePipeline
    from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel

    # BF16 safetensors remain file-backed on CPU until FlexTensor takes ownership.
    # Never move the whole transformer to CUDA first: then FT would discover nothing.
    pipe = LTX2InContextPipeline.from_pretrained(
        BASE_MODEL,
        revision=BASE_REVISION,
        cache_dir=args.model_cache,
        text_encoder=None,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        disable_mmap=False,
    )
    pipe.load_lora_weights(
        LORA_REPO,
        weight_name=LORA_FILE,
        revision=LORA_REVISION,
        cache_dir=args.model_cache,
        adapter_name="outpaint",
    )
    pipe.set_adapters("outpaint", 1.0)
    pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)
    upsampler = LTX2LatentUpsamplerModel.from_pretrained(
        UPSAMPLER_MODEL,
        subfolder="latent_upsampler",
        revision=UPSAMPLER_REVISION,
        cache_dir=args.model_cache,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    hooks = []
    for name in ("connectors", "vae", "audio_vae", "vocoder"):
        module, hook = cpu_offload_with_hook(getattr(pipe, name), torch.device("cuda:0"))
        setattr(pipe, name, module)
        hooks.append(hook)
    upsampler, hook = cpu_offload_with_hook(upsampler, torch.device("cuda:0"))
    hooks.append(hook)
    upsample_pipe = LTX2LatentUpsamplePipeline(vae=pipe.vae, latent_upsampler=upsampler)
    upsample_pipe.set_progress_bar_config(disable=True)
    return pipe, upsample_pipe, hooks


def offload_components(hooks: list[Any]) -> None:
    for hook in reversed(hooks):
        hook.offload()


@contextmanager
def conditioning_lifecycle(transformer: Any, hooks: list[Any]):
    """Release the large connector/VAE components before the first denoiser call."""

    def first_forward(_module: Any, _inputs: Any) -> None:
        hooks[0].offload()
        hooks[1].offload()

    handle = transformer.register_forward_pre_hook(first_forward)
    try:
        yield
    finally:
        handle.remove()
        hooks[0].offload()
        hooks[1].offload()


def generate(
    pipe: Any,
    upsample_pipe: Any,
    hooks: list[Any],
    reference: list[Any],
    conditioning: dict[str, Any],
    args: argparse.Namespace,
    *,
    seed: int = SEED,
    collect_latent: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Run the pinned native two-stage workload, keeping the outpaint LoRA active."""
    import torch
    from diffusers.pipelines.ltx2.pipeline_ltx2_ic_lora import LTX2ReferenceCondition
    from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES

    generator = torch.Generator(device="cuda").manual_seed(seed)
    common = {
        "prompt": None,
        "prompt_embeds": conditioning["prompt_embeds"],
        "prompt_attention_mask": conditioning["prompt_attention_mask"],
        "num_frames": FRAMES,
        "frame_rate": FPS,
        **GUIDANCE,
        "generator": generator,
        "return_dict": False,
    }
    captured = {}

    def capture_final(_pipe: Any, step: int, _timestep: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
        if collect_latent:
            captured["video_latent"] = callback_kwargs["latents"].detach().float().cpu()
            captured["last_callback_step"] = step
        return callback_kwargs

    try:
        with torch.no_grad():
            with conditioning_lifecycle(pipe.transformer, hooks):
                video_latent, audio_latent = pipe(
                    **common,
                    reference_conditions=[LTX2ReferenceCondition(frames=reference, strength=1.0)],
                    conditioning_attention_mask=None,
                    conditioning_attention_strength=1.0,
                    reference_downscale_factor=1,
                    width=args.width,
                    height=args.height,
                    num_inference_steps=len(DISTILLED_SIGMA_VALUES),
                    sigmas=DISTILLED_SIGMA_VALUES,
                    output_type="latent",
                )
            try:
                video_latent = upsample_pipe(latents=video_latent, output_type="latent", return_dict=False)[0]
            finally:
                hooks[-1].offload()
            # The outpaint LoRA remains active in stage 2. There is deliberately no
            # second reference condition: the spatially upsampled latent is refined.
            with conditioning_lifecycle(pipe.transformer, hooks):
                video, audio = pipe(
                    **common,
                    latents=video_latent,
                    audio_latents=audio_latent,
                    width=args.width * 2,
                    height=args.height * 2,
                    num_inference_steps=len(STAGE_2_DISTILLED_SIGMA_VALUES),
                    sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
                    noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
                    output_type="pil",
                    callback_on_step_end=capture_final,
                    callback_on_step_end_tensor_inputs=["latents"],
                )
            if collect_latent:
                captured["audio_waveform"] = audio.detach().float().cpu()
            return video[0], captured
    finally:
        offload_components(hooks)


def configure_flextensor(pipe: Any, args: argparse.Namespace) -> Any:
    import flextensor

    from aitune.torch.jit.flextensor import FlexTensorUnit

    transformer = pipe.transformer
    if len(transformer.transformer_blocks) != BLOCKS:
        raise RuntimeError(f"the pinned pretrained model must contain all {BLOCKS} blocks")
    for index, block in enumerate(transformer.transformer_blocks):
        transformer.transformer_blocks[index] = FlexTensorUnit(block)
    for parameter in transformer.parameters(recurse=False):
        parameter.data = parameter.data.to("cuda:0")
    for name, buffer in transformer.named_buffers(recurse=False):
        setattr(transformer, name, buffer.to("cuda:0"))
    profile_dir = args.work_dir / "flextensor-profile"
    config = flextensor.OffloadConfig(
        discovery_iters=1,
        profiling_iters=args.ft_profile_steps,
        min_blocks=2,
        num_blocks=2,
        max_gpu_mem_fraction=args.gpu_memory_fraction,
        pinned_memory_mode=args.pinned_memory_mode,
        include_patterns=["transformer_blocks.*"],
        profile_mode="view",
        external_compile=args.finalization == "replan",
        profile_storage_dir=str(profile_dir),
        enable_diagnostics=True,
    )
    # This identity compile_fn activates FT's compiled, retained-view-profile path;
    # it does NOT call torch.compile. AITune alone compiles the inner compute later.
    compile_fn = (lambda module: module) if args.finalization == "retain-profile" else None
    if args.reuse_profile:
        pipe.transformer = flextensor.offload_from_profile(
            transformer,
            str(profile_dir),
            config=config,
            name=MANAGER_NAME,
            compile_fn=compile_fn,
        )
    else:
        pipe.transformer = flextensor.offload(transformer, config=config, name=MANAGER_NAME, compile_fn=compile_fn)
    return flextensor.get_offload_manager(MANAGER_NAME)


class MemoryProbe:
    """Sample host RSS and whole-device NVML usage, plus exact PyTorch allocator peaks."""

    def __init__(self) -> None:
        import psutil
        import torch

        self.process = psutil.Process()
        self.peak_rss = self.process.memory_info().rss
        self.peak_device_used = None
        self.stop = threading.Event()
        self.nvml = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            pass
        torch.cuda.reset_peak_memory_stats()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()

    def sample(self) -> None:
        while not self.stop.is_set():
            self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
            if self.nvml is not None:
                used = self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used
                self.peak_device_used = max(self.peak_device_used or 0, used)
            self.stop.wait(0.05)

    def finish(self) -> dict[str, Any]:
        import resource

        import torch

        torch.cuda.synchronize()
        self.stop.set()
        self.thread.join()
        result = {
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "sampled_peak_process_rss_bytes": self.peak_rss,
            "process_lifetime_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "sampled_peak_whole_device_used_bytes": self.peak_device_used,
            "sampling_interval_seconds": 0.05,
        }
        if self.nvml is not None:
            self.nvml.nvmlShutdown()
        return result


def tree_copy(value: Any, device: str) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device, copy=True)
    if isinstance(value, dict):
        return {key: tree_copy(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(tree_copy(item, device) for item in value)
    if isinstance(value, list):
        return [tree_copy(item, device) for item in value]
    return value


def setup_aitune(
    pipe: Any,
    upsample_pipe: Any,
    hooks: list[Any],
    reference: list[Any],
    conditioning: dict[str, Any],
    manager: Any,
    deferred: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import torch

    from aitune.torch.jit.flextensor import register_flextensor_units, replan_flextensor

    registration = register_flextensor_units([manager])
    if len(registration.modules) != BLOCKS:
        raise RuntimeError(f"AITune must register all {BLOCKS} live compute children")
    started = time.perf_counter()
    generate(pipe, upsample_pipe, hooks, reference, conditioning, args)
    recording_seconds = time.perf_counter() - started
    deferred()
    tuning_started = time.perf_counter()
    # AITune runs ordinary backend builds, correctness validation and eager/candidate
    # microbenchmark profiling while each enclosing FT shell holds its weights resident.
    for _ in range(3):
        generate(pipe, upsample_pipe, hooks, reference, conditioning, args)
        if registration.all_tuned:
            break
    tuning_seconds = time.perf_counter() - tuning_started
    if not registration.all_tuned:
        raise RuntimeError(f"all {BLOCKS} targets must tune successfully: {registration.state_counts}")

    representative = {}

    def capture(_module: Any, inputs: Any, kwargs: Any) -> None:
        representative["inputs"] = tree_copy(inputs, "cpu")
        representative["kwargs"] = tree_copy(kwargs, "cpu")

    # Warm every observed stage route, including lazy compilation, before readiness
    # evidence or any optional replan. Capture only ONE last (stage-2) root-forward input.
    handle = pipe.transformer.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        generate(pipe, upsample_pipe, hooks, reference, conditioning, args)
    finally:
        handle.remove()
    finalization_started = time.perf_counter()
    if args.finalization == "replan":
        inputs = tree_copy(representative["inputs"], "cuda:0")
        kwargs = tree_copy(representative["kwargs"], "cuda:0")

        def one_transformer_forward() -> None:
            with torch.no_grad():
                pipe.transformer(*inputs, **kwargs)

        replan_flextensor(manager, registration, one_transformer_forward)
    evidence = backend_evidence(registration)
    return {
        "recording_generation_seconds": recording_seconds,
        "post_deferred_tuning_generation_seconds": tuning_seconds,
        "finalization_seconds": time.perf_counter() - finalization_started,
        "finalization": args.finalization,
        "finalization_note": (
            "Keep FT's pre-AITune identity-compile view-profile strategy; no compiled-timing replan is claimed."
            if args.finalization == "retain-profile"
            else "Replan callback executes exactly one complete transformer forward per requested iteration, using stage 2."
        ),
        "state_counts": registration.state_counts,
        **evidence,
    }


def route_evidence(graph: dict[str, Any], block_index: int) -> dict[str, Any]:
    selected = graph.get("selected_backend") or ""
    if "TorchInductorJitBackend" not in selected or graph.get("exception"):
        raise RuntimeError(f"block {block_index} did not select the requested real Inductor backend: {selected}")
    builds = graph.get("backend_builds", [])
    measured = [build.get("throughput") for build in builds if "TorchInductorJitBackend" in build["backend"]]
    if not graph.get("baseline_throughput") or not any(measured):
        raise RuntimeError(f"block {block_index} lacks real eager/candidate profiling measurements")
    return {
        "graph_name": graph["graph_name"],
        "selected_backend": selected,
        "input_spec": graph.get("input_spec"),
        "baseline_throughput": graph.get("baseline_throughput"),
        "backend_builds": builds,
        "strategy_results": graph.get("strategy_results"),
    }


def backend_evidence(registration: Any) -> dict[str, Any]:
    """Require actual graph backend selections for every one of the 48 real blocks."""
    from aitune.torch import snapshot_tuning_data

    path = snapshot_tuning_data()
    if path is None:
        raise RuntimeError("AITune did not produce backend-selection telemetry")
    snapshot = json.loads(Path(path).read_text())
    modules = {item["module_id"]: item for item in snapshot["modules"]}
    targets = []
    for index, report in enumerate(registration.reports):
        module = modules.get(report.module_id, {})
        graphs = module.get("graphs", [])
        if not graphs or module.get("exception"):
            raise RuntimeError(f"block {index} lacks successful graph telemetry")
        routes = [route_evidence(graph, index) for graph in graphs]
        targets.append({
            "name": f"transformer_blocks.{index}.compute",
            "module_id": report.module_id,
            "state": report.state,
            "num_parameters": report.num_parameters,
            "routes": routes,
        })
    return {"tuning_snapshot": str(path), "selected_target_count": len(targets), "targets": targets}


def runtime_versions() -> dict[str, Any]:
    import importlib.metadata

    import torch

    versions = {
        name: importlib.metadata.version(name)
        for name in ("torch", "diffusers", "transformers", "flextensor", "accelerate", "av", "peft")
    }
    direct_url_text = importlib.metadata.distribution("diffusers").read_text("direct_url.json")
    direct_url = json.loads(direct_url_text) if direct_url_text else {}
    versions.update(
        cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(0),
        sm=list(torch.cuda.get_device_capability(0)),
        expected_diffusers_revision=DIFFUSERS_REVISION,
        installed_diffusers_revision=direct_url.get("vcs_info", {}).get("commit_id"),
        installed_diffusers_source=direct_url.get("url"),
        installed_diffusers_archive_sha256=direct_url.get("archive_info", {}).get("hashes", {}).get("sha256"),
    )
    return versions


def finish_flextensor_startup(
    pipe: Any,
    upsample_pipe: Any,
    hooks: list[Any],
    reference: list[Any],
    conditioning: dict[str, Any],
    manager: Any,
    args: argparse.Namespace,
) -> list[str]:
    import flextensor

    for iteration in range(4):
        if manager.phase.value == "inference":
            break
        print(f"FlexTensor startup generation {iteration + 1}: phase={manager.phase.value}", flush=True)
        generate(pipe, upsample_pipe, hooks, reference, conditioning, args)
    if manager.phase.value != "inference":
        raise RuntimeError("FlexTensor did not finish discovery and profiling")
    names = [name for name, _ in manager.offload_units()]
    if names != [f"transformer_blocks.{index}" for index in range(BLOCKS)]:
        raise RuntimeError(f"expected exactly {BLOCKS} FlexTensor shell units, got {names}")
    if not args.reuse_profile:
        flextensor.save_profile(str(args.work_dir / "flextensor-profile"), name=MANAGER_NAME)
    return names


def configure_compiler(args: argparse.Namespace) -> dict[str, Any]:
    """Bound instance-specialized compilation for all 48 real blocks in this fresh worker."""
    import torch

    names = ("recompile_limit", "accumulated_recompile_limit")
    prior = {name: int(getattr(torch.compiler.config, name)) for name in names}
    requested = {
        "recompile_limit": args.inductor_recompile_limit,
        "accumulated_recompile_limit": args.inductor_accumulated_recompile_limit,
    }
    if min(requested.values()) < 1:
        raise ValueError("compiler limits must be positive finite integers")
    for name, value in requested.items():
        setattr(torch.compiler.config, name, value)
    return {
        "prior": prior,
        "requested": requested,
        "effective": {name: int(getattr(torch.compiler.config, name)) for name in names},
        "environment": {"TORCHINDUCTOR_COMPILE_THREADS": os.environ.get("TORCHINDUCTOR_COMPILE_THREADS")},
    }


def prepare_transformer(
    pipe: Any,
    upsample_pipe: Any,
    hooks: list[Any],
    reference: list[Any],
    conditioning: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[Any | None, list[str]]:
    if args.mode == "eager":
        if args.reuse_profile:
            raise ValueError("resident eager mode does not use a FlexTensor profile")
        pipe.transformer.to("cuda:0")
        return None, []
    manager = configure_flextensor(pipe, args)
    names = finish_flextensor_startup(pipe, upsample_pipe, hooks, reference, conditioning, manager, args)
    return manager, names


def flextensor_evidence(manager: Any | None, names: list[str], args: argparse.Namespace) -> dict[str, Any]:
    if manager is None:
        return {"enabled": False, "placement": "whole unwrapped pretrained transformer resident on CUDA"}
    return {
        "enabled": True,
        "unit_names": names,
        "managed_gpu_memory": asdict(manager.get_gpu_memory_usage()),
        "profile_reused": args.reuse_profile,
        "finalization": args.finalization,
    }


def workload_provenance(args: argparse.Namespace) -> dict[str, Any]:
    from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES

    return {
        "frames": FRAMES,
        "fps": FPS,
        "source_size": list(source_size(args)),
        "stage1_size": [args.width, args.height],
        "stage2_size": [2 * args.width, 2 * args.height],
        "seed": SEED,
        "heldout_seed": SEED + 1,
        "guidance": GUIDANCE,
        "stage1_sigmas": [float(value) for value in DISTILLED_SIGMA_VALUES],
        "stage2_sigmas": [float(value) for value in STAGE_2_DISTILLED_SIGMA_VALUES],
        "stage1_reference_strength": 1.0,
        "stage1_reference_downscale_factor": 1,
        "stage2_reference_condition": None,
        "tf32": False,
        "prompt_sha256": sha256(args.work_dir / "prompt.pt"),
        "source_sha256": sha256(args.work_dir / "source.mp4"),
    }


def validate_video_output(frames: list[Any], args: argparse.Namespace) -> dict[str, Any]:
    if len(frames) != FRAMES:
        raise RuntimeError(f"expected {FRAMES} generated frames, received {len(frames)}")
    if any(frame.mode != "RGB" or frame.size != (args.width * 2, args.height * 2) for frame in frames):
        raise RuntimeError(f"every generated frame must be RGB at {args.width * 2}x{args.height * 2}")
    return {"frames": len(frames), "mode": "RGB", "size": [args.width * 2, args.height * 2]}


def validate_heldout_output(heldout: dict[str, Any]) -> dict[str, Any]:
    import torch

    checks = {}
    for name in ("video_latent", "audio_waveform"):
        tensor = heldout.get(name)
        if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
            raise RuntimeError(f"held-out output {name} must be a nonempty tensor")
        finite = bool(torch.isfinite(tensor).all())
        if not finite:
            raise RuntimeError(f"held-out output {name} contains non-finite values")
        checks[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "finite": finite}
    return checks


def validate_host_pinning(args: argparse.Namespace) -> None:
    """Fail before loading weights if requested public in-place pinning is unavailable."""
    if args.mode != "eager" and args.pinned_memory_mode == "host_register":
        from flextensor.host_pinning import is_available

        available = is_available()
        print(f"FlexTensor host pinning requested=host_register runtime_capability_available={available}", flush=True)
        if not available:
            raise RuntimeError("host_register is unavailable; use --pinned-memory-mode torch for copied-buffer pinning")


def run(args: argparse.Namespace) -> None:
    import numpy as np
    import torch

    if args.runs < 1 or args.warmups < 1:
        raise ValueError("use at least one warmup and one measured generation")
    if not torch.cuda.is_available():
        raise RuntimeError("the real pretrained example requires a CUDA GPU")
    versions = runtime_versions()
    if versions["flextensor"] != "0.5.0":
        raise RuntimeError("this example requires the unchanged FlexTensor 0.5.0 public API")
    validate_host_pinning(args)
    compiler_limits = configure_compiler(args)
    run_dir = args.work_dir / args.mode
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ["AITUNE_CACHE_DIR"] = str((run_dir / "aitune-cache").resolve())
    os.environ["TRITON_CACHE_DIR"] = str((run_dir / "triton-cache").resolve())
    os.environ.setdefault("AITUNE_CONSOLE_OUTPUT", "1")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    deferred = configure_aitune(args) if args.mode == "combined" else None
    conditioning = torch.load(args.work_dir / "prompt.pt", map_location="cpu", weights_only=True)
    provenance = conditioning["provenance"]
    if provenance["revision"] != BASE_REVISION or provenance["prompt"] != PROMPT:
        raise ValueError("prompt artifact is not from this example's pinned encoder and prompt")
    reference = prepare_reference(args.work_dir / "source.mp4", args)
    setup_started = time.perf_counter()
    setup_memory = MemoryProbe()
    print(f"Loading pinned pretrained components for {args.mode}", flush=True)
    pipe, upsample_pipe, hooks = load_pipelines(args)
    pretrained_parameter_count = sum(parameter.numel() for parameter in pipe.transformer.parameters())
    print(f"Preparing {BLOCKS} whole transformer blocks ({pretrained_parameter_count} parameters)", flush=True)
    manager, names = prepare_transformer(pipe, upsample_pipe, hooks, reference, conditioning, args)
    print(
        "Transformer placement ready; starting AITune setup" if deferred else "Transformer placement ready", flush=True
    )
    tuning = (
        setup_aitune(pipe, upsample_pipe, hooks, reference, conditioning, manager, deferred, args) if deferred else {}
    )
    setup_seconds = time.perf_counter() - setup_started
    setup_memory_results = setup_memory.finish()
    print(f"Setup complete in {setup_seconds:.3f}s; warming up {args.warmups} generation(s)", flush=True)
    for _ in range(args.warmups):
        generate(pipe, upsample_pipe, hooks, reference, conditioning, args)
    torch.cuda.synchronize()
    gc.collect()
    measured_memory = MemoryProbe()
    latencies = []
    for _ in range(args.runs):
        torch.cuda.synchronize()
        started = time.perf_counter()
        video, _ = generate(pipe, upsample_pipe, hooks, reference, conditioning, args)
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - started)
    memory = measured_memory.finish()
    measured_video_structure = validate_video_output(video, args)
    save_video(video, run_dir / "outpaint.mp4")
    # Independent seed is not recorded during tuning, warmup, or timed iterations.
    heldout_video, heldout = generate(
        pipe,
        upsample_pipe,
        hooks,
        reference,
        conditioning,
        args,
        seed=SEED + 1,
        collect_latent=True,
    )
    heldout_video_structure = validate_video_output(heldout_video, args)
    heldout_structure = validate_heldout_output(heldout)
    heldout["seed"] = SEED + 1
    torch.save(heldout, run_dir / "heldout.pt")
    save_video(heldout_video, run_dir / "heldout_outpaint.mp4")
    result = {
        "mode": args.mode,
        "versions": versions,
        "compiler_limits": compiler_limits,
        "arguments": vars(args),
        "model": {
            "base": BASE_MODEL,
            "base_revision": BASE_REVISION,
            "upsampler": UPSAMPLER_MODEL,
            "upsampler_revision": UPSAMPLER_REVISION,
            "lora": LORA_REPO,
            "lora_revision": LORA_REVISION,
            "lora_file": LORA_FILE,
            "lora_scale_both_stages": 1.0,
            "pretrained_transformer_parameters": pretrained_parameter_count,
            "blocks": BLOCKS,
            "dtype": "bfloat16",
            "quantization": None,
        },
        "workload": workload_provenance(args),
        "setup": {"seconds": setup_seconds, "memory": setup_memory_results, "aitune": tuning},
        "measurement": {
            "warmups": args.warmups,
            "runs": args.runs,
            "latencies_seconds": latencies,
            "mean_seconds": statistics.mean(latencies),
            "p50_seconds": float(np.percentile(latencies, 50)),
            "p95_seconds": float(np.percentile(latencies, 95)),
            "generations_per_second": 1 / statistics.mean(latencies),
            "memory": memory,
        },
        "flextensor": flextensor_evidence(manager, names, args),
        "output_validation": {
            "measured_video": measured_video_structure,
            "heldout_video": heldout_video_structure,
            "heldout_tensors": heldout_structure,
            "note": "Structure and finiteness only; no pixel-closeness or perceptual-quality gate.",
        },
        "artifacts": {
            "source": str((args.work_dir / "source.mp4").resolve()),
            "outpaint": str((run_dir / "outpaint.mp4").resolve()),
            "heldout": str((run_dir / "heldout.pt").resolve()),
        },
    }
    write_json(run_dir / "results.json", result)
    print(json.dumps({"results": str(run_dir / "results.json"), "mean_seconds": statistics.mean(latencies)}))


def numerical_metrics(baseline: Any, candidate: Any) -> dict[str, Any]:
    import torch

    if baseline.shape != candidate.shape:
        raise ValueError(f"validation shapes differ: {baseline.shape} vs {candidate.shape}")
    a, b = baseline.float(), candidate.float()
    difference = (a - b).abs()
    return {
        "shape": list(a.shape),
        "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
        "max_abs_diff": difference.max().item(),
        "mean_abs_diff": difference.mean().item(),
        "rmse": difference.square().mean().sqrt().item(),
        "max_relative_error": (difference / a.abs().clamp_min(1e-8)).max().item(),
        "max_normalized_error": (difference.max() / a.abs().max().clamp_min(1e-8)).item(),
        "relative_l2_error": ((a - b).norm() / a.norm().clamp_min(1e-8)).item(),
        "cosine_similarity": torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item(),
        "allclose_atol_0_01_rtol_0_01": bool(torch.allclose(a, b, atol=1e-2, rtol=1e-2)),
        "note": "Held-out numerical diagnostic, not an automatic perceptual-quality or deployment gate.",
    }


def require_matching_provenance(left: dict[str, Any], right: dict[str, Any]) -> None:
    for key in ("model", "workload", "versions"):
        if left[key] != right[key]:
            raise ValueError(f"{left['mode']} and {right['mode']} {key} provenance must match exactly")
    if left["compiler_limits"]["effective"] != right["compiler_limits"]["effective"]:
        raise ValueError("effective compiler limits must match across compared workers")
    if left["compiler_limits"]["environment"] != right["compiler_limits"]["environment"]:
        raise ValueError("compiler concurrency environment must match across compared workers")
    for key in ("warmups", "runs"):
        if left["measurement"][key] != right["measurement"][key]:
            raise ValueError(f"{left['mode']} and {right['mode']} measurement {key} must match")


def require_real_offload(result: dict[str, Any]) -> None:
    """Require reported rolling storage and savings, not a configured whole-GPU cap."""
    provider = result["flextensor"]
    memory = provider["managed_gpu_memory"]
    if memory["blocks_bytes"] <= 0 or memory["unmapped_tensors_bytes"] < 0:
        raise ValueError(f"{result['mode']} must report positive rolling storage and nonnegative unmapped storage")
    if memory["total_bytes"] != memory["blocks_bytes"] + memory["unmapped_tensors_bytes"]:
        raise ValueError(f"{result['mode']} managed GPU memory accounting is inconsistent")
    if memory["total_bytes"] >= result["model"]["pretrained_transformer_parameters"] * 2:
        raise ValueError(f"{result['mode']} must use less managed GPU storage than the BF16 weight lower bound")
    if provider["unit_names"] != [f"transformer_blocks.{index}" for index in range(BLOCKS)]:
        raise ValueError(f"{result['mode']} must report all {BLOCKS} ordered whole-block FlexTensor units")


def require_matching_flextensor_policy(baseline: dict[str, Any], combined: dict[str, Any]) -> None:
    """Bind caller settings and the shared profile location, not historical profile bytes."""
    for result in (baseline, combined):
        require_real_offload(result)
    left, right = baseline["arguments"], combined["arguments"]
    for key in ("gpu_memory_fraction", "pinned_memory_mode", "finalization", "ft_profile_steps"):
        if left[key] != right[key]:
            raise ValueError(f"FlexTensor baseline and combined {key} must match")
    if left["finalization"] == "retain-profile" and right.get("reuse_profile") is not True:
        raise ValueError("retain-profile comparison requires combined --reuse-profile")
    if right.get("reuse_profile") and left["work_dir"] != right["work_dir"]:
        raise ValueError("profile reuse requires the same baseline and combined work-dir/profile location")


def compare_heldout(work_dir: Path, left_mode: str, right_mode: str) -> dict[str, Any]:
    import torch

    a = torch.load(work_dir / left_mode / "heldout.pt", map_location="cpu", weights_only=True)
    b = torch.load(work_dir / right_mode / "heldout.pt", map_location="cpu", weights_only=True)
    if a["seed"] != b["seed"] or a["seed"] != SEED + 1:
        raise ValueError("validation must use the same held-out, non-tuning seed")
    return {name: numerical_metrics(a[name], b[name]) for name in ("video_latent", "audio_waveform")}


def eager_comparison(
    args: argparse.Namespace,
    baseline: dict[str, Any],
    combined: dict[str, Any],
) -> dict[str, Any]:
    eager = json.loads((args.work_dir / "eager" / "results.json").read_text())
    require_matching_provenance(eager, baseline)
    require_matching_provenance(eager, combined)
    measured = eager["measurement"]
    return {
        "mean_seconds": measured["mean_seconds"],
        "memory": measured["memory"],
        "ft_baseline_speedup_vs_eager": measured["mean_seconds"] / baseline["measurement"]["mean_seconds"],
        "combined_speedup_vs_eager": measured["mean_seconds"] / combined["measurement"]["mean_seconds"],
        "ft_baseline_peak_allocated_bytes_saved": measured["memory"]["torch_peak_allocated_bytes"]
        - baseline["measurement"]["memory"]["torch_peak_allocated_bytes"],
        "combined_peak_allocated_bytes_saved": measured["memory"]["torch_peak_allocated_bytes"]
        - combined["measurement"]["memory"]["torch_peak_allocated_bytes"],
        "baseline_vs_eager": compare_heldout(args.work_dir, "eager", "baseline"),
        "combined_vs_eager": compare_heldout(args.work_dir, "eager", "combined"),
    }


def compare(args: argparse.Namespace) -> None:
    baseline = json.loads((args.work_dir / "baseline" / "results.json").read_text())
    combined = json.loads((args.work_dir / "combined" / "results.json").read_text())
    require_matching_provenance(baseline, combined)
    require_matching_flextensor_policy(baseline, combined)
    result = {
        "speedup": baseline["measurement"]["mean_seconds"] / combined["measurement"]["mean_seconds"],
        "baseline_mean_seconds": baseline["measurement"]["mean_seconds"],
        "combined_mean_seconds": combined["measurement"]["mean_seconds"],
        "baseline_memory": baseline["measurement"]["memory"],
        "combined_memory": combined["measurement"]["memory"],
        "selected_target_count": combined["setup"]["aitune"]["selected_target_count"],
        "heldout_seed": SEED + 1,
        "numerical_diagnostics": compare_heldout(args.work_dir, "baseline", "combined"),
        "human_review_required": True,
        "human_review_note": "Inspect the source and all available outpaint videos for content, boundary, motion and quality.",
        "tuning_correctness_note": "Normal AITune backend correctness checks ran separately before backend selection.",
    }
    if (args.work_dir / "eager" / "results.json").is_file():
        result["resident_eager_reference"] = eager_comparison(args, baseline, combined)
    write_json(args.work_dir / "comparison.json", result)
    print(json.dumps(result, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare-source", "encode-prompt", "run", "compare"])
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--model-cache", type=Path, default=DEFAULT_WORK_DIR / "models")
    parser.add_argument("--input-video", type=Path)
    parser.add_argument("--start-seconds", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Stage-1 width; final output is twice this")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Stage-1 height; final output is twice this")
    parser.add_argument("--mode", choices=["eager", "baseline", "combined"], default="baseline")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--profile-steps", type=int, default=3)
    parser.add_argument("--ft-profile-steps", type=int, default=10)
    parser.add_argument("--inductor-recompile-limit", type=int, default=128)
    parser.add_argument("--inductor-accumulated-recompile-limit", type=int, default=512)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.65)
    parser.add_argument("--pinned-memory-mode", choices=["torch", "host_register"], default="host_register")
    parser.add_argument("--finalization", choices=["retain-profile", "replan"], default="retain-profile")
    parser.add_argument("--reuse-profile", action="store_true")
    parser.add_argument("--performance-validation", choices=["diagnostic", "enabled"], default="diagnostic")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_size(args)
    if args.phase == "prepare-source":
        prepare_source(args)
    elif args.phase == "encode-prompt":
        encode_prompt(args)
    elif args.phase == "run":
        run(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
