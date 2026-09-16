# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# dependencies = ["transformers"]
# scope = "always"
# allow_failure = false
# [[pip_install]]
# packages = ["onnxruntime-gpu"]
# flags = ["--upgrade", "--index-url", "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/"]
# ///

"""Export pretrained BERT to ONNX, tune, report throughput, and save/load.

Based on bert-base-uncased in trtre/models/kpi/huggingface_models.py:
128-token int64 inputs, with batches capped at 4 for this functional test.
Synthetic token IDs check execution and numerical agreement, not task accuracy.
Run: python -m pytest tests/functional/onnx/006_onnx_bert.py -q -s
Use --basetemp=/path/to/artifacts to choose where ONNX, reports, and checkpoints stay.
"""

import json
from pathlib import Path

import pytest
import torch
from transformers import AutoModel

from aitune.torch import MaxThroughputStrategy, Module, PerformanceValidationMode, load, save, tune
from aitune.torch.backend import ONNXRuntimeBackend, TensorRTBackend, TensorRTBackendConfig
from aitune.torch.dataloader import DynamicShapeDataset
from aitune.torch.module import OnnxModule
from aitune.torch.task.profiling import ProfilingConfig


@torch.inference_mode()
def test_onnx_bert(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    model_name = "bert-base-uncased"
    reference = AutoModel.from_pretrained(model_name, attn_implementation="eager", return_dict=False).eval()
    batch_sizes = [1, 2, 4]
    input_ids = torch.randint(
        reference.config.vocab_size, (max(batch_sizes), 128), generator=torch.Generator().manual_seed(0)
    )
    path = tmp_path / f"{model_name}.onnx"
    torch.onnx.export(
        reference,
        (input_ids[:1],),
        path,
        input_names=["input_ids"],
        output_names=["last_hidden_state", "pooler_output"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "last_hidden_state": {0: "batch"},
            "pooler_output": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
        external_data=False,
    )
    reference_output = dict(zip(["last_hidden_state", "pooler_output"], reference(input_ids[:1]), strict=True))
    del reference

    source = OnnxModule(path)
    module = None
    try:
        requests = input_ids.cuda()
        expected = {batch: source(input_ids=requests[:batch]) for batch in batch_sizes}
        torch.testing.assert_close(
            {name: value.cpu() for name, value in expected[1].items()}, reference_output, rtol=1e-2, atol=1e-2
        )
        strategy = MaxThroughputStrategy(
            [
                ONNXRuntimeBackend(),
                TensorRTBackend(TensorRTBackendConfig(workspace_size=1 << 30)),
            ],
            profiling_config=ProfilingConfig(batch_sizes=batch_sizes),
        )
        strategy.enable_find_max_batch_size(False)
        strategy.enable_performance_validation(PerformanceValidationMode.DIAGNOSTIC)
        module = Module(source, model_name, strategy=strategy)
        tune(
            module,
            DynamicShapeDataset([{"input_ids": tokens} for tokens in requests]),
            batch_sizes=batch_sizes,
            device="cuda",
            ignore_failing_modules=False,
        )
        (backend,) = module.module.backends.values()
        results = strategy.perf_validation_results
        assert len(results) == 2
        assert all(result["success"] for result in strategy.backend_results)
        assert all(result.metric > 0 and result.baseline_metric > 0 for result in results)
        assert max(results, key=lambda result: result.metric).backend_description == backend.describe()
        report = [
            {
                "backend": result.backend_description,
                "baseline_samples_per_second": result.baseline_metric,
                "tuned_samples_per_second": result.metric,
                "speedup": result.speedup,
                "selected": result.backend_description == backend.describe(),
            }
            for result in results
        ]
        (tmp_path / "performance.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"{model_name}: {json.dumps(report, indent=2)}")
        for batch in batch_sizes:
            torch.testing.assert_close(module(input_ids=requests[:batch]), expected[batch], rtol=1e-2, atol=1e-2)

        checkpoint = tmp_path / f"{model_name}.ait"
        save(module, checkpoint)
        assert checkpoint.stat().st_size > 0
        module.deactivate()
        module = load(module, checkpoint)
        for batch in batch_sizes:
            torch.testing.assert_close(module(input_ids=requests[:batch]), expected[batch], rtol=1e-2, atol=1e-2)
        print(f"Artifacts: {tmp_path}")
    finally:
        if module is not None and module.state.name == "TUNED":
            for backend in module.module.backends.values():
                backend._deactivate()
        source.deactivate()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
