# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# scope = "always"
# ///

import json

import torch

import aitune.torch as ait
from aitune.torch.backend import TorchEagerBackend


class CudaBlock(torch.nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.linear = torch.nn.Linear(width, width)
        self.norm = torch.nn.LayerNorm(width)

    def forward(self, x):
        return torch.relu(self.norm(self.linear(x)))


class MultiRegionCudaPipeline:
    def __init__(self, width: int, device: torch.device):
        self.block_1 = self._wrapped_block("cuda-block-1", width, device)
        self.block_2 = self._wrapped_block("cuda-block-2", width, device)
        self.block_3 = self._wrapped_block("cuda-block-3", width, device)

    def __call__(self, x, scale):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        return x * scale

    @staticmethod
    def _wrapped_block(name: str, width: int, device: torch.device):
        block = CudaBlock(width).to(device).eval()
        strategy = ait.OneBackendStrategy(TorchEagerBackend()).enable_find_max_batch_size(False)
        return ait.Module(block, name, strategy=strategy)

    @property
    def modules(self):
        return [self.block_1, self.block_2, self.block_3]


def test_profile_cuda_aot_multiple_regions(tmp_path):
    device = torch.device("cuda")
    pipeline = MultiRegionCudaPipeline(width=64, device=device)
    input_data = {
        "x": torch.randn(16, 64, device=device),
        "scale": torch.tensor(1.25, device=device),
    }

    with torch.inference_mode():
        pipeline(**input_data)

    for module in pipeline.modules:
        module.tune(device=device)
        module.activate()
        assert module.state.value == "tuned"

    def inference_function(**kwargs):
        with torch.inference_mode():
            return pipeline(**kwargs)

    result = ait.profile(
        obj=pipeline,
        input_data=input_data,
        inference_function=inference_function,
        warmup_runs=1,
        measured_runs=3,
        trace_file=tmp_path / "trace.json",
    )

    json_path = tmp_path / "cuda_profile.json"
    markdown_path = tmp_path / "cuda_profile.md"
    json_path.write_text(json.dumps(result.data, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(result.markdown(), encoding="utf-8")

    report = json.loads(json_path.read_text(encoding="utf-8"))
    expected_region_ids = {"aot_module:cuda-block-1", "aot_module:cuda-block-2", "aot_module:cuda-block-3"}

    assert report == result.data
    assert markdown_path.exists()
    assert result.trace_file == (tmp_path / "trace.json").resolve()
    assert result.trace_file.exists()
    assert result.trace_file.stat().st_size > 0
    assert "artifacts" not in report
    assert report["config"]["uses_inference_function"] is True
    assert report["input"]["kwargs"] == ["scale", "x"]
    assert report["profiler"]["activities"] == ["CPU", "CUDA"]
    assert "device_time_total" in report["profiler"]["key_averages"]
    assert len(report["runs"]) == 3
    assert {region["id"] for region in report["regions"]} == expected_region_ids
    assert all(region["wrapper_state"] == "tuned" for region in report["regions"])
    assert report["warnings"] == []

    for run in report["runs"]:
        assert run["timing"]["wall_time_us"] > 0
        assert run["timing"]["cpu_time_us"] > 0
        assert run["timing"]["device_time_us"] > 0
        assert {region["region_id"] for region in run["regions"]} == expected_region_ids
        assert len(run["regions"]) == 3
        assert all(region["calls"] == 1 for region in run["regions"])
        assert all(region["cpu_time_us"] > 0 for region in run["regions"])
        assert all(0 < region["cpu_time_fraction"] <= 1 for region in run["regions"])
        assert all(region["device_time_us"] > 0 for region in run["regions"])
        assert all(0 < region["device_time_fraction"] <= 1 for region in run["regions"])

    trace = result.trace_file.read_text(encoding="utf-8")
    markdown = markdown_path.read_text(encoding="utf-8")
    for region_id in expected_region_ids:
        region_name = region_id.removeprefix("aot_module:")
        assert f"aitune.performance.aot_module:{region_name}" in trace
        assert f"`{region_id}`" in markdown
    assert "## Per-Run Attribution" in markdown
    assert "Device Fraction" in markdown


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        test_profile_cuda_aot_multiple_regions(tmp_path=Path(tmpdir))
