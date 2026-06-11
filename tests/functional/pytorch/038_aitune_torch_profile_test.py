# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# dependencies = ["transformers"]
# scope = "always"
# ///

import json

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from aitune.torch import profile

MODEL_ID = "hf-internal-testing/tiny-random-distilbert"


def test_profile_transformers_kwargs(tmp_path):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()

    encoded_inputs = tokenizer("AITune runtime attribution smoke test.", return_tensors="pt")

    def inference_function(**kwargs):
        with torch.inference_mode():
            return model(**kwargs)

    result = profile(
        obj=model,
        input_data=dict(encoded_inputs),
        inference_function=inference_function,
        warmup_runs=1,
        measured_runs=2,
        trace_file=tmp_path / "trace.json",
    )

    json_path = tmp_path / "transformers_profile.json"
    markdown_path = tmp_path / "transformers_profile.md"
    json_path.write_text(json.dumps(result.data, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(result.markdown(), encoding="utf-8")

    report = json.loads(json_path.read_text(encoding="utf-8"))

    assert json_path.exists()
    assert markdown_path.exists()
    assert result.trace_file == (tmp_path / "trace.json").resolve()
    assert result.trace_file.exists()
    assert result.trace_file.name == "trace.json"
    assert result.trace_file.stat().st_size > 0
    assert report == result.data
    assert "artifacts" not in report

    assert report["config"]["warmup_runs"] == 1
    assert report["config"]["measured_runs"] == 2
    assert report["config"]["uses_inference_function"] is True
    assert report["input"]["kwargs"] == ["attention_mask", "input_ids"]
    assert len(report["runs"]) == 2
    assert all(run["timing"]["wall_time_us"] > 0 for run in report["runs"])
    assert all(run["timing"]["cpu_time_us"] > 0 for run in report["runs"])
    assert report["warnings"] == []

    # DistilBERT has discoverable named children. All should appear as untuned regions
    # (no AITune wrappers exist, so the AOT side is empty).
    assert report["regions"], "expected at least one untuned region for DistilBERT submodules"
    assert all(region["kind"] == "untuned_module" for region in report["regions"])
    assert all("wrapper_state" not in region for region in report["regions"])
    untuned_ids = {region["id"] for region in report["regions"]}

    for run in report["runs"]:
        per_run_ids = {region["region_id"] for region in run["regions"]}
        # Per-run region ids must be a subset of the discovered untuned region ids
        # (no AOT entries since nothing was wrapped).
        assert per_run_ids <= untuned_ids
        assert per_run_ids  # at least one untuned region fired per run
        residual = run["residual"]
        # Residual CPU is positive (framework dispatch + glue code) but strictly less
        # than 100% now that the children are individually attributed.
        assert 0.0 < residual["cpu_time_us"] < run["timing"]["cpu_time_us"]
        assert 0.0 < residual["cpu_time_fraction"] < 1.0
        assert "device_time_us" not in residual
        assert "device_time_fraction" not in residual

    key_average_rows = report["profiler"]["key_averages"]["cpu_time_total"]["events"]
    assert any(row["key"] == "aitune.performance.profiled_run" for row in key_average_rows)
    assert "Performance Profile" in markdown_path.read_text(encoding="utf-8")


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        test_profile_transformers_kwargs(tmp_path=Path(tmpdir))
