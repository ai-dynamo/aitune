# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compiler timing aggregation and installed PyTorch integration."""

from pathlib import Path
import tempfile
import unittest

from flux_schnell_recipe.compilation import CompilationRecorder, interval_seconds


class CompilationTests(unittest.TestCase):
    def test_union_does_not_double_count_nested_or_overlapping_spans(self):
        records = [{"start_time_us": a, "end_time_us": b, "is_runtime": False}
                   for a, b in ((0, 10), (2, 4), (8, 15), (20, 25))]
        records.append({"start_time_us": 0, "end_time_us": 100, "is_runtime": True})
        self.assertEqual(interval_seconds(records), 20 / 1e6)
        self.assertEqual(interval_seconds([]), 0)

    def test_missing_timings_are_rejected(self):
        with self.assertRaises(ValueError):
            interval_seconds([{"start_time_us": None, "end_time_us": None, "is_runtime": False}])

    def test_torch_records_survive_reset_and_exclude_ordinary_inference(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            recorder = CompilationRecorder(Path(directory))
            torch._dynamo.reset()
            compiled = torch.compile(lambda x: x + 1, backend="eager")
            torch.testing.assert_close(compiled(torch.ones(3)), torch.full((3,), 2.0))
            first = recorder.snapshot()
            self.assertGreater(len(first), 0)
            torch._dynamo.reset()
            self.assertEqual(recorder.snapshot(), first)
            result = recorder.finish()
            self.assertGreater(result["compilation_s"], 0)
            self.assertTrue(Path(result["raw_compilation"]).is_file())
            recorder = CompilationRecorder(Path(directory))
            torch.ones(3) + 1
            self.assertEqual(recorder.finish()["compilation_s"], 0)


if __name__ == "__main__":
    unittest.main()
