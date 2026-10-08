# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU regression checks for hard failures and advisory SSIM thresholds."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import yaml

from flux_schnell_recipe.config import Config, load_config
from flux_schnell_recipe.correctness import compare, require_correctness


LOGGER = "flux_schnell_recipe.correctness"
RECIPE = Path(__file__).resolve().parents[1] / "recipes/single-gpu.yaml"


class ThresholdTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.records = [{"id": "first"}, {"id": "second"}]
        for item in self.records:
            Image.new("RGB", (8, 8)).save(self.directory / f"{item['id']}.png")
        self.cfg = {"validation": {"threshold": 0.70, "soft_threshold": 0.95},
                    "workload": {"width": 8, "height": 8}}

    def compare_scores(self, *scores):
        # Supply exact SSIM values to exercise boundaries without depending on image rounding.
        with patch(f"{LOGGER}.score_images", side_effect=scores):
            return compare(self.cfg, self.directory, self.directory, self.records[:len(scores)])

    def test_below_hard_threshold_fails(self):
        with self.assertNoLogs(LOGGER, level="WARNING"):
            result = self.compare_scores(0.699999, 1.0)
        self.assertFalse(result["passed"])
        self.assertFalse(result["scores"][0]["passed"])
        self.assertFalse(result["soft_passed"])

    def test_soft_misses_warn_but_pass_including_hard_boundary(self):
        with self.assertLogs(LOGGER, level="WARNING") as logs:
            result = self.compare_scores(0.70, 0.949999)
        self.assertTrue(result["passed"])
        self.assertFalse(result["soft_passed"])
        self.assertEqual(result["threshold"], 0.70)
        self.assertEqual(result["soft_threshold"], 0.95)
        self.assertEqual(len(logs.output), 2)
        self.assertIn("first: SSIM 0.700000", logs.output[0])
        self.assertIn("soft threshold 0.950000", logs.output[0])
        self.assertIn("hard threshold 0.700000 passed", logs.output[0])
        self.assertTrue(all(item["passed"] and not item["soft_passed"] for item in result["scores"]))

    def test_at_or_above_soft_threshold_passes_without_warning(self):
        with self.assertNoLogs(LOGGER, level="WARNING"):
            result = self.compare_scores(0.95, 1.0)
        self.assertTrue(result["passed"])
        self.assertTrue(result["soft_passed"])

    def test_soft_threshold_is_optional(self):
        for validation in ({"threshold": 0.70}, {"threshold": 0.70, "soft_threshold": None}):
            with self.subTest(validation=validation), self.assertNoLogs(LOGGER, level="WARNING"):
                self.cfg["validation"] = validation
                result = self.compare_scores(0.8)
                self.assertTrue(result["passed"])
                self.assertIsNone(result["soft_threshold"])
                self.assertIsNone(result["soft_passed"])
                self.assertIsNone(result["scores"][0]["soft_passed"])

    def test_soft_change_preserves_existing_hard_pass_evidence(self):
        cfg = Config(self.directory / "recipe.yaml", {
            "output_dir": str(self.directory), "inputs": {"validation": "inputs.jsonl"},
            "validation": {"threshold": 0.70, "soft_threshold": 0.95}})
        metadata = {"contract": {}, "sha256": "artifact"}
        evidence = {"passed": True, "threshold": 0.70, "contract": {},
                    "artifact_sha256": "artifact", "validation_inputs_sha256": "inputs"}
        with patch(f"{LOGGER}.read_json", side_effect=[{"report": "report.json"}, evidence]), \
                patch(f"{LOGGER}.file_sha", return_value="inputs"):
            self.assertEqual(require_correctness(cfg, metadata), evidence)


class ConfigThresholdTests(unittest.TestCase):
    def setUp(self):
        self.raw = yaml.safe_load(RECIPE.read_text())
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "recipe.yaml"

    def load(self, validation):
        raw = deepcopy(self.raw)
        raw["validation"] = validation
        self.path.write_text(yaml.safe_dump(raw))
        return load_config(self.path)

    def test_optional_and_boundary_values(self):
        self.assertEqual(load_config(RECIPE)["validation"]["soft_threshold"], 0.95)
        for value in (None, 0.70, 0.95, 1):
            with self.subTest(value=value):
                cfg = self.load({"metric": "rgb_ssim", "threshold": 0.70, "soft_threshold": value})
                self.assertEqual(cfg["validation"]["soft_threshold"], value)
        self.load({"metric": "rgb_ssim", "threshold": 0.70})

    def test_invalid_soft_thresholds_are_rejected(self):
        for value in (0, 0.69, 1.01, True, False, "0.95", float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "validation.soft_threshold"):
                self.load({"metric": "rgb_ssim", "threshold": 0.70, "soft_threshold": value})


if __name__ == "__main__":
    unittest.main()
