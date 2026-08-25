# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Utilities for validating and benchmarking kernel providers."""

from collections.abc import Callable
from copy import deepcopy
from typing import Any, Literal

import pandas as pd
import torch
import triton

from aitune.torch.module.recording_module import Sample
from aitune.torch.module.sample_metadata import SampleMetadata
from aitune.torch.module.tensor_spec import InfoLevel


class KernelUtils:
    """Kernel utilities."""

    def validate_function(
        self,
        real_function: Callable,
        new_function: Callable,
        samples: list[Sample],
        atol: float = 1e1,
        rtol: float = 1e1,
    ) -> list[str]:
        """Validate the new function against the real function.

        Assumptions is that the real_function works with a given example,
        and the new_function is a candidate for replacement. Both functions
        run under ``torch.no_grad()``.

        Args:
            real_function: the real function to validate against
            new_function: the new function to validate
            samples: the samples to use for validation
            atol: the absolute tolerance for validation
            rtol: the relative tolerance for validation

        Returns:
            A list of errors
        """
        errors = []
        for sample in samples:
            real_output = self._call_function(real_function, sample)
            try:
                new_output = self._call_function(new_function, sample)
            except Exception as e:
                errors.append(f"New function failed: {type(e).__name__}: {e}")
                continue

            new_metadata = SampleMetadata.from_outputs(new_output, strict=True)
            real_metadata = SampleMetadata.from_outputs(real_output, strict=True)

            errors.extend(self._validate_tensors(real_output, new_output, real_metadata, new_metadata, atol, rtol))
            errors.extend(self._validate_other_data(real_metadata, new_metadata))
        return errors

    def benchmark_function(
        self,
        function: Callable,
        samples: list[Sample],
        return_mode: Literal["min", "max", "mean", "median", "all"] = "mean",
        warmup: int = 25,
        repeats: int = 100,
    ) -> float:
        """Benchmark the function with given samples.

        Uses triton.testing.do_bench to benchmark the function under
        ``torch.no_grad()``.

        Args:
            function: the function to benchmark
            samples: the samples to use for benchmarking
            return_mode: the mode to use for benchmarking
            warmup: the number of warmup runs
            repeats: the number of repeat runs

        Returns:
            A float value of the benchmark result.
        """

        def test_fn():
            """Run the benchmarked function once for every sample."""
            for args, kwargs in samples:
                function(*args, **kwargs)

        with torch.no_grad():
            return triton.testing.do_bench(test_fn, warmup=warmup, rep=repeats, return_mode=return_mode)

    @classmethod
    def _validate_tensors(
        cls,
        real_output: Any,
        new_output: Any,
        real_metadata: SampleMetadata,
        new_metadata: SampleMetadata,
        atol: float,
        rtol: float,
    ) -> list[str]:
        """Validate tensors in the real and new metadata."""
        errors = []
        try:
            for (real_locator, real_spec), (new_locator, new_spec) in zip(
                real_metadata.tensor_data,
                new_metadata.tensor_data,
                strict=True,
            ):
                if real_locator != new_locator or real_spec.shape != new_spec.shape:
                    errors.append(
                        f"Tensor metadata mismatch, expected: "
                        f"{real_locator} {real_spec.describe(InfoLevel.MEDIUM)}, got: "
                        f"{new_locator} {new_spec.describe(InfoLevel.MEDIUM)}"
                    )
                    continue
                real_value = real_locator.get_value(real_output)
                new_value = new_locator.get_value(new_output)
                try:
                    torch.testing.assert_close(real_value, new_value, atol=atol, rtol=rtol)
                except Exception as e:
                    errors.append(f"{e}\nAffected tensor: {real_locator}")
        except Exception as e:
            errors.append(f"Tensor validation failed: {e}")

        return errors

    def _validate_other_data(self, real_metadata: SampleMetadata, new_metadata: SampleMetadata) -> list[str]:
        """Validate other data in the real and new metadata."""
        errors = []
        if dict(real_metadata.other_data) != dict(new_metadata.other_data):
            real = pd.DataFrame(real_metadata.other_data, columns=["Locator", "Value"])
            new = pd.DataFrame(new_metadata.other_data, columns=["Locator", "Value"])
            errors.append(f"Other data mismatch, expected:\n{real}, got:\n{new}")
        return errors

    @staticmethod
    def _call_function(function: Callable, sample: Sample) -> Any:
        """Call a function with copied sample inputs."""
        args, kwargs = deepcopy(sample)
        with torch.no_grad():
            return function(*args, **kwargs)
