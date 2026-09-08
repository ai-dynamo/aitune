# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TensorRT plan and optimization-profile artifact records."""

from dataclasses import dataclass

from aitune.records.artifacts.base import Artifact
from aitune.records.shapes import _validate_bound


@dataclass(frozen=True, slots=True, kw_only=True)
class TensorRTProfileInput:
    """Shape range for one input in a TensorRT optimization profile."""

    name: str
    min_shape: tuple[int, ...]
    opt_shape: tuple[int, ...]
    max_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate the profile's concrete shapes and ordering."""
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("TensorRTProfileInput.name must be a non-empty string")
        for field_name in ("min_shape", "opt_shape", "max_shape"):
            _validate_bound(field_name, getattr(self, field_name))
        ranks = {len(self.min_shape), len(self.opt_shape), len(self.max_shape)}
        if len(ranks) != 1:
            raise ValueError(f"TensorRT profile shapes for {self.name!r} must have the same rank")
        for axis, (minimum, optimum, maximum) in enumerate(
            zip(self.min_shape, self.opt_shape, self.max_shape, strict=True)
        ):
            if not minimum <= optimum <= maximum:
                raise ValueError(
                    f"Axis {axis} of TensorRT profile input {self.name!r} must satisfy min <= opt <= max, "
                    f"got min={minimum}, opt={optimum}, max={maximum}"
                )


@dataclass(frozen=True, slots=True, kw_only=True)
class TensorRTOptimizationProfile:
    """Ordered input shape ranges optimized together by TensorRT."""

    inputs: tuple[TensorRTProfileInput, ...]

    def __post_init__(self) -> None:
        """Require a non-empty profile with unique input names."""
        if not self.inputs:
            raise ValueError("A TensorRT optimization profile must contain at least one input")
        names = self.input_names
        if len(names) != len(set(names)):
            raise ValueError(f"TensorRT optimization profile input names must be unique, got {names}")

    @property
    def input_names(self) -> tuple[str, ...]:
        """Return input names in executable order."""
        return tuple(input_shape.name for input_shape in self.inputs)


@dataclass(frozen=True, kw_only=True)
class TensorRTPlanArtifact(Artifact):
    """A serialized TensorRT plan produced by tuning.

    Args:
        optimization_profiles: Exact min/opt/max input ranges stored in the plan.
        use_cuda_graphs: Whether the selected AITune runtime used CUDA graphs.
    """

    optimization_profiles: tuple[TensorRTOptimizationProfile, ...]
    use_cuda_graphs: bool = False

    def __post_init__(self) -> None:
        """Validate TensorRT-specific runtime metadata."""
        super().__post_init__()
        if not self.optimization_profiles:
            raise ValueError("A TensorRT plan must contain at least one optimization profile")
        input_specs = dict(zip(self.input_names, self.inputs, strict=True))
        for profile in self.optimization_profiles:
            if profile.input_names != self.input_names:
                raise ValueError(
                    "TensorRT optimization profile inputs must match the artifact input order, "
                    f"got {profile.input_names}, expected {self.input_names}"
                )
            for profile_input in profile.inputs:
                spec = input_specs[profile_input.name]
                if len(profile_input.min_shape) != len(spec.min_shape):
                    raise ValueError(f"TensorRT profile rank does not match artifact input {profile_input.name!r}")
                if any(
                    profile_min < spec_min or profile_max > spec_max
                    for profile_min, profile_max, spec_min, spec_max in zip(
                        profile_input.min_shape,
                        profile_input.max_shape,
                        spec.min_shape,
                        spec.max_shape,
                        strict=True,
                    )
                ):
                    raise ValueError(
                        f"TensorRT profile bounds for {profile_input.name!r} must stay inside its artifact bounds"
                    )

    @property
    def optimization_profile_count(self) -> int:
        """Return the number of optimization profiles stored in the plan."""
        return len(self.optimization_profiles)
