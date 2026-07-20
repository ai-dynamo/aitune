# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TensorRT Profile for specifying optimization profiles."""

import logging

from polygraphy.backend.trt import Profile

from aitune.torch.utils.tensor import format_tensor_name

# Setup logger
logger = logging.getLogger(__name__)


class TensorRTProfile:
    """Class for representing a TensorRT optimization profile.

    This class provides an interface for defining optimization profiles
    for TensorRT engines with dynamic shapes.
    """

    def __init__(self):
        """Initialize a TensorRT optimization profile."""
        self._profile = Profile()

    def add_input_shape(
        self,
        path: str | tuple[str | int, ...],
        min_shape: tuple[int, ...],
        opt_shape: tuple[int, ...],
        max_shape: tuple[int, ...],
    ) -> "TensorRTProfile":
        """Add a shape binding to the profile.

        Args:
            path: Forward parameter path identifying the input tensor
            min_shape: The minimum shape the profile will support
            opt_shape: The shape for which TensorRT will tune the engine
            max_shape: The maximum shape the profile will support

        Returns:
            The profile object for chaining
        """
        name = format_tensor_name(path, "input")
        self._profile.add(name=name, min=min_shape, opt=opt_shape, max=max_shape)
        logger.debug(
            "Added profile for input '%s': min=%s, opt=%s, max=%s",
            name,
            min_shape,
            opt_shape,
            max_shape,
        )
        return self

    def __eq__(self, other: "TensorRTProfile") -> bool:
        """Check if two TensorRTProfiles are equal."""
        return hash(self) == hash(other)

    def __hash__(self) -> int:
        """Hash the TensorRTProfile."""
        return hash(
            tuple(
                sorted(
                    ((name, min_, opt_, max_) for name, (min_, opt_, max_) in self._profile.items()), key=lambda x: x[0]
                )
            )
        )

    @property
    def profile(self) -> Profile:
        """Get the underlying Polygraphy Profile.

        Returns:
            The Polygraphy Profile object
        """
        return self._profile

    def __str__(self) -> str:
        """Return string representation of the profile.

        Returns:
            String representation
        """
        return str(self._profile)

    def __repr__(self) -> str:
        """Return the official string representation of the profile.

        Returns:
            Official string representation
        """
        return repr(self._profile)

    def to_dict(self) -> dict:
        """Convert TensorRTProfile to dictionary."""
        return self.profile_to_dict(self._profile)

    @classmethod
    def from_dict(cls, data: dict) -> "TensorRTProfile":
        """Create TensorRTProfile from dictionary."""
        profile = cls()
        for name, (min_, opt_, max_) in data.items():
            profile._profile.add(name=name, min=tuple(min_), opt=tuple(opt_), max=tuple(max_))
        return profile

    @classmethod
    def profile_to_dict(cls, profile: Profile) -> dict:
        """Convert Polygraphy Profile to dictionary."""
        return {name: [min_, opt_, max_] for name, (min_, opt_, max_) in profile.items()}
