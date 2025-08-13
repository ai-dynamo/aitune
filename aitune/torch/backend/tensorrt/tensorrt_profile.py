# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TensorRT Profile for specifying optimization profiles."""

import logging

from polygraphy.backend.trt import Profile

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
        self, name: str, min_shape: tuple[int, ...], opt_shape: tuple[int, ...], max_shape: tuple[int, ...]
    ) -> "TensorRTProfile":
        """Add a shape binding to the profile.

        Args:
            name: The name of the input tensor
            min_shape: The minimum shape the profile will support
            opt_shape: The shape for which TensorRT will tune the engine
            max_shape: The maximum shape the profile will support

        Returns:
            The profile object for chaining
        """
        self._profile.add(name=name, min=min_shape, opt=opt_shape, max=max_shape)
        logger.debug(
            "Added profile for input '%s': min=%s, opt=%s, max=%s",
            name,
            min_shape,
            opt_shape,
            max_shape,
        )
        return self

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
