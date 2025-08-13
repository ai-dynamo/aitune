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
"""ONNX model utilities for testing."""

from pathlib import Path


class ToyOnnxModel:
    """Toy ONNX model."""

    def __init__(self, is_linear: bool = True):
        self.is_linear = is_linear

    @property
    def path(self) -> Path:
        """Get the path to a toy ONNX model.

        Args:
            is_linear: If True, returns the path to the linear model.
                    If False, returns the path to the convolutional model.

        Returns:
            Path to the ONNX model file.
        """
        base_path = Path(__file__).parent / "onnx"
        model_name = "toy_linear.onnx" if self.is_linear else "toy_conv.onnx"
        model_path = base_path / model_name

        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found at {model_path}")

        return model_path
