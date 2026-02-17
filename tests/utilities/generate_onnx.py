# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
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
"""Generate ONNX models for testing."""

import argparse
import os
import sys

import torch

# Support both direct execution and package imports
if __name__ == "__main__" and __package__ is None:
    # When run directly
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from tests.toy_models.torch_models import ToyTorchModel
else:
    # When imported as a module
    from ..toy_models.torch_models import ToyTorchModel


def generate_onnx_model(model: torch.nn.Module, input_shape: tuple, output_path: str):
    """Generate an ONNX model from a PyTorch model.

    Args:
        model: The PyTorch model to convert.
        input_shape: The shape of the input tensor.
        output_path: The path where the ONNX model will be saved.

    Returns:
        None
    """
    # Create dummy input
    dummy_input = torch.randn(input_shape)

    # Export the model to ONNX
    torch.onnx.export(
        model,  # PyTorch model
        dummy_input,  # Input tensor
        output_path,  # Output file path
        export_params=True,  # Store the trained weights
        opset_version=12,  # ONNX version to use
        input_names=["input"],  # Input tensor names
        output_names=["output"],  # Output tensor names
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    print(f"ONNX model saved to {output_path}")


def create_toy_model_onnx(is_linear=True, output_path=None):
    """Create a ToyTorchModel and export it to ONNX format.

    Args:
        is_linear: Whether to create a linear model or a conv model.
        output_path: Path where the ONNX model will be saved.
            Defaults to "toy_linear.onnx" or "toy_conv.onnx" based on is_linear.

    Returns:
        The path to the saved ONNX model.
    """
    if output_path is None:
        output_path = "toy_linear.onnx" if is_linear else "toy_conv.onnx"

    # Create model instance
    model = ToyTorchModel(is_linear=is_linear)

    # Set model to evaluation mode
    model.eval()

    # Get sample input shape from the model's samples method
    sample_input = model.inputs(batch_sizes=[2])[0]
    input_shape = sample_input.shape

    # Generate ONNX model
    generate_onnx_model(model, input_shape, output_path)

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ONNX models for testing")
    parser.add_argument(
        "--model", type=str, choices=["linear", "conv", "all"], default="all", help="Type of model to generate"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./onnx_models", help="Directory to save the generated ONNX models"
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    if args.model in ["linear", "all"]:
        linear_path = os.path.join(args.output_dir, "toy_linear.onnx")
        create_toy_model_onnx(is_linear=True, output_path=linear_path)

    if args.model in ["conv", "all"]:
        conv_path = os.path.join(args.output_dir, "toy_conv.onnx")
        create_toy_model_onnx(is_linear=False, output_path=conv_path)
