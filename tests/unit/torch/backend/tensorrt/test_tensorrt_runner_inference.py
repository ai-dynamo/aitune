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
"""Unit tests for TensorRT Runner inference functionality with dynamic batch shapes."""

import torch

from aitune.torch.backend.tensorrt.tensorrt_backend import TensorRTBackend, TensorRTBackendConfig
from tests.toy_models.torch_models import ToyTorchModel
from tests.utilities.helpers import requires_cuda

# Constants for testing
IN_FEATURES = 32


@requires_cuda
def test_runner_integration_with_different_batch_sizes(tmp_path):
    """Integration test with real TensorRT for dynamic batch sizes.

    This test performs actual TensorRT inference with different batch sizes
    to verify end-to-end functionality. It requires CUDA and TensorRT to be
    available and is skipped otherwise.
    """
    # Create backend with default configuration for faster builds
    config = TensorRTBackendConfig()
    backend = TensorRTBackend(config=config)

    # Create model with dynamic batch support
    model = ToyTorchModel().to("cuda").eval()
    batch_sizes = [1, 2, 4]
    samples = model.samples(batch_sizes=batch_sizes, device="cuda")
    graph_spec = model.graph_spec(batch_sizes=batch_sizes, device="cuda")

    # Build TensorRT engine with dynamic batch support
    backend = backend.build(model, graph_spec, samples, device=torch.device("cuda"), cache_dir=tmp_path)
    # Note: build() already calls activate() internally

    try:
        # Test inference with different batch sizes
        for batch_size in batch_sizes:
            # Torch model is offloaded to CPU, use host memory for input tensor
            input_tensor = torch.randn(batch_size, IN_FEATURES, device="cpu")

            # Get reference output from PyTorch model
            with torch.no_grad():
                reference_output = model(input_tensor)

            # Perform TensorRT inference
            trt_output = backend.infer(input_tensor).to("cpu")

            # Verify output shape and accuracy
            assert trt_output.shape == reference_output.shape, f"Shape mismatch for batch size {batch_size}"
            assert torch.allclose(trt_output, reference_output, rtol=1e-2, atol=1e-2), (
                f"Output mismatch for batch size {batch_size}"
            )

    finally:
        # Ensure proper cleanup
        backend.deactivate()
