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
"""Unit tests for TensorRTBackend output formats and tensor handling."""

import pytest
import torch
import torch.nn as nn

from aitune.torch.backend.tensorrt.tensorrt_backend import TensorRTBackend
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.locator import Locator
from aitune.torch.module.sample_metadata import SampleMetadata
from tests.utilities.helpers import requires_cuda

# Test constants
BATCH_SIZE = 2
IN_FEATURES = 32
OUT_FEATURES = 5


class GraphSpecModule(nn.Module):
    """Mixin for models that return a graph spec."""

    def graph_spec(self, samples):
        """Get graph spec for the model."""
        graph_spec = None
        for sample in samples:
            args, kwargs = sample
            outputs = self(*args, **kwargs)
            input_metadata = SampleMetadata.from_inputs(args, kwargs, strict=True)
            output_metadata = SampleMetadata.from_outputs(outputs, strict=True)
            if graph_spec is None:
                graph_spec = GraphSpec(name="toy_model", input_spec=input_metadata, output_spec=output_metadata)
            else:
                graph_spec.update_shapes_seen(input_metadata, output_metadata)
        return graph_spec


class TensorOutputModel(GraphSpecModule):
    """Model that returns a single tensor."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(IN_FEATURES, OUT_FEATURES)

    def forward(self, x):
        return self.linear(x)


class TupleOutputModel(GraphSpecModule):
    """Model that returns a tuple of tensors."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(IN_FEATURES, OUT_FEATURES)
        self.linear2 = nn.Linear(IN_FEATURES, OUT_FEATURES * 2)

    def forward(self, x):
        return self.linear1(x), self.linear2(x)


class ListOutputModel(GraphSpecModule):
    """Model that returns a list of tensors."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(IN_FEATURES, OUT_FEATURES)
        self.linear2 = nn.Linear(IN_FEATURES, OUT_FEATURES * 2)

    def forward(self, x):
        return [self.linear1(x), self.linear2(x)]


class DictOutputModel(GraphSpecModule):
    """Model that returns a dictionary of tensors."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(IN_FEATURES, OUT_FEATURES)
        self.linear2 = nn.Linear(IN_FEATURES, OUT_FEATURES * 2)

    def forward(self, x):
        return {"output1": self.linear1(x), "output2": self.linear2(x)}


class CustomOutput:
    """Custom output class for testing model output handling."""

    def __init__(self, first, tensor1, tensor2, last):
        self.first = first
        self.tensor1 = tensor1
        self.tensor2 = tensor2
        self.last = last

    def __eq__(self, other):
        if not isinstance(other, CustomOutput):
            return False
        return (
            self.tensor1.shape == other.tensor1.shape
            and self.tensor2.shape == other.tensor2.shape
            and self.first == other.first
            and self.last == other.last
        )


class CustomOutputModel(GraphSpecModule):
    """Model that returns a custom object with tensor attributes."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(IN_FEATURES, OUT_FEATURES)
        self.linear2 = nn.Linear(IN_FEATURES, OUT_FEATURES * 2)

    def forward(self, first, x, last):
        return CustomOutput(first, self.linear1(x), self.linear2(x), last)


@requires_cuda
def test_tensor_output(tmp_path):
    """Integration test for single tensor output format."""
    # Create model and data
    device = torch.device("cuda")
    model = TensorOutputModel().to(device).eval()
    test_tensor = torch.randn(BATCH_SIZE, IN_FEATURES, device=device)
    samples = [((test_tensor,), {})]
    graph_spec = model.graph_spec(samples=samples)

    # Build with TensorRT
    backend = TensorRTBackend()
    backend = backend.build(module=model, graph_spec=graph_spec, data=samples, device=device, cache_dir=tmp_path)
    # Note: build() already calls activate() internally

    # Verify output is a tensor
    output = backend.infer(test_tensor)
    assert isinstance(output, torch.Tensor), "Output should be a tensor"
    assert output.shape == (BATCH_SIZE, OUT_FEATURES), "Output shape incorrect"

    backend.deactivate()


@requires_cuda
def test_tuple_output(tmp_path):
    """Integration test for tuple output format."""
    # Create model and data
    device = torch.device("cuda")
    model = TupleOutputModel().to(device).eval()
    test_tensor = torch.randn(BATCH_SIZE, IN_FEATURES, device=device)
    samples = [((test_tensor,), {})]
    graph_spec = model.graph_spec(samples=samples)

    # Build with TensorRT
    backend = TensorRTBackend()
    backend = backend.build(module=model, graph_spec=graph_spec, data=samples, device=device, cache_dir=tmp_path)
    # Note: build() already calls activate() internally

    # Verify output is a tuple
    output = backend.infer(test_tensor)
    try:
        assert isinstance(output, tuple), "Output should be a tuple"
        assert len(output) == 2, "Output should have 2 tensors"
        assert output[0].shape == (BATCH_SIZE, OUT_FEATURES), "First output shape incorrect"
        assert output[1].shape == (BATCH_SIZE, OUT_FEATURES * 2), "Second output shape incorrect"
    finally:
        backend.deactivate()


@requires_cuda
def test_list_output(tmp_path):
    """Integration test for list output format."""
    # Create model and data
    device = torch.device("cuda")
    model = ListOutputModel().to(device).eval()
    test_tensor = torch.randn(BATCH_SIZE, IN_FEATURES, device=device)
    samples = [((test_tensor,), {})]
    graph_spec = model.graph_spec(samples=samples)

    # Build with TensorRT
    backend = TensorRTBackend()
    backend = backend.build(module=model, graph_spec=graph_spec, data=samples, device=device, cache_dir=tmp_path)
    # Note: build() already calls activate() internally

    # Verify output is a list
    output = backend.infer(test_tensor)
    try:
        assert isinstance(output, list), "Output should be a list"
        assert len(output) == 2, "Output should have 2 tensors"
        assert output[0].shape == (BATCH_SIZE, OUT_FEATURES), "First output shape incorrect"
        assert output[1].shape == (BATCH_SIZE, OUT_FEATURES * 2), "Second output shape incorrect"
    finally:
        backend.deactivate()


@requires_cuda
def test_dict_output(tmp_path):
    """Integration test for dictionary output format."""
    # Create model and data
    device = torch.device("cuda")
    model = DictOutputModel().to(device).eval()
    test_tensor = torch.randn(BATCH_SIZE, IN_FEATURES, device=device)
    samples = [((test_tensor,), {})]
    graph_spec = model.graph_spec(samples=samples)

    # Build with TensorRT
    backend = TensorRTBackend()
    backend = backend.build(module=model, graph_spec=graph_spec, data=samples, device=device, cache_dir=tmp_path)
    # Note: build() already calls activate() internally

    # Verify output is a dictionary
    output = backend.infer(test_tensor)
    try:
        assert isinstance(output, dict), "Output should be a dictionary"
        assert len(output) == 2, "Output should have 2 tensors"
        assert "output1" in output, "Missing output1 key"
        assert "output2" in output, "Missing output2 key"
        assert output["output1"].shape == (BATCH_SIZE, OUT_FEATURES), "First output shape incorrect"
        assert output["output2"].shape == (BATCH_SIZE, OUT_FEATURES * 2), "Second output shape incorrect"
    finally:
        backend.deactivate()


@requires_cuda
def test_non_contiguous_input(tmp_path):
    """Integration test with non-contiguous input tensor."""
    # Create model and data
    device = torch.device("cuda")
    model = TensorOutputModel().to(device).eval()
    contiguous_tensor = torch.randn(BATCH_SIZE, IN_FEATURES, device=device)
    samples = [((contiguous_tensor,), {})]
    graph_spec = model.graph_spec(samples=samples)

    # Create a non-contiguous tensor with the correct dimensions
    # Create a tensor with swapped dimensions and transpose to get the right shape but non-contiguous
    non_contiguous_tensor = torch.randn(IN_FEATURES, BATCH_SIZE, device="cuda").transpose(0, 1)
    assert not non_contiguous_tensor.is_contiguous(), "Test tensor should be non-contiguous"
    assert non_contiguous_tensor.shape == (BATCH_SIZE, IN_FEATURES), "Non-contiguous tensor should have correct shape"

    # Build with TensorRT
    backend = TensorRTBackend()
    backend = backend.build(
        module=model,
        graph_spec=graph_spec,
        data=samples,
        device=device,
        cache_dir=tmp_path,
    )
    # Note: build() already calls activate() internally

    # Infer with non-contiguous tensor
    output = backend.infer(non_contiguous_tensor)
    try:
        # Verify output shape is correct
        assert output.shape == (BATCH_SIZE, OUT_FEATURES), "Output shape incorrect"
        assert output.is_cuda, "Output should be on CUDA"
    finally:
        backend.deactivate()


@requires_cuda
def test_cpu_input(tmp_path):
    """Integration test with CPU input tensor that gets moved to CUDA."""
    # Create model and data
    device = torch.device("cuda")
    model = TensorOutputModel().to(device).eval()
    cuda_tensor = torch.randn(BATCH_SIZE, IN_FEATURES, device=device)
    samples = [((cuda_tensor,), {})]
    graph_spec = model.graph_spec(samples=samples)

    # Create a CPU tensor with same values
    cpu_tensor = cuda_tensor.cpu()
    assert not cpu_tensor.is_cuda, "Test tensor should be on CPU"

    # Build with TensorRT
    backend = TensorRTBackend()
    backend = backend.build(module=model, graph_spec=graph_spec, data=samples, device=device, cache_dir=tmp_path)
    # Note: build() already calls activate() internally

    # Infer with CPU tensor (should be automatically moved to CUDA)
    output = backend.infer(cpu_tensor)
    try:
        # Verify output shape is correct
        assert output.shape == (BATCH_SIZE, OUT_FEATURES), "Output shape incorrect"
        assert output.is_cuda, "Output should be on CUDA"
    finally:
        backend.deactivate()


@pytest.mark.skip(reason="torch.onnx.export does not support custom object output")
@requires_cuda
def test_custom_object_output(tmp_path):
    """Integration test with custom object output."""
    Locator.register_user_type(CustomOutput)
    # Create model and data
    device = torch.device("cuda")
    model = CustomOutputModel().to(device).eval()
    cuda_tensor = torch.randn(BATCH_SIZE, IN_FEATURES, device=device)
    samples = [
        (
            (True, cuda_tensor, False),
            {},
        )
    ]
    graph_spec = model.graph_spec(samples=samples)

    # Create a CPU tensor with same values
    cpu_tensor = cuda_tensor.cpu()
    assert not cpu_tensor.is_cuda, "Test tensor should be on CPU"

    # Build with TensorRT
    backend = TensorRTBackend()
    backend = backend.build(module=model, graph_spec=graph_spec, data=samples, device=device, cache_dir=tmp_path)
    # Note: build() already calls activate() internally

    # Infer with CPU tensor (should be automatically moved to CUDA)
    output = backend.infer(cpu_tensor)

    # Verify output shape is correct
    assert output.tensor1 == (BATCH_SIZE, OUT_FEATURES), "Output shape incorrect"
    assert output.tensor2 == (BATCH_SIZE, OUT_FEATURES), "Output shape incorrect"
    assert output.first == "abc", "First output incorrect"
    assert output.last == "xyz", "Last output incorrect"

    backend.deactivate()


# TBD:: add test for nested tensors
