# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Toy model for unit tests."""

import torch

from aitune.torch.module.forward_signature import ForwardSignature
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.recording_module import Sample
from aitune.torch.module.sample_metadata import SampleMetadata

INPUT_SIZE = 32  # must be multiple of 16 for torchao tests
HIDDEN_SIZE = 16
FILTER_SIZE = 129
OUTPUT_SIZE = 5
KERNEL_SIZE = 5
CHANNELS = 1


class ToyTorchModel(torch.nn.Module):
    """Simple toy model."""

    def __init__(self, is_linear: bool = True):
        super().__init__()
        self.is_linear = is_linear
        if self.is_linear:
            self.linear1 = torch.nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
            self.linear2 = torch.nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
        else:
            self.conv1 = torch.nn.Conv2d(CHANNELS, FILTER_SIZE, KERNEL_SIZE)

    def forward(self, x):
        if self.is_linear:
            x = torch.relu(self.linear1(x))
            x = torch.relu(self.linear2(x))
        else:
            x = torch.relu(self.conv1(x))
        return x

    def sample(self, device: str = "cpu"):
        input_shape = (INPUT_SIZE) if self.is_linear else (CHANNELS, FILTER_SIZE, FILTER_SIZE)
        input_tensor = torch.rand(input_shape, dtype=torch.float32, device=device)
        return input_tensor

    def inputs(self, batch_sizes: list[int] | None = None, device: str = "cpu"):
        if batch_sizes is None:
            batch_sizes = [2]
        return [self._input_tensor(batch_size, device) for batch_size in batch_sizes]

    def samples(self, batch_sizes: list[int] | None = None, device: str = "cpu") -> list[Sample]:
        if batch_sizes is None:
            batch_sizes = [2]
        return [((self._input_tensor(batch_size, device),), {}) for batch_size in batch_sizes]

    def graph_spec(self, batch_sizes: list[int] | None = None, device: str = "cpu") -> GraphSpec:
        graph_spec = None
        forward_signature = ForwardSignature.from_callable(self.forward)
        for sample in self.samples(batch_sizes=batch_sizes, device=device):
            args, kwargs = sample

            _output = self(*args, **kwargs)
            forward_inputs = forward_signature.normalize(args, kwargs)
            input_metadata = SampleMetadata.from_inputs(forward_inputs.arguments, batch_size=args[0].shape[0])
            output_metadata = SampleMetadata.from_outputs(_output, batch_size=args[0].shape[0])
            if graph_spec is None:
                graph_spec = GraphSpec(
                    name="toy_model",
                    input_spec=input_metadata,
                    output_spec=output_metadata,
                    forward_signature=forward_signature,
                )
            else:
                graph_spec.update_shapes_seen(input_metadata, output_metadata)

        if not graph_spec:
            raise ValueError("Graph spec is None. This should not happen.")

        return graph_spec

    def _input_tensor(self, batch_size: int = 2, device: str = "cpu"):
        input_shape = (batch_size, INPUT_SIZE) if self.is_linear else (batch_size, CHANNELS, FILTER_SIZE, FILTER_SIZE)
        input_tensor = torch.rand(input_shape, dtype=torch.float32, device=device)
        return input_tensor


class ToyTorchConditionalModel(torch.nn.Module):
    """Simple toy conditional model.

    Contains kwarg in forward so that computational graph is diverging.
    """

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(INPUT_SIZE, OUTPUT_SIZE)

    def forward(self, x, apply_relu=True):
        x = self.linear(x)
        if apply_relu:
            x = torch.relu(x)
        return x

    def samples(
        self, batch_sizes: list[int] | None = None, device: str = "cpu", kwargs: dict | None = None
    ) -> list[Sample]:
        if batch_sizes is None:
            batch_sizes = [2]
        if kwargs is None:
            kwargs = {}
        return [((self._input_tensor(batch_size, device),), kwargs) for batch_size in batch_sizes]

    def inputs(self, batch_sizes: list[int] | None = None, device: str = "cpu"):
        if batch_sizes is None:
            batch_sizes = [2]
        return [self._input_tensor(batch_size, device) for batch_size in batch_sizes]

    def graph_spec(
        self, batch_sizes: list[int] | None = None, device: str = "cpu", kwargs: dict | None = None
    ) -> GraphSpec:
        graph_spec = None
        forward_signature = ForwardSignature.from_callable(self.forward)
        for sample in self.samples(batch_sizes=batch_sizes, device=device, kwargs=kwargs):
            args, kwargs = sample

            _output = self(*args, **kwargs)
            forward_inputs = forward_signature.normalize(args, kwargs)
            input_metadata = SampleMetadata.from_inputs(forward_inputs.arguments, batch_size=args[0].shape[0])
            output_metadata = SampleMetadata.from_outputs(_output, batch_size=args[0].shape[0])
            if graph_spec is None:
                graph_spec = GraphSpec(
                    name="toy_model",
                    input_spec=input_metadata,
                    output_spec=output_metadata,
                    forward_signature=forward_signature,
                )
            else:
                graph_spec.update_shapes_seen(input_metadata, output_metadata)

        if not graph_spec:
            raise ValueError("Graph spec is None. This should not happen.")

        return graph_spec

    def _input_tensor(self, batch_size: int = 2, device: str = "cpu"):
        input_shape = (batch_size, INPUT_SIZE)
        input_tensor = torch.rand(input_shape, dtype=torch.float32, device=device)
        return input_tensor


class ToyPipeline:
    """Toy pipeline for unit tests.

    Does not inherit from torch.nn.Module.
    """

    def __init__(self):
        self.linear1 = torch.nn.Linear(INPUT_SIZE, HIDDEN_SIZE).to("cuda")
        self.linear2 = torch.nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE).to("cuda")
        self.preprocessing = lambda x: x.to("cuda")
        self.config = {"description": "this is a dummy configuration just to imitate a pipeline"}

    def __call__(self, x):
        x = self.preprocessing(x)
        out = self.linear1(x)
        out = self.linear2(out)
        return out

    def to(self, device):
        self.linear1.to(device)
        self.linear2.to(device)
        return self


class ToyComplexPipeline:
    """Toy pipeline for unit tests.

    The pipeline is a composition of a linear model, pre/post processing, and a dummy configuration.

    It resembles HF pipelines where there could be tokenizers, model, post-processors, configuration etc.
    """

    def __init__(self):
        self.net = ToyTorchModel(is_linear=True)
        self.preprocessor = lambda x: 2 * x
        self.postprocessor = lambda x: x + 1
        self.config = {"description": "this is a dummy configuration just to imitate a pipeline configuration"}

    def __call__(self, x):
        x = self.preprocessor(x)
        x = self.net(x)
        x = self.postprocessor(x)
        return x

    def to(self, device):
        self.net.to(device)
        return self

    def inputs(self, batch_sizes: list[int] | None = None, device: str = "cpu"):
        return self.net.inputs(batch_sizes, device)
