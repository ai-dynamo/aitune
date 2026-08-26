# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Toy models shared by backend export and capability tests."""

from typing import Any

import torch
import torch.nn as nn

from aitune.torch.module.forward_signature import ForwardSignature
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_metadata import SampleMetadata
from aitune.torch.module.sample_store import Sample
from tests.toy_models.torch_models import HIDDEN_SIZE, INPUT_SIZE, OUTPUT_SIZE, ToyTorchModel


class ExportModelMixin:
    """Generate static or dynamic samples and their corresponding graph specification."""

    def samples(self, batch_sizes: list[int] | None = None, device: str | torch.device = "cpu") -> list[Sample]:
        """Return representative samples for the requested batch sizes."""
        raise NotImplementedError

    def graph_spec(self, batch_sizes: list[int] | None = None, device: str | torch.device = "cpu") -> GraphSpec:
        """Build a graph specification from the requested samples."""
        batch_sizes = batch_sizes or [2]
        samples = self.samples(batch_sizes=batch_sizes, device=device)
        forward_signature = ForwardSignature.from_callable(self.forward)  # type: ignore[attr-defined]
        graph_spec = None

        for batch_size, (args, kwargs) in zip(batch_sizes, samples, strict=True):
            output = self(*args, **kwargs)  # type: ignore[operator]
            forward_inputs = forward_signature.normalize(args, kwargs)
            input_metadata = SampleMetadata.from_inputs(forward_inputs.arguments, batch_size=batch_size)
            output_metadata = SampleMetadata.from_outputs(output, batch_size=batch_size)
            if graph_spec is None:
                graph_spec = GraphSpec(
                    name=self.__class__.__name__,
                    input_spec=input_metadata,
                    output_spec=output_metadata,
                    forward_signature=forward_signature,
                )
            else:
                graph_spec.update_shapes_seen(input_metadata, output_metadata)

        if graph_spec is None:
            raise ValueError("At least one batch size is required.")
        return graph_spec


class ToyPipelineModel(ExportModelMixin, nn.Module):
    """Pipeline-like composition with preprocessing, a core module, and postprocessing."""

    def __init__(self):
        super().__init__()
        self.preprocessor = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.core = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.postprocessor = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

    def forward(self, sample: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """Run the pipeline with a positional sample and keyword conditioning."""
        hidden = torch.relu(self.preprocessor(sample))
        hidden = torch.tanh(self.core(hidden) + conditioning)
        return self.postprocessor(hidden)

    def samples(self, batch_sizes=None, device="cpu") -> list[Sample]:
        """Return pipeline samples with mixed positional and keyword inputs."""
        batch_sizes = batch_sizes or [2]
        return [
            (
                (torch.randn(batch_size, INPUT_SIZE, device=device),),
                {"conditioning": torch.randn(batch_size, HIDDEN_SIZE, device=device)},
            )
            for batch_size in batch_sizes
        ]


class ToyNestedInputModel(ExportModelMixin, nn.Module):
    """Model with tensors and static leaves spread across nested containers."""

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(INPUT_SIZE, OUTPUT_SIZE)

    def forward(
        self,
        x: torch.Tensor,
        cache: list[Any],
        options: dict[str, Any],
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """Consume a list input and a nested dictionary passed by keyword."""
        if options["metadata"]["mode"] != "add":
            raise ValueError("Unsupported mode")
        hidden = x + cache[0] + options["residuals"][0]
        hidden = hidden * scale
        return self.projection(hidden)

    def samples(self, batch_sizes=None, device="cpu") -> list[Sample]:
        """Return complex samples preserving list, dictionary, ``None``, and string leaves."""
        batch_sizes = batch_sizes or [2]
        samples = []
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, INPUT_SIZE, device=device)
            samples.append((
                (x, [torch.randn(batch_size, INPUT_SIZE, device=device), None]),
                {
                    "options": {
                        "residuals": [torch.randn(batch_size, INPUT_SIZE, device=device), None],
                        "metadata": {"mode": "add"},
                    },
                    "scale": torch.ones(batch_size, 1, device=device),
                },
            ))
        return samples


TOY_EXPORT_MODELS = (
    ("simple", ToyTorchModel),
    ("pipeline", ToyPipelineModel),
    ("complex", ToyNestedInputModel),
)

__all__ = ["TOY_EXPORT_MODELS", "ExportModelMixin", "ToyNestedInputModel", "ToyPipelineModel"]
