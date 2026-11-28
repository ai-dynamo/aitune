# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decorated child forwards compiled after a preprocessing graph break, as in NeMo ASR."""

import torch
import wrapt


def _checks_enabled():
    return False


@wrapt.decorator(enabled=_checks_enabled)
def _checked_forward(wrapped, instance, args, kwargs):
    # NeMo retains its wrapt descriptor when type checking is disabled at runtime.
    return wrapped(*args, **kwargs)


class ToyDecoratedEncoder(torch.nn.Module):
    """A weighted encoder with a decorated forward and a default argument."""

    def __init__(self, seed: int = 0):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.weight = torch.nn.Parameter(torch.randn(8, 8, generator=generator) * 0.1)

    # Keep the wrapt proxy so this fixture exercises NeMo's runtime descriptor.
    @_checked_forward
    def forward(self, audio_signal, scale=2.0):  # pyrefly: ignore[bad-override]
        return torch.nn.functional.linear(audio_signal, self.weight) * scale


class ToyDecoratedSpeechModel(torch.nn.Module):
    """Preprocess inputs before calling an encoder through its decorated descriptor."""

    def __init__(self, seed: int = 0):
        super().__init__()
        self.encoder = ToyDecoratedEncoder(seed=seed)

    def forward(self, audio_signal):
        processed_signal = audio_signal + 1
        # NeMo preprocessing breaks the graph before the decorated encoder call.
        # Keep that boundary explicit so scalar-capture settings cannot remove it.
        torch._dynamo.graph_break()
        return self.encoder(audio_signal=processed_signal)
