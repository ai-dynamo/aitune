# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Toy models for call-tree tests, one per structural case of the reference models."""

import torch
from torch import nn

HIDDEN = 8


class Block(nn.Module):
    """A repeated transformer-like block."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(HIDDEN, HIDDEN)

    def forward(self, x):
        return torch.relu(self.proj(x))


class Llm(nn.Module):
    """Embedding plus a run of identical blocks (rule 1)."""

    def __init__(self, vocab: int = 16, layers: int = 4):
        super().__init__()
        self.embed = nn.Embedding(vocab, HIDDEN)
        self.layers = nn.ModuleList([Block() for _ in range(layers)])
        # A buffer, like BERT's ``position_ids``: offloading moves it too, and a later ``.to()`` must find data.
        self.register_buffer("scale", torch.ones(HIDDEN))

    def forward(self, hidden):
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden * self.scale


class Dit(nn.Module):
    """Action head with a small time embedder."""

    def __init__(self):
        super().__init__()
        self.t_embedder = nn.Linear(1, HIDDEN)
        self.body = nn.Linear(HIDDEN, HIDDEN)

    def forward(self, x, t):
        return self.body(x + self.t_embedder(t))


class NoWeights(nn.Module):
    """Module without parameters (index math)."""

    def forward(self, x):
        return x * 1.0


class VlaToy(nn.Module):
    """Vision, LLM, tied lm_head and a x5 DiT loop in a method, like Xiaomi-Robotics-0."""

    def __init__(self, steps: int = 5):
        super().__init__()
        self.steps = steps
        self.vision = nn.ModuleList([Block(), Block()])
        self.projector = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, HIDDEN))
        self.llm = Llm()
        self.lm_head = nn.Linear(HIDDEN, 16, bias=False)
        self.lm_head.weight = self.llm.embed.weight
        self.positions = NoWeights()
        self.dit = Dit()

    def forward(self, tokens, image):
        for block in self.vision:
            image = block(image)
        image = self.projector(image)
        hidden = self.llm(self.positions(self.llm.embed(tokens) + image))
        self.lm_head(hidden)
        x = torch.zeros_like(hidden)
        for i in range(self.steps):
            t = torch.full((*x.shape[:-1], 1), float(i), device=x.device, dtype=x.dtype)
            x = self.dit_forward(x + hidden, t)
        return x

    def dit_forward(self, x, t):
        return self.dit(x, t)


def vla_inputs(batch: int = 2):
    """Inputs for ``VlaToy``."""
    return torch.randint(0, 16, (batch,)), torch.randn(batch, HIDDEN)


class TwoRoleToy(nn.Module):
    """One module used from two methods, like BAGEL's LLM for prefill and generation."""

    def __init__(self):
        super().__init__()
        self.llm = Llm(layers=3)

    def forward(self, x):
        return self.generate(self.prefill(x))

    def prefill(self, x):
        return self.llm(x)

    def generate(self, x):
        return self.llm(x)


class GlueToy(nn.Module):
    """The root calls the prefix's embedding outside the prefix, like pi0.5.

    ``Llm.forward`` owns ``embed`` but never uses it; the root runs it before calling the prefix.
    """

    def __init__(self):
        super().__init__()
        self.prefix = Llm(layers=3)

    def forward(self, tokens):
        return self.prefix(self.prefix.embed(tokens))


class Holder(nn.Module):
    """Holds an embedding it never uses; its ``forward`` delegates to one child."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(16, HIDDEN)
        self.body = Llm(layers=2)

    def forward(self, hidden):
        return self.body(hidden)


class DelegatingGlueToy(nn.Module):
    """The root looks up ``holder.embed`` before calling ``holder``, whose ``forward`` runs only ``body``."""

    def __init__(self):
        super().__init__()
        self.holder = Holder()

    def forward(self, tokens):
        return self.holder(self.holder.embed(tokens))


class SequentialHeadToy(nn.Module):
    """A head built as an exact ``nn.Sequential`` (skipped by the recorder) and tied to the embedding."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(16, HIDDEN)
        self.head = nn.Sequential(nn.Linear(HIDDEN, 16, bias=False))
        self.head[0].weight = self.embed.weight

    def forward(self, tokens):
        return self.head(self.embed(tokens))


class SequentialSubclassToy(nn.Sequential):
    """``nn.Sequential`` subclass whose ``encode`` moves itself, like E5's SentenceTransformer."""

    def __init__(self):
        super().__init__(Llm(layers=3))

    def encode(self, x):
        self.to(x.device)
        return self(x)


class PlainPipeline:
    """Plain object root with a method step and a checkpoint helper that runs a callable argument."""

    def __init__(self):
        self.encoder = Llm(layers=3)
        self.decoder = nn.Linear(HIDDEN, HIDDEN)

    def __call__(self, x):
        return self.decode(self._apply_checkpoint(lambda y: self.encoder(y), x))

    def decode(self, x):
        return self.decoder(x)

    def _apply_checkpoint(self, func, x):
        return func(x)


class DirectReadToy(nn.Module):
    """The caller reads the unit's weight directly in its own forward, outside any module call."""

    def __init__(self):
        super().__init__()
        self.unit = Llm(layers=2)

    def forward(self, tokens):
        return self.unit(self.unit.embed(tokens)) @ self.unit.scale.T


class DeviceLookupToy(nn.Module):
    """The caller only asks for a weight's device outside the unit, like Cosmos3-Nano's diffusers pipeline in
    ``_get_execution_device``."""

    def __init__(self):
        super().__init__()
        self.unit = Llm(layers=2)

    def forward(self, tokens):
        _ = next(self.unit.parameters()).device
        return self.unit(self.unit.embed(tokens))
