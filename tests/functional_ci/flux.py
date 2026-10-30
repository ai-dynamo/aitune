# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shell of a Flux pipeline for functional tests.

Same component names and call shape as Flux: two text encoders, a transformer,
and a VAE decoder. Each module is a small linear map.
"""

from collections.abc import Callable

import torch
from torch import nn

from tests.functional_ci.models import seeded

DIM = 32
TEXT_LEN = 8
VOCAB = 128
LATENT_CHANNELS = 4


def tokenize(prompts: list[str]) -> torch.Tensor:
    """Encode prompt bytes into fixed-length token values in [0, 1)."""
    ids = torch.zeros((len(prompts), TEXT_LEN))
    for row, prompt in enumerate(prompts):
        values = [byte % VOCAB for byte in prompt.encode("utf-8")[:TEXT_LEN]]
        if values:
            ids[row, : len(values)] = torch.tensor(values, dtype=torch.float32) / VOCAB
    return ids


class ToyClipTextEncoder(nn.Module):
    """Map a token row to one vector."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(TEXT_LEN, DIM)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.proj(tokens)


class ToyT5Encoder(nn.Module):
    """Map a token row to one vector per position."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(TEXT_LEN, DIM)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # in_features must be >1 so int8 weight-only quantization can scale the weight row.
        return self.proj(tokens).unsqueeze(1) + tokens.unsqueeze(-1)


class ToyFluxTransformer(nn.Module):
    """Predict image-token velocity from text, pooled text, and timestep."""

    def __init__(self) -> None:
        super().__init__()
        self.image_in = nn.Linear(LATENT_CHANNELS, DIM)
        self.out = nn.Linear(DIM, LATENT_CHANNELS)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        context = encoder_hidden_states.mean(dim=1) + pooled_projections + timestep[:, None]
        return (self.out(self.image_in(hidden_states) + context[:, None, :]),)


class ToyDecoder(nn.Module):
    """Map latent channels to RGB."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(LATENT_CHANNELS, 3, kernel_size=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.conv(value)


class ToyVae(nn.Module):
    """Hold the decoder. ``forward`` stays unused so inspection selects ``decoder``."""

    def __init__(self) -> None:
        super().__init__()
        self.decoder = ToyDecoder()

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to an image."""
        return self.decoder(latents)


class ToyFluxPipeline:
    """Run the four shell components once."""

    def __init__(self, device: torch.device, seed: int = 0, dtype: torch.dtype = torch.float32) -> None:
        with seeded(device, seed):
            self.text_encoder = ToyClipTextEncoder()
            self.text_encoder_2 = ToyT5Encoder()
            self.transformer: Callable[..., tuple[torch.Tensor]] = ToyFluxTransformer()
            self.vae = ToyVae()
        self.device = device
        self.dtype = dtype
        for component in (self.text_encoder, self.text_encoder_2, self.transformer, self.vae):
            component.to(device=device, dtype=dtype).eval()

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str],
        num_inference_steps: int = 1,
        height: int = 16,
        width: int = 16,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        prompts = [prompt] if isinstance(prompt, str) else prompt
        tokens = tokenize(prompts).to(device=self.device, dtype=self.dtype)
        pooled = self.text_encoder(tokens)
        text = self.text_encoder_2(tokens)
        grid_height, grid_width = height // 4, width // 4
        latents = torch.randn(
            len(prompts),
            grid_height * grid_width,
            LATENT_CHANNELS,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        timestep = torch.full((len(prompts),), 1.0 / num_inference_steps, device=self.device, dtype=self.dtype)
        (velocity,) = self.transformer(latents, text, pooled, timestep)
        latents = (latents + velocity).transpose(1, 2).reshape(len(prompts), LATENT_CHANNELS, grid_height, grid_width)
        return self.vae.decode(latents).clamp(0, 1)
