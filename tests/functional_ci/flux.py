# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small plain-torch FLUX components for functional tests."""

import math
from collections.abc import Callable
from typing import cast

import torch
from torch import nn
from torch.nn import functional

from tests.functional_ci.models import seeded

VOCAB = 128
TEXT_LEN = 8
DIM = 32
HEADS = 2
AXES_DIMS = (4, 4, 8)
LATENT_CHANNELS = 4
PACKED = LATENT_CHANNELS * 4
HEAD_DIM = DIM // HEADS


def tokenize(prompts: list[str]) -> torch.LongTensor:
    """Encode prompt bytes into fixed-length token IDs."""
    ids = torch.zeros((len(prompts), TEXT_LEN), dtype=torch.long)
    for row, prompt in enumerate(prompts):
        values = [byte % VOCAB for byte in prompt.encode("utf-8")[:TEXT_LEN]]
        if values:
            ids[row, : len(values)] = torch.tensor(values)
    return cast(torch.LongTensor, ids)


def _split_heads(value: torch.Tensor) -> torch.Tensor:
    """Split the feature dimension into attention heads."""
    batch, sequence, _ = value.shape
    return value.reshape(batch, sequence, HEADS, HEAD_DIM).transpose(1, 2)


def _merge_heads(value: torch.Tensor) -> torch.Tensor:
    """Merge attention heads into the feature dimension."""
    batch, _, sequence, _ = value.shape
    return value.transpose(1, 2).reshape(batch, sequence, DIM)


def _modulate(norm: nn.Module, value: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive shift and scale after normalization."""
    return norm(value) * (1 + scale[:, None, :]) + shift[:, None, :]


class _AttentionBlock(nn.Module):
    """Pre-normalized attention and MLP block."""

    def __init__(self, causal: bool):
        super().__init__()
        self.causal = causal
        self.norm1 = nn.LayerNorm(DIM)
        self.qkv = nn.Linear(DIM, 3 * DIM)
        self.out = nn.Linear(DIM, DIM)
        self.norm2 = nn.LayerNorm(DIM)
        self.mlp = nn.Sequential(nn.Linear(DIM, 4 * DIM), nn.GELU(), nn.Linear(4 * DIM, DIM))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        query, key, item = self.qkv(self.norm1(value)).chunk(3, dim=-1)
        attended = functional.scaled_dot_product_attention(
            _split_heads(query), _split_heads(key), _split_heads(item), is_causal=self.causal
        )
        value = value + self.out(_merge_heads(attended))
        return value + self.mlp(self.norm2(value))


class ToyClipTextEncoder(nn.Module):
    """Encode token IDs into CLIP-style pooled text features."""

    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB, DIM)
        self.position_embedding = nn.Embedding(TEXT_LEN, DIM)
        self.block = _AttentionBlock(causal=True)
        self.final_norm = nn.LayerNorm(DIM)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        positions = torch.arange(TEXT_LEN, device=input_ids.device)
        hidden_states = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        return self.final_norm(self.block(hidden_states))[:, -1]


class ToyT5Encoder(nn.Module):
    """Encode token IDs into T5-style sequence features."""

    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB, DIM)
        self.block = _AttentionBlock(causal=False)
        self.final_norm = nn.LayerNorm(DIM)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.final_norm(self.block(self.token_embedding(input_ids)))


def rope(ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build rotary cosine and sine embeddings from three-axis token IDs."""
    angles = []
    for axis, axis_dim in enumerate(AXES_DIMS):
        exponents = torch.arange(0, axis_dim, 2, device=ids.device, dtype=torch.float32) / axis_dim
        frequencies = 1 / (10000**exponents)
        angles.append(ids[:, axis, None] * frequencies[None, :])
    all_angles = torch.cat(angles, dim=-1)
    return all_angles.cos(), all_angles.sin()


def apply_rope(value: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent feature pairs using rotary embeddings."""
    pairs = value.reshape(*value.shape[:-1], HEAD_DIM // 2, 2)
    even, odd = pairs.unbind(dim=-1)
    cos = cos.to(dtype=value.dtype)[None, None, :, :]
    sin = sin.to(dtype=value.dtype)[None, None, :, :]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


def timestep_embedding(timestep: torch.Tensor, dim: int) -> torch.Tensor:
    """Build a Flux-style sinusoidal embedding for normalized timesteps."""
    half_dim = dim // 2
    exponents = -math.log(10000) * torch.arange(half_dim, device=timestep.device, dtype=torch.float32) / half_dim
    angles = timestep[:, None] * 1000 * exponents.exp()[None, :]
    return torch.cat((angles.cos(), angles.sin()), dim=-1).to(dtype=timestep.dtype)


class _DoubleStreamBlock(nn.Module):
    """Attend jointly over text and image tokens, then update each stream."""

    def __init__(self):
        super().__init__()
        self.txt_mod = nn.Linear(DIM, 6 * DIM)
        self.img_mod = nn.Linear(DIM, 6 * DIM)
        self.txt_norm1 = nn.LayerNorm(DIM, elementwise_affine=False)
        self.img_norm1 = nn.LayerNorm(DIM, elementwise_affine=False)
        self.txt_qkv = nn.Linear(DIM, 3 * DIM)
        self.img_qkv = nn.Linear(DIM, 3 * DIM)
        self.txt_attn_out = nn.Linear(DIM, DIM)
        self.img_attn_out = nn.Linear(DIM, DIM)
        self.txt_norm2 = nn.LayerNorm(DIM, elementwise_affine=False)
        self.img_norm2 = nn.LayerNorm(DIM, elementwise_affine=False)
        self.txt_mlp = nn.Sequential(nn.Linear(DIM, 4 * DIM), nn.GELU(), nn.Linear(4 * DIM, DIM))
        self.img_mlp = nn.Sequential(nn.Linear(DIM, 4 * DIM), nn.GELU(), nn.Linear(4 * DIM, DIM))

    def forward(
        self,
        text: torch.Tensor,
        image: torch.Tensor,
        temb: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        txt_shift_attn, txt_scale_attn, txt_gate_attn, txt_shift_mlp, txt_scale_mlp, txt_gate_mlp = self.txt_mod(
            functional.silu(temb)
        ).chunk(6, dim=-1)
        img_shift_attn, img_scale_attn, img_gate_attn, img_shift_mlp, img_scale_mlp, img_gate_mlp = self.img_mod(
            functional.silu(temb)
        ).chunk(6, dim=-1)
        txt_norm = _modulate(self.txt_norm1, text, txt_shift_attn, txt_scale_attn)
        img_norm = _modulate(self.img_norm1, image, img_shift_attn, img_scale_attn)
        txt_query, txt_key, txt_value = (_split_heads(part) for part in self.txt_qkv(txt_norm).chunk(3, dim=-1))
        img_query, img_key, img_value = (_split_heads(part) for part in self.img_qkv(img_norm).chunk(3, dim=-1))
        query = apply_rope(torch.cat((txt_query, img_query), dim=2), cos, sin)
        key = apply_rope(torch.cat((txt_key, img_key), dim=2), cos, sin)
        value = torch.cat((txt_value, img_value), dim=2)
        attended = _merge_heads(functional.scaled_dot_product_attention(query, key, value))
        text_attn, image_attn = attended.split((text.shape[1], image.shape[1]), dim=1)
        text = text + txt_gate_attn[:, None, :] * self.txt_attn_out(text_attn)
        image = image + img_gate_attn[:, None, :] * self.img_attn_out(image_attn)
        text = text + txt_gate_mlp[:, None, :] * self.txt_mlp(
            _modulate(self.txt_norm2, text, txt_shift_mlp, txt_scale_mlp)
        )
        image = image + img_gate_mlp[:, None, :] * self.img_mlp(
            _modulate(self.img_norm2, image, img_shift_mlp, img_scale_mlp)
        )
        return text, image


class _SingleStreamBlock(nn.Module):
    """Apply joint rotary attention and a parallel MLP to both streams."""

    def __init__(self):
        super().__init__()
        self.mod = nn.Linear(DIM, 3 * DIM)
        self.norm = nn.LayerNorm(DIM, elementwise_affine=False)
        self.qkv = nn.Linear(DIM, 3 * DIM)
        self.mlp = nn.Sequential(nn.Linear(DIM, 4 * DIM), nn.GELU())
        self.out = nn.Linear(5 * DIM, DIM)

    def forward(self, value: torch.Tensor, temb: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.mod(functional.silu(temb)).chunk(3, dim=-1)
        normalized = _modulate(self.norm, value, shift, scale)
        query, key, item = (_split_heads(part) for part in self.qkv(normalized).chunk(3, dim=-1))
        attended = functional.scaled_dot_product_attention(apply_rope(query, cos, sin), apply_rope(key, cos, sin), item)
        merged = _merge_heads(attended)
        return value + gate[:, None, :] * self.out(torch.cat((merged, self.mlp(normalized)), dim=-1))


class ToyFluxTransformer(nn.Module):
    """Transform packed image latents with joint text and image attention."""

    def __init__(self):
        super().__init__()
        self.x_embedder = nn.Linear(PACKED, DIM)
        self.context_embedder = nn.Linear(DIM, DIM)
        self.time_mlp = nn.Sequential(nn.Linear(DIM, DIM), nn.SiLU(), nn.Linear(DIM, DIM))
        self.pooled_mlp = nn.Sequential(nn.Linear(DIM, DIM), nn.SiLU(), nn.Linear(DIM, DIM))
        self.guidance_mlp = nn.Sequential(nn.Linear(DIM, DIM), nn.SiLU(), nn.Linear(DIM, DIM))
        self.double_stream = _DoubleStreamBlock()
        self.single_stream = _SingleStreamBlock()
        self.final_mod = nn.Linear(DIM, 2 * DIM)
        self.norm_out = nn.LayerNorm(DIM, elementwise_affine=False)
        self.proj_out = nn.Linear(DIM, PACKED)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        timestep: torch.Tensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
        guidance: torch.Tensor | None = None,
        return_dict: bool = False,
    ) -> tuple[torch.Tensor]:
        if return_dict:
            raise ValueError("ToyFluxTransformer only supports return_dict=False")
        temb = self.time_mlp(timestep_embedding(timestep, DIM)) + self.pooled_mlp(pooled_projections)
        if guidance is not None:
            temb = temb + self.guidance_mlp(timestep_embedding(guidance, DIM))
        cos, sin = rope(torch.cat((txt_ids, img_ids), dim=0))
        text = self.context_embedder(encoder_hidden_states)
        image = self.x_embedder(hidden_states)
        text, image = self.double_stream(text, image, temb, cos, sin)
        joint = self.single_stream(torch.cat((text, image), dim=1), temb, cos, sin)
        image = joint[:, text.shape[1] :]
        shift, scale = self.final_mod(functional.silu(temb)).chunk(2, dim=-1)
        sample = self.proj_out(_modulate(self.norm_out, image, shift, scale))
        return (sample,)


class ToyDecoder(nn.Module):
    """Decode four-channel latents into RGB images at twice the spatial size."""

    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(LATENT_CHANNELS, 16, 3, padding=1)
        self.norm1 = nn.GroupNorm(4, 16)
        self.conv_mid = nn.Conv2d(16, 16, 3, padding=1)
        self.norm2 = nn.GroupNorm(4, 16)
        self.conv_out = nn.Conv2d(16, 3, 3, padding=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = functional.silu(self.norm1(self.conv_in(value)))
        value = functional.interpolate(value, scale_factor=2, mode="nearest")
        value = functional.silu(self.norm2(self.conv_mid(value)))
        return self.conv_out(value)


class ToyVae(nn.Module):
    """Hold the decoder and Flux latent scaling parameters."""

    def __init__(self):
        super().__init__()
        self.decoder = ToyDecoder()
        self.scaling_factor = 1.5035
        self.shift_factor = 0.0609

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode scaled latents to an image."""
        return self.decoder(latents / self.scaling_factor + self.shift_factor)


class ToyFluxPipeline:
    """Run seeded toy text encoders, Flux transformer, and VAE decoder."""

    def __init__(self, device: torch.device, seed: int = 0, dtype: torch.dtype = torch.float32):
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
        num_inference_steps: int = 2,
        height: int = 16,
        width: int = 16,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        prompts = [prompt] if isinstance(prompt, str) else prompt
        ids = tokenize(prompts).to(self.device)
        pooled = self.text_encoder(ids)
        text = self.text_encoder_2(ids)
        txt_ids = torch.zeros((TEXT_LEN, 3), device=self.device)
        latent_height, latent_width = height // 2, width // 2
        grid_height, grid_width = latent_height // 2, latent_width // 2
        img_ids = torch.zeros((grid_height * grid_width, 3), device=self.device)
        for row in range(grid_height):
            for col in range(grid_width):
                img_ids[row * grid_width + col] = torch.tensor((0, row, col), device=self.device)
        latents = torch.randn(
            len(prompts), img_ids.shape[0], PACKED, generator=generator, device=self.device, dtype=self.dtype
        )
        sigmas = torch.linspace(1, 0, num_inference_steps + 1).tolist()
        for sigma, sigma_next in zip(sigmas[:-1], sigmas[1:], strict=True):
            (velocity,) = self.transformer(
                hidden_states=latents,
                encoder_hidden_states=text,
                pooled_projections=pooled,
                timestep=torch.full((len(prompts),), sigma, device=self.device, dtype=self.dtype),
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=None,
                return_dict=False,
            )
            latents = latents + (sigma_next - sigma) * velocity
        latents = latents.reshape(len(prompts), grid_height, grid_width, LATENT_CHANNELS, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5).reshape(len(prompts), LATENT_CHANNELS, latent_height, latent_width)
        images = self.vae.decode(latents)
        return (images / 2 + 0.5).clamp(0, 1)
