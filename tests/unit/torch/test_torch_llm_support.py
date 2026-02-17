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
"""Unit tests for testing custom user types e.g. kv cache for LLM support."""

import abc
import logging
from logging import DEBUG, basicConfig

import pytest
import torch

from aitune.torch import Module, OneBackendStrategy
from aitune.torch.backend import TorchInductorBackend
from aitune.torch.module.locator import Locator
from aitune.torch.module.wrapper_module import ModuleState

try:
    from tests.utilities.helpers import requires_cuda
except ImportError:
    requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")

# --- LLM Parameters ---
BATCH_SIZE = 1
HEAD_DIM = 64  # d_k, dimension of Key/Value/Query vectors
VOCAB_SIZE = 100
MAX_PROMPT_LEN = 4
MAX_SEQ_LEN = 5
CACHE_LEN = MAX_PROMPT_LEN + MAX_SEQ_LEN


class BaseKVCache(abc.ABC):
    """Test KV cache."""

    @abc.abstractmethod
    def lazy_initialization(self, key_states: torch.Tensor):
        """Lazy initialization of the cache."""
        pass

    @abc.abstractmethod
    def update(self, keys, values):
        """Update cache with new keys and values."""
        pass

    @abc.abstractmethod
    def get_keys(self):
        """Return cached keys."""
        pass

    @abc.abstractmethod
    def get_values(self):
        """Return cached values."""
        pass

    @abc.abstractmethod
    def is_empty(self):
        """Return True if the cache is empty."""
        pass

    @abc.abstractmethod
    def __len__(self):
        """Number of items in cache."""
        pass


class DynamicKVCache(BaseKVCache):
    def __init__(self):
        # shapes: (B, T, D)
        self.keys = None
        self.values = None

    def lazy_initialization(self, keys: torch.Tensor):
        self.dtype, self.device = keys.dtype, keys.device
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)

    def update(self, keys, values):
        if self.keys is None:
            self.lazy_initialization(keys)

        self.keys = torch.cat([self.keys, keys], dim=-2)
        self.values = torch.cat([self.values, values], dim=-2)
        return self.keys, self.values

    def get_keys(self):
        return self.keys

    def get_values(self):
        return self.values

    def is_empty(self):
        return self.keys is None or self.keys.shape[-2] == 0

    def __len__(self):
        return self.keys.shape[-2] if self.keys is not None else 0


class StaticKVCache(BaseKVCache):
    def __init__(self, max_cache_len: int = CACHE_LEN):
        self.max_cache_len = max_cache_len
        self.index = 0
        self.keys = None
        self.values = None

    def lazy_initialization(self, keys: torch.Tensor):
        shape = list(keys.shape)
        shape[1] = self.max_cache_len
        self.keys = torch.zeros(shape, dtype=keys.dtype, device=keys.device)
        self.values = torch.zeros(shape, dtype=keys.dtype, device=keys.device)

    def update(self, keys: torch.Tensor, values: torch.Tensor):
        """Update cache with new keys and values.

        This method resembles HuggingFace static cache update method.
        This is in order to check their implementation correctness at small test scale.
        """
        if self.keys is None:
            self.lazy_initialization(keys)

        if self.index == self.max_cache_len:
            raise ValueError("Cache is full")

        seq_len = keys.shape[-2]
        cache_position = torch.arange(seq_len, device=keys.device)
        try:
            self.keys.index_copy_(1, cache_position, keys)
            self.values.index_copy_(1, cache_position, values)
        except NotImplementedError:
            self.keys[:, cache_position] = keys
            self.values[:, cache_position] = values
        self.index += seq_len
        return self.keys, self.values

    def get_keys(self):
        return self.keys[:, : self.index] if self.keys is not None else None

    def get_values(self):
        return self.values[:, : self.index] if self.values is not None else None

    def is_empty(self):
        return self.index == 0

    def __len__(self):
        return self.index


class ToyLLMModel(torch.nn.Module):
    """Toy LLM model.

    This is a toy model consisting of just one transformer block with a `generate` method similar to the HF one.

    """

    def __init__(self, embedding, W_Q, W_K, W_V, W_OUT):  # noqa: N803
        super().__init__()
        self.embedding = embedding
        self.W_Q = W_Q
        self.W_K = W_K
        self.W_V = W_V
        self.W_OUT = W_OUT

    def forward(self, x, kv_cache: BaseKVCache):
        is_decode = not kv_cache.is_empty()
        # Shape check: x is (B, T, D)
        q = torch.einsum("btd, dh -> bth", x, self.W_Q)
        k = torch.einsum("btd, dh -> bth", x, self.W_K)
        v = torch.einsum("btd, dh -> bth", x, self.W_V)
        kv_cache.update(k.detach(), v.detach())

        if is_decode:
            k = kv_cache.get_keys()
            v = kv_cache.get_values()

        attn_scores = (q @ k.transpose(1, 2)) / (HEAD_DIM**0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        out = attn_weights @ v

        logits = torch.einsum("bth, hv -> btv", out, self.W_OUT)

        if logits.shape[1] > 0:
            return logits[:, -1, :]  # (B, VOCAB_SIZE)
        return logits

    def generate(self, prompt, kv_cache: BaseKVCache, max_seq_len: int):
        result = []
        # pre-fill stage
        x_prompt = self.embedding(prompt)
        assert x_prompt.shape == (1, 4, HEAD_DIM)

        logits_prefill = self.forward(x_prompt, kv_cache)
        assert logits_prefill.shape == (1, VOCAB_SIZE)

        next_token_id = torch.argmax(logits_prefill, dim=-1, keepdim=True)
        result.append(next_token_id.item())
        assert next_token_id.shape == (1, 1)

        k_cache = kv_cache.get_keys()
        assert k_cache.shape == (1, 4, HEAD_DIM)

        # decoding stage
        for i in range(1, max_seq_len):
            x_next = self.embedding(next_token_id)

            logits_next = self.forward(x_next, kv_cache)
            assert logits_next.shape == (1, VOCAB_SIZE)

            next_token_id = torch.argmax(logits_next, dim=-1, keepdim=True)
            result.append(next_token_id.item())
            assert next_token_id.shape == (1, 1)

            k_cache = kv_cache.get_keys()
            assert k_cache.shape == (1, 4 + i, HEAD_DIM)

        return result, kv_cache


@requires_cuda
@pytest.mark.parametrize("cache_type", [DynamicKVCache, StaticKVCache])
@pytest.mark.parametrize("registered_type", [False, True], ids=["unregistered", "registered"])
def test_llm_dynamic_cache(cache_type, registered_type, torch_device):
    """Test LLM prefill and decode stages with KV cache, including sampling.

    Args:
        cache_type: The type of KV cache to use.
        registered_type: Whether to register the cache type as a user type.
        torch_device: The device to run the test on.

    Note:
    - If `registered_type` is True, the cache type will be registered as a user type, its tensors will be tracked
    - If `registered_type` is False, the cache type will be ignored.
    """
    embedding = torch.nn.Embedding(VOCAB_SIZE, HEAD_DIM).to(torch_device)
    w_k = torch.randn(HEAD_DIM, HEAD_DIM).to(torch_device)
    w_v = torch.randn(HEAD_DIM, HEAD_DIM).to(torch_device)
    w_q = torch.randn(HEAD_DIM, HEAD_DIM).to(torch_device)
    w_out = torch.randn(HEAD_DIM, VOCAB_SIZE).to(torch_device)
    prompt_token_ids = torch.arange(MAX_PROMPT_LEN, device=torch_device).unsqueeze(0)  # (1, MAX_PROMPT_LEN)

    model = ToyLLMModel(embedding, w_q, w_k, w_v, w_out).to(torch_device)
    model.eval()
    kv_cache = cache_type()

    result, kv_cache = model.generate(prompt_token_ids, kv_cache, max_seq_len=MAX_SEQ_LEN)
    assert len(result) == MAX_SEQ_LEN
    assert len(kv_cache) == CACHE_LEN - 1

    if registered_type:
        Locator.register_user_type(cache_type, only_tensors=True)
    else:
        Locator.ignore_type(cache_type)

    model = ToyLLMModel(embedding, w_q, w_k, w_v, w_out).to(torch_device)
    model.eval()
    kv_cache = cache_type()

    model = Module(model, "test_llm")  # wrap the model with AITune
    with torch.no_grad():
        result, kv_cache = model.generate(prompt_token_ids, kv_cache, max_seq_len=MAX_SEQ_LEN)

    if registered_type:
        assert len(model.graph_specs) == 2, "Expected 2 graphs due to lazy initialization of kv cache"
    else:
        assert len(model.graph_specs) == 1, "Expected 1 graph due to ignored kv cache type"

    model.tune(device=torch_device, strategy=OneBackendStrategy(TorchInductorBackend()), dry_run=False)
    assert model.state == ModuleState.TUNED, "Model should be in tuned state after tuning"

    tuned_result, kv_cache = model.generate(prompt_token_ids, cache_type(), max_seq_len=MAX_SEQ_LEN)
    assert len(tuned_result) == MAX_SEQ_LEN
    assert len(kv_cache) == CACHE_LEN - 1
    assert tuned_result == result, "Tuned result should be the same as the original result"


if __name__ == "__main__":
    """This test can be run manually as a python script"""
    basicConfig(level=DEBUG, force=True)
    for cache_type in [DynamicKVCache, StaticKVCache]:
        for registered_type in [False, True]:
            log_msg = f"--- Running test with cache type {cache_type} and registered type {registered_type} ---"
            logging.info("-" * len(log_msg))
            logging.info(log_msg)
            logging.info("-" * len(log_msg))
            test_llm_dynamic_cache(cache_type, registered_type, torch_device=torch.device("cuda"))
