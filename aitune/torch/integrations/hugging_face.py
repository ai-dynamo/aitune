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
"""Hugging Face integrations."""

try:
    from transformers import DynamicCache, StaticCache, StaticLayer, StaticSlidingWindowLayer

    from aitune.torch.module.locator import Locator

    # enable static cache support
    Locator.register_user_type(StaticCache, only_tensors=True)
    Locator.register_user_type(StaticLayer, only_tensors=True)
    Locator.register_user_type(StaticSlidingWindowLayer, only_tensors=True)
    # disable inspection of dynamic cache as it is not supported by neither jit nor aot backends
    # the only exception is TorchEagerBackend as it is basically passthrough with some dtype casts
    Locator.ignore_type(DynamicCache)
except ImportError:
    pass
