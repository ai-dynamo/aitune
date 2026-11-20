# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Transformers integrations."""

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

try:
    from transformers.models.voxtral_realtime.modeling_voxtral_realtime import (
        VoxtralRealtimeConv1dCacheLayer,
        VoxtralRealtimeConv1dPaddingCache,
    )

    from aitune.torch.module.locator import Locator

    # Voxtral Realtime passes a convolution padding cache between generation steps and updates its tensors in place.
    # The cache does not define equality, so as an opaque input it would be compared by object identity: each
    # generate() call would record a separate graph, and a deep copy of a recorded sample would never match the
    # recorded post-call inputs. Track the cache tensors instead, as for static caches.
    Locator.register_user_type(VoxtralRealtimeConv1dPaddingCache, only_tensors=True)
    Locator.register_user_type(VoxtralRealtimeConv1dCacheLayer, only_tensors=True)
except ImportError:
    # Voxtral Realtime cache support is optional and depends on the Transformers version.
    pass

try:
    from transformers.modeling_utils import ModuleUtilsMixin

    from aitune.torch.integrations.device import patch_device_property

    # report the execution device of tuned submodules instead of meta
    patch_device_property(ModuleUtilsMixin)
except ImportError:
    pass
