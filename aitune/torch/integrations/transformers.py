# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Register Transformers cache traversal and report devices for tuned modules.

A StaticCache update can initialize key/value tensors and Python metadata in the same call. Restoring
only tensor results would leave the caller's layer marked uninitialized or missing device and shape
information. StaticLayer and StaticSlidingWindowLayer therefore register the initialization fields
that forward-contract capture must observe in addition to tensors.

For example, a fresh layer can change from keys=None and is_initialized=False to an allocated tensor
and is_initialized=True. The compiled graph supplies the tensor; restoration assigns the recorded
initialization fields to the caller's layer. These non-tensor fields stay out of ordinary SampleMetadata
traversal so they do not become tensor inputs or ordinary graph-matching data. They are recorded values,
not dynamic updates to arbitrary Python state. DynamicCache remains excluded from this traversal.

EncoderDecoderCache needs a different registration: its is_updated flags select whether cross-attention
initializes or reuses its cache, so ordinary strict metadata must see those flags as well as both nested
static caches. Registration does not select cache implementations or compile unrecorded phases.
Strict metadata separates readiness states both with ordinary graph matching and when cache_position
activates LLM phase matching. Phase matching ignores tensor metadata but retains strict non-tensor values,
so cross-attention initialization and reuse can select different graphs within the same decode phase.
"""

try:
    from transformers import DynamicCache, EncoderDecoderCache, StaticCache, StaticLayer, StaticSlidingWindowLayer

    from aitune.torch.module.locator import Locator

    # Lazy initialization sets device/dtype and tensor dimensions as well as the readiness flag.
    # Capture this explicit set so the next call sees a usable layer without observing unrelated Python fields.
    # Transformers 4.x uses head_dim; 5.x uses k_head_dim/v_head_dim and later replaces max_batch_size
    # with batch_size. Traversal captures only fields present on the installed version's layer.
    _STATIC_LAYER_INITIALIZATION_FIELDS = (
        "batch_size",
        "device",
        "dtype",
        "head_dim",
        "is_initialized",
        "k_head_dim",
        "max_batch_size",
        "num_heads",
        "v_head_dim",
    )

    # enable static cache support
    Locator.register_user_type(StaticCache, only_tensors=True)
    Locator.register_user_type(
        StaticLayer,
        only_tensors=True,
        mutation_fields=_STATIC_LAYER_INITIALIZATION_FIELDS,
    )
    Locator.register_user_type(
        StaticSlidingWindowLayer,
        only_tensors=True,
        mutation_fields=_STATIC_LAYER_INITIALIZATION_FIELDS,
    )
    # Whisper's static cache wraps separate self-attention and cross-attention caches.
    # Keep the wrapper's non-tensor is_updated flags visible: they select whether
    # cross-attention initializes its cache or reuses it. Strict metadata separates these
    # states even within the same LLM phase selected by cache_position. Nested StaticCache
    # objects retain tensor-only inspection.
    # Cache selection remains the caller's responsibility; nested DynamicCache
    # objects remain unsupported and ignored below.
    Locator.register_user_type(EncoderDecoderCache)
    # disable inspection of dynamic cache as it is not supported by neither jit nor aot backends
    # the only exception is TorchEagerBackend as it is basically passthrough with some dtype casts
    Locator.ignore_type(DynamicCache)
except ImportError:
    pass

try:
    from transformers.modeling_utils import ModuleUtilsMixin

    from aitune.torch.integrations.device import patch_device_property

    # report the execution device of tuned submodules instead of meta
    patch_device_property(ModuleUtilsMixin)
except ImportError:
    pass
