# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Name the caller of module-moving and weight-touching code when no recorded call is running."""

import sys

OUTSIDE_ANY_CALL = "<outside any call>"

UNRECORDED_PREFIX = "<unrecorded: "

_MOVING_METHODS = frozenset({
    "_apply",
    "to",
    "cuda",
    "cpu",
    "xpu",
    "mps",
    "half",
    "float",
    "double",
    "bfloat16",
    "type",
    "to_empty",
})


def unrecorded_caller() -> str:
    """Name the code that moved modules when no recorded call is running.

    Walks up from the ``_apply`` probe and returns ``<unrecorded: qualname>`` for the first frame that is neither in
    ``torch`` nor in this package, and is not itself a module-moving method (so a library ``.to()`` override such as
    ``PreTrainedModel.to`` is skipped and its caller, e.g. ``SentenceTransformer.encode``, is named).
    """
    frame = sys._getframe(1)
    while frame is not None:
        module = frame.f_globals.get("__name__", "")
        in_torch = module == "torch" or module.startswith("torch.")
        if (
            not in_torch
            and not module.startswith("aitune.torch.call_tree")
            and frame.f_code.co_name not in _MOVING_METHODS
        ):
            return f"{UNRECORDED_PREFIX}{frame.f_code.co_qualname}>"
        frame = frame.f_back
    return OUTSIDE_ANY_CALL
