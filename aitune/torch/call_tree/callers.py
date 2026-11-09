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
    """Name the code that is running when no recorded call is, for example the code that called ``.to()``.

    Code that moves modules is often not a recorded call: a pipeline function outside the model, or library code
    such as ``SentenceTransformer.encode``, which calls ``self.to(device)`` on every call. Instead of reporting all
    of these as ``<outside any call>``, this names the code, e.g. ``<unrecorded: SentenceTransformer.encode>``.

    It walks up the stack from its caller and skips frames:

    - in ``torch`` (the ``.to()`` to ``_apply`` machinery);
    - in ``aitune.torch.call_tree`` (the recorder's own probes);
    - of a function named like a module-moving method (``to``, ``cuda``, ``half``, ...), so a library override such
      as ``PreTrainedModel.to`` is skipped and its caller is named. A user function with such a name is skipped too;
      that only changes which caller is shown.

    Returns:
        ``<unrecorded: qualname>`` of the first frame left, or ``<outside any call>`` if every frame is skipped.
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
