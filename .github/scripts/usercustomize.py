# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prefer local flash_attn package over the system-wide one."""

import sys
from importlib.machinery import PathFinder
from importlib.util import module_from_spec
from pathlib import Path


def _prefer_local_flash_attn():
    local_site = Path(__file__).resolve().parent
    local_package = local_site / "flash_attn"

    if not local_package.is_dir():
        return

    # Do not replace a package already loaded by earlier startup hooks.
    if "flash_attn" in sys.modules:
        print(  # noqa: T201
            "usercustomize: flash_attn already loaded; local override was not applied",
            file=sys.stderr,
        )
        return

    spec = PathFinder.find_spec("flash_attn", [str(local_site)])
    if spec is None:
        return

    package = module_from_spec(spec)
    sys.modules["flash_attn"] = package

    if spec.loader is not None:
        try:
            spec.loader.exec_module(package)
        except BaseException:
            sys.modules.pop("flash_attn", None)
            raise


_prefer_local_flash_attn()
del _prefer_local_flash_attn
