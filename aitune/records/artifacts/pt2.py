# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AOTInductor package artifact records."""

from dataclasses import dataclass

from aitune.records.artifacts.base import Artifact


@dataclass(frozen=True, kw_only=True)
class PT2Artifact(Artifact):
    """An AOTInductor package consumable by Triton's ``torch_aoti`` backend.

    Args:
        structured_call: Whether the package embeds structured inputs or outputs.
    """

    structured_call: bool
