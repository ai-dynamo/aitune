# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validated Torch AOTInductor configuration for Triton."""

from typing import Literal

from pydantic import model_validator

from aitune.triton.config.common import _BaseModelConfig


class TorchAOTIModelConfig(_BaseModelConfig):
    """Triton configuration specialized for AOTInductor PT2 packages."""

    platform: Literal["torch_aoti"] = "torch_aoti"
    structured_call: bool

    @model_validator(mode="after")
    def _validate_structured_batching(self) -> "TorchAOTIModelConfig":
        """Reject batching that cannot preserve a structured PT2 call."""
        if self.structured_call and self.max_batch_size > 0:
            raise ValueError("Triton default batching does not support PT2 artifacts with structured calls")
        return self
