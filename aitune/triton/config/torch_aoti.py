# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validated Torch AOTInductor configuration for Triton."""

from typing import Any, Literal

from pydantic import model_validator

from aitune.records import DeploymentArtifact
from aitune.triton.config.common import BaseModelConfig


class TorchAOTIModelConfig(BaseModelConfig):
    """Triton configuration specialized for AOTInductor PT2 packages."""

    platform: Literal["torch_aoti"] = "torch_aoti"
    structured_call: bool

    @classmethod
    def _artifact_options(cls, artifact: DeploymentArtifact) -> dict[str, Any]:
        """Read the PT2 package's call structure."""
        return {"structured_call": artifact.model.metadata["structured_call"]}

    @model_validator(mode="after")
    def _validate_structured_batching(self) -> "TorchAOTIModelConfig":
        """Reject batching that cannot preserve a structured PT2 call."""
        if self.structured_call and self.dynamic_batching:
            raise ValueError("Triton default batching does not support PT2 artifacts with structured calls")
        return self
