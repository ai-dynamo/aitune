# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strategy extensions for tune strategy.

Extensions should augment tune method instead of implementing _tune method.
"""

from aitune.torch.tune_strategy.extension.find_max_batch_size_extension import (
    FindMaxBatchSizeExtensionConfig,
    TuneStrategyFindMaxBatchSizeExtension,
)

__all__ = [
    "TuneStrategyFindMaxBatchSizeExtension",
    "FindMaxBatchSizeExtensionConfig",
]
