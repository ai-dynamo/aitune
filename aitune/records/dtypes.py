# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frontend-neutral tensor element types used in publication records."""

from enum import Enum


class DType(Enum):
    """Tensor element types supported by the first adapters.

    Frontends and publishers convert their native dtypes at the records boundary.
    """

    BOOL = "bool"
    UINT8 = "uint8"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
