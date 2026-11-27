# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU samples shared by tensor microbenchmarks."""

from typing import Any

import torch


def make_timing_sample(structure, tensor_count, tensor_elements):
    tensors = tuple(torch.empty(tensor_elements, device="cpu") for _ in range(tensor_count))
    if structure == "tensor":
        return tensors[0]
    if structure == "tuple":
        return tensors
    if structure == "cache":
        return {"logits": tensors[0], "cache": tuple(zip(tensors[1::2], tensors[2::2], strict=True))}

    value: dict[str, Any] = {f"tensor_{index}": tensor for index, tensor in enumerate(tensors)}
    if structure == "nested_dict":
        for _ in range(3):
            value = {"nested": value}
    elif structure == "metadata":
        value["metadata"] = [{"tokens": list(range(32)), "label": "decode"} for _ in range(32)]
    else:
        assert structure == "dict"
    return value
