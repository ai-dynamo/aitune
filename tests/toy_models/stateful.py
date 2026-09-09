# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mutation cases shared by export and backend round-trip tests."""

MUTATION_OPERATIONS = ("add_", "mul_", "index_copy_", "scatter_", "zero_")


def update_tensor(cache, value, index, operation):
    if operation == "add_":
        cache.add_(value)
    elif operation == "mul_":
        cache.mul_(value)
    elif operation == "index_copy_":
        cache.index_copy_(0, index, value)
    elif operation == "scatter_":
        cache.scatter_(0, index, value)
    elif operation == "zero_":
        cache.zero_()
    else:
        raise ValueError(f"Unknown mutation operation: {operation}")
