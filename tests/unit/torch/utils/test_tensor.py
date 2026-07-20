# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aitune.torch.utils.tensor import format_tensor_name


def test_format_tensor_name():
    integer_path = ("inputs", 0)

    integer_name = format_tensor_name(integer_path, "input")

    assert integer_name == "input_inputs_0"
    assert integer_name.isidentifier()
    assert format_tensor_name((), "output") == "output"
