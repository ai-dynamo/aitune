# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for torch checkpoint tests."""

from aitune.torch.module.tuned_module import TunedModule


def _backend_state_with_paths(**paths):
    backend_data = {TunedModule.TYPE_KEY: "FakeBackend", **paths}
    return {"wrapped": {TunedModule.BACKENDS_KEY: [({"sample": "metadata"}, backend_data)]}}


def _only_backend_data(state_dict):
    return state_dict["wrapped"][TunedModule.BACKENDS_KEY][0][1]
