# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare helper policies and saved profiles through public FlexTensor APIs on CUDA."""

# /// script
# dependencies = ["flextensor==0.5.0"]
# scope = "always"
# allow_failure = false
# ///

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import flextensor as ft
import torch

from aitune.torch.jit.flextensor import FlexTensorUnit, prepare_flextensor_units

WIDTH = 512
LAYERS = 6
POLICIES = ("names", "classes", "parameters", "mixed", "outside")


class Scale(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(WIDTH))

    def forward(self, value):
        return value * self.weight


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(WIDTH))
        self.projection = torch.nn.Linear(WIDTH, WIDTH)
        self.normalization = Scale()

    def forward(self, value):
        return value + self.normalization(self.projection(value)).relu() * self.scale * 0.1


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.root_scale = torch.nn.Parameter(torch.ones(WIDTH))
        self.layers = torch.nn.ModuleList(Block() for _ in range(LAYERS))

    def forward(self, value):
        value = value * self.root_scale
        for layer in self.layers:
            value = layer(value)
        return value


def selection(policy):
    if policy == "names":
        return ["name:layers.*"], ["layers.0"], tuple(f"layers.{index}" for index in range(1, LAYERS))
    if policy == "classes":
        return ["class:Block"], ["class:Scale"], tuple(f"layers.{index}" for index in range(LAYERS))
    if policy == "parameters":
        paths = tuple(f"layers.{index}.projection" for index in range(LAYERS))
        return ["layers.*.projection.weight"], ["*.bias"], paths
    if policy == "mixed":
        # Dropping the ancestor must NOT expose independently included projections.
        paths = tuple(f"layers.{index}" for index in range(1, LAYERS))
        return ["class:Block", "layers.*.projection.weight", "root_scale"], ["layers.0", "*.bias", "class:Scale"], paths
    paths = tuple(f"layers.{index}.{child}" for index in range(LAYERS) for child in ("projection", "normalization"))
    # Includes direct Block.scale parameters even though the selected units are its children.
    return ["layers.*.*", "root_scale"], [], paths


def install_storage_observers(model, storage):
    """Observe public Torch storage at each original parameter owner's compute boundary."""
    enabled = [False]

    def observer(path):
        def capture(module, args):
            if enabled[0]:
                for name, parameter in module.named_parameters(recurse=False):
                    tensor_storage = parameter.untyped_storage()
                    storage[f"{path}.{name}".lstrip(".")] = (
                        str(parameter.device),
                        tensor_storage.data_ptr(),
                        tensor_storage.nbytes(),
                    )

        return capture

    for path, module in model.named_modules():
        module.register_forward_pre_hook(observer(path))
    return enabled


def worker(policy, setup, profile_dir, result_path):
    torch.manual_seed(73)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = Model().cuda().eval()
    values = tuple(torch.randn(batch, WIDTH, device="cuda") for batch in (1, 2))
    with torch.no_grad():
        expected = tuple(model(value).clone() for value in values)
    parameter_names = tuple(name for name, _ in model.named_parameters())
    weight_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    model = model.cpu()
    # Application-owned root placement remains explicit, as in the LTX workflow.
    model.root_scale.data = model.root_scale.data.cuda()
    if policy == "outside":
        # These parameters are used between the independently managed child boundaries.
        # The application must place them, just as it places root-owned parameters.
        for layer in model.layers:
            layer.scale.data = layer.scale.data.cuda()
    storage = {}
    observe = install_storage_observers(model, storage)
    includes, excludes, expected_paths = selection(policy)
    config = ft.OffloadConfig(
        include_patterns=includes,
        exclude_patterns=excludes,
        external_compile=True,
        profile_mode="view",
        transfer_mode="allocation_block_transfer",
        num_blocks=2,
        min_blocks=2,
        discovery_iters=1,
        profiling_iters=2,
        # A documented, timing-independent strategy makes the storage partition comparable.
        load_strategy=ft.NthLayerStrategy(nth_layer=1, threshold_mb=0.0, n_blocks=2),
    )
    if setup.startswith("helper"):
        owner = torch.nn.Module()
        owner.transformer = model
        prepared = prepare_flextensor_units(
            owner,
            target="transformer",
            ft_config=config,
            load_profile_from=profile_dir if setup == "helper-load" else None,
        )
        assert owner.transformer is prepared.model
        model, manager = owner.transformer, prepared.manager
        assert prepared.unit_paths == expected_paths
        assert prepared.effective_config.load_strategy is config.load_strategy
    else:
        if setup == "manual-shell":
            assert policy == "names", "The preserved manual recipe uses module-name selectors"
            for path in expected_paths:
                model.set_submodule(path, FlexTensorUnit(model.get_submodule(path)))
        model = ft.offload(model, config=config)
        manager = ft.get_offload_manager()
    with torch.no_grad():
        for _ in range(manager.iters_before_inference):
            model(values[0])
        assert manager.phase.value == "inference"
        assert tuple(name for name, _ in manager.offload_units()) == expected_paths
        observe[0] = True
        for value, baseline in zip(values, expected, strict=True):
            storage.clear()
            torch.testing.assert_close(model(value), baseline)
        assert set(storage) == set(parameter_names), "A live original parameter lost its compute-boundary observation"
    by_storage = defaultdict(list)
    for name, identity in storage.items():
        assert identity[0] == "cuda:0", (name, identity)
        by_storage[identity].append(name)
    groups = sorted(sorted(names) for names in by_storage.values())
    assert any(len(names) > 1 for names in groups), "No actual rolling tensor storage was observed"
    usage = manager.get_gpu_memory_usage()
    assert 0 < usage.blocks_bytes < weight_bytes
    assert usage.total_bytes < weight_bytes, "The policy check made the model fully resident"
    if setup in ("helper", "manual-shell"):
        manager.save_profile(str(profile_dir))
    result = {
        "units": [name for name, _ in manager.offload_units()],
        "residents": [name for name, _ in manager.resident_units()],
        "storage_groups": groups,
        "blocks_bytes": usage.blocks_bytes,
        "unmapped_tensors_bytes": usage.unmapped_tensors_bytes,
        "observed_parameters": len(storage),
    }
    result_path.write_text(json.dumps(result))
    print(json.dumps({"result": "PASS", "policy": policy, "setup": setup, **result}), flush=True)


def run_worker(policy, setup, profile_dir, result_path):
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            setup,
            "--policy",
            policy,
            "--profile-dir",
            str(profile_dir),
            "--result-path",
            str(result_path),
        ],
        check=True,
    )
    return json.loads(result_path.read_text())


def run(directory, policies):
    for policy in policies:
        profile_dir = directory / policy / "helper-profile"
        result_path = directory / f"{policy}.json"
        original = run_worker(policy, "original", profile_dir, result_path)
        prepared = run_worker(policy, "helper", profile_dir, result_path)
        assert prepared == original, (policy, original, prepared)
        loaded = run_worker(policy, "helper-load", profile_dir, result_path)
        assert loaded == prepared, (policy, prepared, loaded)
        if policy == "names":
            manual_dir = directory / policy / "manual-profile"
            manual = run_worker(policy, "manual-shell", manual_dir, result_path)
            assert manual == prepared, (manual, prepared)
            manual_loaded = run_worker(policy, "helper-load", manual_dir, result_path)
            assert manual_loaded == manual, (manual, manual_loaded)
    print(json.dumps({"result": "PASS", "policies": list(policies), "fresh_worker_profiles": True}), flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=(*POLICIES, "all"), default="all")
    parser.add_argument("--worker", choices=("original", "helper", "helper-load", "manual-shell"))
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--result-path", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.policy == "all" or args.profile_dir is None or args.result_path is None:
            parser.error("workers require one --policy, --profile-dir and --result-path")
        worker(args.policy, args.worker, args.profile_dir, args.result_path)
    else:
        with tempfile.TemporaryDirectory(prefix="aitune-flextensor-policy-") as directory:
            run(Path(directory), POLICIES if args.policy == "all" else (args.policy,))
