# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared functional-test drivers for Transformers native tensor parallelism.

The initial pytest process downloads the model, creates an isolated cache root, and relaunches the calling test script
with torchrun. Each worker re-enters that script, initializes NCCL on its rank-local GPU, executes the requested AOT or
deferred-JIT workflow, and then destroys its application-owned process group.
"""

import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import torch
import torch.distributed as dist
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.distributed import DistributedConfig

from aitune.torch import OneBackendStrategy
from aitune.torch.backend.backend import Backend

# Qwen3.5-4B has four KV heads, so its native TP plan supports four ranks.
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_PATH_ENV = "AITUNE_TEST_MODEL_PATH"
WORLD_SIZE = 4
DISTRIBUTED_LAUNCH_TIMEOUT_SECONDS = 30 * 60


def run_aot_native_tp_test(backend_class: type[Backend], module_name: str, script_path: str | Path) -> None:
    """Run the native-TP AOT tuning workflow for one backend.

    Each worker loads its tensor-parallel model shard, captures eager output, tunes the wrapped module, and verifies
    that the requested backend was selected and produced non-empty output.

    Args:
        backend_class: Backend class to build on every rank.
        module_name: Stable wrapped-module name used by AITune.
        script_path: Calling functional-test script to relaunch with torchrun.
    """
    from aitune.torch import Module, tune
    from aitune.torch.module.wrapper_module import ModuleState

    def test(device: torch.device) -> None:
        """Execute the AOT workflow in one torchrun worker."""
        tensor_parallel_model, inputs = _load_tensor_parallel_model_and_inputs(device)
        with torch.no_grad():
            expected_outputs = tensor_parallel_model(**inputs)
        strategy = _strategy(backend_class())
        tuned_module = Module(tensor_parallel_model, module_name, strategy=strategy)
        dataset = [{name: tensor[0].cpu() for name, tensor in inputs.items()}]

        def run_tuned_module(**batch: torch.Tensor) -> object:
            """Move a recorded batch to this worker's device and invoke the wrapped module."""
            return tuned_module(**{name: tensor.to(device) for name, tensor in batch.items()})

        tune(
            run_tuned_module,
            dataset,
            batch_sizes=[1],
            max_num_batches_per_batch_size=1,
            device=None,
            disable_external_logging=False,
            ignore_failing_modules=False,
        )

        assert tuned_module.state == ModuleState.TUNED
        selected_backends = {type(backend).__name__ for backend in tuned_module.module.backends.values()}
        assert selected_backends == {backend_class.__name__}
        with torch.no_grad():
            outputs = tuned_module(**inputs)
        dist.barrier()
        _assert_tuned_output(outputs, expected_outputs)

    _run_or_launch(test, script_path, backend_class)


def run_jit_native_tp_test(backend_class: type[Backend], script_path: str | Path) -> None:
    """Run the native-TP deferred-JIT tuning workflow for one backend.

    Each worker patches and loads its tensor-parallel model shard, records an eager sample, enables deferred tuning,
    triggers tuning on the next forward, and verifies the selected backend and subsequent output.

    Args:
        backend_class: Backend class to build on every rank.
        script_path: Calling functional-test script to relaunch with torchrun.
    """
    from aitune.torch.jit.config import JITMode
    from aitune.torch.jit.config import config as jit_config
    from aitune.torch.jit.patched_module import ModuleState as JITModuleState
    from aitune.torch.jit.patched_module import PatchedModule
    from aitune.torch.jit.patcher import prepare_for_jit_tuning
    from aitune.torch.jit.tune import deferred as tune_deferred
    from aitune.torch.module.tuned_module import TunedModule

    def test(device: torch.device) -> None:
        """Execute the deferred-JIT workflow in one torchrun worker."""
        jit_config.mode = JITMode.TUNE_DEFERRED
        jit_config.strategy = _strategy(backend_class())
        jit_config.device = None
        jit_config.dry_run = False
        jit_config.min_samples = 1
        jit_config.batch_axis_required = False
        jit_config.max_depth_level = 1
        jit_config.detect_graph_breaks = False

        with prepare_for_jit_tuning():
            tuned_module, inputs = _load_tensor_parallel_model_and_inputs(device)

        with torch.no_grad():
            expected_outputs = tuned_module(**inputs)
            tune_deferred()
            tuned_module(**inputs)
            outputs = tuned_module(**inputs)

        patched_module = next(head for head in PatchedModule.heads if head.__wrapped__ is tuned_module)
        assert patched_module._state == JITModuleState.TUNED
        assert isinstance(patched_module._wrapper, TunedModule)
        selected_backends = {type(backend).__name__ for backend in patched_module._wrapper.backends.values()}
        assert selected_backends == {backend_class.__name__}
        dist.barrier()
        _assert_tuned_output(outputs, expected_outputs)

    _run_or_launch(test, script_path, backend_class)


def _strategy(backend: Backend) -> OneBackendStrategy:
    """Build a deterministic single-backend strategy without extra profiling passes."""
    strategy = OneBackendStrategy(backend)
    strategy.enable_performance_validation(False)
    strategy.enable_find_max_batch_size(False)
    return strategy


def _load_tensor_parallel_model_and_inputs(device: torch.device) -> tuple[torch.nn.Module, dict[str, torch.Tensor]]:
    """Load rank-local inputs and the native-TP model shard for the active process group."""
    model_path = os.environ[MODEL_PATH_ENV]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    inputs = tokenizer("AITune native tensor parallel inference", return_tensors="pt")
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        distributed_config=DistributedConfig(tp_size=dist.get_world_size()),
        trust_remote_code=False,
    ).eval()
    model.config.use_cache = False
    return model, inputs


def _assert_tuned_output(outputs: object, expected_outputs: object) -> None:
    """Require both executions to return tensor logits and the tuned result to be non-empty."""
    logits = getattr(outputs, "logits", None)
    expected_logits = getattr(expected_outputs, "logits", None)
    assert isinstance(logits, torch.Tensor)
    assert isinstance(expected_logits, torch.Tensor)
    assert logits.numel() > 0


def _run_or_launch(
    test: Callable[[torch.device], None],
    script_path: str | Path,
    backend_class: type[Backend],
) -> None:
    """Run through the functional test's parent and torchrun worker entry points.

    The initial pytest process launches this test script with torchrun. Each worker re-enters with LOCAL_RANK set,
    initializes its distributed state, and runs the provided test on its rank-local device.
    """
    if "LOCAL_RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        _run_distributed(test)
    else:
        _launch_distributed(script_path, backend_class)


def _run_distributed(test: Callable[[torch.device], None]) -> None:
    """Initialize one torchrun worker, execute the test, and release its process group."""
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    try:
        assert dist.get_world_size() == WORLD_SIZE
        test(device)
    finally:
        dist.destroy_process_group()


def _launch_distributed(script_path: str | Path, backend_class: type[Backend]) -> None:
    """Prepare shared inputs and relaunch the functional test script with torchrun."""
    if torch.cuda.device_count() < WORLD_SIZE:
        raise RuntimeError(f"This functional test requires {WORLD_SIZE} visible GPUs")
    model_path = snapshot_download(MODEL_ID)
    with tempfile.TemporaryDirectory(prefix=f"aitune-native-tp-{backend_class.__name__.lower()}-") as cache_dir:
        env = os.environ.copy()
        env[MODEL_PATH_ENV] = model_path
        env["AITUNE_CACHE_DIR"] = cache_dir
        env["AITUNE_JIT_CACHE_DIR"] = cache_dir
        subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc-per-node={WORLD_SIZE}",
                str(Path(script_path).resolve()),
            ],
            check=True,
            timeout=DISTRIBUTED_LAUNCH_TIMEOUT_SECONDS,
            env=env,
        )
