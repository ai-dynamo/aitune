# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inspecting and patching modules."""

import contextlib
import itertools
import time
from collections.abc import Callable, Generator, Sequence
from logging import getLogger
from typing import Any

import torch

from aitune.torch.call_tree.call_tree import CallTree
from aitune.torch.call_tree.recording import (
    CallRecorder,
    StructureRecorder,
    TimingRecorder,
    describe_sample_inputs,
)
from aitune.torch.call_tree.sync import synchronize_call_tree
from aitune.torch.dataloader import DataLoaderFactory, DatasetLike, ensure_enough_samples, samples_generator
from aitune.torch.distributed import coordinator
from aitune.torch.inspecting.module_info import InspectedModulesInfo, ModuleInfo
from aitune.torch.inspecting.module_inspector import ModuleInspector
from aitune.torch.utils.cuda_utils import synchronize
from aitune.utils.logging import setup_logging

logger = getLogger(__name__)

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

DEFAULT_INSPECT_ITERATIONS = 10
DEFAULT_WARMUP_ITERATIONS = 5


def inspect(
    obj: Callable | torch.nn.Module,
    dataset: DatasetLike | DataLoaderFactory | torch.Tensor,
    inference_function: Callable | None = None,
    number_of_iterations: int = DEFAULT_INSPECT_ITERATIONS,
    warmup_iterations: int = DEFAULT_WARMUP_ITERATIONS,
    min_depth: int = 0,
    max_depth: int = 5,
    call_tree: bool = False,
    max_timing_depth: int | None = None,
) -> InspectedModulesInfo:
    """Inspect provided callable object searching for nn.Module members executed as part of forward pass.

    Args:
        obj: Callable object to inspect.
        dataset: List of tuples with batch size and input.
        inference_function: Custom inference function to use for inspection, obj is used by default.
        number_of_iterations: Number of samples from the dataset to inspect, run once per pass. The call tree's
            timing pass runs each sample once per call-path depth; each node's ``calls_per_sample`` has one entry
            per inspected sample.
        warmup_iterations: Number of iterations to run for warmup.
        min_depth: Minimum depth of the modules to inspect, if root level modules is not working, try to increase this value
        max_depth: Maximum depth of the modules to inspect
        call_tree: Also record the call tree (``InspectedModulesInfo.call_tree``): which calls ran inside which,
            with each node's time, tensor bytes, argument signatures and weight accesses. Adds a structure pass and
            a timing pass, which logs its number of replays before it starts. Calls inside parts compiled with
            ``torch.compile`` get no time. For deep models the walk must reach the linear layers, for example
            ``max_depth=8``. Details are in ``docs/guides/aot_inspect.md``.
        max_timing_depth: Deepest call-path depth the timing pass times, with the root call at depth 1; None times
            every depth, and 0 skips the timing pass, so every node keeps ``total_time_ms`` None. Deeper nodes are
            still recorded but keep ``total_time_ms`` None. Limits the timing pass to at most ``max_timing_depth``
            replays per sample.

    Returns:
        InspectedModulesInfo object.

    Note:
        In distributed execution, rank zero's module order and timings are applied to every rank so subsequent
        execution-based filtering selects modules consistently.
    """
    if max_timing_depth is not None and max_timing_depth < 0:
        raise ValueError(f"max_timing_depth must be a non-negative int or None, got {max_timing_depth!r}")

    setup_logging(format_string=LOG_FORMAT)

    logger.info("Inspecting object searching for executed nn.Module members.")
    recorder = StructureRecorder() if call_tree else None
    model_inspector = ModuleInspector(min_depth=min_depth, max_depth=max_depth, recorder=recorder)
    try:
        model_inspector.inspect(obj)

        # If no inference function is provided, use the obj
        if inference_function is None:
            inference_function = obj

        dataset = ensure_enough_samples(dataset, max(number_of_iterations, warmup_iterations))

        # Warmup, run the model on a few samples, to make sure the model is compiled and the cache is warm
        _warmup(inference_function, dataset, warmup_iterations)
        _reset_total_execution_time(model_inspector)

        # Run the model on the dataset for the number of iterations
        total_execution_time = 0.0
        for _, args, kwargs in samples_generator(dataset, [1], max_num_batches_per_batch_size=number_of_iterations):
            total_execution_time += _run_inference(inference_function, args, kwargs)

        model_inspector.warn_about_compiled_modules()
        modules = model_inspector.get_modules()
        total_execution_time, modules = _synchronize_inspection_result(total_execution_time, modules)

        tree = None
        if recorder is not None:
            tree = _record_call_tree(recorder, inference_function, dataset, number_of_iterations, max_timing_depth)

        logger.info("Inspection done. Found %d candidate modules for tuning.", len(modules))

        inspected_modules_info = InspectedModulesInfo(total_execution_time, number_of_iterations, call_tree=tree)
        for module in modules:
            inspected_modules_info.add_module(module)

        return inspected_modules_info
    finally:
        model_inspector.reset()


def _warmup(obj: Callable, dataset: DatasetLike | DataLoaderFactory | torch.Tensor, number_of_iterations: int):
    """Warmup the model by running it on a few samples."""
    for _, args, kwargs in samples_generator(dataset, [1], max_num_batches_per_batch_size=number_of_iterations):
        _run_inference(obj, args, kwargs)


def _run_inference(obj: Callable, args: Sequence[Any], kwargs: dict[str, Any]) -> float:
    """Run one complete inference in lockstep across distributed ranks."""
    coordinator.barrier()
    synchronize()
    start_time = time.perf_counter()
    with coordinator.raise_if_any_rank_fails("Inspection inference"):
        with torch.no_grad():
            obj(*args, **kwargs)
        synchronize()
    coordinator.barrier()
    return time.perf_counter() - start_time


def _record_call_tree(
    recorder: StructureRecorder,
    inference_function: Callable,
    dataset: DatasetLike | DataLoaderFactory | torch.Tensor,
    number_of_iterations: int,
    max_timing_depth: int | None = None,
) -> CallTree:
    """Record the call tree without times, then time it in a second pass, each sample once per depth it reaches."""
    tree = _record_structure(recorder, inference_function, dataset, number_of_iterations)
    if not tree.nodes:
        logger.warning(
            "The call tree recorded no calls: no instrumented module or method ran during the structure pass, "
            "for example because the inference function does not call the inspected object or inspect runs "
            "inside torch.compile. The returned call tree is empty."
        )
    if max_timing_depth == 0:
        logger.info("Skipping the call tree's timing pass (max_timing_depth=0); node times stay None.")
    else:
        _time_call_tree(
            TimingRecorder(tree, max_depth=max_timing_depth), inference_function, dataset, number_of_iterations
        )
    return synchronize_call_tree(tree)


@contextlib.contextmanager
def _pass_on_all_ranks(recorder: CallRecorder, name: str) -> Generator[None, None, None]:
    """Enter and exit ``recorder`` so that a failure on any rank stops every rank."""
    with contextlib.ExitStack() as stack:
        with coordinator.raise_if_any_rank_fails(f"Starting the {name}"):
            stack.enter_context(recorder)
        yield
        with coordinator.raise_if_any_rank_fails(f"Ending the {name}"):
            stack.close()


def _record_structure(
    recorder: StructureRecorder,
    inference_function: Callable,
    dataset: DatasetLike | DataLoaderFactory | torch.Tensor,
    number_of_iterations: int,
) -> CallTree:
    """Record the call tree's structure, one inference per sample."""
    logger.info("Recording the call tree's structure.")
    # Recording reads each weight access eagerly; Dynamo cannot trace the TorchFunctionMode.
    with _pass_on_all_ranks(recorder, "call tree's structure pass"), torch.compiler.set_stance("force_eager"):
        for _, args, kwargs in samples_generator(dataset, [1], max_num_batches_per_batch_size=number_of_iterations):
            with contextlib.ExitStack() as stack:
                # describing runs before ``_run_inference``'s barrier, so its failure must reach every rank
                with coordinator.raise_if_any_rank_fails("Describing a call tree sample"):
                    stack.enter_context(recorder.record_sample(args, kwargs))
                _run_inference(inference_function, args, kwargs)
    with coordinator.raise_if_any_rank_fails("Reading the call tree's structure"):
        return recorder.result()


def _time_call_tree(
    timer: TimingRecorder,
    inference_function: Callable,
    dataset: DatasetLike | DataLoaderFactory | torch.Tensor,
    number_of_iterations: int,
) -> None:
    """Time the tree's nodes in place, replaying each sample once per depth its calls reach."""
    depths_per_sample = timer.depths_per_sample()
    # Ranks must run the same number of inferences or the barriers in _run_inference deadlock.
    coordinator.verify_equal(depths_per_sample, "call tree timing depths")
    if not any(depths_per_sample):
        logger.info("Timing the call tree: no call is timeable, so the timing pass is skipped.")
        return
    logger.info(
        "Timing the call tree: %d samples, %d call-path depths, %d replays "
        "(each sample once per depth its calls reach).",
        number_of_iterations,
        len(set(itertools.chain.from_iterable(depths_per_sample))),
        sum(map(len, depths_per_sample)),
    )
    samples = samples_generator(dataset, [1], max_num_batches_per_batch_size=number_of_iterations)
    different_inputs: set[int] = set()
    with _pass_on_all_ranks(timer, "call tree timing pass"):
        for sample_index, (depths, (_, args, kwargs)) in enumerate(zip(depths_per_sample, samples, strict=True)):
            for depth in depths:
                with coordinator.raise_if_any_rank_fails("Timing a call tree replay"):
                    # checked before every replay: the inference can change its inputs in place; not timed
                    if describe_sample_inputs(args, kwargs) != timer.tree.samples[sample_index].inputs:
                        different_inputs.add(sample_index)
                    with timer.time_replay(depth, sample_index):
                        _run_inference(inference_function, args, kwargs)
    if different_inputs:
        logger.warning(
            "The timing pass read different inputs than the structure pass for %d of %d samples (first: sample "
            "%d), as a shuffling DataLoader, a shape-changing augmentation or an inference that changes its "
            "inputs in place does. Their per-sample times can belong to other inputs than CallTree.samples "
            "describes. Use a dataset that returns the same samples in the same order on every read, and an "
            "inference that leaves its inputs unchanged.",
            len(different_inputs),
            number_of_iterations,
            min(different_inputs),
        )


def _reset_total_execution_time(inspector: ModuleInspector):
    """Reset the timings of the modules."""
    for module in inspector.get_modules():
        module.total_execution_time = 0.0
        module.execution_count = 0


def _synchronize_inspection_result(
    total_execution_time: float,
    modules: list[ModuleInfo],
) -> tuple[float, list[ModuleInfo]]:
    """Order local modules and apply timings using rank zero's inspection result."""
    local_result = (
        total_execution_time,
        tuple(
            (
                module.object_path,
                f"{module.module_type.__module__}.{module.module_type.__qualname__}",
                module.execution_count,
                module.total_execution_time,
            )
            for module in modules
        ),
    )
    rank_zero_total, rank_zero_records = coordinator.broadcast_from_rank0(local_result)

    synchronized_modules = []
    with coordinator.raise_if_any_rank_fails("Synchronizing inspection result"):
        modules_by_path = {module.object_path: module for module in modules}
        rank_zero_paths = tuple(path for path, _, _, _ in rank_zero_records)
        local_paths = set(modules_by_path)
        authoritative_paths = set(rank_zero_paths)
        if local_paths != authoritative_paths:
            missing = tuple(path for path in rank_zero_paths if path not in local_paths)
            unexpected = tuple(
                module.object_path for module in modules if module.object_path not in authoritative_paths
            )
            raise RuntimeError(
                f"Local inspection modules differ from rank zero: missing={missing!r}, unexpected={unexpected!r}"
            )

        for path, module_type, execution_count, execution_time in rank_zero_records:
            module = modules_by_path[path]
            local_module_type = f"{module.module_type.__module__}.{module.module_type.__qualname__}"
            local_signature = (local_module_type, module.execution_count)
            rank_zero_signature = (module_type, execution_count)
            if local_signature != rank_zero_signature:
                raise RuntimeError(
                    f"Local inspection module {path!r} differs from rank zero: "
                    f"local={local_signature!r}, rank zero={rank_zero_signature!r}"
                )
            module.total_execution_time = execution_time
            synchronized_modules.append(module)

    return rank_zero_total, synchronized_modules
