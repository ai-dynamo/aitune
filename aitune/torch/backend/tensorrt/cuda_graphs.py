# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CUDA graph capture and resource ownership for static TensorRT profiles."""

import logging
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from aitune.torch.backend.tensorrt.torch_output_allocator import TorchOutputAllocator

logger = logging.getLogger(__name__)

CudaGraphCachePolicy = Literal["lru", "lfu"]


@dataclass
class CudaGraphProfile:
    """Keep a captured context and its input/output storage alive together."""

    context: Any
    output_allocator: TorchOutputAllocator
    inputs: dict[str, torch.Tensor] = field(default_factory=dict)
    graph: Any = None


class TensorRTCudaGraphCache:
    """Cache static-profile graphs with LRU or aged LFU admission and eviction."""

    _DECAY_INTERVAL = 1024

    def __init__(self):
        """Start with an empty cache; configuration happens on backend activation."""
        self.profiles: OrderedDict[int, CudaGraphProfile] = OrderedDict()
        self.max_graphs = 8
        self.policy: CudaGraphCachePolicy = "lfu"
        self._frequencies: dict[int, int] = {}
        self._requests_since_decay = 0
        self.static_profile_indices: set[int] = set()
        self.active: CudaGraphProfile | None = None
        self.capture_failed = False

    def configure(
        self,
        engine,
        io_tensors,
        input_names,
        optimization_profiles,
        *,
        max_graphs: int = 8,
        policy: CudaGraphCachePolicy = "lfu",
    ):
        """Classify eligible profiles without allocating contexts or capturing graphs."""
        self.clear()
        self.max_graphs = max_graphs
        self.policy = policy
        # Fixed tensor dimensions do not imply fixed values for a shape tensor.
        if any(engine.is_shape_inference_io(name) for name in input_names):
            return
        if optimization_profiles:
            self.static_profile_indices = {
                index
                for index, profile in enumerate(optimization_profiles)
                if set(profile) == set(input_names)
                and all(min_ == opt == max_ and all(dim >= 0 for dim in min_) for min_, opt, max_ in profile.values())
            }
        elif all(all(dim >= 0 for dim in io_tensors[name]["shape"]) for name in input_names):
            self.static_profile_indices = {0}

    def is_eligible(self, index: int) -> bool:
        """Return whether a profile can use graphs for this backend instance."""
        return not self.capture_failed and index in self.static_profile_indices

    def select(
        self, index: int, engine, stream, create_output_allocator: Callable[[Any], TorchOutputAllocator]
    ) -> CudaGraphProfile | None:
        """Select a cached profile, or return None to execute without capture when admission is denied."""
        self.active = None
        if self.policy == "lfu":
            self._record_access(index)
        if index not in self.profiles:
            if not self._admit(index, stream):
                return None
            context = engine.create_execution_context()
            if context is None:
                raise RuntimeError(f"Failed to create TensorRT execution context for profile {index}")
            if not context.set_optimization_profile_async(index, stream.cuda_stream):
                raise RuntimeError(f"TensorRT rejected optimization profile {index}")
            self.profiles[index] = CudaGraphProfile(context, create_output_allocator(context))
        self.profiles.move_to_end(index)
        self.active = self.profiles[index]
        return self.active

    def copy_input(self, name: str, tensor: torch.Tensor, stream) -> torch.Tensor:
        """Copy input data into stable storage on the inference stream."""
        assert self.active is not None
        inputs = self.active.inputs
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            if name not in inputs:
                inputs[name] = torch.empty_like(tensor)
            static_tensor = inputs[name]
            if static_tensor.shape != tensor.shape:
                raise RuntimeError(f"Input {name!r} does not match the captured profile shape")
            static_tensor.copy_(tensor)
        return static_tensor

    def execute(self, stream):
        """Capture on the first request, then replay the selected profile's graph."""
        assert self.active is not None
        if self.active.graph is None:
            self._capture(self.active, stream)
        else:
            self.active.graph.replay()

    def clear(self):
        """Release graphs before their resources, after the caller synchronizes execution."""
        self.active = None
        for profile in self.profiles.values():
            profile.graph = None
            profile.inputs.clear()
        self.profiles.clear()
        self.static_profile_indices.clear()
        self._frequencies.clear()
        self._requests_since_decay = 0

    def _record_access(self, index: int):
        """Count cached and uncached requests, halving history every 1024 eligible requests."""
        self._requests_since_decay += 1
        if self._requests_since_decay == self._DECAY_INTERVAL:
            self._frequencies = {key: count // 2 for key, count in self._frequencies.items()}
            self._requests_since_decay = 0
        self._frequencies[index] = self._frequencies.get(index, 0) + 1

    def _admit(self, index: int, stream) -> bool:
        """Admit into free space, or replace a victim only when the policy allows it."""
        if len(self.profiles) < self.max_graphs:
            return True
        victim = next(iter(self.profiles))
        if self.policy == "lfu":
            # OrderedDict iteration breaks equal-frequency victim ties by least recent use.
            victim = min(self.profiles, key=lambda key: self._frequencies.get(key, 0))
            if self._frequencies[index] <= self._frequencies.get(victim, 0):
                return False
        self._evict(victim, stream)
        return True

    def _evict(self, index: int, stream):
        """Finish pending work before freeing an evicted graph's resources."""
        stream.synchronize()
        profile = self.profiles.pop(index)
        if self.active is profile:
            self.active = None
        profile.graph = None
        profile.inputs.clear()
        profile.output_allocator.clear()
        logger.debug("Evicted CUDA graph for TensorRT profile %d", index)

    def _capture(self, profile: CudaGraphProfile, stream):
        """Warm up, capture, and replay; fall back only when capture itself fails."""
        if not profile.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed during CUDA graph setup")

        try:
            profile.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(profile.graph, stream=stream):
                if not profile.context.execute_async_v3(stream.cuda_stream):
                    raise RuntimeError("TensorRT execution failed during CUDA graph capture")
        except Exception as error:
            self.capture_failed = True
            profile.graph = None
            logger.warning("CUDA graph capture failed; continuing without CUDA graphs: %s", error)
            # Keep static inputs alive until the backend synchronizes fallback execution.
            if not profile.context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError("TensorRT execution failed") from error
            return

        profile.graph.replay()
