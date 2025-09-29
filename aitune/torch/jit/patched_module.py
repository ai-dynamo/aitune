# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module for inspecting PyTorch models and tracking their execution."""

import hashlib
import logging
import random
from collections import Counter, OrderedDict, deque
from enum import Enum
from pathlib import Path
from time import perf_counter
from typing import ClassVar, cast

import torch
import wrapt

from aitune.torch.backend.backend import Backend
from aitune.torch.backend.tensorrt.tensorrt_backend import TensorRTBackend
from aitune.torch.jit.config import config
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.passthrough_module import PassthroughModule
from aitune.torch.module.recording_module import RecordingModule, Sample
from aitune.torch.module.sample_metadata import SampleMetadata, UnsupportedTypeException
from aitune.torch.module.tuned_module import TunedModule
from aitune.torch.tune_strategy.jit_strategy import JitStrategy
from aitune.torch.utils.graph_break_detector import GraphBreakDetector
from aitune.torch.utils.module_utils import count_parameters

PRINT_HIERARCHY_HEADER = "JIT Tuning Hierarchy:"
PRINT_HIERARCHY_NO_MODULES_HEADER = "No modules in hierarchy"


class ModuleState(Enum):
    """Possible states of the Module class."""

    INIT = "init"  # before first forward call, model hierarchy is not fully resolved
    RECORDING = "recording"  # after first forward call, model hierarchy is fully resolved, but not tuned
    TUNED = "tuned"  # model is tuned
    EAGER = "eager"  # model cannot be tuned, moved to eager mode
    SKIPPED = "skipped"  # module is skipped, not tuned
    DETACHED = "detached"  # module is detached i.e. its parent was tuned hence children could be detached


_emoji_state = {
    ModuleState.INIT: "⏳",
    ModuleState.RECORDING: "🔴",
    ModuleState.TUNED: "🎯",
    ModuleState.EAGER: "⚠️",
    ModuleState.SKIPPED: "🚫",
    ModuleState.DETACHED: "☑️",
}


class GraphBreakException(Exception):
    """Exception raised when graph break is detected."""

    pass


class PatchedModule:
    """Patched module.

    Intercepts torch.nn.Module's forward function to record data, tune module and to do inference on a tuned one.

    As opposed to the AITune declarative approach, where the user specifies which module should be tuned,
    JIT approach intercepts all modules on the fly and creates call hierarchy. After a few calls it can perform
    tuning. JIT is actually a bit harder case, since during navigation of the model hierarchy we are not aware
    of the real parent object (there could be python objects in between). Hence we cannot override parent object
    reference meaning we can't inject module as a proxy. This is why only forward function is overridden and
    device attribute is patched with dynamic class creation so that only instance attribute is changed.

    """

    stack: ClassVar[deque["PatchedModule"]] = deque()
    heads: ClassVar[list["PatchedModule"]] = []  # the top level modules
    history: ClassVar[list[str]] = []  # for tracking purposes
    patched_classes: ClassVar[Counter[str]] = Counter()  # for generating unique names
    attempted_tuning: ClassVar[bool] = False

    def __init__(self, module: torch.nn.Module):
        """Initialize the patched module."""
        self.__wrapped__ = module
        # proxy forward, hooks
        self._original_forward = module.forward
        self._original_forward_pre_hooks = module._forward_pre_hooks
        self._original_forward_hooks = module._forward_hooks
        # basic attributes
        self._name = module.__class__.__name__
        # those attributes can't be resolved until first forward call
        self._call_count = 0
        self._level = -1
        self._parent: PatchedModule | None = None
        self._children: list[PatchedModule] = []
        self._allowed_to_tune = False
        # routing maps state to forward method implementation, to split large logic into smaller parts
        self._forward_routing = {
            ModuleState.INIT: self._forward_init,
            ModuleState.RECORDING: self._forward_recording,
            ModuleState.TUNED: self._forward_tuned,
        }
        self._extra_state_info: str = ""  # for tracking purposes
        self._update_state(ModuleState.INIT)

    def tune(
        self,
    ):
        """Tunes the module in blocking mode.

        Args:
            dry_run: if True, only dry run the tuning

        Note:
            if module has already defined strategy/strategies those will take precedence over the provided one.
        """
        PatchedModule.attempted_tuning = True
        self._set_original_forward_for_hierarchy()
        device = torch.device("cuda")
        todo = [self]
        while todo:
            current = todo.pop()
            recording = cast(RecordingModule, current._wrapper)
            backends: OrderedDict[SampleMetadata, Backend] = OrderedDict()
            strategy = JitStrategy(TensorRTBackend())
            try:
                for graph_spec in recording.graph_specs:
                    cache_dir = self._create_graph_cache_dir(graph_spec)
                    data = recording.samples_for_graph_spec(graph_spec)
                    if config.dry_run:
                        self._simulate_dry_run(strategy, current, graph_spec, data, device, cache_dir)
                    else:
                        backend = self._tune(device, current, strategy, graph_spec, cache_dir, data)
                        backends[graph_spec.input_spec] = backend
                if backends:
                    current._wrapper = TunedModule(backends)
                    current._extra_state_info = ", ".join([b.name for b in backends.values()])
                else:
                    current._wrapper = PassthroughModule(current.__wrapped__)
                    current._extra_state_info = "dry-run tuning success"

                # tuning success, we can revert forward for current module, unpatch child modules
                current._update_state(ModuleState.TUNED)
                current._patch_device_attribute(device)
                current._unpatch_hierarchy(include_self=False)
                _to_hist(f"Model tuned: {str(current)}")
            except Exception as e:
                current.__wrapped__.to(device)  # failed backend leaves module on cpu, move it back
                current._unpatch()
                current._update_state(ModuleState.EAGER)
                for child in current._children:
                    child._allowed_to_tune = not child._should_be_skipped()  # child modules are allowed to tune
                    if child._should_be_tuned():
                        todo.append(child)
                if isinstance(e, GraphBreakException):
                    current._extra_state_info = "graph break"
                else:
                    current._extra_state_info = "tuning error"
                _to_hist(f"Failed to tune module, unpatched: {str(current)}")

    def _simulate_dry_run(self, strategy, current, graph_spec, data, device, cache_dir):
        """Simulate tuning in dry-run mode.

        It runs tune_dry_run but it also raises exception with probability of `config.dry_run_failure_probability`.

        This is done to simulate failures during JIT tuning to see how it would behave.
        """
        _to_hist(f"Tuning in dry-run module: {str(current)}")
        strategy.tune_dry_run(current.__wrapped__, current._name, graph_spec, data, device, cache_dir)
        if failure := random.random() < config.dry_run_failure_probability:
            raise Exception(f"Tuning failed in dry-run mode with probability {failure:.2f}")

    def _tune(self, device, current, strategy, graph_spec, cache_dir, data):
        """Tune the module.

        It throws exception if graph break is detected.
        """
        if config.detect_graph_breaks:
            self._throw_if_has_graph_break(current, data)
        backend = self._tune_module(strategy, current, graph_spec, data, device, cache_dir)
        return backend

    def _should_be_skipped(self):
        """Check if the module should be skipped.

        Name contains parameters, so only class names is relevant.
        """
        cls_name = self._name.split(" ")[0]
        return cls_name in config.skip_modules

    def _forward_init(self, wrapped, instance, args, kwargs):
        """Forward call for the first time.

        This method is called when the module is first called.
        It initializes the module and sets the state to RECORDING.
        """
        params = count_parameters(self.__wrapped__)
        if params == "0":
            self._unpatch()
            return self.__wrapped__(*args, **kwargs)

        self._name += f" 📊{params}"
        self._call_count = 1
        if PatchedModule.stack:
            parent = PatchedModule.stack[-1]
            if parent._level + 1 < config.max_depth_level:
                parent._children.append(self)
                self._parent = parent
                self._level = parent._level + 1
                _to_hist(f"New child module: {str(self)}")
            else:
                # too deep, skip the module
                self._unpatch()
                return self.__wrapped__(*args, **kwargs)
        else:
            self._parent = None
            self._level = 0
            self._allowed_to_tune = True  # at the beginning only head modules are allowed
            PatchedModule.heads.append(self)
            _to_hist(f"New top module: {str(self)}")

        self._update_state(ModuleState.RECORDING)
        self._wrapper = RecordingModule(self.__wrapped__, self._name)
        PatchedModule.stack.append(self)
        try:
            self._restore_original_forward()
            result = self._wrapper(*args, **kwargs)
            self._proxy_forward()
        except UnsupportedTypeException as e:
            self._handle_unsupported_type(e)
            result = self.__wrapped__(*args, **kwargs)  # return original result
            # since cannot record parent, let child modules be allowed to tune
            # the order is important - line above will call and register child modules
            for child in self._children:
                child._allowed_to_tune = True
        finally:
            PatchedModule.stack.pop()
        return result

    def _forward_recording(self, wrapped, instance, args, kwargs):
        """Forward call for the recording state.

        In this state we already have the hierarchy resolved for the current module.
        If necessary we can skip the module and its children from processing.
        """
        if self._should_be_skipped():
            self._update_state(ModuleState.SKIPPED)
            _to_hist(f"Module on skip list: {str(self)}")
            self._unpatch_hierarchy(include_self=True)
            return self.__wrapped__(*args, **kwargs)

        try:
            self._restore_original_forward()
            result = self._wrapper(*args, **kwargs)
            self._proxy_forward()
            self._call_count += 1
        except UnsupportedTypeException as e:
            self._handle_unsupported_type(e)
            result = self.__wrapped__(*args, **kwargs)  # return original result
            # since cannot record parent, let child modules be allowed to tune
            for child in self._children:
                child._allowed_to_tune = True

        if self._should_be_tuned():
            self.tune()
        return result

    def _forward_tuned(self, wrapped, instance, args, kwargs):
        """Forward call for the tuned state."""
        try:
            self._restore_original_forward()
            return self._wrapper(*args, **kwargs)
        finally:
            self._proxy_forward()

    def _create_graph_cache_dir(self, graph_spec: GraphSpec) -> Path:
        """Create a cache directory for the graph."""
        cache_dir = config.cache_dir / self._get_hierarchy_hash() / graph_spec.name
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _get_hierarchy_hash(self) -> str:
        """Get the hash of the module hierarchy."""
        todo = [self]
        hierarchy_names = ""
        while todo:
            current = todo.pop()
            hierarchy_names += current._name
            todo.extend(current._children)
        return hashlib.sha256(hierarchy_names.encode()).hexdigest()

    def _handle_unsupported_type(self, e: UnsupportedTypeException):
        """Handle unsupported type exception."""
        self._update_state(ModuleState.EAGER)
        self._unpatch()
        self._extra_state_info = f"Unsupported type: {e.wrong_type.__module__}.{e.wrong_type.__name__}"
        _to_hist(f"Cannot record module: {self._name}, {self._extra_state_info}")

    def _patch_device_attribute(self, device: torch.device):
        """Patch the device attribute of the module and its parents.

        HF pipelines have `device` attribute which checks `next(module.parameters()).device`.
        Since JIT tuning moves the original module to a cpu device, this attribute returns cpu device.
        This causes misplacement issues where some modules are on cuda and other modules or tensors are moved according
        to the device attribute to cpu. To fix this, the method patches dynamically device attribute but only at instance
        (object) level i.e. does not touch class object. This has to be applied not only to the current module but to
        its parents in case parents objects do not have it's own parameters but include child module which has been
        tuned and move to a cpu device.
        """
        todo = [self]
        while todo:
            current = todo.pop()
            name = current._name
            counter = PatchedModule.patched_classes[name]
            # create dynamically a subclass with device attribute overridden
            patched_class = type(
                f"Patched_{name}_{counter}",
                (current.__wrapped__.__class__,),
                {"device": device},
            )  # noqa: N806
            PatchedModule.patched_classes[name] += 1
            # replace class so that python dynamically invokes patched device attribute
            current.__wrapped__.__class__ = patched_class
            if current._parent:
                todo.append(current._parent)

    def __repr__(self):
        """Representation of the module."""
        return self.__str__()

    def __str__(self):
        """String representation of the module."""
        result = f"{self._name} level={self._level}🪜 "
        result += f"state={self._state.value}{_emoji_state[self._state]} "
        if self._extra_state_info:
            result += f"({self._extra_state_info}) "
        result += f"call_count={self._call_count}"
        return result

    def _proxy_forward(self):
        """Proxy the forward calls.

        We need to re-enable hooks, so that they will be called before and after proxied forward.
        """
        self.__wrapped__._forward_pre_hooks = self._original_forward_pre_hooks
        self.__wrapped__.forward = self._proxy_forward_func
        self.__wrapped__._forward_hooks = self._original_forward_hooks

    def _restore_original_forward(self):
        """Restore the original forward and hooks.

        We need to disable hooks, otherwise they will be called twice.
        """
        self.__wrapped__._forward_pre_hooks = OrderedDict()
        self.__wrapped__.forward = self._original_forward
        self.__wrapped__._forward_hooks = OrderedDict()

    def _set_original_forward_for_hierarchy(self):
        """Set the original forward method for the module and its children."""
        todo = [self]
        while todo:
            current = todo.pop()
            current._restore_original_forward()
            todo.extend(current._children)

    def _should_be_tuned(self):
        """Check if the module should be tuned."""
        min_samples_check = self._call_count >= config.min_samples

        dynamic_axes_check = False
        if config.batch_axis_required:
            recording = cast(RecordingModule, self._wrapper)
            for graph_spec in recording.graph_specs:
                if graph_spec.input_spec.detected_dynamic_axis():
                    dynamic_axes_check = True
                    break
        else:
            dynamic_axes_check = True

        return min_samples_check and self._allowed_to_tune and dynamic_axes_check

    def _throw_if_has_graph_break(self, module: "PatchedModule", data: list[Sample]):
        """Throw if graph break is detected.

        Adds to history the duration of the checking.
        """
        detector = GraphBreakDetector()
        args, kwargs = data[0]
        start = perf_counter()
        detector.detect(module.__wrapped__, args, kwargs)
        duration = perf_counter() - start
        if detector.has_graph_breaks():
            message = f"Graph break detected in {module._name} in {duration:.2f}s"
            _to_hist(message)
            raise GraphBreakException(message)
        else:
            _to_hist(f"No graph breaks in {module._name}. Checking took {duration:.2f}s")

    def _tune_module(
        self,
        strategy,
        module: "PatchedModule",
        graph_spec: GraphSpec,
        data: list[Sample],
        device: torch.device,
        cache_dir: Path,
    ):
        """Tune the module.

        Adds to history the duration of the tuning.
        """
        start = perf_counter()
        backend = strategy.tune(module.__wrapped__, module._name, graph_spec, data, device, cache_dir)
        duration = perf_counter() - start
        _to_hist(f"Tuning {module._name} took {duration:.2f}s")
        return backend

    def _unpatch(self):
        """Unpatch the module.

        Removes it also from Patcher object registry.
        """
        self._allowed_to_tune = False
        self._restore_original_forward()

        from aitune.torch.jit.patcher import Patcher  # avoid circular deps

        Patcher.unpatch_module(self)

    def _unpatch_hierarchy(self, include_self=False):
        """Unpatch the module and all its children."""
        if include_self:
            _to_hist(f"Unpatching module: {str(self)}")
            self._unpatch()
        to_unpatch = self._children[:]
        while to_unpatch:
            child = to_unpatch.pop()
            child._update_state(ModuleState.DETACHED)
            _to_hist(f"Unpatching child module: {str(child)}")
            child._unpatch()
            to_unpatch.extend(child._children)

    def _update_state(self, state: ModuleState):
        """Update the state of the module and update the forward.

        Replaces torch.nn.Module.forward method with a proxy forward according to the object state. The substitution
        is done with a wrapt.decorator so that the replaced function has same docstring, signature and other attributes.
        This is crucial as some HF models perform self inspection for method arguments.
        """
        self._state = state
        self._restore_original_forward()
        if replacement_func := self._forward_routing.get(state):
            self._proxy_forward_func = wrapt.decorator(replacement_func)(self.__wrapped__.forward)
            self._proxy_forward()

    @staticmethod
    def print_hierarchy(sink=print):
        """Prints the PatchedModule hierarchy starting from the head module.

        This method traverses the module tree starting from the head module
        and prints each module with indentation to show the hierarchy levels.
        """
        if len(PatchedModule.heads) == 0:
            sink(PRINT_HIERARCHY_NO_MODULES_HEADER)
            return

        def _print_module(module: "PatchedModule", level: int = 0):
            """Recursively print module and its children.

            Args:
                module: The module to print
                level: The current indentation level
            """
            indent = "  " * level
            sink(f"{indent}├─ {str(module)}")

            for child in module._children:
                _print_module(child, level + 1)

        sink(PRINT_HIERARCHY_HEADER)
        for head in PatchedModule.heads:
            _print_module(head)

    @staticmethod
    def print_history(sink=print):
        """Prints history to the sink."""
        sink("\n".join(PatchedModule.history))

    @staticmethod
    def reset():
        """Reset PatchedModule state."""
        PatchedModule.history.clear()
        PatchedModule.heads.clear()
        PatchedModule.stack.clear()

    @staticmethod
    def on_python_exit():
        """Give suggestions if tuning was not attempted."""
        if len(PatchedModule.heads) == 0:
            logging.error("JIT tuning has been enabled but no modules were found.")
        elif not PatchedModule.attempted_tuning:
            to_few_samples = [h._call_count < config.min_samples for h in PatchedModule.heads]
            if all(to_few_samples):
                logging.warning(
                    "JIT tuning has been enabled and requires at least %d samples. None of top level modules had enough samples.",
                    config.min_samples,
                )
            elif config.batch_axis_required:
                logging.warning(
                    """
JIT tuning has been enabled and requires two different batch sizes to detect dynamic axes.
This was not the case. You can either turn this requirement off with:

from aitune.torch.jit.config import config
config.batch_axis_required = False

or provide two different batch sizes when calling the model.
""",
                )


def _to_hist(entry: str):
    """Updates history with new entry."""
    PatchedModule.history.append(entry)
