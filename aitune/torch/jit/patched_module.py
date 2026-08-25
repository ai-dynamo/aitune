# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module for inspecting PyTorch models and tracking their execution."""

import hashlib
import logging
import random
from collections import Counter, OrderedDict, deque
from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from time import perf_counter
from typing import ClassVar, cast

import torch
import wrapt

from aitune.global_context import MODULE_CONTEXT_KEY, global_context
from aitune.torch.backend.backend import Backend
from aitune.torch.config import config as global_config
from aitune.torch.jit.config import JITMode, config
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.passthrough_module import PassthroughModule
from aitune.torch.module.recording_module import RecordingModule
from aitune.torch.module.sample_metadata import SampleMetadata
from aitune.torch.module.sample_store import Sample
from aitune.torch.module.tuned_module import TunedModule
from aitune.torch.tune_data.report_models import ModuleInspectionReport
from aitune.torch.tune_data.reporting import (
    report_graph_tune,
    report_inspection_details,
    report_module_tune,
    report_tune_run_end,
)
from aitune.torch.tune_strategy.mixin import FindMaxBatchSizeMixin
from aitune.torch.tune_strategy.tune_strategy import TuneStrategy
from aitune.torch.utils.graph_break_detector import GraphBreakDetector
from aitune.torch.utils.module import count_parameters, format_num_parameters, get_module_device, offload
from aitune.utils.disk_space import raise_if_out_of_space
from aitune.utils.monitoring import annotate

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
    fq_name_counter: ClassVar[Counter[str]] = Counter()  # for unique fully qualified names (base_fqn -> count)
    attempted_tuning: ClassVar[bool] = False
    deferred_tuning_enabled: ClassVar[bool] = False
    module_counter: ClassVar[int] = 0

    def __init__(self, module: torch.nn.Module):
        """Initialize the patched module."""
        self.__wrapped__ = module
        # proxy forward, hooks
        self._original_forward = module.forward
        self._current_forward_pre_hooks = module._forward_pre_hooks
        self._current_forward_hooks = module._forward_hooks
        # basic attributes
        self._name = module.__class__.__name__
        self._id = PatchedModule.module_counter
        PatchedModule.module_counter += 1
        # those attributes can't be resolved until first forward call
        self._call_count = 0
        self._level = -1
        self._parent: PatchedModule | None = None
        self._children: list[PatchedModule] = []
        self._allowed_to_tune = False
        self._fq_name: str | None = None
        # routing maps state to forward method implementation, to split large logic into smaller parts
        self._forward_routing = {
            ModuleState.INIT: self._forward_init,
            ModuleState.RECORDING: self._forward_recording,
            ModuleState.TUNED: self._forward_tuned,
        }
        self._extra_state_info: str = ""  # for tracking purposes
        self._update_state(ModuleState.INIT)

    @property
    def fq_name(self) -> str:
        """Get the fully qualified name of the module.

        This is guaranteed to be unique after first forward call.
        """
        if self._fq_name is None:
            raise ValueError("Fully qualified name is not available before first forward call.")
        return self._fq_name

    def try_tune(self):
        """Tune the module if readiness conditions for the active JIT mode are met.

        Delegates the readiness check to _should_be_tuned(), which dispatches to the
        mode-specific implementation (eager or deferred).
        """
        if self._should_be_tuned():
            self.tune()

    @annotate(name="tune", color="yellow")
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
        device = config.device or get_module_device(self.__wrapped__)
        if self._should_report_inspection():
            report_inspection_details(self.inspection_subtree_reports(), replace=False)

        todo = [self]
        while todo:
            current = todo.pop()
            recording = cast(RecordingModule, current._wrapper)
            backends: OrderedDict[SampleMetadata, Backend] = OrderedDict()
            strategy = _build_strategy()
            try:
                with report_module_tune(
                    module_name=current.__wrapped__.__class__.__name__,
                    num_parameters=count_parameters(current.__wrapped__),
                    module_id=current._id,
                ):
                    for graph_spec in recording.graph_specs:
                        cache_dir = self._create_graph_cache_dir(graph_spec.name)
                        data = recording.samples_for_graph_spec(graph_spec)
                        if config.dry_run:
                            self._simulate_dry_run(strategy, current, graph_spec, data, device, cache_dir)
                        else:
                            with global_context:
                                global_context.set(MODULE_CONTEXT_KEY, current.fq_name)
                                backend = self._tune(strategy, current, graph_spec, data, device, cache_dir)
                            backends[graph_spec.input_spec] = backend
                    if backends:
                        current._wrapper = TunedModule(
                            backends,
                            module_name=current.fq_name,
                            forward_signature=recording.forward_signature,
                        )
                        current._extra_state_info = ", ".join([b.name for b in backends.values()])
                    else:
                        current._wrapper = PassthroughModule(current.__wrapped__, device=device)
                        current._extra_state_info = "dry-run tuning success"

                    # tuning success, we can revert forward for current module, unpatch child modules
                    current._handle_backend_added_hooks()
                    current._update_state(ModuleState.TUNED)
                    current._patch_device_attribute(device)
                    current._unpatch_hierarchy(include_self=False)
                    _to_hist(f"Model tuned: {str(current)}")
                    self._offload(backends)

            except Exception as e:
                # Out-of-space errors must halt the process — the cache disk is full
                # and silently falling back to eager would mask the real cause.
                raise_if_out_of_space(e, path=global_config.cache_dir)
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
                    current._extra_state_info = "no better tuned version"
                _to_hist(f"Failed to tune module, unpatched: {str(current)}")

    def inspection_report(self) -> ModuleInspectionReport:
        """Build a report snapshot of the JIT data observed for this module."""
        return ModuleInspectionReport(
            module_id=self._id,
            module_name=self._fq_name,
            module_class=f"{self.__wrapped__.__class__.__module__}.{self.__wrapped__.__class__.__qualname__}",
            state=self._state.value,
            level=self._level,
            call_count=self._call_count,
            num_parameters=count_parameters(self.__wrapped__),
            allowed_to_tune=self._allowed_to_tune,
            dtypes=self._module_dtypes(),
            child_module_ids=[child._id for child in self._children],
            parent_module_id=self._parent._id if self._parent is not None else None,
            parent_module_name=self._parent._fq_name if self._parent is not None else None,
            graphs=self._inspection_graphs(),
        )

    def inspection_subtree_reports(self) -> list[ModuleInspectionReport]:
        """Build report snapshots for this module and observed child modules."""
        reports: list[ModuleInspectionReport] = []
        todo = [self]
        while todo:
            current = todo.pop()
            reports.append(current.inspection_report())
            todo.extend(reversed(current._children))
        return reports

    def _module_dtypes(self) -> list[str]:
        """Return unique parameter and buffer dtypes for this module tree."""
        dtypes = {str(tensor.dtype) for tensor in self.__wrapped__.parameters()}
        dtypes.update(str(tensor.dtype) for tensor in self.__wrapped__.buffers())
        return sorted(dtypes)

    def _inspection_graphs(self) -> list[dict]:
        """Return JSON-ready graph specs recorded before deferred tuning."""
        wrapper = getattr(self, "_wrapper", None)
        if not isinstance(wrapper, RecordingModule):
            return []
        return [
            {
                "graph_name": graph_spec.name,
                "input_spec": graph_spec.input_spec.to_json_dict(),
                "output_spec": graph_spec.output_spec.to_json_dict(),
            }
            for graph_spec in wrapper.graph_specs
        ]

    def _simulate_dry_run(
        self,
        strategy: TuneStrategy,
        module: "PatchedModule",
        graph_spec: GraphSpec,
        data: Sequence[Sample],
        device: torch.device,
        cache_dir: Path,
    ):
        """Simulate tuning in dry-run mode.

        It runs tune_dry_run but it also raises exception with probability of `config.dry_run_failure_probability`.

        This is done to simulate failures during JIT tuning to see how it would behave.
        """
        _to_hist(f"Tuning in dry-run module: {str(module)}")
        strategy.tune_dry_run(module.__wrapped__, module._name, graph_spec, data, device, cache_dir)
        if failure := random.random() < config.dry_run_failure_probability:
            raise RuntimeError(f"Tuning failed in dry-run mode with probability {failure:.2f}")

    def _tune(
        self,
        strategy: TuneStrategy,
        module: "PatchedModule",
        graph_spec: GraphSpec,
        data: Sequence[Sample],
        device: torch.device,
        cache_dir: Path,
    ):
        """Tune the module with optional graph break detection and timing."""
        with report_graph_tune(graph_spec, strategy) as graph_result:
            try:
                if config.detect_graph_breaks:
                    self._throw_if_has_graph_break(module, data)

                start = perf_counter()
                backend = strategy.tune(module.__wrapped__, module._name, graph_spec, data, device, cache_dir)
                duration = perf_counter() - start
                _to_hist(f"Tuning {module._name} took {duration:.2f}s")
                graph_result["selected_backend"] = backend.describe()
                return backend
            finally:
                graph_result["strategy_results"] = strategy.backend_results

    def _should_be_skipped(self):
        """Check if the module should be skipped.

        Name contains parameters, so only class names is relevant.
        """
        cls_name = self._name.split(" ")[0]
        return cls_name in config.skip_modules

    def _forward_router(self, wrapped, instance, args, kwargs):
        """Route forward calls to the appropriate state-specific handler.

        This method serves as a stable wrapper throughout the module's lifecycle.
        This design is necessary for compatibility with models that internally cache references
        to module.forward wrappers. Since the wrapper object itself never changes, cached
        references remain valid across state transitions.
        """
        forward_func = self._forward_routing.get(self._state)
        if forward_func is None:
            raise ValueError(f"Module should not be called in the state: {self._state}")
        return forward_func(wrapped, instance, args, kwargs)

    def _forward_init(self, wrapped, instance, args, kwargs):
        """Forward call for the first time.

        This method is called when the module is first called.
        It initializes the module and sets the state to RECORDING.
        """
        params = count_parameters(self.__wrapped__)
        if int(params) <= config.min_parameters:
            self._unpatch()
            return self._original_forward(*args, **kwargs)

        formatted_params = format_num_parameters(params)
        self._name += f" 📊{formatted_params}"
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
                return self._original_forward(*args, **kwargs)
        else:
            self._parent = None
            self._level = 0
            self._allowed_to_tune = True  # at the beginning only head modules are allowed
            PatchedModule.heads.append(self)
            _to_hist(f"New top module: {str(self)}")

        self._fq_name = self._get_fully_qualified_name()
        self._update_state(ModuleState.RECORDING)
        self._wrapper = RecordingModule(self.__wrapped__, self._name)
        PatchedModule.stack.append(self)
        try:
            self._restore_original_forward()
            result = self._wrapper(*args, **kwargs)
            self._proxy_forward()
        finally:
            PatchedModule.stack.pop()

        self._tune_on_init()

        return result

    def _tune_on_init(self):
        """Tune the module on init."""
        if config.mode == JITMode.TUNE_DEFERRED and not PatchedModule.deferred_tuning_enabled:
            return

        # if min_samples is greater than 1, we don't need to tune on init
        if config.mode == JITMode.TUNE_EAGER and config.min_samples > 1:
            return

        # if module is on skip list, we don't need to tune on init
        if self._should_be_skipped():
            self._update_state(ModuleState.SKIPPED)
            _to_hist(f"Module on skip list: {str(self)}")
            self._unpatch_hierarchy(include_self=True)
            return

        self.try_tune()

    def _forward_recording(self, wrapped, instance, args, kwargs):
        """Forward call for the recording state.

        In this state we already have the hierarchy resolved for the current module.
        If necessary we can skip the module and its children from processing.
        """
        if self._should_be_skipped():
            self._update_state(ModuleState.SKIPPED)
            _to_hist(f"Module on skip list: {str(self)}")
            self._unpatch_hierarchy(include_self=True)
            return self._original_forward(*args, **kwargs)

        self._restore_original_forward()
        result = self._wrapper(*args, **kwargs)
        self._proxy_forward()
        self._call_count += 1

        if config.mode == JITMode.TUNE_EAGER or (
            config.mode == JITMode.TUNE_DEFERRED and PatchedModule.deferred_tuning_enabled
        ):
            self.try_tune()
        return result

    @annotate(name="inference", color="green")
    def _forward_tuned(self, wrapped, instance, args, kwargs):
        """Forward call for the tuned state."""
        try:
            self._restore_original_forward()
            return self._wrapper(*args, **kwargs)
        finally:
            self._proxy_forward()

    def _create_graph_cache_dir(self, graph_spec_name: str) -> Path:
        """Create a cache directory for the graph."""
        cache_dir = config.cache_dir / self._get_hierarchy_hash() / graph_spec_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _get_hierarchy_hash(self) -> str:
        """Get the hash of the module hierarchy.

        Assumption: given name (actual class name + number of parameters) + id is unique enough to be used as a hash
        """
        todo = [self]
        hierarchy_names = ""
        while todo:
            current = todo.pop()
            hierarchy_names += current._name + str(current._id)
            todo.extend(current._children)
        return hashlib.sha256(hierarchy_names.encode()).hexdigest()[:10]

    def _get_fully_qualified_name(self) -> str:
        """Return a unique fully qualified name for this module in the hierarchy.

        Builds the path from root to this module by prepending the parent's already-computed
        fq_name (e.g. grand_parent_name.parent_name.name). An index suffix is only added when
        there is a duplicate path (same path seen more than once).
        """
        name = self.__wrapped__.__class__.__name__
        base_qualified = f"{self._parent.fq_name}.{name}" if self._parent is not None else name
        index = PatchedModule.fq_name_counter[base_qualified]
        PatchedModule.fq_name_counter[base_qualified] += 1
        return f"{base_qualified}.{index}" if index > 0 else base_qualified

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
        """Proxy the forward calls with a stable wrapper.

        Replaces torch.nn.Module.forward method with a proxy forward that routes to the appropriate handler.
        The substitution is done with a wrapt.decorator so that the replaced function has the same docstring,
        signature and other attributes. This is crucial as some HF models perform self inspection for method arguments.

        We re-enable hooks so that they are called before and after the proxied forward.
        """
        self.__wrapped__._forward_pre_hooks = self._current_forward_pre_hooks
        self.__wrapped__.forward = wrapt.decorator(self._forward_router)(self._original_forward)
        self.__wrapped__._forward_hooks = self._current_forward_hooks

    def _restore_original_forward(self):
        """Restore the original forward and hooks.

        We need to disable hooks, otherwise they will be called twice.
        """
        self._current_forward_pre_hooks = OrderedDict(self.__wrapped__._forward_pre_hooks)
        self._current_forward_hooks = OrderedDict(self.__wrapped__._forward_hooks)
        self.__wrapped__._forward_pre_hooks = OrderedDict()
        self.__wrapped__.forward = self._original_forward
        self.__wrapped__._forward_hooks = OrderedDict()

    def _handle_backend_added_hooks(self):
        """Handle new hooks if added by a backend.

        Before tuning hooks are cleared (empty OrderedDict). If there are new hooks added by a backend,
        they should be added to the existing ones but before the original hooks so that application layer hooks
        (like HF post processing hooks) are called after the backend hooks.
        """
        if self.__wrapped__._forward_pre_hooks:
            self._current_forward_pre_hooks = self.__wrapped__._forward_pre_hooks | self._current_forward_pre_hooks
        if self.__wrapped__._forward_hooks:
            self._current_forward_hooks = self.__wrapped__._forward_hooks | self._current_forward_hooks

    def _set_original_forward_for_hierarchy(self):
        """Set original forwards for all patched descendants in the module tree.

        The PatchedModule child hierarchy tracks modules observed during execution. It does
        not include registered submodules that were never called for the recorded samples
        (for example eval-only Identity drop-path modules). Backends receive the full
        nn.Module ownership tree, and some of them deepcopy/export that tree, so restore
        every patched descendant before handing the module to a backend.
        """
        from aitune.torch.jit.patcher import Patcher  # avoid circular deps

        for current in Patcher.patched_modules_under(self.__wrapped__):
            current._restore_original_forward()

    def _should_be_tuned(self) -> bool:
        """Check if the module should be tuned, dispatching based on the active JIT mode."""
        if config.mode == JITMode.TUNE_DEFERRED:
            return self._should_be_tuned_deferred()
        elif config.mode == JITMode.TUNE_EAGER:
            return self._should_be_tuned_eager()
        else:
            raise ValueError(f"Unsupported mode: {config.mode}")

    def _should_be_tuned_eager(self) -> bool:
        """Check if the module should be tuned in eager mode.

        Requires a minimum number of forward passes and, when dynamic shapes are expected,
        at least one sample with a detected dynamic axis.
        """
        if config.min_samples == 1:
            return self._allowed_to_tune

        if not self._allowed_to_tune:
            return False

        if self._call_count < config.min_samples:
            return False

        dynamic_axes_check = False
        if config.batch_axis_required:
            recording = cast(RecordingModule, self._wrapper)
            for graph_spec in recording.graph_specs:
                if graph_spec.input_spec.detected_dynamic_axis():
                    dynamic_axes_check = True
                    break
        else:
            dynamic_axes_check = True

        return dynamic_axes_check

    def _should_report_inspection(self) -> bool:
        """Return whether this tuning path should emit per-module inspection details."""
        return config.mode == JITMode.TUNE_EAGER and self._parent is None

    def _should_be_tuned_deferred(self) -> bool:
        """Check if the module should be tuned in deferred mode.

        The explicit enable_tune_deferred() call marks that recording is complete, so the
        next normal forward tunes eligible modules after recording that call.
        """
        return PatchedModule.deferred_tuning_enabled and self._allowed_to_tune and self._call_count >= 1

    def _throw_if_has_graph_break(self, module: "PatchedModule", data: Sequence[Sample]):
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

    def _offload(self, backends: OrderedDict[SampleMetadata, Backend]):
        """Offload the module to the meta device only if no JIT backends are used."""
        if not backends:
            return

        if any(backend.is_jit for backend in backends.values()):
            return

        with annotate("Offloading module to meta device"):
            offload(self.__wrapped__, device=global_config.device_after_tuning)

    def _unpatch(self):
        """Unpatch the module.

        Removes it also from Patcher object registry.
        """
        self._allowed_to_tune = False
        self._restore_original_forward()
        self.__wrapped__._forward_hooks = self._current_forward_hooks
        self.__wrapped__._forward_pre_hooks = self._current_forward_pre_hooks

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
        """Update the state of the module and its forward method.

        For states with custom forward handlers (INIT, RECORDING, TUNED), this method
        installs a proxy forward that routes to the appropriate handler. For other states,
        the original forward is restored.

        Args:
            state: The new state of the module
        """
        self._state = state
        if state in self._forward_routing:
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
        PatchedModule.deferred_tuning_enabled = False

    @staticmethod
    def on_python_exit():
        """Give suggestions if tuning was not attempted."""
        report_tune_run_end()

        if len(PatchedModule.heads) == 0:
            logging.error("JIT tuning has been enabled but no modules were found.")
        elif not PatchedModule.attempted_tuning:
            too_few_samples = [h._call_count < config.min_samples for h in PatchedModule.heads]
            if all(too_few_samples):
                logging.warning(
                    "JIT tuning has been enabled and requires at least %d samples. None of top level modules had enough samples.",
                    config.min_samples,
                )
            elif config.min_samples > 1 and config.batch_axis_required:
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


def _build_strategy() -> TuneStrategy:
    """Build the per-module tune strategy.

    Resolves the strategy via ``config.resolve_strategy()`` and clones it so each module gets
    a fresh instance — strategies hold per-tune state (``backend_results`` etc.) that must
    not leak across modules.

    Find-max-batch-size profiling is disabled because in JIT we cannot run the original
    module separately; the strategy must work from the recorded samples alone.
    """
    strategy = config.resolve_strategy().clone()

    if isinstance(strategy, FindMaxBatchSizeMixin):
        strategy.enable_find_max_batch_size(False)

    return strategy
