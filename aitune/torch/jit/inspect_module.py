# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module for inspecting PyTorch models and tracking their execution."""

import logging
from collections import OrderedDict, deque
from datetime import datetime
from enum import Enum
from pathlib import Path
from time import perf_counter
from typing import Any, ClassVar

import torch
import wrapt

from aitune.torch.jit.config import config
from aitune.torch.jit.html_generator import HTMLGenerator
from aitune.torch.utils.cuda_utils import synchronize as cuda_synchronize
from aitune.torch.utils.module import count_parameters

PRINT_HIERARCHY_HEADER = "JIT Tuning Hierarchy:"
PRINT_HIERARCHY_NO_MODULES_HEADER = "No modules in hierarchy"


class ModuleState(Enum):
    """Possible states of the Module class."""

    INIT = "init"
    INSPECT = "inspect"
    SKIPPED = "skipped"
    DETACHED = "detached"


_emoji_state = {
    ModuleState.INIT: "⏳",
    ModuleState.INSPECT: "🔍",
    ModuleState.SKIPPED: "🚫",
    ModuleState.DETACHED: "☑️",
}


class InspectModule:
    """Inspect module."""

    stack: ClassVar[deque["InspectModule"]] = deque()
    heads: ClassVar[list["InspectModule"]] = []  # the top level modules
    history: ClassVar[list[str]] = []  # for tracking purposes
    report_saved: ClassVar[bool] = False

    def __init__(self, module: torch.nn.Module):
        """Initialize the patched module."""
        self.__wrapped__ = module
        # proxy forward
        self._original_forward = module.forward
        self._original_forward_pre_hooks = module._forward_pre_hooks
        self._original_forward_hooks = module._forward_hooks
        # basic attributes
        self._name = module.__class__.__name__
        self._name_detailed = module.__class__.__module__ + "." + module.__class__.__name__
        # those attributes can't be resolved until first forward call
        self._call_count = 0
        self._level = -1
        self._parent: InspectModule | None = None
        self._children: list[InspectModule] = []
        # execution info
        self._total_execution_time: float = 0.0
        self._args: list[str] = []
        self._kwargs: dict[str, str] = {}
        self._output_types: list[str] | tuple[str] | dict[str, str] | str = None
        self._has_graph_breaks: bool = False

        # routing maps state to forward method implementation, to split large logic into smaller parts
        self._forward_routing = {
            ModuleState.INIT: self._forward_init,
            ModuleState.INSPECT: self._forward_inspect,
        }
        self._update_state(ModuleState.INIT)

    def _should_be_skipped(self):
        """Check if the module should be skipped.

        Name contains parameters, so only class names is relevant.
        """
        cls_name = self._name.split(" ")[0]
        return cls_name in config.skip_modules

    def _forward_init(self, wrapped, instance, args, kwargs):
        """Forward call for the first time.

        This method is called when the module is first called.
        It initializes the module and sets the state to INSPECT.
        """
        params = count_parameters(self.__wrapped__)
        self._name += f" 📊{params}"
        if self._should_be_skipped():
            self._unpatch()
            return self.__wrapped__(*args, **kwargs)

        if InspectModule.stack:
            parent = InspectModule.stack[-1]
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
            InspectModule.heads.append(self)
            _to_hist(f"New top module: {str(self)}")

        self._update_state(ModuleState.INSPECT)
        InspectModule.stack.append(self)
        try:
            result = self._forward_inspect(wrapped, instance, args, kwargs)
        finally:
            InspectModule.stack.pop()
        return result

    def _forward_inspect(self, wrapped, instance, args, kwargs):
        """Forward call for the inspect time."""
        self._call_count += 1
        self._gather_args_kwargs(args, kwargs)
        self._restore_original_forward()
        try:
            cuda_synchronize()
            start_time = perf_counter()
            result = self.__wrapped__(*args, **kwargs)
            cuda_synchronize()
            end_time = perf_counter()
        finally:
            self._proxy_forward()

        self._total_execution_time += end_time - start_time
        self._gather_output_types(result)

        return result

    def _gather_args_kwargs(self, args, kwargs):
        """Gather args and kwargs."""
        self._args = []
        self._kwargs = {}
        for arg in args:
            self._args.append(get_type_info(arg))
        for k, v in kwargs.items():
            self._kwargs[k] = get_type_info(v)

    def _gather_output_types(self, result: Any):
        """Gather output types."""
        if isinstance(result, tuple | list):
            self._output_types = []
            for x in result:
                self._output_types.append(get_type_info(x))
            if isinstance(result, tuple):
                self._output_types = tuple(self._output_types)
        elif isinstance(result, dict):
            self._output_types = {}
            for k, v in result.items():
                self._output_types[k] = get_type_info(v)
        else:
            self._output_types = get_type_info(result)

    def __repr__(self):
        """Representation of the module."""
        return self.__str__()

    def __str__(self):
        """String representation of the module."""
        result = f"{self._name} level={self._level}🪜 "
        result += f"state={self._state.value}{_emoji_state[self._state]} "
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

    def _unpatch(self):
        """Unpatch the module.

        Removes it also from Patcher object registry.
        """
        self._restore_original_forward()

        from aitune.torch.jit.patcher import Patcher  # avoid circular deps

        Patcher.unpatch_module(self)

    def _update_state(self, state: ModuleState):
        """Update the state of the module and update the forward.

        Replaces torch.nn.Module.forward method with a proxy forward according to the object state. The substitution
        is done with a wrapt.decorator so that the replaced function has same docstring, signature and other attributes.
        This is crucial as some HF models perform self inspection for method arguments.
        """
        self._state = state
        self.__wrapped__.forward = self._original_forward  # revert original forward to be ready for decoration
        if replacement_func := self._forward_routing.get(state):
            self._proxy_forward_func = wrapt.decorator(replacement_func)(self.__wrapped__.forward)
            self._proxy_forward()

    @staticmethod
    def print_hierarchy(sink=print):
        """Prints the PatchedModule hierarchy starting from the head module.

        This method traverses the module tree starting from the head module
        and prints each module with indentation to show the hierarchy levels.
        """
        if len(InspectModule.heads) == 0:
            sink(PRINT_HIERARCHY_NO_MODULES_HEADER)
            return

        def _print_module(module: "InspectModule", level: int = 0):
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
        for head in InspectModule.heads:
            _print_module(head)

    @staticmethod
    def print_history(sink=print):
        """Prints history to the sink."""
        sink("\n".join(InspectModule.history))

    @staticmethod
    def reset():
        """Reset PatchedModule state."""
        InspectModule.history.clear()
        InspectModule.heads.clear()
        InspectModule.stack.clear()

    @staticmethod
    def save_report(path: str | Path, model_name: str = "Model"):
        """Save module hierarchy to interactive HTML with folding/unfolding capabilities.

        Creates a comprehensive HTML page that displays the module hierarchy with:
        - Collapsible sections for each module
        - Full module details including name, args, kwargs, result types, call count
        - Modern styling with animations
        - Interactive JavaScript for folding/unfolding
        - Toggle button for switching between short and detailed type information

        Args:
            path: Path where the HTML file should be saved.
            model_name: The name of the model to display in the HTML subtitle.
        """
        html_content = HTMLGenerator.generate_html_content(InspectModule, model_name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_content)

        InspectModule.report_saved = True

    @staticmethod
    def on_python_exit():
        """Print hierarchy and save report if not saved yet when process is exiting."""
        InspectModule.print_hierarchy()
        if len(InspectModule.heads) > 0 and not InspectModule.report_saved:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"jit_inspect_{timestamp}.html"

            InspectModule.save_report(filename, "Unknown")
            logging.warning("Since you didn't save the report, it was saved to: %s.", filename)


def _to_hist(entry: str):
    """Updates history with new entry."""
    InspectModule.history.append(entry)


def get_type_info(obj: Any) -> str:
    """Get the type information of an object."""
    obj_type = type(obj)
    return f"{obj_type.__module__}.{obj_type.__name__}"
