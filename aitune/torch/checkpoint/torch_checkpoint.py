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
"""Torch checkpoint module."""

from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch.nn import Module as TorchModule

from aitune.torch.checkpoint.checkpoint import Checkpoint
from aitune.torch.checkpoint.storage import Storage
from aitune.torch.module.wrapper_module import Module as WrapperModule
from aitune.torch.utils.device import get_device


class TorchCheckpoint(Checkpoint):
    """Class for storing tuned module using torch module's state serialization.

    It uses torch `state_dict` and `load_state_dict` to save and load the module state.
    """

    def __init__(self, storage: Storage):
        """Initialize the TorchCheckpoint.

        Args:
            storage: The storage to use for saving and loading the state dictionary.
        """
        self._storage = storage

    def save(
        self,
        module_or_pipeline: TorchModule | Any,
        path: str | Path,
    ) -> None:
        """Save the module or pipeline state to the specified path.

        This method handles both individual torch modules and pipelines containing modules.
        For individual modules, it directly uses the module's state_dict() method.
        For pipelines, it extracts state dictionaries from all tuned modules in the pipeline.

        The pipeline is expected to be an object with attributes that are tuned modules (WrapperModule instances).
        Only tuned modules are saved in the state dictionary, as regular torch modules don't have tuning information.

        If no tuned modules are found in the pipeline, a ValueError is raised.

        Args:
            module_or_pipeline: The module or pipeline to save. This can be either a torch.nn.Module
                or a pipeline object with module attributes. For pipelines, only tuned modules
                (WrapperModule instances) will be saved.
            path: The path where the module state should be saved.
        """
        if isinstance(module_or_pipeline, TorchModule):
            state_dict = module_or_pipeline.state_dict()
        else:
            state_dict = TorchCheckpoint.state_dict_from_pipeline(module_or_pipeline)
        self._storage.save(path, state_dict)

    def load(
        self,
        module_or_pipeline: TorchModule | Any,
        path: str | Path,
        device_map: dict[str, str | torch.device] | None = None,
    ) -> TorchModule | WrapperModule:
        """Load a module or pipeline from a saved state dictionary.

        This method handles both individual torch modules and pipelines containing modules.
        For individual modules, it uses the module's load_state_dict() method and handles
        any tuned modules appropriately.

        For pipelines, it loads state dictionaries into the corresponding modules in the pipeline.
        The pipeline is expected to be an object with attributes that are torch modules or
        tuned modules (WrapperModule instances).

        Args:
            module_or_pipeline: The module or pipeline to load the state into. This can be
                either a torch.nn.Module or a pipeline object with module attributes.
            path: The path from where the module state should be loaded.
            device_map: The device map to load module to.

        Returns:
            The loaded module, which may be either the original module type or a wrapped module
            if the saved state contains tuning information.

        Raises:
            ValueError: If a module in the state_dict is not found in the pipeline or
                        if a matched attribute is not a torch.nn.Module or WrapperModule.

        """
        # Convert device_map to torch.device
        device_map = {k: get_device(v) for k, v in (device_map or {}).items()}

        state_dict = self._storage.load(path)
        if isinstance(module_or_pipeline, TorchModule):
            module_or_pipeline = TorchCheckpoint.load_state_dict_for_module(module_or_pipeline, state_dict, device_map)
        else:
            module_or_pipeline = TorchCheckpoint.load_state_dict_for_pipeline(
                module_or_pipeline, state_dict, device_map
            )

        if device_map:
            raise ValueError(f"Some modules in the device_map were not found: {list(device_map.keys())}")

        return module_or_pipeline

    @staticmethod
    def get_pipeline_modules(pipeline: Any) -> dict[str, torch.nn.Module]:
        """Get the modules from the pipeline.

        Args:
            pipeline: The pipeline object to extract modules from.

        Returns:
            dict: A dictionary mapping attribute names to Module objects.
        """
        modules = {}

        for name, attr in vars(pipeline).items():
            if isinstance(attr, (torch.nn.Module, WrapperModule)):
                modules[name] = attr

        return modules

    @staticmethod
    def state_dict_from_pipeline(pipeline: Any) -> dict[str, Any]:
        """Extract the state_dict from a pipeline (e.g. HF Diffusers pipeline).

        Args:
            pipeline: The pipeline to extract the state_dict from.

        Returns:
            dict: The state_dict from the pipeline.
        """
        state_dict = OrderedDict()
        pipeline_modules = TorchCheckpoint.get_pipeline_modules(pipeline)

        for name, module in pipeline_modules.items():
            if isinstance(module, WrapperModule):
                state_dict[name] = module.state_dict()
        if not state_dict:
            raise ValueError("No tuned modules found in the pipeline")
        return state_dict

    @staticmethod
    def load_state_dict_for_module(
        module: TorchModule | WrapperModule,
        state_dict: dict,
        device_map: dict[str, torch.device],
        module_name: str = "",
    ) -> TorchModule | WrapperModule:
        """Load a state dictionary into a module.

        This method loads a state dictionary into a module, replacing regular torch modules
        with tuned modules where applicable. It handles both top-level modules and nested
        child modules.

        The methods has two phases:
        1. Traverse the module hierarchy and identify which parts of the state dictionary
        correspond to tuned modules. When a match is found, it replaces the original module
        with a wrapper module initialized from the state dictionary. Corresponding keys are
        removed from the state_dict. The tuned module is compiled is necessary (JIT type).
        2. Handle original torch modules not tuned with `load_state_dict(state_dict, strict=False)`,
        strict=False because wrapped module keys were removed and those modules have been already loaded.

        Args:
            module (torch.nn.Module): The module to load the state dictionary into.
                This can be a regular torch module or a module that contains tuned submodules.
            state_dict (dict): The state dictionary containing the tuned module states.
                The dictionary should have keys that correspond to the module structure,
                with tuned module states stored in a format recognized by WrapperModule.
            device_map (dict): The device map to load module to.
            module_name (str): The name of the module for device map matching.

        Returns:
            torch.nn.Module: The module with tuned components loaded from the state dictionary.
                If the top-level module was tuned, returns the new wrapper module.
                Otherwise, returns the original module with tuned submodules replaced.

        Note:
            This method modifies the module in-place by replacing submodules with their
            tuned versions. Original torch modules that were not tuned are preserved.
        """
        top_module = None

        def replace_tuned_module(
            module: TorchModule,
            local_state_dict: dict,
            device_map: dict[str, torch.device],
            parent_module: TorchModule | None = None,
            prefix: str = "",
            module_name: str = "",
        ) -> None:
            """Traverses the module and replaces torch modules with tuned modules.

            Original torch modules which were not tuned are handled later.

            Args:
                module: The module to traverse.
                local_state_dict: The state dictionary to traverse.
                device_map: The device map to load modules to.
                parent_module: The parent module.
                prefix: The prefix of the module.
                module_name: The name of the module.
            """
            stored_object = local_state_dict.get(prefix, {})
            if WrapperModule.is_state_dict_valid(stored_object):
                # This is a tuned module, replace with wrapper module
                device = device_map.pop(module_name, None)

                wrapper_module = WrapperModule.from_dict(module, stored_object, device)
                if parent_module is None:
                    # We replace top module
                    nonlocal top_module
                    top_module = wrapper_module
                else:
                    # We replace a child module with wrapper module
                    setattr(parent_module, module_name, wrapper_module)
                # Remove state_dict key - we already handled it
                del local_state_dict[prefix]
            else:
                # This is an original torch module
                for child_name, child_module in module.named_children():
                    if child_module is not None:
                        child_prefix = prefix + child_name + "."
                        child_state_dict = {k: v for k, v in local_state_dict.items() if k.startswith(child_prefix)}
                        replace_tuned_module(
                            module=child_module,
                            local_state_dict=child_state_dict,
                            parent_module=module,
                            prefix=child_prefix,
                            module_name=child_name,
                            device_map=device_map,
                        )

        replace_tuned_module(
            module=module,
            local_state_dict=state_dict,
            device_map=device_map,
            module_name=module_name,
        )

        # Handle original torch modules not tuned, strict=False because wrapped module keys removed
        final_module = module if top_module is None else top_module
        final_module.load_state_dict(state_dict, strict=False)
        return final_module

    @staticmethod
    def load_state_dict_for_pipeline(pipeline, state_dict, device_map: dict[str, torch.device]):
        """Load a state dictionary into a pipeline.

        This method loads the state dictionary into the pipeline by matching module names
        in the state dictionary with attributes in the pipeline. Each matched module is
        replaced with the tuned module.

        Args:
            pipeline: The pipeline to load the state dictionary into.
            state_dict: The state dictionary containing the module states.
            device_map: The device map to load module to.

        Returns:
            The pipeline with loaded modules.

        Raises:
            ValueError: If a module in the state_dict is not found in the pipeline or
                        if a matched attribute is not a torch.nn.Module or WrapperModule.
        """
        matched_modules = set()
        for name, module_state_dict in state_dict.items():
            if hasattr(pipeline, name):
                module = getattr(pipeline, name)
                if isinstance(module, (torch.nn.Module, WrapperModule)):
                    matched_modules.add(name)
                    setattr(
                        pipeline,
                        name,
                        TorchCheckpoint.load_state_dict_for_module(module, module_state_dict, device_map, name),
                    )
                else:
                    raise ValueError(f"Module {name} is not a torch.nn.Module or WrapperModule")

        unmatched_keys = set(state_dict.keys()).difference(matched_modules)
        if unmatched_keys:
            raise ValueError(f"Some modules in the state_dict were not found in the pipeline: {unmatched_keys}")

        return pipeline
