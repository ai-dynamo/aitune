# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Torch checkpoint module."""

from collections import OrderedDict
from logging import getLogger
from pathlib import Path
from typing import Any

import torch
from torch.nn import Module as TorchModule

from aitune.torch.checkpoint.checkpoint import Checkpoint
from aitune.torch.checkpoint.storage import Storage
from aitune.torch.module.tuned_module import TunedModule
from aitune.torch.module.wrapper_module import Module as WrapperModule
from aitune.torch.utils.device import get_device

logger = getLogger(__name__)


def _log_module_memory(module_path: str, stage: str) -> None:
    """Report visible GPU memory at module boundaries without interrupting restoration."""
    try:
        for device_index in range(torch.cuda.device_count()):
            device = torch.device("cuda", device_index)
            free, total = torch.cuda.mem_get_info(device)
            mib = 1024**2
            logger.info(
                "%s: %s; device=%s, GPU used=%.1f MiB, free=%.1f MiB, total=%.1f MiB, "
                "PyTorch allocated=%.1f MiB, reserved=%.1f MiB",
                module_path,
                stage,
                device,
                (total - free) / mib,
                free / mib,
                total / mib,
                torch.cuda.memory_allocated(device) / mib,
                torch.cuda.memory_reserved(device) / mib,
            )
    except Exception as error:
        logger.info("%s: %s; GPU memory unavailable: %s", module_path, stage, error)


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
            module_states = {"": state_dict}
        else:
            state_dict = TorchCheckpoint.state_dict_from_pipeline(module_or_pipeline)
            module_states = state_dict
        self._storage.save(path, state_dict)
        for module_path, module_state in module_states.items():
            for prefix, stored_object in module_state.items():
                if not isinstance(stored_object, dict) or WrapperModule.TUNED_MODULE_KEY not in stored_object:
                    continue
                stored_path = ".".join(part for part in (module_path, prefix.rstrip(".")) if part)
                stored_path = stored_path or stored_object[WrapperModule.NAME_KEY]
                logger.info("📦 Saved module %s:", stored_path)
                tuned_state = stored_object[WrapperModule.TUNED_MODULE_KEY]
                for graph_index, (_, backend_state) in enumerate(tuned_state[TunedModule.BACKENDS_KEY]):
                    logger.info("  ✅ graph %s stored with %s", graph_index, backend_state[TunedModule.TYPE_KEY])

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
            ValueError: If a module in the state_dict is not found in the pipeline, including keys that point to an
                attribute that is not a torch.nn.Module or WrapperModule.

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
    def get_pipeline_modules(pipeline: Any) -> dict[str, torch.nn.Module | WrapperModule]:
        """Get the modules from the pipeline.

        Args:
            pipeline: The pipeline object to extract modules from.

        Returns:
            dict: A dictionary mapping dotted module paths to modules. The walk does not descend into
                WrapperModule instances.
        """
        modules: dict[str, torch.nn.Module | WrapperModule] = {}

        def add_module(path: str, module: torch.nn.Module | WrapperModule) -> None:
            modules[path] = module
            if isinstance(module, WrapperModule):
                return
            for child_name, child in module._modules.items():
                if child is not None:
                    add_module(f"{path}.{child_name}", child)

        for name, attr in vars(pipeline).items():
            if isinstance(attr, (torch.nn.Module, WrapperModule)):
                add_module(name, attr)

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
        saved_module_ids = set()

        for name, module in pipeline_modules.items():
            if isinstance(module, WrapperModule) and id(module) not in saved_module_ids:
                state_dict[name] = module.state_dict()
                saved_module_ids.add(id(module))
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
        checkpoint_module_path = module_name

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

                module_path = ".".join(part for part in (checkpoint_module_path, prefix.rstrip(".")) if part)
                module_path = module_path or stored_object[WrapperModule.NAME_KEY]
                logger.info("📦 Loading module %s:", module_path)
                _log_module_memory(module_path, "before restoration")
                wrapper_module = WrapperModule.from_dict(module, stored_object, device)
                _log_module_memory(module_path, "after restoration")
                if parent_module is None:
                    # We replace top module
                    nonlocal top_module
                    top_module = wrapper_module
                else:
                    # We replace a child module with wrapper module
                    setattr(parent_module, module_name, wrapper_module)
                for graph_index, backend in enumerate(wrapper_module.module.backends.values()):
                    logger.info("  ✅ graph %s restored with %s", graph_index, backend.name)
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
            ValueError: If a module in the state_dict is not found in the pipeline, including keys that point to an
                attribute that is not a torch.nn.Module or WrapperModule.
        """
        modules = TorchCheckpoint.get_pipeline_modules(pipeline)
        matched_modules = set()
        restored_modules = {}
        for name, module_state_dict in state_dict.items():
            module = modules.get(name)
            if module is None:
                # reported below as not found
                continue
            matched_modules.add(name)
            if id(module) in restored_modules:
                # Checkpoints can store one key for each alias of a module. Restore the module only once.
                device_map.pop(name, None)
                continue
            restored_modules[id(module)] = TorchCheckpoint.load_state_dict_for_module(
                module, module_state_dict, device_map, name
            )

            # Install each restored module before loading the next one, including all aliases.
            # This preserves completed replacements if a later module fails to load.
            for path, alias_module in modules.items():
                if alias_module is not module:
                    continue
                owner_path, _, attribute = path.rpartition(".")
                owner = modules[owner_path] if owner_path else pipeline
                setattr(owner, attribute, restored_modules[id(module)])

        unmatched_keys = set(state_dict.keys()).difference(matched_modules)
        if unmatched_keys:
            raise ValueError(f"Some modules in the state_dict were not found in the pipeline: {unmatched_keys}")

        return pipeline
