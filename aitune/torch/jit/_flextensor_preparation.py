# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Translate original-model selection into residency shells using public Torch metadata.

FlexTensor 0.5.0 exposes no public pre-offload selector. These integration-owned
rules implement its documented name/class policies without importing provider internals.
"""

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch


@dataclass(frozen=True)
class _Tree:
    """Names and classes needed for policy resolution, without retaining model tensors."""

    modules: tuple[tuple[str, type], ...]
    parameters: tuple[str, ...]


@lru_cache(maxsize=2048)
def _glob(pattern: str, *, recursive: bool = False, atomic: bool = False) -> re.Pattern[str]:
    """Compile only documented star/question globs; brackets remain literal."""
    if atomic:
        return re.compile(re.escape(pattern).replace(r"\*", ".*").replace(r"\?", "."))
    segments = []
    for part in pattern.split("."):
        if recursive and part == "*":
            segments.append(r"[^.]+(?:\.[^.]+)*")
        else:
            segments.append(re.escape(part).replace(r"\*", "[^.]*").replace(r"\?", "[^.]"))
    return re.compile(r"\.".join(segments))


def _matches(path: str, patterns: list[str], *, recursive: bool = False) -> bool:
    """Match complete paths with the appropriate standalone-star semantics."""
    return any(_glob(pattern, recursive=recursive).fullmatch(path) is not None for pattern in patterns)


def _partition(patterns: list[str]) -> tuple[list[str], list[str]]:
    """Separate validated public configuration entries into names and classes."""
    names, classes = [], []
    for pattern in patterns:
        if pattern.startswith("class:"):
            classes.append(pattern.removeprefix("class:"))
        else:
            names.append(pattern.removeprefix("name:"))
    return names, classes


def _class_paths(tree: _Tree, patterns: list[str]) -> set[str]:
    """Resolve short and fully qualified class globs on non-root modules."""
    return {
        path
        for path, cls in tree.modules
        if any(
            _glob(pattern, atomic=True).fullmatch(name) is not None
            for pattern in patterns
            for name in (cls.__name__, f"{cls.__module__}.{cls.__qualname__}")
        )
    }


def _module_patterns(tree: _Tree, names: list[str]) -> list[str]:
    """Resolve module selectors, including module prefixes of parameter selectors."""
    result: set[str] = set()
    module_paths = {path for path, _ in tree.modules}
    for pattern in names:
        parts = pattern.split(".")
        matching_parameters = [path for path in tree.parameters if _matches(path, [pattern], recursive=True)]
        for size in range(len(parts), 0, -1):
            prefix = ".".join(parts[:size])
            if size < len(parts) and prefix == "*":
                continue
            if not any(_matches(path, [prefix]) for path in module_paths):
                continue
            if size == len(parts) or matching_parameters:
                result.add(prefix)
            break
        else:
            result.update(path.rpartition(".")[0] for path in matching_parameters if path.rpartition(".")[0])
    return list(result)


def _prefixes(path: str) -> tuple[str, ...]:
    """Include each non-root ancestor and the path itself."""
    parts = path.split(".")
    return tuple(".".join(parts[:size]) for size in range(1, len(parts) + 1))


def _parameter_match(path: str, names: list[str], classes: set[str]) -> bool:
    """Apply cascading name/class rules to a single original parameter."""
    return any(prefix in classes or _matches(prefix, names, recursive=True) for prefix in _prefixes(path))


def _resolve(
    tree: _Tree, includes: list[str], excludes: list[str]
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[bool, ...]]:
    """Resolve outermost candidates, final units and global parameter eligibility."""
    include_names, include_classes = _partition(includes)
    exclude_names, exclude_classes = _partition(excludes)
    include_class_paths = _class_paths(tree, include_classes)
    exclude_class_paths = _class_paths(tree, exclude_classes)
    patterns = _module_patterns(tree, include_names)
    candidates: list[str] = []
    for path, _ in tree.modules:
        if not (_matches(path, patterns) or path in include_class_paths):
            continue
        if not any(path.startswith(parent + ".") for parent in candidates):
            candidates.append(path)
    units = tuple(
        path
        for path in candidates
        if path not in exclude_class_paths and not _matches(path, exclude_names, recursive=True)
    )
    eligibility = tuple(
        _parameter_match(path, include_names, include_class_paths)
        and not _parameter_match(path, exclude_names, exclude_class_paths)
        for path in tree.parameters
    )
    return tuple(candidates), units, eligibility


def _mapped(path: str, units: tuple[str, ...]) -> str:
    """Insert compute only at an explicitly selected residency boundary."""
    for unit in units:
        if path == unit or path.startswith(unit + "."):
            return unit + ".compute" + path[len(unit) :]
    return path


def _wrapped_tree(tree: _Tree, units: tuple[str, ...], shell_type: type) -> _Tree:
    """Describe the wrapped namespace without constructing or mutating modules."""
    modules: list[tuple[str, type]] = []
    for path, cls in tree.modules:
        if path in units:
            modules.append((path, shell_type))
        modules.append((_mapped(path, units), cls))
    return _Tree(tuple(modules), tuple(_mapped(path, units) for path in tree.parameters))


def _region(path: str, units: tuple[str, ...]) -> str | None:
    """Identify the residency boundary or the remainder of the source model."""
    return next((unit for unit in units if path == unit or path.startswith(unit + ".")), None)


def _check_aliases(model: torch.nn.Module, units: tuple[str, ...]) -> None:
    """Reject modules or tensor storage shared across different residency boundaries."""
    owners: dict[object, str | None] = {}

    def record(key: object, path: str) -> None:
        """Reject a shared identity when its public paths have different owners."""
        owner = _region(path, units)
        if key in owners and owners[key] != owner:
            raise ValueError(f"FlexTensor preparation cannot preserve cross-boundary aliases at {path!r}")
        owners[key] = owner

    for path, module in model.named_modules(remove_duplicate=False):
        record(("module", id(module)), path)
    tensors = (*model.named_parameters(remove_duplicate=False), *model.named_buffers(remove_duplicate=False))
    for path, tensor in tensors:
        record(("tensor", id(tensor)), path)
        # Meta/empty tensors have no meaningful storage pointer. Identity aliases
        # still apply; sparse layouts do not expose strided storage publicly.
        if tensor.numel() and not tensor.is_meta and tensor.layout == torch.strided:
            record(("storage", tensor.device, tensor.untyped_storage().data_ptr()), path)


def _check_boundaries(model: torch.nn.Module, units: tuple[str, ...], shell_type: type) -> None:
    """Validate preparation boundaries before creating any shell or provider state."""
    if any(isinstance(module, shell_type) for module in model.modules()):
        raise ValueError("FlexTensor units are already wrapped; use a fresh unprepared model")
    if not units:
        raise ValueError("FlexTensor preparation selected no units; choose executable module boundaries")
    for path in units:
        module = model.get_submodule(path)
        if getattr(module.forward, "__func__", module.forward) is torch.nn.Module.forward:
            raise ValueError(f"FlexTensor preparation selected non-executable container/module {path!r}; select blocks")
    _check_aliases(model, units)


def _literal(path: str) -> str:
    """Encode exact public FT names without unsafe wildcard or whitespace changes."""
    if "*" in path or "?" in path or path != path.strip():
        raise ValueError(f"FlexTensor cannot encode exact path {path!r}; use the manual preparation path")
    return f"name:{path}"


def _prepare_policy(model: torch.nn.Module, config: Any, shell_type: type) -> tuple[Any, tuple[str, ...]]:
    """Plan a faithful shell/config translation before mutating the original model."""
    tree = _Tree(
        tuple((path, type(module)) for path, module in model.named_modules() if path),
        tuple(path for path, _ in model.named_parameters()),
    )
    candidates, units, eligible = _resolve(tree, config.include_patterns, config.exclude_patterns)
    _check_boundaries(model, units, shell_type)
    includes = list(config.include_patterns) + [_literal(path) for path in candidates]
    excludes = [_literal(path) for path in candidates if path not in units]
    excludes.extend(
        _literal(_mapped(path, units))
        for path, offloaded in zip(tree.parameters, eligible, strict=True)
        if not offloaded
    )
    effective = config.model_copy(update={"include_patterns": includes, "exclude_patterns": excludes}, deep=False)
    virtual = _wrapped_tree(tree, units, shell_type)
    _, actual_units, actual_eligibility = _resolve(virtual, effective.include_patterns, effective.exclude_patterns)
    if actual_units != units or actual_eligibility != eligible:
        raise ValueError(
            "FlexTensor preparation cannot preserve this selection policy; use the manual preparation path"
        )
    return effective, units
