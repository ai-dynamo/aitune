#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Enforce the dependency boundary of ``aitune.records``.

This static, dependency-free checker permits only standard-library modules and
other ``aitune.records`` modules. Run it directly or through its unit tests.
"""

import ast
import sys
from collections.abc import Iterable
from pathlib import Path

PROJECT_DIRECTORY = Path(__file__).resolve().parents[3]
RECORDS_DIRECTORY = PROJECT_DIRECTORY / "aitune" / "records"
RECORDS_PACKAGE = "aitune.records"
STANDARD_LIBRARY_MODULES = frozenset(sys.stdlib_module_names) | frozenset(sys.builtin_module_names)
DYNAMIC_IMPORT_FUNCTIONS = ("__import__", "import_module")


def records_source_paths() -> list[Path]:
    """Return every Python source file in ``aitune.records``."""
    return sorted(RECORDS_DIRECTORY.rglob("*.py"))


def module_name(path: Path) -> str:
    """Return the dotted module name for a repository source file."""
    parts = list(path.relative_to(PROJECT_DIRECTORY).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _relative_import_base(path: Path, level: int) -> str | None:
    """Resolve the package targeted by a relative import."""
    module = module_name(path)
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    parts = package.split(".")
    if level > len(parts):
        return None
    return ".".join(parts[: len(parts) - level + 1])


def _dynamic_import_name(node: ast.Call) -> str | None:
    """Return a statically named dynamic import target, if present."""
    if isinstance(node.func, ast.Name):
        function_name = node.func.id
    elif isinstance(node.func, ast.Attribute):
        function_name = node.func.attr
    else:
        return None
    if function_name not in DYNAMIC_IMPORT_FUNCTIONS:
        return None
    target = (
        node.args[0]
        if node.args
        else next((keyword.value for keyword in node.keywords if keyword.arg in ("name", "module")), None)
    )
    if isinstance(target, ast.Constant) and isinstance(target.value, str):
        return target.value
    return None


def imported_module_names(node: ast.AST, path: Path) -> list[str]:
    """Return absolute module names imported by one AST node."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        if node.level:
            base = _relative_import_base(path, node.level)
            if base is None:
                return ["<unresolved-relative-import>"]
            module = f"{base}.{node.module}" if node.module else base
        else:
            module = node.module
        if module is None:
            return []
        return [f"{module}.{alias.name}" for alias in node.names if alias.name != "*"] or [module]
    if isinstance(node, ast.Call) and (name := _dynamic_import_name(node)) is not None:
        return [name]
    return []


def is_allowed(name: str) -> bool:
    """Return whether ``name`` stays within the records dependency boundary."""
    root = name.partition(".")[0]
    return root in STANDARD_LIBRARY_MODULES or name == RECORDS_PACKAGE or name.startswith(f"{RECORDS_PACKAGE}.")


def direct_violations(paths: Iterable[Path] | None = None) -> list[str]:
    """Return forbidden imports found in records source files."""
    violations = []
    for path in records_source_paths() if paths is None else paths:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            for name in imported_module_names(node, path):
                if not is_allowed(name):
                    violations.append(f"{path}:{node.lineno}: imports forbidden module {name!r}")
    return violations


def main() -> int:
    """Print violations and return whether the check failed."""
    violations = direct_violations()
    if violations:
        print("aitune.records import-boundary violations:", file=sys.stderr)
        print("\n".join(violations), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
