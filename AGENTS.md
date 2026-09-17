---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---
# AGENTS.md

This file provides guidance to AI coding agents working in this repository.

## Development Commands

```bash
make install-dev          # editable install with all dev deps (uses --extra-index-url https://pypi.nvidia.com)
make lint                 # pre-commit --all-files + pytype
make test                 # pytest (unit tests + doctests)
make coverage             # coverage report → htmlcov/
make docs-serve           # Fern docs dev server (run 'make fern-setup' first)

pytest tests/unit/path/to/test_file.py::test_name -v   # single test
pytest -k "keyword" -v                                  # filter by keyword
pytest --no-header -rN tests/unit/ -x                  # fail fast
```

Test paths are `aitune/` and `tests/unit/`; doctests run across production code.
Functional tests under `tests/functional/` require a GPU and real backends.

## Documentation

Backend guides live under `docs/guides/backends/`. Example docs live in
`examples/*/README.md`, with `examples/README.md` as the catalog.
Do not hand-maintain duplicate copies under `docs/examples/`.

## Architecture References

AITune wraps PyTorch `nn.Module` subgraphs and compiles them with optimized backends.
AOT mode uses explicit inspection, wrapping, tuning, and checkpoint APIs. JIT mode
patches modules at import time and controls tuning through JIT configuration.

Use these references when the task touches the corresponding area:

- **AOT APIs and configuration:** `aitune/torch/tuning.py`, `aitune/torch/inspecting/`,
  and `aitune/torch/config.py`.
- **JIT activation, modes, and defaults:** `aitune/torch/jit/`, especially `config.py`
  and `enable.py`.
- **Backends and lifecycle:** `aitune/torch/backend/backend.py`, backend implementations
  under `aitune/torch/backend/`, and `docs/guides/backends/`.
- **Strategy selection and AOT/JIT backend defaults:** `aitune/torch/tune_strategy/`.
- **Module wrapping and state transitions:** `aitune/torch/module/` and
  `aitune/torch/module_registry.py`.
- **Checkpoint persistence:** `aitune/torch/checkpoint/`.
- **Input normalization:** `aitune/torch/dataloader.py`.
- **Profiling:** `aitune/torch/task/profiling/` contains internal backend profiling;
  `aitune/torch/performance/` contains the user-facing performance attribution profiler.
- **Tuning telemetry:** `aitune/torch/tune_data/`.
- **Dynamo serving:** `aitune/dynamo/` (requires `aitune[dynamo]`).
- **Runtime environment variables:** `aitune/utils/env_vars.py`. Inspection debug
  variables are parsed separately in `aitune/torch/inspecting/module_inspector.py`.

## Code Style

Every production file must begin with:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
```

- **Formatter/linter:** `ruff` (line length 120, Google-style docstrings, complexity ≤ 10).
- **Static analysis:** `pytype` (import-error and module-attr checks disabled).
- Keep docstring examples runnable; pytest executes doctests.
- `tests/` and `examples/` are exempt from docstring and `print` rules.

## Test Fixtures

Key fixtures in `conftest.py` (all autouse unless noted):

- `module_registry_cleanup` — clears the global module registry after each test.
- `aitune_cache_dir` — isolates the AITune cache in a temporary path per test.
- `jit_cleanup` — calls `jit_reset()` before and after each test.
- `aitune_logging_setup` — initializes logging; set `AITUNE_TESTS_LOG_LEVEL=DEBUG`
  for verbose test logs.
- `torch_device` (not autouse) — selects the test device; override with
  `AITUNE_TESTS_USE_DEVICE`. Defaults to `cuda:0` if available, otherwise CPU.

Toy models live in `tests/toy_models/`. Dummy backends for strategy and unit tests
(`DummyBackend`, `SuccessBackend`, `FailureBackend`) live in `tests/toy_backends.py`.
