---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Coding Guidelines"
---
# Coding Guidelines

This file lists the coding rules for AITune. It applies to people and to AI agents.

Tools enforce most rules. This file explains the rules that tools do not enforce, and it gives the
reason for each rule. If a rule makes code harder to read in a special case, explain the exception
in the pull request.

Rules that are not obvious have examples:

- **Bad**: do not write code like this.
- **Good**: this is acceptable.
- **Better**: use this form when you can.

## General principles

### Keep it simple

Write the simplest code that solves the current problem.

- Do not add options, parameters, or abstractions for needs that do not exist yet.
- Do not add a class when a function is enough, or a base class with only one subclass.
- Prefer plain Python over clever constructs that need a comment to explain them.

**Why:** Every abstraction is code that a reader must understand and that someone must maintain.
Adding an abstraction later, when a second use exists, is cheap. Removing an unused one is not.

### Use clear names

- Name a thing after what it is or does, not after how it is implemented: `profiles`, not `profile_list`.
- Use full words. Use abbreviations only when they are common in the domain (`trt`, `onnx`, `cfg`).
- Name functions with a verb (`build_engine`) and booleans as a question (`is_dynamic`, `has_cache`).
- Use the same name for the same concept everywhere.

**Why:** A good name removes the need for a comment, and the reader does not have to open the
definition to understand the call.

## Enforced by tools

Run `make lint` before you push. It runs these checks:

- **ruff** and **ruff-format**: lint, formatting (line length 120), import order, Google-style
  docstrings, complexity ≤ 10, mutable default arguments, bare `except:`, `assert` in `aitune/`,
  `else` after `return` or `raise`, and modern syntax such as `list[int]` and `X | None`. The
  configuration is in `pyproject.toml`.
- **pyupgrade**: modern Python syntax for the supported Python versions.
- **pyrefly**: static type checks. CI fails on errors that are not in `pyrefly-baseline.json`.
  - Do not add new entries to the baseline.
  - If you edit a file that has baselined errors, fix them if the fix is small and changes only
    types (no runtime change).
  - After you fix baselined errors, run `pyrefly check --prune-baseline`.
- **SPDX header**: every new file must begin with this header, written as comments for its file
  type. If the file starts with a shebang (`#!`) or a Dockerfile `# syntax=` line, put the header
  after it, because that line must stay first. In Markdown, put the header in the front matter, as
  in `AGENTS.md`. Files without comment syntax,
  such as JSON, do not need it. ruff (`CPY`) checks Python files. The "License headers" job in
  `.github/workflows/ci.yml` checks the other file types, for example shell, Markdown, YAML, TOML,
  HTML, CSS, and `Makefile`. `make lint` does not run that job.

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0
  ```

If a tool and this file do not agree, follow the tool and update this file.
Do not disable a tool rule for a line (`# noqa`, `# pyrefly: ignore`) unless you also write the reason.

## Layout

### Module layout

Use this order in a module:

1. Imports.
2. Constants and `logger = logging.getLogger(__name__)`.
3. Public classes and functions.
4. Private helpers (names that start with `_`).

**Why:** A reader opens a module to find its public API. If the public API is first, the reader
does not have to scroll past implementation details.

### Class layout

Use this order in a class:

1. Class attributes.
2. `__new__`, `__init__`, and `__post_init__`.
3. Public methods, properties, and other special methods (for example `__call__` and `__repr__`).
4. Private methods (names that start with `_`).

**Why:** The constructor tells the reader what state the class has. The public methods tell the
reader what the class does. Private methods are details. With the same order in all classes, a
reviewer knows where to look.

**Bad:**

```python
class Backend:
    def _assert_supported(self, module: nn.Module) -> None: ...

    def build(self, module: nn.Module) -> None:
        self._assert_supported(module)
        ...

    def __init__(self, config: BackendConfig) -> None:
        self._config = config
```

**Good:**

```python
class Backend:
    def __init__(self, config: BackendConfig) -> None:
        self._config = config

    def build(self, module: nn.Module) -> None:
        self._assert_supported(module)
        ...

    def _assert_supported(self, module: nn.Module) -> None: ...
```

## Control flow

### Use guard clauses

Handle errors and special cases first, with `raise`, `return`, or `continue`. Then write the main
path without extra indentation.

ruff reports an `else` after `return` or `raise`. It does not find other nested blocks.

**Why:** Each level of nesting adds a condition that the reader must remember. A guard clause
handles one condition and then the reader can forget it.

**Bad:**

```python
def load(path: Path) -> Checkpoint:
    if path.exists():
        if path.suffix == ".ait":
            return _read(path)
        else:
            raise AITuneUserInputError(f"Expected an .ait file, got {path}.")
    else:
        raise AITuneUserInputError(f"Checkpoint {path} does not exist.")
```

**Better:**

```python
def load(path: Path) -> Checkpoint:
    if not path.exists():
        raise AITuneUserInputError(f"Checkpoint {path} does not exist.")
    if path.suffix != ".ait":
        raise AITuneUserInputError(f"Expected an .ait file, got {path}.")
    return _read(path)
```

### Do not use `for`/`else` or `while`/`else`

**Why:** The `else` block runs when the loop ends without `break`. Many readers think that it runs
when the loop has no items. Use a helper function or `any()`/`all()` instead.

**Bad:**

```python
for idx, profile in enumerate(profiles):
    for name, (min_shape, _, max_shape) in profile.items():
        if not _in_range(inputs[name].shape, min_shape, max_shape):
            break
    else:
        return idx
raise RuntimeError("No TensorRT optimization profile matches the input shapes")
```

**Good:**

```python
for idx, profile in enumerate(profiles):
    matches = True
    for name, (min_shape, _, max_shape) in profile.items():
        if not _in_range(inputs[name].shape, min_shape, max_shape):
            matches = False
            break
    if matches:
        return idx
raise RuntimeError("No TensorRT optimization profile matches the input shapes")
```

**Better:**

```python
for idx, profile in enumerate(profiles):
    if _profile_matches(profile, inputs):
        return idx
raise RuntimeError("No TensorRT optimization profile matches the input shapes")


def _profile_matches(profile: Profile, inputs: dict[str, torch.Tensor]) -> bool:
    return all(_in_range(inputs[name].shape, low, high) for name, (low, _, high) in profile.items())
```

### Use `defaultdict` to group values

**Why:** `defaultdict(list)` shows the default value one time, where the dictionary is created.
`setdefault(key, [])` repeats it at each call and creates a new empty list on each call, also when
the key exists.

### Keep comprehensions simple

Use at most one `for` clause and one `if` clause in a comprehension. Use a loop for more.

**Why:** A nested comprehension reads in a different order than it runs. A loop is easier to read
and to debug.

**Bad:**

```python
names = [name for module in modules for name, param in module.named_parameters() if param.requires_grad if name]
```

**Good:**

```python
names = []
for module in modules:
    names.extend(name for name, param in module.named_parameters() if param.requires_grad and name)
```

## Typing

### Follow the type hints

If a caller passes a value of the wrong type, fix the caller. Do not add code that converts wrong
types at runtime.

**Why:** Hidden conversion makes the type hint incorrect. Then the type checker cannot find the
bug, and the next reader trusts a hint that is not true.

**Bad:**

```python
def set_batch_sizes(batch_sizes: list[int]) -> None:
    if isinstance(batch_sizes, int):
        batch_sizes = [batch_sizes]
    ...
```

**Good:**

```python
def set_batch_sizes(batch_sizes: list[int]) -> None: ...


set_batch_sizes([8])
```

**Better** (only if both forms are part of the public API): make the type hint say so, and convert
the value one time, at the public entry point.

```python
def set_batch_sizes(batch_sizes: int | list[int]) -> None:
    sizes = [batch_sizes] if isinstance(batch_sizes, int) else batch_sizes
    ...
```

### Avoid `Any` and ignore comments

Use the most specific type that is correct. Use `Any` only when no correct type exists, for example
for a third-party object without type hints.

**Why:** `Any` turns off type checks for each value that it touches. An ignore comment hides the
error and also hides new errors on the same line.

### Avoid `cast`

Do not use `typing.cast` to fix a type error. Fix the type that causes the error:

- If an attribute is `X | None` and is set only after a step such as `build()`, check it and raise,
  or pass the value as an argument.
- If a function returns a union and each caller knows which type it gets, make the return type
  more specific, or use `@overload` or a `TypeVar`.
- If the type checker does not follow your check, use `isinstance` where you use the value.

Use `cast` only if a third-party type hint is missing or incorrect. Add a comment that explains why.

**Why:** `cast` tells the type checker to trust you, and it does nothing at runtime. If the value
has a different type, the code fails later with an unclear error. A `cast` usually shows that an
API returns a wider type than its callers get. Fix the API one time, not each caller.

**Bad:**

```python
class OnnxRuntimeBackend(Backend):
    def __init__(self, config: OnnxRuntimeConfig) -> None:
        self._model_artifact: ArtifactPath | None = None

    def _activate(self) -> None:
        model_artifact = cast(ArtifactPath, self._model_artifact)
        self._session = onnxruntime.InferenceSession(str(model_artifact.path))
```

**Good:**

```python
    def _activate(self) -> None:
        if self._model_artifact is None:
            raise RuntimeError("Build the backend before you activate it.")
        self._session = onnxruntime.InferenceSession(str(self._model_artifact.path))
```

## Errors

### Do not use `assert` in production code

ruff checks this rule in `aitune/`, because `python -O` removes `assert` statements. There is one
exception: in a hot path, you can use `assert` to narrow a type for pyrefly when a lifecycle
invariant guarantees the value, for example `assert self._session is not None`. Add
`# noqa: S101` and a comment that names the invariant.

### Raise the correct exception type

- If the user gives an incorrect model, data, or configuration, raise `AITuneUserInputError`
  from `aitune/exceptions.py`.
- For an internal error or an incorrect call between AITune modules, raise a built-in exception,
  for example `ValueError`, `TypeError`, or `RuntimeError`.

**Why:** The exception type tells the user if they must change their input or report a bug.
Users can also catch all AITune input errors in one `except` clause.

**Bad:**

```python
if isinstance(config.profiles, list) and graph_spec.dynamic_shapes:
    raise RuntimeError("Profiles cannot be provided together with dynamic shapes.")
```

**Good:**

```python
if isinstance(config.profiles, list) and graph_spec.dynamic_shapes:
    raise AITuneUserInputError("TensorRT profiles cannot be provided together with module dynamic shape definitions.")
```

### Catch only the exceptions that you expect

Catch the most specific exception type. Keep the `try` block as small as possible. If you must
catch `Exception` (for example, to try the next backend), log the error.

**Why:** A broad `except` also catches bugs, such as `AttributeError` from a typing mistake.
Then the bug looks like a normal failure, and nobody sees it.

**Bad:**

```python
try:
    config = json.loads(path.read_text())
    backend = build_backend(config)
except Exception:
    return None
```

**Good:**

```python
try:
    config = json.loads(path.read_text())
except (OSError, json.JSONDecodeError):
    logger.warning("Cannot read backend config %s", path)
    return None
backend = build_backend(config)
```

## Tests

### Use `assert isinstance` instead of `cast`

**Why:** `cast` only changes the type for the type checker. It does nothing at runtime, so the
test continues with a wrong object and fails later with an unclear error. `assert isinstance`
narrows the type and also checks it.

**Bad:**

```python
(selected,) = cast(TunedModule, head._wrapper).backends.values()
```

**Good:**

```python
assert isinstance(head._wrapper, TunedModule)
(selected,) = head._wrapper.backends.values()
```

### Define shared test helpers at module level

If more than one test uses a helper class or function, define it one time at module level, or in
`tests/toy_models/` and `tests/toy_backends.py` if other test files can use it.

**Why:** Copies of a helper change in different ways over time. One definition is easier to fix.

### Do not write unit tests for `examples/`

The functional workflows validate the examples.

## Comments and docstrings

- Write Google-style docstrings for public modules, classes, and functions.
- Add an `Examples:` section to the docstring of each name in `aitune.torch.__all__`. For other
  public functions, add an example when the usage is not obvious from the signature.
- Make examples runnable doctests (`>>>`). Pytest runs them on CPU in CI. If a statement needs a GPU
  or an optional backend, mark it with `# doctest: +SKIP`. Also mark every later statement that uses
  its result, because `+SKIP` skips only the statement it is on.
- Keep examples short: show the common call and its result, not every option.
- Code in `tests/`, `docs/`, `tools/`, and `examples/` does not need docstrings and can use `print`.
- Write a comment only to explain why the code does something. Do not describe what the code does.

**Why (examples):** An example shows how to call the API faster than a description does. A doctest
fails when the API changes, so the example stays correct.

**Good:**

```python
def format_call_path(path: CallPath) -> str:
    """Format a call path as text.

    Args:
        path: Call labels, outermost first.

    Returns:
        The labels without the `root.` prefix, joined by ` > `.

    Examples:
        >>> format_call_path(("root.forward", "root.vlm.forward"))
        'forward > vlm.forward'
    """
```

**Why (comments):** Good names already tell the reader what the code does. A comment that repeats the code
becomes incorrect when the code changes.

**Bad:**

```python
# Copy the module.
model_copy = deepcopy(module)
```

**Good:**

```python
model_copy = deepcopy(module)
```

**Better** (a comment that explains why):

```python
# ModelOpt quantization changes the model in place, so quantize a copy and keep the caller's module.
model_copy = deepcopy(module)
```
