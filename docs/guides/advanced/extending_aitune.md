---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Extending AITune"
---

Implement a backend to integrate a new compiler, runtime, or optimization. Implement a strategy to define how
candidates are evaluated and selected. Either plugin can live entirely in your application package. Pass the
instances to the normal tuning workflow and AITune supplies the recorded module, graph specification, and samples.

| Plugin | What your code owns |
| --- | --- |
| `Backend` | Supported execution and build modes, inference behavior, resource lifecycle, and serialized state. |
| `TuneStrategy` | Candidate order, validation, measurement, failure behavior, and the decision to return a ready backend. |

A custom strategy can evaluate built-in backends, and built-in strategies can evaluate a custom backend. You can
also use the two custom plugins together, as shown below.

## Implement a backend in your package

This example uses a pretransposed weight and a matrix multiply to implement a plain `nn.Linear`. It illustrates the
interface and lifecycle. Measure it on your workload before selecting it for performance.

Save the following as `my_backend.py` in your application:

```python
from torch import nn

from aitune.torch.backend import Backend, BuildMode, ExecutionMode
from aitune.torch.backend.backend import BackendState


class MatmulLinearBackend(Backend):
    _execution_modes = frozenset({ExecutionMode.SINGLE_GPU})
    _build_mode = BuildMode.AHEAD_OF_TIME

    def _build(self, module, graph_spec, samples, cache_dir):
        if type(module) is not nn.Linear:
            raise TypeError("MatmulLinearBackend requires a plain nn.Linear")
        self.weight_t = module.weight.detach().T.contiguous().clone()
        self.bias = None if module.bias is None else module.bias.detach().clone()
        self._activate()
        return self

    def _infer(self, input):
        output = input @ self.weight_t
        return output if self.bias is None else output + self.bias

    def _activate(self):
        self.weight_t = self.weight_t.to(self.device)
        if self.bias is not None:
            self.bias = self.bias.to(self.device)

    def _deactivate(self):
        self.weight_t = self.weight_t.cpu()
        if self.bias is not None:
            self.bias = self.bias.cpu()

    def _deploy(self):
        self._activate()

    def key(self):
        return "matmul-linear-v1"

    def describe(self):
        return "Linear with a pretransposed weight"

    def to_dict(self):
        return {
            "type": type(self).__name__,
            "weight_t": self.weight_t.cpu(),
            "bias": None if self.bias is None else self.bias.cpu(),
            "device": self.device,
        }

    @classmethod
    def from_dict(cls, module, state_dict):
        backend = cls()
        backend.weight_t = state_dict["weight_t"]
        backend.bias = state_dict["bias"]
        backend._set_device(state_dict["device"])
        backend.state = BackendState.CHECKPOINT_LOADED
        return backend
```

AITune calls the public lifecycle methods, and the backend implements the corresponding hooks. Return an inference-ready
backend from `_build()`. Give different backend configurations distinct cache keys. If rebuilding or restoring
execution requires the original PyTorch module, declare the backend as just-in-time so AITune retains that module.

For checkpoint loading, make the backend package available and import its class before calling `ait.load()`. The
loader resolves backend names among imported direct subclasses of `Backend`, so give the class a unique name. If the
backend needs an engine file or directory, include it in the serialized state using `ArtifactPath` so AITune bundles
and relocates it. Ordinary `Path` values are serialized without copying the files they point to.

### Bundle backend files in a checkpoint

Create an `ArtifactPath` from the backend's cache directory and the engine's relative path. Use the resulting
`.path` as the output location. In this integration example, `build_engine()` stands for your compiler or engine
build function, and `cache_dir` is the directory AITune passes to your backend.

```python
from aitune.torch.backend import ArtifactPath

engine_artifact = ArtifactPath(cache_dir, "tensorrt/model.engine")
build_engine(output_path=engine_artifact.path)


def to_dict(self):
    return {"engine_path": engine_artifact}
```

If another API returns the path of an artifact it has already created, use
`ArtifactPath.from_existing(engine_path, root=cache_dir)`. The explicit root preserves the intended directory
structure and limits the files you can include to that directory. AITune preserves `tensorrt/model.engine` inside
the checkpoint and updates the root when loading it. Use the restored `ArtifactPath` directly in `from_dict()`.

## Implement a strategy in your package

This strategy tries candidates in order and accepts the first one that builds and passes the strategy's correctness
checks. It reuses the base helper for copying the candidate per attempt, build logging, and correctness validation.
A different strategy could add measurements, reject candidates based on application-specific checks, or stop after
a fixed tuning budget.

You can also override `_pre_tune()` and `_post_tune()` to prepare or finalize a tuning attempt.

Save the following as `my_strategy.py`:

```python
from aitune.torch.tune_strategy import TuneStrategy


class OrderedValidationStrategy(TuneStrategy):
    def __init__(self, backends):
        super().__init__()
        self.backends = tuple(backends)

    def to_json_dict(self):
        return {"backends": [backend.describe() for backend in self.backends]}

    def _describe_parts(self):
        return ["Try backends in order and accept the first that passes validation"]

    def _tune(self, module, name, graph_spec, samples, device, cache_dir):
        for backend in self.backends:
            ready = self._build_and_validate_backend(
                backend, module, name, graph_spec, samples, device, cache_dir
            )
            if ready is not None:
                return ready
        raise RuntimeError(f"No backend passed validation for {name}/{graph_spec.name}")
```

## Use the plugins together

The example uses a small CPU workload to show how to wire the plugins. The same interfaces apply when tuning GPU
modules. In an application, supply inputs from your workload and evaluate numerical agreement and end-to-end
performance before deploying.

```python
import torch
import aitune.torch as ait

from my_backend import MatmulLinearBackend
from my_strategy import OrderedValidationStrategy

model = torch.nn.Linear(8, 4).eval()
dataset = torch.randn(8, 8)

with torch.no_grad():
    expected = model(dataset[:2])

strategy = OrderedValidationStrategy([MatmulLinearBackend()])
tuned = ait.Module(model, "projection", strategy=strategy)
ait.tune(tuned, dataset, batch_sizes=[1, 2], device="cpu", ignore_failing_modules=False)

with torch.no_grad():
    torch.testing.assert_close(tuned(dataset[:2]), expected)
```

The [tuning workflow](tuning_workflow.md) explains recording, profiling, and checkpoint details. The
[architecture](architecture.md) shows how these interfaces fit into application execution.
