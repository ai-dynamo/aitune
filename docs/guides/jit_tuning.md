<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Just-in-Time Tuning Guide

Just-in-time tuning enables automatic model tuning without modifying your existing code. You can enable it with an environment variable and run your script - AITune will automatically discover and tune modules during execution.

## Overview

Just-in-time tuning provides a zero-code-change approach to model tuning:

- **Automatic Discovery**: Automatically detects PyTorch modules during execution
- **Zero Code Changes**: No need to modify your existing scripts
- **Hierarchical Tuning**: Recursively tunes modules from top to bottom
- **Configurable**: Fine-tune behavior through environment variables or configuration

## Quick Start

### Enabling Just-in-Time Tuning

The simplest way to enable JIT tuning is through an environment variable:

```bash
export AUTOWRAPT_BOOTSTRAP=aitune_enable_jit_tuning
python your_script.py
```

Your script will run with automatic tuning enabled.

*Note:* Setting the environment variable affects the entire shell session, which may impact other Python processes running in the same shell. We recommend either setting the environment variable immediately before running your script, or using import-based activation inside the script instead.

### Alternative: Import-Based Activation

You can also enable just-in-time tuning by adding a single import at the beginning of your script:

```python
import aitune.torch.jit.enable  # Enable JIT tuning

# Your existing code remains unchanged
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers")
pipe.to("cuda")

# Tuning happens automatically during inference
images = pipe("A beautiful landscape")
```

### Using Annotation (Decorator)

For fine-grained control, you can use the `@patch_for_jit_tuning` decorator on specific functions:

```python
from aitune.torch import patch_for_jit_tuning, jit_config
import timm
import torch
import logging

logging.basicConfig(level=logging.INFO)

jit_config.min_samples = 2
jit_config.batch_axis_required = False

@patch_for_jit_tuning
def create_resnet():
    """This function will have JIT tuning enabled."""
    return timm.create_model("resnet18", pretrained=False).to("cuda")

# Your model
model = create_resnet().cuda()

# Tuning happens automatically when the model is called
with torch.no_grad():
    output = model(torch.randn(1, 3, 224, 224, device="cuda"))
    output = model(torch.randn(2, 3, 224, 224, device="cuda"))
```

This approach allows you to:

- Enable just-in-time tuning for specific functions only
- Keep the rest of your code unchanged

## How Just-in-Time Tuning Works

Just-in-time tuning follows this process:

1. **Initial Runs**: The first few inferences are used to detect model architecture and record input/output shapes
2. **Module Discovery**: Identifies all PyTorch modules in the execution path
3. **Hierarchical Tuning**: Attempts to tune modules starting from the top level:

   - If successful, the module is tuned
   - If a graph break is detected or tuning fails, AITune recursively tunes child modules
4. **Depth Limiting**: Stops at a configurable depth level

### Graph of Execution Detection

AITune has two mechanisms to detect different graphs of execution:

- The first one is based on method signature. If inputs for a particular module change, different `Graphs` are created. Each of them has a separate backend.
- The second one uses the torch dynamo feature to detect graph breaks when a module contains conditional logic based on input data.

Here is an example of a graph break:

```python
import torch

class DynamicModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.child_a = torch.nn.Linear(10, 20)
        self.child_b = torch.nn.Linear(10, 20)

    def forward(self, x):
        if x.sum() > 0:  # Graph break: conditional on input
            return self.child_a(x)
        else:
            return self.child_b(x)
```

When AITune detects a graph break, it skips tuning that module and attempts to tune this module's children.

## Tuning Modes

JIT tuning supports two modes that control *when* tuning is triggered.

### Eager Mode (default)

In eager mode, AITune tunes a module automatically after each forward pass as soon as the required number of samples has been collected.  This works well for pipelines where every module is called a predictable number of times per step (e.g. a simple classifier or a fixed-step inference loop).

```python
from aitune.torch import jit_config
from aitune.torch.jit.config import JITMode

jit_config.mode = JITMode.TUNE_EAGER  # default, explicit assignment not required
```

### Deferred Mode

In deferred mode, AITune records samples during forward passes but does **not** tune automatically.  Tuning is triggered explicitly by calling `aitune.torch.jit.tune.deferred()` after the pipeline has completed at least one full step.

This mode is intended for pipelines where different modules are called a **variable number of times per step** — for example, iterative denoising loops in text-to-image (Stable Diffusion, FLUX) or text-to-video models.  In such cases, eager mode may attempt to tune a module before all modules in the pipeline have been recorded; deferred mode lets you choose a safe synchronisation point after a full pass.

```python
import aitune.torch.jit.enable  # or set AUTOWRAPT_BOOTSTRAP=aitune_enable_jit_tuning

from aitune.torch import jit_config
from aitune.torch.jit.config import JITMode
from aitune.torch.jit.tune import deferred as jit_deferred
from diffusers import DiffusionPipeline

jit_config.mode = JITMode.TUNE_DEFERRED

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers")
pipe.to("cuda")

# First full pipeline step — records samples for every module encountered
pipe("A beautiful landscape")

# Explicitly trigger tuning after the complete pass
jit_deferred()

# All subsequent calls use the tuned pipeline
pipe("A snowy mountain at sunset")
```

## Configuration

The main configuration of the tuning process is in the `config` object - this is the common configuration between ahead-of-time and just-in-time mode. On top of that, there are settings particular for the just-in-time mode in `jit_config`.

This example shows how to get those configuration objects:

```python
import aitune.torch as ait

# main config
ait.config

# just in time config
ait.jit_config
```

### Environment Variables

Control just-in-time tuning behavior with environment variables:

```bash
# Enable just-in-time tuning
export AUTOWRAPT_BOOTSTRAP=aitune_enable_jit_tuning

# Set cache directory
export AITUNE_JIT_CACHE_DIR=/path/to/cache

# Run your script
python your_script.py
```

### Programmatic Configuration

For more control, configure just-in-time tuning in your code:

```python
from aitune.torch import jit_config

# Minimum samples before tuning
jit_config.min_samples = 2

# Require batch axis detection
jit_config.batch_axis_required = True

# Maximum module depth
jit_config.max_depth_level = 2

# Minimum parameters to consider tuning
jit_config.min_parameters = 1000

# Enable/disable graph break detection
jit_config.detect_graph_breaks = True

# Skip specific module types
jit_config.skip_modules = ["BatchNorm2d", "LayerNorm"]

# Add extra package prefixes the JIT patcher must never intercept (on top of the built-in defaults)
jit_config.extra_patch_exclude_packages = ("my_third_party_pkg",)

# Add extra module classes the JIT patcher must never intercept (on top of the built-in defaults)
import torch
jit_config.extra_patch_exclude_modules = (torch.nn.Identity,)

# Set device for tuning
jit_config.device = "cuda"

# Enable dry run mode
jit_config.dry_run = False

# Configure tune strategy (which backends to try and how)
from aitune.torch.backend import TensorRTBackend, TensorRTBackendConfig
from aitune.torch.tune_strategy import FirstWinsStrategy
jit_config.strategy = FirstWinsStrategy(
    backends=[TensorRTBackend(config=TensorRTBackendConfig(use_dynamo=True))],
)
```

### Configuration Options

#### mode

Tuning mode — controls when tuning is triggered.

```python
from aitune.torch.jit.config import JITMode
jit_config.mode = JITMode.TUNE_EAGER  # Default
```

Available values:

- `JITMode.TUNE_EAGER` — tunes automatically after each forward pass once the sample threshold is reached.
- `JITMode.TUNE_DEFERRED` — collects samples but does not tune until `aitune.torch.jit.tune.deferred()` is called explicitly.
- `JITMode.INSPECT` — inspect-only mode; no tuning is performed (see [JIT Inspect](jit_inspect.md)).

#### min_samples

Minimum number of samples to record before attempting tuning.

```python
jit_config.min_samples = 1  # Default: 1
```

AITune needs at minimum 1 sample to perform tuning. Multiple samples are required to detect that the model has dynamic axes. Based on the data seen, it detects input and output shapes with dynamic dimensions and minimum/maximum shapes which are required by some backends. If you cannot control data feed to the model, you can increase this setting so that AITune has enough samples to detect proper edge shapes.

#### max_depth_level

Maximum depth of a module in module hierarchy to be considered for tuning.

```python
jit_config.max_depth_level = 2  # Default: 2
```

**Example**:

- Depth 0: Root module only
- Depth 1: Root and immediate children
- Depth 2: Root, children, and grandchildren

#### min_parameters

Minimum number of parameters for a module to be considered for tuning.

```python
jit_config.min_parameters = 1000  # Default: 0
```

**Why it matters**: Small modules may not benefit from tuning and can be skipped to save time.

#### detect_graph_breaks

Enable graph break detection.

```python
jit_config.detect_graph_breaks = True  # Default: True
```

**Why it matters**: Graph breaks prevent static optimization. When disabled, AITune may attempt to tune modules with dynamic control flow (which will likely fail).

#### skip_modules

List of module class names to skip during tuning.

```python
jit_config.skip_modules = ["BatchNorm2d", "LayerNorm", "Dropout"]
```

**Why it matters**: Some modules (like normalization layers) typically don't benefit from tuning.

#### extra_patch_exclude_packages

Extra package prefixes the JIT patcher must never intercept (matched via `startswith`), on top of the built-in defaults.

```python
jit_config.extra_patch_exclude_packages = ("my_third_party_pkg",)
```

Built-in defaults — always applied — cover internal torch packages (`torch.jit`, `torch._inductor`, `torch._dynamo`, `torch.fx`, `torch.export`) and `torch_tensorrt`. The user-supplied tuple is additive; clearing it does not remove the defaults. Use it to keep AITune away from libraries whose internals must not be wrapped.

#### extra_patch_exclude_modules

Extra module classes the JIT patcher must never intercept (matched via `isinstance`), on top of the built-in defaults.

```python
import torch
jit_config.extra_patch_exclude_modules = (torch.nn.Identity,)
```

Built-in defaults — always applied — cover PyTorch container modules (`ModuleList`, `ModuleDict`, `Sequential`, `ParameterList`, `ParameterDict`); they hold no computation and can be created dynamically during forward passes via slicing. The user-supplied tuple is additive; clearing it does not remove the defaults.

##### `extra_patch_exclude_modules` vs `skip_modules`

Both filters drop modules out of tuning, but at different stages:

| | `extra_patch_exclude_modules` | `skip_modules` |
|---|---|---|
| **Stage** | Patch time — module is never wrapped at all | Tune time — module is wrapped and tracked, but tuning is short-circuited |
| **Match** | `isinstance` against actual classes (subclasses included) | Exact class-name string match |
| **Imports** | Class must be importable when you configure it | Name-only; no import needed |
| **Visibility** | Module is invisible to AITune | Appears in `PatchedModule.print_hierarchy()` as `state=skipped 🚫` |

Use `extra_patch_exclude_modules` when the wrapt proxy itself is the problem (its presence breaks compilation, serialization, etc. — this is exactly why `torch_tensorrt` is in the built-in package exclusions). Use `skip_modules` when you just don't want a class tuned but still want it tracked in inspect output.

#### cache_dir

Directory for caching tuned modules.

```python
from pathlib import Path
jit_config.cache_dir = Path("/path/to/cache")
```

**Note**: Currently, just-in-time mode does not support persistent caching across runs.

#### strategy

Tune strategy to use during tuning. Decides which backends to try and how to choose between them.

```python
from aitune.torch.backend import (
    TensorRTBackend,
    TensorRTBackendConfig,
    TorchInductorJitBackend,
)
from aitune.torch.tune_strategy import FirstWinsStrategy

jit_config.strategy = FirstWinsStrategy(
    backends=[
        TensorRTBackend(config=TensorRTBackendConfig(use_dynamo=True)),
        TorchInductorJitBackend(),
    ],
)
```

Accepts any `TuneStrategy` (e.g. `FirstWinsStrategy`, `MaxThroughputStrategy`, `OneBackendStrategy`). Leave it as `None` (default) to use a `FirstWinsStrategy` over TensorRT (with and without dynamo) and `TorchInductorJitBackend`.

This setting is common for all tuned modules.

## Limitations and Considerations

### 1. No Persistent Caching

Just-in-time tuning does not cache results across runs. Each time you start a new Python interpreter, tuning starts from scratch.

### 2. No Benchmarking

Dynamic axes are detected; however, they cannot be matched against real batch sizes. This is due to missing explicit data source, and hence, AITune cannot control batch size. Without this information, it cannot extrapolate batches to any size, which is required by benchmarking functionality.

### 3. Requires Multiple Samples

At least 2 samples are needed to start tuning. If you need detection of dynamic axes and min/max shape, you should either feed such data into a model or change `min_samples` limits.

## Just-in-Time Tuning vs Ahead-of-Time Tuning

The following table summarizes the difference between those two modes:

| Feature                 | Ahead-of-time         | Just-in-time                  |
|-------------------------|-----------------------|-------------------------------|
| Detecting dynamic axes  | Yes                   | Yes                           |
| Extrapolating batches   | Yes                   | No                            |
| Benchmarking            | Yes                   | No (no extrapolating batches) |
| Modules for tuning      | User has full control | Picked automatically          |
| Selecting tune strategy | Global or per module  | Global                        |
| Available strategies    | All                   | Limited (no benchmarking)     |
| Tune time               | Slow                  | Quick                         |
| Saving artifacts        | Yes                   | No                            |
| Load tuned model time   | Quick                 | Re-tuning required            |
| Code changes required   | Yes                   | No                            |
| Caching                 | Yes                   | No                            |

## Debugging Just-in-Time Tuning

Just-in-time tuning does not require modification of the original python script. To assist or debug the tuning process, there are several features that might be helpful.

### Enable Logging

Make sure your logging level is at least `INFO`. If tuning happens you should be able to see appropriate log entries.

```python
import logging
logging.basicConfig(level=logging.INFO)
```

### Dry Run Mode

To test which modules are captured without an actual tuning, change the following `config` option:

```python
from aitune.torch import jit_config
jit_config.dry_run = True
```

It will log what is about to happen in a real scenario.

### Model Hierarchy

If you would like to see the hierarchy of a model AITune discovered, you can use the following code:

```python
from aitune.torch import PatchedModule

PatchedModule.print_hierarchy()
```

Here is an example from `ResNet`:

```text
JIT Tuning Hierarchy:
├─ ResNet 📊11.7M level=0🪜 state=tuned🎯 (TensorRTBackend) call_count=4
  ├─ Conv2d 📊9.4K level=1🪜 state=detached☑️ call_count=4
  ├─ BatchNorm2d 📊128 level=1🪜 state=detached☑️ call_count=4
  ├─ BasicBlock 📊74.0K level=1🪜 state=detached☑️ call_count=4
    ├─ Conv2d 📊36.9K level=2🪜 state=detached☑️ call_count=4
    ├─ BatchNorm2d 📊128 level=2🪜 state=detached☑️ call_count=4
    ├─ Conv2d 📊36.9K level=2🪜 state=detached☑️ call_count=4
    ├─ BatchNorm2d 📊128 level=2🪜 state=detached☑️ call_count=4

... rest of the hierarchy is abbreviated ...
```

The hierarchy output is a tree view of the model modules that AITune discovered while tracing. Each line represents a module instance, with indentation showing parent-child relationships. The markers provide context about how AITune treats each module:

- Module name (e.g., `ResNet`, `Conv2d`) identifies the layer type or submodule.
- `📊` shows the parameter count for that module (e.g., `11.7M`, `9.4K`).
- `level` is the depth in the module tree (root is `0`).
- `state` indicates how AITune handled the module during tuning (for example `tuned` or `detached`).
- Backend in parentheses (e.g., `TensorRTBackend`) is shown for tuned modules.
- `call_count` is the number of times the module was observed during collection.

Internally a `PatchedModule` can be in the following states:

- INIT: "⏳" before the first forward call; hierarchy is not fully resolved yet.
- RECORDING: "🔴" after the first forward call; hierarchy resolved and collecting samples.
- TUNED: "🎯" tuning succeeded; the module forwards through a tuned backend.
- EAGER: "⚠️" tuning failed or was not possible; module falls back to the original unmodified model.
- SKIPPED: "🚫" explicitly skipped (e.g., in `skip_modules`) and not tuned.
- DETACHED: "☑️" detached because a parent module was tuned, so children are unpatched.

### Tuning History

To investigate what may have failed in the process, you can see the history of just-in-time tuning:

```python
from aitune.torch import PatchedModule

PatchedModule.print_history()
```

Here is abbreviated example from `ResNet`

```text
 'New top module: ResNet 📊11.7M level=0🪜 state=init⏳ call_count=1',
 ...
 'New child module: Linear 📊513.0K level=1🪜 state=init⏳ call_count=1',
 'No graph breaks in ResNet 📊11.7M. Checking took 4.13s',
 'Tuning ResNet 📊11.7M took 8.60s',
 'Unpatching child module: Linear 📊513.0K level=1🪜 state=detached☑️ call_count=4',
 ...
```

Basically, `history` is a list of steps just-in-time tuning took during the process.

## Common Issues and Solutions

### Issue: Modules Not Being Tuned

**Possible causes**:

- Not enough samples (`min_samples` not met)
- Module too small (`min_parameters` threshold)
- Module in `skip_modules` list
- Graph breaks detected
- In some environments the `export AUTOWRAPT_BOOTSTRAP=aitune_enable_jit_tuning` does not start just-in-time tuning. This is a known issue. If it happens, try using a decorator or import to start tuning.

**Solution**:

```python
from aitune.torch import jit_config
import logging

logging.basicConfig(level=logging.DEBUG)
jit_config.min_samples = 2
jit_config.min_parameters = 0
```

### Issue: Strange Errors or Recompilations

Check if the following scenario has happened:

- AITune got enough samples, it tuned a module
- AITune got another sample but with larger min/max shapes

In such a case - backend was tuned for shapes say (1, 10) but it may later get data with shapes (2, 20) - which are out of bound. This may result in backend failure depending on the technology used or a recompilation (e.g. `TorchInductor` backend).

## Next Steps

- Learn about [just-in-time Inspect](jit_inspect.md) for detailed module analysis
- Learn about [AOT Tuning](aot_tuning.md) as an alternative approach
- Explore [Backend Configuration](backends/tensorrt_backend.md)
- Review [Deployment Guide](deployment/deployment.md) for production use
