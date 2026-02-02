<!--
Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
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
from aitune.torch import patch_for_jit_tuning

@patch_for_jit_tuning
def inference_pipeline(model, input_data):
    """This function will have JIT tuning enabled."""
    with torch.no_grad():
        return model(input_data)

# Your model
model = YourModel().cuda()
input_data = torch.randn(1, 3, 224, 224, device="cuda")

# Tuning happens automatically when the decorated function is called
output = inference_pipeline(model, input_data)
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
        self.child_a = Linear(10, 20)
        self.child_b = Linear(10, 20)

    def forward(self, x):
        if x.sum() > 0:  # Graph break: conditional on input
            return self.child_a(x)
        else:
            return self.child_b(x)
```

When AITune detects a graph break, it skips tuning that module and attempts to tune this module's children.

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

# Set device for tuning
jit_config.device = "cuda"

# Enable dry run mode
jit_config.dry_run = False

# Configure backends
from aitune.torch.backend import TensorRTBackend, TensorRTBackendConfig
jit_config.backends = [
    TensorRTBackend(config=TensorRTBackendConfig(use_dynamo=True)),
]
```

### Configuration Options

#### min_samples

Minimum number of samples to record before attempting tuning.

```python
jit_config.min_samples = 2  # Default: 2
```

AITune needs at minimum 2 samples to detect the model's hierarchy. Based on the data seen, it also detects input and output shapes in order to detect dynamic dimensions and minimum/maximum shapes which are required by some backends. If you cannot control data feed to the model, you can increase this setting so that AITune has enough samples to detect proper edge shapes.

#### batch_axis_required

The just-in-time tuning starts based on the `min_samples` and `batch_axis_required`. The latter prevents premature tuning if the AITune has not detected any dynamic axes, i.e., when the data fed was just of the same batch size. If you aren't concerned about batching (constant, in particular `batch size = 1`) you can turn this flag to `False`.

```python
jit_config.batch_axis_required = True  # Default: True
```

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

#### cache_dir

Directory for caching tuned modules.

```python
from pathlib import Path
jit_config.cache_dir = Path("/path/to/cache")
```

**Note**: Currently, just-in-time mode does not support persistent caching across runs.

#### backends

List of backends to try during tuning.

```python
from aitune.torch.backend import (
    TensorRTBackend,
    TensorRTBackendConfig,
    TorchInductorBackend,
)

jit_config.backends = [
    TensorRTBackend(config=TensorRTBackendConfig(precision="fp16")),
    TorchInductorBackend(),
]
```

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
- EAGER: "⚠️" tuning failed or was not possible; module runs in eager mode.
- SKIPPED: "🚫" explicitly skipped (e.g., in `skip_modules`) and not tuned.
- DETACHED: "☑️" detached because a parent module was tuned, so children are unpatched.

### Tuning History

To investigate what may have failed in the process, you can see the history of just-in-time tuning:

```python
from aitune.torch import PatchedModule

print(PatchedModule.history)
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
jit_config.batch_axis_required = False
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
