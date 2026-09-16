---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Ahead-of-Time Tuning Guide"
---

Ahead-of-time tuning is a mode where you explicitly control which modules to tune. This method provides precise control over the tuning process and is recommended for production environments.

## Overview

Ahead-of-time tuning follows a four-step workflow:

1. **Inspect**: Analyze your model or pipeline to identify tuneable modules
2. **Wrap**: Wrap selected modules for tuning
3. **Tune**: Execute the tuning process across different backends
4. **Persist**: Save and load tuned models for later deployment

This approach offers several advantages:

- **Control**: Explicitly choose which modules to tune, pick strategies and backends, and mix different technologies
- **Performance**: Benchmark and select optimal configurations
- **Speed**: Save the tuned model to a deployable artifact to be loaded on the production environment
- **Reproducibility**: Deterministic tuning results

## Quick Start

Here's a complete example using Stable Diffusion:

```python
import aitune.torch as ait
from diffusers import DiffusionPipeline

# Initialize pipeline
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers")
pipe.to("cuda")

# Prepare input data
input_data = [{"prompt": "A beautiful landscape with mountains and a lake"}]

# Step 1: Inspect pipeline to discover modules
modules_info = ait.inspect(pipe, input_data)

# Display discovered modules
modules_info.describe()

# Step 2: Wrap modules for tuning
modules = modules_info.get_modules()
pipe = ait.wrap(pipe, modules)

# Step 3: Tune the pipeline
ait.tune(pipe, input_data)

# Step 4: Save the tuned pipeline
ait.save(pipe, "tuned_pipe.ait")

# Use the tuned pipeline
images = pipe(["A beautiful landscape with mountains and a lake"])
```

## Detailed Workflow

### 1. Inspection Phase

The `inspect` function analyzes your model or pipeline to identify PyTorch modules that can be tuned. For a detailed guide on inspection, see the [AOT Inspect Guide](aot_inspect.md).

### 2. Wrapping Phase

Given the list of modules from the previous step, you can wrap them for tuning. Under the hood, each `torch.nn.Module` is wrapped (imagine a proxy object) with AITune `Module` which intercepts all `forward` calls to get data, tune the module and serve the tuned version.

The following line shows how to wrap modules.

```python
model = ait.wrap(model, modules)
```

You can also specify tuning strategies during wrapping:

```python
import aitune.torch as ait

strategy = ait.OneBackendStrategy(backend=ait.backend.TensorRTBackend())
model = ait.wrap(model, modules, strategy=strategy)
```

If you would like to have more control over picking the modules, you can manually wrap `torch.nn.Module`. When wrapping, you can specify a strategy for each module separately; i.e., you can combine different strategies backends into one model.

```python
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers")
pipe.to("cuda")

pipe.unet = ait.Module(pipe.unet, strategy=strategy_for_unet)
pipe.transformer = ait.Module(pipe.transformer, strategy=strategy_for_transformer)
```

#### User-provided dynamic shapes

When recorded samples do not cover every shape needed in production, pass an explicit shape contract to
`ait.Module`. Each mapping key is a tensor's forward parameter path, and each value describes the tensor's full rank:

- Use an integer for a fixed dimension.
- Use `BatchDim` for the logical batch dimension.
- Use `DynamicDim` for any other bounded dynamic dimension.

`min` and `max` are inclusive. `opt` selects the preferred compilation shape and defaults to `max` when omitted.
Configure `dynamic_shapes` when constructing `ait.Module` directly; `ait.wrap` does not accept per-module shape
definitions.

```python
import aitune.torch as ait

batch = ait.BatchDim("batch", min=1, opt=2, max=8)
height = ait.DynamicDim("spatial", min=224, opt=224, max=512)
width = ait.DynamicDim("spatial", min=224, opt=224, max=512)

model = ait.Module(
    model,
    dynamic_shapes={"x": (batch, 3, height, width)},
)
```

The integer `3` fixes the channel dimension. `height` and `width` are shared because they have the same name, even
though they are separate `DynamicDim` objects. Shared definitions must have the same type and bounds.

Forward input paths are shown in the `input_spec` sample-metadata table in tuning logs. Use the **Semantic Path**
column as the `dynamic_shapes` key. Top-level tensors use their forward parameter name, while tensors nested in
dictionaries, sequences, dataclasses, or supported custom objects include every key, index, or attribute:

```text
Tensors:
Access Path       Semantic Path          Shape
----------------  ---------------------  ----------------
x                 x                      [1, 3, 224, 224]
options["mask"]   ('options', 'mask')    [1, 224, 224]
items[0]          ('items', 0)           [1, 128]
request.image     ('request', 'image')   [1, 3, 512, 512]
```

Use a string such as `"x"` for a top-level path and a tuple such as `("options", "mask")`, `("items", 0)`, or
`("request", "image")` for a nested path. Inputs omitted from the mapping keep using shapes inferred from recorded
samples.

Each dynamic shape definition must match the corresponding input tensor's rank and fixed dimensions, and its ranges
must include all dimension sizes present in the tuning samples.
The explicit definitions then determine the ranges used by supported AOT backends.

See the runnable [ResNet dynamic-shapes example](../../examples/ResNet/README.md#user-provided-dynamic-shapes), which
records `(1, 3, 224, 224)` and runs the tuned model at `(2, 3, 256, 256)`.

### 3. Tuning Phase

The `tune` function executes the actual tuning:

```python
ait.tune(
    func=model,                   # The wrapped callable module or pipeline to tune
    dataset=input_data,           # Dataset to use for tuning (list, Dataset, DataLoaderFactory, or Tensor)
    batch_sizes=[1, 4, 8],         # Optional: Multiple batch sizes. Defaults to [1, 2]
    max_num_batches_per_batch_size=10,  # Max batches per size. Defaults to None (all)
    device="cuda",                  # Device for tuning. Defaults to "cuda:0"
    dry_run=False,                  # Set True to test without tuning
    disable_external_logging=False, # Disable third-party logs
    clear_cache=False,              # Clear AITune cache before tuning
    ignore_failing_modules=True,    # Keep tuning remaining modules when one fails
)
```

#### Tuning Parameters

- **func**: The wrapped callable (model or pipeline)
- **dataset**: Dataset for tuning. It can be a list of samples, `torch.utils.data.Dataset`, `DataLoaderFactory`,  `Tensor` or sequence of tensors, dictionaries, strings
- **batch_sizes**: List of batch sizes to tune against. If not specified, values [1, 2] will be used
- **max_num_batches_per_batch_size**: Maximum number of batches per batch size. If None, all batches will be used
- **device**: Device to use for tuning. Defaults to "cuda:0"
- **dry_run**: If True, performs a dry run without actual tuning
- **disable_external_logging**: Disable logging from external libraries
- **clear_cache**: Clear AITune cache before tuning
- **ignore_failing_modules**: If True, modules that fail tuning fall back to eager execution and tuning continues

Tuning time depends on the tuned modules' size, used strategy, and number of backends. Modules are tuned one by one. If a strategy has many backends to pick from, it takes the one that fulfills specific strategy criteria. Each backend is validated against returning proper numeric results (check against NANs and infinity) and output shapes.

Note: If you specify a batch size that is not a power of 2, it will be used to gather samples but the actual search for maximum throughput will round it up to the nearest power of 2.

### 4. Persistence Phase

Once tuned, you can save your model for later use. This is crucial for production deployments to avoid re-tuning every time:

```python
# Save the tuned model/pipeline
ait.save(pipe, "tuned_model.ait")
```

The tuned artifact will be saved in the `checkpoints` folder. The `save` function creates several files:

- `tuned_model.ait`: The compressed checkpoint containing tuned and original weights
- `tuned_model_sha256_sums.txt`: SHA256 hashes for verification

To do inference, you can load the tuned model/pipeline:

```python
# Note: Initializing the original object is required before loading
pipe = DiffusionPipeline.from_pretrained(...)
pipe = ait.load(pipe, "tuned_model.ait")
# pipe is ready for use
```

## Custom Inference Functions

For complex pipelines, you can provide a custom inference function:

```python
def custom_inference(prompt, num_steps=50):
    """Function forces width, height and number of steps."""
    return pipe(
        prompt=prompt,
        num_inference_steps=num_steps,
        height=1024,
        width=1024,
    )

modules_info = ait.inspect(
    pipe,
    input_data,
    inference_function=custom_inference
)
```

### Inspect and tune with the same workload wrapper

When scalar arguments change module execution or input shapes, put those options in a workload wrapper and use that
same wrapper for both `inspect()` and `tune()`. This is common for diffusion pipelines where you want to tune for
several image sizes, step counts, guidance scales, or sequence-length settings.

The important part is that inspection and tuning must execute the same workload. If you inspect through a wrapper
that uses multiple scalar arguments but later call `ait.tune(pipe, input_data)` directly, tuning records a
different execution path and may miss graph variants or shape ranges.

> **Note for FLUX and Stable Diffusion pipelines:** image size, step count, guidance scale, and sequence-length
> settings are usually scalar keyword arguments on the pipeline call, not tensor samples in the dataset. Put every
> option you want to tune, such as `height`, `width`, `num_inference_steps`, `guidance_scale`, or
> `max_sequence_length`, inside the workload wrapper and pass that wrapper to both `ait.inspect()` and `ait.tune()`.
> Calling `ait.tune(pipe, input_data)` directly after inspecting through a wrapper will not exercise those same
> variants during tuning.

```python
pipe = get_pipeline(model_name=model_name)

sizes = [(256,256), (512,512)]

def call_wrapper(*args, **kwargs):
    for height, width in sizes:
        pipe(
            *args,
            height=height,
            width=width,
            num_inference_steps=28,
            guidance_scale=1.0,
            max_sequence_length=512,
            **kwargs,
        )


input_data = [{"prompt": prompt}]

# Inspect the workload you intend to tune.
modules_info = ait.inspect(
    pipe,
    input_data,
    inference_function=call_wrapper,
)

# Wrap modules selected from that inspection.
pipe = ait.wrap(pipe, modules_info.get_modules())

# Tune through the same wrapper so recorded graphs and scalar options match inspection.
ait.tune(call_wrapper, input_data)
```

With `config.strict_mode=True` (the default), different non-tensor argument values can create separate graphs. Keep
scalar arguments fixed when you want one graph, or exercise each option in the wrapper when those variants should be
tuned.

## Configuration Options

AITune has configuration for the tuning process, and each backend has its configuration.

### Global Configuration

You can configure AITune globally:

```python
from aitune.torch import config

# Set cache directory
config.cache_dir = "/path/to/cache"

# Set minimum samples for tuning
config.min_num_samples = 5

# Set maximum stored samples per graph
config.max_num_samples_stored = 100

# Device to move model after tuning
config.device_after_tuning = "cuda"

# Enable/disable strict mode for input validation
config.strict_mode = True

# Enable/disable HuggingFace integrations
config.enable_transformers_integration = True
config.enable_diffusers_integration = True
```

### Backend-Specific Configuration

Each backend has its own corresponding configuration:

```python
from aitune.torch.backend import TensorRTBackendConfig, TensorRTBackend

config = TensorRTBackendConfig(
    use_cuda_graphs=True,
    workspace_size=1 << 30,  # 1GB
)
backend = TensorRTBackend(config)
```

See backend-specific documentation:

- [TensorRT Backend](backends/tensorrt_backend.md)
- [Torch-Inductor JIT Backend](backends/torch_inductor_jit_backend.md)
- [TorchAO Backend](backends/torchao_backend.md)
- [Torch TensorRT AOT Backend](backends/torch_tensorrt_aot_backend.md)
- [Torch TensorRT JIT Backend](backends/torch_tensorrt_jit_backend.md)

## Dry Run Mode

You can run tuning in dry-run mode. It records samples of data, detects batch and dynamic axes, and detects graphs of execution but does not call the actual backend to tune. This allows debugging if everything is working as expected.

The dry-run mode can be turned on with the proper argument:

```python
import logging

# make sure logging if configured
logging.basicConfig(level=logging.INFO, force=True)
# invoke dry-run tuning
ait.tune(pipe, input_data, dry_run=True)
```

Example output from dry-run
```text
2026-01-26 16:23:44,360 - INFO - ════════════════════════════════════════════════════════════════
2026-01-26 16:23:44,360 - INFO - 🎯 Tuning module: `transformer` (all graphs)
2026-01-26 16:23:44,367 - INFO - ------------------------------------------------------------
2026-01-26 16:23:44,367 - INFO - 🚀 Tuning graph `0` for module `transformer` (DRY RUN):
2026-01-26 16:23:44,368 - INFO -   number of parameters: 2028328000
2026-01-26 16:23:44,368 - INFO -   number of layers: 6
2026-01-26 16:23:44,369 - INFO -   precisions: torch.float16
2026-01-26 16:23:44,369 - INFO -   graph_spec:
2026-01-26 16:23:44,369 - INFO -     input_spec:
 Tensors:
╒═══════════════════════╤═══════════════════════╤══════════════════════════╤═══════════════════╤═══════════════════╤═══════════════╕
│ Access Path           │ Semantic Path         │ Shape                    │ Min Shape         │ Max Shape         │ Dtype         │
╞═══════════════════════╪═══════════════════════╪══════════════════════════╪═══════════════════╪═══════════════════╪═══════════════╡
│ encoder_hidden_states │ encoder_hidden_states │ ['batch0', 333, 4096]    │ [2, 333, 4096]    │ [4, 333, 4096]    │ torch.float16 │
├───────────────────────┼───────────────────────┼──────────────────────────┼───────────────────┼───────────────────┼───────────────┤
│ hidden_states         │ hidden_states         │ ['batch0', 16, 128, 128] │ [2, 16, 128, 128] │ [4, 16, 128, 128] │ torch.float16 │
├───────────────────────┼───────────────────────┼──────────────────────────┼───────────────────┼───────────────────┼───────────────┤
│ pooled_projections    │ pooled_projections    │ ['batch0', 2048]         │ [2, 2048]         │ [4, 2048]         │ torch.float16 │
├───────────────────────┼───────────────────────┼──────────────────────────┼───────────────────┼───────────────────┼───────────────┤
│ timestep              │ timestep              │ ['batch0']               │ [2]               │ [4]               │ torch.float32 │
╘═══════════════════════╧═══════════════════════╧══════════════════════════╧═══════════════════╧═══════════════════╧═══════════════╛
Other:
╒════════════════════════╤════════════════════════╤═════════╕
│ Access Path            │ Semantic Path          │ Value   │
╞════════════════════════╪════════════════════════╪═════════╡
│ joint_attention_kwargs │ joint_attention_kwargs │ None    │
├────────────────────────┼────────────────────────┼─────────┤
│ return_dict            │ return_dict            │ False   │
╘════════════════════════╧════════════════════════╧═════════╛

2026-01-26 16:23:44,370 - INFO -     output_spec:
 Tensors:
╒═══════════════╤═════════════════╤══════════════════════════╤═══════════════════╤═══════════════════╤═══════════════╕
│ Access Path   │ Semantic Path   │ Shape                    │ Min Shape         │ Max Shape         │ Dtype         │
╞═══════════════╪═════════════════╪══════════════════════════╪═══════════════════╪═══════════════════╪═══════════════╡
│ output[0]     │ 0               │ ['batch0', 16, 128, 128] │ [2, 16, 128, 128] │ [4, 16, 128, 128] │ torch.float16 │
╘═══════════════╧═════════════════╧══════════════════════════╧═══════════════════╧═══════════════════╧═══════════════╛

2026-01-26 16:23:44,370 - INFO -   num samples: 1
2026-01-26 16:23:44,370 - INFO -   device: cuda:0
2026-01-26 16:23:44,370 - INFO -   cache_dir: /home/pbazan/.cache/aitune/transformer/0
2026-01-26 16:23:44,371 - INFO -   strategy:
2026-01-26 16:23:44,371 - INFO -     name: First Wins Strategy
2026-01-26 16:23:44,371 - INFO -     description: evaluate backends in order, return first working backend
2026-01-26 16:23:44,371 - INFO -     backends:
2026-01-26 16:23:44,371 - INFO -       TensorRTBackend(quantization_config=None)
2026-01-26 16:23:44,371 - INFO -       TorchInductorJitBackend()
2026-01-26 16:23:44,372 - INFO -       TorchEagerBackend()
2026-01-26 16:23:44,372 - INFO - ✅ Tuning module: `transformer` (all graphs) completed.
```

## Next Steps

- Learn about [Ahead-of-time Inspect](aot_inspect.md) for detailed module analysis
- Learn about [Just-in-Time (JIT) Tuning](jit_tuning.md) as an alternative approach
- Explore [Backend Configuration](backends/tensorrt_backend.md)
- Review [Tune Strategies](tune_strategies/tune_strategies.md)
- See [Deployment Guide](deployment/deployment.md)
