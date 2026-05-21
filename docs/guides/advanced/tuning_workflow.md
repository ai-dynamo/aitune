---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Tuning Workflow"
---

This guide provides an in-depth look at AITune's tuning process, explaining how samples are gathered, how modules are tuned, and how strategies and backends work together to optimize your models.

## Overview

The AITune tuning workflow is structured as follows:

1. **Sample Gathering**: Execute the model with different batch sizes to collect metadata
2. **Graph Detection**: Identify unique computational graphs based on input characteristics
3. **Module Tuning**: Optimize each wrapped module sequentially
4. **Strategy Execution**: Apply a tuning strategy to select the best backend
5. **Backend Activation**: Set up the optimized backends for inference

## The Tuning Process

### 1. Sample Gathering Phase

When you call `ait.tune()`, AITune first enters the sample gathering phase:

```python
import aitune.torch as ait

# Wrap your module
module = ait.Module(model, "my_model")

# Tune with different batch sizes
ait.tune(module, dataset, batch_sizes=[1, 2, 4, 8])
```

During this phase:

- The function/model is executed with samples from the dataset
- Each execution uses a different batch size (from `batch_sizes` parameter)
- **Metadata is recorded** for each execution:
  - Input/Output shapes, dtypes, and structure
  - Batch size information (stored in global context)
- Wrapped modules automatically detect and record this metadata

**Key Point**: At least 2 different batch sizes are required to detect batch dimensions. If only one batch size is provided, AITune assumes static shapes.

#### Sample Generation

AITune uses `samples_generator` to iterate through the dataset:

```python
for batch_size, args, kwargs in samples_generator(dataset, batch_sizes, max_num_batches_per_batch_size):
    with global_context:
        global_context.set(BATCH_SIZE_KEY, batch_size)
        with torch.no_grad():
            func(*args, **kwargs)
```

- **`batch_size`**: Current batch size being processed
- **`args`, `kwargs`**: Actual data samples for this batch
- **`max_num_batches_per_batch_size`**: Limits how many model executions are done per batch size (useful to limit large datasets)

The global context tracks the current batch size, allowing wrapped modules to correlate shape changes with batch size changes.

### 2. Graph Detection

As samples are collected, wrapped modules detect unique computational graphs:

```python
# Example: Different input structures create different graphs
module(torch.randn(1, 10))  # Graph 0
module(torch.randn(1, 10), mask=True)  # Graph 1 (different kwargs)
module(torch.randn(1, 10, 5))  # Graph 2 (different tensor rank)
```

**Graph Identity Rules**:

- Tensors with different ranks → different graphs
- Different tensor shapes but same rank → same graph (dynamic shapes)
- Different non-tensor arguments (in strict mode) → different graphs

Note:

- If strict mode is turned off, only tensor data is taken into account when detecting graphs. The strict mode can be turned off with:

```python
import aitune.torch as ait

ait.config.strict_mode = False  # by default this is turned on.
```

- For a particular graph, there is only a limited number of samples collected to limit memory usage. This threshold can be set with:

```python
import aitune.torch as ait

ait.config.max_num_samples_stored = 10
```

Each unique graph is represented by a `GraphSpec` containing:

- **Name**: Unique identifier (e.g., "0", "1", "2")
- **Input Spec**: `SampleMetadata` describing expected inputs
- **Output Spec**: `SampleMetadata` describing expected outputs

**Batch and Dynamic Dimensions**:

After seeing multiple samples, AITune identifies:

- **Batch dimensions** (e.g., `batch0`): Scale proportionally with batch size
- **Dynamic dimensions** (e.g., `dim0`): Vary independently of batch size
- **Static dimensions**: Never change

Example:

```text
# After seeing samples: with batch_size=1 and batch_size=2
tensor[1, 10, 512] # for batch_size=1
tensor[2, 50, 512] # for batch_size=2

# Detected shape are:
tensor['batch0', 'dim1', 512]  # First dim is batch, second is dynamic, third is static
```

See also [Execution Graphs](execution_graphs.md) - in depth explanation of detecting graphs and sample metadata.

### 3. Module-by-Module Tuning

After sample gathering, AITune tunes each wrapped module sequentially:

```python
for module in MODULE_REGISTRY.modules.values():
    # Deactivate other modules to free memory
    for other_module in MODULE_REGISTRY.modules.values():
        if other_module != module:
            other_module.deactivate()

    # Tune this module
    module.tune(device=device, dry_run=dry_run)
```

**Why Sequential?**

- **Memory Management**: Deactivating other modules frees GPU memory for tuning
- **Isolation**: Prevents interference between module optimizations
- **Predictability**: Each module gets full system resources

For each module, each graph is tuned separately, i.e., a strategy is called for data captured for the specific graph only and:

- Strategy tries to build a backend or backends and select the best one
- Each backend is validated against outputs, i.e., tensor shapes, values, NaNs (not a number)
- Strategies also profile a Torch eager baseline and reject or fall back from correct backends that do not beat the baseline unless baseline validation is disabled.

### 4. Strategy Execution

The tuning strategy determines which backend(s) to try and how to select the best one. AITune provides three built-in strategies:

#### FirstWinsStrategy

Tries backends in priority order and returns the first one that builds, validates correctness, and meets the performance threshold against the Torch eager baseline. Not every backend can handle every model (e.g., TensorRT may fail during ONNX export, Torch Inductor may hit graph breaks), so this strategy provides automatic fallback instead of aborting.

```python
strategy = ait.FirstWinsStrategy([
    ait.backend.TensorRTBackend(),
    ait.backend.TorchInductorJitBackend(),
])
```

**Workflow**:

1. Try each backend in order
2. Build the backend with the module and graph spec
3. Validate correctness by comparing outputs
4. Profile against the Torch eager baseline
5. Return the first backend that passes the performance threshold

**Use Case**: Fast tuning with automatic fallback, especially for models you haven't validated against every backend.

#### OneBackendStrategy

Uses exactly one backend, failing immediately with the original error if it cannot build. Unlike `FirstWinsStrategy` with a single backend, `OneBackendStrategy` surfaces the original exception rather than catching it.

```python
strategy = ait.OneBackendStrategy(ait.backend.TorchInductorJitBackend())
```

**Workflow**:

1. Build the specified backend
2. Validate correctness
3. Profile against the Torch eager baseline
4. Return the backend when it passes the performance threshold

**Use Case**: Production with a validated backend where you want deterministic behavior.

#### MaxThroughputStrategy

Before actual tuning, this strategy tries to estimate `max_batch_size`. It does so by incrementing `batch_size` (in powers of 2) and measuring throughput using the original module. The `max_batch_size` is picked for the best throughput and then is used for selecting the best backend:

```python
strategy = ait.MaxThroughputStrategy([
    ait.backend.TensorRTBackend(),
    ait.backend.TorchInductorJitBackend(),
    ait.backend.TorchEagerBackend(),
])
```

**Workflow**:

1. Estimate `max_batch_size`
2. Try each backend in order
3. Build and validate each working backend
4. Profile throughput for each backend given `max_batch_size`
5. Return the fastest user backend when it beats the Torch eager baseline, otherwise fall back to eager

**Use Case**: When performance is critical and you want the absolute fastest backend.

### 5. Backend Building and Validation

A backend represents a different technology for tuning a torch module, e.g., `TensorRT`, `TorchInductor`, and it is used by a strategy. Before it is used, it acts as a blueprint, i.e., it is copied, and each copy is used to build, validate, and activate a particular tuned module. This is done so that it has no side-effects on different modules or graphs.

Each backend is a small state machine that enforces safe usage:

- `INIT` → `ACTIVE`: `build()` succeeds. A build is allowed only once and must set up a runnable backend.
- `ACTIVE` → `INACTIVE`: `deactivate()` releases resources (and clears compiler/runtime caches).
- `INACTIVE` → `ACTIVE`: `activate()` restores the backend for inference.
- `CHECKPOINT_LOADED` → `ACTIVE` or `DEPLOYED`: A backend created from a checkpoint can be activated for tinkering or
  deployed for final use.
- `ACTIVE` → `DEPLOYED`: `deploy()` finalizes the backend. After this, state changes are not allowed.

The backend's state is governed by the strategy and the user must not change it. After a module is successfully tuned, it can be used to do inference - the backend will be in `ACTIVE` or `DEPLOYED` states.

#### Validation Phase

After building a backend, the strategy tries to validate it. This is enabled by default and can be turned off with `strategy.enable_correctness_check(False)`.

AITune validates correctness by running the tuned backend on sample data. These checks are required to ensure the backend is correctly built:

1. Python basic types `int`, `float` must be finite.
2. Tensors values must be finite.
3. All nested structures are checked against points 1 and 2.
4. Tensor shapes must match against original data i.e. static, dynamic and batch axes must match.

This ensures tuning does not break the module.

### 6. Backend Activation

After tuning is complete, all tuned modules are activated:

```python
for module in MODULE_REGISTRY.modules.values():
    module.activate()
```

**Activation** means:

1. Load the optimized backend into memory
2. Prepare for inference
3. Route future calls to the optimized backend

Now your model is ready for optimized inference!

## Workflow Customization

The workflow can be adapted by implementing a custom strategy or a backend.

### Custom Strategy

If you would like to write a custom strategy, extend the `TuneStrategy` class and implement the `_tune` method:

```python
    def _tune(
        self,
        module: nn.Module,
        name: str,
        graph_spec: GraphSpec,
        data: list[Sample],
        device: torch.device,
        cache_dir: Path,
    ) -> Backend:
        """Tunes given torch module with provided graph_spec and data.

        Note: each tuning operation should perform a deep copy of a backend as tuning could be called multiple times for the
        same module, i.e., if there are different graph specs

        Returns:
            The tuned and activated backend.

        Raises:
            RuntimeError: if the backend fails any check.
        """
        ...
```

If needed, you can also extend the`_pre_tune` and `_post_tune` methods, which are invoked before and after the `_tune`.

### Custom Backend

To write a custom backend, extend the `Backend` base class and `BackendConfig` data class which serves as a config for your custom backend. The following methods are required to be implemented:

- `key()` - Returns a stable identifier for the backend/config combination; used for caching and lookup.
- `describe()` - Returns a short human-readable description of the backend/config changes.
- `to_dict()` - Serializes backend state; includes `Path` objects for artifacts to bundle in checkpoints.
- `from_dict()` - Reconstructs a backend instance from the serialized state.
- `is_jit` - Boolean property indicating whether the backend is of just-in-time type. This information helps AITune manage resources as just-in-time backends require the original torch module for activation. Otherwise, the original module can be offloaded to system memory.
- `_build()` - Builds backend artifacts for a specific module/graph and returns a ready backend.
- `_activate()` - Loads/initializes the backend for inference after it was inactive or checkpoint-loaded.
- `_deactivate()` - Releases runtime resources and makes the backend inactive.
- `_deploy()` - Finalizes the backend for deployment; after this it cannot change state.
- `_infer()` - Executes inference with the backend for the provided inputs.

For serialization with `ait.load` and `ait.save`, the `to_dict` and `from_dict` methods are used. Anything placed in the dictionary will be saved and restored as a checkpoint. Path objects will be copied to the checkpoints folder and can be used by a backend in the `_deploy` method.

## Monitoring the Workflow

AITune provides detailed logging throughout the tuning process.

Example logs from tuning ResNet (an example is placed in the `examples/ResNet` folder):

```text
2026-02-04 13:32:38,340 - INFO - ════════════════════════════════════════════════════════════════
2026-02-04 13:32:38,340 - INFO - 🎯 Tuning module: `example-resnet50` (all graphs)
2026-02-04 13:32:38,342 - INFO - ------------------------------------------------------------
2026-02-04 13:32:38,342 - INFO - 🚀 Tuning graph `0` for module `example-resnet50`:
2026-02-04 13:32:38,342 - INFO -   number of parameters: 25557032
2026-02-04 13:32:38,342 - INFO -   number of layers: 10
2026-02-04 13:32:38,342 - INFO -   precisions: torch.float32
```

Now the logs will show the inputs and outputs of the module:

```text
2026-02-04 13:32:38,342 - INFO -   graph_spec:
2026-02-04 13:32:38,343 - INFO -     input_spec:
 Tensors:
╒═══════════╤════════╤═════════════════════════╤══════════════════╤══════════════════╤═══════════════╕
│ Locator   │ Name   │ Shape                   │ Min Shape        │ Max Shape        │ Dtype         │
╞═══════════╪════════╪═════════════════════════╪══════════════════╪══════════════════╪═══════════════╡
│ [0]       │ args_0 │ ['batch0', 3, 224, 224] │ [1, 3, 224, 224] │ [4, 3, 224, 224] │ torch.float32 │
╘═══════════╧════════╧═════════════════════════╧══════════════════╧══════════════════╧═══════════════╛

2026-02-04 13:32:38,343 - INFO -     output_spec:
 Tensors:
╒═══════════╤═════════╤══════════════════╤═════════════╤═════════════╤═══════════════╕
│ Locator   │ Name    │ Shape            │ Min Shape   │ Max Shape   │ Dtype         │
╞═══════════╪═════════╪══════════════════╪═════════════╪═════════════╪═══════════════╡
│           │ outputs │ ['batch0', 1000] │ [1, 1000]   │ [4, 1000]   │ torch.float32 │
╘═══════════╧═════════╧══════════════════╧═════════════╧═════════════╧═══════════════╛
```

As you can see, AITune detected the batch axis as the first one, hence the name `batch0` and input shapes `3x224x224`, i.e., batch of images, and output shape `1000`, i.e., batch of categories.

Next, you can see the Max Throughput Strategy builds backends one by one:

```text
2026-02-04 13:32:38,343 - INFO -   num samples: 1
2026-02-04 13:32:38,343 - INFO -   device: cuda:0
2026-02-04 13:32:38,343 - INFO -   cache_dir: /home/pbazan/.cache/aitune/example-resnet50/0
2026-02-04 13:32:38,343 - INFO -   strategy:
2026-02-04 13:32:38,343 - INFO -     name: Max Throughput Strategy
2026-02-04 13:32:38,343 - INFO -     description: evaluate all backends, return backend with maximum throughput
2026-02-04 13:32:38,343 - INFO -     backends:
2026-02-04 13:32:38,343 - INFO -       TensorRTBackend(quantization_config=ONNXQuantizationConfig(precision='int8', calibration_method='max', use_model_opt_post_processing=False))
2026-02-04 13:32:38,343 - INFO -       TensorRTBackend(use_dynamo=False,quantization_config=ONNXQuantizationConfig(precision='int8', calibration_method='max', use_model_opt_post_processing=False))
2026-02-04 13:32:38,343 - INFO -       TensorRTBackend(quantization_config=ONNXAutoCastConfig(precision='fp16', keep_io_types=True))
2026-02-04 13:32:38,343 - INFO -       TensorRTBackend(use_dynamo=False,quantization_config=ONNXAutoCastConfig(precision='fp16', keep_io_types=True))
2026-02-04 13:32:38,343 - INFO -       TorchAOBackend(quantization_config=Int8WeightOnlyConfig())
2026-02-04 13:32:38,343 - INFO -       TorchInductorJitBackend(autocast_enabled=True,autocast_dtype=torch.float16)
2026-02-04 13:32:38,343 - INFO -       TorchEagerBackend()
2026-02-04 13:32:38,343 - INFO - ⏳ Executing strategy `MaxThroughputStrategy` on module `example-resnet50` (graph: 0)
2026-02-04 13:32:38,343 - INFO - 🤖 backend: TensorRTBackend(quantization_config=ONNXQuantizationConfig(precision='int8', calibration_method='max', use_model_opt_post_processing=False))
2026-02-04 13:32:38,343 - INFO -     🔄 in progress...please wait
2026-02-04 13:32:58,024 - INFO -     ✅ backend built
2026-02-04 13:32:58,030 - INFO -     ✅ backend validated
2026-02-04 13:32:58,044 - INFO -     ✅ backend profiled - throughput: 10051.17 samples/s, batch size: 4
2026-02-04 13:32:58,044 - INFO -     🎯 new best throughput for TensorRTBackend(quantization_config=ONNXQuantizationConfig(precision='int8', calibration_method='max', use_model_opt_post_processing=False)) is 10051.17 samples/s, batch size: 4
2026-02-04 13:32:58,044 - INFO -     ⏱️ completed in 19.70s
2026-02-04 13:32:58,044 - INFO - 🤖 backend: TensorRTBackend(use_dynamo=False,quantization_config=ONNXQuantizationConfig(precision='int8', calibration_method='max', use_model_opt_post_processing=False))
2026-02-04 13:32:58,044 - INFO -     🔄 in progress...please wait
2026-02-04 13:33:14,828 - INFO -     ✅ backend built
2026-02-04 13:33:14,829 - INFO -     ✅ backend validated
2026-02-04 13:33:14,843 - INFO -     ✅ backend profiled - throughput: 10624.58 samples/s, batch size: 4
2026-02-04 13:33:14,843 - INFO -     🎯 new best throughput for TensorRTBackend(use_dynamo=False,quantization_config=ONNXQuantizationConfig(precision='int8', calibration_method='max', use_model_opt_post_processing=False)) is 10624.58 samples/s, batch size: 4
2026-02-04 13:33:14,844 - INFO -     ⏱️ completed in 16.80s
2026-02-04 13:33:14,845 - INFO - 🤖 backend: TensorRTBackend(quantization_config=ONNXAutoCastConfig(precision='fp16', keep_io_types=True))
2026-02-04 13:33:14,845 - INFO -     🔄 in progress...please wait
2026-02-04 13:33:36,037 - INFO -     ✅ backend built
2026-02-04 13:33:36,038 - INFO -     ✅ backend validated
2026-02-04 13:33:36,054 - INFO -     ✅ backend profiled - throughput: 7812.03 samples/s, batch size: 4
2026-02-04 13:33:36,054 - INFO -     ⏱️ completed in 21.21s
2026-02-04 13:33:36,056 - INFO - 🤖 backend: TensorRTBackend(use_dynamo=False,quantization_config=ONNXAutoCastConfig(precision='fp16', keep_io_types=True))
2026-02-04 13:33:36,056 - INFO -     🔄 in progress...please wait
2026-02-04 13:33:55,150 - INFO -     ✅ backend built
2026-02-04 13:33:55,151 - INFO -     ✅ backend validated
2026-02-04 13:33:55,167 - INFO -     ✅ backend profiled - throughput: 8096.70 samples/s, batch size: 4
2026-02-04 13:33:55,167 - INFO -     ⏱️ completed in 19.11s
2026-02-04 13:33:55,169 - INFO - 🤖 backend: TorchAOBackend(quantization_config=Int8WeightOnlyConfig())
2026-02-04 13:33:55,169 - INFO -     🔄 in progress...please wait
2026-02-04 13:34:04,296 - INFO -     ✅ backend built
2026-02-04 13:34:04,305 - INFO -     ✅ backend validated
2026-02-04 13:34:10,986 - INFO -     ✅ backend profiled - throughput: 23666.16 samples/s, batch size: 4
2026-02-04 13:34:10,986 - INFO -     🎯 new best throughput for TorchAOBackend(quantization_config=Int8WeightOnlyConfig()) is 23666.16 samples/s, batch size: 4
2026-02-04 13:34:10,995 - INFO -     ⏱️ completed in 15.83s
2026-02-04 13:34:10,995 - INFO - 🤖 backend: TorchInductorJitBackend(autocast_enabled=True,autocast_dtype=torch.float16)
2026-02-04 13:34:10,996 - INFO -     🔄 in progress...please wait
2026-02-04 13:34:24,011 - INFO -     ✅ backend built
2026-02-04 13:34:24,013 - INFO -     ✅ backend validated
2026-02-04 13:34:32,533 - INFO -     ✅ backend profiled - throughput: 3637.62 samples/s, batch size: 4
2026-02-04 13:34:32,533 - INFO -     ⏱️ completed in 21.54s
2026-02-04 13:34:32,533 - INFO - 🤖 backend: TorchEagerBackend()
2026-02-04 13:34:32,533 - INFO -     🔄 in progress...please wait
2026-02-04 13:34:32,536 - INFO -     ✅ backend built
2026-02-04 13:34:32,538 - INFO -     ✅ backend validated
2026-02-04 13:34:32,580 - INFO -     ✅ backend profiled - throughput: 2974.79 samples/s, batch size: 4
2026-02-04 13:34:32,580 - INFO -     ⏱️ completed in 47.56ms
```

Finally the winning backend is presented:

```text
2026-02-04 13:34:32,580 - INFO - 🎯 Strategy MaxThroughputStrategy execution finished:
2026-02-04 13:34:32,581 - INFO - ✅ Selected TorchAOBackend(quantization_config=Int8WeightOnlyConfig()) for module example-resnet50 and graph spec Name=0
Input_spec:
Tensors:
╒═══════════╤════════╤═════════════════════════╤══════════════════╤══════════════════╤═══════════════╕
│ Locator   │ Name   │ Shape                   │ Min Shape        │ Max Shape        │ Dtype         │
╞═══════════╪════════╪═════════════════════════╪══════════════════╪══════════════════╪═══════════════╡
│ [0]       │ args_0 │ ['batch0', 3, 224, 224] │ [1, 3, 224, 224] │ [4, 3, 224, 224] │ torch.float32 │
╘═══════════╧════════╧═════════════════════════╧══════════════════╧══════════════════╧═══════════════╛
Output_spec:
Tensors:
╒═══════════╤═════════╤══════════════════╤═════════════╤═════════════╤═══════════════╕
│ Locator   │ Name    │ Shape            │ Min Shape   │ Max Shape   │ Dtype         │
╞═══════════╪═════════╪══════════════════╪═════════════╪═════════════╪═══════════════╡
│           │ outputs │ ['batch0', 1000] │ [1, 1000]   │ [4, 1000]   │ torch.float32 │
╘═══════════╧═════════╧══════════════════╧═════════════╧═════════════╧═══════════════╛
.
2026-02-04 13:34:32,581 - INFO -    Batch size: 4, throughput: 23666.16 samples/s
2026-02-04 13:34:32,581 - INFO - ⏱️ Tune `MaxThroughputStrategy`: completed in 1.90min
2026-02-04 13:34:32,581 - INFO - ✅ Tuning module: `example-resnet50` (all graphs) completed.
2026-02-04 13:34:32,581 - INFO - ════════════════════════════════════════════════════════════════
```

If you do not see logs, make sure the logger is configured to at least `INFO` level:

```python
import logging

logging.basicConfig(level="INFO", force=True)
```

## See Also

- [Execution Graphs](execution_graphs.md) - Understanding graph detection
- [Tune Strategies](../tune_strategies/tune_strategies.md) - Strategy reference
