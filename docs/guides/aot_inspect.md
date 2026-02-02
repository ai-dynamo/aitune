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

# Ahead-of-time Inspect Guide

The `inspect` function is a powerful tool for analyzing PyTorch models and pipelines. It helps you understand model structure, identify tuneable modules, and gather execution statistics. It can also be a first step to pick modules before ahead-of-time tuning.

## Overview

Inspection provides:

- **Module Discovery**: Automatically finds all PyTorch modules in your model or pipeline
- **Execution Tracking**: Identifies which modules are executed during inference
- **Performance Profiling**: Measures execution time for each module
- **Input and output data types**: records model input and its results

## Basic Usage

```python
import aitune.torch as ait

# Inspect a model
modules_info = ait.inspect(model, input_data)

# Display results
modules_info.describe()
```

## Inspection Parameters

### Complete Signature

```python
modules_info = ait.inspect(
    obj=model,                      # Model or pipeline to inspect
    dataset=input_data,             # Representative input data
    inference_function=None,        # Optional custom inference function
    number_of_iterations=10,        # Iterations for profiling
    warmup_iterations=5,            # Warmup iterations
    min_depth=0,                    # Minimum depth for module search
    max_depth=5,                    # Maximum depth for module search
)
```

### Parameter Details

#### obj (Required)

The object to inspect. Can be:

- `torch.nn.Module`: Any PyTorch module
- Callable: Any callable containing PyTorch modules (e.g., HuggingFace pipelines)

```python
# PyTorch module
model = torchvision.models.resnet50()
modules_info = ait.inspect(model, input_data)

# Diffusion pipeline
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("...")
modules_info = ait.inspect(pipe, input_data)
```

#### dataset (Required)

This is the source of data for your model. It can be:

- `torch.Tensor`
- sequence of strings, tensors, or dictionaries. The collate function is used to stack samples into batches.
- `torch.utils.data.Dataset`

```python
# Single tensor with batch dimension
input_data = torch.randn(1, 3, 224, 224)

# List of strings
input_data = [
    "prompt1", "prompt2"
]

# List of tensors
input_data = [torch.randn(3, 224, 224) for _ in range(4)]  # 4 random images, notice there is no batch dimension

# List of dictionaries
input_data = [
    {"input_ids": torch.tensor([...]), "attention_mask": torch.tensor([...])},
]
```

For customization you can use `ait.DataLoaderFactory` class.

#### inference_function (Optional)

Custom function for running inference. Useful for complex execution logic:

```python
def custom_inference(prompt, steps=50):
    """Function forces width, height and number of steps."""
    return pipe(
        prompt=prompt,
        width=1024,
        height=1024,
        num_inference_steps=steps,
    )

modules_info = ait.inspect(
    pipe,
    input_data=[{"prompt": "test"}],
    inference_function=custom_inference
)
```

#### number_of_iterations (Default: 10)

Number of iterations for profiling execution time:

```python
# Quick inspection
modules_info = ait.inspect(model, input_data, number_of_iterations=3)
```

#### warmup_iterations (Default: 5)

Warmup iterations before profiling to stabilize measurements:

```python
modules_info = ait.inspect(
    model,
    input_data,
    warmup_iterations=3  # Speed up inspection by reducing warmup iterations
)
```

#### min_depth (Default: 0)

Minimum depth level for module discovery. Increase if root-level modules don't work:

```python
# Start from root level (default)
modules_info = ait.inspect(model, input_data, min_depth=0)

# Skip root, inspect children
modules_info = ait.inspect(model, input_data, min_depth=1)
modules_info = ait.inspect(model, input_data, min_depth=2)
```

#### max_depth (Default: 5)

If a nested (child) module has a larger depth than `max_depth` it will be skipped from inspection.

Both `min_depth` and `max_depth` narrow the inspection search to a reasonable range.

## Working with InspectedModulesInfo

The `inspect` function returns an `InspectedModulesInfo` object with several useful methods:

### describe()

Display comprehensive information about discovered modules:

```python
modules_info = ait.inspect(model, input_data)
modules_info.describe()
```

Output example for `stable-diffusion-3-medium-diffusers`:

```text
Module Execution Summary:
==========================================================================================================================================
    Module Name        Calls    Total Time (s)    Avg Time (s)    % of Total    # of params      # of layers           precisions
------------------------------------------------------------------------------------------------------------------------------------------
decoder                      1           0.1760           0.1760        2.80%         49545475                6              torch.float16
text_encoder                 2           0.0057           0.0029        0.09%        123650304                2              torch.float16
text_encoder_2               2           0.0124           0.0062        0.20%        694659840                2              torch.float16
text_encoder_3               2           0.0783           0.0391        1.24%       4762310656                2  torch.float16, torch.float32
transformer                 28           5.9934           0.2140       95.17%       2028328000                6              torch.float16
------------------------------------------------------------------------------------------------------------------------------------------
Total execution time: 6.297374 seconds
Number of batch iterations: 1
==========================================================================================================================================
```

### get_modules()

This function allows getting found modules. Its basic usage returns all of them:

```python
# Get all executed modules
modules = modules_info.get_modules()
```

You can place additional criteria:

- **min_execution_percentage** - minimum percentage of total execution count e.g. .9
- **limit** - maximum number of modules to return, e.g., 5

If those criteria are not sufficient, you can manually filter modules:

```python
modules_info = ait.inspect(model, input_data)

# Get only transformer blocks
transformer_modules = [
    m for m in modules_info.get_modules()
    if "transformer" in m.name.lower()
]
```

## Troubleshooting

### Issue: No modules found

```python
# Solution: Increase min_depth or check object structure
modules_info = ait.inspect(model, input_data, min_depth=1)
```

## Summary

Ahead-of-time inspection can be a first step in exploring your model's structure and performance. It can also be used to select modules for tuning. For details on the tuning workflow, head to the [AOT Tuning Guide](aot_tuning.md).
