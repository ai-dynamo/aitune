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

# Execution Graphs

## Overview

During the tuning process, AITune analyzes module inputs to detect unique **execution graphs**—distinct computational paths through your model based on input characteristics. Understanding execution graphs is crucial because:

- **Separate Optimization**: Each graph is tuned independently with its own optimized backend
- **Dynamic Shape Support**: Graphs capture relationships between batch sizes, dynamic dimensions, and static shapes
- **Input Routing**: At inference time, inputs are automatically routed to the correct optimized graph

## What Defines an Execution Graph?

AITune creates a new execution graph when it encounters inputs that differ in these ways:

- **Tensor Rank**: Tensors with different numbers of dimensions

   ```python
   module(torch.randn(1, 10))       # Graph 0: rank-2 tensor
   module(torch.randn(1, 10, 5))    # Graph 1: rank-3 tensor (different graph!)
   ```

- **Argument Structure**: Different combinations of positional and keyword arguments

   ```python
   module(torch.randn(1, 10))              # Graph 0
   module(torch.randn(1, 10), mask=True)   # Graph 1: additional kwarg
   ```

- **Non-Tensor Arguments** (in strict mode): Different primitive values or configurations

   ```python
   module(x, mode="train")    # Graph 0
   module(x, mode="eval")     # Graph 1 (if strict_mode=True)
   ```

**Important**: Tensors with the same rank but different shapes belong to the **same graph**. AITune handles shape variations through dynamic shape tracking (batch axes and dynamic dimensions).

## Graph Detection in the Tuning Workflow

Execution graphs are detected during the **Sample Gathering Phase** of tuning:

1. Your model is executed with samples from the dataset
2. Each wrapped module records input/output metadata using `SampleMetadata`
3. AITune compares metadata to identify unique graph patterns
4. Each unique pattern becomes a separate `GraphSpec`
5. During **Module Tuning**, each graph is optimized independently

For a complete overview of the tuning process, see [Tuning Workflow](tuning_workflow.md).

## What This Guide Covers

This guide explores the technical details of execution graphs through the lens of `SampleMetadata`, the core class responsible for:

- Capturing tensor metadata (shapes, dtypes, structure)
- Detecting batch axes vs. dynamic dimensions
- Tracking shape ranges (min/max) across samples
- Enabling dynamic batching and shape inference

By the end of this guide, you'll understand how AITune:

- Identifies which inputs belong to the same graph
- Learns dynamic shape patterns from multiple samples
- Uses this information to optimize each execution path separately

## Introduction to SampleMetadata

### What is SampleMetadata?

`SampleMetadata` is a class designed to capture and track metadata about function inputs and outputs, particularly focusing on PyTorch tensors. It serves several important purposes:

- **Tensor Tracking**: Automatically discovers and tracks all tensors in complex nested data structures
- **Shape Inference**: Learns about dynamic dimensions and batch axes by observing multiple samples
- **Model Optimization**: Enables optimization backends to understand input/output characteristics
- **Dynamic Batching**: Supports scaling tensors to different batch sizes based on learned patterns

### Key Concepts

1. **Locators**: Navigate through nested structures (tuples, lists, dicts, dataclasses and registered user types) to find tensors
2. **TensorSpec**: Underlying representation that tracks shape, dtype, and batch axis information
3. **Dynamic Dimensions**: Dimensions that vary across samples (e.g., sequence length in NLP)
4. **Batch Axes**: Dimensions that scale proportionally with batch size

Let's start with a simple example:

```python
from dataclasses import dataclass
import torch
from aitune.torch.module.sample_metadata import SampleMetadata, InfoLevel
# Create a simple tensor and capture its metadata
simple_tensor = torch.randn(2, 3, 4)
args = (simple_tensor,)
kwargs = {}

metadata = SampleMetadata.from_inputs(args, kwargs)
print(repr(metadata))
```

```text
Tensors:
╒═══════════╤════════╤═══════════╤═════════════╤═════════════╤═══════════════╕
│ Locator   │ Name   │ Shape     │ Min Shape   │ Max Shape   │ Dtype         │
╞═══════════╪════════╪═══════════╪═════════════╪═════════════╪═══════════════╡
│ [0]       │ args_0 │ [2, 3, 4] │ [2, 3, 4]   │ [2, 3, 4]   │ torch.float32 │
╘═══════════╧════════╧═══════════╧═════════════╧═════════════╧═══════════════╛
```

The output shows that `SampleMetadata` automatically detected the tensor, assigned it a name (`args_0`), and captured its shape `[2, 3, 4]` along with data type information.

## Creating Metadata from Inputs

The primary way to create `SampleMetadata` is through the `from_inputs()` static method. This method accepts:

- `args`: Positional arguments (typically a tuple)
- `kwargs`: Keyword arguments (a dictionary)
- `strict`: Boolean flag controlling whether to track non-tensor data (default: `False`)

Let's explore different input patterns:

```python
# Example 1: Multiple tensors in args
args = (
    torch.randn(2, 3),
    torch.randn(4, 5),
)
kwargs = {}

meta1 = SampleMetadata.from_inputs(args, kwargs)
print("Example 1 - Multiple args:")
print(repr(meta1))

```

```text
Example 1 - Multiple args:
Tensors:
╒═══════════╤════════╤═════════╤═════════════╤═════════════╤═══════════════╕
│ Locator   │ Name   │ Shape   │ Min Shape   │ Max Shape   │ Dtype         │
╞═══════════╪════════╪═════════╪═════════════╪═════════════╪═══════════════╡
│ [0]       │ args_0 │ [2, 3]  │ [2, 3]      │ [2, 3]      │ torch.float32 │
├───────────┼────────┼─────────┼─────────────┼─────────────┼───────────────┤
│ [1]       │ args_1 │ [4, 5]  │ [4, 5]      │ [4, 5]      │ torch.float32 │
╘═══════════╧════════╧═════════╧═════════════╧═════════════╧═══════════════╛
```

```python
# Example 2: Tensors in kwargs
args = ()
kwargs = {
    "input_tensor": torch.randn(3, 4),
    "mask": torch.randn(3, 1),
}

meta2 = SampleMetadata.from_inputs(args, kwargs)
print("Example 2 - Kwargs only:")
print(repr(meta2))
```

```text
Example 2 - Kwargs only:
Tensors:
╒══════════════════╤═════════════════════╤═════════╤═════════════╤═════════════╤═══════════════╕
│ Locator          │ Name                │ Shape   │ Min Shape   │ Max Shape   │ Dtype         │
╞══════════════════╪═════════════════════╪═════════╪═════════════╪═════════════╪═══════════════╡
│ ['input_tensor'] │ kwargs_input_tensor │ [3, 4]  │ [3, 4]      │ [3, 4]      │ torch.float32 │
├──────────────────┼─────────────────────┼─────────┼─────────────┼─────────────┼───────────────┤
│ ['mask']         │ kwargs_mask         │ [3, 1]  │ [3, 1]      │ [3, 1]      │ torch.float32 │
╘══════════════════╧═════════════════════╧═════════╧═════════════╧═════════════╧═══════════════╛
```

```python
# Example 3: Mixed primitives and tensors
args = (
    "some_string",           # Primitive - ignored by default
    torch.randn(2, 2),       # Tensor - tracked
    42,                      # Primitive - ignored by default
)
kwargs = {
    "data": torch.randn(3, 3),
    "learning_rate": 0.001,  # Primitive - ignored by default
}

meta3 = SampleMetadata.from_inputs(args, kwargs)
print("Example 3 - Mixed types (strict=False):")
print(repr(meta3))
print("\nNotice that only tensors are tracked!")

```

```text
Example 3 - Mixed types (strict=False):
Tensors:
╒═══════════╤═════════════╤═════════╤═════════════╤═════════════╤═══════════════╕
│ Locator   │ Name        │ Shape   │ Min Shape   │ Max Shape   │ Dtype         │
╞═══════════╪═════════════╪═════════╪═════════════╪═════════════╪═══════════════╡
│ [1]       │ args_1      │ [2, 2]  │ [2, 2]      │ [2, 2]      │ torch.float32 │
├───────────┼─────────────┼─────────┼─────────────┼─────────────┼───────────────┤
│ ['data']  │ kwargs_data │ [3, 3]  │ [3, 3]      │ [3, 3]      │ torch.float32 │
╘═══════════╧═════════════╧═════════╧═════════════╧═════════════╧═══════════════╛
```

Notice that only tensors are tracked!

## Strict vs. Non-Strict Mode

By default during tuning, AITune operates in **strict mode** (`config.strict_mode=True`), which means `SampleMetadata` captures both tensors and non-tensor data (primitives, strings, etc.). This ensures different argument values create different execution graphs.

However, when calling `SampleMetadata.from_inputs()` directly with `strict=False`, it only tracks tensors and ignores all other data types. This is useful when you only care about tensor shapes for optimization purposes.

**Strict mode** (`strict=True`) is useful for:

- Validating that function signatures match expected patterns
- Debugging data flow through complex pipelines
- Ensuring reproducibility of function calls

Let's compare the two modes:

```python
# Same inputs, different modes
args = (1, 2, 3, torch.randn(2, 2))
kwargs = {"t": torch.randn(2, 3), "other": "abc"}

# Non-strict mode (default)
meta_non_strict = SampleMetadata.from_inputs(args, kwargs, strict=False)
print("Non-Strict Mode (strict=False):")
print(repr(meta_non_strict))
print("\n" + "="*80 + "\n")

# Strict mode
meta_strict = SampleMetadata.from_inputs(args, kwargs, strict=True)
print("Strict Mode (strict=True):")
print(repr(meta_strict))

```

```text
Non-Strict Mode (strict=False):
Tensors:
╒═══════════╤══════════╤═════════╤═════════════╤═════════════╤═══════════════╕
│ Locator   │ Name     │ Shape   │ Min Shape   │ Max Shape   │ Dtype         │
╞═══════════╪══════════╪═════════╪═════════════╪═════════════╪═══════════════╡
│ [3]       │ args_3   │ [2, 2]  │ [2, 2]      │ [2, 2]      │ torch.float32 │
├───────────┼──────────┼─────────┼─────────────┼─────────────┼───────────────┤
│ ['t']     │ kwargs_t │ [2, 3]  │ [2, 3]      │ [2, 3]      │ torch.float32 │
╘═══════════╧══════════╧═════════╧═════════════╧═════════════╧═══════════════╛

================================================================================

Strict Mode (strict=True):
Tensors:
╒═══════════╤══════════╤═════════╤═════════════╤═════════════╤═══════════════╕
│ Locator   │ Name     │ Shape   │ Min Shape   │ Max Shape   │ Dtype         │
╞═══════════╪══════════╪═════════╪═════════════╪═════════════╪═══════════════╡
│ [3]       │ args_3   │ [2, 2]  │ [2, 2]      │ [2, 2]      │ torch.float32 │
├───────────┼──────────┼─────────┼─────────────┼─────────────┼───────────────┤
│ ['t']     │ kwargs_t │ [2, 3]  │ [2, 3]      │ [2, 3]      │ torch.float32 │
╘═══════════╧══════════╧═════════╧═════════════╧═════════════╧═══════════════╛
Other:
╒═══════════╤══════════════╤═════════╕
│ Locator   │ Name         │ Value   │
╞═══════════╪══════════════╪═════════╡
│ [0]       │ args_0       │ 1       │
├───────────┼──────────────┼─────────┤
│ [1]       │ args_1       │ 2       │
├───────────┼──────────────┼─────────┤
│ [2]       │ args_2       │ 3       │
├───────────┼──────────────┼─────────┤
│ ['other'] │ kwargs_other │ abc     │
╘═══════════╧══════════════╧═════════╛
```

Notice that in strict mode, we see an additional "Other" section that includes the primitive values (1, 2, 3, and "abc").

## Working with Nested Structures

One of the most powerful features of `SampleMetadata` is its ability to handle deeply nested data structures. Real-world model inputs often involve complex combinations of:

- **Tuples and Lists**: For variable-length sequences
- **Dictionaries**: For named parameters
- **Dataclasses**: For structured configuration objects

`SampleMetadata` uses **Locators** to navigate these structures and find all tensors, no matter how deeply nested they are.

Let's create a complex nested example:

```python
# Define a custom dataclass
@dataclass
class ModelInput:
    data: torch.Tensor
    metadata: str

# Create complex nested structure
args = [
    "first_arg",
    torch.randn(1),                                      # Simple tensor
    (torch.randn(2), torch.randn(3)),                    # Tuple of tensors
    {"t": torch.randn(4)},                               # Dict with tensor
    ModelInput(data=torch.randn(5), metadata="info"),    # Dataclass with tensor
]

kwargs = {
    "t1": torch.randn(1, 1),
    "t2": [torch.randn(2, 2), torch.randn(3, 3)],        # List of tensors
    "t3": ModelInput(data=torch.randn(4, 4), metadata="xyz"),
    "last": "other",
}

nested_meta = SampleMetadata.from_inputs(args, kwargs, strict=True)
print(repr(nested_meta))

```

```text
Tensors:
╒═════════════╤════════════════╤═════════╤═════════════╤═════════════╤═══════════════╕
│ Locator     │ Name           │ Shape   │ Min Shape   │ Max Shape   │ Dtype         │
╞═════════════╪════════════════╪═════════╪═════════════╪═════════════╪═══════════════╡
│ [1]         │ args_1         │ [1]     │ [1]         │ [1]         │ torch.float32 │
├─────────────┼────────────────┼─────────┼─────────────┼─────────────┼───────────────┤
│ [2][0]      │ args_2_0       │ [2]     │ [2]         │ [2]         │ torch.float32 │
├─────────────┼────────────────┼─────────┼─────────────┼─────────────┼───────────────┤
│ [2][1]      │ args_2_1       │ [3]     │ [3]         │ [3]         │ torch.float32 │
├─────────────┼────────────────┼─────────┼─────────────┼─────────────┼───────────────┤
│ [3]['t']    │ args_3_t       │ [4]     │ [4]         │ [4]         │ torch.float32 │
├─────────────┼────────────────┼─────────┼─────────────┼─────────────┼───────────────┤
│ [4].data    │ args_4.data    │ [5]     │ [5]         │ [5]         │ torch.float32 │
├─────────────┼────────────────┼─────────┼─────────────┼─────────────┼───────────────┤
│ ['t1']      │ kwargs_t1      │ [1, 1]  │ [1, 1]      │ [1, 1]      │ torch.float32 │
├─────────────┼────────────────┼─────────┼─────────────┼─────────────┼───────────────┤
│ ['t2'][0]   │ kwargs_t2_0    │ [2, 2]  │ [2, 2]      │ [2, 2]      │ torch.float32 │
├─────────────┼────────────────┼─────────┼─────────────┼─────────────┼───────────────┤
│ ['t2'][1]   │ kwargs_t2_1    │ [3, 3]  │ [3, 3]      │ [3, 3]      │ torch.float32 │
├─────────────┼────────────────┼─────────┼─────────────┼─────────────┼───────────────┤
│ ['t3'].data │ kwargs_t3.data │ [4, 4]  │ [4, 4]      │ [4, 4]      │ torch.float32 │
╘═════════════╧════════════════╧═════════╧═════════════╧═════════════╧═══════════════╛
Other:
╒═════════════════╤════════════════════╤═══════════╕
│ Locator         │ Name               │ Value     │
╞═════════════════╪════════════════════╪═══════════╡
│ [0]             │ args_0             │ first_arg │
├─────────────────┼────────────────────┼───────────┤
│ [4].metadata    │ args_4.metadata    │ info      │
├─────────────────┼────────────────────┼───────────┤
│ ['last']        │ kwargs_last        │ other     │
├─────────────────┼────────────────────┼───────────┤
│ ['t3'].metadata │ kwargs_t3.metadata │ xyz       │
╘═════════════════╧════════════════════╧═══════════╛
```

### Understanding Locators

The **Locator** column shows the path to each tensor in the nested structure:

- `[1]`: Second element of args (0-indexed)
- `[2][0]`: First element of the tuple at args[2]
- `[3]['t']`: Value at key 't' in the dict at args[3]
- `[4].data`: The 'data' attribute of the dataclass at args[4]
- `['t2'][0]`: First element of the list at kwargs['t2']
- `['t3'].data`: The 'data' attribute of the dataclass at kwargs['t3']

This allows `SampleMetadata` to precisely locate and manipulate tensors in complex structures.

## Describing Metadata - InfoLevel

`SampleMetadata` provides three levels of detail when displaying information, controlled by the `InfoLevel` enum:

1. **`InfoLevel.SHORT`**: Compact representation with just tensor names
2. **`InfoLevel.MEDIUM`**: Includes locators and current shapes (simple table format)
3. **`InfoLevel.FULL`**: Complete details including min/max shapes and dtypes (fancy table format)

Let's see the same metadata displayed at all three levels:

```python
# Create sample metadata
args = (torch.randn(2, 3), torch.randn(4, 5, 6))
kwargs = {"mask": torch.randn(2, 1)}
meta = SampleMetadata.from_inputs(args, kwargs)

print("InfoLevel.SHORT:")
print(meta.describe(InfoLevel.SHORT))
print("\n" + "="*80 + "\n")

print("InfoLevel.MEDIUM:")
print(meta.describe(InfoLevel.MEDIUM))
print("\n" + "="*80 + "\n")

print("InfoLevel.FULL:")
print(meta.describe(InfoLevel.FULL))

```

```text
InfoLevel.SHORT:
Tensors: args_0, args_1, kwargs_mask

================================================================================

InfoLevel.MEDIUM:
Tensors:
Locator    Name         Shape
---------  -----------  ---------
[0]        args_0       [2, 3]
[1]        args_1       [4, 5, 6]
['mask']   kwargs_mask  [2, 1]

================================================================================

InfoLevel.FULL:
Tensors:
╒═══════════╤═════════════╤═══════════╤═════════════╤═════════════╤═══════════════╕
│ Locator   │ Name        │ Shape     │ Min Shape   │ Max Shape   │ Dtype         │
╞═══════════╪═════════════╪═══════════╪═════════════╪═════════════╪═══════════════╡
│ [0]       │ args_0      │ [2, 3]    │ [2, 3]      │ [2, 3]      │ torch.float32 │
├───────────┼─────────────┼───────────┼─────────────┼─────────────┼───────────────┤
│ [1]       │ args_1      │ [4, 5, 6] │ [4, 5, 6]   │ [4, 5, 6]   │ torch.float32 │
├───────────┼─────────────┼───────────┼─────────────┼─────────────┼───────────────┤
│ ['mask']  │ kwargs_mask │ [2, 1]    │ [2, 1]      │ [2, 1]      │ torch.float32 │
╘═══════════╧═════════════╧═══════════╧═════════════╧═════════════╧═══════════════╛
```

The `FULL` level is particularly useful because it shows:

- **Min Shape**: The smallest dimensions seen for each axis
- **Max Shape**: The largest dimensions seen for each axis
- **Dtype**: The PyTorch data type of the tensor

These become interesting when we start tracking multiple samples with different shapes.

## Dynamic Shape Tracking

One of the most sophisticated features of `SampleMetadata` is its ability to learn about **dynamic dimensions** and **batch axes** by observing multiple samples with different shapes.

### How It Works

When you call `update_shapes_seen()` with metadata from a different sample:

1. **Batch Axis Detection**: If a dimension scales proportionally with batch size and the multiplier is an integer, it's marked as a batch axis (e.g., `batch0`, `batch1`)
2. **Dynamic Dimension Detection**: If a dimension changes but not proportionally to batch size, it's marked as a dynamic dimension (e.g., `dim0`, `dim1`)
3. **Min/Max Tracking**: The minimum and maximum values seen for each dimension are tracked

Let's see this in action:

```python
# Create initial metadata with batch size 1
args_initial = [
    torch.randn(1),
    torch.randn(2),
    torch.randn(5),
]
kwargs_initial = {
    "data": torch.randn(1, 10),  # First dim batch, second dynamic
}

meta_initial = SampleMetadata.from_inputs(args_initial, kwargs_initial, strict=False, batch_size=1)
print("Initial Metadata (batch_size=1):")
print(meta_initial.describe(InfoLevel.FULL))

```

```text
Initial Metadata (batch_size=1):
Tensors:
╒═══════════╤═════════════╤═════════╤═════════════╤═════════════╤═══════════════╕
│ Locator   │ Name        │ Shape   │ Min Shape   │ Max Shape   │ Dtype         │
╞═══════════╪═════════════╪═════════╪═════════════╪═════════════╪═══════════════╡
│ [0]       │ args_0      │ [1]     │ [1]         │ [1]         │ torch.float32 │
├───────────┼─────────────┼─────────┼─────────────┼─────────────┼───────────────┤
│ [1]       │ args_1      │ [2]     │ [2]         │ [2]         │ torch.float32 │
├───────────┼─────────────┼─────────┼─────────────┼─────────────┼───────────────┤
│ [2]       │ args_2      │ [5]     │ [5]         │ [5]         │ torch.float32 │
├───────────┼─────────────┼─────────┼─────────────┼─────────────┼───────────────┤
│ ['data']  │ kwargs_data │ [1, 10] │ [1, 10]     │ [1, 10]     │ torch.float32 │
╘═══════════╧═════════════╧═════════╧═════════════╧═════════════╧═══════════════╛
```

```python
# Create second metadata with different shapes and batch size 2
args_second = [
    torch.randn(2),      # Doubled (batch axis)
    torch.randn(5),      # Changed but not proportionally (dynamic)
    torch.randn(15),     # Changed but not proportionally (dynamic)
]
kwargs_second = {
    "data": torch.randn(2, 25),  # First dim doubled, second changed
}

meta_second = SampleMetadata.from_inputs(args_second, kwargs_second, strict=False, batch_size=2)
print("Second Metadata (batch_size=2):")
print(meta_second.describe(InfoLevel.FULL))

```

```text
Second Metadata (batch_size=2):
Tensors:
╒═══════════╤═════════════╤═════════╤═════════════╤═════════════╤═══════════════╕
│ Locator   │ Name        │ Shape   │ Min Shape   │ Max Shape   │ Dtype         │
╞═══════════╪═════════════╪═════════╪═════════════╪═════════════╪═══════════════╡
│ [0]       │ args_0      │ [2]     │ [2]         │ [2]         │ torch.float32 │
├───────────┼─────────────┼─────────┼─────────────┼─────────────┼───────────────┤
│ [1]       │ args_1      │ [5]     │ [5]         │ [5]         │ torch.float32 │
├───────────┼─────────────┼─────────┼─────────────┼─────────────┼───────────────┤
│ [2]       │ args_2      │ [15]    │ [15]        │ [15]        │ torch.float32 │
├───────────┼─────────────┼─────────┼─────────────┼─────────────┼───────────────┤
│ ['data']  │ kwargs_data │ [2, 25] │ [2, 25]     │ [2, 25]     │ torch.float32 │
╘═══════════╧═════════════╧═════════╧═════════════╧═════════════╧═══════════════╛
```

```python
# Update the initial metadata with information from the second sample
meta_initial.update_shapes_seen(meta_second)
print("Updated Metadata (after seeing both samples):")
print(meta_initial.describe(InfoLevel.FULL))

```

```text
Updated Metadata (after seeing both samples):
Tensors:
╒═══════════╤═════════════╤════════════════════╤═════════════╤═════════════╤═══════════════╕
│ Locator   │ Name        │ Shape              │ Min Shape   │ Max Shape   │ Dtype         │
╞═══════════╪═════════════╪════════════════════╪═════════════╪═════════════╪═══════════════╡
│ [0]       │ args_0      │ ['batch0']         │ [1]         │ [2]         │ torch.float32 │
├───────────┼─────────────┼────────────────────┼─────────────┼─────────────┼───────────────┤
│ [1]       │ args_1      │ ['dim0']           │ [2]         │ [5]         │ torch.float32 │
├───────────┼─────────────┼────────────────────┼─────────────┼─────────────┼───────────────┤
│ [2]       │ args_2      │ ['dim0']           │ [5]         │ [15]        │ torch.float32 │
├───────────┼─────────────┼────────────────────┼─────────────┼─────────────┼───────────────┤
│ ['data']  │ kwargs_data │ ['batch0', 'dim1'] │ [1, 10]     │ [2, 25]     │ torch.float32 │
╘═══════════╧═════════════╧════════════════════╧═════════════╧═════════════╧═══════════════╛
```

### Understanding the Results

After updating, notice how the shapes have been transformed:

- **`batch0`**: Dimensions that doubled when batch size doubled (1→2)
- **`dim0`, `dim1`**: Dimensions that changed but not proportionally to batch size
- **Min/Max Shape**: Now show the range of values observed

This information is crucial for:

- **Model compilation**: Backends can create optimized graphs for dynamic shapes
- **Memory planning**: Knowing the range helps allocate appropriate buffers
- **Validation**: Ensuring new inputs fall within expected ranges

## Batch Manipulation

Once `SampleMetadata` has learned about batch axes through `update_shapes_seen()`, it can use the `make_batch()` method to scale tensors to a target batch size.

### How `make_batch()` Works

The method uses **batch axis multipliers** to determine how to scale each dimension:

1. **Multiplier = 1**: Standard batch axis, scales linearly with batch size
2. **Multiplier > 1**: Stacked batch axis (e.g., when inputs are vertically stacked)
3. **Slicing**: If current size > target, slice the tensor
4. **Repeating**: If current size < target, repeat the tensor

Let's see this in action:

```python
# First, create metadata and teach it about batch axes
args1 = [torch.randn(1, 5), torch.randn(2, 3)]
kwargs1 = {"mask": torch.randn(1, 10)}

meta = SampleMetadata.from_inputs(args1, kwargs1, batch_size=1)

# Second sample with batch size 2
args2 = [torch.randn(2, 5), torch.randn(4, 3)]
kwargs2 = {"mask": torch.randn(2, 10)}

meta2 = SampleMetadata.from_inputs(args2, kwargs2, batch_size=2)
meta.update_shapes_seen(meta2)

print("Learned Metadata:")
print(meta.describe(InfoLevel.FULL))

```

```text
Learned Metadata:
Tensors:
╒═══════════╤═════════════╤════════════════╤═════════════╤═════════════╤═══════════════╕
│ Locator   │ Name        │ Shape          │ Min Shape   │ Max Shape   │ Dtype         │
╞═══════════╪═════════════╪════════════════╪═════════════╪═════════════╪═══════════════╡
│ [0]       │ args_0      │ ['batch0', 5]  │ [1, 5]      │ [2, 5]      │ torch.float32 │
├───────────┼─────────────┼────────────────┼─────────────┼─────────────┼───────────────┤
│ [1]       │ args_1      │ ['batch0', 3]  │ [2, 3]      │ [4, 3]      │ torch.float32 │
├───────────┼─────────────┼────────────────┼─────────────┼─────────────┼───────────────┤
│ ['mask']  │ kwargs_mask │ ['batch0', 10] │ [1, 10]     │ [2, 10]     │ torch.float32 │
╘═══════════╧═════════════╧════════════════╧═════════════╧═════════════╧═══════════════╛
```

```python
# Now use make_batch to scale to a larger batch size
original_args = [torch.randn(1, 5), torch.randn(2, 3)]
original_kwargs = {"mask": torch.randn(1, 10)}

print("Original shapes:")
print(f"  args[0]: {original_args[0].shape}")
print(f"  args[1]: {original_args[1].shape}")
print(f"  kwargs['mask']: {original_kwargs['mask'].shape}")
print()

# Scale to batch size 10
batched_args, batched_kwargs = meta.make_batch(original_args, original_kwargs, batch_size=10)

print("After make_batch(batch_size=10):")
print(f"  args[0]: {batched_args[0].shape}")
print(f"  args[1]: {batched_args[1].shape}")
print(f"  kwargs['mask']: {batched_kwargs['mask'].shape}")

```

```text
Original shapes:
  args[0]: torch.Size([1, 5])
  args[1]: torch.Size([2, 3])
  kwargs['mask']: torch.Size([1, 10])

After make_batch(batch_size=10):
  args[0]: torch.Size([10, 5])
  args[1]: torch.Size([20, 3])
  kwargs['mask']: torch.Size([10, 10])
```

Notice how:

- The first dimensions (batch axes) scaled to match the target batch size of 10
- `args[1]` has a multiplier of 2 (it's a stacked batch), so it scaled to 20 (10 × 2)
- Non-batch dimensions (like the 5, 3, 10) remained unchanged

## TensorSpec Deep Dive

`SampleMetadata` is actually a container for multiple `TensorSpec` objects, where each `TensorSpec` represents one tensor in the input/output structure.

### TensorSpec Attributes

- **`name`**: Symbolic name (e.g., `args_0`, `kwargs_mask`)
- **`shape`**: Current shape representation (may include symbolic dimensions)
- **`min_shape`**: Minimum dimensions observed
- **`max_shape`**: Maximum dimensions observed
- **`dtype`**: PyTorch data type
- **`_bs_multipliers`**: Internal batch size multipliers for each axis

Let's inspect TensorSpec objects directly:

```python
# Create metadata with dynamic shapes
args1 = [torch.randn(1, 5)]
kwargs1 = {"data": torch.randn(1, 10)}
meta = SampleMetadata.from_inputs(args1, kwargs1, batch_size=1)

args2 = [torch.randn(2, 5)]
kwargs2 = {"data": torch.randn(2, 20)}
meta2 = SampleMetadata.from_inputs(args2, kwargs2, batch_size=2)
meta.update_shapes_seen(meta2)

# Access individual TensorSpec objects
print("Individual TensorSpec objects:\n")
for locator, tensor_spec in meta.tensor_data:
    print(f"Locator: {locator}")
    print(f"  Name: {tensor_spec.name}")
    print(f"  Shape: {tensor_spec.shape}")
    print(f"  Min Shape: {tensor_spec.min_shape}")
    print(f"  Max Shape: {tensor_spec.max_shape}")
    print(f"  Dtype: {tensor_spec.dtype}")
    print(f"  Has batch axis: {tensor_spec.has_batch_axis()}")
    print(f"  Has dynamic axis: {tensor_spec.has_dynamic_axis()}")
    print(f"  Batch multipliers: {tensor_spec.get_batch_axis_multipliers()}")
```

```text
Individual TensorSpec objects:

Locator: [0]
  Name: args_0
  Shape: ['batch0', 5]
  Min Shape: [1, 5]
  Max Shape: [2, 5]
  Dtype: torch.float32
  Has batch axis: True
  Has dynamic axis: False
  Batch multipliers: {0: 1}
Locator: ['data']
  Name: kwargs_data
  Shape: ['batch0', 'batch1']
  Min Shape: [1, 10]
  Max Shape: [2, 20]
  Dtype: torch.float32
  Has batch axis: True
  Has dynamic axis: False
  Batch multipliers: {0: 1, 1: 10}
```

### Useful TensorSpec Methods

- **`has_batch_axis()`**: Returns True if the tensor has at least one batch dimension
- **`has_dynamic_axis()`**: Returns True if the tensor has at least one dynamic dimension
- **`get_batch_axis_multipliers()`**: Returns a dict mapping axis index to its batch multiplier
- **`matches(other)`**: Checks if two TensorSpecs are compatible

These methods are used internally by `SampleMetadata` to perform operations like `make_batch()`.

## Practical Example - Complete Workflow

Let's put everything together with a realistic scenario: profiling a model with variable-length sequences (like in NLP tasks).

### Scenario

We have a language model that takes:

- Input IDs with shape `(batch_size, sequence_length)`
- Attention mask with shape `(batch_size, sequence_length)`
- Position IDs with shape `(batch_size, sequence_length)`

We'll profile it with different batch sizes and sequence lengths to learn the dynamic shapes.

```python
# Simulate model profiling
@dataclass
class ModelInputs:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor

# Sample 1: batch_size=1, seq_len=10
sample1_args = ()
sample1_kwargs = {
    "inputs": ModelInputs(
        input_ids=torch.randint(0, 1000, (1, 10)),
        attention_mask=torch.ones(1, 10),
        position_ids=torch.arange(10).unsqueeze(0),
    )
}

metadata = SampleMetadata.from_inputs(sample1_args, sample1_kwargs, batch_size=1, strict=False)
print("After Sample 1 (batch=1, seq_len=10):")
print(metadata.describe(InfoLevel.FULL))
```

```text
After Sample 1 (batch=1, seq_len=10):
Tensors:
╒═══════════════════════════╤══════════════════════════════╤═════════╤═════════════╤═════════════╤═══════════════╕
│ Locator                   │ Name                         │ Shape   │ Min Shape   │ Max Shape   │ Dtype         │
╞═══════════════════════════╪══════════════════════════════╪═════════╪═════════════╪═════════════╪═══════════════╡
│ ['inputs'].input_ids      │ kwargs_inputs.input_ids      │ [1, 10] │ [1, 10]     │ [1, 10]     │ torch.int64   │
├───────────────────────────┼──────────────────────────────┼─────────┼─────────────┼─────────────┼───────────────┤
│ ['inputs'].attention_mask │ kwargs_inputs.attention_mask │ [1, 10] │ [1, 10]     │ [1, 10]     │ torch.float32 │
├───────────────────────────┼──────────────────────────────┼─────────┼─────────────┼─────────────┼───────────────┤
│ ['inputs'].position_ids   │ kwargs_inputs.position_ids   │ [1, 10] │ [1, 10]     │ [1, 10]     │ torch.int64   │
╘═══════════════════════════╧══════════════════════════════╧═════════╧═════════════╧═════════════╧═══════════════╛
```

```python
# Sample 2: batch_size=2, seq_len=15
sample2_args = ()
sample2_kwargs = {
    "inputs": ModelInputs(
        input_ids=torch.randint(0, 1000, (2, 15)),
        attention_mask=torch.ones(2, 15),
        position_ids=torch.arange(15).unsqueeze(0).repeat(2, 1),
    )
}

metadata2 = SampleMetadata.from_inputs(sample2_args, sample2_kwargs, batch_size=2, strict=False)
metadata.update_shapes_seen(metadata2)

print("After Sample 2 (batch=2, seq_len=15):")
print(metadata.describe(InfoLevel.FULL))

```

```text
After Sample 2 (batch=2, seq_len=15):
Tensors:
╒═══════════════════════════╤══════════════════════════════╤════════════════════╤═════════════╤═════════════╤═══════════════╕
│ Locator                   │ Name                         │ Shape              │ Min Shape   │ Max Shape   │ Dtype         │
╞═══════════════════════════╪══════════════════════════════╪════════════════════╪═════════════╪═════════════╪═══════════════╡
│ ['inputs'].input_ids      │ kwargs_inputs.input_ids      │ ['batch0', 'dim1'] │ [1, 10]     │ [2, 15]     │ torch.int64   │
├───────────────────────────┼──────────────────────────────┼────────────────────┼─────────────┼─────────────┼───────────────┤
│ ['inputs'].attention_mask │ kwargs_inputs.attention_mask │ ['batch0', 'dim1'] │ [1, 10]     │ [2, 15]     │ torch.float32 │
├───────────────────────────┼──────────────────────────────┼────────────────────┼─────────────┼─────────────┼───────────────┤
│ ['inputs'].position_ids   │ kwargs_inputs.position_ids   │ ['batch0', 'dim1'] │ [1, 10]     │ [2, 15]     │ torch.int64   │
╘═══════════════════════════╧══════════════════════════════╧════════════════════╧═════════════╧═════════════╧═══════════════╛
```

```python
# Sample 3: batch_size=4, seq_len=20
sample3_args = ()
sample3_kwargs = {
    "inputs": ModelInputs(
        input_ids=torch.randint(0, 1000, (4, 20)),
        attention_mask=torch.ones(4, 20),
        position_ids=torch.arange(20).unsqueeze(0).repeat(4, 1),
    )
}

metadata3 = SampleMetadata.from_inputs(sample3_args, sample3_kwargs, batch_size=4, strict=False)
metadata.update_shapes_seen(metadata3)

print("After Sample 3 (batch=4, seq_len=20):")
print(metadata.describe(InfoLevel.FULL))

```

```text
After Sample 3 (batch=4, seq_len=20):
Tensors:
╒═══════════════════════════╤══════════════════════════════╤════════════════════╤═════════════╤═════════════╤═══════════════╕
│ Locator                   │ Name                         │ Shape              │ Min Shape   │ Max Shape   │ Dtype         │
╞═══════════════════════════╪══════════════════════════════╪════════════════════╪═════════════╪═════════════╪═══════════════╡
│ ['inputs'].input_ids      │ kwargs_inputs.input_ids      │ ['batch0', 'dim1'] │ [1, 10]     │ [4, 20]     │ torch.int64   │
├───────────────────────────┼──────────────────────────────┼────────────────────┼─────────────┼─────────────┼───────────────┤
│ ['inputs'].attention_mask │ kwargs_inputs.attention_mask │ ['batch0', 'dim1'] │ [1, 10]     │ [4, 20]     │ torch.float32 │
├───────────────────────────┼──────────────────────────────┼────────────────────┼─────────────┼─────────────┼───────────────┤
│ ['inputs'].position_ids   │ kwargs_inputs.position_ids   │ ['batch0', 'dim1'] │ [1, 10]     │ [4, 20]     │ torch.int64   │
╘═══════════════════════════╧══════════════════════════════╧════════════════════╧═════════════╧═════════════╧═══════════════╛
```

### Analysis

After observing three samples with different batch sizes and sequence lengths:

- **First dimension**: Identified as `batch0` because it scaled proportionally (1→2→4)
- **Second dimension**: Identified as `dim1` because it varied dynamically (10→15→20)
- **Min/Max ranges**: Captured the observed ranges for both dimensions

This information can now be used by optimization backends to compile efficient code for these dynamic shapes.

```python
# Now we can create inputs for any batch size!
test_input_args = ()
test_input_kwargs = {
    "inputs": ModelInputs(
        input_ids=torch.randint(0, 1000, (2, 12)),
        attention_mask=torch.ones(2, 12),
        position_ids=torch.arange(12).unsqueeze(0).repeat(2, 1),
    )
}

print("Original test input shapes:")
print(f"  input_ids: {test_input_kwargs['inputs'].input_ids.shape}")
print(f"  attention_mask: {test_input_kwargs['inputs'].attention_mask.shape}")
print(f"  position_ids: {test_input_kwargs['inputs'].position_ids.shape}")
print()

# Scale to batch size 8
scaled_args, scaled_kwargs = metadata.make_batch(test_input_args, test_input_kwargs, batch_size=8)

print("After scaling to batch_size=8:")
print(f"  input_ids: {scaled_kwargs['inputs'].input_ids.shape}")
print(f"  attention_mask: {scaled_kwargs['inputs'].attention_mask.shape}")
print(f"  position_ids: {scaled_kwargs['inputs'].position_ids.shape}")
print("\nNote: Batch dimension scaled to 8, but sequence length (dim1) remained at 12")

```

```text
Original test input shapes:
  input_ids: torch.Size([2, 12])
  attention_mask: torch.Size([2, 12])
  position_ids: torch.Size([2, 12])

After scaling to batch_size=8:
  input_ids: torch.Size([8, 12])
  attention_mask: torch.Size([8, 12])
  position_ids: torch.Size([8, 12])

Note: Batch dimension scaled to 8, but sequence length (dim1) remained at 12
```

## Summary

### Key Takeaways

1. **Purpose**: `SampleMetadata` captures and tracks metadata about tensors in complex data structures, enabling model optimization and dynamic batching.

2. **Creation**: Use `SampleMetadata.from_inputs(args, kwargs, strict=bool)` to create metadata from function inputs.

3. **Strict Mode**: Controls whether only tensors (`strict=False`) or all data types (`strict=True`) are tracked.

4. **Nested Structures**: Automatically handles tuples, lists, dicts, and dataclasses using Locators.

5. **InfoLevel**: Three display modes (SHORT, MEDIUM, FULL) provide different levels of detail.

6. **Dynamic Shape Learning**: `update_shapes_seen()` learns about batch axes and dynamic dimensions by observing multiple samples.

7. **Batch Manipulation**: `make_batch()` can scale tensors to any batch size based on learned batch axis multipliers.

8. **TensorSpec**: The underlying representation of each tensor, containing shape, dtype, and batch information.

### Use in AI-Tune Pipeline

`SampleMetadata` is a fundamental building block in the AITune library, used by:

- **RecordingModule**: Captures input/output metadata during profiling
- **Backends**: Use metadata to configure optimized execution (TensorRT, TorchScript, etc.)
- **Graph Compilation**: Enables the creation of optimized graphs for dynamic shapes

### Source Code

For more details, see:

- `aitune/torch/module/sample_metadata.py`
- `aitune/torch/module/tensor_spec.py`
- `aitune/torch/module/locator.py`

If you would like to tinker with `SampleMetadata`, you can find this example in `notebooks/sample_metadata_walkthrough.ipynb`.
