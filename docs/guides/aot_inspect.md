---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Ahead-of-time Inspect Guide"
---

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
    call_tree=False,                # Also record the call tree (modules_info.call_tree); recommended: True
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

If the custom function uses scalar arguments that should also be tuned, use the same function later in
`ait.tune()` after wrapping the inspected modules. See
[Inspect and tune with the same workload wrapper](aot_tuning.md#inspect-and-tune-with-the-same-workload-wrapper).

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

- **min_execution_ratio** - minimum ratio of total execution time e.g. .9
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

## Call tree

`get_modules()` sums the calls of each module, so it does not say where a module was called from or which other
calls ran inside it. With `call_tree=True`, `inspect` also records a call tree: every recorded module `forward` and
model-defined method, keyed by the chain of calls that led to it, with its calls and time.

```python
modules_info = ait.inspect(model, input_data, inference_function=run_chunk, min_depth=1, max_depth=8, call_tree=True)
tree = modules_info.call_tree

for node in tree.find("dit"):    # recorded calls whose label contains "dit"
    print(node.path_str, node.calls, node.time_ms, node.bytes_in, node.bytes_out)
# forward > dit_forward 10 ...
# forward > dit_forward > dit.forward 10 ...
```

A node's **path** is its chain of callers, outermost first; `path_str` joins it with ` > `. Each part is an object
path from the inspected object plus a method name: `forward > dit_forward` is `model.dit_forward` called from
`model.forward`. The same module called from two places gives two nodes, for example an LLM used for prefill and for
generation.
`tree.nodes_of(model.dit)` returns the nodes of a module's `forward`, one per place it was called from, and
`tree.nodes_of(model, "dit_forward")` those of another method.

What is recorded:

- `forward` of every module the object walk reaches, and the model's own methods (`dit_forward`, `_forward_flow`),
  on modules and on plain pipeline objects outside any module. Methods of library classes (`torch`,
  `transformers`, `diffusers`, ...) are not recorded, but methods of your own subclasses of them are;
- calls of `.to()`, `.cuda()` and similar that reach a module from code outside it, with the code that made them;
- every torch op that reads a module's parameter or buffer, or looks up its device, with the recorded call it ran in;
- times from CUDA events without synchronizing after every call, or from `time.perf_counter` without CUDA;
- tensor bytes in and out (`bytes_in`, `bytes_out`), summed over a node's calls: the tensors in each call's
  args/kwargs and in its result, nested containers and ModelOutputs included, each tensor counted once per call.
  `UserDict`s, such as transformers' `BatchFeature` and `BatchEncoding`, are walked into their values, and dataclass
  instances into their fields. Other objects that `torch.utils._pytree` does not flatten, such as caches or your own
  classes that aren't dataclasses, are not looked into;
- per node, its `signatures`: each distinct set of arguments and results its calls passed, with the indices of
  the calls that had it. Tensors are listed with dtype and exact shape, other objects by type name only;
  primitives such as numbers and strings are not recorded. `node.calls_passing("x")` and `node.shapes("x")` give
  the calls that passed `x` and the shapes it took;
- per leaf, its `source`: where the same object was last seen in a recorded call, as that node's path, the side
  (`"inputs"` or `"outputs"`) and the leaf's name there. A tensor that one call returns and a later call receives
  links the two, so the leaves on one side of a cut whose source is on the other side are what crosses it. A value
  made in the call's own code, read from an attribute or global, or passed from outside the recorded calls has no
  source; neither has an object without weak-reference support, such as one whose class defines `__slots__`
  without `__weakref__`.

### How the call tree is recorded

`call_tree=True` adds a structure pass of `number_of_iterations` and a timing pass after the per-module pass that
`get_modules()` comes from, so inspection takes longer:

1. **The structure pass** records the tree: every call with its arguments, tensor bytes and weight accesses. It runs
   eagerly, parts compiled with `torch.compile` included, so the calls inside them are recorded too. This tracking
   runs in Python and would distort times, so the structure pass records none.
2. **The timing pass** replays the same samples in the same order, with all tracking off and compiled parts
   compiled. Each sample is replayed once per call-path depth its calls reach, so every call is timed exactly once
   on the sample the structure pass recorded it on, and no timed call runs inside another. The number of iterations —
   at most `number_of_iterations` times the number of call-path depths — is logged before the pass starts.

Some nodes get no time (`time_ms` is `null`):

- calls inside a compiled part, since they run inside the compiled code. The compiled part itself is timed as one
  call, and such nodes are marked `inside_compiled_part` in the export;
- nodes deeper than `max_timing_depth`. They are recorded but not timed, and `inspect` logs no warning for them.

`max_timing_depth` bounds the timing cost: fewer depths per sample means fewer iterations. It counts call-path depths,
with the root call at depth 1, not the attribute depths that `max_depth` counts. The deep leaf levels (linear
layers, norms) are rarely worth timing, and their calls and weight accesses are still recorded.

The times are close to a normal eager run, but not exact: every call inside a timed call still runs through its
recording wrapper, about 0.7 µs each. Where the GPU waits for the CPU, that adds up for calls with many calls inside:
on roberta-base at batch 1, the encoder reads about 10% high; at batch 8, within a few percent. Use the times to
compare nodes, or eager with tuned trees, not as deployed latency.

`get_modules()` and `describe()` are unchanged. The object walk has to reach the linear layers, so deep models need
a larger `max_depth` (8 for BAGEL-sized pipelines).

Limits: exact `nn.Sequential`, `nn.ModuleList` and `nn.ModuleDict` containers are skipped and their children
attached to the caller, unless the container itself is compiled with `module.compile()`. A call that takes a
callable as an argument (a checkpoint helper) is skipped the same way, and free functions are not recorded. CUDA
times come from events on the device that was current when a call started, so work that a call runs on other GPUs
is not fully counted.

### Export the call tree

`tree.to_dict()` returns a JSON-serializable dict. To write it to a file for tools outside the inspecting process
(no torch, no aitune), use Python's `json` module:

```python
import json

with open("call_tree.json", "w", encoding="utf-8") as f:
    json.dump(tree.to_dict(), f, indent=2)
```

The exported data has this structure:

```json
{
  "schema_version": 1,
  "iterations": 2,
  "total_ms": 30.0,
  "nodes": [
    {"call_path": "forward", "owner_class": "Vla", "calls": 2, "time_ms": 30.0, ...},
    {"call_path": "forward > dit_forward > dit.forward", "owner_class": "Dit",
     "calls": 10, "inside_compiled_part": false, "time_ms": 25.0,
     "bytes_in": 3670056, "bytes_out": 3670016,
     "signatures": [
         {"call_indices": [0, 5],
          "inputs": {"hidden_states": {"type": "torch.Tensor", "dtype": "torch.bfloat16",
                                       "shape": [1, 220, 4096],
                                       "source": {"call_path": "forward > dit_forward", "side": "inputs",
                                                  "name": "hidden_states"}},
                     "timestep": {"type": "torch.Tensor", "dtype": "torch.float32", "shape": [1]}},
          "outputs": {"output": {"type": "torch.Tensor", "dtype": "torch.bfloat16",
                                 "shape": [1, 220, 4096]}}},
         {"call_indices": [1, 2, 3, 4, 6, 7, 8, 9],
          "inputs": {"hidden_states": {"type": "torch.Tensor", "dtype": "torch.bfloat16",
                                       "shape": [1, 1, 4096]},
                     "timestep": {"type": "torch.Tensor", "dtype": "torch.float32", "shape": [1]}},
          "outputs": {"output": {"type": "torch.Tensor", "dtype": "torch.bfloat16",
                                 "shape": [1, 1, 4096]}}}],
     "calls_per_iteration": [5, 5]}
  ],
  "modules": {"dit": {"class": "my_pkg.Dit",
                      "parameters": {"weight": {"dtype": "torch.float32", "shape": [4, 4]}},
                      "parameter_bytes": 8192}},
  "weights": {"dit.proj.weight": [{"by": "forward > dit_forward", "kind": "read"}]}
}
```

`iterations` is the number of recorded iterations; `calls`, `bytes_in` and `bytes_out` are totals over them.
Divide by `iterations` for per-iteration values, by `calls` for per-call values. `time_ms` is the node's measured
total over the same iterations. It is `null` when the timing pass never timed the node.
`inside_compiled_part` is true for calls inside a compiled part: they run inside the compiled code in the timing
pass, so their `time_ms` is `null`; the compiled part's own call is timed.
`signatures` lists the node's distinct signatures in first-seen order: each has `call_indices`, the indices of
the calls with exactly that signature, plus `inputs`/`outputs` dicts keyed by argument name (such as `x` or
`inputs['input_ids']`), and `output` for the result, or `output[0]` and `output['logits']` for values nested in it.
Every leaf has `type`, tensors also `dtype` and exact `shape`; `outputs` is null for a signature whose calls raised.
A leaf with a `source` names where its object was last seen: `call_path`, `side` and the leaf `name` there, which can
be in any of that node's signatures. In the example, `dit.forward` received `hidden_states` as `dit_forward`'s own
argument of that name.
`calls_per_iteration` gives how many of the node's calls each iteration made; they add up to `calls`.
`nodes` lists every recorded call node in the order the nodes were first called. `call_path` is the chain of calls
that led to the node and identifies it; the rest follows from it:

```python
parent = call_path.rpartition(" > ")[0]                  # "" for outermost nodes
object_path, _, method = call_path.rpartition(" > ")[2].rpartition(".")
```

The object path is the attribute path from the inspected object to the object whose method was called, the same as
`ModuleInfo.object_path`: when you inspect `model`, `model.visual` is `model.model.visual`, `language_model.layers.0`
is `model.language_model.layers[0]`, and `""` is `model` itself (also for its own methods such as `dit_forward`).
Split at the last dot: method names never contain one, object paths can. The keys of `modules` are object paths, so
`modules[object_path]` gives a module node's class name, its own parameters and buffers in `parameters` (each with
`dtype` and `shape` — a LayerNorm(2048) shows `weight: [2048]`, which is the input width the module expects), and
`parameter_bytes` over the module and its submodules. `weights` lists the recorded accesses
(reads, device lookups, moves) per weight. In distributed runs write the file on one rank.

Later versions may add fields without changing `schema_version`, so readers should ignore keys they don't know.
`schema_version` changes when a field is removed, renamed or changes its meaning.

### Match wrapped modules to call-tree nodes

`ait.wrap` names each wrapped module by its object path — `model.blocks[0].attn` is wrapped as `blocks.0.attn` —
so its name is the `object_path` of its call-tree nodes:

```python
wrapped_name = call_path.rpartition(" > ")[2].rpartition(".")[0]   # "blocks.0.attn"
```

Every node below it is covered by that wrapped module. The same name is the module's `module_name` in the tuning
report and is stored in its checkpoint, so the match also works after `ait.save` and `ait.load`. The root module keeps
its class name and covers the whole tree. A module reachable under two paths appears under the one the inspector
found first.

You can run `inspect(..., call_tree=True)` again after `ait.tune` or `ait.load`. A wrapped module's node keeps its
`call_path`, so the two saved trees can be joined by `call_path` to compare eager and tuned time per node inside the
pipeline. Nodes inside a module tuned with a compiling backend do not appear, because the compiled module runs as
one call.

## Shared weights and wrap

After tuning, AITune offloads a module's original weights (`device_after_tuning`, `"meta"` by default). That breaks a
model when something outside the tuned module still needs one of those weights:

- a head tied to an embedding that another module uses;
- an embedding looked up outside the module, for example by the pipeline before it calls the module;
- a pipeline that reads the module's device from its parameters, as diffusers' `_get_execution_device` does with
  `next(module.parameters()).device`;
- library code that moves the whole model with `.to()` on every call, as `SentenceTransformer.encode` does.

The call tree finds these. `shared_weights_of` takes the call tree and a module, a node path or a list of them, and
returns one entry per weight — a parameter or a buffer — whose `accesses` list the outside uses, sorted by name:
reads (`"read"`), device lookups (`"device"`) and moves (`"move"`). `bytes` is `numel * element_size` per weight; views of
one storage are each counted in full. Reads count per op, so a direct read such as `F.linear(x, model.lm_head.weight)`
in a caller is found. A module outside the unit that itself holds one of the unit's weights counts as reading it
whenever its `forward` runs, because it may pass the weight to a C++ or CUDA extension torch does not see. A module
holds all its parameters and buffers, because offloading moves all of them; a module the call tree has no `forward` of
(an exact `nn.Sequential`, or a `forward` that takes a callable) stands for the recorded calls inside it:

```python
from aitune.torch.call_tree import shared_weights_of

tree = modules_info.call_tree
shared_weights_of(tree, model.lm_head)
# [SharedWeight(name="llm.embed.weight", aliases=["lm_head.weight"], bytes=512,
#               accesses=[WeightAccess("forward > llm.embed.forward", "read")])]
shared_weights_of(tree, model.dit)    # []: can be tuned and offloaded on its own
```

Wrap modules from an inspection with `call_tree=True`, and tuning keeps exactly those weights on the device when it
offloads a wrapped module:

```python
head = [m for m in modules_info.get_modules() if m.module is model.lm_head]
model = ait.wrap(model, head)
ait.tune(run_chunk, input_data)
# Keeping 1 weight (0.0 MB) of lm_head on the device after tuning: read by forward > llm.embed.forward
```

Each wrapped module logs one line for the weights it keeps (and a debug line per weight). When code outside the
module moves it with `.to()`, the moved buffers stay along with its parameters, because that `.to()` moves them
too.

Move or cast the model before `wrap`. Kept weights are tracked as tensor objects: a device or dtype change after
`wrap`, such as `model.to("cuda")` or `model.half()`, gives buffers new tensors, which are then no longer kept.

The cost is the memory of the kept weights. For weights that outside code reads, the model needs that memory
anyway. For weights that outside code only moves, it does not: E5's `SentenceTransformer.encode` moves the
whole encoder with `.to()` on every call, so its 1.34 GB stay on the GPU next to the tuned engine. That is the same
as the `device_after_tuning="cpu"` workaround, whose `.to()` brings them back, but offloading saves nothing there.

`ait.save` stores the names of the kept weights with each tuned module, and `ait.load` keeps them on the device
when it offloads the originals, so a deployment process needs no call tree. Each wrapped module owns the weights it
keeps: wrapping it again replaces its set, and a weight stays kept while any wrapped module keeps it, also across
models in the same process.

Modules from `inspect` without `call_tree=True`, or `ModuleInfo` built by hand, keep nothing, so `wrap` and tuning
behave as before.

The call tree has these limits:

- It only knows the paths the recording ran. A weight read or a device lookup on another path (another branch,
  another input size) is not seen.
- It assumes the model runs on one thread, like the rest of AITune: weight reads on other threads are not seen,
  and a `.to()` on another thread during recording can hide reads.
- It keeps following a parameter that gets new storage without `.to()`, for example through `param.data = ...` or
  accelerate's `set_module_tensor_to_device`. It loses track of a buffer that is replaced the same way.

## Troubleshooting

### Issue: No modules found

```python
# Solution: Increase min_depth or check object structure
modules_info = ait.inspect(model, input_data, min_depth=1)
```

### Models with parts compiled by `torch.compile`

A part compiled with `torch.compile` or `nn.Module.compile()` counts as one call: `inspect` times the compiled module
but not the modules it calls, so `get_modules()` cannot return them, and `inspect` logs a warning for each such part.
To inspect inside compiled parts, run `inspect` with compilation turned off; compiled parts then run eagerly, and their
times are eager times:

```python
with torch.compiler.set_stance("force_eager"):
    modules_info = ait.inspect(model, input_data, min_depth=2)
```

A child compiled with `torch.compile(module)` is an `OptimizedModule` that holds the original module as `_orig_mod`, so
its submodules are one level deeper: `.encoder._orig_mod.proj` instead of `.encoder.proj`.

## Summary

Ahead-of-time inspection can be a first step in exploring your model's structure and performance. It can also be used to select modules for tuning. For details on the tuning workflow, head to the [AOT Tuning Guide](aot_tuning.md).
