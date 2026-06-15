..
    SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
    SPDX-License-Identifier: Apache-2.0

NVIDIA AITune
=============

|License| |Python| |PyTorch|

**NVIDIA AITune** is an inference toolkit designed for tuning and deploying Deep Learning models with a focus on NVIDIA GPUs. It provides model tuning capabilities through compilation and conversion paths that can significantly improve inference speed and efficiency across various AI workloads including Computer Vision, Natural Language Processing, Speech Recognition, and Generative AI.

The toolkit enables seamless tuning of PyTorch models and pipelines using various backends such as TensorRT, Torch-TensorRT, TorchAO, Torch Inductor, and ONNX Runtime through a single Python API. The resulting tuned models are ready for deployment in production environments.

NVIDIA AITune works with your environment — relying first on your software versions — and selects the best-performing backend for your software and hardware setup, guiding you to supported technologies.

**Note**: This is the first release. The API may change in future versions.

  **NOTICE AND DISCLAIMER: This software automatically retrieves, accesses or interacts with external materials.
  Those retrieved materials are not distributed with this software and are governed solely by separate terms,
  conditions and licenses. You are solely responsible for finding, reviewing and complying with all applicable terms,
  conditions, and licenses, and for verifying the security, integrity and suitability of any retrieved materials for
  your specific use case. This software is provided "AS IS", without warranty of any kind. The author makes no
  representations or warranties regarding any retrieved materials, and assumes no liability for any losses,
  damages, liabilities or legal consequences from your use or inability to use this software or any retrieved
  materials. Use this software and the retrieved materials at your own risk.**

Features at Glance
------------------

The distinct capabilities of NVIDIA AITune are summarized in the feature matrix:

+------------------------+----------------------------------------------------------------------------------+
| Feature                | Description                                                                      |
+========================+==================================================================================+
| Ease-of-use            | Single line of code to run all possible tuning paths directly from your source   |
|                        | code                                                                             |
+------------------------+----------------------------------------------------------------------------------+
| Wide Backend Support   | Compatible with various tuning backends including TensorRT, Torch-TensorRT,      |
|                        | TorchAO, Torch Inductor, and ONNX Runtime                                        |
+------------------------+----------------------------------------------------------------------------------+
| Model Tuning           | Enhance the performance of models such as ResNET and BERT for efficient          |
|                        | inference deployment                                                             |
+------------------------+----------------------------------------------------------------------------------+
| Pipeline Tuning        | Streamline Python code pipelines for models such as Stable Diffusion and Flux    |
|                        | using seamless model wrapping and tuning                                         |
+------------------------+----------------------------------------------------------------------------------+
| Model Export and       | Automate the process of exporting and converting models between various formats  |
| Conversion             | with focus on TensorRT, Torch-TensorRT, Torch Inductor, and ONNX Runtime         |
+------------------------+----------------------------------------------------------------------------------+
| Correctness Testing    | Ensures tuned models produce correct outputs by validating on provided data      |
|                        | samples                                                                          |
+------------------------+----------------------------------------------------------------------------------+
| Performance Profiling  | Profiles models to select the optimal backend based on performance metrics such  |
|                        | as latency and throughput                                                        |
+------------------------+----------------------------------------------------------------------------------+
| Model Persistence      | Save and load tuned models for production deployment with flexible storage       |
|                        | options                                                                          |
+------------------------+----------------------------------------------------------------------------------+
| JIT tuning             | Just-in-time tuning of a model or a pipeline without any code changes required   |
+------------------------+----------------------------------------------------------------------------------+


When to Use AITune
------------------

AITune provides compute graph optimizations for PyTorch models at the ``nn.Module`` level. Use AITune when you want automated inference optimization with minimal code changes.

If your model is supported by a dedicated serving framework and benefits from runtime optimizations (e.g. continuous batching, speculative decoding), use frameworks like TensorRT-LLM, vLLM, or SGLang for best performance. Use AITune for general PyTorch models and pipelines that lack such specialized tooling.

Prerequisites
-------------

Before proceeding with the installation of NVIDIA AITune, ensure your system meets the following criteria:

* **Operating System**: Linux (Ubuntu 22.04+ recommended)
* **Python**: Version ``3.10`` or newer
* **PyTorch**: Version ``2.7`` or newer
* **TensorRT**: Version ``10.5.0`` or higher (for TensorRT backend)
* **NVIDIA GPU**: Required for GPU-accelerated tuning

You can use NGC Containers for PyTorch which contain all necessary dependencies:

* `PyTorch NGC Container <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch>`_

Install
-------

NVIDIA AITune can be installed from ``pypi.org``.

Installing from PyPI (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    pip install --extra-index-url https://pypi.nvidia.com aitune

For PyTorch 2.10 with CUDA 13 support, install the ``torch210`` extra:

.. code-block:: bash

    pip install --extra-index-url https://pypi.nvidia.com --index-url https://download.pytorch.org/whl/cu130 "aitune[torch210]"

Installing from Source
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/ai-dynamo/aitune
    cd aitune
    pip install --extra-index-url https://pypi.nvidia.com .

or use editable mode for development:

.. code-block:: bash

    pip install --extra-index-url https://pypi.nvidia.com -e .

Quick Start
-----------

This quick start provides examples of tuning and deployment paths available in NVIDIA AITune.

NVIDIA AITune enables seamless tuning of models for deployment (for example, converting them to TensorRT) without requiring changes to your original Python pipelines.

NVIDIA AITune supports two modes:

* Ahead-of-time tuning — provide a model or a pipeline, and a dataset/dataloader. You can either rely on ``inspect`` to detect promising modules to tune or manually select them.
* Just-in-time tuning — set a special environment variable, run your script without changes, and AITune will, on the fly, detect modules and tune them one by one.

Ahead-of-time mode is more powerful and allows you to tweak more settings, whereas just-in-time works out of the box but offers less control over the tuning process. For a more detailed comparison, see the `Comparison between ahead-of-time and just-in-time tuning`_ section below.

Enabling logging
~~~~~~~~~~~~~~~~

The tuning process guides the user through decisions and steps that are performed to tune every selected module.

We recommend to enable the INFO logging level for better verbosity.

.. code-block:: python

    import logging

    logging.basicConfig(level=logging.INFO, force=True)

Ahead-of-time tuning
~~~~~~~~~~~~~~~~~~~~

The code below demonstrates Stable Diffusion pipeline tuning.

You can annotate ``torch.nn.Module``\ s manually or use the ``inspect`` functionality to have modules picked automatically; you can then verify them and schedule them for tuning.

First, install the required third-party dependencies:

.. code-block:: bash

    pip install transformers diffusers torch

Then initialize the pipeline:

.. code-block:: python

    # HuggingFace dependencies
    import torch
    from diffusers import DiffusionPipeline

    # Import AITune
    import aitune.torch as ait

    # Initialize pipeline
    pipe = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
    pipe.to("cuda")

Next, `inspect` the pipeline components and display the summary:

.. code-block:: python

    # Prepare input data
    input_data = [{"prompt": "A beautiful landscape with mountains and a lake"}]

    # Inspect pipeline to get modules
    modules_info = ait.inspect(pipe, input_data)


    # Optional: inference function, if you need more control over execution
    def infer(prompt):
        return pipe(prompt, width=1024, height=1024, num_inference_steps=10)


    modules_info = ait.inspect(pipe, input_data, inference_function=infer)

    # Display modules info
    modules_info.describe()

Finally, `wrap` the selected modules and `tune` within the pipeline:

.. code-block:: python

    # Wrap modules for tuning
    modules = modules_info.get_modules()
    pipe = ait.wrap(pipe, modules)

    # Tune pipeline
    ait.tune(pipe, input_data)

At this point, you can use the pipeline to generate predictions with the tuned models directly in Python:

.. code-block:: python

    # Run inference on tuned pipeline
    images = pipe(["A beautiful landscape with mountains and a lake"])
    image = images[0][0]

    # Save image for preview
    image.save("landscape.png")

Once the pipeline has been tuned, you can save the best-performing version of the modules for later deployment:

.. code-block:: python

    ait.save(pipe, "tuned_pipe.ait")

And load the tuned pipeline directly:

.. code-block:: python

    pipe = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
    pipe.to("cuda")
    ait.load(pipe, "tuned_pipe.ait")

Just-in-time tuning
~~~~~~~~~~~~~~~~~~~

In this mode, there is no need to modify the user's code. AITune records inference calls until ``jit_config.min_samples`` is met, then tries to tune modules one by one starting from the top. If there is one of the following conditions:

* a graph break is detected, i.e., ``torch.nn.Module`` contains conditional logic on inputs, meaning there is no guarantee of a static, correct graph of computations, or
* there is an error during tuning

that module is left unchanged and AITune tries to tune its children. This process continues until the module depth reaches a configured limit.

First, install the required third-party dependencies:

.. code-block:: bash

    pip install transformers diffusers torch

Prepare the example script for tuning ``my_script.py``:

.. code-block:: python

    # Enable JIT tuning
    import aitune.torch.jit.enable

    # HuggingFace dependencies
    import torch
    from diffusers import DiffusionPipeline

    # Initialize pipeline
    pipe = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
    pipe.to("cuda")

    # First call - tuning the model
    pipe("A beautiful landscape with mountains and a lake")

    # Second call - using tuned model
    pipe("A beautiful landscape with mountains and a lake")

You can then run your script:

.. code-block:: bash

    python my_script.py

*Note*: The ``import aitune.torch.jit.enable`` must be a first import in your code. The alternative option is to use ``export AUTOWRAPT_BOOTSTRAP=aitune_enable_jit_tuning`` to avoid any source code modification.

Configuring just-in-time tuning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If there is a need to adjust just-in-time options, you can do it but currently this requires modifying code to import the JIT config:

.. code-block:: python

    from aitune.torch.jit.config import config as jit_config
    from aitune.torch.backend import TensorRTBackend
    from aitune.torch.tune_strategy import FirstWinsStrategy

    jit_config.max_depth_level = 1 # change the default maximum depth level for nested modules to be tuned
    jit_config.detect_graph_breaks = False # turn off graph break detection
    jit_config.strategy = FirstWinsStrategy(backends=[TensorRTBackend()]) # change the tune strategy

.. _comparison-aot-jit:

Comparison between ahead-of-time and just-in-time tuning
--------------------------------------------------------

The ahead-of-time tuning gives you the most control over the tuning process:

* it detects the batch axis and dynamic axes (axes that change shape independently of batch size, e.g., sequence length in LLMs)
* allows picking modules to tune
* you can pick a tuning strategy (e.g., best throughput) for the whole process or per-module
* you can pick tuning backends (e.g., TensorRT, TorchInductor, TorchAO, ONNXRuntime) which will be used by the strategy
* you can mix different backends in the same model/pipeline
* you can manually verify the tuning process (note: AITune performs basic checks for NaNs and errors)
* you can save the resulting artifact and later read it from disk

The big advantage of just-in-time tuning is that you don't need to modify the user's script to tune a model. However, it has some disadvantages — since it cannot access data directly (you don't provide a dataloader):

* it cannot deduce batch size nor do benchmarking
* input/output shapes depend on the data seen, so for example, TRT backend will build a profile only for that data
* it needs at least one inference call to record inputs before tuning; later calls use tuned modules where tuning succeeded
* if you need dynamic axes (e.g., TRT backend), you need to provide two different batch sizes
* benchmarking-based strategies are limited because JIT cannot extrapolate to controlled batch sizes
* you can specify a global tune strategy for the whole model

The following table summarizes the difference between modes:

+------------------------+-----------------------+-------------------------------+
| Feature                | Ahead-of-time         | Just-in-time                  |
+========================+=======================+===============================+
| Detecting dynamic axes | Yes                   | Yes                           |
+------------------------+-----------------------+-------------------------------+
| Extrapolating batches  | Yes                   | No                            |
+------------------------+-----------------------+-------------------------------+
| Benchmarking           | Yes                   | No (no extrapolating batches) |
+------------------------+-----------------------+-------------------------------+
| Modules for tuning     | User has full control | Picked automatically          |
+------------------------+-----------------------+-------------------------------+
| Selecting tune strategy| Global or per module  | Global                        |
+------------------------+-----------------------+-------------------------------+
| Available strategies   | All                   | Global only                   |
+------------------------+-----------------------+-------------------------------+
| Tune time              | Slow                  | Quick                         |
+------------------------+-----------------------+-------------------------------+
| Saving artifacts       | Yes                   | No                            |
+------------------------+-----------------------+-------------------------------+
| Load tuned model time  | Quick                 | Re-tuning required            |
+------------------------+-----------------------+-------------------------------+
| Code changes required  | Yes                   | No                            |
+------------------------+-----------------------+-------------------------------+
| Caching                | Yes                   | Build artifacts only; no reuse|
+------------------------+-----------------------+-------------------------------+

Note: JIT mode writes build artifacts and logs under ``jit_config.cache_dir`` / ``AITUNE_JIT_CACHE_DIR``, but it does not reuse them as tuned checkpoints across Python interpreter runs. Every new process starts tuning from scratch.

Core Functionalities
--------------------

Inspect for AOT tuning
~~~~~~~~~~~~~~~~~~~~~~

The ``inspect`` function allows you to analyze PyTorch models and pipelines to understand their structure, parameters, and execution flow. It provides detailed insights into model architecture and helps identify tuning opportunities.

.. code-block:: python

    import aitune.torch as ait
    import torch.nn as nn

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(100, 10)

        def forward(self, x):
            return self.linear(x)

    model = SimpleModel()

    # Inspect the model
    ait.inspect(model, dataset)

Inspect for JIT tuning
~~~~~~~~~~~~~~~~~~~~~~

JIT tuning also has a corresponding ``inspect`` mode which gathers information about the model/pipeline and allows checking model input and output arguments, hierarchy of the model, etc.

Here is a short snippet how to use it:

.. code-block:: python

    # required imports
    import aitune.torch.jit.enable_inspection as inspection

    # your code goes here
    # ...

    # you can export report to html file
    inspection.save_report("filename.html", "YOUR_MODEL_NAME")

Tune
~~~~

The ``tune`` function is the core functionality that automatically tunes your PyTorch models and pipelines for optimal inference performance. It supports various backends and automatically selects the best performing configuration.

.. code-block:: python

    import aitune.torch as ait
    import torch

    # Define your model
    model = SimpleModel()

    # Wrap the model
    model = ait.Module(model)

    # Define inference function
    def inference_fn(x):
        return model(x)

    # Tune the model
    ait.tune(
        func=inference_fn,
        dataset=torch.randn(1, 100),
    )

Save
~~~~

The ``save`` function allows you to persist tuned models for later use. It stores tuned and
original module weights together in a single file with a ``.ait`` extension. Apart from the checkpoint file,
there is also a SHA hash file.

.. code-block:: python

    # Save the tuned model
    import aitune.torch as ait
    ait.save(model, "tuned_model.ait")

Example output:

.. code-block:: bash

    checkpoints/
    ├── tuned_model
    ├── tuned_model.ait
    └── tuned_model_sha256_sums.txt

You can copy the checkpoint file ``tuned_model.ait`` and SHA sums file to a target host or folder to use it for inference.

*Note:* We recommend to deploy ``*.ait`` package on the same hardware as tuning has been performed for functional and performance compatibility.


Load
~~~~

The ``load`` function enables you to load previously tuned models from a checkpoint file.

.. code-block:: python

    # Load the tuned model
    import aitune.torch as ait
    tuned_model = ait.load(model, "tuned_model.ait")

On first load, the checkpoint file is decompressed and the tuned and original module weights are loaded. Subsequent loads will use the decompressed weights from the same folder.

Backends
--------

NVIDIA AITune supports multiple tuning backends, each with different characteristics and use cases. The backends align with a common interface for the build and inference process.

TensorRT Backend
~~~~~~~~~~~~~~~~

The TensorRT backend provides highly optimized inference using NVIDIA's TensorRT engine. It offers the best performance for production deployments. The backend integrates `TensorRT Model Optimizer <https://github.com/NVIDIA/TensorRT-Model-Optimizer>`_ in a seamless flow.

.. code-block:: python

    from aitune.torch.backend import TensorRTBackend, TensorRTBackendConfig, ONNXAutoCastConfig

    config = TensorRTBackendConfig(quantization_config=ONNXAutoCastConfig())  # FP16 autocast through ModelOpt
    backend = TensorRTBackend(config)

CUDA Graphs Support
~~~~~~~~~~~~~~~~~~~

The TensorRT backend supports CUDA Graphs for reduced CPU overhead and improved inference performance. CUDA Graphs automatically capture and replay GPU operations, eliminating kernel launch overhead for repeated inference calls. This feature is disabled by default.

Keep in mind that graphs are automatically recaptured when input shapes change.

.. code-block:: python

    from aitune.torch.backend import TensorRTBackend, TensorRTBackendConfig

    config = TensorRTBackendConfig(use_cuda_graphs=True)
    backend = TensorRTBackend(config)

Torch-TensorRT Backend (JIT)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Torch-TensorRT JIT backend integrates TensorRT tuning directly into PyTorch, providing seamless tuning without model conversion through
``torch.compile``.

.. code-block:: python

    import torch
    from aitune.torch.backend import TorchTensorRTJitBackend, TorchTensorRTJitBackendConfig, TorchTensorRTConfig

    config = TorchTensorRTJitBackendConfig(compile_config=TorchTensorRTConfig(enabled_precisions={torch.float16}))
    backend = TorchTensorRTJitBackend(config)

Torch-TensorRT Backend (AOT)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Torch-TensorRT AOT backend integrates TensorRT tuning directly into PyTorch, providing seamless tuning without model conversion through ``torch_tensorrt.compile``.

.. code-block:: python

    import torch
    from aitune.torch.backend import TorchTensorRTAotBackend, TorchTensorRTAotBackendConfig, TorchTensorRTConfig

    config = TorchTensorRTAotBackendConfig(compile_config=TorchTensorRTConfig(enabled_precisions={torch.float16}))
    backend = TorchTensorRTAotBackend(config)

TorchAO Backend
~~~~~~~~~~~~~~~

TorchAO backend leverages PyTorch's AO (Accelerated Optimization) framework for model tuning.

.. code-block:: python

    from aitune.torch.backend import TorchAOBackend

    backend = TorchAOBackend()

Torch Inductor Backend (JIT)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Torch Inductor JIT backend uses PyTorch's Inductor compiler through ``torch.compile`` for model tuning.

.. code-block:: python

    from aitune.torch.backend import TorchInductorJitBackend

    backend = TorchInductorJitBackend()

Torch Inductor Backend (AOT)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Torch Inductor AOT backend uses PyTorch's AOT Inductor compiler to produce a compiled artifact that can be saved and loaded with AITune checkpoints.

.. code-block:: python

    from aitune.torch.backend import TorchInductorAotBackend

    backend = TorchInductorAotBackend()

ONNXRuntime Backend
~~~~~~~~~~~~~~~~~~~

ONNXRuntime backend exports the selected PyTorch module to ONNX and runs inference through ONNX Runtime with CUDA or TensorRT execution providers.

.. code-block:: python

    from aitune.torch.backend import ONNXRuntimeBackend, ONNXRuntimeBackendConfig, ONNXExecutionProvider

    config = ONNXRuntimeBackendConfig(execution_provider=ONNXExecutionProvider.CUDA)
    backend = ONNXRuntimeBackend(config)

Tune Strategies
---------------

NVIDIA AITune provides different strategies for selecting the optimal backend configuration. The strategies align with a common interface for the tuning process.

Not every backend can tune every model — each relies on different compilation technology with its own
limitations (e.g., ONNX export for TensorRT, graph breaks in Torch Inductor, unsupported layers in TorchAO).
Strategies control how AITune handles this.

Strategies also validate performance against a Torch eager baseline. Use
``strategy.enable_performance_validation(False)`` when you want to keep a correct backend regardless of speed
and skip baseline profiling, candidate performance checks, and speedup reporting.

FirstWinsStrategy
~~~~~~~~~~~~~~~~~

Tries backends in priority order and returns the first one that builds, validates correctness, and beats the
Torch eager baseline by the configured threshold. If a backend fails or is slower than baseline, the strategy
moves on to the next candidate instead of aborting.

.. code-block:: python

    from aitune.torch.tune_strategy import FirstWinsStrategy

    strategy = FirstWinsStrategy(backends=[TensorRTBackend(), TorchInductorJitBackend()])

OneBackendStrategy
~~~~~~~~~~~~~~~~~~

Uses exactly one backend, failing immediately with the original error if it cannot build or validate
correctness. If the backend is correct but does not beat the eager performance gate, it falls back to
``TorchEagerBackend``. Unlike ``FirstWinsStrategy`` with a single backend, ``OneBackendStrategy`` surfaces build
and correctness exceptions rather than catching them.

.. code-block:: python

    from aitune.torch.tune_strategy import OneBackendStrategy

    strategy = OneBackendStrategy(backend=TensorRTBackend())

MaxThroughputStrategy
~~~~~~~~~~~~~~~~~~~~~~~~~

Profiles all compatible backends and selects the fastest one that beats the Torch eager baseline, falling back
to eager when no user backend is faster. Use this when maximum throughput matters and you can afford longer
tuning time.

.. code-block:: python

    from aitune.torch.tune_strategy import MaxThroughputStrategy

    strategy = MaxThroughputStrategy(backends=[TensorRTBackend(), TorchInductorJitBackend(), TorchEagerBackend()])

Profiling with NVTX
-------------------

NVIDIA AITune includes NVTX (NVIDIA Tools Extension) annotations for profiling and debugging. NVTX marks key operations in the code, making them visible in profiling tools like NVIDIA Nsight Systems.

**Note**: NVTX annotations are disabled by default to avoid overhead in production environments.

Enabling NVTX
~~~~~~~~~~~~~

To enable NVTX profiling, set the environment variable before running your script:

.. code-block:: bash

    export AITUNE_NVTX_EVENTS=1
    python your_script.py

Using with Nsight Systems
~~~~~~~~~~~~~~~~~~~~~~~~~

Once enabled, you can profile your application with Nsight Systems:

.. code-block:: bash

    AITUNE_NVTX_EVENTS=1 nsys profile -o output.nsys-rep -trace=cuda,nvtx,osrt python your_script.py

The NVTX annotations will appear as colored regions in the timeline, helping you identify:

* Backend inference calls (TensorRT, Torch-TensorRT, TorchAO, etc.)
* Tuning operation
* Performance bottlenecks

Hardware Metrics
----------------

NVIDIA AITune can collect hardware metrics during tuning and inference, giving you visibility into resource utilization per module and backend. Metrics are collected in a background process and reported at program exit.

**Note**: Hardware metrics collection is disabled by default to avoid overhead in production environments.

Enabling Hardware Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~

Set the environment variable before running your script:

.. code-block:: bash

    export AITUNE_HARDWARE_METRICS=1
    python your_script.py

Collected Metrics
~~~~~~~~~~~~~~~~~

The following metrics are sampled continuously (every 100 ms by default) and aggregated per module and backend:

+------------------------------------+--------------------------------------------+
| Category                           | Metrics                                    |
+====================================+============================================+
| **GPU memory** (per device)        | ``cuda:N`` used memory [GB]                |
+------------------------------------+--------------------------------------------+
| **GPU utilization** (per device)   | ``cuda:N`` utilization mean / max [%]      |
+------------------------------------+--------------------------------------------+
| **GPU power** (per device)         | ``cuda:N`` power mean / max [W]            |
+------------------------------------+--------------------------------------------+
| **Host CPU**                       | CPU utilization [%]                        |
+------------------------------------+--------------------------------------------+
| **Host memory**                    | Used / free system memory                  |
+------------------------------------+--------------------------------------------+
| **PyTorch allocator**              | Allocated and reserved CUDA memory         |
+------------------------------------+--------------------------------------------+

GPU metrics require NVML (available when running on a system with NVIDIA drivers). If NVML is unavailable, only host and PyTorch metrics are collected.

Output
~~~~~~

At program exit, AITune logs a summary table and writes a CSV file to the working directory.

By default a timestamped filename is used:

.. code-block:: text

    hardware_metrics_20260402_153012.csv

To write to a fixed path instead, set ``AITUNE_HARDWARE_METRICS_PATH``:

.. code-block:: bash

    export AITUNE_HARDWARE_METRICS_PATH=hardware_metrics.csv

The log summary looks like:

.. code-block:: text

    INFO Hardware metrics summary:
    ╒════════════════════════╤══════════════════════════════╤════════════╤════════════╤══════════════╤═════════════╤═════════════╤═════════════╕
    │ Module                 │ Backend                      │    Host    │   Cuda:0   │    Cuda:0    │   Cuda:0    │  Power [W]  │  Power [W]  │
    │                        │                              │  Mem [GB]  │  Mem [GB]  │  Util% mean  │  Util% max  │    mean     │     max     │
    ╞════════════════════════╪══════════════════════════════╪════════════╪════════════╪══════════════╪═════════════╪═════════════╪═════════════╡
    │ CLIPTextModel          │ TensorRTBackend(             │   15.53    │    1.73    │     1.03     │      7      │    72.33    │   112.26    │
    │                        │     quantization_config=None │            │            │              │             │             │             │
    │                        │ )                            │            │            │              │             │             │             │
    ├────────────────────────┼──────────────────────────────┼────────────┼────────────┼──────────────┼─────────────┼─────────────┼─────────────┤
    │ Decoder                │ TensorRTBackend(             │   15.43    │    1.81    │      12      │     56      │   100.88    │   148.19    │
    │                        │     quantization_config=None │            │            │              │             │             │             │
    │                        │ )                            │            │            │              │             │             │             │
    ├────────────────────────┼──────────────────────────────┼────────────┼────────────┼──────────────┼─────────────┼─────────────┼─────────────┤
    │ Decoder                │ TensorRTBackend(             │   15.46    │    1.8     │    33.38     │     60      │   117.22    │   167.79    │
    │                        │     use_dynamo=False,        │            │            │              │             │             │             │
    │                        │     quantization_config=None │            │            │              │             │             │             │
    │                        │ )                            │            │            │              │             │             │             │
    ├────────────────────────┼──────────────────────────────┼────────────┼────────────┼──────────────┼─────────────┼─────────────┼─────────────┤
    │ Decoder                │ TorchInductorJitBackend()    │   15.53    │    1.7     │     3.12     │     85      │    85.92    │   179.21    │
    ├────────────────────────┼──────────────────────────────┼────────────┼────────────┼──────────────┼─────────────┼─────────────┼─────────────┤
    │ FluxTransformer2DModel │ TensorRTBackend(             │   14.36    │    1.79    │      0       │      0      │    67.84    │    71.79    │
    │                        │     quantization_config=None │            │            │              │             │             │             │
    │                        │ )                            │            │            │              │             │             │             │
    ├────────────────────────┼──────────────────────────────┼────────────┼────────────┼──────────────┼─────────────┼─────────────┼─────────────┤
    │ FluxTransformer2DModel │ TensorRTBackend(             │   14.35    │    1.79    │      0       │      0      │    63.46    │    63.46    │
    │                        │     use_dynamo=False,        │            │            │              │             │             │             │
    │                        │     quantization_config=None │            │            │              │             │             │             │
    │                        │ )                            │            │            │              │             │             │             │
    ├────────────────────────┼──────────────────────────────┼────────────┼────────────┼──────────────┼─────────────┼─────────────┼─────────────┤
    │ FluxTransformer2DModel │ TorchInductorJitBackend()    │   15.53    │    1.79    │     2.44     │     85      │    84.09    │   179.21    │
    ├────────────────────────┼──────────────────────────────┼────────────┼────────────┼──────────────┼─────────────┼─────────────┼─────────────┤
    │ T5EncoderModel         │ TensorRTBackend(             │   16.65    │    1.77    │     1.76     │     85      │    70.57    │   179.21    │
    │                        │     quantization_config=None │            │            │              │             │             │             │
    │                        │ )                            │            │            │              │             │             │             │
    ╘════════════════════════╧══════════════════════════════╧════════════╧════════════╧══════════════╧═════════════╧═════════════╧═════════════╛

Combining with NVTX
~~~~~~~~~~~~~~~~~~~

Hardware metrics and NVTX profiling can be enabled together:

.. code-block:: bash

    AITUNE_HARDWARE_METRICS=1 AITUNE_NVTX_EVENTS=1 nsys profile -o output.nsys-rep -trace=cuda,nvtx,osrt python your_script.py

Examples
--------

We offer comprehensive examples that showcase the utilization of NVIDIA AITune's diverse features. These examples are designed to elucidate the processes of tuning, profiling, testing, and deployment of models.

For detailed examples and step-by-step guides, please visit our `Examples Catalog <https://github.com/ai-dynamo/aitune/blob/main/examples/README.md>`_. The catalog includes practical implementations for various AI workloads including computer vision, natural language processing, speech recognition, and generative AI models.

Useful Links
------------

* `Changelog <https://github.com/ai-dynamo/aitune/blob/main/CHANGELOG.md>`_
* `Contributing <https://github.com/ai-dynamo/aitune/blob/main/CONTRIBUTING.md>`_
* `License <https://github.com/ai-dynamo/aitune/blob/main/LICENSE>`_


.. |License| image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :target: https://github.com/ai-dynamo/aitune/blob/main/LICENSE
.. |Python| image:: https://img.shields.io/badge/python-3.10+-blue.svg
   :target: https://www.python.org/downloads/
.. |PyTorch| image:: https://img.shields.io/badge/PyTorch-2.7+-red.svg
   :target: https://pytorch.org/
