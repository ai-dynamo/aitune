..
    SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
    SPDX-License-Identifier: Apache-2.0

NVIDIA AITune
=============

|License| |Python| |PyTorch|

**NVIDIA AITune** is an inference toolkit designed for tuning and deploying Deep Learning models with a focus on NVIDIA GPUs. It provides model tuning capabilities through compilation and conversion paths that can significantly improve inference speed and efficiency across various AI workloads including Computer Vision, Natural Language Processing, Speech Recognition, and Generative AI.

The toolkit enables seamless tuning of PyTorch models and pipelines using various backends such as TensorRT, Torch-TensorRT, TorchAO, and Torch Inductor through a single Python API. The resulting tuned models are ready for deployment in production environments.

NVIDIA AITune works with your environment — relying first on your software versions — and selects the best-performing backend for your software and hardware setup, guiding you to supported technologies.

**Note**: This is the first release. The API may change in future versions.

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
|                        | TorchAO, and Torch Inductor                                                      |
+------------------------+----------------------------------------------------------------------------------+
| Model Tuning           | Enhance the performance of models such as ResNET and BERT for efficient          |
|                        | inference deployment                                                             |
+------------------------+----------------------------------------------------------------------------------+
| Pipeline Tuning        | Streamline Python code pipelines for models such as Stable Diffusion and Flux    |
|                        | using seamless model wrapping and tuning                                         |
+------------------------+----------------------------------------------------------------------------------+
| Model Export and       | Automate the process of exporting and converting models between various formats  |
| Conversion             | with focus on TensorRT and Torch-TensorRT                                        |
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

Installing from Source
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/ai-dynamo/aitune
    cd aitune
    pip install --extra-index-url https://pypi.nvidia.com .

    # Or use editable mode for development
    pip install --extra-index-url https://pypi.nvidia.com -e .

Quick Start
-----------

This quick start provides examples of tuning and deployment paths available in NVIDIA AITune.

NVIDIA AITune enables seamless tuning of models for deployment (for example, converting them to TensorRT) without requiring changes to your original Python pipelines.

The code below demonstrates Stable Diffusion pipeline tuning. First, install the required third-party dependencies:

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

Core Functionalities
--------------------

Inspect
~~~~~~~

The ``inspect`` function allows you to analyze PyTorch models and pipelines to understand their structure, parameters, and execution flow. It provides detailed insights into model architecture and helps identify tuning opportunities.

.. code-block:: python

    import aitune.torch as ait
    import torch
    import torch.nn as nn

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(100, 10)

        def forward(self, x):
            return self.linear(x)

    model = SimpleModel()
    dataset = torch.randn(1, 100)

    # Inspect the model
    ait.inspect(model, dataset)

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

*Note:* We recommend deploying ``*.ait`` package on the same hardware as tuning has been performed for functional and performance compatibility.


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

    config = TensorRTBackendConfig(quantization_config=ONNXAutoCastConfig())
    backend = TensorRTBackend(config)

CUDA Graphs Support
~~~~~~~~~~~~~~~~~~~

The TensorRT backend supports CUDA Graphs for reduced CPU overhead and improved inference performance. This feature is disabled by default.

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

Torch-TensorRT backend integrates TensorRT tuning directly into PyTorch, providing seamless tuning without model conversion through ``torch_tensorrt.compile``.

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

Torch Inductor Backend
~~~~~~~~~~~~~~~~~~~~~~

Torch Inductor backend uses PyTorch's Inductor compiler for model tuning.

.. code-block:: python

    from aitune.torch.backend import TorchInductorBackend

    backend = TorchInductorBackend()

Tune Strategies
---------------

NVIDIA AITune provides different strategies for selecting the optimal backend configuration. The strategies align with a common interface for the tuning process.

Not every backend can tune every model — each relies on different compilation technology with its own
limitations (e.g., ONNX export for TensorRT, graph breaks in Torch Inductor, unsupported layers in TorchAO).
Strategies control how AITune handles this.

FirstWinsStrategy
~~~~~~~~~~~~~~~~~

Tries backends in priority order and returns the first one that succeeds. If a backend fails, the strategy
moves on to the next candidate instead of aborting.

.. code-block:: python

    from aitune.torch.tune_strategy import FirstWinsStrategy

    strategy = FirstWinsStrategy(backends=[TensorRTBackend(), TorchInductorBackend()])

OneBackendStrategy
~~~~~~~~~~~~~~~~~~

Uses exactly one backend, failing immediately with the original error if it cannot build. Use this when you
have already validated that a backend works and want deterministic behavior. Unlike ``FirstWinsStrategy`` with
a single backend, ``OneBackendStrategy`` surfaces the original exception rather than catching it.

.. code-block:: python

    from aitune.torch.tune_strategy import OneBackendStrategy

    strategy = OneBackendStrategy(backend=TensorRTBackend())

HighestThroughputStrategy
~~~~~~~~~~~~~~~~~~~~~~~~~

Profiles all compatible backends and selects the fastest. Use this when maximum throughput matters and you
can afford longer tuning time.

.. code-block:: python

    from aitune.torch.tune_strategy import HighestThroughputStrategy

    strategy = HighestThroughputStrategy(backends=[TensorRTBackend(), TorchInductorBackend(), TorchEagerBackend()])

Examples
--------

We offer comprehensive examples that showcase the utilization of NVIDIA AITune's diverse features. These examples are designed to elucidate the processes of tuning, profiling, testing, and deployment of models.

For detailed examples and step-by-step guides, please visit our `Examples Catalog <https://github.com/ai-dynamo/aitune/tree/main/examples>`_. The catalog includes practical implementations for various AI workloads including computer vision, natural language processing, speech recognition, and generative AI models.

Useful Links
------------

* `Changelog <https://github.com/ai-dynamo/aitune/blob/main/CHANGELOG.md>`_
* `Contributing <https://github.com/ai-dynamo/aitune/blob/main/CONTRIBUTING.md>`_
* `License <https://github.com/ai-dynamo/aitune/blob/main/LICENSE>`_

Links valid after first official release:

* `Documentation <https://github.com/ai-dynamo/aitune>`_
* `GitHub Issues <https://github.com/ai-dynamo/aitune/issues>`_

.. |License| image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :target: https://github.com/ai-dynamo/aitune/blob/main/LICENSE
.. |Python| image:: https://img.shields.io/badge/python-3.10+-blue.svg
   :target: https://www.python.org/downloads/
.. |PyTorch| image:: https://img.shields.io/badge/PyTorch-2.7+-red.svg
   :target: https://pytorch.org/
