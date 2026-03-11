# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# dependencies = ["timm"]
# ///

import tempfile
from logging import INFO, basicConfig, getLogger
from pathlib import Path

import timm
import torch

from aitune.torch.backend.tensorrt.tensorrt_backend import TensorRTBackend
from aitune.torch.backend.torch_inductor_backend import TorchInductorBackend
from aitune.torch.backend.torch_tensorrt_aot_backend import TorchTensorRTAotBackend, TorchTensorRTAotBackendConfig
from aitune.torch.backend.torch_tensorrt_jit_backend import (
    TorchTensorRTConfig,
    TorchTensorRTJitBackend,
    TorchTensorRTJitBackendConfig,
)
from aitune.torch.module.wrapper_module import Module
from aitune.torch.module_registry import MODULE_REGISTRY
from aitune.torch.tune_strategy.one_backend_strategy import OneBackendStrategy
from aitune.torch.tuning import load, save, tune

basicConfig(level=INFO, force=True)
logger = getLogger(__name__)


def _tune_save_load_helper(model_factory, samples, output_dir, wrap_model_fn):
    """Helper function to handle common tune-save-load-verify workflow.

    Args:
        model_factory: Function that creates a fresh model instance.
        samples (torch.Tensor): Input samples for model inference.
        output_dir (Path): Directory to save the tuned model.
        wrap_model_fn: Function that takes a model and returns a wrapped version
            ready for tuning.

    Returns:
        None

    Raises:
        AssertionError: If model predictions after load don't match expected results.
    """
    model = model_factory()
    model.eval()
    test_data = samples.repeat(10, 1, 1, 1)
    with torch.no_grad():
        expected = model(test_data)

    wrapped_model = wrap_model_fn(model)
    logger.info("Tuning model")
    tune(wrapped_model, samples, batch_sizes=[1, 10], dry_run=False, disable_external_logging=False)
    logger.info("Saving model")
    save(wrapped_model, output_dir)

    # create a new model instance and load the tuned model
    model = model_factory()
    model.eval()
    logger.info("Loading model")
    model_loaded = load(model, output_dir, disable_external_logging=False)

    logger.info("Verifying model")
    with torch.no_grad():
        preds = model_loaded(test_data)
    torch.testing.assert_close(preds, expected, rtol=1e-3, atol=1e-3)


def tune_save_load_whole_model(model_factory, samples, output_dir, backend_factory):
    """Test tuning, saving and loading a whole model wrapped in a single Module."""

    def wrap_whole_model(model):
        strategy = OneBackendStrategy(backend_factory())
        strategy.enable_find_max_batch_size(enable=False)
        return Module(model, "demo-simple", strategy=strategy)

    _tune_save_load_helper(model_factory, samples, output_dir, wrap_whole_model)


def tune_save_load_part_of_model(model_factory, samples, output_dir, backend_factory):
    """Test tuning, saving and loading a model with individual layers wrapped in Modules."""

    def wrap_partial_model(model):
        strategy = OneBackendStrategy(backend_factory())
        strategy.enable_find_max_batch_size(enable=False)
        model.layer1 = Module(model.layer1, "demo-simple1", strategy=strategy)
        model.layer2 = Module(model.layer2, "demo-simple2", strategy=strategy)
        return model

    _tune_save_load_helper(model_factory, samples, output_dir, wrap_partial_model)


def tensorrt_jit_backend():
    return TorchTensorRTJitBackend(
        config=TorchTensorRTJitBackendConfig(
            compile_config=TorchTensorRTConfig(
                enabled_precisions={torch.float32, torch.float16, torch.bfloat16},
                use_python_runtime=False,
            ),
            dynamic_shapes=False,
        )
    )


def tensorrt_aot_backend():
    return TorchTensorRTAotBackend(
        config=TorchTensorRTAotBackendConfig(
            compile_config=TorchTensorRTConfig(
                enabled_precisions={torch.float32, torch.float16, torch.bfloat16},
            ),
        ),
    )


def test_backends_serialization():
    model_factory = lambda: timm.create_model("resnet18", pretrained=False).cuda()  # noqa: E731
    samples = torch.randn(3, 224, 224).cuda()

    for backend_factory in [
        TensorRTBackend,
        tensorrt_aot_backend,
        tensorrt_jit_backend,
        TorchInductorBackend,
    ]:
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info("Testing %s", backend_factory.__name__)
            logger.info("Testing whole model")
            output_path = Path(temp_dir) / "top_model_test.pt"
            tune_save_load_whole_model(model_factory, samples, output_path, backend_factory)
            MODULE_REGISTRY.clear()
            torch.cuda.empty_cache()
            logger.info("Testing partial model")
            output_path = Path(temp_dir) / "partial_model_test.pt"
            tune_save_load_part_of_model(model_factory, samples, output_path, backend_factory)
            MODULE_REGISTRY.clear()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    test_backends_serialization()
