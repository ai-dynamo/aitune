# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for model state storage functionality."""

from unittest.mock import Mock

import pytest
import torch

from aitune.torch.backend.tensorrt.tensorrt_backend import TensorRTBackend
from aitune.torch.backend.torch_inductor_jit_backend import TorchInductorJitBackend
from aitune.torch.checkpoint.local_torch_storage import LocalTorchStorage
from aitune.torch.checkpoint.torch_checkpoint import TorchCheckpoint
from aitune.torch.module.wrapper_module import Module
from aitune.torch.tune_strategy.one_backend_strategy import OneBackendStrategy
from aitune.torch.tuning import tune
from tests.toy_models.torch_models import ToyComplexPipeline, ToyPipeline, ToyTorchModel
from tests.utilities.helpers import requires_cuda


@pytest.fixture
def checkpoint(tmp_path):
    return TorchCheckpoint(
        LocalTorchStorage(
            tmp_path,
            remove_checkpoint_after_tune=True,  # True to check unzip functionality
        )
    )


@pytest.fixture
def model_factory():
    return lambda: ToyTorchModel(is_linear=True).to("cuda")


@pytest.fixture
def pipeline_factory():
    return lambda: ToyPipeline().to("cuda")


@pytest.fixture
def complex_pipeline_factory():
    return lambda: ToyComplexPipeline().to("cuda")


@pytest.fixture
def sample(model_factory):
    yield model_factory().sample().to("cuda")


def _tune_save_load_helper(model_factory, samples, output_dir, wrap_model_fn, checkpoint, device_map=None):
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
    test_data = samples.repeat(10, 1)  # make a bs=10
    with torch.no_grad():
        expected = model(test_data)

    wrapped_model = wrap_model_fn(model)
    tune(wrapped_model, samples, batch_sizes=[1, 2], dry_run=False, disable_external_logging=False)
    checkpoint.save(wrapped_model, output_dir)

    # create a new model instance and load the tuned model
    model = model_factory()
    model_loaded = checkpoint.load(model, output_dir, device_map=device_map)

    with torch.no_grad():
        preds = model_loaded(test_data)
    torch.testing.assert_close(preds, expected, rtol=1e-3, atol=1e-3)


@requires_cuda
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(TorchInductorJitBackend(), id="torch_inductor"),
        pytest.param(TensorRTBackend(), id="tensorrt"),
    ],
)  # make one test at least for jit and aot backend
def test_tune_save_load_whole_model(model_factory, sample, backend, checkpoint):
    """Test tuning, saving and loading a whole model wrapped in a single Module."""

    def wrap_whole_model(model):
        return Module(model, "demo-simple", strategy=OneBackendStrategy(backend))

    _tune_save_load_helper(model_factory, sample, "top_model_test.ait", wrap_whole_model, checkpoint)


@requires_cuda
def test_tune_save_load_whole_model_with_device_map(model_factory, sample, torch_device, checkpoint):
    """Test tuning, saving and loading a whole model wrapped in a single Module."""

    def wrap_whole_model(model):
        return Module(model, "demo-simple", strategy=OneBackendStrategy(TorchInductorJitBackend()))

    _tune_save_load_helper(
        model_factory,
        sample,
        "top_model_test.ait",
        wrap_whole_model,
        checkpoint,
        device_map={"": torch_device},
    )


@requires_cuda
def test_tune_save_load_whole_model_with_invalid_device_map(model_factory, sample, torch_device, checkpoint):
    """Test tuning, saving and loading a whole model wrapped in a single Module."""

    def wrap_whole_model(model):
        return Module(model, "demo-simple", strategy=OneBackendStrategy(TorchInductorJitBackend()))

    with pytest.raises(ValueError, match="Some modules in the device_map were not found: .*"):
        _tune_save_load_helper(
            model_factory,
            sample,
            "top_model_test.ait",
            wrap_whole_model,
            checkpoint,
            device_map={"linear123": torch_device},
        )


@requires_cuda
def test_tune_save_load_part_of_model(model_factory, sample, checkpoint):
    """Test tuning, saving and loading a model with individual layers wrapped in Modules."""

    def wrap_partial_model(model):
        model.linear1 = Module(model.linear1, "demo-simple1", strategy=OneBackendStrategy(TorchInductorJitBackend()))
        model.linear2 = Module(model.linear2, "demo-simple2", strategy=OneBackendStrategy(TorchInductorJitBackend()))
        return model

    _tune_save_load_helper(model_factory, sample, "partial_model_test.ait", wrap_partial_model, checkpoint)


@requires_cuda
def test_tune_save_load_part_of_model_with_full_device_map(model_factory, sample, torch_device, checkpoint):
    """Test tuning, saving and loading a model with individual layers wrapped in Modules."""

    def wrap_partial_model(model):
        model.linear1 = Module(model.linear1, "demo-simple1", strategy=OneBackendStrategy(TorchInductorJitBackend()))
        model.linear2 = Module(model.linear2, "demo-simple2", strategy=OneBackendStrategy(TorchInductorJitBackend()))
        return model

    _tune_save_load_helper(
        model_factory,
        sample,
        "partial_model_test.ait",
        wrap_partial_model,
        checkpoint,
        device_map={"linear1": torch_device, "linear2": torch_device},
    )


@requires_cuda
def test_tune_save_load_part_of_model_with_partial_device_map(model_factory, sample, torch_device, checkpoint):
    """Test tuning, saving and loading a model with individual layers wrapped in Modules."""

    def wrap_partial_model(model):
        model.linear1 = Module(model.linear1, "demo-simple1", strategy=OneBackendStrategy(TorchInductorJitBackend()))
        model.linear2 = Module(model.linear2, "demo-simple2", strategy=OneBackendStrategy(TorchInductorJitBackend()))
        return model

    _tune_save_load_helper(
        model_factory,
        sample,
        "partial_model_test.ait",
        wrap_partial_model,
        checkpoint,
        device_map={"linear1": torch_device},
    )


@requires_cuda
def test_tune_save_load_part_of_model_with_invalid_device_map(model_factory, sample, torch_device, checkpoint):
    """Test tuning, saving and loading a model with individual layers wrapped in Modules."""

    def wrap_partial_model(model):
        model.linear1 = Module(model.linear1, "demo-simple1", strategy=OneBackendStrategy(TorchInductorJitBackend()))
        model.linear2 = Module(model.linear2, "demo-simple2", strategy=OneBackendStrategy(TorchInductorJitBackend()))
        return model

    with pytest.raises(ValueError, match="Some modules in the device_map were not found: .*"):
        _tune_save_load_helper(
            model_factory,
            sample,
            "partial_model_test.ait",
            wrap_partial_model,
            checkpoint,
            device_map={"linear123": torch_device},
        )


@requires_cuda
def test_tune_save_load_pipeline(pipeline_factory, sample, checkpoint):
    """Test tuning, saving and loading a pipeline."""

    def wrap_pipeline(pipeline):
        pipeline.linear1 = Module(
            pipeline.linear1, "demo-simple3", strategy=OneBackendStrategy(TorchInductorJitBackend())
        )
        pipeline.linear2 = Module(
            pipeline.linear2, "demo-simple4", strategy=OneBackendStrategy(TorchInductorJitBackend())
        )
        return pipeline

    _tune_save_load_helper(pipeline_factory, sample, "pipeline_test.ait", wrap_pipeline, checkpoint)


@requires_cuda
def test_tune_save_load_pipeline_with_full_device_map(pipeline_factory, sample, torch_device, checkpoint):
    """Test tuning, saving and loading a pipeline."""

    def wrap_pipeline(pipeline):
        pipeline.linear1 = Module(
            pipeline.linear1, "demo-simple3", strategy=OneBackendStrategy(TorchInductorJitBackend())
        )
        pipeline.linear2 = Module(
            pipeline.linear2, "demo-simple4", strategy=OneBackendStrategy(TorchInductorJitBackend())
        )
        return pipeline

    _tune_save_load_helper(
        pipeline_factory,
        sample,
        "pipeline_test.ait",
        wrap_pipeline,
        checkpoint,
        device_map={"linear1": torch_device, "linear2": torch_device},
    )


@requires_cuda
def test_tune_save_load_pipeline_with_partial_device_map(pipeline_factory, sample, torch_device, checkpoint):
    """Test tuning, saving and loading a pipeline."""

    def wrap_pipeline(pipeline):
        pipeline.linear1 = Module(
            pipeline.linear1, "demo-simple3", strategy=OneBackendStrategy(TorchInductorJitBackend())
        )
        pipeline.linear2 = Module(
            pipeline.linear2, "demo-simple4", strategy=OneBackendStrategy(TorchInductorJitBackend())
        )
        return pipeline

    _tune_save_load_helper(
        pipeline_factory,
        sample,
        "pipeline_test.ait",
        wrap_pipeline,
        checkpoint,
        device_map={"linear1": torch_device},
    )


@requires_cuda
def test_tune_save_load_pipeline_with_invalid_device_map(pipeline_factory, sample, torch_device, checkpoint):
    """Test tuning, saving and loading a pipeline."""

    def wrap_pipeline(pipeline):
        pipeline.linear1 = Module(
            pipeline.linear1, "demo-simple3", strategy=OneBackendStrategy(TorchInductorJitBackend())
        )
        pipeline.linear2 = Module(
            pipeline.linear2, "demo-simple4", strategy=OneBackendStrategy(TorchInductorJitBackend())
        )
        return pipeline

    with pytest.raises(ValueError, match="Some modules in the device_map were not found: .*"):
        _tune_save_load_helper(
            pipeline_factory,
            sample,
            "pipeline_test.ait",
            wrap_pipeline,
            checkpoint,
            device_map={"linear123": torch_device},
        )


@requires_cuda
def test_tune_save_load_complex_pipeline(complex_pipeline_factory, sample, torch_device, checkpoint):
    """Test tuning, saving and loading a pipeline."""
    pipeline = complex_pipeline_factory()
    test_data = sample.repeat(10, 1)  # make a bs=10
    with torch.no_grad():
        expected = pipeline(test_data)

    pipeline.net.linear1 = Module(
        pipeline.net.linear1, "demo-simple3", strategy=OneBackendStrategy(TorchInductorJitBackend())
    )
    pipeline.net.linear2 = Module(
        pipeline.net.linear2, "demo-simple4", strategy=OneBackendStrategy(TorchInductorJitBackend())
    )

    tune(pipeline, sample, batch_sizes=[1, 2], dry_run=False, disable_external_logging=False)

    checkpoint.save(pipeline.net.linear1, "linear1_test.ait")
    checkpoint.save(pipeline.net.linear2, "linear2_test.ait")

    # create a new model instance and load the tuned model
    pipeline = complex_pipeline_factory()
    pipeline.net.linear1 = checkpoint.load(pipeline.net.linear1, "linear1_test.ait", device_map={"": torch_device})
    pipeline.net.linear2 = checkpoint.load(pipeline.net.linear2, "linear2_test.ait", device_map={"": torch_device})

    with torch.no_grad():
        preds = pipeline(test_data)
    torch.testing.assert_close(preds, expected, rtol=1e-4, atol=1e-4)


@requires_cuda
def test_tune_save_load_complex_pipeline_with_device_map(complex_pipeline_factory, sample, torch_device, checkpoint):
    """Test tuning, saving and loading a pipeline."""
    pipeline = complex_pipeline_factory()
    test_data = sample.repeat(10, 1)  # make a bs=10
    with torch.no_grad():
        expected = pipeline(test_data)

    pipeline.net.linear1 = Module(
        pipeline.net.linear1, "demo-simple3", strategy=OneBackendStrategy(TorchInductorJitBackend())
    )
    pipeline.net.linear2 = Module(
        pipeline.net.linear2, "demo-simple4", strategy=OneBackendStrategy(TorchInductorJitBackend())
    )

    tune(pipeline, sample, batch_sizes=[1, 2], dry_run=False, disable_external_logging=False)

    checkpoint.save(pipeline.net.linear1, "linear1_test.ait")
    checkpoint.save(pipeline.net.linear2, "linear2_test.ait")

    # create a new model instance and load the tuned model
    pipeline = complex_pipeline_factory()
    pipeline.net.linear1 = checkpoint.load(pipeline.net.linear1, "linear1_test.ait", device_map={})
    pipeline.net.linear2 = checkpoint.load(pipeline.net.linear2, "linear2_test.ait", device_map={})

    with torch.no_grad():
        preds = pipeline(test_data)
    torch.testing.assert_close(preds, expected, rtol=1e-4, atol=1e-4)


@requires_cuda
def test_get_pipeline_modules(pipeline_factory):
    pipeline = pipeline_factory()
    modules = TorchCheckpoint.get_pipeline_modules(pipeline)

    # Check that only torch modules are included
    assert set(modules.keys()) == {"linear1", "linear2"}
    assert len(modules) == 2
    assert isinstance(modules["linear1"], torch.nn.Linear)
    assert isinstance(modules["linear2"], torch.nn.Linear)


@requires_cuda
def test_get_pipeline_modules_empty():
    class EmptyPipeline:
        def __init__(self):
            self.non_module = "not a module"
            self.none_value = None

    modules = TorchCheckpoint.get_pipeline_modules(EmptyPipeline())

    assert len(modules) == 0
    assert isinstance(modules, dict)


@requires_cuda
def test_state_dict_from_pipeline(pipeline_factory):
    pipeline = pipeline_factory()
    pipeline.linear1 = Mock(spec=Module)
    # second module is still torch.nn.Module i.e. not tuned
    state_dict = TorchCheckpoint.state_dict_from_pipeline(pipeline)
    assert isinstance(state_dict, dict)
    assert len(state_dict) == 1
    assert "linear1" in state_dict


@requires_cuda
def test_load_state_dict_for_pipeline(pipeline_factory):
    pipeline = pipeline_factory()
    pipeline.linear1 = Mock(spec=Module)
    # second module is still torch.nn.Module i.e. not tuned
    state_dict = TorchCheckpoint.state_dict_from_pipeline(pipeline)
    assert isinstance(state_dict, dict)
    assert len(state_dict) == 1


def test_state_dict_from_pipeline_preserves_direct_passthrough_wrapper_state():
    """Pipeline save treats direct passthrough wrappers as original modules."""

    class Pipeline:
        pass

    pipeline = Pipeline()
    pipeline.failed = Module(torch.nn.Linear(2, 2), "failed")
    expected_state_dict = {
        name: value.detach().clone() for name, value in pipeline.failed.__wrapped__.state_dict().items()
    }
    pipeline.failed.enable_passthrough()

    state_dict = TorchCheckpoint.state_dict_from_pipeline(pipeline)

    assert list(state_dict.keys()) == ["failed"]
    assert list(state_dict["failed"].keys()) == ["weight", "bias"]
    for name, value in state_dict["failed"].items():
        torch.testing.assert_close(value, expected_state_dict[name])


@requires_cuda
def test_state_dict_from_pipeline_no_tuned_modules(pipeline_factory):
    """Test that ValueError is raised when trying to get state dict from a pipeline with no tuned modules."""
    pipeline = pipeline_factory()
    # Both modules are regular torch.nn.Module, not wrapped in Module
    with pytest.raises(ValueError, match="No tuned modules found in the pipeline"):
        TorchCheckpoint.state_dict_from_pipeline(pipeline)
