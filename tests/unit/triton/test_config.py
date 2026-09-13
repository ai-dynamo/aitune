# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from google.protobuf import text_format
from pydantic import ValidationError
from tritonclient.grpc import model_config_pb2 as pb

from aitune.triton import (
    DynamicBatcher,
    ExecutionAccelerator,
    InstanceGroup,
    ModelWarmup,
    ONNXRuntimeModelConfig,
    QueuePolicy,
    SequenceBatcher,
    TensorRTModelConfig,
    TorchAOTIModelConfig,
)
from aitune.triton.config import TritonDataType, TritonTensorConfig


def tensor(**kwargs):
    return TritonTensorConfig(**{"name": "x", "data_type": "TYPE_FP32", "dims": (4,)} | kwargs)


def config(backend="onnx", **kwargs):
    options = {"name": "model", "max_batch_size": 8, "inputs": (tensor(),), "outputs": (tensor(name="y"),)} | kwargs
    if backend == "trt":
        return TensorRTModelConfig(**{"optimization_profile_indices": (0,)} | options)
    if backend == "pt2":
        return TorchAOTIModelConfig(**{"structured_call": False} | options)
    return ONNXRuntimeModelConfig(**{"execution_provider": "cuda"} | options)


def roundtrip(model):
    message = text_format.Parse(model.to_pbtxt(), pb.ModelConfig())
    assert message == model.to_protobuf()
    return message


@pytest.mark.parametrize(
    "backend,platform", [("onnx", "onnxruntime_onnx"), ("trt", "tensorrt_plan"), ("pt2", "torch_aoti")]
)
@pytest.mark.parametrize("batched", [False, True])
def test_backend_roundtrip(backend, platform, batched):
    result = roundtrip(config(backend, max_batch_size=8 if batched else 0, dynamic_batching=batched))
    assert result.platform == platform
    assert result.max_batch_size == (8 if batched else 0)
    assert result.HasField("dynamic_batching") == batched
    assert result.input[0].name == "x"
    assert result.output[0].dims == [4]


@pytest.mark.parametrize("dtype", list(TritonDataType))
def test_tensor_datatypes(dtype):
    result = roundtrip(config(inputs=(tensor(data_type=dtype),)))
    assert result.input[0].data_type == pb.DataType.Value(dtype.value)


def test_tensor_options():
    result = roundtrip(
        config(
            "trt",
            inputs=(
                tensor(
                    dims=(-1,),
                    reshape=(),
                    optional=True,
                    format="FORMAT_NHWC",
                    allow_ragged_batch=True,
                    is_shape_tensor=True,
                    is_non_linear_format_io=True,
                ),
            ),
            outputs=(tensor(name="y", reshape=(2, 2), label_filename="labels.txt", is_shape_tensor=True),),
        )
    )
    assert result.input[0].HasField("reshape")
    assert list(result.input[0].reshape.shape) == []
    assert result.input[0].optional
    assert result.input[0].allow_ragged_batch
    assert result.input[0].format == pb.ModelInput.FORMAT_NHWC
    assert result.input[0].is_shape_tensor
    assert result.input[0].is_non_linear_format_io
    assert result.output[0].reshape.shape == [2, 2]
    assert result.output[0].label_filename == "labels.txt"


def test_dynamic_batching_policies():
    result = roundtrip(
        config(
            dynamic_batching=DynamicBatcher(
                preferred_batch_size=(2, 8),
                max_queue_delay_microseconds=123,
                preserve_ordering=True,
                priority_levels=3,
                default_priority_level=2,
                default_queue_policy=QueuePolicy(
                    timeout_action="DELAY",
                    max_queue_size=17,
                    default_timeout_microseconds=456,
                    allow_timeout_override=True,
                ),
                priority_queue_policy={1: QueuePolicy(max_queue_size=9)},
            )
        )
    )
    batcher = result.dynamic_batching
    assert batcher.preferred_batch_size == [2, 8]
    assert batcher.max_queue_delay_microseconds == 123
    assert batcher.preserve_ordering
    assert batcher.priority_levels == 3
    assert batcher.default_priority_level == 2
    assert batcher.default_queue_policy.timeout_action == pb.ModelQueuePolicy.DELAY
    assert batcher.default_queue_policy.default_timeout_microseconds == 456
    assert batcher.default_queue_policy.allow_timeout_override
    assert batcher.default_queue_policy.max_queue_size == 17
    assert batcher.priority_queue_policy[1].max_queue_size == 9


@pytest.mark.parametrize("strategy", ["direct", "oldest"])
def test_sequence_batching(strategy):
    result = roundtrip(
        config(
            sequence_batching=SequenceBatcher(**{
                strategy: {"max_queue_delay_microseconds": 42},
                "max_sequence_idle_microseconds": 1000,
                "control_input": (
                    {"name": "START", "control": [{"kind": "CONTROL_SEQUENCE_START", "int32_false_true": [0, 1]}]},
                ),
                "state": (
                    {
                        "input_name": "in_state",
                        "output_name": "out_state",
                        "data_type": "TYPE_FP32",
                        "dims": [4],
                        "initial_state": [
                            {"name": "initial", "data_type": "TYPE_FP32", "dims": [4], "data_file": "state.bin"}
                        ],
                    },
                ),
            })
        )
    )
    batcher = result.sequence_batching
    assert batcher.WhichOneof("strategy_choice") == strategy
    assert getattr(batcher, strategy).max_queue_delay_microseconds == 42
    assert batcher.control_input[0].control[0].int32_false_true == [0, 1]
    assert batcher.state[0].initial_state[0].data_file == "state.bin"
    assert batcher.max_sequence_idle_microseconds == 1000


@pytest.mark.parametrize("enabled", [False, True])
def test_common_options_and_explicit_false(enabled):
    result = roundtrip(
        config(
            parameters={"key": "value"},
            response_cache=enabled,
            decoupled=enabled,
            default_model_filename="custom.onnx",
            version_policy={"specific": {"versions": [1, 3]}},
            optimization={"input_pinned_memory": {"enable": False}},
            metric_tags={"team": "serving"},
            instance_groups=(
                InstanceGroup(
                    kind="KIND_CPU",
                    count=2,
                    name="cpu",
                    host_policy="host",
                    rate_limiter={"priority": 2, "resources": [{"name": "resource", "count": 1}]},
                ),
            ),
            warmup=(
                ModelWarmup(name="zeros", inputs={"x": {"data_type": "TYPE_FP32", "dims": [4], "zero_data": False}}),
            ),
        )
    )
    assert result.parameters["key"].string_value == "value"
    assert result.HasField("response_cache") and result.response_cache.enable == enabled
    assert result.HasField("model_transaction_policy") and result.model_transaction_policy.decoupled == enabled
    assert result.default_model_filename == "custom.onnx"
    assert result.version_policy.specific.versions == [1, 3]
    assert result.optimization.HasField("input_pinned_memory")
    assert not result.optimization.input_pinned_memory.enable
    assert result.metric_tags["team"] == "serving"
    assert result.instance_group[0].count == 2
    assert result.instance_group[0].host_policy == "host"
    assert result.instance_group[0].rate_limiter.resources[0].name == "resource"
    assert result.model_warmup[0].inputs["x"].WhichOneof("input_data_type") == "zero_data"


def test_tensorrt_profile_selection_and_optimization():
    result = roundtrip(
        config(
            "trt",
            optimization_profile_indices=(1, 3),
            cuda_graphs=True,
            eager_batching=True,
            gather_kernel_buffer_threshold=16,
            instance_groups=(InstanceGroup(kind="KIND_GPU", gpus=(1,), profile=("3",)),),
        )
    )
    assert len(result.instance_group) == 1
    assert result.instance_group[0].profile == ["3"]
    assert result.instance_group[0].gpus == [1]
    assert result.optimization.cuda.graphs
    assert result.optimization.eager_batching
    assert result.optimization.gather_kernel_buffer_threshold == 16


def test_onnx_provider_from_string_enables_tensorrt():
    result = roundtrip(config(execution_provider="".join(("tensor", "rt"))))
    assert [item.name for item in result.optimization.execution_accelerators.gpu_execution_accelerator] == ["tensorrt"]


def test_onnx_accelerators_preserve_parameters():
    result = roundtrip(
        config(
            execution_provider="tensorrt",
            gpu_execution_accelerators=(ExecutionAccelerator(name="tensorrt", parameters={"precision_mode": "FP16"}),),
            cpu_execution_accelerators=(ExecutionAccelerator(name="openvino", parameters={"num_of_threads": "2"}),),
        )
    )
    accelerators = result.optimization.execution_accelerators
    assert len(accelerators.gpu_execution_accelerator) == 1
    assert accelerators.gpu_execution_accelerator[0].parameters["precision_mode"] == "FP16"
    assert accelerators.cpu_execution_accelerator[0].name == "openvino"
    assert accelerators.cpu_execution_accelerator[0].parameters["num_of_threads"] == "2"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dims": ()},
        {"dims": (0,)},
        {"dims": (-2,)},
        {"reshape": (-2,)},
        {"reshape": (-1, -1)},
        {"name": ""},
        {"data_type": "TYPE_INVALID"},
        {"format": "NHWC"},
        {"unknown": True},
    ],
)
def test_invalid_tensor_options(kwargs):
    with pytest.raises(ValidationError):
        tensor(**kwargs)


@pytest.mark.parametrize(
    "factory,kwargs",
    [
        (QueuePolicy, {"max_queue_size": -1}),
        (QueuePolicy, {"timeout_action": "DROP"}),
        (DynamicBatcher, {"preferred_batch_size": (0,)}),
        (DynamicBatcher, {"max_queue_delay_microseconds": -1}),
        (DynamicBatcher, {"priority_levels": 2}),
        (DynamicBatcher, {"default_priority_level": 1}),
        (DynamicBatcher, {"priority_levels": 2, "default_priority_level": 3}),
        (DynamicBatcher, {"priority_levels": 2, "default_priority_level": 1, "priority_queue_policy": {3: {}}}),
        (SequenceBatcher, {"direct": {}, "oldest": {}}),
        (InstanceGroup, {"count": 0}),
        (InstanceGroup, {"gpus": (-1,)}),
        (InstanceGroup, {"gpus": (0, 0)}),
        (InstanceGroup, {"kind": "KIND_CPU", "gpus": (0,)}),
        (ModelWarmup, {"name": "warm", "inputs": {}}),
        (ModelWarmup, {"name": "warm", "batch_size": 0, "inputs": {"x": {}}}),
        (ExecutionAccelerator, {"name": ""}),
        (ExecutionAccelerator, {"name": "tensorrt", "parameters": {"key": 1}}),
    ],
)
def test_invalid_helper_options(factory, kwargs):
    with pytest.raises(ValidationError):
        factory(**kwargs)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"max_batch_size": -1}, "greater than or equal"),
        ({"max_batch_size": 0, "dynamic_batching": True}, "positive max_batch_size"),
        ({"dynamic_batching": True, "sequence_batching": SequenceBatcher()}, "mutually exclusive"),
        ({"dynamic_batching": DynamicBatcher(preferred_batch_size=(9,))}, "cannot exceed"),
        ({"outputs": (tensor(optional=True),)}, "input-only"),
        ({"inputs": (tensor(label_filename="labels.txt"),)}, "output-only"),
        ({"instance_groups": (InstanceGroup(profile=("0",)),)}, "only supported by TensorRT"),
        ({"optimization": {"unknown": True}}, "no field named"),
        ({"version_policy": {"latest": {}, "all": {}}}, "oneof"),
        ({"sequence_batching": SequenceBatcher(direct={"unknown": True})}, "no field named"),
        ({"warmup": (ModelWarmup(name="w", inputs={"x": {"zero_data": True, "random_data": True}}),)}, "oneof"),
        ({"unknown": True}, "Extra inputs"),
    ],
)
def test_invalid_model_options(kwargs, match):
    with pytest.raises(ValidationError, match=match):
        config(**kwargs)


@pytest.mark.parametrize(
    "filename", ["", ".", "..", "../model.onnx", "/model.onnx", "sub/model.onnx", "sub\\model.onnx"]
)
def test_invalid_default_filename(filename):
    with pytest.raises(ValidationError, match="single filename"):
        config(default_model_filename=filename)


@pytest.mark.parametrize("profiles", [(), (-1,), (0, 0)])
def test_invalid_tensorrt_profiles(profiles):
    with pytest.raises(ValidationError, match="profiles"):
        config("trt", optimization_profile_indices=profiles)


def test_backend_constraints():
    with pytest.raises(ValidationError, match="structured"):
        config("pt2", structured_call=True)
    with pytest.raises(ValidationError, match="KIND_GPU"):
        config("trt", instance_groups=(InstanceGroup(kind="KIND_CPU"),))
    with pytest.raises(ValidationError, match="included"):
        config("trt", instance_groups=(InstanceGroup(profile=("1",)),))
    with pytest.raises(ValidationError, match="Duplicate execution accelerator"):
        config(gpu_execution_accelerators=(ExecutionAccelerator(name="tensorrt"),) * 2)


def test_warmup_limits():
    warmup = ModelWarmup(
        name="warm", batch_size=2, inputs={"x": {"data_type": "TYPE_FP32", "dims": [4], "zero_data": True}}
    )
    with pytest.raises(ValidationError, match="unique"):
        config(warmup=(warmup, warmup))
    with pytest.raises(ValidationError, match="batch limit"):
        config(max_batch_size=0, warmup=(warmup,))


def test_batch_io_mappings():
    result = roundtrip(
        config(
            batch_input=(
                {
                    "kind": "BATCH_ELEMENT_COUNT",
                    "target_name": ["counts"],
                    "data_type": "TYPE_INT32",
                    "source_input": ["x"],
                },
            ),
            batch_output=({"kind": "BATCH_SCATTER_WITH_INPUT_SHAPE", "target_name": ["y"], "source_input": ["x"]},),
        )
    )
    assert result.batch_input[0].kind == pb.BatchInput.BATCH_ELEMENT_COUNT
    assert result.batch_input[0].target_name == ["counts"]
    assert result.batch_input[0].source_input == ["x"]
    assert result.batch_output[0].kind == pb.BatchOutput.BATCH_SCATTER_WITH_INPUT_SHAPE
    assert result.batch_output[0].target_name == ["y"]


@pytest.mark.parametrize("field", ["batch_input", "batch_output"])
def test_batch_io_rejects_unknown_enum(field):
    with pytest.raises(ValidationError, match="Invalid enum value"):
        config(**{field: ({"kind": "UNKNOWN"},)})


@pytest.mark.parametrize("backend", ["onnx", "trt", "pt2"])
def test_default_optional_messages_absent(backend):
    result = roundtrip(config(backend))
    for field in ("response_cache", "model_transaction_policy", "sequence_batching", "version_policy"):
        assert not result.HasField(field)


def test_accelerator_duplicate_between_mapping_and_helper():
    with pytest.raises(ValidationError, match="Duplicate execution accelerator"):
        config(
            optimization={"execution_accelerators": {"gpu_execution_accelerator": [{"name": "tensorrt"}]}},
            gpu_execution_accelerators=(ExecutionAccelerator(name="tensorrt"),),
        )
