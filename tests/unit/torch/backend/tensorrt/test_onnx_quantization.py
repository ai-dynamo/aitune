# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for ONNXQuantizer."""

import shutil
from pathlib import Path

from aitune.torch.backend.tensorrt.onnx_quantization import ONNXQuantizer
from tests.toy_models.onnx_models import ToyOnnxModel
from tests.toy_models.torch_models import ToyTorchModel


def test_apply_model_opt_post_processing(tmp_path):
    """Test the application of ModelOpt post-processing optimizations to an ONNX model."""
    model = ToyOnnxModel()
    onnx_copy_path = Path(tmp_path) / model.path.name
    shutil.copy(model.path, onnx_copy_path)
    quantizer = ONNXQuantizer()

    quantizer.apply_model_opt_post_processing(onnx_copy_path)

    assert onnx_copy_path.exists()
    assert onnx_copy_path.is_file()
    assert onnx_copy_path.stat().st_size > 0


def test_prepare_calibration_data():
    quantizer = ONNXQuantizer()
    torch_model = ToyTorchModel()

    cpu_samples = torch_model.samples()
    graph_spec = torch_model.graph_spec()

    calibration_samples = quantizer._prepare_calibration_data(
        data=cpu_samples,
        graph_spec=graph_spec,
    )

    assert calibration_samples[0]["args_0"][0].shape == cpu_samples[0][0][0][0].shape
    assert calibration_samples[0]["args_0"][1].shape == cpu_samples[0][0][0][1].shape
    assert len(calibration_samples[0]["args_0"]) == len(cpu_samples[0][0][0]) == 2  # two batched samples.
