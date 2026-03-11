# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ONNXQuantizer."""

import shutil
from pathlib import Path

from aitune.torch.backend.tensorrt.onnx_quantization import ONNXQuantizer
from tests.toy_models.onnx_models import ToyOnnxModel


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
