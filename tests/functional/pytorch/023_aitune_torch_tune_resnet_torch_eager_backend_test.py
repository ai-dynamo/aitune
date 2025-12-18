# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

# /// script
# dependencies = ["timm"]
# scope = "always"
# allow_failure = true
# ///


from logging import DEBUG, basicConfig, getLogger

import timm
import torch

from aitune.torch import tune
from aitune.torch.backend.torch_eager import TorchEagerBackend, TorchEagerBackendConfig
from aitune.torch.module.wrapper_module import Module
from aitune.torch.module_registry import MODULE_REGISTRY
from aitune.torch.tune_strategy.one_backend_strategy import OneBackendStrategy

logger = getLogger(__name__)


def do_test(backend: TorchEagerBackend):
    # given
    device = torch.device("cuda")

    model = timm.create_model("resnet18", pretrained=False)
    model.to(device)
    model.eval()
    data = torch.randn((3, 224, 224), device=device)

    with torch.no_grad():
        out = model(data.unsqueeze(0))
    expected_probs = torch.nn.functional.softmax(out[0], dim=0)

    module = Module(model, "functional-resnet18", strategy=OneBackendStrategy(backend))
    # when
    tune(module, data, batch_sizes=[2, 1], dry_run=False, disable_external_logging=False)
    # then - verify tuning
    out = module(data.unsqueeze(0))
    actual_probs = torch.nn.functional.softmax(out[0], dim=0)
    torch.testing.assert_close(actual_probs, expected_probs, rtol=1e-4, atol=1e-5)


def test_tune_resnet_torch_eager():
    errors = []

    try:
        logger.info("Testing eager")
        config = TorchEagerBackendConfig()
        do_test(TorchEagerBackend(config=config))
    except Exception as e:
        logger.error("Error with eager: %s", e)
        errors.append(f"Error with eager: {e}")
    finally:
        MODULE_REGISTRY.clear()

    if errors:
        raise RuntimeError("There were some errors:\n" + "\n".join(errors))


def test_tune_resnet_torch_eager_autocast():
    errors = []

    dtypes = [torch.float16, torch.bfloat16]
    for dtype in dtypes:
        try:
            logger.info("Testing autocast dtype: %s", dtype)
            config = TorchEagerBackendConfig(autocast_enabled=True, autocast_dtype=dtype)
            do_test(TorchEagerBackend(config=config))
        except Exception as e:
            logger.error("Error with autocast and dtype %s: %s", dtype, e)
            errors.append(f"Error with autocast and dtype {dtype}: {e}")
        finally:
            MODULE_REGISTRY.clear()

    if errors:
        raise RuntimeError("There were some errors:\n" + "\n".join(errors))


if __name__ == "__main__":
    basicConfig(level=DEBUG, force=True)
    test_tune_resnet_torch_eager()
    test_tune_resnet_torch_eager_autocast()
