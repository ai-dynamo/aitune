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
"""Tune ESM2 model."""

import logging

import torch
from transformers import AutoTokenizer, EsmForMaskedLM

import aitune.torch as ait
from aitune.torch.backend import TensorRTBackend, TensorRTBackendConfig, TorchEagerBackend, TorchInductorBackend

DEVICE = torch.device("cuda")
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
LOG_LEVEL = "INFO"
SAMPLE_SEQUENCE = "MQIFVKTLTGKTITLEVEPS<mask>TIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"

logger = logging.getLogger(__name__)


def get_model(model_name: str = MODEL_NAME, device: torch.device = DEVICE):
    logger.info("Loading model '%s' on %s...", model_name, device)
    model = EsmForMaskedLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model


def prepare_sample(device=DEVICE):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    input_data = tokenizer([SAMPLE_SEQUENCE], return_tensors="pt")
    return {
        "input_ids": input_data["input_ids"].squeeze(0).to(device),
        "attention_mask": input_data["attention_mask"].squeeze(0).to(device),
    }


def tune(
    model_path: str = "esm2_tuned",
    model_name: str = MODEL_NAME,
    device: torch.device = DEVICE,
    batch_sizes: list[int] | None = None,
):
    logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", force=True)
    logger.info("Tuning ESM2 model...")
    with torch.no_grad():
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16, 32]

        model = get_model(model_name=model_name, device=device)
        input_data = prepare_sample(device=device)

        logger.info("Inspecting model...")
        modules_info = ait.inspect(model, [input_data], number_of_iterations=1, warmup_iterations=1)
        modules_info.describe()

        strategy = ait.FirstWinsStrategy(
            backends=[
                TensorRTBackend(),
                TensorRTBackend(TensorRTBackendConfig(use_dynamo=False)),
                TorchInductorBackend(),
                TorchEagerBackend(),
            ]
        )
        strategy.enable_find_max_batch_size(enable=False)

        logger.info("Wrapping modules...")
        model = ait.wrap(model, modules_info.get_modules(), strategy=strategy)

        logger.info("Tuning model...")
        ait.tune(model, [input_data], batch_sizes=batch_sizes)

        logger.info("Saving model...")
        ait.save(model, model_path)


if __name__ == "__main__":
    tune()
