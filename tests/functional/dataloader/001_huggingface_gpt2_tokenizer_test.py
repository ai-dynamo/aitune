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
# /// script
# scope = "nightly"
# ///

from pathlib import Path

import datasets
from transformers import AutoTokenizer

from aitune.torch.dataloader import DataLoaderFactory

PROMPTS_PATH = Path(__file__).parent.parent.parent / "fixtures/chatgpt_prompts_100.json"


def test_huggingface_dataset():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return {
            "input_ids": tokenizer(
                examples["prompt"],  # filter just text column
                padding="max_length",  # Pad to max_length
                truncation=True,  # Truncate if needed
                max_length=128,  # Set max sequence length
                return_tensors="pt",  # Return PyTorch tensors
            )["input_ids"].squeeze(0)  # Note: NOT COOL!
        }

    # Load from local fixture instead of HuggingFace API
    dataset = datasets.load_dataset("json", data_files=str(PROMPTS_PATH), split="train").map(
        tokenize_function, remove_columns=["prompt", "act"]
    )

    dataloader = DataLoaderFactory(dataset).create_dataloader(batch_size=4)
    samples = list(dataloader)

    assert len(samples) == 25

    kwargs = samples[0]

    assert len(kwargs) == 1

    assert len(kwargs["input_ids"]) == 4

    assert kwargs["input_ids"].shape == (4, 128)  # Batch size 4, sequence length 128


if __name__ == "__main__":
    test_huggingface_dataset()
