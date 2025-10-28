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

from pathlib import Path

import datasets
from transformers import AutoTokenizer
from transformers.data.data_collator import DataCollatorWithPadding

from aitune.torch.dataloader import DataLoaderFactory

PROMPTS_PATH = Path(__file__).parent.parent.parent / "fixtures/chatgpt_prompts_100.json"


def get_dataloader():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    max_sequence_length = 128

    def tokenize_function(examples):
        return tokenizer(examples["prompt"], truncation=True, max_length=max_sequence_length)

    dataset = datasets.load_dataset("json", data_files=str(PROMPTS_PATH), split="train[:99]").map(
        tokenize_function, remove_columns=["prompt", "act"]
    )

    return DataLoaderFactory(
        dataset, collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, max_length=max_sequence_length)
    ).create_dataloader(batch_size=4)


def test_transformers_distilbert():
    dataloader = get_dataloader()
    data = list(dataloader)

    assert len(data) == 24
    assert data[0]["input_ids"].shape == (4, 128)
    assert data[0]["attention_mask"].shape == (4, 128)


if __name__ == "__main__":
    test_transformers_distilbert()
