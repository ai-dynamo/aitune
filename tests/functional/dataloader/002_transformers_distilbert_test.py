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

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.data.data_collator import DataCollatorWithPadding

from aitune.torch.dataloader import DataLoaderFactory


def get_dataloader():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    max_sequence_length = 128

    def preprocess_function(examples):
        return tokenizer(examples["prompt"], truncation=True, max_length=max_sequence_length)

    dataset = load_dataset("fka/awesome-chatgpt-prompts", split="train[:99]").map(
        preprocess_function, remove_columns=["act", "prompt"]
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
