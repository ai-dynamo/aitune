# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import datasets
import numpy as np
import pytest
import torch
from transformers import AutoTokenizer
from transformers.data.data_collator import DataCollatorWithPadding

from aitune.torch.dataloader import (
    DataLoaderFactory,
    DatasetLike,
    InputConfig,
    MinMaxRandomDataset,
    ensure_enough_samples,
    samples_generator,
)
from tests.utilities.prompts import PROMPTS_PATH


def simulate_tuning_loop(dataset: DatasetLike, batch_sizes: list[int]):
    """Prepare dataset for tuning.

    Steps:
    1. Create a dataloader from the dataset
    2. Extract the batches with collated samples from the dataloader

    Args:
        dataset: DatasetLike
        batch_sizes: list of batch sizes to use in the tuning loop, for tests it should be a list with a single element

    Returns:
        list of batches to assert in the tests and additional params
    """
    return [*samples_generator(dataset, batch_sizes)]


def test_simple_sequence_map():
    dataset = [{"input": torch.randn(3, 24, 24), "labels": torch.randint(0, 10, (1,))} for _ in range(10)]
    samples = simulate_tuning_loop(dataset, batch_sizes=[4])

    assert len(samples) == 2

    bs, args, kwargs = samples[0]
    assert bs == 4
    assert len(args) == 0
    assert len(kwargs) == 2

    assert kwargs["input"].shape == (4, 3, 24, 24)
    assert kwargs["labels"].shape == (4, 1)


def test_simple_sequence_tensors():
    dataset = [torch.randn(3, 24, 24) for _ in range(10)]
    samples = simulate_tuning_loop(dataset, batch_sizes=[4])

    assert len(samples) == 2

    bs, args, kwargs = samples[0]
    assert bs == 4
    assert len(kwargs) == 0
    assert len(args) == 1
    assert args[0].shape == (4, 3, 24, 24)


def test_simple_sequence_strings():
    dataset = ["random string" for _ in range(10)]
    samples = simulate_tuning_loop(dataset, batch_sizes=[4])

    assert len(samples) == 2

    bs, args, kwargs = samples[0]
    assert bs == 4
    assert len(kwargs) == 0
    assert len(args) == 1
    assert len(args[0]) == 4


def test_simple_mapping_strings():
    dataset = [{"input": "random string"} for _ in range(10)]
    samples = simulate_tuning_loop(dataset, batch_sizes=[4])

    assert len(samples) == 2

    bs, args, kwargs = samples[0]
    assert bs == 4
    assert len(args) == 0
    assert len(kwargs["input"]) == 4


def test_resnet_dataset_synthetic():
    synthetic_dataset = [
        {
            "pixel_values": torch.rand(3, 224, 224),  # Random RGB image tensor
            "labels": torch.randint(0, 1000, (1,)).item(),  # Random label (ImageNet has 1000 classes)
        }
        for _ in range(100)
    ]

    samples = simulate_tuning_loop(synthetic_dataset, batch_sizes=[4])

    assert len(samples) == 25

    bs, args, kwargs = samples[0]
    assert bs == 4
    assert len(args) == 0
    assert len(kwargs) == 2

    assert len(kwargs["pixel_values"]) == 4
    assert kwargs["pixel_values"].shape == (4, 3, 224, 224)
    assert kwargs["labels"].shape == (4,)


def test_dynamic_shapes():
    dataset = MinMaxRandomDataset(
        25,
        [
            InputConfig(min_input=(3, 24, 24), max_input=(3, 48, 48), kwarg_name="input"),
            InputConfig(min_input=(1,), max_input=(1,), kwarg_name="labels"),
        ],
        include_min_max_shapes=False,
    )

    samples = simulate_tuning_loop(dataset, batch_sizes=[4])

    assert len(samples) == 25

    bs, args, kwargs = samples[0]
    assert bs == 4
    assert len(args) == 0
    assert len(kwargs) == 2

    assert len(kwargs["input"]) == 4
    assert kwargs["input"].shape[0] == 4
    assert kwargs["input"].shape[1] == 3
    possible_shapes = list(range(24, 49))
    assert kwargs["input"].shape[2] in possible_shapes
    assert kwargs["input"].shape[3] in possible_shapes
    assert kwargs["labels"].shape[0] == 4


def test_dynamic_shapes_with_min_max_only():
    dataset = MinMaxRandomDataset(
        2,
        [
            InputConfig(min_input=(3, 24, 24), max_input=(3, 48, 48), kwarg_name="input"),
            InputConfig(min_input=(1,), max_input=(1,), kwarg_name="labels"),
        ],
        include_min_max_shapes=True,
    )

    samples = simulate_tuning_loop(dataset, batch_sizes=[4])

    assert len(samples) == 2

    bs, args, kwargs = samples[0]
    assert bs == 4
    assert len(args) == 0
    assert len(kwargs) == 2

    assert len(kwargs["input"]) == 4
    assert kwargs["input"].shape[0] == 4
    assert kwargs["input"].shape[1] == 3
    assert kwargs["input"].shape[2] == 24
    assert kwargs["input"].shape[3] == 24

    assert kwargs["labels"].shape[0] == 4

    bs, args, kwargs = samples[1]
    assert bs == 4
    assert len(args) == 0
    assert len(kwargs) == 2

    assert len(kwargs["input"]) == 4
    assert kwargs["input"].shape[0] == 4
    assert kwargs["input"].shape[1] == 3
    assert kwargs["input"].shape[2] == 48
    assert kwargs["input"].shape[3] == 48


def test_dynamic_shapes_number_of_samples_fails():
    MinMaxRandomDataset(
        1,
        [InputConfig(min_input=(3, 24, 24), max_input=(3, 48, 48), kwarg_name="input")],
        include_min_max_shapes=False,
    )

    with pytest.raises(ValueError, match="num_samples must be at least 2"):
        MinMaxRandomDataset(
            1,
            [InputConfig(min_input=(3, 24, 24), max_input=(3, 48, 48), kwarg_name="input")],
            include_min_max_shapes=True,
        )


def test_dynamic_shapes_fails():
    with pytest.raises(ValueError, match="All input configs must have either a kwarg_name or none"):
        MinMaxRandomDataset(
            2,
            [
                InputConfig(min_input=(3, 24, 24), max_input=(3, 48, 48), kwarg_name="input"),
                InputConfig(min_input=(1,), max_input=(1,)),  # missing kwarg_name
            ],
            include_min_max_shapes=False,
        )


def test_llm_padding_collator():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Add padding token (fix for gpt2)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(examples):
        return tokenizer(
            examples["prompt"],
            truncation=True,
            padding=True,
            max_length=128,  # Add reasonable max length
            return_tensors=None,  # Important: don't return tensors here
        )

    # Load raw dataset first
    dataset = datasets.load_dataset("json", data_files=str(PROMPTS_PATH), split="train[:100]").map(
        tokenize, remove_columns=["prompt", "act"]
    )

    dl_config = DataLoaderFactory(dataset, collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, padding=True))

    samples = simulate_tuning_loop(dl_config, batch_sizes=[4])

    assert len(samples) == 25

    bs, args, kwargs = samples[0]
    assert bs == 4
    assert len(args) == 0
    assert len(kwargs) == 2

    assert len(kwargs["input_ids"]) == 4
    assert len(kwargs["attention_mask"]) == 4

    assert kwargs["input_ids"].shape[0] == 4  # Batch size
    assert kwargs["attention_mask"].shape[0] == 4  # Batch size
    assert kwargs["input_ids"].shape == kwargs["attention_mask"].shape  # Same shape for both tensors


def test_custom_dataloader_shuffled_dataset():
    dataset = [{"input": torch.ones(3, 24, 24) * i, "labels": torch.randint(0, 10, (1,))} for i in range(100)]

    class CustomDataloaderFactory(DataLoaderFactory):
        custom_used = False

        def create_dataloader(self, batch_size):
            self.custom_used = True
            return torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    dataloader = CustomDataloaderFactory(dataset)
    samples = simulate_tuning_loop(dataloader, batch_sizes=[4])

    bs, args, kwargs = samples[0]
    assert bs == 4
    assert len(args) == 0
    assert len(kwargs) == 2
    assert kwargs["input"].shape == (4, 3, 24, 24)
    assert kwargs["labels"].shape == (4, 1)

    assert dataloader.custom_used


def test_torch_stack_dataset():
    # Create example tensors
    tensors = [
        torch.randn(3, 4, 4),  # Simulating image data
        torch.randn(3, 4, 4),
        torch.randn(3, 4, 4),
        torch.randn(3, 4, 4),
    ]

    # Create dataset and dataloader
    dataset = torch.utils.data.StackDataset(tensors)

    samples = [*samples_generator(dataset, batch_sizes=[1, 2, 4], max_num_batches_per_batch_size=1)]

    assert len(samples) == 3

    bs, args, kwargs = samples[0]
    assert bs == 1
    assert len(args) == 1
    assert len(kwargs) == 0
    assert args[0].shape == (1, 3, 4, 4)

    bs, args, kwargs = samples[1]
    assert bs == 2
    assert len(args) == 1
    assert len(kwargs) == 0
    assert args[0].shape == (2, 3, 4, 4)

    bs, args, kwargs = samples[2]
    assert bs == 4
    assert len(args) == 1
    assert len(kwargs) == 0
    assert args[0].shape == (4, 3, 4, 4)


def test_dataset_label_tensor():
    dataset = [{"label": i} for i in range(10)]
    samples = simulate_tuning_loop(dataset, batch_sizes=[4])

    bs, args, kwargs = samples[0]
    assert bs == 4
    assert len(args) == 0
    assert len(kwargs) == 1

    assert "labels" in kwargs
    assert "label" not in kwargs

    assert kwargs["labels"].shape == (4,)
    assert kwargs["labels"].tolist() == [0, 1, 2, 3]


def test_dataset_label_tensor_with_label_ids_tensor():
    dataset = [{"label_ids": torch.randint(0, 10, (1,))} for _ in range(10)]
    samples = simulate_tuning_loop(dataset, batch_sizes=[4])

    bs, args, kwargs = samples[0]
    assert bs == 4
    assert len(args) == 0
    assert len(kwargs) == 1

    assert "labels" in kwargs
    assert "label_ids" not in kwargs


def test_dataset_label_tensor_with_label_ids_ints():
    dataset = [{"label_ids": [i, i, i]} for i in range(10)]
    samples = simulate_tuning_loop(dataset, batch_sizes=[4])

    bs, args, kwargs = samples[0]
    assert bs == 4
    assert len(args) == 0
    assert len(kwargs) == 1

    assert "labels" in kwargs
    assert "label_ids" not in kwargs


def test_single_tensor_dataset():
    dataset = torch.randn(3, 24, 24)

    samples = simulate_tuning_loop(dataset, batch_sizes=[4, 8])

    bs, args, kwargs = samples[0]
    assert bs == 4
    assert len(args) == 1
    assert len(kwargs) == 0
    assert args[0].shape == (4, 3, 24, 24)

    bs, args, kwargs = samples[1]
    assert bs == 8
    assert len(args) == 1
    assert len(kwargs) == 0
    assert args[0].shape == (8, 3, 24, 24)


def test_dataloader_sample_as_dict():
    """
    from datasets import load_dataset

    dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")

    print(dataset[0])
    """
    dataset = [
        {
            "audio": {
                "path": "0d38672e0bbdbdc460af55b8bb84a15b2730db2819f2af64f9c777d4d586f2de",
                "array": np.array([0.00238037, 0.0020752, 0.00198364, 0.00024414, 0.00048828, 0.0005188]),
                "sampling_rate": 16000,
            }
        }
    ]

    samples = simulate_tuning_loop(dataset, batch_sizes=[4])

    bs, args, kwargs = samples[0]
    assert bs == 4
    assert len(args) == 0
    assert len(kwargs) == 1

    assert "audio" in kwargs
    assert "path" in kwargs["audio"]
    assert "array" in kwargs["audio"]
    assert "sampling_rate" in kwargs["audio"]

    assert len(kwargs["audio"]["path"]) == 4
    assert len(kwargs["audio"]["array"]) == 4
    assert len(kwargs["audio"]["sampling_rate"]) == 4


def test_ensure_enough_samples_tensor():
    dataset = torch.ones(3, 24, 24)

    dataset = ensure_enough_samples(dataset, 10)

    assert len(dataset) == 10
    assert all(s.shape == (3, 24, 24) for s in dataset)


def test_ensure_enough_samples_iterable():
    dataset = [{"input": torch.ones(3, 24, 24), "labels": torch.randint(0, 10, (1,))}]

    dataset = ensure_enough_samples(dataset, 10)

    assert len(dataset) == 10

    assert all("input" in s for s in dataset)
    assert all("labels" in s for s in dataset)


def test_ensure_enough_samples_torch_dataset():
    dataset = torch.utils.data.TensorDataset(torch.randn(3, 24, 24))

    dataset = ensure_enough_samples(dataset, 10)

    assert len(dataset) == 10


def test_ensure_enough_samples_dataloader_factory():
    dataset = [{"input": torch.ones(3, 24, 24), "labels": torch.randint(0, 10, (1,))}]
    dataloader = DataLoaderFactory(dataset)

    dataset = ensure_enough_samples(dataloader, 10).dataset

    assert len(dataset) == 10

    assert all("input" in s for s in dataset)
    assert all("labels" in s for s in dataset)


def test_ensure_enough_samples_empty_dataset():
    dataset = []

    dataset = ensure_enough_samples(dataset, 10)

    assert len(dataset) == 0

    assert dataset == []
