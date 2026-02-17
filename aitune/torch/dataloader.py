# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
# Copyright 2020 The HuggingFace Team. All rights reserved.
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
"""Dataset definition to feed data for tuning.

User should provide a dataset with iterable samples, e.g.
    - list of tensors,
    - list of list of tensors,
    - list of dicts,
    - torch.utils.data.Dataset
    - HuggingFace datasets.

Example with a list of tensors:

>>> dataset = [torch.randn(3, 224, 224) for _ in range(4)]  # 4 random images
>>> for batch_size, args, kwargs in samples_generator(dataset, [4]):
...     args[0].shape  # First argument is a batch of 2 images
torch.Size([4, 3, 224, 224])

Example with a list of dictionaries:

>>> dataset = [{"input": torch.randn(3, 224, 224)} for _ in range(4)]
>>> for batch_size, args, kwargs in samples_generator(dataset, [4]):
...     kwargs["input"].shape
torch.Size([4, 3, 224, 224])


Example with dynamic shapes use our DataLoaderFactory and MinMaxRandomDataset.

>>> random_dataset = MinMaxRandomDataset(2, [InputConfig((3, 244, 244), (3, 488, 488), kwarg_name="image")])
>>> dataloader_factory = DataLoaderFactory(random_dataset)
>>> for batch_size, args, kwargs in samples_generator(dataloader_factory, [4]):
...     kwargs["image"].shape
torch.Size([4, 3, 244, 244])
torch.Size([4, 3, 488, 488])

Samples are collated into a batch with a transformers default collator with additional support for strings.

We are using torch.utils.data.Dataloader internally, we feed tuning with
different batch sizes of samples.

Non-batchable inputs should be handled by the user on module level.

>>> def my_module(*args, **kwargs):
...     print(len(kwargs["prompt"]), kwargs["width"], kwargs["height"])
>>> dataset = [{"prompt": "Hello, world!"} for _ in range(4)]
>>> for batch_size, args, kwargs in samples_generator(dataset, [4]):
...     my_module(*args, **kwargs, width=1024, height=758)
...     my_module(*args, **kwargs, width=2048, height=1536)
4 1024 758
4 2048 1536

"""

import itertools
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

DatasetLike = Sequence | torch.utils.data.Dataset


class DataLoaderFactory:
    """Factory for the torch DataLoader.

    We need samples with required batch sizes thus we create a DataLoader on
    demand during tuning, validation or benchmarking.

    If you need custom collate function you can pass it to the constructor.
    By default we use default_data_collator that is based on transformers default
    collator with additional support for strings and lists.

    NOTE: A batch of strings is a list. (Tokenization might be done using huggingface datasets and mapping functionality)

    More customization can be done by extending DataLoaderFactory and overriding
    create_dataloader method.
    """

    dataset: DatasetLike
    collate_fn: Callable | None = None
    num_workers: int = 0

    def __init__(self, dataset: DatasetLike, collate_fn: Callable | None = None, num_workers: int = 0):
        """Initialize the DataLoaderFactory.

        Args:
            dataset: The dataset or DataloaderFactory with the dataset
            collate_fn: The collate function to make a tensor batch from the inputs default is default_data_collator
            num_workers: The number of workers to use for the torch.utils.data.DataLoader
        """
        self.dataset = dataset
        self.collate_fn = collate_fn or default_data_collator
        self.num_workers = num_workers

    def create_dataloader(self, batch_size: int) -> torch.utils.data.DataLoader:
        """Create a DataLoader from the configuration."""
        if isinstance(self.dataset, DynamicShapeDataset):
            return torch.utils.data.DataLoader(
                _DynamicShapeDatasetWrapper(self.dataset, batch_size),
                batch_size=batch_size,
                collate_fn=self.collate_fn,
                drop_last=True,
                num_workers=self.num_workers,
            )

        if len(self.dataset) < batch_size:
            return torch.utils.data.DataLoader(
                SingleBatchDatasetWrapper(self.dataset, batch_size),
                batch_size=batch_size,
                collate_fn=self.collate_fn,
                drop_last=True,
                num_workers=self.num_workers,
            )

        return torch.utils.data.DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            drop_last=True,
            num_workers=self.num_workers,
        )


def _make_dataloader_factory(datasetlike: DatasetLike | DataLoaderFactory) -> DataLoaderFactory:
    """Create a DataLoaderFactory from dataset or if DataLoaderFactory is given return it.

    Convenience method to deal only with DataLoaderFactory.
    """
    if isinstance(datasetlike, DataLoaderFactory):
        return datasetlike

    return DataLoaderFactory(dataset=datasetlike)


def samples_generator(
    dataset: DatasetLike | DataLoaderFactory | torch.Tensor,
    batch_sizes: list[int] | Generator[int, None, None],
    max_num_batches_per_batch_size: int | None = None,
) -> Generator[tuple[int, list[Any], dict[str, Any]], None, None]:
    """Generate samples from the dataset with the given batch sizes and number of samples.

    It is convenience utility to iterate over all samples with different batch sizes in one go.

    >>> dataset = [{"prompt": "Hello, world!"} for _ in range(10)]
    >>> for batch_size, args, kwargs in samples_generator(dataset, [4, 8], 1):
    ...     print(batch_size, len(kwargs["prompt"]))
    4 4
    8 8

    >>> for batch_size, args, kwargs in samples_generator(torch.randn(10, 10), [1, 2]):
    ...     print(batch_size, args[0].shape)
    1 torch.Size([1, 10, 10])
    2 torch.Size([2, 10, 10])

    NOTE: for each batch size we will iterate over all samples in the dataset.

    Args:
        dataset: The dataset to generate samples from. It can be DataLoaderFactory or any dataset/iterable and even torch.Tensor.
            Tensor will be treated as a single sample dataset.
        batch_sizes: The batch sizes to generate samples with.
        max_num_batches_per_batch_size: The maximum number of batches to use for tuning per batch size.

    Returns:
        A generator of tuples of batch size, args and kwargs
    """
    if isinstance(dataset, torch.Tensor):
        dataset = [dataset]

    max_num_batches_per_batch_size = max_num_batches_per_batch_size or float("inf")
    dataloader_factory = _make_dataloader_factory(dataset)
    for batch_size in batch_sizes:
        dataloader = dataloader_factory.create_dataloader(batch_size)
        for i, batch in enumerate(dataloader):
            if i >= max_num_batches_per_batch_size:
                break

            args, kwargs = [], {}
            if isinstance(batch, Mapping):
                kwargs = batch
            elif len(batch) == 2 and isinstance(batch[1], Mapping):  # support for list of samples
                args, kwargs = batch
            elif isinstance(batch, Sequence):
                args = batch

            yield batch_size, args, kwargs


@dataclass
class InputConfig:
    """Configuration for an input tensor.

    Example basic usage:

    >>> InputConfig(min_input=(3, 24, 24), max_input=(3, 48, 48), kwarg_name="input")
    InputConfig(min_input=(3, 24, 24), max_input=(3, 48, 48), dtype=torch.float32, kwarg_name='input', min_value=0, max_value=1)

    Example with just min_input for both min and max:

    >>> InputConfig(min_input=(3, 24, 24), kwarg_name="input2", min_value=0, max_value=3)
    InputConfig(min_input=(3, 24, 24), max_input=(3, 24, 24), dtype=torch.float32, kwarg_name='input2', min_value=0, max_value=3)
    """

    min_input: torch.Size
    max_input: torch.Size | None = None
    dtype: torch.dtype = torch.float32
    kwarg_name: str | None = None
    min_value: int | float = 0
    max_value: int | float = 1

    def __post_init__(self):
        """Post-initialization hook."""
        if self.max_input is None:
            self.max_input = self.min_input


class DynamicShapeDataset(list, torch.utils.data.Dataset):
    """Each sample of this dataset is of a different shape.

    Example of list of tensors:

    >>> dataset = DynamicShapeDataset([torch.randn(10, 10), torch.randn(20, 20)])
    >>> for batch_size, args, _ in samples_generator(dataset, [4, 8]):
    ...     print(batch_size, args[0].shape)
    4 torch.Size([4, 10, 10])
    4 torch.Size([4, 20, 20])
    8 torch.Size([8, 10, 10])
    8 torch.Size([8, 20, 20])

    Example of list of dicts:

    >>> dataset = DynamicShapeDataset([{"input": torch.randn(10, 10)}, {"input": torch.randn(20, 20)}])
    >>> for batch_size, _, kwargs in samples_generator(dataset, [4, 8]):
    ...     print(batch_size, kwargs["input"].shape)
    4 torch.Size([4, 10, 10])
    4 torch.Size([4, 20, 20])
    8 torch.Size([8, 10, 10])
    8 torch.Size([8, 20, 20])
    """

    def __getitem__(self, index):
        """Get an item from the dataset."""
        return super().__getitem__(index)

    def __len__(self):
        """Get the length of the dataset."""
        return super().__len__()


class MinMaxRandomDataset(DynamicShapeDataset):
    """Dataset that contains `num_samples` samples with random shapes given by the input configs."""

    def __init__(
        self,
        num_samples: int,
        input_configs: list[InputConfig],
        include_min_max_shapes: bool = True,
    ):
        """Initialize the random min-max dataset.

        Args:
            num_samples: number of samples to generate
            input_configs: list of input configs
            include_min_max_shapes: if True, include min and max shapes in the samples
        Throws:
            ValueError: if num_samples is less than 2 as min and max shapes are included
            ValueError: if there mixed kwarg_names in input_configs - all input configs must have either a kwarg_name or none
        """
        min_samples = 2 if include_min_max_shapes else 1
        if num_samples < min_samples:
            raise ValueError(f"num_samples must be at least {min_samples}")

        self.num_samples = num_samples
        self.input_configs = input_configs
        self.include_min_max_shapes = include_min_max_shapes
        self.is_dict = self._validate_names()
        self.samples = self._generate_samples()

    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, index):
        """Get a sample from the dataset."""
        return self.samples[index]

    def _validate_names(self):
        all_has_name = all(cfg.kwarg_name is not None for cfg in self.input_configs)
        none_has_name = all(cfg.kwarg_name is None for cfg in self.input_configs)
        if not (all_has_name or none_has_name):
            raise ValueError("All input configs must have either a kwarg_name or none")
        return all_has_name

    def _generate_sample_dict(self, gen_tensor_fn: Callable):
        return {cfg.kwarg_name: gen_tensor_fn(cfg) for cfg in self.input_configs}

    def _generate_sample_list(self, gen_tensor_fn: Callable):
        return [gen_tensor_fn(cfg) for cfg in self.input_configs]

    def _generate_samples(self):
        gen = self._generate_sample_dict if self.is_dict else self._generate_sample_list
        samples = [gen(self._get_tensor) for _ in range(self.num_samples)]
        if self.include_min_max_shapes:
            samples = [gen(self._get_min_tensor), gen(self._get_max_tensor), *samples[:-2]]
        return samples

    def _get_tensor(self, cfg: InputConfig):
        min_x, max_x = cfg.min_input, cfg.max_input
        max_shape = np.array(max_x) + 1
        shapes = np.random.randint(min_x, max_shape).tolist()
        tensor = torch.empty(shapes, dtype=cfg.dtype).uniform_(cfg.min_value, cfg.max_value)
        return tensor

    def _get_min_tensor(self, cfg: InputConfig):
        return torch.empty(cfg.min_input, dtype=cfg.dtype).uniform_(cfg.min_value, cfg.max_value)

    def _get_max_tensor(self, cfg: InputConfig):
        return torch.empty(cfg.max_input, dtype=cfg.dtype).uniform_(cfg.min_value, cfg.max_value)


class SingleBatchDatasetWrapper(torch.utils.data.Dataset):
    """Makes sure that set has enough samples for batch size by repeating the elements of the dataset."""

    def __init__(self, dataset: DatasetLike, batch_size: int):
        """Initialize the wrapper.

        Args:
            dataset: The dataset to wrap.
            batch_size: The batch size.
        """
        self.dataset = dataset
        self.batch_size = batch_size

    def __len__(self):
        """Return the length of the dataset which is exactly batch_size."""
        return self.batch_size

    def __getitem__(self, index):
        """Return the item at the given index."""
        return self.dataset[index % len(self.dataset)]


class _DynamicShapeDatasetWrapper(torch.utils.data.Dataset):
    """Samples of different shapes cannot be batched together.

    This wrapper allows creation of batches by using same sample multiple times.

    Returns given dynamic shape samples batch_size times.

    NOTE: Increases dataset size by batch_size times.

    >>> dataset = [{"input": torch.randn(10, 10)}, {"input": torch.randn(20, 20)}]
    >>> wrapped_dataset = _DynamicShapeDatasetWrapper(dataset, 2)
    >>> len(wrapped_dataset)
    4
    >>> wrapped_dataset[0]["input"].shape
    torch.Size([10, 10])
    >>> wrapped_dataset[1]["input"].shape
    torch.Size([10, 10])
    >>> wrapped_dataset[2]["input"].shape
    torch.Size([20, 20])
    >>> wrapped_dataset[3]["input"].shape
    torch.Size([20, 20])
    """

    def __init__(self, dataset: DynamicShapeDataset, batch_size: int):
        """Initialize the wrapper.

        Args:
            dataset: The dataset to wrap.
            batch_size: The batch size.
        """
        self.dataset = dataset
        self._batch_size = batch_size

    def __len__(self) -> int:
        """Returns larger dataset so that we can create batch of different sample shapes."""
        return len(self.dataset) * self._batch_size

    def __getitem__(self, index: int) -> Any:
        """Return the item at the given index.

        Args:
            index: The index of the item to return.

        Returns:
            The item at the given index.
        """
        return self.dataset[index // self._batch_size]


def default_data_collator(batch: list) -> tuple | list | dict:
    """Default data collator that simply concatenates the inputs.

    Returns:
        List of concatenated tensors or dict of concatenated tensors.
    """
    first = batch[0]

    # checking for Sample (args, kwargs)
    if isinstance(first, (list, tuple)):
        if len(first) == 2 and isinstance(first[1], Mapping):
            return _list_data_collator([[tensor] for tensor in first[0]]), _map_data_collator([first[1]])

    if isinstance(first, torch.Tensor):
        return _list_data_collator([[tensor] for tensor in batch])
    elif isinstance(first, str):
        return _list_data_collator([[s] for s in batch])
    elif isinstance(first, (tuple, list)):
        return _list_data_collator(batch)
    elif isinstance(first, Mapping):
        return _map_data_collator(batch)
    else:
        raise ValueError(f"Unsupported batch type: {type(first)}")


def _list_data_collator(features: list[list]) -> list:
    """Default data collator that simply concatenates the inputs.

    Example of list of mixed tensors and numpy arrays:

    >>> features = [[torch.randn(10, 10), np.zeros((15, 15))], [torch.randn(10, 10), np.zeros((15, 15))]]
    >>> result = _list_data_collator(features)
    >>> result[0].shape, result[1].shape
    (torch.Size([2, 10, 10]), torch.Size([2, 15, 15]))

    Example of list of strings:

    >>> features = [["Hello World"], ["Hello World"]]
    >>> result = _list_data_collator(features)
    >>> result
    [['Hello World', 'Hello World']]

    Args:
        features: List of lists of samples

    Returns:
        List of concatenated tensors.
    """
    batch = []
    num_inputs = len(features[0])
    for i in range(num_inputs):
        first = features[0][i]
        if isinstance(first, torch.Tensor):
            batch.append(torch.stack([feature[i] for feature in features]))
        elif isinstance(first, np.ndarray):
            batch.append(torch.from_numpy(np.stack([feature[i] for feature in features])))
        elif isinstance(first, str):
            batch.append([feature[i] for feature in features])
        else:
            raise ValueError(f"Unsupported batch type: {type(first)}")

    return batch


def _map_data_collator(features: list) -> dict:
    """Transformers default data collector copied from source.

    NOTE: This is a copy of the transformers default data collector with additional support for strings.
    https://github.com/huggingface/transformers/blob/8f137b242762eb9295a431ec6eb8cd9ee673daf9/src/transformers/data/data_collator.py#L127
    Transformers are Apache 2.0 licensed.

    Copy has been made to add string support and avoid dependency on transformers.
    """
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if isinstance(first["label_ids"][0], int) else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None:
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.from_numpy(np.stack([f[k] for f in features]))
            elif isinstance(v, str):
                batch[k] = [f[k] for f in features]
            # Supporting dict of dicts for audio datasets: [{"audio": {"path": "...", "array": "...", "sampling_rate": "..."}}]
            elif isinstance(v, dict):
                # Note: Tensors and numpy arrays are not concatenated, as most frameworks does it by themselves
                batch[k] = {sub_k: [f[k][sub_k] for f in features] for sub_k in v.keys()}
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    return batch


def ensure_enough_samples(
    dataset: DatasetLike | DataLoaderFactory | torch.Tensor,
    number_of_samples: int,
) -> DatasetLike | DataLoaderFactory:
    """Ensures there is enough samples in the dataset to run the model for the given number of iterations.

    NOTE: This function trims the dataset to the given number of samples.

    Args:
        dataset: The dataset to make iterable.
        number_of_samples: The number of samples to ensure.

    Returns:
        The dataset with enough samples.
    """
    if number_of_samples <= 0:
        raise ValueError("number_of_samples must be greater than 0")

    if isinstance(dataset, torch.Tensor):
        dataset = [dataset]

    if isinstance(dataset, Iterable):
        return list(itertools.islice(itertools.cycle(dataset), number_of_samples))

    if isinstance(dataset, torch.utils.data.Dataset):
        # torch Dataset has only __getitem__ and only potentially __len__
        limited_dataset = []

        # getting enough samples by iterating over the dataset, cycling over the dataset if needed
        index = 0
        while len(limited_dataset) < number_of_samples:
            try:
                limited_dataset.append(dataset[index])
                index += 1
            except IndexError:
                index = 0

        return limited_dataset

    if isinstance(dataset, DataLoaderFactory):
        dataset.dataset = ensure_enough_samples(dataset.dataset, number_of_samples)
        return dataset
