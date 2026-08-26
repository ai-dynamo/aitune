# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Disk-backed sequence of recorded module inputs."""

from collections.abc import Generator, Iterator, Sequence
from pathlib import Path
from typing import Any, overload

import torch

from aitune.torch.checkpoint.artifact import ArtifactPath

Sample = tuple[tuple, dict]

_SAMPLE_FILE_TEMPLATE = "sample-{index:05d}.pt"


class SampleStore(Sequence[Sample]):
    """Store samples as ``.pt`` files and load them only when requested.

    Each sample is a serialized ``(args, kwargs)`` pair in a sequentially numbered
    ``sample-N.pt`` file. Only the next sample index stays in memory during tuning.
    The sample directory must not contain other entries. Iteration loads one sample
    file at a time.

    ``iter_samples(device=...)`` restores CUDA-recorded tensors on the requested
    device. CPU-recorded tensors remain on CPU. Tuning consumers share the completed
    sample files and receive newly loaded sample objects.

    Example:
        Create a store, append positional and keyword inputs, and read them back:

        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     store = SampleStore.create(Path(directory), "samples")
        ...     store.append(((torch.tensor([1.0]),), {"scale": 2.0}))
        ...     len(store)
        ...     args, kwargs = store[0]
        ...     args[0] * kwargs["scale"]
        1
        tensor([2.])

    """

    def __init__(self, artifact_path: ArtifactPath) -> None:
        """Initialize a sample store.

        Args:
            artifact_path: Directory containing the stored samples.
        """
        self._artifact = artifact_path
        paths = list(self._artifact.path.iterdir())
        sample_count = len(paths)
        expected_filenames = {self._sample_filename(index) for index in range(sample_count)}
        if any(not path.is_file() for path in paths) or {path.name for path in paths} != expected_filenames:
            raise ValueError(f"Sample directory {artifact_path} must contain only contiguous sample-N.pt files")
        self._next_sample_index = sample_count

    @classmethod
    def create(cls, root: Path, relative_path: str | Path) -> "SampleStore":
        """Create an empty sample store below an artifact root."""
        artifact = ArtifactPath(root, relative_path)
        artifact.path.mkdir(parents=True, exist_ok=True)
        store = cls(artifact)
        for index in range(len(store)):
            (artifact.path / store._sample_filename(index)).unlink()
        store._next_sample_index = 0
        return store

    @classmethod
    def from_samples(cls, samples: Sequence[Sample], root: Path, relative_path: str | Path) -> "SampleStore":
        """Persist an existing sample sequence in a new store."""
        store = cls.create(root, relative_path)
        for sample in samples:
            store.append(sample)
        return store

    def append(self, sample: Sample) -> None:
        """Persist one sample."""
        filename = self._sample_filename(self._next_sample_index)
        torch.save(sample, self._artifact.path / filename)
        self._next_sample_index += 1

    def to_dict(self) -> dict[str, Any]:
        """Return the sample metadata stored in a backend checkpoint."""
        return {"artifact": self._artifact}

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "SampleStore":
        """Restore a store from backend checkpoint metadata."""
        return cls(state["artifact"])

    def iter_samples(self, device: torch.device | None = None) -> Generator[Sample, None, None]:
        """Load one sample at a time without retaining the iterator's reference."""
        for index in range(len(self)):
            sample = torch.load(
                self._artifact.path / self._sample_filename(index),
                map_location=self._map_location(device),
                weights_only=False,
            )
            try:
                yield sample
            finally:
                # Consumers may retain the sample; the iterator does not.
                del sample

    @staticmethod
    def _map_location(device: torch.device | None):
        """Preserve CPU tensors while remapping serialized CUDA storage."""
        if device is None:
            return None

        def remap(storage, location):
            if location.startswith("cuda"):
                return torch.serialization.default_restore_location(storage, str(device))
            return storage

        return remap

    def __iter__(self) -> Iterator[Sample]:
        """Iterate samples on their recorded devices."""
        return self.iter_samples()

    def __len__(self) -> int:
        """Return the number of stored samples."""
        return self._next_sample_index

    @overload
    def __getitem__(self, index: int) -> Sample: ...

    @overload
    def __getitem__(self, index: slice) -> list[Sample]: ...

    def __getitem__(self, index: int | slice) -> Sample | list[Sample]:
        """Load one sample or a slice of samples."""
        if isinstance(index, slice):
            indices = range(*index.indices(len(self)))
            return [self._load_sample(item) for item in indices]
        return self._load_sample(index)

    def _load_sample(self, index: int) -> Sample:
        normalized_index = range(len(self))[index]
        return torch.load(self._artifact.path / self._sample_filename(normalized_index), weights_only=False)

    @staticmethod
    def _sample_filename(index: int) -> str:
        return _SAMPLE_FILE_TEMPLATE.format(index=index)
