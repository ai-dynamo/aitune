# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Disk-backed storage for recorded tuning samples.

AITune records each module invocation as a ``(args, kwargs)`` pair. Keeping all
of those tensors alive would make tuning memory usage grow with the number and
size of recorded samples. :class:`SampleStore` writes every pair to a separate
``.pt`` file and exposes the files through the standard sequence interface, so
backend code can continue to index and iterate over samples without retaining
the whole collection in memory.

The store is internal to AITune. Users normally control how many entries it
contains with ``aitune.torch.config.max_num_samples_stored`` rather than creating
a store directly.
"""

import json
from collections.abc import Generator, Iterator, Sequence
from pathlib import Path
from typing import Any, overload

import torch

from aitune.torch.checkpoint.artifact import ArtifactPath

Sample = tuple[tuple, dict]

_MANIFEST_NAME = "manifest.json"
_SAMPLE_FILE_TEMPLATE = "sample-{index:05d}.pt"
_STORE_VERSION = 1


class SampleStore(Sequence[Sample]):
    """A sequence of samples stored on disk and loaded on demand.

    ``create`` initializes an empty directory containing ``manifest.json``.
    ``append`` serializes one ``(args, kwargs)`` pair at a time, while indexing
    and iteration deserialize fresh objects. Consequently, mutations made by a
    backend do not modify the recorded copy. Iteration loads only one sample at
    a time, and deepcopy shares the immutable persisted files instead of copying
    their tensor contents.

    When ``iter_samples(device=...)`` is used, serialized CUDA storage is moved
    to the requested CUDA device while CPU storage remains on CPU. The optional
    ``owner`` keeps temporary artifact roots alive for the lifetime of the store.

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

    Note:
        Disk reads and sample batching occur before AITune starts an inference
        measurement, so they do not contribute to reported backend latency or
        throughput. They can still add time to recording, backend construction,
        correctness validation, and the overall tuning workflow.
    """

    def __init__(
        self,
        artifact_path: ArtifactPath,
        filenames: list[str] | None = None,
        owner: Any | None = None,
    ) -> None:
        """Initialize a sample store.

        Args:
            artifact_path: Directory containing the stored samples.
            filenames: Existing sample filenames. When omitted, they are read from the manifest.
            owner: Optional object controlling the lifetime of the artifact root.
        """
        self._artifact = artifact_path
        self._owner = owner
        if filenames is None:
            manifest = json.loads((artifact_path.path / _MANIFEST_NAME).read_text(encoding="utf-8"))
            if manifest.get("version") != _STORE_VERSION:
                raise ValueError(f"Unsupported sample store version: {manifest.get('version')}")
            filenames = manifest["samples"]
        self._filenames = filenames

    @classmethod
    def create(cls, root: Path, relative_path: str | Path, owner: Any | None = None) -> "SampleStore":
        """Create an empty sample store below an owning artifact root."""
        artifact = ArtifactPath(root, relative_path)
        artifact.path.mkdir(parents=True, exist_ok=True)
        for stale_sample in artifact.path.glob("sample-*.pt"):
            stale_sample.unlink()
        store = cls(artifact, filenames=[], owner=owner)
        store._write_manifest()
        return store

    @classmethod
    def from_samples(cls, samples: Sequence[Sample], root: Path, relative_path: str | Path) -> "SampleStore":
        """Persist an existing sample sequence in a new store."""
        store = cls.create(root, relative_path)
        for sample in samples:
            store.append(sample)
        return store

    @property
    def artifact(self) -> ArtifactPath:
        """Return the checkpoint artifact containing the samples."""
        return self._artifact

    def append(self, sample: Sample) -> None:
        """Persist one sample."""
        filename = _SAMPLE_FILE_TEMPLATE.format(index=len(self))
        torch.save(sample, self._artifact.path / filename)
        self._filenames.append(filename)
        self._write_manifest()

    def iter_samples(self, device: torch.device | None = None) -> Generator[Sample, None, None]:
        """Load samples one at a time, optionally remapping accelerator storage to a device."""
        for filename in self._filenames:
            sample = torch.load(
                self._artifact.path / filename,
                map_location=self._map_location(device),
                weights_only=False,
            )
            try:
                yield sample
            finally:
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
        return len(self._filenames)

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
        filename = self._filenames[index]
        return torch.load(self._artifact.path / filename, weights_only=False)

    def __deepcopy__(self, memo: dict[int, Any]) -> "SampleStore":
        """Share immutable persisted samples instead of duplicating their files."""
        return self

    def _write_manifest(self) -> None:
        manifest = {"version": _STORE_VERSION, "samples": self._filenames}
        (self._artifact.path / _MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def ensure_sample_store(samples: Sequence[Sample], cache_dir: Path) -> SampleStore:
    """Return a shared store, persisting eager samples only for direct backend use."""
    if isinstance(samples, SampleStore):
        return samples
    return SampleStore.from_samples(samples, cache_dir, "samples")
