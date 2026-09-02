# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Durable references to tuned files that publishers can consume."""

import hashlib
import os
import string
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from aitune.records.shapes import TunedTensorSpec

_HASH_CHUNK_SIZE = 1024 * 1024


class ArtifactIntegrityError(RuntimeError):
    """Raised when an artifact file is unavailable or differs from its recorded bytes."""


@dataclass(frozen=True, kw_only=True)
class Artifact:
    """Reference a finalized tuning result and its tensor interface.

    An artifact exists only after tuning has established concrete tensor bounds.
    The record neither owns nor serializes its cache file. The AITune Package
    design will define source provenance and persistence.

    Args:
        inputs: Ordered tuned input specifications.
        outputs: Ordered tuned output specifications.
        path: Tuned file in the AITune cache.
        fingerprint: Lowercase SHA-256 digest of the file at build time.
    """

    inputs: tuple[TunedTensorSpec, ...]
    outputs: tuple[TunedTensorSpec, ...]
    path: Path
    fingerprint: str

    def __post_init__(self) -> None:
        """Validate invariants that the field types cannot express."""
        if (
            len(self.fingerprint) != 64
            or self.fingerprint != self.fingerprint.lower()
            or any(character not in string.hexdigits for character in self.fingerprint)
        ):
            raise ValueError("Artifact fingerprint must be a lowercase SHA-256 hexadecimal digest")

        for label, tensors in (("input", self.inputs), ("output", self.outputs)):
            names = tuple(tensor.name for tensor in tensors)
            if len(names) != len(set(names)):
                raise ValueError(f"{label} tensor names must be unique, got {names}")

    @property
    def input_names(self) -> tuple[str, ...]:
        """Return input names in executable order."""
        return tuple(tensor.name for tensor in self.inputs)

    @property
    def output_names(self) -> tuple[str, ...]:
        """Return output names in executable order."""
        return tuple(tensor.name for tensor in self.outputs)

    @property
    def max_batch_size(self) -> int | None:
        """Return the largest batch size supported by every tensor, if known.

        Every input and output must have a known batch axis and support batch
        size one. Differing maximums are intersected by using the smallest one.
        """
        tensors = self.inputs + self.outputs
        if not tensors:
            return None

        maximums = []
        for tensor in tensors:
            if tensor.batch_axis is None or tensor.min_batch_size != 1:
                return None
            maximums.append(tensor.max_shape[tensor.batch_axis])
        return min(maximums)

    def verify(self) -> None:
        """Verify that the file is readable and matches its recorded hash.

        Raises:
            ArtifactIntegrityError: If the file cannot be read or its contents changed.
        """
        self._compare_fingerprint(self._fingerprint_file())

    def export_file(self, path: str | os.PathLike[str]) -> Path:
        """Copy verified artifact bytes to ``path``.

        Args:
            path: Destination file. Missing parent directories are created.

        Returns:
            The destination path.

        Raises:
            ArtifactIntegrityError: If the source cannot be read or changed since build.
        """
        destination = Path(path)
        if destination.resolve() == self.path.resolve():
            self.verify()
            return destination

        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        # Write beside the destination so replacement is atomic and failures preserve any existing file.
        output = tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        )
        temporary_path = Path(output.name)
        try:
            with output:
                for chunk in self._read_chunks():
                    digest.update(chunk)
                    output.write(chunk)

            self._compare_fingerprint(digest.hexdigest())
            os.replace(temporary_path, destination)
        except (ArtifactIntegrityError, OSError):
            temporary_path.unlink(missing_ok=True)
            raise
        return destination

    def _fingerprint_file(self) -> str:
        """Return the SHA-256 digest of the current artifact file."""
        digest = hashlib.sha256()
        for chunk in self._read_chunks():
            digest.update(chunk)
        return digest.hexdigest()

    def _read_chunks(self) -> Iterator[bytes]:
        """Yield source bytes while translating read failures."""
        try:
            with self.path.open("rb") as source:
                while chunk := source.read(_HASH_CHUNK_SIZE):
                    yield chunk
        except OSError as error:
            raise self._unreadable(error) from error

    def _compare_fingerprint(self, actual: str) -> None:
        """Raise unless ``actual`` is the recorded artifact fingerprint."""
        if actual != self.fingerprint:
            raise ArtifactIntegrityError(
                f"The {type(self).__name__} at {self.path} has changed since it was built: "
                f"expected {self.fingerprint[:12]}, found {actual[:12]}. Tune again to rebuild it."
            )

    def _unreadable(self, error: OSError) -> ArtifactIntegrityError:
        """Describe why the referenced cache file cannot be read."""
        return ArtifactIntegrityError(
            f"The {type(self).__name__} cannot be read at {self.path}. "
            f"The AITune cache may have been cleared or changed. Cause: {error}"
        )


@dataclass(frozen=True, kw_only=True)
class TensorRTPlanArtifact(Artifact):
    """A serialized TensorRT plan produced by tuning."""


@dataclass(frozen=True, kw_only=True)
class ONNXArtifact(Artifact):
    """An ONNX model finalized as a tuning result for ONNX Runtime."""
