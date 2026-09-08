# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Durable references to tuned files that publishers can consume."""

import hashlib
import os
import string
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from aitune.records.shapes import BoundedTensorSpec

_HASH_CHUNK_SIZE = 1024 * 1024


class ArtifactIntegrityError(RuntimeError):
    """Raised when an artifact file is unavailable or differs from its recorded bytes."""


def _validate_fingerprint(fingerprint: str) -> None:
    """Validate a SHA-256 digest stored in an artifact record."""
    if (
        len(fingerprint) != 64
        or fingerprint != fingerprint.lower()
        or any(character not in string.hexdigits for character in fingerprint)
    ):
        raise ValueError("Artifact fingerprint must be a lowercase SHA-256 hexadecimal digest")


@dataclass(frozen=True, slots=True)
class ArtifactFile:
    """Describe a file required beside an artifact's entry file.

    Args:
        relative_path: File location relative to the entry file's directory.
        fingerprint: Lowercase SHA-256 digest of the file at build time.
    """

    relative_path: Path
    fingerprint: str

    def __post_init__(self) -> None:
        """Reject paths that could escape the artifact directory."""
        if self.relative_path == Path() or self.relative_path.is_absolute() or ".." in self.relative_path.parts:
            raise ValueError(
                f"Artifact companion path must stay inside the artifact directory, got {self.relative_path}"
            )
        _validate_fingerprint(self.fingerprint)


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
        companions: Additional files resolved relative to ``path.parent``.
    """

    inputs: tuple[BoundedTensorSpec, ...]
    outputs: tuple[BoundedTensorSpec, ...]
    path: Path
    fingerprint: str
    companions: tuple[ArtifactFile, ...] = ()

    def __post_init__(self) -> None:
        """Validate invariants that the field types cannot express."""
        _validate_fingerprint(self.fingerprint)

        for label, tensors in (("input", self.inputs), ("output", self.outputs)):
            names = tuple(tensor.name for tensor in tensors)
            if len(names) != len(set(names)):
                raise ValueError(f"{label} tensor names must be unique, got {names}")

        companion_paths = tuple(companion.relative_path for companion in self.companions)
        if len(companion_paths) != len(set(companion_paths)):
            raise ValueError(f"Artifact companion paths must be unique, got {companion_paths}")
        if Path(self.path.name) in companion_paths:
            raise ValueError("An artifact companion cannot also be its entry file")

    @property
    def files(self) -> tuple[tuple[Path, str], ...]:
        """Return the complete artifact file set, with the entry file first."""
        return (
            (self.path, self.fingerprint),
            *((self.path.parent / file.relative_path, file.fingerprint) for file in self.companions),
        )

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
        """Verify that every artifact file is readable and matches its recorded hash.

        Raises:
            ArtifactIntegrityError: If the file cannot be read or its contents changed.
        """
        for path, fingerprint in self.files:
            self._compare_fingerprint(self._fingerprint_file(path), path, fingerprint)

    def export_file(self, path: str | os.PathLike[str]) -> Path:
        """Copy the verified entry file and its companions to ``path``.

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

        planned = [(self.path, destination, self.fingerprint)]
        planned.extend(
            (
                self.path.parent / companion.relative_path,
                destination.parent / companion.relative_path,
                companion.fingerprint,
            )
            for companion in self.companions
        )
        destinations = tuple(target for _, target, _ in planned)
        if len(destinations) != len(set(destinations)):
            raise ValueError("The exported entry file would overwrite one of its companions")

        staged: list[tuple[Path, Path]] = []
        try:
            for source, target, fingerprint in planned:
                target.parent.mkdir(parents=True, exist_ok=True)
                output = tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=target.parent,
                    prefix=f".{target.name}.",
                    suffix=".tmp",
                    delete=False,
                )
                temporary_path = Path(output.name)
                staged.append((temporary_path, target))
                digest = hashlib.sha256()
                with output:
                    for chunk in self._read_chunks(source):
                        digest.update(chunk)
                        output.write(chunk)
                self._compare_fingerprint(digest.hexdigest(), source, fingerprint)

            for temporary_path, target in staged:
                os.replace(temporary_path, target)
        except (ArtifactIntegrityError, OSError):
            for temporary_path, _ in staged:
                temporary_path.unlink(missing_ok=True)
            raise
        return destination

    def _fingerprint_file(self, path: Path) -> str:
        """Return the SHA-256 digest of the current artifact file."""
        digest = hashlib.sha256()
        for chunk in self._read_chunks(path):
            digest.update(chunk)
        return digest.hexdigest()

    def _read_chunks(self, path: Path) -> Iterator[bytes]:
        """Yield source bytes while translating read failures."""
        try:
            with path.open("rb") as source:
                while chunk := source.read(_HASH_CHUNK_SIZE):
                    yield chunk
        except OSError as error:
            raise self._unreadable(path, error) from error

    def _compare_fingerprint(self, actual: str, path: Path, expected: str) -> None:
        """Raise unless ``actual`` is the recorded artifact fingerprint."""
        if actual != expected:
            raise ArtifactIntegrityError(
                f"The {type(self).__name__} at {path} has changed since it was built: "
                f"expected {expected[:12]}, found {actual[:12]}. Tune again to rebuild it."
            )

    def _unreadable(self, path: Path, error: OSError) -> ArtifactIntegrityError:
        """Describe why the referenced cache file cannot be read."""
        return ArtifactIntegrityError(
            f"The {type(self).__name__} cannot be read at {path}. "
            f"The AITune cache may have been cleared or changed. Cause: {error}"
        )


@dataclass(frozen=True, kw_only=True)
class TensorRTPlanArtifact(Artifact):
    """A serialized TensorRT plan produced by tuning.

    Args:
        optimization_profile_count: Number of profiles stored in the plan.
        use_cuda_graphs: Whether the selected AITune runtime used CUDA graphs.
    """

    optimization_profile_count: int = 1
    use_cuda_graphs: bool = False

    def __post_init__(self) -> None:
        """Validate TensorRT-specific runtime metadata."""
        super().__post_init__()
        if self.optimization_profile_count < 1:
            raise ValueError("A TensorRT plan must contain at least one optimization profile")


class ONNXExecutionProvider(str, Enum):
    """ONNX Runtime execution provider selected during tuning."""

    CUDA = "cuda"
    TENSORRT = "tensorrt"


@dataclass(frozen=True, kw_only=True)
class ONNXArtifact(Artifact):
    """An ONNX model finalized as a tuning result for ONNX Runtime."""

    execution_provider: ONNXExecutionProvider = ONNXExecutionProvider.CUDA


@dataclass(frozen=True, kw_only=True)
class PT2Artifact(Artifact):
    """An AOTInductor package consumable by Triton's ``torch_aoti`` backend.

    Args:
        structured_call: Whether the package embeds structured inputs or outputs.
    """

    structured_call: bool
