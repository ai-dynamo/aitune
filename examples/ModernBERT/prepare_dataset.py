# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rebuild the committed FiQA length dataset from the pinned corpus."""

import hashlib
import json
from collections.abc import Iterator

from generator import DATA_PATH, TOKEN_LENGTHS, TOKENIZER_REPO, TOKENIZER_REVISION, load_tokenizer
from huggingface_hub import hf_hub_download
from pyarrow.parquet import ParquetFile

CORPUS_REPO = "BeIR/fiqa"
CORPUS_REVISION = "979c07a7cb5ccc6ca009792241fa1250b98055dd"
CORPUS_CONFIG = "corpus"
CORPUS_SPLIT = "corpus"
CORPUS_SIZE = 57_638


def prepare_dataset() -> None:
    """Build the length dataset from the pinned FiQA Parquet file."""
    _write_dataset(_parquet_batches(), "pinned_parquet")


def _write_dataset(
    batches: Iterator[list[dict[str, str]]], download: str, *, ascending_id_prefix: bool = False
) -> None:
    tokenizer = load_tokenizer()
    selected: dict[int, tuple[int, str]] = {}
    num_documents = 0
    for rows in batches:
        num_documents += len(rows)
        passages = ["\n".join(part for part in (row["title"], row["text"]) if part) for row in rows]
        encodings = tokenizer.encode_batch(passages)
        for row, passage, encoding in zip(rows, passages, encodings, strict=True):
            length = len(encoding.ids)
            if length not in TOKEN_LENGTHS:
                continue
            doc_id = int(row["_id"])
            if length not in selected or doc_id < selected[length][0]:
                selected[length] = (doc_id, passage)

    if not ascending_id_prefix and num_documents != CORPUS_SIZE:
        raise ValueError(f"Expected {CORPUS_SIZE} FiQA documents; got {num_documents}.")
    if missing := sorted(set(TOKEN_LENGTHS) - selected.keys()):
        raise ValueError(f"FiQA has no passages for these token lengths: {missing}.")
    dataset = "".join(
        json.dumps({"id": str(doc_id), "text": text, "num_tokens": length}, ensure_ascii=False) + "\n"
        for length, (doc_id, text) in sorted(selected.items())
    ).encode("utf-8")
    metadata = {
        "source": {
            "repo": CORPUS_REPO,
            "revision": CORPUS_REVISION,
            "config": CORPUS_CONFIG,
            "split": CORPUS_SPLIT,
            "num_documents": CORPUS_SIZE,
            "num_documents_scanned": num_documents,
            "download": download,
        },
        "tokenizer": {"repo": TOKENIZER_REPO, "revision": TOKENIZER_REVISION},
        "num_passages": len(selected),
        "min_tokens": TOKEN_LENGTHS.start,
        "max_tokens": TOKEN_LENGTHS.stop - 1,
        "sha256": hashlib.sha256(dataset).hexdigest(),
    }
    if ascending_id_prefix:
        metadata["source"]["selection_assumption"] = (
            "FiQA corpus rows are in strictly ascending numeric document ID order."
        )
    DATA_PATH.parent.mkdir(exist_ok=True)
    DATA_PATH.write_bytes(dataset)
    DATA_PATH.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def _parquet_batches() -> Iterator[list[dict[str, str]]]:
    corpus_path = hf_hub_download(
        CORPUS_REPO,
        f"{CORPUS_CONFIG}/{CORPUS_SPLIT}-00000-of-00001.parquet",
        repo_type="dataset",
        revision=CORPUS_REVISION,
    )
    for batch in ParquetFile(corpus_path).iter_batches(batch_size=512, columns=["_id", "title", "text"]):
        yield batch.to_pylist()


if __name__ == "__main__":
    prepare_dataset()
