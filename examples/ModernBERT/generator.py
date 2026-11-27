# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate seeded ModernBERT batches from the committed FiQA passages."""

import argparse
import json
from pathlib import Path
from random import Random

from tokenizers import Tokenizer

DATA_PATH = Path(__file__).parent / "data" / "fiqa_32_256.jsonl"
TOKENIZER_REPO = "answerdotai/ModernBERT-base"
TOKENIZER_REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"
TOKEN_LENGTHS = range(32, 257)
PAD_ID = 50283
LENGTH_WEIGHTS = {
    range(32, 61): 100_000,
    range(61, 85): 880_000,
    range(85, 110): 600_000,
    range(110, 134): 385_000,
    range(134, 159): 205_000,
    range(159, 183): 125_000,
    range(183, 208): 80_000,
    range(208, 232): 55_000,
    range(232, 257): 135_000,
}


def load_tokenizer() -> Tokenizer:
    """Load the pinned tokenizer without truncation or padding."""
    tokenizer = Tokenizer.from_pretrained(TOKENIZER_REPO, revision=TOKENIZER_REVISION)
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer


class FiQABatchGenerator:
    """Sample passages with replacement from weighted length bins."""

    def __init__(self, seed: int) -> None:
        self._random = Random(seed)
        with DATA_PATH.open(encoding="utf-8") as stream:
            passages = [json.loads(line) for line in stream]
        if sorted(passage["num_tokens"] for passage in passages) != list(TOKEN_LENGTHS):
            raise ValueError("FiQA must contain exactly one passage for each token length from 32 through 256.")

        tokenizer = load_tokenizer()
        encodings = tokenizer.encode_batch([passage["text"] for passage in passages])
        self._token_ids: dict[int, list[int]] = {}
        for passage, encoding in zip(passages, encodings, strict=True):
            length = passage["num_tokens"]
            if len(encoding.ids) != length:
                raise ValueError(f"FiQA passage {passage['id']} has {len(encoding.ids)} tokens; expected {length}.")
            self._token_ids[length] = encoding.ids

    def batch(self, batch_size: int) -> dict[str, list[list[int]]]:
        """Return input IDs and attention masks padded to the longest passage in the batch."""
        if batch_size <= 0:
            raise ValueError("Batch size must be positive.")
        bins = self._random.choices(list(LENGTH_WEIGHTS), weights=list(LENGTH_WEIGHTS.values()), k=batch_size)
        lengths = [self._random.choice(length_bin) for length_bin in bins]
        width = max(lengths)
        return {
            "input_ids": [self._token_ids[length] + [PAD_ID] * (width - length) for length in lengths],
            "attention_mask": [[1] * length + [0] * (width - length) for length in lengths],
        }


def _check() -> None:
    generator = FiQABatchGenerator(seed=0)
    batch = generator.batch(32)
    assert batch == FiQABatchGenerator(seed=0).batch(32)
    lengths = [sum(mask) for mask in batch["attention_mask"]]
    width = max(lengths)
    assert min(lengths) < width
    for row, mask, length in zip(batch["input_ids"], batch["attention_mask"], lengths, strict=True):
        assert row == generator._token_ids[length] + [PAD_ID] * (width - length)
        assert mask == [1] * length + [0] * (width - length)
    print("FiQA self-check passed: 225 token lengths, seeded sampling, and right padding with attention masks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", required=True, help="Validate the dataset and a sampled batch.")
    parser.parse_args()
    _check()
