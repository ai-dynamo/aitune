---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "ModernBERT"
---

# ModernBERT

The committed FiQA data contains one passage per token length from 32 through 256, including
`[CLS]` and `[SEP]`. `FiQABatchGenerator(seed=…)` samples from the weighted Bloomberg request length
bins. Use different seeds for tuning and measurement. Each batch includes input IDs and an
attention mask, with rows right-padded to the longest passage in that batch.

Install `tokenizers` and `huggingface-hub`, then run `python generator.py --check` from this directory.
To regenerate the data, install `huggingface-hub[hf-xet]` and `pyarrow`, then run
`python prepare_dataset.py`.

The passages are a length-filtered selection from [FiQA-2018](https://sites.google.com/view/fiqa/),
distributed in [BEIR's FiQA dataset](https://huggingface.co/datasets/BeIR/fiqa) under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). This subset keeps the original
passage text and joins titles and bodies with a newline when both are present. See
[`fiqa_32_256.meta.json`](data/fiqa_32_256.meta.json) for source and tokenizer revisions.
