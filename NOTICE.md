<!--
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->

# Third-party notices

AITune incorporates modified code from the projects below under the shared
[Apache License 2.0](LICENSE).

| Component | Incorporated code | Copyright | License |
| --- | --- | --- | --- |
| Hugging Face Transformers | `_map_data_collator` in `aitune/torch/dataloader.py` | Copyright 2020 The HuggingFace Team. All rights reserved. Copyright 2018- The Hugging Face team. All rights reserved. | [Apache-2.0](LICENSE) |
| Ray | `examples/common/aitune_examples_common/batching.py` | Copyright 2023 Ray Authors | [Apache-2.0](LICENSE) |

Upstream sources and licenses:

- Transformers: [collator](https://github.com/huggingface/transformers/blob/8f137b242762eb9295a431ec6eb8cd9ee673daf9/src/transformers/data/data_collator.py), [license](https://github.com/huggingface/transformers/blob/8f137b242762eb9295a431ec6eb8cd9ee673daf9/LICENSE).
- Ray: [batching and supporting helpers](https://github.com/ray-project/ray/tree/ea987b5cfc6464ee74ace6ce81bbfed1792d3d93/python/ray), [license](https://github.com/ray-project/ray/blob/ea987b5cfc6464ee74ace6ce81bbfed1792d3d93/LICENSE).
