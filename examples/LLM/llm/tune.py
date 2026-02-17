# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
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
"""Tune ResNet model."""

import logging
from logging import basicConfig

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from aitune.torch import Module, OneBackendStrategy
from aitune.torch import tune as aitune
from aitune.torch.backend import TorchEagerBackend, TorchInductorBackend, TorchInductorBackendConfig
from llm.cmd_args import get_tune_parser


def get_model_and_tokenizer(model_id="microsoft/Phi-3-mini-4k-instruct"):
    """Get the model and tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token  # some models does not have it set
    return model.to("cuda"), tokenizer


def print_conversation(output, title="CONVERSATION OUTPUT"):
    """Print the conversation in a readable format.

    Args:
        output: The pipeline output containing generated text with messages.
    """
    print("=" * 60)
    print(title)
    print("=" * 60)
    for message in output[0]["generated_text"]:
        role = message["role"].upper()
        content = message["content"]
        print(f"{role}: {content}\n")


def tune_model(model, tokenizer, cache="static"):
    """Tune the model.

    There are different strategies to tune the model depending on the cache implementation:
    - Dynamic cache: is not compile friendly and is ignored, hence we can't distinguish between prefill and decode phases.
    - Static cache: For prefill phase we have to use torch eager backend because torch.compile would trigger every time
    as the prompt sequence length changes. However for decode phase we can use torch inductor backend as there is
    always static sequence length = 1 and static cache is torch.compile friendly.

    Args:
        model: The model to tune.
        tokenizer: The tokenizer to use.
        cache: The cache implementation to use.

    Returns:
        The tuned model.

    """
    if cache == "dynamic":
        model = Module(
            model,
            model.__class__.__name__,
            strategy=OneBackendStrategy(TorchEagerBackend()).enable_find_max_batch_size(
                False
            ),  # just one strategy, there is no prefill/decode distinction
        )
    elif cache == "static":
        model = Module(
            model,
            model.__class__.__name__,
            strategies=[
                OneBackendStrategy(TorchEagerBackend()).enable_find_max_batch_size(False),  # for prefill phase
                OneBackendStrategy(
                    TorchInductorBackend(TorchInductorBackendConfig(mode="reduce-overhead"))
                ).enable_find_max_batch_size(False),  # for decode phase
            ],
        )

    generate_args = {
        "do_sample": False,
        "disable_compile": True,
        "cache_implementation": cache,
        "use_cache": True,
    }

    def pipe(messages):
        inputs = tokenizer(messages, return_tensors="pt", padding=True)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            return model.generate(**inputs, **generate_args)

    aitune(pipe, ["2+2?", "How big is the universe?"], batch_sizes=[1, 2], device="cuda", dry_run=False)
    return model


def main():
    """Entry point for the script."""
    basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S", force=True)
    args = get_tune_parser().parse_args()

    generate_args = {
        "do_sample": False,  # make deterministic
        "disable_compile": True,
        "cache_implementation": args.cache,
        "use_cache": True,
        "max_new_tokens": 512,
    }
    messages = [
        {
            "role": "user",
            "content": "How big is the universe?",
        }
    ]
    model, tokenizer = get_model_and_tokenizer(args.model_id)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    original_output = pipe(messages, **generate_args)

    model, tokenizer = get_model_and_tokenizer(args.model_id)
    tuned_model = tune_model(model, tokenizer, args.cache)
    pipe = pipeline("text-generation", model=tuned_model, tokenizer=tokenizer)
    tuned_output = pipe(messages, **generate_args)

    print_conversation(original_output, title="Original model output")
    print_conversation(tuned_output, title="Tuned model output")


if __name__ == "__main__":
    main()
