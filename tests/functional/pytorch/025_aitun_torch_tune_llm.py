# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# dependencies = ["transformers"]
# scope = "always"
# allow_failure = true
# [environment]
# TQDM_DISABLE=1
# ///

import logging
from logging import basicConfig
from typing import Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from aitune.torch import Module, OneBackendStrategy
from aitune.torch import tune as aitune
from aitune.torch.backend import TorchEagerBackend, TorchInductorJitBackend


def get_model_and_tokenizer(model_id="Qwen/Qwen2.5-0.5B-Instruct"):
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
        _decode_strategy = OneBackendStrategy(TorchInductorJitBackend())
        _decode_strategy.enable_performance_validation(False)
        _decode_strategy.enable_find_max_batch_size(False)
        model = Module(
            model,
            model.__class__.__name__,
            strategies=[
                OneBackendStrategy(TorchEagerBackend()).enable_find_max_batch_size(False),  # for prefill phase
                _decode_strategy,  # for decode phase
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


def run_tuning_for_cache(cache: Literal["dynamic", "static"]):
    """Run tuning for a given cache implementation."""
    generate_args = {
        "do_sample": False,  # make deterministic
        "disable_compile": True,
        "cache_implementation": cache,
        "use_cache": True,
        "max_new_tokens": 128,
    }
    messages = [
        {
            "role": "user",
            "content": "4*7=?",
        }
    ]
    model, tokenizer = get_model_and_tokenizer()
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    original_output = pipe(messages, **generate_args)

    model, tokenizer = get_model_and_tokenizer()
    tuned_model = tune_model(model, tokenizer, cache)
    pipe = pipeline("text-generation", model=tuned_model, tokenizer=tokenizer)
    tuned_output = pipe(messages, **generate_args)

    if original_output == tuned_output:
        print("Success: Original and tuned model outputs are the same for cache implementation: ", cache)
        print_conversation(tuned_output, title="Model output")
    else:
        print("Error: Original and tuned model outputs are not the same for cache implementation: ", cache)
        print_conversation(original_output, title="Expected model output")
        print_conversation(tuned_output, title="Tuned model output")


def test_static_cache():
    """Test static cache."""
    logging.info("Testing static cache...")
    run_tuning_for_cache("static")


def test_dynamic_cache():
    """Test dynamic cache."""
    logging.info("Testing dynamic cache...")
    run_tuning_for_cache("dynamic")


if __name__ == "__main__":
    basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S", force=True)
    test_static_cache()
    test_dynamic_cache()
