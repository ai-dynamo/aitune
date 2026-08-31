# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tune a causal language model on one or multiple GPUs."""

import logging
from logging import basicConfig

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline as create_text_pipeline

from aitune.torch import Module, OneBackendStrategy
from aitune.torch import tune as aitune_tune
from aitune.torch.backend import (
    TorchEagerBackend,
    TorchInductorJitBackend,
    TorchInductorJitBackendConfig,
)
from llm.cmd_args import get_tune_parser
from llm.distributed import initialize as initialize_distributed
from llm.distributed import is_rank_zero
from llm.distributed import shutdown as shutdown_distributed


def get_model_and_tokenizer(model_id="Qwen/Qwen3-0.6B", multi_gpu=False):
    """Load a model and tokenizer for single-GPU or native-TP execution."""
    model_args = {"dtype": "auto", "trust_remote_code": False}
    if multi_gpu:
        model_args["tp_plan"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_args)
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token  # Some causal models do not define a padding token.
    if not multi_gpu:
        model = model.to("cuda")
    return model, tokenizer


def generation_options(cache: str, max_new_tokens: int) -> dict:
    """Return the generation options shared by reference, tuning, and tuned runs."""
    return {
        "do_sample": False,
        "disable_compile": True,
        "cache_implementation": None if cache == "no_cache" else cache,
        "use_cache": cache != "no_cache",
        "max_new_tokens": max_new_tokens,
    }


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


def tune_model(model, tokenizer, cache="static", max_new_tokens=20):
    """Tune the model.

    The selected strategies reflect how each cache mode changes generation. Dynamic cache stays in TorchEager. Static
    cache uses TorchEager for variable-length prefill and TorchInductor for single-token decode. Generation without a
    cache uses TorchInductor for its recorded graph.

    Args:
        model: The model to tune.
        tokenizer: The tokenizer to use.
        cache: The cache implementation to use.
        max_new_tokens: Number of tokens generated while tuning.

    Returns:
        The tuned model.

    """
    if cache == "no_cache":
        model = Module(
            model,
            model.__class__.__name__,
            strategy=OneBackendStrategy(TorchInductorJitBackend()).enable_find_max_batch_size(False),
        )
    elif cache == "dynamic":
        model = Module(
            model,
            model.__class__.__name__,
            strategy=OneBackendStrategy(TorchEagerBackend()).enable_find_max_batch_size(False),
        )
    elif cache == "static":
        model = Module(
            model,
            model.__class__.__name__,
            strategies=[
                # Prefill has a variable prompt length.
                OneBackendStrategy(TorchEagerBackend()).enable_find_max_batch_size(False),
                # Decode has a single-token input and a compile-friendly static cache.
                OneBackendStrategy(
                    TorchInductorJitBackend(TorchInductorJitBackendConfig(mode="reduce-overhead"))
                ).enable_find_max_batch_size(False),
            ],
        )

    generate_args = generation_options(cache, max_new_tokens)

    def run_generation(messages):
        inputs = tokenizer(messages, return_tensors="pt", padding=True)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            return model.generate(**inputs, **generate_args)

    aitune_tune(
        run_generation,
        ["2+2?", "How big is the universe?"],
        batch_sizes=[1, 2],
        device="cuda",
        dry_run=False,
    )
    return model


def run_example(args) -> None:
    """Run the example while owning its distributed and model lifecycles."""
    initialize_distributed(args.multi_gpu)
    try:
        generate_args = generation_options(args.cache, args.max_new_tokens)
        messages = [{"role": "user", "content": "How big is the universe?"}]

        # Establish the reference output before AITune wraps the model.
        model, tokenizer = get_model_and_tokenizer(args.model_id, args.multi_gpu)
        original_pipeline = create_text_pipeline("text-generation", model=model, tokenizer=tokenizer)
        original_output = original_pipeline(messages, **generate_args)

        # Tune a fresh model, then repeat the same generation for comparison.
        model, tokenizer = get_model_and_tokenizer(args.model_id, args.multi_gpu)
        tuned_model = tune_model(model, tokenizer, args.cache, args.max_new_tokens)
        tuned_pipeline = create_text_pipeline("text-generation", model=tuned_model, tokenizer=tokenizer)
        tuned_output = tuned_pipeline(messages, **generate_args)

        if is_rank_zero():
            print_conversation(original_output, title="Original model output")
            print_conversation(tuned_output, title="Tuned model output")
    finally:
        shutdown_distributed()


def main():
    """Initialize distributed execution, run the example, and release its resources."""
    basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S", force=True)
    args = get_tune_parser().parse_args()
    run_example(args)


if __name__ == "__main__":
    main()
