# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
"""Benchmark functions."""

import logging
from dataclasses import dataclass
from logging import basicConfig
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from llm.cmd_args import get_benchmark_parser
from llm.tune import get_model_and_tokenizer, tune_model

torch.random.manual_seed(0)
torch.set_float32_matmul_precision("high")


@dataclass
class BenchmarkArgs:
    """Benchmark arguments.

    Attributes:
        model_id: Model ID to load from Hugging Face Hub.
        sequence_lengths: List of tuples (ISL, OSL)
        iterations: Number of benchmark iterations.
        warmup_iters: Number of warmup iterations.
        cache: Cache implementation type.
        scenario: Benchmark scenario to run.
    """

    model_id: str
    sequence_lengths: list[tuple[int, int]]
    iterations: int
    warmup_iters: int
    cache: str
    scenario: str


def get_inputs_from_chat_template(messages, tokenizer, device):
    """Generate model inputs from chat messages.

    Args:
        messages: The list of message dictionaries with 'role' and 'content' keys.
        tokenizer: The tokenizer to apply chat template.
        device: The device to move tensors to.

    Returns:
        dict: Model inputs with tensors moved to the specified device.
    """
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs


def get_random_inputs(isl: int, tokenizer, device, batch_size: int = 1):
    """Generate random model inputs.

    Args:
        isl: Input sequence length.
        tokenizer: The tokenizer to determine valid token IDs.
        device: The device to move tensors to.

    Returns:
        dict: Model inputs with random input_ids and attention_mask.
    """
    vocab_size = len(tokenizer)
    input_ids = torch.randint(0, vocab_size, (batch_size, isl), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
    }


def benchmark(
    model,
    tokenizer,
    sequence_lengths: list[tuple[int, int]],
    iterations: int = 1,
    warmup_iters: int = 3,
    generation_args: dict | None = None,
):
    """Benchmark the model.

    Args:
        model: The model to benchmark.
        tokenizer: The tokenizer to use.
        sequence_lengths: The list of tuples (ISL, OSL)
        iterations: The number of iterations to run the benchmark.
        warmup_iters: The number of warmup iterations to run the benchmark.
        generation_args: The generation arguments to use.

    Returns:
        dict: The benchmark result.
            - isl: The input sequence length.
            - osl: The output sequence length.
            - mean_time: The mean time of the benchmark.
            - min_time: The min time of the benchmark.
            - max_time: The max time of the benchmark.
            - std_time: The std time of the benchmark.
            - generation_args: The generation arguments used.
    """
    gen_args = generation_args.copy() if generation_args else {}
    gen_args["do_sample"] = False  # make it deterministic
    gen_args["eos_token_id"] = len(tokenizer) + 1  # change eos token to generate up to osl

    def run_inference(isl, osl, iterations):
        """Run inference for a isl, osl pair."""
        inputs = get_random_inputs(isl, tokenizer, model.device, batch_size=1)
        gen_args["max_new_tokens"] = osl
        times = []
        with torch.inference_mode():
            for _ in range(iterations):
                start = perf_counter()
                torch.cuda.synchronize()
                model.generate(**inputs, **gen_args)
                torch.cuda.synchronize()
                times.append(perf_counter() - start)
        return times

    results = []
    for isl, osl in sequence_lengths:
        # warmup
        logging.info("Warming up %d times with isl=%d, osl=%d", warmup_iters, isl, osl)
        run_inference(isl, osl, warmup_iters)

        # measure
        logging.info("Benchmarking sequence %d times with isl=%d, osl=%d", iterations, isl, osl)
        times = run_inference(isl, osl, iterations)
        results.append(
            {
                "isl": isl,
                "osl": osl,
                "mean_time": np.mean(times),
                "min_time": np.min(times),
                "max_time": np.max(times),
                "std_time": np.std(times),
            }
            | generation_args
        )

    return results


def benchmark_scenario(args: BenchmarkArgs):
    """Run benchmark scenario.

    Args:
        args: Benchmark arguments.
    """
    # raise the default limit to reveal recompilation issues
    torch._dynamo.config.recompile_limit = 100
    torch.compiler.reset()

    model, tokenizer = get_model_and_tokenizer(args.model_id)

    generate_args = {}

    if args.cache == "no_cache":
        generate_args["cache_implementation"] = None
        generate_args["use_cache"] = False
    else:
        generate_args["cache_implementation"] = args.cache
        generate_args["use_cache"] = True

    if args.scenario == "aot":
        generate_args["disable_compile"] = True  # prevent compilation by HF
        model = tune_model(model, tokenizer, args.cache)
    else:
        generate_args["disable_compile"] = False

    results = benchmark(
        model,
        tokenizer,
        sequence_lengths=args.sequence_lengths,
        iterations=args.iterations,
        warmup_iters=args.warmup_iters,
        generation_args=generate_args,
    )

    for result in results:
        result["model_id"] = args.model_id
        result["cache"] = args.cache
        result["scenario"] = args.scenario

    return results


def main():
    basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S", force=True)

    parser = get_benchmark_parser()
    parsed_args = parser.parse_args()

    args = BenchmarkArgs(**vars(parsed_args))
    logging.info("Benchmarking scenario with args: %s", args)
    result = benchmark_scenario(args)
    df = pd.DataFrame(result)
    logging.info("Benchmarking result:\n%s", df)


if __name__ == "__main__":
    main()
