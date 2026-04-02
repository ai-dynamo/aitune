# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tune SentenceTransformer model."""

import os
from logging import INFO, basicConfig, getLogger
from pathlib import Path

import torch
from aitune.torch import MaxThroughputStrategy, inspect, save, tune, wrap
from aitune.torch.backend import TensorRTBackend, TensorRTBackendConfig
from aitune.torch.config import config as global_config

from .cmd_args import get_parser
from .model import get_model

logger = getLogger(__name__)

basicConfig(level=INFO, force=True)

DEFAULT_OUTPUT_DIR = os.environ.get("AITUNE_OUTPUT_DIR", ".")
DEFAULT_OUTPUT_FILE = Path(DEFAULT_OUTPUT_DIR) / "e5large_tuned.pt"


def main():
    """Main function."""
    args = get_parser().parse_args()

    tune_model(
        model_name=args.model_name,
        output_path=args.tuned_model_path,
        max_batch_size=args.max_batch_size,
    )


def tune_model(
    model_name: str,
    output_path: Path,
    max_batch_size: int,
):
    """Tune the SentenceTransformer model.

    Args:
        model_name: The name of the model to tune.
        output_path: The path to save the tuned model.
        max_batch_size: The maximum batch size to tune.
    """
    model = get_model(model_name=model_name)

    input_texts = [
        {"sentences": "query: how much protein should a female eat"},
        {"sentences": "query: summit define"},
        {
            "sentences": "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day."
        },
        {
            "sentences": "passage: Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments."
        },
    ]
    input_list = [i["sentences"] for i in input_texts]

    embeddings = model.encode(sentences=input_list, batch_size=4, normalize_embeddings=True, convert_to_tensor=True)

    def call_wrapper(*args, **kwargs):
        return model.encode(
            *args,
            **kwargs,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=False,
            convert_to_tensor=True,
            batch_size=4,
            device="cuda",
        )

    # NOTE: without min_depth=2, inspector finds a wrapper module `pipeline._modules["0"]` and fails with incorrect input in tunning
    #     with error: RuntimeError: Only Tensors created explicitly by the user (graph leaves) support the deepcopy protocol at the moment. @ torch/_tensor.py", line 136, in __deepcopy__
    #     during make_batch and deepcopy of arguments with all kwags arguments passed as dict in args[0]
    inspected_modules_info = inspect(model, input_texts, inference_function=call_wrapper, min_depth=2)
    inspected_modules_info.describe()

    # NOTE: workaround for SentenceTransformer to work with tuned model see below first dirty hack
    #       By default, we move modules to meta device after tuning to save memory. Our wrapper returns correct device for the module.
    #       However, SentenceTransformer overrides .device property, has one more reference to the same module we tuned, returns meta device by this additional reference.
    #       In encode() method, it calls .to(self.device) and tries to move modules that are already on right devices, breaking the inference.
    #       We need to move modules to cpu device after tuning to work with SentenceTransformer.
    global_config.device_after_tuning = "cpu"

    modules = inspected_modules_info.get_modules(min_execution_percentage=0.1)
    model = wrap(
        model,
        modules,
        strategy=MaxThroughputStrategy(
            backends=[
                TensorRTBackend(),  # fails, symbolic_shapes.ConstraintViolationError - probably requires user specified dynamic shapes
                TensorRTBackend(TensorRTBackendConfig(use_dynamo=False)),
            ]
        ).enable_find_max_batch_size(True),
    )

    tune(call_wrapper, input_texts, batch_sizes=[1], dry_run=True)
    logger.info("Tuning module: %s", model_name)

    tune(call_wrapper, input_texts, batch_sizes=[2**n for n in range(max_batch_size.bit_length())], dry_run=False)
    logger.info("Tuning completed.")

    logger.info("Running inference on the tuned model...")
    results = call_wrapper(sentences=input_list)

    torch.testing.assert_close(results, embeddings, rtol=1e-3, atol=1e-3)

    save(model, output_path)


if __name__ == "__main__":
    main()
