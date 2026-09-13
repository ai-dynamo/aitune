# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate a Triton model store from a tuned ResNet package."""

import argparse
import faulthandler
import os
import threading
from pathlib import Path
from typing import cast

import aitune.triton
from aitune.torch import Module, load
from resnet.model import get_model


def get_parser():
    """Create the model-store generation parser."""
    parser = argparse.ArgumentParser(description="Generate a Triton model store from a tuned ResNet package")
    parser.add_argument("--model-name", default="resnet50", help="Name of the tuned ResNet model")
    parser.add_argument("--tuned-model-path", type=Path, default=Path("resnet50.ait"), help="Tuned AITune package")
    parser.add_argument(
        "--model-repository",
        type=Path,
        default=Path("model_repository"),
        help="Triton model repository (default: model_repository)",
    )
    parser.add_argument(
        "--model-analyzer-configs",
        type=Path,
        default=Path("model_analyzer"),
        help="Model Analyzer configuration directory (default: model_analyzer)",
    )
    return parser


def main():
    """Load a tuned package and generate its Triton deployment files."""
    args = get_parser().parse_args()
    print(f"Loading tuned model: {args.tuned_model_path} (pid={os.getpid()})", flush=True)
    tuned_model = cast(Module, load(get_model(args.model_name, pretrained=False), args.tuned_model_path))
    try:
        print("Tuned model loaded; extracting deployment artifact", flush=True)
        artifact = tuned_model.artifact()
        print(f"Publishing artifact: format={artifact.model.format}, runtime={artifact.runtime.name}", flush=True)
        model_path = aitune.triton.publish(
            artifact,
            path=args.model_repository,
            model_name=args.model_name,
            max_batch_size=artifact.max_batch_size,
        )
        print(f"Triton model: {model_path}", flush=True)
        print("Generating Model Analyzer configurations", flush=True)
        configs_path = aitune.triton.generate_model_analyzer_configs(
            artifact,
            model_path=model_path,
            path=args.model_analyzer_configs,
        )
        print(f"Model Analyzer configurations: {configs_path}", flush=True)
    finally:
        # Leave the watchdog armed through interpreter shutdown to diagnose exit hangs.
        faulthandler.dump_traceback_later(60, repeat=True)
        print("Deactivating tuned module (thread stacks will be dumped if cleanup stalls)", flush=True)
        tuned_model.deactivate()
        print("Tuned module deactivated", flush=True)
    threads = [(thread.name, thread.ident, thread.daemon) for thread in threading.enumerate()]
    print(f"Python threads before exit (name, ident, daemon): {threads}", flush=True)
    print("Model-store generation complete; returning from main", flush=True)


if __name__ == "__main__":
    main()
