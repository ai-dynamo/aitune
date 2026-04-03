# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# dependencies = ["transformers", "diffusers", "coloredlogs", "flatbuffers", "numpy", "packaging", "protobuf", "sympy"]
# scope = "always"
#
# [[pip_install]]
# packages = ["onnxruntime-gpu"]
# flags = ["--upgrade", "--pre", "--index-url", "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/"]
# ///

import tempfile
from logging import INFO, basicConfig, getLogger
from pathlib import Path
from time import perf_counter

import diffusers
import torch

from aitune.torch import inspect, load, save, tune, wrap
from aitune.torch.backend.onnx_runtime_backend import (
    ONNXExecutionProvider,
    ONNXRuntimeBackend,
    ONNXRuntimeBackendConfig,
)
from aitune.torch.module_registry import MODULE_REGISTRY
from aitune.torch.tune_strategy.one_backend_strategy import OneBackendStrategy

basicConfig(level=INFO, force=True)
logger = getLogger(__name__)


def test_stable_diffusion_dynamic_batch_onnx_runtime_tensorrt_ep():
    """Test ONNXRuntimeBackend with TensorRT EP on the StableDiffusion pipeline.

    Mirrors test 033 (ONNX Runtime CUDA EP) but uses the TensorRT Execution
    Provider so the full lifecycle is exercised end-to-end with TRT-accelerated
    ONNX inference:
        inspect → wrap → tune (batch_sizes=[1,2]) → save → load → infer

    Two batch sizes are verified (bs=1 and bs=2) to confirm that dynamic_axes
    in the exported ONNX model carry through the TensorRT EP correctly.

    Requires ``onnxruntime-gpu`` built with TensorRT support and a compatible
    TensorRT installation on the system.
    """
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    prompt = "A futuristic cityscape with neon lights and flying cars"
    sizes = [(256, 256), (512, 512)]
    steps = 10

    logger.info("Starting StableDiffusion ONNX Runtime TensorRT EP test")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        tuned_model_path = temp_path / "tuned_stable_diffusion_onnx_trt_ep.pt"
        output_dir = temp_path / "generated_images"
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Step 1: Load pipeline
            logger.info("Loading StableDiffusion pipeline")
            pipeline = diffusers.StableDiffusionPipeline.from_pretrained(model_id)
            pipeline.to("cuda")
            pipeline.set_progress_bar_config(disable=True)

            # Step 2: Inspect to identify modules
            logger.info("Inspecting pipeline modules")
            input_data = [{"prompt": prompt}]
            modules_info = inspect(pipeline, input_data)

            modules = modules_info.get_modules(limit=1)
            assert len(modules) > 0, "Expected at least one module"
            for module in modules:
                logger.info("Module: %s, execution time: %s", module.name, module.total_execution_time)

            # Step 3: Wrap with ONNXRuntimeBackend + TensorRT EP
            logger.info("Wrapping with ONNXRuntimeBackend (TensorRT EP)")
            strategy = OneBackendStrategy(
                ONNXRuntimeBackend(config=ONNXRuntimeBackendConfig(execution_provider=ONNXExecutionProvider.TENSORRT))
            )
            strategy.enable_find_max_batch_size(enable=False)
            pipeline = wrap(pipeline, modules, strategy=strategy)

            def call_wrapper(*args, **kwargs):
                for height, width in sizes:
                    pipeline(*args, height=height, width=width, num_inference_steps=steps, **kwargs)

            # Step 4: Dry run then tune
            tune(call_wrapper, input_data, dry_run=True)
            logger.info("Dry run completed")

            logger.info("Tuning with batch_sizes=[1, 2]")
            tune(
                call_wrapper,
                input_data,
                batch_sizes=[1, 2],
                dry_run=False,
                disable_external_logging=False,
                ignore_failing_modules=False,
            )
            logger.info("Tuning completed")

            # Step 5: Save
            logger.info("Saving tuned model to %s", tuned_model_path)
            save(pipeline, tuned_model_path)

            MODULE_REGISTRY.clear()

            # Step 6: Load in a fresh pipeline
            logger.info("Loading tuned model")
            fresh_pipeline = diffusers.StableDiffusionPipeline.from_pretrained(model_id)
            fresh_pipeline.to("cuda")
            fresh_pipeline.set_progress_bar_config(disable=True)
            loaded_pipeline = load(fresh_pipeline, tuned_model_path, disable_external_logging=False)

            check_size = 256
            check_steps = 50

            # Step 7: Inference at batch_size=1
            logger.info("Testing inference with batch_size=1")
            start = perf_counter()
            images_bs1 = loaded_pipeline(
                prompt=[prompt],
                height=check_size,
                width=check_size,
                num_inference_steps=check_steps,
                generator=torch.Generator(device="cuda").manual_seed(42),
            )
            end = perf_counter()

            actual_bs1 = images_bs1.images if hasattr(images_bs1, "images") else images_bs1
            assert len(actual_bs1) == 1, f"Expected 1 image, got {len(actual_bs1)}"
            logger.info("bs=1, res=%d, steps=%d, duration: %.2f s", check_size, check_steps, end - start)
            actual_bs1[0].save(output_dir / "sd_onnx_trt_ep_bs1.jpg")

            # Step 8: Inference at batch_size=2 (dynamic batch)
            logger.info("Testing inference with batch_size=2")
            start = perf_counter()
            images_bs2 = loaded_pipeline(
                prompt=[prompt, prompt],
                height=check_size,
                width=check_size,
                num_inference_steps=check_steps,
                generator=torch.Generator(device="cuda").manual_seed(42),
            )
            end = perf_counter()

            actual_bs2 = images_bs2.images if hasattr(images_bs2, "images") else images_bs2
            assert len(actual_bs2) == 2, f"Expected 2 images, got {len(actual_bs2)}"
            logger.info("bs=2, res=%d, steps=%d, duration: %.2f s", check_size, check_steps, end - start)
            for i, image in enumerate(actual_bs2):
                image.save(output_dir / f"sd_onnx_trt_ep_bs2_{i}.jpg")

            # Step 9: Verify all images were written and are non-trivial
            saved_images = list(output_dir.glob("*.jpg"))
            assert len(saved_images) == 3, f"Expected 3 saved images, found {len(saved_images)}"
            for img_path in saved_images:
                assert img_path.stat().st_size > 1000, f"Image {img_path} seems too small"

            logger.info("All ONNX Runtime TensorRT EP StableDiffusion tests passed")

        except Exception as e:
            logger.error("Test failed: %s", e)
            raise
        finally:
            MODULE_REGISTRY.clear()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    test_stable_diffusion_dynamic_batch_onnx_runtime_tensorrt_ep()
