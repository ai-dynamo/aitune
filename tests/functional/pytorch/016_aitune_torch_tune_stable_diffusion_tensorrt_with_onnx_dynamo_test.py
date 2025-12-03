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

# /// script
# dependencies = ["transformers<5", "diffusers"]
#
# # Optional, default "always", determines how often test is generated, always, nightly, weekly, monthly
# scope = "always"
# docker_image = "nvcr.io/nvidia/pytorch:25.08-py3"
# ///

import tempfile
from logging import INFO, basicConfig, getLogger
from pathlib import Path

import diffusers
import torch

from aitune.torch import inspect, load, save, tune, wrap
from aitune.torch.backend.tensorrt.tensorrt_backend import TensorRTBackend, TensorRTBackendConfig
from aitune.torch.module_registry import MODULE_REGISTRY
from aitune.torch.tune_strategy.one_backend_strategy import OneBackendStrategy

basicConfig(level=INFO, force=True)
logger = getLogger(__name__)


def test_stable_diffusion_dynamic_batch_tensorrt_dynamo():
    """Test dynamic shape inference of TensorRT backend with StableDiffusion.

    This test:
    1. Loads StableDiffusion pipeline
    2. Inspects the pipeline to identify key modules
    3. Wraps the main UNet module with TensorRT backend configured for dynamic shapes
    4. Tunes the model with batch_size=2 to enable dynamic shape support
    5. Saves the tuned model
    6. Loads the tuned model and tests inference with different batch sizes (bs=1 and bs=2)
    7. Saves generated images to verify functionality
    """
    # Test configuration
    model_id = "stabilityai/stable-diffusion-2-1"
    prompt = "A futuristic cityscape with neon lights and flying cars"
    sizes = [(256, 256), (512, 512)]  # Multiple sizes for faster testing
    steps = 10  # Reduced steps for faster testing

    logger.info("Starting StableDiffusion dynamic batch TensorRT test")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        tuned_model_path = temp_path / "tuned_stable_diffusion.pt"
        output_dir = temp_path / "generated_images"
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Step 1: Load StableDiffusion pipeline
            logger.info("Loading StableDiffusion pipeline")
            pipeline = diffusers.StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            pipeline.to("cuda")
            pipeline.set_progress_bar_config(disable=True)  # Disable progress bar for cleaner logs

            # Step 2: Inspect pipeline to identify modules
            logger.info("Inspecting pipeline modules")
            input_data = [{"prompt": prompt}]
            modules_info = inspect(pipeline, input_data)

            # Get the all modules for tuning
            modules = modules_info.get_modules()
            assert len(modules) > 0, "Expected at least one module"

            # Print modules info
            for module in modules:
                logger.info("Module: %s with execution time %s", module.name, module.total_execution_time)

            # Step 3: Configure TensorRT backend with dynamic shapes support
            logger.info("Configuring TensorRT backend with dynamic shapes")
            tensorrt_config = TensorRTBackendConfig(use_dynamo=True)
            tensorrt_backend = TensorRTBackend(config=tensorrt_config)

            # Step 4: Wrap the UNet module with TensorRT backend
            logger.info("Wrapping all modules with TensorRT backend")
            strategy = OneBackendStrategy(tensorrt_backend)
            strategy.enable_find_max_batch_size(enable=False)  # WAR: Disabled to avoid OOM
            pipeline = wrap(pipeline, modules, strategy=strategy)

            # Create wrapper function for both tuning and inference
            def call_wrapper(*args, **kwargs):
                """Wrapper function for pipeline calls that supports different batch sizes."""
                for height, width in sizes:
                    pipeline(
                        *args,
                        height=height,
                        width=width,
                        num_inference_steps=steps,
                        **kwargs,
                    )

            # Step 5: First do a dry run for testing
            tune(call_wrapper, input_data, dry_run=True)
            logger.info("Dry run completed successfully")

            # Now do the actual tuning with batch_size=2 to enable dynamic shape inference
            logger.info("Tuning StableDiffusion with TensorRT backend (batch_size=[1, 2])")
            tune(
                call_wrapper,
                input_data,
                batch_sizes=[1, 2],  # Tune with batch_size=2 to enable dynamic shapes
                dry_run=False,
                disable_external_logging=False,
            )
            logger.info("Tuning completed successfully")

            # Step 6: Save the tuned model
            logger.info("Saving tuned model to %s", tuned_model_path)
            save(pipeline, tuned_model_path)
            logger.info("Model saved successfully")

            # Clear registry to simulate loading in new session
            MODULE_REGISTRY.clear()

            # Step 7: Load the tuned model
            logger.info("Loading tuned StableDiffusion model")
            fresh_pipeline = diffusers.StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            fresh_pipeline.to("cuda")
            fresh_pipeline.set_progress_bar_config(disable=True)

            # Load the tuned components
            loaded_pipeline = load(fresh_pipeline, tuned_model_path, disable_external_logging=False)

            # Step 8: Test inference with batch_size=1
            logger.info("Testing inference with batch_size=1")
            single_prompt = [prompt]
            images_bs1 = loaded_pipeline(
                prompt=single_prompt,
                height=256,
                width=256,
                num_inference_steps=steps,
                generator=torch.Generator(device="cuda").manual_seed(42),
            )

            # Verify output structure for batch_size=1
            assert hasattr(images_bs1, "images") or isinstance(images_bs1, (list, tuple)), "Expected images output"
            if hasattr(images_bs1, "images"):
                actual_images_bs1 = images_bs1.images
            else:
                actual_images_bs1 = images_bs1

            assert len(actual_images_bs1) == 1, f"Expected 1 image for batch_size=1, got {len(actual_images_bs1)}"
            logger.info("Batch_size=1 inference successful")

            # Save image for batch_size=1
            output_path_bs1 = output_dir / "stable_diffusion_bs1.jpg"
            actual_images_bs1[0].save(output_path_bs1)
            logger.info("Saved batch_size=1 image to %s", output_path_bs1)

            # Step 9: Test inference with batch_size=2 (dynamic shape inference)
            logger.info("Testing dynamic shape inference with batch_size=2")
            double_prompt = [prompt, prompt]  # Same prompt twice for batch_size=2
            images_bs2 = loaded_pipeline(
                prompt=double_prompt,
                height=512,
                width=512,
                num_inference_steps=steps,
                generator=torch.Generator(device="cuda").manual_seed(42),
            )

            # Verify output structure for batch_size=2
            assert hasattr(images_bs2, "images") or isinstance(images_bs2, (list, tuple)), "Expected images output"
            if hasattr(images_bs2, "images"):
                actual_images_bs2 = images_bs2.images
            else:
                actual_images_bs2 = images_bs2

            assert len(actual_images_bs2) == 2, f"Expected 2 images for batch_size=2, got {len(actual_images_bs2)}"
            logger.info("Batch_size=2 dynamic shape inference successful")

            # Save images for batch_size=2
            for i, image in enumerate(actual_images_bs2):
                output_path_bs2 = output_dir / f"stable_diffusion_bs2_image_{i}.jpg"
                image.save(output_path_bs2)
                logger.info("Saved batch_size=2 image %s to %s", i, output_path_bs2)

            # Step 10: Verify that all images were generated successfully
            saved_images = list(output_dir.glob("*.jpg"))
            assert len(saved_images) == 3, f"Expected 3 saved images, found {len(saved_images)}"

            # Verify image file sizes (basic sanity check)
            for image_path in saved_images:
                assert image_path.stat().st_size > 1000, f"Image {image_path} seems too small"

            logger.info("All dynamic shape inference tests passed successfully!")
            logger.info("Generated images saved to: %s", output_dir)

        except Exception as e:
            logger.error("Test failed with error: %s", e)
            raise
        finally:
            # Cleanup
            MODULE_REGISTRY.clear()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    test_stable_diffusion_dynamic_batch_tensorrt_dynamo()
