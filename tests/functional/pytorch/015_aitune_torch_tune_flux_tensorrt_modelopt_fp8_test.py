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
# dependencies = ["diffusers"]
#
# # Optional, default "always", determines how often test is generated, always, nightly, weekly, monthly
# scope = "nightly"
# additional_tags = ["gpu/a100"]
# ///

import tempfile
from logging import INFO, basicConfig, getLogger
from pathlib import Path

import torch
from diffusers import FluxPipeline

from aitune.torch import FirstWinsStrategy, inspect, tune, wrap
from aitune.torch.backend import TensorRTBackend, TorchEagerBackend, TorchInductorBackend
from aitune.torch.backend.tensorrt.tensorrt_backend import TensorRTBackendConfig
from aitune.torch.backend.tensorrt.torch_quantization import TorchQuantizationConfig
from aitune.torch.module_registry import MODULE_REGISTRY

basicConfig(level=INFO, force=True)
logger = getLogger(__name__)


def get_flux_pipeline(model_name: str = "black-forest-labs/FLUX.1-dev", device: str = "cuda"):
    """Get a pretrained Flux model from HuggingFace.

    Args:
        model_name: HuggingFace model name or path
        device: Device to load the model on

    Returns:
        FluxPipeline: The loaded Flux pipeline
    """
    pipe = FluxPipeline.from_pretrained(model_name, torch_dtype=torch.float32).to(torch.float16).to(device)
    torch.cuda.empty_cache()
    return pipe


def test_flux_tensorrt_modelopt_fp8():
    """Test FLUX pipeline with TensorRT backend and ModelOpt FP8 quantization.

    This test:
    1. Loads FLUX pipeline from HuggingFace
    2. Inspects the pipeline to identify key modules
    3. Configures TensorRT backend with FP8 quantization using ModelOpt
    4. Wraps modules with FirstWinsStrategy for FP8 quantization with FP16 fallback
    5. Tunes the model with different image sizes
    6. Tests inference with the tuned model
    7. Verifies the number of generated images
    8. Verifies the dimensions of generated images
    9. Saves generated images to verify functionality
    """
    # Test configuration
    model_name = "black-forest-labs/FLUX.1-dev"
    prompt = "A serene mountain landscape with a crystal clear lake reflecting the sky"
    sizes = [(512, 512), (768, 768)]  # Different image sizes for testing
    steps = 4  # Reduced steps for faster testing
    guidance_scale = 3.5
    max_sequence_length = 256

    logger.info("Starting FLUX TensorRT ModelOpt FP8 test")

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "generated_images"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Step 1: Load FLUX pipeline
            logger.info("Loading FLUX pipeline")
            pipeline = get_flux_pipeline(model_name=model_name)
            logger.info("FLUX pipeline loaded successfully")

            # Step 2: Inspect pipeline to identify modules
            logger.info("Inspecting pipeline modules")
            input_data = [{"prompt": prompt}]
            modules_info = inspect(pipeline, input_data, number_of_iterations=1, warmup_iterations=1)

            # Get all modules for comprehensive tuning
            modules = modules_info.get_modules()
            logger.info("Found %d modules to tune", len(modules))

            # Log key modules for debugging
            for module in modules:
                if module.total_execution_time > 0.1:  # Log modules with significant execution time
                    logger.info("Module: %s, execution time: %.2fs", module.name, module.total_execution_time)

            # Step 3: Configure TensorRT backend with FP8 quantization and fallbacks
            logger.info("Configuring TensorRT backend with FP8 quantization")
            strategy = FirstWinsStrategy(
                backends=[
                    # Primary: TensorRT with FP8 quantization
                    TensorRTBackend(
                        TensorRTBackendConfig(
                            quantization_config=TorchQuantizationConfig(
                                quantization_config="FP8_DEFAULT_CFG",
                                device="cuda",
                                verbose=False,
                            ),
                        )
                    ),
                    # Fallback 1: TensorRT without quantization (FP16)
                    TensorRTBackend(),
                    # Fallback 2: Torch Inductor backend
                    TorchInductorBackend(),
                    # Fallback 3: Torch Eager backend
                    TorchEagerBackend(),
                ]
            )
            # Disable automatic batch size finding for more predictable behavior
            strategy.enable_find_max_batch_size(enable=False)

            # Step 4: Wrap modules with the strategy
            logger.info("Wrapping modules with FirstWinsStrategy")
            pipeline = wrap(pipeline, modules, strategy=strategy)

            # Create wrapper function for tuning that handles different image sizes
            def call_wrapper(*args, **kwargs):
                """Wrapper function for pipeline calls that tests different image sizes."""
                results = []
                for height, width in sizes:
                    logger.info("Generating image with size %dx%d", height, width)
                    result = pipeline(
                        *args,
                        height=height,
                        width=width,
                        num_inference_steps=steps,
                        guidance_scale=guidance_scale,
                        max_sequence_length=max_sequence_length,
                        generator=torch.Generator("cpu").manual_seed(42),  # Fixed seed for reproducibility
                        **kwargs,
                    )
                    results.append(result)
                return results

            # Step 5: First do a dry run for testing
            logger.info("Running dry run for testing")
            tune(call_wrapper, input_data, batch_sizes=[1], dry_run=True)
            logger.info("Dry run completed successfully")

            # Step 6: Perform actual tuning
            logger.info("Tuning FLUX with TensorRT FP8 quantization")
            tune(
                call_wrapper,
                input_data,
                batch_sizes=[1],  # FLUX typically works with batch_size=1
                dry_run=False,
                disable_external_logging=False,
            )
            logger.info("Tuning completed successfully")

            # Step 7: Test inference with tuned model and generate images
            logger.info("Testing inference with tuned model")

            # Generate images for verification using the tuned pipeline
            def inference_wrapper(*args, **kwargs):
                """Wrapper function for inference testing."""
                return pipeline(
                    *args,
                    height=512,
                    width=512,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    max_sequence_length=max_sequence_length,
                    generator=torch.Generator("cpu").manual_seed(42),  # Fixed seed for reproducibility
                    **kwargs,
                )

            # Generate test image
            result = inference_wrapper(prompt=prompt)

            # Verify output structure
            assert hasattr(result, "images") or isinstance(result, (list, tuple)), "Expected images output"
            if hasattr(result, "images"):
                images = result.images
            else:
                images = result

            assert len(images) >= 1, f"Expected at least 1 image, got {len(images)}"
            logger.info("Inference test successful")

            # Step 8: Verify number of generated images
            expected_num_images = 1  # Single prompt should generate 1 image
            actual_num_images = len(images)
            assert actual_num_images == expected_num_images, (
                f"Expected {expected_num_images} images, but got {actual_num_images}"
            )
            logger.info("✓ Number of images verification passed: %d images generated", actual_num_images)

            # Step 9: Verify image dimensions
            expected_height = 512
            expected_width = 512
            for i, image in enumerate(images):
                actual_width, actual_height = image.size  # PIL Image.size returns (width, height)
                assert actual_width == expected_width, (
                    f"Image {i} width mismatch: expected {expected_width}, got {actual_width}"
                )
                assert actual_height == expected_height, (
                    f"Image {i} height mismatch: expected {expected_height}, got {actual_height}"
                )
                logger.info("✓ Image %d dimensions verification passed: %dx%d", i, actual_width, actual_height)

            # Step 10: Save generated images for verification
            for i, image in enumerate(images):
                output_path = output_dir / f"flux_fp8_generated_{i}.jpg"
                image.save(output_path)
                logger.info("Generated image %d saved to %s", i, output_path)

                # Basic verification of saved image
                assert output_path.exists(), f"Generated image file {output_path} does not exist"
                assert output_path.stat().st_size > 10000, (
                    f"Generated image seems too small: {output_path.stat().st_size} bytes"
                )

            logger.info("All FLUX TensorRT ModelOpt FP8 tests passed successfully!")
            logger.info("Generated %d images saved to: %s", len(images), output_dir)

        except Exception as e:
            logger.error("Test failed with error: %s", e)
            raise
        finally:
            # Cleanup
            MODULE_REGISTRY.clear()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    test_flux_tensorrt_modelopt_fp8()
