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
"""Client for AI Dynamo service with ResNet model."""

import argparse
import base64
import logging
from pathlib import Path
from typing import Any

import requests  # pytype: disable=import-error,pyi-error

logger = logging.getLogger(__name__)


def encode_image_to_base64(image_path: Path) -> str:
    """Encode image file to base64 string."""
    with image_path.open("rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def predict_image(image_data: str, prediction_url: str = "http://localhost:8000/predict") -> Any:
    """Send prediction request to ResNet service."""
    headers = {"accept": "text/event-stream", "Content-Type": "application/json"}
    payload = {"image_data": image_data}

    try:
        response = requests.post(prediction_url, headers=headers, json=payload, stream=True)
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                print(line.decode("utf-8"))  # noqa: T201

    except requests.exceptions.RequestException as e:
        logger.error("Request failed: %s", e)
        raise


def main() -> None:
    """Main function to handle command line arguments and execute prediction."""
    parser = argparse.ArgumentParser(description="ResNet inference client")
    parser.add_argument(
        "--image-path",
        type=str,
        required=True,
        help="Path to the image file for prediction",
    )
    parser.add_argument(
        "--prediction-url",
        type=str,
        default="http://localhost:8000/predict",
        help="URL of the ResNet AI Dynamo inference service (default: http://localhost:8000/predict)",
    )

    args = parser.parse_args()

    image_path = Path(args.image_path)

    # Validate image file exists
    if not image_path.exists():
        logger.error("Image file not found: %s", image_path)
        return

    try:
        # Encode image to base64
        image_data = encode_image_to_base64(image_path)

        # Send prediction request
        predict_image(image_data, args.prediction_url)

    except Exception as e:
        logger.error("Prediction failed: %s", e)


if __name__ == "__main__":
    main()
