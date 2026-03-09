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
"""Sample audio data for the Parakeet CTC example."""

import logging
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

SAMPLE_AUDIO_URL = "https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav"
SAMPLE_AUDIO_FILENAME = "2086-149220-0033.wav"


def ensure_sample_audio(target_dir: Path | None = None) -> Path:
    """Download the sample audio file if it does not already exist.

    Args:
        target_dir: Directory to download the file into.
            Defaults to the example root directory (one level above this package).

    Returns:
        Path to the sample audio file.
    """
    if target_dir is None:
        target_dir = Path(__file__).parent.parent

    target = target_dir / SAMPLE_AUDIO_FILENAME
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading sample audio from %s ...", SAMPLE_AUDIO_URL)
        urllib.request.urlretrieve(SAMPLE_AUDIO_URL, target)  # noqa: S310
        logger.info("Saved to %s", target)
    return target
