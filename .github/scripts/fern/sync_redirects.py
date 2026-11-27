# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sync the AITune-owned redirect block without rewriting the website's Fern configuration."""

import re
import sys
from pathlib import Path

SOURCE_START = "  # BEGIN AITUNE PAGE REDIRECTS\n"
SOURCE_END = "  # END AITUNE PAGE REDIRECTS\n"
REDIRECT = re.compile(r'  - source: "([^"]+)"\n    destination: "([^"]+)"\n')


def sync_redirects(source: Path, website: Path, version: str) -> None:
    """Update either dev routes or the unversioned routes that follow the latest release."""
    source_text = source.read_text()
    if SOURCE_START not in source_text:
        return  # Older tags do not own a redirect block.
    managed, _ = source_text.split(SOURCE_START, 1)[1].split(SOURCE_END, 1)
    pairs = REDIRECT.findall(managed)
    if not pairs or "".join(f'  - source: "{old}"\n    destination: "{new}"\n' for old, new in pairs) != managed:
        raise ValueError(f"Invalid managed redirect block in {source}")

    start = f"  # BEGIN AITUNE {version.upper()} REDIRECTS\n"
    end = f"  # END AITUNE {version.upper()} REDIRECTS\n"
    routes = [pair for pair in pairs if pair[0].startswith("/aitune/dev/") == (version == "dev")]
    before, separator, after = website.read_text().partition(start)
    if separator:
        _, after = after.split(end, 1)
    text = before + after
    redirect_header = re.search(r"^redirects:[ \t]*(?:\[\])?[ \t]*$", text, re.MULTILINE)
    if redirect_header is None:
        text = text.rstrip("\n") + "\n\nredirects:\n"
        redirect_header = re.search(r"^redirects:$", text, re.MULTILINE)
        assert redirect_header is not None

    for old, _ in routes:
        if re.search(rf"""^\s*- source:\s*["']?{re.escape(old)}["']?\s*$""", text, re.MULTILINE):
            raise ValueError(f"Redirect for {old} already exists outside the managed block in {website}")

    block = "".join(f'  - source: "{old}"\n    destination: "{new}"\n' for old, new in routes)
    header_end = redirect_header.end()
    suffix = text[header_end:].removeprefix("\n")
    prefix = text[: redirect_header.start()] + "redirects:\n"
    dev_end = "  # END AITUNE DEV REDIRECTS\n"
    if version == "latest" and dev_end in suffix:
        dev_block, _, suffix = suffix.partition(dev_end)
        prefix += dev_block + dev_end
    website.write_text(prefix + start + block + end + suffix)


if __name__ == "__main__":
    version_arg = sys.argv[3]
    if version_arg not in {"dev", "latest"}:
        raise ValueError(f"Unknown redirect version: {version_arg}")
    sync_redirects(Path(sys.argv[1]), Path(sys.argv[2]), version_arg)
