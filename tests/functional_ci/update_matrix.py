#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read or update the matrix-data JSON block in the functional features matrix HTML.

  update_matrix.py matrix.html --dump > data.json      # print the current data
  update_matrix.py matrix.html updates.json            # merge updates into the file

updates.json: {"run": {...}, "cells": {"<cell id>": {"state": "failed", "note": "...", "run": "..."}}}
Only the given cells and keys change. "run" replaces the run metadata.
"""

import argparse
import json
import re
import sys
from pathlib import Path

STATES = {"covered", "weak", "legacy", "unit", "known", "gap", "na", "passed", "failed", "skipped"}
BLOCK = re.compile(r'(<script type="application/json" id="matrix-data">)(.*?)(</script>)', re.S)


def read_matrix(path: Path) -> dict:
    """Return the matrix-data JSON object from `path`."""
    text = path.read_text(encoding="utf-8")
    match = BLOCK.search(text)
    if not match:
        sys.exit(f"{path}: no matrix-data block")
    return json.loads(match.group(2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("html", type=Path)
    parser.add_argument("updates", type=Path, nargs="?")
    parser.add_argument("--dump", action="store_true", help="print the current matrix-data JSON and exit")
    args = parser.parse_args()

    text = args.html.read_text(encoding="utf-8")
    data = read_matrix(args.html)
    if args.dump:
        print(json.dumps(data, indent=1))
        return
    if not args.updates:
        parser.error("give an updates file or --dump")

    updates = json.loads(args.updates.read_text(encoding="utf-8"))
    for cell_id, change in updates.get("cells", {}).items():
        if cell_id not in data["cells"]:
            sys.exit(f"unknown cell id: {cell_id}")
        if change.get("state", data["cells"][cell_id]["state"]) not in STATES:
            sys.exit(f"{cell_id}: unknown state {change['state']!r}, expected one of {sorted(STATES)}")
        data["cells"][cell_id].update(change)
    if "run" in updates:
        data["run"] = updates["run"]

    match = BLOCK.search(text)
    assert match
    payload = json.dumps(data, indent=1).replace("</", "<\\/")
    args.html.write_text(text[: match.start(2)] + "\n" + payload + "\n" + text[match.end(2) :], encoding="utf-8")
    print(f"updated {len(updates.get('cells', {}))} cells in {args.html}")


if __name__ == "__main__":
    main()
