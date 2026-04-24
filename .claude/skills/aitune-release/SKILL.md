---
name: aitune-release
description: Use when creating a new release tag for the project
allowed-tools: Bash
license: Apache-2.0
---
<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Release

## Pre-flight checks

Before tagging, verify:

1. Working tree is clean:
   ```bash
   git status --short
   ```
   If there are uncommitted changes, stop and ask the user to commit or stash them first.

2. Current branch is `main` and up to date:
   ```bash
   git branch --show-current
   git fetch origin && git status
   ```

3. Latest pipeline on `main` is passing — check in CI/CD pipelines before tagging. A tag pushed against a failing pipeline will trigger a broken deploy. Stop and ask the user to confirm the pipeline is green.

## Workflow

1. Read the latest tag from origin:
   ```bash
   git fetch --tags origin
   git tag --sort=-version:refname | head -5
   ```

2. Ask the user: **major, minor, or patch?**

3. Compute the next version:
   - Current: `v1.2.3`
   - Major → `v2.0.0`, Minor → `v1.3.0`, Patch → `v1.2.4`

4. Propose the tag (e.g. `v1.3.0`) and wait for confirmation.

5. On confirmation, create an annotated tag and push:
   ```bash
   git tag -a v1.3.0 -m "Release v1.3.0"
   git push origin v1.3.0
   ```

Pushing the tag triggers the GitLab CI tag pipeline automatically.
