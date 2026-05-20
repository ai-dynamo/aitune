---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
name: aitune-docs-update
description: Run full documentation sync (not limited to recent git changes)
allowed-tools: Read, Write, Edit, Grep, Glob, Bash
license: Apache-2.0
---

Run the docs-auditor agent in **full mode** to perform a comprehensive documentation audit across the entire project.

**Focus area** (optional): $ARGUMENTS

If a focus area is provided, prioritize that area in the report and edits but still perform the full scan for context. If no focus area is provided, give equal attention to all areas.

**Two-step workflow with user review:**

1. **Run Phase 1 (Report) only** using the docs-auditor agent. Present the full audit report with proposed changes to the user.
2. **Stop and wait for user review.** Use AskUserQuestion to ask the user which proposed changes to proceed with.
3. **Run Phase 2 (Edit)** only after the user responds, executing only the approved changes.

Do NOT run Phase 2 automatically. The user must review and approve proposed changes first.
