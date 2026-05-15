---
name: aitune-commit
description: Use when creating a git commit to ensure the message follows the Conventional Commits specification
license: Apache-2.0
---
<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Conventional Commit

## Workflow

1. Run `git diff HEAD`, `git log --oneline`, `git status` to understand the changes
2. Compose **one** commit message covering all staged changes
3. Present it to the user for confirmation — do NOT commit yet
4. After confirmation, run `git commit -m "..."`
5. After the commit succeeds, suggest one MR title that summarises the branch's overall purpose (derived from `git log origin/main..HEAD --oneline`). Rules:
   - Follow Conventional Commits format: `type(scope): description`
   - Pick the **dominant** type (most impactful: feat > fix > refactor > chore/docs/test)
   - Pick the **primary** scope (most commits, or most user-visible change)
   - If commits span unrelated areas, omit scope and describe the branch goal abstractly
   - Present as: `**MR title suggestion:** <title>`

**Never offer multiple commit options or ask whether to split.** Always produce a single message.

## Format

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

## Types

| Type | When to use |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `chore` | Maintenance, deps, tooling (no prod code change) |
| `docs` | Documentation only |
| `refactor` | Code change that is neither fix nor feature |
| `test` | Adding or fixing tests |
| `perf` | Performance improvement |
| `ci` | CI/CD configuration changes |
| `revert` | Revert a previous commit |

## Rules

- **type** and **description** are required
- **scope** is optional — use the affected module/area (e.g. `feat(auth):`)
- Description is lowercase, imperative mood, no period at end
- **Single line only** — no body, no footer
- Keep it short and informative — aim for 50–72 chars, hard limit 120
- Breaking changes: add `!` after type/scope (`feat!:`)
- Exception: `BREAKING CHANGE:` footer only if `!` alone is insufficient

## Examples

```
feat(plan): add diffusers benchmark group
fix(ci): correct needs dependency in generate job
chore: update aitune-benchmarks to main branch
docs: add plan file format to CLAUDE.md
refactor(builds): consolidate hub Dockerfiles
feat!: rename platform key for Colossus GPU

BREAKING CHANGE: platform key changed from RTX-PRO-6000 to RTX-PRO-6000-MAXQ-PCIE-96GB-Colossus
```

## Common Mistakes

- `fix: Fixed the bug` → should be `fix: fix the bug` (imperative, lowercase)
- `feat: add new feature.` → no trailing period
- `update: something` → `update` is not a valid type; use `chore` or `refactor`
- First line > 72 chars → move detail to body
- Offering multiple messages or asking to split → always pick one and present it
