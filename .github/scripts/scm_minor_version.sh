#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -eu

# Keep release builds exact, even when a newer release exists on another branch.
exact=$(git tag --points-at HEAD --sort=-version:refname | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | head -n 1)
tag=${exact:-$(git tag --sort=-version:refname | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | head -n 1)}
if [ -z "$tag" ]; then
    exec git describe --dirty --tags --long --match '*[0-9]*'
fi

distance=$(git rev-list --count "$tag..HEAD")
# An older checkout must not masquerade as a release it does not contain.
if [ -z "$exact" ] && [ "$distance" -eq 0 ]; then
    distance=1
fi
node=$(git rev-parse --short HEAD)
dirty=
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    dirty=-dirty
fi

next_tag=$(echo "$tag" | awk -F. -v OFS=. '{
    sub(/^v/, "", $1);
    printf("v%d.%d.%d", $1, $2, $3+1);
}')

# example v0.6.1.dev18+g449503ba.d20260921
printf '%s.dev%s+g%s.d%s\n' "$next_tag" "$distance" "$node" "$(date +%Y%m%d)"
