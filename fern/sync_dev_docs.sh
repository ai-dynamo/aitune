#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Sync the docs of an aitune checkout into the dev version of a docs-website checkout.
# Same steps as sync_dev_docs in ai-tune-ci docs.gitlab-ci.yml (release-docs).
# Usage: fern/sync_dev_docs.sh <aitune-checkout> <docs-website-checkout>
set -euo pipefail

src=$(realpath "$1")
dst=$(realpath "$2")/fern
test -f "$dst/docs.yml"

rm -rf "$dst/pages-dev"
mkdir -p "$dst/versions"
cp -r "$src/docs" "$dst/pages-dev"
find "$dst/pages-dev" \( -name digest -o -name pypi.rst -o -name index.yml \) -prune -exec rm -rf {} +

sed -E \
  -e 's#^(\s*-?\s*(path|folder): )\.\./docs/#\1../pages-dev/#' \
  -e 's#\.\./examples/#../pages-dev/examples/#g' \
  -e 's#\.\./(CHANGELOG\.md|CONTRIBUTING\.md|LICENSE)#../pages-dev/\1#g' \
  "$src/docs/index.yml" >"$dst/versions/dev.yml"

for file in fern.config.json .gitignore convert_callouts.py main.css; do
  if [ -f "$src/fern/$file" ]; then cp "$src/fern/$file" "$dst/$file"; fi
done

if [ -d "$src/docs/_components" ]; then rm -rf "$dst/_components"; fi
for dir in fern/components docs/assets; do
  if [ -d "$src/$dir" ]; then
    rm -rf "${dst:?}/$(basename "$dir")"
    cp -r "$src/$dir" "$dst/$(basename "$dir")"
  fi
done

(cd "$src" && find examples -name '*.md' -exec cp --parents {} "$dst/pages-dev/" \;)

for file in CHANGELOG.md CONTRIBUTING.md LICENSE; do
  if [ -f "$src/$file" ]; then cp "$src/$file" "$dst/pages-dev/$file"; fi
done
