#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Add the docs of an aitune tag checkout as a release version of a docs-website checkout.
# Same steps as release-docs in ai-tune-ci docs.gitlab-ci.yml, without the dev sync (nightly does it).
# Usage: .github/scripts/fern/sync_tag_docs.sh <aitune-checkout> <docs-website-checkout> <vX.Y.Z>
set -euo pipefail

src=$(realpath "$1")
dst=$(realpath "$2")/fern
tag=$3
[[ $tag =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "Not a release tag: $tag" >&2; exit 1; }
test -f "$dst/docs.yml"
if [ -e "$dst/pages-$tag" ] || [ -e "$dst/versions/$tag.yml" ] || grep -q "slug: $tag$" "$dst/docs.yml"; then
  echo "Version $tag already exists in docs-website" >&2
  exit 1
fi

# Build the snapshot with the dev sync in a scratch dir, so shared files and pages-dev stay as they are.
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
mkdir "$tmp/fern" && touch "$tmp/fern/docs.yml"
"$(dirname "$0")/sync_dev_docs.sh" "$src" "$tmp"

mv "$tmp/fern/pages-dev" "$dst/pages-$tag"
sed "s#\.\./pages-dev/#../pages-$tag/#g" "$tmp/fern/versions/dev.yml" >"$dst/versions/$tag.yml"
find "$dst/pages-$tag" \( -name '*.md' -o -name '*.mdx' \) -exec sed -i -E \
  "s#github\.com/ai-dynamo/aitune/(tree|blob)/main#github.com/ai-dynamo/aitune/\1/$tag#g" {} +

newest=$(find "$dst/versions" -name 'v*.yml' -printf '%f\n' | sed 's/\.yml$//' | sort -V | tail -n 1)
latest_args=()
if [ "$newest" = "$tag" ]; then
  latest_args=(
    -e "s#^(    path: \./versions/).*#\1$tag.yml#"
    -e "/^      - display-name: Latest /{s#\(.*\)#($tag)#;n;s#(path: \./versions/).*#\1$tag.yml#}"
  )
fi
sed -i -E "${latest_args[@]}" \
  -e "/^        slug: dev$/{n;s#\$#\n      - display-name: $tag\n        path: ./versions/$tag.yml\n        slug: $tag\n        availability: stable#}" \
  -e "s#^(    - \./pages-dev/_components)\$#\1\n    - ./pages-$tag/_components#" \
  "$dst/docs.yml"
grep -q "slug: $tag$" "$dst/docs.yml" && grep -q "pages-$tag/_components" "$dst/docs.yml"
