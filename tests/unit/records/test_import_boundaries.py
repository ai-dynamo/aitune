# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tests.unit.records.import_boundaries import direct_violations


def test_records_import_only_stdlib_and_records_modules():
    violations = direct_violations()

    assert not violations, "Invalid aitune.records imports:\n" + "\n".join(violations)


def test_checker_rejects_third_party_and_other_aitune_imports(tmp_path):
    source = tmp_path / "forbidden.py"
    source.write_text("import numpy\nfrom aitune import torch\n")

    violations = direct_violations([source])
    violations.sort()

    assert len(violations) == 2
    assert "numpy" in violations[0]
    assert "aitune.torch" in violations[1]
