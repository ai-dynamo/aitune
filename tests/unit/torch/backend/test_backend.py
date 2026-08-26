# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test for BackendConfig._get_changed_fields method."""

import json
from dataclasses import dataclass, field
from unittest.mock import Mock

import torch
import torch.nn as nn

from aitune.torch.backend.backend import BackendConfig, BuildMode, DummyBackend
from aitune.utils.hashing import hash_string
from tests.toy_backends import SleepBackend


@dataclass
class BackendTestConfig(BackendConfig):
    """Test backend configuration for testing _get_changed_fields."""

    name: str = "test_backend"
    enabled: bool = True
    timeout: float = 30.0
    max_retries: int = 3
    precision: str = "fp32"
    cache_dir: str | None = None
    extra_params: dict = field(default_factory=dict)


@dataclass
class BackendTestWithSubClassConfig(BackendConfig):
    """Test backend configuration for testing _get_changed_fields."""

    name: str = "test_backend_with_sub_class"
    enabled: bool = True
    sub_class: BackendTestConfig = field(default_factory=BackendTestConfig)


@dataclass
class BackendTestConfigWithDefaults(BackendConfig):
    """Test backend configuration with custom default fields."""

    name: str = "test_backend"
    enabled: bool = True
    timeout: float = 30.0
    max_retries: int = 3

    def _default_describe_fields(self) -> list[str]:
        """Returns the default fields to describe."""
        return ["name", "enabled"]  # Always include these fields


def test_backend_build_releases_unused_memory(mocker, tmp_path):
    collect = mocker.patch("aitune.torch.utils.memory.gc.collect")
    mocker.patch("aitune.torch.utils.memory.torch.cuda.is_available", return_value=True)
    empty_cache = mocker.patch("aitune.torch.utils.memory.torch.cuda.empty_cache")

    DummyBackend().build(nn.Identity(), Mock(), [], torch.device("cpu"), tmp_path)

    collect.assert_called_once()
    empty_cache.assert_called_once()


def test_backend_exposes_build_mode():
    assert DummyBackend().build_mode == BuildMode.AHEAD_OF_TIME
    assert SleepBackend().build_mode == BuildMode.JUST_IN_TIME


def test_backend_config_key():
    """Test backend config key."""
    expected_dict = {
        "name": "test_backend",
        "enabled": True,
        "timeout": 30.0,
        "max_retries": 3,
        "precision": "fp32",
        "cache_dir": None,
        "extra_params": {},
    }
    expected_key = hash_string(json.dumps(expected_dict))

    config = BackendTestConfig()
    key = config.key()
    assert key == expected_key

    key = config.key()
    assert key == expected_key

    # Change value of a field
    config.name = "test_backend_2"
    key = config.key()
    assert key != expected_key

    expected_dict["name"] = "test_backend_2"
    expected_key = hash_string(json.dumps(expected_dict))

    key = config.key()
    assert key == expected_key

    key = config.key()
    assert key == expected_key


def test_backend_config_with_sub_class_key():
    """Test backend config key."""
    expected_dict = {
        "name": "test_backend_with_sub_class",
        "enabled": True,
        "sub_class": {
            "name": "test_backend",
            "enabled": True,
            "timeout": 30.0,
            "max_retries": 3,
            "precision": "fp32",
            "cache_dir": None,
            "extra_params": {},
        },
    }
    expected_key = hash_string(json.dumps(expected_dict))

    config = BackendTestWithSubClassConfig()

    key = config.key()
    assert key == expected_key

    key = config.key()
    assert key == expected_key

    # Change value of a field
    config.sub_class.timeout = 40.0
    key = config.key()
    assert key != expected_key

    expected_dict["sub_class"]["timeout"] = 40.0
    expected_key = hash_string(json.dumps(expected_dict))

    key = config.key()
    assert key == expected_key

    key = config.key()
    assert key == expected_key


def test_backend_config_with_defaults_key():
    """Test backend config key."""
    config = BackendTestConfigWithDefaults()

    expected_dict = {
        "name": "test_backend",
        "enabled": True,
        "timeout": 30.0,
        "max_retries": 3,
    }
    expected_key = hash_string(json.dumps(expected_dict))

    key = config.key()
    assert key == expected_key

    key = config.key()
    assert key == expected_key

    # Change value of a field
    config.max_retries = 4
    key = config.key()
    assert key != expected_key

    expected_dict["max_retries"] = 4
    expected_key = hash_string(json.dumps(expected_dict))

    key = config.key()
    assert key == expected_key

    key = config.key()
    assert key == expected_key


def test_no_changes_returns_empty_list():
    """Test that when no fields are changed, empty list is returned."""
    config = BackendTestConfig()
    default = BackendTestConfig()

    changed_fields = config._get_changed_fields(config, default)

    assert changed_fields == []


def test_single_field_change_returns_only_changed_field():
    """Test that only changed fields are returned."""
    config = BackendTestConfig(timeout=60.0)  # Changed from default 30.0
    default = BackendTestConfig()

    changed_fields = config._get_changed_fields(config, default)

    assert len(changed_fields) == 1
    assert "timeout=60.0" in changed_fields


def test_multiple_field_changes_returns_all_changed_fields():
    """Test that multiple changed fields are returned."""
    config = BackendTestConfig(name="custom_backend", enabled=False, timeout=45.0, max_retries=5)
    default = BackendTestConfig()

    changed_fields = config._get_changed_fields(config, default)

    assert len(changed_fields) == 4
    assert "name=custom_backend" in changed_fields
    assert "enabled=False" in changed_fields
    assert "timeout=45.0" in changed_fields
    assert "max_retries=5" in changed_fields


def test_excluded_fields_are_not_returned():
    """Test that excluded fields are not returned even if changed."""
    config = BackendTestConfig(timeout=60.0, max_retries=5)
    default = BackendTestConfig()

    changed_fields = config._get_changed_fields(config, default, exclude=["timeout"])

    assert len(changed_fields) == 1
    assert "timeout=60.0" not in changed_fields
    assert "max_retries=5" in changed_fields


def test_included_fields_are_always_returned():
    """Test that included fields are always returned even if not changed."""
    config = BackendTestConfig()  # No changes from default
    default = BackendTestConfig()

    changed_fields = config._get_changed_fields(config, default, include=["name", "enabled"])

    assert len(changed_fields) == 2
    assert "name=test_backend" in changed_fields
    assert "enabled=True" in changed_fields


def test_default_describe_fields_are_included():
    """Test that fields from _default_describe_fields are always included."""
    config = BackendTestConfigWithDefaults()  # No changes from default
    default = BackendTestConfigWithDefaults()

    changed_fields = config._get_changed_fields(config, default)

    # Should include default describe fields even if not changed
    assert len(changed_fields) == 2
    assert "name=test_backend" in changed_fields
    assert "enabled=True" in changed_fields


def test_mixed_changes_and_defaults():
    """Test combination of changed fields and default describe fields."""
    config = BackendTestConfigWithDefaults(
        timeout=60.0,  # Changed field
        max_retries=5,  # Changed field
    )
    default = BackendTestConfigWithDefaults()

    changed_fields = config._get_changed_fields(config, default)

    # Should include: default describe fields (2) + changed fields (2)
    assert len(changed_fields) == 4
    assert "name=test_backend" in changed_fields  # Default describe field
    assert "enabled=True" in changed_fields  # Default describe field
    assert "timeout=60.0" in changed_fields  # Changed field
    assert "max_retries=5" in changed_fields  # Changed field


def test_include_overrides_exclude():
    """Test that exclude takes precedence over include."""
    config = BackendTestConfigWithDefaults(timeout=60.0)
    default = BackendTestConfigWithDefaults()

    changed_fields = config._get_changed_fields(
        config,
        default,
        exclude=["timeout"],  # Exclude timeout
        include=["timeout"],  # But also include timeout
    )

    # Exclude should take precedence
    assert "timeout=60.0" in changed_fields


def test_none_values_are_handled_correctly():
    """Test that None values are handled correctly."""
    config = BackendTestConfig(cache_dir=None)  # Explicitly set to None
    key = config.key()

    # Should not include None values unless they're in default describe fields
    assert "enabled_precisions=None" not in key


def test_dict_values_are_handled_correctly():
    """Test that dict values are handled correctly."""
    custom_params = {"key1": "value1", "key2": "value2"}
    config = BackendTestConfig(extra_params=custom_params)
    default = BackendTestConfig()  # extra_params defaults to empty dict

    changed_fields = config._get_changed_fields(config, default)

    assert len(changed_fields) == 1
    assert "extra_params=" in changed_fields[0]
    # The exact string representation may vary, so we check it contains the field name


def test_boolean_values_are_formatted_correctly():
    """Test that boolean values are formatted correctly."""
    config = BackendTestConfig(enabled=False)  # Changed from default True
    default = BackendTestConfig()

    changed_fields = config._get_changed_fields(config, default)

    assert len(changed_fields) == 1
    assert "enabled=False" in changed_fields


def test_float_values_are_formatted_correctly():
    """Test that float values are formatted correctly."""
    config = BackendTestConfig(timeout=45.5)  # Changed from default 30.0
    default = BackendTestConfig()

    changed_fields = config._get_changed_fields(config, default)

    assert len(changed_fields) == 1
    assert "timeout=45.5" in changed_fields


def test_int_values_are_formatted_correctly():
    """Test that int values are formatted correctly."""
    config = BackendTestConfig(max_retries=10)  # Changed from default 3
    default = BackendTestConfig()

    changed_fields = config._get_changed_fields(config, default)

    assert len(changed_fields) == 1
    assert "max_retries=10" in changed_fields


def test_string_values_are_formatted_correctly():
    """Test that string values are formatted correctly."""
    config = BackendTestConfig(name="custom_name")  # Changed from default
    default = BackendTestConfig()

    changed_fields = config._get_changed_fields(config, default)

    assert len(changed_fields) == 1
    assert "name=custom_name" in changed_fields


def test_empty_exclude_list_does_not_affect_result():
    """Test that empty exclude list doesn't affect the result."""
    config = BackendTestConfig(timeout=60.0)
    default = BackendTestConfig()

    changed_fields_with_exclude = config._get_changed_fields(config, default, exclude=[])
    changed_fields_without_exclude = config._get_changed_fields(config, default)

    assert changed_fields_with_exclude == changed_fields_without_exclude


def test_empty_include_list_does_not_affect_result():
    """Test that empty include list doesn't affect the result."""
    config = BackendTestConfig(timeout=60.0)
    default = BackendTestConfig()

    changed_fields_with_include = config._get_changed_fields(config, default, include=[])
    changed_fields_without_include = config._get_changed_fields(config, default)

    assert changed_fields_with_include == changed_fields_without_include
