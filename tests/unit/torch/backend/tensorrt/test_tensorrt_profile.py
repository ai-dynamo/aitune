# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TensorRTProfile."""

import pytest

from aitune.exceptions import AITuneUserInputError
from aitune.torch.backend.tensorrt.tensorrt_profile import TensorRTProfile
from aitune.torch.utils.tensor import format_tensor_name


@pytest.fixture
def mock_polygraphy_profile(mocker):
    """Fixture that mocks Polygraphy Profile."""
    mock_profile = mocker.patch("aitune.torch.backend.tensorrt.tensorrt_profile.Profile")
    mock_profile_instance = mock_profile.return_value
    mock_profile_instance.add.return_value = mock_profile_instance
    return mock_profile_instance


def test_tensorrt_profile_init():
    """Test TensorRTProfile initialization."""
    profile = TensorRTProfile()
    assert profile is not None


def test_tensorrt_profile_add_input_shape(mock_polygraphy_profile):
    """Test add_input_shape method."""
    # Test data
    input_path = ("inputs", 0)
    min_shape = (1, 3, 224, 224)
    opt_shape = (4, 3, 224, 224)
    max_shape = (8, 3, 224, 224)

    # Create profile and add shape
    profile = TensorRTProfile()
    profile._profile = mock_polygraphy_profile  # Replace with mock
    result = profile.add_input_shape(input_path, min_shape, opt_shape, max_shape)

    # Verify interactions
    mock_polygraphy_profile.add.assert_called_once_with(
        name=format_tensor_name(input_path, "input"), min=min_shape, opt=opt_shape, max=max_shape
    )

    # Verify chainable API
    assert result == profile


def test_tensorrt_profile_rejects_invalid_input_path():
    with pytest.raises(AITuneUserInputError, match="Forward input path"):
        TensorRTProfile().add_input_shape(("", 0), (1,), (2,), (4,))


def test_tensorrt_profile_property():
    """Test profile property."""
    profile = TensorRTProfile()
    assert profile.profile is profile._profile


def test_tensorrt_profile_round_trip_preserves_formatted_name():
    profile = TensorRTProfile().add_input_shape(("inputs", 0), (1, 3), (2, 3), (4, 3))

    restored = TensorRTProfile.from_dict(profile.to_dict())

    assert restored == profile


def test_tensorrt_profile_str(mocker):
    """Test string representation."""
    mock_profile = mocker.patch("aitune.torch.backend.tensorrt.tensorrt_profile.Profile")
    mock_profile_instance = mock_profile.return_value
    mock_profile_instance.__str__.return_value = "MockProfileString"

    profile = TensorRTProfile()
    assert str(profile) == "MockProfileString"


def test_tensorrt_profile_repr(mocker):
    """Test repr representation."""
    mock_profile = mocker.patch("aitune.torch.backend.tensorrt.tensorrt_profile.Profile")
    # Create a MagicMock with a custom __repr__ method
    mock_instance = mocker.MagicMock()
    # The self parameter is automatically passed when the method is called
    mock_instance.__repr__ = lambda *args: "MockProfileRepr"
    mock_profile.return_value = mock_instance

    profile = TensorRTProfile()
    assert repr(profile) == "MockProfileRepr"


def test_tensorrt_profile_multiple_inputs(mock_polygraphy_profile):
    """Test adding multiple input shapes to a profile."""
    # Create profile
    profile = TensorRTProfile()
    profile._profile = mock_polygraphy_profile  # Replace with mock

    # Add first input shape
    input1_name = "input1"
    input1_min = (1, 3, 224, 224)
    input1_opt = (4, 3, 224, 224)
    input1_max = (8, 3, 224, 224)

    # Add second input shape
    input2_name = "input2"
    input2_min = (1, 1, 112, 112)
    input2_opt = (4, 1, 112, 112)
    input2_max = (8, 1, 112, 112)

    # Add both shapes to the profile
    profile.add_input_shape(input1_name, input1_min, input1_opt, input1_max)
    profile.add_input_shape(input2_name, input2_min, input2_opt, input2_max)

    # Verify both inputs were added to the profile
    assert mock_polygraphy_profile.add.call_count == 2

    # Check first call arguments
    first_call = mock_polygraphy_profile.add.call_args_list[0]
    assert first_call[1]["name"] == format_tensor_name(input1_name, "input")
    assert first_call[1]["min"] == input1_min
    assert first_call[1]["opt"] == input1_opt
    assert first_call[1]["max"] == input1_max

    # Check second call arguments
    second_call = mock_polygraphy_profile.add.call_args_list[1]
    assert second_call[1]["name"] == format_tensor_name(input2_name, "input")
    assert second_call[1]["min"] == input2_min
    assert second_call[1]["opt"] == input2_opt
    assert second_call[1]["max"] == input2_max
