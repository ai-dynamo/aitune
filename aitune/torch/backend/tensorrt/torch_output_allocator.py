# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PyTorch-based TensorRT output allocator."""

import logging
import math
import sys
import traceback

import torch
from wrapt import lazy_import

trt = lazy_import("tensorrt")

# Setup logger
logger = logging.getLogger(__name__)

# Element size mapping for each data type (in bytes)
DTYPE_ELEMENT_SIZE = {
    torch.float32: 4,
    torch.float16: 2,
    torch.int32: 4,
    torch.int8: 1,
    torch.bool: 1,
    torch.uint8: 1,
    torch.bfloat16: 2,
    torch.int64: 8,
}


def _get_trt_to_torch_dtype_mapping():
    return {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
        trt.int8: torch.int8,
        trt.bool: torch.bool,
        trt.uint8: torch.uint8,
    }


class TorchOutputAllocator(trt.IOutputAllocator):
    """PyTorch-based output allocator for TensorRT.

    This allocator allows TensorRT to control output allocation and reshaping
    while using PyTorch tensors underneath. It provides better integration
    between TensorRT and PyTorch by letting TensorRT handle dynamic shapes
    while maintaining PyTorch tensor semantics.
    """

    def __init__(self, engine_info=None):
        """Initialize the PyTorch output allocator.

        Args:
            engine_info: Optional TensorRTEngineInfo object containing output dtype information
        """
        super().__init__()
        self._raw_outputs = {}
        self._output_shapes = {}
        self._engine_info = engine_info
        logger.debug("Initialized TorchOutputAllocator")

    def reallocate_output(self, tensor_name: str, memory: int, size: int, alignment: int) -> int | None:
        """Reallocate output tensor memory.

        Called by TensorRT when it needs to allocate or reallocate memory for
        an output tensor. This method creates or resizes PyTorch tensors and
        returns the memory pointer for TensorRT to use.

        Args:
            tensor_name: Name of the output tensor
            memory: Existing memory pointer (0 if new allocation)
            size: Size in bytes needed for the tensor
            alignment: Memory alignment requirement

        Returns:
            Memory pointer to the allocated tensor data
        """
        logger.debug("Reallocating output tensor '%s': size=%s, alignment=%s", tensor_name, size, alignment)
        trt_to_torch_dtype_mapping = _get_trt_to_torch_dtype_mapping()
        try:
            # Determine the data type for this tensor
            dtype = torch.float32  # Default fallback
            if self._engine_info and hasattr(self._engine_info, "output_dtypes"):
                if tensor_name in self._engine_info.output_dtypes:
                    engine_dtype = self._engine_info.output_dtypes[tensor_name]

                    # First check if it's already a torch dtype
                    if isinstance(engine_dtype, torch.dtype):
                        dtype = engine_dtype
                        logger.debug("Using torch dtype %s for tensor '%s' from engine info", dtype, tensor_name)
                    # Otherwise try to map from TensorRT dtype to torch dtype
                    elif engine_dtype in trt_to_torch_dtype_mapping:
                        dtype = trt_to_torch_dtype_mapping[engine_dtype]
                        logger.debug(
                            "Mapped TensorRT dtype %s to torch dtype %s for tensor '%s'",
                            engine_dtype,
                            dtype,
                            tensor_name,
                        )
                    else:
                        raise ValueError("Unknown dtype '%s' for tensor '%s'", engine_dtype, tensor_name)

            # Calculate number of elements based on size and dtype
            element_size = DTYPE_ELEMENT_SIZE[dtype]
            num_elements = size // element_size

            # Create new PyTorch tensor with proper alignment
            # For dynamic shapes, we create a 1D tensor and let TensorRT handle reshaping
            # Torch empty return contiguous tensor
            tensor = torch.empty(num_elements, dtype=dtype, device="cuda")

            # Store the tensor so it doesn't get garbage collected
            self._raw_outputs[tensor_name] = tensor

            # Get memory pointer
            ptr = tensor.data_ptr()
            logger.debug("Allocated tensor '%s' with %s elements (%s) at ptr=%s", tensor_name, num_elements, dtype, ptr)

            return ptr

        except Exception as e:
            exc_type, _, exc_traceback = sys.exc_info()
            logger.debug(
                "Failed to reallocate output tensor '%s': %s\nException type: %s\nTraceback:\n%s",
                tensor_name,
                e,
                exc_type.__name__,
                "".join(traceback.format_tb(exc_traceback)),
            )

            return 0

    def notify_shape(self, tensor_name: str, shape: trt.Dims) -> None:
        """Notify the allocator of the actual output shape.

        Called by TensorRT to inform the allocator of the actual shape
        of an output tensor after inference. This method only saves the
        shape information; reshaping happens when outputs are accessed.

        Args:
            tensor_name: Name of the output tensor
            shape: Actual shape of the tensor as determined by TensorRT
        """
        logger.debug("Shape notification for '%s': %s", tensor_name, shape)

        # Convert TensorRT Dims to tuple and store
        shape_tuple = tuple(shape)
        self._output_shapes[tensor_name] = shape_tuple

    @property
    def outputs(self) -> dict[str, torch.Tensor]:
        """Get the allocated output tensors with proper shapes.

        This property performs reshaping on access, using the shape information
        provided by notify_shape calls.

        Returns:
            Dictionary mapping tensor names to PyTorch tensors with correct shapes
        """
        reshaped_outputs = {}

        for tensor_name, tensor in self._raw_outputs.items():
            if tensor_name in self._output_shapes:
                shape_tuple = self._output_shapes[tensor_name]
                try:
                    # Calculate total elements in the target shape
                    total_elements = math.prod(shape_tuple)

                    # Only reshape if we have enough elements
                    if tensor.numel() >= total_elements:
                        # Create a view with the correct shape
                        reshaped_tensor = tensor[:total_elements].view(shape_tuple)
                        reshaped_outputs[tensor_name] = reshaped_tensor
                        logger.debug("Reshaped tensor %s to %s", tensor_name, shape_tuple)
                    else:
                        logger.debug("Tensor %s has insufficient elements for shape %s", tensor_name, shape_tuple)
                        reshaped_outputs[tensor_name] = tensor

                except Exception as e:
                    logger.debug("Failed to reshape tensor '%s' to %s: %s", tensor_name, shape_tuple, e)
                    reshaped_outputs[tensor_name] = tensor
            else:
                # No shape information available, return original tensor
                reshaped_outputs[tensor_name] = tensor

        return reshaped_outputs

    def clear(self) -> None:
        """Clear all allocated outputs.

        This should be called after inference to free memory.
        """
        logger.debug("Clearing output allocator")
        self._raw_outputs.clear()
        self._output_shapes.clear()

    @property
    def output_shapes(self) -> dict[str, tuple[int, ...]]:
        """Get the shapes of all output tensors.

        Returns:
            Dictionary mapping tensor names to their actual shapes
        """
        return self._output_shapes
