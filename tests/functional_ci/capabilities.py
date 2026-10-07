# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runner and software capability declarations for functional CI."""

from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class FunctionalProfile:
    """Requirements and optional feature declarations for one test run."""

    name: str
    min_gpus: int
    min_compute_capability: tuple[int, int] | None
    required_imports: tuple[str, ...]
    capabilities: frozenset[str]
    exclusions: dict[str, str]


_REQUIRED_IMPORTS = ("torch", "torch_tensorrt", "tensorrt", "onnx", "onnxruntime", "torchao", "modelopt", "polygraphy")
_KNOWN_CAPABILITIES = frozenset({"fp8", "fp4", "torch_flash_sdpa", "distributed"})
PR_PROFILE = FunctionalProfile(
    name="pr",
    min_gpus=1,
    min_compute_capability=(12, 0),
    required_imports=_REQUIRED_IMPORTS,
    capabilities=frozenset({"fp8", "fp4", "torch_flash_sdpa"}),
    exclusions={"distributed": "PR profile advertises one GPU"},
)


def _module_available(name: str) -> bool:
    """Return whether an importable top-level module is installed."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _gpu_environment() -> tuple[list[dict[str, Any]], str | None, str | None]:
    """Read GPU data without importing AITune."""
    try:
        import torch

        devices = []
        for index in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(index)
            devices.append({"index": index, "name": prop.name, "compute_capability": [prop.major, prop.minor]})
        driver = None
        try:
            from pynvml import nvmlInit, nvmlSystemGetDriverVersion

            nvmlInit()
            driver = nvmlSystemGetDriverVersion()
        except Exception:
            pass
        return devices, driver, torch.version.cuda
    except Exception:
        return [], None, None


def _mx_capability_error(capability: str) -> str | None:
    gpus, _, _ = _gpu_environment()
    if not gpus or min(tuple(gpu["compute_capability"]) for gpu in gpus) < (10, 0):
        return f"{capability} requires an SM100 or newer GPU"
    try:
        from torchao.prototype.mx_formats.inference_workflow import (
            MXDynamicActivationMXWeightConfig,
            NVFP4DynamicActivationNVFP4WeightConfig,
        )
        from torchao.quantization.quantize_.common import KernelPreference

        del MXDynamicActivationMXWeightConfig, NVFP4DynamicActivationNVFP4WeightConfig, KernelPreference
    except Exception:
        return f"{capability} requires TorchAO MX format support"
    try:
        import torch

        if not torch.cuda.is_available():
            return f"{capability} requires CUDA"
    except Exception as error:
        return f"{capability} could not verify CUDA support: {error}"
    return None


def _capability_error(profile: FunctionalProfile, capability: str) -> str | None:
    if capability in {"fp8", "fp4"}:
        return _mx_capability_error(capability)
    if capability == "torch_flash_sdpa":
        try:
            import torch

            return (
                None
                if torch.backends.cuda.is_flash_attention_available()
                else "requires PyTorch flash attention support"
            )
        except Exception as error:
            return f"torch_flash_sdpa could not verify PyTorch support: {error}"
    if capability == "distributed":
        gpus, _, _ = _gpu_environment()
        return None if len(gpus) >= 2 else "distributed requires at least two visible GPUs"
    raise ValueError(f"Unknown functional capability: {capability}")


def check_environment(profile: FunctionalProfile) -> dict[str, Any]:
    """Validate required software and advertised capabilities, then return environment facts."""
    unknown = (profile.capabilities | profile.exclusions.keys()) - _KNOWN_CAPABILITIES
    if unknown:
        raise ValueError(f"Unknown functional capability declaration: {', '.join(sorted(unknown))}")
    gpus, driver, cuda = _gpu_environment()
    if len(gpus) < profile.min_gpus:
        raise RuntimeError(f"Functional profile {profile.name!r} requires {profile.min_gpus} GPU(s), found {len(gpus)}")
    if profile.min_compute_capability and any(
        tuple(gpu["compute_capability"]) < profile.min_compute_capability for gpu in gpus
    ):
        raise RuntimeError(
            f"Functional profile {profile.name!r} requires compute capability {profile.min_compute_capability}"
        )
    missing = [name for name in profile.required_imports if not _module_available(name)]
    if missing:
        raise RuntimeError(f"Missing required dependency imports: {', '.join(missing)}")
    for capability in profile.capabilities:
        reason = _capability_error(profile, capability)
        if reason:
            raise RuntimeError(f"Profile {profile.name!r} advertises unavailable capability {capability!r}: {reason}")
    return {
        "gpus": gpus,
        "gpu_count": len(gpus),
        "driver_version": driver,
        "cuda_version": cuda,
    }


def capability_reason(profile: FunctionalProfile, capability: str) -> str | None:
    """Return the declared exclusion reason, failing if advertised support is broken."""
    if capability not in _KNOWN_CAPABILITIES:
        raise ValueError(f"Unknown functional capability: {capability}")
    if capability in profile.exclusions:
        return profile.exclusions[capability]
    if capability not in profile.capabilities:
        raise ValueError(f"Capability {capability!r} is neither advertised nor explicitly excluded")
    reason = _capability_error(profile, capability)
    if reason:
        raise RuntimeError(f"Profile {profile.name!r} advertises unavailable capability {capability!r}: {reason}")
    return None


def profile_data(profile: FunctionalProfile) -> dict[str, Any]:
    """Return a JSON-compatible profile declaration."""
    data = asdict(profile)
    data["capabilities"] = sorted(profile.capabilities)
    data["min_compute_capability"] = list(profile.min_compute_capability) if profile.min_compute_capability else None
    return data


def get_functional_profile(name: str | None = None) -> FunctionalProfile:
    """Select PR requirements in CI, or derive a local profile from visible devices."""
    import os

    selected = name or os.environ.get("AITUNE_FUNCTIONAL_PROFILE", "local")
    if selected == "pr":
        return PR_PROFILE
    if selected != "local":
        raise ValueError(f"Unknown functional profile: {selected}")
    gpus, _, _ = _gpu_environment()
    cc = tuple(min(tuple(gpu["compute_capability"]) for gpu in gpus)) if gpus else None
    capabilities = set()
    exclusions = {"distributed": "local profile has fewer than two visible GPUs"} if len(gpus) < 2 else {}
    for capability in ("fp8", "fp4", "torch_flash_sdpa"):
        if cc and _capability_error(PR_PROFILE, capability) is None:
            capabilities.add(capability)
        else:
            exclusions[capability] = "local hardware or software does not provide this capability"
    if len(gpus) >= 2:
        capabilities.add("distributed")
    return FunctionalProfile("local", 0, None, _REQUIRED_IMPORTS, frozenset(capabilities), exclusions)
