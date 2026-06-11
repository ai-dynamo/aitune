# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect

import pytest
import torch

import aitune.torch as ait
from aitune.torch.backend import TorchEagerBackend
from aitune.torch.jit.patcher import Patcher
from aitune.torch.performance.attribution_hooks import (
    _CONTEXT_ATTR,
    _discover_aot_targets,
    _discover_untuned_targets,
    _RegionInstaller,
    _RegionTarget,
    _user_defined_method_names,
)
from aitune.torch.performance.context import (
    AOT_MODULE_REGION_PREFIX,
    UNTUNED_MODULE_REGION_PREFIX,
)


class _TwoChildModel(torch.nn.Module):
    """Model with two direct nn.Module children, used for discovery tests."""

    def __init__(self):
        super().__init__()
        self.left = torch.nn.Linear(4, 4)
        self.right = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.left(x) + self.right(x)


class _PipelineLike:
    """Plain class holding nn.Module attributes and a non-Module list of modules."""

    def __init__(self):
        self.encoder = torch.nn.Linear(4, 4)
        self.decoder = torch.nn.Linear(4, 4)
        self.layers = [torch.nn.Linear(4, 4)]  # should NOT be recursed in v1
        self.config = {"key": "value"}


def test_discover_returns_direct_children_of_nn_module():
    model = _TwoChildModel()
    targets = _discover_untuned_targets(model)
    paths = sorted(target.path for target in targets)
    assert paths == ["left", "right"]
    assert all(isinstance(target.module, torch.nn.Linear) for target in targets)
    assert all(target.module_type == "torch.nn.modules.linear.Linear" for target in targets)


def test_discover_returns_direct_nn_module_attributes_of_pipeline_like():
    pipeline = _PipelineLike()
    targets = _discover_untuned_targets(pipeline)
    paths = sorted(target.path for target in targets)
    assert paths == ["decoder", "encoder"]
    # The "layers" list attribute is NOT recursed in v1; its contents should not appear.
    assert not any("layers" in target.path for target in targets)


def test_discover_skips_aot_managed_children():
    model = _TwoChildModel()
    model.left = ait.Module(
        model.left,
        "tuned-left",
        strategy=ait.OneBackendStrategy(TorchEagerBackend()).enable_find_max_batch_size(False),
    )

    targets = _discover_untuned_targets(model)
    paths = [target.path for target in targets]
    # "left" is AOT-managed (itself in MODULE_REGISTRY). Only "right" remains.
    assert paths == ["right"]


def test_discover_does_not_yield_aot_only_branch_when_no_untuned_siblings_exist():
    """A parent containing only an AOT-managed descendant yields nothing on recursion.

    The parent is not instrumented (would double-count over the wrapper); recursion
    descends into the parent but finds only the AOT-managed child, so no untuned
    region is emitted for that branch.
    """

    class _Outer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = torch.nn.Linear(4, 4)

    class _Root(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.parent = _Outer()
            self.sibling = torch.nn.Linear(4, 4)

    root = _Root()
    root.parent.inner = ait.Module(
        root.parent.inner,
        "tuned-inner",
        strategy=ait.OneBackendStrategy(TorchEagerBackend()).enable_find_max_batch_size(False),
    )

    targets = _discover_untuned_targets(root)
    paths = [target.path for target in targets]
    assert paths == ["sibling"]


def test_installer_install_and_restore():
    model = _TwoChildModel()
    targets = [
        _RegionTarget(
            path="left",
            module=model.left,
            module_type="torch.nn.modules.linear.Linear",
            annotation_prefix=UNTUNED_MODULE_REGION_PREFIX,
        ),
        _RegionTarget(
            path="right",
            module=model.right,
            module_type="torch.nn.modules.linear.Linear",
            annotation_prefix=UNTUNED_MODULE_REGION_PREFIX,
        ),
    ]

    with _RegionInstaller(targets):
        # During the context, the modules have forward hooks registered.
        # A direct forward call should still produce a valid output.
        out = model(torch.randn(1, 4))
        assert out.shape == (1, 4)

    # After __exit__, no leftover context attribute remains.
    for target in targets:
        assert not hasattr(target.module, _CONTEXT_ATTR)


def test_installer_exception_safety():
    class _Broken(torch.nn.Module):
        def forward(self, x):
            raise RuntimeError("forward failed")

    broken = _Broken()
    target = _RegionTarget(
        path="broken", module=broken, module_type="x.Broken", annotation_prefix=UNTUNED_MODULE_REGION_PREFIX
    )

    with _RegionInstaller([target]):
        with pytest.raises(RuntimeError, match="forward failed"):
            broken(torch.tensor(0.0))

    # always_call=True must have invoked the post-hook; no leftover context.
    assert not hasattr(broken, _CONTEXT_ATTR)


def test_installer_reentry_skip_emits_only_outer_span():
    """Recursive forward calls on the same module emit exactly one outer record_function span."""

    class _Recursive(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._depth = 0

        def forward(self, x):
            if self._depth == 0:
                self._depth = 1
                try:
                    return self(x)  # recursive call into the same module
                finally:
                    self._depth = 0
            return x

    recursive = _Recursive()
    target = _RegionTarget(
        path="recursive", module=recursive, module_type="x.Recursive", annotation_prefix=UNTUNED_MODULE_REGION_PREFIX
    )

    with _RegionInstaller([target]):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
            recursive(torch.tensor(0.0))

    event_names = [getattr(event, "name", "") for event in profiler.events() or []]
    # Exactly one outer span — the inner recursive entry was suppressed by the depth guard.
    assert event_names.count("aitune.performance.untuned_module:recursive") == 1
    assert not hasattr(recursive, _CONTEXT_ATTR)


def test_installer_records_function_events_in_profiler():
    """End-to-end inside a torch.profiler context: untuned regions appear in events()."""
    model = _TwoChildModel()
    targets = _discover_untuned_targets(model)

    with _RegionInstaller(targets):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
            model(torch.randn(1, 4))

    event_names = {getattr(event, "name", "") for event in profiler.events() or []}
    assert "aitune.performance.untuned_module:left" in event_names
    assert "aitune.performance.untuned_module:right" in event_names


def test_discover_returns_empty_for_non_module_non_pipeline_target():
    # A bare callable has no children and no __dict__ with nn.Module values -> no targets.
    targets = _discover_untuned_targets(lambda x: x)
    assert targets == []


def test_discover_recurses_into_root_containing_aot_managed_descendant():
    """When a top-level child contains an AOT-managed descendant, recurse into it.

    Siblings of the managed descendant at any depth should be instrumented individually
    so the parent's untuned siblings do not silently fall into residual.
    """

    class _Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tuned_branch = torch.nn.Linear(4, 4)
            self.untuned_sibling = torch.nn.Linear(4, 4)

        def forward(self, x):
            return self.tuned_branch(x) + self.untuned_sibling(x)

    class _Root(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.block = _Block()
            self.tail = torch.nn.Linear(4, 4)

        def forward(self, x):
            return self.tail(self.block(x))

    root = _Root()
    root.block.tuned_branch = ait.Module(
        root.block.tuned_branch,
        "tuned-branch",
        strategy=ait.OneBackendStrategy(TorchEagerBackend()).enable_find_max_batch_size(False),
    )

    targets = _discover_untuned_targets(root)
    paths = sorted(target.path for target in targets)
    # `block` is no longer skipped wholesale: recursion finds `block.untuned_sibling`.
    # `block.tuned_branch` is AOT-managed and skipped.
    # `tail` is a top-level clean sibling.
    assert paths == ["block.untuned_sibling", "tail"]


def test_discover_recurses_through_multiple_levels():
    """Recursion descends as deep as needed to escape the AOT subtree."""

    class _Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tuned_leaf = torch.nn.Linear(4, 4)
            self.untuned_leaf = torch.nn.Linear(4, 4)

    class _Mid(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = _Inner()
            self.helper = torch.nn.Linear(4, 4)

    class _Outer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mid = _Mid()

    outer = _Outer()
    outer.mid.inner.tuned_leaf = ait.Module(
        outer.mid.inner.tuned_leaf,
        "deep-tuned",
        strategy=ait.OneBackendStrategy(TorchEagerBackend()).enable_find_max_batch_size(False),
    )

    targets = _discover_untuned_targets(outer)
    paths = sorted(target.path for target in targets)
    # Descent: mid (has descendant) -> mid.inner (has descendant) -> mid.inner.untuned_leaf (clean).
    # mid.helper is also clean (sibling of mid.inner with no managed descendants).
    assert paths == ["mid.helper", "mid.inner.untuned_leaf"]


def test_discover_aot_targets_returns_registered_wrappers():
    class _Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4)

        def forward(self, x):
            return self.linear(x)

    wrapper = ait.Module(
        _Tiny(),
        "tiny-wrapper",
        strategy=ait.OneBackendStrategy(TorchEagerBackend()).enable_find_max_batch_size(False),
    )

    targets = _discover_aot_targets()
    paths = [target.path for target in targets]
    assert "tiny-wrapper" in paths

    target = next(target for target in targets if target.path == "tiny-wrapper")
    assert target.module is wrapper
    assert target.annotation_prefix == AOT_MODULE_REGION_PREFIX
    # module_type is the type of the underlying wrapped instance, not the wrapt proxy.
    assert target.module_type.endswith("_Tiny")


def test_discover_recurses_through_modulelist():
    """nn.ModuleList is transparent during discovery — its children become regions, not the list itself.

    nn.ModuleList's ``forward`` raises NotImplementedError; user code iterates its
    entries. Installing a hook on the list would never fire and child compute
    would silently land in residual.
    """

    class _WithList(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    targets = _discover_untuned_targets(_WithList())
    paths = sorted(target.path for target in targets)
    # Children attributed individually; the ModuleList container itself is not a region.
    assert paths == ["layers.0", "layers.1"]
    assert all(isinstance(target.module, torch.nn.Linear) for target in targets)


def test_discover_recurses_through_moduledict():
    """nn.ModuleDict is transparent during discovery, same reasoning as ModuleList."""

    class _WithDict(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.branches = torch.nn.ModuleDict({
                "left": torch.nn.Linear(4, 4),
                "right": torch.nn.Linear(4, 4),
            })

        def forward(self, x):
            return self.branches["left"](x) + self.branches["right"](x)

    targets = _discover_untuned_targets(_WithDict())
    paths = sorted(target.path for target in targets)
    assert paths == ["branches.left", "branches.right"]


def test_discover_aot_targets_instruments_all_registered_wrappers():
    """Discovery returns every registered wrapper unconditionally.

    Nested AOT suppression is per-event at classification time (see
    ``test_profile_filters_nested_aot_events`` in test_profiling),
    not at install time — install-time filtering would drop legitimate attribution
    when the inner wrapper is profiled directly without the outer firing.
    """

    class _Outer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = torch.nn.Linear(4, 4)

        def forward(self, x):
            return self.inner(x)

    outer = _Outer()
    outer.inner = ait.Module(
        outer.inner,
        "inner",
        strategy=ait.OneBackendStrategy(TorchEagerBackend()).enable_find_max_batch_size(False),
    )
    ait.Module(
        outer,
        "outer",
        strategy=ait.OneBackendStrategy(TorchEagerBackend()).enable_find_max_batch_size(False),
    )

    targets = _discover_aot_targets()
    paths = sorted(target.path for target in targets)
    # Both wrappers get hooks installed; nested suppression happens later, per event.
    assert paths == ["inner", "outer"]


def test_installer_emits_aot_annotation_during_profiler():
    """Hooks installed on an AITune Module wrapper fire through the wrapt proxy."""

    class _Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4)

        def forward(self, x):
            return self.linear(x)

    wrapper = ait.Module(
        _Tiny(),
        "emit-aot",
        strategy=ait.OneBackendStrategy(TorchEagerBackend()).enable_find_max_batch_size(False),
    )
    targets = _discover_aot_targets()
    aot_target = next(target for target in targets if target.path == "emit-aot")

    with _RegionInstaller([aot_target]):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
            wrapper(torch.randn(1, 4))

    event_names = {getattr(event, "name", "") for event in profiler.events() or []}
    assert "aitune.performance.aot_module:emit-aot" in event_names


class _ModuleWithDecode(torch.nn.Module):
    """Toy stand-in for diffusers-style modules: forward + user-defined decode/encode methods.

    ``decode`` and ``encode`` invoke an internal Linear child via __call__ so we can also
    assert correct interaction with forward hooks if needed.
    """

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(4, 4)
        self._decoded = 0

    def forward(self, x):
        return self.layer(x)

    def decode(self, x):
        return self.layer(x) + 1

    def encode(self, x):
        return self.layer(x) - 1


def test_user_defined_method_names_filters_inherited_and_dunders():
    """Returns methods declared on the subclass, excluding nn.Module-inherited names and forward."""
    module = _ModuleWithDecode()
    names = set(_user_defined_method_names(module))
    assert {"decode", "encode"}.issubset(names)
    assert "forward" not in names
    assert "__init__" not in names
    assert "__call__" not in names
    # nn.Module-inherited names must not leak through.
    assert "to" not in names
    assert "eval" not in names
    assert "parameters" not in names


def test_user_defined_method_names_skips_non_function_descriptors():
    """Properties, classmethods, and staticmethods are excluded."""

    class _WithDescriptors(torch.nn.Module):
        some_class_var = 1

        @property
        def derived(self):
            return self.some_class_var * 2

        @classmethod
        def from_config(cls, config):
            return cls()

        @staticmethod
        def helper(x):
            return x

        def regular_method(self, x):
            return x

    module = _WithDescriptors()
    names = set(_user_defined_method_names(module))
    assert "regular_method" in names
    assert "derived" not in names
    assert "from_config" not in names
    assert "helper" not in names


def test_installer_wraps_method_emits_annotation():
    """Calling a wrapped user-defined method emits an untuned_module:<path>.<method> region."""
    module = _ModuleWithDecode()
    target = _RegionTarget(path="m", module=module, module_type="x.M", annotation_prefix=UNTUNED_MODULE_REGION_PREFIX)

    with _RegionInstaller([target]):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
            module.decode(torch.randn(1, 4))

    event_names = {getattr(event, "name", "") for event in profiler.events() or []}
    assert "aitune.performance.untuned_module:m.decode" in event_names

    # After teardown the instance attribute is gone (class lookup wins again).
    assert "decode" not in vars(module)


def test_installer_method_wrap_handles_exception():
    """If a wrapped method raises, the shared record_function context is closed cleanly."""

    class _RaisingDecode(torch.nn.Module):
        def forward(self, x):
            return x

        def decode(self, x):
            raise RuntimeError("decode failed")

    module = _RaisingDecode()
    target = _RegionTarget(path="m", module=module, module_type="x.M", annotation_prefix=UNTUNED_MODULE_REGION_PREFIX)

    with _RegionInstaller([target]):
        with pytest.raises(RuntimeError, match="decode failed"):
            module.decode(torch.tensor(0.0))
        # Span closed inside the finally — no leftover shared context attribute.
        assert not hasattr(module, _CONTEXT_ATTR)


def test_installer_method_wrap_reentry_depth():
    """A method calling itself only opens one outer span (nested entry suppressed by shared guard)."""

    class _SelfCalling(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._depth = 0

        def forward(self, x):
            return x

        def decode(self, x):
            if self._depth == 0:
                self._depth = 1
                try:
                    return self.decode(x)
                finally:
                    self._depth = 0
            return x

    module = _SelfCalling()
    target = _RegionTarget(path="m", module=module, module_type="x.M", annotation_prefix=UNTUNED_MODULE_REGION_PREFIX)

    with _RegionInstaller([target]):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
            module.decode(torch.tensor(0.0))

    event_names = [getattr(event, "name", "") for event in profiler.events() or []]
    # Only one outer span for the outer call; the inner recursive call did not open a nested one.
    assert event_names.count("aitune.performance.untuned_module:m.decode") == 1
    assert not hasattr(module, _CONTEXT_ATTR)


def test_installer_method_calling_forward_shares_guard_no_double_count():
    """When a wrapped method calls self(...), the forward hook must NOT open a nested span.

    Regression guard for the same-module overlap that would otherwise double-count
    method-vs-forward time. Only the outer (method) span is emitted; the inner
    forward entry is suppressed by the shared depth counter.
    """

    class _DecodeCallsSelf(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(4, 4)

        def forward(self, x):
            return self.layer(x)

        def decode(self, x):
            return self(x) + 1  # invokes __call__ → forward hook fires on self

    module = _DecodeCallsSelf()
    target = _RegionTarget(path="m", module=module, module_type="x.M", annotation_prefix=UNTUNED_MODULE_REGION_PREFIX)

    with _RegionInstaller([target]):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
            module.decode(torch.randn(1, 4))

    event_names = [getattr(event, "name", "") for event in profiler.events() or []]
    # Outer method span is emitted; inner forward entry is suppressed by the shared guard.
    assert event_names.count("aitune.performance.untuned_module:m.decode") == 1
    assert "aitune.performance.untuned_module:m" not in event_names
    assert not hasattr(module, _CONTEXT_ATTR)


def test_installer_forward_calling_method_shares_guard_no_double_count():
    """When forward() calls a wrapped method, the method wrapper must NOT open a nested span."""

    class _ForwardCallsDecode(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(4, 4)

        def forward(self, x):
            return self.decode(x)  # invokes wrapped method while forward span is active

        def decode(self, x):
            return self.layer(x) + 1

    module = _ForwardCallsDecode()
    target = _RegionTarget(path="m", module=module, module_type="x.M", annotation_prefix=UNTUNED_MODULE_REGION_PREFIX)

    with _RegionInstaller([target]):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
            module(torch.randn(1, 4))

    event_names = [getattr(event, "name", "") for event in profiler.events() or []]
    # Outer forward span is emitted; inner method entry is suppressed.
    assert event_names.count("aitune.performance.untuned_module:m") == 1
    assert "aitune.performance.untuned_module:m.decode" not in event_names
    assert not hasattr(module, _CONTEXT_ATTR)


def test_discover_skips_jit_patched_modules():
    """JIT-managed modules are excluded from untuned discovery in V1.

    Documents current behavior: when a module is in ``Patcher._patched_modules``
    (i.e. JIT has decided to manage it), ``_is_managed`` returns True for that
    module, and untuned discovery skips it. AOT discovery (`MODULE_REGISTRY`)
    doesn't pick it up either. Net effect: the module's compute lands in
    residual.

    Closing this gap is the planned ``jit_module:`` attribution follow-up — see
    spec's Deferred Acceptance Criteria. This test locks in the current
    behavior so the follow-up MR has a clear inversion point.
    """

    class _Root(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.jit_managed = torch.nn.Linear(4, 4)
            self.untouched = torch.nn.Linear(4, 4)

        def forward(self, x):
            return self.jit_managed(x) + self.untouched(x)

    root = _Root()

    # Minimal stand-in for a JIT PatchedModule: discovery only needs
    # ``.__wrapped__`` to satisfy the ``patched.__wrapped__ is module`` check
    # in ``_is_managed``. Avoid importing the real PatchedModule (and its
    # compile-path machinery) just to test the discovery contract.
    class _FakePatched:
        def __init__(self, wrapped):
            self.__wrapped__ = wrapped

    fake = _FakePatched(root.jit_managed)
    Patcher._patched_modules.append(fake)
    try:
        targets = _discover_untuned_targets(root)
        paths = sorted(target.path for target in targets)
        # jit_managed is skipped; only the untouched sibling is discovered.
        assert paths == ["untouched"]
    finally:
        Patcher._patched_modules.remove(fake)


def test_discover_dedupes_aliased_modules():
    """Two pipeline attributes pointing to the same nn.Module yield one region, not two.

    Without dedupe, the second wrap would see the first wrapper as the "original
    instance attribute" and `_restore_methods` would leave an instance-level wrapper
    behind on teardown, leaving the module silently instrumented after profiling.
    """

    class _Aliased:
        def __init__(self):
            shared = _ModuleWithDecode()
            self.vae = shared
            self.legacy_vae = shared  # alias to the same instance

    pipeline = _Aliased()
    targets = _discover_untuned_targets(pipeline)
    assert [target.path for target in targets] == ["vae"]


def test_installer_restores_methods_cleanly_with_aliased_attributes():
    """End-to-end: after the installer exits, no wrapper remains on aliased modules."""

    class _Aliased:
        def __init__(self):
            shared = _ModuleWithDecode()
            self.vae = shared
            self.legacy_vae = shared

    pipeline = _Aliased()
    original_class_decode = type(pipeline.vae).decode

    with _RegionInstaller(_discover_untuned_targets(pipeline)):
        # While instrumented, the instance has a wrapper. Calling it works.
        _ = pipeline.vae.decode(torch.randn(1, 4))

    # After teardown, no leftover instance-level wrapper — class lookup wins.
    assert "decode" not in vars(pipeline.vae)
    assert type(pipeline.vae).decode is original_class_decode


def test_installer_skips_method_when_instance_attribute_is_not_callable():
    """If an instance shadows a class method with a non-callable, leave it alone.

    Wrapping would (1) silently change runtime behavior by replacing the non-callable
    with a callable wrapper and (2) crash on call when the wrapper tries to invoke
    a string/int/etc. Safer to skip the name entirely.
    """
    module = _ModuleWithDecode()
    module.decode = "disabled"  # type: ignore[assignment]
    target = _RegionTarget(path="m", module=module, module_type="x.M", annotation_prefix=UNTUNED_MODULE_REGION_PREFIX)

    with _RegionInstaller([target]):
        # Instance attribute is unchanged by the installer; no wrapper installed.
        assert module.decode == "disabled"

    # And remains unchanged after teardown.
    assert module.decode == "disabled"


def test_installer_method_wrap_preserves_signature_and_metadata():
    """Wrapped methods must still report their original signature/name/doc via inspect.

    Frameworks (HF transformers, diffusers) commonly introspect method signatures to
    filter kwargs or dispatch behavior. Profiling must not alter what they see.
    """

    class _WithSignature(torch.nn.Module):
        def forward(self, x):
            return x

        def decode(self, latents: torch.Tensor, *, return_dict: bool = True) -> torch.Tensor:
            """Decode latents to a sample."""
            return latents

    module = _WithSignature()
    original_signature = inspect.signature(module.decode)
    target = _RegionTarget(path="m", module=module, module_type="x.M", annotation_prefix=UNTUNED_MODULE_REGION_PREFIX)

    with _RegionInstaller([target]):
        wrapped_signature = inspect.signature(module.decode)
        assert wrapped_signature == original_signature
        assert "latents" in wrapped_signature.parameters
        assert "return_dict" in wrapped_signature.parameters
        assert module.decode.__name__ == "decode"
        assert module.decode.__doc__ == "Decode latents to a sample."


def test_installer_method_wrap_preserves_pre_existing_instance_override():
    """If a module instance already has an instance-level method patch, it is restored on __exit__."""

    module = _ModuleWithDecode()
    # User-provided instance-level patch BEFORE we install attribution hooks.
    sentinel = object()
    user_decode = lambda x: sentinel  # noqa: E731 — intentionally a plain callable
    module.decode = user_decode

    target = _RegionTarget(path="m", module=module, module_type="x.M", annotation_prefix=UNTUNED_MODULE_REGION_PREFIX)

    with _RegionInstaller([target]):
        # Our wrapper now sits in front of the user's patch; calling it goes through both.
        result = module.decode(torch.tensor(0.0))
        assert result is sentinel  # user's patch ran inside our record_function wrapper

    # After teardown, the user's instance-level patch is restored — not the class method.
    assert module.decode is user_decode


def test_installer_method_wrap_and_forward_hook_coexist():
    """Calling forward and a wrapped method on the same module produces distinct regions."""
    module = _ModuleWithDecode()
    target = _RegionTarget(path="m", module=module, module_type="x.M", annotation_prefix=UNTUNED_MODULE_REGION_PREFIX)

    with _RegionInstaller([target]):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
            module(torch.randn(1, 4))  # forward path → hook fires
            module.decode(torch.randn(1, 4))  # method path → method wrap fires

    event_names = {getattr(event, "name", "") for event in profiler.events() or []}
    assert "aitune.performance.untuned_module:m" in event_names
    assert "aitune.performance.untuned_module:m.decode" in event_names


def test_installer_does_not_wrap_aot_module_methods():
    """AOT targets get forward hooks only; method wrapping would overlap with the forward span."""

    class _TinyWithMethod(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4)

        def forward(self, x):
            return self.linear(x)

        def custom(self, x):
            return self.linear(x) * 2

    wrapper = ait.Module(
        _TinyWithMethod(),
        "no-method-wrap",
        strategy=ait.OneBackendStrategy(TorchEagerBackend()).enable_find_max_batch_size(False),
    )
    targets = _discover_aot_targets()
    aot_target = next(target for target in targets if target.path == "no-method-wrap")

    with _RegionInstaller([aot_target]):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
            wrapper.custom(torch.randn(1, 4))

    event_names = {getattr(event, "name", "") for event in profiler.events() or []}
    # No method-level region emitted for an AOT target.
    assert "aitune.performance.aot_module:no-method-wrap.custom" not in event_names


def test_installer_aot_post_hook_fires_through_wrapper_clear_restore_cycle():
    """always_call=True survives the wrapper's _restore_original_forward / _proxy_forward cycle.

    The wrapper temporarily clears _forward_pre_hooks / _forward_hooks while the dispatcher
    runs, then restores them in `finally`. If the dispatcher raises, the post-hook must
    still fire and release the per-module context attribute on its way out.
    """

    class _Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4)

        def forward(self, x):
            raise RuntimeError("dispatcher failed")

    wrapper = ait.Module(
        _Tiny(),
        "exception-aot",
        strategy=ait.OneBackendStrategy(TorchEagerBackend()).enable_find_max_batch_size(False),
    )
    targets = _discover_aot_targets()
    aot_target = next(target for target in targets if target.path == "exception-aot")

    with _RegionInstaller([aot_target]):
        with pytest.raises(RuntimeError, match="dispatcher failed"):
            wrapper(torch.randn(1, 4))

    # The post-hook must have run via always_call=True, releasing the context attribute.
    assert not hasattr(wrapper.__wrapped__, _CONTEXT_ATTR)
    assert not hasattr(wrapper, _CONTEXT_ATTR)
