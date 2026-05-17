"""WR-08 regression: ``ester_family`` D-16 un-wrap contract.

Phase 160.2 Plan-04-03a WR-08 closure per 160.1-REVIEW.md WR-08.

The composite handler's exception-raising path is documented at
``handlers/ester_family.py:121-139``: when ``pool.best()`` returns ``None``
(pool rejected the only candidate per quality threshold / wildcard /
ratio-floor), ``pool.best().name`` raises ``AttributeError``, mirroring
the pre-Plan-03-01 inline cascade at ``composer.py:875``. The
``AttributeError`` propagates UN-WRAPPED through ``dispatch_inner`` per
the D-16 exception list at ``inner_dispatch.py:346-349``:

    except (AttributeError, KeyError, IndexError, TypeError):
        raise  # un-wrapped per D-16

Without this regression test, a future refactor could:

1. Catch ``AttributeError`` inside ``ester_family.name_ester_family``
   defensively (looks innocuous), which would break the descriptive-
   fallback signaling path in ``namer.py:1888`` for wildcard SMILES.
2. Catch the exception in ``dispatch_inner`` without listing it in the
   un-wrap tuple, again breaking the signal chain.

This file enforces both contracts by exercising the actual un-wrap path
through ``dispatch_inner`` with a controlled handler that raises
``AttributeError`` mirroring the live ``pool.best().name`` path.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from orthonym.assembly import inner_dispatch as _ind
from orthonym.assembly.inner_dispatch import (
    INNER_DISPATCH_TABLE,
    InnerDispatchEntry,
    dispatch_inner,
)


def _install_raising_handler(handler_id: str, exc: BaseException):
    """Install a synthetic handler that raises *exc* and return restore-fn.

    Mirrors the table-replace pattern in
    ``TestInnerDispatchTypeErrorWrapping``: snapshot the existing
    table+cache, clear, install ONE entry that always-matches and raises
    *exc*, return a callable that restores the original state.
    """
    def _always_true(features):
        return True

    def _raises(features, mol, style="pin"):
        raise exc

    entry = InnerDispatchEntry(
        handler_id=handler_id,
        priority=1,
        predicate=_always_true,
        handler=_raises,
        iupac_section="P-X (synthetic)",
        description=f"synthetic handler raising {type(exc).__name__}",
        side_effect_inventory=(),
    )

    original_table = INNER_DISPATCH_TABLE.copy()
    original_cache = _ind._SORTED_ENTRIES_CACHE
    INNER_DISPATCH_TABLE.clear()
    INNER_DISPATCH_TABLE[handler_id] = entry
    _ind._SORTED_ENTRIES_CACHE = None

    def restore():
        INNER_DISPATCH_TABLE.clear()
        INNER_DISPATCH_TABLE.update(original_table)
        _ind._SORTED_ENTRIES_CACHE = original_cache

    return restore


def test_ester_family_attribute_error_unwrapped():
    """D-16 un-wrap contract: ``AttributeError`` propagates un-wrapped
    through ``dispatch_inner``.

    The handler raises ``AttributeError("'NoneType' object has no
    attribute 'name'")`` mirroring the live
    ``pool.best().name`` shape (``handlers/ester_family.py:172``). The
    un-wrap allow-list at ``inner_dispatch.py:346-349`` MUST re-raise
    it un-wrapped (NOT wrap in ``RuntimeError``); namer's broad
    ``except (TypeError, KeyError, IndexError, AttributeError)`` at
    ``namer.py:1888`` then catches it and falls through to
    ``_descriptive_fallback``.
    """
    original = AttributeError("'NoneType' object has no attribute 'name'")
    restore = _install_raising_handler(
        "synthetic_ester_family_d16", original,
    )
    try:
        with pytest.raises(AttributeError) as excinfo:
            dispatch_inner(object())
        # Un-wrapped: the AttributeError surfaces directly (NOT wrapped in
        # RuntimeError). The string is exactly the original message.
        assert excinfo.value is original or str(excinfo.value) == str(original), (
            f"AttributeError was wrapped instead of re-raised: {excinfo.value!r}"
        )
    finally:
        restore()


def test_keyerror_unwrapped_through_dispatch_inner():
    """D-16 un-wrap contract: ``KeyError`` propagates un-wrapped.

    Mirrors the AttributeError case for the other entries in the allow-list
    tuple at ``inner_dispatch.py:346``
    (``except (AttributeError, KeyError, IndexError, TypeError):``).
    """
    original = KeyError("synthetic_missing_handler_key")
    restore = _install_raising_handler(
        "synthetic_ester_family_keyerror", original,
    )
    try:
        with pytest.raises(KeyError) as excinfo:
            dispatch_inner(object())
        # KeyError str() wraps in quotes — check identity instead.
        assert excinfo.value is original, (
            f"KeyError was wrapped instead of re-raised: {excinfo.value!r}"
        )
    finally:
        restore()


def test_indexerror_unwrapped_through_dispatch_inner():
    """D-16 un-wrap contract: ``IndexError`` propagates un-wrapped."""
    original = IndexError("synthetic list index out of range")
    restore = _install_raising_handler(
        "synthetic_ester_family_indexerror", original,
    )
    try:
        with pytest.raises(IndexError) as excinfo:
            dispatch_inner(object())
        assert excinfo.value is original, (
            f"IndexError was wrapped instead of re-raised: {excinfo.value!r}"
        )
    finally:
        restore()


def test_valueerror_wrapped_as_runtime_error():
    """Negative control: exceptions OUTSIDE the un-wrap allow-list ARE
    wrapped as ``RuntimeError`` per ``inner_dispatch.py:350-357``.

    ``ValueError`` is NOT in the allow-list ``(AttributeError, KeyError,
    IndexError, TypeError)``; ``dispatch_inner`` MUST wrap it as
    ``RuntimeError`` with the original chained via ``__cause__``. This
    test pins the boundary between the D-16 list and the broader wrap.
    """
    original = ValueError("synthetic not-in-allow-list error")
    restore = _install_raising_handler(
        "synthetic_ester_family_valueerror", original,
    )
    try:
        with pytest.raises(RuntimeError) as excinfo:
            dispatch_inner(object())
        assert excinfo.value.__cause__ is original, (
            f"ValueError was NOT wrapped via __cause__: "
            f"cause={excinfo.value.__cause__!r}"
        )
        assert "ValueError" in str(excinfo.value), (
            f"RuntimeError message missing ValueError tag: "
            f"{excinfo.value!s}"
        )
    finally:
        restore()
