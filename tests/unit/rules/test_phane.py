"""Phase 155.A unit tests for src/orthonym/rules/phane.py.

Wave 0 scaffold: one green smoke test that asserts the module's public API
exports; four skip-marked placeholder classes (TestIsCyclophane,
TestClassifyPhaneTopology, TestBuildCompositeLocant, TestNameCyclophane)
that Task 3 fills with concrete cases per 155-CONTEXT.md D-03/D-04/D-05/D-16
and IUPAC 2013 Blue Book P-26.4.

Source:
- 155-CONTEXT.md D-03/D-04/D-05/D-16/D-22 (test layout).
- 155-PATTERNS.md "tests/unit/rules/test_phane.py" section.
"""
from __future__ import annotations

import pytest

from orthonym.rules.phane import (
    PhaneTopology,
    _build_composite_locant,
    _classify_phane_topology,
    is_cyclophane,
    name_cyclophane,
)


def test_phane_module_exports_public_api() -> None:
    """Smoke test: phane.py exports the six public symbols Task 3 will fill.

    Source: 155-CONTEXT.md D-03/D-04/D-05; D-22 (test layout); 155-PATTERNS.md.
    """
    assert callable(is_cyclophane)
    assert callable(name_cyclophane)
    assert callable(_classify_phane_topology)
    assert callable(_build_composite_locant)
    assert PhaneTopology.PARACYCLOPHANE.value == "paracyclophane"
    assert PhaneTopology.METACYCLOPHANE.value == "metacyclophane"
    assert PhaneTopology.ORTHOCYCLOPHANE.value == "orthocyclophane"
    assert PhaneTopology.GENERIC_CYCLOPHANE.value == "generic"


@pytest.mark.skip(reason="Phase 155-01 Task 3 fills this")
class TestIsCyclophane:
    """Topology gate per 155-CONTEXT.md D-03 (Task 3 fills positive + negative cases)."""

    def test_placeholder(self) -> None:  # pragma: no cover -- skipped at collection
        raise AssertionError("Task 3 replaces this stub.")


@pytest.mark.skip(reason="Phase 155-01 Task 3 fills this")
class TestClassifyPhaneTopology:
    """Sub-class enum dispatch per 155-CONTEXT.md D-04 (Task 3 fills enum cases)."""

    def test_placeholder(self) -> None:  # pragma: no cover
        raise AssertionError("Task 3 replaces this stub.")


@pytest.mark.skip(reason="Phase 155-01 Task 3 fills this")
class TestBuildCompositeLocant:
    """Composite-locant ASCII / Unicode emission per 155-CONTEXT.md D-05."""

    def test_placeholder(self) -> None:  # pragma: no cover
        raise AssertionError("Task 3 replaces this stub.")


@pytest.mark.skip(reason="Phase 155-01 Task 3 fills this")
class TestNameCyclophane:
    """Top-level handler per 155-CONTEXT.md D-04 + D-05 (Task 3 fills cases)."""

    def test_placeholder(self) -> None:  # pragma: no cover
        raise AssertionError("Task 3 replaces this stub.")
