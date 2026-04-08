"""Tests for HandlerResult coverage metric (Phase 139 ARCH-06).

Verifies the HandlerResult dataclass computes coverage correctly
and is importable from the public API.
"""

import pytest


class TestHandlerResult:
    """Unit tests for HandlerResult dataclass."""

    def test_handler_result_importable(self):
        """HandlerResult is importable from orthonym.assembly.composer."""
        from orthonym.assembly.composer import HandlerResult
        assert HandlerResult is not None

    def test_coverage_correct_ratio(self):
        """HandlerResult.coverage returns correct ratio."""
        from orthonym.assembly.composer import HandlerResult
        hr = HandlerResult(
            name="test",
            handler_id="test",
            parent_atoms={0, 1, 2},
            accounted_atoms={0, 1, 2, 3, 4},
            total_heavy_atoms=10,
        )
        assert hr.coverage == pytest.approx(0.5)

    def test_coverage_zero_for_empty_total(self):
        """HandlerResult.coverage returns 0.0 for total_heavy_atoms=0."""
        from orthonym.assembly.composer import HandlerResult
        hr = HandlerResult(
            name="test",
            handler_id="test",
            total_heavy_atoms=0,
        )
        assert hr.coverage == 0.0

    def test_coverage_full(self):
        """HandlerResult.coverage returns 1.0 when all atoms accounted."""
        from orthonym.assembly.composer import HandlerResult
        hr = HandlerResult(
            name="ethanol",
            handler_id="chain",
            parent_atoms={0, 1},
            accounted_atoms={0, 1, 2},
            total_heavy_atoms=3,
        )
        assert hr.coverage == pytest.approx(1.0)

    def test_dataclass_fields(self):
        """HandlerResult can be instantiated with all fields."""
        from orthonym.assembly.composer import HandlerResult
        hr = HandlerResult(
            name="test-name",
            handler_id="test-handler",
            parent_atoms={1, 2, 3},
            accounted_atoms={1, 2, 3, 4},
            total_heavy_atoms=10,
        )
        assert hr.name == "test-name"
        assert hr.handler_id == "test-handler"
        assert hr.parent_atoms == {1, 2, 3}
        assert hr.accounted_atoms == {1, 2, 3, 4}
        assert hr.total_heavy_atoms == 10
        assert hr.coverage == pytest.approx(0.4)

    def test_coverage_three_of_ten(self):
        """Verify the specific acceptance criteria example: 3/10 = 0.3."""
        from orthonym.assembly.composer import HandlerResult
        hr = HandlerResult(
            name="test",
            handler_id="test",
            total_heavy_atoms=10,
            accounted_atoms={1, 2, 3},
        )
        assert hr.coverage == pytest.approx(0.3)
