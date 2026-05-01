"""Phase 151-04 anti-regression locks for the composer dispatch.

The 151-02 SUMMARY documented a git-stash incident that silently dropped
the elif ring_type == 'mixed-spiro-fused' branch from
_assemble_complex_ring_name. This test makes such a regression visible:
if the elif branch ever disappears from composer.py, this test fails
BEFORE the integration suite catches the symptom.
"""
import inspect
from pathlib import Path

import pytest


COMPOSER_PATH = Path(__file__).resolve().parents[3] / "src" / "orthonym" / "assembly" / "composer.py"


def _composer_source() -> str:
    with open(COMPOSER_PATH) as f:
        return f.read()


def _get_assemble_complex_ring_name_source() -> str:
    """Extract the source of _assemble_complex_ring_name only — scopes the
    grep to the function body so a stray comment elsewhere in composer.py
    cannot satisfy the assertion."""
    from orthonym.assembly.composer import _assemble_complex_ring_name
    return inspect.getsource(_assemble_complex_ring_name)


class TestComposerDispatchInvariants:
    """Phase 151-04 BLK-01 anti-regression locks."""

    @pytest.mark.unit
    def test_mixed_spiro_fused_elif_branch_present_in_function_body(self):
        """The elif branch MUST exist in the body of
        _assemble_complex_ring_name. A function-scoped inspect.getsource
        grep proves the branch is in the dispatcher, not in a comment
        elsewhere in the module."""
        src = _get_assemble_complex_ring_name_source()
        # Strip comments to defeat the self-invalidating-grep-gate problem
        non_comment = "\n".join(
            line for line in src.split("\n") if not line.strip().startswith("#")
        )
        assert "elif ring_type == 'mixed-spiro-fused':" in non_comment, (
            "BLK-01 regression: the elif ring_type == 'mixed-spiro-fused' "
            "branch is missing from _assemble_complex_ring_name. The 151-02 "
            "SUMMARY noted a git-stash incident that dropped this same edit "
            "once before — please restore it from "
            "*/151-04-PLAN.md Task 2."
        )

    @pytest.mark.unit
    def test_name_mixed_spiro_fused_imported_in_function_body(self):
        """The dispatcher must import name_mixed_spiro_fused. A bare elif
        tag without the import is also a regression."""
        src = _get_assemble_complex_ring_name_source()
        non_comment = "\n".join(
            line for line in src.split("\n") if not line.strip().startswith("#")
        )
        assert "name_mixed_spiro_fused" in non_comment, (
            "BLK-01 regression: name_mixed_spiro_fused is not imported "
            "in _assemble_complex_ring_name. The elif branch is dead "
            "without the import."
        )

    @pytest.mark.unit
    def test_elif_lands_between_spiro_and_ortho_fused(self):
        """The elif branch must dispatch BEFORE 'ortho-fused' AND AFTER 'spiro'.
        Wrong ordering would mean a mixed-spiro-fused mol is misrouted by
        an earlier branch.

        We use line-number ordering on the function body to verify."""
        src = _get_assemble_complex_ring_name_source()
        lines = src.split("\n")
        spiro_line = None
        mixed_line = None
        ortho_line = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "elif ring_type == 'spiro':" in stripped:
                spiro_line = i
            elif "elif ring_type == 'mixed-spiro-fused':" in stripped:
                mixed_line = i
            elif "elif ring_type in ('ortho-fused', 'ortho-peri-fused'):" in stripped:
                ortho_line = i
        assert spiro_line is not None, "elif 'spiro' branch missing — bigger regression"
        assert mixed_line is not None, "BLK-01: elif 'mixed-spiro-fused' branch missing"
        assert ortho_line is not None, "elif 'ortho-fused' branch missing — bigger regression"
        assert spiro_line < mixed_line < ortho_line, (
            f"BLK-01: elif ordering wrong. spiro at line {spiro_line}, "
            f"mixed-spiro-fused at line {mixed_line}, ortho-fused at "
            f"line {ortho_line}. Required: spiro < mixed-spiro-fused < ortho-fused."
        )

    @pytest.mark.unit
    def test_only_one_elif_for_mixed_spiro_fused_in_module(self):
        """Belt-and-braces: the entire composer.py module should contain
        exactly ONE non-comment occurrence of `elif ring_type == 'mixed-spiro-fused':`.
        Two would mean a copy-paste accident; zero would mean the regression
        recurred."""
        src = _composer_source()
        non_comment = "\n".join(
            line for line in src.split("\n") if not line.strip().startswith("#")
        )
        count = non_comment.count("elif ring_type == 'mixed-spiro-fused':")
        assert count == 1, (
            f"BLK-01 anti-regression: expected exactly 1 elif "
            f"ring_type == 'mixed-spiro-fused' in composer.py, found {count}."
        )
