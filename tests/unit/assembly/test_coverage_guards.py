"""Tests for assembly coverage guard fixes (,).

: Lactone/lactam coverage guard relaxed from ring_size + 4 to ring_size + 8.
: Hard 20 HA fragment size gate removed; quality filters handle garbled names.
"""

import pytest
from rdkit import Chem

from orthonym.namer import name_compound


# ============================================================================
#: Lactone coverage guard uses ring_size + 8
# ============================================================================

@pytest.mark.unit
class TestLactoneGuardRelaxed:
    """: Lactone coverage guard uses ring_size + 8."""

    def test_substituted_lactone_uses_lactone_path(self):
        """5-membered lactone with pentyl (11 HA total, old guard=9, new=13).

        O=C1OCC(CCCCC)C1 is a gamma-butyrolactone with pentyl substituent.
        With old ring_size+4=9, total 11 HA > 9 would reject lactone guard.
        With new ring_size+8=13, total 11 HA <= 13 allows lactone path.
        The lactone handler produces oxolan-2-one naming.
        """
        result = name_compound("O=C1OCC(CCCCC)C1")
        assert result is not None
        # Should use lactone naming (oxolan-2-one pattern)
        result_lower = result.lower()
        assert "oxolan" in result_lower or "furan" in result_lower or "olide" in result_lower or "lacton" in result_lower, \
            f"Expected lactone-path name, got: {result}"

    def test_small_lactone_still_works(self):
        """Simple butyrolactone (no extra substituents) still names correctly."""
        result = name_compound("O=C1OCCC1")
        assert result is not None

    def test_macrocycle_exception_unchanged(self):
        """Macrocyclic lactones (ring > 8) still bypass guard."""
        # 9-membered lactone ring
        result = name_compound("O=C1OCCCCCCC1")
        assert result is not None


# ============================================================================
#: Fragments >20 HA get naming attempt
# ============================================================================

@pytest.mark.unit
class TestLargeFragmentNaming:
    """: Fragments >20 HA get naming attempt instead of being dropped."""

    def test_large_fragment_not_silently_dropped(self):
        """A ring compound with a large substituent should attempt naming.

        Cyclohexane with a long chain substituent (>20 HA in fragment).
        Quality filters may still reject garbled output, but the hard
        size gate should not prevent the attempt.
        """
        # Cyclohexane with a C21 chain substituent
        smiles = "C1CCCCC1" + "C" * 21
        result = name_compound(smiles)
        # Should get SOME name (not silently dropped)
        assert result is not None
