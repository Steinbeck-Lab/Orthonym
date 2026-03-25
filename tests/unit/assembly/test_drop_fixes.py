"""Unit tests for DROP-07 and DROP-17 fixes in composer.py.

DROP-07: FG-only ring substituents (halogens, -OH, -NH2, =O as non-principal)
should produce correct locanted prefixes using oriented_ring locants, not be
silently skipped to the global FG loop (which lacks ring locant context).

Key symptom: monosubstituted halocycloalkanes get a spurious "1-" locant
(e.g., "1-fluorocyclohexane" instead of "fluorocyclohexane") because the
global FG loop does not apply should_omit_locant_one().

DROP-17: When BUG-B overfilter removes all FG matches, at least one match
should be restored to prevent total FG loss.
"""

import pytest
from orthonym import name_compound


# ---------------------------------------------------------------------------
# DROP-07: FG-only ring substituents with correct locant handling
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDROP07FGOnlyRingSubstituents:
    """DROP-07 fix: FG-only ring substituents emitted with correct ring locants."""

    def test_fluorocyclohexane_no_spurious_locant(self):
        """Monosubstituted fluorocyclohexane: locant 1 must be elided.
        Expected: 'fluorocyclohexane', NOT '1-fluorocyclohexane'."""
        result = name_compound("C1CCCCC1F")
        assert result is not None
        low = result.lower()
        assert "fluoro" in low, f"Expected 'fluoro' in '{result}'"
        # Monosubstituted carbocyclic ring: locant-1 must be omitted
        assert not low.startswith("1-"), (
            f"Spurious locant: expected 'fluorocyclohexane', got '{result}'"
        )

    def test_chlorocyclopentane_no_spurious_locant(self):
        """Monosubstituted chlorocyclopentane: locant 1 must be elided."""
        result = name_compound("C1CCCC1Cl")
        assert result is not None
        low = result.lower()
        assert "chloro" in low, f"Expected 'chloro' in '{result}'"
        assert not low.startswith("1-"), (
            f"Spurious locant: expected 'chlorocyclopentane', got '{result}'"
        )

    def test_bromocyclohexane_no_spurious_locant(self):
        """Monosubstituted bromocyclohexane: locant 1 must be elided."""
        result = name_compound("C1CCCCC1Br")
        assert result is not None
        low = result.lower()
        assert "bromo" in low
        assert not low.startswith("1-"), (
            f"Spurious locant: expected 'bromocyclohexane', got '{result}'"
        )

    def test_hydroxycyclohexanone_contains_hydroxy(self):
        """4-Hydroxycyclohexan-1-one (OC1CCCC(=O)C1): -OH is non-principal,
        must appear as 'hydroxy' prefix with correct ring locant."""
        result = name_compound("OC1CCCC(=O)C1")
        assert result is not None
        assert "hydroxy" in result.lower(), f"Expected 'hydroxy' in '{result}'"

    def test_fluoro_and_methyl_on_ring(self):
        """1-fluoro-4-methylcyclohexane: both 'fluoro' and 'methyl' present."""
        result = name_compound("CC1CCC(F)CC1")
        assert result is not None
        low = result.lower()
        assert "fluoro" in low, f"Expected 'fluoro' in '{result}'"
        assert "methyl" in low, f"Expected 'methyl' in '{result}'"

    def test_no_double_emission_fluorocyclohexane(self):
        """Fluorocyclohexane must NOT have 'difluoro' -- only one fluorine atom."""
        result = name_compound("C1CCCCC1F")
        assert result is not None
        assert "difluoro" not in result.lower(), (
            f"Double-emission: expected single 'fluoro', got '{result}'"
        )

    def test_aminocyclohexanone_amino_as_prefix(self):
        """3-Aminocyclohexan-1-one: amino is non-principal (ketone is higher),
        must appear as 'amino' prefix."""
        result = name_compound("NC1CCCC(=O)C1")
        assert result is not None
        assert "amino" in result.lower(), f"Expected 'amino' in '{result}'"

    def test_iodocyclopentane_no_spurious_locant(self):
        """Monosubstituted iodocyclopentane: locant 1 must be elided."""
        result = name_compound("C1CCCC1I")
        assert result is not None
        low = result.lower()
        assert "iodo" in low, f"Expected 'iodo' in '{result}'"
        assert not low.startswith("1-"), (
            f"Spurious locant: expected 'iodocyclopentane', got '{result}'"
        )


# ---------------------------------------------------------------------------
# DROP-17: BUG-B overfilter recovery
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDROP17OverfilterRecovery:
    """DROP-17 fix: when BUG-B removes all FG matches, at least one is restored."""

    def test_hydroxy_present_with_hydroxymethyl_branch(self):
        """A compound with -OH and -CH2OH on cyclohexane: hydroxyl info not lost."""
        result = name_compound("OC1CCCCC1CO")
        assert result is not None
        low = result.lower()
        assert "hydroxy" in low or "ol" in low, (
            f"Expected hydroxyl info in '{result}'"
        )

    def test_lactic_acid_hydroxy_present(self):
        """2-hydroxypropanoic acid: hydroxy on small branch must survive."""
        result = name_compound("CC(O)C(=O)O")
        assert result is not None
        low = result.lower()
        assert "hydroxy" in low or "ol" in low, (
            f"Expected hydroxyl info in '{result}'"
        )

    def test_drop17_no_total_fg_loss(self):
        """4-amino-2-(hydroxymethyl)cyclohexan-1-ol: amino must survive
        even if some matches are on small branches."""
        result = name_compound("NC1CCC(O)C(CO)C1")
        assert result is not None
        low = result.lower()
        assert "amino" in low or "amine" in low, (
            f"Expected amino info in '{result}'"
        )
