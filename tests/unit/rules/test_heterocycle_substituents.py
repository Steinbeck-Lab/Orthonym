"""Tests for heterocycle substituent detection and naming.

Phase 125-03 Task 1: Verify that halogens and heteroatom-containing groups
on heterocyclic rings appear in the generated IUPAC name.

Root cause: _identify_hetero_substituent() only recognized 5 patterns
(amino, hydroxy, nitro, sulfanyl, oxo), silently dropping halogens and
all multi-atom heteroatom groups.
"""

import pytest
from orthonym import name_compound


class TestHeterocycleHalogens:
    """Halogens on heterocyclic rings must appear in names."""

    def test_chloropyridine(self):
        """3-chloropyridine: Cl on pyridine must produce 'chloro' prefix."""
        result = name_compound("Clc1ccncc1")
        assert "chloro" in result, f"Expected 'chloro' in '{result}'"

    def test_fluoroimidazole(self):
        """Fluoroimidazole: F on imidazole must produce 'fluoro' prefix."""
        result = name_compound("Fc1cnc[nH]1")
        assert "fluoro" in result, f"Expected 'fluoro' in '{result}'"

    def test_bromothiophene(self):
        """3-bromothiophene: Br on thiophene must produce 'bromo' prefix."""
        result = name_compound("Brc1ccsc1")
        assert "bromo" in result, f"Expected 'bromo' in '{result}'"

    def test_iodopyridine(self):
        """3-iodopyridine: I on pyridine must produce 'iodo' prefix."""
        result = name_compound("Ic1ccncc1")
        assert "iodo" in result, f"Expected 'iodo' in '{result}'"


class TestHeterocycleExistingPatterns:
    """Existing patterns (amino, hydroxy, etc.) must not regress."""

    def test_aminopyridine(self):
        """3-aminopyridine: NH2 on pyridine."""
        result = name_compound("Nc1ccncc1")
        assert "amino" in result, f"Expected 'amino' in '{result}'"

    def test_hydroxypyridine(self):
        """3-hydroxypyridine: OH on pyridine."""
        result = name_compound("Oc1ccncc1")
        assert "hydroxy" in result or "pyridinol" in result, (
            f"Expected 'hydroxy' or 'pyridinol' in '{result}'"
        )


class TestUnsubstitutedHeterocycles:
    """Unsubstituted heterocycles must remain unchanged."""

    def test_pyridine(self):
        result = name_compound("c1ccncc1")
        assert result == "pyridine"

    def test_thiophene(self):
        result = name_compound("c1ccsc1")
        assert result == "thiophene"

    def test_furan(self):
        result = name_compound("c1ccoc1")
        assert result == "furan"


class TestLargeSubstituentsOnComplexRings:
    """Large substituents (>12 HA) on complex rings should not be silently dropped."""

    def test_long_chain_on_cyclohexane(self):
        """A 10-carbon chain on cyclohexane should be named."""
        smi = "CCCCCCCCCC(C1CCCCC1)C"
        result = name_compound(smi)
        # Should contain some indication of the ring and the chain
        assert "cyclohexyl" in result or "decyl" in result or "methyl" in result, (
            f"Expected substituent names in '{result}'"
        )
