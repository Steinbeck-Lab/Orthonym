"""Tests for naming cascade quality improvements.

Phase 125-03 Tasks 2-4: Verify cycloalkyl retained names, FRAGMENT_NAME_CACHE
entries, and Tier 5 descriptive fallback for compound substituents.
"""

import pytest
from rdkit import Chem
from orthonym import name_compound


class TestCycloalkylRetainedNames:
    """Cycloalkyl ring substituents should be detected by Tier 1."""

    def test_cyclopentyl_in_compound(self):
        """Cyclopentyl group should appear in compound names."""
        # propylcyclopentane or cyclopentylpropane
        result = name_compound("C1CCCC1CCC")
        assert "cyclopentyl" in result or "propylcyclopentane" in result, (
            f"Expected 'cyclopentyl' or 'propylcyclopentane' in '{result}'"
        )

    def test_cyclohexyl_in_compound(self):
        """Cyclohexyl group should appear in compound names."""
        # ethylcyclohexane or cyclohexylethane
        result = name_compound("C1CCCCC1CC")
        assert "cyclohexyl" in result or "ethylcyclohexane" in result, (
            f"Expected 'cyclohexyl' or 'ethylcyclohexane' in '{result}'"
        )

    def test_cyclopropyl_in_compound(self):
        """Cyclopropyl group should appear in compound names."""
        result = name_compound("C1CC1CCCC")
        assert "cyclopropyl" in result or "butylcyclopropane" in result, (
            f"Expected 'cyclopropyl' or 'butylcyclopropane' in '{result}'"
        )


class TestCycloalkylRetainedNameUnit:
    """Unit tests for _check_retained_substituent() cycloalkyl detection."""

    def test_cyclopentyl_direct(self):
        """Cyclopentyl ring fragment should be detected by retained name check."""
        from orthonym.assembly.substituent_naming import _check_retained_substituent

        mol = Chem.MolFromSmiles("C1(CCCC1)C")  # methylcyclopentane
        ring_info = mol.GetRingInfo()
        ring = list(ring_info.AtomRings()[0])
        result = _check_retained_substituent(mol, ring, ring[0])
        assert result == "cyclopentyl", f"Expected 'cyclopentyl', got '{result}'"

    def test_cyclohexyl_direct(self):
        """Cyclohexyl ring fragment should be detected by retained name check."""
        from orthonym.assembly.substituent_naming import _check_retained_substituent

        mol = Chem.MolFromSmiles("C1(CCCCC1)C")  # methylcyclohexane
        ring_info = mol.GetRingInfo()
        ring = list(ring_info.AtomRings()[0])
        result = _check_retained_substituent(mol, ring, ring[0])
        assert result == "cyclohexyl", f"Expected 'cyclohexyl', got '{result}'"

    def test_cyclopropyl_direct(self):
        """Cyclopropyl ring fragment should be detected by retained name check."""
        from orthonym.assembly.substituent_naming import _check_retained_substituent

        mol = Chem.MolFromSmiles("C1(CC1)C")  # methylcyclopropane
        ring_info = mol.GetRingInfo()
        ring = list(ring_info.AtomRings()[0])
        result = _check_retained_substituent(mol, ring, ring[0])
        assert result == "cyclopropyl", f"Expected 'cyclopropyl', got '{result}'"


class TestFragmentNameCache:
    """Verify common fragments are in the FRAGMENT_NAME_CACHE."""

    def test_cache_has_branched_alkanes(self):
        """Branched alkanes should be in cache for Tier 2 hits."""
        from orthonym.assembly.fragment_naming import FRAGMENT_NAME_CACHE

        assert "CC(C)C" in FRAGMENT_NAME_CACHE  # 2-methylpropane
        assert "CCC(C)C" in FRAGMENT_NAME_CACHE  # 2-methylbutane
        assert "CC(C)(C)C" in FRAGMENT_NAME_CACHE  # 2,2-dimethylpropane

    def test_cache_has_cycloalkanes(self):
        """Cycloalkanes should be in cache."""
        from orthonym.assembly.fragment_naming import FRAGMENT_NAME_CACHE

        assert "C1CC1" in FRAGMENT_NAME_CACHE  # cyclopropane
        assert "C1CCC1" in FRAGMENT_NAME_CACHE  # cyclobutane
        assert "C1CCCC1" in FRAGMENT_NAME_CACHE  # cyclopentane
        assert "C1CCCCC1" in FRAGMENT_NAME_CACHE  # cyclohexane

    def test_cache_has_substituted_aromatics(self):
        """Common substituted aromatics should be in cache."""
        from orthonym.assembly.fragment_naming import FRAGMENT_NAME_CACHE

        assert "Cc1ccccc1" in FRAGMENT_NAME_CACHE  # toluene
        assert "CCc1ccccc1" in FRAGMENT_NAME_CACHE  # ethylbenzene
        assert "COc1ccccc1" in FRAGMENT_NAME_CACHE  # anisole

    def test_cache_has_substituted_heterocycles(self):
        """Common substituted heterocycles should be in cache."""
        from orthonym.assembly.fragment_naming import FRAGMENT_NAME_CACHE

        assert "Cc1ccncc1" in FRAGMENT_NAME_CACHE  # 4-methylpyridine
        assert "Cc1ccccn1" in FRAGMENT_NAME_CACHE  # 2-methylpyridine
        assert "Cc1cccnc1" in FRAGMENT_NAME_CACHE  # 3-methylpyridine

    def test_cache_has_dicarboxylic_acids(self):
        """Common dicarboxylic acids should be in cache."""
        from orthonym.assembly.fragment_naming import FRAGMENT_NAME_CACHE

        assert "O=C(O)CC(=O)O" in FRAGMENT_NAME_CACHE  # propanedioic acid
        assert "O=C(O)CCC(=O)O" in FRAGMENT_NAME_CACHE  # butanedioic acid
        assert "O=C(O)CCCC(=O)O" in FRAGMENT_NAME_CACHE  # pentanedioic acid
