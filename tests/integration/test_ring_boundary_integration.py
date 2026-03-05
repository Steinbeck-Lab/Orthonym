"""
Integration tests for ring boundary fix (Phase 89.2).

Tests that the complete ring system boundary prevents fabricated substituents
on fused/bridged/spiro systems, while preserving ring-as-substituent detection
on chain parents.
"""

import pytest
from rdkit import Chem

from orthonym.namer import Orthonym


@pytest.fixture
def namer():
    return Orthonym()


class TestNoFabricatedSubstituents:
    """Verify no fabricated substituents on fused/bridged/spiro ring systems."""

    def test_caffeine_no_fabricated_butyl(self, namer):
        """Caffeine should not have fabricated 'butyl' substituents.

        Root cause: BFS from imidazole ring walked into pyrimidine ring,
        counted 4 carbons as 'butyl'.
        """
        name = namer.name("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
        assert name is not None
        # Must NOT contain fabricated 'butyl' or 'dibutyl'
        name_lower = name.lower()
        assert "butyl" not in name_lower, f"Fabricated 'butyl' in caffeine name: {name}"
        # Methyl groups should still be present
        assert "methyl" in name_lower or "trimethyl" in name_lower or "caffeine" in name_lower, \
            f"Missing methyl groups in caffeine name: {name}"

    def test_naphthalene_carboxylic_acid_no_fabricated_subs(self, namer):
        """Naphthalene-2-carboxylic acid should not have fabricated ring substituents."""
        name = namer.name("OC(=O)c1ccc2ccccc2c1")
        assert name is not None
        name_lower = name.lower()
        # Should NOT contain fabricated 'butyl', 'pentyl', 'hexyl' from ring leak
        for bad_sub in ["butyl", "pentyl", "hexyl"]:
            assert bad_sub not in name_lower, \
                f"Fabricated '{bad_sub}' in naphthalene-2-carboxylic acid name: {name}"

    def test_indole_no_fabricated_subs(self, namer):
        """Indole should not have fabricated substituents from ring leak."""
        name = namer.name("c1ccc2[nH]ccc2c1")
        assert name is not None
        name_lower = name.lower()
        for bad_sub in ["butyl", "propyl", "pentyl"]:
            assert bad_sub not in name_lower, \
                f"Fabricated '{bad_sub}' in indole name: {name}"

    def test_norbornane_no_fabricated_subs(self, namer):
        """Norbornane (bridged bicyclic) should not have fabricated substituents."""
        name = namer.name("C1CC2CC1CC2")
        assert name is not None
        name_lower = name.lower()
        # Norbornane is bicyclo[2.2.1]heptane -- no substituents
        for bad_sub in ["methyl", "ethyl", "propyl", "butyl"]:
            assert bad_sub not in name_lower, \
                f"Fabricated '{bad_sub}' in norbornane name: {name}"


class TestRingAsSubstituentPreserved:
    """Verify ring-as-substituent on chain parents still works (Pitfall 3)."""

    def test_phenylpentanoic_acid(self, namer):
        """5-phenylpentanoic acid: phenyl ring must still be detected on chain parent."""
        name = namer.name("OC(=O)CCCCc1ccccc1")
        assert name is not None
        name_lower = name.lower()
        # Should contain "phenyl" as a ring substituent on the chain
        assert "phenyl" in name_lower, \
            f"Ring substituent 'phenyl' missing from name: {name}"

    def test_cyclohexylacetic_acid(self, namer):
        """Cyclohexylacetic acid: cyclohexyl ring as substituent on chain."""
        name = namer.name("OC(=O)CC1CCCCC1")
        assert name is not None
        name_lower = name.lower()
        assert "cyclohexyl" in name_lower, \
            f"Ring substituent 'cyclohexyl' missing from name: {name}"


class TestRingSubstituentOnRingParent:
    """Verify ring substituent on ring parent still detected (Pitfall 4)."""

    def test_4_methylnaphthalene(self, namer):
        """4-methylnaphthalene: methyl detected on fused ring parent."""
        name = namer.name("Cc1cccc2ccccc12")
        assert name is not None
        name_lower = name.lower()
        assert "methyl" in name_lower, \
            f"Missing 'methyl' on naphthalene: {name}"

    def test_2_methylindole(self, namer):
        """2-methylindole: methyl detected on fused heterocyclic parent."""
        name = namer.name("Cc1cc2ccccc2[nH]1")
        assert name is not None
        name_lower = name.lower()
        assert "methyl" in name_lower, \
            f"Missing 'methyl' on indole: {name}"


class TestSpiroCompounds:
    """Verify spiro ring atoms are handled correctly."""

    def test_spiro_decane_no_fabricated_subs(self, namer):
        """Spiro[4.5]decane should not have fabricated substituents."""
        name = namer.name("C1CCCC11CCCCC1")
        assert name is not None
        name_lower = name.lower()
        # Should be some form of spiro name, not have fabricated subs
        for bad_sub in ["butyl", "pentyl"]:
            assert bad_sub not in name_lower, \
                f"Fabricated '{bad_sub}' in spiro[4.5]decane name: {name}"
