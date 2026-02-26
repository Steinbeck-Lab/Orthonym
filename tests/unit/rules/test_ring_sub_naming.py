"""
Unit tests for ring-as-substituent naming (ring_substituents.py).

Phase 79: Verifies RING_SUBSTITUENT_NAMES dict completeness and that
identify_ring_system() + get_ring_substituent_name() produce correct
substituent prefix names for all monocyclic ring systems.
"""

import pytest
from rdkit import Chem

from orthonym.rules.ring_substituents import (
    RING_SUBSTITUENT_NAMES,
    identify_ring_system,
    get_ring_substituent_name,
)


class TestRingSubstituentNamesCompleteness:
    """Every ring name returned by identify_ring_system() has a RING_SUBSTITUENT_NAMES entry."""

    @pytest.mark.parametrize(
        "smiles, expected_ring_name, expected_sub_name",
        [
            # Carbocyclic saturated
            ("C1CC1", "cyclopropane", "cyclopropyl"),
            ("C1CCC1", "cyclobutane", "cyclobutyl"),
            ("C1CCCC1", "cyclopentane", "cyclopentyl"),
            ("C1CCCCC1", "cyclohexane", "cyclohexyl"),
            ("C1CCCCCC1", "cycloheptane", "cycloheptyl"),
            ("C1CCCCCCC1", "cyclooctane", "cyclooctyl"),
            # Carbocyclic aromatic
            ("c1ccccc1", "benzene", "phenyl"),
            # Aromatic heterocyclic
            ("c1ccncc1", "pyridine", "pyridyl"),
            ("c1ccoc1", "furan", "furyl"),
            ("c1ccsc1", "thiophene", "thienyl"),
            ("c1cc[nH]c1", "pyrrole", "pyrrolyl"),
            ("c1cnc[nH]1", "imidazole", "imidazolyl"),
            ("c1ccnc(n1)", "pyrimidine", "pyrimidinyl"),
            ("c1cnccn1", "pyrazine", "pyrazinyl"),
            ("c1ccnnc1", "pyridazine", "pyridazinyl"),
            # Saturated heterocyclic (6-membered)
            ("C1CCNCC1", "piperidine", "piperidinyl"),
            ("C1COCCN1", "morpholine", "morpholinyl"),
            ("C1CNCCN1", "piperazine", "piperazinyl"),
            ("C1CCOCC1", "tetrahydropyran", "tetrahydropyranyl"),
            # Saturated heterocyclic (5-membered)
            ("C1CCOC1", "oxolane", "oxolanyl"),
            ("C1CCNC1", "pyrrolidine", "pyrrolidinyl"),
            # Saturated heterocyclic (4-membered)
            ("C1CCO1", "oxetane", "oxetanyl"),
            ("C1CCN1", "azetidine", "azetidinyl"),
            # Saturated heterocyclic (3-membered)
            ("C1CO1", "oxirane", "oxiranyl"),
            ("C1CN1", "aziridine", "aziridinyl"),
        ],
        ids=lambda x: x if isinstance(x, str) and len(x) < 20 else "",
    )
    def test_ring_identification_and_naming(self, smiles, expected_ring_name, expected_sub_name):
        """Verify identify_ring_system returns correct name and dict has matching entry."""
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, f"Failed to parse SMILES: {smiles}"

        ring_info = mol.GetRingInfo()
        rings = ring_info.AtomRings()
        assert len(rings) >= 1, f"No rings found in {smiles}"

        # Use the first (and usually only) ring
        ring_atoms = tuple(rings[0])

        # Test identify_ring_system
        ring_name = identify_ring_system(mol, ring_atoms)
        assert ring_name == expected_ring_name, (
            f"identify_ring_system returned '{ring_name}', expected '{expected_ring_name}'"
        )

        # Test dict completeness
        assert ring_name in RING_SUBSTITUENT_NAMES, (
            f"Ring name '{ring_name}' not in RING_SUBSTITUENT_NAMES dict"
        )
        assert RING_SUBSTITUENT_NAMES[ring_name] == expected_sub_name, (
            f"RING_SUBSTITUENT_NAMES['{ring_name}'] = '{RING_SUBSTITUENT_NAMES[ring_name]}', "
            f"expected '{expected_sub_name}'"
        )

        # Test get_ring_substituent_name
        sub_name = get_ring_substituent_name(mol, ring_atoms)
        assert sub_name == expected_sub_name, (
            f"get_ring_substituent_name returned '{sub_name}', expected '{expected_sub_name}'"
        )

    def test_dict_has_minimum_entries(self):
        """RING_SUBSTITUENT_NAMES should have at least 24 entries covering all identified rings."""
        assert len(RING_SUBSTITUENT_NAMES) >= 24, (
            f"RING_SUBSTITUENT_NAMES has only {len(RING_SUBSTITUENT_NAMES)} entries, "
            f"expected >= 24"
        )

    def test_all_values_end_with_yl(self):
        """All substituent names must end with -yl per IUPAC convention."""
        for ring_name, sub_name in RING_SUBSTITUENT_NAMES.items():
            assert sub_name.endswith("yl"), (
                f"RING_SUBSTITUENT_NAMES['{ring_name}'] = '{sub_name}' "
                f"does not end with 'yl'"
            )


class TestGetRingSubstituentNameFallback:
    """Test fallback behavior for rings not in the dict."""

    def test_generic_large_cycloalkane(self):
        """Large all-carbon ring should get generic cycloXyl name."""
        mol = Chem.MolFromSmiles("C1CCCCCCCCC1")  # cyclodecane
        ring_info = mol.GetRingInfo()
        ring_atoms = tuple(ring_info.AtomRings()[0])
        name = get_ring_substituent_name(mol, ring_atoms)
        assert name.startswith("cyclo"), f"Expected cyclo* prefix, got '{name}'"
        assert name.endswith("yl"), f"Expected *yl suffix, got '{name}'"

    def test_unknown_ring_returns_string(self):
        """Even for unknown rings, a non-empty string should be returned."""
        mol = Chem.MolFromSmiles("C1CCCCCCCCC1")
        ring_info = mol.GetRingInfo()
        ring_atoms = tuple(ring_info.AtomRings()[0])
        name = get_ring_substituent_name(mol, ring_atoms)
        assert isinstance(name, str) and len(name) > 0
