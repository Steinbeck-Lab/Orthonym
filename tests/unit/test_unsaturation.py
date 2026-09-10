"""
Unit tests for unsaturation naming (alkenes, alkynes, enynes).

Tests cover:
- Simple alkene naming with locants (FOUND-12)
- Simple alkyne naming with locants (FOUND-13)
- Enyne naming with double bond priority (FOUND-14)
- Branched unsaturated compounds
"""

import pytest
from orthonym import name_compound


class TestAlkeneNaming:
    """Test IUPAC alkene naming with locants (FOUND-12)."""

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,expected", [
        ("C=C", "ethene"),
        ("C=CC", "propene"),   # (d): unsubstituted trinuclear omits '1'
        ("CC=C", "propene"),   # Same molecule, different SMILES
        ("CC=CC", "but-2-ene"),
        ("C=CCC", "but-1-ene"),
        ("C=CCCC", "pent-1-ene"),
        ("CC=CCC", "pent-2-ene"),
        ("C=CCCCC", "hex-1-ene"),
        ("CC=CCCC", "hex-2-ene"),
        ("CCC=CCC", "hex-3-ene"),
    ])
    def test_simple_alkenes(self, smiles, expected):
        """Test simple alkene naming with locants."""
        assert name_compound(smiles) == expected

    @pytest.mark.unit
    def test_ethene_no_locant(self):
        """Ethene should not have a locant (only one possible position)."""
        result = name_compound("C=C")
        assert result == "ethene"
        assert "-" not in result  # No hyphen means no locant

    @pytest.mark.unit
    def test_propene_no_locant(self):
        """Unsubstituted propene omits the bond locant (d) PIN);
        a substituent restores it (3-chloroprop-1-ene)."""
        result = name_compound("C=CC")
        assert result == "propene"
        assert "-" not in result
        assert name_compound("C=CCCl") == "3-chloroprop-1-ene"

    @pytest.mark.unit
    def test_butene_locant_position(self):
        """Test but-1-ene vs but-2-ene locant assignment."""
        assert name_compound("C=CCC") == "but-1-ene"
        assert name_compound("CC=CC") == "but-2-ene"


class TestAlkyneNaming:
    """Test IUPAC alkyne naming with locants (FOUND-13)."""

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,expected", [
        ("C#C", "acetylene"),
        ("C#CC", "propyne"),   # (d): unsubstituted trinuclear omits '1'
        ("CC#C", "propyne"),   # Same molecule, different SMILES
        ("CC#CC", "but-2-yne"),
        ("C#CCC", "but-1-yne"),
        ("C#CCCC", "pent-1-yne"),
        ("CC#CCC", "pent-2-yne"),
    ])
    def test_simple_alkynes(self, smiles, expected):
        """Test simple alkyne naming with locants."""
        assert name_compound(smiles) == expected

    @pytest.mark.unit
    def test_ethyne_no_locant(self):
        """C#C returns retained name 'acetylene' PIN)."""
        result = name_compound("C#C")
        assert result == "acetylene"
        assert "-" not in result

    @pytest.mark.unit
    def test_propyne_no_locant(self):
        """Unsubstituted propyne omits the bond locant (d) PIN)."""
        result = name_compound("C#CC")
        assert result == "propyne"
        assert "-" not in result


class TestEnyneNaming:
    """Test IUPAC enyne naming (FOUND-14).

    For enynes (compounds with both double and triple bonds),
    IUPAC 2013 gives preference to double bonds when locant sets tie.
    """

    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected", [
        ("C=CC#C", "but-1-en-3-yne"),  # Double at 1, triple at 3
        ("C#CC=C", "but-1-en-3-yne"),  # Same molecule, reversed SMILES
        ("C=CCC#C", "pent-1-en-4-yne"),  # Double at 1, triple at 4
    ])
    def test_simple_enynes(self, smiles, expected):
        """Test simple enyne naming."""
        assert name_compound(smiles) == expected

    @pytest.mark.integration
    def test_enyne_double_bond_priority(self):
        """When locant sets tie, double bonds should get lower locants.

        For C#CC=C (but-1-en-3-yne):
        - Forward numbering: double at 3, triple at 1 -> [1, 3]
        - Reverse numbering: double at 1, triple at 3 -> [1, 3]
        Both have same combined locant set, but IUPAC 2013 says
        double bonds get lower locants, so reverse (double at 1) wins.
        """
        # The SMILES C#CC=C could number as:
        # - triple at 1, double at 3
        # - double at 1, triple at 3
        # We want double at 1 (lower locant)
        result = name_compound("C#CC=C")
        assert result == "but-1-en-3-yne"
        # Verify double bond is at position 1
        assert "-1-en" in result
        # Verify triple bond is at position 3
        assert "-3-yn" in result


class TestUnsaturationWithSubstituents:
    """Test unsaturated compounds with alkyl substituents."""

    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected", [
        ("CC(C)=CC", "2-methylbut-2-ene"),
        ("C=C(C)C", "2-methylprop-1-ene"),
        ("CC(C)=C", "2-methylprop-1-ene"),  # Same molecule
    ])
    def test_branched_alkenes(self, smiles, expected):
        """Test branched alkene naming."""
        assert name_compound(smiles) == expected

    @pytest.mark.integration
    def test_2_methylpropene(self):
        """2-methylprop-1-ene (isobutylene)."""
        # Both SMILES should give same name
        assert name_compound("C=C(C)C") == "2-methylprop-1-ene"
        assert name_compound("CC(C)=C") == "2-methylprop-1-ene"

    @pytest.mark.integration
    def test_2_methylbutene(self):
        """2-methylbut-2-ene."""
        assert name_compound("CC(C)=CC") == "2-methylbut-2-ene"


class TestUnsaturationWithFunctionalGroups:
    """Test unsaturation combined with functional groups."""

    @pytest.mark.integration
    def test_alkenol_prop_2_en_1_ol(self):
        """Allyl alcohol - IUPAC 2013 prefers retained name.

        'allyl alcohol' is the IUPAC 2013 preferred retained name.
        'prop-2-en-1-ol' is the systematic equivalent.
        """
        result = name_compound("C=CCO")
        # IUPAC 2013: retained name "allyl alcohol" is preferred
        assert result in ["allyl alcohol", "prop-2-en-1-ol"]

    @pytest.mark.integration
    def test_alkenol_prop_1_en_1_ol(self):
        """prop-1-en-1-ol."""
        result = name_compound("CC=CO")
        assert result == "prop-1-en-1-ol"

    @pytest.mark.integration
    def test_saturated_still_works(self):
        """Ensure saturated compounds still work correctly."""
        assert name_compound("CCCO") == "propan-1-ol"
        assert name_compound("CC(O)C") == "propan-2-ol"


class TestBondLocantHelpers:
    """Test the get_bond_locants helper function."""

    @pytest.mark.unit
    def test_get_bond_locants_single_double(self):
        """Test locant resolution for single double bond."""
        from orthonym.rules.locants import get_bond_locants, build_atom_to_locant

        chain = [0, 1, 2, 3]
        double_bonds = [(1, 2)]  # Bond between atoms 1 and 2
        atom_to_locant = build_atom_to_locant(chain)

        locants = get_bond_locants(chain, double_bonds, atom_to_locant)
        # Bond (1,2) -> lower atom by chain position is 1, locant is 2
        assert locants == [2]

    @pytest.mark.unit
    def test_get_bond_locants_multiple_bonds(self):
        """Test locant resolution for multiple bonds."""
        from orthonym.rules.locants import get_bond_locants, build_atom_to_locant

        chain = [0, 1, 2, 3, 4]
        bonds = [(0, 1), (3, 4)]  # Two bonds
        atom_to_locant = build_atom_to_locant(chain)

        locants = get_bond_locants(chain, bonds, atom_to_locant)
        assert locants == [1, 4]  # Sorted

    @pytest.mark.unit
    def test_get_bond_locants_empty(self):
        """Test with no bonds."""
        from orthonym.rules.locants import get_bond_locants, build_atom_to_locant

        chain = [0, 1, 2, 3]
        bonds = []
        atom_to_locant = build_atom_to_locant(chain)

        locants = get_bond_locants(chain, bonds, atom_to_locant)
        assert locants == []
