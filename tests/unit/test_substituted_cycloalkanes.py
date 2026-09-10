"""
Unit tests for substituted cycloalkane naming.

Tests cover:
- Monosubstituted cycloalkanes (locant omission)
- Disubstituted cycloalkanes (locants required)
- Ring vs chain parent selection
- Substituted cycloalkenes
- Ring orientation for lowest locants
"""

import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.rules.cycloalkanes import (
    orient_cycloalkane,
    get_ring_substituents,
    select_ring_or_chain_parent,
    get_substituent_name,
)


# ============================================================================
# Monosubstituted Cycloalkane Tests
# ============================================================================

class TestMonosubstitutedCycloalkanes:
    """Tests for monosubstituted cycloalkanes (locant 1 is implied)."""

    @pytest.mark.unit
    def test_methylcyclopropane(self):
        """CC1CC1 -> methylcyclopropane (locant omitted)."""
        assert name_compound("CC1CC1") == "methylcyclopropane"

    @pytest.mark.unit
    def test_methylcyclobutane(self):
        """CC1CCC1 -> methylcyclobutane."""
        assert name_compound("CC1CCC1") == "methylcyclobutane"

    @pytest.mark.unit
    def test_methylcyclopentane(self):
        """CC1CCCC1 -> methylcyclopentane."""
        assert name_compound("CC1CCCC1") == "methylcyclopentane"

    @pytest.mark.unit
    def test_methylcyclohexane(self):
        """CC1CCCCC1 -> methylcyclohexane."""
        assert name_compound("CC1CCCCC1") == "methylcyclohexane"

    @pytest.mark.unit
    def test_ethylcyclohexane(self):
        """CCC1CCCCC1 -> ethylcyclohexane."""
        assert name_compound("CCC1CCCCC1") == "ethylcyclohexane"

    @pytest.mark.unit
    def test_propylcyclohexane(self):
        """CCCC1CCCCC1 -> propylcyclohexane."""
        assert name_compound("CCCC1CCCCC1") == "propylcyclohexane"

    @pytest.mark.unit
    def test_butylcyclohexane(self):
        """CCCCC1CCCCC1 -> butylcyclohexane."""
        assert name_compound("CCCCC1CCCCC1") == "butylcyclohexane"

    @pytest.mark.unit
    def test_ethylcyclopentane(self):
        """CCC1CCCC1 -> ethylcyclopentane."""
        assert name_compound("CCC1CCCC1") == "ethylcyclopentane"


# ============================================================================
# Disubstituted Cycloalkane Tests
# ============================================================================

class TestDisubstitutedCycloalkanes:
    """Tests for disubstituted cycloalkanes (locants required)."""

    @pytest.mark.unit
    def test_dimethylcyclohexane_1_2(self):
        """CC1CCCCC1C -> 1,2-dimethylcyclohexane."""
        assert name_compound("CC1CCCCC1C") == "1,2-dimethylcyclohexane"

    @pytest.mark.unit
    def test_dimethylcyclohexane_1_3(self):
        """CC1CCCC(C)C1 -> 1,3-dimethylcyclohexane."""
        assert name_compound("CC1CCCC(C)C1") == "1,3-dimethylcyclohexane"

    @pytest.mark.unit
    def test_dimethylcyclohexane_1_4(self):
        """CC1CCC(C)CC1 -> 1,4-dimethylcyclohexane."""
        assert name_compound("CC1CCC(C)CC1") == "1,4-dimethylcyclohexane"

    @pytest.mark.unit
    def test_dimethylcyclopentane_1_2(self):
        """CC1CCCC1C -> 1,2-dimethylcyclopentane."""
        assert name_compound("CC1CCCC1C") == "1,2-dimethylcyclopentane"

    @pytest.mark.unit
    def test_dimethylcyclopentane_1_3(self):
        """CC1CCC(C)C1 -> 1,3-dimethylcyclopentane."""
        assert name_compound("CC1CCC(C)C1") == "1,3-dimethylcyclopentane"

    @pytest.mark.unit
    def test_ethylmethylcyclohexane(self):
        """Different substituents - alphabetized."""
        # 1-ethyl-2-methylcyclohexane: ethyl before methyl alphabetically
        result = name_compound("CCC1CCCCC1C")
        assert "ethyl" in result
        assert "methyl" in result
        # Check alphabetical order
        ethyl_pos = result.find("ethyl")
        methyl_pos = result.find("methyl")
        assert ethyl_pos < methyl_pos


# ============================================================================
# Ring vs Chain Parent Selection Tests
# ============================================================================

class TestRingVsChainSelection:
    """Tests for ring vs chain parent selection per IUPAC 2013 Method 1."""

    @pytest.mark.unit
    def test_ring_as_parent_methylcyclohexane(self):
        """Ring is parent when attached group is small."""
        # Methylcyclohexane: 6C ring + 1C substituent -> ring is parent
        assert name_compound("CC1CCCCC1") == "methylcyclohexane"

    @pytest.mark.unit
    def test_ring_as_parent_ethylcyclopentane(self):
        """5C ring + 2C chain -> ring is parent."""
        assert name_compound("CCC1CCCC1") == "ethylcyclopentane"

    @pytest.mark.unit
    def test_ring_as_parent_propylcyclobutane(self):
        """4C ring + 3C chain -> ring is parent."""
        assert name_compound("CCCC1CCC1") == "propylcyclobutane"

    @pytest.mark.unit
    def test_select_ring_or_chain_function(self):
        """Test select_ring_or_chain_parent function directly."""
        mol = Chem.MolFromSmiles("CC1CCCCC1")  # methylcyclohexane
        ring = mol.GetRingInfo().AtomRings()[0]
        # Find the chain (methyl substituent)
        ring_set = set(ring)
        chain = [i for i in range(mol.GetNumAtoms()) if i not in ring_set]

        result = select_ring_or_chain_parent(mol, [ring], chain)
        assert result == 'ring'


# ============================================================================
# Ring Substituent Detection Tests
# ============================================================================

class TestRingSubstituentDetection:
    """Tests for get_ring_substituents function."""

    @pytest.mark.unit
    def test_single_substituent_detection(self):
        """Detect single methyl substituent on cyclohexane."""
        mol = Chem.MolFromSmiles("CC1CCCCC1")
        ring = mol.GetRingInfo().AtomRings()[0]
        subs = get_ring_substituents(mol, ring)

        # Should have exactly 1 substituted position
        assert len(subs) == 1

        # Get the substituent atoms
        sub_list = list(subs.values())[0]
        assert len(sub_list) == 1  # One substituent at that position

        # Substituent should have 1 carbon (methyl)
        sub_atoms = sub_list[0]
        carbon_count = sum(
            1 for idx in sub_atoms
            if mol.GetAtomWithIdx(idx).GetSymbol() == 'C'
        )
        assert carbon_count == 1

    @pytest.mark.unit
    def test_two_substituent_detection(self):
        """Detect two substituents on cyclohexane."""
        mol = Chem.MolFromSmiles("CC1CCCCC1C")  # 1,2-dimethylcyclohexane
        ring = mol.GetRingInfo().AtomRings()[0]
        subs = get_ring_substituents(mol, ring)

        # Should have 2 substituted positions
        assert len(subs) == 2

    @pytest.mark.unit
    def test_substituent_name_methyl(self):
        """Get name 'methyl' for 1-carbon substituent."""
        mol = Chem.MolFromSmiles("CC1CCCCC1")
        ring = mol.GetRingInfo().AtomRings()[0]
        subs = get_ring_substituents(mol, ring)

        # Get the first substituent
        sub_list = list(subs.values())[0]
        sub_atoms = sub_list[0]

        name = get_substituent_name(mol, sub_atoms)
        assert name == "methyl"

    @pytest.mark.unit
    def test_substituent_name_ethyl(self):
        """Get name 'ethyl' for 2-carbon substituent."""
        mol = Chem.MolFromSmiles("CCC1CCCCC1")
        ring = mol.GetRingInfo().AtomRings()[0]
        subs = get_ring_substituents(mol, ring)

        sub_list = list(subs.values())[0]
        sub_atoms = sub_list[0]

        name = get_substituent_name(mol, sub_atoms)
        assert name == "ethyl"


# ============================================================================
# Ring Orientation Tests
# ============================================================================

class TestCycloalkaneOrientation:
    """Tests for orient_cycloalkane function."""

    @pytest.mark.unit
    def test_single_substituent_at_position_1(self):
        """Single substituent should be at position 1."""
        mol = Chem.MolFromSmiles("CC1CCCCC1")
        ring = mol.GetRingInfo().AtomRings()[0]
        subs = get_ring_substituents(mol, ring)

        oriented = orient_cycloalkane(mol, ring, subs)

        # First atom should be the substituted one
        assert oriented[0] in subs

    @pytest.mark.unit
    def test_orientation_preserves_atoms(self):
        """Oriented ring contains same atoms as original."""
        mol = Chem.MolFromSmiles("CC1CCCC(C)C1")  # 1,3-dimethylcyclohexane
        ring = mol.GetRingInfo().AtomRings()[0]
        subs = get_ring_substituents(mol, ring)

        oriented = orient_cycloalkane(mol, ring, subs)

        assert set(oriented) == set(ring)

    @pytest.mark.unit
    def test_lowest_locants_1_3_not_1_5(self):
        """1,3-dimethyl should be preferred over 1,5-dimethyl."""
        mol = Chem.MolFromSmiles("CC1CCCC(C)C1")
        ring = mol.GetRingInfo().AtomRings()[0]
        subs = get_ring_substituents(mol, ring)

        oriented = orient_cycloalkane(mol, ring, subs)

        # Build locant set
        atom_to_locant = {atom: i + 1 for i, atom in enumerate(oriented)}
        locants = sorted(
            atom_to_locant[atom] for atom in subs.keys()
        )

        # Should be [1, 3] not [1, 5] (lowest by first-point-of-difference)
        assert locants == [1, 3]


# ============================================================================
# Substituted Cycloalkene Tests
# ============================================================================

class TestSubstitutedCycloalkenes:
    """Tests for cycloalkenes with substituents."""

    @pytest.mark.unit
    def test_methylcyclohexene(self):
        """Methyl on cyclohexene."""
        # CC1=CCCCC1 - methyl on same carbon as double bond
        result = name_compound("CC1=CCCCC1")
        assert "methyl" in result
        assert "cyclohex" in result
        assert "ene" in result

    @pytest.mark.unit
    def test_methylcyclopentene(self):
        """Methyl on cyclopentene."""
        result = name_compound("CC1=CCCC1")
        assert "methyl" in result
        assert "cyclopent" in result
        assert "ene" in result


# ============================================================================
# Locant Omission Verification Tests
# ============================================================================

class TestMonosubstitutedLocantOmission:
    """Tests verifying that monosubstituted cycloalkanes omit locant 1."""

    @pytest.mark.unit
    def test_no_locant_methylcyclohexane(self):
        """Result should be 'methylcyclohexane' not '1-methylcyclohexane'."""
        result = name_compound("CC1CCCCC1")
        assert result == "methylcyclohexane"
        assert not result.startswith("1-")

    @pytest.mark.unit
    def test_no_locant_ethylcyclopentane(self):
        """Result should be 'ethylcyclopentane' not '1-ethylcyclopentane'."""
        result = name_compound("CCC1CCCC1")
        assert result == "ethylcyclopentane"
        assert not result.startswith("1-")

    @pytest.mark.unit
    def test_di_substituted_requires_locants(self):
        """Disubstituted cycloalkanes MUST include locants."""
        result = name_compound("CC1CCCCC1C")
        assert "1,2" in result


# ============================================================================
# Edge Cases and Alternative SMILES
# ============================================================================

class TestSubstitutedCycloalkaneEdgeCases:
    """Tests for edge cases in substituted cycloalkane naming."""

    @pytest.mark.unit
    def test_different_smiles_same_name(self):
        """Same compound with different SMILES should give same name."""
        # Both represent methylcyclohexane
        assert name_compound("CC1CCCCC1") == "methylcyclohexane"
        assert name_compound("C1CCCCC1C") == "methylcyclohexane"

    @pytest.mark.unit
    def test_symmetrical_substitution(self):
        """Symmetrical substitution patterns."""
        # 1,4-dimethylcyclohexane
        result = name_compound("CC1CCC(C)CC1")
        assert result == "1,4-dimethylcyclohexane"


# ============================================================================
# Trisubstituted Cycloalkane Tests
# ============================================================================

class TestTrisubstitutedCycloalkanes:
    """Tests for trisubstituted cycloalkanes."""

    @pytest.mark.unit
    def test_trimethylcyclohexane_1_2_4(self):
        """1,2,4-trimethylcyclohexane."""
        result = name_compound("CC1CCC(C)C(C)C1")
        assert "trimethyl" in result
        assert "cyclohexane" in result
        # Locants should be present
        assert "1" in result

    @pytest.mark.unit
    def test_trimethylcyclohexane_1_3_5(self):
        """1,3,5-trimethylcyclohexane (symmetrical)."""
        result = name_compound("CC1CC(C)CC(C)C1")
        assert "trimethyl" in result
        assert "cyclohexane" in result
