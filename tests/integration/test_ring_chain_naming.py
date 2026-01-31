"""
Integration tests for ring-chain parent selection naming.

Tests IUPAC P-44.1 parent selection and P-61.5 ring substituent naming
for molecules containing both ring and chain systems.

When a functional group is on the chain (not the ring), the chain becomes
the parent and the ring becomes a substituent (phenyl, cyclohexyl, etc.).
"""

import pytest
from orthonym.namer import name_compound


class TestRingChainParentSelection:
    """Test parent selection chooses correct parent (ring or chain)."""

    def test_phenylbutanoic_acid_chain_is_parent(self):
        """4-Phenylbutanoic acid: chain is parent, benzene is phenyl substituent."""
        result = name_compound('c1ccc(CCCC(=O)O)cc1')
        assert result == '4-phenylbutanoic acid'

    def test_phenylpropanoic_acid_chain_is_parent(self):
        """3-Phenylpropanoic acid: chain is parent, benzene is phenyl substituent."""
        result = name_compound('c1ccc(CCC(=O)O)cc1')
        assert result == '3-phenylpropanoic acid'

    def test_phenylacetic_acid_chain_is_parent(self):
        """Phenylacetic acid: 2-carbon chain with FG, chain is parent."""
        result = name_compound('c1ccc(CC(=O)O)cc1')
        # IUPAC systematic: 2-phenylethanoic acid (ethanoic = acetic)
        assert result == '2-phenylethanoic acid'

    def test_benzoic_acid_ring_is_parent(self):
        """Benzoic acid: FG directly on ring, so ring is parent.

        Benzoic acid is a retained name. The carboxylic acid is directly
        attached to the benzene ring (carbonyl carbon bonded to ring carbon).
        """
        result = name_compound('c1ccc(C(=O)O)cc1')
        assert result == 'benzoic acid'

    def test_phenylpropanol_chain_is_parent(self):
        """3-Phenylpropan-1-ol: alcohol on chain."""
        result = name_compound('c1ccc(CCCO)cc1')
        assert result == '3-phenylpropan-1-ol'

    def test_cyclohexylbutanoic_acid(self):
        """4-Cyclohexylbutanoic acid: cycloalkane as substituent."""
        result = name_compound('C1CCCCC1CCCC(=O)O')
        assert result == '4-cyclohexylbutanoic acid'

    def test_cyclohexylpropanoic_acid(self):
        """3-Cyclohexylpropanoic acid: cycloalkane as substituent."""
        result = name_compound('C1CCCCC1CCC(=O)O')
        assert result == '3-cyclohexylpropanoic acid'

    def test_cyclopentylbutanoic_acid(self):
        """4-Cyclopentylbutanoic acid: cyclopentane as substituent."""
        result = name_compound('C1CCCC1CCCC(=O)O')
        assert result == '4-cyclopentylbutanoic acid'


class TestRingSubstituentNames:
    """Test correct ring substituent names are used."""

    def test_phenyl_from_benzene(self):
        """Benzene substituent is named phenyl."""
        result = name_compound('c1ccc(CCC(=O)O)cc1')
        assert 'phenyl' in result.lower()

    def test_cyclohexyl_from_cyclohexane(self):
        """Cyclohexane substituent is named cyclohexyl."""
        result = name_compound('C1CCCCC1CCC(=O)O')
        assert 'cyclohexyl' in result.lower()

    def test_cyclopentyl_from_cyclopentane(self):
        """Cyclopentane substituent is named cyclopentyl."""
        result = name_compound('C1CCCC1CCC(=O)O')
        assert 'cyclopentyl' in result.lower()

    def test_cyclopropyl_from_cyclopropane(self):
        """Cyclopropane substituent is named cyclopropyl."""
        result = name_compound('C1CC1CCC(=O)O')
        assert 'cyclopropyl' in result.lower()

    def test_cyclobutyl_from_cyclobutane(self):
        """Cyclobutane substituent is named cyclobutyl."""
        result = name_compound('C1CCC1CCC(=O)O')
        assert 'cyclobutyl' in result.lower()


class TestRetainedNamesPreserved:
    """Test that retained names are not broken by parent selection."""

    def test_benzene_alone(self):
        """Unsubstituted benzene still named benzene."""
        result = name_compound('c1ccccc1')
        assert result == 'benzene'

    def test_toluene(self):
        """Toluene retained name preserved."""
        result = name_compound('Cc1ccccc1')
        assert result == 'toluene'

    def test_benzoic_acid_retained(self):
        """Benzoic acid retained name preserved."""
        result = name_compound('c1ccc(C(=O)O)cc1')
        assert result == 'benzoic acid'

    def test_phenol_retained(self):
        """Phenol retained name preserved."""
        result = name_compound('c1ccc(O)cc1')
        assert result == 'phenol'

    def test_cyclohexane_alone(self):
        """Unsubstituted cyclohexane still named cyclohexane."""
        result = name_compound('C1CCCCC1')
        assert result == 'cyclohexane'

    def test_methylcyclohexane(self):
        """Methylcyclohexane naming preserved."""
        result = name_compound('CC1CCCCC1')
        assert result == 'methylcyclohexane'


class TestLocantAssignment:
    """Test locants are correctly assigned for ring substituents."""

    def test_4_phenylbutanoic_acid_locant(self):
        """Phenyl at position 4 of butanoic acid."""
        result = name_compound('c1ccc(CCCC(=O)O)cc1')
        assert '4-phenyl' in result

    def test_3_phenylpropanoic_acid_locant(self):
        """Phenyl at position 3 of propanoic acid."""
        result = name_compound('c1ccc(CCC(=O)O)cc1')
        assert '3-phenyl' in result

    def test_2_phenylethanoic_acid_locant(self):
        """Phenyl at position 2 of ethanoic acid (systematic name for acetic acid)."""
        result = name_compound('c1ccc(CC(=O)O)cc1')
        # IUPAC systematic: 2-phenylethanoic acid
        assert result == '2-phenylethanoic acid'

    def test_4_cyclohexylbutanoic_acid_locant(self):
        """Cyclohexyl at position 4 of butanoic acid."""
        result = name_compound('C1CCCCC1CCCC(=O)O')
        assert '4-cyclohexyl' in result


class TestEdgeCases:
    """Test edge cases in ring-chain naming."""

    def test_phenylalanine_amino_acid(self):
        """Phenylalanine should use trivial name (amino acid)."""
        # NC(Cc1ccccc1)C(=O)O = phenylalanine
        result = name_compound('NC(Cc1ccccc1)C(=O)O')
        assert result == 'phenylalanine'

    def test_phenyl_with_alcohol(self):
        """Phenyl substituent with alcohol on chain."""
        result = name_compound('c1ccc(CCO)cc1')
        # PIN style: 2-phenylethan-1-ol (with locant for -ol)
        assert result == '2-phenylethan-1-ol'

    def test_short_chain_no_fg(self):
        """Short chain with ring but no functional group."""
        # Ethylbenzene should use retained name path, not ring-chain selection
        result = name_compound('CCc1ccccc1')
        # Should be ethylbenzene (ring is parent, chain is substituent)
        assert result == 'ethylbenzene'

    def test_long_alkyl_on_benzene(self):
        """Propylbenzene: still ring is parent for simple alkyl."""
        result = name_compound('CCCc1ccccc1')
        # Propylbenzene or 1-propylbenzene
        assert 'benzene' in result.lower()


class TestAlphabetizationWithRingSubstituents:
    """Test alphabetization works correctly with ring substituents."""

    def test_methyl_and_phenyl_alphabetized(self):
        """Multiple substituents including ring are alphabetized."""
        # 3-methyl-4-phenylbutanoic acid: methyl before phenyl
        result = name_compound('c1ccc(CC(C)C(=O)O)cc1')
        # Should have methyl and phenyl in alphabetical order
        if 'methyl' in result and 'phenyl' in result:
            methyl_pos = result.find('methyl')
            phenyl_pos = result.find('phenyl')
            assert methyl_pos < phenyl_pos, f"Expected methyl before phenyl in {result}"
