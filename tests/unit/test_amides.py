"""Tests for amide naming (POLY-03).

Tests the amide naming module for:
- Primary amides (formamide, acetamide, propanamide)
- Secondary amides (N-methylacetamide, N-ethylacetamide)
- Tertiary amides (N,N-dimethylformamide, N-ethyl-N-methylacetamide)
- Ring-attached amides (cyclohexanecarboxamide)

Based on IUPAC 2013 Blue Book P-66.1.
"""
import pytest
from rdkit import Chem

from orthonym.rules.amides import (
    get_amide_type,
    get_n_substituents,
    format_n_substitution,
    is_ring_attached_amide,
    get_amide_chain_length,
    name_amide,
)


class TestGetAmideType:
    """Test amide type detection (primary, secondary, tertiary)."""

    def test_primary_amide(self):
        """Acetamide (CC(=O)N) is primary."""
        mol = Chem.MolFromSmiles("CC(=O)N")
        pattern = Chem.MolFromSmarts("[CX3](=O)[NX3]")
        matches = mol.GetSubstructMatches(pattern)
        assert len(matches) == 1
        assert get_amide_type(mol, matches[0]) == "primary"

    def test_secondary_amide(self):
        """N-methylacetamide (CNC(C)=O) is secondary."""
        mol = Chem.MolFromSmiles("CNC(C)=O")
        pattern = Chem.MolFromSmarts("[CX3](=O)[NX3]")
        matches = mol.GetSubstructMatches(pattern)
        assert len(matches) == 1
        assert get_amide_type(mol, matches[0]) == "secondary"

    def test_tertiary_amide(self):
        """N,N-dimethylformamide (CN(C)C=O) is tertiary."""
        mol = Chem.MolFromSmiles("CN(C)C=O")
        pattern = Chem.MolFromSmarts("[CX3](=O)[NX3]")
        matches = mol.GetSubstructMatches(pattern)
        assert len(matches) == 1
        assert get_amide_type(mol, matches[0]) == "tertiary"

    def test_formamide_primary(self):
        """Formamide (NC=O) is primary."""
        mol = Chem.MolFromSmiles("NC=O")
        pattern = Chem.MolFromSmarts("[CX3](=O)[NX3]")
        matches = mol.GetSubstructMatches(pattern)
        assert len(matches) == 1
        assert get_amide_type(mol, matches[0]) == "primary"


class TestGetNSubstituents:
    """Test N-substituent detection."""

    def test_no_substituents_primary(self):
        """Primary amides have no N-substituents."""
        mol = Chem.MolFromSmiles("CC(=O)N")
        pattern = Chem.MolFromSmarts("[CX3](=O)[NX3]")
        matches = mol.GetSubstructMatches(pattern)
        subs = get_n_substituents(mol, matches[0])
        assert len(subs) == 0

    def test_one_methyl_substituent(self):
        """N-methylacetamide has one methyl on N."""
        mol = Chem.MolFromSmiles("CNC(C)=O")
        pattern = Chem.MolFromSmarts("[CX3](=O)[NX3]")
        matches = mol.GetSubstructMatches(pattern)
        subs = get_n_substituents(mol, matches[0])
        assert len(subs) == 1
        assert subs[0]["name"] == "methyl"

    def test_two_methyl_substituents(self):
        """N,N-dimethylformamide has two methyls on N."""
        mol = Chem.MolFromSmiles("CN(C)C=O")
        pattern = Chem.MolFromSmarts("[CX3](=O)[NX3]")
        matches = mol.GetSubstructMatches(pattern)
        subs = get_n_substituents(mol, matches[0])
        assert len(subs) == 2
        assert all(s["name"] == "methyl" for s in subs)

    def test_ethyl_substituent(self):
        """N-ethylacetamide has one ethyl on N."""
        mol = Chem.MolFromSmiles("CCNC(C)=O")
        pattern = Chem.MolFromSmarts("[CX3](=O)[NX3]")
        matches = mol.GetSubstructMatches(pattern)
        subs = get_n_substituents(mol, matches[0])
        assert len(subs) == 1
        assert subs[0]["name"] == "ethyl"


class TestFormatNSubstitution:
    """Test N-substitution prefix formatting."""

    def test_single_methyl(self):
        """Single methyl -> N-methyl"""
        subs = [{"atoms": [0], "name": "methyl", "carbon_count": 1}]
        result = format_n_substitution(subs)
        assert result == "N-methyl"

    def test_two_identical_methyls(self):
        """Two methyls -> N,N-dimethyl"""
        subs = [
            {"atoms": [0], "name": "methyl", "carbon_count": 1},
            {"atoms": [1], "name": "methyl", "carbon_count": 1},
        ]
        result = format_n_substitution(subs)
        assert result == "N,N-dimethyl"

    def test_two_different_substituents(self):
        """Ethyl + methyl -> N-ethyl-N-methyl (alphabetized)"""
        subs = [
            {"atoms": [0], "name": "methyl", "carbon_count": 1},
            {"atoms": [1, 2], "name": "ethyl", "carbon_count": 2},
        ]
        result = format_n_substitution(subs)
        # Ethyl comes before methyl alphabetically
        assert result == "N-ethyl-N-methyl"

    def test_two_identical_ethyls(self):
        """Two ethyls -> N,N-diethyl"""
        subs = [
            {"atoms": [0, 1], "name": "ethyl", "carbon_count": 2},
            {"atoms": [2, 3], "name": "ethyl", "carbon_count": 2},
        ]
        result = format_n_substitution(subs)
        assert result == "N,N-diethyl"


class TestIsRingAttachedAmide:
    """Test ring-attached amide detection."""

    def test_chain_amide_not_ring_attached(self):
        """Acetamide is not ring-attached."""
        mol = Chem.MolFromSmiles("CC(=O)N")
        pattern = Chem.MolFromSmarts("[CX3](=O)[NX3]")
        matches = mol.GetSubstructMatches(pattern)
        assert is_ring_attached_amide(mol, matches[0]) is False

    def test_cyclohexane_carboxamide_is_ring_attached(self):
        """Cyclohexanecarboxamide is ring-attached."""
        mol = Chem.MolFromSmiles("NC(=O)C1CCCCC1")
        pattern = Chem.MolFromSmarts("[CX3](=O)[NX3]")
        matches = mol.GetSubstructMatches(pattern)
        assert is_ring_attached_amide(mol, matches[0]) is True


class TestGetAmideChainLength:
    """Test amide chain length detection."""

    def test_formamide_chain_length(self):
        """Formamide (NC=O) has 1 carbon."""
        mol = Chem.MolFromSmiles("NC=O")
        pattern = Chem.MolFromSmarts("[CX3](=O)[NX3]")
        matches = mol.GetSubstructMatches(pattern)
        assert get_amide_chain_length(mol, matches[0]) == 1

    def test_acetamide_chain_length(self):
        """Acetamide (CC(=O)N) has 2 carbons."""
        mol = Chem.MolFromSmiles("CC(=O)N")
        pattern = Chem.MolFromSmarts("[CX3](=O)[NX3]")
        matches = mol.GetSubstructMatches(pattern)
        assert get_amide_chain_length(mol, matches[0]) == 2

    def test_propanamide_chain_length(self):
        """Propanamide (CCC(=O)N) has 3 carbons."""
        mol = Chem.MolFromSmiles("CCC(=O)N")
        pattern = Chem.MolFromSmarts("[CX3](=O)[NX3]")
        matches = mol.GetSubstructMatches(pattern)
        assert get_amide_chain_length(mol, matches[0]) == 3


class TestNameAmide:
    """Test amide name generation."""

    def test_name_formamide(self):
        """NC=O -> formamide"""
        mol = Chem.MolFromSmiles("NC=O")
        pattern = Chem.MolFromSmarts("[CX3](=O)[NX3]")
        matches = mol.GetSubstructMatches(pattern)
        assert name_amide(mol, matches[0]) == "formamide"

    def test_name_acetamide(self):
        """CC(=O)N -> acetamide"""
        mol = Chem.MolFromSmiles("CC(=O)N")
        pattern = Chem.MolFromSmarts("[CX3](=O)[NX3]")
        matches = mol.GetSubstructMatches(pattern)
        assert name_amide(mol, matches[0]) == "acetamide"

    def test_name_propanamide(self):
        """CCC(=O)N -> propanamide"""
        mol = Chem.MolFromSmiles("CCC(=O)N")
        pattern = Chem.MolFromSmarts("[CX3](=O)[NX3]")
        matches = mol.GetSubstructMatches(pattern)
        assert name_amide(mol, matches[0]) == "propanamide"

    def test_name_n_methylacetamide(self):
        """CNC(C)=O -> N-methylacetamide"""
        mol = Chem.MolFromSmiles("CNC(C)=O")
        pattern = Chem.MolFromSmarts("[CX3](=O)[NX3]")
        matches = mol.GetSubstructMatches(pattern)
        assert name_amide(mol, matches[0]) == "N-methylacetamide"

    def test_name_n_n_dimethylformamide(self):
        """CN(C)C=O -> N,N-dimethylformamide"""
        mol = Chem.MolFromSmiles("CN(C)C=O")
        pattern = Chem.MolFromSmarts("[CX3](=O)[NX3]")
        matches = mol.GetSubstructMatches(pattern)
        assert name_amide(mol, matches[0]) == "N,N-dimethylformamide"

    def test_name_n_ethyl_n_methylacetamide(self):
        """CCN(C)C(C)=O -> N-ethyl-N-methylacetamide"""
        mol = Chem.MolFromSmiles("CCN(C)C(C)=O")
        pattern = Chem.MolFromSmarts("[CX3](=O)[NX3]")
        matches = mol.GetSubstructMatches(pattern)
        result = name_amide(mol, matches[0])
        assert result == "N-ethyl-N-methylacetamide"

    def test_name_cyclohexanecarboxamide(self):
        """NC(=O)C1CCCCC1 -> cyclohexanecarboxamide"""
        mol = Chem.MolFromSmiles("NC(=O)C1CCCCC1")
        pattern = Chem.MolFromSmarts("[CX3](=O)[NX3]")
        matches = mol.GetSubstructMatches(pattern)
        assert name_amide(mol, matches[0]) == "cyclohexanecarboxamide"


class TestAmideIntegration:
    """Integration tests using the full naming pipeline."""

    def test_formamide_retained(self):
        """NC=O should be named formamide."""
        from orthonym import name_compound
        result = name_compound("NC=O")
        assert result == "formamide"

    def test_acetamide_retained(self):
        """CC(=O)N should be named acetamide."""
        from orthonym import name_compound
        result = name_compound("CC(=O)N")
        assert result == "acetamide"

    def test_propanamide(self):
        """CCC(=O)N should be named propanamide."""
        from orthonym import name_compound
        result = name_compound("CCC(=O)N")
        assert result == "propanamide"

    def test_butanamide(self):
        """CCCC(=O)N should be named butanamide."""
        from orthonym import name_compound
        result = name_compound("CCCC(=O)N")
        assert result == "butanamide"


class TestSecondaryAmidesIntegration:
    """Integration tests for N-monosubstituted amides."""

    def test_n_methylformamide(self):
        """CNC=O should be named N-methylformamide."""
        from orthonym import name_compound
        result = name_compound("CNC=O")
        assert result == "N-methylformamide"

    def test_n_methylacetamide(self):
        """CNC(C)=O should be named N-methylacetamide."""
        from orthonym import name_compound
        result = name_compound("CNC(C)=O")
        assert result == "N-methylacetamide"

    def test_n_ethylacetamide(self):
        """CCNC(C)=O should be named N-ethylacetamide."""
        from orthonym import name_compound
        result = name_compound("CCNC(C)=O")
        assert result == "N-ethylacetamide"

    def test_n_methylpropanamide(self):
        """CNC(=O)CC should be named N-methylpropanamide."""
        from orthonym import name_compound
        result = name_compound("CNC(=O)CC")
        assert result == "N-methylpropanamide"


class TestTertiaryAmidesIntegration:
    """Integration tests for N,N-disubstituted amides."""

    def test_n_n_dimethylformamide(self):
        """CN(C)C=O should be named N,N-dimethylformamide."""
        from orthonym import name_compound
        result = name_compound("CN(C)C=O")
        assert result == "N,N-dimethylformamide"

    def test_n_n_dimethylacetamide(self):
        """CN(C)C(C)=O should be named N,N-dimethylacetamide."""
        from orthonym import name_compound
        result = name_compound("CN(C)C(C)=O")
        assert result == "N,N-dimethylacetamide"

    def test_n_ethyl_n_methylacetamide(self):
        """CCN(C)C(C)=O -> N-ethyl-N-methylacetamide"""
        from orthonym import name_compound
        result = name_compound("CCN(C)C(C)=O")
        assert result == "N-ethyl-N-methylacetamide"

    def test_n_n_diethylacetamide(self):
        """CCN(CC)C(C)=O should be named N,N-diethylacetamide."""
        from orthonym import name_compound
        result = name_compound("CCN(CC)C(C)=O")
        assert result == "N,N-diethylacetamide"


class TestRingAttachedAmidesIntegration:
    """Integration tests for ring-attached amides."""

    def test_cyclohexanecarboxamide(self):
        """NC(=O)C1CCCCC1 -> cyclohexanecarboxamide"""
        from orthonym import name_compound
        result = name_compound("NC(=O)C1CCCCC1")
        assert result == "cyclohexanecarboxamide"


class TestHydrazidesV42P43:
    """-3 hydrazide PIN-spelling wins (P-66.3).

    Every expectation is a Blue-Book verbatim (PIN) and OPSIN round-trips.
    """

    def test_oxalohydrazide_retained(self):
        # P-66.3.1.2.1 (the Blue Book): 'oxalohydrazide (PIN)' — one of the five
        # retained hydrazide PINs; the systematic form is 'ethanedihydrazide'.
        from orthonym import name_compound
        assert name_compound("C(C(=O)NN)(=O)NN") == "oxalohydrazide"

    def test_dicarbonic_dihydrazide(self):
        # P-66.3.5.2 (the Blue Book): 'dicarbonic dihydrazide (PIN)' — the hydrazide
        # of dicarbonic acid, functional-class over the O-bridged dicarbonic
        # skeleton (was 'bis(1-hydrazinylmethanoic) anhydride').
        from orthonym import name_compound
        assert name_compound("C(=O)(OC(=O)NN)NN") == "dicarbonic dihydrazide"

    def test_plain_hydrazides_regression(self):
        # Regression: the pre-existing systematic/retained hydrazides are intact.
        from orthonym import name_compound
        assert name_compound("C(CCCC)(=O)NN") == "pentanehydrazide"
        assert name_compound("C(C)(=O)NN") == "acetohydrazide"
        assert name_compound("C(C1=CC=CC=C1)(=O)NN") == "benzohydrazide"
