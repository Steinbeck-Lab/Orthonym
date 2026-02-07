"""
Unit tests for skeletal replacement ("a") nomenclature.

Tests IUPAC P-15.4 replacement naming for chains with embedded heteroatoms.
"""

import pytest
from rdkit import Chem

from orthonym.rules.skeletal_replacement import (
    try_skeletal_replacement_name,
    REPLACEMENT_TERMS,
)
from orthonym import name_compound


# ============================================================================
# Module-level tests
# ============================================================================

@pytest.mark.unit
class TestReplacementTerms:
    """Test the REPLACEMENT_TERMS table."""

    def test_oxa_present(self):
        assert REPLACEMENT_TERMS['O'] == 'oxa'

    def test_aza_present(self):
        assert REPLACEMENT_TERMS['N'] == 'aza'

    def test_thia_present(self):
        assert REPLACEMENT_TERMS['S'] == 'thia'

    def test_selena_present(self):
        assert REPLACEMENT_TERMS['Se'] == 'selena'

    def test_phospha_present(self):
        assert REPLACEMENT_TERMS['P'] == 'phospha'

    def test_sila_present(self):
        assert REPLACEMENT_TERMS['Si'] == 'sila'


# ============================================================================
# Basic dioxa chains
# ============================================================================

@pytest.mark.unit
class TestDioxaChains:
    """Test replacement naming for chains with embedded oxygen atoms."""

    def test_dioxahexane(self):
        """COCCOC -> 2,5-dioxahexane"""
        mol = Chem.MolFromSmiles('COCCOC')
        assert try_skeletal_replacement_name(mol) == '2,5-dioxahexane'

    def test_dioxaoctane(self):
        """CCOCCOCC -> 3,6-dioxaoctane"""
        mol = Chem.MolFromSmiles('CCOCCOCC')
        assert try_skeletal_replacement_name(mol) == '3,6-dioxaoctane'

    def test_trioxanonane(self):
        """COCCOCCOC -> 2,5,8-trioxanonane"""
        mol = Chem.MolFromSmiles('COCCOCCOC')
        assert try_skeletal_replacement_name(mol) == '2,5,8-trioxanonane'

    def test_trioxaundecane(self):
        """CCOCCOCCOCC -> 3,6,9-trioxaundecane"""
        mol = Chem.MolFromSmiles('CCOCCOCCOCC')
        assert try_skeletal_replacement_name(mol) == '3,6,9-trioxaundecane'


# ============================================================================
# Aza chains
# ============================================================================

@pytest.mark.unit
class TestAzaChains:
    """Test replacement naming for chains with embedded nitrogen atoms."""

    def test_azahexane(self):
        """CCNCCC -> 3-azahexane (single N, chain >= 5)"""
        mol = Chem.MolFromSmiles('CCNCCC')
        assert try_skeletal_replacement_name(mol) == '3-azahexane'

    def test_diazaheptane(self):
        """CNCCNCC -> 2,5-diazaheptane"""
        mol = Chem.MolFromSmiles('CNCCNCC')
        assert try_skeletal_replacement_name(mol) == '2,5-diazaheptane'

    def test_single_n_5atom_not_replacement(self):
        """CCNCC -> None (single N, chain = 5, below threshold of 6)"""
        mol = Chem.MolFromSmiles('CCNCC')
        assert try_skeletal_replacement_name(mol) is None


# ============================================================================
# Thia chains
# ============================================================================

@pytest.mark.unit
class TestThiaChains:
    """Test replacement naming for chains with embedded sulfur atoms."""

    def test_dithiahexane(self):
        """CSCCSC -> 2,5-dithiahexane"""
        mol = Chem.MolFromSmiles('CSCCSC')
        assert try_skeletal_replacement_name(mol) == '2,5-dithiahexane'


# ============================================================================
# Mixed heteroatom chains
# ============================================================================

@pytest.mark.unit
class TestMixedHeteroatomChains:
    """Test replacement naming for chains with different heteroatom types."""

    def test_oxa_aza_octane(self):
        """COCCNCCC -> 2-oxa-5-azaoctane (ascending locant order)"""
        mol = Chem.MolFromSmiles('COCCNCCC')
        assert try_skeletal_replacement_name(mol) == '2-oxa-5-azaoctane'


# ============================================================================
# Boundary cases -- should return None
# ============================================================================

@pytest.mark.unit
class TestBoundaryCases:
    """Test cases that should NOT trigger skeletal replacement."""

    def test_dimethyl_ether_too_short(self):
        """COC -> None (3-atom chain, single O, too short)"""
        mol = Chem.MolFromSmiles('COC')
        assert try_skeletal_replacement_name(mol) is None

    def test_diethyl_ether_too_short(self):
        """COCC -> None (4-atom chain, single O, too short)"""
        mol = Chem.MolFromSmiles('COCC')
        assert try_skeletal_replacement_name(mol) is None

    def test_ethoxyethane_single_o_5atom(self):
        """CCOCC -> None (5-atom chain, single O, uses substitutive naming)"""
        mol = Chem.MolFromSmiles('CCOCC')
        assert try_skeletal_replacement_name(mol) is None

    def test_ethanol_terminal_oh(self):
        """CCO -> None (terminal OH = alcohol)"""
        mol = Chem.MolFromSmiles('CCO')
        assert try_skeletal_replacement_name(mol) is None

    def test_acetic_acid_priority_fg(self):
        """CC(=O)O -> None (carboxylic acid is priority FG)"""
        mol = Chem.MolFromSmiles('CC(=O)O')
        assert try_skeletal_replacement_name(mol) is None

    def test_benzene_cyclic(self):
        """c1ccccc1 -> None (cyclic molecule)"""
        mol = Chem.MolFromSmiles('c1ccccc1')
        assert try_skeletal_replacement_name(mol) is None

    def test_cyclohexane_cyclic(self):
        """C1CCCCC1 -> None (cyclic molecule)"""
        mol = Chem.MolFromSmiles('C1CCCCC1')
        assert try_skeletal_replacement_name(mol) is None

    def test_acetaldehyde_priority_fg(self):
        """CC=O -> None (aldehyde is priority FG)"""
        mol = Chem.MolFromSmiles('CC=O')
        assert try_skeletal_replacement_name(mol) is None

    def test_acetone_priority_fg(self):
        """CC(=O)C -> None (ketone is priority FG)"""
        mol = Chem.MolFromSmiles('CC(=O)C')
        assert try_skeletal_replacement_name(mol) is None

    def test_methylamine_terminal_nh2(self):
        """CN -> None (terminal NH2 = amine)"""
        mol = Chem.MolFromSmiles('CN')
        assert try_skeletal_replacement_name(mol) is None

    def test_branched_chain_rejected(self):
        """COCCOCC(C)C -> None (has substituent branch)"""
        mol = Chem.MolFromSmiles('COCCOCC(C)C')
        assert try_skeletal_replacement_name(mol) is None

    def test_none_molecule(self):
        """None mol -> None"""
        assert try_skeletal_replacement_name(None) is None

    def test_pure_hydrocarbon(self):
        """CCCCCC -> None (no heteroatoms)"""
        mol = Chem.MolFromSmiles('CCCCCC')
        assert try_skeletal_replacement_name(mol) is None

    def test_nitrile_priority_fg(self):
        """CC#N -> None (nitrile is priority FG)"""
        mol = Chem.MolFromSmiles('CC#N')
        assert try_skeletal_replacement_name(mol) is None

    def test_thiol_terminal(self):
        """CCS -> None (terminal SH = thiol)"""
        mol = Chem.MolFromSmiles('CCS')
        assert try_skeletal_replacement_name(mol) is None


# ============================================================================
# End-to-end via name_compound
# ============================================================================

@pytest.mark.unit
class TestEndToEnd:
    """Test skeletal replacement via the full name_compound pipeline."""

    def test_dioxahexane_e2e(self):
        """name_compound('COCCOC') -> '2,5-dioxahexane'"""
        assert name_compound('COCCOC') == '2,5-dioxahexane'

    def test_azahexane_e2e(self):
        """name_compound('CCNCCC') -> '3-azahexane'"""
        assert name_compound('CCNCCC') == '3-azahexane'

    def test_dioxaoctane_e2e(self):
        """name_compound('CCOCCOCC') -> '3,6-dioxaoctane'"""
        assert name_compound('CCOCCOCC') == '3,6-dioxaoctane'

    def test_trioxanonane_e2e(self):
        """name_compound('COCCOCCOC') -> '2,5,8-trioxanonane'"""
        assert name_compound('COCCOCCOC') == '2,5,8-trioxanonane'

    def test_mixed_oxa_aza_e2e(self):
        """name_compound('COCCNCCC') -> '2-oxa-5-azaoctane'"""
        assert name_compound('COCCNCCC') == '2-oxa-5-azaoctane'

    def test_coc_not_replacement(self):
        """COC should NOT produce a replacement name."""
        result = name_compound('COC')
        assert 'oxa' not in result
        assert result == 'methoxymethane'

    def test_ethanol_not_replacement(self):
        """CCO should still be 'ethanol'."""
        assert name_compound('CCO') == 'ethanol'

    def test_dithiahexane_e2e(self):
        """name_compound('CSCCSC') -> '2,5-dithiahexane'"""
        assert name_compound('CSCCSC') == '2,5-dithiahexane'
