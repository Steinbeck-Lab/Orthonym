"""Unit tests for the chalcogen-chain parent-hydride namer (v23 Phase 7).

P-21.2.2: a homogeneous O/S/Se/Te chain -> ``<multiplier><stem>`` (dioxidane,
trisulfane, …). Carbon substitution admitted only for >=3 chalcogens (sulfides /
disulfides are a distinct functional class). Fail-closed everywhere else.
"""
import pytest
from rdkit import Chem

from orthonym.rules.polychalcogen import name_chalcogen_chain


def _name(smiles):
    return name_chalcogen_chain(Chem.MolFromSmiles(smiles))


class TestBareChains:
    @pytest.mark.parametrize("smiles,expected", [
        ("OO", "dioxidane"),
        ("OOO", "trioxidane"),
        ("OOOO", "tetraoxidane"),
        ("SS", "disulfane"),
        ("SSS", "trisulfane"),
        ("SSSS", "tetrasulfane"),
        ("SSSSS", "pentasulfane"),
        ("[SeH][SeH]", "diselane"),
        ("[TeH][TeH]", "ditellane"),
    ])
    def test_bare_chains(self, smiles, expected):
        assert _name(smiles) == expected


class TestCarbonSubstituted:
    @pytest.mark.parametrize("smiles,expected", [
        ("CSSS", "1-methyltrisulfane"),
        ("CSSSC", "1,3-dimethyltrisulfane"),
    ])
    def test_terminal_organyl_min_three(self, smiles, expected):
        assert _name(smiles) == expected


class TestFailClosed:
    @pytest.mark.parametrize("smiles,why", [
        ("CSC", "1-S carbon chain is a sulfide, not a sulfane"),
        ("CSSC", "2-S carbon chain is a disulfide, not a disulfane"),
        ("COC", "dimethyl ether (1 O carbon chain)"),
        ("CSCC", "ethyl methyl sulfide"),
        ("O=S(=O)(O)O", "sulfuric acid: S bears =O"),
        ("OS(=O)(=O)O", "a sulfur oxoacid form"),
        ("OOC", "2-O carbon chain (methyl hydroperoxide / dioxidane is bare-only at n=2)"),
        ("[O-]OO", "charged"),
        ("OCO", "carbon between the oxygens (not an O-O chain)"),
    ])
    def test_declines(self, smiles, why):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            pytest.skip(f"RDKit rejects SMILES ({why})")
        assert name_chalcogen_chain(mol) is None, why

    def test_none_input(self):
        assert name_chalcogen_chain(None) is None
