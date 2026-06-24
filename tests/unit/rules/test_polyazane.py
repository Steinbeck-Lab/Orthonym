"""Unit tests for the polyazane parent-hydride namer (v23 Phase 7).

P-68.3.1.1 / P-68.3.1.3 / P-21.2.2: chains of N atoms joined by N-N bonds.
hydrazine / diazene are retained PINs; longer members systematic. Azo R-N=N-R is
a substituted diazene. Fail-closed on amines / diamines / hydroxylamine /
hydrazones / azides / rings.
"""
import pytest
from rdkit import Chem

from orthonym.rules.polyazane import name_polyazane


def _name(smiles):
    return name_polyazane(Chem.MolFromSmiles(smiles))


class TestBareSaturated:
    @pytest.mark.parametrize("smiles,expected", [
        ("NN", "hydrazine"),
        ("NNN", "triazane"),
        ("NNNN", "tetraazane"),
        ("NNNNN", "pentaazane"),
        ("NNNNNN", "hexaazane"),
    ])
    def test_saturated(self, smiles, expected):
        assert _name(smiles) == expected


class TestBareUnsaturated:
    @pytest.mark.parametrize("smiles,expected", [
        ("N=N", "diazene"),
        ("N=NN", "triaz-1-ene"),
        ("N=NNN", "tetraaz-1-ene"),
    ])
    def test_unsaturated(self, smiles, expected):
        assert _name(smiles) == expected


class TestSubstituted:
    @pytest.mark.parametrize("smiles,expected", [
        ("CNN", "methylhydrazine"),              # mono -> no locant (BB phenylhydrazine precedent)
        ("CCNN", "ethylhydrazine"),
        ("c1ccccc1NN", "phenylhydrazine"),
        ("CNNC", "1,2-dimethylhydrazine"),
        ("CN(C)N", "1,1-dimethylhydrazine"),
        ("CN=N", "methyldiazene"),               # mono diazene -> no locant
        ("CN=NC", "1,2-dimethyldiazene"),
        ("CCN=NCC", "1,2-diethyldiazene"),
        ("c1ccccc1N=Nc1ccccc1", "1,2-diphenyldiazene"),  # azobenzene (NOT a PIN)
    ])
    def test_substituted(self, smiles, expected):
        assert _name(smiles) == expected


class TestFailClosed:
    @pytest.mark.parametrize("smiles,why", [
        ("CN", "methylamine: no N-N bond"),
        ("NCCN", "ethylenediamine: N-C-C-N, no N-N bond"),
        ("NCCCN", "1,3-diaminopropane"),
        ("NO", "hydroxylamine: N-O, not N-N"),
        ("CC(C)=NN", "acetone hydrazone: C=N-N ylidene substituent"),
        ("[N-]=[N+]=N", "azide: charged + two cumulated double bonds"),
        ("c1cc[nH]n1", "pyrazole: N-N in a ring"),
        ("c1ccccc1N", "aniline: single N"),
    ])
    def test_declines(self, smiles, why):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            pytest.skip(f"RDKit rejects SMILES ({why})")
        assert name_polyazane(mol) is None, why

    def test_none_input(self):
        assert name_polyazane(None) is None
