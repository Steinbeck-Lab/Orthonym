"""Tetracene is spelled 'tetracene', never 'naphthacene' "Polyacenes").

the Blue Book: "tetracene (PIN) (formerly naphthacene) pentacene (PIN)". The fused-ring
catalogue named the parent 'naphthacene', so every substituted tetracene shipped as pin_verified
with a general-nomenclature name (d825, d1491, d1148 of dev2000, 'naphthacene-5,12-dione').
The name must round-trip through OPSIN to the input structure (change-asserted-value: the
quote above is the primary source; OPSIN reads both spellings, so the read-back checks the
molecule, the quote checks the spelling).
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym.validation.opsin_roundtrip import opsin_parse

pytestmark = pytest.mark.opsin_gate

TETRACENE_DIONE = "O=C1c2ccccc2C(=O)c2cc3ccccc3cc12"
D825 = "CC1(O)CC(O)Cc2cc3c(c(O)c21)C(=O)c1c(O)cc(O)cc1C3(C)C"
D1491 = ("COc1cc2c(c(O)c1Cl)C(=O)c1c(cc3c(Cl)c(O)cc(-c4c(C)c(Cl)c(O)c(Cl)c4O)c3c1O)C2(C)C")


def _same(smi, name):
    out = opsin_parse(name)
    m_in, m_out = Chem.MolFromSmiles(smi), Chem.MolFromSmiles(out) if out else None
    return m_out is not None and inchi.MolToInchiKey(m_in) == inchi.MolToInchiKey(m_out)


def test_bare_tetracene(namer):
    assert namer.name("c1ccc2cc3cc4ccccc4cc3cc2c1") == "tetracene"


def test_tetracene_dione(namer):
    name = namer.name(TETRACENE_DIONE)
    assert name == "tetracene-5,12-dione", name
    assert _same(TETRACENE_DIONE, name), name


@pytest.mark.parametrize("smiles,expected", [
    (D825, "2,4,6,7,9-pentahydroxy-7,12,12-trimethyl-7,8,9,10-tetrahydrotetracen-5(12H)-one"),
    (D1491, "3,10-dichloro-7-(3,5-dichloro-2,4-dihydroxy-6-methylphenyl)-4,6,9-trihydroxy-2-methoxy-"
            "12,12-dimethyltetracen-5(12H)-one"),
])
def test_substituted_tetracene_spelling_and_readback(namer, smiles, expected):
    name = namer.name(smiles)
    assert name == expected, name
    assert "naphthacen" not in name
    assert _same(smiles, name), name


def test_pentacene_still_not_a_tetracene(namer):
    # the coverage guard keeps a 5-ring acene from matching the 4-ring catalogue entry
    assert namer.name("c1ccc2cc3cc4cc5ccccc5cc4cc3cc2c1") == "pentacene"
