""" — (R-CO)2NH named as N-acyl derivative of the SENIOR primary amide.

BB (Diacylamines): a diacylamide is named as the N-acyl derivative of the
senior primary amide; verbatim example 'N-acetylbenzamide (PIN)'. The senior acid is
chosen by (here both are carboxylic acids, so ring>chain: benzoic >
acetic), so benzamide is the parent and acetyl the N-acyl prefix -- NOT the other way.

Root cause was decomposition/engine.py::_select_best_bond choosing the cleaved acid by
an atom-balance heuristic with no seniority awareness. The fix re-picks within a
shared-nitrogen diacylamide sibling pair to cleave the LESS senior acid.

Every emitted name OPSIN-round-trips to the input (0-wrong).
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.validation.opsin_roundtrip import opsin_parse

_FLAGS = dict(general_fallback=True, general_fallback_unverified=True,
              allow_aromatic_general=True)


@pytest.fixture(scope="module")
def engine():
    return Orthonym(**_FLAGS)


def _inchikey(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchiKey(mol) if mol is not None else None


@pytest.mark.parametrize("smiles,expected", [
    ("CC(=O)NC(=O)c1ccccc1", "N-acetylbenzamide"),                        # def_id 66.1.2.1
    ("CC(=O)N(C(=O)c1ccccc1)c1ccc2ccccc2c1",
     "N-acetyl-N-(naphthalen-2-yl)benzamide"),                            # def_id 66.1.2.1
])
def test_diacylamide_names_senior_acid_as_parent(engine, smiles, expected):
    got = engine.name_tiered(smiles)["name"]
    assert got == expected, f"{smiles}: got {got!r}"
    rt = opsin_parse(got)
    assert rt and _inchikey(rt) == _inchikey(smiles), f"RT failed for {got!r}"


@pytest.mark.parametrize("smiles,expected", [
    # identical acyls -> genuine seniority tie -> balance heuristic unchanged
    ("O=CNC=O", "N-formylformamide"),
    ("CC(=O)N(C(C)=O)C1CCCC1", "N-acetyl-N-cyclopentylacetamide"),
    # not a diacylamide sibling pair (urea + amide) -> untouched
    ("CC(=O)NCCNC(N)=O", "N-[2-(carbamoylamino)ethyl]acetamide"),
])
def test_seniority_tiebreak_does_not_disturb_ties_or_nonsiblings(engine, smiles, expected):
    assert engine.name_tiered(smiles)["name"] == expected
