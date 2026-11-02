"""A =O on a ring atom in the lambda state of a fused ring system is a heterone suffix.

 HETERONES (the Blue Book): heterones "are named in the same way as ketones". The ring examples of the
book: '1H-1λ4-thiophen-1-one (PIN)' (:29440), '2H-2λ6-naphtho[1,8-cd][1,2]oxathiole-2,2-dione (PIN)' (:32160),
'1λ6-naphtho[1,8-cd][1,2]thiazole-1,1(2H)-dione (PIN)' (:33254). The lambda atom keeps its bonding number in the
mancude parent, so it counts in the indicated hydrogen, and its =O takes two of its hydrogen atoms.
A benzo name cites the lambda token at its heteroatom locant: '1λ4,5-benzodithiepine (PIN)' (:3326).

These parents are not printed with a lambda state in the book, so the names are labelled below the PIN
(``data.halogen_ring_parents``) and ship at the best-effort tier.
"""
import pytest
from rdkit import Chem

from tests.support.rt_assert import assert_full_rt, name_best_effort

pytestmark = pytest.mark.unit


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", [
    ("OI1(=O)OC(=O)c2ccccc12", "1-hydroxy-1λ5,2-benziodoxole-1,3-dione"),
    ("O=C1NS(=O)(=O)c2ccccc12", "1H-1λ6,2-benzothiazole-1,1,3(2H)-trione"),
    ("CN1C(=O)c2ccccc2S1(=O)=O", "2-methyl-1H-1λ6,2-benzothiazole-1,1,3(2H)-trione"),
    ("O=S1C=Cc2ccccc21", "1H-1λ4-benzothiophen-1-one"),
    ("O=S1(=O)C=Cc2ccccc21", "1H-1λ6-benzothiophene-1,1-dione"),
    ("O=S1(=O)CCc2ccccc21", "2,3-dihydro-1H-1λ6-benzothiophene-1,1-dione"),
    ("O=S1CCc2ccccc21", "2,3-dihydro-1H-1λ4-benzothiophen-1-one"),
])
def test_fused_ring_heterone(smiles, name):
    row = name_best_effort(smiles)
    assert row["name"] == name and row["tier"] == "systematic_verified", row
    assert_full_rt(row["name"], smiles, what="best-effort: ")


def test_the_lambda_token_joins_the_heteroatom_locants_of_a_benzo_name():
    from orthonym.rules.lambda_convention import merge_lambda_into_benzo_locants
    mol = Chem.MolFromSmiles("O=C1NS(=O)(=O)c2ccccc12")
    s = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == 'S')
    n = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == 'N')
    ring = {i for r in mol.GetRingInfo().AtomRings() for i in r}
    locs = {i: 9 for i in ring}
    locs.update({s: 1, n: 2})
    assert merge_lambda_into_benzo_locants("1,2-benzothiazole", mol, ring, locs) == "1λ6,2-benzothiazole"
    # a bracketed fusion name keeps the token at its front
    assert merge_lambda_into_benzo_locants("naphtho[1,8-cd][1,2]thiazole", mol, ring, locs) is None


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", [
    ("O=Br1(O)OC(=O)c2ccccc12", "1-hydroxy-1λ5,2-benzobromoxole-1,3-dione"),
    ("O=Cl1(O)OC(=O)c2ccccc12", "1-hydroxy-1λ5,2-benzochloroxole-1,3-dione"),
])
def test_the_bromine_and_chlorine_analogues_keep_a_name(smiles, name):
    """Review fix: the iodine drawing's 'benziodoxole' keeps the 'o' of 'benzo' before a consonant (f),
    the Blue Book, the elision is before a vowel only)."""
    row = name_best_effort(smiles)
    assert row["name"] == name and row["tier"] == "systematic_verified", row
