"""A substituted ylidene on an amino group is enclosed: '[(aminomethylidene)amino]', not '(aminomethylideneamino)'.

 "Parentheses" (the Blue Book), (:7232): parentheses around compound prefixes;
'(diaminomethylidene)amino (preferred prefix)' (:17838), '(aminomethylidene)hydrazinyl' (:34519). The functional-group
namer of the wider tiers cited a one-word substituted ylidene bare before 'amino'.
"""
import pytest

from tests.support.pin_tiers import tier_row
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles,name", [
    ("NC=Nc1ccc(C(=O)O)cc1", "4-[(aminomethylidene)amino]benzoic acid"),
    ("ClC=Nc1ccc(C(=O)O)cc1", "4-[(chloromethylidene)amino]benzoic acid"),
    ("NNC=Nc1ccc(C(=O)O)cc1", "4-[(hydrazinylmethylidene)amino]benzoic acid"),
    # simple ylidenes stay bare
    ("CC=Nc1ccc(C(=O)O)cc1", "4-(ethylideneamino)benzoic acid"),
    ("O=C(O)c1ccc(N=Cc2ccccc2)cc1", "4-(benzylideneamino)benzoic acid"),
])
def test_substituted_ylidene_amino_marks(smiles, name):
    row = tier_row(smiles, "best-effort")
    assert row["name"] == name, row
    assert_full_rt(row["name"], smiles)
