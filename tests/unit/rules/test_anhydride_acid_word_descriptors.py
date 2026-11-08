"""Each acid word of an anhydride carries its own descriptors (program 20, 2026-11-08).

 "Mixed anhydrides" (the Blue Book) names an anhydride by its acid words; (:32376) gives
'bis[(2S)-2-aminopropanoic] hexanedioic dianhydride', each acid word with its own descriptors; (:44643) puts a
descriptor before the part of the name it relates to. The engine used to put one block in front of the whole name:
'(2S)-acetic 2-methylbutanoic anhydride' (labelled pin_verified, the descriptor on 'acetic') and
'(2S)-methanesulfonic (2S)-2-methylbutanoic anhydride' (cited twice).
"""
import pytest

from orthonym import Orthonym

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles, name", [
    ("CC[C@H](C)C(=O)OC(=O)C", "acetic (2S)-2-methylbutanoic anhydride"),
    ("CC(=O)OC(=O)/C=C/C", "acetic (2E)-but-2-enoic anhydride"),
    ("N[C@@H](C)C(=O)OC(=O)C", "acetic (2S)-2-aminopropanoic anhydride"),
    ("CC[C@H](C)C(=O)OC(=O)[C@H](C)CC", "(2R)-2-methylbutanoic (2S)-2-methylbutanoic anhydride"),
    ("CC[C@H](C)C(=O)OC(=O)[C@@H](C)CC", "bis[(2S)-2-methylbutanoic] anhydride"),
    ("CC[C@H](C)C(=O)OS(C)(=O)=O", "methanesulfonic (2S)-2-methylbutanoic anhydride"),
    # review 2026-11-08: every route that names acid words (thio-, peroxy-, cyanic anhydrides)
    ("CC[C@H](C)C(=O)SC(=O)C", "acetic (2S)-2-methylbutanoic thioanhydride"),
    ("CC[C@H](C)C(=O)OOC(=O)C", "acetic (2S)-2-methylbutanoic peroxyanhydride"),
    ("CC[C@H](C)C(=O)OC#N", "cyanic (2S)-2-methylbutanoic anhydride"),
    ("CC[C@H](C)C(=O)OOC(=O)[C@@H](C)CC", "bis[(2S)-2-methylbutanoic] peroxyanhydride"),
    # a one-word name (a cyclic anhydride as a ring dione) keeps the parent's block
    ("C[C@@H]1CC(=O)OC1=O", "(3R)-3-methyloxolane-2,5-dione"),
    ("C[C@@H]1C(=O)OC(=O)[C@H]1C", "(3S,4S)-3,4-dimethyloxolane-2,5-dione"),
    # without a stereo element: unchanged
    ("CC(Cl)C(=O)OC(=O)C", "acetic 2-chloropropanoic anhydride"),
    ("ClCC(=O)OC(=O)CCl", "bis(chloroacetic) anhydride"),
    ("CC(=O)OC(=O)C", "acetic anhydride"),
])
def test_each_acid_word_carries_its_own_descriptors(smiles, name):
    row = Orthonym().name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "pin_verified")
