"""A mixed carboxylic-sulfonic anhydride whose carboxylic side is an amino acid cites the acid word of that side.

 "Mixed anhydrides" (the Blue Book): the two acid names without the word 'acid', in alphanumerical order,
then 'anhydride' ('chloroacetic 4-nitrobenzene-1-sulfonic anhydride (PIN)',:32270). Glycine has no word 'acid', so its
side is cited by its acid word, as in the carboxylic anhydride 'acetic aminoacetic anhydride'.
"""
import pytest

from tests.support.rt_assert import name_best_effort, name_is_rt_exact

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,name", [
    ("NCC(=O)OS(C)(=O)=O", "aminoacetic methanesulfonic anhydride"),
    ("C[C@H](N)C(=O)OS(C)(=O)=O", "(2S)-2-aminopropanoic methanesulfonic anhydride"),
    ("ClCC(=O)OS(=O)(=O)c1ccc([N+](=O)[O-])cc1", "chloroacetic 4-nitrobenzene-1-sulfonic anhydride"),
])
def test_the_amino_acid_side_takes_its_acid_word(smiles, name):
    r = name_best_effort(smiles)
    assert r["name"] == name and r["tier"] == "pin_verified", r
    assert name_is_rt_exact(name, smiles)
