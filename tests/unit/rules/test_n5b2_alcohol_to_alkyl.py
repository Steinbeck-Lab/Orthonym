"""'...furan-3a(4H)-ol' becomes '...furan-3a(4H)-yl' in the ester and amide decomposition.

``_alcohol_to_alkyl`` read the locant of a '-ol' name as digits only, so a fusion letter or the
'added indicated hydrogen' of a ring parent, the Blue Book: 'quinolin-2(1H)-one')
fell through to the contracted elision and dropped the hyphen: '...3a(4H)yl acetate',
'decahydronaphthalen-4ayl'."""
import pytest

from orthonym.decomposition.fragment_assembly import _alcohol_to_alkyl


@pytest.mark.parametrize("alcohol,alkyl", [
    ("dihydro-1H,3H-furo[3,4-c]furan-3a(4H)-ol", "dihydro-1H,3H-furo[3,4-c]furan-3a(4H)-yl"),
    ("octahydronaphthalen-4a(2H)-ol", "octahydronaphthalen-4a(2H)-yl"),
    ("decahydronaphthalen-4a-ol", "decahydronaphthalen-4a-yl"),
    ("octahydro-1H-inden-3a-ol", "octahydro-1H-inden-3a-yl"),
    ("heptan-3-ol", "heptan-3-yl"),                     # unchanged
    ("propan-1-ol", "propyl"),                          # unchanged
    ("ethanol", "ethyl"),                               # unchanged
    ("1,2,3,4-tetrahydronaphthalen-1-ol", "1,2,3,4-tetrahydronaphthalen-1-yl"),   # unchanged
])
def test_the_locant_keeps_its_letters_and_added_hydrogen(alcohol, alkyl):
    assert _alcohol_to_alkyl(alcohol) == alkyl


def test_the_ester_of_a_fusion_atom_alcohol_has_its_hyphen():
    from tests.support.default_tier import default_tier_row
    from tests.support.rt_assert import assert_full_rt
    smiles = "COc1cc(C2OCC3(OC(C)=O)C(c4ccc(O)c(OC)c4)OCC23)ccc1O"
    name = default_tier_row(smiles)["name"]
    assert ")yl" not in name and "3a(4H)-yl acetate" in name, name
    assert_full_rt(name, smiles)
