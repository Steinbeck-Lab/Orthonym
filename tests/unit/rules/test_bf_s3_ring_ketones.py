"""Slice S3 (Task S3.3): a ketone on a ring atom of the fused parent of a bridged fused name.

The ketone needs a >CH2 group, the Blue Book-:28388): indicated hydrogen
of the mancude bridged parent holds it when the parent allows:24768,
'1,2,3,7,8,8a-hexahydro-4H-3a,7-methanoazulene-4,9-dione (PIN)':24792), otherwise 'added
indicated hydrogen' follows the suffix locants:24693,:24695,
'3,4-dihydronaphthalen-1(2H)-one (PIN)':3276), and a pair of ketones that removes one double
bond needs none:24721, '2,3-dihydronaphthalene-1,4-dione (PIN)':24884). When a
ring reading and a bridge reading of one parent tie, (c) (:3256) puts the ketone in the
ring. Every name was read back by OPSIN 2.9.0 to the input's full InChIKey (S3 planning
notes, ledger)."""
import pytest
from rdkit import Chem

from orthonym.rules import bridged_fused_pin as pkg
from orthonym.rules.bridged_fused_pin import build

RING_KETONES = [
    ("O=C1C=CC2CC3CCCC13C2=O", "1,2,3,7,8,8a-hexahydro-4H-3a,7-methanoazulene-4,9-dione"),
    ("O=C1CC2CC1c1ccccc12", "3,4-dihydro-1,4-methanonaphthalen-2(1H)-one"),
    ("O=C1CC2CCC1c1ccccc12", "3,4-dihydro-1,4-ethanonaphthalen-2(1H)-one"),
    ("O=C1CC2CCC1C1CCCCC12", "octahydro-1,4-ethanonaphthalen-2(1H)-one"),
    ("O=C1C(=O)C2CCC1c1ccccc12", "1,4-dihydro-1,4-ethanonaphthalene-2,3-dione"),
    ("O=C1C=CC(=O)C2C3CCC(C3)C12", "1,2,3,4,4a,8a-hexahydro-1,4-methanonaphthalene-5,8-dione"),
    ("O=C1CCC(=O)C2C3CCC(C3)C12", "octahydro-1,4-methanonaphthalene-5,8-dione"),
    ("O=c1ccc2c([nH]1)C1CCC2C1", "5,6,7,8-tetrahydro-5,8-methanoquinolin-2(1H)-one"),
]


@pytest.mark.parametrize("smiles,name", RING_KETONES)
def test_ring_ketone_name(smiles, name):
    res = build(Chem.MolFromSmiles(smiles))
    assert res is not None and res[0] == name, res


def test_added_hydrogen_on_a_ring_nitrogen_is_tautomer_checked(monkeypatch):
    # spec section 8: the standard InChIKey does not tell quinolin-2(1H)-one from its
    # tautomers, so an added hydrogen on a ring N (or next to one) is checked like an
    # indicated hydrogen; the builder declines when the fixed-H check fails
    smiles = "O=c1ccc2c([nH]1)C1CCC2C1"
    monkeypatch.setattr(pkg, "_same_tautomer", lambda mol, name: False)
    assert build(Chem.MolFromSmiles(smiles)) is None
    monkeypatch.setattr(pkg, "_same_tautomer", lambda mol, name: True)
    assert build(Chem.MolFromSmiles(smiles))[0] == "5,6,7,8-tetrahydro-5,8-methanoquinolin-2(1H)-one"
