import pytest
from rdkit import Chem
from orthonym.rules.acid_ester_anion import name_acid_ester_anion

def _be():
    from orthonym.namer import Orthonym
    return Orthonym(general_fallback=True, general_fallback_unverified=True,
                     allow_aromatic_general=True)

# --- current-correct poly-anion shapes that route_charged/oxoacid own: must stay byte-identical
@pytest.mark.parametrize("smi,expected", [
    ("[O-]C(=O)CC(=O)[O-]", "propanedioate"),          # dicarboxylate dianion (GUARD 2)
    ("CP(=O)([O-])[O-]", "methylphosphonate"),          # phosphonate dianion (CR-02)
])
def test_nonester_dianion_unchanged(smi, expected):
    assert _be().name(smi) == expected

# --- the producer is CORRECT when called directly (proves it is a routing bug, not scope)
@pytest.mark.parametrize("smi,expected", [
    ("O=P([O-])([O-])OC(CO)CO", "1,3-dihydroxypropan-2-yl phosphate"),
    ("O=P([O-])([O-])OCC(O)CO", "2,3-dihydroxypropyl phosphate"),
])
def test_producer_correct_when_called_directly(smi, expected):
    m = Chem.MolFromSmiles(smi)
    assert name_acid_ester_anion(m) == expected

# --- the TARGET: end-to-end these currently abstain; after the fix they must emit the word
@pytest.mark.parametrize("smi,expected", [
    ("O=P([O-])([O-])OC(CO)CO", "1,3-dihydroxypropan-2-yl phosphate"),
    ("O=P([O-])([O-])OCC(O)CO", "2,3-dihydroxypropyl phosphate"),
])
def test_glycerophosphate_class_end_to_end(smi, expected):
    assert _be().name(smi) == expected
