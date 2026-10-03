"""Polyazane and polyazene substituent prefixes carry their locants.

 Polyazanes (the Blue Book): 'triazan-1-yl (preselected prefix)' (:39029),
'triaz-2-en-1-yl (preselected prefix) (not triaz-2-eno)' (:39031), '4-(triaz-2-en-1-yl)benzoic
acid (PIN)' (:39035), 'ethyl (tetraazan-1-yl)acetate (PIN)' (:39037); the final 'a' of the
multiplier is not elided before 'azane' (:39017). The default tier shipped '4-triazenylbenzoic
acid' (pin_verified), which OPSIN 2.9.0 reads as the other tautomer (-N=N-NH2: same standard
InChIKey, different fixed-H InChI), and 'triazanyl' / 'tetraazanyl' without the free-valence
locant; best-effort wrote '4-diazenylaminobenzoic acid' and 'aminoaminoamino...'. Every name
below reads back to the input's full InChIKey and fixed-H InChI.
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym.assembly.substituent_naming import parent_to_prefix
from orthonym.assembly.substituent_naming import ATTACH_LOCANT_UNKNOWN
from tests.support.pin_tiers import assert_pin_at_both_tiers
from tests.support.rt_assert import _independent_parse

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,pin", [
    ("OC(=O)c1ccc(NN=N)cc1", "4-(triaz-2-en-1-yl)benzoic acid"),
    ("Oc1ccc(NN=N)cc1", "4-(triaz-2-en-1-yl)phenol"),
    ("OC(=O)c1ccc(NNN)cc1", "4-(triazan-1-yl)benzoic acid"),
    ("CCOC(=O)CNNNN", "ethyl (tetraazan-1-yl)acetate"),
])
def test_polyazane_prefix_at_both_tiers(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)
    fixed = lambda s: inchi.MolToInchi(Chem.MolFromSmiles(s), options="/FixedH")
    assert fixed(_independent_parse(pin)) == fixed(smiles)


@pytest.mark.parametrize("hydride", ["triazene", "triazane", "tetraazane"])
def test_the_string_converter_declines_a_polyazane_without_locants(hydride):
    assert parent_to_prefix(hydride, 0, attach_locant=ATTACH_LOCANT_UNKNOWN) is None
