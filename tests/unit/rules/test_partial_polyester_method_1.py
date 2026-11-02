"""A partial polyester of a polyol with different acids is named by method (1), its free hydroxy a prefix.

 (the Blue Book): "When anions are different [...] (1) names of anions are cited in
alphanumerical order preceded by a locant [...] Method (1) generates preferred IUPAC names", 'propane-1,2,3-triyl
1,2-diacetate 3-propanoate (PIN)'; a free hydroxy is a prefix of the organyl group as in '2-hydroxypropane-1,3-diyl
diacetate'; a stereodescriptor of an anion is enclosed with it, '... 3-[(9Z)-octadec-9-enoate] (PIN)' (:31846).
OPSIN reads the form through its acyloxy twin (``validation.polyester_twin``); a general tier ships the
ester-family name OPSIN reads directly.
"""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.validation.polyester_twin import polyester_twin

pytestmark = pytest.mark.unit


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", [
    ("CCCCCCCC/C=C\\CCCCCCCC(=O)OC[C@H](CO)OC(=O)CCCCCCCCC",
     "(2S)-3-hydroxypropane-1,2-diyl 2-decanoate 1-[(9Z)-octadec-9-enoate]"),
    ("CCC(=O)OCC(O)COC(C)=O", "2-hydroxypropane-1,3-diyl 1-acetate 3-propanoate"),
    # canary DK-GLYSUB (was the acyloxy-on-an-alcohol name '(2R)-1-(octadecanoyloxy)-2-[...]propan-3-ol')
    ("CCCCCCCC/C=C\\CCCCCCCC(=O)O[C@H](CO)COC(=O)CCCCCCCCCCCCCCCCC",
     "(2R)-3-hydroxypropane-1,2-diyl 1-octadecanoate 2-[(9Z)-octadec-9-enoate]"),
    ("CCC(=O)OCC(OC(C)=O)COC(C)=O", "propane-1,2,3-triyl 1,2-diacetate 3-propanoate"),
])
def test_pin_tier(smiles, name):
    row = Orthonym(style="pin", **_emit_tier_flags("pin")).name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row


def test_the_twin_keeps_the_hydroxy_prefix():
    assert polyester_twin("2-hydroxypropane-1,3-diyl 1-acetate 3-propanoate") == \
        "2-hydroxy-1-[acetyloxy]-3-[propanoyloxy]propane"
