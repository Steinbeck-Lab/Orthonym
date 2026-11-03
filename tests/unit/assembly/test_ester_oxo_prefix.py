"""An ester beside a principal group senior to it is the pair 'alkoxy... oxo' on the chain (``assembly.ester_oxo_prefix``).

``## **** Esters cited as prefixes`` (the Blue Book): '3-ethoxy-N,N,N-trimethyl-3-oxopropan-1-aminium
bromide (PIN)' (:31708). An anion is senior to an ester, so the olate takes the suffix ('3-methoxy-3-oxopropoxide',
 :28180 for the retained word); a sulfonic acid too. A neutral alcohol or amine beside the ester keeps the ester.
"""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles, name", [
    ("CCOC(=O)CC[N+](C)(C)C.[Br-]", "3-ethoxy-N,N,N-trimethyl-3-oxopropan-1-aminium bromide"),
    ("[O-]CCC(=O)OC", "3-methoxy-3-oxopropan-1-olate"),     # a group senior to OH: not the alcohol's parent,
    ("OS(=O)(=O)CCC(=O)OC", "3-methoxy-3-oxopropane-1-sulfonic acid"),
    ("OC(=O)CCC(=O)OC", "4-methoxy-4-oxobutanoic acid"),
    ("OCCC(=O)OC", "methyl 3-hydroxypropanoate"),
    ("CCOC(=O)CCN(C)C", "ethyl 3-(dimethylamino)propanoate"),
], ids=lambda v: v)
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_ester_beside_a_senior_group(smiles, name, tier):
    row = Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)
    assert row["name"] == name, row
    assert row["tier"] == "pin_verified", row
