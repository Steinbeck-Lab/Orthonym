"""Alkoxy prefixes on a ring assembly are spelled by the substituent namer.

 "Retained names" (the Blue Book): "Some contracted names are retained for R-O- substituent
groups.... They are: methoxy, ethoxy, propoxy, butoxy, phenoxy, tert-butoxy." A longer chain takes the
systematic '{alkyl}oxy' form ('hexyloxy'); the ring-assembly route used to build 'hexoxy' and 'octoxy' from a
carbon count, which OPSIN reads, so '4,4'-dihexoxy-1,1'-biphenyl' shipped labelled pin_verified. Counting carbons
also missed branches and multiple bonds; the substituent namer reads the chain.

Every expected name below is read back by an independent OPSIN call (``assert_pin_at_both_tiers``).
"""
import pytest

from tests.support.pin_tiers import assert_pin_at_both_tiers

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles, pin", [
    ("CCCCCCOc1ccc(cc1)-c1ccc(OCCCCCC)cc1", "4,4'-bis(hexyloxy)-1,1'-biphenyl"),
    ("CCCCCCCCOc1ccc(cc1)-c1ccc(C#N)cc1", "4'-(octyloxy)[1,1'-biphenyl]-4-carbonitrile"),
    ("CCCCCCOc1ccc(cc1)-c1ccccc1", "4-(hexyloxy)-1,1'-biphenyl"),
    ("CC(C)CCOc1ccc(cc1)-c1ccccc1", "4-(3-methylbutoxy)-1,1'-biphenyl"),
    ("C=CCCCCOc1ccc(cc1)-c1ccccc1", "4-[(hex-5-en-1-yl)oxy]-1,1'-biphenyl"),
    # branched groups used to decline at the default tier; other ring assemblies
    ("CC(C)Oc1ccc(cc1)-c1ccccc1", "4-[(propan-2-yl)oxy]-1,1'-biphenyl"),
    ("CC(C)(C)Oc1ccc(cc1)-c1ccc(OC(C)(C)C)cc1", "4,4'-di-tert-butoxy-1,1'-biphenyl"),
    ("CCCCCCOc1ccc(nc1)-c1ccc(OCCCCCC)cn1", "5,5'-bis(hexyloxy)-2,2'-bipyridine"),
    # the retained contracted prefixes stay
    ("COc1ccc(cc1)-c1ccc(OC)cc1", "4,4'-dimethoxy-1,1'-biphenyl"),
    ("CCCCOc1ccc(cc1)-c1ccccc1", "4-butoxy-1,1'-biphenyl"),
])
def test_alkoxy_prefix_on_a_ring_assembly(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)
