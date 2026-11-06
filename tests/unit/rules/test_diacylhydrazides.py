"""1,2-Diacylhydrazines: the senior hydrazide substituted by the other acyl group.

 "Substituted hydrazides", (the Blue Book): "Acyl, diacyl, and triacyl derivatives of
hydrazides are named by substituting the senior hydrazide by the appropriate acyl groups (the senior hydrazide is
the one derived from the senior acid)." The senior acid is a ring before a chain (1),:19340) and,
between chains, the longer one. The Blue Book rows (66.3.3.3) are byte-identical; the two dev2000 rows are
d1330 and d1706 (the acetyl prefix of a substituted acetic acid carries no locant,:40694; there is no (2E) on
d1706 because the input has no double-bond stereo).
"""
import pytest

from orthonym.assembly.inner_dispatch import INNER_DISPATCH_TABLE
from tests.support.read_back import same_key

pytestmark = pytest.mark.opsin_gate


ROWS = [
    # Blue Book rows (benchmarks/bb_conformance, def_id 66.3.3.3)
    ("O=C(NNC(=O)c1ccccc1)c1ccccc1", "N'-benzoylbenzohydrazide"),
    ("CCC(=O)N(C)N(CC)C(C)=O", "N'-acetyl-N'-ethyl-N-methylpropanehydrazide"),
    # dev2000 d1330 and d1706 (a parent prefix and the N' prefix in one alphanumerical block)
    ("COc1ccc(C(=O)NNC(=O)CSc2nnc(-c3cccnc3)n2C)cc1",
     "4-methoxy-N'-({[4-methyl-5-(pyridin-3-yl)-4H-1,2,4-triazol-3-yl]sulfanyl}acetyl)benzohydrazide"),
    ("O=C(C=Cc1ccccc1)NNC(=O)c1sccc1-n1cccc1",
     "N'-(3-phenylprop-2-enoyl)-3-(1H-pyrrol-1-yl)thiophene-2-carbohydrazide"),
    # the senior acid decides the parent
    ("CC(=O)NNC(=O)c1ccccc1", "N'-acetylbenzohydrazide"),          # ring before chain
    ("CCC(=O)NNC(C)=O", "N'-acetylpropanehydrazide"),              # the longer chain
    ("CC(=O)NNC(C)=O", "N'-acetylacetohydrazide"),                 #:33940, two identical acyls
]


@pytest.mark.parametrize("smiles,expected", ROWS)
def test_name_tier_and_readback(namer, smiles, expected):
    row = namer.name_tiered(smiles)
    assert row["name"] == expected and row["tier"] == "pin_verified", row
    assert row["producer"] == "inner:diacylhydrazide"
    assert same_key(smiles, expected), expected


@pytest.mark.parametrize("smiles", [
    "CC(=O)N(C(C)=O)NC(C)=O",                  # three acyl groups: not built
    "O=C1NNC(=O)c2ccccc12",                    # the N-N bond is in a ring
    "COC(=O)NNC(C)=O",                         # an alkoxycarbonyl is no acyl: a carbazate
    "O=CNNC(C)=O",                             # formyl: not built
    "C[C@H](Cl)C(=O)NNC(C)=O",                 # a stereo element would be dropped: declined
    "COc1ccc(Cl)cc1C(=O)NNC(C)=O",             # two prefix groups on the parent: the merge is not built
    "CC(=O)NNC(=O)CSC",                        # two chains of one length: no safe seniority, declined
    "CC(=O)NNC(=O)O",                          # an acid is senior
])
def test_not_claimed_outside_the_scope(namer, smiles):
    assert namer.name_tiered(smiles)["producer"] != "inner:diacylhydrazide"


@pytest.mark.parametrize("smiles,expected", [
    ("NNC(=O)C", "acetohydrazide"),
    ("NNC(=O)c1ccccc1", "benzohydrazide"),
    ("NNC(=O)CCC", "butanehydrazide"),
])
def test_monoacyl_hydrazides_keep_their_names(namer, smiles, expected):
    assert namer.name(smiles) == expected


def test_registered_ahead_of_the_claimants():
    entry = INNER_DISPATCH_TABLE["diacylhydrazide"]
    assert entry.priority == 1480
    assert entry.priority < INNER_DISPATCH_TABLE["chain_diamide"].priority < INNER_DISPATCH_TABLE["ester_family"].priority
