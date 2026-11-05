"""A suffix name of the principal characteristic group is not replaced by a prefix-only name at the wider tiers (lane/l17
item 15, the class drops of the end check).

 "SENIORITY ORDER FOR CLASSES" (the Blue Book), Table 4.1, and "The senior parent structure has the maximum
number of substituents corresponding to the principal characteristic group (suffix)" (:18875): the principal group is the
suffix of the parent, so a name that cites every acid, amide or ring ketone as a prefix ('3,4,5-tricarboxy-...octane') drops
the class of the molecule.

Cause of the zaragozic acid rows: the chain producer of 'a chain prefix with an exo-methylene' (a436fc3dbe, item 3) names the
substituent '4-hydroxy-5-methyl-3-methylidene-6-phenylhexyl' by delegating a chain that no longest chain's C=C reaches to the
saturated branched producer, which did not hand on the numbering of its chain (``pos_out``). The two stereocentres of the
substituent then had no locant to be cited at "NAMING OF STEREOISOMERS": "They are preceded by a numerical or letter
locant... when such locants are present"), the stereo layer of the whole name was stripped, the engine's own name was refused by
the certificate and the suffix-free name of the last-resort rung won. The numbering is handed on.
"""
import pytest

from orthonym.cli import _tiered_namer
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

_ZARAGOZIC = [
    ("C=C(CC[C@]12O[C@H](C(=O)O)[C@@](O)(C(=O)O)[C@](C(=O)O)(O1)[C@H](OC(=O)/C=C/[C@@H](C)C[C@@H](C)CC)[C@H]2O)[C@@H](O)[C@H](C)Cc1ccccc1",
     "(1S,3S,4S,5R,6R,7R)-6-{[(2E,4S,6S)-4,6-dimethyloct-2-enoyl]oxy}-4,7-dihydroxy-1-[(4S,5R)-4-hydroxy-5-methyl-3-methylidene-6-phenylhexyl]-2,8-dioxabicyclo[3.2.1]octane-3,4,5-tricarboxylic acid"),
    ("C=C(CC[C@]12O[C@H](C(=O)O)[C@@](O)(C(=O)O)[C@](C(=O)O)(O1)[C@H](OC(=O)/C=C/C(C)=C/C(C)CC)[C@H]2O)[C@@H](OC(C)=O)[C@H](C)Cc1ccccc1",
     "(1S,3S,4S,5R,6R,7R)-1-[(4S,5R)-4-(acetyloxy)-5-methyl-3-methylidene-6-phenylhexyl]-6-{[(2E,4E)-4,6-dimethylocta-2,4-dienoyl]oxy}-4,7-dihydroxy-2,8-dioxabicyclo[3.2.1]octane-3,4,5-tricarboxylic acid"),
]

# The rows below are named by their prefixes at best-effort (the suffix of the amide, the lactone or the lactam is cited as
# 'carbamoyl', '(oxo)methyl}amino' or 'oxo'), where the L16 name has the suffix. They are open: the principal group that the
# seniority of lane/l17 gives (the amide parent of f60ed2e5d9, the lactone as a heterocycle of 054d0fac9d) is declined by the
# engine rung and the ring cascade, and the suffix-free producer wins.
_OPEN = [
    ("CC(C)CC(NC(=O)C(O)C(O)C1CC(=O)NC(C)(C)N1)C1Cc2cccc(O)c2C(=O)O1",
     "6-(1,2-dihydroxy-3-{[1-(8-hydroxy-1-oxo-3,4-dihydro-1H-2-benzopyran-3-yl)-3-methylbutyl]amino}-3-oxopropyl)-2,2-dimethyl-1,3-diazinan-4-one"),
    ("C/C=C1\\NC(=O)[C@H](C)NC(=O)[C@H](C(C)C)NC(=O)[C@H](NC(=O)[C@H](Cc2ccccc2)NC(=O)[C@@H](Cc2ccccc2)NC(C)=O)[C@@H](C)OC1=O",
     "(3Z,6S,9S,12R,13R)-12-{(2S)-2-[(2R)-2-acetamido-3-phenylpropanamido]-3-phenylpropanamido}-3-ethylidene-6,13-dimethyl-9-(propan-2-yl)-1-oxa-4,7,10-triazacyclotridecane-2,5,8,11-tetrone"),
    ("CCC1NC(=O)CC(C(O)C(O)C(=O)NC(CC(C)C)C2Cc3cccc(O)c3C(=O)O2)N1",
     "6-(1,2-dihydroxy-3-{[1-(8-hydroxy-1-oxo-3,4-dihydro-1H-2-benzopyran-3-yl)-3-methylbutyl]amino}-3-oxopropyl)-2-ethyl-1,3-diazinan-4-one"),
    ("CO[C@H]1[C@@H](O)[C@H](n2ccc(=O)[nH]c2=O)O[C@@H]1[C@@H](O[C@H]1OC(C(=O)N[C@H]2CCC[C@@H](C)NC2=O)=C[C@H](O)[C@@H]1O)C(N)=O",
     "(2R)-2-{[(2S,3S,4S)-3,4-dihydroxy-6-{[(3S,7R)-7-methyl-2-oxoazepan-3-yl]carbamoyl}-3,4-dihydro-2H-pyran-2-yl]oxy}-2-[(2S,3S,4R,5R)-5-(2,4-dioxo-3,4-dihydropyrimidin-1(2H)-yl)-4-hydroxy-3-methoxyoxolan-2-yl]acetamide"),
    ("C[C@@H]1CCC[C@H](NC(=O)C2=C[C@H](O)[C@H](O)[C@@H](O[C@@H](C(N)=O)[C@H]3O[C@@H](n4ccc(=O)[nH]c4=O)[C@H](O)[C@@H]3O)O2)C(=O)N1",
     "(2R)-2-{[(2S,3S,4S)-3,4-dihydroxy-6-{[(3S,7R)-7-methyl-2-oxoazepan-3-yl]carbamoyl}-3,4-dihydro-2H-pyran-2-yl]oxy}-2-[(2S,3S,4R,5R)-5-(2,4-dioxo-3,4-dihydropyrimidin-1(2H)-yl)-3,4-dihydroxyoxolan-2-yl]acetamide"),
    # Found by the class-drop scan of end check 3 (present since end check 1): the lactone of a steroid side chain becomes
    # '2-oxooxane', and the phosphocholine ester of a ceramide becomes a suffix-free 'octadec-7-ene'.
    ("CC(=O)OC[C@]12CC[C@H]3[C@@H](CCC4=CC(=O)C=C[C@@]43C)[C@@H]1CC[C@@H]2[C@H](C)[C@H]1C[C@H](C)[C@H](C)C(=O)O1",
     "(3S,4S,6R)-6-[(1S)-1-{(8R,9S,10R,13R,14S,17R)-13-[(acetyloxy)methyl]-10-methyl-3-oxo-6,7,8,9,10,11,12,13,14,15,16,17-dodecahydro-3H-cyclopenta[a]phenanthren-17-yl}ethyl]-3,4-dimethyloxan-2-one"),
    ("CCCCCC/C=C\\CCCCCCCCCC(=O)N[C@@H](COP(=O)([O-])OCC[N+](C)(C)C)[C@H](O)CCCCCCCCCCCCCCC",
     "(2S,3R)-3-hydroxy-2-[(11Z)-octadec-11-enamido]octadecyl 2-(trimethylazaniumyl)ethyl phosphate"),
]


@pytest.mark.parametrize("smiles, expected", _ZARAGOZIC)
def test_chain_prefix_with_exo_methylene_keeps_the_stereo_and_the_suffix(smiles, expected):
    row = _tiered_namer("pin", "best-effort").name_tiered(smiles)
    assert row["name"] == expected
    assert_full_rt(row["name"], smiles, "zaragozic acid chain, tricarboxylic acid suffix")


@pytest.mark.xfail(strict=True, reason="class drop open: suffix-free producer wins over the principal group of lane/l17")
@pytest.mark.parametrize("smiles, expected", _OPEN)
def test_open_class_drop_rows_keep_the_suffix(smiles, expected):
    row = _tiered_namer("pin", "best-effort").name_tiered(smiles)
    assert row["name"] == expected
