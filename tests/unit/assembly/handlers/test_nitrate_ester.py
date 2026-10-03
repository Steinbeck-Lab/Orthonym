"""Esters of nitric acid, functional-class names ('<organyl> nitrate').

 (heading 'Esters of mononuclear noncarbon oxoacids',
the Blue Book),:35918: "Esters of mononuclear noncarbon acids are named in
the same way as esters of organic acids (see. Alkyl groups, aryl
groups, etc. are cited as separate words, in alphanumerical order when more than
one, and followed by the name of the appropriate anion." Sibling PIN: 'pentyl
nitrite (PIN)' (:35922). The group's prefix 'nitrooxy',:36405) is
kept where a senior class names the parent.

The PIN tier used to decline these (the only name built was the substitutive
'(nitrooxy)ethane', labelled below the PIN); each name below is read back by a
fresh OPSIN call to the input's full InChIKey.
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.assembly.handlers.nitrate_ester import _is_nitrate_ester
from tests.support.rt_assert import assert_full_rt, name_best_effort

pytestmark = pytest.mark.opsin_gate

NITRATE_ESTERS = {
    "methyl": ("CO[N+](=O)[O-]", "methyl nitrate"),
    "ethyl": ("CCO[N+](=O)[O-]", "ethyl nitrate"),
    "pentyl": ("CCCCCO[N+](=O)[O-]", "pentyl nitrate"),
    "phenyl": ("c1ccccc1O[N+](=O)[O-]", "phenyl nitrate"),
    "benzyl": ("c1ccccc1CO[N+](=O)[O-]", "benzyl nitrate"),
    "cyclohexyl": ("C1CCCCC1O[N+](=O)[O-]", "cyclohexyl nitrate"),
    "chloroethyl": ("ClCCO[N+](=O)[O-]", "2-chloroethyl nitrate"),
    "butan_2_yl": ("C[C@H](CC)O[N+](=O)[O-]", "(2R)-butan-2-yl nitrate"),
}


@pytest.mark.parametrize("smiles,expected", list(NITRATE_ESTERS.values()),
                         ids=list(NITRATE_ESTERS))
def test_nitrate_ester_pin(smiles, expected):
    row = Orthonym(style="pin").name_tiered(smiles)
    assert row["name"] == expected, row
    assert row["tier"] == "pin_verified", row
    assert_full_rt(row["name"], smiles)
    assert name_best_effort(smiles)["name"] == expected


NITRITE_ESTERS = {
    "pentyl": ("CCCCCON=O", "pentyl nitrite"),
    "ethyl": ("CCON=O", "ethyl nitrite"),
}


@pytest.mark.parametrize("smiles,expected", list(NITRITE_ESTERS.values()),
                         ids=list(NITRITE_ESTERS))
def test_nitrite_esters_unchanged(smiles, expected):
    row = Orthonym(style="pin").name_tiered(smiles)
    assert row["name"] == expected and row["tier"] == "pin_verified", row


# Outside the producer: a senior class names the parent and keeps the prefix
# ('3-(nitrooxy)propanoic acid', 'ethyl 3-(nitrooxy)propanoate'), and a polyol
# ester, whose PIN names the alcohol component by a multivalent group
#,:31815), keeps its best-effort name.
SENIOR_PARENT = {
    "acid": ("OC(=O)CCO[N+](=O)[O-]", "3-(nitrooxy)propanoic acid"),
    "ester": ("CCOC(=O)CCO[N+](=O)[O-]", "ethyl 3-(nitrooxy)propanoate"),
}


@pytest.mark.parametrize("smiles,expected", list(SENIOR_PARENT.values()),
                         ids=list(SENIOR_PARENT))
def test_senior_parent_keeps_the_nitrooxy_prefix(smiles, expected):
    row = Orthonym(style="pin").name_tiered(smiles)
    assert row["name"] == expected and row["tier"] == "pin_verified", row


# A polyol ester with another group (a free OH) is not claimed. The clean polyol
# ester '[O-][N+](=O)OCCO[N+](=O)[O-]' is: 'ethane-1,2-diyl dinitrate'
#, the Blue Book; tests/unit/rules/test_polyol_identical_anions.py).
@pytest.mark.parametrize("smiles", [
    "OCC(O[N+](=O)[O-])CO[N+](=O)[O-]",
])
def test_polyol_ester_is_not_claimed(smiles):
    from orthonym.errors import is_failure_name
    be = name_best_effort(smiles)["name"]
    assert_full_rt(be, smiles)
    assert not be.endswith("nitrate"), be
    pin = Orthonym(style="pin").name(smiles)
    assert is_failure_name(pin) or not pin.endswith("nitrate"), pin


# The alcohol part of a carboxylic ester is named as a fragment and turned into
# a substituent prefix; it keeps the substitutive 'nitrooxy' form there, so the
# carboxylic ester (the senior ester, class 9 in the order of the acids)
# still names the parent. Two different anions on one alcohol component make it
# method (2) of (the Blue Book: "Method (1) generates
# preferred IUPAC names"), so the name ships at best-effort, below the PIN, and
# the prefix carries its locant ('2-(nitrooxy)ethyl',.
@pytest.mark.parametrize("smiles,acid_word", [
    ("CC(=O)OCCO[N+](=O)[O-]", "acetate"),
    ("C1=CC=C(C(=C1)CC(=O)OCCO[N+](=O)[O-])NC2=C(C=CC=C2Cl)Cl", "acetate"),
])
def test_alcohol_part_of_a_carboxylic_ester_keeps_the_prefix(smiles, acid_word):
    row = name_best_effort(smiles)
    name = assert_full_rt(row["name"], smiles)
    assert name.startswith("2-(nitrooxy)ethyl ") and name.endswith(acid_word), name
    assert row["tier"] == "systematic_verified", row
    assert Orthonym(style="pin").name_tiered(smiles)["tier"] == "abstain"


# An acyl group on the nitrate oxygen makes a mixed anhydride of nitric acid, not
# an ester: (the Blue Book, 'Mixed anhydrides'),:32272 "Mixed
# anhydrides with carbonic acid, cyanic acid, and inorganic acids are named as
# anhydrides."; 'CH3-CO-O-CN acetic cyanic anhydride (PIN)' (:32276); the acids
# are cited in alphabetical order (:32255). names an alcohol
# component ("Alkyl groups, aryl groups, etc.",:35918), not an acyl group.
ACYL_NITRATES = {
    "acetyl": ("CC(=O)O[N+](=O)[O-]", "acetic nitric anhydride"),
    "propanoyl": ("CCC(=O)O[N+](=O)[O-]", "nitric propanoic anhydride"),
    "benzoyl": ("O=C(O[N+](=O)[O-])c1ccccc1", "benzoic nitric anhydride"),
}


@pytest.mark.parametrize("smiles,expected", list(ACYL_NITRATES.values()),
                         ids=list(ACYL_NITRATES))
def test_acyl_nitrate_is_not_named_as_an_ester(smiles, expected):
    pin = Orthonym(style="pin").name_tiered(smiles)["name"]
    be = name_best_effort(smiles)["name"]
    for name in (pin, be):
        assert not name.endswith(" nitrate"), name
    assert_full_rt(be, smiles)


@pytest.mark.parametrize("smiles,expected", list(ACYL_NITRATES.values()),
                         ids=list(ACYL_NITRATES))
def test_acyl_nitrate_mixed_anhydride_pin(smiles, expected):
    row = Orthonym(style="pin").name_tiered(smiles)
    assert row["name"] == expected and row["tier"] == "pin_verified", row
    assert_full_rt(row["name"], smiles)


# A senior acid keeps the substitutive read of the same linkage (best-effort,
# read back exact), as before.
@pytest.mark.parametrize("smiles", [
    "OC(=O)CC(=O)O[N+](=O)[O-]",
])
def test_acyl_nitrate_under_a_senior_acid_keeps_its_name(smiles):
    be = name_best_effort(smiles)["name"]
    assert_full_rt(be, smiles)
    assert "nitrooxy" in be, be


class _F:
    def __init__(self, fg, pg=None):
        self.functional_groups = fg
        self.principal_group = pg


def test_predicate():
    one = [(0, 1, 2, 3, 4)]
    assert _is_nitrate_ester(_F({"nitrooxy": one}))
    assert not _is_nitrate_ester(_F({"nitrooxy": one}, pg="ester"))
    assert not _is_nitrate_ester(_F({"nitrooxy": one, "nitrite": [(5, 6, 7, 8)]}))
    # several nitrate groups reach the producer, which names only a clean polyol
    # ester ('ethane-1,2-diyl dinitrate',, the Blue Book)
    assert _is_nitrate_ester(_F({"nitrooxy": one + [(5, 6, 7, 8, 9)]}))
    assert not _is_nitrate_ester(_F({}))


def test_registered_beside_the_nitrite_producer():
    from orthonym.assembly.inner_dispatch import INNER_DISPATCH_TABLE
    entry = INNER_DISPATCH_TABLE["nitrate_ester"]
    assert entry.priority == 2962
    assert entry.side_effect_inventory == ()
    assert Chem.MolFromSmiles("CCO[N+](=O)[O-]") is not None
