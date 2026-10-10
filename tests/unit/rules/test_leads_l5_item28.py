"""Leads program item 28 (lane L5): a ring that carries ring C=O groups is the parent over a chain
that carries a ketone.

 (the Blue Book): "There is no seniority order difference between ketones and
pseudoketones. When necessary, the maximum number of carbonyl groups or doubly bonded oxygen atoms,
the seniority order between chains and rings, and between rings and ring systems, are considered".
 (:24092, 'Selection between a ring and a chain as parent hydride'; sentence:24096): "Within
the same heteroatom class and for the same number of characteristic groups cited as the principal
characteristic group, a ring is always selected as the parent hydride to construct a preferred IUPAC
name." (:19336) (1): "Within the same class, a ring or ring system has seniority over a
chain."

``perception.functional_groups`` registered a mancude-ring heterone (ring C=O beside a ring
heteroatom, or on an aromatic ring: invisible to the ketone SMARTS) only when no SMARTS ketone existed,
so beside a chain ketone the ring C=O was dropped and the chain won. The class is now the union of the
SMARTS ketones and the heterones, each carbonyl carbon once.

Two hazards that union unmasked are pinned here as well:

* H1: the fusion producer ``fused_rings._apply_suffix_to_core`` writes a ring '-one' without its
  added indicated hydrogen ('phthalazin-1-one'); (:24687, 'Added indicated hydrogen'),
  (:24695) and 'quinolin-2(1H)-one (PIN)' (:28334) require 'phthalazin-1(2H)-one'. The name was labelled a
  PIN; ``tier_a_ring`` now audits it against the shared hydrogen rule and the label is withdrawn.
* H2: the fragment namer left the E/Z descriptor out of an oxo-bearing alkenyl substituent of a heterocycle
  and the stereo reclaim put it back at the FRONT of the name with the substituent's own locant
  ('(1E)-...-6-(3-oxobut-1-en-1-yl)-2H-pyran-2-one'); 'NAMING OF STEREOISOMERS' (:44639; sentence:44643: "When they relate to substituent groups, they are
  cited at the front of the corresponding prefix") cites it at the front of the prefix.
"""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from tests.support.rt_assert import assert_full_rt

pytestmark = pytest.mark.opsin_gate

PIN = Orthonym(style="pin")
BEST = Orthonym(style="pin", **_emit_tier_flags("best-effort"))
VALID = Orthonym(style="pin", **_emit_tier_flags("valid"))

# (SMILES, PIN) -- each name is read back by OPSIN 2.9.0 to the input's full InChIKey (assert_full_rt)
RING_PARENT_ROWS = [
    ("CC(=O)CCCc1cccc(=O)o1", "6-(4-oxopentyl)-2H-pyran-2-one"),
    ("CC(=O)CCc1cc2ccc(=O)oc2cc1O", "7-hydroxy-6-(3-oxobutyl)-2H-1-benzopyran-2-one"),
    ("CC(=O)c1c(O)cc(C)oc1=O", "3-acetyl-4-hydroxy-6-methyl-2H-pyran-2-one"),
    ("CC(=O)CCn1ccc(=O)[nH]c1=O", "1-(3-oxobutyl)pyrimidine-2,4(1H,3H)-dione"),
]


@pytest.mark.parametrize("smiles,expected", RING_PARENT_ROWS)
def test_the_ring_that_carries_ring_carbonyls_is_the_parent(smiles, expected):
    row = PIN.name_tiered(smiles)
    assert row["name"] == expected and row["tier"] == "pin_verified", row
    assert_full_rt(expected, smiles)
    assert BEST.name_tiered(smiles)["name"] == expected


@pytest.mark.parametrize("smiles,expected", [
    # a saturated ring ketone, lactone and lactam were always read by the ketone SMARTS or the
    # lactone handler: the ring was the parent before the union and stays it
    ("CC(=O)CCC1CCCCC1=O", "2-(3-oxobutyl)cyclohexan-1-one"),
    ("CC(=O)CCC1CCOC1=O", "3-(3-oxobutyl)oxolan-2-one"),
    ("CC(=O)CCC1CCCNC1=O", "3-(3-oxobutyl)piperidin-2-one"),
    # no chain ketone: the heterone was always registered
    ("O=c1ccc2ccccc2o1", "2H-1-benzopyran-2-one"),
])
def test_controls_keep_their_names(smiles, expected):
    row = PIN.name_tiered(smiles)
    assert row["name"] == expected and row["tier"] == "pin_verified", row
    assert_full_rt(expected, smiles)


def test_the_union_counts_each_carbonyl_carbon_once():
    # SMARTS ketones and heterones are one class: a ring that the SMARTS already reads (a plain
    # cyclohexanone) is not registered twice by the heterone loop
    from rdkit import Chem

    from orthonym.perception.functional_groups import detect_functional_groups
    mol = Chem.MolFromSmiles("CC(=O)CCc1cc(=O)oc2ccccc12")
    ketones = detect_functional_groups(mol)["ketone"]
    carbons = [m[1] for m in ketones]
    assert len(carbons) == len(set(carbons)) == 2, ketones
    mol = Chem.MolFromSmiles("O=C1CCCC=C1CC(C)=O")
    carbons = [m[1] for m in detect_functional_groups(mol)["ketone"]]
    assert len(carbons) == len(set(carbons)) == 2, carbons


@pytest.mark.parametrize("smiles", [
    "CC(=O)c1cccc(CC2=NNC(=O)c3ccccc23)c1",     # '4-[(3-acetylphenyl)methyl]phthalazin-1-one'
    "CC(=O)CCc1cc(=O)[nH]c2ccccc12",            # '4-(3-oxobutyl)quinolin-2-one'
])
def test_h1_a_ring_one_without_its_added_hydrogen_is_not_labelled_a_pin(smiles):
    # the fusion producer still writes the hydrogen-less spelling (a separate item: the producer is
    # fused_rings._apply_suffix_to_core); the PIN tier now declines it and the wider tiers ship it
    # below the PIN, never as pin_verified
    assert PIN.name_tiered(smiles)["tier"] == "abstain"
    row = VALID.name_tiered(smiles)
    assert row["tier"] != "pin_verified", row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles", [
    # a substituent that spells ITS OWN indicated hydrogen ('1H-pyrrol-2-yl', '1H-indol-6-yl') says
    # nothing about the parent: the parent '-one' still lacks 'quinolin-2(1H)-one' / 'phthalazin-1(2H)-one'
    "CC(=O)c1ccc(CC2=NNC(=O)c3ccccc23)[nH]1",                  # '4-[(5-acetyl-1H-pyrrol-2-yl)methyl]phthalazin-1-one'
    "CC(=O)CCc1cc(=O)[nH]c2cc(-c3cc[nH]c3)ccc12",              # '4-(3-oxobutyl)-7-(1H-pyrrol-3-yl)quinolin-2-one'
    "CC(=O)c1cccc(CC2=NNC(=O)c3ccccc23)c1-c1ccc[nH]1",         # '4-{[3-acetyl-2-(1H-pyrrol-2-yl)phenyl]methyl}phthalazin-1-one'
    "CC(=O)CCc1cc(=O)[nH]c2ccc(Cc3ccc4cc[nH]c4c3)cc12",        # '6-[(1H-indol-6-yl)methyl]-4-(3-oxobutyl)quinolin-2-one'
    "CC(=O)c1ccc(Cc2cc(=O)[nH]c3ccccc23)n1C",                  # '4-[(5-acetyl-1-methyl-1H-pyrrol-2-yl)methyl]quinolin-2-one'
])
def test_h1_a_substituent_hydrogen_does_not_stand_for_the_parents(smiles):
    # 'Added indicated hydrogen' (the Blue Book): the PARENT cites it; the audit reads the
    # parent's own part of the name, so the PIN tier declines these rows and no tier labels them pin_verified
    assert PIN.name_tiered(smiles)["tier"] == "abstain"
    for namer in (VALID, BEST):
        row = namer.name_tiered(smiles)
        assert row["tier"] != "pin_verified", row
        assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("name,spelled", [
    # the parent's own spelling: an added-hydrogen group or an indicated hydrogen outside every substituent
    ("quinolin-2(1H)-one", True),
    ("4-benzylphthalazin-1(2H)-one", True),
    ("pyrimidine-2,4(1H,3H)-dione", True),
    ("2H-1-benzopyran-2-one", True),
    ("3-[(1H-indol-6-yl)methyl]-2H-1-benzopyran-2-one", True),
    ("6-[(1H-indol-6-yl)methyl]-4-(3-oxobutyl)quinolin-2(1H)-one", True),
    # a hydrogen spelled only inside a substituent is not the parent's
    ("phthalazin-1-one", False),
    ("4-[(5-acetyl-1H-pyrrol-2-yl)methyl]phthalazin-1-one", False),
    ("4-(3-oxobutyl)-7-(1H-pyrrol-3-yl)quinolin-2-one", False),
    ("4-{[3-acetyl-2-(1H-pyrrol-2-yl)phenyl]methyl}phthalazin-1-one", False),
    ("6-[(1H-indol-6-yl)methyl]-4-(3-oxobutyl)quinolin-2-one", False),
])
def test_h1_the_audit_reads_the_parents_own_hydrogen(name, spelled):
    from orthonym.assembly.handlers.tier_a_ring import _parent_part_spells_ring_hydrogen
    assert _parent_part_spells_ring_hydrogen(name) is spelled


@pytest.mark.parametrize("smiles,expected", [
    # the controls the audit must not touch: the ring hydrogen is spelled
    ("O=C1NN=C(Cc2ccccc2)c2ccccc12", "4-benzylphthalazin-1(2H)-one"),
    ("O=c1ccc2ccccc2[nH]1", "quinolin-2(1H)-one"),
    ("O=C1CCc2ccccc2C1", "3,4-dihydronaphthalen-2(1H)-one"),
])
def test_h1_controls_keep_their_pin_labels(smiles, expected):
    row = PIN.name_tiered(smiles)
    assert row["name"] == expected and row["tier"] == "pin_verified", row
    assert_full_rt(expected, smiles)


@pytest.mark.parametrize("smiles,expected", [
    ("CC(=O)/C=C/c1cc(O)cc(=O)o1", "4-hydroxy-6-[(1E)-3-oxobut-1-en-1-yl]-2H-pyran-2-one"),
    ("CC(=O)/C=C/c1c(C)[nH]c(=O)[nH]c1=O", "6-methyl-5-[(1E)-3-oxobut-1-en-1-yl]pyrimidine-2,4(1H,3H)-dione"),
    ("COc1cc(=O)oc(/C=C/C=C/C(C)=O)c1C", "4-methoxy-5-methyl-6-[(1E,3E)-5-oxohexa-1,3-dien-1-yl]-2H-pyran-2-one"),
])
def test_h2_the_descriptor_of_an_oxo_alkenyl_substituent_is_cited_in_its_prefix(smiles, expected):
    for namer in (PIN, BEST):
        row = namer.name_tiered(smiles)
        assert row["name"] == expected and row["tier"] == "pin_verified", row
    assert_full_rt(expected, smiles)


@pytest.mark.parametrize("smiles,expected", [
    # the substituent keeps its R/S descriptor and a chain ketone without a ring C=O stays the parent
    ("CC(=O)C[C@H](O)Cc1cc(O)c(C)c(=O)o1", "4-hydroxy-6-[(2R)-2-hydroxy-4-oxopentyl]-3-methyl-2H-pyran-2-one"),
    ("CC(=O)/C=C/c1ccccn1", "(3E)-4-(pyridin-2-yl)but-3-en-2-one"),
])
def test_h2_controls(smiles, expected):
    row = PIN.name_tiered(smiles)
    assert row["name"] == expected and row["tier"] == "pin_verified", row
    assert_full_rt(expected, smiles)


# --- separate items the lane records as strict xfail rows (each flips when its producer lands) ---

@pytest.mark.xfail(strict=True, reason=(
    "leads item 28 hazard H1, producer half (separate item): fused_rings._apply_suffix_to_core writes the "
    "ring '-one' of a fusion parent without added indicated hydrogen when the certified fusion name "
    "declines; tier_a_ring audits the finished name and the PIN tier declines it. The PIN is "
    "'phthalazin-1(2H)-one' / 'quinolin-2(1H)-one' (P-58.2.2, BlueBookV2.md:24687; 'quinolin-2(1H)-one "
    "(PIN)' :28334). When the producer spells the hydrogen, delete this marker and the "
    "test_h1_a_ring_one_without_its_added_hydrogen row it replaces."))
@pytest.mark.parametrize("smiles,expected", [
    ("CC(=O)c1cccc(CC2=NNC(=O)c3ccccc23)c1", "4-[(3-acetylphenyl)methyl]phthalazin-1(2H)-one"),
    ("CC(=O)CCc1cc(=O)[nH]c2ccccc12", "4-(3-oxobutyl)quinolin-2(1H)-one"),
])
def test_h1_producer_spells_the_added_hydrogen(smiles, expected):
    row = PIN.name_tiered(smiles)
    assert row["name"] == expected and row["tier"] == "pin_verified", row
    assert_full_rt(expected, smiles)


@pytest.mark.xfail(strict=True, reason=(
    "leads item 28 hazard H2, carbocycle half: assembly/substituent_naming._name_unsaturated_oxo_substituent "
    "(lane L3's file) returns '3-oxobut-1-en-1-yl' with no E/Z descriptor, the stereo reclaim puts it "
    "at the FRONT of the name with the substituent's own locant and the PIN tier declines. Patch "
    "L5-oxo-alkenyl-stereo.patch (p3/patches) cites the descriptor in the prefix as the sibling writers do "
    "(P-91.3 'NAMING OF STEREOISOMERS', BlueBookV2.md:44639, sentence :44643). Delete this marker "
    "when the patch is applied."))
def test_h2_carbocycle_the_descriptor_is_cited_in_its_prefix():
    smiles, expected = "CC(=O)/C=C/C1CCCCC1=O", "2-[(1E)-3-oxobut-1-en-1-yl]cyclohexan-1-one"
    row = PIN.name_tiered(smiles)
    assert row["name"] == expected and row["tier"] == "pin_verified", row
    assert_full_rt(expected, smiles)
