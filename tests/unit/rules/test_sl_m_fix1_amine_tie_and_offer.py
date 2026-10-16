"""SL lane M, fix a performance pass: (1) a tie between parents of an amine nitrogen is no PIN; (2) the best-effort
tier prefers the clean name of a junior parent to a chain-parent name that breaks a spelling rule.

(1) "CRITERIA RELATED TO NUMBER AND LOCATION OF SUBSTITUENT GROUPS" (the Blue Book,:21604) and
 (:21698) choose the senior parent by the number of prefixes and their locants. When the candidate
parents through the nitrogen -- the benzylic carbons of N-benzyl-N-methyl-(4-chlorobenzyl)amine, the CH2Cl and
the CH2Ph of ClCN(C)Cc1ccccc1 -- tie on both, (:21791) "the lower locant or set of locants for
substituents cited as prefixes... in their order of citation in the name" decides, and the N-locants of the
nitrogen's own prefixes are part of that order ('N' is lower than a numeral,,:3195). The selector's
last key (the alphanumerical key of the prefixes,,:22234) does not read them. That choice is not the
rule's, so the name is not certified as the PIN: the default tier declines it, the best-effort tier keeps it
below pin_verified. Neither reading of is built here. The tests read the structure
(``chains.amine_parent_tie``), not a name string.

(2) A name already below the PIN is not read by the spelling checks (they run on a would-be pin_verified
name), so a chain-parent name that broke (:7232), (:7272) or (:44643) shipped.
``assembly.spelling_offer`` names the junior parents of the pool in turn.

Changed asserted values: none, every row is new.
"""
import random

import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.perception.chains import amine_parent_tie, chain_parent_prefix_seniority
from orthonym.validation.pin_spelling import check_pin_spelling

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


def _features(smiles):
    namer = Orthonym(_disable_opsin_validity_gate=True)
    mol = Chem.MolFromSmiles(smiles)
    canonical = Chem.MolToSmiles(mol)
    features = namer._perceive(mol, canonical, canonical)
    namer._classify(features)
    return features


def _spellings(smiles, n=6, seed=5):
    mol = Chem.MolFromSmiles(smiles)
    rng = random.Random(seed)
    out = [smiles]
    for _ in range(n):
        order = list(range(mol.GetNumAtoms()))
        rng.shuffle(order)
        out.append(Chem.MolToSmiles(Chem.RenumberAtoms(mol, order), canonical=False))
    return list(dict.fromkeys(out))


TIED = [
    "ClCN(C)Cc1ccccc1",                                    # CH2Cl against CH2Ph: three prefixes each
    "CN(Cc1ccccc1)Cc1ccc(Cl)cc1",                          # two benzylic carbons, different rings
    "CN(Cc1ccccc1)CC1CC1",                                 # benzyl against cyclopropylmethyl
    "CN(CC1=CC(=C(C=C1)Br)[N+](=O)[O-])CC2=CC=CC=C2F",
    "ClCNCc1ccccc1",                                       # a secondary amine
    "ClCN(C)CBr",                                          # no ring: the CH2Cl and the CH2Br, the selector's last key
    "FCN(C)CCl",
]
NOT_TIED = [
    "CN(C)Cc1ccccc1",                                      # one benzylic carbon, two methyls: decides
    "CNCc1ccccc1",
    "CN(Cc1ccccc1)Cc1ccccc1",                              # the two benzylic carbons are one chain by symmetry
    "CCN(C)Cc1ccccc1",                                     # (b): the longer chain
    "CCN(CC)Cc1ccccc1",
    "CN(C)C(c1ccccc1)c1ccccc1",
    "CN(C)CCc1ccccc1",
    "CN(C)C", "CCN(C)C", "ClCN(C)C", "CCN(CC)CCl",          # no ring: one candidate, a longer chain, or a symmetry
]


@pytest.mark.parametrize("smiles", TIED)
def test_a_tie_between_parents_of_the_nitrogen_is_undecided(smiles):
    features = _features(smiles)
    assert features.chain_is_parent or not features.is_cyclic
    assert amine_parent_tie(features) is True
    assert chain_parent_prefix_seniority(features) == "undecided"


@pytest.mark.parametrize("smiles", NOT_TIED)
def test_a_choice_the_prefix_count_decides_is_no_tie(smiles):
    features = _features(smiles)
    assert amine_parent_tie(features) is False
    assert chain_parent_prefix_seniority(features) is None


@pytest.mark.parametrize("smiles", TIED[:3] + ["ClCN(C)CBr"] + NOT_TIED[:3])
def test_the_tie_does_not_depend_on_the_order_of_the_atoms(smiles):
    expected = smiles in TIED
    assert {amine_parent_tie(_features(s)) for s in _spellings(smiles)} == {expected}


def test_the_group_must_be_an_amine_with_a_chain_for_parent():
    assert amine_parent_tie(_features("OCc1ccccc1")) is False          # an alcohol
    assert amine_parent_tie(_features("CC(C)(C)Nc1ccccc1")) is False   # a ring is the parent
    assert amine_parent_tie(_features("CC(C)NC1CCCCC1")) is False


@pytest.mark.parametrize("smiles", TIED[:4] + ["ClCN(C)CBr", "FCN(C)CCl", "ClC[N+](C)(C)CBr"])
def test_the_tied_name_loses_pin_verified_at_both_tiers(smiles):
    default = Orthonym().name_tiered(smiles)
    assert default["tier"] == "abstain" and default["limit_code"] == "NO_VERIFIED_PIN"
    best_effort = Orthonym(general_fallback=True, general_fallback_unverified=True,
                           allow_aromatic_general=True).name_tiered(smiles)
    assert best_effort["tier"] == "pin_unverified" and not best_effort["is_pin"]
    assert best_effort["name"] and best_effort["verified"] == "opsin"


@pytest.mark.parametrize("smiles", [
    "ClC[N+](C)(C)Cc1ccccc1", "CN(Cc1ccccc1)Cc1ccc(Cl)cc1", "C[N+](C)(Cc1ccccc1)Cc1ccc(Cl)cc1",
    "ClC[NH+](C)Cc1ccccc1", "C[NH+](Cc1ccccc1)Cc1ccc(Cl)cc1",
])
def test_the_cations_of_a_tie_lose_it_too(smiles):
    """The quaternary route never reaches the candidate pool, and the protonated amine is named as
    its neutral amine whose name changes its ending: the label is read for the shipped name."""
    assert Orthonym().name_tiered(smiles)["tier"] == "abstain"


@pytest.mark.parametrize("smiles, tier", [
    ("CN(C)Cc1ccccc1", "pin_verified"), ("C[N+](C)(C)Cc1ccccc1", "pin_verified"),
    ("C[NH+](C)Cc1ccccc1", "pin_verified"), ("CN(Cc1ccccc1)Cc1ccccc1", "pin_verified"),
    ("CCN(C)Cc1ccccc1", "pin_verified"),
])
def test_a_decided_choice_keeps_its_label(smiles, tier):
    assert Orthonym().name_tiered(smiles)["tier"] == tier


# ---- the best-effort offer ----------------------------------------------------------------------------

BEST_EFFORT = dict(general_fallback=True, general_fallback_unverified=True, allow_aromatic_general=True)

#: (SMILES, a spelling rule the chain-parent name broke)
OFFERED = [
    ("COCCNCC=CC1=CC=C(O1)Cl", "P-14.5 / P-16.5.1.1"),
    ("COCCNCC=Cc1ccccc1", "P-16.5.1.1"),
    ("CN1C=C(C=N1)C2=CC3=C(N=C2)O[C@@H](CN3)[C@@H](C4=CC=CC=C4)NCCC5=CC=C(C=C5)OC", "P-91.3"),
]


@pytest.mark.parametrize("smiles, rule", OFFERED)
def test_the_best_effort_tier_ships_a_name_that_breaks_no_spelling_rule(smiles, rule):
    row = Orthonym(**BEST_EFFORT).name_tiered(smiles)
    assert row["tier"] != "abstain" and row["verified"] == "opsin"
    assert check_pin_spelling(Chem.MolFromSmiles(smiles), row["name"]) == []
    assert row["spelling_failures"] == []


@pytest.mark.parametrize("smiles, rule", OFFERED)
def test_the_pin_tier_is_not_touched(smiles, rule):
    assert Orthonym().name_tiered(smiles)["tier"] == "abstain"


def test_without_a_clean_junior_name_the_row_stays_and_lists_what_it_breaks(monkeypatch):
    """A name that no member of the pool gives clean ships as it was, its failures listed."""
    from orthonym.validation import pin_spelling
    stub = [pin_spelling.SpellingFailure("P-14.5", "stub")]
    monkeypatch.setattr(pin_spelling, "check_pin_spelling", lambda mol, name, **kw: list(stub))
    row = Orthonym(**BEST_EFFORT).name_tiered("COCCNCC=Cc1ccccc1")
    assert row["tier"] != "abstain" and row["name"]
    assert row["spelling_failures"] == [{"rule": "P-14.5", "detail": "stub"}]


def test_a_molecule_that_is_not_a_chain_parent_amine_is_never_offered():
    """The offer reads a flag that the chain-parent label of a secondary or tertiary amine sets for its own
    molecule: a row of another molecule is returned as it is."""
    from orthonym.assembly import spelling_offer
    row = {"tier": "systematic_verified", "name": "whatever", "verified": "opsin", "spelling_failures": []}
    namer = Orthonym(**BEST_EFFORT)
    assert spelling_offer.prefer_clean_name(namer, "CC(C)(C)Nc1ccccc1", row) is row
    assert spelling_offer.prefer_clean_name(namer, "OCc1ccccc1", row) is row


def test_a_junior_parent_is_never_adopted_as_pin_verified(monkeypatch):
    """A junior parent is not the PIN while the PIN tier declines the molecule: an alternative labelled
    pin_verified is refused, one labelled below it is taken."""
    from orthonym import namer as namer_module
    from orthonym.assembly import spelling_offer
    smiles = "COCCNCC=Cc1ccccc1"
    chain_name = "N-2-methoxyethyl-3-phenylprop-2-en-1-amine"            # breaks
    clean = "{3-[(2-methoxyethyl)amino]prop-1-en-1-yl}benzene"

    class FakeNamer:
        _general_fallback_unverified = True
        _forced_parent_rank = 0
        _top_parent_pool_size = 2
        style = "pin"

    def fake_orthonym(tier):
        class Inner:
            def __init__(self, **kw):
                assert kw["_forced_parent_rank"] == 1
            def name_tiered(self, smi):
                return {"name": clean, "tier": tier, "verified": "opsin", "spelling_failures": []}
        return Inner

    def row():
        return {"name": chain_name, "tier": "systematic_verified", "verified": "opsin", "spelling_failures": []}

    spelling_offer._AMINE_CHAIN_PARENT.set(Chem.MolToSmiles(Chem.MolFromSmiles(smiles)))
    monkeypatch.setattr(namer_module, "Orthonym", fake_orthonym("pin_verified"))
    refused = spelling_offer.prefer_clean_name(FakeNamer(), smiles, row())
    assert refused["name"] == chain_name and refused["spelling_failures"][0]["rule"] == "P-16.5.1.1"
    monkeypatch.setattr(namer_module, "Orthonym", fake_orthonym("systematic_verified"))
    assert spelling_offer.prefer_clean_name(FakeNamer(), smiles, row())["name"] == clean


def test_the_flag_of_an_earlier_naming_is_cleared_before_the_next():
    from orthonym.assembly import spelling_offer
    spelling_offer._AMINE_CHAIN_PARENT.set("CCO")
    seen = []
    row = {"name": "x", "tier": "abstain"}
    out = spelling_offer.name_tiered_with_offer(
        None, "CCO", lambda smi: seen.append(spelling_offer._AMINE_CHAIN_PARENT.get()) or row)
    assert seen == [None] and out is row


# ---- the second cited simple prefix keeps its multiplier outside the marks,:7272) ----

@pytest.mark.parametrize("part, expected", [
    ("diphenyl", "di(phenyl)"), ("triphenyl", "tri(phenyl)"), ("dimethyl", "di(methyl)"),
    ("phenyl", "(phenyl)"), ("chloro", "(chloro)"),
    ("diazenyl", "(diazenyl)"),                 # one simple group whose name begins with 'di'
    ("dimethylamino", "(dimethylamino)"),       # a compound group: whole
])
def test_the_multiplier_of_a_second_simple_prefix_stays_outside_the_marks(part, expected):
    from orthonym.assembly.prefix_marks import enclose_second_prefix
    assert enclose_second_prefix(part, -1) == expected


@pytest.mark.parametrize("smiles, expected", [
    ("CN(C)C(c1ccccc1)c1ccccc1", "N,N-dimethyldi(phenyl)methanamine"),
    ("CN(C)C(c1ccccc1)(c1ccccc1)c1ccccc1", "N,N-dimethyltri(phenyl)methanamine"),
    ("CN(C)Cc1ccccc1", "N,N-dimethyl(phenyl)methanamine"),
    ("CN(C)C(C)(c1ccccc1)c1ccccc1", "N,N-dimethyl-1,1-diphenylethan-1-amine"),   # two positions: locants cited
])
def test_the_amine_producer_spells_the_second_multiplied_prefix_by_the_book(smiles, expected):
    """ (:7272): "When the simple substituent groups are accompanied by multiplicative prefixes such
    as 'di' and 'tri', the multiplicative prefixes are not included in the parentheses" ('ethyldi(methyl)phosphane
    (PIN)',:7290); the locants of the C-prefix are omitted by the licence of as for the rows above."""
    assert Orthonym().name_tiered(smiles)["name"] == expected
    assert Orthonym().name_tiered(smiles)["tier"] == "pin_verified"
