"""The flat name with the input's stereo composed on it: the last offer of the best-effort tier.

``assembly/stereo_composed_offer.py``. A molecule with stereo that every path of the best-effort tier leaves
without a name is named through its stereo-free structure: the name of that structure, with the stereo of the
input composed on it in OPSIN's numbering, when OPSIN reads the composed name back to the input's full InChIKey.

The expected names below are the composed names the engine gives; what is checked independently of the engine is
the read-back (``assert_full_rt``: a fresh OPSIN call, the full key, the radical graph) and the label. The Blue
Book has no PIN for these molecules and the composed name is not one: NAMING OF STEREOISOMERS
(the Blue Book), sentence:44643, "In preferred IUPAC names, stereodescriptors are placed immediately at the
front of the part of the name to which they relate... They are preceded by a numerical or letter locant to describe
the position of the stereogenic unit". The composed name cites every descriptor at the front of the complete name, so
a substituent's descriptor can repeat a locant of the parent. A repeated locant does not give the position of a
stereogenic unit: ``(3S,3R,...)`` is read to the input and ``(3R,3S,...)`` to a diastereomer, so such a name is
offered only when every descriptor at the locant is the same and the descriptors state every stereo unit of the
input (``(4S,4S,4S,4'S,...)``). The spelling is not checked against another reader: UNVERIFIED beyond OPSIN 2.9.0's
reading.
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.assembly import stereo_composed_offer as offer
from orthonym.cli import _emit_tier_flags
from orthonym.errors import is_failure_name
from tests.support.rt_assert import assert_full_rt, name_best_effort

pytestmark = pytest.mark.opsin_gate

# PubChem CID178930232: the run with the stereo ends at a refusal and the stereo-free run names it, but the composed
# name cites a locant twice with two descriptors, (3S,3R,...): the offer declines it
_AZABICYCLO = "C[C@H]1CCOC[C@@H]2N1C3=NC(=NC4=C3C(=NC(=C4F)Cl)N2C5CC5)OC[C@@]67CCCN6C[C@@H](C7)F"
_AZABICYCLO_REPEATED = (
    "(3S,3R,5S,8R)-12-chloro-9-cyclopropyl-13-fluoro-16-[(3-fluoro-1-azabicyclo[3.3.0]octan-5-yl)methoxy]-3-methyl-"
    "6-oxa-2,9,11,15,17-pentaazatetracyclo[8.7.1.0^2,8.0^14,18]octadeca-1(18),10,12,14,16-pentaene")
# PubChem CID57339753: one descriptor at each repeated locant, and every stereo unit stated (16 of 16)
_SULFOOXY = (
    "C1=C[C@H]([C@H]([C@@H]([C@H]1N([C@H]2C=C[C@H]([C@H]([C@@H]2OS(=O)(=O)O)OS(=O)(=O)O)OS(=O)(=O)O)"
    "C3=CC=C(C=C3)N([C@H]4C=C[C@H]([C@H]([C@@H]4OS(=O)(=O)O)OS(=O)(=O)O)OS(=O)(=O)O)"
    "[C@H]5C=C[C@H]([C@H]([C@@H]5OS(=O)(=O)O)OS(=O)(=O)O)OS(=O)(=O)O)OS(=O)(=O)O)OS(=O)(=O)O)OS(=O)(=O)O")
_SULFOOXY_NAME = (
    "(1S,1'S,1'S,1'''S,4R,4'R,4'R,4'''R,5R,5'R,5'R,5'''R,6R,6'R,6'R,6'''R)-"
    "1,4-bis{bis[4,5,6-tris(sulfooxy)cyclohex-2-en-1-yl]amino}benzene")
# PubChem CID140605103: two stereo units, two locants, no repeat
_AZANIDE = "C1CC[C@H]([C@@H](C1)NC2=NC3=C(S2)C=C(C=C3)CN4C5=CC=CC=C5N=C4[NH-])O"
_AZANIDE_NAME = ("(1R,2R)-[1-({2-[(2-hydroxycyclohexyl)amino]-1,3-benzothiazol-6-yl}methyl)-1H-1,3-benzimidazol-2-yl]"
                 "azanide")
# PubChem CID102144947: eight oxazoline stereocentres; the composed name has one descriptor at each repeated locant
_OXAZOLINES = (
    "CC(C)[C@H]1COC(=N1)N(CC2=CN(N=N2)CC(CN3C=C(N=N3)CN(C4=N[C@H](CO4)C(C)C)C5=N[C@H](CO5)C(C)C)"
    "(CN6C=C(N=N6)CN(C7=N[C@H](CO7)C(C)C)C8=N[C@H](CO8)C(C)C)CN9C=C(N=N9)CN(C1=N[C@H](CO1)C(C)C)"
    "C1=N[C@H](CO1)C(C)C)C1=N[C@H](CO1)C(C)C")
_OXAZOLINES_PREFIX = "(4S,4S,4S,4'S,4'S,4'S,4'S,4'''S)-"
# PubChem CID59055008: a P(III) stereocentre; the lone-pair check withdraws every name of it
_PHOSPHOLE = "CC1C2=CC=CC=C2[P@](N1C)C"


@pytest.mark.parametrize("smiles,name,producer", [
    (_SULFOOXY, _SULFOOXY_NAME, "engine:general"),
    (_AZANIDE, _AZANIDE_NAME, "outer:anion_small"),             # the producer of the stereo-free name
])
def test_the_composed_name_is_offered_and_reads_back_in_full(smiles, name, producer):
    row = name_best_effort(smiles)
    assert row["name"] == name, row
    assert_full_rt(row["name"], smiles)
    # a verified systematic name of the general engine, never the PIN
    assert row["tier"] == "systematic_verified" and row["is_pin"] is False, row
    assert row["gate_outcome"] == "stereo_omission_full_key_recomposed" and row["verified"] == "opsin", row
    assert row["producer"] == producer and row["limit_code"] is None, row


def test_name_and_name_tiered_agree():
    namer = Orthonym(style="pin", **_emit_tier_flags("best-effort"))
    assert namer.name(_AZANIDE) == _AZANIDE_NAME


def test_a_repeated_locant_with_two_descriptors_is_declined():
    # (the Blue Book): the descriptors are "preceded by a numerical or letter locant to describe the
    # position of the stereogenic unit". The composed name is (3S,3R,5S,8R)-...: OPSIN reads it to the input and
    # (3R,3S,5S,8R)-... to a diastereomer, so it rests on the order of reading and the abstention stands.
    row = name_best_effort(_AZABICYCLO)
    assert row["tier"] == "abstain" and row["name"] is None and row["limit_code"] == "UNNAMEABLE", row
    namer = Orthonym(style="pin", **_emit_tier_flags("best-effort"))
    assert is_failure_name(namer.name(_AZABICYCLO, on_failure="label"))


@pytest.mark.parametrize("name,smiles,expected", [
    (_AZABICYCLO_REPEATED, _AZABICYCLO, False),                                        # 3S,3R: two descriptors at 3
    (_AZABICYCLO_REPEATED.replace("(3S,3R,", "(3R,3S,"), _AZABICYCLO, False),
    (_SULFOOXY_NAME, _SULFOOXY, True),                                                 # one descriptor at each, 16 of 16
    (_OXAZOLINES_PREFIX + "x", _OXAZOLINES, True),                                     # 4S x3, 4'S x4, 4'''S: 8 of 8
    (_OXAZOLINES_PREFIX.replace("4S,4S,4S,", "4S,4S,") + "x", _OXAZOLINES, False),     # a unit is left out: 7 of 8
    (_OXAZOLINES_PREFIX + "x", _OXAZOLINES.replace("[C@H]1COC(=N1)N(CC2", "C1COC(=N1)N(CC2", 1), False),  # one unspecified
    (_OXAZOLINES_PREFIX + "x", _OXAZOLINES + " |&1:3|", False),                        # an AND group: not absolute
    (_AZANIDE_NAME, _AZANIDE, True),                                                   # no locant repeated
    ("(1R,2R)-x", "C[C@H](N)C[C@H](N)CC", True),
    ("(1R,2R)-x", "C[C@H](N)C[C@H](N)CC |&1:1|", True),                                # no repeat: nothing to decide
    ("(2Z)-4-(propan-2-yl)-x", "C/C=C\\C", True),
])
def test_a_repeated_locant_names_one_structure_only_under_both_conditions(name, smiles, expected):
    assert offer.repeated_locants_are_one_structure(name, smiles) is expected


@pytest.mark.parametrize("name,groups", [
    ("(4aR,8aS)-x", [[("4a", "R"), ("8a", "S")]]),
    ("(2Z)-4-[(1S)-x](propan-2-yl)(1H)", [[("2", "Z")], [("1", "S")]]),
    ("(Z)-x-(Ra,3Sa)-y", [[("", "Z")], [("", "Ra"), ("3", "Sa")]]),
    ("2-methylpropan-1-ol", []),
])
def test_the_groups_of_stereodescriptors_of_a_name(name, groups):
    assert offer.stereo_groups(name) == groups


@pytest.mark.parametrize("marker", ["|&1:3|", "|o1:3|"])
def test_an_input_with_an_enhanced_stereo_group_is_not_offered(marker):
    # the centres of an AND or OR group are not absolute: an absolute descriptor would assert they are. The
    # same input without the group is offered (the control of the first assertions).
    namer = Orthonym(style="pin", **_emit_tier_flags("best-effort"))
    assert offer.has_stereo_groups(f"{_AZANIDE} {marker}") and not offer.has_stereo_groups(_AZANIDE)
    assert offer.stereo_composed_row(namer, f"{_AZANIDE} {marker}") is None
    assert offer.stereo_composed_row(namer, Chem.MolFromSmiles(f"{_AZANIDE} {marker}")) is None
    assert offer.stereo_composed_row(namer, _AZANIDE)["name"] == _AZANIDE_NAME


def test_a_lone_pair_stereocentre_stays_without_a_name():
    # the stereo-free run names the molecule and the composition refuses it (a probe of the composition alone
    # gives None for this input); the configuration check of the lone pair stands behind it: the abstention
    # stands, with its reason
    row = name_best_effort(_PHOSPHOLE)
    assert row["tier"] == "abstain" and row["name"] is None and row["limit_code"] == "UNNAMEABLE", row
    assert is_failure_name(Orthonym(style="pin", **_emit_tier_flags("best-effort")).name(_PHOSPHOLE, on_failure="label"))


def test_the_pin_tier_and_the_other_tiers_make_no_offer():
    # the offer is the best-effort tier's: no naming is made for any other engine
    for tier in ("pin", "valid", "complete"):
        namer = Orthonym(style="pin", **_emit_tier_flags(tier))
        assert offer.stereo_composed_row(namer, _AZANIDE) is None, tier
    assert Orthonym(style="pin").name_tiered(_AZANIDE)["tier"] == "abstain"


def test_no_offer_without_the_reclaim(monkeypatch):
    import orthonym.namer as namer_module
    monkeypatch.setattr(namer_module, "_STEREO_OMISSION_RECLAIM", False)
    namer = Orthonym(style="pin", **_emit_tier_flags("best-effort"))
    assert offer.stereo_composed_row(namer, _AZANIDE) is None


@pytest.mark.parametrize("smiles,flat", [
    ("C[C@H](N)C(=O)O", "CC(N)C(=O)O"),
    ("C/C=C/C", "CC=CC"),
    ("C[C@@H]1CC[C@H](O)CC1", "CC1CCC(O)CC1"),
    ("[13CH3][C@H](N)C(=O)O", "[13CH3]C(N)C(=O)O"),              # the isotope stays: only stereo is removed
])
def test_the_stereo_free_structure(smiles, flat):
    assert offer.stereo_free_smiles(smiles) == Chem.MolToSmiles(Chem.MolFromSmiles(flat))


@pytest.mark.parametrize("smiles", ["CC(N)C(=O)O", "c1ccccc1", "not a smiles"])
def test_no_stereo_no_stereo_free_structure(smiles):
    # a molecule with no stereo is the stereo-free structure itself: the offer cannot name it again
    assert offer.stereo_free_smiles(smiles) is None


def test_an_offer_that_fails_leaves_the_abstention_and_the_provenance(monkeypatch):
    import orthonym.namer as namer_module
    from orthonym.metrics import provenance as pv
    from orthonym.metrics.abstention import (
        AbstentionCode, clear_abstention, peek_abstention, record_abstention)
    monkeypatch.setattr(namer_module, "_try_compose_input_stereo", lambda *args, **kwargs: None)
    clear_abstention()
    record_abstention(AbstentionCode.OTHER, "kept")
    pv.clear_provenance()
    pv.record_source("pin_path")

    def state():
        # the name-scoped record of non-PIN fragments only grows within a call (``restore_provenance``)
        return peek_abstention(), {k: v for k, v in pv.get_provenance().items() if k != "non_pin_fragments"}

    before = state()
    namer = Orthonym(style="pin", **_emit_tier_flags("best-effort"))
    assert offer.stereo_composed_row(namer, _AZANIDE) is None         # the stereo-free run names it; no composition
    assert state() == before


def test_the_stereo_free_structure_is_named_by_an_engine_of_the_callers_tier(monkeypatch):
    # one naming of the stereo-free structure, never of the input, by an engine with the flags of the caller
    from orthonym.assembly import default_tier_offer
    seen = []
    real = default_tier_offer.general_tier_engine

    def spy(namer):
        engine = real(namer)
        name_tiered = engine.name_tiered
        engine.name_tiered = lambda smiles: seen.append((smiles, engine._general_fallback_unverified)) or name_tiered(smiles)
        return engine

    monkeypatch.setattr(default_tier_offer, "general_tier_engine", spy)
    namer = Orthonym(style="pin", **_emit_tier_flags("best-effort"))
    assert offer.stereo_composed_row(namer, _AZANIDE)["name"] == _AZANIDE_NAME
    assert seen == [(offer.stereo_free_smiles(_AZANIDE), True)]


def test_a_composed_name_that_the_exit_check_withdraws_is_not_offered(monkeypatch):
    # the exit check of every other name (OPSIN reads it as ambiguous, a lone-pair configuration) is the offer's too,
    # and what it withdraws leaves the abstention and the provenance of the caller as they were
    from orthonym.metrics import provenance as pv
    from orthonym.metrics.abstention import (
        AbstentionCode, clear_abstention, peek_abstention, record_abstention)
    exit_check = Orthonym._exit_round_trip_check

    def withdraws_the_composed_name(self, name, smiles):
        return "unknown organic compound" if name == _AZANIDE_NAME else exit_check(self, name, smiles)

    monkeypatch.setattr(Orthonym, "_exit_round_trip_check", withdraws_the_composed_name)
    clear_abstention()
    record_abstention(AbstentionCode.OTHER, "kept")
    pv.clear_provenance()
    pv.record_source("pin_path")
    namer = Orthonym(style="pin", **_emit_tier_flags("best-effort"))
    assert offer.stereo_composed_row(namer, _AZANIDE) is None
    assert peek_abstention().detail == "kept" and pv.get_provenance()["source"] == "pin_path"
