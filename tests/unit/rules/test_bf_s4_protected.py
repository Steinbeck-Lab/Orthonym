"""Slice S4 protection: what the morphine-class build must leave alone, at every task.

- Natural products without a bridged fused reading keep their names and labels at both
  tiers: 'cholesterol' (the exact-match table), 'cholesta-4,6-dien-3beta-ol' (a
  steroid name; a dev split and milestone1500), 'germacrane' (the exact-match list of
  parents, a monocycle). 'hetisan', an exact-match list parent the bridged fused builder does
  not name, keeps its name at the default tier (best-effort ships a verified von Baeyer
  name, as before). The names of the exact-match list are no PINs: INTRODUCTION
  (the Blue Book;:50943) "Preferred IUPAC names (PINs) are not identified for the compounds
  in this Chapter." Their label is systematic_verified, verified 'identity' (lead decision
  2026-10-17, the user delegating it to the Blue Book; names and emission unchanged).
- The best-effort tier keeps the name of every class member the builder does not complete
  (an ester, a morphinan without the 4,5-epoxy bridge; the ring ketones are named since slice
  S3, ``test_bf_s3_morphinan.py``), each read back to the
  input's full InChIKey; only its label may change (Task S4.4)."""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots
from tests.support.rt_assert import name_is_rt_exact

# NPL 2026-11-01 (these rows kept 'cholesterol' and 'cholesta-4,6-dien-3β-ol' at pin_verified): INTRODUCTION
# (the Blue Book) "Preferred IUPAC names (PINs) are not identified for the compounds in this Chapter"; the route
# offers the systematic name of Chapters to first (:50943), so both take it, still pin_verified at both tiers
# (OPSIN 2.9.0 reads each back to the full InChIKey; ``test_the_systematic_names_read_back``).
KEEP_BOTH_TIERS = [
    ("C[C@H](CCCC(C)C)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C",
     "(3S,8S,9S,10R,13R,14S,17R)-10,13-dimethyl-17-[(2R)-6-methylheptan-2-yl]-"
     "2,3,4,7,8,9,10,11,12,13,14,15,16,17-tetradecahydro-1H-cyclopenta[a]phenanthren-3-ol"),
    ("CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3C=CC4=C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C",
     "(3S,8S,9S,10R,13R,14S,17R)-10,13-dimethyl-17-[(2R)-6-methylheptan-2-yl]-"
     "2,3,8,9,10,11,12,13,14,15,16,17-dodecahydro-1H-cyclopenta[a]phenanthren-3-ol"),
]
# the exact-match list parent of (a monocycle): the name is kept at both tiers, below the PIN
KEEP_LIST_NAME_BOTH_TIERS = [
    ("CC(C)[C@@H]1CC[C@H](C)CCC[C@H](C)CC1", "germacrane"),
]
HETISAN = "C=C1C[C@]23C[C@H]4[C@@H]5[C@@]6(C)CCC[C@]57C(C2C[C@H]1C[C@H]37)N4C6"

KEEP_BEST_EFFORT_NAME = [
    ("CN1CC[C@]23c4c5ccc(OC(C)=O)c4O[C@H]2[C@@H](OC(C)=O)C=C[C@H]3[C@H]1C5", "diamorphine"),
    ("COc1ccc2c(c1)[C@@]13CCCC[C@@H]3[C@@H](C2)N(C)CC1", "(9R,13S,14S)-3-methoxy-17-methylmorphinan"),
    ("Oc1ccc2c(c1)[C@@]13CCCC[C@@H]3[C@@H](C2)N(C)CC1", "(9R,13S,14S)-17-methylmorphinan-3-ol"),
]


def _row(smiles, tier):
    with jvm_slots(1, purpose="bf-s4"):
        if tier == "pin":
            return Orthonym().name_tiered(smiles)
        return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles,name", KEEP_BOTH_TIERS)
def test_natural_products_outside_the_class_keep_name_and_label(smiles, name, tier):
    row = _row(smiles, tier)
    assert (row.get("name"), row["tier"]) == (name, "pin_verified"), (row.get("name"), row["tier"])


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", KEEP_BOTH_TIERS)
def test_the_systematic_names_read_back(smiles, name):
    assert name_is_rt_exact(name, smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles,name", KEEP_LIST_NAME_BOTH_TIERS)
def test_an_exact_match_list_name_is_kept_at_both_tiers_below_the_pin(smiles, name, tier):
    """ INTRODUCTION (the Blue Book;:50943): "Preferred IUPAC names (PINs) are not identified for
    the compounds in this Chapter." The name is the list name (verified 'identity'), labelled systematic_verified;
    mutation: the label assertion fails on the commit before (pin_verified).
    NPL 2026-11-01: the default tier now gives the systematic name of Chapters to (:50943),
    '(1R,4s,7S)-1,7-dimethyl-4-(propan-2-yl)cyclodecane', pin_unverified (OPSIN 2.9.0 parses no lowercase
    pseudoasymmetric descriptor; its stereo-stripped form reads back to the constitution); the best-effort tier keeps
    the list name."""
    row = _row(smiles, tier)
    # 2026-11-01 (changed asserted value; advisor ruling, internal notes): was 'germacrane' at best-effort. The default tier ships the systematic name through its stereo-layer carve-out, and a name of the PIN path outranks a list name OPSIN cannot read INTRODUCTION, the Blue Book; user decision 2026-10-31, systematic name first). A name still ships at every tier (user decision 2026-09-30).
    if tier in ("pin", "best-effort"):
        assert (row.get("name"), row["tier"]) == (
            "(1R,4s,7S)-1,7-dimethyl-4-(propan-2-yl)cyclodecane", "pin_unverified"), (row.get("name"), row["tier"])
        assert row["is_pin"] is False, row
        return
    assert (row.get("name"), row["tier"]) == (name, "systematic_verified"), (row.get("name"), row["tier"])
    assert row["verified"] == "identity" and row["is_pin"] is False, row


@pytest.mark.opsin_gate
def test_an_exact_match_list_parent_the_builder_does_not_name_keeps_its_name():
    """'hetisan': no PIN either,:50943); the name is kept at the default tier, labelled systematic_verified."""
    row = _row(HETISAN, "pin")
    assert (row.get("name"), row["tier"]) == ("hetisan", "systematic_verified"), (row.get("name"), row["tier"])
    assert row["verified"] == "identity" and row["is_pin"] is False, row


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", KEEP_BEST_EFFORT_NAME)
def test_best_effort_keeps_the_name_of_members_the_builder_does_not_complete(smiles, name):
    row = _row(smiles, "best-effort")
    assert row.get("name") == name, (row.get("name"), row["tier"])
    assert name_is_rt_exact(name, smiles)
