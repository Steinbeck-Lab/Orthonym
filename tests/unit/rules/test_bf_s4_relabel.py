"""Slice S4 relabel: a natural-product name whose molecule has a bridged fused PIN the builder
does not complete is not the PIN; the wider tiers keep it labelled below the PIN and the
default tier declines it.

 (the Blue Book): "Preferred IUPAC names (PINs) are not identified for the
compounds in this Chapter"; (:14241): "When a polycyclic ring system cannot be
named completely as a fused ring system, possible ways for naming it as a bridged fused system
are considered". The user's decision on natural-product names (PIN class program Task 32):
"the default tier declines a natural-product name only where no systematic PIN is built".
Here: the esters (diamorphine) and the
morphinans without the 4,5-epoxy bridge, whose PIN needs the 'azanoethano' bridge prefix
,:14155 "-NH-CH2-CH2- (azanoethano) (preferred prefix)"): OPSIN 2.9.0 reads the
general-nomenclature 'epiminoethano' spelling but not the preferred 'azanoethano', so these
morphinans can get a verified bridged fused name below the PIN but never a PIN label. A natural
product without a bridged fused reading (cholesterol) keeps its name and label
(``test_bf_s4_routing.py``). The ring ketones (oxycodone, hydromorphone) have their PIN since
slice S3 (``test_bf_s3_morphinan.py``)."""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots
from orthonym.rules.bridged_fused_pin import has_bridged_fused_reading
from tests.support.rt_assert import name_is_rt_exact

#: (SMILES, the natural-product name the wider tiers keep)
BELOW_PIN = [
    ("CN1CC[C@]23c4c5ccc(OC(C)=O)c4O[C@H]2[C@@H](OC(C)=O)C=C[C@H]3[C@H]1C5", "diamorphine"),
    ("COc1ccc2c(c1)[C@@]13CCCC[C@@H]3[C@@H](C2)N(C)CC1", "(9R,13S,14S)-3-methoxy-17-methylmorphinan"),
    ("Oc1ccc2c(c1)[C@@]13CCCC[C@@H]3[C@@H](C2)N(C)CC1", "(9R,13S,14S)-17-methylmorphinan-3-ol"),
]
CHOLESTEROL = "C[C@H](CCCC(C)C)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C"


def _row(smiles, tier):
    with jvm_slots(1, purpose="bf-s4"):
        if tier == "pin":
            return Orthonym().name_tiered(smiles)
        return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


@pytest.mark.parametrize("smiles,name", BELOW_PIN)
def test_the_molecule_has_a_bridged_fused_reading(smiles, name):
    assert has_bridged_fused_reading(Chem.MolFromSmiles(smiles))


def test_a_steroid_has_no_bridged_fused_reading():
    # a fused ring system (cyclopenta[a]phenanthrene) is not bridged: (:14241)
    # does not apply
    assert not has_bridged_fused_reading(Chem.MolFromSmiles(CHOLESTEROL))


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", BELOW_PIN)
def test_the_default_tier_declines_the_natural_product_name(smiles, name):
    row = _row(smiles, "pin")
    assert (row["tier"], row.get("limit_code")) == ("abstain", "NO_VERIFIED_PIN"), row


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", BELOW_PIN)
def test_best_effort_keeps_the_natural_product_name_below_the_pin(smiles, name):
    row = _row(smiles, "best-effort")
    assert (row.get("name"), row["tier"]) == (name, "systematic_verified"), row
    assert name_is_rt_exact(name, smiles)



@pytest.mark.opsin_gate
def test_the_trivial_option_emits_a_retained_trivial_name_where_no_pin_is_built():
    # '--trivial': "When no preferred name can be built, also allow a retained trivial name
    # that is not a preferred name"; the user's decision: "trivial names as retained trivial
    # names". A semisystematic name is not a retained trivial name: still declined.
    with jvm_slots(1, purpose="bf-s4"):
        dia = Orthonym(trivial_fallback=True).name_tiered(BELOW_PIN[0][0])
        dxm = Orthonym(trivial_fallback=True).name_tiered(BELOW_PIN[1][0])
    assert (dia.get("name"), dia["tier"], dia["source"]) == (
        "diamorphine", "systematic_verified", "trivial_retained"), dia
    assert (dxm["tier"], dxm.get("limit_code")) == ("abstain", "NO_VERIFIED_PIN"), dxm


@pytest.mark.opsin_gate
def test_an_exact_match_list_parent_keeps_its_label():
    # 'hasubanan', the exact-match list) has a bridged fused reading the
    # builder does not complete; list names keep the label of the paper's measured run. A
    # policy carve-out (the paper's deliberate exception for list names, namer.py
    # _default_tier_emits), not a Blue Book reading: (:50943) names no PIN for it
    smiles = "c1ccc2c(c1)CC[C@@]13CCCC[C@@]21CCN3"
    assert has_bridged_fused_reading(Chem.MolFromSmiles(smiles))
    row = _row(smiles, "pin")
    assert (row.get("name"), row["tier"]) == ("hasubanan", "pin_verified"), row
