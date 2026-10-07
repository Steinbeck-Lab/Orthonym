"""Slice S2c-1 makes two-component fused heterocycles nameable as spiro components; the spiro
writers keep the PIN label only for the book's spellings of such names.

 (the Blue Book, heading "MONOSPIRO RING SYSTEMS WITH DIFFERENT RING COMPONENTS"):
"When Roman letters are inadequate to distinguish alphabetically between two ring components,
criteria based on italic fusion letters and numbers, heteroatom locants, and von Baeyer
descriptor numbers are used"; the printed order of heteroatom locants is the order of the
lowest set,:3191): 'spiro[[3,1]benzoxazine-7,6'-[2,3]benzoxazine] (PIN)
(3,1-benzoxazine before 2,3-benzoxazine)' (:10318), 'spiro[[1,2]benzodithiole-3,2'-
[1,3]benzodithiole] (PIN)' (:10312); fusion locants and letters as written:
'2'H,5H-spiro[thieno[2,3-b]furan-4,3'-thieno[3,2-b]furan] (PIN)' (:10310),
'1H,2'H-spiro[benzo[g]isoquinoline-8,9'-benzo[h]isoquinoline] (PIN)' (:10301).

 (:10152): "Indicated hydrogen of individual components is not cited... If
indicated hydrogen is needed, it is cited in front of the spiro atom locants"
('2'H,4H-2,4'-spirobi[[1,3]dioxolo[4,5-c]pyran] (PIN)',:10172). The spirobi writer's branch
without a front citation declines a component name that still carries indicated hydrogen, and
the von Baeyer spiro writer labels such a name below the PIN (as it does for a component's own
hydro prefixes,.
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots
from orthonym.rules import spiro
from tests.support.rt_assert import name_is_rt_exact

BENZOXAZINES = "C1=CC2(C=CC3=COC=NC3=C2)C=C2C=NOC=C12"
DIOXOLOPYRANS = "C1=CC2=C(CO1)OC1(OC=CC3=C1OCO3)O2"


def _row(smiles, tier):
    with jvm_slots(1, purpose="s2c1-spiro"):
        if tier == "pin":
            return Orthonym().name_tiered(smiles)
        return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


@pytest.mark.parametrize("a,b,first", [
    ("[2,3]benzoxazine", "[3,1]benzoxazine", "[3,1]benzoxazine"),          #:10318
    ("[1,3]benzodithiole", "[1,2]benzodithiole", "[1,2]benzodithiole"),    #:10312
    ("thieno[3,2-b]furan", "thieno[2,3-b]furan", "thieno[2,3-b]furan"),    #:10310
    ("benzo[h]isoquinoline", "benzo[g]isoquinoline", "benzo[g]isoquinoline"),  #:10301
    ("[1,3]oxazole", "indene", "indene"),                                  # Roman letters first
])
def test_the_order_of_two_spiro_components(a, b, first):
    assert min((a, b), key=spiro._pin_component_alpha_key) == first
    assert min((b, a), key=spiro._pin_component_alpha_key) == first


@pytest.mark.opsin_gate
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_two_benzoxazine_components_take_the_book_order(tier):
    row = _row(BENZOXAZINES, tier)
    name = "spiro[[3,1]benzoxazine-7,6'-[2,3]benzoxazine]"
    assert (row.get("name"), row["tier"]) == (name, "pin_verified"), (row.get("name"), row["tier"])
    assert name_is_rt_exact(name, BENZOXAZINES)


@pytest.mark.opsin_gate
def test_a_spirobi_component_hydrogen_inside_the_bracket_is_not_the_pin():
    # the PIN is '2'H,4H-2,4'-spirobi[[1,3]dioxolo[4,5-c]pyran]' (:10172); no writer
    # builds that front citation for this component yet, so the default tier declines and
    # best-effort keeps the read-back name below the PIN
    assert spiro._name_spirobi_core(Chem.MolFromSmiles(DIOXOLOPYRANS)) is None
    assert _row(DIOXOLOPYRANS, "pin")["tier"] == "abstain"
    best = _row(DIOXOLOPYRANS, "best-effort")
    assert best.get("name") and best["tier"] != "pin_verified", (best.get("name"), best["tier"])
    assert name_is_rt_exact(best["name"], DIOXOLOPYRANS)
