"""Slice S2b protection: names the phane check must leave alone, and the book's
phane row that must stay without a bridged fused name, at both tiers.

- W2E-P5BR-2 (a gate target): '5,10-ethenobenzo[8]annulene' for the formal
  1,4-buta[1,3]dienonaphthalene. Its etheno bridge is two carbon atoms of the naphthalene, but
  every atom of the ring system carries a ring double bond (one mancude polycycle, no chain),
  as in the book's (b) example '6,7-(epiprop[1]en[1]yl[3]ylidene)benzo[a]-
  cyclohepta[e][8]annulene (I) (PIN)' (the Blue Book).
- '2,6-hexanonaphthalene' was listed here as a name the phane check must keep (the S2 TRIAGE concern
  11 reading: bridged fused, as the pentano bridge of:19904). It is no PIN: (:15033) prints
  '1(3,10)-fluoranthenacyclononaphane (I) (PIN)' (:15039), a phane PIN of a macrocycle closed by a chain
  across one polycyclic system, so a naphthalene joined at the nonadjacent positions 2 and 6 meets
   (1) (:23828) and the bridged fused name is not preferred,:23831); the two
  tests at the end of this file state the label it has now.:19904 is a bridged system, not a mancude
  one, and '1,4-ethanonaphthalene' (:14407) and '9,10-ethanoanthracene' (:14183) close macrocycles of
  fewer than ten atoms, which the structure test does not judge.
-:23895 '(I) 3,7-dithia-1(1,7),5(7,1)-dinaphthalenacyclooctaphane (PIN; a phane name)': no
  bridged fused name at the PIN tier; best-effort keeps its verified von Baeyer name."""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots
from tests.support.opsin_marks import needs_phane_opsin, release_opsin_only
from tests.support.rt_assert import name_is_rt_exact

KEEP = [
    ("C12=CC=C(C3=CC=CC=C13)C=CC=C2", "5,10-ethenobenzo[8]annulene"),       # W2E-P5BR-2
]
BOOK_PHANE = "C1SCc2ccc3cccc(CSCc4ccc5cccc1c5c4)c3c2"                      #:23895


def _row(smiles, tier):
    with jvm_slots(1, purpose="bf-s2b"):
        if tier == "pin":
            return Orthonym().name_tiered(smiles)
        return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles,name", KEEP)
def test_names_the_phane_check_must_keep(smiles, name, tier):
    row = _row(smiles, tier)
    assert (row.get("name"), row["tier"]) == (name, "pin_verified"), (row.get("name"), row["tier"])


@release_opsin_only
@pytest.mark.opsin_gate
def test_the_book_phane_row_gets_no_bridged_fused_name():
    pin = _row(BOOK_PHANE, "pin")
    assert pin["tier"] == "abstain", (pin.get("name"), pin["tier"])
    best = _row(BOOK_PHANE, "best-effort")
    assert best["tier"] == "systematic_verified" and "dithiapentacyclo" in best["name"], best


#: a hexano chain across the 2,6-positions of naphthalene: the bridged fused name was a PIN of the base
HEXANO = "C1CCCc2ccc3cc(ccc3c2)CC1"
HEXANO_BRIDGED_FUSED = "2,6-hexanonaphthalene"


@release_opsin_only
@pytest.mark.opsin_gate
def test_a_hexano_chain_across_naphthalene_has_no_bridged_fused_pin():
    # UNVERIFIED reading of the label, resting on:15039 and (1) (see the module text); the structure
    # is a cyclophane, so the default tier declines the bridged fused name and best-effort keeps it below the PIN
    assert _row(HEXANO, "pin")["tier"] == "abstain"
    best = _row(HEXANO, "best-effort")
    assert (best["name"], best["tier"]) == (HEXANO_BRIDGED_FUSED, "systematic_verified"), best
    assert name_is_rt_exact(HEXANO_BRIDGED_FUSED, HEXANO)


@needs_phane_opsin
@pytest.mark.opsin_gate
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_a_hexano_chain_across_naphthalene_is_named_as_a_phane(tier):
    # (:15033): the lower attachment locant first, '1(2,6)' as the book's '1(3,10)-fluoranthena...'; OPSIN
    # reads the name to the full InChIKey of the input
    name = "1(2,6)-naphthalenacycloheptaphane"
    row = _row(HEXANO, tier)
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert name_is_rt_exact(name, HEXANO)
