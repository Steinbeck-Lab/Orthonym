"""Slice S2b protection: names the phane check must leave alone, and the book's
phane row that must stay without a bridged fused name, at both tiers.

- W2E-P5BR-2 (a gate target): '5,10-ethenobenzo[8]annulene' for the formal
  1,4-buta[1,3]dienonaphthalene. Its etheno bridge is two carbon atoms of the naphthalene, but
  every atom of the ring system carries a ring double bond (one mancude polycycle, no chain),
  as in the book's (b) example '6,7-(epiprop[1]en[1]yl[3]ylidene)benzo[a]-
  cyclohepta[e][8]annulene (I) (PIN)' (the Blue Book).
- '2,6-hexanonaphthalene': a naphthalene with a saturated chain across nonadjacent atoms that
  stays the bridge (the S2 TRIAGE concern 11 reading: bridged fused, as the pentano bridge of
  '1,4-methano-10,13-pentanonaphtho[2,3-c][1]benzazocine (PIN)':19904).
-:23895 '(I) 3,7-dithia-1(1,7),5(7,1)-dinaphthalenacyclooctaphane (PIN; a phane name)': no
  bridged fused name at the PIN tier; best-effort keeps its verified von Baeyer name."""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots

KEEP = [
    ("C12=CC=C(C3=CC=CC=C13)C=CC=C2", "5,10-ethenobenzo[8]annulene"),       # W2E-P5BR-2
    ("C1CCCc2ccc3cc(ccc3c2)CC1", "2,6-hexanonaphthalene"),
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


@pytest.mark.opsin_gate
def test_the_book_phane_row_gets_no_bridged_fused_name():
    pin = _row(BOOK_PHANE, "pin")
    assert pin["tier"] == "abstain", (pin.get("name"), pin["tier"])
    best = _row(BOOK_PHANE, "best-effort")
    assert best["tier"] == "systematic_verified" and "dithiapentacyclo" in best["name"], best
