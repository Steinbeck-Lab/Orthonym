"""A wider tier gives the PIN tier's name for a molecule its own naming names in a lower class (``assembly.pin_first``).

The wider tiers name a molecule in one pass with the breadth producers in it, so the ring cascade could name it on a
ring parent that cites the principal characteristic group as a prefix: an amide as a prefix of a phenol, the two
ring C=O of a bicyclic lactam as 'dioxo' on an indole parent.: the parent has the maximum number of
principal characteristic groups. The cascade notes such a name and the PIN tier is asked for that molecule only; its
row is taken when it is a PIN read back directly. A PIN that only a twin spelling verifies is not taken: the wider
tiers ship the name OPSIN reads (user decision 2026-10-22).
"""
import pytest

from orthonym import name_compound
from orthonym.cli import _tiered_namer
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

_PIN_CASES = [
    # the principal group (an amide) sits on no ring
    ("O=CN/C=C\\c1ccc(O)cc1", "N-[(Z)-2-(4-hydroxyphenyl)ethenyl]formamide"),
    # the principal group sits on a ring that is not the cascade's parent
    ("CN1C(=O)[C@@]2(O)NC(=O)[C@]1(C)SS[C@@H]2c1c[nH]c2ccccc12",
     "(1S,4R,5S)-5-hydroxy-4-(1H-indol-3-yl)-1,8-dimethyl-2,3-dithia-6,8-diazabicyclo[3.2.2]nonane-7,9-dione"),
]


@pytest.mark.parametrize("tier", ["valid", "complete", "best-effort"])
@pytest.mark.parametrize("smiles,name", _PIN_CASES)
def test_a_wider_tier_gives_the_pin_of_the_pin_tier(smiles, name, tier):
    row = _tiered_namer("pin", tier).name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles,name", _PIN_CASES)
def test_every_entry_point_gives_the_same_pin(smiles, name):
    namer = _tiered_namer("pin", "best-effort")
    assert name_compound(smiles) == name
    record = namer.name_with_confidence(smiles)
    assert (record["name"], record["tier"]) == (name, "pin_verified"), record


def test_a_twin_verified_pin_leaves_the_readable_name_to_the_wider_tier():
    """'2-(N,N-dimethylmethanaminiumyl)ethyl dodecyl phosphate' is the PIN-tier name, read back only through a twin
    spelling; the wider tier keeps the name OPSIN reads."""
    smiles = "CCCCCCCCCCCCOP(=O)([O-])OCC[N+](C)(C)C"
    pin = _tiered_namer("pin", "pin").name_tiered(smiles)
    assert pin["opsin"] == "verified_twin", pin
    row = _tiered_namer("pin", "best-effort").name_tiered(smiles)
    assert row["name"] == "dodecyl 2-(trimethylazaniumyl)ethyl phosphate", row
    assert_full_rt(row["name"], smiles)


def test_a_name_no_producer_notes_is_not_named_again(monkeypatch):
    """The PIN tier runs only for a molecule a producer noted: an ordinary molecule is named once."""
    from orthonym.assembly import pin_first
    calls = []
    monkeypatch.setattr(pin_first, "_pin_tier_row", lambda namer, smiles: calls.append(smiles))
    assert name_compound("CCO") == "ethanol"
    assert _tiered_namer("pin", "best-effort").name_tiered("OC(=O)c1ccccc1")["name"] == "benzoic acid"
    assert calls == []
