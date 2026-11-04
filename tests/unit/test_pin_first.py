"""A wider tier gives the PIN tier's name for a molecule its own naming names in a lower class (``assembly.pin_first``).

The wider tiers name a molecule in one pass with the breadth producers in it, so the ring cascade could name it on a
ring parent that cites the principal characteristic group as a prefix: an amide as a prefix of a phenol, the two
ring C=O of a bicyclic lactam as 'dioxo' on an indole parent.: the parent has the maximum number of
principal characteristic groups. The cascade notes such a name and the PIN tier is asked for that molecule only; its
row is taken when it is a PIN read back directly. A PIN that only a twin spelling verifies is not taken: the wider
tiers ship the name OPSIN reads (user decision 2026-10-22).

The last-resort floor of the best-effort engine names a molecule the main path and every rung declined, and spells
an azanium cation on the ring that carries the least ('...-2,3-dimethylcyclohexane' with an 'azaniumyl' prefix). It
asks the PIN tier too: SENIORITY ORDER FOR CLASSES, Table 4.1, puts cations (6) above amides (11), and
Cationic centers on characteristic groups: "the largest neutral parent possible is used", so the cation is the parent
and the amide a prefix, the name the PIN tier gives ('...-1-oxopropan-2-aminium'). The ask is made for a molecule with a
formal positive charge on a nitrogen atom, at valid and complete only: best-effort has no target for it and pays double.
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
    # the final floor names an azanium cation on a ring parent: a cation is senior to an amide)
    ("CCNC(=O)C[NH+](CC)[C@@H](C)C(=O)N[C@@H]1CCC[C@@H]([C@@H]1C)C",
     "(2S)-1-{[(1R,2S,3S)-2,3-dimethylcyclohexyl]amino}-N-ethyl-N-[2-(ethylamino)-2-oxoethyl]-1-oxopropan-2-aminium"),
    ("CCNC(=O)C[NH+](CC)[C@@H](C)C(=O)NC1CCCC(C)C1C",
     "(2S)-1-[(2,3-dimethylcyclohexyl)amino]-N-ethyl-N-[2-(ethylamino)-2-oxoethyl]-1-oxopropan-2-aminium"),
]


@pytest.mark.parametrize("tier", ["valid", "complete", "best-effort"])
@pytest.mark.parametrize("smiles,name", _PIN_CASES)
def test_a_wider_tier_gives_the_pin_of_the_pin_tier(smiles, name, tier):
    row = _tiered_namer("pin", tier).name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles,name", _PIN_CASES)
def test_the_pin_tier_names_the_pin_itself(smiles, name):
    row = _tiered_namer("pin", "pin").name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row


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


@pytest.mark.parametrize("tier", ["valid", "complete", "best-effort"])
def test_a_molecule_the_main_path_names_keeps_its_name_and_is_not_named_again(tier, monkeypatch):
    """The control of the final-floor cases: the diethyl cation is named by the main path, no floor name is noted, and
    the PIN tier is not asked."""
    from orthonym.assembly import pin_first
    calls = []
    real = pin_first._pin_tier_row
    monkeypatch.setattr(pin_first, "_pin_tier_row", lambda namer, smiles: calls.append(smiles) or real(namer, smiles))
    smiles = "CC[NH+](CC)[C@@H](C)C(=O)N[C@@H]1CCC[C@@H]([C@@H]1C)C"
    row = _tiered_namer("pin", tier).name_tiered(smiles)
    assert (row["name"], row["tier"]) == (
        "(2S)-1-{[(1R,2S,3S)-2,3-dimethylcyclohexyl]amino}-N,N-diethyl-1-oxopropan-2-aminium", "pin_verified"), row
    assert calls == []


_FLOOR_NITROGEN_CATION = "CC(C)[C@@H](C(=O)O)[n+]1ccc2c(c1)C(=O)C1=CC(C)(C)CC1C2(C)O"


def _asks_of_the_pin_tier(monkeypatch, smiles, tier):
    from orthonym.assembly import pin_first
    asks = []
    real = pin_first.note_pin_tier_asked
    monkeypatch.setattr(pin_first, "note_pin_tier_asked", lambda: asks.append(1) or real())
    row = _tiered_namer("pin", tier).name_tiered(smiles)
    return len(asks), row


@pytest.mark.parametrize("tier", ["valid", "complete"])
def test_a_final_floor_name_of_a_nitrogen_cation_asks_the_pin_tier(tier, monkeypatch):
    """ (cationic centers on characteristic groups): the class the final-floor ask was built for, a formal
    positive charge on a nitrogen atom, is still asked at the tiers that run it."""
    asks, row = _asks_of_the_pin_tier(monkeypatch, _FLOOR_NITROGEN_CATION, tier)
    assert asks >= 1, row


def test_a_final_floor_name_does_not_ask_the_pin_tier_at_best_effort(monkeypatch):
    """At best-effort the main path's PIN promotion gives the PIN; the ask had no target there and doubled the time of
    the rows it fired on (1.2x to 3.1x on ten engine:t4 rows)."""
    asks, row = _asks_of_the_pin_tier(monkeypatch, _FLOOR_NITROGEN_CATION, "best-effort")
    assert asks == 0, row


@pytest.mark.parametrize("smiles", [
    "[C+]#N",
    "CC1=CC(=O)CC(C)(C)[C@@]1(O)/C=C/C(C)=C\\C(=O)[O-]",
])
@pytest.mark.parametrize("tier", ["valid", "complete"])
def test_a_final_floor_name_without_a_nitrogen_cation_does_not_ask(smiles, tier, monkeypatch):
    """The floor names these molecules, and the PIN tier is not asked: no formal positive charge sits on a nitrogen."""
    asks, row = _asks_of_the_pin_tier(monkeypatch, smiles, tier)
    assert asks == 0, row


@pytest.mark.parametrize("smiles,expected", [
    ("C[NH+](C)C", True),                              # an azanium
    ("c1cc[n+](C)cc1", True),                          # a pyridinium
    ("C[N+](C)(C)CC(=O)[O-]", True),                   # a betaine: the carboxylate oxygen is not bonded to the nitrogen
    ("[O-][N+](=O)c1ccccc1", False),                   # a nitro group
    ("C[N+]([O-])(C)C", False),                        # an N-oxide
    ("CN=[N+]=[N-]", False),                           # an azide
    ("CCO", False),
])
def test_only_a_nitrogen_cation_asks_the_pin_tier(smiles, expected):
    """ (cationic centres): a charge-separated nitro, N-oxide or azide drawing is no cation, so the final-floor
    ask (valid and complete only) is not made for it."""
    from rdkit import Chem

    from orthonym.assembly.pin_first import has_nitrogen_cation
    assert has_nitrogen_cation(Chem.MolFromSmiles(smiles)) is expected

