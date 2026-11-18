"""Fused parents that OPSIN 2.9.0 and OPSIN main number differently are not used (data.opsin_misnumbered_parents).

A name must describe the input when OPSIN's main branch reads it too. Two causes, measured 2026-11-18 against
OPSIN main 760d57c0f132:

- OPSIN main lays out some fused ring systems otherwise than 2.9.0 does (its commits 093a195c and 75ff41fe), so a
  parent numbered by 2.9.0's reading ('benzo[4,5]cyclodeca[1,2-b]oxirene') puts its locants on other atoms when
  main reads the name. While the OPSIN in use is not a main build, such a parent is not used.
- OPSIN main reads locant 8 of pyrrolizine as the fusion carbon 7a (its PR #316). "Numbering of bridge
  atoms" (the Blue Book: "Bridge atoms are numbered continuing from the highest locant of the fused ring
  system") gives the first bridge atom of '1,5-ethanopyrrolizine' locant 8, so OPSIN main reads such a name as
  another structure. A bridged pyrrolizine is not used with any OPSIN.

The molecule then takes its von Baeyer name, below the PIN, as it does under OPSIN main."""
import pytest

from orthonym import Orthonym, jars
from orthonym.cli import _emit_tier_flags
from orthonym.data.opsin_misnumbered_parents import BRIDGE_LOCANT_ALIASED_PARENTS, LAYOUT_MISNUMBERED_PARENTS
from orthonym.jvm_budget import jvm_slots
from orthonym.rules.bridged_fused_pin import parents
from tests.support.opsin_marks import release_opsin_in_use
from tests.support.rt_assert import name_is_rt_exact

#: the tiers below are the shipped ones: OPSIN's validity gate on (tests/conftest.py)
pytestmark = pytest.mark.opsin_gate

#: pubchem1M:9857358, '(9R)-9-(3,4-dichlorophenyl)tetrahydro-1H-1,5-ethanopyrrolizin-2(3H)-one' before
PYRROLIZINE_BRIDGED = "C1CC2C3[C@@H](CC1N2CC3=O)C4=CC(=C(C=C4)Cl)Cl"
#: CHEBI:206340 (chebifull:86812), '(1aR,11R,11aR)-7-hydroxy-10-methoxy-5-methyl-3,6,11,11a-tetrahydro-2H-
#: 11,1a-(epoxymethano)benzo[4,5]cyclodeca[1,2-b]oxiren-13-one' before
LAYOUT_ROW = "COc1ccc(O)c2c1[C@H]1OC(=O)[C@]3(CCC=C(C)C2)O[C@H]13"


def _active(monkeypatch, label):
    monkeypatch.setattr(jars, "active_opsin", lambda download=True: {"label": label, "commit": None, "path": "x"})


def test_layout_parent_is_withheld_unless_opsin_main(monkeypatch):
    name = "benzo[4,5]cyclodeca[1,2-b]oxirene"
    assert name in LAYOUT_MISNUMBERED_PARENTS
    for label, withheld in (("2.9.0", True), ("custom", True), ("main", False)):
        _active(monkeypatch, label)
        assert parents._opsin_misnumbers(name) is withheld, label


def test_unlisted_parent_is_kept(monkeypatch):
    _active(monkeypatch, "2.9.0")
    assert not parents._opsin_misnumbers("naphthalene")
    assert not parents._opsin_misnumbers("pyrrolizine")        # withheld only as the parent of a bridged name


def _row(smiles, tier):
    with jvm_slots(1, purpose="opsin-misnumbered-parents"):
        return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


def test_bridged_pyrrolizine_takes_its_von_baeyer_name():
    assert "pyrrolizine" in BRIDGE_LOCANT_ALIASED_PARENTS
    row = _row(PYRROLIZINE_BRIDGED, "best-effort")
    assert (row["name"], row["tier"]) == (
        "(3R)-3-(3,4-dichlorophenyl)-7-azatricyclo[5.3.0.0^4,8]decan-5-one", "systematic_verified")
    assert name_is_rt_exact(row["name"], PYRROLIZINE_BRIDGED)
    assert "pyrrolizin" not in (_row(PYRROLIZINE_BRIDGED, "pin").get("name") or "")


@pytest.mark.skipif(not release_opsin_in_use(), reason="OPSIN main numbers this parent as the engine then does")
def test_layout_parent_row_takes_its_von_baeyer_name_on_opsin_2_9_0():
    row = _row(LAYOUT_ROW, "best-effort")
    assert (row["name"], row["tier"]) == (
        "(1R,13R,14R)-8-hydroxy-11-methoxy-5-methyl-16-oxo-15,17-dioxatetracyclo[11.2.2.0^1,14.0^7,12]heptadeca-"
        "4,7,9,11-tetraene", "systematic_verified")
    assert name_is_rt_exact(row["name"], LAYOUT_ROW)
    assert "cyclodeca" not in (_row(LAYOUT_ROW, "pin").get("name") or "")


def test_plain_pyrrolizine_keeps_its_pin():
    row = _row("C1CC2=CC=CN2C1", "pin")
    assert (row["name"], row["tier"]) == ("2,3-dihydro-1H-pyrrolizine", "pin_verified")


@pytest.mark.parametrize("smiles,name", [
    ("C1C2=CC=CN2C3=C1SC=C3", "8H-thieno[2,3-b]pyrrolizine"),
    ("C1=CC=C2C(=C1)C(=C3C(=N2)C4=CC=CN4C3=O)OC5=CC=C(C=C5)Cl", "9-(4-chlorophenoxy)-10H-pyrrolizino[1,2-b]quinolin-10-one"),
])
def test_pyrrolizine_as_a_fusion_component_keeps_its_pin(smiles, name):
    """A fused system numbers its own atoms, so main's alias on pyrrolizine's C7a is not reached;
    (the Blue Book) keeps the retained component name ('pyrrolizino', not 'pyrrolo[1',2':1,2]pyrrolo')."""
    row = _row(smiles, "pin")
    assert (row["name"], row["tier"]) == (name, "pin_verified")


@pytest.mark.skipif(not release_opsin_in_use(), reason="OPSIN main names the parent dibenzo[b,i][1]oxacycloundecine")
def test_withheld_bare_parent_takes_the_bracketed_benzo_name():
    """With 'dibenzo[b,i][1]oxacycloundecine' withheld, the bare parent is named by benzo annulation; the base
    component's heteroatom locant takes brackets after the prefix, the Blue Book:
    'benzo[c][2]benzopyran')."""
    smiles = "C1=CC=CC2=C1OC=C1C(=CC=CC=C2)C=CC=C1"
    row = _row(smiles, "best-effort")
    assert (row["name"], row["tier"]) == ("benzo[c][2]benzoxacycloundecine", "systematic_verified")
    assert name_is_rt_exact(row["name"], smiles)
