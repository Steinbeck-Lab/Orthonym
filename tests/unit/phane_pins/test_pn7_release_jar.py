"""With an OPSIN that reads no phane names the engine is as before: no phane name is produced and the legacy
phane PIN ships as a grammar-gap carve-out. Runs with OPSIN 2.9.0 and with main."""
import pytest

from orthonym import namer
from orthonym import opsin_capabilities as caps

from ._helpers import default_tier_row, name_best_effort

pytestmark = pytest.mark.opsin_gate

CYCLIC_NOT_LEGACY = "c1cc2cc(c1)CCC1CCCC(CC2)C1"            #:23857
DITHIA = "c1cc2c3cc(ccc3c1)CSCc1cccc3ccc(cc13)CSC2"         #:23895
LEGACY = "c1cc2cc(c1)CCc1cccc(c1)CC2"                       # [2.2]metacyclophane,:15021


@pytest.fixture(autouse=True)
def release_capabilities(monkeypatch):
    monkeypatch.setattr(caps, "opsin_capabilities", lambda: {"phane": False, "halogen_uide": False})


def test_names_stay_without_the_capability():
    assert default_tier_row(CYCLIC_NOT_LEGACY)["tier"] == "abstain"
    assert name_best_effort(CYCLIC_NOT_LEGACY)["name"] == "tricyclo[9.3.1.1^4,8]hexadeca-1(15),11,13-triene"
    assert name_best_effort(DITHIA)["name"].startswith("3,14-dithiapentacyclo[14.6.2.2^5,8.0^7,12.0^19,23]")


def test_the_carve_out_ships_a_phane_name_only_with_its_read_back():
    # no capability enters here: validation/phane_read_back builds the structure the name states, with either jar
    assert namer._grammar_gap_carveout("1,4(1,3)-dibenzenacyclohexaphane", LEGACY) == "phane"
    assert namer._grammar_gap_carveout("1,4(1,3)-dibenzenacyclohexaphane", "C1Cc2ccc(cc2)CCc2ccc1cc2") is None
    assert namer._grammar_gap_carveout("1,4(1,3)-dibenzenacyclohexaphane") is None


@pytest.mark.skipif(caps.phane_enabled(), reason="the label of the carve-out name is the release jar's")
def test_the_legacy_phane_pin_ships_as_a_carve_out():
    row = default_tier_row(LEGACY)
    assert row["name"] == "1,4(1,3)-dibenzenacyclohexaphane" and row["tier"] == "pin_unverified", row


def test_the_p10_classes_dispatch_before_the_phane_classes():
    # the spelling-check skip is kept (decision of the queue file): carbohydrates, natural products, peptides,
    # retained names and amino acids (priorities 1000-1400) name a structure before the phane classes
    from orthonym.routing.dispatch_table import DISPATCH_TABLE, StoutClass
    assert DISPATCH_TABLE[StoutClass.LINEAR_PHANE].priority == 1650
    assert DISPATCH_TABLE[StoutClass.CYCLOPHANE].priority == 1600
    for cls in (StoutClass.CARBOHYDRATE_LOOKUP, StoutClass.NATURAL_PRODUCT, StoutClass.PEPTIDE,
                StoutClass.RETAINED_NAME, StoutClass.AMINO_ACID):
        assert DISPATCH_TABLE[cls].priority < DISPATCH_TABLE[StoutClass.CYCLOPHANE].priority
