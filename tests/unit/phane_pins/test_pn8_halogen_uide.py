import pytest

from ._helpers import default_tier_row, needs_phane_opsin

pytestmark = [pytest.mark.opsin_gate, needs_phane_opsin]


def test_the_name_round_trips_with_the_main_jar():
    # "Anions formed by addition of hydride ions" (the Blue Book), row:41110:
    # diphenyliodanuide (PIN); OPSIN main reads it, so no twin spelling is needed
    row = default_tier_row("[I-](c1ccccc1)c1ccccc1")
    assert row["name"] == "diphenyliodanuide" and row["tier"] == "pin_verified", row
