"""The P/S oxoacid prefix is a wider-tier form: the default (PIN) tier names these rows as before.

`allow_mancude` is passed by PIN-path callers too (phosphorus.py:1110/1225/1352, purine.py:239),
so Tier 0.56 is gated on the wider-tier context and the book-spelling switch. Before the gate the
phosphonate esters below left the default tier as `pin_verified` names with an unenclosed compound
prefix after 'diethyl', the Blue Book: "Parentheses are used around compound...
prefixes"; 'methyl [(dimethoxyphosphoryl)oxy]phosphonate (PIN)',:36989)."""
import pytest

from tests.support.default_tier import default_tier_row
from tests.support.rt_assert import assert_full_rt, name_best_effort

pytestmark = pytest.mark.opsin_gate

ROWS = [
    "CCOP(=O)(CCP(=O)(OCC)OCC)OCC",
    "CCOP(=O)(OCC)C(=Cc1ccccc1)P(=O)(OCC)OCC",
    "CCOP(=O)(OCC)C(=Cc1cc(C(C)(C)C)c(O)c(C(C)(C)C)c1)P(=O)(OCC)OCC",
]


@pytest.mark.parametrize("smiles", ROWS)
def test_default_tier_does_not_name_the_row_as_a_pin(smiles):
    row = default_tier_row(smiles)
    assert row["tier"] == "abstain" and row["is_pin"] is False, row


@pytest.mark.parametrize("smiles", ROWS)
def test_best_effort_still_names_it_and_reads_back(smiles):
    row = name_best_effort(smiles)
    assert row["name"] and row["tier"] != "abstain", row
    assert_full_rt(row["name"], smiles)
