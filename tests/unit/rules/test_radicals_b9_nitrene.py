"""Nitrenes R-N:,.

'(CH3)2N-N: dimethylhydrazinylidene (PIN)' (the Blue Book) is built at the
PIN tier. The '-aminylidene'/'-amidylidene' PINs ('benzenaminylidene (PIN)':40570,
'acetamidylidene (PIN)':40649) are not OPSIN-parseable, so the PIN tier gives no
name and the best-effort tier ships '<R>azanylidene' with a non-PIN label. Every
shipped name passes a strict OPSIN -r round trip.
"""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots
from tests.support.read_back import same_canonical_smiles


def _name(smiles, tier):
    with jvm_slots(1, purpose="test-radical-b9"):
        return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


@pytest.mark.opsin_gate
def test_hydrazinylidene_pin():
    row = _name("CN(C)[N]", "pin")
    assert row["name"] == "dimethylhydrazinylidene" and row["tier"] == "pin_verified"
    assert same_canonical_smiles(row["name"], "CN(C)[N]")


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,be_name", [
    ("[N]c1ccccc1", "phenylazanylidene"),
    ("CC(=O)[N]", "acetylazanylidene"),
])
def test_unparseable_pin_nitrenes(smiles, be_name):
    assert _name(smiles, "pin")["tier"] == "abstain"
    be = _name(smiles, "best-effort")
    assert be["name"] == be_name and be["tier"] not in ("abstain", "pin_verified")
    assert same_canonical_smiles(be_name, smiles)
