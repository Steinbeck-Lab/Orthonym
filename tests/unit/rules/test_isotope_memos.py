"""The two memos of the isotope decorator (still-unnamed PubChem plan of 2026-11-11, Topic 7): a skeleton named again
in the same engine and state, and a candidate read back again for the same structure and reading state, are served
from the per-call memo. The names are the ones of the memo switched off."""
import pytest

import orthonym.assembly.memo as memo
from orthonym import Orthonym

pytestmark = pytest.mark.opsin_gate

ROWS = ["[2H]C([2H])([2H])Oc1ccccc1", "C[13CH2]O", "CC(=O)N[C@@H](C([2H])([2H])[2H])C(=O)O",
        "[2H]N([2H])c1ccccc1", "OC([2H])([2H])C([2H])([2H])O", "[3H]C([3H])([3H])N1CCC(CC1)C(=O)O"]


@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_memo_on_and_off_give_the_same_names(monkeypatch, tier):
    from orthonym.cli import _emit_tier_flags
    names = {}
    for mode in ("off", "on", "verify"):
        monkeypatch.setattr(memo, "_MODE", mode)
        namer = Orthonym(style="pin", **_emit_tier_flags(tier))
        names[mode] = [(namer.name_tiered(s)["name"], namer.name_tiered(s)["tier"]) for s in ROWS]
    assert names["on"] == names["off"] == names["verify"]
