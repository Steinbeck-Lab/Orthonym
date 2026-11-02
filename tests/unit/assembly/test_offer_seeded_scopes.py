"""The default tier's offers start with the memo entries of the naming that left the molecule without a name.

An offer (``default_tier_offer.default_tier_row_for``) names the molecule again with the breadth flags of its own
engine. It ran in fresh memo scopes and recomputed what the naming had just computed: 7.5% of the CPU of the panel of
400 typical PubChem rows and 10.7% of the 60 slowest. Each offer scope now starts with a copy of the naming's entries
(``memo.scope_entries``; ``spelling_offer.name_tiered_with_offer`` for ``name_tiered``, ``default_tier_name_scope``
for ``name``). Every key carries what its value depends on, so the row is the row of a fresh offer, and a copy keeps
the offer's writes out of the naming's scopes.
"""
import pytest

from orthonym import Orthonym
from orthonym.assembly import default_tier_offer as dto
from orthonym.assembly import memo
from orthonym.cli import _emit_tier_flags

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

# named by the offer of the best-effort tier (the naming itself abstains)
OFFERED = ["C(C([Si])Cl)[Si]", "[B]C1CCCC(C1)[B]", "CCCC[Si][Si][Si]C"]


def _best_effort():
    return Orthonym(style="pin", **_emit_tier_flags("best-effort"))


def test_scope_entries_copies_every_entry_and_side_record():
    a = {("ns1", 1): "x", memo._SIDE_KEY: {("ns1", 1): "s"}}
    b = {("ns2", 2): "y"}
    out = memo.scope_entries([a, b])
    assert out[("ns1", 1)] == "x" and out[("ns2", 2)] == "y" and out[memo._SIDE_KEY] == {("ns1", 1): "s"}
    assert memo.scope_entries([]) == {memo._SIDE_KEY: {}}


@pytest.mark.parametrize("smiles", OFFERED)
def test_the_offer_starts_with_the_namings_entries_and_gives_the_fresh_row(smiles, monkeypatch):
    real = dto.default_tier_row_for
    seen = {}

    def spy(namer, s):
        seen["seed"] = memo._seed_var.get()
        return real(namer, s)
    monkeypatch.setattr(dto, "default_tier_row_for", spy)
    row = _best_effort().name_tiered(smiles)
    assert seen["seed"] and len(seen["seed"]) > 1, "the offer ran without the naming's entries"
    monkeypatch.setattr(memo, "scope_entries", lambda scopes: {memo._SIDE_KEY: {}})
    fresh = _best_effort().name_tiered(smiles)
    assert (row["name"], row["tier"]) == (fresh["name"], fresh["tier"]) and row["name"]
