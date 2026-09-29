"""Breadth -- ZINC loss (2026-09-28): a failure rescue that exhausts the molecule's hang
budget declines; the main exit, with its round-trip-gated floor offer, still runs.

The row 'COC(=O)c1ccc2[nH]c(=N[C@H]3C[C@@H](NC(=O)[C@H]4CCOC4)C3)sc2c1' was
named round-trip exact at the best-effort tier up to 171c54d5e (the last-resort floor,
source 't4_floor') and abstained from 818c36360 on. Measured: its naming charges the
per-molecule analysis-call budget (``fragment_naming._ANALYSIS_CALL_BUDGET``, 500) with
410 calls at 171c54d5e and 518-520 from 818c36360 on, where ``rules.amides.name_amide``
declines the oxolane ring it used to spell as cyclopentane (a 0-wrong decline that stays;
with only that decline reverted in a scratch copy the count is 412) and the rescues do
more work. The call that crosses 500
sits inside the clean general fall-through, one of the optional rescues ``name`` runs
once its main path has produced a failure. ``PerfBudgetExceeded`` then unwound to the
``_budget_scope`` boundary, which abstains with the floor offer suppressed, so the floor
name the main exit offers every other failure was never computed. With the budget
raised to 600 the same code names the molecule; with the PIN tier's re-run switched off
it still abstains (the re-run is not the cause).

Now a rescue that exhausts the budget declines (``namer._FailureRescues``), the rescues
after it are skipped, and ``name`` leaves through its main exit on a fresh budget,
after the bounded rescue the boundary itself offers. A trip in the MAIN path (the
hang class of ``tests/unit/rules/test_m25_workbudget.py``) still reaches the boundary.

Every name is read back by a fresh OPSIN call that does not go through the engine
(``tests.support.rt_assert``), full InChIKey. ZINC / dev inputs only, no a holdout split row.
"""
import sys

import pytest

from orthonym import Orthonym
from orthonym import decomposition as decomposition_module
from orthonym import namer as namer_module
from orthonym.assembly import fragment_naming as fn
from orthonym.cli import _emit_tier_flags
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.integration, pytest.mark.opsin_gate]

WITNESS = "COC(=O)c1ccc2[nH]c(=N[C@H]3C[C@@H](NC(=O)[C@H]4CCOC4)C3)sc2c1"
# The floor name 171c54d5e shipped (best_effort, t4_floor); OPSIN full-InChIKey exact.
WITNESS_NAME = (
    "8-[1-(cis-3-{2-[(1S)-3-oxacyclopentan-1-yl]-3-oxa-1-azaprop-2-en-1-yl}"
    "cyclobutan-1-yl)-1-azamethan-1-ylidene]-4-(1-oxo-2-oxapropan-1-yl)-7-thia-"
    "9-azabicyclo[4.3.0]nona-1(6),2,4-triene")


def _be(smiles):
    return Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(smiles)


def test_zinc_witness_is_named_again_at_best_effort():
    row = _be(WITNESS)
    assert row["name"] == WITNESS_NAME, row
    assert row["tier"] == "best_effort" and row["source"] == "t4_floor", row
    assert_full_rt(row["name"], WITNESS)


# The class: ANY failure rescue of the outermost frame that runs out of the budget. Each
# one is made to exhaust it where ``name`` calls it as a rescue, in the frame that owns
# the budget (a nested frame, and a call from the main path, run the real function), and
# the molecule must still leave through the main exit.
_RESCUES = [
    (namer_module.Orthonym, "_try_general_engine_recovery"),
    (namer_module.Orthonym, "_try_demote_senior_group_rescue"),
    (decomposition_module, "try_decompose"),
    (namer_module.Orthonym, "_try_alternate_parent_rescue"),
    (namer_module.Orthonym, "_try_besteffort_clean_general_fallthrough"),
]


@pytest.mark.parametrize("owner,attr", _RESCUES, ids=[a for _o, a in _RESCUES])
def test_a_rescue_that_exhausts_the_budget_declines(monkeypatch, owner, attr):
    real = getattr(owner, attr)
    tripped = []

    def _exhausts(*args, **kwargs):
        caller = sys._getframe(1)
        if caller.f_code.co_name == "run":        # through _FailureRescues.run
            caller = caller.f_back
        if (fn.name_scope_depth() == 1 and caller.f_code.co_name == "name"
                and caller.f_code.co_filename.endswith("namer.py")):
            tripped.append(attr)
            raise fn.PerfBudgetExceeded()
        return real(*args, **kwargs)

    # The exit runs on a FRESH budget (a re-explosion there must still stop): record
    # the hang budgets the floor offer of ``_finish`` sees.
    real_floor = namer_module.Orthonym._maybe_append_t4_floor_offer
    floor_budgets = []

    def _floor_spy(self, *args, **kwargs):
        g = fn._fragment_guard
        floor_budgets.append((getattr(g, "perf_budget", None),
                              getattr(g, "analysis_budget", None)))
        return real_floor(self, *args, **kwargs)

    monkeypatch.setattr(owner, attr, _exhausts)
    monkeypatch.setattr(namer_module.Orthonym, "_maybe_append_t4_floor_offer", _floor_spy)
    row = _be(WITNESS)
    assert tripped, f"{attr} never ran in the outermost frame -- the test is vacuous"
    assert row["name"] == WITNESS_NAME, row
    assert row["tier"] == "best_effort" and row["source"] == "t4_floor", row
    assert_full_rt(row["name"], WITNESS)
    assert floor_budgets, "the floor offer never ran"
    assert all(p and a for p, a in floor_budgets), floor_budgets


# ---------------------------------------------------------------------------
# The helper's contract (no naming, no JVM)
# ---------------------------------------------------------------------------

def _raise():
    raise fn.PerfBudgetExceeded()


def test_the_budget_owner_declines_and_skips_later_rescues(monkeypatch):
    monkeypatch.setattr(fn._fragment_guard, "name_call_depth", 1, raising=False)
    rescues = namer_module._FailureRescues()
    assert rescues.run(lambda: "a name") == "a name" and not rescues.exhausted
    assert rescues.run(_raise, fallback="kept") == "kept"
    assert rescues.exhausted
    later = []
    assert rescues.run(lambda: later.append(1) or "late") is None
    assert later == [], "a rescue after the trip must not run"


def test_a_nested_frame_re_raises(monkeypatch):
    monkeypatch.setattr(fn._fragment_guard, "name_call_depth", 2, raising=False)
    rescues = namer_module._FailureRescues()
    with pytest.raises(fn.PerfBudgetExceeded):
        rescues.run(_raise)
    assert not rescues.exhausted
