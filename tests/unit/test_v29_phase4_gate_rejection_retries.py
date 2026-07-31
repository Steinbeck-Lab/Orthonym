"""v29 Phase 4 — a GATE REJECTION re-enters the dispatch cascade.

`_name_impl`'s cascade already falls through when a handler returns ``None``. But
the OPSIN/SELF-01 gates run out in ``name()``, AFTER the cascade has exited, so a
handler that produced a name the gate then rejected aborted the whole molecule with
no retry. Phase 4 closes that: "keep the first that clears both gates, fall through
instead of aborting the molecule."

★ THE ROADMAP NAMED THE WRONG MECHANISM, and this test file exists partly to record
it. Phase 4's text describes filtering over ``CandidatePool``. Measured 2026-07-31:
the pool holds **one** candidate in **158 of 158** spied ``best()`` calls over 120
dev500 rows, and ``_best_two_tier`` records zero. The pool is not the generator of
alternatives — the DISPATCH CASCADE is, and Phase 4 works with it directly.

SAFETY ARGUMENT (why this cannot ship a wrong name)
---------------------------------------------------
The retry runs *only* when the gate replaced a real name with the descriptive
fallback, and returns that same fallback unless a later class produces a name that
PASSES the same gate. So it can only ever convert an ABSTENTION into a
gate-cleared name; a name that already shipped is never altered. 0-wrong is
preserved by construction.
"""

import pytest

from orthonym import Orthonym
from orthonym.errors import _DESCRIPTIVE_FALLBACK_NAMES


@pytest.fixture(autouse=True)
def _reenable_the_validity_gate(monkeypatch):
    """⚠ REQUIRED. ``tests/conftest.py`` has an autouse fixture that disables the
    OPSIN validity gate suite-wide (it asserts raw output and will not pay a
    per-name OPSIN call). Phase 4 is *defined* by what happens on a gate
    rejection, so with the gate off every test here is green-but-blind — and
    worse, ``CC(=O)N(CC1CO1)C(C)C`` then SHIPS `(5-carbamoylpentyl)oxirane`, a
    different molecule, because nothing suppresses it. Re-enable it here.
    """
    import orthonym.namer as _namer
    monkeypatch.setattr(_namer, "_DISABLE_VALIDITY_GATE", False, raising=False)
    yield

# A ceramide that abstained before Phase 4: its first-matching class produced a
# name the gate rejected, and the molecule died there. A later class names it
# correctly.
CERAMIDE = ("CCCCCCCCCCCCCCCCCCCC[C@@H](O)C(=O)N[C@@H](CO)"
            "[C@H](O)[C@H](O)CCCCCCCCCCCCCC")
CERAMIDE_NAME = "(2R)-2-hydroxy-N-(1,3,4-trihydroxyoctadecan-2-yl)docosanamide"


def test_a_gate_rejection_now_falls_through_to_a_later_class():
    assert Orthonym(style="pin").name(CERAMIDE) == CERAMIDE_NAME


@pytest.mark.parametrize("smiles,expected", [
    ("CCO", "ethanol"),
    ("c1ccccc1", "benzene"),
    ("BrCc1cccc(CBr)n1", "2,6-bis(bromomethyl)pyridine"),
])
def test_names_that_already_passed_are_untouched(smiles, expected):
    """The retry is unreachable for a name that clears the gate first time."""
    assert Orthonym(style="pin").name(smiles) == expected


def test_it_still_abstains_when_no_class_can_clear_the_gate():
    """The retry must not manufacture a name. This molecule's every candidate is
    gate-rejected (its first is `(5-carbamoylpentyl)oxirane`, a DIFFERENT
    molecule), so the fallback must survive the retry unchanged."""
    assert Orthonym(style="pin").name("CC(=O)N(CC1CO1)C(C)C") \
        in _DESCRIPTIVE_FALLBACK_NAMES


class TestNoOrderDependence:
    """The exclusion set is per-molecule scratch state on the instance, which is
    exactly the shape that caused a real cross-molecule contamination bug before
    (the stereo/confidence thread-locals — see the reset block in `name()`). These
    three properties are the ones that would break if it leaked."""

    def test_idempotent_on_one_instance(self):
        o = Orthonym(style="pin")
        assert o.name(CERAMIDE) == o.name(CERAMIDE) == o.name(CERAMIDE)

    def test_unaffected_by_what_was_named_before(self):
        fresh = Orthonym(style="pin").name(CERAMIDE)
        used = Orthonym(style="pin")
        for s in ("CC(=O)N(CC1CO1)C(C)C", "CCO", "c1ccccc1"):
            used.name(s)
        assert used.name(CERAMIDE) == fresh

    def test_does_not_poison_the_next_molecule(self):
        others = ["CC(=O)N(CC1CO1)C(C)C", "CCO", "c1ccccc1"]
        poisoned = Orthonym(style="pin")
        poisoned.name(CERAMIDE)          # exercises the retry, excluding classes
        after = [poisoned.name(s) for s in others]
        clean = Orthonym(style="pin")
        assert after == [clean.name(s) for s in others]
