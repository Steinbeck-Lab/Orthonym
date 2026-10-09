""" a phase — a GATE REJECTION re-enters the dispatch cascade.

`_name_impl`'s cascade already falls through when a handler returns ``None``. But
the OPSIN/ gates run out in ``name``, AFTER the cascade has exited, so a
handler that produced a name the gate then rejected aborted the whole molecule with
no retry. a phase closes that: "keep the first that clears both gates, fall through
instead of aborting the molecule."

★ THE ROADMAP NAMED THE WRONG MECHANISM, and this test file exists partly to record
it. a phase's text describes filtering over ``CandidatePool``. Measured 2026-07-31:
the pool holds **one** candidate in **158 of 158** traced ``best`` calls over 120
a dev split rows, and ``_best_two_tier`` records zero. The pool is not the generator of
alternatives — the DISPATCH CASCADE is, and a phase works with it directly.

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


# ⚠ REQUIRED. ``tests/conftest.py`` disables the OPSIN validity gate suite-wide
# (most tests assert raw output and will not pay a per-name OPSIN call). a phase
# is *defined* by what happens on a gate rejection, so with the gate off every
# test here is green-but-blind — and worse, ``CC(=O)N(CC1CO1)C(C)C`` then SHIPS
# `(5-carbamoylpentyl)oxirane`, a different molecule, because nothing suppresses
# it. The marker re-enables the gate, and additionally SKIPS if the OPSIN jar is
# absent — without a jar the gate fails OPEN  and these tests would be
# blind a second way. See ``tests/unit/test_opsin_gate_test_harness.py``.
pytestmark = pytest.mark.opsin_gate

# A ceramide that abstained before a phase: its first-matching class produced a
# name the gate rejected, and the molecule died there. A later class names it
# correctly.
CERAMIDE = ("CCCCCCCCCCCCCCCCCCCC[C@@H](O)C(=O)N[C@@H](CO)"
            "[C@H](O)[C@H](O)CCCCCCCCCCCCCC")
# The N-substituent now carries its three stereodescriptors. The former expectation,
# '(2R)-2-hydroxy-N-(1,3,4-trihydroxyoctadecan-2-yl)docosanamide', omitted them: it matched the
# input only on the InChIKey skeleton, and a name that omits defined stereo is a mismatch
# since 0c4d4a2d3 ("reject stereo OMISSION as a mismatch"). OPSIN 2.9.0 parses this name to the
# input's full InChIKey.
CERAMIDE_NAME = "(2R)-2-hydroxy-N-[(2S,3S,4R)-1,3,4-trihydroxyoctadecan-2-yl]docosanamide"


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


# The former witness ``CC(=O)N(CC1CO1)C(C)C`` is named now ('N-(oxiranylmethyl)-N-(propan-2-yl)acetamide',
# OPSIN 2.9.0 full-InChIKey exact, so there is nothing for the gate to reject). This dicarbamate is a
# molecule whose first candidate is a DIFFERENT molecule that the gate rejects
# ('3-{1-[(carbamoyloxy)methyl]-1-octylcyclohexyl}-N,N-dimethylpropan-1-amine'); the retry then
# runs and no later class clears the gate.
NO_CLASS_CLEARS = "CN(C)CCCNC(=O)OCC1(CCCCC1)COC(=O)NC2CCCCC2"


def test_it_still_abstains_when_no_class_can_clear_the_gate(monkeypatch):
    """The retry must not manufacture a name. This molecule's candidates are gate-rejected (its
    first is a different molecule), so the fallback must survive the retry unchanged.

    The retry is traced on, so the test cannot go green-but-blind: it asserts that the retry RAN
    past a gate-rejected real name, excluded at least one dispatch class, and returned the
    fallback."""
    from orthonym.errors import is_failure_name

    seen = []
    real = Orthonym._retry_cascade_on_gate_rejection

    def spy(self, smiles, pre_gate, gated):
        out = real(self, smiles, pre_gate, gated)
        seen.append((pre_gate, gated, out, set(self._excluded_dispatch_classes)))
        return out

    monkeypatch.setattr(Orthonym, "_retry_cascade_on_gate_rejection", spy)
    assert Orthonym(style="pin").name(NO_CLASS_CLEARS) in _DESCRIPTIVE_FALLBACK_NAMES
    ran = [t for t in seen if t[0] and not is_failure_name(t[0]) and is_failure_name(t[1])]
    assert ran, "the gate never rejected a real name, so the retry was not exercised"
    assert all(is_failure_name(t[2]) for t in ran)
    assert ran[0][3], "the retry excluded no dispatch class"


class TestNoOrderDependence:
    """The exclusion set is per-molecule scratch state on the instance, which is
    exactly the shape that caused a real cross-molecule contamination bug before
    (the stereo/confidence thread-locals — see the reset block in `name`). These
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
