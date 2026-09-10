""".2 — HW / partial-saturation heteromonocycle component namer.

Sub-lever C (cross-ref SP2.4a). Two signatures for a partially-saturated
heteromonocycle spiro component:

  (1) an OPSIN-unparseable fully-mancude stem (``oxine``/``[1,3]thiazole``)
      emitted for a ring that is only PARTIALLY saturated;
  (2) the ring unsaturation / spiro locant scrambled so the built name denotes
      a DIFFERENT constitution (the thiazoline ``C=N`` case).

Root cause (VERIFIED CT.2 trace, HEAD 6ceb6a71): in the ``mixed-spiro-fused``
assembly path, ``_name_side_ring`` builds the component NAME via
``name_heterocycle`` (correct: ``4,5-dihydro-1,3-thiazole``) but takes the spiro
locant from an INDEPENDENT ``_walk_side_ring_locants`` walk that can disagree
with that numbering — citing the spiro atom at ``2'`` (a heteroatom-adjacent
carbon that cannot be the sp3 spiro junction) instead of ``5'``. The fix unifies
the side-ring locant map onto the SAME numbering authority the stem uses
(``_mancude_hydro_numbering``).

Contract: name RT-exact or fail closed (abstain) — NEVER a mis-saturated
constitution. 0-wrong is absolute.

Run ONLY this file (whole-suite deadlocks on an OPSIN pipe):
    .venv/bin/python -m pytest tests/unit/rules/test_ct2_hw_partial_saturation.py -q
"""
import signal
import pytest

from orthonym.jvm_budget import jvm_slots

pytestmark = pytest.mark.opsin_gate


@pytest.fixture(scope="module", autouse=True)
def _jvm_slot():
    with jvm_slots(1, purpose="test_ct2_hw_partial_saturation"):
        yield


class _Timeout(Exception):
    pass


def _alarm(seconds=60):
    def _raise(sig, frm):
        raise _Timeout()
    signal.signal(signal.SIGALRM, _raise)
    signal.alarm(seconds)


def _name(smiles):
    from orthonym import name_compound
    _alarm(60)
    try:
        return name_compound(smiles)
    finally:
        signal.alarm(0)


def _rt(smiles, name):
    from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check
    _alarm(60)
    try:
        return opsin_roundtrip_check(smiles, name)["passed"]
    finally:
        signal.alarm(0)


# The priority signature-2 witness: 2-(methylsulfanyl)-thiazoline spiro-fused to a
# 1-methoxy-2-oxindole. NO defined stereo (the spiro C is undefined in the input),
# so a constitution-correct name is RT-exact. Today it abstains because the built
# name scrambles the ring C=N -> N-CH2 (wrong constitution); the fix must NAME it.
THIAZOLINE = "CON1C(=O)C2(CN=C(SC)S2)c2ccccc21"


def test_thiazoline_partial_saturation_names_rt_exact():
    """Signature 2 (the fix target): the thiazoline spiro-oxindole must NAME
    (not abstain) and round-trip exactly. Preserving ``C=N`` and citing the
    spiro locant consistently with the ``4,5-dihydro-1,3-thiazole`` stem."""
    n = _name(THIAZOLINE)
    assert not n.startswith("unknown"), (
        f"still abstaining on the thiazoline partial-saturation component: {n!r}")
    assert _rt(THIAZOLINE, n), f"named but does not round-trip: {n!r}"


# 0-wrong family guard: every HW / partial-saturation heteromonocycle spiro
# component witness must ABSTAIN or ROUND-TRIP — never ship a mis-saturated
# constitution. (The oxine sig-1 witness stays a NAMED BLOCKER: even a correct
# stem leaves 6 undescribed stereocentres, so it abstains via the stereo backstop;
# that is 0-wrong, not a wrong-molecule ship. Residual = stereo, ST-series.)
@pytest.mark.parametrize("smiles", [
    THIAZOLINE,
    # oxine / partial-sat O-lactone HW-misclassification signature (stereo-blocked)
    "CC1=CC[C@]2(OC[C@]34CCC5=C(CC[C@@H]6C(=C5)C=CC(=O)OC6(C)C)[C@]3(C)CC[C@@H]4[C@@H]2C)OC1=O",
    # spiro-oxindole partial-saturation family
    "CCOC(=O)OC1=C(c2cc(C)ccc2C)C(=O)N[C@]12CC[C@@H](OC)CC2",
])
def test_hw_partial_saturation_zero_wrong(smiles):
    n = _name(smiles)
    assert n.startswith("unknown") or _rt(smiles, n), \
        f"shipped a non-round-tripping HW partial-saturation name: {n!r}"
