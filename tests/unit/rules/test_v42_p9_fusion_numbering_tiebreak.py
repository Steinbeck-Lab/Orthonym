""" a phase — co-optimal fusion-numbering tie-breaks (child locant citation).

Two ROOT-CAUSE algorithmic fixes to fused-ring numbering, each fixing the whole
class rather than one molecule:

Fix 1 (2-ring path, ``generate_systematic_name_for_fused_pair``): when a child
component's senior heteroatom is EQUIDISTANT from the fusion bond (a "symmetric"
child) its two numberings tie on heteroatom + fusion-bond locants, and the code
used to break that tie arbitrarily. Now the co-optimal child numberings are
enumerated and the fusion descriptor lowest in citation order is chosen, per
 (the Blue Book, "the letter as early in the alphabet as
possible... these numbers are chosen to be as low as is consistent with the
numbering of the compound and their order conforms to the direction of lettering
of the parent component") + (d) (:13403/:13409, "the locant set '4,5'
is lower than '5,4'"). Fixes ``selenopheno[3,4-b]selenophene`` (was [4,3-b]).

Fix 2 (>=3-ring star path, ``_pcf_name_with_base``): the same symmetric-child tie
for one attachment of a polycomponent star. Fixes the second (primed) furan of
``difuro[3,2-b:3',4'-e]pyridine`` (was 4',3'-e). BB PIN verbatim at
 (:13467 heading /:13479 example).

Every emission below is OPSIN-round-trip-gated on the full InChIKey (0-wrong by
construction). Fresh process per naming probe (feedback_spy_before_you_refute).
Heteroatom-first component numbering: (a):12543 / (b):12558.
"""
import json
import subprocess
import sys

import pytest

_THIS = __file__


def _name_one(smiles):
    """Name ONE smiles in a FRESH subprocess via the public API; return
    {name, rt} where rt is the full-InChIKey OPSIN round-trip verdict."""
    out = subprocess.run(
        [sys.executable, _THIS, smiles],
        capture_output=True, text=True,
    )
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("{")]
    assert line, f"no JSON from subprocess: {out.stdout!r} / {out.stderr[-800:]!r}"
    return json.loads(line[-1])


# --- Fix 1: selenopheno[3,4-b] symmetric-child tie (the [4,3]->[3,4] bug) -----
def test_selenopheno_3_4_b_symmetric_child_pin():
    """meta-Se child is symmetric across the fusion bond -> tie broken to the
    lower cited pair (3,4)<(4,3). PIN at the Blue Book."""
    res = _name_one("c1cc2c[se]cc2[se]1")
    assert res.get("name") == "selenopheno[3,4-b]selenophene", (
        f"expected selenopheno[3,4-b]selenophene, got {res.get('name')!r}")
    assert res.get("rt") is True


# --- Fix 1 regression pins: decisive (asymmetric) children stay byte-identical
def test_selenopheno_3_2_b_regression_pin():
    """ortho-Se child is decisive (asymmetric) -> single numbering -> the
    citation is the lettering-forced descending pair. PIN at:11936."""
    res = _name_one("c1cc2[se]ccc2[se]1")
    assert res.get("name") == "selenopheno[3,2-b]selenophene", (
        f"expected selenopheno[3,2-b]selenophene, got {res.get('name')!r}")
    assert res.get("rt") is True


def test_selenopheno_2_3_b_regression_pin():
    """ortho-Se child, other side -> decisive -> [2,3-b]. PIN at:11921."""
    res = _name_one("c1cc2cc[se]c2[se]1")
    assert res.get("name") == "selenopheno[2,3-b]selenophene", (
        f"expected selenopheno[2,3-b]selenophene, got {res.get('name')!r}")
    assert res.get("rt") is True


def test_thieno_2_3_b_furan_regression_pin():
    """Single-O base + single-S child, both decisive -> unique numbering each ->
    the co-optimal product is a singleton and output is unchanged."""
    res = _name_one("c1cc2ccsc2o1")
    assert res.get("name") == "thieno[2,3-b]furan", (
        f"expected thieno[2,3-b]furan, got {res.get('name')!r}")
    assert res.get("rt") is True


# --- Fix 2: difuro symmetric second furan (the 4',3'->3',4' bug) --------------
def test_difuro_3_2_b_3p_4p_e_pyridine_pin():
    """Second furan is symmetric (O in the middle, meta to the fusion bond) ->
    tie broken to the lower cited pair (3,4)<(4,3) -> 3',4'-e. PIN verbatim at
     the Blue Book."""
    res = _name_one("c1cc2nc3cocc3cc2o1")
    assert res.get("name") == "difuro[3,2-b:3',4'-e]pyridine", (
        f"expected difuro[3,2-b:3',4'-e]pyridine, got {res.get('name')!r}")
    assert res.get("rt") is True


def test_difuro_3_2_b_2p_3p_e_pyridine_regression_pin():
    """Second furan is ASYMMETRIC (O adjacent to the fusion bond) -> single
    numbering -> citation unchanged at 2',3'-e. The definitive over-fire pin
    for Fix 2 (must STAY byte-identical). BB def 25.3.4.1.2."""
    res = _name_one("c1cc2nc3ccoc3cc2o1")
    assert res.get("name") == "difuro[3,2-b:2',3'-e]pyridine", (
        f"expected difuro[3,2-b:2',3'-e]pyridine, got {res.get('name')!r}")
    assert res.get("rt") is True


# ---------------------------------------------------------------------------
# subprocess entrypoint (fresh process per witness)
# ---------------------------------------------------------------------------
def _run(smiles):
    from orthonym.jvm_budget import jvm_slots
    with jvm_slots(1, purpose="v42-p9-fusnum-test"):
        from orthonym import name_compound
        from orthonym.validation import opsin_roundtrip_check
        name = name_compound(smiles)
        rt = None
        if name and "unknown" not in name:
            rt = bool(opsin_roundtrip_check(smiles, name)["passed"])
        return {"name": name, "rt": rt}


if __name__ == "__main__":
    print(json.dumps(_run(sys.argv[1])))
