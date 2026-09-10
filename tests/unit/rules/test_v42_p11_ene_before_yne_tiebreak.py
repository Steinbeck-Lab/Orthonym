""" a phase (11B5) — ene-before-yne locant tie-break (cross-chapter).

ROOT-CAUSE numbering fix, one criterion tier added to two ring numberers so the
whole class is fixed rather than any single molecule:

When a ring's double bonds (-ene) and triple bonds (-yne) share the *same lowest
COMBINED multiple-bond locant SET*, the remaining choice -- which bonds get the
lower locants -- is given to the DOUBLE bonds. So `cycloicos-1-en-3-yne`, NOT
`cycloicos-3-en-1-yne`; `bicyclo[11.3.1]heptadec-2-en-11-yne`, NOT
`...-11-en-2-yne`.

Governing rule: the general unsaturation ordering the Blue Book states at
P-31.1.4.2.4 (assign low locants first to the combined ene+yne set, then to the
double bonds) is inherited by ring / von-Baeyer parents through P-31.1.4.3.4.
The acyclic chain numberer already implements it as `orient_chain` criteria
(b) [combined set] -> (c) [double bonds specifically]; this task ports the same
(b)->(c) pair into the two cyclic numberers that were missing criterion (c):

  * `assembly/general_engine.py::_orient_carbocycle` (all-carbon monocycle)
  * `rules/bicyclo.py::get_bicyclo_numbering` / `_key_lists` (von Baeyer)

The new tier is a PURE tie-break: it only decides orientations that already tied
on the combined ene+yne set, and is provably inert on rings with no triple bond
(then the doubles-only set equals the combined set for every orientation).

Every emission below is named in the bb_conformance engine config
(general_fallback + unverified + aromatic_general -- the tier that reaches the
all-carbon monocycle path) in a FRESH subprocess, and OPSIN-round-trip-gated on
the full InChIKey (0-wrong by construction).
"""
import json
import subprocess
import sys

_THIS = __file__


def _name_one(smiles):
    """Name ONE smiles in a FRESH subprocess (no warm cache) via the
    bb_conformance engine config; return {name, rt} where rt is the
    full-InChIKey OPSIN round-trip verdict."""
    out = subprocess.run(
        [sys.executable, _THIS, smiles],
        capture_output=True, text=True,
    )
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("{")]
    assert line, f"no JSON from subprocess: {out.stdout!r} / {out.stderr[-800:]!r}"
    return json.loads(line[-1])


# --- Targets: combined ene+yne set ties -> ene wins the low member (P-31.1.4.2.4)


def test_cycloicos_1_en_3_yne_monocycle():
    """{1,3} either way; the ene must take 1. def 44.4.1.2."""
    res = _name_one("C1#CCCCCCCCCCCCCCCCCC=C1")
    assert res.get("name") == "cycloicos-1-en-3-yne", res
    assert res.get("rt") is True


def test_cycloicosa_1_3_dien_5_yne_monocycle():
    """Two enes + one yne, combined set {1,3,5} either way -> enes take {1,3}.
    def 44.4.1.10.1."""
    res = _name_one("C1#CCCCCCCCCCCCCCCC=CC=C1")
    assert res.get("name") == "cycloicosa-1,3-dien-5-yne", res
    assert res.get("rt") is True


def test_cyclopentadec_1_en_4_yne_monocycle():
    """{1,4} either way -> ene takes 1. def 16.7.1."""
    res = _name_one("C1#CCCCCCCCCCCC=CC1")
    assert res.get("name") == "cyclopentadec-1-en-4-yne", res
    assert res.get("rt") is True


def test_cyclododeca_pentaen_11_yne_mancude_monocycle():
    """Mancude ring: five enes {1,3,5,7,9} + yne at 11 (combined {1,3,5,7,9,11}
    either way -> enes take the low members). def 31.2.4.1."""
    res = _name_one("C1#CC=CC=CC=CC=CC=C1")
    assert res.get("name") == "cyclododeca-1,3,5,7,9-pentaen-11-yne", res
    assert res.get("rt") is True


def test_bicyclo_11_3_1_heptadec_2_en_11_yne_von_baeyer():
    """von Baeyer: combined set {2,11} either way -> the ene takes 2, the yne 11
    (P-31.1.4.3.4 inherits the ene-before-yne tie-break). def 31.1.4.3."""
    res = _name_one("C1#CC2CCCC(C=CCCCCCCC1)C2")
    assert res.get("name") == "bicyclo[11.3.1]heptadec-2-en-11-yne", res
    assert res.get("rt") is True


# --- Regression pins: ene already has the low locant naturally -> the new tier
# is inert and the name MUST stay byte-identical.


def test_regression_bicyclo_14_3_1_trien_2_yne_von_baeyer():
    """von-Baeyer pin sharing get_bicyclo_numbering with the target: the enes
    {11,13,18} already sit below the yne at 2 by the COMBINED-set tier (2<11 for
    the yne, but the ene set as a whole is lower), so the doubles-only tier never
    fires. Must stay byte-identical. def 31.1.4.3."""
    res = _name_one("C1#CC2C=CCC(CC=CC=CCCCCCCC1)C2")
    assert res.get("name") == "bicyclo[14.3.1]icosa-11,13,18-trien-2-yne", res
    assert res.get("rt") is True


def test_regression_cycloicosa_1_7_dien_3_yne_monocycle():
    """Monocycle pin sharing _orient_carbocycle with the targets: the ene set
    {1,7} is strictly lower than any numbering giving the yne 1, so the combined
    tier already decides and the doubles-only tier is inert. Must stay
    byte-identical. def 44.4.1.10.1."""
    res = _name_one("C1#CCCC=CCCCCCCCCCCCCC=C1")
    assert res.get("name") == "cycloicosa-1,7-dien-3-yne", res
    assert res.get("rt") is True


def test_regression_pent_3_en_1_yne_acyclic():
    """Acyclic reference (orient_chain criteria (b)->(c)) the cyclic fix mirrors;
    must be unaffected. P-31.1.4.2.4 verbatim direction."""
    res = _name_one("C#CC=CC")
    assert res.get("name") == "pent-3-en-1-yne", res
    assert res.get("rt") is True


# ---------------------------------------------------------------------------
# subprocess entrypoint (fresh process per witness)
# ---------------------------------------------------------------------------
def _run(smiles):
    from orthonym.jvm_budget import jvm_slots
    with jvm_slots(1, purpose="v42-p11-eneyne-test"):
        from orthonym import Orthonym
        from orthonym.validation import opsin_roundtrip_check
        namer = Orthonym(general_fallback=True, general_fallback_unverified=True,
                          allow_aromatic_general=True)
        name = namer.name_tiered(smiles).get("name")
        rt = None
        if name and "unknown" not in name:
            rt = bool(opsin_roundtrip_check(smiles, name)["passed"])
        return {"name": name, "rt": rt}


if __name__ == "__main__":
    print(json.dumps(_run(sys.argv[1])))
