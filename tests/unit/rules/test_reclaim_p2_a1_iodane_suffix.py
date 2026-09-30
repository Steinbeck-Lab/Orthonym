"""Reclaim P2/A1 — λ-halogen substituent suffix ``iodyl`` → ``iodanyl``.

A NEUTRAL hypervalent-iodine(III/V) SUBSTITUENT (real iodine(III) reagent chemistry)
builds a CORRECT structure but the substituent suffix was misspelled: the parent
``λ3-iodane`` formed its ``-yl`` by stripping the whole ``ane`` (``iodyl``, which OPSIN
rejects → the molecule abstained) instead of eliding only the final ``e``
(``iodanyl``,.

Root cause (Task-0 producer trace, 2026-09-14): the producer is
``assembly/substituent_naming.py::parent_to_prefix``, via TWO branches —
(1) the generic ``if name.endswith('ane'): stem = name[:-3]`` fallback, and
(2) the ``if 'cyclo' in name and name.endswith('ane')`` branch, which false-matched
``cyclopentyl(methyl)-λ3-iodane`` (the ``cyclo`` is in a substituent, the parent is
``iodane``). Both now route through the shared ``_HYDRIDE_ELIDE_E_STEMS`` elide-only-``e``
rule. The plan's assumed *second* producer in ``rules/mononuclear_hydrides.py`` does NOT
exist; the only other ``iodyl`` emitter is ``charged_router._apply_radical_suffix``,
which fires solely on the CHARGED iodanuide (net −1) class that is out of scope here and
must keep abstaining (0-wrong canary).

Measured gated reclaim (RT-verified) on the 10M abstention census: 18 neutral rows
(the RT-fail residue is 74 whole-molecule locant/skeleton differences the gate correctly
abstains — NOT spelling; the plan's "~76–127" was the blind-string-rewrite ceiling).

P-rule: (a substituent ``-yl`` from a mononuclear hydride elides only the
final ``e``); λ-convention hypervalent iodine /.

Verification uses the NORMAL gated path (gate ON, jar present) in a FRESH subprocess per
molecule, and confirms full round-trip to the input InChIKey.
"""
import json
import subprocess
import sys

import pytest

# --- (a) NEUTRAL λ3/λ5-iodane substituent rows: reclaim + full-RT --------------
# From the 10M abstention census (charge 0, corrected name OPSIN-RTs to input).
# Mix of the ``C=I`` methylidene frame and the ``I(C)C`` direct-organyl frame.
RECLAIM_SMILES = [
    "CC1C2C1=NC=C(O2)I=C",                            # 164079807 methylidene-λ3-iodanyl
    "C=IC1=C(C2(CCNCC2)CN1C(=O)NCC3CCOCC3)PP",       # 143067024 methylidene-λ3-iodanyl
    "CC1=CC(=CI=C(CC1)C)C(F)(F)I=C",                  # 162593342 difluoro(methylidene-λ3-iodanyl)
    "CC1[C@@H](C=C([C@@H]2CC2C1N)F)I=C",             # 163447970 methylidene-λ3-iodanyl (stereo)
    "C=IC(OC1=CN=C(N=C1)OC2=CCNC(N=C2)N)I=C",        # 164019062 di(methylidene-λ3-iodanyl)
    "CC(C)C1(C=NC(C=NS1)(C)I(C)C)C",                  # 142900755 dimethyl-λ3-iodanyl (I(C)C frame)
    "CI(C)N1CCCC(C1)OC(=O)NC2=CC=CC=C2C3=CC=CC=C3",  # 140973462 dimethyl-λ3-iodanyl (I(C)C frame)
    "CC1CCCCC(CC1)(N)I(C)C2CCCC2",                    # 163540094 cyclopentyl(methyl)-λ3-iodanyl
]

# --- (b) the "pre-formed" token shapes the plan called frame #2 --------------
# ``dimethyl-/cyclopentyl(methyl)-λ3-iodanyl`` come from ``I(C)C`` neutral iodine, NOT
# from a separate pre-formed producer. The cyclopentyl(methyl) row specifically exercises
# the SECOND ``parent_to_prefix`` branch (the ``'cyclo' in name`` false-match) that the
# one-line elide-list edit alone did not cover — proving BOTH branches are fixed.
FRAME2_SHAPE_SMILES = [
    "CC(C)C1(C=NC(C=NS1)(C)I(C)C)C",                  # 142900755 dimethyl-λ3-iodanyl (generic-ane branch)
    "CC1CCCCC(CC1)(N)I(C)C2CCCC2",                    # 163540094 cyclopentyl(methyl)-λ3-iodanyl (cyclo branch)
]

# --- (c) 0-wrong canary: CHARGED iodanuide (net −1) stays ABSTAINED ----------
# The corrected ``…-iodanyl`` denotes a NEUTRAL iodine; the input is charged, so the
# RT/InChIKey gate rejects it → abstain. Plus a hypervalent-Cl input RDKit cannot even
# sanitize → abstains at perception. Adding chlorane/bromane/fluorane to the elide list
# must not turn any of these into an emission.
CANARY_ABSTAIN_SMILES = [
    "CN(CC1CC1)[I-]C2CC2CNI",                         # 164025616 iodanuide (q=-1)
    "C1CC(CC(C1)[I-]NC2=NC=CS2)NI",                   # 163765206 iodanuide (q=-1)
    "CC1([C@@H]2CCC=I[C@@H]2CC1[I-]C)C",              # 149413697 iodanuide (q=-1)
    "CC=ClC1=CC=CC=C1",                               # hypervalent Cl -> RDKit reject -> abstain
]

# --- (d) determinism rows -----------------------------------------------------
DETERMINISM_SMILES = [
    "CC1C2C1=NC=C(O2)I=C",
    "CC(C)C1(C=NC(C=NS1)(C)I(C)C)C",
    "CC1CCCCC(CC1)(N)I(C)C2CCCC2",
]


_CHILD = r"""
import json, sys
from rdkit import Chem
from rdkit.Chem import inchi
from orthonym import name_compound
from orthonym import errors
from orthonym.validation.opsin_roundtrip import opsin_parse

def ik(smi):
    try:
        m = Chem.MolFromSmiles(smi)
        return inchi.MolToInchiKey(m) if m else None
    except Exception:
        return None

smi = sys.argv[1]
try:
    name = name_compound(smi)
except Exception as e:
    name = None
emits = bool(name) and not errors.is_failure_name(name)
rt_ok = False
if emits:
    parsed = opsin_parse(name)
    rt_ok = bool(parsed) and (ik(parsed) == ik(smi))
# the default tier's decline code, the strict path's name (the emission rule off) and
# the best-effort name, each with its round trip
import orthonym.namer as nm
from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
limit = strict = be = None
try:
    limit = Orthonym().name_tiered(smi)["limit_code"]
    tok = nm._DEFAULT_TIER_POLICY_OFF.set(True)
    try:
        strict = Orthonym().name(smi)
    finally:
        nm._DEFAULT_TIER_POLICY_OFF.reset(tok)
    be = Orthonym(**_emit_tier_flags("best-effort")).name_tiered(smi)["name"]
except Exception:   # an input RDKit cannot read (the hypervalent Cl canary)
    pass
def rt(n):
    p = opsin_parse(n) if n and not errors.is_failure_name(n) else None
    return bool(p) and ik(p) == ik(smi)
print(json.dumps({"name": name, "emits": emits, "rt_ok": rt_ok, "limit": limit,
                  "strict": strict, "strict_rt": rt(strict), "be": be, "be_rt": rt(be)}))
"""

# Default tier: the paper, Methods, "Tiers" (L73): "The default configuration emits a
# name only when the pipeline can build the preferred IUPAC name (PIN); otherwise, it
# declines." User decision 2026-09-30 ("Ship it in 1.0.2"): this row's name carries a
# part the code records as not the PIN, so the default tier declines it
# (NO_VERIFIED_PIN); the strict path's name and the best-effort name carry the
# 'iodanyl' spelling and round-trip.
DEFAULT_TIER_DECLINES = frozenset({"CI(C)N1CCCC(C1)OC(=O)NC2=CC=CC=C2C3=CC=CC=C3"})


def _name_fresh(smi):
    """Name ``smi`` in a FRESH subprocess on the NORMAL gated path (gate ON)."""
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, smi],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, f"child failed for {smi!r}: {proc.stderr[-500:]}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.roundtrip
@pytest.mark.parametrize("smi", RECLAIM_SMILES)
def test_a1_neutral_iodane_substituent_reclaims_and_roundtrips(smi):
    r = _name_fresh(smi)
    if smi in DEFAULT_TIER_DECLINES:
        assert not r["emits"] and r["limit"] == "NO_VERIFIED_PIN", r
        assert "iodanyl" in r["be"] and r["be_rt"], r
        r = dict(r, name=r["strict"], emits=True, rt_ok=r["strict_rt"])
    assert r["emits"], f"expected an emission, got abstain: {r}"
    assert "iodanyl" in r["name"], f"expected 'iodanyl' spelling, got {r['name']!r}"
    assert "iodyl" not in r["name"].replace("iodanyl", ""), \
        f"unexpected 'iodyl' token remains: {r['name']!r}"
    assert r["rt_ok"], f"name did not round-trip to input InChIKey: {r}"


@pytest.mark.roundtrip
@pytest.mark.parametrize("smi", FRAME2_SHAPE_SMILES)
def test_a1_dimethyl_shape_also_reclaims(smi):
    # The plan feared these needed a separate producer; the single parent_to_prefix
    # fix reaches them (Task-0 trace). Proves both token shapes are covered.
    r = _name_fresh(smi)
    assert r["emits"] and r["rt_ok"], f"expected reclaim+RT, got {r}"
    assert "iodanyl" in r["name"], r["name"]


@pytest.mark.roundtrip
@pytest.mark.parametrize("smi", CANARY_ABSTAIN_SMILES)
def test_a1_charged_and_hypervalent_cl_stay_abstained(smi):
    # 0-wrong canary: charged iodanuide and RDKit-rejected hypervalent Cl must NOT
    # be turned into an emission by the elide-list change.
    r = _name_fresh(smi)
    assert not r["emits"], f"0-wrong canary regressed — expected abstain, got {r}"


@pytest.mark.roundtrip
@pytest.mark.parametrize("smi", DETERMINISM_SMILES)
def test_a1_determinism_across_fresh_subprocesses(smi):
    names = {_name_fresh(smi)["name"] for _ in range(3)}
    assert len(names) == 1, f"non-deterministic name for {smi!r}: {names}"
