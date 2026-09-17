"""Phase-1 abstention-reclaim — general-engine stereo COUNT-VETO reclaim (0-wrong).

The general-engine stereo gate ``general_engine_stereo_complete(mol, name)``
(``rules/stereochemistry.py``) is a raw token-count identity
``count_expressed_stereo_descriptors(name) == count_defined_stereo_elements(mol)``.
Consumed by ``namer._stereo_emit_decision``, its ``(False, False)`` branch used to
TERMINALLY ABSTAIN every count-mismatch general-fallback candidate BEFORE it could
reach the already-built stereo-composition channel. a phase lets those candidates
be OFFERED (as-is, or with the input's stereo COMPOSED on) and verified by the existing
EXACT-ISOMERIC round-trip ``_rt_match(smiles, cand_smi, stereo_flagged=False)``.

Governing rule (heading + decisive sentence):
   "Recommended stereodescriptors" -> (the Blue Book):
  "In preferred IUPAC names, stereodescriptors, preceded by a locant, MUST BE cited to
  specify each stereogenic unit, as illustrated in." An exocyclic C=C (ylidene)
  is not one of the omission classes, so its descriptor is REQUIRED; dropping
  it is a defect that composition repairs, not a licit omission.

0-wrong ABSOLUTE: every reclaim ships UNFLAGGED so the caller's exact-isomeric
``_rt_match`` (CanonSmiles==CanonSmiles AND full-InChIKey) is the emission authority;
any genuine CONFLICT / InChI-blind under-composition abstains.

DETERMINISM: each SMILES is named in a FRESH subprocess (names drift in a warm worker
loop near hang/perf-budget kill points). Config: best-effort tier, validity gate
ENABLED, env ``ORTHONYM_SELF_CONSISTENCY_GATE=off`` ONLY (NEVER
``ORTHONYM_DISABLE_OPSIN_VALIDITY_GATE`` — the reclaim self-suppresses with a dead gate).
"""
import json
import os
import re
import subprocess
import sys

import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym.namer import _validity_gate_jar_present, _validity_gate_name_to_smiles
from orthonym.errors import is_failure_name
from orthonym.validation.opsin_roundtrip import opsin_parse


pytestmark = [
    pytest.mark.roundtrip,
    pytest.mark.slow,
    pytest.mark.skipif(not _validity_gate_jar_present(),
                       reason="OPSIN jar absent — the reclaim requires a live validity gate"),
]


# --- fresh-subprocess naming (determinism near kill points) ------------------

# Named in a child so hang/perf budgets, warm caches and OPSIN state never leak
# between molecules. Best-effort tier, validity gate ENABLED, off.
_DRIVER = r"""
import os, sys, json
os.environ["ORTHONYM_SELF_CONSISTENCY_GATE"] = "off"
os.environ.pop("ORTHONYM_DISABLE_OPSIN_VALIDITY_GATE", None)
from orthonym.namer import Orthonym
smi = sys.argv[1]
eng = Orthonym(general_fallback=True, general_fallback_unverified=True,
                allow_aromatic_general=True)
try:
    name = eng.name(smi)
except Exception:
    name = None
sys.stdout.write("OSTOUT_RESULT<<" + json.dumps(name) + ">>OSTOUT_END\n")
"""

_RESULT_RE = re.compile(r"OSTOUT_RESULT<<(.*?)>>OSTOUT_END", re.DOTALL)


def _name_fresh(smiles):
    """Name one SMILES in a fresh subprocess; return the name string or None."""
    env = dict(os.environ)
    env["ORTHONYM_SELF_CONSISTENCY_GATE"] = "off"
    env.pop("ORTHONYM_DISABLE_OPSIN_VALIDITY_GATE", None)
    from orthonym.jvm_budget import jvm_slots
    with jvm_slots(1, purpose="p1-reclaim-stereo-test"):
        proc = subprocess.run(
            [sys.executable, "-c", _DRIVER, smiles],
            capture_output=True, text=True, env=env, timeout=420)
    m = _RESULT_RE.search(proc.stdout)
    if m is None:
        raise AssertionError(
            f"driver produced no result for {smiles!r}\n"
            f"STDOUT tail:\n{proc.stdout[-800:]}\nSTDERR tail:\n{proc.stderr[-800:]}")
    return json.loads(m.group(1))


# --- oracles -----------------------------------------------------------------

def _full_ik(smi):
    m = Chem.MolFromSmiles(smi)
    return inchi.MolToInchiKey(m) if m else None


def _iso_canon(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m, isomericSmiles=True) if m else None


def full_rt(name, smi):
    """True iff OPSIN parses ``name`` and it round-trips to ``smi`` (full InChIKey)."""
    if not name or is_failure_name(name):
        return False
    r = opsin_parse(name)
    if not r:
        return False
    mr, ms = Chem.MolFromSmiles(r), Chem.MolFromSmiles(smi)
    if mr is None or ms is None:
        return False
    return Chem.MolToInchiKey(mr) == Chem.MolToInchiKey(ms)


def _emitted(name):
    return bool(name) and not is_failure_name(name)


# ============================================================================
# (a) FALSE-ABSTAIN rows now emit a full-InChIKey name (count-veto reclaim:
# as-is complete-but-miscounted + exocyclic-ylidene composition).
# ============================================================================
BUCKET_A = [
    # 101644479 — symmetric bis-penicillin: the candidate already carries the
    # (2S,5R,6R) block and full-RTs, but bis(...) doubles the DEFINED count so the
    # raw count under-counts (3 < 6) and the veto used to fire. AS-IS reclaim.
    ("101644479",
     "CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)CC3=CC4=C(C=C3)OC5C4(C=CC(=O)C5)CC(=O)"
     "N[C@H]6[C@@H]7N(C6=O)[C@H](C(S7)(C)C)C(=O)OC)C(=O)OC)C"),
    # 122493921 — exocyclic ylidene C=C E/Z dropped -> composed to (1Z)-...
    ("122493921", r"CC1=C\2C(=CC=C1)CCO/C2=C\3/CCCN3"),
    # 166852113 — ylidene -> (2Z)-...
    ("166852113", r"CC1/C(=C/2\CCN(C2)C)/N1N"),
    # 24215632 — ylidene dithiole -> (3Z)-...
    ("24215632", r"CC\1=C(SS/C1=C\2/C(=C(SS2)C3=CC=CC=C3)C)C4=CC=CC=C4"),
    # 173868686 — fix-spec DEVIATION: labeled a conflict ("over-specification"),
    # but verified (2026-09-14) a genuine 0-wrong reclaim -> (1Z,3R,5S)-... which
    # FULL round-trips (input==recon an InChIKey). All its
    # centres ARE specified in the input; the reclaim completes them, not over-cites.
    ("173868686",
     r"CCC1=C(C=CC(=C1)NC(=NC)C2=NC=C(N2C)C(=C)C(C)[C@@H](C)/C(=C(\C)/F)/OC)"
     "C(=C)N(CC[C@H](C)C(=O)O)C=C"),
]


@pytest.mark.parametrize("cid,smi", BUCKET_A, ids=[c for c, _ in BUCKET_A])
def test_a_false_abstain_reclaimed(cid, smi):
    """CID {cid}: the count-veto abstention becomes a full-InChIKey emission."""
    name = _name_fresh(smi)
    assert _emitted(name), f"CID {cid}: expected a reclaim, got {name!r}"
    assert full_rt(name, smi), f"CID {cid}: {name!r} does not full-round-trip"


# ============================================================================
# (b) DEVIATION from the fix-spec (verified 2026-09-14, one fresh subprocess each):
# the two census-labeled "const_only OMISSION" rows below are NOT general-engine
# count-veto cases — they are named by CLASS HANDLERS, so `name_general`
# returns None and they NEVER reach the 6 Phase-1 sites `_stereo_emit_decision`
# governs:
# * 98390559 — spiro/complex_ring handler: builds the RIGHT constitution
# (skeleton InChIKey matches the input) but drops all 6 R/S; the ^-locant
# pentacyclo composition is on the pre-existing `_final_opsin_validity_
# gate` reclaim path (NOT Phase-1) and does not complete.
# * 169629145 — pyridazinone handler: gate-off it ships an `...2-ethylidene...
# (3Z)...` name that drops the exocyclic ethylidene E/Z (unexpressible).
# Both correctly ABSTAIN under Phase-1 (0-wrong; the fix must not make a
# handler-path molecule emit an incomplete/wrong isomer). This is exactly the
# census over-attribution the fix-spec warns of. The general-engine
# OMISSION->compose path (STEP 4b) IS covered — by bucket (a)'s three ylidene
# compositions (122493921/166852113/24215632).
# ============================================================================
BUCKET_B_OUT_OF_SCOPE = [
    ("169629145",
     r"C/C=C(/CN1C(=O)C=CC(=N1)C2=CN=C(N=C2)N3CCCC3C(F)(F)F)\C=C/C#C"),
    ("98390559", "C1COC2(O1)C3[C@H]4[C@@H]5[C@@H]4C2([C@@H]6[C@@H]5[C@@H]63)Br"),
]


@pytest.mark.parametrize("cid,smi", BUCKET_B_OUT_OF_SCOPE,
                         ids=[c for c, _ in BUCKET_B_OUT_OF_SCOPE])
def test_b_handler_path_omission_abstains_out_of_scope(cid, smi):
    """CID {cid}: a handler-path omission (outside Phase-1's 6 general-engine sites)
    must ABSTAIN 0-wrong — never ship an incomplete/wrong stereoisomer."""
    name = _name_fresh(smi)
    if _emitted(name):
        out = _validity_gate_name_to_smiles(name)
        assert out is not None and _iso_canon(out) == _iso_canon(smi), (
            f"CID {cid}: shipped a WRONG/incomplete stereoisomer {name!r}")
    else:
        assert not _emitted(name)  # abstained -> correct (out of Phase-1 scope)


# ============================================================================
# (c) CONFLICT rows still ABSTAIN (a wrong-config / over-specified candidate
# must never ship — the emission is None or a failure sentinel).
# ============================================================================
BUCKET_C = [
    # 122677887 — von-Baeyer bridgehead R/S inversion (S->R).
    ("122677887",
     "CC1([C@H]2CC[C@@]1(C3=NN=C(C=C23)C4=C(C=CC=C4F)F)C5=NC(=CC=C5)"
     "S(=O)(=O)NC6(COC6)CO)C"),
    # 169166535 — fusion-carbon R vs S conflict.
    ("169166535",
     "CC(C)(C)C1=NC=C(O1)C(=O)N2CCC3=C([C@@H]2C4=CC5=CC=CC=C5C=N4)N=CN3"),
    # 144005043 — ring C=C E vs Z ("Stereo backstop" wrong sign).
    ("144005043",
     r"C\1C/C(=C\2/C=NC(=N/C2=C/C=C1)NCC3=CCC(C=C3)O)/N4CCC(CC4)C(=O)NC5=CN=CC=C5"),
]


@pytest.mark.parametrize("cid,smi", BUCKET_C, ids=[c for c, _ in BUCKET_C])
def test_c_conflict_abstains(cid, smi):
    """CID {cid}: a stereo CONFLICT must abstain (0-wrong), never ship an isomer."""
    name = _name_fresh(smi)
    if _emitted(name):
        # If anything IS emitted it must denote EXACTLY the input (never a wrong
        # isomer) — the airtight 0-wrong floor. A true conflict cannot achieve this
        # and must abstain, so this branch should not be reached for these rows.
        out = _validity_gate_name_to_smiles(name)
        assert out is not None and _iso_canon(out) == _iso_canon(smi), (
            f"CID {cid}: shipped a WRONG stereoisomer {name!r}")
        pytest.fail(f"CID {cid}: expected ABSTAIN, emitted {name!r} (verify producer)")
    # abstained -> correct.


# ============================================================================
# (d) Non-vacuous enantiomer/isomer distinctness (compare NAME STRINGS, never
# verify_or_none): a completion must use ABSOLUTE descriptors, so two
# distinct stereoisomers get two DIFFERENT names.
# ============================================================================
def _flip_first_stereo_double_bond(smi):
    """Return a SMILES with the first stereo double bond's geometry flipped, or
    None if RDKit realizes no distinct isomer."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    swap = {Chem.BondStereo.STEREOZ: Chem.BondStereo.STEREOE,
            Chem.BondStereo.STEREOE: Chem.BondStereo.STEREOZ,
            Chem.BondStereo.STEREOCIS: Chem.BondStereo.STEREOTRANS,
            Chem.BondStereo.STEREOTRANS: Chem.BondStereo.STEREOCIS}
    for b in mol.GetBonds():
        if b.GetStereo() in swap:
            b.SetStereo(swap[b.GetStereo()])
            out = Chem.MolToSmiles(mol)
            if _full_ik(out) and _full_ik(out) != _full_ik(smi):
                return out
            return None
    return None


def test_d_ylidene_ez_distinct():
    """The ylidene reclaim uses an absolute (nE)/(nZ) descriptor: flipping the
    geometry yields a DIFFERENT name (and still 0-wrong)."""
    orig = r"CC1=C\2C(=CC=C1)CCO/C2=C\3/CCCN3"  # 122493921
    orig_name = _name_fresh(orig)
    assert _emitted(orig_name), f"122493921 did not emit: {orig_name!r}"
    flip = _flip_first_stereo_double_bond(orig)
    if flip is None:
        pytest.skip("RDKit realized no distinct geometric flip of the ylidene bond")
    flip_name = _name_fresh(flip)
    if not _emitted(flip_name):
        pytest.skip(f"flipped isomer abstained ({flip_name!r}); distinctness untestable")
    assert flip_name != orig_name, (
        f"E/Z isomers got the SAME name {orig_name!r} — descriptor is not absolute")
    assert full_rt(flip_name, flip), f"flipped emission {flip_name!r} is not 0-wrong"


def test_d_tetra_flip_distinct():
    """Flipping one tetrahedral centre of the bis-penicillin changes the emitted
    descriptor block (absolute R/S), so the name string differs."""
    orig = ("CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)CC3=CC4=C(C=C3)OC5C4(C=CC(=O)C5)"
            "CC(=O)N[C@H]6[C@@H]7N(C6=O)[C@H](C(S7)(C)C)C(=O)OC)C(=O)OC)C")  # 101644479
    flip = orig.replace("[C@@H]", "[C@H]", 1)  # invert the first centre
    assert _full_ik(flip) and _full_ik(flip) != _full_ik(orig), "flip did not change the isomer"
    orig_name = _name_fresh(orig)
    flip_name = _name_fresh(flip)
    assert _emitted(orig_name), f"101644479 did not emit: {orig_name!r}"
    if not _emitted(flip_name):
        pytest.skip(f"flipped diastereomer abstained ({flip_name!r}); distinctness untestable")
    assert flip_name != orig_name, (
        "diastereomers got the SAME name — descriptor block is not absolute")
    assert full_rt(flip_name, flip), f"flipped emission {flip_name!r} is not 0-wrong"


def test_d_adv1_enantiomer_absolute_not_relative():
    """Adv1 engine-level 50/50 guard: a true enantiomeric pair must get two
    DIFFERENT ABSOLUTE-descriptor names, never one shared bare relative token."""
    a = "C[C@@H]1CCCC[C@H]1O"
    b = "C[C@H]1CCCC[C@@H]1O"  # enantiomer of a
    assert _full_ik(a) and _full_ik(a) != _full_ik(b), "control pair is not distinct"
    na, nb = _name_fresh(a), _name_fresh(b)
    assert _emitted(na) and _emitted(nb), f"pair did not both emit: {na!r} / {nb!r}"
    abs_re = re.compile(r"\(\d+[a-z]?[RS]")   # e.g. (1R / (2S / (7aR
    rel_re = re.compile(r"(?:^|[-([{ \t])(?:cis|trans|rel|rac)-", re.IGNORECASE)
    for n in (na, nb):
        assert abs_re.search(n), f"{n!r} carries no absolute R/S descriptor"
        assert not rel_re.search(n), f"{n!r} carries a bare RELATIVE token (vacuous on a pair)"
    assert na != nb, f"enantiomers got the SAME name {na!r}"


# ============================================================================
# (e) Adv3 regression guard — InChI-blind incompleteness must ABSTAIN.
# ============================================================================
def test_e_inchi_blind_incompleteness_abstains():
    """CID 177393776: 3 defined stereo elements — 2 C=N imine (standard InChIKey is
    BLIND to them) + 1 ylidene. Composition drops the 2 C=N, so a full-InChIKey gate
    would false-confirm the under-specified name. The exact-isomeric CanonSmiles gate
    catches it -> the row MUST abstain, never ship a mismatched isomer."""
    smi = ("C1=CC=C2N/C(=N\\C3=NC4=CC=CC=C4S3)/C(=C\\5/C(=N/C6=NC7=CC=CC=C7S6)/"
           "NC8=CC=CC=C58)/C2=C1")
    name = _name_fresh(smi)
    if _emitted(name):
        # If (unexpectedly) a name is emitted it must be the EXACT input isomer under
        # canonical ISOMERIC SMILES — never an InChI-blind mismatched isomer.
        out = _validity_gate_name_to_smiles(name)
        assert out is not None and _iso_canon(out) == _iso_canon(smi), (
            f"shipped an InChI-blind under-composed isomer: {name!r}")
    else:
        assert not _emitted(name)  # abstained -> correct


# ============================================================================
# fast unit test (no OPSIN) — the relative-stereo-token guard (STEP 0 / Adv1).
# ============================================================================
def test_has_relative_stereo_token():
    from orthonym.namer import _has_relative_stereo_token
    # relative / ambiguous descriptors -> True (must be guarded)
    assert _has_relative_stereo_token("trans-2-methylcyclohexan-1-ol")
    assert _has_relative_stereo_token("cis-4-tert-butylcyclohexan-1-ol")
    assert _has_relative_stereo_token("rel-(1R,2S)-2-methylcyclohexan-1-ol")
    assert _has_relative_stereo_token("(±)-mandelic acid")
    assert _has_relative_stereo_token("(RS)-butan-2-ol")
    assert _has_relative_stereo_token("rac-alanine")
    assert _has_relative_stereo_token("(1R*,2S*)-2-methylcyclohexan-1-ol")
    # absolute / config-prefix / no-stereo names -> False (must NOT be guarded)
    assert not _has_relative_stereo_token("(1Z)-2-(pyrrolidin-2-ylidene)ethyl")
    assert not _has_relative_stereo_token("(2S,5R,6R)-penicillanic acid")
    assert not _has_relative_stereo_token("beta-D-glucopyranosyl bromide")
    assert not _has_relative_stereo_token("butan-2-ol")
    assert not _has_relative_stereo_token("")
