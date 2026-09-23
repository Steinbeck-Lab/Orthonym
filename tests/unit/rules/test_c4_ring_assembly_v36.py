""" Milestone C4 -- glycan / multi-ring-system assembly (compose-across-linker).

RED baseline + fresh-process round-trip harness for the 22 addressable
"ring-assembly" witnesses the C4 grounding trace confirmed
(internal notes): molecules made of two-or-more
individually-nameable ring systems joined by a linker (C-chain / ether /
ester / glycosidic-O / thioether-sulfinyl). Every one currently emits
``unknown organic compound`` at the best-effort tier.

WHY FRESH PROCESS PER WITNESS (do not "optimise" this away):
the ``isolated_naming_session`` warm-cache hazard (memory
``feedback_spy_before_you_refute``; the C4/C1/C2/C6 reviews all hit it) means
a witness can emit a correct name in a WARM process and abstain COLD. The RED
baseline is only trustworthy measured in a fresh interpreter per witness, so
``_c4_rt`` shells out to a one-shot ``python -c`` child (its own JVM, its own
empty cache) exactly as the contributor guide's "-m orthonym one-shot" guidance
prescribes. Result is memoised per SMILES so each witness spawns exactly one
child regardless of how many tests read it.

⚠ IMPORTANT FINDING (this harness is a baseline, NOT a red-then-green TDD gate
for the plan's Task 2 as written). The C4 plan's dominant lever -- "give
``decomposition/weave.py`` a ring-hub mode" -- was measured OFF-PATH before any
code was written (a project rule, ``feedback_choke_point_off_path``):

  * ``weave.py`` is a carbon-only *core-and-arms* polyol composer
    (glycerophospholipid-shaped). With its ring guard (``weave.py:517-519``)
    removed by hand, ``_build_from_core`` still declines on **0/22** witnesses
    -- none is core-and-arms shaped, so weave can never emit the substitutive
    ring-parent names these need.
  * The ordinary whole-molecule substitutive namer ALREADY composes a ring
    across a linker for many shapes (benzyl phenyl sulfide, 2-benzylpyridine,
    diphenyl sulfoxide, 3,3'-methylenebis(1H-indole) all round-trip today).
    The linker composition is NOT the gap.
  * The real gap is deep and per-witness in the CORE namer. For the plan's
    VERIFIED target (pantoprazole, w05/w17) ``name_pipeline_only`` returns
    ``5-(difluoromethoxy)-1H-benzimidazole`` -- it names the benzimidazole
    parent and SILENTLY DROPS the whole C2 sulfinyl-pyridyl half
    (``_enrich_complex_ring_with_subs`` ``continue``s past a substituent it
    cannot render, composer.py:3947-3957). correctly suppresses the
    partial (0-wrong holds -> abstain). Forcing the von-Baeyer fallback does
    not rescue it: ``name_ortho_fused_bicyclic`` returns None for the
    substituted benzimidazole. No clean root-cause fix; a dedicated core-namer
    effort. Full findings: the C4 session report + ``V36-a trace-C4.md``.

So the per-witness "target" tests below are marked xfail (the intended
end-state -- name + full round-trip) and remain xfail at HEAD; the
"abstains-today" tests document and guard the measured baseline.
"""

from __future__ import annotations

import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]

# One-shot child: build a fresh best-effort Orthonym (the exact flags
# gen_ost_full.py / the C4 trace used), name the SMILES, and OPSIN-round-trip the
# result to a full InChI in the SAME child (one JVM). Prints a single status
# token on the last stdout line: RT-OK | RT-MISMATCH | ABSTAIN | NO-PARSE | BAD.
_CHILD = r'''
import sys, logging
logging.disable(logging.CRITICAL)
from orthonym.jvm_budget import jvm_slots
from orthonym import namer as _N
from orthonym.validation.opsin_roundtrip import opsin_parse
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchi
smi = sys.argv[1]
with jvm_slots(1, purpose="c4-rt-harness"):
    ost = _N.Orthonym(general_fallback=True,
                       general_fallback_unverified=True,
                       allow_aromatic_general=True)
    try:
        nm = ost.name(smi)
    except Exception:
        nm = None
    if not nm or "unknown" in nm.lower():
        print("ABSTAIN"); sys.exit(0)
    try:
        osm = opsin_parse(nm)
        if not osm:
            print("NO-PARSE"); sys.exit(0)
        p = Chem.MolFromSmiles(osm); m = Chem.MolFromSmiles(smi)
        if p is None or m is None:
            print("BAD"); sys.exit(0)
        print("RT-OK" if MolToInchi(p) == MolToInchi(m) else "RT-MISMATCH")
    except Exception:
        print("ERR")
'''


@lru_cache(maxsize=None)
def _c4_rt(smiles: str) -> str:
    """Name *smiles* at the best-effort tier in a FRESH process and report its
    OPSIN round-trip status. Memoised: one child per distinct SMILES."""
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, smiles],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ("NO-OUTPUT: " + proc.stderr[-200:])


# The 22 C4-addressable witnesses (V36-a trace-C4.md; the 2 real-peptide rows
# excluded per the trace's carve-out). Linker sub-class is descriptive only --
# the trace flags its atom-symbol classifier as approximate, not a P-rule bond
# typing (V36-a trace-C4.md "Caveats").
WITNESSES = [
    ("w00", "c_chain", "OCC[NH+]1CC[NH+](CC/C=C2/c3ccccc3Sc3ccc(C(F)(F)F)cc32)CC1"),
    ("w01", "glycosidic", "CC1CCC2(OC1)OC1CC3C4CCC5CC(OC6OC(CO)C(O)C(OC7OC(CO)C(O)C(O)C7O)C6O)CCC5(C)C4CCC3(C)C1C2C"),
    ("w02", "c_chain", "O=C(c1nc(-c2c[nH]c3cc(Br)ccc23)c[nH]1)c1c[nH]c2ccccc12"),
    ("w03", "ester_ether", "CC1(C)C(C(=O)OC(C#N)c2cccc(Oc3ccccc3)c2)C12C=Cc1ccccc12"),
    ("w04", "glycosidic", "CCC(C)/C=C/CCCC(C)C(O)C/C=C/C=C/C(=O)OC1C(OC2OC(CO)C(O)C(O)C2O)C(CO)OC2(OCc3cc(O)cc(O)c32)C1O"),
    ("w05", "thioether_sulfinyl", "COc1ccnc(CS(=O)(=O)c2nc3ccc(OC(F)F)cc3[nH]2)c1OC"),
    ("w08", "c_chain", "C=CC(C)(C)c1[nH]c2cc(CC=C(C)C)ccc2c1/C=c1\\[nH]c(=O)c(=C)[nH]c1=O"),
    ("w09", "c_chain", "OCCc1c([C@@H](c2c[nH]c3ccccc23)[C@H](O)CO)[nH]c2ccccc12"),
    ("w10", "c_chain", "CN1C(=CC=Cc2cc[n+](CCCC(CCCC(CCC[n+]3ccc(C=CC=C4Sc5ccccc5N4C)c4ccccc43)=[N+](C)C)=[N+](C)C)c3ccccc23)Sc2ccccc21"),
    ("w11", "c_chain", "COC(CO)[C@@H](c1[nH]c2ccccc2c1CC(=O)O)c1c[nH]c2ccccc12"),
    ("w12", "c_chain", "CC1CCC2C(C)(C)C(O)CCC2(C)C12Cc1c(cc3c(c1O)CN(CCCCC(C(=O)O)N1Cc4c(cc5c(c4O)CC4(O5)C(C)CCC5C(C)(C)C(O)CCC54C)C1=O)C3=O)O2"),
    ("w13", "c_chain", "CC(=O)OC1=C(Cc2c(CC=C(C)C)[nH]c3ccccc23)C(=O)C(O)=C(Cc2c(CC=C(C)C)[nH]c3ccccc23)C1=O"),
    ("w14", "c_chain", "O=C1CC2(C(=O)N1)C(=O)N(Cc1ccc(Br)cc1F)C(=O)c1ccc(F)cc12"),
    ("w15", "glycosidic", "CC(=O)OC(CC1C=C(C)C(=O)O1)C(C)C1CCC2C3(C)CCC(OC4OC(CO)C(O)C(O)C4OC4OC(C)C(O)C(O)C4O)C(C)(C)C3CCC2(C)C12COC(=O)C2"),
    ("w16", "c_chain", "OCCc1c([C@@H](CO)c2c[nH]c3ccccc23)[nH]c2ccccc12"),
    ("w17", "thioether_sulfinyl", "COc1ccnc(CS(=O)c2nc3cc(OC(F)F)ccc3[nH]2)c1OC"),
    ("w18", "amide_mixed", "O=C(/C=C/c1ccccc1)NC(NC(=S)Nc1cccc2cccnc12)C(Cl)(Cl)Cl"),
    ("w19", "c_chain", "OCCc1c([C@@H](c2c[nH]c3ccccc23)C(O)CO)[nH]c2ccccc12"),
    ("w20", "ester_ether", "O=C(ON1C(=O)CCC1=O)c1cc(Cl)c2c(c1Cl)C1(OC2=O)c2cc(Cl)c(O)cc2Oc2cc(O)c(Cl)cc21"),
    ("w21", "ester_ether", "C=CC(C)(C)c1[nH]c2ccccc2c1C=C1NC(=O)[C@]23C[C@H](c4c(c(C)cc(O)c4C(=O)OC)O2)[C@]2(CC(=O)C=C(OC)C2=O)N3C1=O"),
    ("w22", "c_chain", "O=C(O)[C@H](O)Cc1c(Cc2c[nH]c3ccccc23)[nH]c2ccccc12"),
    ("w23", "ester_ether", "O=C1OC2(c3ccc(O)cc3Oc3cc(Oc4ccc(O)cc4)ccc32)c2ccccc21"),
]

_IDS = [f"{wid}-{linker}" for wid, linker, _ in WITNESSES]

# Witnesses split by CURRENT measured round-trip status (fresh process per
# SMILES, re-measured 2026-09-22). The -C4 baseline was "all 22 ABSTAIN";
# later milestones made 21 of them name AND OPSIN-round-trip to the CORRECT
# molecule (RT-OK -- the child compares OPSIN's InChI of the emitted name against
# the input's InChI, so RT-OK is an independent correct-constitution check, not a
# claim about the code under test). Per this file's own protocol, a witness that
# flips to RT-OK is promoted from the abstain baseline to a plain round-trip
# assert. Only w21 (a substituted spiro-lactone ring assembly) still abstains.
_ABSTAIN_IDS = {"w21"}
RT_OK_WITNESSES = [w for w in WITNESSES if w[0] not in _ABSTAIN_IDS]
ABSTAIN_WITNESSES = [w for w in WITNESSES if w[0] in _ABSTAIN_IDS]
_RT_OK_TEST_IDS = [f"{wid}-{linker}" for wid, linker, _ in RT_OK_WITNESSES]
_ABSTAIN_TEST_IDS = [f"{wid}-{linker}" for wid, linker, _ in ABSTAIN_WITNESSES]


def test_harness_validates_against_a_known_positive():
    """Guard against a silently-broken harness (``feedback_harness_that_
    reports_success``): a molecule Orthonym names today MUST report RT-OK, or
    every ABSTAIN below is meaningless."""
    assert _c4_rt("C(c1ccccc1)Sc1ccccc1") == "RT-OK"  # benzyl phenyl sulfide


@pytest.mark.parametrize("wid,linker,smiles", RT_OK_WITNESSES, ids=_RT_OK_TEST_IDS)
def test_c4_witness_names_and_round_trips(wid, linker, smiles):
    """-C4 PROGRESS: these witnesses now name to a full-InChI OPSIN round-trip
    (RT-OK). Promoted from the original "all 22 abstain" baseline as later
    milestones composed ring-across-linker. The OPSIN round-trip (name -> InChI,
    compared to the input's InChI) is the correctness check, independent of the
    naming code -- so this asserts a correct constitution, not merely a non-empty
    string. A witness that stops round-tripping fails here (teeth)."""
    assert _c4_rt(smiles) == "RT-OK"


@pytest.mark.parametrize("wid,linker,smiles", ABSTAIN_WITNESSES, ids=_ABSTAIN_TEST_IDS)
def test_c4_witness_abstains_today(wid, linker, smiles):
    """BASELINE remnant: this witness still emits ``unknown organic compound``
    -> ABSTAIN (fail closed, 0-wrong). A change that flips it to RT-OK is real C4
    progress and should move it into ``RT_OK_WITNESSES`` above."""
    assert _c4_rt(smiles) == "ABSTAIN"


@pytest.mark.parametrize("wid,linker,smiles", ABSTAIN_WITNESSES, ids=_ABSTAIN_TEST_IDS)
@pytest.mark.xfail(
    reason="v36-C4 target: compose ring-across-linker to a full-RT name. "
    "Deep core-namer gap (weave lever refuted off-path); abstains at HEAD.",
    strict=False,
)
def test_c4_witness_names_and_round_trips_TARGET(wid, linker, smiles):
    """TARGET end-state for the still-abstaining witness: name to a full-InChI
    OPSIN round-trip. xfail while it abstains; xpasses when C4 reaches it."""
    assert _c4_rt(smiles) == "RT-OK"
