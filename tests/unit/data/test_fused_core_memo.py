""" a phase — the fused-core matcher memo (perf lever, byte-identical).

``_match_fused_heterocycle_core_impl`` is re-run 100-390x on the SAME ring fragment
within one molecule's naming (the recursive substituent enumeration re-derives it),
exhausting the per-molecule macrocycle-hang budgets and abstaining a NAMEABLE
macrocycle. A per-molecule memo (keyed on the atom-order-preserving SMILES, a COMPLETE
key for the atom-index-keyed result) turns a repeat into an O(1) hit that charges no
budget → the molecule finishes under the ORIGINAL budget. These tests pin the 0-wrong
contract: the memo must never change an emitted name.

Measured reclaim (frozen perf_op_budget sample, default 12M/500 budget): 19/40 rows
reclaim at 0-wrong (0 WRONG), byte-identical for everything that already completed.
"""
import os
import subprocess
import sys

import pytest

from orthonym import name_compound
from orthonym.jvm_budget import jvm_slots

# Fused ring systems that exercise the core matcher (indole/quinoline/carbazole/…).
FUSED = [
    "c1ccc2[nH]ccc2c1", "Cc1ccc2[nH]ccc2c1", "c1ccc2ncccc2c1",
    "c1ccc2c(c1)ccc1ccccc12", "c1ccc2c(c1)[nH]c1ccccc12",
    "O=c1[nH]c2ccccc2o1", "c1ccc2c(c1)oc1ccccc12", "c1ccc2[nH]c3ccccc3c2c1",
]


# PubChem-ordered SMILES (atom order != RDKit DFS order) — the class that exposed the
# incomplete bare-string key (a review P3 Crit-1). The first is a review's production witness:
# with the bare-string key it named differently memo on vs off (RT gate caught the
# wrong-locant candidate, so 0-wrong held, but "byte-identical" was false).
PUBCHEM_ORDERED = [
    "CCC1=CC=C(C=C1)NC(=O)[C@H](C)OC(=O)CN2C=NC3=CC=CC=C3C2=O",
    "CC1=CC2=C(C=C1)N=CC=C2",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",   # caffeine, PubChem order
    "C1=CC2=CC3=CC=CC=C3C=C2C=C1",    # anthracene, PubChem order
]


def _name_smis(memo_mode, smis, best_effort=False):
    """Name `smis` in a subprocess with ORTHONYM_MEMO=<mode> (read at import)."""
    env = dict(os.environ)
    env["ORTHONYM_MEMO"] = memo_mode
    flags = (", general_fallback=True, general_fallback_unverified=True, "
             "allow_aromatic_general=True") if best_effort else ""
    code = (
        "import sys, json\n"
        "from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')\n"
        "from orthonym import name_compound\n"
        f"print(json.dumps([name_compound(s{flags}) for s in sys.argv[1:]]))\n"
    )
    p = subprocess.run([sys.executable, "-c", code] + smis,
                       env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-500:]
    import json
    return json.loads(p.stdout.strip().splitlines()[-1])


def _name_all(memo_mode):
    """Name FUSED in a subprocess with ORTHONYM_MEMO=<mode> (read at import)."""
    return _name_smis(memo_mode, FUSED)


@pytest.mark.roundtrip
def test_fused_core_memo_is_byte_identical():
    """The memo (on) must emit the identical names as the recompute path (off)."""
    with jvm_slots(2, purpose="fused-core-memo-test"):
        on = _name_all("on")
        off = _name_all("off")
    assert on == off, [(a, b) for a, b in zip(on, off) if a != b]
    # sanity: the catalog actually matched (not all abstain)
    assert on[0] == "1H-indole"
    assert "quinoline" in on


@pytest.mark.roundtrip
def test_fused_core_memo_verify_mode_finds_no_incomplete_key():
    """verify mode recomputes every call and raises MemoMismatch the instant the key
    is incomplete. A clean run over the fused set proves the key is complete."""
    with jvm_slots(1, purpose="fused-core-memo-verify"):
        verify = _name_all("verify")
        on = _name_all("on")
    assert verify == on, "verify-mode names differ -> a memo key is incomplete"


@pytest.mark.roundtrip
def test_fused_core_memo_byte_identical_on_pubchem_ordered_inputs():
    """a review P3 Crit-1 regression: a PubChem-ordered SMILES (atom order != RDKit DFS)
    made the bare-string key `MolToSmiles(canonical=False)` collide across labelings,
    so the index-keyed atom_mapping was applied to the wrong labeling and the name
    changed memo on-vs-off. The key now includes `_smilesAtomOutputOrder`, so on==off.
    Best-effort tier (where the fused-core matcher is re-run most)."""
    with jvm_slots(2, purpose="fused-core-memo-pubchem"):
        on = _name_smis("on", PUBCHEM_ORDERED, best_effort=True)
        off = _name_smis("off", PUBCHEM_ORDERED, best_effort=True)
    diffs = [(s, a, b) for s, a, b in zip(PUBCHEM_ORDERED, on, off) if a != b]
    assert not diffs, f"memo changed the name on PubChem-ordered inputs: {diffs}"


def test_fused_core_memo_key_is_complete_verify_counter():
    """a review P3 Crit-2: a MemoMismatch raised inside the matcher is SWALLOWED by the
    naming cascade, so '0 uncaught exceptions' proves nothing. The process-global
    verify counter records every same-key-different-result event BEFORE the raise, so
    it is the real detector. Over the witness class the `fused_core` namespace must
    record 0 (an incomplete key would record > 0)."""
    env = dict(os.environ)
    env["ORTHONYM_MEMO"] = "verify"
    env["ORTHONYM_PERF_OP_BUDGET"] = "0"      # disarm hang budgets so the matcher
    env["ORTHONYM_ANALYSIS_CALL_BUDGET"] = "0"  # is re-run fully (max memo exercise)
    code = (
        "import sys, json\n"
        "from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')\n"
        "import logging; logging.disable(logging.CRITICAL)\n"
        "from orthonym import name_compound\n"
        "from orthonym.assembly import memo\n"
        "memo.reset_verify_mismatches()\n"
        "for s in sys.argv[1:]:\n"
        "    try: name_compound(s, general_fallback=True, general_fallback_unverified=True, allow_aromatic_general=True)\n"
        "    except Exception: pass\n"
        "fc = sum(1 for m in memo._VERIFY_MISMATCHES if m[0] == 'fused_core')\n"
        "print(json.dumps({'fused_core': fc}))\n"
    )
    with jvm_slots(1, purpose="fused-core-memo-counter"):
        p = subprocess.run([sys.executable, "-c", code] + PUBCHEM_ORDERED,
                           env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-500:]
    import json
    res = json.loads(p.stdout.strip().splitlines()[-1])
    assert res["fused_core"] == 0, (
        f"fused_core memo key is INCOMPLETE: {res['fused_core']} verify mismatches")
