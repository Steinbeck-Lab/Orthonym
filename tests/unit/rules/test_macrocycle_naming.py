"""v41 macrocyclic-limonoid naming — the three-fix milestone (F2 ene compound-locant,
F1 P-16.3 enclosing marks, F3 decoration composition).

These deep NP cages emit a whole-molecule oxa-von-Baeyer ``complex_ring`` candidate whose
skeleton numbering is correct (audit 0 diff edges vs OPSIN) but whose name string does not
OPSIN-parse, so the engine abstains (0-wrong). F2 is the dominant blocker: a ring double
bond spanning a bridge (its two locants are non-consecutive) must be cited with the
P-31.1.4.2(1) compound locant ``15(36)``; the current path emits a bare ``15`` which OPSIN
reads as the 15=16 bond, and 16 is a carbonyl → cumulated ketene → ``C valency: 5``.

Building the whole-cage candidate for a 53-atom cage takes ~90 s with the macrocycle-hang
work budget disabled (the budget otherwise fast-abstains these before the candidate is
built), so the end-to-end wiring test is ``@slow``. The parent-name builder is captured at
``polycyclic.py`` (``_build_parent_with_unsaturation``, right after the F2 unsaturation
step) — deterministic and exactly the string F2 changes.
"""
import json
from pathlib import Path

import pytest

from orthonym import Orthonym

_SMILES = json.loads((Path(__file__).parent / "_macrocycle_smiles.json").read_text())

# One representative formula-complete macrocyclic cage with a bridge-spanning ring double
# bond (numbering (1,24) — non-consecutive). The full 3-cage + witness sweep is in the
# milestone measurement (MACROCYCLE-* planning docs), not this fast-suite file.
F2_WITNESS_CID = "25180764"


# Subprocess: build the whole-cage candidate with the macrocycle-hang budget disabled
# (read at import -> must be a subprocess), wrap the named site, print the captured
# strings. `SITE` selects what to capture: "parent" = _build_parent_with_unsaturation
# (F2's ene locant); "candidate" = the whole complex_ring candidate name (F1's brackets).
_SUBPROC = r'''
import os, sys, json
os.environ["ORTHONYM_PERF_OP_BUDGET"] = "0"
os.environ["ORTHONYM_ANALYSIS_CALL_BUDGET"] = "0"
import logging; logging.getLogger().setLevel(logging.CRITICAL)
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
from orthonym.rules import polycyclic as poly
from orthonym.assembly import candidate_pool as cp_mod
from orthonym import Orthonym
site, smiles = sys.argv[1], sys.argv[2]
captured = []
if site == "parent":
    orig = poly._build_parent_with_unsaturation
    def wrapped(total_atoms, unsaturation, fg_suffix=None):
        out = orig(total_atoms, unsaturation, fg_suffix=fg_suffix)
        if out:
            captured.append(out)
        return out
    poly._build_parent_with_unsaturation = wrapped
else:  # candidate
    orig = cp_mod.CandidatePool.add
    def wrapped(self, name, handler_id, features, *a, **kw):
        if handler_id == "complex_ring" and name:
            captured.append(name)
        return orig(self, name, handler_id, features, *a, **kw)
    cp_mod.CandidatePool.add = wrapped
Orthonym(general_fallback=True, general_fallback_unverified=True,
          allow_aromatic_general=True).name(smiles)
print("CAPTURED_JSON:" + json.dumps(captured))
'''


def _capture(site, smiles):
    """Build the whole-cage candidate in a SUBPROCESS (budget off) and return the strings
    captured at ``site`` ("parent" or "candidate"). ~90 s (hence ``@slow``)."""
    import subprocess
    import sys
    proc = subprocess.run([sys.executable, "-c", _SUBPROC, site, smiles],
                          capture_output=True, text=True, timeout=300)
    for line in proc.stdout.splitlines():
        if line.startswith("CAPTURED_JSON:"):
            return json.loads(line[len("CAPTURED_JSON:"):])
    return []


@pytest.mark.unit
@pytest.mark.slow
def test_f2_ring_double_bond_uses_compound_locant():
    """v41 F2 (P-31.1.4.2(1)): the built whole-cage parent name cites a ring double bond
    whose end-locants are NOT consecutive as ``lo(hi)``, not a bare ``lo``. The bare form
    makes OPSIN read the wrong bond and raise ``C valency: 5``.

    RED at HEAD before F2 (parent name `...tetracosa-1,11-diene...`, bare); GREEN after
    (`...tetracosa-1(24),11-diene...`)."""
    import re
    names = _capture("parent", _SMILES[F2_WITNESS_CID])
    assert names, f"{F2_WITNESS_CID}: no polycyclic parent name was built"
    assert any(re.search(r"\d+\(\d+\)", n) for n in names), (
        f"{F2_WITNESS_CID}: expected a P-31.1.4.2(1) compound ring-ene locant N(M) in a "
        f"built parent name; got {names!r}")


@pytest.mark.unit
@pytest.mark.slow
def test_f1_compound_substituent_prefix_is_enclosed():
    """v41 F1 (P-16.3.3 / P-29.6.1): a COMPOUND substituent prefix on the whole-cage
    candidate (e.g. the ylidene ``3-methoxy-3-oxopropan-2-ylidene``) is cited in
    enclosing marks — ``20-(3-methoxy-3-oxopropan-2-ylidene)`` — not bare. The bare form
    leaves OPSIN unable to assign the substituent's internal locants.

    RED at HEAD before F1 (candidate carries `...-20-3-methoxy-3-oxopropan-2-ylidene-...`,
    unbracketed); GREEN after (`...-20-(3-methoxy-3-oxopropan-2-ylidene)-...`)."""
    import re
    cands = _capture("candidate", _SMILES[F2_WITNESS_CID])
    assert cands, f"{F2_WITNESS_CID}: no complex_ring candidate was built"
    # A compound '-ylidene' prefix (it carries an internal locant) must be bracketed:
    # look for '(' immediately before a compound propan-2-ylidene-style prefix.
    assert any(re.search(r"\(\d*-?\w*methoxy\w*ylidene\)", c) or
               re.search(r"-\(\S+ylidene\)", c) for c in cands), (
        f"{F2_WITNESS_CID}: expected the compound ylidene prefix in enclosing marks "
        f"(P-16.3.3); got {cands!r}")


@pytest.mark.unit
def test_f3b_hydroperoxy_token_is_parseable():
    """v41 F3b (P-21.2.2 / P-29.3.3): a hydroperoxy substituent ``-OOH`` (parent
    ``dioxidane``) names as the OPSIN-parseable ``dioxidanyl``, not the malformed
    ``dioxidyl`` (which drops the ``-an-`` of the heteroatom mononuclear-hydride stem).
    Fast — no cage naming."""
    from rdkit import Chem
    from orthonym.assembly.substituent_enumerator import name_substituent
    from orthonym.assembly.substituent_naming import parent_to_prefix
    # direct: the hydride family keeps its 'an' stem
    assert parent_to_prefix("dioxidane", 2, attach_locant=None) == "dioxidanyl"
    assert parent_to_prefix("oxidane", 1, attach_locant=None) == "oxidanyl"
    # end to end: -OOH on a ring -> a parseable token
    m = Chem.MolFromSmiles("OOC1CCCCC1")
    ring = set()
    for r in m.GetRingInfo().AtomRings():
        ring.update(r)
    attach = next(a.GetIdx() for a in m.GetAtoms()
                  if a.GetIdx() not in ring
                  and any(n.GetIdx() in ring for n in a.GetNeighbors()))
    branch = [i for i in range(m.GetNumAtoms()) if i not in ring]
    tok = name_substituent(m, branch, attach)
    assert tok == "dioxidanyl", f"expected dioxidanyl, got {tok!r}"
