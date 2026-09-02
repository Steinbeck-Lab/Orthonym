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


_SUBPROC = r'''
import os, sys, json
os.environ["ORTHONYM_PERF_OP_BUDGET"] = "0"       # disable macrocycle-hang budgets
os.environ["ORTHONYM_ANALYSIS_CALL_BUDGET"] = "0"  # (read at import -> so, subprocess)
import logging; logging.getLogger().setLevel(logging.CRITICAL)
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
from orthonym.rules import polycyclic as poly
from orthonym import Orthonym
smiles = sys.argv[1]
captured = []
orig = poly._build_parent_with_unsaturation
def wrapped(total_atoms, unsaturation, fg_suffix=None):
    out = orig(total_atoms, unsaturation, fg_suffix=fg_suffix)
    if out:
        captured.append(out)
    return out
poly._build_parent_with_unsaturation = wrapped
Orthonym(general_fallback=True, general_fallback_unverified=True,
          allow_aromatic_general=True).name(smiles)
print("PARENT_NAMES_JSON:" + json.dumps(captured))
'''


def _capture_parent_names(smiles):
    """Build the whole-cage candidate in a SUBPROCESS with the macrocycle-hang work
    budget disabled via env (read at import; otherwise these 53-atom cages fast-abstain
    before the candidate is built). Returns every parent-name string built by
    ``_build_parent_with_unsaturation`` — the site F2 changes. Subprocess isolation keeps
    the disabled budget out of the rest of the suite. ~90 s (hence ``@slow``)."""
    import subprocess
    import sys
    proc = subprocess.run([sys.executable, "-c", _SUBPROC, smiles],
                          capture_output=True, text=True, timeout=300)
    for line in proc.stdout.splitlines():
        if line.startswith("PARENT_NAMES_JSON:"):
            return json.loads(line[len("PARENT_NAMES_JSON:"):])
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
    names = _capture_parent_names(_SMILES[F2_WITNESS_CID])
    assert names, f"{F2_WITNESS_CID}: no polycyclic parent name was built"
    assert any(re.search(r"\d+\(\d+\)", n) for n in names), (
        f"{F2_WITNESS_CID}: expected a P-31.1.4.2(1) compound ring-ene locant N(M) in a "
        f"built parent name; got {names!r}")
