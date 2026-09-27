"""M4 : the best-effort floor must COMPLETE ring stereochemistry.

A saturated ring stereocentre needs a relative descriptor (cis/trans, or
an OPSIN-numbering-anchored absolute descriptor — NOT the bare `(1r,3R)-` absolute
R/S block `collect_stereodescriptors` emits, which OPSIN cannot parse. When the
ring block is unparseable the whole stereo layer is dropped and the molecule
abstains (offer full-InChIKey gate voids the stereo-incomplete name).

The floor now runs candidate competition: it offers ring relative-stereo (cis/trans)
composed with the branch/chain absolute R/S, and the offer full-InChIKey gate keeps
whichever round-trips (0-wrong by construction). Finding:
internal notes.
"""
import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import inchi

# The checkout under test, derived from this file (TRIAGE g6 C25): the path used to
# be the main tree's, hard-coded, so a run from another checkout (a worktree, CI)
# named with the main tree's src in the child and tested the wrong code.
_REPO = Path(__file__).resolve().parents[3]
_SRC = _REPO / "src"
sys.path.insert(0, str(_SRC))
from orthonym.jvm_flags import java_cmd  # noqa: E402
from tests.support.jars import jar_or_skip


def _full_rt(name, smiles):
    """OPSIN-parse name, compare full InChIKey to the input (constitution+stereo)."""
    if not name or name.startswith("unknown") or "not supported" in name:
        return False
    p = subprocess.run(java_cmd() + ["-jar", jar_or_skip(), "-o", "smi"],
                       input=name + "\n", capture_output=True, text=True)
    out = p.stdout.strip().splitlines()
    out = out[0] if out else ""
    mo = Chem.MolFromSmiles(out) if out else None
    if mo is None:
        return False
    return inchi.MolToInchiKey(mo) == inchi.MolToInchiKey(Chem.MolFromSmiles(smiles))


def _best_effort_name(smiles):
    # Name in a FRESH subprocess: the whole-molecule pipeline is warm-state
    # sensitive (a shared OPSIN/CIP cache across a pytest session can leak an
    # unrelated intermediate candidate), so the deterministic production behaviour
    # (one molecule per process, as the eval harness runs) is only reproduced in a
    # clean interpreter -- the same subprocess pattern test_c1c2c6 uses.
    code = (
        "import logging; logging.disable(logging.CRITICAL)\n"
        "from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')\n"
        "from orthonym.namer import Orthonym\n"
        "eng = Orthonym(general_fallback=True, general_fallback_unverified=True, allow_aromatic_general=True)\n"
        f"print(repr(eng.name({smiles!r})))\n"
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        x for x in (str(_SRC), os.environ.get("PYTHONPATH", "")) if x))
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       cwd=str(_REPO), env=env)
    out = p.stdout.strip().splitlines()
    # the child prints repr of the name string; ast.literal_eval parses that
    # string literal safely (never eval on subprocess output).
    return ast.literal_eval(out[-1]) if out else None


# 2-substituted ring stereocentres (cis/trans) composed with a chain branch centre.
RING_STEREO_WITNESSES = [
    # 1,3-cyclobutane (trans) + a chain (3R): the pinned witness.
    "N#CC1(C(=O)N[C@H]2C[C@H](NC(=O)[C@@H](F)OC(F)(F)F)C2)CC1",
]


@pytest.mark.parametrize("smiles", RING_STEREO_WITNESSES)
def test_floor_completes_ring_stereo_and_full_rt(smiles):
    name = _best_effort_name(smiles)
    assert name and not name.startswith("unknown"), f"abstained: {name!r}"
    assert _full_rt(name, smiles), f"stereo not completed / no full-InChIKey RT: {name!r}"
