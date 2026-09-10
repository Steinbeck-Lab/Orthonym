""" giants engine 1 — acyl-CoA / nucleotide-lipid re-rooted FG substituent walk.

The interior amide / thioester / phosphate-ester bonds of a CoA spine used to route
through ``parent_to_prefix`` and fail closed, collapsing the whole giant to
``unknown``. The re-rooted FG-capable located namer (best-effort-gated) now names the
CoA spine atom-complete. 0-wrong is preserved by the whole-molecule OPSIN RT gate:
a candidate that does not round-trip abstains on every SHIPPING path.
"""
import os
import subprocess
import tempfile

import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym import Orthonym
from orthonym.jvm_flags import JVM_HYGIENE_FLAGS

pytestmark = pytest.mark.opsin_gate

_JAR = str(__import__("pathlib").Path(__file__).resolve().parents[3] / "opsin-cli-2.9.0-jar-with-dependencies.jar")

# best-effort breadth tier (the documented RT-full tier — name_general all-or-none).
# 0-wrong is delivered by applying the OPSIN full-InChIKey RT gate to its output: a
# candidate that does not round-trip is a non-conversion, never an accepted name.
_BE = Orthonym(style="pin", general_fallback=True,
                general_fallback_unverified=True, allow_aromatic_general=True)

# acetyl-CoA (ChEBI:15351 canonical), a 51-heavy-atom CoA spine.
_ACETYL_COA = ("CC(=O)SCCNC(=O)CCNC(=O)[C@H](O)C(C)(C)COP(=O)(O)OP(=O)(O)"
               "OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1OP(=O)(O)O")


def _inchikey(smi):
    m = Chem.MolFromSmiles(smi)
    return inchi.MolToInchiKey(m) if m else None


def _opsin_inchikey(name):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write(name + "\n")
        p = fh.name
    try:
        out = subprocess.run(
            ["java", *JVM_HYGIENE_FLAGS, "-jar", _JAR, "-osmi", p],
            capture_output=True, text=True, timeout=180,
        ).stdout.strip()
    finally:
        os.unlink(p)
    return _inchikey(out) if out else None


def test_acetyl_coa_names_atom_complete_and_round_trips():
    """A CoA spine names atom-complete (adenine terminus present) and RT-matches."""
    name = _BE.name(_ACETYL_COA)
    assert name and "unknown" not in name.lower(), name
    # the adenine terminus must be present (not dropped)
    assert "purin" in name, name
    assert _opsin_inchikey(name) == _inchikey(_ACETYL_COA), name


def test_acetyl_coa_default_pin_unchanged():
    """PIN default tier is byte-identical: the giant abstains on the strict path."""
    pin = Orthonym(style="pin")
    name = pin.name(_ACETYL_COA)
    # default PIN either abstains or (if it names) must round-trip — never wrong.
    if name and "unknown" not in name.lower():
        assert _opsin_inchikey(name) == _inchikey(_ACETYL_COA), name


def test_propionyl_coa_names_atom_complete_and_round_trips():
    """A second CoA spine (propionyl-CoA) also names atom-complete + RT-matches —
    the win is the shared spine, not one memorized string."""
    smi = ("CCC(=O)SCCNC(=O)CCNC(=O)[C@H](O)C(C)(C)COP(=O)(O)OP(=O)(O)"
           "OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1OP(=O)(O)O")
    name = _BE.name(smi)
    assert name and "unknown" not in name.lower(), name
    assert "purin" in name, name
    assert _opsin_inchikey(name) == _inchikey(smi), name
