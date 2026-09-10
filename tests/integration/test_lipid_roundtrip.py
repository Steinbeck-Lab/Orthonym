"""OPSIN round-trip integration tests for the lipid assembler (a phase).

Generate name → OPSIN parse → compare canonical SMILES to the original. RT is
the project's accuracy oracle (-01). Stereo-aware where the backbone carries
defined configuration; the negative guards a non-lipid stays correct.

WAVE 0 CONTRACT: RED until Waves 1-4 build the subsystem.
"""

import subprocess
from pathlib import Path

import pytest
from rdkit import Chem

from orthonym import name_compound

PROJECT_ROOT = Path(__file__).parent.parent.parent
OPSIN_JAR = PROJECT_ROOT / "opsin-cli-2.9.0-jar-with-dependencies.jar"


def _opsin_available() -> bool:
    if not OPSIN_JAR.exists():
        return False
    try:
        proc = subprocess.run(["java", "-version"], capture_output=True, text=True, timeout=5)
        return proc.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _opsin_parse(name: str) -> str:
    if not name or not OPSIN_JAR.exists():
        return ""
    try:
        proc = subprocess.run(["java", "-jar", str(OPSIN_JAR), "-osmi"],
                              input=name, capture_output=True, text=True, timeout=15)
        return proc.stdout.strip()
    except (subprocess.TimeoutExpired, Exception):
        return ""


def _canon(smiles: str, *, stereo: bool = True) -> str:
    if not smiles:
        return ""
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return ""
    return Chem.MolToSmiles(m, isomericSmiles=stereo)


def _roundtrips(input_smiles: str, *, stereo: bool = True):
    name = name_compound(input_smiles)
    if not name or name == "unknown organic compound":
        return (name, False)
    opsin_smi = _opsin_parse(name)
    if not opsin_smi:
        return (name, False)
    return (name, _canon(input_smiles, stereo=stereo) == _canon(opsin_smi, stereo=stereo))


pytestmark = [pytest.mark.integration,
              pytest.mark.skipif(not _opsin_available(), reason="OPSIN/Java not available")]


LIPID_CASES = [
    ("TAG-saturated",
     "CCCCCCCCCCCCCCCC(=O)OCC(COC(=O)CCCCCCCCCCCCCCC)OC(=O)CCCCCCCCCCCCCCC"),
    ("TAG-unsaturated",
     "CCCCCCCC/C=C\\CCCCCCCC(=O)OCC(COC(=O)CCCCCCC/C=C\\CCCCCCCC)OC(=O)CCCCCCC/C=C\\CCCCCCCC"),
    ("1-MAG", "CCCCCCCCCCCCCCCC(=O)OCC(O)CO"),
    ("1,2-DAG", "CCCCCCCCCCCCCCCC(=O)OCC(CO)OC(=O)CCCCCCCCCCCCCCC"),
    ("PC", "CCCCCCCCCCCCCCCC(=O)OC[C@H](COP([O-])(=O)OCC[N+](C)(C)C)OC(=O)CCCCCCCCCCCCCCC"),
    ("PE", "CCCCCCCCCCCCCCCC(=O)OC[C@H](COP(O)(=O)OCCN)OC(=O)CCCCCCCCCCCCCCC"),
    ("ceramide", "CCCCCCCCCCCCC/C=C/[C@@H](O)[C@H](CO)NC(=O)CCCCCCCCCCCCCCC"),
    ("galactosylceramide",
     "CCCCCCCCCCCCC/C=C/[C@@H](O)[C@H](CO[C@@H]1O[C@H](CO)[C@H](O)[C@H](O)[C@H]1O)NC(=O)CCCCCCCCCCCCCCC"),
    ("galactosyldiacylglycerol",
     "CCCCCCCCCCCCCCCCCC(=O)OC[C@@H](OC(=O)CCCCCCCCCCCCCCCCC)CO[C@@H]1O[C@H](CO)[C@H](O)[C@H](O)[C@H]1O"),
]


@pytest.mark.parametrize("label,smiles", LIPID_CASES, ids=[c[0] for c in LIPID_CASES])
def test_lipid_roundtrip(label, smiles):
    name, ok = _roundtrips(smiles)
    assert ok, f"{label}: name={name!r} did not round-trip"


def test_non_lipid_unaffected():
    """A non-lipid ester must keep round-tripping (zero-regression guard)."""
    name, ok = _roundtrips("CCOC(=O)C")  # ethyl acetate
    assert ok, f"non-lipid regression: ethyl acetate name={name!r}"
