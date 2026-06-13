"""OPSIN round-trip integration tests for the steroid α/β assembler (Phase 181, WSC-02).

Generate name → OPSIN parse → compare canonical SMILES to the original. RT is the
project's accuracy oracle. The 9 gold exemplars are conjugate-free so the RT-flip is
attributable to Phase 181 (conjugated steroids await Phase 182, Pitfall 5). The negative
guard confirms a non-steroid stays correct.

WAVE 0 CONTRACT: RED until Waves 1-2 build + wire the converter.
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


# The 9 conjugate-free α/β exemplars (181-00 <gold_smiles>) — RT attributable to Phase 181.
STEROID_CASES = [
    ("5alpha-cholestan-3beta-ol",
     "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC[C@H]4C[C@H](CC[C@]4(C)[C@H]3CC[C@]12C)O"),
    ("5alpha-cholestan-3alpha-ol",
     "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC[C@H]4C[C@@H](CC[C@]4(C)[C@H]3CC[C@]12C)O"),
    ("5beta-cholestan-3alpha-ol",
     "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC[C@@H]4C[C@@H](CC[C@]4(C)[C@H]3CC[C@]12C)O"),
    ("3beta-hydroxy-5alpha-androstan-17beta-ol",
     "O[C@@H]1C[C@@H]2CC[C@H]3[C@@H]4CC[C@@H]([C@@]4(C)CC[C@@H]3[C@]2(CC1)C)O"),
    ("androst-5-en-3beta-ol",
     "C[C@]12CC[C@H]3[C@@H](CC=C4C[C@@H](O)CC[C@]34C)[C@@H]1CCC2"),
    ("5alpha-pregnane-3beta,20-diol",
     "CC([C@H]1CC[C@H]2[C@@H]3CC[C@H]4C[C@H](CC[C@]4(C)[C@H]3CC[C@]12C)O)O"),
    ("(20R,22R)-cholest-5-ene-3beta,20,22-triol",
     "CC(C)CC[C@H]([C@@](C)([C@H]1CC[C@H]2[C@@H]3CC=C4C[C@H](CC[C@]4(C)[C@H]3CC[C@]12C)O)O)O"),
    ("5alpha-androstan-17beta-ol",
     "C[C@@]12[C@H](CC[C@H]1[C@@H]1CC[C@H]3CCCC[C@]3(C)[C@H]1CC2)O"),
    ("17beta-hydroxy-5alpha-androstan-3-one",
     "O=C1C[C@@H]2CC[C@H]3[C@@H]4CC[C@@H]([C@@]4(C)CC[C@@H]3[C@]2(CC1)C)O"),
]


@pytest.mark.parametrize("label,smiles", STEROID_CASES, ids=[c[0] for c in STEROID_CASES])
def test_steroid_roundtrip(label, smiles):
    name, ok = _roundtrips(smiles)
    assert ok, f"{label}: name={name!r} did not round-trip"


def test_non_steroid_unaffected():
    """A non-steroid ester must keep round-tripping (zero-regression guard)."""
    name, ok = _roundtrips("CCOC(=O)C")  # ethyl acetate
    assert ok, f"non-steroid regression: ethyl acetate name={name!r}"
