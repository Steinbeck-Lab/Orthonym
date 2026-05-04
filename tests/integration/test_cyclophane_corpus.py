"""Phase 155.A integration tests: cyclophane corpus + Blue Book fixtures with
OPSIN layer-1 InChI round-trip per 155-CONTEXT.md D-15.

Wave 0 scaffold: the OPSIN-RT helper trio (`_opsin_available`, `_opsin_parse`,
`_inchi_l1`) is copied verbatim from
``tests/integration/test_skeletal_replacement_corpus.py`` (Phase 154 D-22 idiom).
Fixtures load from ``tests/fixtures/cyclophane/{blue_book_examples,corpus_mined}.json``;
Task 2 audit populates them.  The parametrize-over-empty-list contract keeps
pytest collection clean while the JSON files are empty stubs.

Source: 155-CONTEXT.md D-15 (OPSIN-RT mandatory); 155-PATTERNS.md
"tests/integration/test_cyclophane_corpus.py".
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from rdkit import Chem

from orthonym import name_compound


_OPSIN_JAR = (
    Path(__file__).resolve().parents[2]
    / "opsin-cli-2.9.0-jar-with-dependencies.jar"
)


def _opsin_available() -> bool:
    return _OPSIN_JAR.exists() and shutil.which("java") is not None


def _opsin_parse(name: str) -> str | None:
    if not _opsin_available():
        return None
    try:
        proc = subprocess.run(
            ["java", "-jar", str(_OPSIN_JAR), "-osmi"],
            input=name + "\n",
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception:
        return None
    out = proc.stdout.strip().splitlines()
    if not out:
        return None
    last = out[-1].strip()
    low = last.lower()
    if "unparsable" in low or "is unparsable" in low or last.startswith("Run the jar"):
        return None
    return last


def _inchi_l1(smiles: str) -> str:
    """Return InChI layer-1 (skeleton; strips connectivity onward).

    Per Phase 151 D-23 + 155-CONTEXT.md D-15: the ``/c`` split isolates the
    formula portion of the InChI string -- the binding correctness oracle for
    skeletal nomenclature.  Stereo / charge / isotope layers (everything
    after ``/c``) are intentionally dropped.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    inchi = Chem.MolToInchi(mol)
    return inchi.split("/c")[0]


_FIXTURE_DIR = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "fixtures"
    / "cyclophane"
)
_BLUE_BOOK_FIXTURES = json.loads((_FIXTURE_DIR / "blue_book_examples.json").read_text())
_CORPUS_MINED_FIXTURES = json.loads((_FIXTURE_DIR / "corpus_mined.json").read_text())
_CORPUS_FIXTURES = list(_BLUE_BOOK_FIXTURES) + list(_CORPUS_MINED_FIXTURES)


@pytest.mark.skipif(not _opsin_available(), reason="OPSIN/Java not available")
@pytest.mark.integration
@pytest.mark.skip(
    reason="Phase 155-01 Task 3 fills this -- fixtures populated by Task 2 audit"
)
@pytest.mark.parametrize(
    "fixture",
    _CORPUS_FIXTURES,
    ids=[f.get("fixture_id", f"row_{i}") for i, f in enumerate(_CORPUS_FIXTURES)],
)
def test_cyclophane_corpus_opsin_roundtrip(fixture):
    """D-15: name_compound(smi) -> OPSIN -> InChI L1 == input InChI L1.

    Task 3 will un-skip this test and exercise it on the populated fixtures.
    """
    raise AssertionError("Task 3 replaces this stub.")
