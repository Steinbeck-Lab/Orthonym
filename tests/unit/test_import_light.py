"""``import orthonym`` loads no library that naming does not need: ``requests`` (only the optional PubChem lookup uses
it, 0.12 s to import) and numpy (its OpenBLAS starts one spinning thread per CPU; ``test_cli_blas_threads.py``).
Each check runs in a fresh child process."""
import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit


def _child(code: str) -> str:
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip().splitlines()[-1]


def test_import_orthonym_does_not_import_requests():
    assert _child("import sys, orthonym; print('requests' in sys.modules)") == "False"


def test_the_pubchem_lookup_still_reaches_requests():
    assert _child("from orthonym.validation import pubchem_validator as p; print(p.requests.__name__)") == "requests"
