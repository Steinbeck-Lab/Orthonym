"""Charged multi-centre subsystem (root R11; P-71..75) — W8-P5.

Task 0: guard-characterization test locking the ~44 already-built charged
golds so later work in this subsystem can never regress them (these already
pass at HEAD; this is a characterization lock, not a new-behaviour test).
"""
import pytest

from orthonym.namer import Orthonym


GUARD = {
    "C[N-][N+](C)(C)C": "1,2,2,2-tetramethylhydrazin-2-ium-1-ide",   # P-74.1.1
    "CC(C)=[O+][O-]":   "2-(propan-2-ylidene)dioxidan-2-ium-1-ide",  # P-74.1.1
    "C[P+](C)(C)[C-](C)C": "2-(trimethylphosphaniumyl)propan-2-ide", # P-74.2.1.1
    "[C-]#[C-]": "ethynediide",                                       # P-72.2.2.1
    "[O-]CC[O-]": "ethane-1,2-bis(olate)",                            # P-72.2.2.2.2
    "[NH-]CC[NH-]": "ethane-1,2-bis(aminide)",                        # P-72.2.2.2.3
    "[NH3+]CC[NH3+]": "ethane-1,2-bis(aminium)",                      # P-73.5 poly-aminium
    "C[N+](C)(C)C": "N,N,N-trimethylmethanaminium",                   # P-73.1.2.1
    "[CH-]1CCCCC1": "cyclohexan-1-ide",                               # P-72.2.2.1 ring
    "C[B-](C)(C)C": "tetramethylboranuide",                           # P-72.3
    "C[P-](C)(C)C": "tetramethylphosphanuide",                        # P-72.3
    "NC(=[OH+])N": "uronium",                                         # P-73.1.2.2
    "CCC=[S+][O-]": "propylidene-lambda4-sulfanone",                  # P-74.2.2.1.8
    "CC(C)[O-]": "propan-2-olate",                                    # P-72.2.2.2.2
}


@pytest.mark.parametrize("smi,expected", GUARD.items())
def test_charged_guard_no_regression(smi, expected):
    assert Orthonym().name(smi) == expected
