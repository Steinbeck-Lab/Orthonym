"""P-22.2.7.1: λ-convention on Hantzsch-Widman heteromonocycles.

All expected PINs OPSIN-2.9.0-verified 2026-07-09 (ASCII 'lambda' spelling,
RDKit-canonical round-trip equality with the input SMILES).
"""
import pytest
from orthonym.namer import name_compound

LAMBDA_HW_CASES = [
    # (SMILES, expected PIN)  — BB P-22.2.7.1 examples verbatim
    ("[SH2]1C=CC=C1", "1H-1lambda4-thiophene"),
    ("[IH]1CCCCC1", "1lambda3-iodinane"),
    ("O1C=[PH2]C=C1", "1,3lambda5-oxaphosphole"),
    ("S1=CN=CC=C1", "1lambda4,3-thiazine"),
]


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", LAMBDA_HW_CASES)
def test_lambda_heteromonocycle_pin(smiles, expected):
    assert name_compound(smiles) == expected


@pytest.mark.unit
def test_standard_heteromonocycles_unchanged():
    # protects — recorded at HEAD 2026-07-09, all OPSIN-RT clean
    assert name_compound("c1ccsc1") == "thiophene"
    assert name_compound("C1CCOC1") == "oxolane"
    assert name_compound("c1ccncc1") == "pyridine"


@pytest.mark.unit
def test_partially_saturated_lambda_ring_fails_closed():
    # 2,3-dihydro form of 1lambda4-thiophene has sp3 ring CARBONS -> not a
    # mancude lambda parent; hydro-naming for lambda rings is unbuilt, so the
    # namer must refuse (P-31.1.4) — never emit a lambda-less or H-mislabeled name.
    assert "unknown" in name_compound("[SH2]1CCC=C1").lower()
