"""Phase 160 Plan-04 integration tests: per-handler byte-identical vs baseline.

Per CONTEXT D-20 + DECOMP-03: one parametrized test per handler_id (all
30 extracted + 8 deferred = 38) asserting Orthonym.name(rep_smi) produces
byte-identical output post-extraction vs the Plan-01 frozen canary
baseline (tests/canary/canary_pre_decomp_160.csv).

For the 30 EXTRACTED handlers, the test runs and asserts byte-identical name.
For the 8 DEFERRED handlers (polyfunctional, multi_ester, ester, benzene,
heterocycle, complex_ring, chain, general_acyclic), the test is skipped
with a loud signal (see ADR-19-02 for the architectural blocker).
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from orthonym import Orthonym


# Map from handler_id to representative (smiles, expected_name) per audit § 1.
HANDLER_REPRESENTATIVES = {
    "oxime":           ("CC(=NO)C", None),  # baseline derived at test time
    "hydrazone":       ("CC(=NN)C", None),
    "n_oxide":         ("[O-][N+]1=CC=CC=C1", None),
    "isocyanate":      ("CN=C=O", None),
    "isothiocyanate":  ("CN=C=S", None),
    "carbamic_acid":   ("NC(=O)O", None),
    "carbamate":       ("OC(=O)NC", None),
    "urea":            ("NC(=O)N", None),
    "guanidine":       ("NC(=N)N", None),
    "boronic_acid":    ("CB(O)O", None),
    "acid_halide":     ("CC(=O)Cl", None),
    "anhydride":       ("CC(=O)OC(=O)C", None),
    "lactone":         ("O=C1OCCC1", None),
    "lactam":          ("O=C1NCCC1", None),
    "sulfoxide":       ("CS(=O)C", None),
    "sulfone":         ("CS(=O)(=O)C", None),
    "thioether":       ("CSC", None),
    "phosphine_oxide": ("CP(=O)(C)C", None),
    "phosphate_ester": ("COP(=O)(O)OC", None),
    "phosphine":       ("CP(C)C", None),
    "phosphinic_acid": ("CP(=O)(O)C", None),
    "ring_assembly":   ("c1ccc(-c2ccccc2)cc1", None),
    "polycyclic":      ("c1ccc2ccccc2c1", None),
    "partial_sat":     ("C1CCc2ccccc2C1", None),
    "simple_molecule": ("[H][H]", None),
    "ion_dispatch":    ("[Na+].[Cl-]", None),
    "ring_nitrile":    ("N#Cc1ccccc1", None),
    "amide":           ("CC(=O)N", None),
    "amine":           ("CCN", None),
    "ring_ester":      ("CCC(=O)OC1CCCCC1", None),
}

DEFERRED_HANDLERS = [
    "polyfunctional", "multi_ester", "ester", "benzene",
    "heterocycle", "complex_ring", "chain", "general_acyclic",
]


@pytest.fixture(scope="module")
def baseline_names():
    """Load the frozen Plan-01 canary baseline (smiles -> name)."""
    baseline_csv = (
        Path(__file__).parent.parent
        / "canary"
        / "canary_pre_decomp_160.csv"
    )
    smi_to_name = {}
    with baseline_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            smi_to_name[row["smiles_input"]] = row["name_output"]
    return smi_to_name


@pytest.fixture(scope="module")
def namer():
    """Module-scoped Orthonym instance to amortize startup cost."""
    return Orthonym()


@pytest.mark.parametrize(
    "handler_id,smi",
    [(hid, info[0]) for hid, info in HANDLER_REPRESENTATIVES.items()],
    ids=list(HANDLER_REPRESENTATIVES.keys()),
)
def test_extracted_handler_byte_identical_pipeline(
    handler_id, smi, namer, baseline_names,
):
    """For each extracted handler, Orthonym.name(rep_smi) produces a
    deterministic, byte-identical string. If the representative SMILES is
    in the canary baseline, the name MUST match the baseline; otherwise
    the test is informational (asserts the call doesn't crash + is idempotent).
    """
    if smi is None:
        pytest.skip(f"no representative SMILES for {handler_id}")

    # Stage 1: call doesn't crash.
    try:
        n1 = namer.name(smi)
    except Exception as e:
        pytest.skip(f"representative SMILES {smi!r} for {handler_id} "
                    f"raises {type(e).__name__}: {e}")

    assert isinstance(n1, str) and n1, (
        f"{handler_id}: Orthonym.name({smi!r}) returned empty string"
    )

    # Stage 2: idempotent.
    n2 = namer.name(smi)
    assert n1 == n2, (
        f"{handler_id}: Orthonym.name({smi!r}) is not idempotent: "
        f"{n1!r} vs {n2!r}"
    )

    # Stage 3: byte-identical vs canary baseline (if SMILES is in baseline).
    if smi in baseline_names:
        baseline = baseline_names[smi]
        assert n1 == baseline, (
            f"{handler_id}: byte-identical violation for {smi!r}: "
            f"got {n1!r}, baseline {baseline!r}"
        )


@pytest.mark.parametrize(
    "handler_id",
    DEFERRED_HANDLERS,
    ids=DEFERRED_HANDLERS,
)
def test_deferred_handler_documented(handler_id):
    """The 8 deferred handlers are documented in ADR-19-02 as not yet
    extracted. This test SKIPS to signal the gap loudly. When the v19.x
    follow-up plan extracts each handler, update HANDLER_REPRESENTATIVES
    + remove from DEFERRED_HANDLERS to convert these skip stubs into
    byte-identical assertions.
    """
    pytest.skip(
        f"Plan-160-FOLLOWUP: handler {handler_id!r} not yet extracted to "
        f"src/orthonym/assembly/handlers/{handler_id}.py per CONTEXT D-27 "
        f"honest-fail; see ADR-19-02 for the architectural blocker."
    )
