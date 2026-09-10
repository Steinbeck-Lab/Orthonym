"""``eval/audit_candidates.py``'s atom-drop class must exclude rows that SHIPPED.

Regression test for the precision defect found and fixed 2026-08-05. The first
best-effort a dev split run reported **78** atom-drop rows; **11 of them shipped a
constitutionally correct name** (9 ``CORRECT_EMITTED`` + 2 ``STEREO_ONLY``) and were not
defects at all. The honest class size is 67.

The mechanism is the field's own construction, which is why it needs a permanent test:
``classify`` builds ``mol_names`` from molecule-scope entries **excluding stage
'emitted'**, so on a row that shipped, the "best candidate" it measures is some lesser
intermediate rather than the name that actually went out. Filtering on the audit CLASS
would still be wrong -- that is what reported 69 -- because ``STEREO_ONLY`` also ships.
The filter must be on ``emitted_constitution_ok``.

No JVM: ``opsin_batch`` is a module-level name in ``audit_candidates``, so it is stubbed
with a name -> SMILES dict. That keeps this test in the fast suite, which matters because
the OPSIN pipe is what makes the broad suites deadlock.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "eval"))

import audit_candidates as A  # noqa: E402


def _ledger(*entries):
    """(site, stage, scope, name) tuples -> the dict shape ``classify`` consumes."""
    return [
        {"site": s, "stage": st, "scope": sc, "name": n, "detail": None, "depth": 0}
        for s, st, sc, n in entries
    ]


@pytest.fixture
def stub_opsin(monkeypatch):
    """Route every candidate name to a SMILES, so RDKit supplies real InChIKeys."""

    def _install(mapping):
        def fake_batch(names):
            return [mapping.get(n, "") for n in names]

        monkeypatch.setattr(A, "opsin_batch", fake_batch)

    return _install


def test_shipped_correct_row_is_not_counted_as_an_atom_drop(stub_opsin):
    """A row that emits the right constitution never enters the costly class, even
    though a smaller molecule-scope candidate was recorded for it."""
    # propan-1-ol (3 HA) shipped and is correct; a 1-carbon intermediate was also built.
    #
    # The correct name is recorded ONLY at stage 'emitted', which is what makes the
    # signature fire: ``classify`` excludes stage 'emitted' from ``mol_names``, so the
    # best *remaining* molecule-scope candidate is the 1-carbon one. That is exactly
    # the shape of the 9 real CORRECT_EMITTED rows. (If the correct name were also
    # recorded as 'produced' it would be the max-HA candidate, delta would be 0, and
    # there would be nothing to exclude -- a different case, not this one.)
    stub_opsin({"propan-1-ol": "CCCO", "methane": "C"})
    rows = [{
        "smiles": "CCCO",
        "emitted": "propan-1-ol",
        "error": None,
        "ledger": _ledger(
            ("chain", "produced", "molecule", "methane"),
            ("namer._finish", "emitted", "molecule", "propan-1-ol"),
        ),
    }]
    out, summary = A.classify(rows, verbose=False)
    (row,) = out

    assert row["audit_class"] == A.CORRECT_EMITTED
    assert row["emitted_constitution_ok"] is True
    # The signature still matches -- a smaller candidate really was built --
    # but it must not be charged as a cost.
    assert summary["atom_drop"]["n_rows_matching_signature"] == 1
    assert summary["atom_drop"]["n_rows"] == 0
    assert summary["atom_drop"]["n_rows_also_correct"] == 1


def test_stereo_only_row_is_excluded_too(stub_opsin):
    """The bug that made the honest number 67 rather than 69: STEREO_ONLY rows ship a
    constitutionally correct name, so a class-name filter misses them."""
    # Emitted name has the right constitution, wrong stereo; a truncated candidate exists.
    stub_opsin({
        "butan-2-ol": "CCC(C)O",           # same skeleton as the input, no stereo
        "propan-1-ol": "CCCO",             # the atom-short intermediate
    })
    rows = [{
        "smiles": "CC[C@H](C)O",
        "emitted": "butan-2-ol",
        "error": None,
        "ledger": _ledger(
            ("chain", "produced", "molecule", "propan-1-ol"),
            ("namer._finish", "emitted", "molecule", "butan-2-ol"),
        ),
    }]
    out, summary = A.classify(rows, verbose=False)
    (row,) = out

    assert row["audit_class"] == A.STEREO_ONLY
    assert row["emitted_constitution_ok"] is True, (
        "a stereo-only mismatch still shipped the right constitution"
    )
    assert summary["atom_drop"]["n_rows"] == 0, (
        "STEREO_ONLY ships; filtering on audit_class alone would count it and "
        "reproduce the 69-vs-67 error"
    )


def test_genuine_drop_is_counted_and_attributed_to_its_producer(stub_opsin):
    """The control: a row that abstained while an atom-short candidate was built IS the
    defect class, and blame goes to the PRODUCER, not to the gate that caught it."""
    stub_opsin({"methane": "C"})
    rows = [{
        "smiles": "COS(=O)(=O)O",          # 6 heavy atoms
        "emitted": "unknown organic compound",
        "error": None,
        "ledger": _ledger(
            ("chain", "produced", "molecule", "methane"),
            ("AbstentionCode.GATE_SUPPRESSED", "suppressed", "molecule", "methane"),
            ("namer._finish", "emitted", "molecule", "unknown organic compound"),
        ),
    }]
    out, summary = A.classify(rows, verbose=False)
    (row,) = out

    assert row["is_atom_drop"] is True
    assert row["emitted_constitution_ok"] is False
    assert row["atom_delta"] == -5, "methane covers 1 of 6 heavy atoms"
    assert row["candidate_producer"] == "chain", (
        "the producer built it; AbstentionCode.GATE_SUPPRESSED only caught it. "
        "Attributing to the suppressor credits every drop to SELF-01 and names "
        "no producer to fix."
    )
    assert summary["atom_drop"]["n_rows"] == 1
    assert summary["atom_drop"]["by_producer"] == {"chain": 1}
    assert summary["atom_drop"]["signature_precision"] == 1.0


def test_precision_is_published_whenever_the_signature_matches(stub_opsin):
    """Invariant 14: the tool must report precision, not just a count."""
    stub_opsin({"methane": "C", "propan-1-ol": "CCCO"})
    rows = [
        {   # genuine drop
            "smiles": "COS(=O)(=O)O",
            "emitted": "unknown organic compound",
            "error": None,
            "ledger": _ledger(("chain", "produced", "molecule", "methane")),
        },
        {   # shipped correct, smaller candidate also built
            "smiles": "CCCO",
            "emitted": "propan-1-ol",
            "error": None,
            "ledger": _ledger(
                ("chain", "produced", "molecule", "methane"),
                ("namer._finish", "emitted", "molecule", "propan-1-ol"),
            ),
        },
    ]
    _, summary = A.classify(rows, verbose=False)
    ad = summary["atom_drop"]

    assert ad["n_rows_matching_signature"] == 2
    assert ad["n_rows"] == 1
    assert ad["signature_precision"] == 0.5
