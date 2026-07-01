"""
Task 1.5 / R1 — canonicalize retained-name keys at load (P-65.1.1.1 family).

Verifies that drifted SMILES keys in RETAINED_NAMES resolve correctly after
the load-time canonicalization fix in data/__init__.py.

Before the fix:
  - The stored key for oxalic acid is `OC(=O)C(O)=O` (non-canonical); the
    incoming SMILES `OC(=O)C(=O)O` canonicalizes to `O=C(O)C(=O)O`, which
    misses the lookup -> wrong anion path emits `dihydroxalate`.
  - Tartaric acid was absent entirely -> `dihydrotartrate`.
"""

import pytest
from orthonym.namer import name_compound


@pytest.mark.unit
def test_r1_drifted_retained_keys_resolve():
    """Drifted retained-name keys must resolve correctly end-to-end (R1 fix)."""
    # Oxalic acid: stored key was OC(=O)C(O)=O (non-canonical);
    # RDKit canonical is O=C(O)C(=O)O. After load-time canonicalization the
    # lookup hits and returns the Blue-Book retained name (P-65.1.1.1).
    result_oxalic = name_compound("OC(=O)C(=O)O", style="pin")
    assert result_oxalic == "oxalic acid", (
        f"Expected 'oxalic acid', got {result_oxalic!r}. "
        "Retained key was non-canonical; fix: canonicalize keys at load."
    )

    # Tartaric acid: was absent from retained names -> wrong anion path.
    # Either the retained name or the systematic name is acceptable.
    result_tartaric = name_compound("OC(=O)C(O)C(O)C(=O)O", style="pin")
    assert result_tartaric in ("tartaric acid", "2,3-dihydroxybutanedioic acid"), (
        f"Expected 'tartaric acid' or '2,3-dihydroxybutanedioic acid', "
        f"got {result_tartaric!r}."
    )
