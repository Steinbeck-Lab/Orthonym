"""Phase 150 SC-2 - OPSIN round-trip integration tests (Tier 2).

Per CONTEXT D-08 Tier 2: sample 50 entries per source dict (cyclic, NP, aryl,
simple); 200+ round-trip tests total. Each test runs OPSIN CLI, computes
InChI L1 for both source SMILES and OPSIN parse output, asserts L1 match.

Determinism per Phase 145.2: sample is FROZEN as a sorted list at planning
time (seed=42), NOT re-sampled at test-run time.

Source: 150-CONTEXT.md D-04 + D-08 Tier 2.
Source: 150-RESEARCH.md section 10.2.
Source: 150-PATTERNS.md "NEW test_opsin_retained_roundtrip.py".
Source: https://iupac.qmul.ac.uk/BlueBook/P2.html (P-22 retained-name PIN tier).
"""

import os
import random
import subprocess

import pytest
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchi

from orthonym.data.opsin_imports import (
    OPSIN_ARYL_GROUPS,
    OPSIN_SIMPLE_GROUPS,
    OPSIN_CYCLIC_GROUPS,
    OPSIN_NATURAL_PRODUCTS,
)


# ---------------------------------------------------------------------------
# Helpers REUSED INLINE from tests/integration/test_opsin_roundtrip_validation.py
# (NOT cross-imported per CONTEXT D-08 Tier 2 acceptance: "helpers reused
# verbatim as inline definitions, NOT cross-test-file import")
# ---------------------------------------------------------------------------


def _java_available() -> bool:
    try:
        subprocess.run(["java", "-version"], capture_output=True, timeout=10)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _opsin_jar_path():
    candidates = [
        "opsin-cli-2.9.0-jar-with-dependencies.jar",
        "opsin/opsin-cli-2.9.0-jar-with-dependencies.jar",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def _opsin_name_to_smiles(name: str, jar_path: str) -> str:
    result = subprocess.run(
        ["java", "-jar", jar_path, "-osmi"],
        input=name,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout.strip()


def _inchi_match(smi1: str, smi2: str) -> bool:
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)
    if mol1 is None or mol2 is None:
        return False
    i1 = MolToInchi(mol1)
    i2 = MolToInchi(mol2)
    if i1 is None or i2 is None:
        return False

    def strip(x: str) -> str:
        return "/".join(p for p in x.split("/") if not p.startswith(("t", "b", "m", "s")))

    return strip(i1) == strip(i2)


# ---------------------------------------------------------------------------
# Frozen 50-entry sample per source (sorted + seed=42; Phase 145.2 determinism)
# ---------------------------------------------------------------------------


def _sample_source(source: dict, n: int = 50, seed: int = 42):
    """Sort items by SMILES key (canonicalize order), then seeded sample."""
    items = sorted(source.items())
    if not items:
        return []
    rng = random.Random(seed)
    return rng.sample(items, min(n, len(items)))


def _select_primary_name(names):
    """Match _build_retained_names::_select_primary_name (RESEARCH Pitfall 5)."""
    spaced = [n for n in names if " " in n]
    if spaced:
        return max(spaced, key=len)
    return names[0]


def _build_param(source_dict):
    """Build (smiles, name) parametrize tuples from sampled entries."""
    sample = _sample_source(source_dict, 50)
    params = []
    for key, meta in sample:
        if meta.get("subType") in ("saltComponent", "chalcogenide"):
            continue
        smiles = meta.get("smiles", key.split("||")[0] if "||" in key else key)
        names = meta.get("names", [])
        if names:
            params.append((smiles, _select_primary_name(names)))
    return params


ARYL_PARAMS = _build_param(OPSIN_ARYL_GROUPS)
SIMPLE_PARAMS = _build_param(OPSIN_SIMPLE_GROUPS)
CYCLIC_PARAMS = _build_param(OPSIN_CYCLIC_GROUPS)
NP_PARAMS = _build_param(OPSIN_NATURAL_PRODUCTS)


# ---------------------------------------------------------------------------
# Tests (4 parametrized; ~50 cases each → 200+ total per CONTEXT D-08)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize(
    "smiles,name", ARYL_PARAMS, ids=[n for _, n in ARYL_PARAMS]
)
def test_aryl_groups_roundtrip(smiles, name):
    """Phase 150 SC-2: aryl_groups entries round-trip via OPSIN L1.

    Source: https://iupac.qmul.ac.uk/BlueBook/P2.html P-22.1.x
    Source: 150-CONTEXT.md D-04 + D-08 Tier 2.
    """
    jar = _opsin_jar_path()
    if jar is None or not _java_available():
        pytest.skip("OPSIN CLI or Java not available")
    opsin_smi = _opsin_name_to_smiles(name, jar)
    if not opsin_smi or "unparsable" in opsin_smi.lower() or "ambiguous" in opsin_smi.lower():
        pytest.xfail(f"OPSIN cannot parse {name!r}: {opsin_smi}")
    assert _inchi_match(smiles, opsin_smi), (
        f"L1 mismatch for {name!r}: source {smiles!r} vs OPSIN {opsin_smi!r}"
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    "smiles,name", SIMPLE_PARAMS, ids=[n for _, n in SIMPLE_PARAMS]
)
def test_simple_groups_roundtrip(smiles, name):
    """Phase 150 SC-2: simple_groups entries round-trip via OPSIN L1.

    Source: https://iupac.qmul.ac.uk/BlueBook/P2.html P-22.x
    Source: 150-CONTEXT.md D-04 + D-08 Tier 2.
    """
    jar = _opsin_jar_path()
    if jar is None or not _java_available():
        pytest.skip("OPSIN CLI or Java not available")
    opsin_smi = _opsin_name_to_smiles(name, jar)
    if not opsin_smi or "unparsable" in opsin_smi.lower() or "ambiguous" in opsin_smi.lower():
        pytest.xfail(f"OPSIN cannot parse {name!r}: {opsin_smi}")
    assert _inchi_match(smiles, opsin_smi), (
        f"L1 mismatch for {name!r}: source {smiles!r} vs OPSIN {opsin_smi!r}"
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    "smiles,name", CYCLIC_PARAMS, ids=[n for _, n in CYCLIC_PARAMS]
)
def test_cyclic_groups_roundtrip(smiles, name):
    """Phase 150 SC-2: cyclic_groups entries round-trip via OPSIN L1.

    Source: https://iupac.qmul.ac.uk/BlueBook/P2.html P-22.2.1
    Source: 150-CONTEXT.md D-04 + D-08 Tier 2.
    """
    jar = _opsin_jar_path()
    if jar is None or not _java_available():
        pytest.skip("OPSIN CLI or Java not available")
    opsin_smi = _opsin_name_to_smiles(name, jar)
    if not opsin_smi or "unparsable" in opsin_smi.lower() or "ambiguous" in opsin_smi.lower():
        pytest.xfail(f"OPSIN cannot parse {name!r}: {opsin_smi}")
    assert _inchi_match(smiles, opsin_smi), (
        f"L1 mismatch for {name!r}: source {smiles!r} vs OPSIN {opsin_smi!r}"
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    "smiles,name", NP_PARAMS, ids=[n for _, n in NP_PARAMS]
)
def test_natural_products_roundtrip(smiles, name):
    """Phase 150 SC-2: natural_products entries round-trip via OPSIN L1.

    Source: https://iupac.qmul.ac.uk/BlueBook/P2.html (NP retained names)
    Source: 150-CONTEXT.md D-04 + D-08 Tier 2.
    """
    jar = _opsin_jar_path()
    if jar is None or not _java_available():
        pytest.skip("OPSIN CLI or Java not available")
    opsin_smi = _opsin_name_to_smiles(name, jar)
    if not opsin_smi or "unparsable" in opsin_smi.lower() or "ambiguous" in opsin_smi.lower():
        pytest.xfail(f"OPSIN cannot parse {name!r}: {opsin_smi}")
    assert _inchi_match(smiles, opsin_smi), (
        f"L1 mismatch for {name!r}: source {smiles!r} vs OPSIN {opsin_smi!r}"
    )
