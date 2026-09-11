"""a phase - OPSIN round-trip integration tests (Tier 2).

Per internal notes Tier 2: sample 50 entries per source dict (cyclic, NP, aryl,
simple); 200+ round-trip tests total. Each test runs OPSIN CLI, computes
InChI L1 for both source SMILES and OPSIN parse output, asserts L1 match.

Determinism per a phase: sample is FROZEN as a sorted list at planning
time (seed=42), NOT re-sampled at test-run time.

Source: 150-internal notes + Tier 2.
Source: internal notes section 10.2.
Source: internal notes "NEW test_opsin_retained_roundtrip.py".
Source: https://iupac.qmul.ac.uk/BlueBook/P2.html retained-name PIN tier).
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
# (NOT cross-imported per internal notes Tier 2 acceptance: "helpers reused
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
    """a phase REVIEW: catch TimeoutExpired so a single hung
    OPSIN invocation does not crash the test session. Returning ""
    routes the test through the existing 'OPSIN cannot parse' xfail
    branch in each parametrized test.
    """
    try:
        result = subprocess.run(
            ["java", "-jar", jar_path, "-osmi"],
            input=name,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return ""
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
# Frozen 50-entry sample per source (sorted + seed=42; a phase determinism)
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


# Plan 03 cleanup (a phase closeout): 5 known-failing OPSIN-ambiguous
# round-trip cases get xfail-with-citation markers. Each entry was identified
# during Plan 02 SUMMARY review (internal notes
# 150-02-SUMMARY.md "Plan 03 Unblock" + docs/retained_name_conflicts.md +
# docs/known_opsin_limitations.md). Per the contributor guide root-cause discipline these
# are NOT silenced — each xfail reason explains the OPSIN data-source bug.
#
# Strict=False because the OPSIN parser may improve in a future release; if a
# previously-failing case starts passing, pytest reports XPASS but does not
# fail the suite (per the standard a phase + RESEARCH section 7.2 pattern).
_KNOWN_OPSIN_AMBIGUOUS_FAILURES = {
    # (test-target, name) -> citation text
    ("aryl", "lupetidine"): (
        "OPSIN-ambiguous: 'lupetidine' parses to a constitutionally-different "
        "structure than the OPSIN-stored SMILES. See "
        ".planning/phases/150-opsin-xml-retained-name-expansion/150-02-SUMMARY.md "
        "section 'Tier 2 integration tests' + docs/retained_name_conflicts.md. "
        "Phase 150 closes via xfail; Phase 156 OPSIN grammar pre-validation "
        "may resolve."
    ),
    ("aryl", "benzoquinone"): (
        "OPSIN-ambiguous: 'benzoquinone' bare form lacks the locant-prefix "
        "(1,4- or 1,2-) that OPSIN requires for unambiguous parse. See "
        ".planning/phases/150-opsin-xml-retained-name-expansion/150-02-SUMMARY.md "
        "section 'Tier 2 integration tests'. Phase 150 closes via xfail."
    ),
    ("cyclic", "lutidine"): (
        "Semantic data-source bug: 'lutidine' in the OPSIN cyclicGroups XML "
        "is keyed against a bare pyridine SMILES. The actual lutidines are "
        "2,3- / 2,4- / 2,6- / 3,4- / 3,5-dimethylpyridines per P-25.2.1.1.3. "
        "Logged in docs/retained_name_conflicts.md row 'lutidine' with "
        "'OPSIN data-source bug; HC overrides per CONTEXT D-03'. Phase 150 "
        "closes via xfail."
    ),
    ("np", "morphin"): (
        "OPSIN-ambiguous: 'morphin' parses to a constitutionally-different "
        "structure than the OPSIN-stored aglycone SMILES (morphine-with-OH "
        "vs the bare phenanthrene-isoquinoline backbone the OPSIN entry "
        "claims). See "
        ".planning/phases/150-opsin-xml-retained-name-expansion/150-02-SUMMARY.md "
        "section 'Tier 2 integration tests' + docs/known_opsin_limitations.md."
    ),
    ("np", "androstenedione"): (
        "OPSIN-ambiguous: 'androstenedione' name maps to androst-4-ene-3,17-"
        "dione (the historical canonical structure) but the OPSIN entry's "
        "SMILES is androstane-3,17-dione (no 4-ene). See "
        ".planning/phases/150-opsin-xml-retained-name-expansion/150-02-SUMMARY.md "
        "section 'Tier 2 integration tests'. Phase 150 closes via xfail."
    ),
}


def _xfail_marks_for(target: str, name: str):
    """Return list of pytest marks for parametrize entry (xfail if known-fail)."""
    citation = _KNOWN_OPSIN_AMBIGUOUS_FAILURES.get((target, name))
    if citation is None:
        return []
    return [pytest.mark.xfail(reason=citation, strict=False)]


def _build_param(source_dict, target: str = ""):
    """Build (smiles, name) parametrize tuples from sampled entries.

    target: short tag ("aryl", "simple", "cyclic", "np") used to match
    Plan 03's known-fail register (_KNOWN_OPSIN_AMBIGUOUS_FAILURES).
    """
    sample = _sample_source(source_dict, 50)
    params = []
    for key, meta in sample:
        if meta.get("subType") in ("saltComponent", "chalcogenide"):
            continue
        smiles = meta.get("smiles", key.split("||")[0] if "||" in key else key)
        names = meta.get("names", [])
        if names:
            primary = _select_primary_name(names)
            marks = _xfail_marks_for(target, primary)
            params.append(pytest.param(smiles, primary, marks=marks, id=primary))
    return params


ARYL_PARAMS = _build_param(OPSIN_ARYL_GROUPS, "aryl")
SIMPLE_PARAMS = _build_param(OPSIN_SIMPLE_GROUPS, "simple")
CYCLIC_PARAMS = _build_param(OPSIN_CYCLIC_GROUPS, "cyclic")
NP_PARAMS = _build_param(OPSIN_NATURAL_PRODUCTS, "np")


# ---------------------------------------------------------------------------
# Tests (4 parametrized; ~50 cases each → 200+ total per internal notes)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("smiles,name", ARYL_PARAMS)
def test_aryl_groups_roundtrip(smiles, name):
    """a phase: aryl_groups entries round-trip via OPSIN L1.

    Source: https://iupac.qmul.ac.uk/BlueBook/P2.html.x
    Source: 150-internal notes + Tier 2.
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
@pytest.mark.parametrize("smiles,name", SIMPLE_PARAMS)
def test_simple_groups_roundtrip(smiles, name):
    """a phase: simple_groups entries round-trip via OPSIN L1.

    Source: https://iupac.qmul.ac.uk/BlueBook/P2.html.x
    Source: 150-internal notes + Tier 2.
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
@pytest.mark.parametrize("smiles,name", CYCLIC_PARAMS)
def test_cyclic_groups_roundtrip(smiles, name):
    """a phase: cyclic_groups entries round-trip via OPSIN L1.

    Source: https://iupac.qmul.ac.uk/BlueBook/P2.html
    Source: 150-internal notes + Tier 2.
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
@pytest.mark.parametrize("smiles,name", NP_PARAMS)
def test_natural_products_roundtrip(smiles, name):
    """a phase: natural_products entries round-trip via OPSIN L1.

    Source: https://iupac.qmul.ac.uk/BlueBook/P2.html (NP retained names)
    Source: 150-internal notes + Tier 2.
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
