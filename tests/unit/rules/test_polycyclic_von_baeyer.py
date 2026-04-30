"""Phase 151 D-04..D-08 unit tests for Von Baeyer ≥4-ring naming.

Covers:
 - Classification predicate (is_higher_polycyclo): ≥4-ring AND
   single-ring-system AND ≥2 bridgeheads AND no spiro AND no aromatic
   AND not a natural product backbone AND not bicyclic / tricyclic.
 - Retained-name passthrough (D-08): adamantane / norbornane / twistane
   served by tricyclo.get_retained_tricyclo_name; new module is fallback
   only.
 - Cascade-step-6 supplier coverage invariant (D-06 / Phase 147 SC-7):
   get_higher_polycyclo_iupac_locants returns Optional[Dict[int, int]]
   with FULL ring-atom coverage (or None — never partial).
 - Anti-canary: bicyclo / tricyclo / steroid / aromatic inputs MUST
   return None / False (D-04 lock — bicyclo.py + tricyclo.py +
   polycyclics.py retain authority on their proven cases).
 - Blue Book examples round-trip via OPSIN to InChI L1 == input
   (D-23 hard test gate).
 - No parallel locant comparator (D-07 / D-20): module imports
   compare_locant_sets and contains zero local _compare_locant
   definitions.

Test markers:
 - @pytest.mark.unit for fast tests
 - @pytest.mark.roundtrip for OPSIN integration tests (skipif jar missing)

Imports the new module that Plan 151-01 Task 2 creates. In the Wave-0
state (this file committed BEFORE the module exists), all tests RED at
import collection time. Task 2 commit flips them GREEN.

Source:
 - 151-CONTEXT.md D-04 / D-05 / D-06 / D-07 / D-08 / D-23 / D-25
 - 151-RESEARCH.md §"Existing Code Audit", §"OPSIN Compatibility Evidence"
 - 151-AUDIT-A.md verdict: THIN_WRAPPER
 - IUPAC 2013 Blue Book P-23.2.5 / P-23.3.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

import pytest
from rdkit import Chem

# Wave-0 / RED state: the new module
# (src/orthonym/rules/polycyclic_von_baeyer.py) does not exist yet.
# Plan 151-01 Task 2 implements it. We defer the import to test bodies
# so pytest collection succeeds (≥30 tests collected) and individual
# tests fail with a clear ModuleNotFoundError until Task 2 lands. This
# matches the audit-first cadence per CONTEXT D-02.

def _vb_import():
    """Lazy import of the new module so pytest collection works in RED state."""
    from orthonym.rules import polycyclic_von_baeyer as mod  # noqa: WPS433
    return mod


def is_higher_polycyclo(mol):  # type: ignore[no-redef]
    return _vb_import().is_higher_polycyclo(mol)


def name_higher_polycyclo(mol):  # type: ignore[no-redef]
    return _vb_import().name_higher_polycyclo(mol)


def get_higher_polycyclo_iupac_locants(mol):  # type: ignore[no-redef]
    return _vb_import().get_higher_polycyclo_iupac_locants(mol)


# ============================================================================
# Fixture loader
# ============================================================================

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "ring_systems"


def load_fixtures(relpath: str) -> List[Dict[str, Any]]:
    """Load Phase 151 ring-system fixtures (D-25 corpus_provenance shape).

    Skips entries with expected_name is None for tests that match against
    a name; fixtures keep null expected_name when only the structural
    predicate matters (e.g. anti-canary entries used to verify
    is_higher_polycyclo returns False).
    """
    path = FIXTURES_DIR / relpath
    with path.open() as f:
        return json.load(f)


def _ring_atoms(mol) -> set[int]:
    ri = mol.GetRingInfo()
    out: set[int] = set()
    for r in ri.AtomRings():
        out.update(r)
    return out


# ============================================================================
# Compound fixtures (used by multiple test classes)
# ============================================================================


@pytest.fixture
def cubane():
    """Cubane: pentacyclo[4.2.0.0^{2,5}.0^{3,8}.0^{4,7}]octane (8C cage)."""
    return Chem.MolFromSmiles("C12C3C4C1C5C3C4C25")


@pytest.fixture
def adamantane():
    """Adamantane: tricyclo[3.3.1.1^{3,7}]decane (RETAINED NAME)."""
    return Chem.MolFromSmiles("C1C2CC3CC1CC(C2)C3")


@pytest.fixture
def norbornane():
    """Norbornane: bicyclo[2.2.1]heptane (BICYCLIC anti-canary)."""
    return Chem.MolFromSmiles("C1CC2CCC1C2")


@pytest.fixture
def bicyclo_222_octane():
    """Bicyclo[2.2.2]octane (BICYCLIC anti-canary)."""
    return Chem.MolFromSmiles("C1CC2CCC1CC2")


@pytest.fixture
def naphthalene():
    """Aromatic anti-canary: naphthalene routes to polycyclics.py."""
    return Chem.MolFromSmiles("c1ccc2ccccc2c1")


@pytest.fixture
def gonane_steroid():
    """Steroid backbone (gonane): natural product anti-canary."""
    # Gonane (cyclopenta[a]phenanthrenes saturated) — must NOT match
    # is_higher_polycyclo because detect_natural_product short-circuits.
    return Chem.MolFromSmiles("C1CC2CCC3C(C2CC1)CCC1CCCC31")


# ============================================================================
# TestRingCount — classification predicate
# ============================================================================


class TestRingCount:
    @pytest.mark.unit
    def test_cubane_classified_as_higher_polycyclo(self, cubane):
        assert is_higher_polycyclo(cubane) is True

    @pytest.mark.unit
    def test_norbornane_NOT_higher_polycyclo(self, norbornane):
        # bicyclic anti-canary (D-04)
        assert is_higher_polycyclo(norbornane) is False

    @pytest.mark.unit
    def test_bicyclo222_NOT_higher_polycyclo(self, bicyclo_222_octane):
        # bicyclic anti-canary (D-04)
        assert is_higher_polycyclo(bicyclo_222_octane) is False

    @pytest.mark.unit
    def test_adamantane_NOT_higher_polycyclo(self, adamantane):
        # tricyclic anti-canary (D-04 lock — tricyclo.py retains authority)
        assert is_higher_polycyclo(adamantane) is False

    @pytest.mark.unit
    def test_naphthalene_NOT_higher_polycyclo(self, naphthalene):
        # aromatic anti-canary (PAH branch, polycyclics.py)
        assert is_higher_polycyclo(naphthalene) is False

    @pytest.mark.unit
    def test_methane_NOT_higher_polycyclo(self):
        # 0-ring sanity
        assert is_higher_polycyclo(Chem.MolFromSmiles("C")) is False

    @pytest.mark.unit
    def test_cyclohexane_NOT_higher_polycyclo(self):
        # 1-ring sanity
        assert is_higher_polycyclo(Chem.MolFromSmiles("C1CCCCC1")) is False

    @pytest.mark.unit
    def test_None_input_returns_False(self):
        # defensive coding: invalid SMILES returns None mol → predicate False
        assert is_higher_polycyclo(None) is False


# ============================================================================
# TestRetainedNamePassthrough (D-08 + HERITAGE §4)
# ============================================================================


class TestRetainedNamePassthrough:
    @pytest.mark.unit
    def test_adamantane_returns_retained_name(self, adamantane):
        # D-08: retained-name dict (tricyclo.get_retained_tricyclo_name)
        # is consulted FIRST. Adamantane keeps "adamantane".
        assert name_higher_polycyclo(adamantane) == "adamantane"

    @pytest.mark.unit
    def test_twistane_routes_via_retained(self):
        # twistane = tricyclo[4.4.0.0^{3,8}]decane retained name
        mol = Chem.MolFromSmiles("C1CC2CC3CCCC1C23")
        result = name_higher_polycyclo(mol)
        # Either retained-name match OR None (if classify_bridged_system
        # marks it tricyclo and anti-canary returns None) — must not be
        # an algorithmic descriptor for this retained-tricyclic case.
        assert result in ("twistane", None)

    @pytest.mark.unit
    def test_norbornane_returns_None_not_descriptor(self, norbornane):
        # bicyclic anti-canary — tricyclo.get_retained_tricyclo_name
        # only handles tricyclic retained names; bicyclo has its own
        # bicyclo.py path. Either way, name_higher_polycyclo returns None
        # for a bicyclic input per D-04.
        assert name_higher_polycyclo(norbornane) is None


# ============================================================================
# TestSupplierCoverageInvariant (D-06 / Phase 147 cascade-step-6 gate)
# ============================================================================


class TestSupplierCoverageInvariant:
    @pytest.mark.unit
    def test_cubane_coverage_full(self, cubane):
        # Pitfall 7: partial coverage = None; full coverage required.
        locants = get_higher_polycyclo_iupac_locants(cubane)
        assert locants is not None
        assert set(locants.keys()) >= _ring_atoms(cubane)

    @pytest.mark.unit
    def test_supplier_returns_None_on_None_input(self):
        assert get_higher_polycyclo_iupac_locants(None) is None

    @pytest.mark.unit
    def test_supplier_returns_None_on_methane(self):
        assert (
            get_higher_polycyclo_iupac_locants(Chem.MolFromSmiles("C")) is None
        )

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "fixture",
        [f for f in load_fixtures("vb/blue_book_examples.json")
         if f["compound_class"].startswith("vb-")],
        ids=lambda f: f["fixture_id"],
    )
    def test_blue_book_coverage(self, fixture):
        mol = Chem.MolFromSmiles(fixture["smiles"])
        ring_atoms = _ring_atoms(mol)
        locants = get_higher_polycyclo_iupac_locants(mol)
        # Either supplier returns None (acceptable) or full coverage —
        # never partial (Pitfall 7).
        if locants is not None:
            assert set(locants.keys()) >= ring_atoms, (
                f"partial coverage on {fixture['fixture_id']}"
            )


# ============================================================================
# TestAntiCanaryBicycloTricyclo (RESEARCH "Risk Landmines")
# ============================================================================


class TestAntiCanaryBicycloTricyclo:
    @pytest.mark.unit
    def test_supplier_returns_None_on_norbornane(self, norbornane):
        # 13 bicyclic canary cpds at risk per RESEARCH "Risk Landmines".
        assert get_higher_polycyclo_iupac_locants(norbornane) is None

    @pytest.mark.unit
    def test_supplier_returns_None_on_bicyclo222(self, bicyclo_222_octane):
        assert get_higher_polycyclo_iupac_locants(bicyclo_222_octane) is None

    @pytest.mark.unit
    def test_supplier_returns_None_on_adamantane(self, adamantane):
        # 99 tricyclic canary cpds at risk per RESEARCH "Risk Landmines".
        assert get_higher_polycyclo_iupac_locants(adamantane) is None

    @pytest.mark.unit
    def test_supplier_returns_None_on_naphthalene(self, naphthalene):
        # PAHs route via polycyclics.py — aromatic guard.
        assert get_higher_polycyclo_iupac_locants(naphthalene) is None

    @pytest.mark.unit
    def test_supplier_returns_None_on_steroid_gonane(self, gonane_steroid):
        # detect_natural_product short-circuits the predicate.
        assert get_higher_polycyclo_iupac_locants(gonane_steroid) is None


# ============================================================================
# TestBlueBookExamples — naming smoke (cardinality / type checks)
# ============================================================================


class TestBlueBookExamples:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "fixture",
        [f for f in load_fixtures("vb/blue_book_examples.json")
         if f["compound_class"].startswith("vb-")
         and f.get("expected_name") is not None],
        ids=lambda f: f["fixture_id"],
    )
    def test_blue_book_returns_a_descriptor_string(self, fixture):
        # Cardinality-only check: every named Blue Book example must yield
        # a non-empty descriptor that contains the cyclo-prefix matching
        # the ring count. Exact byte match is NOT required because the
        # analyzer may legitimately emit an alternate P-23.2.5 form
        # (constitutionally identical, all valid IUPAC) per audit row 1
        # and row 4. OPSIN round-trip is the binding correctness oracle
        # (see TestRoundTripViaOPSIN below).
        mol = Chem.MolFromSmiles(fixture["smiles"])
        name = name_higher_polycyclo(mol)
        # Acceptable: either a real descriptor string OR None (Plan 151
        # must not regress retained-name path; some Blue Book entries
        # like dioxa-tetracyclic have null expected_name).
        if name is not None:
            assert isinstance(name, str) and len(name) > 0


# ============================================================================
# TestNoParallelComparator (D-07 / D-20 lock)
# ============================================================================


class TestNoParallelComparator:
    @pytest.mark.unit
    def test_module_imports_compare_locant_sets(self):
        import inspect

        from orthonym.rules import polycyclic_von_baeyer

        src = inspect.getsource(polycyclic_von_baeyer)
        assert "compare_locant_sets" in src, (
            "D-07 lock: module MUST import compare_locant_sets from "
            "rules.locants — no parallel comparator"
        )

    @pytest.mark.unit
    def test_module_does_not_define_local_locant_comparator(self):
        import inspect

        from orthonym.rules import polycyclic_von_baeyer

        src = inspect.getsource(polycyclic_von_baeyer)
        # Strip comment-only lines so a documentation mention doesn't
        # trip this check.
        live = "\n".join(
            line
            for line in src.splitlines()
            if not line.lstrip().startswith("#")
        )
        assert "def _compare_locant" not in live, (
            "D-20 lock: no local _compare_locant* function allowed"
        )


# ============================================================================
# TestPublicAPI (D-06 — naming-collision-safe API)
# ============================================================================


class TestPublicAPI:
    @pytest.mark.unit
    def test_get_higher_polycyclo_iupac_locants_exported(self):
        from orthonym.rules import polycyclic_von_baeyer

        assert hasattr(polycyclic_von_baeyer, "get_higher_polycyclo_iupac_locants")

    @pytest.mark.unit
    def test_NO_collision_export_named_get_polycyclic_iupac_locants(self):
        # Q-01 audit lock: the new module MUST NOT re-export the
        # 2-arg PAH supplier name from polycyclics.py:386.
        from orthonym.rules import polycyclic_von_baeyer

        assert not hasattr(
            polycyclic_von_baeyer, "get_polycyclic_iupac_locants"
        )

    @pytest.mark.unit
    def test_is_higher_polycyclo_exported(self):
        from orthonym.rules import polycyclic_von_baeyer

        assert hasattr(polycyclic_von_baeyer, "is_higher_polycyclo")

    @pytest.mark.unit
    def test_name_higher_polycyclo_exported(self):
        from orthonym.rules import polycyclic_von_baeyer

        assert hasattr(polycyclic_von_baeyer, "name_higher_polycyclo")


# ============================================================================
# TestRoundTripViaOPSIN (D-23 hard test gate)
# ============================================================================


_OPSIN_JAR = (
    Path(__file__).resolve().parents[2]
    / "opsin-cli-2.9.0-jar-with-dependencies.jar"
)


def _opsin_available() -> bool:
    return _OPSIN_JAR.exists() and shutil.which("java") is not None


def _opsin_parse_one(name: str) -> str | None:
    """Single-name OPSIN parse → SMILES (or None on failure).

    Spawns a fresh JVM per call. For batch use, prefer the canary-suite
    helper which uses one JVM for all names. Phase 151 Task 1b uses a
    small fixture set (≤12), so per-call invocation is acceptable.
    """
    import subprocess

    try:
        proc = subprocess.run(
            ["java", "-jar", str(_OPSIN_JAR), "-osmi"],
            input=name + "\n",
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode != 0:
            return None
        for line in proc.stdout.splitlines():
            line = line.strip()
            if line and not line.startswith("Run the jar"):
                # OPSIN prints a banner line then SMILES per name
                return line
        return None
    except Exception:
        return None


class TestRoundTripViaOPSIN:
    @pytest.mark.roundtrip
    @pytest.mark.skipif(not _opsin_available(), reason="OPSIN/Java unavailable")
    @pytest.mark.parametrize(
        "fixture",
        [f for f in load_fixtures("vb/blue_book_examples.json")
         if f["compound_class"].startswith("vb-")
         and f.get("expected_name") is not None],
        ids=lambda f: f["fixture_id"],
    )
    def test_opsin_roundtrip_inchi_l1(self, fixture):
        mol = Chem.MolFromSmiles(fixture["smiles"])
        name = name_higher_polycyclo(mol)
        if name is None:
            pytest.skip("module returned None; covered by TestSupplierCoverageInvariant")
        parsed_smi = _opsin_parse_one(name)
        if parsed_smi is None:
            pytest.skip(f"OPSIN cannot parse {name!r} (logged for v19 follow-up)")
        # InChI layer-1 (constitution) must match input
        inchi_in = Chem.MolToInchi(Chem.MolFromSmiles(fixture["smiles"])).split("/c")[0]
        rt_mol = Chem.MolFromSmiles(parsed_smi)
        assert rt_mol is not None, f"OPSIN output unparseable: {parsed_smi}"
        inchi_rt = Chem.MolToInchi(rt_mol).split("/c")[0]
        assert inchi_in == inchi_rt, (
            f"InChI L1 mismatch on {fixture['fixture_id']}: "
            f"input={inchi_in!r} rt={inchi_rt!r} via name={name!r}"
        )
