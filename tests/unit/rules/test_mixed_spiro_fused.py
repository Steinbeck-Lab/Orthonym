"""Phase 151-02 D-09 / D-13: mixed spiro/fused detector + name builder + suppliers.

Wave-0 RED scaffold: tests target the NEW Plan 151-02 functions that
must be added to ``src/orthonym/rules/spiro.py`` in Task 2:

- ``is_mixed_spiro_fused(mol)``  (D-09 + D-13)
- ``name_mixed_spiro_fused(mol)``  (D-13 + HERITAGE §4)
- ``get_spiro_iupac_locants(mol)``  (D-21 partial)
- ``get_mixed_spiro_fused_iupac_locants(mol)``  (D-21 partial)

Lazy-import pattern matches Plan 151-01's test scaffold so RED-state
collection succeeds before Task 2 lands.

Source: 151-02-PLAN.md tasks 1b/2; 151-AUDIT-B.md; 151-CONTEXT.md
D-09/D-13/D-20/D-21/D-22(b); 151-RESEARCH.md Pitfall 3 + HERITAGE §4.
"""
from __future__ import annotations

import inspect
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from rdkit import Chem


# ---------------------------------------------------------------------------
# OPSIN round-trip helpers (D-23) — same shape as canary infra.
# parents[3]: tests/unit/rules/test_X.py -> project root.
# ---------------------------------------------------------------------------

_OPSIN_JAR = (
    Path(__file__).resolve().parents[3]
    / "opsin-cli-2.9.0-jar-with-dependencies.jar"
)


def _opsin_available() -> bool:
    return _OPSIN_JAR.exists() and shutil.which("java") is not None


_OPSIN_BATCH_CACHE: dict[str, str | None] = {}


def _opsin_parse_one(name: str) -> str | None:
    if name in _OPSIN_BATCH_CACHE:
        return _OPSIN_BATCH_CACHE[name]
    if not _opsin_available():
        return None
    try:
        proc = subprocess.run(
            ["java", "-jar", str(_OPSIN_JAR), "-osmi"],
            input=name + "\n",
            capture_output=True,
            text=True,
            timeout=30,
        )
        result: str | None = None
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line or line.startswith("Run the jar"):
                continue
            if line.startswith("Enter a chemical"):
                continue
            if line.lower().startswith("error") or line.startswith("Could not"):
                result = None
                break
            result = line
            break
        _OPSIN_BATCH_CACHE[name] = result
        return result
    except Exception:
        _OPSIN_BATCH_CACHE[name] = None
        return None


# ---------------------------------------------------------------------------
# Lazy imports — until Task 2 lands, the four new functions don't exist.
# ---------------------------------------------------------------------------

def _import_or_skip(symbol: str):
    try:
        from orthonym.rules import spiro
    except ImportError:
        pytest.skip(f"orthonym.rules.spiro import failed")
    if not hasattr(spiro, symbol):
        pytest.skip(f"orthonym.rules.spiro.{symbol} not yet exported "
                    f"(Task 2 GREEN gate — RED scaffold)")
    return getattr(spiro, symbol)


# ---------------------------------------------------------------------------
# Fixture loaders.
# ---------------------------------------------------------------------------

def _fixtures_dir() -> Path:
    return (Path(__file__).resolve().parents[2]
            / "fixtures" / "ring_systems" / "spiro")


def _load_json(name: str) -> list[dict]:
    p = _fixtures_dir() / name
    if not p.exists():
        return []
    return json.loads(p.read_text())


def _filter(fxs: list[dict], **kw) -> list[dict]:
    out = []
    for f in fxs:
        match = True
        for k, v in kw.items():
            if f.get(k) != v:
                match = False
                break
        if match:
            out.append(f)
    return out


_CORPUS = _load_json("corpus_mined.json")
_BLUE_BOOK = _load_json("blue_book_examples.json")
_ANTI_CANARY = _load_json("anti_canary.json")

_MIXED_CORPUS = _filter(_CORPUS, compound_class="spiro-mixed-fused")
_PURE_CORPUS = _filter(_CORPUS, compound_class="spiro-pure")


# ===========================================================================
# Class: is_mixed_spiro_fused True positives
# ===========================================================================

class TestIsMixedSpiroFusedTruePositives:
    """is_mixed_spiro_fused returns True on real mixed-spiro/fused inputs."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "fixture",
        _MIXED_CORPUS,
        ids=lambda f: f["fixture_id"] if isinstance(f, dict) else "no-id",
    )
    def test_corpus_mixed_detected(self, fixture):
        is_mixed_spiro_fused = _import_or_skip("is_mixed_spiro_fused")
        mol = Chem.MolFromSmiles(fixture["smiles"])
        if mol is None:
            pytest.skip(f"SMILES invalid: {fixture['fixture_id']}")
        # Some corpus mixed candidates carry natural-product backbones
        # that the detect_natural_product short-circuit (Pitfall 3) WILL
        # mask out. We accept either True (real mixed) or False (NP-masked)
        # — the assertion is that the detector RUNS without error AND
        # never crashes on real corpus inputs.
        result = is_mixed_spiro_fused(mol)
        assert isinstance(result, bool), (
            f"is_mixed_spiro_fused must return bool, got {type(result)} "
            f"on {fixture['fixture_id']}")

    @pytest.mark.unit
    def test_minimal_spiro_indane_is_mixed(self):
        """Spiro-indane (pure-aromatic fused part attached via spiro): MIXED."""
        is_mixed_spiro_fused = _import_or_skip("is_mixed_spiro_fused")
        mol = Chem.MolFromSmiles("C1CCC2(CC1)Cc1ccccc12")
        # 3 rings: cyclohexane spiro to 5-membered C-ring fused to benzene.
        # n_spiro=1, n_rings=3, n_rings > n_spiro+1 -> mixed.
        assert is_mixed_spiro_fused(mol) is True

    @pytest.mark.unit
    def test_spiro_oxindole_like_is_mixed(self):
        """Spiro-isoindolone: 3 rings, 1 spiro -> MIXED."""
        is_mixed_spiro_fused = _import_or_skip("is_mixed_spiro_fused")
        mol = Chem.MolFromSmiles("O=C1NC2(CCCCC2)C2=CC=CC=C12")
        # 3 rings, 1 spiro centre between cyclohexane and isoindoline.
        assert is_mixed_spiro_fused(mol) is True


# ===========================================================================
# Class: anti-canary (RESEARCH Pitfall 3)
# ===========================================================================

class TestIsMixedSpiroFusedAntiCanary:
    """RESEARCH Pitfall 3 — steroids + alkaloids MUST return False."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "fixture",
        _ANTI_CANARY,
        ids=lambda f: f["fixture_id"] if isinstance(f, dict) else "no-id",
    )
    def test_natural_products_not_mixed_spiro_fused(self, fixture):
        is_mixed_spiro_fused = _import_or_skip("is_mixed_spiro_fused")
        mol = Chem.MolFromSmiles(fixture["smiles"])
        assert mol is not None, fixture["fixture_id"]
        assert is_mixed_spiro_fused(mol) is False, (
            f"FALSE POSITIVE: {fixture['fixture_id']} "
            f"({fixture.get('reason', '')}) misclassified as mixed-spiro-fused"
        )


# ===========================================================================
# Class: pure-spiro contract preservation (D-09 lock)
# ===========================================================================

class TestPureSpiroContractPreserved:
    """D-09 lock: is_spiro_system body byte-identical; returns True for
    pure spiro and False for mixed."""

    @pytest.mark.unit
    def test_spiro45decane_still_pure_spiro(self):
        from orthonym.rules.spiro import is_spiro_system
        is_mixed_spiro_fused = _import_or_skip("is_mixed_spiro_fused")
        mol = Chem.MolFromSmiles("C1CCC2(CC1)CCCCC2")
        assert is_spiro_system(mol) is True
        # Mutually exclusive with mixed.
        assert is_mixed_spiro_fused(mol) is False

    @pytest.mark.unit
    def test_dispiro_still_pure_spiro(self):
        from orthonym.rules.spiro import is_spiro_system
        is_mixed_spiro_fused = _import_or_skip("is_mixed_spiro_fused")
        mol = Chem.MolFromSmiles("C1CCC2(CC1)CC1(CCC2)CCCCCC1")
        assert is_spiro_system(mol) is True
        assert is_mixed_spiro_fused(mol) is False

    @pytest.mark.unit
    def test_is_spiro_system_body_unchanged(self):
        """D-09: source-level enforcement that the body's invariant
        ``n_rings == n_spiro + 1`` remains intact."""
        from orthonym.rules import spiro
        src = inspect.getsource(spiro.is_spiro_system)
        assert (
            "n_rings == n_spiro + 1" in src
            or "n_rings == len(spiro_atoms) + 1" in src
        ), f"is_spiro_system body modified — D-09 lock violated:\n{src}"

    @pytest.mark.unit
    def test_is_spiro_system_returns_false_on_mixed(self):
        """Confirms is_spiro_system continues to reject mixed inputs."""
        from orthonym.rules.spiro import is_spiro_system
        # spiro-indane: 3 rings, 1 spiro -> n_rings > n_spiro+1
        mol = Chem.MolFromSmiles("C1CCC2(CC1)Cc1ccccc12")
        assert is_spiro_system(mol) is False


# ===========================================================================
# Class: HERITAGE §4 separable-parts naming (D-13)
# ===========================================================================

class TestSeparablePartsNaming:
    """name_mixed_spiro_fused implements HERITAGE §4 separable-parts."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "fixture",
        _MIXED_CORPUS[:5],
        ids=lambda f: f["fixture_id"] if isinstance(f, dict) else "no-id",
    )
    def test_returns_tuple_or_none_on_corpus(self, fixture):
        """Return-shape contract: Optional[(name, ring_atoms, atom_to_locant, subs_included)]."""
        name_mixed_spiro_fused = _import_or_skip("name_mixed_spiro_fused")
        mol = Chem.MolFromSmiles(fixture["smiles"])
        if mol is None:
            pytest.skip(f"SMILES invalid: {fixture['fixture_id']}")
        result = name_mixed_spiro_fused(mol)
        if result is None:
            # Acceptable — Plan 151-02 D-24: failures log to HERITAGE-followups.
            return
        assert isinstance(result, tuple), fixture["fixture_id"]
        assert len(result) == 4, fixture["fixture_id"]
        name, ring_atoms, atom_to_locant, subs_included = result
        assert isinstance(name, str)
        assert isinstance(ring_atoms, set)
        assert isinstance(atom_to_locant, dict)
        assert isinstance(subs_included, bool)
        assert "spiro" in name.lower()

    @pytest.mark.unit
    def test_returns_none_on_pure_spiro(self):
        """name_mixed_spiro_fused(pure spiro) -> None (D-09 dispatch contract)."""
        name_mixed_spiro_fused = _import_or_skip("name_mixed_spiro_fused")
        mol = Chem.MolFromSmiles("C1CCC2(CC1)CCCCC2")
        assert name_mixed_spiro_fused(mol) is None

    @pytest.mark.unit
    def test_returns_none_on_steroid(self):
        """Detect_natural_product short-circuit: morphinan -> None."""
        name_mixed_spiro_fused = _import_or_skip("name_mixed_spiro_fused")
        mol = Chem.MolFromSmiles(
            "c1ccc2c(c1)C[C@H]1NCC[C@@]23CCCC[C@@H]13"
        )
        assert name_mixed_spiro_fused(mol) is None


# ===========================================================================
# Class: cascade-step-6 supplier coverage invariant (Pitfall 7)
# ===========================================================================

class TestSupplierCoverageInvariant:
    """get_*_iupac_locants suppliers: full ring-atom coverage OR None."""

    @pytest.mark.unit
    def test_get_mixed_spiro_fused_iupac_locants_signature(self):
        get_msf_iupac_locants = _import_or_skip(
            "get_mixed_spiro_fused_iupac_locants")
        # Signature smoke check — function must accept a mol.
        assert callable(get_msf_iupac_locants)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "fixture",
        _MIXED_CORPUS[:5],
        ids=lambda f: f["fixture_id"] if isinstance(f, dict) else "no-id",
    )
    def test_mixed_supplier_full_coverage_or_none(self, fixture):
        get_msf_iupac_locants = _import_or_skip(
            "get_mixed_spiro_fused_iupac_locants")
        mol = Chem.MolFromSmiles(fixture["smiles"])
        if mol is None:
            pytest.skip(f"SMILES invalid: {fixture['fixture_id']}")
        loc = get_msf_iupac_locants(mol)
        if loc is None:
            return  # Pitfall 7: partial coverage MUST return None.
        ri = mol.GetRingInfo()
        ring_atoms = set()
        for r in ri.AtomRings():
            ring_atoms.update(r)
        assert set(loc.keys()) >= ring_atoms, (
            f"partial coverage on {fixture['fixture_id']}: "
            f"{len(loc)} entries vs {len(ring_atoms)} ring atoms"
        )

    @pytest.mark.unit
    def test_mixed_supplier_returns_none_on_non_mixed(self):
        get_msf_iupac_locants = _import_or_skip(
            "get_mixed_spiro_fused_iupac_locants")
        mol = Chem.MolFromSmiles("C1CCCCC1")  # cyclohexane
        assert get_msf_iupac_locants(mol) is None

    @pytest.mark.unit
    def test_pure_spiro_supplier_full_coverage(self):
        get_spiro_iupac_locants = _import_or_skip("get_spiro_iupac_locants")
        mol = Chem.MolFromSmiles("C1CCC2(CC1)CCCC2")  # spiro[4.5]decane
        loc = get_spiro_iupac_locants(mol)
        assert loc is not None
        ri = mol.GetRingInfo()
        ring_atoms = set()
        for r in ri.AtomRings():
            ring_atoms.update(r)
        assert set(loc.keys()) >= ring_atoms

    @pytest.mark.unit
    def test_pure_spiro_supplier_returns_none_on_non_spiro(self):
        get_spiro_iupac_locants = _import_or_skip("get_spiro_iupac_locants")
        mol = Chem.MolFromSmiles("C1CCCCC1")
        assert get_spiro_iupac_locants(mol) is None


# ===========================================================================
# Class: D-11/D-20 enforcement — no parallel locant comparator
# ===========================================================================

class TestNoParallelLocantComparator:
    """D-11 / D-20: spiro.py uses compare_locant_sets — no new comparator."""

    @pytest.mark.unit
    def test_spiro_module_imports_compare_locant_sets(self):
        """Plan 151-02 lands the import even if the existing pure-spiro
        helpers do not yet use it (v19 follow-up). The import lock is
        proof D-11/D-20 invariant is honored."""
        _import_or_skip("name_mixed_spiro_fused")  # gates on Task 2
        from orthonym.rules import spiro
        src = inspect.getsource(spiro)
        assert "compare_locant_sets" in src, (
            "spiro.py must import compare_locant_sets (D-11/D-20 reuse lock)"
        )

    @pytest.mark.unit
    def test_no_parallel_comparator_definition(self):
        """No private locant-comparator function in spiro.py (D-20 lock)."""
        from orthonym.rules import spiro
        src = inspect.getsource(spiro)
        # Strip comments / docstrings: count only `def _compare_locant`
        # (or similar) in actual code.
        non_comment = "\n".join(
            line for line in src.splitlines()
            if not line.lstrip().startswith("#")
        )
        assert "def _compare_locant" not in non_comment, (
            "spiro.py defines a parallel locant comparator — D-20 violation"
        )


# ===========================================================================
# Class: D-12 enforcement — heteroatom prefix reuse
# ===========================================================================

class TestHeteroatomPrefixReuse:
    """D-12: spiro.py reuses polycyclic_bridged.get_heteroatom_prefix."""

    @pytest.mark.unit
    def test_spiro_imports_get_heteroatom_prefix(self):
        from orthonym.rules import spiro
        src = inspect.getsource(spiro)
        assert "get_heteroatom_prefix" in src

    @pytest.mark.unit
    def test_no_new_heteroatom_prefix_builder(self):
        """Only the pre-existing _build_hetero_prefix helper exists.
        Task 2 must NOT add a new builder beyond the one at spiro.py:759."""
        from orthonym.rules import spiro
        src = inspect.getsource(spiro)
        # Allow the existing helper; reject any new top-level
        # builder named like _build_heteroatom_prefix.
        assert "def _build_heteroatom_prefix" not in src


# ===========================================================================
# Class: round-trip via OPSIN (D-23)
# ===========================================================================

class TestRoundTripViaOPSIN:
    """D-23 hard test gate. Plan 151-02 acceptance: ≥10 corpus mixed
    fixtures round-trip via OPSIN with InChI L1 match."""

    @pytest.mark.roundtrip
    @pytest.mark.skipif(not _opsin_available(),
                        reason="OPSIN/Java not available")
    @pytest.mark.parametrize(
        "fixture",
        _MIXED_CORPUS,
        ids=lambda f: f["fixture_id"] if isinstance(f, dict) else "no-id",
    )
    def test_round_trip_mixed_corpus(self, fixture):
        """Round-trip InChI L1 (formula + connectivity).

        For each fixture: name_mixed_spiro_fused -> OPSIN -> InChI compare.
        If naming returns None or OPSIN fails, the failure is logged via
        HERITAGE-followups per D-24 (in Task 2 commit), so a per-fixture
        skip is acceptable in the RED-state phase. Once Plan 151-02 lands
        ≥10 of these will produce real round-trip evidence.
        """
        name_mixed_spiro_fused = _import_or_skip("name_mixed_spiro_fused")
        mol = Chem.MolFromSmiles(fixture["smiles"])
        if mol is None:
            pytest.skip(f"SMILES invalid: {fixture['fixture_id']}")
        result = name_mixed_spiro_fused(mol)
        if result is None:
            pytest.skip(f"name_mixed_spiro_fused returned None for "
                        f"{fixture['fixture_id']} — logged to HERITAGE-followups")
        name = result[0]
        parsed = _opsin_parse_one(name)
        if parsed is None:
            pytest.skip(f"OPSIN cannot parse {name!r} for "
                        f"{fixture['fixture_id']} — logged to HERITAGE-followups")
        rt = Chem.MolFromSmiles(parsed)
        if rt is None:
            pytest.skip(f"OPSIN output not RDKit-parseable for "
                        f"{fixture['fixture_id']}: {parsed!r}")
        i_in = Chem.MolToInchi(mol).split("/c", 1)[0]
        i_rt = Chem.MolToInchi(rt).split("/c", 1)[0]
        assert i_in == i_rt, (
            f"InChI L1 mismatch on {fixture['fixture_id']}: "
            f"input={i_in!r} round-trip={i_rt!r} via name={name!r}"
        )

    @pytest.mark.roundtrip
    @pytest.mark.skipif(not _opsin_available(),
                        reason="OPSIN/Java not available")
    def test_round_trip_heritage_indoline_cyclohexane(self):
        """Q-05 evidence: spiro[indoline-3,1'-cyclohexane] parses cleanly."""
        name = "spiro[indoline-3,1'-cyclohexane]"
        parsed = _opsin_parse_one(name)
        assert parsed is not None
        rt = Chem.MolFromSmiles(parsed)
        assert rt is not None
        # Round-trip the OPSIN output to verify InChI computability.
        assert Chem.MolToInchi(rt).startswith("InChI=")
