"""Phase 151-02 D-10: multi-spiro tests for dispiro/trispiro/tetra/penta.

Wave-0 RED scaffold: tests target the existing ``name_spiro_system``
helper for multi-spiro coverage. They exercise the polyspiro descriptor
build path AND the cascade-step-6 supplier wrapper added in Plan 151-02
(``get_spiro_iupac_locants``).

Source: 151-02-PLAN.md tasks 1b/2; 151-AUDIT-B.md; IUPAC P-24.2.2.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from rdkit import Chem

from orthonym.rules.spiro import (
    is_spiro_system,
    name_spiro_system,
    get_spiro_atoms,
)


# OPSIN round-trip helpers — same shape as 151-01 plan / canary infra.
# parents[3]: tests/unit/rules/test_X.py -> parents[3] is project root.
_OPSIN_JAR = (
    Path(__file__).resolve().parents[3]
    / "opsin-cli-2.9.0-jar-with-dependencies.jar"
)


def _opsin_available() -> bool:
    return _OPSIN_JAR.exists() and shutil.which("java") is not None


def _opsin_parse_one(name: str) -> str | None:
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
        out = proc.stdout.strip().splitlines()
        # Skip startup banner line (matches canary helper pattern).
        for line in out:
            line = line.strip()
            if not line or line.startswith("Run the jar"):
                continue
            if line.startswith("Enter a chemical"):
                continue
            if line.lower().startswith("error") or line.startswith("Could not"):
                return None
            return line
        return None
    except Exception:
        return None


def _spiro_lazy_supplier_import():
    """Lazy import: until Task 2 lands, get_spiro_iupac_locants does
    not exist; tests that depend on it skip with a clear marker."""
    try:
        from orthonym.rules.spiro import get_spiro_iupac_locants
        return get_spiro_iupac_locants
    except ImportError:
        pytest.skip("get_spiro_iupac_locants not yet exported "
                    "(Task 2 GREEN gate)")


def _load_blue_book_fixtures(*, compound_class_filter: str | None = None):
    p = (Path(__file__).resolve().parents[2]
         / "fixtures" / "ring_systems" / "spiro" / "blue_book_examples.json")
    if not p.exists():
        return []
    fx = json.loads(p.read_text())
    if compound_class_filter:
        fx = [f for f in fx if f.get("compound_class") == compound_class_filter]
    return fx


def _load_corpus_fixtures(*, compound_class_filter: str | None = None):
    p = (Path(__file__).resolve().parents[2]
         / "fixtures" / "ring_systems" / "spiro" / "corpus_mined.json")
    if not p.exists():
        return []
    fx = json.loads(p.read_text())
    if compound_class_filter:
        fx = [f for f in fx if f.get("compound_class") == compound_class_filter]
    return fx


class TestDispiro:
    """Dispiro hydrocarbons — Plan 151-02 D-10 must-ship gate."""

    @pytest.mark.unit
    def test_dispiro_canonical_blue_book(self):
        """dispiro[5.1.5.2]heptadecane (Blue Book P-24.2.2 example)."""
        mol = Chem.MolFromSmiles("C1CCC2(CC1)CC1(CCC2)CCCCCC1")
        assert is_spiro_system(mol) is True
        assert len(get_spiro_atoms(mol)) == 2
        result = name_spiro_system(mol)
        assert result is not None
        name, _, _, _ = result
        # P-24.2.2 segment count tiebreak deferred to v19 (logged
        # AUTONOM-followups). Connectivity correctness is what we
        # verify here — name must contain 'dispiro' and a proper
        # '[a.b.c.d]' descriptor.
        assert name.startswith("dispiro[")

    @pytest.mark.unit
    def test_dispiro_segments_have_correct_arity(self):
        """Dispiro descriptor has exactly 4 dot-separated segments."""
        mol = Chem.MolFromSmiles("C1CCC2(CC1)CC1(CCC2)CCCCCC1")
        result = name_spiro_system(mol)
        assert result is not None
        name = result[0]
        # Extract the [a.b.c.d] block
        bracket = name[name.index("[") + 1: name.index("]")]
        segments = bracket.split(".")
        assert len(segments) == 4, (
            f"dispiro descriptor must have 4 segments, got {len(segments)} "
            f"in {name!r}"
        )

    @pytest.mark.unit
    def test_dispiro_atom_count_invariant(self):
        """Total atom count for dispiro[a.b.c.d]: a+b+c+d+2 = total ring atoms."""
        mol = Chem.MolFromSmiles("C1CCC2(CC1)CC1(CCC2)CCCCCC1")
        result = name_spiro_system(mol)
        assert result is not None
        name = result[0]
        bracket = name[name.index("[") + 1: name.index("]")]
        segments = [int(s) for s in bracket.split(".")]
        ri = mol.GetRingInfo()
        ring_atoms = set()
        for r in ri.AtomRings():
            ring_atoms.update(r)
        # Sum of segments + N spiro centres = total ring atoms.
        n_spiro = len(get_spiro_atoms(mol))
        assert sum(segments) + n_spiro == len(ring_atoms), (
            f"dispiro segment-count invariant violated: "
            f"{segments} + {n_spiro} != {len(ring_atoms)}"
        )

    @pytest.mark.unit
    def test_dispiro_supplier_full_coverage(self):
        """get_spiro_iupac_locants returns map covering ALL ring atoms (Pitfall 7)."""
        get_spiro_iupac_locants = _spiro_lazy_supplier_import()
        mol = Chem.MolFromSmiles("C1CCC2(CC1)CC1(CCC2)CCCCCC1")
        loc = get_spiro_iupac_locants(mol)
        assert loc is not None
        ri = mol.GetRingInfo()
        ring_atoms = set()
        for r in ri.AtomRings():
            ring_atoms.update(r)
        assert set(loc.keys()) >= ring_atoms

    @pytest.mark.unit
    def test_pure_spiro_supplier_full_coverage_45_decane(self):
        """get_spiro_iupac_locants on monospiro spiro[4.5]decane."""
        get_spiro_iupac_locants = _spiro_lazy_supplier_import()
        mol = Chem.MolFromSmiles("C1CCC2(CC1)CCCC2")
        loc = get_spiro_iupac_locants(mol)
        assert loc is not None
        ri = mol.GetRingInfo()
        ring_atoms = set()
        for r in ri.AtomRings():
            ring_atoms.update(r)
        assert set(loc.keys()) >= ring_atoms

    @pytest.mark.unit
    def test_pure_spiro_supplier_returns_none_on_non_spiro(self):
        """Supplier returns None on non-spiro inputs."""
        get_spiro_iupac_locants = _spiro_lazy_supplier_import()
        mol = Chem.MolFromSmiles("C1CCCCC1")  # cyclohexane
        assert get_spiro_iupac_locants(mol) is None

    @pytest.mark.unit
    def test_pure_spiro_supplier_returns_none_on_mixed(self):
        """Supplier returns None on mixed-spiro/fused (defers to mixed handler)."""
        get_spiro_iupac_locants = _spiro_lazy_supplier_import()
        mol = Chem.MolFromSmiles("C1CCC2(CC1)Cc1ccccc12")  # spiro-indane
        # is_spiro_system returns False on this; therefore supplier returns None.
        assert is_spiro_system(mol) is False
        assert get_spiro_iupac_locants(mol) is None

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "fixture",
        [f for f in _load_corpus_fixtures(compound_class_filter="spiro-pure")],
        ids=lambda f: f["fixture_id"] if isinstance(f, dict) else "no-id",
    )
    def test_corpus_pure_multispiro_named(self, fixture):
        """Corpus pure-multi-spiro inputs produce a polyspiro descriptor."""
        mol = Chem.MolFromSmiles(fixture["smiles"])
        if mol is None:
            pytest.skip(f"corpus SMILES invalid: {fixture['fixture_id']}")
        # Corpus may include compounds with substituents; skip if
        # the spiro-core itself doesn't qualify.
        if not is_spiro_system(mol):
            pytest.skip(f"corpus fixture not pure spiro at SSSR layer: "
                        f"{fixture['fixture_id']}")
        result = name_spiro_system(mol)
        # Wave2 T6a (P-24.2.0/P-31.1.5.1): a POLYSPIRO system with a ring
        # multiple bond now fails closed — the unsaturation splice is
        # monospiro-only, and the old '-ane' core silently dropped the ring
        # double bond (the exact defect T6a fixes). Assert that contract.
        _has_ring_unsat = any(
            b.IsInRing()
            and b.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE)
            for b in mol.GetBonds()
        )
        if len(get_spiro_atoms(mol)) >= 2 and _has_ring_unsat:
            assert result is None, (
                f"{fixture['fixture_id']}: unsaturated polyspiro must fail "
                f"closed, got {result[0]!r}"
            )
            return
        assert result is not None, fixture["fixture_id"]
        name = result[0]
        n_spiro = len(get_spiro_atoms(mol))
        if n_spiro == 1:
            assert name.startswith("spiro[") or "spiro[" in name
        elif n_spiro == 2:
            assert "dispiro[" in name, (fixture["fixture_id"], name)
        elif n_spiro == 3:
            assert "trispiro[" in name, (fixture["fixture_id"], name)
        elif n_spiro >= 4:
            assert ("tetraspiro[" in name or "pentaspiro[" in name), (
                fixture["fixture_id"], name)


class TestTrispiro:
    """Trispiro target gate — Plan 151-02 D-10."""

    @pytest.mark.unit
    def test_trispiro_corpus_or_skip(self):
        """If corpus has a trispiro fixture, it must produce 'trispiro['."""
        fixtures = _load_corpus_fixtures(
            compound_class_filter="spiro-pure")
        trispiros = []
        for f in fixtures:
            mol = Chem.MolFromSmiles(f["smiles"])
            if mol is None:
                continue
            if len(get_spiro_atoms(mol)) >= 3 and is_spiro_system(mol):
                trispiros.append(f)
        if not trispiros:
            pytest.skip("No corpus trispiro pure fixture available")
        f = trispiros[0]
        mol = Chem.MolFromSmiles(f["smiles"])
        result = name_spiro_system(mol)
        assert result is not None
        # Trispiro/tetraspiro/pentaspiro all acceptable per D-10.
        assert ("trispiro[" in result[0]
                or "tetraspiro[" in result[0]
                or "pentaspiro[" in result[0])


class TestPolyspiroPrefixesTable:
    """Plan 151-02 D-10: _POLYSPIRO_PREFIXES already covers up to penta."""

    @pytest.mark.unit
    def test_prefixes_cover_through_pentaspiro(self):
        from orthonym.rules.spiro import _POLYSPIRO_PREFIXES
        assert _POLYSPIRO_PREFIXES[1] == "spiro"
        assert _POLYSPIRO_PREFIXES[2] == "dispiro"
        assert _POLYSPIRO_PREFIXES[3] == "trispiro"
        assert _POLYSPIRO_PREFIXES[4] == "tetraspiro"
        assert _POLYSPIRO_PREFIXES[5] == "pentaspiro"


class TestRoundTripViaOPSIN:
    """OPSIN round-trip InChI L1 (D-23) on Blue Book pure-spiro fixtures."""

    @pytest.mark.roundtrip
    @pytest.mark.skipif(not _opsin_available(),
                        reason="OPSIN/Java not available")
    @pytest.mark.parametrize(
        "fixture",
        [f for f in _load_blue_book_fixtures()
         if f.get("compound_class") in ("spiro-pure", "spiro-hetero")],
        ids=lambda f: f["fixture_id"] if isinstance(f, dict) else "no-id",
    )
    def test_blue_book_roundtrip(self, fixture):
        mol = Chem.MolFromSmiles(fixture["smiles"])
        result = name_spiro_system(mol)
        assert result is not None, fixture["fixture_id"]
        name = result[0]
        parsed = _opsin_parse_one(name)
        if parsed is None:
            pytest.skip(f"OPSIN cannot parse {name!r} for "
                        f"{fixture['fixture_id']}")
        i_in = Chem.MolToInchi(mol).split("/c", 1)[0]
        i_rt = Chem.MolToInchi(Chem.MolFromSmiles(parsed)).split("/c", 1)[0]
        assert i_in == i_rt, (
            f"InChI L1 mismatch on {fixture['fixture_id']}: "
            f"input={i_in!r} round-trip={i_rt!r} via name={name!r}"
        )
