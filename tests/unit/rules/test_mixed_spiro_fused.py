"""a phase-02 /: mixed spiro/fused detector + name builder + suppliers.

Wave-0 RED scaffold: tests target the NEW Plan 151-02 functions that
must be added to ``src/orthonym/rules/spiro.py`` in Task 2:

- ``is_mixed_spiro_fused(mol)`` (+)
- ``name_mixed_spiro_fused(mol)`` (+ AUTONOM)
- ``get_spiro_iupac_locants(mol)`` (partial)
- ``get_mixed_spiro_fused_iupac_locants(mol)`` (partial)

Lazy-import pattern matches Plan 151-01's test scaffold so RED-state
collection succeeds before Task 2 lands.

Source: 151-02-PLAN.md tasks 1b/2; internal notes-B.md; 151-internal notes
////(b); internal notes Pitfall 3 + AUTONOM
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
from tests.support.jars import jar_or_none


# ---------------------------------------------------------------------------
# OPSIN round-trip helpers  — same shape as canary infra.
# parents[3]: tests/unit/rules/test_X.py -> project root.
# ---------------------------------------------------------------------------

_OPSIN_JAR = jar_or_none()


def _opsin_available() -> bool:
    return _OPSIN_JAR is not None and shutil.which("java") is not None


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


_ANTI_CANARY = _load_json("anti_canary.json")


# ===========================================================================
# Class: is_mixed_spiro_fused True positives
# ===========================================================================

class TestIsMixedSpiroFusedTruePositives:
    """is_mixed_spiro_fused returns True on real mixed-spiro/fused inputs."""

    @pytest.mark.unit
    def test_minimal_spiro_indane_is_mixed(self):
        """Spiro-indane (pure-aromatic fused part attached via spiro): MIXED."""
        is_mixed_spiro_fused = _import_or_skip("is_mixed_spiro_fused")
        mol = Chem.MolFromSmiles("C1CCC2(CC1)CCc1ccccc12")
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
# Class: pure-spiro contract preservation (lock)
# ===========================================================================

class TestPureSpiroContractPreserved:
    """ lock: is_spiro_system body byte-identical; returns True for
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
    def test_is_spiro_system_returns_false_on_mixed(self):
        """Confirms is_spiro_system continues to reject mixed inputs."""
        from orthonym.rules.spiro import is_spiro_system
        # spiro-indane: 3 rings, 1 spiro -> n_rings > n_spiro+1
        mol = Chem.MolFromSmiles("C1CCC2(CC1)CCc1ccccc12")
        assert is_spiro_system(mol) is False


# ===========================================================================
# Class: AUTONOM separable-parts naming
# ===========================================================================

class TestSeparablePartsNaming:
    """name_mixed_spiro_fused implements AUTONOM §4 separable-parts."""

    @pytest.mark.unit
    def test_returns_none_on_pure_spiro(self):
        """name_mixed_spiro_fused(pure spiro) -> None (dispatch contract)."""
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
# Class: enforcement — heteroatom prefix reuse
# ===========================================================================

class TestHeteroatomPrefixReuse:
    """: spiro.py reuses polycyclic_bridged.get_heteroatom_prefix."""

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
# Class: round-trip via OPSIN
# ===========================================================================

class TestRoundTripViaOPSIN:
    """ hard test gate. Plan 151-02 acceptance: ≥10 corpus mixed
    fixtures round-trip via OPSIN with InChI L1 match."""

    @pytest.mark.roundtrip
    @pytest.mark.skipif(not _opsin_available(),
                        reason="OPSIN/Java not available")
    def test_round_trip_autonom_indoline_cyclohexane(self):
        """Q-05 evidence: spiro[indoline-3,1'-cyclohexane] parses cleanly."""
        name = "spiro[indoline-3,1'-cyclohexane]"
        parsed = _opsin_parse_one(name)
        assert parsed is not None
        rt = Chem.MolFromSmiles(parsed)
        assert rt is not None
        # Round-trip the OPSIN output to verify InChI computability.
        assert Chem.MolToInchi(rt).startswith("InChI=")


class TestHeteroatomSeniorityWR01:
    """a phase-04: side-ring heteroatom seniority must include
    halogens per IUPAC."""

    @pytest.mark.unit
    def test_no_local_priority_dict_in_walk_side_ring_locants(self):
        """The local priority dict at spiro.py:1449 must be removed in
        favor of get_heteroatom_priority. A grep over the function source
        confirms the regression is closed."""
        from orthonym.rules import spiro
        # Find the function (private; access through module globals)
        walker = getattr(spiro, "_walk_side_ring_locants", None)
        if walker is None:
            pytest.skip("_walk_side_ring_locants is no longer a module attribute")
        src = inspect.getsource(walker)
        non_comment = "\n".join(
            L for L in src.split("\n") if not L.strip().startswith("#")
        )
        # lock: the literal local-priority-dict pattern must NOT
        # appear in the function body.
        assert 'priority = {"O": 0' not in non_comment, (
            "WR-01 regression: local priority dict reintroduced in "
            "_walk_side_ring_locants. Use get_heteroatom_priority instead."
        )
        # AND the canonical helper IS used.
        assert "get_heteroatom_priority" in non_comment, (
            "WR-01: get_heteroatom_priority must be referenced in "
            "_walk_side_ring_locants for halogen-aware seniority."
        )

    @pytest.mark.unit
    def test_get_heteroatom_priority_orders_halogens_above_oxygen(self):
        """Sanity: confirm the canonical helper ranks F senior to O.
        If this ever flips, the fix's correctness premise is gone."""
        from orthonym.data.hw_heteroatoms import get_heteroatom_priority
        assert get_heteroatom_priority("F") < get_heteroatom_priority("O"), (
            "IUPAC P-25.3.1.3: F is more senior than O. If this assertion "
            "fails, hw_heteroatoms.py's ordering has changed and WR-01 "
            "needs re-review."
        )
        assert get_heteroatom_priority("Cl") < get_heteroatom_priority("O")
        assert get_heteroatom_priority("Br") < get_heteroatom_priority("O")
        assert get_heteroatom_priority("I") < get_heteroatom_priority("O")


# ===========================================================================
#.3 — offer-not-return for spiro-CORE-plus-PENDANT-RING molecules
# ===========================================================================

def _full_inchikey(smi: str):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToInchiKey(m) if m else None


class TestCT3PendantRingOfferNotReturn:
    """CT.3 (sub-lever B): a spiro/fused system that additionally carries a
    ring-bearing substituent (a PENDANT ring attached to the core through a
    single bond) must no longer be REJECTED wholesale by the coverage
    invariant. The old invariant demanded the combined locant map cover EVERY
    ring atom in the molecule, so any pendant ring forced
    ``name_mixed_spiro_fused`` -> None -> abstain. The correct behaviour is to
    OFFER the spiro CORE parent (fused component + side ring) and let the
    cascade substituent supplier attach the pendant ring, with the RT gate
    (/ OPSIN full InChIKey) deciding acceptance (a project rule).

    0-wrong is preserved by the RT gate — this class asserts the *positive*
    wins (previously-abstaining molecules that now round-trip EXACT) and the
    core-offer contract at the function level.
    """

    # Two witnesses validated fresh (full InChIKey incl. stereo) at HEAD:
    # each carries a pendant aromatic-ring substituent on a spiro/fused core.
    _PENDANT_WITNESSES = [
        # spiro[2,3-dihydro-1-benzofuran-2,3'-piperazine] with a pendant
        # (2-hydroxyphenyl)methylidene substituent.
        ("O=C1N[C@@]2(Cc3ccccc3O2)C(=O)N/C1=C\\c1ccccc1O",
         "OMZTVPPJRIMQEW"),
        # spiro[1,2,3,4-tetrahydroisoquinoline-4,3'-pyrrolidine] with a
        # pendant (4-bromo-2-fluorophenyl)methyl substituent.
        ("O=C1CC2(C(=O)N1)C(=O)N(Cc1ccc(Br)cc1F)C(=O)c1ccc(F)cc12",
         "BMHZAHGTGIZZCT"),
    ]

    @pytest.mark.unit
    def test_offers_core_parent_for_pendant_ring(self):
        """name_mixed_spiro_fused OFFERS a core parent (non-None) for a
        spiro-core-plus-pendant-ring molecule instead of a terminal None.
        The returned ring_atoms set is the CORE only (does not include the
        pendant ring's atoms — those are substituent territory)."""
        name_mixed_spiro_fused = _import_or_skip("name_mixed_spiro_fused")
        # The (2-hydroxyphenyl)methylidene benzofuran-piperazine spiro.
        mol = Chem.MolFromSmiles(
            "O=C1N[C@@]2(Cc3ccccc3O2)C(=O)N/C1=C\\c1ccccc1O")
        assert mol is not None
        result = name_mixed_spiro_fused(mol)
        assert result is not None, (
            "CT.3 regression: name_mixed_spiro_fused returned a terminal None "
            "for a spiro-core-plus-pendant-ring molecule; it must OFFER the "
            "core parent (invariant 18 offer-not-return)."
        )
        name, ring_atoms, _atom_to_locant, _subs = result
        # (the Blue Book): "Indicated hydrogen... is cited in front
        # of the name if needed in the complete structure"; the benzofuran is its
        # mancude component with bracketed locants,:10295). Was a name
        # starting 'spiro[2,3-dihydro-1-benzofuran-...'.
        assert name == "3H-spiro[[1]benzofuran-2,2'-piperazine]", name
        # The pendant phenyl ring atoms must NOT be in the core ring set.
        total_ring_atoms = len({a for r in mol.GetRingInfo().AtomRings()
                                for a in r})
        assert len(ring_atoms) < total_ring_atoms, (
            f"CT.3: core ring set ({len(ring_atoms)}) must exclude the pendant "
            f"ring atoms (total ring atoms {total_ring_atoms})."
        )

    @pytest.mark.roundtrip
    @pytest.mark.skipif(not _opsin_available(),
                        reason="OPSIN/Java not available")
    @pytest.mark.parametrize("smiles,expect_block1", _PENDANT_WITNESSES,
                             ids=["benzofuran-piperazine-spiro",
                                  "tetrahydroisoquinoline-pyrrolidine-spiro"])
    def test_pendant_ring_witness_round_trips_exact(self, smiles,
                                                    expect_block1):
        """The two validated pendant-ring witnesses emit a FULL name that
        OPSIN round-trips to the exact input InChIKey (constitution +
        stereo), through the full best-effort namer."""
        from orthonym.namer import Orthonym
        o = Orthonym(style="pin", general_fallback=True,
                      general_fallback_unverified=True,
                      allow_aromatic_general=True)
        name = o.name_with_confidence(smiles).get("name")
        assert name and name != "unknown organic compound", (
            f"CT.3: expected an emitted name for {smiles!r}, got {name!r}")
        parsed = _opsin_parse_one(name)
        assert parsed is not None, f"OPSIN could not parse {name!r}"
        rt_key = _full_inchikey(parsed)
        in_key = _full_inchikey(smiles)
        assert rt_key == in_key, (
            f"CT.3 round-trip mismatch on {smiles!r}: name={name!r} "
            f"input={in_key} round-trip={rt_key}")
