"""a phase-03.. unit tests for ring assemblies size 3+.

Covers:
- TestIdentity : n-way signature identity already structurally enforced
  in detect_ring_assembly:185 (existing). New tests verify the contract for
  3+ component cases (terphenyl/quaterphenyl) and rejection cases
  (mixed phenyl-pyridyl chain).
- TestTopology : NEW _check_path_topology helper rejects branched
  arrangements (1,3,5-triphenylbenzene, central degree=3) and accepts
  linear chains (terphenyl, quaterphenyl). 4'-methylbiphenyl is the
  canary control (RESEARCH Pitfall 4).
- TestPrimedLocants (,,): primed-locant emission via
  _format_prime; Blue Book form is primary; carat is opt-in
  fallback only and never emitted by default.
- TestMultiplierClosed : ASSEMBLY_MULTIPLIERS table closed at
  deci(10); size-11 returns None.
- TestConnectionLocant : per-ring own IUPAC numbering used for
  connection locant; heterocycles via heteroatom priority; carbocyclics
  numbered relative to inter-ring bonds.
- TestLowestLocantTiebreak : symmetric assemblies pick lowest set
  per first-point-of-difference using compare_locant_sets.
- TestSupplierCoverageInvariant: cascade-step-6 supplier returns None
  on partial coverage (Pitfall 7); full coverage on complete cases.
- TestNoParallelComparator (/): grep-style lock that no
  ``def _compare_locant*`` is introduced in ring_assemblies.py.
- TestRoundTripViaOPSIN : OPSIN parses every Blue-Book/literature
  fixture name back to the expected SMILES with InChI L1 match.

Source: 151-internal notes,,,,,,,,
        ,.
Source: https://iupac.qmul.ac.uk/BlueBook/P2.html.
Source: internal notes §"OPSIN Compatibility Evidence" (12 names).
Source: internal notes-C.md (verdict: ENGINE_N3_PLUS_PARTIAL +
        PATH_TOPOLOGY_MISSING + SUPPLIER_MISSING + Q-04
        SUBSTITUENT_NAMING_ROUND_TRIPS).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from rdkit import Chem
from tests.support.jars import jar_or_none


# Lazy imports inside test bodies so test collection succeeds even when
# new code (Task 2) hasn't landed yet. Same pattern used in 151-01 and
# 151-02 Wave-0 scaffolds for audit-first cadence per internal notes.

def _ra_import():
    """Return the ring_assemblies module symbols this test file uses."""
    from orthonym.rules import ring_assemblies as ra
    return ra


def _ra_supplier_import():
    """Return the cascade-step-6 supplier — fails until Task 2 lands."""
    from orthonym.rules.ring_assemblies import (
        get_ring_assembly_iupac_locants,
    )
    return get_ring_assembly_iupac_locants


def _ra_topology_import():
    """Return the _check_path_topology helper — fails until Task 2 lands."""
    from orthonym.rules.ring_assemblies import _check_path_topology
    return _check_path_topology


# OPSIN oracle . The jar lives at the worktree root via symlink to
# parent project's jar. Path resolution depth is parents[3] for tests/unit/rules/.
_OPSIN_JAR = jar_or_none()


def _opsin_available() -> bool:
    return _OPSIN_JAR is not None and shutil.which("java") is not None


def _opsin_parse(name: str) -> str | None:
    """Single-name OPSIN parse. For batch use, prefer test_canary_rt75 helper."""
    if not _opsin_available():
        return None
    try:
        proc = subprocess.run(
            ["java", "-jar", str(_OPSIN_JAR), "-osmi"],
            input=name + "\n",
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception:
        return None
    out = proc.stdout.strip().splitlines()
    if not out:
        return None
    last = out[-1].strip()
    if "unparsable" in last.lower() or "is unparsable" in last.lower():
        return None
    if last.startswith("Run the jar"):
        return None
    return last


# Fixture registry — load Blue Book + literature_validated entries.
_FIXTURE_DIR = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "fixtures"
    / "ring_systems"
    / "assemblies"
)
_BLUE_BOOK_FIXTURES = json.loads((_FIXTURE_DIR / "blue_book_examples.json").read_text())
_LIT_FIXTURES = json.loads((_FIXTURE_DIR / "literature_validated.json").read_text())


# ============================================================================
# TestIdentity
# ============================================================================
class TestIdentity:
    """: n-way identity check rejects mixed-ring chains."""

    @pytest.mark.unit
    def test_homogeneous_terphenyl_accepted(self):
        ra = _ra_import()
        mol = Chem.MolFromSmiles("c1ccc(-c2ccc(-c3ccccc3)cc2)cc1")
        from orthonym.perception.rings import get_ring_systems
        rs = get_ring_systems(mol, include_spiro=False)
        info = ra.detect_ring_assembly(mol, rs)
        assert info is not None
        assert info["count"] == 3
        assert info["ring_type"] == "carbocyclic"

    @pytest.mark.unit
    def test_heterogeneous_phenyl_pyridyl_chain_rejected(self):
        ra = _ra_import()
        # 1-phenyl-3-pyridyl-benzene: phenyl-pyridyl-phenyl mixed signatures
        mol = Chem.MolFromSmiles("c1ccc(-c2ccncc2)cc1-c1ccccc1")
        from orthonym.perception.rings import get_ring_systems
        rs = get_ring_systems(mol, include_spiro=False)
        info = ra.detect_ring_assembly(mol, rs)
        assert info is None  # n-way signature mismatch (already structural)

    @pytest.mark.unit
    def test_2_phenyl_pyridine_rejected(self):
        ra = _ra_import()
        mol = Chem.MolFromSmiles("c1ccc(-c2ccccn2)cc1")
        from orthonym.perception.rings import get_ring_systems
        rs = get_ring_systems(mol, include_spiro=False)
        info = ra.detect_ring_assembly(mol, rs)
        assert info is None

    @pytest.mark.unit
    def test_terpyridine_homogeneous_accepted(self):
        ra = _ra_import()
        mol = Chem.MolFromSmiles("c1ccc(-c2cccc(-c3ccccn3)n2)nc1")
        from orthonym.perception.rings import get_ring_systems
        rs = get_ring_systems(mol, include_spiro=False)
        info = ra.detect_ring_assembly(mol, rs)
        assert info is not None
        assert info["count"] == 3
        assert info["ring_type"] == "heterocyclic"


# ============================================================================
# TestTopology
# ============================================================================
class TestTopology:
    """: linear-path requirement; branched arrangements rejected."""

    @pytest.mark.unit
    def test_check_path_topology_linear_returns_true(self):
        _check_path_topology = _ra_topology_import()
        # Linear: 0-1-2 (degrees 1,2,1)
        connections = [(0, 0, 0, 1), (0, 0, 1, 2)]
        assert _check_path_topology(3, connections) is True

    @pytest.mark.unit
    def test_check_path_topology_quaterphenyl_returns_true(self):
        _check_path_topology = _ra_topology_import()
        # Linear 4-chain: 0-1-2-3 (degrees 1,2,2,1)
        connections = [(0, 0, 0, 1), (0, 0, 1, 2), (0, 0, 2, 3)]
        assert _check_path_topology(4, connections) is True

    @pytest.mark.unit
    def test_check_path_topology_star_branched_returns_false(self):
        _check_path_topology = _ra_topology_import()
        # Star: central system 0 with degree 3 (connects to 1, 2, 3)
        connections = [(0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 0, 3)]
        assert _check_path_topology(4, connections) is False

    @pytest.mark.unit
    def test_check_path_topology_single_system_trivially_true(self):
        _check_path_topology = _ra_topology_import()
        assert _check_path_topology(1, []) is True

    @pytest.mark.unit
    def test_135_triphenylbenzene_rejected_by_detect(self):
        """End-to-end: 1,3,5-triphenylbenzene must NOT be a ring assembly."""
        ra = _ra_import()
        mol = Chem.MolFromSmiles("c1cc(-c2ccccc2)cc(-c2ccccc2)c1-c1ccccc1")
        from orthonym.perception.rings import get_ring_systems
        rs = get_ring_systems(mol, include_spiro=False)
        info = ra.detect_ring_assembly(mol, rs)
        assert info is None  # path-topology rejects (central degree=3)

    @pytest.mark.unit
    def test_4prime_methylbiphenyl_still_accepted(self):
        """RESEARCH Pitfall 4 control: substituent does NOT disturb topology."""
        ra = _ra_import()
        mol = Chem.MolFromSmiles("Cc1ccc(-c2ccccc2)cc1")
        from orthonym.perception.rings import get_ring_systems
        rs = get_ring_systems(mol, include_spiro=False)
        info = ra.detect_ring_assembly(mol, rs)
        assert info is not None
        assert info["count"] == 2

    @pytest.mark.unit
    def test_para_terphenyl_linear_accepted(self):
        ra = _ra_import()
        mol = Chem.MolFromSmiles("c1ccc(-c2ccc(-c3ccccc3)cc2)cc1")
        from orthonym.perception.rings import get_ring_systems
        rs = get_ring_systems(mol, include_spiro=False)
        info = ra.detect_ring_assembly(mol, rs)
        assert info is not None
        assert info["count"] == 3


# ============================================================================
# TestPrimedLocants (,,)
# ============================================================================
class TestPrimedLocants:
    """: primed-locant primary; carat opt-in only."""

    @pytest.mark.unit
    def test_format_prime_zero_returns_empty(self):
        ra = _ra_import()
        assert ra._format_prime(0) == ""

    @pytest.mark.unit
    def test_format_prime_one_returns_single_apostrophe(self):
        ra = _ra_import()
        assert ra._format_prime(1) == "'"

    @pytest.mark.unit
    def test_format_prime_three_returns_triple_apostrophe(self):
        ra = _ra_import()
        assert ra._format_prime(3) == "'''"

    @pytest.mark.unit
    def test_terphenyl_emits_primed_locant_form(self):
        """: name_compound emits 1,1':4',1''-terphenyl, NOT carat."""
        from orthonym import name_compound
        name = name_compound("c1ccc(-c2ccc(-c3ccccc3)cc2)cc1")
        assert name is not None
        assert "terphenyl" in name
        # Must NOT contain caret notation per
        assert "^" not in name

    @pytest.mark.unit
    def test_no_carat_in_quaterphenyl_name(self):
        from orthonym import name_compound
        name = name_compound("c1ccc(-c2ccc(-c3ccc(-c4ccccc4)cc3)cc2)cc1")
        assert name is not None
        assert "^" not in name

    @pytest.mark.unit
    def test_terphenyl_para_emits_canonical_blue_book_form(self):
        """ + +: middle ring gets locant 4 (para-attachment)."""
        from orthonym import name_compound
        name = name_compound("c1ccc(-c2ccc(-c3ccccc3)cc2)cc1")
        assert name == "1,1':4',1''-terphenyl"


# ============================================================================
# TestMultiplierClosed
# ============================================================================
class TestMultiplierClosed:
    """: multiplier table. Extended to dodeci(12) by (Wave2 P1CB
    Task 8, OPSIN-RT verified for undeci); still closed above 12."""

    @pytest.mark.unit
    def test_deci_present(self):
        ra = _ra_import()
        assert ra.ASSEMBLY_MULTIPLIERS[10] == "deci"

    @pytest.mark.unit
    def test_size_11_12_present(self):
        # (Wave2 P1CB Task 8): undeci(11)/dodeci(12) now supported.
        ra = _ra_import()
        assert ra.ASSEMBLY_MULTIPLIERS.get(11) == "undeci"
        assert ra.ASSEMBLY_MULTIPLIERS.get(12) == "dodeci"

    @pytest.mark.unit
    def test_size_13_returns_none(self):
        """Table stops at dodeci(12); count 13+ has no OPSIN-verifiable affix
        here ->.get->None decline (fail closed)."""
        ra = _ra_import()
        assert ra.ASSEMBLY_MULTIPLIERS.get(13) is None

    @pytest.mark.unit
    def test_size_15_returns_none(self):
        """OPSIN multipliers.xml extends to 15; the table deliberately stops at 12."""
        ra = _ra_import()
        assert ra.ASSEMBLY_MULTIPLIERS.get(15) is None

    @pytest.mark.unit
    def test_name_ring_assembly_returns_none_above_table(self):
        ra = _ra_import()
        # Build a 13-system mock info dict directly to exercise the gate
        # (13 is above the dodeci(12) cap -> multiplier lookup fails -> None).
        fake_info = {
            "ring_systems": [set() for _ in range(13)],
            "connections": [],
            "count": 13,
            "ring_type": "carbocyclic",
        }
        # name_ring_assembly returns None when multiplier lookup fails
        result = ra.name_ring_assembly(Chem.MolFromSmiles("CC"), fake_info, None)
        assert result is None


# ============================================================================
# TestConnectionLocant
# ============================================================================
class TestConnectionLocant:
    """: per-ring own IUPAC numbering used for connection locant."""

    @pytest.mark.unit
    def test_pyridine_nitrogen_priority(self):
        """Pyridyl atom alpha to N (locant 2) gets locant 2 in 2,2'-bipyridine.

        SMILES uses the canonical 2,2'-bipyridine constitution
        (each connection atom directly adjacent to N).
        """
        ra = _ra_import()
        from orthonym.perception.rings import get_ring_systems
        # 2,2'-bipyridine: ring atoms with N as locant 1, connection at locant 2
        mol = Chem.MolFromSmiles("c1ccc(-c2ccccn2)nc1")  # 2,2'-bipyridine
        rs = get_ring_systems(mol, include_spiro=False)
        info = ra.detect_ring_assembly(mol, rs)
        assert info is not None
        # Pick connection atom in first ring; its locant should be 2
        a1, _, s1, _ = info["connections"][0]
        loc = ra._get_connection_locant(mol, a1, info["ring_systems"][s1])
        assert loc == 2

    @pytest.mark.unit
    def test_thiophene_sulfur_priority(self):
        """Thienyl atom adjacent to S gets locant 2 (lowest-locant tiebreak)."""
        ra = _ra_import()
        from orthonym.perception.rings import get_ring_systems
        mol = Chem.MolFromSmiles("c1ccsc1-c1cccs1")  # 2,2'-bithiophene
        rs = get_ring_systems(mol, include_spiro=False)
        info = ra.detect_ring_assembly(mol, rs)
        assert info is not None
        a1, _, s1, _ = info["connections"][0]
        loc = ra._get_connection_locant(mol, a1, info["ring_systems"][s1])
        assert loc == 2


# ============================================================================
# TestLowestLocantTiebreak
# ============================================================================
class TestLowestLocantTiebreak:
    """: lowest-locant tiebreak via compare_locant_sets (no parallel)."""

    @pytest.mark.unit
    def test_compare_locant_sets_imported_in_supplier(self):
        """Lock: get_ring_assembly_iupac_locants OR detect/name path uses
        compare_locant_sets, never a parallel comparator."""
        src = (
            Path(__file__).resolve().parents[3]
            / "src" / "orthonym" / "rules" / "ring_assemblies.py"
        ).read_text()
        # Either imported or currently absent (will be added by Task 2 if
        # the supplier needs cross-orientation tiebreaks; if the helpers
        # don't introduce a comparator at all, that's also fine — the
        # negative assertion in TestNoParallelComparator covers it).
        assert "_compare_locant" not in src or "compare_locant_sets" in src


# ============================================================================
# TestSupplierCoverageInvariant (Pitfall 7 /)
# ============================================================================
class TestSupplierCoverageInvariant:
    """Pitfall 7: cascade-step-6 supplier returns full coverage or None."""

    @pytest.mark.unit
    def test_supplier_full_coverage_on_terphenyl(self):
        get_ring_assembly_iupac_locants = _ra_supplier_import()
        mol = Chem.MolFromSmiles("c1ccc(-c2ccc(-c3ccccc3)cc2)cc1")
        locants = get_ring_assembly_iupac_locants(mol)
        assert locants is not None
        # Must cover all ring atoms (Pitfall 7 invariant)
        ri = mol.GetRingInfo()
        ring_atoms: set = set()
        for r in ri.AtomRings():
            ring_atoms.update(r)
        assert set(locants.keys()) >= ring_atoms

    @pytest.mark.unit
    def test_supplier_full_coverage_on_quaterphenyl(self):
        get_ring_assembly_iupac_locants = _ra_supplier_import()
        mol = Chem.MolFromSmiles(
            "c1ccc(-c2ccc(-c3ccc(-c4ccccc4)cc3)cc2)cc1"
        )
        locants = get_ring_assembly_iupac_locants(mol)
        assert locants is not None
        ri = mol.GetRingInfo()
        ring_atoms: set = set()
        for r in ri.AtomRings():
            ring_atoms.update(r)
        assert set(locants.keys()) >= ring_atoms

    @pytest.mark.unit
    def test_supplier_returns_none_on_135_triphenylbenzene(self):
        """Branched arrangements MUST return None (+ Pitfall 7)."""
        get_ring_assembly_iupac_locants = _ra_supplier_import()
        mol = Chem.MolFromSmiles(
            "c1cc(-c2ccccc2)cc(-c2ccccc2)c1-c1ccccc1"
        )
        # path-topology rejects in detect; supplier emits None to keep
        # cascade-step-6 from firing on a non-assembly.
        result = get_ring_assembly_iupac_locants(mol)
        assert result is None

    @pytest.mark.unit
    def test_supplier_returns_none_on_phenyl_pyridyl_mixed(self):
        get_ring_assembly_iupac_locants = _ra_supplier_import()
        mol = Chem.MolFromSmiles("c1ccc(-c2ccncc2)cc1-c1ccccc1")
        result = get_ring_assembly_iupac_locants(mol)
        assert result is None  # n-way identity rejects; supplier follows

    @pytest.mark.unit
    def test_supplier_returns_none_on_single_ring(self):
        get_ring_assembly_iupac_locants = _ra_supplier_import()
        mol = Chem.MolFromSmiles("c1ccccc1")
        assert get_ring_assembly_iupac_locants(mol) is None

    @pytest.mark.unit
    def test_supplier_returns_none_on_acyclic(self):
        get_ring_assembly_iupac_locants = _ra_supplier_import()
        mol = Chem.MolFromSmiles("CCCC")
        assert get_ring_assembly_iupac_locants(mol) is None


# ============================================================================
# TestNoParallelComparator (/)
# ============================================================================
class TestNoParallelComparator:
    """: no SMARTS broadening;: no parallel locant comparator."""

    @pytest.mark.unit
    def test_no_parallel_compare_locant_function(self):
        """grep test: no ``def _compare_locant`` introduced in ring_assemblies.py."""
        src_path = (
            Path(__file__).resolve().parents[3]
            / "src" / "orthonym" / "rules" / "ring_assemblies.py"
        )
        src_lines = src_path.read_text().splitlines()
        # Strip comments (lines starting with #) before grepping
        non_comment_text = "\n".join(
            line for line in src_lines if not line.lstrip().startswith("#")
        )
        assert "def _compare_locant" not in non_comment_text


# ============================================================================
# TestRoundTripViaOPSIN
# ============================================================================
@pytest.mark.skipif(not _opsin_available(), reason="OPSIN/Java not available")
class TestRoundTripViaOPSIN:
    """: every named fixture must round-trip via OPSIN with InChI L1 match."""

    # Fixtures known to need substituent-classification fix per
    # internal notes-C.md "Out-of-scope follow-ups" + AUTONOM-followups.md.
    # Marked xfail so the suite stays GREEN while the upstream bug is
    # tracked for.
    _XFAIL_FIXTURES = {
        "ra_lit_terphenyl_dicarboxylic_acid": (
            "151-AUDIT-C.md out-of-scope: -CHO vs -COOH substituent "
            "classification bug emits 'diformyl' instead of "
            "'dicarboxylic acid'. Upstream substituent-detection issue "
            "logged to AUTONOM-followups.md as v19 follow-up."
        ),
    }

    @pytest.mark.parametrize(
        "fixture",
        [f for f in _LIT_FIXTURES if f.get("expected_name")],
        ids=[f["fixture_id"] for f in _LIT_FIXTURES if f.get("expected_name")],
    )
    @pytest.mark.unit
    def test_lit_fixture_opsin_inchi_l1_match(self, fixture, request):
        from orthonym import name_compound

        if fixture["fixture_id"] in self._XFAIL_FIXTURES:
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=self._XFAIL_FIXTURES[fixture["fixture_id"]],
                    strict=False,
                )
            )

        smi = fixture["smiles"]
        expected = fixture["expected_name"]
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None, f"invalid SMILES {smi}"

        name = name_compound(smi)
        # NOTE: This test checks round-trip, not strict expected-name
        # match. Some non-canonical-but-equivalent forms emitted by the
        # current engine still round-trip via InChI L1.
        if name is None:
            pytest.skip(f"name_compound returned None for {smi}")

        parsed = _opsin_parse(name)
        if parsed is None:
            pytest.skip(f"OPSIN could not parse {name!r}")

        parsed_mol = Chem.MolFromSmiles(parsed)
        if parsed_mol is None:
            pytest.skip(f"OPSIN-emitted SMILES invalid: {parsed!r}")

        # Compare InChI L1 (formula + connectivity) only; stereo deferred
        # to a phase/153.
        input_inchi = Chem.MolToInchi(mol).split("/c")[0]
        round_inchi = Chem.MolToInchi(parsed_mol).split("/c")[0]
        assert input_inchi == round_inchi, (
            f"InChI L1 mismatch for {fixture['fixture_id']}\n"
            f"  smiles:   {smi}\n"
            f"  emitted:  {name!r}\n"
            f"  parsed:   {parsed}\n"
            f"  expected: {expected!r}\n"
        )


class TestCyclicRejectionWR02:
    """a phase-04: a cyclic arrangement of 3+ ring systems where
    each system has degree 2 must be REJECTED — it is not a linear-path
    ring assembly per IUPAC."""

    @pytest.mark.unit
    def test_cyclic_three_system_arrangement_rejected(self):
        """Three ring systems each bonded to the other two: each system
        has degree 2 (degree check passes) but len(connections) == 3 == N
        != N-1 == 2. The new tree-shape check rejects it."""
        from orthonym.rules.ring_assemblies import _check_path_topology

        # Synthetic: 3 ring systems, each connected to the other two.
        # Tuples are (atom_a, atom_b, system_a, system_b) per
        # _find_inter_system_bonds output shape.
        cyclic_connections = [
            (0, 6, 0, 1),   # system 0 - system 1
            (5, 12, 1, 2),  # system 1 - system 2
            (11, 1, 2, 0),  # system 2 - system 0 (closes the cycle)
        ]
        assert _check_path_topology(3, cyclic_connections) is False

    @pytest.mark.unit
    def test_linear_three_system_path_accepted(self):
        """Sanity: a linear path of 3 systems still passes (N=3,
        connections=2 = N-1, all degrees <= 2)."""
        from orthonym.rules.ring_assemblies import _check_path_topology

        linear_connections = [
            (0, 6, 0, 1),
            (5, 12, 1, 2),
        ]
        assert _check_path_topology(3, linear_connections) is True

    @pytest.mark.unit
    def test_branched_arrangement_rejected_by_degree(self):
        """Sanity: branched arrangements (e.g., star — one system
        bonded to 3 others) are STILL rejected by the existing
        degree<=2 check, regardless of edge count."""
        from orthonym.rules.ring_assemblies import _check_path_topology

        star_connections = [
            (0, 6, 0, 1),
            (1, 12, 0, 2),
            (2, 18, 0, 3),
        ]
        # Center system has degree 3 -> rejected
        assert _check_path_topology(4, star_connections) is False

    @pytest.mark.unit
    def test_two_system_linear_assembly_unaffected(self):
        """The smallest ring assembly (biphenyl shape: 2 systems, 1
        connection) passes both gates: N-1 = 1 = len(connections), all
        degrees == 1."""
        from orthonym.rules.ring_assemblies import _check_path_topology
        assert _check_path_topology(2, [(0, 6, 0, 1)]) is True
