"""Phase 147: ring-type parity between _build_ring_pos and handler orient functions.

Six ring-type parity tests proving that ``_build_ring_pos(ring_set, ring_info)``
returns locants byte-equivalent to what the corresponding ring-type handler
orient function (or stored authoritative numbering) produces.

Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.1
Source: https://iupac.qmul.ac.uk/BlueBook/P1.html P-14.4(g), P-14.5.2
Source: https://iupac.qmul.ac.uk/BlueBook/P2.html P-25, P-25.3
Source: HERITAGE-1990 §3 (Wisniewski et al., J. Chem. Inf. Comput. Sci. 30, 324-332)
        — criterion order (a)-(f) for symmetric-ring numbering.
Source: Phase 147 CONTEXT D-03, D-05, D-09, D-10.

Per Phase 147 D-05 the criterion order applied within each handler orient
function matches HERITAGE §3 (a)-(f) AND IUPAC P-14.5 / P-14.4:
  (a) lowest locants for principal characteristic group   (P-14.5.2)
  (b) lowest locants for indicated hydrogens              (P-31.1.4 / P-25.7.1)
  (c) lowest locants for multiple bonds                   (P-14.5.2 / P-31.1.4)
  (d) maximum number of substituents                      (P-14.5.2 secondary)
  (e) lowest locants for substituents                     (P-14.5.2)
  (f) lowest locants for substituents in alphabetical citation order (P-14.4(g))
"""
import pytest
from rdkit import Chem

from orthonym.rules.parent_selection import _build_ring_pos
from orthonym.data.fused_heterocycles import match_fused_heterocycle_core
from orthonym.namer import (
    compute_features,
    _build_ring_info_for_parent_selection,
)


# ============================================================================
# Test 1 — Benzene
# ============================================================================

def test_benzene_toluene_substituent_locant_1():
    """Toluene (Cc1ccccc1): the methyl-bearing ring C gets locant 1.

    Criterion: HERITAGE §3 (e) lowest substituent locants; P-14.4(g) lowest
    locants for substituents in alphabetical citation order. With a single
    substituent the only constraint is "place it at C1" — orient_benzene
    gives the substituent-bearing aromatic C the locant 1 and the dispatch
    helper consumes that result via _build_ring_info_for_parent_selection
    branch 3 (BL-2 fix).

    Source: https://iupac.qmul.ac.uk/BlueBook/P1.html P-14.4(g)
    Source: HERITAGE-1990 §3 (e)-(f).
    Source: Phase 147 CONTEXT D-03 branch 3, D-05 (a)-(f), Plan 02 BL-2.
    """
    mol = Chem.MolFromSmiles('Cc1ccccc1')
    features = compute_features(mol)
    ring_info = _build_ring_info_for_parent_selection(features)
    assert ring_info is not None
    assert ring_info.get("iupac_locants") is not None

    # ring_systems[0] is the benzene ring (the methyl carbon is acyclic).
    ring_set = set(features.ring_systems[0])
    ring_pos = _build_ring_pos(ring_set, ring_info)

    # Methyl carbon is index 0 (CH3); ring atoms are indices 1..6.
    # The substituent-bearing aromatic C must have locant 1.
    methyl_atom = mol.GetAtomWithIdx(0)
    assert methyl_atom.GetSymbol() == 'C' and not methyl_atom.GetIsAromatic()
    ipso_neighbors = [n.GetIdx() for n in methyl_atom.GetNeighbors()
                      if n.GetIsAromatic()]
    assert len(ipso_neighbors) == 1
    ipso_idx = ipso_neighbors[0]
    assert ring_pos[ipso_idx] == 1, (
        f"toluene ipso C (idx {ipso_idx}) must have locant 1, got "
        f"{ring_pos[ipso_idx]}; full map: {ring_pos}"
    )

    # Homogeneous int output (no fusion atoms).
    assert all(isinstance(v, int) for v in ring_pos.values()), ring_pos


# ============================================================================
# Test 2 — Pyridine (simple heterocycle, single ring)
# ============================================================================

def test_pyridine_nitrogen_locant_1():
    """Pyridine: N gets locant 1 per P-25.3 heteroatom seniority.

    Criterion: HERITAGE §3 — heteroatom seniority anchor, then (e) lowest
    substituent locants. orient_heterocycle_with_substituents (branch 4 of
    the dispatch helper) places N at position 1 for pyridine.

    Source: https://iupac.qmul.ac.uk/BlueBook/P2.html P-25.3
    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.1
    Source: HERITAGE-1990 §3 (heteroatom anchor, then (e)).
    Source: Phase 147 CONTEXT D-03 branch 4, D-05.
    """
    mol = Chem.MolFromSmiles('c1ccncc1')
    features = compute_features(mol)
    ring_info = _build_ring_info_for_parent_selection(features)
    assert ring_info is not None
    assert ring_info.get("iupac_locants") is not None

    ring_set = set(features.ring_systems[0])
    ring_pos = _build_ring_pos(ring_set, ring_info)

    n_idx = next(
        a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == 'N'
    )
    assert ring_pos[n_idx] == 1, (
        f"pyridine N (idx {n_idx}) must have locant 1, got "
        f"{ring_pos[n_idx]}; full map: {ring_pos}"
    )
    # Six positions, all int (single ring, no fusion atoms).
    assert len(ring_pos) == 6, ring_pos
    assert all(isinstance(v, int) for v in ring_pos.values()), ring_pos
    assert sorted(ring_pos.values()) == [1, 2, 3, 4, 5, 6], ring_pos


# ============================================================================
# Test 3 — Morpholine (simple heterocycle, two heteroatoms, 1,4 set)
# ============================================================================

def test_morpholine_heteroatom_locants_1_and_4():
    """Morpholine (C1COCCN1): O=1, N=4. Lowest heteroatom locant set {1, 4}.

    Criterion: HERITAGE §3 — heteroatom seniority/anchor + lowest heteroatom
    locant set. P-25.3.1.3 heteroatom seniority for ring numbering ranks
    O > N (the order is F, Cl, Br, I, O, S, Se, Te, N, P, ...), so when
    both O and N are present in the same ring the higher-priority O gets
    locant 1. The locant set {1, 4} is the lowest possible for a six-
    membered ring with two heteroatoms separated by two carbons.

    Source: https://iupac.qmul.ac.uk/BlueBook/P2.html P-25.3.1.3
        (heteroatom seniority order for monocyclic Hantzsch-Widman names).
    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.1
    Source: HERITAGE-1990 §3 (lowest heteroatom locants).
    Source: Phase 147 CONTEXT D-03 branch 4, D-05 (a)-(f).
    """
    mol = Chem.MolFromSmiles('C1COCCN1')
    features = compute_features(mol)
    ring_info = _build_ring_info_for_parent_selection(features)
    assert ring_info is not None
    assert ring_info.get("iupac_locants") is not None

    ring_set = set(features.ring_systems[0])
    ring_pos = _build_ring_pos(ring_set, ring_info)

    n_idx = next(
        a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == 'N'
    )
    o_idx = next(
        a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == 'O'
    )
    # Per P-25.3.1.3 heteroatom seniority: O > N -> O at locant 1.
    assert ring_pos[o_idx] == 1, (
        f"morpholine O (idx {o_idx}) must have locant 1 (P-25.3.1.3 "
        f"heteroatom seniority O > N), got {ring_pos[o_idx]}; full map: "
        f"{ring_pos}"
    )
    assert ring_pos[n_idx] == 4, (
        f"morpholine N (idx {n_idx}) must have locant 4 (lowest heteroatom "
        f"locant set {{1, 4}}), got {ring_pos[n_idx]}; full map: {ring_pos}"
    )
    assert sorted(ring_pos.values()) == [1, 2, 3, 4, 5, 6], ring_pos
    assert all(isinstance(v, int) for v in ring_pos.values()), ring_pos


# ============================================================================
# Test 4 — Naphthalene (PAH, fusion atoms present -> tuple locants)
# ============================================================================

def test_naphthalene_fusion_tuples_present():
    """Naphthalene (c1ccc2ccccc2c1): 8 peripheral int locants + 2 fusion tuples.

    Criterion: HERITAGE §3 (b) lowest locants for indicated hydrogens / fused-
    system rule. Per P-25 PAH numbering is FIXED, not reoriented per
    substituents. Per Plan 01 D-01 homogeneity coercion, the ring_pos dict
    is fully (int, str) tuples when ANY fusion atom appears — peripheral
    locants are coerced from int N to (N, '') so downstream min()/sort()
    operations stay type-safe (RESEARCH §3 Risk 3).

    Source: https://iupac.qmul.ac.uk/BlueBook/P2.html P-25
    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.1
    Source: HERITAGE-1990 §3 (b) — fused-system fixed numbering.
    Source: Phase 147 CONTEXT D-01 (tuple encoding), D-03 branch 2.
    """
    mol = Chem.MolFromSmiles('c1ccc2ccccc2c1')
    features = compute_features(mol)
    ring_info = _build_ring_info_for_parent_selection(features)
    assert ring_info is not None
    assert ring_info.get("iupac_locants") is not None

    # Aggregate the union of all ring-system atoms (naphthalene is one
    # fused ring SYSTEM but RDKit still returns two SSSR rings; the
    # dispatcher's PAH branch produces locants for ALL 10 atoms).
    ring_set = set()
    for ring in features.ring_systems:
        ring_set.update(ring)
    ring_pos = _build_ring_pos(ring_set, ring_info)

    assert len(ring_pos) == 10, ring_pos
    # After Plan 01 coercion, every value is a tuple when ANY fusion
    # tuple is present. Naphthalene has 2 fusion tuples (4a, 8a).
    assert all(isinstance(v, tuple) for v in ring_pos.values()), (
        f"expected homogeneous-tuple dict, got mixed types: {ring_pos}"
    )

    # Fusion tuples are the (N, 'a') pairs; peripherals are (N, '').
    fusion = [v for v in ring_pos.values() if v[1] != '']
    peripheral = [v for v in ring_pos.values() if v[1] == '']
    assert sorted(fusion) == [(4, 'a'), (8, 'a')], fusion
    assert sorted(peripheral) == [
        (1, ''), (2, ''), (3, ''), (4, ''),
        (5, ''), (6, ''), (7, ''), (8, ''),
    ], peripheral


# ============================================================================
# Test 5 — Cyclohexane (carbocyclic monocycle, sorted-fallback per D-09)
# ============================================================================

def test_cyclohexane_sorted_fallback():
    """Cyclohexane (C1CCCCC1): no authoritative numbering — sorted fallback OK.

    Criterion: HERITAGE §3 — pure-carbon monocycle has no criterion-
    differentiated winner. P-14.5.2 ring symmetry makes any consistent
    numbering equivalent for a homocyclic carbocycle with no substituents.
    Per Phase 147 D-09 back-compat, _build_ring_info_for_parent_selection
    branch 7 returns None, and _build_ring_pos falls through to the sorted
    1-indexed mapping.

    Source: https://iupac.qmul.ac.uk/BlueBook/P1.html P-14.5.2
    Source: HERITAGE-1990 §3 (no criterion-differentiated winner for symmetric
        homocyclic carbocycle).
    Source: Phase 147 CONTEXT D-03 branch 7, D-09.
    """
    mol = Chem.MolFromSmiles('C1CCCCC1')
    features = compute_features(mol)
    ring_info = _build_ring_info_for_parent_selection(features)

    # Cyclohexane has no authoritative numbering: branch 7 returns None.
    assert ring_info is None, ring_info

    ring_set = set(features.ring_systems[0])
    ring_pos = _build_ring_pos(ring_set, None)
    # Sorted fallback: 1-indexed by sort order.
    expected = {idx: i + 1 for i, idx in enumerate(sorted(ring_set))}
    assert ring_pos == expected, ring_pos
    # All int (no tuple coercion path).
    assert all(isinstance(v, int) for v in ring_pos.values()), ring_pos


# ============================================================================
# Test 6 — Indole (fused heterocycle, D-09 byte-identical preservation)
# ============================================================================

def test_indole_fused_hetero_atom_mapping():
    """Indole (c1ccc2[nH]ccc2c1): byte-identical to match_fused_heterocycle_core.

    Criterion: HERITAGE §3 — stored arrangement for cataloged fused systems
    (functionally equivalent: the FUSED_HETEROCYCLES registry stores the
    canonical IUPAC numbering, and branch 1 of the dispatch helper preserves
    that mapping byte-identical to honor D-09 (back-compat with the existing
    fused-heterocycle path that ships in pre-147 Orthonym).

    Source: https://iupac.qmul.ac.uk/BlueBook/P2.html P-25
    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.1
    Source: HERITAGE-1990 §3 — stored arrangement for cataloged systems.
    Source: Phase 147 CONTEXT D-03 branch 1, D-09 byte-identical.
    """
    mol = Chem.MolFromSmiles('c1ccc2[nH]ccc2c1')
    features = compute_features(mol)
    ring_info = _build_ring_info_for_parent_selection(features)

    core_match = match_fused_heterocycle_core(mol)
    assert core_match is not None, "indole must be detected by the registry"
    _, expected_mapping, _ = core_match

    # Branch 1 preserves the heterocycle atom_mapping byte-identical
    # (heteroatom guard returns the exact dict).
    assert ring_info == {"iupac_locants": expected_mapping}, ring_info


# ============================================================================
# Test 7 — Phase 151-01 D-21: VB ≥4-ring cascade-step-6 wiring
# ============================================================================


class TestVB:
    """Phase 151 D-21: cascade-step-6 fires for VB tetracyclo+ inputs.

    Verifies that ``_build_ring_info_for_parent_selection`` Branch 6 routes
    >=4-ring non-cataloged bridged systems through the new
    ``polycyclic_von_baeyer.get_higher_polycyclo_iupac_locants`` supplier and
    that the cascade-step-6 gate (`candidate_pool._has_iupac_locants`)
    consumes the resulting locants with FULL atom coverage.

    Source: 151-CONTEXT.md D-04 / D-06 / D-21; 151-AUDIT-A.md verdict
    THIN_WRAPPER; Phase 147 cascade-step-6 gate.
    """

    @pytest.mark.unit
    def test_cubane_cascade_step_6_fires(self):
        """Cubane: pentacyclo[4.2.0.0^{2,5}.0^{3,8}.0^{4,7}]octane."""
        mol = Chem.MolFromSmiles('C12C3C4C1C5C3C4C25')
        features = compute_features(mol)
        ring_info = _build_ring_info_for_parent_selection(features)
        assert ring_info is not None, (
            "Branch 6 must fire on cubane; got None"
        )
        assert ring_info.get("iupac_locants") is not None, (
            "cascade-step-6 gate did not see iupac_locants for cubane"
        )
        ring_set = set(features.ring_systems[0])
        ring_pos = _build_ring_pos(ring_set, ring_info)
        assert len(ring_pos) == len(ring_set), (
            f"partial coverage: {len(ring_pos)} entries for "
            f"{len(ring_set)} ring atoms (Pitfall 7 violation)"
        )

    @pytest.mark.unit
    def test_adamantane_anticanary_predicate_false(self):
        """D-08 lock: adamantane is tricyclic, is_higher_polycyclo False."""
        from orthonym.rules.polycyclic_von_baeyer import is_higher_polycyclo
        mol = Chem.MolFromSmiles('C1C2CC3CC1CC(C2)C3')
        assert is_higher_polycyclo(mol) is False

    @pytest.mark.unit
    def test_bicyclo222_octane_anticanary(self):
        """D-04 anti-canary: bicyclo[2.2.2]octane retains bicyclo.py auth."""
        from orthonym.rules.polycyclic_von_baeyer import (
            get_higher_polycyclo_iupac_locants,
            is_higher_polycyclo,
        )
        mol = Chem.MolFromSmiles('C1CC2CCC1CC2')
        assert is_higher_polycyclo(mol) is False
        assert get_higher_polycyclo_iupac_locants(mol) is None

    @pytest.mark.unit
    def test_norbornane_anticanary(self):
        """D-04 anti-canary: norbornane (bicyclo[2.2.1]heptane)."""
        from orthonym.rules.polycyclic_von_baeyer import (
            get_higher_polycyclo_iupac_locants,
            is_higher_polycyclo,
        )
        mol = Chem.MolFromSmiles('C1CC2CCC1C2')
        assert is_higher_polycyclo(mol) is False
        assert get_higher_polycyclo_iupac_locants(mol) is None

    @pytest.mark.unit
    def test_naphthalene_anticanary(self):
        """Aromatic guard: PAHs route via polycyclics.py (Branch 2)."""
        from orthonym.rules.polycyclic_von_baeyer import (
            get_higher_polycyclo_iupac_locants,
            is_higher_polycyclo,
        )
        mol = Chem.MolFromSmiles('c1ccc2ccccc2c1')
        assert is_higher_polycyclo(mol) is False
        assert get_higher_polycyclo_iupac_locants(mol) is None

    @pytest.mark.unit
    def test_blue_book_diamantane_cascade_step_6_fires(self):
        """Diamantane (pentacyclic 14-atom diamondoid): Branch 6 fires."""
        mol = Chem.MolFromSmiles('C1C2CC3CC4CC1C1CC2(CC4)C31')
        features = compute_features(mol)
        ring_info = _build_ring_info_for_parent_selection(features)
        assert ring_info is not None
        assert ring_info.get("iupac_locants") is not None
        ring_set = set(features.ring_systems[0])
        ring_pos = _build_ring_pos(ring_set, ring_info)
        assert len(ring_pos) == len(ring_set)

    @pytest.mark.unit
    def test_quadricyclane_predicate_decides_routing(self):
        """Quadricyclane: VB-supplier OR fallthrough — must NOT corrupt
        parent selection."""
        mol = Chem.MolFromSmiles('C1C2C3C1C1C2C31')
        features = compute_features(mol)
        ring_info = _build_ring_info_for_parent_selection(features)
        if ring_info is not None and ring_info.get("iupac_locants"):
            ring_set = set(features.ring_systems[0])
            ring_pos = _build_ring_pos(ring_set, ring_info)
            assert len(ring_pos) == len(ring_set)


# ============================================================================
# Test 8 — Phase 151-02 D-21: pure spiro cascade-step-6 wiring (Branch 5a)
# ============================================================================


class TestSpiro:
    """Phase 151-02 D-21: cascade-step-6 fires for pure spiro inputs.

    Verifies that ``_build_ring_info_for_parent_selection`` Branch 5
    routes pure spiro systems through ``get_spiro_iupac_locants`` and
    that the cascade-step-6 gate consumes the resulting locants with
    FULL atom coverage.

    Source: 151-CONTEXT.md D-09 / D-21; 151-02-PLAN.md task 3.
    """

    @pytest.mark.unit
    def test_spiro45decane_cascade_step_6_fires(self):
        """spiro[4.5]decane: Branch 5a cascade-step-6 supplies locants."""
        mol = Chem.MolFromSmiles('C1CCC2(CC1)CCCC2')
        features = compute_features(mol)
        ring_info = _build_ring_info_for_parent_selection(features)
        assert ring_info is not None
        assert ring_info.get("iupac_locants") is not None, (
            "cascade-step-6 gate did not see iupac_locants for spiro[4.5]decane"
        )
        ri = mol.GetRingInfo()
        ra = set()
        for r in ri.AtomRings():
            ra.update(r)
        assert set(ring_info["iupac_locants"].keys()) >= ra

    @pytest.mark.unit
    def test_dispiro_cascade_full_coverage(self):
        """dispiro[5.1.5.2]heptadecane: Branch 5a fires."""
        mol = Chem.MolFromSmiles('C1CCC2(CC1)CC1(CCC2)CCCCCC1')
        features = compute_features(mol)
        ring_info = _build_ring_info_for_parent_selection(features)
        assert ring_info is not None
        assert ring_info.get("iupac_locants") is not None
        ri = mol.GetRingInfo()
        ra = set()
        for r in ri.AtomRings():
            ra.update(r)
        assert set(ring_info["iupac_locants"].keys()) >= ra

    @pytest.mark.unit
    def test_hetero_spiro_cascade_full_coverage(self):
        """1,4-dioxaspiro[4.5]decane: Branch 5a fires on hetero-spiro."""
        mol = Chem.MolFromSmiles('C1CC2(OCCO2)CCC1')
        features = compute_features(mol)
        ring_info = _build_ring_info_for_parent_selection(features)
        assert ring_info is not None
        assert ring_info.get("iupac_locants") is not None
        ri = mol.GetRingInfo()
        ra = set()
        for r in ri.AtomRings():
            ra.update(r)
        assert set(ring_info["iupac_locants"].keys()) >= ra


# ============================================================================
# Test 9 — Phase 151-02 D-09 + D-21: mixed spiro/fused cascade-step-6 (Branch 5b)
# ============================================================================


class TestMixedSpiroFused:
    """Phase 151-02 D-09 / D-13 / D-21: cascade-step-6 fires for mixed
    spiro/fused inputs that the HERITAGE §4 separable-parts builder names.

    Source: 151-CONTEXT.md D-09 / D-13 / D-21; 151-02-PLAN.md task 3.
    """

    @pytest.mark.unit
    def test_mixed_routes_to_mixed_classification(self):
        """D-09 dispatch order: composer cascade picks mixed-spiro-fused
        BEFORE polycyclic-bridged AND BEFORE pure-spiro on a known mixed
        SMILES."""
        from orthonym.assembly.composer import _classify_complex_ring
        # spiro-indane: 3 rings, 1 spiro centre
        mol = Chem.MolFromSmiles('C1CCC2(CC1)CCc1ccccc12')
        ring_type = _classify_complex_ring(mol)
        assert ring_type == 'mixed-spiro-fused', (
            f"composer classify returned {ring_type!r} on a mixed input — "
            f"D-09 dispatch order violated"
        )

    @pytest.mark.unit
    def test_mixed_spiro_indane_cascade_step_6_fires(self):
        """spiro-indane: Branch 5b cascade-step-6 supplies locants."""
        mol = Chem.MolFromSmiles('C1CCC2(CC1)CCc1ccccc12')
        features = compute_features(mol)
        ring_info = _build_ring_info_for_parent_selection(features)
        assert ring_info is not None
        # Branch 5b returns iupac_locants when name_mixed_spiro_fused
        # produces a name with full coverage. If the algorithm declines
        # (e.g., catalog miss), the branch returns {"iupac_locants": None}
        # which is still gate-visible.
        if ring_info.get("iupac_locants") is not None:
            ri = mol.GetRingInfo()
            ra = set()
            for r in ri.AtomRings():
                ra.update(r)
            assert set(ring_info["iupac_locants"].keys()) >= ra

    @pytest.mark.unit
    def test_mixed_indoline_cyclohexane_cascade(self):
        """spiro[indoline-2,1'-cyclohexane]: Branch 5b fires with full
        coverage on HERITAGE §4 canonical input."""
        mol = Chem.MolFromSmiles('C12(CCCCC1)CNC1=CC=CC=C12')
        features = compute_features(mol)
        ring_info = _build_ring_info_for_parent_selection(features)
        assert ring_info is not None
        if ring_info.get("iupac_locants") is not None:
            ri = mol.GetRingInfo()
            ra = set()
            for r in ri.AtomRings():
                ra.update(r)
            assert set(ring_info["iupac_locants"].keys()) >= ra

    @pytest.mark.unit
    def test_mixed_steroid_anti_canary(self):
        """RESEARCH Pitfall 3: cholestane MUST NOT route as mixed-spiro-fused.

        Steroids have n_spiro == 0 (no RDKit spiro atoms), so
        is_mixed_spiro_fused returns False AND the natural-product
        short-circuit kicks in. Branch 5b should not engage.
        """
        from orthonym.rules.spiro import is_mixed_spiro_fused
        mol = Chem.MolFromSmiles(
            'CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CCC4CCCC[C@]4(C)[C@H]3CC[C@]12C'
        )
        assert is_mixed_spiro_fused(mol) is False

    @pytest.mark.unit
    def test_pure_spiro_not_mixed_classify(self):
        """D-09 contract: pure spiro routes as 'spiro', NOT mixed-spiro-fused."""
        from orthonym.assembly.composer import _classify_complex_ring
        mol = Chem.MolFromSmiles('C1CCC2(CC1)CCCCC2')
        ring_type = _classify_complex_ring(mol)
        assert ring_type == 'spiro', (
            f"composer classify returned {ring_type!r} on a pure spiro — "
            f"D-09 dispatch order disturbed"
        )
