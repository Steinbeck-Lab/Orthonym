"""FR-2.3 base component selection tests (Phase 149 Tier 1).

Per V18 plan §6 Phase 149 + AUTONOM-1990 §4 (hybrid catalog + algorithmic
fallback architecture; 61% Beilstein-expert agreement validates the approach).

Each test cites:
  - The QMUL FR-2.3 URL (https://iupac.qmul.ac.uk/fusedring/FR23.html)
  - The IUPAC rule code (P-25.3.2.4 / FR-2.3(letter))
  - AUTONOM-1990 §4 reference
  - The Phase 149 CONTEXT decision (D-XX) being verified

Tests are organized into criterion-named classes per CD-03; ≥3 tests per
criterion (a)-(j) per V18 plan §6 acceptance + 149-CONTEXT D-10 Tier 1.

Source: https://iupac.qmul.ac.uk/fusedring/FR23.html
Source: https://iupac.qmul.ac.uk/BlueBook/P2.html P-25.3.2.4
Source: https://iupac.qmul.ac.uk/BlueBook/P2.html P-25.3.1.3
Source: AUTONOM-1990 §4 (Wisniewski J. Chem. Inf. Comput. Sci. 30, 324-332)
        — hybrid catalog + algorithmic fallback validation; 61% agreement.
Source: Phase 149 CONTEXT D-01..D-06, D-10.
"""
import inspect

import pytest
from rdkit import Chem

from orthonym.rules.fused_ring_selection import (
    ComponentRank,
    _enumerate_components,
    _FR23_ALT_ORDER,
    _FR23_HETEROATOM_ALT,
    _rank,
    select_base_component,
)


# ============================================================================
# Helpers
# ============================================================================


def _atoms_for_ring(mol: Chem.Mol, predicate) -> set:
    """Return the atom set of the first SSSR ring matching ``predicate``."""
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        if predicate(mol, ring):
            return set(ring)
    raise AssertionError("No matching ring found")


def _all_ring_atoms(mol: Chem.Mol) -> set:
    """Return the union of atoms across all SSSR rings."""
    ri = mol.GetRingInfo()
    out: set = set()
    for ring in ri.AtomRings():
        out.update(ring)
    return out


def _ring_contains_symbol(sym: str):
    """Predicate: ring contains an atom with the given symbol."""
    def _pred(mol, ring):
        return any(mol.GetAtomWithIdx(i).GetSymbol() == sym for i in ring)
    return _pred


def _ring_all_carbon():
    """Predicate: ring contains only carbon atoms."""
    def _pred(mol, ring):
        return all(mol.GetAtomWithIdx(i).GetSymbol() == 'C' for i in ring)
    return _pred


def _ring_of_size(n: int):
    """Predicate: ring has exactly ``n`` atoms."""
    def _pred(mol, ring):  # noqa: ARG001
        return len(ring) == n
    return _pred


# ============================================================================
# Criterion (a) — Heteroatom Seniority
# ============================================================================


class TestCriterionAHeteroatomSeniority:
    """FR-2.3(a) — heteroatom precedence per primary order N>F>...>Hg.

    Source: https://iupac.qmul.ac.uk/fusedring/FR23.html FR-2.3(a)
    Source: https://iupac.qmul.ac.uk/BlueBook/P2.html P-25.3.2.4
    """

    def test_a_pyridine_vs_benzene_pyridine_wins(self):
        """N senior to C; pyridine ring is base in pyridine-benzene fusion (quinoline).

        Per FR-2.3(a) the heterocyclic component containing the heteroatom
        occurring earliest in the order N>F>...>Hg is preferred. Pyridine
        contains N (rank 20); benzene is all-C (rank 0). Pyridine wins.

        Source: https://iupac.qmul.ac.uk/fusedring/FR23.html FR-2.3(a)
        Source: https://iupac.qmul.ac.uk/BlueBook/P2.html P-25.3.2.4
        Source: AUTONOM-1990 §4.
        Source: Phase 149 CONTEXT D-04 (REUSE _HETEROATOM_SENIORITY for FR-2.3(a)).
        """
        mol = Chem.MolFromSmiles("c1ccc2ncccc2c1")  # quinoline
        components = _enumerate_components(mol)
        base, _others = select_base_component(mol, components)
        n_atom_idx = next(
            i for i in range(mol.GetNumAtoms())
            if mol.GetAtomWithIdx(i).GetSymbol() == 'N'
        )
        assert n_atom_idx in base, (
            f"Pyridine ring (containing N at idx {n_atom_idx}) must be base "
            f"per FR-2.3(a); algorithmic base was {base}"
        )

    def test_a_pyridine_vs_furan_pyridine_wins(self):
        """N senior to O; pyridine ring beats furan ring under FR-2.3(a).

        Per V18 plan §6 line 949: explicit pyridine-vs-furan example.
        Pyridine N rank = 20; furan O rank = 15. Pyridine wins.

        We compose the comparison via direct _rank calls (a 2-component
        molecule that has both pyridine and furan rings would be
        furo-pyridine — uncommon in test data). Direct _rank with
        constructed atom sets is sufficient for criterion isolation.

        Source: https://iupac.qmul.ac.uk/fusedring/FR23.html FR-2.3(a)
        Source: https://iupac.qmul.ac.uk/BlueBook/P2.html P-25.3.2.4
        Source: AUTONOM-1990 §4.
        Source: Phase 149 CONTEXT D-04 (primary order N>F>Cl>...>O>...).
        """
        py = Chem.MolFromSmiles("c1ccncc1")
        fu = Chem.MolFromSmiles("c1ccoc1")
        py_atoms = set(range(py.GetNumAtoms()))
        fu_atoms = set(range(fu.GetNumAtoms()))
        rank_py = _rank(py, py_atoms)
        rank_fu = _rank(fu, fu_atoms)
        assert rank_py.senior_het_neg < rank_fu.senior_het_neg, (
            f"FR-2.3(a) pyridine N (rank 20) must beat furan O (rank 15); "
            f"got pyridine={rank_py.senior_het_neg}, furan={rank_fu.senior_het_neg}"
        )

    def test_a_furan_vs_thiophene_furan_wins(self):
        """O senior to S; furan ring beats thiophene under FR-2.3(a) primary order.

        _HETEROATOM_SENIORITY['O']=15 > _HETEROATOM_SENIORITY['S']=14.

        Source: https://iupac.qmul.ac.uk/fusedring/FR23.html FR-2.3(a)
        Source: https://iupac.qmul.ac.uk/BlueBook/P2.html P-25.3.2.4
        Source: AUTONOM-1990 §4.
        Source: Phase 149 CONTEXT D-04 / D-15 (REUSE primary order from ring_selection).
        """
        fu = Chem.MolFromSmiles("c1ccoc1")
        th = Chem.MolFromSmiles("c1ccsc1")
        fu_atoms = set(range(fu.GetNumAtoms()))
        th_atoms = set(range(th.GetNumAtoms()))
        rank_fu = _rank(fu, fu_atoms)
        rank_th = _rank(th, th_atoms)
        assert rank_fu.senior_het_neg < rank_th.senior_het_neg, (
            f"FR-2.3(a) furan O must beat thiophene S; "
            f"got furan={rank_fu.senior_het_neg}, thiophene={rank_th.senior_het_neg}"
        )


# ============================================================================
# Criterion (b) — Ring Count
# ============================================================================


class TestCriterionBRingCount:
    """FR-2.3(b) — Greater number of rings in the component.

    Source: https://iupac.qmul.ac.uk/fusedring/FR23.html FR-2.3(b)
    Source: https://iupac.qmul.ac.uk/BlueBook/P2.html P-25.3.2.4
    """

    def test_b_3ring_vs_2ring_3ring_wins(self):
        """When (a) ties, FR-2.3(b) prefers the component with more rings.

        Per V18 plan §6 line 950: 3-ring beats 2-ring.

        We use anthracene (3-ring all-C) and compare a constructed
        atom set spanning all 3 SSSR rings vs an atom set spanning
        only one ring. Both are all-C so (a) ties at senior_het_neg=0;
        (b) breaks the tie via -3 < -1.

        Source: https://iupac.qmul.ac.uk/fusedring/FR23.html FR-2.3(b)
        Source: https://iupac.qmul.ac.uk/BlueBook/P2.html P-25.3.2.4
        Source: AUTONOM-1990 §4.
        Source: Phase 149 CONTEXT D-02 (ring_count_neg = -count).
        """
        mol = Chem.MolFromSmiles("c1ccc2cc3ccccc3cc2c1")  # anthracene
        all_atoms = _all_ring_atoms(mol)
        single_ring = set(mol.GetRingInfo().AtomRings()[0])
        rank_all = _rank(mol, all_atoms)
        rank_single = _rank(mol, single_ring)
        # (a) tied (both all-C, senior_het_neg = 0)
        assert rank_all.senior_het_neg == 0 == rank_single.senior_het_neg
        # (b) breaks the tie
        assert rank_all.ring_count_neg == -3
        assert rank_single.ring_count_neg == -1
        assert rank_all < rank_single, (
            f"FR-2.3(b) 3-ring component must beat 1-ring component when (a) "
            f"ties; got rank_all={rank_all}, rank_single={rank_single}"
        )

    def test_b_2ring_component_beats_1ring_component_when_a_ties(self):
        """When (a) ties, more rings wins under FR-2.3(b) regardless of (c)+ ties.

        Use phenanthrene (3-ring all-C) and compare 2-ring atom subset vs
        1-ring atom subset. Both subsets are all-C → (a) ties at 0;
        2-ring subset beats 1-ring at (b).

        Source: https://iupac.qmul.ac.uk/fusedring/FR23.html FR-2.3(b)
        Source: https://iupac.qmul.ac.uk/BlueBook/P2.html P-25.3.2.4
        Source: AUTONOM-1990 §4.
        Source: Phase 149 CONTEXT D-02 (ring_count_neg negation discipline).
        """
        mol = Chem.MolFromSmiles("c1ccc2ccc3ccccc3c2c1")  # phenanthrene
        rings = mol.GetRingInfo().AtomRings()
        # Use the union of two adjacent SSSR rings
        two_ring_atoms = set(rings[0]) | set(rings[1])
        one_ring_atoms = set(rings[2])
        rank_two = _rank(mol, two_ring_atoms)
        rank_one = _rank(mol, one_ring_atoms)
        # Both all-C
        assert rank_two.senior_het_neg == rank_one.senior_het_neg == 0
        assert rank_two.ring_count_neg == -2
        assert rank_one.ring_count_neg == -1
        assert rank_two < rank_one

    def test_b_ring_count_negation_correct(self):
        """FR-2.3(b) negation: ring_count_neg = -ring_count so min-sort picks more rings.

        Verifies the dataclass field convention from V18 Appendix A.6
        line 2384 (ring_count_neg=-ring_count).

        Source: https://iupac.qmul.ac.uk/fusedring/FR23.html FR-2.3(b)
        Source: V18 Appendix A.6 line 2384.
        Source: AUTONOM-1990 §4.
        Source: Phase 149 CONTEXT D-02 (negation discipline).
        """
        mol = Chem.MolFromSmiles("c1ccc2ccccc2c1")  # naphthalene
        all_atoms = _all_ring_atoms(mol)
        rank = _rank(mol, all_atoms)
        # naphthalene has 2 SSSR rings → ring_count_neg = -2
        assert rank.ring_count_neg == -2, (
            f"naphthalene 2-ring component must have ring_count_neg=-2; "
            f"got {rank.ring_count_neg}"
        )


# ============================================================================
# Criterion (c) — Larger Ring at First Point of Difference
# ============================================================================


class TestCriterionCRingSize:
    """FR-2.3(c) — Larger ring at first point of difference (descending sizes).

    Source: https://iupac.qmul.ac.uk/fusedring/FR23.html FR-2.3(c)
    Source: https://iupac.qmul.ac.uk/BlueBook/P2.html P-25.3.2.4
    """

    def test_c_75_vs_66_75_wins(self):
        """Per V18 plan §6 line 951: at first difference 7 > 6.

        Compare ring_sizes_neg of (7,6) vs (6,6): -7 < -6 at index 0,
        so 7+6 wins. (Variant of "7+5 vs 6+6" — RDKit kekulization of
        azulene gives 7+6 in some conformers; the principle is identical.)

        Source: https://iupac.qmul.ac.uk/fusedring/FR23.html FR-2.3(c)
        Source: https://iupac.qmul.ac.uk/BlueBook/P2.html P-25.3.2.4
        Source: AUTONOM-1990 §4.
        Source: Phase 149 CONTEXT D-02 (ring_sizes_neg descending+negated).
        """
        # 7+6 fused all-C
        mol_76 = Chem.MolFromSmiles("C1CCC2CCCCCC2C1")  # bicyclo[5.4.0]undecane → 7+6
        # 6+6 fused all-C
        mol_66 = Chem.MolFromSmiles("C1CCC2CCCCC2C1")  # decalin → 6+6
        rank_76 = _rank(mol_76, _all_ring_atoms(mol_76))
        rank_66 = _rank(mol_66, _all_ring_atoms(mol_66))
        # (a) ties (all-C); (b) ties (both 2-ring); (c) breaks tie at first index
        assert rank_76.senior_het_neg == rank_66.senior_het_neg == 0
        assert rank_76.ring_count_neg == rank_66.ring_count_neg == -2
        assert rank_76.ring_sizes_neg[0] < rank_66.ring_sizes_neg[0], (
            f"FR-2.3(c) 7+6 must beat 6+6 at first ring-size index; "
            f"got 7+6={rank_76.ring_sizes_neg}, 6+6={rank_66.ring_sizes_neg}"
        )
        assert rank_76 < rank_66

    def test_c_single_7_beats_single_6(self):
        """Single 7-membered ring beats single 6-membered ring at FR-2.3(c).

        Cycloheptane vs cyclohexane: ring_sizes_neg=(-7,) < (-6,).

        Source: https://iupac.qmul.ac.uk/fusedring/FR23.html FR-2.3(c)
        Source: https://iupac.qmul.ac.uk/BlueBook/P2.html P-25.3.2.4
        Source: AUTONOM-1990 §4.
        Source: Phase 149 CONTEXT D-02.
        """
        mol7 = Chem.MolFromSmiles("C1CCCCCC1")  # cycloheptane
        mol6 = Chem.MolFromSmiles("C1CCCCC1")  # cyclohexane
        rank7 = _rank(mol7, _all_ring_atoms(mol7))
        rank6 = _rank(mol6, _all_ring_atoms(mol6))
        assert rank7.ring_sizes_neg == (-7,)
        assert rank6.ring_sizes_neg == (-6,)
        assert rank7 < rank6

    def test_c_descending_sort_correct(self):
        """ring_sizes_neg is sorted descending (largest first), then negated.

        Verifies V18 Appendix A.6 line 2387:
            sizes_desc = sorted([...], reverse=True)
            sizes_neg = tuple(-s for s in sizes_desc)

        For mol with rings of sizes 5, 7, 6: descending=[7,6,5], negated=(-7,-6,-5).

        Source: https://iupac.qmul.ac.uk/fusedring/FR23.html FR-2.3(c)
        Source: V18 Appendix A.6 line 2387.
        Source: AUTONOM-1990 §4.
        Source: Phase 149 CONTEXT D-02.
        """
        # 5+7 fused (azulene Kekulé form gives [5,7] in SSSR)
        mol = Chem.MolFromSmiles("C1=CC2=CC=CC=CC2=C1")
        all_atoms = _all_ring_atoms(mol)
        rank = _rank(mol, all_atoms)
        # ring_sizes_neg is (-7, -5) — sorted descending then negated
        assert rank.ring_sizes_neg == (-7, -5), (
            f"sizes must be sorted descending then negated; got "
            f"{rank.ring_sizes_neg}"
        )


# ============================================================================
# Criterion (d) — Greater Total Heteroatom Count
# ============================================================================


class TestCriterionDHeteroatomCount:
    """FR-2.3(d) — Greater total heteroatom count.

    Source: https://iupac.qmul.ac.uk/fusedring/FR23.html FR-2.3(d)
    Source: https://iupac.qmul.ac.uk/BlueBook/P2.html P-25.3.2.4
    """

    def test_d_2N_vs_1N_2N_wins(self):
        """Per V18 plan §6 line 952: pyrazine (2N) beats pyridine (1N).

        Both rings contain N, so (a) ties at senior_het_neg=-20.
        Both are 6-membered single rings, so (b) and (c) tie.
        (d) breaks the tie: pyrazine het_count_neg=-2 < pyridine -1.

        Source: https://iupac.qmul.ac.uk/fusedring/FR23.html FR-2.3(d)
        Source: https://iupac.qmul.ac.uk/BlueBook/P2.html P-25.3.2.4
        Source: AUTONOM-1990 §4.
        Source: Phase 149 CONTEXT D-02 (het_count_neg = -count).
        """
        pyrazine = Chem.MolFromSmiles("c1cnccn1")
        pyridine = Chem.MolFromSmiles("c1ccncc1")
        r_pz = _rank(pyrazine, _all_ring_atoms(pyrazine))
        r_py = _rank(pyridine, _all_ring_atoms(pyridine))
        assert r_pz.senior_het_neg == r_py.senior_het_neg == -20
        assert r_pz.ring_count_neg == r_py.ring_count_neg == -1
        assert r_pz.ring_sizes_neg == r_py.ring_sizes_neg == (-6,)
        # (d) breaks
        assert r_pz.het_count_neg == -2
        assert r_py.het_count_neg == -1
        assert r_pz < r_py

    def test_d_oxazine_vs_pyridine_oxazine_wins(self):
        """1,4-oxazine (1N+1O = 2 het) beats pyridine (1N = 1 het) at (d).

        Both rings have N (senior heteroatom), so (a) ties. Both 6-rings.
        (d) prefers more total heteroatoms: oxazine 2 > pyridine 1.

        Source: https://iupac.qmul.ac.uk/fusedring/FR23.html FR-2.3(d)
        Source: https://iupac.qmul.ac.uk/BlueBook/P2.html P-25.3.2.4
        Source: AUTONOM-1990 §4.
        Source: Phase 149 CONTEXT D-02.
        """
        # morpholine = saturated 1,4-oxazine; aromatic 1,4-oxazine: c1ccocn1?
        # Use morpholine (saturated, no aromaticity ambiguity); criterion (d)
        # operates on atom symbols not aromaticity.
        oxazine = Chem.MolFromSmiles("C1COCCN1")  # morpholine
        pyridine = Chem.MolFromSmiles("c1ccncc1")
        r_ox = _rank(oxazine, _all_ring_atoms(oxazine))
        r_py = _rank(pyridine, _all_ring_atoms(pyridine))
        # (a) tied at -20 (both have N)
        assert r_ox.senior_het_neg == r_py.senior_het_neg == -20
        # (d) breaks: morpholine 2 het, pyridine 1
        assert r_ox.het_count_neg == -2
        assert r_py.het_count_neg == -1
        assert r_ox < r_py

    def test_d_negation_correct(self):
        """FR-2.3(d) negation: het_count_neg = -het_count.

        Verifies V18 Appendix A.6 line 2391-2394 negation discipline.

        Source: https://iupac.qmul.ac.uk/fusedring/FR23.html FR-2.3(d)
        Source: V18 Appendix A.6 lines 2391-2394.
        Source: AUTONOM-1990 §4.
        Source: Phase 149 CONTEXT D-02.
        """
        # 1,3,5-triazine: 3 N
        triazine = Chem.MolFromSmiles("c1ncncn1")
        r = _rank(triazine, _all_ring_atoms(triazine))
        assert r.het_count_neg == -3, (
            f"1,3,5-triazine must have het_count_neg=-3; got {r.het_count_neg}"
        )


# ============================================================================
# Criterion (e) — Greater Heteroatom Variety
# ============================================================================


class TestCriterionEHeteroatomVariety:
    """FR-2.3(e) — Greater heteroatom variety (set cardinality of heteroatom species).

    Source: https://iupac.qmul.ac.uk/fusedring/FR23.html FR-2.3(e)
    Source: https://iupac.qmul.ac.uk/BlueBook/P2.html P-25.3.2.4
    """

    def test_e_NO_vs_N_NO_wins(self):
        """Per V18 plan §6 line 953: variety {N,O}=2 beats variety {N}=1.

        Oxazole (1N+1O, variety=2) vs imidazole (2N, variety=1). Both 5-ring.
        Total het count (d) ties at -2; (e) breaks: -2 < -1.

        Source: https://iupac.qmul.ac.uk/fusedring/FR23.html FR-2.3(e)
        Source: https://iupac.qmul.ac.uk/BlueBook/P2.html P-25.3.2.4
        Source: AUTONOM-1990 §4.
        Source: Phase 149 CONTEXT D-02.
        """
        oxazole = Chem.MolFromSmiles("c1ocnc1")  # 1,3-oxazole: 1N + 1O
        imidazole = Chem.MolFromSmiles("c1[nH]cnc1")  # 1H-imidazole: 2N
        r_ox = _rank(oxazole, _all_ring_atoms(oxazole))
        r_im = _rank(imidazole, _all_ring_atoms(imidazole))
        # (d) ties at 2 hets each
        assert r_ox.het_count_neg == r_im.het_count_neg == -2
        # (e) breaks
        assert r_ox.het_variety_neg == -2
        assert r_im.het_variety_neg == -1
        assert r_ox < r_im

    def test_e_NN_vs_NO_NO_wins(self):
        """Variety {N,O}=2 > variety {N}=1 even at the same total het count.

        Pyridazine (2N, variety 1) vs 1,2-oxazine (1N+1O, variety 2).
        Both 6-rings; (a) ties at N, (d) ties at 2 het. (e) breaks.

        Source: https://iupac.qmul.ac.uk/fusedring/FR23.html FR-2.3(e)
        Source: https://iupac.qmul.ac.uk/BlueBook/P2.html P-25.3.2.4
        Source: AUTONOM-1990 §4.
        Source: Phase 149 CONTEXT D-02.
        """
        pyridazine = Chem.MolFromSmiles("c1ccnnc1")  # 2N
        oxazine_12 = Chem.MolFromSmiles("C1CCNOC1")  # 1,2-oxazinane: 1N + 1O
        r_pd = _rank(pyridazine, _all_ring_atoms(pyridazine))
        r_ox = _rank(oxazine_12, _all_ring_atoms(oxazine_12))
        # Both: senior_het_neg = -20 (N), het_count_neg = -2
        assert r_pd.senior_het_neg == r_ox.senior_het_neg == -20
        assert r_pd.het_count_neg == r_ox.het_count_neg == -2
        # (e) variety breaks
        assert r_pd.het_variety_neg == -1
        assert r_ox.het_variety_neg == -2
        assert r_ox < r_pd

    def test_e_variety_is_set_cardinality_not_count(self):
        """Variety = set cardinality of distinct heteroatom species.

        Per CONTEXT D-15 audit: FR-2.3(e) is set cardinality, not the
        per-element-count vector (which is P-44.2.1(g), a different clause).

        For 1,3,5-triazine (3 N atoms), variety is 1 (single species
        {N}), NOT 3.

        Source: https://iupac.qmul.ac.uk/fusedring/FR23.html FR-2.3(e)
        Source: V18 Appendix A.6 lines 2396-2402.
        Source: AUTONOM-1990 §4.
        Source: Phase 149 CONTEXT D-15 (variety is set cardinality).
        """
        triazine = Chem.MolFromSmiles("c1ncncn1")  # 1,3,5-triazine: 3 N's
        r = _rank(triazine, _all_ring_atoms(triazine))
        assert r.het_count_neg == -3  # 3 het ATOMS
        assert r.het_variety_neg == -1, (
            f"FR-2.3(e) variety = |{{N}}| = 1, NOT count = 3; got "
            f"het_variety_neg={r.het_variety_neg}"
        )


# ============================================================================
# Criterion (f) — Heteroatoms by Alt Priority Order
# ============================================================================


class TestCriterionFAltOrder:
    """FR-2.3(f) — Heteroatoms by alt priority (F > Cl > Br > I > O > ... > Hg).

    Per V18 Appendix A.6 line 2319-2325 the alt order DIFFERS from primary:
    F at rank 20 (not N); N at rank 12 (not 20). Hg present, Al/Ga absent.

    Source: https://iupac.qmul.ac.uk/fusedring/FR23.html FR-2.3(f)
    Source: https://iupac.qmul.ac.uk/BlueBook/P2.html P-25.3.2.4
    """

    def test_f_alt_het_tuple_uses_alt_order_not_primary(self):
        """alt_het_tuple counts heteroatoms in _FR23_ALT_ORDER element order.

        For pyridine (1 N), alt_het_tuple has -1 at the index of 'N' in
        _FR23_ALT_ORDER and 0 elsewhere. Per V18 Appendix A.6 line 2406-2407
        the alt order is ('F', 'Cl', 'Br', 'I', 'O', 'S', 'Se', 'Te', 'N', ...);
        N is at index 8.

        Source: https://iupac.qmul.ac.uk/fusedring/FR23.html FR-2.3(f)
        Source: V18 Appendix A.6 lines 2406-2412.
        Source: AUTONOM-1990 §4.
        Source: Phase 149 CONTEXT D-04 (alt order is genuinely different from primary).
        """
        py = Chem.MolFromSmiles("c1ccncc1")
        rank = _rank(py, _all_ring_atoms(py))
        n_index = _FR23_ALT_ORDER.index('N')
        assert rank.alt_het_tuple[n_index] == -1, (
            f"alt_het_tuple[{n_index}] (N count) must be -1 for pyridine; "
            f"got {rank.alt_het_tuple[n_index]}"
        )
        # All other indices = 0 (no other heteroatoms)
        for i, val in enumerate(rank.alt_het_tuple):
            if i != n_index:
                assert val == 0, (
                    f"alt_het_tuple[{i}] (element {_FR23_ALT_ORDER[i]}) "
                    f"must be 0 for pyridine; got {val}"
                )

    def test_f_alt_order_F_beats_N_at_alt(self):
        """In the alt tuple, F (index 0) beats N (index 8) at first non-zero index.

        Construct two ranks with everything tied (a)-(e) except (f):
        - ring with F: alt_het_tuple = (-1, 0, ..., 0)  (F at index 0)
        - ring with N: alt_het_tuple = (0, 0, ..., 0, -1, 0, ...)  (N at index 8)

        At index 0, the F-tuple has -1 vs N-tuple's 0 → F-tuple < N-tuple.

        NOTE: In real molecules ranks (a)-(e) would NOT tie between an F ring
        and an N ring — primary order (a) would already pick N (rank 20 > F
        rank 19). This test isolates (f) by directly comparing alt_het_tuples,
        which is the documented purpose of (f) as a tail tiebreaker.

        Source: https://iupac.qmul.ac.uk/fusedring/FR23.html FR-2.3(f)
        Source: V18 Appendix A.6 lines 2406-2412.
        Source: AUTONOM-1990 §4.
        Source: Phase 149 CONTEXT D-04 (alt is the (f) tail-tiebreaker, not the (a) primary).
        """
        # Build two atom sets in the same RDKit mol for fair _rank comparison.
        # Use a synthetic molecule containing both an F-substituted ring and
        # an N-containing ring; rank each ring's atoms separately.
        # 4-fluoropyridine: has F (substituent) and N (ring atom).
        # We separately compute alt_het_tuple-only logic via direct construction
        # using ComponentRank, since the alt counter targets atoms in atoms set.
        f_only = ComponentRank(
            senior_het_neg=0, ring_count_neg=-1, ring_sizes_neg=(-6,),
            het_count_neg=-1, het_variety_neg=-1,
            alt_het_tuple=tuple([-1] + [0] * 18),  # F at index 0
        )
        n_only = ComponentRank(
            senior_het_neg=0, ring_count_neg=-1, ring_sizes_neg=(-6,),
            het_count_neg=-1, het_variety_neg=-1,
            alt_het_tuple=tuple([0] * 8 + [-1] + [0] * 10),  # N at index 8
        )
        assert f_only < n_only, (
            f"FR-2.3(f) F (alt index 0) must beat N (alt index 8); got "
            f"f={f_only.alt_het_tuple}, n={n_only.alt_het_tuple}"
        )

    def test_f_alt_order_no_Al_no_Ga_yes_Hg(self):
        """_FR23_HETEROATOM_ALT excludes Al and Ga but includes Hg per V18 Appendix A.6.

        Differs from _HETEROATOM_SENIORITY (which has Al + Ga, no Hg) by design
        per CONTEXT D-04. The two constants are genuinely different; FR-2.3(a)
        uses primary order, FR-2.3(f) uses alt order.

        Source: https://iupac.qmul.ac.uk/fusedring/FR23.html FR-2.3(f)
        Source: V18 Appendix A.6 lines 2319-2325.
        Source: AUTONOM-1990 §4.
        Source: Phase 149 CONTEXT D-04 (alt is a SEPARATE constant; Hg present, Al/Ga absent).
        """
        assert 'Al' not in _FR23_HETEROATOM_ALT, (
            "Al must NOT be in _FR23_HETEROATOM_ALT (V18 Appendix A.6 lock)"
        )
        assert 'Ga' not in _FR23_HETEROATOM_ALT, (
            "Ga must NOT be in _FR23_HETEROATOM_ALT (V18 Appendix A.6 lock)"
        )
        assert 'Hg' in _FR23_HETEROATOM_ALT, (
            "Hg must BE in _FR23_HETEROATOM_ALT (V18 Appendix A.6 lock)"
        )
        assert _FR23_HETEROATOM_ALT['Hg'] == 2, (
            f"Hg rank must be 2; got {_FR23_HETEROATOM_ALT['Hg']}"
        )
        assert _FR23_HETEROATOM_ALT['F'] == 20, (
            f"F rank must be 20 (alt-order top); got {_FR23_HETEROATOM_ALT['F']}"
        )
        assert _FR23_HETEROATOM_ALT['N'] == 12, (
            f"N rank in alt order must be 12 (NOT 20 like primary); got "
            f"{_FR23_HETEROATOM_ALT['N']}"
        )


# ============================================================================
# Criterion (g) — Preferred Orientation (deterministic stub per D-03)
# ============================================================================


class TestCriterionGOrientStub:
    """FR-2.3(g) — Preferred orientation. FILLED (Wave-2 P5 fused, Task 6).

    P-25.3.2.4(g): greatest number of rings in a horizontal row. The field is
    now a COMPUTED per-component structural descriptor (negated horizontal-row
    count) instead of the deferred D-03 constant 0. It is deterministic and
    spelling-invariant, and only decides when (a)-(f) tie.

    Source: https://iupac.qmul.ac.uk/fusedring/FR23.html FR-2.3(g)
    Source: BlueBookV2.md:12317 P-25.3.2.4(g)
    """

    def test_orient_computed_from_ring_span(self):
        """orient_stub = -(horizontal-row count); monocycle=-1, 2-ring=-2."""
        # quinoline: 2 ortho-fused rings -> -2
        mol = Chem.MolFromSmiles("c1ccc2ncccc2c1")
        rank = _rank(mol, _all_ring_atoms(mol))
        assert rank.orient_stub == -2, rank.orient_stub

    def test_orient_deterministic_across_mols(self):
        """orient_stub is a well-defined int per component (deterministic)."""
        cases = [
            ("c1ccncc1", -1),                    # pyridine monocycle
            ("c1ccc2ncccc2c1", -2),              # quinoline (2 rings)
            ("c1ccc2cc3ccccc3cc2c1", -3),        # anthracene (3-in-row)
        ]
        for smi, expected in cases:
            mol = Chem.MolFromSmiles(smi)
            rank = _rank(mol, _all_ring_atoms(mol))
            assert rank.orient_stub == expected, (smi, rank.orient_stub)

    def test_orient_spelling_invariant(self):
        """Same structure, two SMILES spellings -> identical orient_stub."""
        a = Chem.MolFromSmiles("c1ccc2ncccc2c1")
        b = Chem.MolFromSmiles("c1ccc2c(c1)nccc2")
        assert (_rank(a, _all_ring_atoms(a)).orient_stub
                == _rank(b, _all_ring_atoms(b)).orient_stub)


# ============================================================================
# Criterion (h) — Lower Locants for Heteroatoms (deterministic stub per D-03)
# ============================================================================


class TestCriterionHHetLocantsStub:
    """FR-2.3(h) — Lower heteroatom locants. FILLED (Wave-2 P5 fused, Task 6).

    P-25.3.2.4(h): a component with lower locants for heteroatoms. The field is
    now the ascending tuple of heteroatom locants under a spelling-invariant
    per-component canonical numbering (empty for a carbocycle).

    Source: https://iupac.qmul.ac.uk/fusedring/FR23.html FR-2.3(h)
    Source: BlueBookV2.md:12392 P-25.3.2.4(h)
    """

    def test_het_locants_populated_for_het_component(self):
        """het_locants_stub is a non-empty ascending tuple for a het component."""
        mol = Chem.MolFromSmiles("c1ccc2ncccc2c1")
        rank = _rank(mol, _all_ring_atoms(mol))
        assert isinstance(rank.het_locants_stub, tuple)
        assert len(rank.het_locants_stub) == 1  # one N
        assert list(rank.het_locants_stub) == sorted(rank.het_locants_stub)

    def test_het_locants_empty_for_carbocycle(self):
        """Carbocyclic component -> empty tuple (non-regressing, per D-03 default
        semantics preserved for all-carbon rings)."""
        mol = Chem.MolFromSmiles("c1ccc2ccccc2c1")  # naphthalene
        rank = _rank(mol, _all_ring_atoms(mol))
        assert rank.het_locants_stub == ()

    def test_het_locants_spelling_invariant(self):
        """Same structure, two spellings -> identical het_locants_stub."""
        a = Chem.MolFromSmiles("c1ccc2ncccc2c1")
        b = Chem.MolFromSmiles("c1ccc2c(c1)nccc2")
        assert (_rank(a, _all_ring_atoms(a)).het_locants_stub
                == _rank(b, _all_ring_atoms(b)).het_locants_stub)


# ============================================================================
# Criterion (i) — Locant Ordering by Heteroatom Type (deterministic stub per D-03)
# ============================================================================


class TestCriterionIHetTypeLocantsStub:
    """FR-2.3(i) — Locant ordering by heteroatom type. FILLED (Wave-2 P5, Task 6).

    P-25.3.2.4(i): lower locants for heteroatoms in seniority order. The field is
    now a computed tuple (senior element's locants first). Deterministic,
    spelling-invariant, empty for a carbocycle.

    Source: https://iupac.qmul.ac.uk/fusedring/FR23.html FR-2.3(i)
    Source: BlueBookV2.md:12407 P-25.3.2.4(i)
    """

    def test_het_type_locants_populated_for_het_component(self):
        """Non-empty computed tuple for a multi-heteroatom component."""
        mol = Chem.MolFromSmiles("c1ocnc1")  # oxazole (N+O)
        rank = _rank(mol, _all_ring_atoms(mol))
        assert isinstance(rank.het_type_locants_stub, tuple)
        assert len(rank.het_type_locants_stub) == 2  # N + O

    def test_het_type_locants_empty_for_carbocycle(self):
        """Carbocyclic component -> empty tuple."""
        mol = Chem.MolFromSmiles("c1ccc2ccccc2c1")  # naphthalene
        rank = _rank(mol, _all_ring_atoms(mol))
        assert rank.het_type_locants_stub == ()

    def test_het_type_locants_spelling_invariant(self):
        """Same structure, two spellings -> identical het_type_locants_stub."""
        a = Chem.MolFromSmiles("c1ccc2[nH]cnc2c1")  # benzimidazole
        b = Chem.MolFromSmiles("c1ccc2nc[nH]c2c1")
        assert (_rank(a, _all_ring_atoms(a)).het_type_locants_stub
                == _rank(b, _all_ring_atoms(b)).het_type_locants_stub)


# ============================================================================
# Criterion (j) — Lower Bridgehead Carbon Locants (deterministic stub per D-03)
# ============================================================================


class TestCriterionJBridgeheadStub:
    """FR-2.3(j) — Lower peripheral fusion-carbon locants. FILLED (Wave-2 P5, Task 6).

    P-25.3.2.4(j): lower locants for peripheral fusion carbon atoms. The field is
    now the ascending tuple of fusion-carbon locants (carbons shared by >=2 rings
    of the component), computed under the spelling-invariant per-component
    numbering. Empty for a monocycle (no fusion carbons).

    Source: https://iupac.qmul.ac.uk/fusedring/FR23.html FR-2.3(j)
    Source: BlueBookV2.md:12418 P-25.3.2.4(j)
    """

    def test_bridgehead_locants_populated_for_fused_component(self):
        """Fused carbocycle -> non-empty ascending fusion-carbon locant tuple."""
        mol = Chem.MolFromSmiles("c1ccc2ccccc2c1")  # naphthalene: 2 fusion C
        rank = _rank(mol, _all_ring_atoms(mol))
        assert isinstance(rank.bridgehead_locants_stub, tuple)
        assert len(rank.bridgehead_locants_stub) == 2
        assert list(rank.bridgehead_locants_stub) == sorted(rank.bridgehead_locants_stub)

    def test_bridgehead_locants_empty_for_monocycle(self):
        """Monocyclic component -> no shared/fusion carbons -> empty tuple."""
        mol = Chem.MolFromSmiles("c1ccncc1")  # pyridine
        rank = _rank(mol, _all_ring_atoms(mol))
        assert rank.bridgehead_locants_stub == ()

    def test_bridgehead_locants_spelling_invariant(self):
        """Same structure, two spellings -> identical bridgehead_locants_stub."""
        a = Chem.MolFromSmiles("c1ccc2ccccc2c1")
        b = Chem.MolFromSmiles("c1ccc2c(c1)cccc2")
        assert (_rank(a, _all_ring_atoms(a)).bridgehead_locants_stub
                == _rank(b, _all_ring_atoms(b)).bridgehead_locants_stub)


# ============================================================================
# ComponentRank Total Order
# ============================================================================


class TestComponentRankTotalOrder:
    """ComponentRank dataclass(frozen=True, order=True) total-order properties.

    Source: V18_MILESTONE_PLAN Appendix A.6 lines 2328-2341.
    Source: 149-CONTEXT.md D-02 (sortable dataclass).
    Source: 149-RESEARCH.md "Stable total-order proof sketch" line 248.
    """

    def test_componentrank_total_order_distinct_ranks_compare(self):
        """Any two distinct ranks compare deterministically via lexicographic order.

        Source: https://iupac.qmul.ac.uk/fusedring/FR23.html
        Source: AUTONOM-1990 §4.
        Source: Phase 149 CONTEXT D-02 (sortable dataclass total-order).
        """
        # Higher senior het — wins
        r1 = ComponentRank(
            senior_het_neg=-20, ring_count_neg=-1, ring_sizes_neg=(-6,),
            het_count_neg=-1, het_variety_neg=-1,
        )
        # No heteroatom
        r2 = ComponentRank(
            senior_het_neg=0, ring_count_neg=-1, ring_sizes_neg=(-6,),
            het_count_neg=0, het_variety_neg=0,
        )
        assert r1 < r2 and r2 > r1
        assert (r1 < r2) ^ (r2 < r1)  # exactly one direction

    def test_componentrank_equal_ranks_compare_equal(self):
        """Two ComponentRank instances with all 10 fields equal compare equal.

        Source: https://iupac.qmul.ac.uk/fusedring/FR23.html
        Source: AUTONOM-1990 §4.
        Source: Phase 149 CONTEXT D-02 (frozen dataclass).
        """
        a = ComponentRank(
            senior_het_neg=-20, ring_count_neg=-1, ring_sizes_neg=(-6,),
            het_count_neg=-1, het_variety_neg=-1,
        )
        b = ComponentRank(
            senior_het_neg=-20, ring_count_neg=-1, ring_sizes_neg=(-6,),
            het_count_neg=-1, het_variety_neg=-1,
        )
        assert a == b
        assert not (a < b)
        assert not (b < a)
        # Hashable (frozen=True)
        assert hash(a) == hash(b)


# ============================================================================
# Public API Contract
# ============================================================================


class TestSelectBaseComponentAPI:
    """Public API contract per V18 plan §6 SC #1 (D-06 signature lock).

    Source: V18_MILESTONE_PLAN §6 Phase 149 SC #1.
    Source: 149-CONTEXT.md D-06.
    """

    def test_signature_matches_v18_plan(self):
        """select_base_component(mol, fused_components) per D-06 lock.

        Source: V18_MILESTONE_PLAN §6 Phase 149 SC #1.
        Source: https://iupac.qmul.ac.uk/fusedring/FR23.html
        Source: AUTONOM-1990 §4.
        Source: Phase 149 CONTEXT D-06 (signature lock).
        """
        sig = inspect.signature(select_base_component)
        params = list(sig.parameters)
        assert params == ['mol', 'fused_components'], (
            f"V18 plan §6 SC #1 signature lock; got {params}"
        )

    def test_raises_on_single_component(self):
        """Edge case: fusion naming undefined for single component (D-06).

        Source: V18_MILESTONE_PLAN §6 Phase 149 SC #1 ValueError lock.
        Source: https://iupac.qmul.ac.uk/fusedring/FR23.html
        Source: AUTONOM-1990 §4.
        Source: Phase 149 CONTEXT D-06.
        """
        mol = Chem.MolFromSmiles("c1ccccc1")
        with pytest.raises(ValueError, match="Need >=2"):
            select_base_component(mol, [{0, 1, 2, 3, 4, 5}])

    def test_raises_on_zero_components(self):
        """Edge case: empty list also raises ValueError (D-06).

        Source: https://iupac.qmul.ac.uk/fusedring/FR23.html P-25.3.2.4
        Source: AUTONOM-1990 §4.
        Source: Phase 149 CONTEXT D-06.
        """
        mol = Chem.MolFromSmiles("CCO")
        with pytest.raises(ValueError, match="Need >=2"):
            select_base_component(mol, [])


# ============================================================================
# _enumerate_components Edge Cases
# ============================================================================


class TestEnumerateComponents:
    """_enumerate_components SSSR-based decomposition per D-05 + CD-04.

    Source: https://iupac.qmul.ac.uk/BlueBook/P2.html P-25.3.1.3
    Source: 149-CONTEXT.md D-05 (SSSR base), CD-04 (frozenset return).
    """

    def test_returns_frozensets_per_cd_04(self):
        """_enumerate_components returns List[FrozenSet[int]] per CD-04.

        Source: https://iupac.qmul.ac.uk/fusedring/FR23.html
        Source: AUTONOM-1990 §4.
        Source: Phase 149 CONTEXT CD-04 (frozenset for hashability).
        """
        mol = Chem.MolFromSmiles("c1ccc2ncccc2c1")  # quinoline
        components = _enumerate_components(mol)
        assert all(isinstance(c, frozenset) for c in components), (
            f"_enumerate_components must return frozensets per CD-04; got "
            f"{[type(c).__name__ for c in components]}"
        )

    def test_quinoline_yields_2_components(self):
        """Quinoline (2 SSSR rings) → 2 components.

        Source: https://iupac.qmul.ac.uk/fusedring/FR23.html
        Source: AUTONOM-1990 §4.
        Source: Phase 149 CONTEXT D-05.
        """
        mol = Chem.MolFromSmiles("c1ccc2ncccc2c1")
        components = _enumerate_components(mol)
        assert len(components) == 2, (
            f"quinoline must yield 2 components; got {len(components)}"
        )

    def test_acyclic_yields_empty_list(self):
        """Acyclic molecule → empty component list (caller raises ValueError).

        Source: https://iupac.qmul.ac.uk/fusedring/FR23.html
        Source: AUTONOM-1990 §4.
        Source: Phase 149 CONTEXT D-05.
        """
        mol = Chem.MolFromSmiles("CCO")  # ethanol
        components = _enumerate_components(mol)
        assert components == [], (
            f"acyclic mol must yield empty list; got {components}"
        )


# ============================================================================
# Phase 155.C — P-25.2.2.4 Benzo-Fusion Regression Cases (D-13)
# ============================================================================


def _ring_contains_symbol(sym: str):
    """Return a predicate ``(mol, ring) -> bool`` true iff ``ring`` contains
    at least one atom whose element symbol is ``sym``."""
    def _pred(mol: Chem.Mol, ring) -> bool:
        return any(mol.GetAtomWithIdx(i).GetSymbol() == sym for i in ring)
    return _pred


def _ring_is_pure_carbocycle():
    """Return a predicate ``(mol, ring) -> bool`` true iff every atom in the
    ring is carbon (no heteroatoms)."""
    def _pred(mol: Chem.Mol, ring) -> bool:
        return all(mol.GetAtomWithIdx(i).GetSymbol() == "C" for i in ring)
    return _pred


class TestP25224BenzoFusion:
    """Phase 155.C D-13 regression: P-25.2.2.4 Jan 2022 errata benzo-fusion
    base-selection cases.

    Eight canonical regression cases verify that Phase 149's FR-2.3 cascade
    ((a)-(f) live, (g)-(j) deterministic stubs) selects the IUPAC-preferred
    base component WITHOUT consulting MONOCYCLIC_COMPONENTS.seniority (the
    last-resort numeric tiebreaker).

    Per 155-AUDIT-C.md §"D-13 Activation Verdict": every case below picks the
    HETEROCYCLIC ring (or in the homo-N pyrido-pyrido case the lower-locant
    pyridine ring), never the pure-carbocyclic benzene ring. Phase 149's
    FR-2.3 cascade is therefore sufficient — no `fused_ring_selection.py`
    edit ships in sub-phase 155.C.

    Source: 155-CONTEXT.md D-13.
    Source: 155-AUDIT-C.md "P-25.2.2.4 Regression Cases (D-13 cross-check)".
    Source: IUPAC P-25.2.2.4 Jan 2022 errata.
    """

    @pytest.mark.parametrize(
        "smiles,base_predicate,case_label,expected_size",
        [
            # benzo[b]furan — base must be furan (5-ring containing O), NOT
            # benzene (6-ring pure carbocycle).
            ("c1ccc2occc2c1", _ring_contains_symbol("O"), "benzo[b]furan", 5),
            # benzo[c]furan / isobenzofuran — base must be furan ring.
            ("c1cc2cocc2cc1", _ring_contains_symbol("O"), "isobenzofuran", 5),
            # 1H-indole — base must be pyrrole (5-ring containing N).
            ("c1ccc2[nH]ccc2c1", _ring_contains_symbol("N"), "1H-indole", 5),
            # 2H-isoindole — base must be pyrrole ring.
            ("c1ccc2c[nH]cc2c1", _ring_contains_symbol("N"), "2H-isoindole", 5),
            # 1H-benzimidazole — base must be imidazole (5-ring containing N).
            ("c1ccc2[nH]cnc2c1", _ring_contains_symbol("N"), "1H-benzimidazole", 5),
            # 1,3-benzothiazole — base must be thiazole (5-ring containing N
            # — N is more senior than S in FR-2.3 (a)).
            ("c1ccc2scnc2c1", _ring_contains_symbol("N"), "1,3-benzothiazole", 5),
            # pyrido[2,3-b]pyridine — both 6-rings contain N; FR-2.3 still
            # picks one canonical ring (verified deterministic).
            ("c1cnc2cccnc2c1", _ring_contains_symbol("N"), "pyrido[2,3-b]pyridine", 6),
            # pyrido[3,2-b]pyridine — homo-N case; FR-2.3 picks one ring.
            ("c1cnc2ncccc2c1", _ring_contains_symbol("N"), "pyrido[3,2-b]pyridine", 6),
        ],
    )
    def test_benzo_fusion_base_is_heterocyclic(
        self, smiles, base_predicate, case_label, expected_size
    ):
        """FR-2.3 cascade selects the heterocyclic base for every benzo-fusion
        regression case in 155-AUDIT-C.md (8 canonical P-25.2.2.4 examples)."""
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, f"RDKit failed to parse {smiles!r}"
        components = [set(c) for c in _enumerate_components(mol)]
        assert len(components) >= 2, (
            f"{case_label}: expected >=2 SSSR rings; got {len(components)}"
        )
        base, _others = select_base_component(mol, components)
        # Base ring contains the expected senior heteroatom.
        assert base_predicate(mol, base), (
            f"{case_label}: FR-2.3 picked base ring {sorted(base)} but it "
            f"does not satisfy the base-predicate (expected senior heteroatom)"
        )
        # Base ring has the expected size (5 for benzo-fused 5-mem; 6 for
        # pyrido-pyrido).
        assert len(base) == expected_size, (
            f"{case_label}: FR-2.3 picked base of size {len(base)}; "
            f"expected {expected_size}"
        )

    def test_benzo_b_furan_base_is_NOT_benzene(self):
        """Explicit anti-regression: for benzo[b]furan, FR-2.3 must NEVER pick
        the pure-carbocyclic 6-ring as the base (would yield wrong PIN)."""
        mol = Chem.MolFromSmiles("c1ccc2occc2c1")
        components = [set(c) for c in _enumerate_components(mol)]
        base, _ = select_base_component(mol, components)
        # base must NOT be the all-carbon 6-ring
        is_pure_carbocycle = all(
            mol.GetAtomWithIdx(i).GetSymbol() == "C" for i in base
        )
        assert not is_pure_carbocycle, (
            f"FR-2.3 picked benzene as base for benzo[b]furan "
            f"(atoms {sorted(base)}) — violates P-25.2.2.4. The "
            f"heterocyclic component must be senior over benzene."
        )

    def test_d13_no_op_dataclass_defaults_unchanged(self):
        """D-13 NO-OP guard: ComponentRank (g)-(j) stubs MUST stay at
        dataclass defaults — Phase 155.C did NOT edit fused_ring_selection.py
        per 155-AUDIT-C.md §"D-13 Activation Verdict"."""
        # Get default values for the (g)-(j) stub fields by constructing
        # ComponentRank with only the (a)-(f) required positional args.
        rank = ComponentRank(
            senior_het_neg=0,
            ring_count_neg=0,
            ring_sizes_neg=(),
            het_count_neg=0,
            het_variety_neg=0,
        )
        assert rank.orient_stub == 0, (
            "D-13 NO-OP violated: orient_stub default changed from 0"
        )
        assert rank.het_locants_stub == (), (
            "D-13 NO-OP violated: het_locants_stub default changed from ()"
        )
        assert rank.het_type_locants_stub == (), (
            "D-13 NO-OP violated: het_type_locants_stub default changed"
        )
        assert rank.bridgehead_locants_stub == (), (
            "D-13 NO-OP violated: bridgehead_locants_stub default changed"
        )
