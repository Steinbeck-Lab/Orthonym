"""
Unit tests for term-by-term heteroatom variety comparison in ring_system_score.

IUPAC P-44.2.1(g): heteroatom variety is compared term-by-term by seniority,
NOT as a weighted sum. The comparison examines counts of each element in
seniority order (N, F, Cl, Br, I, O, S, Se, Te, P).

Reference: IUPAC 2013 Blue Book, P-44.2.1(g)
"""

import pytest
from rdkit import Chem

from orthonym.rules.ring_selection import ring_system_score


def _make_ring_with_heteroatoms(size: int, heteroatom_positions: dict) -> tuple:
    """Build a ring molecule with specified heteroatoms and return (mol, ring_atoms).

    Args:
        size: Ring size (number of atoms)
        heteroatom_positions: Dict mapping 0-based position to element symbol

    Returns:
        (mol, ring_atom_set) tuple
    """
    from rdkit.Chem import RWMol, BondType
    rw = RWMol()
    atoms = ["C"] * size
    for pos, symbol in heteroatom_positions.items():
        atoms[pos] = symbol

    for symbol in atoms:
        atom = Chem.Atom(Chem.GetPeriodicTable().GetAtomicNumber(symbol))
        rw.AddAtom(atom)

    for i in range(size - 1):
        rw.AddBond(i, i + 1, BondType.SINGLE)
    rw.AddBond(size - 1, 0, BondType.SINGLE)

    try:
        Chem.SanitizeMol(rw)
    except Exception:
        pass

    mol = rw.GetMol()
    ring_atoms = set(range(size))
    return mol, ring_atoms


class TestHeteroatomVarietyTermByTerm:
    """Tests for P-44.2.1(g) term-by-term heteroatom variety comparison."""

    def test_two_O_one_Te_beats_one_O_two_S(self):
        """CRITICAL: Ring with 2O+1Te beats 1O+2S at criterion (g).

        This is the case where weighted sum gives the WRONG answer:
        - Ring A: 1O + 2S -> sum = 1*5 + 2*4 = 13 (weighted sum says A wins)
        - Ring B: 2O + 1Te -> sum = 2*5 + 1*2 = 12

        But term-by-term (correct):
        - N(0=0), F(0=0), Cl(0=0), Br(0=0), I(0=0)
        - O: 1 vs 2 -> Ring B wins (more O, the most senior element present)

        Criteria (a)-(f) all tie: both have heteroatoms (True), no N,
        same senior element (O, rank 5), 1 ring, 6 atoms, 3 heteroatoms.
        """
        # Ring A: 1 O + 2 S in 6-membered ring
        mol_A, atoms_A = _make_ring_with_heteroatoms(
            6, {0: "O", 2: "S", 4: "S"}
        )
        # Ring B: 2 O + 1 Te in 6-membered ring
        mol_B, atoms_B = _make_ring_with_heteroatoms(
            6, {0: "O", 2: "O", 4: "Te"}
        )

        score_A = ring_system_score(mol_A, atoms_A)
        score_B = ring_system_score(mol_B, atoms_B)

        # Ring B (2O+1Te) should be more senior (lower score) than A (1O+2S)
        assert score_B < score_A, (
            f"Ring with 2O+1Te should beat 1O+2S at criterion (g): "
            f"B={score_B}, A={score_A}"
        )

    def test_one_O_two_Te_beats_three_Se(self):
        """Ring with 1O+2Te beats 3Se (resolved at criterion (c), not (g)).

        Ring with O has more senior heteroatom (O, rank 5) vs Se (rank 3).
        """
        mol_O_Te, atoms_O_Te = _make_ring_with_heteroatoms(
            6, {0: "O", 2: "Te", 4: "Te"}
        )
        mol_Se, atoms_Se = _make_ring_with_heteroatoms(
            6, {0: "Se", 2: "Se", 4: "Se"}
        )

        score_O_Te = ring_system_score(mol_O_Te, atoms_O_Te)
        score_Se = ring_system_score(mol_Se, atoms_Se)

        assert score_O_Te < score_Se, (
            f"Ring with 1O+2Te should beat 3Se: O_Te={score_O_Te}, Se={score_Se}"
        )

    def test_one_O_one_Se_beats_two_S(self):
        """Ring with 1O+1Se beats 2S (resolved at criterion (c), not (g)).

        Senior heteroatom O (rank 5) beats S (rank 4).
        """
        mol_O_Se, atoms_O_Se = _make_ring_with_heteroatoms(
            6, {0: "O", 3: "Se"}
        )
        mol_S, atoms_S = _make_ring_with_heteroatoms(
            6, {0: "S", 3: "S"}
        )

        score_O_Se = ring_system_score(mol_O_Se, atoms_O_Se)
        score_S = ring_system_score(mol_S, atoms_S)

        assert score_O_Se < score_S, (
            f"Ring with 1O+1Se should beat 2S: O_Se={score_O_Se}, S={score_S}"
        )

    def test_nitrogen_beats_oxygen(self):
        """Ring with 1N beats 1O (N is most senior heteroatom)."""
        mol_N, atoms_N = _make_ring_with_heteroatoms(6, {0: "N"})
        mol_O, atoms_O = _make_ring_with_heteroatoms(6, {0: "O"})

        score_N = ring_system_score(mol_N, atoms_N)
        score_O = ring_system_score(mol_O, atoms_O)

        assert score_N < score_O, (
            f"Ring with 1N should beat 1O: N={score_N}, O={score_O}"
        )

    def test_two_N_beats_one_N(self):
        """Ring with 2N beats 1N (more of most-senior element)."""
        mol_2N, atoms_2N = _make_ring_with_heteroatoms(6, {0: "N", 3: "N"})
        mol_1N, atoms_1N = _make_ring_with_heteroatoms(6, {0: "N"})

        score_2N = ring_system_score(mol_2N, atoms_2N)
        score_1N = ring_system_score(mol_1N, atoms_1N)

        assert score_2N < score_1N, (
            f"Ring with 2N should beat 1N: 2N={score_2N}, 1N={score_1N}"
        )

    def test_pure_carbon_ring_zero_variety(self):
        """Pure carbon ring should have zero for all heteroatom variety fields."""
        mol = Chem.MolFromSmiles("C1CCCCC1")  # cyclohexane
        ring_atoms = set(range(6))
        score = ring_system_score(mol, ring_atoms)
        assert score[0] == 0  # -int(has_heteroatom) = 0
        assert score[1] == 0  # -int(has_nitrogen) = 0

    def test_existing_behavior_N_ring_beats_no_heteroatom(self):
        """Preservation: pyridine ring beats benzene ring (existing behavior)."""
        mol_pyridine = Chem.MolFromSmiles("c1ccncc1")
        mol_benzene = Chem.MolFromSmiles("c1ccccc1")

        from orthonym.perception.rings import get_ring_systems
        rs_pyr = get_ring_systems(mol_pyridine)
        rs_benz = get_ring_systems(mol_benzene)

        score_pyr = ring_system_score(mol_pyridine, rs_pyr[0])
        score_benz = ring_system_score(mol_benzene, rs_benz[0])

        assert score_pyr < score_benz

    def test_score_tuple_length(self):
        """Score tuple should have the correct new length (17 elements).

        6 original fields + 10 variety fields + 1 type_rank = 17.
        """
        mol = Chem.MolFromSmiles("c1ccncc1")
        from orthonym.perception.rings import get_ring_systems
        rs = get_ring_systems(mol)
        score = ring_system_score(mol, rs[0])
        assert len(score) == 17, f"Expected 17-element tuple, got {len(score)}"

    def test_empty_system_sentinel_length(self):
        """Empty system sentinel should match score tuple length (17)."""
        mol = Chem.MolFromSmiles("C")
        score = ring_system_score(mol, set())
        assert len(score) == 17, f"Expected 17-element sentinel, got {len(score)}"
