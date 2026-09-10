"""Unit tests for discover_substituents -- universal substituent discovery.

Tests the new unified substituent discovery function that replaces 6 parallel
systems with a single entry point. Covers ring parents, chain parents,
auto-detection, completeness invariant, compound fragments, geminal
substituents, exocyclic bonds, and large substituents.

a phase, Plan 01 -- TDD RED: all tests expected to fail initially.
"""

import pytest
from rdkit import Chem

from orthonym.assembly.substituent_enumerator import (
    SubstituentInfo,
    discover_substituents,
)


# ============================================================================
# Helpers
# ============================================================================


def _make_ring_parent(smiles):
    """Build (mol, parent_atoms_set, oriented_ring) for a ring molecule.

    Extracts the first ring from RDKit ring info and uses it as both the
    parent atom set and the oriented ring (IUPAC numbering order).

    Returns:
        (mol, parent_atoms_set, oriented_ring_list)
    """
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Invalid SMILES: {smiles}"
    rings = mol.GetRingInfo().AtomRings()
    assert len(rings) >= 1, f"No rings found in {smiles}"
    ring = list(rings[0])
    return mol, set(ring), ring


def _make_chain_parent(smiles, chain_indices):
    """Build (mol, parent_atoms_set, principal_chain) for a chain molecule.

    Args:
        smiles: SMILES string.
        chain_indices: List of atom indices forming the principal chain
            (in order, 0-indexed).

    Returns:
        (mol, parent_atoms_set, principal_chain_list)
    """
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Invalid SMILES: {smiles}"
    return mol, set(chain_indices), chain_indices


def _all_non_parent_heavy(mol, parent_atoms):
    """Get set of all non-parent heavy atom indices in mol."""
    return {
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() > 1 and atom.GetIdx() not in parent_atoms
    }


def _assert_completeness(mol, parent_atoms, results):
    """Assert completeness invariant: every non-parent heavy atom assigned once."""
    expected = _all_non_parent_heavy(mol, parent_atoms)
    actual = set()
    for sub in results:
        # Check no overlap (no double-assignment)
        overlap = actual & set(sub.frag_atoms)
        assert not overlap, f"Double-assigned atoms: {overlap}"
        actual.update(sub.frag_atoms)
    assert expected == actual, (
        f"Atom coverage mismatch: missing={expected - actual}, "
        f"extra={actual - expected}"
    )


# ============================================================================
# Ring Parent Tests
# ============================================================================


@pytest.mark.unit
def test_ring_parent_basic():
    """discover_substituents on methylcyclohexane (CC1CCCCC1) with ring parent
    returns 1 SubstituentInfo with methyl fragment atoms."""
    mol, parent_atoms, oriented_ring = _make_ring_parent("CC1CCCCC1")
    results = discover_substituents(
        mol, parent_atoms, parent_type="ring", oriented_ring=oriented_ring
    )
    assert len(results) == 1
    sub = results[0]
    assert isinstance(sub, SubstituentInfo)
    # Atom 0 is the methyl carbon (not in ring)
    assert 0 in sub.frag_atoms
    assert sub.locant is not None
    assert isinstance(sub.locant, int)


@pytest.mark.unit
def test_auto_detect_ring():
    """Auto-detection identifies methylcyclohexane as a ring parent."""
    mol, parent_atoms, oriented_ring = _make_ring_parent("CC1CCCCC1")
    results = discover_substituents(
        mol, parent_atoms, parent_type="auto", oriented_ring=oriented_ring
    )
    assert len(results) == 1
    assert 0 in results[0].frag_atoms


@pytest.mark.unit
def test_geminal_substituents_ring():
    """gem-dimethylcyclohexane (CC1(C)CCCCC1) produces 2 separate SubstituentInfo
    entries at the same locant."""
    mol, parent_atoms, oriented_ring = _make_ring_parent("CC1(C)CCCCC1")
    results = discover_substituents(
        mol, parent_atoms, parent_type="ring", oriented_ring=oriented_ring
    )
    assert len(results) == 2
    # Both should be at the same locant (the gem position)
    locants = [s.locant for s in results]
    assert locants[0] == locants[1]
    # Each methyl should have exactly 1 atom
    for sub in results:
        assert len(sub.frag_atoms) == 1
    # Atoms 0 and 2 are the two methyls
    all_sub_atoms = set()
    for sub in results:
        all_sub_atoms.update(sub.frag_atoms)
    assert all_sub_atoms == {0, 2}


@pytest.mark.unit
def test_exocyclic_double_bond():
    """=O on cyclohexanone (O=C1CCCCC1) is included in discovery."""
    mol, parent_atoms, oriented_ring = _make_ring_parent("O=C1CCCCC1")
    results = discover_substituents(
        mol, parent_atoms, parent_type="ring", oriented_ring=oriented_ring
    )
    assert len(results) == 1
    # Atom 0 is the oxygen
    assert 0 in results[0].frag_atoms


@pytest.mark.unit
def test_empty_parent_no_subs():
    """Cyclohexane (C1CCCCC1) with no substituents returns empty list."""
    mol, parent_atoms, oriented_ring = _make_ring_parent("C1CCCCC1")
    results = discover_substituents(
        mol, parent_atoms, parent_type="ring", oriented_ring=oriented_ring
    )
    assert results == []


@pytest.mark.unit
def test_completeness_invariant_ring():
    """For tri-substituted cyclohexane, ALL non-ring heavy atoms are assigned
    to exactly one SubstituentInfo."""
    # CCC1CC(C)CC(C)C1 = 1-ethyl-3,5-dimethylcyclohexane
    mol, parent_atoms, oriented_ring = _make_ring_parent("CCC1CC(C)CC(C)C1")
    results = discover_substituents(
        mol, parent_atoms, parent_type="ring", oriented_ring=oriented_ring
    )
    # Should find 3 substituents: 1 ethyl + 2 methyls
    assert len(results) == 3
    _assert_completeness(mol, parent_atoms, results)


@pytest.mark.unit
def test_fg_compound_fragment():
    """For OC1CCC(CCO)CC1, the -CCO substituent is discovered as ONE
    SubstituentInfo (not split into separate O and CC fragments)."""
    mol, parent_atoms, oriented_ring = _make_ring_parent("OC1CCC(CCO)CC1")
    results = discover_substituents(
        mol, parent_atoms, parent_type="ring", oriented_ring=oriented_ring
    )
    # Should find 2 substituents: -OH (atom 0) and -CH2CH2OH (atoms 5,6,7)
    assert len(results) == 2
    # Find the compound fragment (the one with >1 atom)
    compound_subs = [s for s in results if len(s.frag_atoms) > 1]
    assert len(compound_subs) == 1
    compound = compound_subs[0]
    # Should contain the CH2, CH2, and OH as one fragment
    assert len(compound.frag_atoms) == 3  # C, C, O
    _assert_completeness(mol, parent_atoms, results)


@pytest.mark.unit
def test_heteroatom_not_split():
    """For methoxycyclohexane (COC1CCCCC1), the -OCH3 is one compound fragment."""
    mol, parent_atoms, oriented_ring = _make_ring_parent("COC1CCCCC1")
    results = discover_substituents(
        mol, parent_atoms, parent_type="ring", oriented_ring=oriented_ring
    )
    assert len(results) == 1
    sub = results[0]
    # Fragment should contain both O (atom 1) and C (atom 0)
    assert 0 in sub.frag_atoms  # methyl C
    assert 1 in sub.frag_atoms  # oxygen
    assert len(sub.frag_atoms) == 2
    _assert_completeness(mol, parent_atoms, results)


# ============================================================================
# Chain Parent Tests
# ============================================================================


@pytest.mark.unit
def test_chain_parent_basic():
    """discover_substituents on 2-methylpentane with chain parent returns 1
    SubstituentInfo with methyl at locant 2.

    SMILES: CCCC(C)C -> principal chain [5, 3, 2, 1, 0] (pentane), sub at pos 2.
    RDKit canonical CCCC(C)C: atoms 0,1,2,3,4,5
    Chain = [5, 3, 2, 1, 0] (C-5 through C-1 to get pentane in order)

    Actually for simplicity: chain = [0, 1, 2, 3, 5] covers 5 carbons.
    Substituent = atom 4 (the methyl branch).
    """
    # CCCC(C)C: atoms 0-5, chain = longest = [0,1,2,3,5], sub = {4}
    mol, parent_atoms, chain = _make_chain_parent("CCCC(C)C", [0, 1, 2, 3, 5])
    results = discover_substituents(
        mol, parent_atoms, parent_type="chain", principal_chain=chain
    )
    assert len(results) == 1
    sub = results[0]
    assert isinstance(sub, SubstituentInfo)
    assert 4 in sub.frag_atoms  # the methyl branch
    # Locant should be 4 (position of atom 3 in chain = index 3, 1-indexed = 4)
    assert sub.locant == 4


@pytest.mark.unit
def test_auto_detect_chain():
    """Auto-detection identifies a pure chain molecule as chain parent."""
    mol, parent_atoms, chain = _make_chain_parent("CCCC(C)C", [0, 1, 2, 3, 5])
    results = discover_substituents(
        mol, parent_atoms, parent_type="auto", principal_chain=chain
    )
    assert len(results) == 1
    assert 4 in results[0].frag_atoms


@pytest.mark.unit
def test_completeness_invariant_chain():
    """For 2,3-dimethylpentane, ALL non-chain heavy atoms are assigned exactly once.

    CC(C)C(C)CC: atoms 0-6
    Chain = [0, 1, 3, 5, 6] (pentane backbone), subs = {2, 4}
    """
    mol, parent_atoms, chain = _make_chain_parent("CC(C)C(C)CC", [0, 1, 3, 5, 6])
    results = discover_substituents(
        mol, parent_atoms, parent_type="chain", principal_chain=chain
    )
    assert len(results) == 2
    _assert_completeness(mol, parent_atoms, results)


@pytest.mark.unit
def test_multiple_attachment_points():
    """Multiple substituents on different chain positions are all discovered.

    3-ethylpentane: CCC(CC)CC
    atoms: 0(C), 1(C), 2(C), 3(C), 4(C), 5(C), 6(C)
    chain = [0, 1, 2, 5, 6] (pentane), subs = {3,4} as one ethyl at position 3
    """
    mol, parent_atoms, chain = _make_chain_parent("CCC(CC)CC", [0, 1, 2, 5, 6])
    results = discover_substituents(
        mol, parent_atoms, parent_type="chain", principal_chain=chain
    )
    assert len(results) == 1  # One ethyl substituent
    sub = results[0]
    assert sub.frag_atoms == frozenset({3, 4})
    assert sub.locant == 3  # At position 3 of the chain
    _assert_completeness(mol, parent_atoms, results)


@pytest.mark.unit
def test_chain_ring_in_substituent():
    """Phenyl group on pentane chain is discovered as single substituent containing
    the ring atoms.

    3-phenylpentane: CCC(CC)c1ccccc1
    atoms: 0-4 (chain carbons), 5-10 (phenyl ring)
    chain = [0, 1, 2, 3, 4] (pentane), sub = {5,6,7,8,9,10} (phenyl)

    Actually RDKit: CCC(CC)c1ccccc1:
    0(C),1(C),2(C),3(C),4(C),5(c),6(c),7(c),8(c),9(c),10(c)
    chain = [4, 3, 2, 1, 0] or [0, 1, 2, 3, 4], sub = {5..10}
    """
    mol, parent_atoms, chain = _make_chain_parent(
        "CCC(CC)c1ccccc1", [0, 1, 2, 3, 4]
    )
    results = discover_substituents(
        mol, parent_atoms, parent_type="chain", principal_chain=chain
    )
    assert len(results) == 1
    sub = results[0]
    # All 6 ring carbons should be in the substituent
    assert len(sub.frag_atoms) == 6
    assert sub.frag_atoms == frozenset({5, 6, 7, 8, 9, 10})
    _assert_completeness(mol, parent_atoms, results)


@pytest.mark.unit
def test_bfs_walks_through_heteroatoms():
    """For a chain with -NHCH2COOH substituent, BFS collects the entire
    heteroatom-containing fragment as one unit.

    CCCC(NCC(O)=O)CC:
    atoms: 0(C),1(C),2(C),3(C),4(N),5(C),6(C),7(O),8(O),9(C),10(C)
    chain = [0,1,2,3,9,10] (hexane backbone), subs = {4,5,6,7,8}
    """
    mol, parent_atoms, chain = _make_chain_parent(
        "CCCC(NCC(O)=O)CC", [0, 1, 2, 3, 9, 10]
    )
    results = discover_substituents(
        mol, parent_atoms, parent_type="chain", principal_chain=chain
    )
    assert len(results) == 1
    sub = results[0]
    # Should be all 5 atoms: N, C, C, O, O
    assert sub.frag_atoms == frozenset({4, 5, 6, 7, 8})
    _assert_completeness(mol, parent_atoms, results)


@pytest.mark.unit
def test_no_size_guard():
    """A substituent with >25 atoms is discovered successfully (no size limit).

    Chain parent with a 30-carbon substituent:
    CCCCC(CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC)CCCCC
    """
    smiles = "CCCCC(CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC)CCCCC"
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    # Build chain manually: the longest chain is 10 atoms (5+5)
    # atoms 0-3 = first 4C, atom 4 = branch, atoms 35-39 = last 5C
    # Actually just identify the atoms properly
    num_atoms = mol.GetNumAtoms()

    # Find the branching carbon (degree 3+)
    branch_atom = None
    for atom in mol.GetAtoms():
        if atom.GetDegree() >= 3 and atom.GetSymbol() == 'C':
            branch_atom = atom.GetIdx()
            break
    assert branch_atom is not None

    # BFS from atom 0 to find the chain through the branch point
    # For the test, just use a chain of 10 atoms that includes the branch point
    # and verify the substituent has >25 atoms
    from collections import deque

    # Find the two terminal chain pieces (shorter paths through the branch atom)
    branch = mol.GetAtomWithIdx(branch_atom)
    neighbors = [n.GetIdx() for n in branch.GetNeighbors()]

    # Build small paths from each neighbor of branch_atom (not through branch)
    def _bfs_path(mol, start, exclude):
        visited = []
        queue = deque([start])
        seen = {exclude}
        while queue:
            idx = queue.popleft()
            if idx in seen:
                continue
            seen.add(idx)
            visited.append(idx)
            for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
                if nbr.GetIdx() not in seen:
                    queue.append(nbr.GetIdx())
        return visited

    # Get all branches from the branch atom
    branches = []
    for nbr_idx in neighbors:
        path = _bfs_path(mol, nbr_idx, branch_atom)
        branches.append(path)

    # Sort by length: the two shortest are the chain ends, longest is substituent
    branches.sort(key=len)
    # Chain = branch_atom + two shortest branches
    chain = branches[0] + [branch_atom] + branches[1]
    parent_set = set(chain)
    sub_atoms = branches[-1]  # longest branch = substituent

    results = discover_substituents(
        mol, parent_set, parent_type="chain", principal_chain=chain
    )
    assert len(results) >= 1
    # The big substituent should have >25 atoms
    big_subs = [s for s in results if len(s.frag_atoms) > 25]
    assert len(big_subs) == 1, f"Expected 1 big substituent, got {len(big_subs)}"
    _assert_completeness(mol, parent_set, results)
