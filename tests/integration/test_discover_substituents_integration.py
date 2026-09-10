"""Integration tests for discover_substituents -- real-world molecule validation.

Validates the universal substituent discovery function (a phase) against
real-world molecules from the benchmark. Tests three critical properties:

1. **Real-molecule coverage**: discover_substituents correctly handles
   substituted benzene, fused rings, heterocycles, chain+FG, chain+ring-sub,
   and >25-atom substituents.

2. **A/B canary parity**: naming output is UNCHANGED after a phase (additive
   only). 20+ canary compounds produce identical names.

3. ** elimination**: molecules that previously triggered silent drops
   at (ring+heteroatom branches) are handled correctly by
   discover_substituents.

a phase, Plan 02 -- integration gate before a phase builds naming on top.
"""

import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.assembly.substituent_enumerator import (
    SubstituentInfo,
    discover_substituents,
)


# ============================================================================
# Helpers
# ============================================================================


def _ring_parent_info(smiles):
    """Build (mol, parent_atoms, oriented_ring) using ALL ring atoms as parent.

    For fused ring systems, unions all ring atoms into a single parent set.
    Returns (mol, parent_set, oriented_ring_list).
    """
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Invalid SMILES: {smiles}"
    ring_info = mol.GetRingInfo()
    all_ring_atoms = set()
    for ring in ring_info.AtomRings():
        all_ring_atoms.update(ring)
    assert all_ring_atoms, f"No rings in {smiles}"
    oriented = sorted(all_ring_atoms)
    return mol, all_ring_atoms, oriented


def _chain_parent_info(smiles, chain_indices):
    """Build (mol, parent_set, principal_chain) for a chain parent.

    Args:
        smiles: SMILES string.
        chain_indices: List of atom indices forming the principal chain.

    Returns (mol, parent_set, principal_chain_list).
    """
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Invalid SMILES: {smiles}"
    return mol, set(chain_indices), chain_indices


def _assert_completeness(mol, parent_atoms, results):
    """Assert every non-parent heavy atom is assigned to exactly one SubstituentInfo."""
    expected = {
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetAtomicNum() > 1 and a.GetIdx() not in parent_atoms
    }
    actual = set()
    for sub in results:
        overlap = actual & set(sub.frag_atoms)
        assert not overlap, f"Double-assigned atoms: {overlap}"
        actual.update(sub.frag_atoms)
    assert expected == actual, (
        f"Atom coverage mismatch: missing={expected - actual}, "
        f"extra={actual - expected}"
    )


def _count_non_parent_heavy(mol, parent_atoms):
    """Count non-parent heavy atoms in mol."""
    return sum(
        1 for a in mol.GetAtoms()
        if a.GetAtomicNum() > 1 and a.GetIdx() not in parent_atoms
    )


# ============================================================================
# Class 1: TestDiscoveryOnRealMolecules
# ============================================================================


@pytest.mark.integration
class TestDiscoveryOnRealMolecules:
    """Test discover_substituents on real-world molecules with edge cases."""

    def test_substituted_benzene(self):
        """Toluene (Cc1ccccc1): ring parent with single methyl substituent."""
        mol, parent, oriented = _ring_parent_info("Cc1ccccc1")
        results = discover_substituents(
            mol, parent, parent_type="ring", oriented_ring=oriented
        )
        assert len(results) == 1
        sub = results[0]
        assert isinstance(sub, SubstituentInfo)
        assert len(sub.frag_atoms) == 1  # methyl = 1 carbon
        assert sub.locant is not None
        _assert_completeness(mol, parent, results)

    def test_disubstituted_benzene(self):
        """4-Methylphenol (Cc1ccc(O)cc1): ring parent with methyl + OH."""
        mol, parent, oriented = _ring_parent_info("Cc1ccc(O)cc1")
        results = discover_substituents(
            mol, parent, parent_type="ring", oriented_ring=oriented
        )
        assert len(results) == 2
        # One substituent is CH3 (1 atom), the other is OH (1 atom)
        sizes = sorted(len(s.frag_atoms) for s in results)
        assert sizes == [1, 1]
        _assert_completeness(mol, parent, results)

    def test_naphthalene_substituent(self):
        """1-Methylnaphthalene (Cc1cccc2ccccc12): fused ring parent, 10 ring atoms."""
        mol, parent, oriented = _ring_parent_info("Cc1cccc2ccccc12")
        assert len(parent) == 10, "Naphthalene should have 10 ring atoms"
        results = discover_substituents(
            mol, parent, parent_type="ring", oriented_ring=oriented
        )
        assert len(results) == 1
        assert len(results[0].frag_atoms) == 1  # methyl
        _assert_completeness(mol, parent, results)

    def test_heteroatom_ring_substituent(self):
        """4-Methylpyridine (Cc1ccncc1): heterocyclic ring parent with methyl."""
        mol, parent, oriented = _ring_parent_info("Cc1ccncc1")
        results = discover_substituents(
            mol, parent, parent_type="ring", oriented_ring=oriented
        )
        assert len(results) == 1
        assert len(results[0].frag_atoms) == 1
        _assert_completeness(mol, parent, results)

    def test_chain_with_multiple_subs(self):
        """Lactic acid (CC(O)C(=O)O): chain parent with OH + COOH atoms.

        Chain = [0, 1, 3] (propanoic acid C-C-C backbone).
        Substituents: OH at C-2, =O at C-3, -OH at C-3.
        """
        mol, parent, chain = _chain_parent_info("CC(O)C(=O)O", [0, 1, 3])
        results = discover_substituents(
            mol, parent, parent_type="chain", principal_chain=chain
        )
        # Non-parent atoms: 2(OH), 4(=O), 5(-OH)
        assert len(results) >= 2, f"Expected >= 2 substituents, got {len(results)}"
        # All non-parent atoms should be covered
        _assert_completeness(mol, parent, results)

    def test_chain_with_phenyl_sub(self):
        """Butylbenzene (CCCCc1ccccc1): chain parent with phenyl ring substituent."""
        mol, parent, chain = _chain_parent_info("CCCCc1ccccc1", [0, 1, 2, 3])
        results = discover_substituents(
            mol, parent, parent_type="chain", principal_chain=chain
        )
        assert len(results) == 1
        sub = results[0]
        # Phenyl ring has 6 carbons
        assert len(sub.frag_atoms) == 6
        _assert_completeness(mol, parent, results)

    def test_long_chain_branch(self):
        """Ring with >25-atom substituent: no size guard prevents discovery.

        27-carbon chain on benzene (CCCCCCCCCCCCCCCCCCCCCCCCCCCc1ccccc1).
        """
        smiles = "CCCCCCCCCCCCCCCCCCCCCCCCCCCc1ccccc1"
        mol, parent, oriented = _ring_parent_info(smiles)
        results = discover_substituents(
            mol, parent, parent_type="ring", oriented_ring=oriented
        )
        assert len(results) == 1
        assert len(results[0].frag_atoms) == 27, (
            f"Expected 27-atom substituent, got {len(results[0].frag_atoms)}"
        )
        _assert_completeness(mol, parent, results)

    def test_fused_het_parent_subs(self):
        """5-Methylindole (Cc1ccc2[nH]ccc2c1): fused heterocycle parent with methyl."""
        mol, parent, oriented = _ring_parent_info("Cc1ccc2[nH]ccc2c1")
        # Indole has 9 ring atoms (5+6 fused, sharing 2)
        assert len(parent) == 9, f"Indole parent should have 9 atoms, got {len(parent)}"
        results = discover_substituents(
            mol, parent, parent_type="ring", oriented_ring=oriented
        )
        assert len(results) == 1
        assert len(results[0].frag_atoms) == 1  # methyl
        _assert_completeness(mol, parent, results)


# ============================================================================
# Class 2: TestCanaryABComparison
# ============================================================================

# 20+ representative canary compounds covering diverse compound classes.
# Each (SMILES, expected_name) pair is from the golden canary suite.
AB_CANARY_COMPOUNDS = [
    # Simple alkane
    ("CCCCCCCCCCC(C)C(=O)O", "2-methyldodecanoic acid"),
    ("CCCCCCCCC(C)CC(C)C", "2,4-dimethyldodecane"),
    # Alcohol
    ("CC(O)C(=O)O", "2-hydroxypropanoic acid"),
    ("OCC(O)CO", "glycerol"),
    # Carboxylic acid (chain)
    ("CC(=O)O", "acetic acid"),
    ("O=C(O)CCC(=O)O", "butanedioic acid"),
    # Ketone
    ("C=CC(=O)CCCC", "hept-1-en-3-one"),
    # Aldehyde
    ("CCCCC=O", "pentanal"),
    # Ring compound - benzene derivative
    ("Cc1ccc(O)cc1C", "3,4-dimethylphenol"),  #: phenol suffix routing
    ("COc1ccc(OC)c(OC)c1", "1,2,4-trimethoxybenzene"),
    # Heterocycle
    ("CCCc1nc(C)c(C)nc1C", "2,3,6-trimethyl-5-propylpyrazine"),
    ("Oc1ccnc2ccccc12", "4-hydroxyquinoline"),
    # Ester
    ("CCCCCCCCCCCCCCCCCCCCCC(=O)OCC", "ethyl docosanoate"),
    # Stereo compound
    ("C=C[C@@H](O)CCCCC", "(3S)-oct-1-en-3-ol"),
    # Amide
    ("CC(N)=O", "acetamide"),
    # Long unsaturated chain
    ("CCCCCCCC/C=C/CCCCCCCC=O", "(9E)-octadec-9-enal"),
    # Dicarboxylic acid
    ("OC(=O)CC(O)=O", "propanedioic acid"),
    # Fused aromatic
    ("c1ccc2c(c1)ccc1ccccc12", "phenanthrene"),
    # Heterocycle (retained name)
    ("c1ccc2[nH]ccc2c1", "1H-indole"),
    # Acid + ring
    ("OC(=O)c1ccc(O)cc1", "4-hydroxybenzoic acid"),
    # Hydroxy acid
    ("O=C(O)CCO", "3-hydroxypropanoic acid"),
]

_AB_IDS = [
    name[:40].replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")
    for _, name in AB_CANARY_COMPOUNDS
]


@pytest.mark.integration
class TestCanaryABComparison:
    """A/B comparison: verify naming is unchanged after a phase."""

    @pytest.mark.parametrize("smiles,expected_name", AB_CANARY_COMPOUNDS, ids=_AB_IDS)
    def test_naming_unchanged(self, smiles, expected_name):
        """Name must match golden canary expectation exactly."""
        result = name_compound(smiles)
        assert result == expected_name, (
            f"A/B REGRESSION: {smiles}\n"
            f"  Expected: {expected_name}\n"
            f"  Got:      {result}"
        )

    @pytest.mark.parametrize("smiles,expected_name", AB_CANARY_COMPOUNDS, ids=_AB_IDS)
    def test_discovery_completeness_on_canary(self, smiles, expected_name):
        """discover_substituents completeness invariant holds for canary molecules.

        For ring-parent molecules, uses ring atoms as parent.
        For chain-only molecules, uses longest chain as parent.
        Verifies every non-parent heavy atom is assigned exactly once.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            pytest.skip(f"Invalid SMILES: {smiles}")

        ring_info = mol.GetRingInfo()
        all_ring_atoms = set()
        for ring in ring_info.AtomRings():
            all_ring_atoms.update(ring)

        if all_ring_atoms:
            # Ring parent
            oriented = sorted(all_ring_atoms)
            results = discover_substituents(
                mol, all_ring_atoms, parent_type="ring", oriented_ring=oriented
            )
            parent = all_ring_atoms
        else:
            # Chain parent - find longest path as principal chain
            # Use simple heuristic: atom 0 to furthest atom BFS
            from collections import deque

            def _bfs_furthest(mol, start):
                visited = {}
                queue = deque([(start, 0)])
                furthest = start
                max_dist = 0
                while queue:
                    idx, dist = queue.popleft()
                    if idx in visited:
                        continue
                    visited[idx] = dist
                    if dist > max_dist:
                        max_dist = dist
                        furthest = idx
                    for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
                        if nbr.GetAtomicNum() > 1 and nbr.GetIdx() not in visited:
                            queue.append((nbr.GetIdx(), dist + 1))
                return furthest, visited

            # Two-pass BFS to find longest chain endpoints
            end1, _ = _bfs_furthest(mol, 0)
            end2, dists = _bfs_furthest(mol, end1)

            # Reconstruct path from end1 to end2
            path = []
            current = end2
            while current != end1:
                path.append(current)
                best_nbr = None
                best_dist = dists.get(current, 0)
                for nbr in mol.GetAtomWithIdx(current).GetNeighbors():
                    nbr_idx = nbr.GetIdx()
                    if nbr_idx in dists and dists[nbr_idx] < best_dist:
                        best_dist = dists[nbr_idx]
                        best_nbr = nbr_idx
                if best_nbr is None:
                    break
                current = best_nbr
            path.append(end1)
            path.reverse()

            parent = set(path)
            results = discover_substituents(
                mol, parent, parent_type="chain", principal_chain=path
            )

        # If there are non-parent atoms, verify completeness
        non_parent_count = _count_non_parent_heavy(mol, parent)
        if non_parent_count > 0:
            _assert_completeness(mol, parent, results)


# ============================================================================
# Class 3: TestDropPointElimination
# ============================================================================


@pytest.mark.integration
class TestDropPointElimination:
    """Test that discover_substituents has no silent drops."""

    def test_no_drop04_ring_heteroatom_branch_phenoxyacetic(self):
        """Phenoxyacetic acid backbone: -OCC(=O)O on benzene is one fragment.

        Previously triggered (ring+heteroatom branch silent drop).
        discover_substituents must return the full -OCC(=O)O as one
        SubstituentInfo.
        """
        mol, parent, oriented = _ring_parent_info("c1ccc(OCC(=O)O)cc1")
        results = discover_substituents(
            mol, parent, parent_type="ring", oriented_ring=oriented
        )
        assert len(results) == 1, f"Expected 1 substituent, got {len(results)}"
        sub = results[0]
        assert len(sub.frag_atoms) > 0
        # Fragment should contain O, C, C, O, O (5 atoms)
        assert len(sub.frag_atoms) == 5, (
            f"Expected 5-atom fragment (OCC(=O)O), got {len(sub.frag_atoms)}"
        )
        _assert_completeness(mol, parent, results)

    def test_no_drop04_ring_heteroatom_branch_amine(self):
        """Aminoalkyl on benzene: -NCCCO on benzene is one fragment.

        c1ccc(NCCCO)cc1 = 4-(3-hydroxypropyl)amino-type molecule.
        Previously triggered (ring+heteroatom branch).
        """
        mol, parent, oriented = _ring_parent_info("c1ccc(NCCCO)cc1")
        results = discover_substituents(
            mol, parent, parent_type="ring", oriented_ring=oriented
        )
        assert len(results) == 1
        sub = results[0]
        # Fragment: N, C, C, C, O = 5 atoms
        assert len(sub.frag_atoms) == 5, (
            f"Expected 5-atom fragment (NCCCO), got {len(sub.frag_atoms)}"
        )
        _assert_completeness(mol, parent, results)

    def test_no_size_guard_real_molecule(self):
        """A real molecule with a >25 heavy atom substituent is discovered correctly.

        27-carbon alkyl chain on benzene ring.
        """
        smiles = "CCCCCCCCCCCCCCCCCCCCCCCCCCCc1ccccc1"
        mol, parent, oriented = _ring_parent_info(smiles)
        results = discover_substituents(
            mol, parent, parent_type="ring", oriented_ring=oriented
        )
        assert len(results) >= 1
        big_subs = [s for s in results if len(s.frag_atoms) > 25]
        assert len(big_subs) == 1, (
            f"Expected 1 substituent with >25 atoms, got {len(big_subs)}"
        )
        _assert_completeness(mol, parent, results)

    def test_compound_fragment_not_split(self):
        """-SCH2COOH on a ring is one SubstituentInfo, not split.

        c1ccc(SCC(=O)O)cc1 = phenylthioacetic acid-type.
        The sulfur-linked fragment must stay as one unit.
        """
        mol, parent, oriented = _ring_parent_info("c1ccc(SCC(=O)O)cc1")
        results = discover_substituents(
            mol, parent, parent_type="ring", oriented_ring=oriented
        )
        assert len(results) == 1, (
            f"Expected 1 compound fragment, got {len(results)} separate fragments"
        )
        sub = results[0]
        # Fragment: S, C, C, O, O = 5 atoms
        assert len(sub.frag_atoms) == 5, (
            f"Expected 5-atom compound fragment (SCC(=O)O), got {len(sub.frag_atoms)}"
        )
        # Verify all atoms have non-empty frag_atoms
        for s in results:
            assert len(s.frag_atoms) > 0, "SubstituentInfo has empty frag_atoms"
        _assert_completeness(mol, parent, results)

    def test_all_returned_subs_have_frag_atoms(self):
        """Every SubstituentInfo from various molecules has non-empty frag_atoms."""
        test_smiles = [
            "c1ccc(OCC(=O)O)cc1",
            "c1ccc(NCCCO)cc1",
            "c1ccc(SCC(=O)O)cc1",
            "Cc1ccc(O)cc1",
            "Cc1cccc2ccccc12",
        ]
        for smiles in test_smiles:
            mol, parent, oriented = _ring_parent_info(smiles)
            results = discover_substituents(
                mol, parent, parent_type="ring", oriented_ring=oriented
            )
            for sub in results:
                assert len(sub.frag_atoms) > 0, (
                    f"Empty frag_atoms on {smiles} at locant {sub.locant}"
                )
