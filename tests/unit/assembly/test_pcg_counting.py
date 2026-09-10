"""Unit tests for a phase _count_pcgs_in_parent (resolution).

Tests the PCG counting algorithm that resolves per RESEARCH
A PCG counts when its attachment atom is inside parent_atom_indices OR
bonded to an atom inside parent_atom_indices. Substituent PCGs (attachment
neither inside parent nor bonded to parent atom) do NOT count.

Source for algorithm: internal notes / 146-internal notes.
Source for IUPAC rule: https://iupac.qmul.ac.uk/BlueBook/P4.html
"""
from types import SimpleNamespace

import pytest
from rdkit import Chem

from orthonym.assembly.candidate_pool import _count_pcgs_in_parent
from orthonym.assembly.coverage_scoring import CandidateName


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_features(smiles, principal_group, principal_group_atoms):
    """Build a synthetic MolecularFeatures-like object via SimpleNamespace."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return SimpleNamespace(
        mol=mol,
        principal_group=principal_group,
        principal_group_atoms=principal_group_atoms,
    )


def _make_cand(parent_atom_indices):
    """Build a CandidateName with a parent_atom_indices field."""
    cand = CandidateName(
        name="x", handler="chain", confidence=0.5, factors={},
    )
    cand.parent_atom_indices = parent_atom_indices
    return cand


# ---------------------------------------------------------------------------
# Test scenarios per resolution
# ---------------------------------------------------------------------------

class TestCountPCGsInParent:
    """ algorithm: count PCGs whose attachment atom is INSIDE parent
    OR whose attachment atom has a NEIGHBOR inside parent.

    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html
    """

    def test_implicit_pcg_attached_to_chain_parent(self):
        """ case 1: butanoic acid CCCC(=O)O.
        principal_group_atoms=[(3, 4, 5)] — attach=atom 3 (carbonyl C).
        parent_atom_indices={0,1,2,3} (4-carbon chain).
        Attach atom 3 IS in parent_set -> direct attachment -> count=1.
        """
        features = _make_features(
            "CCCC(=O)O",
            principal_group="carboxylic_acid",
            principal_group_atoms=[(3, 4, 5)],
        )
        cand = _make_cand({0, 1, 2, 3})
        assert _count_pcgs_in_parent(cand, features) == 1, (
            "P-44.1.1: butanoic acid carbonyl C in chain -> count=1 (direct)"
        )

    def test_explicit_pcg_neighbor_attachment(self):
        """ case 2: nitrile substituent CC(C#N)CC.
        Atom indices: 0=C, 1=C, 2=C(of CN), 3=N, 4=C, 5=C.
        principal_group_atoms=[(2, 3)] — attach=atom 2 (the nitrile C).
        parent_atom_indices={0, 1, 4, 5} (excludes the nitrile C+N).
        Attach atom 2 NOT in parent_set, BUT atom 2's neighbors include
        atom 1 which IS in parent_set -> indirect attachment -> count=1.
        """
        features = _make_features(
            "CC(C#N)CC",
            principal_group="nitrile",
            principal_group_atoms=[(2, 3)],
        )
        cand = _make_cand({0, 1, 4, 5})
        assert _count_pcgs_in_parent(cand, features) == 1, (
            "P-44.1.1: nitrile C bonded to parent atom -> count=1 (indirect)"
        )

    def test_pcg_on_substituent_not_attached_to_parent(self):
        """ case 3: methyl propanoate CC(C)CC(=O)OC.
        Atom indices: 0=C, 1=C, 2=C, 3=C, 4=C(carbonyl), 5=O(=O), 6=O, 7=C.
        principal_group_atoms=[(4, 5, 6)] — attach=atom 4.
        parent_atom_indices={6, 7} (HYPOTHETICAL methoxy-side parent).
        Attach atom 4 NOT in parent_set; atom 4's neighbors are {3, 5, 6}.
        Atom 6 IS in parent_set -> indirect -> count=1.
        Confirms the algorithm WOULD count via indirect path even on
        a hypothetical methoxy-side parent.
        """
        features = _make_features(
            "CC(C)CC(=O)OC",
            principal_group="ester",
            principal_group_atoms=[(4, 5, 6)],
        )
        cand = _make_cand({6, 7})
        assert _count_pcgs_in_parent(cand, features) == 1, (
            "P-44.1.1: ester carbonyl bonded to methoxy O (in parent) "
            "-> count=1 (indirect via O neighbor)"
        )

    def test_pcg_truly_disconnected_from_parent(self):
        """ case 3 alternate: PCG attach atom NEITHER in parent NOR
        bonded to parent atom.
        Use methyl propanoate atoms but parent_atom_indices={0, 1, 2}
        (the isopropyl arm only). Attach atom 4 not in parent; atom 4's
        neighbors {3, 5, 6} — none in parent_set -> count=0.
        """
        features = _make_features(
            "CC(C)CC(=O)OC",
            principal_group="ester",
            principal_group_atoms=[(4, 5, 6)],
        )
        cand = _make_cand({0, 1, 2})
        assert _count_pcgs_in_parent(cand, features) == 0, (
            "P-44.1.1: PCG attach not in parent and no neighbor in parent "
            "-> count=0 (substituent PCG)"
        )

    def test_no_principal_group_returns_zero(self):
        """ case 4: features.principal_group is None -> count=0."""
        features = _make_features(
            "CCCC(=O)O",
            principal_group=None,
            principal_group_atoms=[(3, 4, 5)],
        )
        cand = _make_cand({0, 1, 2, 3})
        assert _count_pcgs_in_parent(cand, features) == 0, (
            "CD-01: no principal_group -> nothing to count"
        )

    def test_empty_principal_group_atoms_returns_zero(self):
        """ case 5: principal_group_atoms= -> count=0."""
        features = _make_features(
            "CCCC(=O)O",
            principal_group="carboxylic_acid",
            principal_group_atoms=[],
        )
        cand = _make_cand({0, 1, 2, 3})
        assert _count_pcgs_in_parent(cand, features) == 0, (
            "CD-01: empty principal_group_atoms -> count=0"
        )

    def test_none_parent_atom_indices_returns_zero(self):
        """ case 6: candidate.parent_atom_indices=None -> 0
        (no-decision sentinel)."""
        features = _make_features(
            "CCCC(=O)O",
            principal_group="carboxylic_acid",
            principal_group_atoms=[(3, 4, 5)],
        )
        cand = _make_cand(None)
        assert _count_pcgs_in_parent(cand, features) == 0, (
            "CD-01: None parent_atom_indices -> 0 (no-decision sentinel)"
        )

    def test_multiple_pcgs_all_in_parent(self):
        """ case 7: hexanedioic acid OC(=O)CCCCC(=O)O.
        Atom indices: 0=O, 1=C(carbonyl), 2=O, 3-6=C, 7=C(carbonyl), 8=O, 9=O.
        principal_group_atoms=[(1, 0, 2), (7, 8, 9)] — attach atoms 1 and 7.
        parent_atom_indices={1, 3, 4, 5, 6, 7} (the 6-carbon chain).
        Both attach atoms 1 and 7 IN parent_set -> count=2.
        """
        features = _make_features(
            "OC(=O)CCCCC(=O)O",
            principal_group="carboxylic_acid",
            principal_group_atoms=[(1, 0, 2), (7, 8, 9)],
        )
        cand = _make_cand({1, 3, 4, 5, 6, 7})
        assert _count_pcgs_in_parent(cand, features) == 2, (
            "P-44.1.1: hexanedioic acid both carbonyls in parent -> count=2"
        )

    def test_multiple_pcgs_only_one_attaches(self):
        """ case 8: multiple PCGs but only one attaches to parent.
        Use hexanedioic acid atoms but parent_atom_indices={1, 3, 4} only.
        Attach atom 1 IN parent -> +1.
        Attach atom 7 NOT in parent; its neighbors are {6, 8, 9} — none
        in {1, 3, 4} -> +0.
        Total count = 1.
        """
        features = _make_features(
            "OC(=O)CCCCC(=O)O",
            principal_group="carboxylic_acid",
            principal_group_atoms=[(1, 0, 2), (7, 8, 9)],
        )
        cand = _make_cand({1, 3, 4})
        assert _count_pcgs_in_parent(cand, features) == 1, (
            "P-44.1.1: only first PCG inside parent -> count=1"
        )

    def test_mol_attribute_missing_returns_zero(self):
        """ case 9: features.mol is None -> count=0 (no exception)."""
        features = SimpleNamespace(
            mol=None,
            principal_group="carboxylic_acid",
            principal_group_atoms=[(3, 4, 5)],
        )
        cand = _make_cand({0, 1, 2, 3})
        assert _count_pcgs_in_parent(cand, features) == 0, (
            "CD-01: mol=None -> graceful 0 (no exception)"
        )

    def test_principal_group_atoms_contains_empty_tuple(self):
        """ case 10: principal_group_atoms=[, (3,4,5)] — empty
        tuple skipped, only second tuple counted -> count=1."""
        features = _make_features(
            "CCCC(=O)O",
            principal_group="carboxylic_acid",
            principal_group_atoms=[(), (3, 4, 5)],
        )
        cand = _make_cand({0, 1, 2, 3})
        assert _count_pcgs_in_parent(cand, features) == 1, (
            "CD-01: empty tuple in principal_group_atoms is skipped"
        )

    def test_indirect_attachment_via_carbonyl_oxygen(self):
        """ algorithm coverage: attach atom NOT in parent but neighbor
        oxygen IS in parent. Tests that the neighbor check correctly walks
        through the PCG attach atom to the parent set.

        Hexanedioic acid OC(=O)CCCCC(=O)O. Use parent_atom_indices={0}
        (just the leftmost OH oxygen). Attach atom 1 not in parent; atom
        1's neighbors are {0, 2, 3} -> 0 IS in parent -> count=1 for
        first PCG. Attach atom 7 not in parent; neighbors {6, 8, 9} —
        none in {0} -> 0. Total = 1.
        """
        features = _make_features(
            "OC(=O)CCCCC(=O)O",
            principal_group="carboxylic_acid",
            principal_group_atoms=[(1, 0, 2), (7, 8, 9)],
        )
        cand = _make_cand({0})
        assert _count_pcgs_in_parent(cand, features) == 1, (
            "P-44.1.1: PCG attach C bonded to parent O -> count=1 (indirect)"
        )

    def test_invalid_attach_atom_index_handled_gracefully(self):
        """ defensive: attach atom index out of range for the mol
        should be caught by the try/except and not raise."""
        features = _make_features(
            "CCCC(=O)O",
            principal_group="carboxylic_acid",
            # Atom 999 doesn't exist in the 6-atom mol.
            principal_group_atoms=[(999, 1000, 1001)],
        )
        cand = _make_cand({0, 1, 2, 3})
        # Attach 999 not in parent_set; the GetAtomWithIdx call raises
        # internally, caught by except -> contributes 0. Total = 0.
        assert _count_pcgs_in_parent(cand, features) == 0, (
            "CD-01: invalid atom index handled gracefully -> 0"
        )
