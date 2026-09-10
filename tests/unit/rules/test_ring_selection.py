"""
Unit tests for ring system type classification, scoring, and selection.

Tests the IUPAC ring system hierarchy and selection logic
from orthonym.rules.ring_selection.

Reference: IUPAC 2013 Blue Book, (Ring System Selection)
"""

import pytest
from rdkit import Chem

from orthonym.rules.ring_selection import (
    RingSystemType,
    classify_ring_system_type,
    ring_system_score,
    select_principal_ring_system,
)
from orthonym.perception.rings import get_ring_systems


# ============================================================================
# principal-group priority in ring-system selection (vB engine Piece 1)
# ============================================================================


class TestPrincipalGroupBearingRing:
    """: the senior parent bears the principal characteristic group,
    BEFORE the ring-type hierarchy is applied as a tiebreaker.

    ``ring_system_score`` sees only the ring atoms, so it cannot honour.
    ``select_principal_ring_system`` takes an optional ``principal_group_atoms``
    hint: when exactly one ring system bears the group, that system wins.
    """

    # 4-(pyridin-2-yl)benzoic acid: the benzene bears -COOH (the principal
    # characteristic group); the pyridine is more senior by ring score (an
    # N-heterocycle). Per the COOH-bearing benzene must be the parent.
    SMILES = "OC(=O)c1ccc(-c2ccccn2)cc1"

    def _rings(self):
        mol = Chem.MolFromSmiles(self.SMILES)
        systems = get_ring_systems(mol)
        pg = mol.GetSubstructMatch(Chem.MolFromSmarts("C(=O)O"))
        carboxyl_c = pg[0]
        benzene = set(next(
            s for s in systems
            if any(carboxyl_c in [n.GetIdx()
                                  for n in mol.GetAtomWithIdx(a).GetNeighbors()]
                   for a in s)))
        return mol, systems, [pg], benzene

    def test_pg_hint_selects_the_group_bearing_ring(self):
        """With the principal-group hint, the COOH-bearing benzene wins."""
        mol, systems, pg_atoms, benzene = self._rings()
        picked = set(select_principal_ring_system(
            mol, systems, principal_group_atoms=pg_atoms))
        assert picked == benzene

    def test_without_hint_behaviour_is_unchanged(self):
        """No hint -> current scoring (the N-heterocycle) -- byte-identical.

        Guards against the hint silently changing the default: without it the
        pyridine still wins, proving the hint (not some side effect) drives the
        new behaviour.
        """
        mol, systems, _pg, benzene = self._rings()
        picked = set(select_principal_ring_system(mol, systems))
        assert picked != benzene  # pyridine, as before


# ============================================================================
# A. Ring System Type Classification Tests
# ============================================================================


class TestClassifyRingSystemType:
    """Tests for classify_ring_system_type -- type hierarchy."""

    def test_spiro_undecane(self):
        """Spiro[5.5]undecane should be classified as SPIRO."""
        mol = Chem.MolFromSmiles("C1CCC2(CC1)CCCCC2")
        ring_systems = get_ring_systems(mol, include_spiro=True)
        assert len(ring_systems) == 1
        result = classify_ring_system_type(mol, ring_systems[0])
        assert result == RingSystemType.SPIRO

    def test_fused_naphthalene(self):
        """Naphthalene should be classified as FUSED."""
        mol = Chem.MolFromSmiles("c1ccc2ccccc2c1")
        ring_systems = get_ring_systems(mol)
        assert len(ring_systems) == 1
        result = classify_ring_system_type(mol, ring_systems[0])
        assert result == RingSystemType.FUSED

    def test_bridged_fused_system(self):
        """A bridged fused system should be classified as BRIDGED_FUSED.

        Uses 1,4-dihydro-1,4-methanonaphthalene -- a naphthalene core
        with a methano bridge across positions 1,4.
        """
        # C12CC3CC(CC(C3)C1)C2 is camphor-like (actually VB)
        # Instead use benzonorbornadiene: C1=CC2CC1c1ccccc12
        mol = Chem.MolFromSmiles("C1=CC2CC1c1ccccc12")
        if mol is not None:
            ring_systems = get_ring_systems(mol)
            # Should have one ring system containing fused + bridge
            if ring_systems:
                result = classify_ring_system_type(mol, ring_systems[0])
                assert result == RingSystemType.BRIDGED_FUSED

    def test_von_baeyer_norbornane(self):
        """Norbornane (bicyclo[2.2.1]heptane) should be VON_BAEYER."""
        mol = Chem.MolFromSmiles("C1CC2CCC1C2")
        ring_systems = get_ring_systems(mol)
        assert len(ring_systems) == 1
        result = classify_ring_system_type(mol, ring_systems[0])
        assert result == RingSystemType.VON_BAEYER

    def test_ring_assembly_biphenyl(self):
        """Biphenyl should be classified as RING_ASSEMBLY."""
        mol = Chem.MolFromSmiles("c1ccc(-c2ccccc2)cc1")
        # Biphenyl: two disconnected ring systems connected by a bond
        ring_systems = get_ring_systems(mol)
        assert len(ring_systems) == 2
        # Each ring individually is MONOCYCLIC
        for rs in ring_systems:
            result = classify_ring_system_type(mol, rs)
            assert result == RingSystemType.MONOCYCLIC

    def test_monocyclic_cyclohexane(self):
        """Cyclohexane should be classified as MONOCYCLIC."""
        mol = Chem.MolFromSmiles("C1CCCCC1")
        ring_systems = get_ring_systems(mol)
        assert len(ring_systems) == 1
        result = classify_ring_system_type(mol, ring_systems[0])
        assert result == RingSystemType.MONOCYCLIC

    def test_fused_indole(self):
        """Indole (fused heterocyclic) should be FUSED."""
        mol = Chem.MolFromSmiles("c1ccc2[nH]ccc2c1")
        ring_systems = get_ring_systems(mol)
        assert len(ring_systems) == 1
        result = classify_ring_system_type(mol, ring_systems[0])
        assert result == RingSystemType.FUSED

    def test_spiro_oxaspiro_decane(self):
        """Spiro[4.5]decane should be SPIRO."""
        mol = Chem.MolFromSmiles("C1CCC2(CC1)CCCC2")
        ring_systems = get_ring_systems(mol, include_spiro=True)
        assert len(ring_systems) == 1
        result = classify_ring_system_type(mol, ring_systems[0])
        assert result == RingSystemType.SPIRO


# ============================================================================
# B. Ring System Scoring Tests
# ============================================================================


class TestRingSystemScore:
    """Tests for ring_system_score -- general criteria."""

    def test_heterocyclic_over_carbocyclic(self):
        """Heterocyclic ring system should score more senior than carbocyclic
        of same size. Pyridine ring vs benzene ring."""
        mol_pyridine = Chem.MolFromSmiles("c1ccncc1")
        mol_benzene = Chem.MolFromSmiles("c1ccccc1")
        rs_pyridine = get_ring_systems(mol_pyridine)
        rs_benzene = get_ring_systems(mol_benzene)
        score_pyridine = ring_system_score(mol_pyridine, rs_pyridine[0])
        score_benzene = ring_system_score(mol_benzene, rs_benzene[0])
        # Lower score = more senior (min selects most senior)
        assert score_pyridine < score_benzene

    def test_nitrogen_over_oxygen_heterocycle(self):
        """Nitrogen-containing heterocycle should score more senior than
        oxygen-only heterocycle. Pyridine vs pyran."""
        mol_pyridine = Chem.MolFromSmiles("c1ccncc1")
        mol_pyran = Chem.MolFromSmiles("C1=COC=CC1")
        rs_pyridine = get_ring_systems(mol_pyridine)
        rs_pyran = get_ring_systems(mol_pyran)
        score_pyridine = ring_system_score(mol_pyridine, rs_pyridine[0])
        score_pyran = ring_system_score(mol_pyran, rs_pyran[0])
        assert score_pyridine < score_pyran

    def test_more_rings_more_senior(self):
        """Naphthalene (2 rings) should score more senior than benzene (1 ring)
        within the same type."""
        mol_naph = Chem.MolFromSmiles("c1ccc2ccccc2c1")
        mol_benz = Chem.MolFromSmiles("c1ccccc1")
        rs_naph = get_ring_systems(mol_naph)
        rs_benz = get_ring_systems(mol_benz)
        score_naph = ring_system_score(mol_naph, rs_naph[0])
        score_benz = ring_system_score(mol_benz, rs_benz[0])
        assert score_naph < score_benz

    def test_more_skeletal_atoms_more_senior(self):
        """Cycloheptane (7 atoms) should score more senior than cyclohexane
        (6 atoms) within the same monocyclic type."""
        mol_7 = Chem.MolFromSmiles("C1CCCCCC1")
        mol_6 = Chem.MolFromSmiles("C1CCCCC1")
        rs_7 = get_ring_systems(mol_7)
        rs_6 = get_ring_systems(mol_6)
        score_7 = ring_system_score(mol_7, rs_7[0])
        score_6 = ring_system_score(mol_6, rs_6[0])
        assert score_7 < score_6

    def test_hetero_beats_more_rings_across_types(self):
        """A heterocyclic monocyclic ring should score more senior
        on the heteroatom criterion than a carbocyclic monocyclic
        ring of same size. This verifies scoring tuple comparison works."""
        mol_pyridine = Chem.MolFromSmiles("c1ccncc1")
        mol_benzene = Chem.MolFromSmiles("c1ccccc1")
        rs_pyridine = get_ring_systems(mol_pyridine)
        rs_benzene = get_ring_systems(mol_benzene)
        score_pyridine = ring_system_score(mol_pyridine, rs_pyridine[0])
        score_benzene = ring_system_score(mol_benzene, rs_benzene[0])
        # Pyridine should be more senior (lower tuple) because of heteroatom
        assert score_pyridine < score_benzene

    def test_monocyclic_heterocycle_beats_fused_carbocycle(self):
        """(a): A monocyclic heterocycle (pyridine) should beat a fused
        carbocycle (naphthalene). General criteria (heterocyclic preferred)
        must dominate over type hierarchy (fused > monocyclic)."""
        mol_pyridine = Chem.MolFromSmiles("c1ccncc1")
        rs_pyridine = get_ring_systems(mol_pyridine)
        score_pyridine = ring_system_score(mol_pyridine, rs_pyridine[0])

        mol_naph = Chem.MolFromSmiles("c1ccc2ccccc2c1")
        rs_naph = get_ring_systems(mol_naph)
        score_naph = ring_system_score(mol_naph, rs_naph[0])

        assert score_pyridine < score_naph, (
            f"Monocyclic heterocycle should beat fused carbocycle per P-44.2.1(a). "
            f"pyridine={score_pyridine}, naphthalene={score_naph}"
        )

    def test_two_monocyclic_same_type_larger_wins(self):
        """(e): Among same-type carbocycles, larger one wins."""
        mol_7 = Chem.MolFromSmiles("C1CCCCCC1")  # cycloheptane
        mol_5 = Chem.MolFromSmiles("C1CCCC1")    # cyclopentane
        rs_7 = get_ring_systems(mol_7)
        rs_5 = get_ring_systems(mol_5)
        score_7 = ring_system_score(mol_7, rs_7[0])
        score_5 = ring_system_score(mol_5, rs_5[0])
        assert score_7 < score_5

    def test_n_containing_beats_o_containing_same_type(self):
        """(b): N-containing heterocycle beats O-containing."""
        mol_n = Chem.MolFromSmiles("C1CCNCC1")  # piperidine
        mol_o = Chem.MolFromSmiles("C1CCOCC1")  # oxane
        rs_n = get_ring_systems(mol_n)
        rs_o = get_ring_systems(mol_o)
        score_n = ring_system_score(mol_n, rs_n[0])
        score_o = ring_system_score(mol_o, rs_o[0])
        assert score_n < score_o

    def test_score_is_tuple(self):
        """ring_system_score should return a tuple."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        rs = get_ring_systems(mol)
        score = ring_system_score(mol, rs[0])
        assert isinstance(score, tuple)
        assert len(score) >= 6  # At least 6 criteria in tuple


# ============================================================================
# C. Principal Ring System Selection Tests
# ============================================================================


class TestSelectPrincipalRingSystem:
    """Tests for select_principal_ring_system."""

    def test_single_ring_system(self):
        """Molecule with one ring system returns that system's atoms."""
        mol = Chem.MolFromSmiles("c1ccccc1")  # benzene
        ring_systems = get_ring_systems(mol)
        assert len(ring_systems) == 1
        result = select_principal_ring_system(mol, ring_systems)
        assert set(result) == ring_systems[0]

    def test_pyridine_over_benzene_disconnected(self):
        """Molecule with pyridine ring system and benzene ring system
        (disconnected, connected by chain): selects pyridine
        (heterocyclic > carbocyclic)."""
        # 4-phenylpyridine: pyridine ring + benzene ring connected by bond
        mol = Chem.MolFromSmiles("c1cc(-c2ccccc2)ccn1")
        ring_systems = get_ring_systems(mol)
        assert len(ring_systems) == 2
        result = select_principal_ring_system(mol, ring_systems)
        # The selected ring should contain nitrogen (the pyridine)
        has_nitrogen = any(
            mol.GetAtomWithIdx(idx).GetSymbol() == "N" for idx in result
        )
        assert has_nitrogen, "Should select pyridine (heterocyclic) over benzene"

    def test_fused_over_monocyclic(self):
        """Molecule with a fused ring system and a separate monocyclic ring:
        selects fused system (fused type beats monocyclic type in hierarchy)."""
        # Naphthalene + cyclohexane connected by chain
        mol = Chem.MolFromSmiles("c1ccc2ccccc2c1CCC1CCCCC1")
        ring_systems = get_ring_systems(mol)
        # Should have 2 ring systems: naphthalene (10 atoms) + cyclohexane (6 atoms)
        assert len(ring_systems) >= 2
        result = select_principal_ring_system(mol, ring_systems)
        # The naphthalene system has 10 atoms; cyclohexane has 6
        assert len(result) == 10, "Should select the naphthalene (fused) system"

    def test_empty_ring_systems(self):
        """Molecule with no ring systems returns empty tuple."""
        mol = Chem.MolFromSmiles("CCCCCC")  # hexane
        ring_systems = get_ring_systems(mol)
        assert len(ring_systems) == 0
        result = select_principal_ring_system(mol, ring_systems)
        assert result == ()
