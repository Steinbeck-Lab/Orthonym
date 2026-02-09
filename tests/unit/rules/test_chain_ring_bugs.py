"""
Regression tests for ring-atom leakage into chain-based naming paths.

Bug 1: Ring atoms included in chain-finding when they should be excluded,
        producing wrong chain lengths (e.g., hexyl instead of phenyl).
Bug 2: Duplicate substituent naming where ring atoms produce both a ring
        prefix (phenyl) and an alkyl prefix (hexyl) for the same atoms.
Bug 3: Amino acid systematic naming counts all carbons including ring
        carbons, producing wrong chain stems (e.g., nonanoic for tyrosine).
"""
import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.perception.chains import find_principal_chain
from orthonym.perception.functional_groups import detect_functional_groups
from orthonym.rules.seniority import get_principal_group


class TestChainFindingExcludesRingAtoms:
    """Test that find_principal_chain correctly excludes ring atoms."""

    def test_phenylbutanoic_acid_chain_with_exclusion(self):
        """Chain for 4-phenylbutanoic acid should be 4 atoms, not 10."""
        mol = Chem.MolFromSmiles("c1ccc(CCCC(=O)O)cc1")
        fgs = detect_functional_groups(mol)
        pg_name, _ = get_principal_group(mol, fgs)

        # Get ring atoms
        ri = mol.GetRingInfo()
        ring_atoms = set()
        for ring in ri.AtomRings():
            ring_atoms.update(ring)

        chain = find_principal_chain(mol, fgs, pg_name, exclude_atoms=ring_atoms)
        assert len(chain) == 4, (
            f"Expected chain length 4 (butanoic), got {len(chain)}"
        )

    def test_phenylbutanoic_acid_chain_without_exclusion_includes_ring(self):
        """Without exclusion, chain traverses through ring (demonstrates bug)."""
        mol = Chem.MolFromSmiles("c1ccc(CCCC(=O)O)cc1")
        fgs = detect_functional_groups(mol)
        pg_name, _ = get_principal_group(mol, fgs)

        chain = find_principal_chain(mol, fgs, pg_name, exclude_atoms=None)
        # Without exclusion, chain includes ring atoms (>4)
        assert len(chain) > 4, (
            "Without ring exclusion, chain should traverse through ring atoms"
        )

    def test_simple_acyclic_molecule_unaffected(self):
        """Pentanoic acid (no rings) works correctly with defensive guard."""
        mol = Chem.MolFromSmiles("CCCCC(=O)O")
        fgs = detect_functional_groups(mol)
        pg_name, _ = get_principal_group(mol, fgs)

        # No ring atoms
        ri = mol.GetRingInfo()
        ring_atoms = set()
        for ring in ri.AtomRings():
            ring_atoms.update(ring)
        assert len(ring_atoms) == 0

        chain = find_principal_chain(
            mol, fgs, pg_name,
            exclude_atoms=ring_atoms if ring_atoms else None
        )
        assert len(chain) == 5, (
            f"Expected chain length 5 (pentanoic), got {len(chain)}"
        )


class TestNamerChainRingOutput:
    """Test that the namer produces correct names for chain-ring compounds."""

    def test_phenylbutanoic_acid_name_contains_butanoic(self):
        """4-phenylbutanoic acid should contain 'butanoic', not 'decanoic'."""
        name = name_compound("c1ccc(CCCC(=O)O)cc1")
        assert "butanoic" in name, (
            f"Expected 'butanoic' in name, got: {name}"
        )

    def test_phenylbutanoic_acid_name_has_phenyl(self):
        """4-phenylbutanoic acid should contain 'phenyl'."""
        name = name_compound("c1ccc(CCCC(=O)O)cc1")
        assert "phenyl" in name, (
            f"Expected 'phenyl' in name, got: {name}"
        )

    def test_pentanoic_acid_still_works(self):
        """Pentanoic acid (acyclic) should still be named correctly."""
        name = name_compound("CCCCC(=O)O")
        assert "pentanoic acid" in name, (
            f"Expected 'pentanoic acid' in name, got: {name}"
        )


class TestNoDuplicateSubstituents:
    """Test that ring-containing chain compounds don't get duplicate substituents."""

    def test_phenylbutanoic_acid_no_hexyl(self):
        """4-phenylbutanoic acid should NOT contain 'hexyl'."""
        name = name_compound("c1ccc(CCCC(=O)O)cc1")
        assert "hexyl" not in name, (
            f"'hexyl' (ring atoms as alkyl) found in name: {name}"
        )

    def test_phenylpropanediol_no_hexyl(self):
        """2-phenylethane-1,2-diol should NOT contain 'hexyl'."""
        name = name_compound("OC(CO)c1ccccc1")
        assert "hexyl" not in name, (
            f"'hexyl' (ring atoms as alkyl) found in name: {name}"
        )

    def test_phenylpropanediol_has_phenyl(self):
        """2-phenylethane-1,2-diol should contain 'phenyl'."""
        name = name_compound("OC(CO)c1ccccc1")
        assert "phenyl" in name, (
            f"Expected 'phenyl' in name, got: {name}"
        )

    def test_propylbenzene_no_duplicate(self):
        """Propylbenzene should not contain 'hexyl'."""
        name = name_compound("c1ccc(CCC)cc1")
        assert "hexyl" not in name, (
            f"'hexyl' found in name: {name}"
        )


class TestAminoAcidRingAtomLeakage:
    """Test that amino acid naming doesn't count ring carbons in chain stem."""

    def test_tyrosine_not_nonanoic(self):
        """Tyrosine should NOT produce 'nonanoic' (9 carbons incl. ring)."""
        name = name_compound("N[C@@H](Cc1ccc(O)cc1)C(=O)O")
        assert "nonanoic" not in name.lower(), (
            f"Ring carbons leaked into amino acid chain: {name}"
        )

    def test_bromophenylalanine_not_nonanoic(self):
        """3-(4-bromophenyl)alanine should NOT produce 'nonanoic'."""
        name = name_compound("N[C@@H](Cc1ccc(Br)cc1)C(=O)O")
        assert "nonanoic" not in name.lower(), (
            f"Ring carbons leaked into amino acid chain: {name}"
        )

    def test_tyrosine_has_phenyl_or_hydroxyphenyl(self):
        """Tyrosine name should reference the hydroxyphenyl substituent."""
        name = name_compound("N[C@@H](Cc1ccc(O)cc1)C(=O)O")
        assert "phenyl" in name.lower(), (
            f"Expected 'phenyl' in tyrosine name, got: {name}"
        )
