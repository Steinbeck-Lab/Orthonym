"""
Unit tests for multi-ring substituent prefix generation (Phase 82).

Tests cover:
- Ring assembly substituent prefixes (IUPAC P-28.3): [1,1'-biphenyl]-4-yl format
- Mixed-ring compound substituent prefixes (IUPAC P-31): senior ring as parent
- Multi-ring fragment grouping: _merge_connected_ring_groups correctness
- Regression: single ring and fused het substituents still work
"""

import pytest
from rdkit import Chem

from orthonym.namer import name_compound
from orthonym.rules.ring_assemblies import (
    detect_ring_assembly,
    name_ring_assembly_prefix,
    name_mixed_ring_prefix,
    _find_inter_system_bonds,
)
from orthonym.assembly.composer import _merge_connected_ring_groups


# ============================================================================
# Ring Assembly Prefix Tests (IUPAC P-28.3)
# ============================================================================


@pytest.mark.unit
def test_biphenyl_on_chain_acid():
    """Biphenylbutanoic acid produces [1,1'-biphenyl]-4-yl prefix."""
    name = name_compound("OC(=O)CCCc1ccc(-c2ccccc2)cc1")
    assert "[1,1'-biphenyl]-4-yl" in name, (
        f"Expected [1,1'-biphenyl]-4-yl in name, got: {name}"
    )
    assert "butanoic acid" in name


@pytest.mark.unit
def test_biphenyl_on_chain_ethanoic():
    """Biphenyl ethanoic acid uses the same prefix format at different chain length."""
    name = name_compound("OC(=O)Cc1ccc(-c2ccccc2)cc1")
    assert "[1,1'-biphenyl]-4-yl" in name, (
        f"Expected [1,1'-biphenyl]-4-yl in name, got: {name}"
    )


@pytest.mark.unit
def test_biphenyl_substituent_format():
    """Ring assembly prefix uses square bracket enclosing: [assembly_name]-N-yl."""
    name = name_compound("OC(=O)CCCc1ccc(-c2ccccc2)cc1")
    # Must contain square bracket pattern: [<assembly>]-<locant>-yl
    assert "[" in name and "]-" in name, (
        f"Expected square bracket enclosing in name, got: {name}"
    )
    # Must have parentheses around the entire prefix per P-16.5.1.1
    assert "([" in name, (
        f"Expected parenthesized bracket prefix, got: {name}"
    )


@pytest.mark.unit
def test_bipyridyl_on_chain():
    """Bipyridine assembly on chain produces [X,X'-bipyridin]-N-yl prefix."""
    name = name_compound("OC(=O)CCCc1ccnc(-c2ccccn2)c1")
    assert "bipyridin" in name, (
        f"Expected bipyridin in name, got: {name}"
    )
    assert "[" in name and "]-" in name, (
        f"Expected bracket format, got: {name}"
    )


@pytest.mark.unit
def test_ring_assembly_prefix_function_directly():
    """Unit test name_ring_assembly_prefix() with a biphenyl molecule."""
    mol = Chem.MolFromSmiles("OC(=O)CCCc1ccc(-c2ccccc2)cc1")
    ri = mol.GetRingInfo()
    rings = ri.AtomRings()

    ring_systems = [set(r) for r in rings]

    assembly_info = detect_ring_assembly(mol, ring_systems)
    assert assembly_info is not None, "detect_ring_assembly should find biphenyl"
    assert assembly_info['count'] == 2

    # Find the atom that connects to the chain (non-ring atom neighbor)
    all_ring_atoms = set()
    for rs in ring_systems:
        all_ring_atoms.update(rs)

    attach_atom = None
    for ra in all_ring_atoms:
        atom = mol.GetAtomWithIdx(ra)
        for nbr in atom.GetNeighbors():
            if nbr.GetIdx() not in all_ring_atoms:
                # Check if this non-ring atom is a carbon (chain atom)
                if nbr.GetSymbol() == 'C':
                    attach_atom = ra
                    break
        if attach_atom is not None:
            break

    assert attach_atom is not None
    prefix = name_ring_assembly_prefix(mol, assembly_info, attach_atom)
    assert prefix is not None
    assert "biphenyl" in prefix
    assert prefix.startswith("[") and "]-" in prefix
    assert prefix.endswith("-yl")


# ============================================================================
# Mixed-Ring Compound Prefix Tests (IUPAC P-31)
# ============================================================================


@pytest.mark.unit
def test_phenyl_pyridyl_on_chain():
    """Phenyl-pyridyl on chain produces compound prefix with correct locants."""
    # Chain attaches to benzene, pyridine is substituent on benzene
    name = name_compound("OC(=O)CCCCCc1ccc(-c2ccccn2)cc1")
    # Should contain pyridyl or pyridin as part of the compound prefix
    assert "pyrid" in name.lower(), (
        f"Expected pyridyl/pyridin in name, got: {name}"
    )
    # Should still have a chain acid suffix
    assert "hexanoic acid" in name


@pytest.mark.unit
def test_phenyl_pyridyl_chain_on_pyridine():
    """When chain attaches to pyridine (senior), phenyl is the substituent."""
    name = name_compound("OC(=O)CCCCCc1ccnc(-c2ccccc2)c1")
    # Pyridine is parent of compound prefix, phenyl is substituent
    assert "phenyl" in name.lower(), (
        f"Expected phenyl in name, got: {name}"
    )
    assert "pyridin" in name.lower(), (
        f"Expected pyridin stem in name, got: {name}"
    )
    # The pyridin stem should have a -yl suffix (compound prefix)
    assert "-yl)" in name, (
        f"Expected compound prefix with -yl), got: {name}"
    )


@pytest.mark.unit
def test_mixed_ring_prefix_function_directly():
    """Unit test name_mixed_ring_prefix() with a phenyl-pyridyl molecule."""
    mol = Chem.MolFromSmiles("OC(=O)CCCCCc1ccnc(-c2ccccc2)c1")
    ri = mol.GetRingInfo()
    rings = ri.AtomRings()

    ring_systems = [set(r) for r in rings]
    inter_bonds = _find_inter_system_bonds(mol, ring_systems)

    assert len(inter_bonds) > 0, "Should find inter-ring bond"

    # Find which ring the chain attaches to
    all_ring_atoms = set()
    for rs in ring_systems:
        all_ring_atoms.update(rs)

    attach_atom = None
    for ra in all_ring_atoms:
        atom = mol.GetAtomWithIdx(ra)
        for nbr in atom.GetNeighbors():
            if nbr.GetIdx() not in all_ring_atoms:
                if nbr.GetSymbol() == 'C':
                    attach_atom = ra
                    break
        if attach_atom is not None:
            break

    assert attach_atom is not None
    prefix = name_mixed_ring_prefix(mol, ring_systems, inter_bonds, attach_atom)
    assert prefix is not None, "name_mixed_ring_prefix should return a prefix"
    assert prefix.startswith("(") and prefix.endswith(")"), (
        f"Compound prefix should be parenthesized, got: {prefix}"
    )
    assert "phenyl" in prefix, (
        f"Expected 'phenyl' (junior ring) in prefix, got: {prefix}"
    )
    assert "pyridin" in prefix, (
        f"Expected 'pyridin' (senior ring stem) in prefix, got: {prefix}"
    )


# ============================================================================
# Multi-Ring Fragment Grouping Tests
# ============================================================================


@pytest.mark.unit
def test_merge_connected_ring_groups_biphenyl():
    """Two phenyl ring tuples connected by single bond merge into one fragment."""
    mol = Chem.MolFromSmiles("c1ccc(-c2ccccc2)cc1")
    ri = mol.GetRingInfo()
    ring_groups = [tuple(r) for r in ri.AtomRings()]

    assert len(ring_groups) == 2, "Biphenyl should have 2 SSSR rings"

    multi, single = _merge_connected_ring_groups(mol, ring_groups)
    assert len(multi) == 1, "Should produce 1 multi-ring fragment"
    assert len(single) == 0, "Should have 0 single-ring groups"
    assert len(multi[0]) == 2, "Multi-ring fragment should contain 2 ring tuples"


@pytest.mark.unit
def test_merge_connected_ring_groups_no_merge():
    """Two unconnected ring substituents remain as single groups."""
    # Two separate phenyl groups on a chain (not connected to each other)
    mol = Chem.MolFromSmiles("c1ccc(CCCC(c2ccccc2)CC)cc1")
    ri = mol.GetRingInfo()
    ring_groups = [tuple(r) for r in ri.AtomRings()]

    assert len(ring_groups) == 2, "Should have 2 SSSR rings"

    multi, single = _merge_connected_ring_groups(mol, ring_groups)
    assert len(multi) == 0, "No multi-ring fragments expected"
    assert len(single) == 2, "Both rings should remain as single groups"


@pytest.mark.unit
def test_merge_connected_ring_groups_triple():
    """Three connected rings (terphenyl) merge into one multi-ring fragment."""
    mol = Chem.MolFromSmiles("c1ccc(-c2ccc(-c3ccccc3)cc2)cc1")
    ri = mol.GetRingInfo()
    ring_groups = [tuple(r) for r in ri.AtomRings()]

    assert len(ring_groups) == 3, "Terphenyl should have 3 SSSR rings"

    multi, single = _merge_connected_ring_groups(mol, ring_groups)
    assert len(multi) == 1, "Should produce 1 multi-ring fragment"
    assert len(single) == 0, "No single-ring groups expected"
    assert len(multi[0]) == 3, "Multi-ring fragment should contain 3 ring tuples"


# ============================================================================
# Regression / Fallback Tests
# ============================================================================


@pytest.mark.unit
def test_single_ring_still_works():
    """Simple phenyl substituent on chain still produces 'phenyl'."""
    name = name_compound("OC(=O)CCCc1ccccc1")
    assert "phenyl" in name, (
        f"Expected 'phenyl' in name, got: {name}"
    )
    assert "butanoic acid" in name
    # Must NOT contain bracket format (single ring, not assembly)
    assert "[" not in name, (
        f"Single ring should not have bracket format, got: {name}"
    )


@pytest.mark.unit
def test_fused_het_still_works():
    """Fused heterocycle substituent (quinoline) still produces correct prefix.

    Fused rings share edges (not single bonds between separate rings),
    so they should NOT be consumed by the multi-ring detection pass.
    """
    # Quinoline on a chain
    name = name_compound("OC(=O)CCCCCc1ccc2ncccc2c1")
    # Should contain quinolin (fused het prefix)
    assert "quinolin" in name.lower(), (
        f"Expected 'quinolin' prefix for fused het, got: {name}"
    )
    # Should NOT be treated as a ring assembly
    assert "[" not in name or "bi" not in name, (
        f"Fused het should not be treated as ring assembly, got: {name}"
    )
