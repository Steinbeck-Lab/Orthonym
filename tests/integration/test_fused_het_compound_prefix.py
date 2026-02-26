"""
Integration tests for compound prefix generation of substituted fused heterocycles.

Phase 78 Plan 03 — When a fused heterocycle ring has its own substituents AND
is itself a substituent on a chain parent, the output should be a compound prefix
like "(5-methyl-1H-indol-3-yl)".

Tests verify FHET-06 requirement.
"""

import subprocess
import pytest
from pathlib import Path
from rdkit import Chem

from orthonym.data.fused_heterocycles import (
    get_substituted_fused_het_prefix,
    FUSED_HETEROCYCLE_PREFIX_STEMS,
    match_fused_heterocycle_core,
)


@pytest.mark.integration
class TestSubstitutedFusedHetCompoundPrefix:
    """Test compound prefix generation for substituted fused het substituents."""

    def test_get_substituted_fused_het_prefix_basic(self):
        """Basic compound prefix: inner subs + stem + locant + yl in parens."""
        # 5-methyl-1H-indole core
        mol = Chem.MolFromSmiles('Cc1ccc2[nH]ccc2c1')
        result = match_fused_heterocycle_core(mol)
        assert result is not None
        name, atom_mapping, core_smiles = result

        # Find C-3 attachment point (a carbon in the pyrrole ring)
        # For this test, use a known attachment point
        # C-3 in indole - find the atom mapped to IUPAC locant 3
        attach_idx = None
        for mol_idx, locant in atom_mapping.items():
            if locant == 3:
                attach_idx = mol_idx
                break

        if attach_idx is not None:
            prefix = get_substituted_fused_het_prefix(
                core_smiles, attach_idx, atom_mapping,
                "5-methyl-"
            )
            assert prefix is not None
            assert prefix == "(5-methyl-1H-indol-3-yl)"

    def test_compound_prefix_chloro_quinoline(self):
        """6-chloroquinoline as substituent → (6-chloroquinolin-X-yl)."""
        # 6-chloroquinoline
        mol = Chem.MolFromSmiles('Clc1ccc2ncccc2c1')
        result = match_fused_heterocycle_core(mol)
        assert result is not None
        name, atom_mapping, core_smiles = result

        # Find attachment locant 2
        attach_idx = None
        for mol_idx, locant in atom_mapping.items():
            if locant == 2:
                attach_idx = mol_idx
                break

        if attach_idx is not None:
            prefix = get_substituted_fused_het_prefix(
                core_smiles, attach_idx, atom_mapping,
                "6-chloro"
            )
            assert prefix is not None
            assert "chloro" in prefix
            assert "quinolin" in prefix
            assert prefix.startswith("(") and prefix.endswith(")")


@pytest.mark.integration
class TestCompoundPrefixParenthesized:
    """Verify compound prefixes are always enclosed in parentheses."""

    def test_parenthesized(self):
        """All compound prefixes should be wrapped in parentheses."""
        smiles = 'c1ccc2[nH]ccc2c1'  # indole core
        stem = FUSED_HETEROCYCLE_PREFIX_STEMS.get(smiles)
        assert stem is not None

        mol = Chem.MolFromSmiles('Cc1ccc2[nH]ccc2c1')  # 5-methyl-1H-indole
        result = match_fused_heterocycle_core(mol)
        assert result is not None
        _, atom_mapping, core_smiles = result

        # Get any valid attachment point
        attach_idx = None
        for mol_idx, locant in atom_mapping.items():
            if isinstance(locant, int):
                attach_idx = mol_idx
                break

        if attach_idx is not None:
            prefix = get_substituted_fused_het_prefix(
                core_smiles, attach_idx, atom_mapping,
                "5-methyl-"
            )
            assert prefix is not None
            assert prefix.startswith("(")
            assert prefix.endswith(")")


@pytest.mark.integration
class TestUnsubstitutedStillWorks:
    """Regression: unsubstituted fused het substituents still produce simple prefixes."""

    def test_simple_prefix_still_works(self):
        """get_fused_heterocycle_prefix still returns simple prefix."""
        from orthonym.data.fused_heterocycles import get_fused_heterocycle_prefix
        # Quinoline at C-2
        mol = Chem.MolFromSmiles('Cc1ccc2ccccc2n1')
        result = match_fused_heterocycle_core(mol)
        assert result is not None
        name, atom_mapping, core_smiles = result
        methyl_atom = mol.GetAtomWithIdx(0)
        ring_attach = None
        for nbr in methyl_atom.GetNeighbors():
            ring_attach = nbr.GetIdx()
            break
        prefix = get_fused_heterocycle_prefix(core_smiles, ring_attach, atom_mapping)
        assert prefix is not None
        assert "quinolin" in prefix
        assert "-yl" in prefix

    def test_benzothiazole_simple_prefix(self):
        """1,3-benzothiazole simple prefix."""
        from orthonym.data.fused_heterocycles import get_fused_heterocycle_prefix
        mol = Chem.MolFromSmiles('Cc1nc2ccccc2s1')
        result = match_fused_heterocycle_core(mol)
        assert result is not None
        _, atom_mapping, core_smiles = result
        methyl_atom = mol.GetAtomWithIdx(0)
        ring_attach = None
        for nbr in methyl_atom.GetNeighbors():
            ring_attach = nbr.GetIdx()
            break
        prefix = get_fused_heterocycle_prefix(core_smiles, ring_attach, atom_mapping)
        assert prefix is not None
        assert "benzothiazol" in prefix

    def test_indole_simple_prefix(self):
        """1H-indole simple prefix."""
        from orthonym.data.fused_heterocycles import get_fused_heterocycle_prefix
        mol = Chem.MolFromSmiles('Cc1c[nH]c2ccccc12')
        result = match_fused_heterocycle_core(mol)
        assert result is not None
        _, atom_mapping, core_smiles = result
        methyl_atom = mol.GetAtomWithIdx(0)
        ring_attach = None
        for nbr in methyl_atom.GetNeighbors():
            ring_attach = nbr.GetIdx()
            break
        prefix = get_fused_heterocycle_prefix(core_smiles, ring_attach, atom_mapping)
        assert prefix is not None
        assert "1H-indol" in prefix


@pytest.mark.integration
class TestCompoundPrefixNoRecursion:
    """Verify compound prefix uses static lookup, no recursive naming."""

    def test_no_recursive_naming_in_compound_prefix(self):
        """get_substituted_fused_het_prefix does not import or call name_compound."""
        import inspect
        src = inspect.getsource(get_substituted_fused_het_prefix)
        assert "name_compound" not in src
