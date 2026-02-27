"""
Integration tests for expanded fused heterocycle inner substituent detection.

Phase 80.5 Plan 01 -- Validates that _detect_fused_het_inner_subs() correctly
identifies cyano, nitro, trifluoromethyl, general alkyl (propyl/butyl), and
general alkoxy (ethoxy/propoxy) substituents on fused heterocycle cores.

Tests verify FHET-01 and FHET-06 requirements.
"""

import pytest
from collections import defaultdict
from rdkit import Chem

from orthonym.data.fused_heterocycles import match_fused_heterocycle_core
from orthonym.assembly.composer import _detect_fused_het_inner_subs


def _setup_inner_sub_test(smiles):
    """Helper: parse SMILES, match fused het core, return args for _detect_fused_het_inner_subs.

    Returns (mol, core_atom_set, atom_mapping, ring_atom_set, chain_set, attach_ring_idx)
    or None if no fused het core found.
    """
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Failed to parse SMILES: {smiles}"
    result = match_fused_heterocycle_core(mol)
    if result is None:
        return None
    name, atom_mapping, core_smiles = result
    core_atom_set = set(atom_mapping.keys())
    # ring_atom_set is same as core_atom_set for simple fused het
    ring_atom_set = set(core_atom_set)
    # chain_set is empty for isolated ring tests
    chain_set = set()
    # Find attachment point: pick a core atom that has NO non-core, non-H neighbors
    # (i.e., not the atom with the substituent). This simulates a chain attachment point.
    attach_ring_idx = None
    for ci in sorted(core_atom_set):
        atom = mol.GetAtomWithIdx(ci)
        has_non_core_nbr = any(
            n.GetIdx() not in core_atom_set for n in atom.GetNeighbors()
        )
        if not has_non_core_nbr:
            attach_ring_idx = ci
            break
    if attach_ring_idx is None:
        # Fallback: use the first core atom (some may still work)
        attach_ring_idx = min(core_atom_set)
    return mol, core_atom_set, atom_mapping, ring_atom_set, chain_set, attach_ring_idx


@pytest.mark.integration
class TestCyanoInnerSub:
    """Cyano substituent detection on fused het cores."""

    def test_cyano_inner_sub(self):
        """5-cyano-1H-indole core should return prefix containing '5-cyano-'."""
        # 5-cyano-1H-indole
        smiles = 'N#Cc1ccc2[nH]ccc2c1'
        args = _setup_inner_sub_test(smiles)
        assert args is not None, "Failed to match fused het core"
        mol, core_atom_set, atom_mapping, ring_atom_set, chain_set, attach_ring_idx = args
        result = _detect_fused_het_inner_subs(
            mol, core_atom_set, atom_mapping, ring_atom_set, chain_set, attach_ring_idx
        )
        assert result is not None and result != ""
        assert 'cyano' in result, f"Expected 'cyano' in result, got: {result}"
        assert '5-cyano' in result, f"Expected '5-cyano' in result, got: {result}"


@pytest.mark.integration
class TestNitroInnerSub:
    """Nitro substituent detection on fused het cores."""

    def test_nitro_inner_sub(self):
        """5-nitro-1H-indole core should return prefix containing '5-nitro-'."""
        # 5-nitro-1H-indole
        smiles = 'O=[N+]([O-])c1ccc2[nH]ccc2c1'
        args = _setup_inner_sub_test(smiles)
        assert args is not None, "Failed to match fused het core"
        mol, core_atom_set, atom_mapping, ring_atom_set, chain_set, attach_ring_idx = args
        result = _detect_fused_het_inner_subs(
            mol, core_atom_set, atom_mapping, ring_atom_set, chain_set, attach_ring_idx
        )
        assert result is not None and result != ""
        assert 'nitro' in result, f"Expected 'nitro' in result, got: {result}"
        assert '5-nitro' in result, f"Expected '5-nitro' in result, got: {result}"


@pytest.mark.integration
class TestTrifluoromethylInnerSub:
    """Trifluoromethyl (haloalkyl) substituent detection on fused het cores."""

    def test_trifluoromethyl_inner_sub(self):
        """5-(trifluoromethyl)-1H-indole core should return prefix containing 'trifluoromethyl'."""
        # 5-(trifluoromethyl)-1H-indole
        smiles = 'FC(F)(F)c1ccc2[nH]ccc2c1'
        args = _setup_inner_sub_test(smiles)
        assert args is not None, "Failed to match fused het core"
        mol, core_atom_set, atom_mapping, ring_atom_set, chain_set, attach_ring_idx = args
        result = _detect_fused_het_inner_subs(
            mol, core_atom_set, atom_mapping, ring_atom_set, chain_set, attach_ring_idx
        )
        assert result is not None and result != ""
        assert 'trifluoromethyl' in result, f"Expected 'trifluoromethyl' in result, got: {result}"


@pytest.mark.integration
class TestGeneralAlkylInnerSub:
    """General alkyl (propyl, butyl) substituent detection on fused het cores."""

    def test_propyl_inner_sub(self):
        """5-propyl-1H-indole core should return prefix containing '5-propyl-'."""
        # 5-propyl-1H-indole
        smiles = 'CCCc1ccc2[nH]ccc2c1'
        args = _setup_inner_sub_test(smiles)
        assert args is not None, "Failed to match fused het core"
        mol, core_atom_set, atom_mapping, ring_atom_set, chain_set, attach_ring_idx = args
        result = _detect_fused_het_inner_subs(
            mol, core_atom_set, atom_mapping, ring_atom_set, chain_set, attach_ring_idx
        )
        assert result is not None and result != ""
        assert 'propyl' in result, f"Expected 'propyl' in result, got: {result}"
        assert 'methyl' not in result, f"Unexpected 'methyl' in result, got: {result}"

    def test_butyl_inner_sub(self):
        """5-butyl-1H-indole core should return prefix containing '5-butyl-'."""
        # 5-butyl-1H-indole
        smiles = 'CCCCc1ccc2[nH]ccc2c1'
        args = _setup_inner_sub_test(smiles)
        assert args is not None, "Failed to match fused het core"
        mol, core_atom_set, atom_mapping, ring_atom_set, chain_set, attach_ring_idx = args
        result = _detect_fused_het_inner_subs(
            mol, core_atom_set, atom_mapping, ring_atom_set, chain_set, attach_ring_idx
        )
        assert result is not None and result != ""
        assert 'butyl' in result, f"Expected 'butyl' in result, got: {result}"
        assert 'methyl' not in result, f"Unexpected 'methyl' in result, got: {result}"


@pytest.mark.integration
class TestGeneralAlkoxyInnerSub:
    """General alkoxy (ethoxy, propoxy) substituent detection on fused het cores."""

    def test_ethoxy_inner_sub(self):
        """5-ethoxy-1H-indole core should return prefix containing '5-ethoxy-'."""
        # 5-ethoxy-1H-indole
        smiles = 'CCOc1ccc2[nH]ccc2c1'
        args = _setup_inner_sub_test(smiles)
        assert args is not None, "Failed to match fused het core"
        mol, core_atom_set, atom_mapping, ring_atom_set, chain_set, attach_ring_idx = args
        result = _detect_fused_het_inner_subs(
            mol, core_atom_set, atom_mapping, ring_atom_set, chain_set, attach_ring_idx
        )
        assert result is not None and result != ""
        assert 'ethoxy' in result, f"Expected 'ethoxy' in result, got: {result}"
        assert 'methoxy' not in result, f"Unexpected 'methoxy' in result, got: {result}"

    def test_propoxy_inner_sub(self):
        """5-propoxy-1H-indole core should return prefix containing '5-propoxy-'."""
        # 5-propoxy-1H-indole
        smiles = 'CCCOc1ccc2[nH]ccc2c1'
        args = _setup_inner_sub_test(smiles)
        assert args is not None, "Failed to match fused het core"
        mol, core_atom_set, atom_mapping, ring_atom_set, chain_set, attach_ring_idx = args
        result = _detect_fused_het_inner_subs(
            mol, core_atom_set, atom_mapping, ring_atom_set, chain_set, attach_ring_idx
        )
        assert result is not None and result != ""
        assert 'propoxy' in result, f"Expected 'propoxy' in result, got: {result}"
        assert 'methoxy' not in result, f"Unexpected 'methoxy' in result, got: {result}"


@pytest.mark.integration
class TestExpandedInnerSubsEndToEnd:
    """End-to-end tests: name_compound() produces correct compound prefixes."""

    def test_e2e_cyano_indole_on_diacid(self):
        """5-cyano-1H-indole on heptanedioic acid produces '5-cyano-1H-indol-3-yl'."""
        from orthonym.namer import name_compound
        result = name_compound('OC(=O)CCC(CCC(=O)O)c1c[nH]c2cc(C#N)ccc12')
        assert result is not None
        assert 'cyano' in result, f"Expected 'cyano' in name, got: {result}"
        assert 'methyl' not in result.lower() or 'trifluoromethyl' in result.lower(), \
            f"Unexpected 'methyl' leakage in name, got: {result}"

    def test_e2e_nitro_indole_on_diacid(self):
        """5-nitro-1H-indole on heptanedioic acid produces '5-nitro-1H-indol-3-yl'."""
        from orthonym.namer import name_compound
        result = name_compound('OC(=O)CCC(CCC(=O)O)c1c[nH]c2cc([N+](=O)[O-])ccc12')
        assert result is not None
        assert 'nitro' in result, f"Expected 'nitro' in name, got: {result}"

    def test_e2e_trifluoromethyl_indole_on_diacid(self):
        """5-(trifluoromethyl)-1H-indole on diacid produces 'trifluoromethyl' in name."""
        from orthonym.namer import name_compound
        result = name_compound('OC(=O)CCC(CCC(=O)O)c1c[nH]c2cc(C(F)(F)F)ccc12')
        assert result is not None
        assert 'trifluoromethyl' in result, f"Expected 'trifluoromethyl' in name, got: {result}"
        # No trifluoro leakage on the parent chain name
        # The parent chain portion should not contain 'trifluoro' outside of the compound prefix
        # Check: 'trifluoro' should only appear as part of 'trifluoromethyl'
        import re
        trifluoro_count = len(re.findall(r'trifluoro', result))
        trifluoromethyl_count = len(re.findall(r'trifluoromethyl', result))
        assert trifluoro_count == trifluoromethyl_count, \
            f"'trifluoro' leakage on parent chain: {result}"

    def test_e2e_propyl_indole_on_diacid(self):
        """5-propyl-1H-indole on diacid produces '5-propyl-1H-indol-3-yl'."""
        from orthonym.namer import name_compound
        result = name_compound('OC(=O)CCC(CCC(=O)O)c1c[nH]c2cc(CCC)ccc12')
        assert result is not None
        assert 'propyl' in result, f"Expected 'propyl' in name, got: {result}"
        # Should NOT misclassify as methyl or ethyl
        # (propyl contains these as substrings so just check prefix form)
        assert '5-propyl' in result or 'propyl-1H-indol' in result, \
            f"Expected propyl as inner sub, got: {result}"


@pytest.mark.integration
class TestExistingSubsStillWork:
    """Regression: existing substituent types (methyl, ethyl, halogen, hydroxy, methoxy, amino) still work."""

    def test_methyl_still_works(self):
        """5-methyl-1H-indole should still produce '5-methyl-' prefix."""
        smiles = 'Cc1ccc2[nH]ccc2c1'
        args = _setup_inner_sub_test(smiles)
        assert args is not None
        mol, core_atom_set, atom_mapping, ring_atom_set, chain_set, attach_ring_idx = args
        result = _detect_fused_het_inner_subs(
            mol, core_atom_set, atom_mapping, ring_atom_set, chain_set, attach_ring_idx
        )
        assert 'methyl' in result

    def test_chloro_still_works(self):
        """6-chloro-quinoline should still produce 'chloro' prefix."""
        smiles = 'Clc1ccc2ncccc2c1'
        args = _setup_inner_sub_test(smiles)
        assert args is not None
        mol, core_atom_set, atom_mapping, ring_atom_set, chain_set, attach_ring_idx = args
        result = _detect_fused_het_inner_subs(
            mol, core_atom_set, atom_mapping, ring_atom_set, chain_set, attach_ring_idx
        )
        assert 'chloro' in result

    def test_hydroxy_still_works(self):
        """5-hydroxy-1H-indole should still produce 'hydroxy' prefix."""
        smiles = 'Oc1ccc2[nH]ccc2c1'
        args = _setup_inner_sub_test(smiles)
        assert args is not None
        mol, core_atom_set, atom_mapping, ring_atom_set, chain_set, attach_ring_idx = args
        result = _detect_fused_het_inner_subs(
            mol, core_atom_set, atom_mapping, ring_atom_set, chain_set, attach_ring_idx
        )
        assert 'hydroxy' in result
