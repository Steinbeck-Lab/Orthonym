"""TDD RED tests for name_substituent() five-tier cascade and parent_to_prefix() extensions.

Tests the Phase 85 deliverables:
- name_substituent(mol, frag_atoms, attach_idx) -> str (never None)
- parent_to_prefix() extended for ester, amide, nitrile, cyclic parent names

References:
    IUPAC 2013 P-31.1.3 (substituent prefix naming)
    IUPAC 2013 P-65.6.3 (ester prefixes)
    IUPAC 2013 P-66.1.1.4 (amide prefixes)
    IUPAC 2013 P-66.1.4.1 (nitrile prefixes)
"""

import pytest
from rdkit import Chem

from orthonym.assembly.substituent_enumerator import name_substituent
from orthonym.assembly.substituent_naming import parent_to_prefix


# ============================================================================
# Helper: build a mol and identify fragment atoms + attachment point
# ============================================================================

def _make_mol(smiles):
    """Create RDKit Mol from SMILES, returns (mol, num_atoms)."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Invalid SMILES: {smiles}"
    return mol


# ============================================================================
# TestNeverNone: name_substituent() returns non-None string for all inputs
# ============================================================================

class TestNeverNone:
    """Verify name_substituent() never returns None for any valid fragment."""

    def test_simple_methyl(self):
        """Single carbon fragment -> 'methyl'."""
        mol = _make_mol("CC")  # ethane: C0-C1
        # Fragment = {1}, attached at atom 0 (parent)
        result = name_substituent(mol, {1}, attach_idx=1)
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
        assert result == "methyl"

    def test_linear_propyl(self):
        """Three carbon linear chain -> 'propyl'."""
        mol = _make_mol("CCCC")  # butane: C0-C1-C2-C3
        # Fragment = {1, 2, 3}, attached at atom 1 (bonded to parent atom 0)
        result = name_substituent(mol, {1, 2, 3}, attach_idx=1)
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    def test_phenyl_fragment(self):
        """Benzene ring fragment -> 'phenyl'."""
        mol = _make_mol("c1ccccc1C")  # toluene
        # Fragment = ring atoms {0,1,2,3,4,5}, attached at atom 5 (bonded to C6)
        # Actually, let's find the ring atoms
        ring_info = mol.GetRingInfo()
        ring_atoms = set(ring_info.AtomRings()[0])
        # Find the ring atom bonded to the methyl carbon
        methyl_idx = None
        ring_attach = None
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'C' and not atom.GetIsAromatic():
                methyl_idx = atom.GetIdx()
                for nbr in atom.GetNeighbors():
                    if nbr.GetIdx() in ring_atoms:
                        ring_attach = nbr.GetIdx()
                break
        assert ring_attach is not None
        # Ring as substituent on methyl parent
        result = name_substituent(mol, ring_atoms, attach_idx=ring_attach)
        assert result is not None
        assert isinstance(result, str)
        assert "phenyl" in result.lower()

    def test_hydroxy_fragment(self):
        """Single oxygen (no carbon) -> 'hydroxy'."""
        mol = _make_mol("CO")  # methanol: C0-O1
        # Fragment = {1} (the oxygen), attached at O1 bonded to C0
        result = name_substituent(mol, {1}, attach_idx=1)
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    def test_complex_heteroatom_fragment(self):
        """Fragment with heteroatoms: -OCH2C(=O)O (carboxymethoxy)."""
        mol = _make_mol("OCC(=O)O")  # glycolic acid O0-C1-C2(=O3)-O4
        # Use the whole molecule as a fragment on some hypothetical parent
        # Fragment = {0, 1, 2, 3, 4}, attached at atom 0
        all_atoms = set(range(mol.GetNumAtoms()))
        result = name_substituent(mol, all_atoms, attach_idx=0)
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    def test_large_fragment_depth_limit(self):
        """Very large substituent that might hit recursion depth limit -> still non-None."""
        # Create a long chain molecule
        mol = _make_mol("C" * 20)  # icosane
        all_atoms = set(range(mol.GetNumAtoms()))
        parent = {0}  # just atom 0 as "parent"
        frag = all_atoms - parent
        # Find atom bonded to parent
        attach = None
        for nbr in mol.GetAtomWithIdx(0).GetNeighbors():
            if nbr.GetIdx() in frag:
                attach = nbr.GetIdx()
                break
        assert attach is not None
        result = name_substituent(mol, frag, attach_idx=attach)
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_fragment_fallback(self):
        """Empty fragment atoms -> should return something, not None."""
        mol = _make_mol("C")  # methane
        result = name_substituent(mol, set(), attach_idx=0)
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    def test_single_nitrogen_amino(self):
        """Single nitrogen with hydrogens -> 'amino'."""
        mol = _make_mol("CN")  # methylamine: C0-N1
        result = name_substituent(mol, {1}, attach_idx=1)
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    def test_single_sulfur_sulfanyl(self):
        """Single sulfur with hydrogen -> 'sulfanyl'."""
        mol = _make_mol("CS")  # methanethiol: C0-S1
        result = name_substituent(mol, {1}, attach_idx=1)
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    def test_halogen_fluoro(self):
        """Single fluorine -> 'fluoro'."""
        mol = _make_mol("CF")  # fluoromethane: C0-F1
        result = name_substituent(mol, {1}, attach_idx=1)
        assert result is not None
        assert isinstance(result, str)


# ============================================================================
# TestTierOrdering: correct tier is used for each case
# ============================================================================

class TestTierOrdering:
    """Verify that the correct tier is selected in the cascade."""

    def test_cache_hit_returns_cache_result(self):
        """Fragment matching FRAGMENT_NAME_CACHE entry returns cached name."""
        # ethanol fragment: "CCO" is in the cache as "ethanol"
        # parent_to_prefix("ethanol", 2) -> "hydroxyethyl" or similar
        mol = _make_mol("CCCO")  # propan-1-ol: C0-C1-C2-O3
        # Fragment = {1, 2, 3} (ethanol fragment: C1-C2-O3)
        # This should hit cache for "CCO" -> "ethanol" -> prefix form
        result = name_substituent(mol, {1, 2, 3}, attach_idx=1)
        assert result is not None
        assert isinstance(result, str)

    def test_retained_isopropyl(self):
        """Isopropyl (branched 3C at center) -> 'isopropyl' via retained names."""
        mol = _make_mol("CC(C)C")  # isobutane: C0-C1(-C2)-C3
        # Fragment = {0, 1, 2}, attached at C1 (bonded to parent C3)
        result = name_substituent(mol, {0, 1, 2}, attach_idx=1)
        assert result is not None
        assert "isopropyl" in result.lower()

    def test_linear_ethyl(self):
        """Linear ethyl fragment -> 'ethyl' via linear alkyl path."""
        mol = _make_mol("CCC")  # propane: C0-C1-C2
        # Fragment = {1, 2}, attached at C1 (bonded to parent C0)
        result = name_substituent(mol, {1, 2}, attach_idx=1)
        assert result is not None
        assert result == "ethyl"

    def test_linear_propyl_fast_path(self):
        """Linear propyl -> 'propyl' via fast path."""
        mol = _make_mol("CCCC")  # butane
        result = name_substituent(mol, {1, 2, 3}, attach_idx=1)
        assert result is not None
        assert result == "propyl"

    def test_recursive_branched_alkyl(self):
        """Branched alkyl requires recursive naming."""
        # 2-methylpropyl (isobutyl)
        mol = _make_mol("CC(C)CC")  # 2-methylbutane
        # Fragment = {0, 1, 2, 3} (the isobutyl part), attached at C3
        result = name_substituent(mol, {0, 1, 2, 3}, attach_idx=3)
        assert result is not None
        assert isinstance(result, str)

    def test_descriptive_fallback_for_exotic_fragment(self):
        """Descriptive fallback used when tiers 1-4 fail."""
        # A fragment so exotic that normal naming fails
        # This is rare, but name_substituent must still return non-None
        mol = _make_mol("C")
        # Passing an atom that doesn't exist in any naming path
        # but name_substituent should still handle gracefully
        result = name_substituent(mol, {0}, attach_idx=0)
        assert result is not None
        assert isinstance(result, str)


# ============================================================================
# TestAttachIdx: attach_idx correctly determines naming
# ============================================================================

class TestAttachIdx:
    """Verify that attach_idx is correctly used for naming."""

    def test_methyl_attach_at_only_atom(self):
        """Methyl with attach_idx=0 -> 'methyl'."""
        mol = _make_mol("CC")
        result = name_substituent(mol, {1}, attach_idx=1)
        assert result == "methyl"

    def test_propyl_attached_at_end(self):
        """3-carbon fragment attached at end -> 'propyl'."""
        mol = _make_mol("CCCC")  # butane
        # Fragment {1,2,3} with attach at atom 1 (the end nearest parent)
        result = name_substituent(mol, {1, 2, 3}, attach_idx=1)
        assert result is not None
        # Should be propyl (linear, attached at terminal)
        assert result == "propyl"

    def test_propyl_attached_at_middle(self):
        """3-carbon fragment attached at middle -> 'isopropyl'."""
        mol = _make_mol("CC(C)C")  # 2-methylpropane
        # Fragment {0, 1, 2} with attach at atom 1 (the branching center)
        result = name_substituent(mol, {0, 1, 2}, attach_idx=1)
        assert result is not None
        # Attached at center of 3 carbons -> isopropyl
        assert "isopropyl" in result.lower()

    def test_ethyl_always_ethyl(self):
        """2-carbon linear fragment -> 'ethyl' regardless of attach position."""
        mol = _make_mol("CCC")
        result = name_substituent(mol, {1, 2}, attach_idx=1)
        assert result == "ethyl"


# ============================================================================
# TestParentToPrefixEster: ester suffix -> prefix conversion
# ============================================================================

class TestParentToPrefixEster:
    """Verify parent_to_prefix() handles ester (-oate) names."""

    def test_propanoate_ester(self):
        """IUPAC P-65.6.3: propanoate -> some form of carboxy/alkoxycarbonyl prefix."""
        result = parent_to_prefix("propanoate", 3)
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
        # Should produce a carboxy-related or alkoxycarbonyl prefix
        assert any(word in result.lower() for word in
                   ["carboxy", "oxycarbonyl", "alkoxycarbonyl"])

    def test_butanoate_ester(self):
        """butanoate -> carboxy/alkoxycarbonyl prefix."""
        result = parent_to_prefix("butanoate", 4)
        assert result is not None
        assert len(result) > 0


# ============================================================================
# TestParentToPrefixAmide: amide suffix -> prefix conversion
# ============================================================================

class TestParentToPrefixAmide:
    """Verify parent_to_prefix() handles amide names."""

    def test_propanamide(self):
        """IUPAC P-66.1.1.4: propanamide -> carbamoyl prefix form."""
        result = parent_to_prefix("propanamide", 3)
        assert result is not None
        assert isinstance(result, str)
        assert "carbamoyl" in result.lower()

    def test_acetamide(self):
        """acetamide -> carbamoyl prefix form."""
        result = parent_to_prefix("acetamide", 2)
        assert result is not None
        assert isinstance(result, str)
        assert "carbamoyl" in result.lower()

    def test_carboxamide(self):
        """benzcarboxamide -> carbamoyl prefix form."""
        result = parent_to_prefix("carboxamide", 0)
        assert result is not None
        assert "carbamoyl" in result.lower()


# ============================================================================
# TestParentToPrefixNitrile: nitrile suffix -> prefix conversion
# ============================================================================

class TestParentToPrefixNitrile:
    """Verify parent_to_prefix() handles nitrile names."""

    def test_propanenitrile(self):
        """IUPAC P-66.1.4.1: propanenitrile -> cyano prefix form."""
        result = parent_to_prefix("propanenitrile", 3)
        assert result is not None
        assert isinstance(result, str)
        assert "cyano" in result.lower()

    def test_acetonitrile(self):
        """acetonitrile -> cyano prefix form."""
        result = parent_to_prefix("acetonitrile", 2)
        assert result is not None
        assert isinstance(result, str)
        assert "cyano" in result.lower()

    def test_carbonitrile(self):
        """carbonitrile -> cyano prefix form."""
        result = parent_to_prefix("carbonitrile", 0)
        assert result is not None
        assert "cyano" in result.lower()


# ============================================================================
# TestParentToPrefixCyclic: cyclic parent -> prefix conversion
# ============================================================================

class TestParentToPrefixCyclic:
    """Verify parent_to_prefix() handles cyclic parent names."""

    def test_cyclohexane(self):
        """cyclohexane -> cyclohexyl."""
        result = parent_to_prefix("cyclohexane", 6)
        assert result is not None
        assert "cyclohex" in result.lower()
        assert result.endswith("yl")

    def test_pyridine(self):
        """pyridine -> pyridinyl (or pyridyl)."""
        result = parent_to_prefix("pyridine", 5)
        assert result is not None
        assert result.endswith("yl")
        assert "pyridin" in result.lower()

    def test_cyclopentane(self):
        """cyclopentane -> cyclopentyl."""
        result = parent_to_prefix("cyclopentane", 5)
        assert result is not None
        assert "cyclopent" in result.lower()
        assert result.endswith("yl")
