"""TDD RED tests for name_substituent five-tier cascade and parent_to_prefix(, attach_locant=ATTACH_LOCANT_UNKNOWN) extensions.

Tests the a phase deliverables:
- name_substituent(mol, frag_atoms, attach_idx) -> str (never None)
- parent_to_prefix(, attach_locant=ATTACH_LOCANT_UNKNOWN) extended for ester, amide, nitrile, cyclic parent names

References:
    IUPAC 2013 (substituent prefix naming)
    IUPAC 2013 (ester prefixes)
    IUPAC 2013 (amide prefixes)
    IUPAC 2013 (nitrile prefixes)
"""

import pytest
from rdkit import Chem

from orthonym.assembly.substituent_enumerator import name_substituent
from orthonym.assembly.substituent_naming import (
    ATTACH_LOCANT_UNKNOWN, parent_to_prefix)


# ============================================================================
# Helper: build a mol and identify fragment atoms + attachment point
# ============================================================================

def _make_mol(smiles):
    """Create RDKit Mol from SMILES, returns (mol, num_atoms)."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Invalid SMILES: {smiles}"
    return mol


# ============================================================================
# TestNeverNone: name_substituent returns non-None string for all inputs
# ============================================================================

class TestNeverNone:
    """Verify name_substituent never returns None for any valid fragment."""

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

    def test_retained_isopropyl(self):
        """Branched 3C at the centre -> 'propan-2-yl' (F-T9/DD6: the PIN is the
        located form; 'isopropyl' is general-only and no longer emitted)."""
        mol = _make_mol("CC(C)C")  # isobutane: C0-C1(-C2)-C3
        # Fragment = {0, 1, 2}, attached at C1 (bonded to parent C3)
        result = name_substituent(mol, {0, 1, 2}, attach_idx=1)
        assert result is not None
        assert "propan-2-yl" in result.lower()

    def test_cache_hit_returns_cache_result(self):
        """Fragment matching FRAGMENT_NAME_CACHE entry returns cached name (tier 2)."""
        # ethanol fragment: "CCO" is in the cache as "ethanol"
        # parent_to_prefix("ethanol", 2, attach_locant=ATTACH_LOCANT_UNKNOWN) -> "hydroxyethyl" or similar
        mol = _make_mol("CCCO")  # propan-1-ol: C0-C1-C2-O3
        # Fragment = {1, 2, 3} (ethanol fragment: C1-C2-O3)
        # This should hit cache for "CCO" -> "ethanol" -> prefix form
        result = name_substituent(mol, {1, 2, 3}, attach_idx=1)
        assert result is not None
        assert isinstance(result, str)

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
        """3-carbon fragment attached at middle -> 'propan-2-yl' (F-T9/DD6)."""
        mol = _make_mol("CC(C)C")  # 2-methylpropane
        # Fragment {0, 1, 2} with attach at atom 1 (the branching center)
        result = name_substituent(mol, {0, 1, 2}, attach_idx=1)
        assert result is not None
        # Attached at centre of 3 carbons -> propan-2-yl (located PIN, not 'isopropyl')
        assert "propan-2-yl" in result.lower()

    def test_ethyl_always_ethyl(self):
        """2-carbon linear fragment -> 'ethyl' regardless of attach position."""
        mol = _make_mol("CCC")
        result = name_substituent(mol, {1, 2}, attach_idx=1)
        assert result == "ethyl"


# ============================================================================
# TestParentToPrefixEster: ester suffix -> prefix conversion
# ============================================================================

class TestParentToPrefixEster:
    """Verify parent_to_prefix handles ester (-oate) names."""

    def test_ethanoate_ester_one_position_stem(self):
        """ +: a one-position stem needs no locant, so the
        carbon count cannot mis-place one. 'ethanoate' -> 'carboxymethyl'."""
        result = parent_to_prefix("ethanoate", 2, attach_locant=ATTACH_LOCANT_UNKNOWN)
        assert result == "carboxymethyl"

    @pytest.mark.parametrize("parent,clen", [("propanoate", 3), ("butanoate", 4)])
    def test_longer_esters_decline_the_count_derived_locant(self, parent, clen):
        """ residue Task A: the carboxy locant was read off the whole-fragment
        carbon COUNT, which is not a proof of the fragment's shape."""
        assert parent_to_prefix(
            parent, clen, attach_locant=ATTACH_LOCANT_UNKNOWN) is None


# ============================================================================
# TestParentToPrefixAmide: amide suffix -> prefix conversion
# ============================================================================

class TestParentToPrefixAmide:
    """Verify parent_to_prefix handles amide names."""

    def test_propanamide_declines_the_count_derived_locant(self):
        """ residue Task A -- see test_longer_esters_decline... above.
        The whole-molecule name is unaffected where the chain is proven."""
        assert parent_to_prefix(
            "propanamide", 3, attach_locant=ATTACH_LOCANT_UNKNOWN) is None

    def test_acetamide(self):
        """acetamide -> the method (1) prefix of (the Blue Book,
        "method (1) is preferred for chains"): '5-(2-amino-2-oxoethyl)furan-2-carboxylic
        acid (PIN)' (:32940), not '(carbamoylmethyl)' (:32941)."""
        result = parent_to_prefix("acetamide", 2, attach_locant=ATTACH_LOCANT_UNKNOWN)
        assert result == "2-amino-2-oxoethyl"

    def test_carboxamide(self):
        """benzcarboxamide -> carbamoyl prefix form."""
        result = parent_to_prefix("carboxamide", 0, attach_locant=ATTACH_LOCANT_UNKNOWN)
        assert result is not None
        assert "carbamoyl" in result.lower()


# ============================================================================
# TestParentToPrefixNitrile: nitrile suffix -> prefix conversion
# ============================================================================

class TestParentToPrefixNitrile:
    """Verify parent_to_prefix handles nitrile names."""

    def test_propanenitrile_declines_the_count_derived_locant(self):
        """ residue Task A -- see test_longer_esters_decline... above."""
        assert parent_to_prefix(
            "propanenitrile", 3, attach_locant=ATTACH_LOCANT_UNKNOWN) is None

    def test_acetonitrile(self):
        """acetonitrile -> cyano prefix form."""
        result = parent_to_prefix("acetonitrile", 2, attach_locant=ATTACH_LOCANT_UNKNOWN)
        assert result is not None
        assert isinstance(result, str)
        assert "cyano" in result.lower()

    def test_carbonitrile(self):
        """carbonitrile -> cyano prefix form."""
        result = parent_to_prefix("carbonitrile", 0, attach_locant=ATTACH_LOCANT_UNKNOWN)
        assert result is not None
        assert "cyano" in result.lower()


# ============================================================================
# TestParentToPrefixCyclic: cyclic parent -> prefix conversion
# ============================================================================

class TestParentToPrefixCyclic:
    """Verify parent_to_prefix handles cyclic parent names."""

    def test_cyclohexane(self):
        """cyclohexane -> cyclohexyl."""
        result = parent_to_prefix("cyclohexane", 6, attach_locant=ATTACH_LOCANT_UNKNOWN)
        assert result is not None
        assert "cyclohex" in result.lower()
        assert result.endswith("yl")

    def test_pyridine(self):
        """pyridine -> pyridinyl (or pyridyl)."""
        result = parent_to_prefix("pyridine", 5, attach_locant=ATTACH_LOCANT_UNKNOWN)
        assert result is not None
        assert result.endswith("yl")
        assert "pyridin" in result.lower()

    def test_cyclopentane(self):
        """cyclopentane -> cyclopentyl."""
        result = parent_to_prefix("cyclopentane", 5, attach_locant=ATTACH_LOCANT_UNKNOWN)
        assert result is not None
        assert "cyclopent" in result.lower()
        assert result.endswith("yl")


# ============================================================================
# TestFusedRingSubstituentFallback: verify gap closure in fused_rings.py
# ============================================================================

class TestFusedRingSubstituentFallback:
    """Verify _identify_fused_substituent no longer returns None for simple C subs."""

    def test_bfs_collect_all_basic(self):
        """_bfs_collect_all collects all reachable atoms excluding excluded set."""
        from orthonym.rules.fused_rings import _bfs_collect_all
        mol = _make_mol("CCCC")  # butane: C0-C1-C2-C3
        # Collect from atom 1, excluding atom 0
        result = _bfs_collect_all(mol, 1, {0})
        assert result == {1, 2, 3}

    def test_bfs_collect_all_with_heteroatoms(self):
        """_bfs_collect_all traverses through heteroatoms."""
        from orthonym.rules.fused_rings import _bfs_collect_all
        mol = _make_mol("CCNCO")  # C0-C1-N2-C3-O4
        result = _bfs_collect_all(mol, 1, {0})
        assert result == {1, 2, 3, 4}

    def test_bfs_collect_all_empty_excluded(self):
        """_bfs_collect_all with no excluded atoms collects everything."""
        from orthonym.rules.fused_rings import _bfs_collect_all
        mol = _make_mol("CCC")
        result = _bfs_collect_all(mol, 0, set())
        assert result == {0, 1, 2}

    def test_fallback_names_propyl_on_naphthalene(self):
        """Propyl attached to naphthalene should get a name via fallback."""
        from orthonym.rules.fused_rings import _identify_fused_substituent
        mol = _make_mol("CCCc1ccc2ccccc2c1")
        ri = mol.GetRingInfo()
        core = set()
        for ring in ri.AtomRings():
            core.update(ring)
        # Atom 0 is the first carbon of propyl chain
        if 0 not in core:
            result = _identify_fused_substituent(mol, 0, core)
            assert result is not None
            assert 'name' in result
            assert 'propyl' in result['name'].lower()


# ============================================================================
# TestRingNameTokensReplacement: verify structural check in composer.py
# ============================================================================

class TestRingNameTokensReplacement:
    """Verify _RING_NAME_TOKENS replaced with structural check."""

    def test_has_ring_atoms_helper_ring(self):
        """_has_ring_atoms returns True for ring atoms."""
        from orthonym.assembly.composer import _has_ring_atoms
        mol = _make_mol("c1ccccc1C")
        ring_atoms = set(mol.GetRingInfo().AtomRings()[0])
        assert _has_ring_atoms(mol, ring_atoms) is True

    def test_has_ring_atoms_helper_non_ring(self):
        """_has_ring_atoms returns False for non-ring atoms."""
        from orthonym.assembly.composer import _has_ring_atoms
        mol = _make_mol("c1ccccc1C")
        # Find the methyl carbon (not in ring)
        methyl_idx = None
        for atom in mol.GetAtoms():
            if not atom.IsInRing():
                methyl_idx = atom.GetIdx()
                break
        assert methyl_idx is not None
        assert _has_ring_atoms(mol, {methyl_idx}) is False

    def test_name_reflects_ring_positive(self):
        """_name_reflects_ring detects ring tokens."""
        from orthonym.assembly.composer import _name_reflects_ring
        assert _name_reflects_ring("cyclohexyl") is True
        assert _name_reflects_ring("phenyl") is True
        assert _name_reflects_ring("piperidinyl") is True
        assert _name_reflects_ring("morpholinyl") is True
        assert _name_reflects_ring("indolyl") is True

    def test_name_reflects_ring_negative(self):
        """_name_reflects_ring rejects non-ring names."""
        from orthonym.assembly.composer import _name_reflects_ring
        assert _name_reflects_ring("methyl") is False
        assert _name_reflects_ring("ethyl") is False
        assert _name_reflects_ring("propyl") is False
        assert _name_reflects_ring("butyl") is False
        assert _name_reflects_ring("hydroxy") is False


class TestWR06MultiStereocenterSubstituentLocants:
    """ (code review 2026-06-02) — TRACKING xfail for the known
    multi-stereocenter substituent locant defect.

    A substituent that itself carries >=2 stereocenters currently gets a
    double-applied, atom-index-numbered (heteroatom-counting) descriptor, e.g.
    '(2R,3R)-(2R)-2-bromochloropropyl'. The proper fix threads the substituent's
    OWN IUPAC numbering out of name_fragment_recursively and de-duplicates the
    recursive stereo descriptor — high blast radius, ~0 corpus reach, and the
    malformed output is already OPSIN-gated by in production. This xfail
    pins the defect; it flips green when the root-cause fix lands.
    """

    @pytest.mark.xfail(
        reason="WR-06 DEFERRED (169.5 code review): multi-stereocenter substituent "
               "uses raw atom-index locants AND double-applies the recursive stereo "
               "descriptor. Correct fix needs the substituent's own numbering threaded "
               "out of name_fragment_recursively (recursion-contract change, high blast "
               "radius; branch fires ~25/1282 canary rows that currently pass, so a "
               "behavior change risks regression; malformed cases SUB-03-gated). "
               "Tracked here + in the V20 audit.",
        strict=False,
    )
    def test_multi_stereocenter_substituent_single_attachment_rooted_descriptor(self):
        from rdkit.Chem import rdCIPLabeler
        from orthonym.assembly.substituent_naming import name_substituent_fragment
        # benzene bearing -CH(Cl)CH(Br)CH3 (two stereocenters in the substituent)
        mol = Chem.MolFromSmiles("Cl[C@@H]([C@H](Br)C)c1ccccc1")
        rdCIPLabeler.AssignCIPLabels(mol)
        sub_atoms = [a.GetIdx() for a in mol.GetAtoms()
                     if not a.GetIsAromatic() and a.GetSymbol() != 'H']
        attach = next(a.GetIdx() for a in mol.GetAtoms()
                      if a.GetSymbol() == 'C' and not a.GetIsAromatic()
                      and any(nb.GetIsAromatic() for nb in a.GetNeighbors()))
        ring = [a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()]
        name = name_substituent_fragment(mol, sub_atoms, attach, ring)
        # Correctness property: exactly ONE leading stereo descriptor (no double
        # application), and attachment-rooted numbering starts at locant 1.
        assert name.count(")-") == 1, f"double/garbled descriptor: {name!r}"
        assert "(1" in name, f"numbering not attachment-rooted: {name!r}"
