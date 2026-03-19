"""
Unit tests for axial chirality detection and Ra/Sa descriptor formatting.

Tests the detect_axial_chirality() function in perception/stereo.py for
atropisomeric biaryls (STEREOATROPCW/CCW) and allenes (CHI_ALLENE), plus
the integration of Ra/Sa descriptors into format_stereodescriptor_string().

IUPAC Reference: P-93.5 (axial chirality descriptors)
RDKit: STEREOATROPCW/CCW for atropisomers, CHI_ALLENE for allenes
Mapping: RDKit CIP P -> Ra, RDKit CIP M -> Sa
"""

import pytest
from rdkit import Chem
from rdkit.Chem import rdCIPLabeler

from src.orthonym.perception.stereo import (
    detect_axial_chirality,
    _manual_allene_cip,
)
from src.orthonym.rules.stereochemistry import (
    collect_stereodescriptors,
    format_stereodescriptor_string,
)


# =============================================================================
# Helpers for constructing test molecules
# =============================================================================

def _make_biaryl_atropisomer(stereo_type):
    """Create a biaryl molecule with specified atropisomer stereo.

    Uses 2,2'-dimethylbiphenyl (Cc1ccccc1-c1ccccc1C) as the scaffold.
    Sets the specified stereo on the biaryl single bond connecting
    the two aromatic rings.

    Args:
        stereo_type: Chem.BondStereo.STEREOATROPCW or STEREOATROPCCW

    Returns:
        RDKit Mol object with atropisomer stereo set and CIP assigned.
    """
    mol = Chem.RWMol(Chem.MolFromSmiles('Cc1ccccc1-c1ccccc1C'))
    for bond in mol.GetBonds():
        b = bond.GetBeginAtom()
        e = bond.GetEndAtom()
        if (b.IsInRing() and e.IsInRing() and not bond.IsInRing()
                and b.GetIsAromatic() and e.GetIsAromatic()):
            bond.SetStereo(stereo_type)
            b_nbrs = [n.GetIdx() for n in b.GetNeighbors()
                       if n.GetIdx() != e.GetIdx()]
            e_nbrs = [n.GetIdx() for n in e.GetNeighbors()
                       if n.GetIdx() != b.GetIdx()]
            bond.SetStereoAtoms(b_nbrs[0], e_nbrs[0])
            break
    final_mol = mol.GetMol()
    rdCIPLabeler.AssignCIPLabels(final_mol)
    return final_mol


def _make_allene_with_chi_allene():
    """Create a chiral allene with CHI_ALLENE tag on central carbon.

    Uses ClC=C=CBr -- 4 distinct terminal substituents (Cl, H on one
    end; Br, H on other end) making the allene chiral.

    Returns:
        RDKit Mol object with CHI_ALLENE set on central C.
    """
    mol = Chem.RWMol(Chem.MolFromSmiles('ClC=C=CBr'))
    # Find the central allene carbon (has 2 double bonds)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'C':
            dbl_count = sum(
                1 for b in atom.GetBonds()
                if b.GetBondType() == Chem.BondType.DOUBLE
            )
            if dbl_count == 2:
                atom.SetChiralTag(Chem.ChiralType.CHI_ALLENE)
                break
    return mol.GetMol()


# =============================================================================
# Test detect_axial_chirality -- no axial chirality
# =============================================================================

class TestDetectAxialChiralityNone:
    """Tests for molecules without axial chirality."""

    def test_no_axial_chirality_ethanol(self):
        """Ethanol (CCO) has no axial chirality -- returns empty list."""
        mol = Chem.MolFromSmiles('CCO')
        rdCIPLabeler.AssignCIPLabels(mol)
        result = detect_axial_chirality(mol)
        assert result == []

    def test_no_axial_chirality_biphenyl_without_stereo(self):
        """Plain biphenyl without atropisomer stereo set returns empty list."""
        mol = Chem.MolFromSmiles('c1ccc(-c2ccccc2)cc1')
        rdCIPLabeler.AssignCIPLabels(mol)
        result = detect_axial_chirality(mol)
        assert result == []

    def test_no_axial_chirality_allene_without_tag(self):
        """Allene without CHI_ALLENE tag returns empty list (stereo not encoded)."""
        mol = Chem.MolFromSmiles('CC=C=CC')
        rdCIPLabeler.AssignCIPLabels(mol)
        # No CHI_ALLENE tag set -- should return empty
        result = detect_axial_chirality(mol)
        assert result == []


# =============================================================================
# Test detect_axial_chirality -- atropisomers
# =============================================================================

class TestDetectAxialChiralityAtropisomer:
    """Tests for atropisomeric biaryl detection."""

    def test_stereoatropcw_returns_sa(self):
        """STEREOATROPCW bond -> CIP M -> IUPAC Sa."""
        mol = _make_biaryl_atropisomer(Chem.BondStereo.STEREOATROPCW)
        result = detect_axial_chirality(mol)
        assert len(result) == 1
        entry = result[0]
        assert entry['type'] == 'atropisomer'
        assert entry['cip'] == 'Sa'
        assert entry['locant_atom'] is not None

    def test_stereoatropccw_returns_ra(self):
        """STEREOATROPCCW bond -> CIP P -> IUPAC Ra."""
        mol = _make_biaryl_atropisomer(Chem.BondStereo.STEREOATROPCCW)
        result = detect_axial_chirality(mol)
        assert len(result) == 1
        entry = result[0]
        assert entry['type'] == 'atropisomer'
        assert entry['cip'] == 'Ra'
        assert entry['locant_atom'] is not None

    def test_atropisomer_has_correct_keys(self):
        """Atropisomer result dict has all required keys."""
        mol = _make_biaryl_atropisomer(Chem.BondStereo.STEREOATROPCW)
        result = detect_axial_chirality(mol)
        assert len(result) == 1
        entry = result[0]
        assert 'type' in entry
        assert 'idx' in entry
        assert 'cip' in entry
        assert 'locant_atom' in entry

    def test_molecule_with_point_and_atropisomer_chirality(self):
        """Molecule with both R/S stereocenters and atropisomer chirality.

        detect_axial_chirality should ONLY return the atropisomer element;
        point chirality (R/S) is handled separately by get_stereocenters().
        """
        # Create a biaryl with a stereocenter: use a chiral center on one ring
        mol = Chem.RWMol(Chem.MolFromSmiles('C[C@H](O)c1ccccc1-c1ccccc1C'))
        # Find the biaryl bond and set atropisomer stereo
        for bond in mol.GetBonds():
            b = bond.GetBeginAtom()
            e = bond.GetEndAtom()
            if (b.IsInRing() and e.IsInRing() and not bond.IsInRing()
                    and b.GetIsAromatic() and e.GetIsAromatic()):
                bond.SetStereo(Chem.BondStereo.STEREOATROPCW)
                b_nbrs = [n.GetIdx() for n in b.GetNeighbors()
                           if n.GetIdx() != e.GetIdx()]
                e_nbrs = [n.GetIdx() for n in e.GetNeighbors()
                           if n.GetIdx() != b.GetIdx()]
                bond.SetStereoAtoms(b_nbrs[0], e_nbrs[0])
                break

        final_mol = mol.GetMol()
        rdCIPLabeler.AssignCIPLabels(final_mol)

        result = detect_axial_chirality(final_mol)
        # Should only return atropisomer entries, not point chirality
        assert all(entry['type'] == 'atropisomer' for entry in result)
        assert len(result) >= 1


# =============================================================================
# Test detect_axial_chirality -- allenes
# =============================================================================

class TestDetectAxialChiralityAllene:
    """Tests for allene axial chirality detection."""

    def test_allene_with_chi_allene_detected(self):
        """Allene with CHI_ALLENE tag is detected with type='allene'."""
        mol = _make_allene_with_chi_allene()
        result = detect_axial_chirality(mol)
        assert len(result) == 1
        entry = result[0]
        assert entry['type'] == 'allene'
        assert entry['cip'] is not None  # manual CIP should determine Ra or Sa

    def test_allene_returns_ra_or_sa(self):
        """Allene CIP is either 'Ra' or 'Sa' (not None, R, S, P, M)."""
        mol = _make_allene_with_chi_allene()
        result = detect_axial_chirality(mol)
        assert len(result) == 1
        assert result[0]['cip'] in ('Ra', 'Sa')


# =============================================================================
# Test _manual_allene_cip
# =============================================================================

class TestManualAlleneCIP:
    """Tests for manual allene CIP determination."""

    def test_manual_cip_returns_ra_or_sa(self):
        """Manual CIP for chiral allene returns 'Ra' or 'Sa'."""
        mol = _make_allene_with_chi_allene()
        # Find the central allene C (has CHI_ALLENE)
        central_idx = None
        for atom in mol.GetAtoms():
            if atom.GetChiralTag() == Chem.ChiralType.CHI_ALLENE:
                central_idx = atom.GetIdx()
                break
        assert central_idx is not None
        result = _manual_allene_cip(mol, central_idx)
        assert result in ('Ra', 'Sa')

    def test_manual_cip_symmetrical_allene_returns_none(self):
        """Symmetrical allene (identical terminal groups) -> None (achiral)."""
        # H2C=C=CH2 (propadiene) -- both terminals have 2 H
        mol = Chem.RWMol(Chem.MolFromSmiles('C=C=C'))
        # Set CHI_ALLENE on central C
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'C':
                dbl_count = sum(
                    1 for b in atom.GetBonds()
                    if b.GetBondType() == Chem.BondType.DOUBLE
                )
                if dbl_count == 2:
                    atom.SetChiralTag(Chem.ChiralType.CHI_ALLENE)
                    central_idx = atom.GetIdx()
                    break
        final_mol = mol.GetMol()
        result = _manual_allene_cip(final_mol, central_idx)
        assert result is None

    def test_manual_cip_four_distinct_groups(self):
        """Allene with 4 completely distinct terminal groups returns Ra or Sa."""
        # FC=C=CCl -- F,H on one end; Cl,H on other
        mol = Chem.RWMol(Chem.MolFromSmiles('FC=C=CCl'))
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'C':
                dbl_count = sum(
                    1 for b in atom.GetBonds()
                    if b.GetBondType() == Chem.BondType.DOUBLE
                )
                if dbl_count == 2:
                    atom.SetChiralTag(Chem.ChiralType.CHI_ALLENE)
                    central_idx = atom.GetIdx()
                    break
        final_mol = mol.GetMol()
        result = _manual_allene_cip(final_mol, central_idx)
        assert result in ('Ra', 'Sa')


# =============================================================================
# Test format_stereodescriptor_string with Ra/Sa codes
# =============================================================================

class TestFormatWithAxialDescriptors:
    """Tests for format_stereodescriptor_string handling multi-char CIP codes."""

    def test_single_ra_descriptor(self):
        """[(1, 'Ra')] -> '(1Ra)-'"""
        result = format_stereodescriptor_string([(1, 'Ra')])
        assert result == '(1Ra)-'

    def test_single_sa_descriptor(self):
        """[(3, 'Sa')] -> '(3Sa)-'"""
        result = format_stereodescriptor_string([(3, 'Sa')])
        assert result == '(3Sa)-'

    def test_mixed_ra_and_s(self):
        """[(1, 'Ra'), (2, 'S')] -> '(1Ra,2S)-'"""
        result = format_stereodescriptor_string([(1, 'Ra'), (2, 'S')])
        assert result == '(1Ra,2S)-'

    def test_mixed_r_and_sa(self):
        """[(1, 'R'), (3, 'Sa')] -> '(1R,3Sa)-'"""
        result = format_stereodescriptor_string([(1, 'R'), (3, 'Sa')])
        assert result == '(1R,3Sa)-'

    def test_all_stereo_types_combined(self):
        """[(1, 'Ra'), (2, 'S'), (3, 'E')] -> '(1Ra,2S,3E)-'"""
        result = format_stereodescriptor_string([(1, 'Ra'), (2, 'S'), (3, 'E')])
        assert result == '(1Ra,2S,3E)-'


# =============================================================================
# Test collect_stereodescriptors with axial chirality integration
# =============================================================================

class TestCollectStereodescriptorsAxial:
    """Tests for axial chirality flowing through collect_stereodescriptors."""

    def test_atropisomer_in_atom_to_locant_returns_descriptor(self):
        """Atropisomer bond atom in atom_to_locant -> Ra/Sa included."""
        mol = _make_biaryl_atropisomer(Chem.BondStereo.STEREOATROPCW)
        # Find the biaryl bond begin atom index
        begin_atom = None
        for bond in mol.GetBonds():
            if bond.GetStereo() == Chem.BondStereo.STEREOATROPCW:
                begin_atom = bond.GetBeginAtomIdx()
                break
        assert begin_atom is not None

        # Create atom_to_locant including the begin atom of the atropisomer bond
        # Map all ring atoms to sequential locants
        atom_to_locant = {}
        for i, atom in enumerate(mol.GetAtoms()):
            atom_to_locant[i] = i + 1

        descriptors = collect_stereodescriptors(mol, atom_to_locant)
        # Should contain an Sa descriptor (STEREOATROPCW -> M -> Sa)
        axial_descs = [(loc, cip) for loc, cip in descriptors if cip in ('Ra', 'Sa')]
        assert len(axial_descs) >= 1
        assert axial_descs[0][1] == 'Sa'

    def test_atropisomer_not_in_locant_map_filtered_out(self):
        """Atropisomer bond atom NOT in atom_to_locant -> filtered out."""
        mol = _make_biaryl_atropisomer(Chem.BondStereo.STEREOATROPCW)
        # Empty locant map -- axial chirality should be filtered out
        atom_to_locant = {}
        descriptors = collect_stereodescriptors(mol, atom_to_locant)
        axial_descs = [(loc, cip) for loc, cip in descriptors if cip in ('Ra', 'Sa')]
        assert len(axial_descs) == 0

    def test_point_and_axial_chirality_both_collected(self):
        """Molecule with R/S + atropisomer chirality gets all descriptors."""
        # Build biaryl with stereocenter: C[C@H](O)c1ccccc1-c1ccccc1C
        mol = Chem.RWMol(Chem.MolFromSmiles('C[C@H](O)c1ccccc1-c1ccccc1C'))
        for bond in mol.GetBonds():
            b = bond.GetBeginAtom()
            e = bond.GetEndAtom()
            if (b.IsInRing() and e.IsInRing() and not bond.IsInRing()
                    and b.GetIsAromatic() and e.GetIsAromatic()):
                bond.SetStereo(Chem.BondStereo.STEREOATROPCCW)
                b_nbrs = [n.GetIdx() for n in b.GetNeighbors()
                           if n.GetIdx() != e.GetIdx()]
                e_nbrs = [n.GetIdx() for n in e.GetNeighbors()
                           if n.GetIdx() != b.GetIdx()]
                bond.SetStereoAtoms(b_nbrs[0], e_nbrs[0])
                break
        final_mol = mol.GetMol()
        rdCIPLabeler.AssignCIPLabels(final_mol)

        # Map all atoms to locants
        atom_to_locant = {i: i + 1 for i in range(final_mol.GetNumAtoms())}
        descriptors = collect_stereodescriptors(final_mol, atom_to_locant)

        # Should have both R/S and Ra/Sa descriptors
        rs_descs = [(l, c) for l, c in descriptors if c in ('R', 'S')]
        axial_descs = [(l, c) for l, c in descriptors if c in ('Ra', 'Sa')]
        assert len(rs_descs) >= 1, "Should have R/S descriptor"
        assert len(axial_descs) >= 1, "Should have Ra/Sa descriptor"

    def test_end_to_end_format_ra_with_s(self):
        """End-to-end: collect_stereodescriptors -> format produces '(NRa,MS)-' style."""
        # Create a molecule with both types and format the output
        mol = Chem.RWMol(Chem.MolFromSmiles('C[C@H](O)c1ccccc1-c1ccccc1C'))
        for bond in mol.GetBonds():
            b = bond.GetBeginAtom()
            e = bond.GetEndAtom()
            if (b.IsInRing() and e.IsInRing() and not bond.IsInRing()
                    and b.GetIsAromatic() and e.GetIsAromatic()):
                bond.SetStereo(Chem.BondStereo.STEREOATROPCCW)
                b_nbrs = [n.GetIdx() for n in b.GetNeighbors()
                           if n.GetIdx() != e.GetIdx()]
                e_nbrs = [n.GetIdx() for n in e.GetNeighbors()
                           if n.GetIdx() != b.GetIdx()]
                bond.SetStereoAtoms(b_nbrs[0], e_nbrs[0])
                break
        final_mol = mol.GetMol()
        rdCIPLabeler.AssignCIPLabels(final_mol)

        atom_to_locant = {i: i + 1 for i in range(final_mol.GetNumAtoms())}
        descriptors = collect_stereodescriptors(final_mol, atom_to_locant)
        result = format_stereodescriptor_string(descriptors)

        # Should be a parenthesized prefix like "(2S,7Ra)-" or similar
        assert result.startswith('(')
        assert result.endswith(')-')
        assert 'Ra' in result or 'Sa' in result, "Should contain axial descriptor"
        # Should also have R or S for the point chirality
        # (need to check for R/S not preceded by another letter to avoid matching Ra/Sa)
        import re
        has_point_chirality = bool(re.search(r'\d[RS](?![a])', result))
        assert has_point_chirality, f"Should contain R/S point chirality descriptor in {result}"

    def test_opsin_limitation_documented(self):
        """OPSIN cannot interpret Ra/Sa descriptors.

        This test documents the known OPSIN limitation: OPSIN's
        StereochemistryHandler.java throws StereochemistryException
        for AXIAL_TYPE_VAL. Names containing Ra/Sa will fail OPSIN
        round-trip validation. This is NOT an Orthonym bug.

        See: opsin/opsin-core/src/main/java/.../StereochemistryHandler.java
        See: 115-RESEARCH.md Pitfall 4
        """
        # This is a documentation-only test -- it always passes.
        # The actual OPSIN limitation is documented in the docstring.
        pass
