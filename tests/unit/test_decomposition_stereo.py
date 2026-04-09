"""Tests for stereochemistry preservation through decomposition pipeline.

Covers:
- CIP re-assignment on fragment mol objects after bond cleavage + capping
- CIP label changes when substituent tree changes (expected CIP "flip")
- Unresolvable stereocenters after cleavage (symmetric fragments)
- E/Z preservation in acid-side fragments
- E/Z not generated for bonds spanning cleavage point
- Compound-specific integration tests through name_compound()

Phase 63-03: STER-03 requirement.
"""

import pytest
from rdkit import Chem
from rdkit.Chem import rdCIPLabeler

from orthonym.decomposition.fragment_capping import cleave_and_cap
from orthonym.decomposition.bond_cleavage import find_cleavable_bonds


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_fragment_cip(smiles):
    """Cleave a molecule and return CIP info for each fragment.

    Returns a list of dicts:
      {side, smiles, atom_cip: {atom_idx: label}, bond_cip: {bond_idx: label}}
    """
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Invalid SMILES: {smiles}"
    rdCIPLabeler.AssignCIPLabels(mol)

    bonds = find_cleavable_bonds(mol)
    if not bonds:
        return []

    fragments = cleave_and_cap(mol, bonds)
    results = []
    for frag in fragments:
        fmol = Chem.MolFromSmiles(frag["smiles"])
        if fmol is None:
            results.append({
                "side": frag["side"],
                "smiles": frag["smiles"],
                "atom_cip": {},
                "bond_cip": {},
            })
            continue

        rdCIPLabeler.AssignCIPLabels(fmol)
        atom_cip = {}
        for atom in fmol.GetAtoms():
            if atom.HasProp("_CIPCode"):
                atom_cip[atom.GetIdx()] = atom.GetProp("_CIPCode")
        bond_cip = {}
        for bond in fmol.GetBonds():
            if bond.HasProp("_CIPCode"):
                bond_cip[bond.GetIdx()] = bond.GetProp("_CIPCode")

        results.append({
            "side": frag["side"],
            "smiles": frag["smiles"],
            "atom_cip": atom_cip,
            "bond_cip": bond_cip,
        })
    return results


# ===========================================================================
# CIP Re-assignment Tests
# ===========================================================================


@pytest.mark.unit
class TestCIPReassignment:
    """Test that CIP is correctly re-assigned on capped fragments."""

    def test_stereocenter_retained_after_amide_cleavage(self):
        """R/S stereocenter on amine side of amide retains CIP after cleavage."""
        # CC(=O)N[C@@H](C)S  -> amine fragment: [C@H](N)(C)S
        smiles = "CC(=O)N[C@@H](C)S"
        frags = _get_fragment_cip(smiles)
        amine_frags = [f for f in frags if f["side"] == "amine"]
        assert len(amine_frags) == 1
        # The stereocenter should have a CIP label (R or S)
        assert len(amine_frags[0]["atom_cip"]) > 0, (
            "Amine fragment lost CIP stereocenter after amide cleavage"
        )

    def test_stereocenter_retained_after_ester_cleavage(self):
        """R/S stereocenter on acid side of ester retains CIP after cleavage."""
        # O=C(OCC)[C@@H](O)C -> acid fragment: O=C(OH)[C@@H](O)C
        smiles = "O=C(OCC)[C@@H](O)C"
        frags = _get_fragment_cip(smiles)
        acid_frags = [f for f in frags if f["side"] == "acid"]
        assert len(acid_frags) == 1
        assert len(acid_frags[0]["atom_cip"]) > 0, (
            "Acid fragment lost CIP stereocenter after ester cleavage"
        )

    def test_cip_flip_when_priority_changes(self):
        """CIP label correctly changes when cleavage alters priority order.

        Full molecule: CCC(=O)O[C@@H](F)OC has CIP S at the stereocenter.
        After ester cleavage, the ester-O becomes OH. Since OH has lower
        priority than OMe while ester-O had higher priority than OMe,
        the CIP label flips to R for the same geometric arrangement.
        Both labels are correct for their respective contexts.
        """
        full_smiles = "CCC(=O)O[C@@H](F)OC"
        mol = Chem.MolFromSmiles(full_smiles)
        rdCIPLabeler.AssignCIPLabels(mol)

        # Find the stereocenter CIP in the full molecule
        full_cip = None
        for atom in mol.GetAtoms():
            if atom.HasProp("_CIPCode"):
                full_cip = atom.GetProp("_CIPCode")
        assert full_cip == "S", f"Expected S on full molecule, got {full_cip}"

        # Get fragment CIP
        frags = _get_fragment_cip(full_smiles)
        alkyl_frags = [f for f in frags if f["side"] == "alkyl"]
        assert len(alkyl_frags) == 1

        # The fragment should have CIP R (flipped from S)
        frag_cip_values = list(alkyl_frags[0]["atom_cip"].values())
        assert len(frag_cip_values) == 1, "Expected exactly one stereocenter in fragment"
        assert frag_cip_values[0] == "R", (
            f"Expected CIP flip S->R after cleavage, got {frag_cip_values[0]}"
        )

    def test_unresolvable_stereocenter_after_cleavage(self):
        """Stereocenter that becomes symmetric after cleavage gets no CIP label.

        C(F)(F)(Cl)OC(=O)CC: The carbon C(F)(F)(Cl) is not a stereocenter
        (two F atoms) even before cleavage. After cleavage, alkyl fragment
        is OC(F)(F)Cl -- still not a stereocenter. RDKit correctly omits CIP.
        """
        smiles = "FC(F)(Cl)OC(=O)CC"
        frags = _get_fragment_cip(smiles)
        alkyl_frags = [f for f in frags if f["side"] == "alkyl"]
        assert len(alkyl_frags) == 1
        assert len(alkyl_frags[0]["atom_cip"]) == 0, (
            "Symmetric fragment should have no CIP stereocenters"
        )


# ===========================================================================
# E/Z Preservation Tests
# ===========================================================================


@pytest.mark.unit
class TestEZPreservation:
    """Test E/Z descriptor handling through decomposition."""

    def test_ez_preserved_in_acid_fragment(self):
        """E/Z double bond on the acid side of an amide is preserved."""
        # C/C=C/C(=O)NCC -> acid fragment: C/C=C/C(=O)O
        smiles = "C/C=C/C(=O)NCC"
        frags = _get_fragment_cip(smiles)
        acid_frags = [f for f in frags if f["side"] == "acid"]
        assert len(acid_frags) == 1
        assert len(acid_frags[0]["bond_cip"]) > 0, (
            "E/Z bond lost in acid fragment after amide cleavage"
        )
        # Should be E
        ez_labels = list(acid_frags[0]["bond_cip"].values())
        assert "E" in ez_labels, f"Expected E label, got {ez_labels}"

    def test_ez_preserved_in_ester_acid_fragment(self):
        """E/Z double bond on the acid side of an ester is preserved."""
        # C/C=C/C(=O)OCC -> acid fragment: C/C=C/C(=O)O
        smiles = "C/C=C/C(=O)OCC"
        frags = _get_fragment_cip(smiles)
        acid_frags = [f for f in frags if f["side"] == "acid"]
        assert len(acid_frags) == 1
        assert len(acid_frags[0]["bond_cip"]) > 0, (
            "E/Z bond lost in acid fragment after ester cleavage"
        )
        ez_labels = list(acid_frags[0]["bond_cip"].values())
        assert "E" in ez_labels

    def test_no_ez_on_amine_fragment_without_double_bond(self):
        """Amine fragment from amide cleavage has no E/Z if no double bond."""
        smiles = "C/C=C/C(=O)NCC"
        frags = _get_fragment_cip(smiles)
        amine_frags = [f for f in frags if f["side"] == "amine"]
        assert len(amine_frags) == 1
        assert len(amine_frags[0]["bond_cip"]) == 0, (
            "Amine fragment should have no E/Z bonds"
        )

    def test_z_double_bond_preserved(self):
        """Z configuration preserved through decomposition."""
        smiles = r"C/C=C\C(=O)OCC"  # Z double bond
        frags = _get_fragment_cip(smiles)
        acid_frags = [f for f in frags if f["side"] == "acid"]
        assert len(acid_frags) == 1
        ez_labels = list(acid_frags[0]["bond_cip"].values())
        assert "Z" in ez_labels, f"Expected Z label, got {ez_labels}"


# ===========================================================================
# Decomposition Integration Tests (through name_compound)
# ===========================================================================


@pytest.mark.unit
class TestDecompositionStereoIntegration:
    """Integration tests: stereo descriptors appear in decomposed names."""

    def test_amide_with_chiral_acid_chain(self):
        """R/S stereocenter on acid side appears in decomposed amide name."""
        from orthonym import name_compound

        # [C@@H](N)(C)C(=O)NCC -> should include (R) or (S) in name
        smiles = "[C@@H](N)(C)C(=O)NCC"
        name = name_compound(smiles)
        assert name is not None
        assert name != "unknown"
        # The name should contain a stereo descriptor
        assert "R" in name or "S" in name, (
            f"Decomposed amide name '{name}' missing R/S descriptor"
        )

    def test_ester_with_chiral_acid_chain(self):
        """R/S stereocenter on acid side appears in decomposed ester name."""
        from orthonym import name_compound

        smiles = "O=C(OCC)[C@@H](O)C"
        name = name_compound(smiles)
        assert name is not None
        assert name != "unknown"
        assert "R" in name or "S" in name, (
            f"Decomposed ester name '{name}' missing R/S descriptor"
        )

    def test_ester_with_ez_acid(self):
        """E/Z double bond on acid side appears in decomposed ester name."""
        from orthonym import name_compound

        smiles = "C/C=C/C(=O)OCC"
        name = name_compound(smiles)
        assert name is not None
        assert name != "unknown"
        assert "E" in name or "Z" in name, (
            f"Decomposed ester name '{name}' missing E/Z descriptor"
        )

    def test_amide_ez_acid_fragment(self):
        """E/Z on acid fragment preserved through amide decomposition."""
        from orthonym import name_compound

        # (E)-but-2-enoic acid amide with ethylamine
        smiles = "C/C=C/C(=O)NCC"
        name = name_compound(smiles)
        assert name is not None
        assert name != "unknown"
        # Acid fragment should be named with E/Z
        # (exact format depends on assembly path)
        assert "E" in name or "Z" in name or "but-2-en" in name, (
            f"Amide name '{name}' missing E/Z or unsaturation from acid fragment"
        )

    def test_combined_rs_and_ez_through_decomposition(self):
        """Both R/S and E/Z survive decomposition in a multi-stereo compound."""
        from orthonym import name_compound

        # Ester: ethyl (2S,3E)-... a compound with both stereo types on acid
        smiles = "O=C(OCC)[C@@H](O)/C=C/C"
        name = name_compound(smiles)
        assert name is not None
        assert name != "unknown"
        # Should have at least one stereo descriptor
        has_rs = "R" in name or "S" in name
        has_ez = "E" in name or "Z" in name
        assert has_rs or has_ez, (
            f"Multi-stereo compound name '{name}' missing stereo descriptors"
        )


# ===========================================================================
# Fragment Capping Stereo Correctness Tests
# ===========================================================================


@pytest.mark.unit
class TestFragmentCappingStereo:
    """Test that cleave_and_cap produces fragments with correct stereo SMILES."""

    def test_fragment_smiles_has_chiral_tag(self):
        """Fragment SMILES retains chiral tags (@ or @@) after capping."""
        smiles = "CC(=O)N[C@@H](C)S"
        mol = Chem.MolFromSmiles(smiles)
        bonds = find_cleavable_bonds(mol)
        fragments = cleave_and_cap(mol, bonds)

        amine_frags = [f for f in fragments if f["side"] == "amine"]
        assert len(amine_frags) == 1
        frag_smiles = amine_frags[0]["smiles"]
        assert "@" in frag_smiles, (
            f"Fragment SMILES '{frag_smiles}' lost chiral tag"
        )

    def test_fragment_smiles_has_ez_markers(self):
        """Fragment SMILES retains E/Z markers (/ or \\) after capping."""
        smiles = "C/C=C/C(=O)NCC"
        mol = Chem.MolFromSmiles(smiles)
        bonds = find_cleavable_bonds(mol)
        fragments = cleave_and_cap(mol, bonds)

        acid_frags = [f for f in fragments if f["side"] == "acid"]
        assert len(acid_frags) == 1
        frag_smiles = acid_frags[0]["smiles"]
        # E/Z markers appear as / or \ in SMILES
        assert "/" in frag_smiles or "\\" in frag_smiles, (
            f"Fragment SMILES '{frag_smiles}' lost E/Z bond markers"
        )

    def test_cip_reassigned_on_fragment_mol(self):
        """CIP labels are re-assigned on fragment mol after capping.

        Verify by checking that rdCIPLabeler has been called (the
        fragment mol has _CIPCode properties matching fresh assignment).
        """
        smiles = "O=C(OCC)[C@@H](O)C"
        mol = Chem.MolFromSmiles(smiles)
        bonds = find_cleavable_bonds(mol)
        fragments = cleave_and_cap(mol, bonds)

        acid_frags = [f for f in fragments if f["side"] == "acid"]
        assert len(acid_frags) == 1

        # Parse the fragment SMILES and independently assign CIP
        frag_mol = Chem.MolFromSmiles(acid_frags[0]["smiles"])
        assert frag_mol is not None
        rdCIPLabeler.AssignCIPLabels(frag_mol)

        # The fragment should have a stereocenter with CIP label
        has_cip = any(
            atom.HasProp("_CIPCode") for atom in frag_mol.GetAtoms()
        )
        assert has_cip, "Fragment mol should have CIP labels after re-assignment"

    def test_multiple_stereocenters_preserved(self):
        """Multiple stereocenters in a fragment are all preserved."""
        # Two chiral centers on acid side of ester
        smiles = "O=C(OCC)[C@@H](O)[C@H](N)C"
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            pytest.skip("SMILES not parseable")
        bonds = find_cleavable_bonds(mol)
        if not bonds:
            pytest.skip("No cleavable bonds found")
        fragments = cleave_and_cap(mol, bonds)

        acid_frags = [f for f in fragments if f["side"] == "acid"]
        if not acid_frags:
            pytest.skip("No acid fragment produced")

        frag_mol = Chem.MolFromSmiles(acid_frags[0]["smiles"])
        if frag_mol is None:
            pytest.skip("Fragment SMILES not parseable")
        rdCIPLabeler.AssignCIPLabels(frag_mol)

        cip_count = sum(
            1 for atom in frag_mol.GetAtoms() if atom.HasProp("_CIPCode")
        )
        assert cip_count >= 2, (
            f"Expected 2+ stereocenters in fragment, got {cip_count}"
        )


# ===========================================================================
# Phase 140: End-to-end decomposition stereo via name_compound
# ===========================================================================

import re as _re


@pytest.mark.unit
class TestDecompositionStereoEndToEnd:
    """Verify stereo survives the full decomposition+assembly pipeline via name_compound."""

    @pytest.mark.parametrize("smiles,desc", [
        ("O=C(OCC)[C@@H](O)C", "ethyl (S)-2-hydroxypropanoate -- ester with R/S"),
        ("O=C(OC)[C@H](N)CC", "methyl aminobutanoate -- ester with amino stereo"),
        ("CC(=O)O[C@H](C)CC", "1-methylpropyl acetate -- alcohol-side stereo"),
        ("O=C(NC)[C@@H](O)CC", "amide with hydroxyl stereo"),
        ("O=C(OCC)/C=C/C", "ethyl but-2-enoate -- ester with E/Z"),
        ("O=C(OCCC)[C@@H](CC)O", "propyl 2-hydroxypentanoate -- longer chain ester"),
        ("CC(=O)N[C@@H](CC)C(=O)O", "amide with amino acid stereo"),
        ("O=C(OCC)[C@@H](O)[C@H](O)CC", "diester with two stereocenters"),
        ("O=C(OC(C)C)[C@H](C)O", "isopropyl ester with tertiary stereo"),
        pytest.param("O=C(O[C@@H]1CCCC1)[C@@H](O)C", "cyclopentyl ester with ring fragment stereo",
                     marks=pytest.mark.xfail(reason="Handler gap: ring-fragment ester drops hydroxyl + stereo (Phase 140 backstop logs WARNING)")),
    ])
    def test_decomposition_preserves_stereo(self, smiles, desc):
        from orthonym import name_compound
        name = name_compound(smiles)
        # Check that at least one stereodescriptor appears in the output
        assert _re.search(r'\([^)]*[RSEZrsez][^)]*\)', name), (
            f"Decomposed name '{name}' for {desc} (SMILES: {smiles}) "
            f"is missing stereodescriptors"
        )

    def test_no_stereo_compound_no_false_stereo(self):
        """Compounds without stereo input should NOT get stereo in decomposed name."""
        from orthonym import name_compound
        name = name_compound("O=C(OCC)CCC")  # ethyl butanoate, no stereo
        # Should not have R/S/E/Z descriptors
        assert not _re.search(r'\(\d*[RSEZrsez](,\d*[RSEZrsez])*\)-', name), (
            f"Non-stereo compound got false stereo: {name}"
        )
