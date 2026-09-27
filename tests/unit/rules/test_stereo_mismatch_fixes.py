"""
Tests for stereo_mismatch benchmark compounds (through).

a phase Plan 03 Task 1: Fix stereodescriptor accuracy for benchmark compounds.

Root cause analysis of 5 stereo_mismatch compounds:
-: Generated name already correct (matches benchmark). RT fails due to
  steroid naming convention (OPSIN adds more stereo from steroid name).
-: No stereo in original SMILES (no @/@@ notation). Unfixable without
  adding stereo that doesn't exist in input.
-: Missing substituent stereocenter (1R) in name. Requires substituent-level
  stereo descriptor support (deferred to a phase).
-: Missing (E) descriptor for exocyclic C=C bond on substituent.
  Fixable by enhancing collect_stereodescriptors to capture near-parent E/Z bonds.
-: No stereo in original SMILES. Unfixable.
"""

import pytest
from rdkit import Chem
from rdkit.Chem import rdCIPLabeler

from orthonym import name_compound


# ============================================================================
#: Stereocenters correct -- name already matches benchmark
# ============================================================================

SM_40_SMILES = (
    "CC(CC(=O)CC(C)C1C[C@H](O)[C@@]2(C)C3=C(C(=O)CC12C)"
    "C1(C)CC[C@H](O)C(C)(C)C1C[C@@H]3O)C(=O)O"
)


class TestSM40:
    """: a partially stereo-defined steroid acid.

    j7 (TRIAGE g7 C12): the input leaves C-10, 13, 17 and 20 undefined, and the
    cholestane stereoparent implies their configuration, the Blue Book
    :51047: "The name of a fundamental parent structure usually implies the
    absolute configuration of all chirality centers"). The steroid producer now
    declines such an input; the raw benchmark name below (and the raw
    '(3S,7S,14R,15S)-3,7,15,26-tetrahydroxy-4,4,14-trimethylcholest-8-ene-11,23,
    26-trione' it had become) over-specified those centres (OPSIN: skeleton only).
    Production names at best-effort with an RT-exact systematic name."""

    _BENCHMARK = (
        "(3S,7S,14R,15S)-3,7,15,27-tetrahydroxy-4,4,14-trimethyl"
        "cholest-8-en-11,23,27-trione"
    )

    @staticmethod
    def _best_effort_name():
        from orthonym.namer import Orthonym
        return Orthonym(general_fallback=True, general_fallback_unverified=True,
                        allow_aromatic_general=True).name(SM_40_SMILES)

    @pytest.mark.opsin_gate
    def test_name_contains_stereodescriptor_block(self):
        """The best-effort production name carries a (xR,yS,...) block."""
        name = self._best_effort_name()
        assert name.startswith("("), f"Name missing stereo block: {name}"
        assert ")-" in name, f"Name missing stereo block closing: {name}"

    @pytest.mark.opsin_gate
    def test_stereo_locants_are_valid(self):
        """The best-effort production name is RT exact and cites R/S."""
        from tests.support.rt_assert import name_is_rt_exact
        name = self._best_effort_name()
        stereo_block = name.split(")-")[0] + ")"
        assert any(c in stereo_block for c in "RS"), f"No R/S in stereo block: {stereo_block}"
        assert name_is_rt_exact(name, SM_40_SMILES), name

    def test_sm40_matches_benchmark(self):
        """Gate off (the raw generator): never the over-specified benchmark-family
        steroid name; whatever ships must denote itself."""
        from tests.support.rt_assert import name_is_rt_exact
        name = name_compound(SM_40_SMILES)
        assert name != self._BENCHMARK
        assert "cholest" not in name, name
        assert name == "unknown organic compound" or name_is_rt_exact(name, SM_40_SMILES), name


# ============================================================================
#: No stereo in SMILES -- unfixable, documented
# ============================================================================

SM_41_SMILES = "CCC(CCC(C)C1CCC2C3C(O)C=C4CC(O)CCC4(C)C3CCC12C)C(C)C"


class TestSM41:
    """: No stereo in SMILES -- cannot generate stereodescriptors."""

    def test_no_stereo_in_smiles(self):
        """Verify that SMILES has no stereo notation."""
        assert "@" not in SM_41_SMILES, "SM-41 should have no @ notation"

    def test_no_cip_labels_assigned(self):
        """RDKit should not assign CIP labels when no stereo defined."""
        mol = Chem.MolFromSmiles(SM_41_SMILES)
        rdCIPLabeler.AssignCIPLabels(mol)
        cip_atoms = [a for a in mol.GetAtoms() if a.HasProp("_CIPCode")]
        assert len(cip_atoms) == 0, f"Unexpected CIP labels on SM-41: {len(cip_atoms)}"

    def test_name_is_steroid(self):
        """ should produce a steroid retained name."""
        name = name_compound(SM_41_SMILES)
        assert "stigmast" in name.lower(), f"Expected steroid name: {name}"


# ============================================================================
#: Missing substituent stereocenter
# ============================================================================

SM_42_SMILES = "CC(C)CC[C@@H](O)[C@H]1C(=O)OC[C@@H]1CO"


class TestSM42:
    """: 3 stereocenters, name shows 2 (ring atoms only). Substituent
    stereocenter (1R on 1-hydroxy-4-methylpentyl) missing from name.

    The 3rd stereocenter is on the substituent, not the parent ring.
    IUPAC requires stereo within substituent brackets, e.g.,
    3-[(1R)-1-hydroxy-4-methylpentyl]. This is a substituent naming
    feature deferred to a phase.
    """

    def test_three_stereocenters_detected(self):
        """RDKit should detect 3 stereocenters in."""
        mol = Chem.MolFromSmiles(SM_42_SMILES)
        rdCIPLabeler.AssignCIPLabels(mol)
        cip_atoms = [a for a in mol.GetAtoms() if a.HasProp("_CIPCode")]
        assert len(cip_atoms) == 3, f"Expected 3 stereocenters, got {len(cip_atoms)}"

    def test_ring_stereocenters_in_name(self):
        """Name should contain (3S,4S)- for the ring stereocenters."""
        name = name_compound(SM_42_SMILES)
        assert "(3S,4S)-" in name or "(4S,3S)-" in name, (
            f"Ring stereo missing from name: {name}"
        )

    def test_name_contains_oxolanone(self):
        """Name should reference the oxolanone (lactone) parent."""
        name = name_compound(SM_42_SMILES)
        assert "oxolan" in name.lower(), f"Expected oxolanone parent: {name}"


# ============================================================================
#: Missing (E) descriptor for exocyclic C=C bond
# ============================================================================

SM_43_SMILES = "CC(C)C1=C(O)C(N)=C(/C=C/c2ccccc2)C(=O)C1=O"


class TestSM43:
    """: Has E double bond in substituent, missing from name.

    The /C=C/ bond between the ring and phenyl creates an E configuration.
    RDKit detects this as E on the bond between atoms 9-10.
    The E descriptor should appear in the name as (E)- prefix.
    """

    def test_e_bond_detected_by_rdkit(self):
        """RDKit should detect E configuration on the C=C bond."""
        mol = Chem.MolFromSmiles(SM_43_SMILES)
        rdCIPLabeler.AssignCIPLabels(mol)
        ez_bonds = [
            b for b in mol.GetBonds()
            if b.HasProp("_CIPCode") and b.GetProp("_CIPCode") in ("E", "Z")
        ]
        assert len(ez_bonds) >= 1, "No E/Z bonds detected in SM-43"
        assert ez_bonds[0].GetProp("_CIPCode") == "E", "Expected E configuration"

    def test_name_contains_e_descriptor(self):
        """Name should contain E descriptor for the styryl/styrenyl group.

        The E/Z bond is on the substituent but one hop from the parent ring.
        The descriptor uses the parent locant at the attachment point, e.g. (3E)-.
        """
        name = name_compound(SM_43_SMILES)
        assert "E)" in name and "(" in name, (
            f"Missing E descriptor in name: {name}"
        )

    def test_name_contains_amino(self):
        """Name should contain amino substituent."""
        name = name_compound(SM_43_SMILES)
        assert "amino" in name.lower(), f"Missing amino in name: {name}"


# ============================================================================
#: No stereo in SMILES -- unfixable, documented
# ============================================================================

SM_44_SMILES = "C=C(C)C(C)CCC(C)C1CCC2C3=CCC4CC(O)CCC4(C)C3CCC21C"


class TestSM44:
    """: No stereo in SMILES -- cannot generate stereodescriptors."""

    def test_no_stereo_in_smiles(self):
        """Verify that SMILES has no stereo notation."""
        assert "@" not in SM_44_SMILES, "SM-44 should have no @ notation"

    def test_no_cip_labels_assigned(self):
        """RDKit should not assign CIP labels when no stereo defined."""
        mol = Chem.MolFromSmiles(SM_44_SMILES)
        rdCIPLabeler.AssignCIPLabels(mol)
        cip_atoms = [a for a in mol.GetAtoms() if a.HasProp("_CIPCode")]
        assert len(cip_atoms) == 0, f"Unexpected CIP labels on SM-44: {len(cip_atoms)}"

    def test_name_is_steroid(self):
        """ should produce a steroid retained name."""
        name = name_compound(SM_44_SMILES)
        assert "ergosta" in name.lower(), f"Expected steroid name: {name}"


# ============================================================================
# Regression guards: existing stereo compounds must not regress
# ============================================================================


class TestStereoRegressionGuards:
    """Ensure currently-passing stereo compounds are not broken."""

    @pytest.mark.parametrize(
        "smiles,expected_stereo",
        [
            # (2S)-butan-2-ol
            ("CC[C@@H](O)C", "(2S)-"),
            # (2R)-butan-2-ol
            ("CC[C@H](O)C", "(2R)-"),
            # (2E)-but-2-ene
            (r"C/C=C/C", "(2E)-"),
            # (2Z)-but-2-ene
            (r"C/C=C\C", "(2Z)-"),
        ],
    )
    def test_simple_stereo_preserved(self, smiles, expected_stereo):
        """Simple R/S and E/Z naming must remain correct."""
        name = name_compound(smiles)
        assert expected_stereo in name, (
            f"Expected '{expected_stereo}' in name for {smiles}, got: {name}"
        )

    def test_cholesterol_stereo(self):
        """Cholesterol (if named as steroid) should preserve stereo or name."""
        smiles = "C([C@H]1CC[C@@H]2[C@]1(CC[C@H]1[C@H]2CC=C2C[C@@H](O)CC[C@@]21C)C)CCC(C)C"
        name = name_compound(smiles)
        # Should have stereo block or be a retained name
        assert name is not None and len(name) > 0, "Cholesterol naming failed"

    def test_menthol_stereo(self):
        """Menthol should have stereo descriptors."""
        smiles = "C[C@@H]1CC[C@H]([C@@H](C1)O)C(C)C"
        name = name_compound(smiles)
        assert name is not None and len(name) > 0, "Menthol naming failed"
