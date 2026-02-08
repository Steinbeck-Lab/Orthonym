"""Phase 27: Isoindoline locant mapping tests.

Tests that the iupac_locants mapping in FUSED_HETEROCYCLE_DATA for isoindoline
produces correct IUPAC locants from the perception layer -- no postprocessor
band-aids needed.

Root cause fix: fused_heterocycles.py isoindoline entry had 5-ring atom
positions shifted by 1. Corrected so N -> locant 2 (not 3).
"""

import pytest

from orthonym.namer import name_compound


# ---------------------------------------------------------------------------
# Section 1: Locant correctness -- dione (phthalimide) compounds
# ---------------------------------------------------------------------------


class TestIsoindolineDioneLocants:
    """Verify isoindoline-1,3-dione locants are correct from source."""

    @pytest.mark.unit
    def test_bare_phthalimide(self):
        """Bare phthalimide produces isoindoline-1,3-dione (not 2,4)."""
        name = name_compound("O=C1NC(=O)c2ccccc21")
        assert "isoindoline-1,3-dione" in name

    @pytest.mark.unit
    def test_hydroxy_phthalimide(self):
        """6-hydroxy substituted phthalimide has correct locants."""
        name = name_compound("O=C1CCC(N2C(=O)c3ccc(O)cc3C2=O)C(=O)N1")
        assert "6-hydroxyisoindoline-1,3-dione" in name

    @pytest.mark.unit
    def test_phthalimide_with_pyrazole(self):
        """Phthalimide bearing pyrazole substituent."""
        name = name_compound("Cc1cc(N2C(=O)c3ccccc3C2=O)n(C)n1")
        assert "isoindoline-1,3-dione" in name


# ---------------------------------------------------------------------------
# Section 2: Locant correctness -- mono-one (isoindolinone) compounds
# ---------------------------------------------------------------------------


class TestIsoindolinMonoOneLocants:
    """Verify mono-one isoindoline locants are correct from source."""

    @pytest.mark.unit
    def test_basic_isoindolinone(self):
        """Basic isoindolinone produces isoindolin-3-one or isoindolin-1-one."""
        name = name_compound("O=C1NCc2ccccc21")
        # C=O adjacent to benzo junction should be position 1 or 3
        assert "isoindolin-" in name
        assert "-one" in name
        # Must NOT produce isoindolin-2-one (position 2 is nitrogen)
        assert "isoindolin-2-one" not in name


# ---------------------------------------------------------------------------
# Section 3: Previously unfixable compound (index 200) -- now correct
# ---------------------------------------------------------------------------


class TestIsoindolineIndex200:
    """Index 200 was previously unfixable by postprocessor.

    Root cause: prefix locants (e.g., methoxy position) were wrong due to
    shifted 5-ring atom-to-locant mapping. With corrected mapping, both
    prefix and suffix locants are now correct.
    """

    @pytest.mark.unit
    def test_index_200_correct_prefix_locants(self):
        """Index 200: prefix locants now correct (4-methoxy, not 1-methoxy)."""
        name = name_compound("COc1c(C)c(O)cc2c1C(=O)N[C@H]2C")
        # Methoxy should be at position 4 (not 1)
        assert "4-methoxy" in name
        # Position 1 should be the C=O (suffix), not methoxy
        assert "1-methoxy" not in name

    @pytest.mark.unit
    def test_index_200_correct_suffix_locant(self):
        """Index 200: suffix locant is isoindolin-1-one (not 2-one)."""
        name = name_compound("COc1c(C)c(O)cc2c1C(=O)N[C@H]2C")
        assert "isoindolin-1-one" in name
        assert "isoindolin-2-one" not in name

    @pytest.mark.unit
    def test_index_200_opsin_parseable(self):
        """Index 200: corrected name is OPSIN-parseable."""
        import subprocess

        name = name_compound("COc1c(C)c(O)cc2c1C(=O)N[C@H]2C")
        result = subprocess.run(
            [
                "java", "-jar",
                "opsin-cli-2.8.0-jar-with-dependencies.jar",
                "-osmi",
            ],
            input=name,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.stdout.strip() != "", (
            f"OPSIN cannot parse: {name}"
        )

    @pytest.mark.unit
    def test_index_200_full_name(self):
        """Index 200: full name with all substituents correctly placed."""
        name = name_compound("COc1c(C)c(O)cc2c1C(=O)N[C@H]2C")
        assert "6-hydroxy" in name
        assert "4-methoxy" in name
        assert "isoindolin-1-one" in name


# ---------------------------------------------------------------------------
# Section 4: Locant mapping verification via substructure matching
# ---------------------------------------------------------------------------


class TestIsoindolineLocantMapping:
    """Verify the locant mapping data in FUSED_HETEROCYCLE_DATA is correct."""

    @pytest.mark.unit
    def test_locant_mapping_n_at_position_2(self):
        """Isoindoline IUPAC numbering: N is at position 2."""
        from rdkit import Chem
        from orthonym.data.fused_heterocycles import (
            match_fused_heterocycle_core,
        )

        mol = Chem.MolFromSmiles("O=C1NC(=O)c2ccccc21")
        result = match_fused_heterocycle_core(mol)
        assert result is not None
        name, atom_mapping, _ = result
        assert name == "isoindoline"

        # Find the N atom and verify its IUPAC locant is 2
        for atom_idx, locant in atom_mapping.items():
            atom = mol.GetAtomWithIdx(atom_idx)
            if atom.GetSymbol() == "N":
                assert locant == 2, (
                    f"N atom at mol idx {atom_idx} has locant {locant}, "
                    f"expected 2"
                )

    @pytest.mark.unit
    def test_locant_mapping_carbonyl_positions(self):
        """Phthalimide C=O atoms should be at positions 1 and 3."""
        from rdkit import Chem
        from orthonym.data.fused_heterocycles import (
            match_fused_heterocycle_core,
        )

        mol = Chem.MolFromSmiles("O=C1NC(=O)c2ccccc21")
        _, atom_mapping, _ = match_fused_heterocycle_core(mol)

        carbonyl_locants = []
        for atom_idx, locant in atom_mapping.items():
            atom = mol.GetAtomWithIdx(atom_idx)
            if atom.GetSymbol() == "C" and not atom.GetIsAromatic():
                # Non-aromatic C in 5-ring -- check for =O neighbor
                for nbr in atom.GetNeighbors():
                    if (
                        nbr.GetSymbol() == "O"
                        and nbr.GetIdx() not in atom_mapping
                    ):
                        carbonyl_locants.append(locant)

        assert sorted(carbonyl_locants) == [1, 3], (
            f"Carbonyl locants {carbonyl_locants}, expected [1, 3]"
        )


# ---------------------------------------------------------------------------
# Section 5: No postprocessor needed (regression guard)
# ---------------------------------------------------------------------------


class TestNoPostprocessorNeeded:
    """Verify that correct locants come from source, not string hacks."""

    @pytest.mark.unit
    def test_no_postprocess_function_exists(self):
        """_postprocess_name should not exist -- root cause is fixed."""
        import orthonym.namer as namer
        assert not hasattr(namer, "_postprocess_name"), (
            "_postprocess_name still exists -- root cause fix in "
            "fused_heterocycles.py should make it unnecessary"
        )
