"""Unit tests for multi-amide quality gate detection (PEP-01).

N-acyl multi-amide chains (e.g., CC(=O)NCC(=O)NCC(=O)NCC(=O)O) with no
free NH2 terminus bypass the peptide namer and produce partial names like
"2-(ethanoylamino)ethanoic acid" that only name 1 of N amide units. The
quality gate must detect this under-naming and trigger decomposition.
"""

import re

import pytest
from rdkit import Chem

from orthonym.decomposition.engine import _name_quality_is_acceptable
from orthonym.decomposition.bond_cleavage import find_cleavable_bonds
from orthonym.namer import name_compound


# ---------------------------------------------------------------------------
# Quality gate direct tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMultiAmideQualityGate:
    """Test quality gate detection of multi-amide under-naming."""

    def test_quality_gate_rejects_multi_amide_with_partial_name(self):
        """4-amide chain with partial name should be rejected by quality gate."""
        smiles = "CC(=O)NCC(=O)NCC(=O)NCC(=O)O"
        mol = Chem.MolFromSmiles(smiles)
        bonds = find_cleavable_bonds(mol)
        amide_count = sum(1 for b in bonds if b.get("type") == "amide")
        assert amide_count >= 3, f"Expected >=3 amide bonds, got {amide_count}"

        # Partial name covers only 1 amide -- should be rejected
        result = _name_quality_is_acceptable("2-(ethanoylamino)ethanoic acid", mol)
        assert result is False, (
            "Quality gate should reject partial name for a 4-amide chain"
        )

    def test_quality_gate_accepts_single_amide_name(self):
        """Single amide (N-acetylglycine) with full name should be accepted."""
        smiles = "CC(=O)NCC(=O)O"
        mol = Chem.MolFromSmiles(smiles)

        result = _name_quality_is_acceptable("2-(ethanoylamino)ethanoic acid", mol)
        assert result is True, (
            "Quality gate should accept a name that fully covers a single amide"
        )

    def test_quality_gate_accepts_well_formed_name(self):
        """Well-formed name with digits and hyphens should be accepted."""
        smiles = "CCCCCCCCCCCCCCCC"  # hexadecane, 16 heavy atoms
        mol = Chem.MolFromSmiles(smiles)

        result = _name_quality_is_acceptable("hexadecane", mol)
        assert result is True, (
            "Quality gate should accept well-formed names for non-amide molecules"
        )


# ---------------------------------------------------------------------------
# End-to-end naming tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMultiAmideEndToEnd:
    """Test end-to-end naming for multi-amide chains."""

    def test_4_amide_chain_names_multiple_amides(self):
        """4-amide chain should produce a name referencing more than 1 amide."""
        smiles = "CC(=O)NCC(=O)NCC(=O)NCC(=O)O"
        result = name_compound(smiles)

        # The name should be substantially different from the partial name
        # that only covers the terminal amide bond
        partial_name = "2-(ethanoylamino)ethanoic acid"
        assert result != partial_name, (
            f"4-amide chain should not produce the single-amide partial name. "
            f"Got: {result}"
        )

        # The name should be longer (more complete) or contain multiple
        # amide-related references (glycyl/glycine each stand for one amide
        # unit in peptide-style decomposition names)
        amide_refs = len(re.findall(
            r'amino|amido|amide|acetamid|formamid|acyl|glycyl|glycine',
            result, re.IGNORECASE
        ))
        long_enough = len(result) > len(partial_name) + 5
        assert amide_refs > 1 or long_enough, (
            f"Expected name referencing multiple amides or longer than partial. "
            f"Got: {result}"
        )

    def test_3_amide_chain_names_multiple_amides(self):
        """3-amide chain should reflect more amide connectivity than just one."""
        smiles = "CC(=O)NCC(=O)NCC(=O)O"
        result = name_compound(smiles)

        partial_name = "2-(ethanoylamino)ethanoic acid"
        assert result != partial_name, (
            f"3-amide chain should not produce the single-amide partial name. "
            f"Got: {result}"
        )

    def test_single_amide_n_acyl_unchanged(self):
        """Single amide N-acetylglycine should still produce correct name."""
        smiles = "CC(=O)NCC(=O)O"
        result = name_compound(smiles)

        # Should still be the correct single-amide name (P-66.1.1.4.3
        # method (1): acetamido is the preferred prefix)
        assert result == "2-acetamidoethanoic acid", (
            f"Single amide should keep its correct name. Got: {result}"
        )

    def test_peptide_with_terminal_nh2_unchanged(self):
        """Gly-Gly (terminal NH2) routes through the peptide dispatch. v38: its
        PIN is the SUBSTITUTIVE form (V38-PEPTIDE-PIN-VERDICT.md; peptide names
        are non-PIN). Full-InChIKey round-trip verified."""
        smiles = "NCC(=O)NCC(=O)O"
        result = name_compound(smiles)

        assert result == "(2-aminoacetamido)acetic acid", (
            f"Gly-Gly should produce the substitutive PIN. Got: {result}"
        )

    def test_2_amide_chain_without_terminal_nh2(self):
        """2-amide chain (N-acetyl dipeptide backbone) should reflect more
        amide content than just the first bond."""
        smiles = "CC(=O)NCC(=O)NCC(=O)O"
        result = name_compound(smiles)

        # This is the same as the 3-amide test but naming it "2-amide"
        # because there are 2 internal amide bonds plus the terminal acid
        partial_name = "2-(ethanoylamino)ethanoic acid"
        assert result != partial_name, (
            f"Multi-amide chain should not produce the single-amide partial name. "
            f"Got: {result}"
        )
