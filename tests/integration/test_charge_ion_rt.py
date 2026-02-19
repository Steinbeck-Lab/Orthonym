"""
Phase 64: Charge and Ion Naming - Regression Tests

Phase 64 Accounting:
  Total compounds addressed: 31 of 39
  Categories fixed:
    A1 (single anions): 6 compounds -- -oate suffix via neutralize-then-name
    A2 (single cations): 1 compound -- 2-methylpropan-2-aminium correct at v7
    A3 (salts): 4 compounds -- guanidinium, hydrogen prefix, hydrochloride
    A4 (multi-charged anions): 4 compounds -- phosphate dianion neutralize path
    B1 (phospholipid zwitterions): 8 compounds -- routed to neutral pipeline
    B2 (amino acid zwitterions): 8 compounds -- retained names + neutral naming
  Names improved: 27 (non-empty, non-garbled names produced)
  OPSIN parse: 9 (OPSIN accepts the name from Plan 01 fixes)
  InChI RT: 4 (full round-trip match from Plan 01 fixes)
  Still failing: 8 (steroid vocabulary, large molecule naming, metal complexes)

Tests for compounds fixed in Phase 64:
- Plan 01: A1 single anions, A3 salts (guanidinium, hydrogen prefix, etc.)
- Plan 02: A4 phosphate dianions, B1 phospholipid zwitterions, amino acid zwitterions

Each test verifies a specific compound produces the expected name pattern.
When names improve in future phases, update the expected values.
"""

import pytest

from orthonym import name_compound


# ---------------------------------------------------------------------------
# A1: Single Anion Compounds
# These should produce -oate or -ate names for carboxylate anions.
# ---------------------------------------------------------------------------

class TestSingleAnionNaming:
    """Test single carboxylate anion compounds produce -oate names."""

    def test_a1_2_amino_acid_anion(self):
        """A1#2: amino acid carboxylate should produce -oate (not aminide)."""
        result = name_compound('CC[C@H](C)[C@H](N)C(=O)[O-]')
        assert 'oate' in result, f"Expected -oate suffix, got: {result}"
        assert 'aminide' not in result, f"Should not be aminide: {result}"

    def test_a1_3_ester_bearing_anion(self):
        """A1#3: ester-bearing carboxylate should produce -oate."""
        result = name_compound('C[C@H](CC(=O)[O-])OC(=O)C[C@@H](C)O')
        assert 'oate' in result, f"Expected -oate suffix, got: {result}"
        assert result != '', "Should not be empty"

    def test_a1_4_phenoxy_anion(self):
        """A1#4: phenoxybutanoate anion."""
        result = name_compound('O=C([O-])CCCOc1ccc(Cl)cc1Cl')
        assert 'oate' in result or 'ate' in result, \
            f"Expected -oate/-ate suffix, got: {result}"

    def test_a1_6_poly_anion_dioate(self):
        """A1#6: poly-anion should produce -dioate (not -dioic acid)."""
        result = name_compound('O=C([O-])CC=CC(=O)C(=O)[O-]')
        assert 'dioate' in result, f"Expected -dioate suffix, got: {result}"
        assert 'dioic acid' not in result, \
            f"Should be -dioate not -dioic acid: {result}"

    def test_a1_7_poly_anion_pentanedioate(self):
        """A1#7: poly-anion pentanedioate."""
        result = name_compound('O=C([O-])C(=O)C[C@H](O)C(=O)[O-]')
        assert 'dioate' in result, f"Expected -dioate suffix, got: {result}"
        assert 'dioic acid' not in result, \
            f"Should be -dioate not -dioic acid: {result}"

    def test_a1_8_poly_anion_hexanedioate(self):
        """A1#8: poly-anion hexanedioate with stereo."""
        result = name_compound(
            'O=C([O-])[C@@H](O)[C@H](O)[C@@H](O)[C@H](O)C(=O)[O-]'
        )
        assert 'dioate' in result, f"Expected -dioate suffix, got: {result}"
        assert 'dioic acid' not in result, \
            f"Should be -dioate not -dioic acid: {result}"


# ---------------------------------------------------------------------------
# A3: Salt Compounds
# These should produce correct "cation anion" format names.
# ---------------------------------------------------------------------------

class TestSaltNaming:
    """Test salt compounds produce correct IUPAC salt names."""

    def test_a3_1_sodium_hydrogen_fumarate(self):
        """A3#1: partial salt should have 'hydrogen' prefix."""
        result = name_compound('O=C([O-])/C=C/C(=O)O.[Na+]')
        assert 'sodium' in result, f"Expected 'sodium', got: {result}"
        assert 'hydrogen' in result, \
            f"Expected 'hydrogen' prefix for partial salt, got: {result}"

    def test_a3_2_guanidinium_salt(self):
        """A3#2: guanidinium should be correctly identified."""
        result = name_compound('NC(N)=[NH2+].O=C([O-])C(=O)O')
        assert 'guanidinium' in result, \
            f"Expected 'guanidinium', got: {result}"

    def test_a3_3_sodium_amino_acid_salt(self):
        """A3#3: sodium + amino acid anion salt."""
        result = name_compound(
            '[NH3+][C@@H](CCC(=O)[O-])C(=O)[O-].[Na+]'
        )
        assert 'sodium' in result, f"Expected 'sodium', got: {result}"
        assert 'amino' in result or 'oate' in result, \
            f"Expected amino acid anion name, got: {result}"

    def test_a3_4_hydrochloride_salt(self):
        """A3#4: neutral organic + H+ + Cl- = hydrochloride."""
        result = name_compound(
            'COc1ccc(C(CN(C)C)C2(O)CCCCC2)cc1.[Cl-].[H+]'
        )
        assert 'hydrochloride' in result, \
            f"Expected 'hydrochloride', got: {result}"

    def test_basic_sodium_acetate(self):
        """Basic salt: sodium acetate must not regress."""
        result = name_compound('[Na+].[O-]C(C)=O')
        assert result == 'sodium acetate', f"Got: {result}"

    def test_basic_potassium_chloride(self):
        """Basic salt: potassium chloride must not regress."""
        result = name_compound('[K+].[Cl-]')
        assert result == 'potassium chloride', f"Got: {result}"

    def test_basic_ammonium_chloride(self):
        """Basic salt: ammonium chloride must not regress."""
        result = name_compound('[NH4+].[Cl-]')
        assert result == 'ammonium chloride', f"Got: {result}"

    def test_guanidinium_retained_name(self):
        """Guanidinium cation lookup works correctly."""
        from orthonym.data.ion_retained_names import get_cation_name
        assert get_cation_name('NC(N)=[NH2+]') == 'guanidinium'


# ---------------------------------------------------------------------------
# Retained ion name stability
# ---------------------------------------------------------------------------

class TestRetainedIonNames:
    """Ensure retained ion names are stable after changes."""

    def test_acetate(self):
        assert name_compound('CC(=O)[O-]') == 'acetate'

    def test_benzoate(self):
        assert name_compound('O=C([O-])c1ccccc1') == 'benzoate'

    def test_naphthoate(self):
        result = name_compound('O=C([O-])c1ccc2ccccc2c1')
        assert result == 'naphthoate', f"Got: {result}"

    def test_4_chlorobenzoate(self):
        result = name_compound('O=C([O-])c1ccc(Cl)cc1')
        assert result == '4-chlorobenzoate', f"Got: {result}"

    def test_oxalate(self):
        result = name_compound('O=C([O-])C(=O)[O-]')
        assert result == 'oxalate', f"Got: {result}"


# ---------------------------------------------------------------------------
# A4: Multi-Charged Anion Compounds (Plan 02)
# Phosphate dianions -- neutralize O- to OH, name neutral form.
# ---------------------------------------------------------------------------

class TestMultiChargedAnionNaming:
    """Test phosphate dianion compounds produce names via neutralize path."""

    def test_a4_1_glycerol_3_phosphate(self):
        """A4#1: glycerol-3-phosphate dianion."""
        result = name_compound('O=P([O-])([O-])OC[C@@H](O)CO')
        assert result, "Should produce a name"
        assert 'phosph' in result.lower(), \
            f"Should contain phosph reference: {result}"

    def test_a4_3_phosphate_ester_dianion(self):
        """A4#3: phosphate ester dianion on long chain."""
        result = name_compound(
            'O=C(O)CCCCCCCCCCCCCCCCCC(=O)OCCOP(=O)([O-])[O-]'
        )
        assert result, "Should produce a name"
        # Should not be empty -- the poly-anion path neutralizes then names
        assert len(result) > 5, f"Name too short: {result}"


# ---------------------------------------------------------------------------
# B1: Phospholipid Zwitterion Compounds (Plan 02)
# These are now routed to the neutral pipeline instead of producing
# garbled "ammonium Xanoate" names.
# ---------------------------------------------------------------------------

class TestPhospholipidZwitterionNaming:
    """Test phospholipid zwitterions produce structural names, not garbled."""

    def test_b1_1_dipalmitoyl_pc(self):
        """B1: dipalmitoyl phosphatidylcholine zwitterion."""
        result = name_compound(
            'CCCCCCCCCCCCCCCC(=O)OC(COC(=O)CCCCCCCCCCCCCCC)'
            'COP(=O)([O-])OCC[N+](C)(C)C'
        )
        assert result, "Should produce a name"
        assert 'ammonium' not in result.lower(), \
            f"Should NOT produce ammonium name: {result}"

    def test_b1_2_dihexacosanoyl_pc(self):
        """B1: dihexacosanoyl phosphatidylcholine zwitterion."""
        result = name_compound(
            'CCCCCCCCCCCCCCCCCCCCCCCCCC(=O)OC(COC(=O)'
            'CCCCCCCCCCCCCCCCCCCCCCCCC)COP(=O)([O-])OCC[N+](C)(C)C'
        )
        assert result, "Should produce a name"
        assert 'ammonium' not in result.lower(), \
            f"Should NOT produce ammonium name: {result}"


# ---------------------------------------------------------------------------
# Amino Acid Zwitterion Naming (Plan 02)
# Amino acid zwitterions should produce retained trivial names or
# neutral-form systematic names, NOT ionic "2-azaniumylXanoate".
# ---------------------------------------------------------------------------

class TestAminoAcidZwitterionNaming:
    """Test amino acid zwitterions produce correct names."""

    def test_glycine_zwitterion(self):
        """Glycine zwitterion should produce 'glycine'."""
        result = name_compound('[NH3+]CC([O-])=O')
        assert result == 'glycine', f"Expected glycine, got: {result}"

    def test_l_alanine_zwitterion(self):
        """L-alanine zwitterion should produce 'L-alanine'."""
        result = name_compound('C[C@H]([NH3+])C([O-])=O')
        assert result == 'L-alanine', f"Expected L-alanine, got: {result}"

    def test_l_valine_zwitterion(self):
        """L-valine zwitterion."""
        result = name_compound('CC(C)[C@@H]([NH3+])C([O-])=O')
        assert result == 'L-valine', f"Expected L-valine, got: {result}"

    def test_l_leucine_zwitterion(self):
        """L-leucine zwitterion."""
        result = name_compound('CC(C)C[C@@H]([NH3+])C([O-])=O')
        assert result == 'L-leucine', f"Expected L-leucine, got: {result}"

    def test_l_proline_zwitterion(self):
        """L-proline zwitterion."""
        result = name_compound('O=C([O-])[C@@H]1CCC[NH2+]1')
        assert result == 'L-proline', f"Expected L-proline, got: {result}"

    def test_l_phenylalanine_zwitterion(self):
        """L-phenylalanine zwitterion."""
        result = name_compound('[NH3+][C@@H](Cc1ccccc1)C([O-])=O')
        assert result == 'L-phenylalanine', \
            f"Expected L-phenylalanine, got: {result}"

    def test_l_glutamic_acid_zwitterion(self):
        """L-glutamic acid zwitterion (one COOH protonated)."""
        result = name_compound('[NH3+][C@@H](CCC(=O)O)C(=O)[O-]')
        assert result == 'L-glutamic acid', \
            f"Expected L-glutamic acid, got: {result}"

    def test_l_aspartic_acid_zwitterion(self):
        """L-aspartic acid zwitterion (one COOH protonated)."""
        result = name_compound('[NH3+][C@@H](CC(=O)O)C(=O)[O-]')
        assert result == 'L-aspartic acid', \
            f"Expected L-aspartic acid, got: {result}"

    def test_racemic_alanine_zwitterion(self):
        """Racemic alanine zwitterion."""
        result = name_compound('[NH3+]C(C)C([O-])=O')
        assert result == 'alanine', f"Expected alanine, got: {result}"

    def test_no_azaniumyl_for_amino_acids(self):
        """Amino acid zwitterions should NOT produce azaniumyl ionic names."""
        for smi in [
            '[NH3+]CC([O-])=O',
            'C[C@H]([NH3+])C([O-])=O',
            'CC(C)[C@@H]([NH3+])C([O-])=O',
        ]:
            result = name_compound(smi)
            assert 'azaniumyl' not in result.lower(), \
                f"Should not use ionic form for {smi}: {result}"
