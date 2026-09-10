"""Integration tests for recursive substituent naming (a phase-02).

Tests end-to-end naming of molecules with complex (branched/functionalized)
substituents through the full pipeline, plus unit-level checks on the
utility functions (apply_enclosing_marks, get_bracket_depth, format_substituent_prefix).

Test categories:
1. Regression guards: simple molecules that MUST NOT change
2. Branched alkyl substituent detection on chains
3. Enclosing marks (RSN-02): parentheses, brackets, braces
4. Compound substituent formatting and multipliers (RSN-03)
5. Utility function behavior (is_complex_substituent, format_substituent_prefix)
"""

import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.assembly.naming_utils import (
    apply_enclosing_marks,
    get_bracket_depth,
    is_complex_substituent,
    format_substituent_prefix,
)
from orthonym.assembly.substituent_naming import (
    ATTACH_LOCANT_UNKNOWN,
    _is_linear_alkyl,
    name_substituent_fragment,
    parent_to_prefix,
)


# ============================================================================
# Category 1: Regression Guards -- simple molecules must not change
# ============================================================================

class TestRegressionGuards:
    """Simple molecules that MUST produce the same names as before."""

    @pytest.mark.integration
    def test_ethane(self):
        assert name_compound("CC") == "ethane"

    @pytest.mark.integration
    def test_butane(self):
        assert name_compound("CCCC") == "butane"

    @pytest.mark.integration
    def test_pentane(self):
        assert name_compound("CCCCC") == "pentane"

    @pytest.mark.integration
    def test_2_methylpropane(self):
        assert name_compound("CC(C)C") == "2-methylpropane"

    @pytest.mark.integration
    def test_3_ethylpentane(self):
        assert name_compound("CCC(CC)CC") == "3-ethylpentane"

    @pytest.mark.integration
    def test_2_methylbutane(self):
        assert name_compound("CC(C)CC") == "2-methylbutane"

    @pytest.mark.integration
    def test_2_2_dimethylpropane(self):
        assert name_compound("CC(C)(C)C") == "2,2-dimethylpropane"

    @pytest.mark.integration
    def test_acetic_acid(self):
        assert name_compound("CC(=O)O") == "acetic acid"

    @pytest.mark.integration
    def test_propan_1_ol(self):
        assert name_compound("CCCO") == "propan-1-ol"

    @pytest.mark.integration
    def test_butan_2_one(self):
        assert name_compound("CC(=O)CC") == "butan-2-one"

    @pytest.mark.integration
    def test_2_4_dimethylhexane(self):
        assert name_compound("CC(C)CC(C)CC") == "2,4-dimethylhexane"

    @pytest.mark.integration
    def test_3_methylpentane(self):
        assert name_compound("CCC(C)CC") == "3-methylpentane"

    @pytest.mark.integration
    def test_decane(self):
        assert name_compound("CCCCCCCCCC") == "decane"


# ============================================================================
# Category 2: Branched alkyl substituent detection on chains
# ============================================================================

class TestBranchedAlkylDetection:
    """Test that branched substituents are correctly identified."""

    @pytest.mark.integration
    def test_linear_detection_simple_chain(self):
        """Linear substituent atoms are detected as linear."""
        mol = Chem.MolFromSmiles("CCC(CC)CC")  # 3-ethylpentane
        # ethyl branch atoms [3, 4] are linear
        assert _is_linear_alkyl(mol, [3, 4]) is True

    @pytest.mark.integration
    def test_linear_detection_methyl(self):
        """Single-atom methyl is linear."""
        mol = Chem.MolFromSmiles("CC(C)C")  # 2-methylpropane
        # Each methyl is a single carbon
        assert _is_linear_alkyl(mol, [0]) is True

    @pytest.mark.integration
    def test_branched_detection_tert_butyl(self):
        """tert-butyl fragment (4C with 3 branches) is NOT linear."""
        mol = Chem.MolFromSmiles("CCCC(C)(C)C")
        # Find the quaternary carbon (degree 4)
        for atom in mol.GetAtoms():
            if atom.GetDegree() == 4:
                # tert-butyl fragment includes this atom and its 3 methyl neighbors
                sub = [atom.GetIdx()] + [
                    n.GetIdx() for n in atom.GetNeighbors()
                    if n.GetSymbol() == 'C'
                ]
                # Only include atoms NOT in the main chain attachment
                # For simplicity, test that a 4-carbon star is not linear
                if len(sub) == 4:
                    assert _is_linear_alkyl(mol, sub) is False
                    break

    @pytest.mark.integration
    def test_4_ethyloctane_linear(self):
        """4-ethyloctane: ethyl substituent is linear."""
        result = name_compound("CCCC(CC)CCCC")
        assert result == "4-ethyloctane"

    @pytest.mark.integration
    def test_3_ethyl_2_methylpentane(self):
        """Both ethyl and methyl are simple linear subs on pentane."""
        result = name_compound("CCC(CC)C(C)CC")
        # Should have ethyl and methyl, not a compound sub
        assert "ethyl" in result
        assert "methyl" in result


# ============================================================================
# Category 3: Enclosing Marks (RSN-02)
# ============================================================================

class TestEnclosingMarks:
    """Test IUPAC P-16.5.1.1 enclosing mark nesting."""

    @pytest.mark.integration
    def test_parentheses_depth_0(self):
        assert apply_enclosing_marks("2-methylpropyl", 0) == "(2-methylpropyl)"

    @pytest.mark.integration
    def test_brackets_depth_1(self):
        assert apply_enclosing_marks("2-methylpropyl", 1) == "[2-methylpropyl]"

    @pytest.mark.integration
    def test_braces_depth_2(self):
        assert apply_enclosing_marks("2-methylpropyl", 2) == "{2-methylpropyl}"

    @pytest.mark.integration
    def test_wrap_around_depth_3(self):
        """Depth 3 wraps around to parentheses again."""
        assert apply_enclosing_marks("test", 3) == "(test)"

    @pytest.mark.integration
    def test_get_bracket_depth_none(self):
        assert get_bracket_depth("methyl") == 0

    @pytest.mark.integration
    def test_get_bracket_depth_parens(self):
        assert get_bracket_depth("(2-methylpropyl)") == 1

    @pytest.mark.integration
    def test_get_bracket_depth_brackets(self):
        assert get_bracket_depth("[test]") == 2

    @pytest.mark.integration
    def test_get_bracket_depth_braces(self):
        assert get_bracket_depth("{test}") == 3


# ============================================================================
# Category 4: Compound Substituent Formatting and Multipliers (RSN-03)
# ============================================================================

class TestCompoundSubstituentFormatting:
    """Test format_substituent_prefix with compound substituent names."""

    @pytest.mark.integration
    def test_compound_single_parens(self):
        """Single compound substituent gets parentheses."""
        result = format_substituent_prefix("2-methylpropyl", [3], 1)
        assert result == "3-(2-methylpropyl)"

    @pytest.mark.integration
    def test_compound_bis_multiplier(self):
        """Two identical compound substituents get bis multiplier."""
        result = format_substituent_prefix("2-methylpropyl", [3, 5], 2)
        assert result == "3,5-bis(2-methylpropyl)"

    @pytest.mark.integration
    def test_compound_tris_multiplier(self):
        """Three identical compound substituents get tris multiplier."""
        result = format_substituent_prefix("1-methylethyl", [2, 4, 7], 3)
        assert result == "2,4,7-tris(1-methylethyl)"

    @pytest.mark.integration
    def test_simple_di_multiplier(self):
        """Simple substituents use di multiplier (no parens)."""
        result = format_substituent_prefix("methyl", [2, 3], 2)
        assert result == "2,3-dimethyl"

    @pytest.mark.integration
    def test_simple_single_no_parens(self):
        """Single simple substituent has no parentheses."""
        result = format_substituent_prefix("methyl", [2], 1)
        assert result == "2-methyl"

    @pytest.mark.integration
    def test_isopropyl_no_parens(self):
        """Isopropyl (no digits/hyphens) gets no parentheses."""
        result = format_substituent_prefix("isopropyl", [4], 1)
        assert result == "4-isopropyl"

    @pytest.mark.integration
    def test_sec_butyl_parens(self):
        """sec-butyl is a SIMPLE substituent -> NO enclosing marks (P-16.3.3(b);
        cf. CLAUDE.md '3-tert-butyl-...'). The old parens-for-any-hyphen rule
        was stale."""
        result = format_substituent_prefix("sec-butyl", [3], 1)
        assert result == "3-sec-butyl"

    @pytest.mark.integration
    def test_already_wrapped_no_double(self):
        """A name carrying an UN-closed enclosing mark (parens around oxan-2-yl,
        bare trailing 'oxy') is a compound substituent; per the enclosing-mark
        nesting order (P-16.5.4.1) it escalates to the next bracket level ->
        '1-[(oxan-2-yl)oxy]', not a bare/double paren. Stale expectation."""
        result = format_substituent_prefix("(oxan-2-yl)oxy", [1], 1)
        assert result == "1-[(oxan-2-yl)oxy]"


# ============================================================================
# Category 5: is_complex_substituent Detection
# ============================================================================

class TestIsComplexSubstituent:
    """Test that is_complex_substituent correctly classifies names."""

    @pytest.mark.integration
    def test_simple_methyl(self):
        assert is_complex_substituent("methyl") is False

    @pytest.mark.integration
    def test_simple_ethyl(self):
        assert is_complex_substituent("ethyl") is False

    @pytest.mark.integration
    def test_simple_isopropyl(self):
        assert is_complex_substituent("isopropyl") is False

    @pytest.mark.integration
    def test_complex_1_methylethyl(self):
        assert is_complex_substituent("1-methylethyl") is True

    @pytest.mark.integration
    def test_complex_2_methylpropyl(self):
        assert is_complex_substituent("2-methylpropyl") is True

    @pytest.mark.integration
    def test_complex_sec_butyl(self):
        # P-16.3.3(b): sec-butyl is a SIMPLE substituent (the italic 'sec-' is
        # not a locant/complexity marker), so it is NOT complex.
        assert is_complex_substituent("sec-butyl") is False

    @pytest.mark.integration
    def test_complex_tert_butyl(self):
        # P-16.3.3(b): tert-butyl is SIMPLE (cf. CLAUDE.md '3-tert-butyl-...').
        assert is_complex_substituent("tert-butyl") is False


# ============================================================================
# Category 6: parent_to_prefix Conversion
# ============================================================================

class TestParentToPrefix:
    """Test parent name to substituent prefix conversion."""

    @pytest.mark.integration
    def test_alkane_to_yl(self):
        assert parent_to_prefix("propane", 3, attach_locant=ATTACH_LOCANT_UNKNOWN) == "propyl"

    @pytest.mark.integration
    def test_branched_alkane_to_yl(self):
        assert parent_to_prefix("2-methylpropane", 3, attach_locant=ATTACH_LOCANT_UNKNOWN) == "2-methylpropyl"

    @pytest.mark.integration
    def test_alcohol_to_hydroxy(self):
        assert parent_to_prefix("ethanol", 2, attach_locant=ATTACH_LOCANT_UNKNOWN) == "hydroxyethyl"

    @pytest.mark.integration
    def test_locanted_alcohol_to_hydroxy(self):
        # residue Task A: (name, count) is not injective over fragments --
        # '-CH2CH2CH2OH' and '-CH(OH)CH2CH3' both cap to 'propan-1-ol' with
        # count 3, and OPSIN 2.9.0 makes the single old answer EXACT for one
        # and a DIFFERENT MOLECULE for the other. P-46.1.8 (the Blue Book) can only
        # be honoured by a caller holding the molecule, so this declines; the
        # structural namers still give the right whole-molecule name.
        assert parent_to_prefix("propan-2-ol", 3, attach_locant=ATTACH_LOCANT_UNKNOWN) is None

    @pytest.mark.integration
    def test_ketone_to_oxo(self):
        assert parent_to_prefix("butan-2-one", 4, attach_locant=ATTACH_LOCANT_UNKNOWN) is None

    @pytest.mark.integration
    def test_aldehyde_to_oxo(self):
        # a phase MBA-02: the absorbed -CHO carbon sits at the terminus opposite
        # the attachment (= chain_length), so oxo is at C3, not the acyl C1.
        # '1-oxopropyl' is NOT a preferred IUPAC prefix (the Blue Book Table-28.1 note m).
        # Task A: `chain_length` is a COUNT, not a proof of chain length -- on
        # '2-methylpropanal' (count 4, stem 'prop') it spliced locant 4 onto a
        # three-position stem. Declines; the end-to-end name is unaffected.
        assert parent_to_prefix("propanal", 3, attach_locant=ATTACH_LOCANT_UNKNOWN) is None

    @pytest.mark.integration
    def test_amine_to_amino(self):
        assert parent_to_prefix("propan-1-amine", 3, attach_locant=ATTACH_LOCANT_UNKNOWN) is None

    @pytest.mark.integration
    def test_carboxylic_acid_to_carboxy(self):
        assert parent_to_prefix("butanoic acid", 4, attach_locant=ATTACH_LOCANT_UNKNOWN) is None
