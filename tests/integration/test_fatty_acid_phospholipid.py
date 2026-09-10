"""
Integration tests for a phase-03: Fatty acid retained names and phospholipid FG detection.

Tests:
- Fatty acid trivial name lookups (acylate, acyloxy)
- Fatty acid ester naming (methyl palmitate, methyl stearate)
- Phospholipid FG detection (phosphate_monoester vs phosphonic_acid)
- Regression: existing simple esters unchanged
"""
import pytest
from rdkit import Chem
from orthonym.namer import name_compound
from orthonym.data.trivial_acids import get_acylate_name
from orthonym.rules.esters import get_acyloxy_prefix
from orthonym.perception.functional_groups import detect_functional_groups


# ===========================================================================
# Fatty acid trivial name lookups
# ===========================================================================


@pytest.mark.integration
class TestFattyAcidLookups:
    """Fatty acid retained names in acylate and acyloxy tables.

    NOTE : these assert a SPELLING conversion, stem -> acyl word,
    and are NOT an endorsement of the trivial name as a PIN.
    (the Blue Book) retains these "for general nomenclature with
    functionalization... the formation of esters leads to names such as methyl
    butyrate", so the conversion itself is legitimate general nomenclature.
    The PIN decision belongs at the STEM PRODUCER, not here: no PIN-path
    producer hands 'palmitic' to this table any more. See
    TestFattyAcidEsterNaming below for what the default path must emit.
    """

    def test_get_acylate_palmitic(self):
        """Palmitic -> palmitate."""
        assert get_acylate_name("palmitic") == "palmitate"

    def test_get_acylate_stearic(self):
        """Stearic -> stearate."""
        assert get_acylate_name("stearic") == "stearate"

    def test_get_acylate_lauric(self):
        """Lauric -> laurate."""
        assert get_acylate_name("lauric") == "laurate"

    def test_get_acylate_myristic(self):
        """Myristic -> myristate."""
        assert get_acylate_name("myristic") == "myristate"

    def test_get_acylate_oleic(self):
        """Oleic -> oleate."""
        assert get_acylate_name("oleic") == "oleate"

    def test_get_acylate_arachidic(self):
        """Arachidic -> arachidate."""
        assert get_acylate_name("arachidic") == "arachidate"

    def test_get_acyloxy_palmitic(self):
        """Palmitic -> palmitoyloxy."""
        assert get_acyloxy_prefix("palmitic") == "palmitoyloxy"

    def test_get_acyloxy_stearic(self):
        """Stearic -> stearoyloxy."""
        assert get_acyloxy_prefix("stearic") == "stearoyloxy"

    def test_get_acyloxy_lauric(self):
        """Lauric -> lauroyloxy."""
        assert get_acyloxy_prefix("lauric") == "lauroyloxy"

    def test_get_acyloxy_myristic(self):
        """Myristic -> myristoyloxy."""
        assert get_acyloxy_prefix("myristic") == "myristoyloxy"

    def test_get_acyloxy_oleic(self):
        """Oleic -> oleoyloxy."""
        assert get_acyloxy_prefix("oleic") == "oleoyloxy"

    def test_get_acyloxy_arachidic(self):
        """Arachidic -> arachidoyloxy."""
        assert get_acyloxy_prefix("arachidic") == "arachidoyloxy"


# ===========================================================================
# Fatty acid ester naming (end-to-end)
# ===========================================================================


@pytest.mark.integration
class TestFattyAcidEsterNaming:
    """Methyl esters of saturated fatty acids use the SYSTEMATIC acid stem.

    This class formerly asserted the opposite -- "Methyl esters of fatty acids
    use trivial acid names" -- and so codified a non-PIN emission as the spec.
    Corrected by.

     "Retained names as preferred IUPAC names" (the Blue Book)
    -- "Only the following five carboxylic acids retained names and are also
    preferred IUPAC names": formic, oxalic, acetic, benzoic, oxamic. No fatty
    acid is among them.

     "Systematic names" (heading:29858, rule:29860) -- "Except for
    formic acid, acetic acid, oxalic acid (see, and oxamic acid
    (see, systematically formed names are preferred IUPAC names;
    the names given in are retained names for use in general
    nomenclature."

    The book prints the marker on the systematic side::29787 "palmitic acid
    hexadecanoic acid (PIN)",:29791 "stearic acid octadecanoic acid (PIN)".

     (:31659) -- "All preferred IUPAC names for esters are named by
    functional class nomenclature" -- takes the acyl word from the PIN acid,
    which is why it prints "ethyl methyl butanedioate (PIN)" and not
    "succinate" even though succinic acid is a retained name.

    The ACID path already emitted "hexadecanoic acid" for the same chain, so
    the two paths contradicted each other; these assertions were the wrong half.
    """

    def test_methyl_hexadecanoate_name(self):
        """C16:0 methyl ester -- PIN stem, not 'palmitate'."""
        result = name_compound("CCCCCCCCCCCCCCCC(=O)OC")
        assert result == "methyl hexadecanoate", f"got: {result!r}"

    def test_methyl_octadecanoate_name(self):
        """C18:0 methyl ester -- PIN stem, not 'stearate'."""
        result = name_compound("CCCCCCCCCCCCCCCCCC(=O)OC")
        assert result == "methyl octadecanoate", f"got: {result!r}"

    def test_methyl_dodecanoate_name(self):
        """C12:0 methyl ester -- PIN stem, not 'laurate'."""
        result = name_compound("CCCCCCCCCCCC(=O)OC")
        assert result == "methyl dodecanoate", f"got: {result!r}"

    def test_methyl_tetradecanoate_name(self):
        """C14:0 methyl ester -- PIN stem, not 'myristate'."""
        result = name_compound("CCCCCCCCCCCCCC(=O)OC")
        assert result == "methyl tetradecanoate", f"got: {result!r}"

    def test_regression_methyl_acetate(self):
        """Methyl acetate is unchanged."""
        assert name_compound("COC(C)=O") == "methyl acetate"

    def test_regression_ethyl_propanoate(self):
        """Ethyl propanoate is unchanged."""
        assert name_compound("CCOC(=O)CC") == "ethyl propanoate"


# ===========================================================================
# Phospholipid functional group detection
# ===========================================================================


@pytest.mark.integration
class TestPhospholipidFGDetection:
    """Phosphate monoester detection suppresses phosphonic_acid for C-O-P bonds."""

    def test_glycerol_phosphate_detects_monoester(self):
        """Glycerol phosphate: C-O-P(=O)(OH)2 detects phosphate_monoester.
        Note: phosphonic_acid also matches (same P atom, different bond interpretation)
        and is intentionally kept for backward-compatible naming (phosphono prefix)."""
        mol = Chem.MolFromSmiles("OCC(O)COP(=O)(O)O")
        fgs = detect_functional_groups(mol)
        assert "phosphate_monoester" in fgs, (
            f"Expected phosphate_monoester in FGs: {list(fgs.keys())}"
        )

    def test_phospholipid_diacetate_detects_monoester(self):
        """Phospholipid diacetate: detects phosphate_monoester for C-O-P bond."""
        mol = Chem.MolFromSmiles("CC(=O)OCC(COP(=O)(O)O)OC(=O)C")
        fgs = detect_functional_groups(mol)
        assert "phosphate_monoester" in fgs, (
            f"Expected phosphate_monoester in FGs: {list(fgs.keys())}"
        )

    def test_phosphonic_acid_direct_cp_bond(self):
        """True phosphonic acid: direct C-P bond is still detected as phosphonic_acid."""
        mol = Chem.MolFromSmiles("CP(=O)(O)O")  # methylphosphonic acid
        fgs = detect_functional_groups(mol)
        assert "phosphonic_acid" in fgs, (
            f"Expected phosphonic_acid for direct C-P bond: {list(fgs.keys())}"
        )
        assert "phosphate_monoester" not in fgs, (
            f"phosphate_monoester should NOT match direct C-P bond: {list(fgs.keys())}"
        )

    def test_phosphate_diester_detected(self):
        """Phosphate diester: C-O-P(=O)(OH)(O-C) is phosphate_diester."""
        mol = Chem.MolFromSmiles("COP(=O)(O)OC")
        fgs = detect_functional_groups(mol)
        assert "phosphate_diester" in fgs, (
            f"Expected phosphate_diester in FGs: {list(fgs.keys())}"
        )

    def test_phosphate_triester_detected(self):
        """Phosphate triester: all three O's have C attached."""
        mol = Chem.MolFromSmiles("COP(=O)(OC)OC")
        fgs = detect_functional_groups(mol)
        assert "phosphate_triester" in fgs, (
            f"Expected phosphate_triester in FGs: {list(fgs.keys())}"
        )
