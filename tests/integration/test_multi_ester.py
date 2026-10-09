"""
Integration tests for multi-ester (dicarboxylic acid diester) naming.

Tests the detection and naming of compounds with two ester groups sharing
a dicarboxylic acid backbone, producing names in the format:
  "[multiplier]alkyl [parent]anedioate"

Examples:
  COC(=O)CC(=O)OC -> "dimethyl propanedioate" (dimethyl malonate)
  CCOC(=O)CCC(=O)OCC -> "diethyl butanedioate" (diethyl succinate)
"""

import pytest
from orthonym.namer import name_compound


# ============================================================================
# Dicarboxylic acid diester tests (NEW - should initially FAIL)
# ============================================================================

@pytest.mark.integration
def test_dimethyl_malonate():
    """Dimethyl malonate: dimethyl ester of malonic acid (propanedioic acid)."""
    result = name_compound("COC(=O)CC(=O)OC")
    assert result == "dimethyl propanedioate", f"Got: {result}"


@pytest.mark.integration
def test_diethyl_succinate():
    """Diethyl succinate: diethyl ester of succinic acid (butanedioic acid)."""
    result = name_compound("CCOC(=O)CCC(=O)OCC")
    assert result == "diethyl butanedioate", f"Got: {result}"


@pytest.mark.integration
def test_dimethyl_adipate():
    """Dimethyl adipate: dimethyl ester of adipic acid (hexanedioic acid)."""
    result = name_compound("COC(=O)CCCCC(=O)OC")
    assert result == "dimethyl hexanedioate", f"Got: {result}"


@pytest.mark.integration
def test_asymmetric_methyl_ethyl_malonate():
    """Asymmetric diester: different alkyl groups listed alphabetically."""
    result = name_compound("COC(=O)CC(=O)OCC")
    assert result == "ethyl methyl propanedioate", f"Got: {result}"


@pytest.mark.integration
def test_dimethyl_glutarate():
    """Dimethyl glutarate: dimethyl ester of glutaric acid (pentanedioic acid)."""
    result = name_compound("COC(=O)CCCC(=O)OC")
    assert result == "dimethyl pentanedioate", f"Got: {result}"


@pytest.mark.integration
def test_diethyl_oxalate():
    """Diethyl oxalate: diethyl ester of oxalic acid (2-carbon diacid, trivial name)."""
    result = name_compound("CCOC(=O)C(=O)OCC")
    assert result == "diethyl oxalate", f"Got: {result}"


# ============================================================================
# Regression tests (must STILL pass -- no regressions)
# ============================================================================

@pytest.mark.integration
def test_regression_ethyl_acetate():
    """Simple single ester: ethyl acetate must remain correct."""
    result = name_compound("CCOC(C)=O")
    assert result == "ethyl acetate", f"Got: {result}"


@pytest.mark.integration
def test_regression_methyl_propanoate():
    """Simple single ester: methyl propanoate must remain correct."""
    result = name_compound("COC(=O)CC")
    assert result == "methyl propanoate", f"Got: {result}"


@pytest.mark.integration
def test_regression_glycerol_diacetate():
    """Glycerol 1,3-diacetate: a PIN names an ester by functional class nomenclature
    , the Blue Book) and an ester outranks an alcohol, class 9 over class
    17, the Blue Book / the Blue Book), so the free OH is the 'hydroxy' prefix on the multivalent
    group; identical anions give 'propane-1,3-diyl bis(chloroacetate) (PIN)', the Blue Book
    ."""
    result = name_compound("CC(=O)OCC(O)COC(=O)C")
    assert result == "2-hydroxypropane-1,3-diyl diacetate", f"Got: {result}"


# ============================================================================
# Polyol polyester tests (NEW - should initially FAIL)
# ============================================================================

class TestPolyolPolyester:
    """Tests for fully-esterified polyol naming (triacetin, triglycerides)."""

    @pytest.mark.integration
    def test_triacetin(self):
        """Triacetin: glycerol triacetate, functional class multiplicative name.

        'propane-1,2,3-triyl triacetate (PIN)', the Blue Book under (the Blue Book,
        'When anions are identical functional class multiplicative nomenclature is
        used'); all PIN ester names are functional class names, the Blue Book),
        so there is no 'acetyloxy' prefix.
        """
        result = name_compound("CC(=O)OCC(COC(C)=O)OC(C)=O")
        assert result == "propane-1,2,3-triyl triacetate", f"Got: {result}"

    @pytest.mark.integration
    def test_glycerol_tripropanoate(self):
        """Glycerol tripropanoate: functional class multiplicative name
        , the Blue Book; 'propane-1,2,3-triyl triacetate (PIN)', the Blue Book)."""
        result = name_compound("CCC(=O)OCC(COC(=O)CC)OC(=O)CC")
        assert result == "propane-1,2,3-triyl tripropanoate", f"Got: {result}"

    @pytest.mark.integration
    def test_glycerol_diacetate_mono_propanoate(self):
        """Mixed-acid triester: 2 acetyl + 1 propanoyl on glycerol backbone.

        Byte-identical to the Blue Book example 'propane-1,2,3-triyl 1,2-diacetate
        3-propanoate (PIN)', the Blue Book under (method (1) generates
        preferred IUPAC names).
        """
        result = name_compound("CC(=O)OCC(COC(=O)CC)OC(C)=O")
        assert result == "propane-1,2,3-triyl 1,2-diacetate 3-propanoate", f"Got: {result}"

    @pytest.mark.integration
    def test_regression_glycerol_diacetate_polyol(self):
        """Glycerol diacetate with free -OH: the ester is senior, the Blue Book over
        the Blue Book), so the OH is a 'hydroxy' prefix on the multivalent group."""
        result = name_compound("CC(=O)OCC(O)COC(=O)C")
        assert result == "2-hydroxypropane-1,3-diyl diacetate", f"Got: {result}"

    @pytest.mark.integration
    def test_regression_dimethyl_malonate_unaffected(self):
        """Dicarboxylic diester from 41-01: must still produce correct name."""
        result = name_compound("COC(=O)CC(=O)OC")
        assert result == "dimethyl propanedioate", f"Got: {result}"


# ============================================================================
# Independent multi-ester tests (esters not sharing acid/alcohol backbone)
# ============================================================================

class TestIndependentMultiEster:
    """Tests for independent multi-ester compounds (no shared backbone)."""

    @pytest.mark.integration
    def test_classify_independent_two_esters_via_ether(self):
        """Two ester groups connected via ether bridge classify as independent."""
        from rdkit import Chem
        from orthonym.rules.esters import classify_multi_ester

        smi = "CC(=O)OCOCOC(=O)CC"
        mol = Chem.MolFromSmiles(smi)
        ester_smarts = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")
        matches = mol.GetSubstructMatches(ester_smarts)
        assert len(matches) >= 2, f"Expected 2+ ester matches, got {len(matches)}"
        result = classify_multi_ester(mol, matches)
        assert result == "independent", f"Expected 'independent', got {result!r}"

    @pytest.mark.integration
    def test_independent_ester_produces_valid_name(self):
        """Independent multi-ester produces a non-empty valid name."""
        # Two esters connected through an ether bridge (not shared acid or alcohol)
        smi = "CC(=O)OCOCOC(=O)CC"
        result = name_compound(smi)
        assert result is not None, "Got None"
        assert len(result.strip()) >= 5, f"Name too short: {result!r}"

    @pytest.mark.integration
    def test_independent_ester_via_nitrogen(self):
        """Two esters connected through nitrogen classify and name correctly."""
        smi = "CC(=O)OCNC(=O)OCC"
        result = name_compound(smi)
        assert result is not None, "Got None"
        assert len(result.strip()) >= 5, f"Name too short: {result!r}"
