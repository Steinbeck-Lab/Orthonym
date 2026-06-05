"""
Tests for phosphorus SMARTS pattern detection in functional_groups.py.

Verifies all 9 phosphorus-related SMARTS patterns have correct positive and
negative detection, including the phosphate specificity chain
(triester > diester > monoester) via collision resolution.

Requirement: PERC-06
"""

import pytest
from rdkit import Chem
from orthonym.perception.functional_groups import detect_functional_groups


# ---------------------------------------------------------------------------
# Positive detection: each phosphorus SMARTS should match its target molecule
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "smiles, expected_fg, description",
    [
        # Phosphonic acid: R-P(=O)(OH)2
        ("CP(=O)(O)O", "phosphonic_acid", "methylphosphonic acid"),
        # Phosphinic acid: R2P(=O)(OH) — two C neighbors + one OH
        ("CP(=O)(C)O", "phosphinic_acid", "dimethylphosphinic acid"),
        # Phosphate monoester: R-O-P(=O)(OH)2
        ("COP(=O)(O)O", "phosphate_monoester", "methyl dihydrogen phosphate"),
        # Phosphate diester: (RO)2P(=O)(OH)
        ("COP(=O)(OC)O", "phosphate_diester", "dimethyl hydrogen phosphate"),
        # Phosphate triester: (RO)3P(=O)
        ("COP(=O)(OC)OC", "phosphate_triester", "trimethyl phosphate"),
        # Primary phosphine: R-PH2
        ("CP", "primary_phosphine", "methylphosphine"),
        # Secondary phosphine: R2PH
        ("CPC", "secondary_phosphine", "dimethylphosphine"),
        # Tertiary phosphine: R3P
        ("CP(C)C", "tertiary_phosphine", "trimethylphosphine"),
        # Phosphine oxide: R3P(=O)
        ("CP(=O)(C)C", "phosphine_oxide", "trimethylphosphine oxide"),
    ],
    ids=lambda val: val if isinstance(val, str) and len(val) > 10 else None,
)
def test_phosphorus_positive_detection(smiles, expected_fg, description):
    """Each phosphorus SMARTS pattern detects its target compound."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Invalid SMILES: {smiles}"
    fgs = detect_functional_groups(mol)
    assert expected_fg in fgs, (
        f"{description} ({smiles}): expected '{expected_fg}' in detected FGs, "
        f"got {sorted(fgs.keys())}"
    )


# ---------------------------------------------------------------------------
# Negative detection: phosphorus patterns should NOT fire on non-P molecules
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "smiles, excluded_fg, description",
    [
        # Trimethylamine should not match any phosphorus pattern
        ("CN(C)C", "primary_phosphine", "trimethylamine vs primary phosphine"),
        ("CN(C)C", "secondary_phosphine", "trimethylamine vs secondary phosphine"),
        ("CN(C)C", "tertiary_phosphine", "trimethylamine vs tertiary phosphine"),
        ("CN(C)C", "phosphine_oxide", "trimethylamine vs phosphine oxide"),
        # Carboxylic acid should not match phosphonic acid
        ("CC(=O)O", "phosphonic_acid", "acetic acid vs phosphonic acid"),
        # Ester should not match phosphate ester
        ("CC(=O)OC", "phosphate_monoester", "methyl acetate vs phosphate monoester"),
        ("CC(=O)OC", "phosphate_diester", "methyl acetate vs phosphate diester"),
        ("CC(=O)OC", "phosphate_triester", "methyl acetate vs phosphate triester"),
        # Sulfonic acid should not match phosphonic acid
        ("CS(=O)(=O)O", "phosphonic_acid", "methanesulfonic acid vs phosphonic acid"),
        # Simple ether should not match phosphate ester
        ("COC", "phosphate_monoester", "dimethyl ether vs phosphate monoester"),
        # Simple ketone should not match phosphine oxide
        ("CC(=O)C", "phosphine_oxide", "acetone vs phosphine oxide"),
    ],
)
def test_phosphorus_negative_detection(smiles, excluded_fg, description):
    """Non-phosphorus molecules should not trigger phosphorus FG patterns."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Invalid SMILES: {smiles}"
    fgs = detect_functional_groups(mol)
    assert excluded_fg not in fgs, (
        f"{description} ({smiles}): '{excluded_fg}' should NOT be detected, "
        f"but found {fgs.get(excluded_fg, [])}"
    )


# ---------------------------------------------------------------------------
# Phosphate specificity chain: triester > diester > monoester
# ---------------------------------------------------------------------------

class TestPhosphateSpecificityChain:
    """After collision resolution, more-specific phosphate ester suppresses less-specific."""

    def test_triester_suppresses_diester_and_monoester(self):
        """Trimethyl phosphate should detect ONLY triester, not diester or monoester."""
        mol = Chem.MolFromSmiles("COP(=O)(OC)OC")
        assert mol is not None
        fgs = detect_functional_groups(mol)
        assert "phosphate_triester" in fgs, (
            f"Expected phosphate_triester, got {sorted(fgs.keys())}"
        )
        assert "phosphate_diester" not in fgs, (
            "phosphate_diester should be suppressed by triester collision resolution"
        )
        assert "phosphate_monoester" not in fgs, (
            "phosphate_monoester should be suppressed by triester collision resolution"
        )

    def test_diester_suppresses_monoester(self):
        """Dimethyl hydrogen phosphate should detect ONLY diester, not monoester."""
        mol = Chem.MolFromSmiles("COP(=O)(OC)O")
        assert mol is not None
        fgs = detect_functional_groups(mol)
        assert "phosphate_diester" in fgs, (
            f"Expected phosphate_diester, got {sorted(fgs.keys())}"
        )
        assert "phosphate_monoester" not in fgs, (
            "phosphate_monoester should be suppressed by diester collision resolution"
        )

    def test_monoester_alone(self):
        """Methyl dihydrogen phosphate should detect monoester."""
        mol = Chem.MolFromSmiles("COP(=O)(O)O")
        assert mol is not None
        fgs = detect_functional_groups(mol)
        assert "phosphate_monoester" in fgs, (
            f"Expected phosphate_monoester, got {sorted(fgs.keys())}"
        )


# ---------------------------------------------------------------------------
# Documented intentional behavior: phosphoric acid matches phosphonic_acid
# ---------------------------------------------------------------------------

class TestPhosphoricAcidDocumentedBehavior:
    """
    Phosphoric acid OP(=O)(O)O is the free inorganic oxoacid (P-67), NOT a
    carbon phosphonic acid (P-65.3, which requires a C-P bond).

    BBR-PERC / DEF-4 (Phase 169.7) FIXED the prior false match: the phosphonic_acid
    SMARTS now carries a recursive-env `$([PX4][#6])` carbon-attachment constraint,
    and a dedicated `phosphoric_acid` SMARTS perceives the free oxoacid. This
    corrects the audit's documented `OP(=O)(O)O -> trihydrophosphate` mis-cast.
    """

    def test_phosphoric_acid_is_not_phosphonic_acid(self):
        """OP(=O)(O)O perceives `phosphoric_acid`, NOT the carbon-acid `phosphonic_acid` (DEF-4 fix)."""
        mol = Chem.MolFromSmiles("OP(=O)(O)O")
        assert mol is not None
        fgs = detect_functional_groups(mol)
        # Post-169.7: the C-attachment constraint excludes the inorganic oxoacid.
        assert "phosphonic_acid" not in fgs, (
            f"Phosphoric acid must NOT match the carbon-acid phosphonic_acid SMARTS "
            f"(P-65.3 requires a C-P bond); got {sorted(fgs.keys())}"
        )
        assert "phosphoric_acid" in fgs, (
            f"Phosphoric acid must perceive its own `phosphoric_acid` class (P-67); "
            f"got {sorted(fgs.keys())}"
        )

    def test_carbon_phosphonic_acid_still_matches(self):
        """A genuine C-attached phosphonic acid (CCP(=O)(O)O) still matches (regression guard)."""
        mol = Chem.MolFromSmiles("CCP(=O)(O)O")
        fgs = detect_functional_groups(mol)
        assert "phosphonic_acid" in fgs and "phosphoric_acid" not in fgs


# ---------------------------------------------------------------------------
# Benchmark spot-check: real phosphorus compounds from ChEBI
# ---------------------------------------------------------------------------

# Real phosphorus-containing SMILES from the ChEBI dataset and common compounds
_CHEBI_PHOSPHORUS_SMILES = [
    "NCC(O)P(=O)(O)O",  # 2-amino-1-hydroxyethylphosphonic acid
    "O=P(O)(O)CCO",  # 2-hydroxyethylphosphonic acid
    "COP(=O)(OC)OC",  # trimethyl phosphate (triester)
    "CP(=O)(O)O",  # methylphosphonic acid
    "COP(=O)(O)O",  # methyl dihydrogen phosphate (monoester)
    "CCOP(=O)(OCC)O",  # diethyl hydrogen phosphate (diester)
    "CCP",  # ethylphosphine (primary phosphine)
    "CP(=O)(C)C",  # trimethylphosphine oxide
]


@pytest.mark.parametrize("smiles", _CHEBI_PHOSPHORUS_SMILES)
def test_phosphorus_benchmark_spot_check(smiles):
    """Real phosphorus-containing SMILES from ChEBI produce non-empty phosphorus FG detection."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Invalid SMILES: {smiles}"
    fgs = detect_functional_groups(mol)

    # At least one phosphorus-related FG should be detected
    phosphorus_fgs = {
        "phosphonic_acid", "phosphinic_acid",
        "phosphate_monoester", "phosphate_diester", "phosphate_triester",
        "primary_phosphine", "secondary_phosphine", "tertiary_phosphine",
        "phosphine_oxide",
    }
    detected_p_fgs = set(fgs.keys()) & phosphorus_fgs
    assert detected_p_fgs, (
        f"Expected at least one phosphorus FG for {smiles}, "
        f"got only: {sorted(fgs.keys())}"
    )
