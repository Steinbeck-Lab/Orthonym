"""
OPSIN format validation tests for Orthonym.

Tests that generated IUPAC names are parseable by OPSIN and do not contain
known OPSIN-incompatible patterns. These tests ensure that naming improvements
do not introduce regressions in OPSIN compatibility.

Test categories:
1. Pattern-based regression tests (no OPSIN needed)
2. OPSIN-specific regression tests (require OPSIN JAR)
3. Format validation tests (no OPSIN needed)
"""

import os
import subprocess
import re
import pytest
from pathlib import Path

from orthonym import name_compound
from tests.support.jars import jar_or_none


# === OPSIN JAR DETECTION ===

PROJECT_ROOT = Path(__file__).parent.parent.parent
OPSIN_JAR = jar_or_none()


def _opsin_available() -> bool:
    """Check if OPSIN CLI JAR is available and Java is installed."""
    if OPSIN_JAR is None:
        return False
    try:
        proc = subprocess.run(
            ["java", "-version"],
            capture_output=True, text=True, timeout=5
        )
        return proc.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _opsin_parse(name: str) -> str:
    """Parse an IUPAC name through OPSIN and return SMILES (or empty string)."""
    if not name or OPSIN_JAR is None:
        return ""
    try:
        proc = subprocess.run(
            ["java", "-jar", str(OPSIN_JAR), "-osmi"],
            input=name, capture_output=True, text=True, timeout=10
        )
        return proc.stdout.strip()
    except (subprocess.TimeoutExpired, Exception):
        return ""


HAS_OPSIN = _opsin_available()
skip_no_opsin = pytest.mark.skipif(
    not HAS_OPSIN, reason="OPSIN JAR not available"
)


# === OPSIN-INCOMPATIBLE PATTERN TESTS ===
# These test that generated names do NOT contain known bad patterns.
# No OPSIN needed - pure string validation.


class TestOpsinIncompatiblePatterns:
    """Test that names do not contain patterns known to break OPSIN parsing."""

    @pytest.mark.integration
    def test_no_generic_ion_names(self):
        """No name should contain 'cation-ium' or 'anion-ide' fallback strings."""
        test_smiles = [
            "[NH4+]",        # ammonium
            "CC(=O)[O-]",   # acetate
            "[Na+].[Cl-]",  # sodium chloride
            "[SeH+]",       # selenonium
            "CC[O-]",       # ethoxide
        ]
        for smi in test_smiles:
            result = name_compound(smi)
            if result:
                assert "cation-" not in result, (
                    f"Generic 'cation-' in name for {smi}: {result}"
                )
                assert "anion-" not in result, (
                    f"Generic 'anion-' in name for {smi}: {result}"
                )

    @pytest.mark.integration
    def test_no_oxine_in_names(self):
        """Names should not contain 'oxine' (OPSIN-blocked HW name).

        IUPAC 2013 prefers '2H-pyran' over the HW systematic 'oxine'.
        OPSIN rejects 'oxine' as a blocked Hantzsch-Widman system.
        """
        test_smiles = [
            "O1C=CC=CC1",            # 2H-pyran
            "C1=COC=CC1",            # 4H-pyran
            "c1ccncc1",              # pyridine (should not produce azine)
        ]
        for smi in test_smiles:
            result = name_compound(smi)
            if result:
                # Allow 'oxine' only as substring of other words like 'dioxin'
                assert "oxine" not in result.split("-") and not result.endswith("oxine"), (
                    f"OPSIN-incompatible 'oxine' in name for {smi}: {result}"
                )

    @pytest.mark.integration
    def test_multiplier_locant_agreement(self):
        """Multiplied prefixes must have matching locant counts.

        E.g., 'diamino' needs 2 locants: '2,6-diamino', not '3-diamino'.
        """
        test_smiles = [
            "NC(N)CC(O)=O",             # 3,3-diaminopropanoic acid
            "NCCCCC(N)C(O)=O",          # 2,6-diaminoheptanoic acid
            "OC(O)CC(O)=O",             # 3-hydroxypropanedioic acid
        ]
        # Pattern to find multiplied prefixes with locants
        # Matches "3,3-diamino", "2,4-dichloro", etc.
        di_tri_pattern = re.compile(
            r"([\d,]+)-(di|tri|tetra|penta|hexa)"
            r"(amino|hydroxy|chloro|bromo|fluoro|oxo)"
        )
        mult_map = {
            "di": 2, "tri": 3, "tetra": 4,
            "penta": 5, "hexa": 6
        }
        for smi in test_smiles:
            result = name_compound(smi)
            if result:
                for match in di_tri_pattern.finditer(result):
                    locant_str = match.group(1)
                    multiplier_prefix = match.group(2)
                    n_locants = len(locant_str.split(","))
                    expected_count = mult_map.get(multiplier_prefix, 0)
                    if expected_count:
                        assert n_locants == expected_count, (
                            f"Locant count ({n_locants}) != multiplier "
                            f"count ({expected_count}) in "
                            f"'{match.group(0)}' for {smi}: {result}"
                        )

    @pytest.mark.integration
    def test_no_unknown_in_salt_names(self):
        """Salt names should not contain 'unknown', 'cation', or 'anion' as words."""
        test_smiles = [
            "[Na+].[O-]C(C)=O",     # sodium acetate
            "[Na+].[Cl-]",          # sodium chloride
            "[K+].[O-]C=O",         # potassium formate
        ]
        bad_words = {"unknown", "cation", "anion"}
        for smi in test_smiles:
            result = name_compound(smi)
            if result:
                words = set(result.lower().split())
                found_bad = words & bad_words
                assert not found_bad, (
                    f"Generic placeholder '{found_bad}' in salt name "
                    f"for {smi}: {result}"
                )


# === OPSIN REGRESSION CASES ===
# Specific SMILES that previously failed OPSIN, now expected to pass.

OPSIN_REGRESSION_CASES = [
    # (SMILES, expected_name_pattern, description)
    ("CCO", "ethanol", "simple alcohol"),
    ("CCCC(=O)O", "butanoic acid", "simple acid"),
    ("c1ccccc1", "benzene", "benzene retained name"),
    ("c1ccncc1", "pyridine", "pyridine retained name"),
    ("CC(=O)[O-]", "acetate", "carboxylate anion"),
    ("[NH4+]", "azanium", "azanium cation (P-73.1.1.2 PIN; was 'ammonium')"),
    ("CC(O)CC(=O)O", None, "hydroxy acid (polyfunctional)"),
    ("NC(N)CC(O)=O", "3,3-diaminopropanoic acid", "geminal diamino"),
    ("OCC(O)CO", None, "triol"),
    ("[Na+].[Cl-]", "sodium chloride", "simple salt"),
    ("CC(=O)NC", None, "simple amide"),
    ("CCOC(=O)C", None, "simple ester"),
    ("ClC(Cl)(Cl)Cl", None, "tetrachloromethane"),
    ("C1CCCCC1", "cyclohexane", "cyclohexane"),
    ("C1CC1", "cyclopropane", "cyclopropane"),
    ("CC=CC", None, "simple alkene"),
    ("C#CC", None, "propyne"),
    ("CCCCCCCC", "octane", "straight chain alkane"),
    ("CC(C)CC", None, "branched alkane"),
    ("[Na+].[O-]C(C)=O", "sodium acetate", "sodium acetate salt"),
]


class TestOpsinRegressionCases:
    """Test that specific compounds generate OPSIN-parseable names."""

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "smiles,expected_pattern,description",
        OPSIN_REGRESSION_CASES,
        ids=[c[2] for c in OPSIN_REGRESSION_CASES]
    )
    def test_name_generation(self, smiles, expected_pattern, description):
        """Each regression case should produce a non-empty name."""
        result = name_compound(smiles)
        assert result, f"Failed to name {smiles} ({description})"
        assert result != "unknown", f"Got 'unknown' for {smiles} ({description})"
        if expected_pattern:
            assert expected_pattern in result, (
                f"Expected '{expected_pattern}' in name for {smiles}, "
                f"got '{result}' ({description})"
            )

    @skip_no_opsin
    @pytest.mark.integration
    @pytest.mark.parametrize(
        "smiles,expected_pattern,description",
        OPSIN_REGRESSION_CASES,
        ids=[c[2] for c in OPSIN_REGRESSION_CASES]
    )
    def test_opsin_parseable(self, smiles, expected_pattern, description):
        """Each regression case name should be parseable by OPSIN."""
        result = name_compound(smiles)
        if not result or result == "unknown":
            pytest.skip(f"No name generated for {smiles}")

        opsin_smi = _opsin_parse(result)
        assert opsin_smi, (
            f"OPSIN failed to parse '{result}' for {smiles} ({description})"
        )


# === FORMAT VALIDATION TESTS ===
# Structural format rules that all generated names should follow.


class TestNameFormatValidation:
    """Test that generated names follow IUPAC format rules."""

    @pytest.mark.integration
    def test_no_double_hyphens(self):
        """Names should not contain '--' (double hyphens)."""
        test_smiles = [
            "CC(O)C(=O)O",
            "CCCC(=O)NC",
            "CC(Cl)CC(=O)O",
        ]
        for smi in test_smiles:
            result = name_compound(smi)
            if result:
                assert "--" not in result, (
                    f"Double hyphen in name for {smi}: {result}"
                )

    @pytest.mark.integration
    def test_no_trailing_hyphens(self):
        """Names should not end with a hyphen."""
        test_smiles = [
            "CCCCO",
            "CCCC=O",
            "CC(=O)CC",
        ]
        for smi in test_smiles:
            result = name_compound(smi)
            if result:
                assert not result.endswith("-"), (
                    f"Trailing hyphen in name for {smi}: {result}"
                )

    @pytest.mark.integration
    def test_no_leading_hyphens(self):
        """Names should not start with a hyphen."""
        test_smiles = [
            "CCCCO",
            "c1ccccc1O",
            "CC(=O)O",
        ]
        for smi in test_smiles:
            result = name_compound(smi)
            if result:
                assert not result.startswith("-"), (
                    f"Leading hyphen in name for {smi}: {result}"
                )

    @pytest.mark.integration
    def test_balanced_parentheses(self):
        """Names should have balanced parentheses."""
        test_smiles = [
            "CC(=O)OC",
            "CC(=O)NCC",
            "CCCC(=O)N(CC)CC",
            "c1ccc(O)cc1",
        ]
        for smi in test_smiles:
            result = name_compound(smi)
            if result:
                open_count = result.count("(")
                close_count = result.count(")")
                assert open_count == close_count, (
                    f"Unbalanced parentheses in name for {smi}: {result} "
                    f"(open={open_count}, close={close_count})"
                )

    @pytest.mark.integration
    def test_suffix_multiplier_within_chain_capacity(self):
        """Suffix multiplier count should not exceed parent chain/ring size.

        E.g., ethane (2C) cannot have tetraol (4 OH groups).
        """
        test_smiles = [
            ("CCO", 2),          # ethanol - max 2 OH on ethane
            ("CCCO", 3),         # propanol - max 3 OH on propane
            ("C1CCCCC1O", 6),    # cyclohexanol - max 6 OH on cyclohexane
        ]
        suffix_mult_re = re.compile(
            r"(di|tri|tetra|penta|hexa|hepta|octa)(ol|one|amine|al)"
        )
        mult_count = {
            "di": 2, "tri": 3, "tetra": 4, "penta": 5,
            "hexa": 6, "hepta": 7, "octa": 8
        }
        for smi, max_capacity in test_smiles:
            result = name_compound(smi)
            if result:
                for match in suffix_mult_re.finditer(result):
                    prefix = match.group(1)
                    count = mult_count.get(prefix, 0)
                    assert count <= max_capacity, (
                        f"Suffix multiplier '{match.group(0)}' exceeds "
                        f"chain capacity ({max_capacity}) for {smi}: {result}"
                    )


# === SPECIES TYPE DETECTION TESTS ===


class TestSpeciesTypeDetection:
    """Test that species types are correctly detected for naming routing."""

    @pytest.mark.integration
    def test_charged_heteroatom_is_ion_not_radical(self):
        """Charged heteroatoms should be detected as ions, not radicals.

        RDKit may assign radical electrons to certain charged heteroatoms
        (e.g., [SeH+] gets 2 radical electrons). The species detector
        should recognize the formal charge and route to ion naming.
        """
        from orthonym.perception.ions import detect_species_type
        from rdkit import Chem

        ion_cases = [
            ("[SeH+]", "ion"),
            ("[NH4+]", "ion"),
            ("CC(=O)[O-]", "ion"),
        ]
        for smi, expected in ion_cases:
            mol = Chem.MolFromSmiles(smi)
            assert mol is not None
            species = detect_species_type(mol)
            assert species == expected, (
                f"Expected species type '{expected}' for {smi}, "
                f"got '{species}'"
            )

    @pytest.mark.integration
    def test_radicals_still_detected(self):
        """True radicals (no formal charge) should still be detected."""
        from orthonym.perception.ions import detect_species_type
        from rdkit import Chem

        radical_cases = [
            ("[CH3]", "radical"),
            ("[CH2]", "radical"),
        ]
        for smi, expected in radical_cases:
            mol = Chem.MolFromSmiles(smi)
            assert mol is not None
            species = detect_species_type(mol)
            assert species == expected, (
                f"Expected species type '{expected}' for {smi}, "
                f"got '{species}'"
            )


# === STEREODESCRIPTOR CASE PRESERVATION TESTS (a phase-03) ===


class TestStereoCasePreservation:
    """Verify that E/Z stereodescriptors retain uppercase in acyloxy/ion names.

    Bug: get_acyloxy_prefix and ion naming functions used.lower on the
    entire acid name, converting (11Z,14Z) to (11z,14z). Fixed by preserving
    original case and using lowercase only for comparison/lookup.
    """

    @pytest.mark.integration
    def test_acyloxy_preserves_stereo_case(self):
        """E/Z descriptors in acyloxy prefixes must be uppercase."""
        from orthonym.rules.esters import get_acyloxy_prefix

        # Systematic acid with E/Z stereodescriptor
        result = get_acyloxy_prefix("(11Z,14Z)-icosa-11,14-dienoic")
        assert "(11Z,14Z)" in result, (
            f"Expected uppercase (11Z,14Z) in acyloxy prefix, got: {result}"
        )

    @pytest.mark.integration
    def test_acyloxy_still_works_for_trivial_acids(self):
        """Trivial acid lookup must still work (case-insensitive)."""
        from orthonym.rules.esters import get_acyloxy_prefix

        assert get_acyloxy_prefix("acetic") == "acetyloxy"
        assert get_acyloxy_prefix("Acetic") == "acetyloxy"
        assert get_acyloxy_prefix("benzoic") == "benzoyloxy"
        assert get_acyloxy_prefix("propanoic") == "propanoyloxy"

    @pytest.mark.integration
    def test_carboxylate_anion_preserves_stereo_case(self):
        """Carboxylate anion naming preserves stereodescriptor case."""
        from orthonym.rules.ions import name_carboxylate_anion

        result = name_carboxylate_anion("(2E)-but-2-enoic acid")
        assert "(2E)" in result, (
            f"Expected uppercase (2E) in anion name, got: {result}"
        )

    @pytest.mark.integration
    @skip_no_opsin
    def test_e2e_lipid_stereo_uppercase(self):
        """End-to-end: unsaturated lipid names have uppercase E/Z in acyloxy."""
        smi = (
            r"CCCCC/C=C\C/C=C\CCCCCCCCCC(=O)OC"
            r"(COC(=O)CCCCCCC/C=C\C/C=C\CCCCC)"
            r"COC(=O)CCCCCCC/C=C\C/C=C\CCCCC"
        )
        name = name_compound(smi)
        # Name must contain uppercase Z (not lowercase z) in stereo prefix
        import re
        stereo_matches = re.findall(r'\(\d+[EZ]', name)
        lowercase_matches = re.findall(r'\(\d+[ez]', name)
        assert stereo_matches, (
            f"Expected E/Z stereodescriptors in name: {name}"
        )
        assert not lowercase_matches, (
            f"Found lowercase stereo descriptors in name: {name}"
        )
