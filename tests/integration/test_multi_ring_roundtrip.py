"""
OPSIN round-trip integration tests for multi-ring substituent naming.

Phase 82 Plan 02 — Verifies that ring assembly prefixes ([1,1'-biphenyl]-4-yl)
and mixed-ring compound prefixes round-trip through OPSIN, and that enclosing
marks follow IUPAC P-16.3.3 nesting hierarchy.
"""

import subprocess

import pytest
from rdkit import Chem

from orthonym import name_compound


# === OPSIN Setup ===

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
OPSIN_JAR = PROJECT_ROOT / "opsin-cli-2.8.0-jar-with-dependencies.jar"


def _opsin_available() -> bool:
    """Check if OPSIN CLI JAR is available and Java is installed."""
    if not OPSIN_JAR.exists():
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
    if not name or not OPSIN_JAR.exists():
        return ""
    try:
        proc = subprocess.run(
            ["java", "-jar", str(OPSIN_JAR), "-osmi"],
            input=name, capture_output=True, text=True, timeout=15
        )
        return proc.stdout.strip()
    except (subprocess.TimeoutExpired, Exception):
        return ""


def _canonical(smiles: str) -> str:
    """Canonicalize SMILES via RDKit."""
    if not smiles:
        return ""
    return Chem.CanonSmiles(smiles) or ""


def _roundtrip_matches(input_smiles: str) -> tuple:
    """Generate name and check OPSIN round-trip. Returns (name, matches, opsin_smi)."""
    name = name_compound(input_smiles)
    if not name:
        return (name, False, "")
    opsin_smi = _opsin_parse(name)
    if not opsin_smi:
        return (name, False, "")
    can_orig = _canonical(input_smiles)
    can_opsin = _canonical(opsin_smi)
    return (name, can_orig == can_opsin, opsin_smi)


HAS_OPSIN = _opsin_available()
skip_no_opsin = pytest.mark.skipif(
    not HAS_OPSIN, reason="OPSIN JAR not available"
)


# ============================================================================
# Ring Assembly Prefix Round-Trip Tests (IUPAC P-28.3)
# ============================================================================

@pytest.mark.integration
class TestRingAssemblyRoundTrip:
    """Round-trip tests for identical ring assembly substituent prefixes."""

    @skip_no_opsin
    def test_biphenylbutanoic_acid_roundtrip(self):
        """4-([1,1'-biphenyl]-4-yl)butanoic acid round-trips through OPSIN."""
        smi = "OC(=O)CCCc1ccc(-c2ccccc2)cc1"
        name, matches, opsin_smi = _roundtrip_matches(smi)
        assert name is not None, "name_compound returned None"
        assert "biphenyl" in name, f"Expected 'biphenyl' in '{name}'"
        assert matches, (
            f"Round-trip failed: '{name}' -> OPSIN -> '{opsin_smi}'\n"
            f"Expected canonical: {_canonical(smi)}\n"
            f"Got canonical:      {_canonical(opsin_smi)}"
        )

    @skip_no_opsin
    def test_biphenylacetic_acid_roundtrip(self):
        """([1,1'-biphenyl]-4-yl)acetic acid variant round-trips."""
        smi = "OC(=O)Cc1ccc(-c2ccccc2)cc1"
        name, matches, opsin_smi = _roundtrip_matches(smi)
        assert name is not None
        assert "biphenyl" in name, f"Expected 'biphenyl' in '{name}'"
        assert matches, (
            f"Round-trip failed: '{name}' -> OPSIN -> '{opsin_smi}'"
        )

    @skip_no_opsin
    def test_biphenyl_meta_attachment_roundtrip(self):
        """Biphenyl with meta-position chain attachment round-trips."""
        smi = "OC(=O)CCCc1cccc(-c2ccccc2)c1"
        name, matches, opsin_smi = _roundtrip_matches(smi)
        assert name is not None
        assert "biphenyl" in name, f"Expected 'biphenyl' in '{name}'"
        assert matches, (
            f"Round-trip failed: '{name}' -> OPSIN -> '{opsin_smi}'"
        )

    @skip_no_opsin
    def test_bipyridyl_substituent_roundtrip(self):
        """[2,2'-bipyridin]-4-yl on chain round-trips through OPSIN."""
        smi = "OC(=O)CCCCCc1ccnc(-c2ccccn2)c1"
        name, matches, opsin_smi = _roundtrip_matches(smi)
        assert name is not None
        assert "bipyridin" in name, f"Expected 'bipyridin' in '{name}'"
        assert matches, (
            f"Round-trip failed: '{name}' -> OPSIN -> '{opsin_smi}'"
        )


# ============================================================================
# Mixed-Ring Compound Prefix Round-Trip Tests (IUPAC P-31)
# ============================================================================

@pytest.mark.integration
class TestMixedRingRoundTrip:
    """Round-trip tests for non-identical ring compound substituent prefixes."""

    @skip_no_opsin
    def test_phenylpyridyl_roundtrip(self):
        """Phenyl-pyridyl compound prefix on chain round-trips."""
        smi = "OC(=O)CCCCCc1ccc(-c2ccccn2)cc1"
        name, matches, opsin_smi = _roundtrip_matches(smi)
        assert name is not None
        assert matches, (
            f"Round-trip failed: '{name}' -> OPSIN -> '{opsin_smi}'\n"
            f"Expected canonical: {_canonical(smi)}\n"
            f"Got canonical:      {_canonical(opsin_smi)}"
        )

    @skip_no_opsin
    def test_pyridylphenyl_reversed_smiles_roundtrip(self):
        """Same compound with reversed SMILES atom order still round-trips."""
        # Pyridine written first, phenyl second — same molecule
        smi = "OC(=O)CCCCCc1ccc(-c2ccccn2)cc1"
        smi_alt = "OC(=O)CCCCCc1ccc(cc1)-c1ccccn1"
        # Both should canonicalize to the same thing
        assert _canonical(smi) == _canonical(smi_alt)
        name, matches, opsin_smi = _roundtrip_matches(smi_alt)
        assert name is not None
        assert matches, (
            f"Round-trip failed: '{name}' -> OPSIN -> '{opsin_smi}'"
        )


# ============================================================================
# Enclosing Mark Format Tests (IUPAC P-16.3.3)
# ============================================================================

@pytest.mark.integration
class TestEnclosingMarks:
    """Verify correct enclosing mark usage for multi-ring prefixes."""

    def test_biphenyl_prefix_has_square_brackets(self):
        """Biphenyl assembly prefix contains square brackets for locants."""
        smi = "OC(=O)CCCc1ccc(-c2ccccc2)cc1"
        name = name_compound(smi)
        assert name is not None
        assert "[" in name and "]" in name, (
            f"Expected square brackets in biphenyl prefix, got: '{name}'"
        )
        assert "'-biphenyl]" in name, (
            f"Expected '-biphenyl]' pattern in '{name}'"
        )

    def test_biphenyl_prefix_has_outer_parentheses(self):
        """Biphenyl prefix wrapped in parentheses: ([1,1'-biphenyl]-N-yl).

        Per IUPAC P-16.3.3: parentheses () enclose square brackets [].
        """
        smi = "OC(=O)CCCc1ccc(-c2ccccc2)cc1"
        name = name_compound(smi)
        assert name is not None
        # Should contain pattern: ([ ... ])
        assert "([" in name, (
            f"Expected '([' in name for outer parentheses, got: '{name}'"
        )
        assert "])" in name or "]-" in name, (
            f"Expected bracket-paren nesting in '{name}'"
        )

    def test_simple_phenyl_no_brackets(self):
        """Single phenyl substituent does NOT get square brackets (regression)."""
        smi = "OC(=O)CCCc1ccccc1"
        name = name_compound(smi)
        assert name is not None
        assert "[" not in name, (
            f"Single phenyl should not have square brackets, got: '{name}'"
        )
        assert "phenyl" in name.lower()

    def test_bipyridyl_prefix_has_brackets(self):
        """Bipyridyl assembly prefix also uses square brackets."""
        smi = "OC(=O)CCCCCc1ccnc(-c2ccccn2)c1"
        name = name_compound(smi)
        assert name is not None
        assert "[" in name and "]" in name, (
            f"Expected square brackets in bipyridyl prefix, got: '{name}'"
        )


# ============================================================================
# Canary Stability Test
# ============================================================================

@pytest.mark.integration
class TestCanaryStability:
    """Verify representative canary compounds still produce expected names."""

    CANARY_COMPOUNDS = [
        # (SMILES, expected_substring) — diverse compound classes
        ("CCO", "ethanol"),
        ("CC(=O)O", "acetic acid"),
        ("c1ccccc1", "benzene"),
        ("CC(C)CC", "2-methylbutane"),
        ("CC=CC", "but-2-ene"),
        ("OC(=O)CCCCC", "hexanoic acid"),
        ("CC(=O)CC", "butan-2-one"),
        ("c1ccncc1", "pyridine"),
        ("C1CCCC1", "cyclopentane"),
        ("c1ccc2ccccc2c1", "naphthalene"),
        ("CC(O)CC", "butan-2-ol"),
        ("CCCC(=O)CC", "hexan-3-one"),
        ("c1ccc(cc1)O", "phenol"),
        ("CC(=O)N", "acetamide"),
        ("OC(=O)c1ccccc1", "benzoic acid"),
        ("C1CCOCC1", "tetrahydropyran"),
        ("c1ccoc1", "furan"),
        ("C1CCNCC1", "piperidine"),
        ("OC(=O)CCCc1ccccc1", "phenylbutanoic acid"),
        ("CC(C)(C)C", "2,2-dimethylpropane"),
    ]

    @pytest.mark.parametrize("smi,expected", CANARY_COMPOUNDS)
    def test_canary_compound(self, smi, expected):
        """Canary compound produces expected name substring."""
        name = name_compound(smi)
        assert name is not None, f"name_compound('{smi}') returned None"
        assert expected.lower() in name.lower(), (
            f"Expected '{expected}' in '{name}' for SMILES '{smi}'"
        )
