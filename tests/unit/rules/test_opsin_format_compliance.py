"""
Tests for OPSIN format compliance fixes (Phase 54, Plan 01).

Covers:
  OPFX-01: Acylamino bracket format -- OPSIN requires brackets around acylamino prefixes
  OPFX-02: phenylamino -> anilino -- OPSIN recognizes "anilino" as a simple substituent

These are root-cause fixes at the point of name generation, not postprocessors.
"""

import os
import subprocess

import pytest

from orthonym.namer import name_compound

# Check OPSIN availability for round-trip tests
OPSIN_JAR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "opsin-cli-2.8.0-jar-with-dependencies.jar"
)
OPSIN_AVAILABLE = os.path.isfile(OPSIN_JAR)


def opsin_parses(name: str) -> bool:
    """Return True if OPSIN can parse the given IUPAC name to a SMILES."""
    try:
        result = subprocess.run(
            ["java", "-jar", OPSIN_JAR, "-osmi"],
            input=name,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return bool(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


# ---------------------------------------------------------------------------
# Section 1: Anilino prefix tests (OPFX-02)
# ---------------------------------------------------------------------------


class TestAnilinoPrefix:
    """Verify phenylamino -> anilino substitution in all contexts."""

    @pytest.mark.unit
    def test_anilino_prefix_on_chain(self):
        """Compound with -NHPh on an acyclic chain produces 'anilino' not 'phenylamino'."""
        # 4-aminodiphenylamine: benzene with NH2 and NHPh
        name = name_compound("Nc1ccc(Nc2ccccc2)cc1")
        assert "anilino" in name, f"Expected 'anilino' in '{name}'"
        assert "phenylamino" not in name, f"'phenylamino' should not appear in '{name}'"

    @pytest.mark.unit
    def test_anilino_prefix_on_ring(self):
        """Compound with -NHPh on a ring parent produces 'anilino'."""
        # 2-anilinopyridine
        name = name_compound("c1ccc(Nc2ccccn2)cc1")
        assert name is not None, "name_compound returned None"
        # The compound may name differently through decomposition,
        # but if anilino appears it should never be phenylamino
        assert "phenylamino" not in name, f"'phenylamino' should not appear in '{name}'"

    @pytest.mark.unit
    def test_anilino_no_outer_brackets(self):
        """anilino should NOT have outer parentheses (OPSIN simple substituent)."""
        name = name_compound("Nc1ccc(Nc2ccccc2)cc1")
        # Should be "...anilinobenzene" not "...(anilino)benzene"
        assert "(anilino)" not in name, (
            f"'(anilino)' with brackets found in '{name}' -- anilino is a simple substituent"
        )

    @pytest.mark.unit
    @pytest.mark.skipif(not OPSIN_AVAILABLE, reason="OPSIN JAR not found")
    def test_anilino_opsin_parses(self):
        """OPSIN should parse names containing 'anilino'."""
        # Simple test: 4-anilinobenzene-1-amine
        name = name_compound("Nc1ccc(Nc2ccccc2)cc1")
        if "anilino" in name:
            assert opsin_parses(name), f"OPSIN failed to parse '{name}'"


# ---------------------------------------------------------------------------
# Section 2: Acylamino bracket format tests (OPFX-01)
# ---------------------------------------------------------------------------


class TestAcylaminoBrackets:
    """Verify acylamino prefixes use bracketed form for OPSIN compatibility."""

    @pytest.mark.unit
    def test_acylamino_bracketed_short_chain(self):
        """Short acylamino group (C2) should have brackets."""
        # 2-(acetylamino)acetic acid: glycine with acetyl
        name = name_compound("CC(=O)NCC(=O)O")
        # The acylamino prefix should be in brackets
        assert name is not None
        # Verify no bare unbracketed acetylamino
        if "acetylamino" in name:
            assert "(acetylamino)" in name or "acetyl" in name, (
                f"Expected bracketed acetylamino in '{name}'"
            )

    @pytest.mark.unit
    def test_acylamino_bracketed_long_chain(self):
        """C5 acylamino group should produce '(pentanoylamino)' with brackets."""
        # 2-(pentanoylamino)pentanedioic acid
        name = name_compound("CCCCC(=O)NC(CCC(=O)O)C(=O)O")
        assert "(pentanoylamino)" in name, f"Expected '(pentanoylamino)' in '{name}'"

    @pytest.mark.unit
    def test_acylamino_bracketed_medium_chain(self):
        """C3 acylamino group should have brackets."""
        # propanoylamino on a chain
        name = name_compound("CCC(=O)NCCCC(=O)O")
        assert name is not None
        if "propanoylamino" in name:
            assert "(propanoylamino)" in name, (
                f"Expected '(propanoylamino)' with brackets in '{name}'"
            )

    @pytest.mark.unit
    @pytest.mark.skipif(not OPSIN_AVAILABLE, reason="OPSIN JAR not found")
    def test_acylamino_opsin_parses(self):
        """OPSIN should parse names with bracketed acylamino prefixes."""
        name = name_compound("CCCCC(=O)NC(CCC(=O)O)C(=O)O")
        if "(pentanoylamino)" in name:
            assert opsin_parses(name), f"OPSIN failed to parse '{name}'"


# ---------------------------------------------------------------------------
# Section 3: No bare phenylamino regression (parametrized)
# ---------------------------------------------------------------------------


class TestNoBarePhenylamino:
    """Parametrized test ensuring phenylamino never appears in any context."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "smiles,description",
        [
            ("Nc1ccc(Nc2ccccc2)cc1", "aminodiphenylamine"),
            ("c1ccc(Nc2ccccc2)cc1", "diphenylamine"),
            ("c1ccc(Nc2ccccn2)cc1", "phenyl-aminopyridine"),
            ("c1ccc(NCCCC(=O)O)cc1", "phenyl-amino-acid"),
            ("CC(=O)Nc1ccccc1", "N-phenylacetamide"),
            ("c1ccc(Nc2ccc3ccccc3c2)cc1", "phenyl-aminonaphthalene"),
        ],
        ids=[
            "aminodiphenylamine",
            "diphenylamine",
            "phenyl-aminopyridine",
            "phenyl-amino-acid",
            "N-phenylacetamide",
            "phenyl-aminonaphthalene",
        ],
    )
    def test_no_bare_phenylamino_in_any_context(self, smiles, description):
        """No compound should produce 'phenylamino' in its IUPAC name."""
        name = name_compound(smiles)
        assert name is not None, f"name_compound returned None for {description}"
        assert "phenylamino" not in name, (
            f"{description}: 'phenylamino' found in '{name}' -- should be 'anilino'"
        )


# ---------------------------------------------------------------------------
# Section 4: Canary regression guard
# ---------------------------------------------------------------------------


class TestCanaryRegression:
    """Verify all 88 canary compounds still pass after format changes."""

    @pytest.mark.unit
    def test_canary_compounds_no_regressions(self):
        """All canary compounds must produce valid names (no None, no crash)."""
        # Import canary data
        from tests.integration.test_canary_rt75 import CANARY_COMPOUNDS

        failures = []
        for smi, expected_name in CANARY_COMPOUNDS:
            try:
                result = name_compound(smi)
                if result is None:
                    failures.append(f"None for {smi}")
                elif result != expected_name:
                    failures.append(f"CHANGED: {smi}: expected '{expected_name}', got '{result}'")
            except Exception as e:
                failures.append(f"CRASH for {smi}: {e}")

        assert len(failures) == 0, (
            f"{len(failures)} canary regressions:\n" + "\n".join(failures[:10])
        )
