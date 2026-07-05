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
    os.path.dirname(__file__), "..", "..", "..", "opsin-cli-2.9.0-jar-with-dependencies.jar"
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
        """Nc1ccc(Nc2ccccc2)cc1 is the PIN N-phenylbenzene-1,4-diamine.

        SUPERSEDED (C4b, 2026-07-03): this test previously required the OLD
        non-PIN general name '...anilinobenzene' for this molecule. Under the
        C4/C4b aromatic-diamine fix both amino N's are the principal group, so
        the PIN is the benzene-1,4-diamine parent with the aryl group cited as
        an italic-N prefix -> 'N-phenylbenzene-1,4-diamine' (OPSIN round-trip
        verified). No 'anilino' fragment appears; that is correct PIN behaviour,
        not a regression. 'phenylamino' must still never appear.
        """
        name = name_compound("Nc1ccc(Nc2ccccc2)cc1")
        assert name == "N-phenylbenzene-1,4-diamine", (
            f"Expected PIN 'N-phenylbenzene-1,4-diamine', got '{name}'"
        )
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
        """anilino should NOT have outer parentheses (OPSIN simple substituent).

        Uses 4-anilinobenzoic acid: the senior CO2H keeps the -NHPh amine
        demoted to the 'anilino' PREFIX (P-62.2.2 seniority), so this still
        exercises the anilino-prefix path. (The bare NH2/NHPh diamine now names
        as the PIN N-phenylbenzene-1,4-diamine — see test_anilino_prefix_on_chain.)
        """
        name = name_compound("OC(=O)c1ccc(Nc2ccccc2)cc1")
        assert "anilino" in name, f"Expected 'anilino' prefix in '{name}'"
        # Should be "...anilinobenzoic acid" not "...(anilino)benzoic acid"
        assert "(anilino)" not in name, (
            f"'(anilino)' with brackets found in '{name}' -- anilino is a simple substituent"
        )

    @pytest.mark.unit
    @pytest.mark.skipif(not OPSIN_AVAILABLE, reason="OPSIN JAR not found")
    def test_anilino_opsin_parses(self):
        """OPSIN should parse names containing 'anilino'."""
        name = name_compound("OC(=O)c1ccc(Nc2ccccc2)cc1")
        if "anilino" in name:
            assert opsin_parses(name), f"OPSIN failed to parse '{name}'"


# ---------------------------------------------------------------------------
# Section 2: Acylamino bracket format tests (OPFX-01)
# ---------------------------------------------------------------------------


class TestAcylaminoBrackets:
    """N-acyl prefixes use the amido form (P-66.1.1.4.3 method (1) = PIN).

    Wave2 T1c: the method-(2) '(pentanoylamino)' bracketed forms were
    replaced by the preferred amido family — formamido/acetamido/
    {stem}anamido — which are simple prefixes and take NO enclosing marks
    (Blue Book: 4-formamidobenzoic acid, 4-acetamidobenzoic acid).
    """

    @pytest.mark.unit
    def test_amido_short_chain(self):
        """C2 acyl on glycine -> acetamido, unbracketed."""
        name = name_compound("CC(=O)NCC(=O)O")
        assert name == "2-acetamidoethanoic acid", (
            f"Expected '2-acetamidoethanoic acid', got '{name}'"
        )

    @pytest.mark.unit
    def test_amido_long_chain(self):
        """C5 acyl on glutamic acid -> pentanamido, unbracketed."""
        name = name_compound("CCCCC(=O)NC(CCC(=O)O)C(=O)O")
        assert name == "2-pentanamidopentanedioic acid", (
            f"Expected '2-pentanamidopentanedioic acid', got '{name}'"
        )

    @pytest.mark.unit
    def test_amido_medium_chain(self):
        """C3 acyl on GABA -> propanamido, unbracketed."""
        name = name_compound("CCC(=O)NCCCC(=O)O")
        assert name == "4-propanamidobutanoic acid", (
            f"Expected '4-propanamidobutanoic acid', got '{name}'"
        )

    @pytest.mark.unit
    @pytest.mark.skipif(not OPSIN_AVAILABLE, reason="OPSIN JAR not found")
    def test_amido_opsin_parses(self):
        """OPSIN should parse names with amido prefixes."""
        name = name_compound("CCCCC(=O)NC(CCC(=O)O)C(=O)O")
        assert "pentanamido" in name, f"Expected 'pentanamido' in '{name}'"
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
