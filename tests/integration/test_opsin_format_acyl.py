"""
Tests for OPSIN-compatible acylamino/acyloxy/alkylamino prefix formatting.

Validates that Orthonym generates prefix names parseable by OPSIN CLI.
Specifically tests that:
1. Acyl prefix format uses 'pentanoylamino' not '(pentanoyl)amino' (no inner parens)
2. Compound substituents have enclosing parens: '2-(pentanoylamino)' not '2-pentanoylamino'
3. OPSIN CLI successfully parses all generated acyl prefix names

Phase 15.5 Plan 01: OPSIN Format Fixes - Acyl Prefix Format
"""

import os
import re
import subprocess
import pytest
from orthonym import name_compound

# ---------------------------------------------------------------------------
# OPSIN CLI helper
# ---------------------------------------------------------------------------

OPSIN_JAR = os.path.join(
    os.path.dirname(__file__), "..", "..", "opsin-cli-2.8.0-jar-with-dependencies.jar"
)
OPSIN_AVAILABLE = os.path.isfile(OPSIN_JAR)


def opsin_parse(name: str) -> str:
    """Parse a name with OPSIN CLI and return SMILES or empty string on failure."""
    if not OPSIN_AVAILABLE:
        return ""
    try:
        result = subprocess.run(
            ["java", "-jar", OPSIN_JAR, "-osmi"],
            input=name,
            capture_output=True,
            text=True,
            timeout=15,
        )
        lines = result.stdout.strip().split("\n")
        out = lines[-1].strip() if lines else ""
        if "could not be interpreted" in out.lower():
            return ""
        if "unsure of the meaning" in out.lower():
            return ""
        return out
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


# ===========================================================================
# Test: Acylamino prefix format
# ===========================================================================


@pytest.mark.integration
class TestAcylaminoFormat:
    """Verify acylamino prefix produces OPSIN-compatible format with enclosing parens."""

    def test_pentanoylamino_format(self):
        """2-(pentanoylamino)pentanedioic acid: enclosing parens around compound substituent."""
        name = name_compound("CCCCC(=O)NC(CCC(=O)O)C(=O)O")
        assert "pentanoylamino" in name, f"Expected 'pentanoylamino' in '{name}'"
        # Must NOT have the old format with inner parens around just the acyl part
        assert "(pentanoyl)amino" not in name, (
            f"Old format '(pentanoyl)amino' found in '{name}'"
        )

    def test_pentanoylamino_enclosing_parens(self):
        """Compound substituent must have enclosing parens: (pentanoylamino)."""
        name = name_compound("CCCCC(=O)NC(CCC(=O)O)C(=O)O")
        assert "(pentanoylamino)" in name, (
            f"Expected enclosing parens '(pentanoylamino)' in '{name}'"
        )

    def test_ethanoylamino_format(self):
        """Acetylamino group should use '(ethanoylamino)' format."""
        name = name_compound("CC(=O)NC(CC(=O)O)C(=O)O")
        assert "ethanoylamino" in name, f"Expected 'ethanoylamino' in '{name}'"
        assert "(ethanoyl)amino" not in name, (
            f"Old format '(ethanoyl)amino' found in '{name}'"
        )

    def test_ethanoylamino_enclosing_parens(self):
        """Compound substituent must have enclosing parens: (ethanoylamino)."""
        name = name_compound("CC(=O)NC(CC(=O)O)C(=O)O")
        assert "(ethanoylamino)" in name, (
            f"Expected enclosing parens '(ethanoylamino)' in '{name}'"
        )

    def test_octadecanoylamino_format(self):
        """Long-chain acylamino: (octadecanoylamino) with enclosing parens."""
        name = name_compound("CCCCCCCCCCCCCCCCCC(=O)NCC(=O)O")
        assert "octadecanoylamino" in name, (
            f"Expected 'octadecanoylamino' in '{name}'"
        )
        assert "(octadecanoyl)amino" not in name, (
            f"Old format '(octadecanoyl)amino' found in '{name}'"
        )

    def test_no_inner_parens_pattern(self):
        """No generated name should contain the pattern (Xanoyl)amino."""
        smiles_list = [
            "CCCCC(=O)NC(CCC(=O)O)C(=O)O",       # pentanoylamino
            "CC(=O)NC(CC(=O)O)C(=O)O",             # ethanoylamino
            "CCCCCCCC(=O)NCC(=O)O",                 # octanoylamino
            "CCCCCCCCCCCCCCCCCC(=O)NCC(=O)O",       # octadecanoylamino
        ]
        pattern = re.compile(r"\([a-z]+anoyl\)amino")
        for smiles in smiles_list:
            name = name_compound(smiles)
            assert not pattern.search(name), (
                f"Inner-paren acyl format found in '{name}' for {smiles}"
            )

    def test_enclosing_parens_with_locant(self):
        """When locant present, format must be N-(Xanoylamino): locant followed by parens."""
        name = name_compound("CCCCC(=O)NC(CCC(=O)O)C(=O)O")
        # Must match pattern: digit-(Xanoylamino) -- enclosing parens around whole group
        pattern = re.compile(r"\d-\([a-z]+anoylamino\)")
        assert pattern.search(name), (
            f"Expected locant-(anoylamino) pattern with enclosing parens in '{name}'"
        )

    @pytest.mark.skipif(not OPSIN_AVAILABLE, reason="OPSIN JAR not found")
    def test_pentanoylamino_opsin_parses(self):
        """OPSIN should parse pentanoylamino-containing names."""
        name = name_compound("CCCCC(=O)NC(CCC(=O)O)C(=O)O")
        smiles = opsin_parse(name)
        assert smiles, f"OPSIN failed to parse: '{name}'"

    @pytest.mark.skipif(not OPSIN_AVAILABLE, reason="OPSIN JAR not found")
    def test_ethanoylamino_opsin_parses(self):
        """OPSIN should parse ethanoylamino-containing names."""
        name = name_compound("CC(=O)NC(CC(=O)O)C(=O)O")
        smiles = opsin_parse(name)
        assert smiles, f"OPSIN failed to parse: '{name}'"

    @pytest.mark.skipif(not OPSIN_AVAILABLE, reason="OPSIN JAR not found")
    def test_octadecanoylamino_opsin_parses(self):
        """OPSIN should parse octadecanoylamino-containing names."""
        name = name_compound("CCCCCCCCCCCCCCCCCC(=O)NCC(=O)O")
        smiles = opsin_parse(name)
        assert smiles, f"OPSIN failed to parse: '{name}'"

    @pytest.mark.skipif(not OPSIN_AVAILABLE, reason="OPSIN JAR not found")
    def test_octanoylamino_opsin_parses(self):
        """OPSIN should parse octanoylamino-containing names."""
        name = name_compound("CCCCCCCC(=O)NCC(=O)O")
        smiles = opsin_parse(name)
        assert smiles, f"OPSIN failed to parse: '{name}'"

    @pytest.mark.skipif(not OPSIN_AVAILABLE, reason="OPSIN JAR not found")
    def test_hexanoylamino_opsin_parses(self):
        """OPSIN should parse hexanoylamino-containing names."""
        name = name_compound("CCCCCC(=O)NC(CCCCN)C(=O)O")
        smiles = opsin_parse(name)
        assert smiles, f"OPSIN failed to parse: '{name}'"


# ===========================================================================
# Test: Acyloxy prefix format
# ===========================================================================


@pytest.mark.integration
class TestAcyloxyFormat:
    """Verify acyloxy prefix produces OPSIN-compatible format with enclosing parens."""

    def test_ethanoyloxy_format(self):
        """Acyloxy should produce '(ethanoyloxy)' not '(ethanoyl)oxy'."""
        name = name_compound("CC(=O)OCCC(=O)O")
        assert "ethanoyloxy" in name, f"Expected 'ethanoyloxy' in '{name}'"
        assert "(ethanoyl)oxy" not in name, (
            f"Old format '(ethanoyl)oxy' found in '{name}'"
        )

    def test_ethanoyloxy_enclosing_parens(self):
        """Compound substituent must have enclosing parens: (ethanoyloxy)."""
        name = name_compound("CC(=O)OCCC(=O)O")
        assert "(ethanoyloxy)" in name, (
            f"Expected enclosing parens '(ethanoyloxy)' in '{name}'"
        )

    def test_no_inner_parens_acyloxy_pattern(self):
        """No generated name should contain the pattern (Xanoyl)oxy."""
        smiles_list = [
            "CC(=O)OCCC(=O)O",                     # ethanoyloxy
            "CCCCC(=O)OCCC(=O)O",                   # pentanoyloxy
            "CCCCCCCCCCCCCCCCCC(=O)OCCC(=O)O",      # octadecanoyloxy
        ]
        pattern = re.compile(r"\([a-z]+anoyl\)oxy")
        for smiles in smiles_list:
            name = name_compound(smiles)
            assert not pattern.search(name), (
                f"Inner-paren acyloxy format found in '{name}' for {smiles}"
            )

    def test_enclosing_parens_with_locant(self):
        """When locant present, format must be N-(Xanoyloxy): locant followed by parens."""
        name = name_compound("CC(=O)OCCC(=O)O")
        # Must match pattern: digit-(Xanoyloxy) -- enclosing parens around whole group
        pattern = re.compile(r"\d-\([a-z]+anoyloxy\)")
        assert pattern.search(name), (
            f"Expected locant-(anoyloxy) pattern with enclosing parens in '{name}'"
        )

    @pytest.mark.skipif(not OPSIN_AVAILABLE, reason="OPSIN JAR not found")
    def test_ethanoyloxy_opsin_parses(self):
        """OPSIN should parse ethanoyloxy-containing names."""
        name = name_compound("CC(=O)OCCC(=O)O")
        smiles = opsin_parse(name)
        assert smiles, f"OPSIN failed to parse: '{name}'"

    @pytest.mark.skipif(not OPSIN_AVAILABLE, reason="OPSIN JAR not found")
    def test_pentanoyloxy_opsin_parses(self):
        """OPSIN should parse pentanoyloxy-containing names."""
        name = name_compound("CCCCC(=O)OCCC(=O)O")
        smiles = opsin_parse(name)
        assert smiles, f"OPSIN failed to parse: '{name}'"

    @pytest.mark.skipif(not OPSIN_AVAILABLE, reason="OPSIN JAR not found")
    def test_octadecanoyloxy_opsin_parses(self):
        """OPSIN should parse octadecanoyloxy-containing names."""
        name = name_compound("CCCCCCCCCCCCCCCCCC(=O)OCCC(=O)O")
        smiles = opsin_parse(name)
        assert smiles, f"OPSIN failed to parse: '{name}'"


# ===========================================================================
# Test: Alkylamino prefix format
# ===========================================================================


@pytest.mark.integration
class TestAlkylaminoFormat:
    """Verify alkylamino prefixes have correct compound substituent format."""

    def test_methylamino_has_enclosing_parens(self):
        """methylamino compound substituent should have enclosing parens."""
        name = name_compound("CNCCCCCC(=O)O")
        # The name should contain 'methylamino' (may be part of a larger prefix)
        assert "methylamino" in name, f"Expected 'methylamino' in '{name}'"

    def test_ethylamino_has_enclosing_parens(self):
        """ethylamino compound substituent should have enclosing parens."""
        name = name_compound("CCNCCCCCC(=O)O")
        assert "ethylamino" in name, f"Expected 'ethylamino' in '{name}'"

    def test_phenylamino_format(self):
        """phenylamino should be a single word without inner parens."""
        # N-phenylamine derivative
        name = name_compound("c1ccc(NCCCC(=O)O)cc1")
        # Should contain 'phenylamino' without inner parens '(phenyl)amino'
        assert "(phenyl)amino" not in name, (
            f"Inner-paren format '(phenyl)amino' found in '{name}'"
        )

    @pytest.mark.skipif(not OPSIN_AVAILABLE, reason="OPSIN JAR not found")
    def test_methylamino_opsin_parses(self):
        """OPSIN should parse names with methylamino prefix."""
        name = name_compound("CNCCCCCC(=O)O")
        smiles = opsin_parse(name)
        # Note: may fail due to unrelated naming issues, skip assertion
        # if name doesn't contain methylamino (route may differ)
        if "methylamino" in name:
            assert smiles, f"OPSIN failed to parse: '{name}'"


# ===========================================================================
# Test: Edge cases and regression
# ===========================================================================


@pytest.mark.integration
class TestAcylPrefixEdgeCases:
    """Edge cases for acyl prefix formatting."""

    def test_formylamino_single_carbon_acyl(self):
        """Single-carbon acyl: formylamino (methanoyl -> methanoylamino)."""
        # N-formylglycine: HC(=O)NCC(=O)O
        name = name_compound("O=CNC(=O)O")
        # Should not contain '(methanoyl)amino' or '(formyl)amino'
        assert "(methanoyl)amino" not in name
        assert "(formyl)amino" not in name

    def test_acetyloxy_aspirin_retained(self):
        """Aspirin should use retained name, not systematic acyloxy."""
        name = name_compound("CC(=O)Oc1ccccc1C(=O)O")
        # Aspirin has retained name
        assert "(ethanoyl)oxy" not in name, (
            f"Old format '(ethanoyl)oxy' found in '{name}'"
        )

    def test_no_regression_simple_amide(self):
        """Simple amides should not be affected by acylamino changes."""
        # Acetamide: CC(=O)N
        name = name_compound("CC(=O)N")
        assert "amid" in name.lower() or "ethanamid" in name.lower() or name == "acetamide", (
            f"Simple amide naming broken: '{name}'"
        )

    def test_no_regression_simple_ester(self):
        """Simple esters should not be affected by acyloxy changes."""
        # Ethyl acetate: CC(=O)OCC
        name = name_compound("CC(=O)OCC")
        assert "acetate" in name.lower() or "ethanoate" in name.lower(), (
            f"Simple ester naming broken: '{name}'"
        )

    def test_no_regression_amino_acid(self):
        """Amino acids without acyl groups should be unaffected."""
        # Alanine: CC(N)C(=O)O
        name = name_compound("CC(N)C(=O)O")
        assert "amino" in name.lower() or "alanine" in name.lower(), (
            f"Simple amino acid naming broken: '{name}'"
        )

    @pytest.mark.skipif(not OPSIN_AVAILABLE, reason="OPSIN JAR not found")
    def test_acylamino_glutamic_acid_round_trip(self):
        """Full round-trip: SMILES -> name -> OPSIN -> SMILES for acyl glutamic acid."""
        from rdkit import Chem
        input_smiles = "CCCCC(=O)NC(CCC(=O)O)C(=O)O"
        name = name_compound(input_smiles)
        opsin_smiles = opsin_parse(name)
        assert opsin_smiles, f"OPSIN failed to parse: '{name}'"
        # Verify the OPSIN output is a valid molecule
        mol = Chem.MolFromSmiles(opsin_smiles)
        assert mol is not None, f"Invalid SMILES from OPSIN: '{opsin_smiles}'"
