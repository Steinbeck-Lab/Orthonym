"""
Integration tests for a phase: Macrocyclic & Misc Format fixes.
Tests (replacement prefix hyphen joining) and (large ring chain prefix verification).
"""
import re
import subprocess

import pytest

from orthonym.namer import name_compound


def opsin_parse(name: str) -> str | None:
    """Parse IUPAC name through OPSIN, return SMILES or None.

    a phase cleanup: routed through `_find_opsin_jar` (a phase
    canonical helper). The hardcoded `opsin-cli-2.8.0-...jar` was dead
    since the project upgraded to opsin-cli-2.9.0.
    """
    from tests.support.jars import jar_or_none

    opsin_jar = jar_or_none()
    if opsin_jar is None:
        return None
    try:
        result = subprocess.run(
            ["java", "-jar", opsin_jar, "-osmi"],
            input=name,
            capture_output=True,
            text=True,
            timeout=30,
        )
        # OPSIN prints the prompt + result; take last non-empty line
        lines = [l.strip() for l in result.stdout.split("\n") if l.strip()]
        smiles = lines[-1] if lines else ""
        # Filter out the prompt line that doesn't have parseable SMILES
        if "help" in smiles.lower() or not smiles:
            return None
        return smiles
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


@pytest.mark.integration
class TestReplacementPrefixHyphen:
    """: Mixed-element macrocyclic replacement prefixes use hyphen separator."""

    def test_oxa_aza_cyclododecane(self):
        """O+N macrocyclic must have hyphen between oxa and aza groups."""
        # 12-membered ring with O and N
        smiles = "C1CCNCCCCCCOC1"
        result = name_compound(smiles)
        assert result is not None
        # Must NOT have 'oxa' immediately followed by a digit without hyphen
        assert not re.search(r"oxa\d", result), (
            f"Found 'oxa' followed by digit without hyphen in '{result}'"
        )
        # Verify correct hyphen-separated format
        assert "oxa-" in result, (
            f"Expected hyphen after 'oxa' group in '{result}'"
        )

    def test_dioxa_diaza_macrocycle(self):
        """Crown ether with O+N must have hyphens between element groups."""
        # 15-ring with 2 O, 2 N
        smiles = "C1CNCCCOCCCNCCCO1"
        result = name_compound(smiles)
        assert result is not None
        # Should have 'dioxa-' followed by locants for aza group
        assert not re.search(r"oxa\d", result), (
            f"Found concatenated prefix groups (missing hyphen) in '{result}'"
        )

    def test_single_element_unchanged(self):
        """Single-element replacement prefix (only O, only N) has no extra hyphen."""
        # 1,4,7,10-tetraoxacyclododecane (12-crown-4)
        smiles = "C1COCCOCCOCCO1"
        result = name_compound(smiles)
        assert result is not None
        assert "tetraoxa" in result, f"Expected 'tetraoxa' in '{result}'"
        assert "cyclododecane" in result, f"Expected 'cyclododecane' in '{result}'"

    def test_oxa_thia_macrocycle(self):
        """O+S macrocyclic must have hyphen between oxa and thia groups."""
        smiles = "C1CSCCCCCCOCC1"  # 12-ring with O and S
        result = name_compound(smiles)
        assert result is not None
        # Should not have oxa/thia followed directly by digit
        assert not re.search(r"(oxa|thia)\d", result), (
            f"Missing hyphen between replacement prefix groups in '{result}'"
        )

    def test_oxa_aza_exact_format(self):
        """O+N 12-ring produces correctly formatted name."""
        smiles = "C1CCNCCCCCCOC1"
        result = name_compound(smiles)
        assert result is not None
        assert "oxa" in result
        assert "aza" in result
        assert "cyclododecane" in result

    def test_dioxa_diaza_exact_format(self):
        """O+N 15-ring produces correctly formatted name with multipliers."""
        smiles = "C1CNCCCOCCCNCCCO1"
        result = name_compound(smiles)
        assert result is not None
        assert "dioxa" in result
        assert "diaza" in result
        assert "cyclopentadecane" in result

    def test_oxa_thia_exact_format(self):
        """O+S 12-ring produces correctly formatted name."""
        smiles = "C1CSCCCCCCOCC1"
        result = name_compound(smiles)
        assert result is not None
        assert "oxa" in result
        assert "thia" in result
        assert "cyclododecane" in result


@pytest.mark.integration
class TestReplacementPrefixOPSINRoundTrip:
    """: OPSIN round-trip tests for macrocyclic replacement names."""

    @pytest.mark.roundtrip
    def test_tetraoxacyclododecane_opsin(self):
        """12-crown-4 name parses through OPSIN."""
        smiles = "C1COCCOCCOCCO1"
        result = name_compound(smiles)
        assert result is not None
        parsed = opsin_parse(result)
        assert parsed is not None, f"OPSIN rejected '{result}'"

    @pytest.mark.roundtrip
    def test_mixed_heteroatom_macrocycle_opsin(self):
        """Mixed O+N macrocyclic name parses through OPSIN."""
        smiles = "C1CCNCCCCCCOC1"
        result = name_compound(smiles)
        assert result is not None
        parsed = opsin_parse(result)
        assert parsed is not None, f"OPSIN rejected '{result}'"

    @pytest.mark.roundtrip
    def test_oxa_thia_macrocycle_opsin(self):
        """Mixed O+S macrocyclic name parses through OPSIN."""
        smiles = "C1CSCCCCCCOCC1"
        result = name_compound(smiles)
        assert result is not None
        parsed = opsin_parse(result)
        assert parsed is not None, f"OPSIN rejected '{result}'"


@pytest.mark.integration
class TestLargeRingChainPrefix:
    """: Verify get_chain_prefix produces correct names for large rings."""

    def test_cyclononadecane(self):
        """19-membered ring produces correct name."""
        smiles = "C1" + "C" * 17 + "C1"  # cyclononadecane
        result = name_compound(smiles)
        assert result is not None
        assert "cyclononadecane" in result, f"Expected 'cyclononadecane' in '{result}'"

    def test_cycloicosane(self):
        """20-membered ring produces correct name."""
        smiles = "C1" + "C" * 18 + "C1"
        result = name_compound(smiles)
        assert result is not None
        assert "cycloicosane" in result, f"Expected 'cycloicosane' in '{result}'"

    def test_cyclohenicosane(self):
        """21-membered ring produces correct name."""
        smiles = "C1" + "C" * 19 + "C1"
        result = name_compound(smiles)
        assert result is not None
        assert "cyclohenicosane" in result, f"Expected 'cyclohenicosane' in '{result}'"

    def test_cyclodotriacontane(self):
        """32-membered ring produces correct name."""
        smiles = "C1" + "C" * 30 + "C1"
        result = name_compound(smiles)
        assert result is not None
        assert "cyclodotriacontane" in result, (
            f"Expected 'cyclodotriacontane' in '{result}'"
        )

    def test_cyclotetracosane(self):
        """24-membered ring produces correct name."""
        smiles = "C1" + "C" * 22 + "C1"
        result = name_compound(smiles)
        assert result is not None
        assert "cyclotetracosane" in result, (
            f"Expected 'cyclotetracosane' in '{result}'"
        )

    def test_cyclotriacontane(self):
        """30-membered ring produces correct name."""
        smiles = "C1" + "C" * 28 + "C1"
        result = name_compound(smiles)
        assert result is not None
        assert "cyclotriacontane" in result, (
            f"Expected 'cyclotriacontane' in '{result}'"
        )

    @pytest.mark.roundtrip
    def test_cyclononadecane_opsin(self):
        """19-ring name parses through OPSIN."""
        smiles = "C1" + "C" * 17 + "C1"
        result = name_compound(smiles)
        assert result is not None
        parsed = opsin_parse(result)
        assert parsed is not None, f"OPSIN rejected '{result}'"

    @pytest.mark.roundtrip
    def test_cyclohenicosane_opsin(self):
        """21-ring name parses through OPSIN."""
        smiles = "C1" + "C" * 19 + "C1"
        result = name_compound(smiles)
        assert result is not None
        parsed = opsin_parse(result)
        assert parsed is not None, f"OPSIN rejected '{result}'"

    @pytest.mark.roundtrip
    def test_cyclodotriacontane_opsin(self):
        """32-ring name parses through OPSIN."""
        smiles = "C1" + "C" * 30 + "C1"
        result = name_compound(smiles)
        assert result is not None
        parsed = opsin_parse(result)
        assert parsed is not None, f"OPSIN rejected '{result}'"
