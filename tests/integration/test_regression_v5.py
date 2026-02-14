"""Regression tests for v5.0 OPSIN parse regressions fixed in Phase 58.

Tests cover regressions 5, 7, 8, and 9 from the v5.0 gap analysis.
Each test verifies the specific fix (coverage gate whitelist, bracket
hyphenation, substituent locant format) without depending on exact
name strings where possible.
"""
import pytest
from orthonym import name_compound


class TestCoverageGateWhitelist:
    """Regressions 5 and 9: adenine retained name not rejected by quality gate."""

    @pytest.mark.integration
    def test_regression_5_adenine_nucleotide(self):
        """Adenine nucleotide cofactor should produce 'adenine', not garbled systematic."""
        smiles = (
            "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OC(=O)CCCC"
            "[C@@H]2SC[C@@H]3NC(=O)N[C@@H]32)[C@@H](O)[C@H]1O"
        )
        name = name_compound(smiles)
        assert name == "adenine", f"Expected 'adenine', got '{name}'"

    @pytest.mark.integration
    def test_regression_9_coa_thioester(self):
        """CoA thioester should produce 'adenine', not garbled decomposition."""
        smiles = (
            r"CCC/C=C\C/C=C\CCCCCCCC(=O)SCCNC(=O)CCNC(=O)"
            r"[C@H](O)C(C)(C)COP(=O)(O)OP(=O)(O)OC[C@H]1OC"
            r"(n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1OP(=O)(O)O"
        )
        name = name_compound(smiles)
        assert name == "adenine", f"Expected 'adenine', got '{name}'"

    @pytest.mark.integration
    def test_whitelist_does_not_weaken_gate_for_small_rings(self):
        """Coverage gate still rejects oversimplified names for large molecules.

        Common small-ring names that are NOT nucleobases should NOT bypass
        the coverage gate, even for molecules containing those substructures.
        """
        from orthonym.decomposition.engine import _RETAINED_CORE_NAMES
        # Nucleobase names should be in the whitelist
        assert "adenine" in _RETAINED_CORE_NAMES
        assert "guanine" in _RETAINED_CORE_NAMES
        # Common ring names should NOT be in the whitelist
        assert "benzene" not in _RETAINED_CORE_NAMES
        assert "cyclohexane" not in _RETAINED_CORE_NAMES
        assert "pyridine" not in _RETAINED_CORE_NAMES

    @pytest.mark.integration
    def test_nucleobase_bypass_does_not_affect_dense_polycyclics(self):
        """Dense polycyclic molecules containing indole should NOT return
        just '1H-indole' -- the coverage gate should reject it even though
        1H-indole is in the engine quality gate whitelist.

        This guards against Pitfall 1 from the Phase 58 research:
        loosening coverage gates must not reintroduce oversimplification.
        """
        # Complex polycyclic with indole substructure
        smiles = (
            "CN1C(=O)[C@]23SSS[C@@]1(CO)C(=O)N2[C@H]1Nc2ccccc2"
            "[C@@]1(c1c[nH]c2ccccc12)[C@@H]3O"
        )
        name = name_compound(smiles)
        assert name != "1H-indole", (
            f"Dense polycyclic should not be named '1H-indole' -- "
            f"coverage gate should reject this oversimplification"
        )
        assert len(name) > 15, (
            f"Name '{name}' too short for 34-atom polycyclic"
        )

    @pytest.mark.integration
    def test_adenine_monophosphate_produces_adenine(self):
        """AMP-like molecule should produce 'adenine' via nucleobase bypass."""
        smiles = "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)O)[C@@H](O)[C@H]1O.O"
        name = name_compound(smiles)
        assert name == "adenine", f"Expected 'adenine', got '{name}'"


class TestBracketHyphenation:
    """Regressions 7 and 8: missing hyphens after brackets and in substituent chains."""

    @pytest.mark.integration
    def test_regression_8_bracket_locant_hyphen(self):
        """Bracket followed by locant must have hyphen: ]-2 not ]2.

        This was originally caused by a TypeError crash in fused heterocycle
        locant sorting (mixed int/str), not just bracket hyphenation.
        """
        smiles = "COc1c(Cl)c2c(c(C(=O)O)c1Cl)C[C@H](C)O2"
        name = name_compound(smiles)
        # The name should not contain "]2" (missing hyphen)
        assert "]2" not in name and "]3" not in name and "]4" not in name, (
            f"Missing hyphen after bracket in: '{name}'"
        )
        # Should produce a valid fused heterocycle name
        assert name is not None and name != "unknown"
        assert len(name) > 10, f"Name too short: '{name}'"

    @pytest.mark.integration
    def test_regression_7_substituent_chain_format(self):
        """Substituent chain should have proper hyphenation between
        hydroxy prefix and locant: '1-hydroxy-3-methylbut-2-enyl'
        not 'hydroxy3-methylbut-2-en-1-yl'.
        """
        smiles = "CC(C)=C[C@H](O)C1=CC(=O)[C@@H](O)[C@H](O)[C@H]1O"
        name = name_compound(smiles)
        # Should NOT contain "hydroxy3-" (missing hyphen)
        assert "hydroxy3" not in name, (
            f"Missing hyphen in substituent: '{name}'"
        )
        # Should contain properly formatted name
        assert name is not None and name != "unknown"

    @pytest.mark.integration
    def test_bracket_hyphenation_in_join_prefixes(self):
        """Verify that _join_prefixes inserts hyphen after ] when
        followed by a digit."""
        from orthonym.assembly.composer import _join_prefixes
        result = _join_prefixes(["5-[(S)-isopropoxy]", "2,4-dichloro"])
        assert "]-" in result or "]2" not in result, (
            f"Missing hyphen after ] in: '{result}'"
        )

    @pytest.mark.integration
    def test_bracket_hyphenation_in_join_prefix_to_name(self):
        """Verify that _join_prefix_to_name inserts hyphen after ] when
        followed by a digit."""
        from orthonym.assembly.composer import _join_prefix_to_name
        result = _join_prefix_to_name("5-[(S)-isopropoxy]", "2,4-dichloro")
        assert result == "5-[(S)-isopropoxy]-2,4-dichloro", (
            f"Expected '5-[(S)-isopropoxy]-2,4-dichloro', got '{result}'"
        )
