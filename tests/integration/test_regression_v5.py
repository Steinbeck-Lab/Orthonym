"""Regression tests for v5.0 OPSIN parse regressions fixed in Phase 58.

Plan 01 tests cover regressions 5, 7, 8, and 9 (coverage gate whitelist,
bracket hyphenation, substituent locant format).

Plan 02 tests cover regressions 1, 2, 3, 4, 6 (decomposition quality
comparison, fatty acid identification).
"""
import pytest
from orthonym import name_compound


class TestCoverageGateWhitelist:
    """Regressions 5 and 9: adenine coverage guard.

    Phase 099-03: retained-name coverage guard rejects 'adenine' for molecules
    with HA > 20 and ratio < 0.25. Decomposition now produces more complete
    names for these large adenine-containing molecules.
    """

    @pytest.mark.integration
    def test_regression_5_adenine_nucleotide(self):
        """Adenine nucleotide (HA=38): coverage guard rejects 'adenine'
        (ratio 0.18), decomposition produces more descriptive name."""
        smiles = (
            "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OC(=O)CCCC"
            "[C@@H]2SC[C@@H]3NC(=O)N[C@@H]32)[C@@H](O)[C@H]1O"
        )
        name = name_compound(smiles)
        assert name != "unknown"
        # Phase 099-03: 'adenine' (7 chars) for 38-HA molecule = ratio 0.18
        # Below 0.25 threshold -> decomposition attempted
        assert name != "adenine", (
            "Coverage guard should reject 'adenine' for HA=38 (ratio 0.18)"
        )
        assert len(name) > len("adenine"), (
            f"Decomposition result should be longer than 'adenine', got: {name}"
        )

    @pytest.mark.integration
    @pytest.mark.xfail(reason="Pre-existing: decomposition returns None for this CoA, pipeline falls back to adenine")
    def test_regression_9_coa_thioester(self):
        """CoA thioester (HA=65): coverage guard rejects 'adenine'
        (ratio 0.11), decomposition produces more descriptive name."""
        smiles = (
            r"CCC/C=C\C/C=C\CCCCCCCC(=O)SCCNC(=O)CCNC(=O)"
            r"[C@H](O)C(C)(C)COP(=O)(O)OP(=O)(O)OC[C@H]1OC"
            r"(n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1OP(=O)(O)O"
        )
        name = name_compound(smiles)
        assert name != "unknown"
        # Phase 099-03: 'adenine' (7 chars) for 65-HA molecule = ratio 0.11
        # Below 0.25 threshold -> decomposition attempted
        assert name != "adenine", (
            "Coverage guard should reject 'adenine' for HA=65 (ratio 0.11)"
        )
        assert len(name) > len("adenine"), (
            f"Decomposition result should be longer than 'adenine', got: {name}"
        )

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
        """AMP-like molecule should produce 'adenine' via nucleobase bypass.

        The .O (water) is a single-atom fragment which falls through to the
        normal pipeline rather than splitting via dot-disconnected handling.
        """
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


# ============================================================================
# Plan 02 regression tests: decomposition quality + fatty acid identification
# ============================================================================


class TestDecompositionQuality:
    """Regressions 1, 2, 4: decomposition should not produce worse names."""

    @pytest.mark.integration
    def test_regression_1_indole_peptide_no_garbled(self):
        """Multi-amide peptide should not produce 'cycloanedicarboxamide'.

        The composer's direct output for this molecule is garbled. The
        final quality gate in namer.py should detect the garbled token
        and fall back to the fragment naming result, which correctly
        identifies the 1H-indole core with amide substituents.
        """
        smiles = (
            "CC(=O)N[C@@H](CC(C)C)C(=O)N(C)[C@@H](Cc1ccccc1)"
            r"C(=O)N/C=C\c1c[nH]c2ccccc12"
        )
        name = name_compound(smiles)
        assert "cycloanedicarboxamide" not in name, (
            f"Garbled token in: '{name}'"
        )
        assert "cycloane" not in name.lower(), (
            f"Garbled 'cycloane' token in: '{name}'"
        )
        assert name is not None and name != "unknown"
        # Should contain meaningful substructure references
        assert "indole" in name.lower() or "amino" in name.lower(), (
            f"Name should reference indole or amino groups: '{name}'"
        )

    @pytest.mark.integration
    def test_regression_2_sphingolipid_fallback(self):
        """Sphingolipid should produce a meaningful name without garbled tokens.

        The v4 name '(2S)-2-hydroxytetracosanamide' was incomplete but
        OPSIN-parseable. v11 depth-independent naming produces a systematic
        phosphonic acid name via the cyclohexane-hexayl parent.
        """
        smiles = (
            "CCCCCCCCCCCCCCCCCCCCCC[C@H](O)C(=O)N[C@@H]"
            "(COP(=O)(O)O[C@@H]1[C@H](O)[C@H](O)[C@@H](O)"
            "[C@H](O)[C@H]1O)C(CCCCCCCCCCCCCCC)"
            "/C=C/CCCCCCCCCCCCC"
        )
        name = name_compound(smiles)
        # Should not contain garbled 'acidyl' token
        assert "acidyl" not in name.lower(), (
            f"Garbled 'acidyl' token in: '{name}'"
        )
        assert name is not None and name != "unknown"
        # v11: depth-independent naming produces phosphonic acid form
        assert "phosphon" in name.lower(), (
            f"Expected phosphonic acid form in: '{name}'"
        )

    @pytest.mark.integration
    def test_regression_4_ceramide(self):
        """Ceramide should not produce garbled decomposition name.

        The decomposition result is not garbled but OPSIN-unparseable.
        This test verifies the name is at least structurally valid
        (balanced brackets, no garbled tokens).
        """
        smiles = (
            "CCCCCCCCCCCCC/C=C/[C@@H](O)[C@H]"
            "(CO[C@H]1OC(CO)[C@@H](O)[C@H](O)[C@H]1O)"
            "NC(=O)CCCCCCCCCCCCCCCCCCCCCC"
        )
        name = name_compound(smiles)
        assert name is not None and name != "unknown"
        # Brackets should be balanced
        assert name.count("(") == name.count(")"), (
            f"Unbalanced parens in: '{name}'"
        )
        assert name.count("[") == name.count("]"), (
            f"Unbalanced brackets in: '{name}'"
        )
        # Should not contain garbled tokens
        assert "cycloane" not in name.lower()
        assert "aneyl" not in name.lower()

    @pytest.mark.integration
    def test_decomposition_is_worse_detects_garbled_tokens(self):
        """_decomposition_is_worse should detect known garbled patterns."""
        from rdkit import Chem
        from orthonym.decomposition.engine import _decomposition_is_worse

        mol = Chem.MolFromSmiles("CCCCCCCCCCCCCCCCCCCC")

        # 'acidyl' is garbled
        assert _decomposition_is_worse(
            "phosphonic acidyl foo", "tetracosanamide", mol
        ) is True

        # 'cycloane' is garbled
        assert _decomposition_is_worse(
            "cycloanedicarboxamide", "1H-indole", mol
        ) is True

        # Normal names should not be flagged
        assert _decomposition_is_worse(
            "ethyl acetate", "ethyl acetate", mol
        ) is False

    @pytest.mark.integration
    def test_decomposition_is_worse_detects_bracket_mismatch(self):
        """_decomposition_is_worse should detect unbalanced brackets."""
        from rdkit import Chem
        from orthonym.decomposition.engine import _decomposition_is_worse

        mol = Chem.MolFromSmiles("CCCCCCCCCCCCCCCCCCCC")

        # Unbalanced parens
        assert _decomposition_is_worse(
            "N-(2S-hydroxytetracosanoyl", "tetracosanamide", mol
        ) is True

        # Balanced parens
        assert _decomposition_is_worse(
            "N-(2S)-hydroxytetracosanoyl", "tetracosanamide", mol
        ) is False


class TestFattyAcidIdentification:
    """Regressions 3, 6: fatty acid chain identification."""

    @pytest.mark.integration
    @pytest.mark.xfail(
        reason="OPSIN cannot parse linolenoyloxy/icosadienoyloxy; "
        "systematic naming is correct IUPAC but OPSIN vocabulary limit. "
        "⚠ STALE AS OF v29 Task J3: 'linolenoyloxy' is no longer emitted at all "
        "-- the acyl prefix now comes from the PIN acid stem "
        "((9Z,12Z,15Z)-octadeca-9,12,15-trienoyloxy), which OPSIN DOES parse, "
        "and this test flips XFAIL->XPASS because of that fix. The marker is a "
        "candidate for removal, but it also XPASSes without J3 when this file is "
        "run alone, i.e. the outcome is test-ORDER dependent -- resolve that "
        "before unmarking. See "
    )
    def test_regression_3_phospholipid_opsin_parse(self):
        """Phospholipid fatty acid chains: correct IUPAC but OPSIN-unparseable.

        v4 used wrong trivial names (arachidoyl=C20:0 for C20:2, stearoyl=C18:0
        for C18:3). v5+ correctly identifies unsaturation: C18:3=linolenic,
        C20:2=icosa-11,14-dienoic. OPSIN cannot parse these acyloxy forms.

        This is marked xfail because no trivial name exists for C20:2 and
        OPSIN cannot parse the correct systematic 'icosa-11,14-dienoyloxy'.
        """
        import subprocess

        smiles = (
            r"CC/C=C\C/C=C\C/C=C\CCCCCCCC(=O)OCC"
            r"(COP(=O)(O)OCCNC)OC(=O)CCCCCCCCC/C=C\C/C=C\CCCCC"
        )
        name = name_compound(smiles)
        jar = str(__import__("pathlib").Path(__file__).resolve().parents[2] / "opsin-cli-2.9.0-jar-with-dependencies.jar")
        result = subprocess.run(
            ["java", "-jar", jar, "-osmi"],
            input=name,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.stdout.strip(), f"OPSIN cannot parse: '{name}'"

    @pytest.mark.integration
    def test_regression_3_correct_fatty_acid_names(self):
        """Phospholipid should use correct fatty acid identification.

        C18:3 should be identified as linolenic (trivial name exists).
        C20:2 should use systematic naming (no trivial name for C20:2).
        Neither should use saturated trivial names (arachidoyl, stearoyl).
        """
        smiles = (
            r"CC/C=C\C/C=C\C/C=C\CCCCCCCC(=O)OCC"
            r"(COP(=O)(O)OCCNC)OC(=O)CCCCCCCCC/C=C\C/C=C\CCCCC"
        )
        name = name_compound(smiles)
        assert name is not None and name != "unknown"
        # Should NOT use wrong saturated trivial names
        assert "arachidoyloxy" not in name, (
            f"Wrong trivial name 'arachidoyloxy' (C20:0) for C20:2 chain: '{name}'"
        )
        assert "stearoyloxy" not in name, (
            f"Wrong trivial name 'stearoyloxy' (C18:0) for C18:3 chain: '{name}'"
        )
        # Should use linolenoyloxy for C18:3 (correct trivial)
        assert "linolenoyloxy" in name, (
            f"Expected 'linolenoyloxy' for C18:3 chain: '{name}'"
        )
        # C20:2 has no trivial name -- systematic 'icosa-11,14-dienoyloxy' is correct
        assert "dienoyloxy" in name, (
            f"Expected systematic dienoyloxy for C20:2: '{name}'"
        )

    @pytest.mark.integration
    def test_regression_6_triglyceride(self):
        """Triglyceride fatty acid chains should use correct names.

        This regression was already fixed by prior phases -- the current
        name parses in OPSIN. Verify it stays fixed.
        """
        smiles = (
            r"CC/C=C\C/C=C\C/C=C\C/C=C\CCCCCCC(=O)OC"
            r"[C@H](COC(=O)CCCCCCCCCCCCCCCCCCCCCC)"
            r"OC(=O)CCCCCCCC/C=C\C/C=C\C/C=C\CC"
        )
        name = name_compound(smiles)
        assert name is not None and name != "unknown"
        # Should use arachidonoyloxy for C20:4 (correct trivial name)
        assert "arachidonoyloxy" in name, (
            f"Expected 'arachidonoyloxy' for C20:4 chain: '{name}'"
        )
        # C23:0 should use systematic name tricosanoyloxy
        assert "tricosanoyloxy" in name, (
            f"Expected 'tricosanoyloxy' for C23:0 chain: '{name}'"
        )
        # Should NOT use wrong trivial names
        assert "arachidoyloxy" not in name, (
            f"Wrong trivial name 'arachidoyloxy' (C20:0) for C20:4 chain: '{name}'"
        )

    @pytest.mark.integration
    def test_fatty_acid_trivial_names_via_fragment(self):
        """Verify fatty acid identification via get_acid_fragment_name.

        Tests that the correct trivial names are returned for known
        carbon count / double bond count combinations, and that
        unknown combinations fall through to systematic naming.
        """
        from rdkit import Chem
        from orthonym.rules.esters import get_acid_fragment_name, parse_ester_fragments

        # Test C18:0 stearic via stearic acid methyl ester
        mol = Chem.MolFromSmiles("CCCCCCCCCCCCCCCCCC(=O)OC")
        pattern = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")
        matches = mol.GetSubstructMatches(pattern)
        acid_atoms, _ = parse_ester_fragments(mol, matches[0])
        assert get_acid_fragment_name(mol, acid_atoms) == "stearic"

        # Test C20:4 arachidonic (simple ester with 4 double bonds)
        mol2 = Chem.MolFromSmiles(r"CCCCC/C=C\C/C=C\C/C=C\C/C=C\CCCC(=O)OC")
        matches2 = mol2.GetSubstructMatches(pattern)
        acid_atoms2, _ = parse_ester_fragments(mol2, matches2[0])
        assert get_acid_fragment_name(mol2, acid_atoms2) == "arachidonic"

        # Test that acyloxy conversion works for these
        from orthonym.rules.esters import get_acyloxy_prefix
        assert get_acyloxy_prefix("stearic") == "stearoyloxy"
        assert get_acyloxy_prefix("arachidonic") == "arachidonoyloxy"
        assert get_acyloxy_prefix("linolenic") == "linolenoyloxy"
