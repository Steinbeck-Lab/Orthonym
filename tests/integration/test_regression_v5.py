"""Regression tests for v5.0 OPSIN parse regressions fixed in a phase.

Plan 01 tests cover regressions 5, 7, 8, and 9 (coverage gate whitelist,
bracket hyphenation, substituent locant format).

Plan 02 tests cover regressions 1, 2, 3, 4, 6 (decomposition quality
comparison, fatty acid identification).
"""
import pytest
from orthonym import name_compound
from tests.support.rt_assert import assert_full_rt, assert_tier_contract
from tests.support.default_tier import (  # noqa: E402
    declined_pin_row,
    default_tier_rule_applies,
)

# Default tier: the paper, Methods, "Tiers" (L73): "The default configuration emits a
# name only when the pipeline can build the preferred IUPAC name (PIN); otherwise, it
# declines." User decision 2026-09-30 ("Ship it in 1.0.2"): a name the code records
# as not the PIN is declined at the default tier with NO_VERIFIED_PIN; for the
# molecules below the test asserts that decline, the strict path's name and label,
# and the same name at the best-effort tier (tests/support/default_tier.py).
DEFAULT_TIER_DECLINES = frozenset({
    "CC/C=C\\C/C=C\\C/C=C\\C/C=C\\CCCCCCC(=O)OC[C@H](COC(=O)CCCCCCCCCCCCCCCCCCCCCC)OC(=O)CCCCCCCC/C=C\\C/C=C\\C/C=C\\CC",
})
#... whose best-effort name is another one (it reads back exactly)
BEST_EFFORT_NAMES_IT_OTHERWISE = frozenset()


def _declined_pin_row(smiles):
    return declined_pin_row(
        smiles, best_effort_same=smiles not in BEST_EFFORT_NAMES_IT_OTHERWISE)


def _dt_name_compound(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)["name"]
    return name_compound(smiles)


def _dt_name(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)["name"]
    return Orthonym(style="pin").name(smiles)


def _dt_row(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)
    return Orthonym(style="pin").name_tiered(smiles)



class TestCoverageGateWhitelist:
    """Regressions 5 and 9: adenine coverage guard.

    a phase-03: retained-name coverage guard rejects 'adenine' for molecules
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
        name = _dt_name_compound(smiles)
        assert name != "unknown"
        # a phase-03: 'adenine' (7 chars) for 38-HA molecule = ratio 0.18
        # Below 0.25 threshold -> decomposition attempted
        assert name != "adenine", (
            "Coverage guard should reject 'adenine' for HA=38 (ratio 0.18)"
        )
        assert len(name) > len("adenine"), (
            f"Decomposition result should be longer than 'adenine', got: {name}"
        )

    @pytest.mark.integration
    @pytest.mark.opsin_gate
    def test_regression_9_coa_thioester(self):
        """CoA thioester (HA=65): the coverage guard rejects 'adenine' (ratio 0.11),
        and the molecule is named by the tier contract.

        Task 12 fix a performance pass (wp6-tests): the non-strict xfail XPASSed, but vacuously:
        the raw output is the failure sentinel 'unknown organic compound', which
        passed every old assertion (!= 'unknown', != 'adenine', longer than
        'adenine'). The PIN tier abstains (no CoA-thioester PIN producer) and the
        best-effort tier names it RT-exact (systematic_verified; probe 2026-09-26),
        so the test asserts the tier contract, gate ON (tests/support/rt_assert.py),
        and keeps the coverage-guard check."""
        smiles = (
            r"CCC/C=C\C/C=C\CCCCCCCC(=O)SCCNC(=O)CCNC(=O)"
            r"[C@H](O)C(C)(C)COP(=O)(O)OP(=O)(O)OC[C@H]1OC"
            r"(n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1OP(=O)(O)O"
        )
        pin, be = assert_tier_contract(smiles)
        # a phase-03: 'adenine' (7 chars) for 65-HA molecule = ratio 0.11
        assert pin != "adenine" and be != "adenine", (
            "Coverage guard should reject 'adenine' for HA=65 (ratio 0.11)")

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

        This guards against Pitfall 1 from the a phase research:
        loosening coverage gates must not reintroduce oversimplification.
        """
        # Complex polycyclic with indole substructure
        smiles = (
            "CN1C(=O)[C@]23SSS[C@@]1(CO)C(=O)N2[C@H]1Nc2ccccc2"
            "[C@@]1(c1c[nH]c2ccccc12)[C@@H]3O"
        )
        name = _dt_name_compound(smiles)
        assert name != "1H-indole", (
            f"Dense polycyclic should not be named '1H-indole' -- "
            f"coverage gate should reject this oversimplification"
        )
        assert len(name) > 15, (
            f"Name '{name}' too short for 34-atom polycyclic"
        )

    @pytest.mark.integration
    def test_adenine_monophosphate_produces_adenine(self):
        """AMP hydrate: the whole molecule is named, never the bare nucleobase.

        The old expectation 'adenine' described a different molecule (it drops the
        ribose phosphate and the water; OPSIN full InChIKey: wrong).
        """
        smiles = "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)O)[C@@H](O)[C@H]1O.O"
        name = _dt_name_compound(smiles)
        # 2026-09-26 (pre-existing-failures plan, Task 11 carry, TRIAGE row 61)
        # change-asserted-value. Controller ruling, option A (-09-24 'From T10'):
        # the retained nucleotide name stays at the PIN tier, the same as the D-a ruling
        # for sugars and amino acids. Blue Book: "## **** RETAINED NAMES" --
        # "The following are traditional names for esters of nucleosides with phosphoric
        # acid." (the Blue Book heading,:55005 sentence), listing 5'-adenylic acid (:55007); and
        # "### ** INTRODUCTION**" (:50939) -- "Preferred IUPAC names (PINs) are not identified
        # for the compounds in this Chapter." (:50943). Consistent with the three AMP gold
        # rows (DD7-natprod-2, -protect-amp, P14-NUC-AMP-PROTECT). OPSIN 2.9.0 full
        # InChIKey exact (an InChIKey), checked in Task 11 with a batch
        # OPSIN call outside the engine; the round trip is also asserted below.
        # Task 12 fix a performance pass (wp6-tests; whole-branch review items 13 and 17): this is
        # NOT a Blue Book ruling that the name is a PIN. (:50943) identifies no
        # PIN for the chapter, and (:7954) says "no PIN label will be
        # assigned in names including" water. The name is kept at the PIN tier by
        # decision (controller option A, like D-a): "no PIN identified, kept by
        # decision D-a". The test id ('..._produces_adenine') is kept so the TRIAGE row
        # still joins; it no longer describes the test, which asserts that the bare
        # nucleobase is NOT the answer.
        assert name == "5'-adenylic acid—water (1/1)", (
            f"Expected \"5'-adenylic acid—water (1/1)\", got '{name}'")
        assert_full_rt(name, smiles, "AMP hydrate: ")


class TestBracketHyphenation:
    """Regressions 7 and 8: missing hyphens after brackets and in substituent chains."""

    @pytest.mark.integration
    def test_regression_8_bracket_locant_hyphen(self):
        """Bracket followed by locant must have hyphen: ]-2 not ]2.

        This was originally caused by a TypeError crash in fused heterocycle
        locant sorting (mixed int/str), not just bracket hyphenation.
        """
        smiles = "COc1c(Cl)c2c(c(C(=O)O)c1Cl)C[C@H](C)O2"
        name = _dt_name_compound(smiles)
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
        name = _dt_name_compound(smiles)
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
    @pytest.mark.opsin_gate
    def test_regression_1_indole_peptide_no_garbled(self):
        """Multi-amide peptide should not produce 'cycloanedicarboxamide'.

        The composer's direct output for this molecule is garbled. The
        final quality gate in namer.py should detect the garbled token
        and fall back to the fragment naming result, which correctly
        identifies the 1H-indole core with amide substituents.

        2026-09-25 (pre-existing-failures plan, Task 4, TRIAGE row 62): with
        the gate off the PIN tier shipped 'N-[(1Z)-2-(1H-indol-3-yl)eth-1-en-
        1-yl](2S)-3-phenylpropanamide', which drops the acetyl-leucyl-N-methyl
        part (a residual gate-off producer, TRIAGE.md, Task 4). Checked here
        under the tier contract with the gate ON: the PIN tier, which has no
        producer for this peptide's substitutive PIN, fails closed, and the
        best-effort name is RT-exact (full InChIKey).
        """
        smiles = (
            "CC(=O)N[C@@H](CC(C)C)C(=O)N(C)[C@@H](Cc1ccccc1)"
            r"C(=O)N/C=C\c1c[nH]c2ccccc12"
        )
        _pin, name = assert_tier_contract(smiles)
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
    @pytest.mark.xfail(strict=True, reason=(
        "DEFECT (breadth + raw 0-wrong), pre-existing at 4e0e5c29b: the raw (gate-off) "
        "decomposition drops the inositol phosphodiester and the unsaturated sphingoid "
        "chain ('N-[(2S)-2-hydroxytetracosanoyl](1R,2R,3S,4S,5R,6S)-aminocyclohexane-"
        "1,2,3,4,5-pentol', OPSIN-unparseable); with the gate on the PIN tier AND the "
        "best-effort tier abstain, so no tier names this molecule (probe 2026-09-26). "
        "Needs a ceramide-phosphoinositol build. .planning/TODO-2026-09-24.md 'Open "
        "from T12 fix round 2 (wp6)'."))
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
        name = _dt_name_compound(smiles)
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
        name = _dt_name_compound(smiles)
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
    def test_regression_3_phospholipid_opsin_parse(self):
        """Phospholipid fatty acid chains: the name OPSIN-parses back to the molecule.

        v4 used wrong trivial names (arachidoyl=C20:0 for C20:2, stearoyl=C18:0
        for C18:3). v5+ correctly identifies unsaturation: C18:3=linolenic,
        C20:2=icosa-11,14-dienoic.

        Task 12 fix a performance pass (wp6-tests): the non-strict xfail ('OPSIN cannot parse
        linolenoyloxy/icosadienoyloxy') was stale -- made the acyl
        prefixes come from the PIN acid stems, which OPSIN parses, and the test
        XPASSed. The order dependence its reason warned about was checked: the test
        passes alone and inside the file (2026-09-26). The assertion is strengthened
        from "OPSIN parses it" to the full-InChIKey round trip (independent OPSIN
        call). The spelling is not pinned: whether this acyloxy-on-propane /
        hydroxyphosphoryl form is the PIN is open (the 'hydrogen phosphate' form for
        phospholipids, TRIAGE.md ' outcome').
        """
        smiles = (
            r"CC/C=C\C/C=C\C/C=C\CCCCCCCC(=O)OCC"
            r"(COP(=O)(O)OCCNC)OC(=O)CCCCCCCCC/C=C\C/C=C\CCCCC"
        )
        name = _dt_name_compound(smiles)
        from tests.support.jars import jar_or_skip
        jar_or_skip()
        assert_full_rt(name, smiles, "phospholipid: ")

    @pytest.mark.integration
    def test_regression_3_correct_fatty_acid_names(self):
        """Phospholipid should use correct fatty acid identification.

        Both chains use systematic acyl prefixes (C18:3 octadeca-9,12,15-trienoyl,
        C20:2 icosa-11,14-dienoyl); no retained or saturated trivial names
        (linolenoyl, arachidoyl, stearoyl).
        """
        smiles = (
            r"CC/C=C\C/C=C\C/C=C\CCCCCCCC(=O)OCC"
            r"(COP(=O)(O)OCCNC)OC(=O)CCCCCCCCC/C=C\C/C=C\CCCCC"
        )
        name = _dt_name_compound(smiles)
        assert name is not None and name != "unknown"
        # Should NOT use wrong saturated trivial names
        assert "arachidoyloxy" not in name, (
            f"Wrong trivial name 'arachidoyloxy' (C20:0) for C20:2 chain: '{name}'"
        )
        assert "stearoyloxy" not in name, (
            f"Wrong trivial name 'stearoyloxy' (C18:0) for C18:3 chain: '{name}'"
        )
        # 2026-09-25 (pre-existing-failures plan, Task 5, TRIAGE row 65, ruling
        # R20): the retained fatty-acid names are not used in PINs --
        # (the Blue Book), "stearic acid... octadecanoic acid (PIN)" (:29791)
        # -- so no 'linolenoyl'. Each acyl is cited in its own marks with 'oxy'
        # outside: '3-[(pyridine-3-carbonyl)oxy]propanoic acid (PIN)',
        #:31723), and the stereo-led acyl takes '[' then '{' by the order
        # (:7446). OPSIN round trip of the whole name: full-InChIKey exact.
        assert "linolenoyl" not in name, (
            f"Retained 'linolenoyl' is not a PIN acyl prefix (R20): '{name}'"
        )
        assert "{[(9Z,12Z,15Z)-octadeca-9,12,15-trienoyl]oxy}" in name, (
            f"Expected systematic C18:3 acyloxy prefix: '{name}'"
        )
        # fix-all 2026-10-09: the C20:2 ester is the functional-class word now, the
        # alcohol component cited before it 'Esters', the Blue Book,
        # 'All preferred IUPAC names for esters are named by functional class
        # nomenclature'); OPSIN 2.9.0 reads the whole name back full-InChIKey exact.
        assert "propan-2-yl (11Z,14Z)-icosa-11,14-dienoate" in name, (
            f"Expected the systematic C20:2 ester word: '{name}'"
        )

    @pytest.mark.integration
    @pytest.mark.opsin_gate
    def test_regression_6_triglyceride(self):
        """Triglyceride fatty acid chains should use correct names.

        This regression was already fixed by prior phases -- the current
        name parses in OPSIN. Verify it stays fixed.

        2026-09-25 (pre-existing-failures plan, Task 4, TRIAGE row 66):
        - The C20:4 chain here is (8Z,11Z,14Z,17Z)-icosa-8,11,14,17-tetraenoyl,
          not arachidonoyl (5Z,8Z,11Z,14Z), and trivial acyl prefixes are not
          PIN prefixes anyway (plan ruling R20:, the Blue Book,
          'stearic acid octadecanoic acid (PIN)':29791). So the old
          'arachidonoyloxy' expectation is dropped and a trivial acyl prefix is
          refused instead.
        - With the gate off the glyceride producer ships the
          method (1) form without the enclosing marks around its stereo-bearing
          anions, which OPSIN cannot parse (a residual, TRIAGE.md, Task 4). The
          shipped name (gate ON) is checked under the tier contract: RT-exact
          by full InChIKey at both tiers.
        """
        smiles = (
            r"CC/C=C\C/C=C\C/C=C\C/C=C\CCCCCCC(=O)OC"
            r"[C@H](COC(=O)CCCCCCCCCCCCCCCCCCCCCC)"
            r"OC(=O)CCCCCCCC/C=C\C/C=C\C/C=C\CC"
        )
        name, _be = assert_tier_contract(smiles)
        if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
            name = _declined_pin_row(smiles)["name"]
        # C23:0 should use systematic name tricosanoyloxy
        assert "tricosanoyloxy" in name, (
            f"Expected 'tricosanoyloxy' for C23:0 chain: '{name}'"
        )
        # No trivial fatty-acyl prefix (R20)
        for trivial in ("arachid", "linole", "palmitoyl", "stearoyl", "oleoyl"):
            assert trivial not in name, f"trivial acyl prefix {trivial!r} in: '{name}'"

    @pytest.mark.integration
    def test_fatty_acid_trivial_names_via_fragment(self):
        """Verify fatty acid identification via get_acid_fragment_name.

        The acid stems are the systematic PIN stems (plan ruling R20), not the
        trivial 'stearic'/'arachidonic'. The get_acyloxy_prefix asserts below test
        a spelling converter on its own input and are unchanged.
        """
        from rdkit import Chem
        from orthonym.rules.esters import get_acid_fragment_name, parse_ester_fragments

        # Test C18:0 stearic via stearic acid methyl ester
        mol = Chem.MolFromSmiles("CCCCCCCCCCCCCCCCCC(=O)OC")
        pattern = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")
        matches = mol.GetSubstructMatches(pattern)
        acid_atoms, _ = parse_ester_fragments(mol, matches[0])
        # PIN per R20: "The following names are retained for general
        # nomenclature with functionalization but no substitution is allowed."
        # the Blue Book; "stearic acid octadecanoic acid (PIN)":29791. OPSIN RT
        # exact: 'octadecanoic acid' -> CCCCCCCCCCCCCCCCCC(=O)O, full InChIKey (Task 7/8).
        assert get_acid_fragment_name(mol, acid_atoms) == "octadecanoic"

        # Test C20:4 arachidonic (simple ester with 4 double bonds)
        mol2 = Chem.MolFromSmiles(r"CCCCC/C=C\C/C=C\C/C=C\C/C=C\CCCC(=O)OC")
        matches2 = mol2.GetSubstructMatches(pattern)
        acid_atoms2, _ = parse_ester_fragments(mol2, matches2[0])
        # PIN per R20: 'arachidonic acid' is not even a retained name in the Blue Book (0
        # hits); "Only the following five carboxylic acids retained names and are
        # also preferred IUPAC names." the Blue Book; stereo format as in "oleic acid
        # (9Z)-octadec-9-enoic acid (PIN)":29785. OPSIN RT exact:
        # '(5Z,8Z,11Z,14Z)-icosa-5,8,11,14-tetraenoic acid' -> the acid of this ester, full
        # InChIKey (Task 7/8).
        assert get_acid_fragment_name(mol2, acid_atoms2) == "(5Z,8Z,11Z,14Z)-icosa-5,8,11,14-tetraenoic"

        # Test that acyloxy conversion works for these
        from orthonym.rules.esters import get_acyloxy_prefix
        assert get_acyloxy_prefix("stearic") == "stearoyloxy"
        assert get_acyloxy_prefix("arachidonic") == "arachidonoyloxy"
        assert get_acyloxy_prefix("linolenic") == "linolenoyloxy"
