"""Integration tests for end-to-end glycoside naming.

Tests the full pipeline from SMILES to glycoside name using name_compound()
from orthonym.namer. Validates that known sugar moieties produce glycosyloxy
prefixes, format is correct, no systematic oxane names appear, and non-glycoside
molecules are unaffected.
"""

import pytest
from orthonym.namer import name_compound


# ============================================================================
# Basic glycoside naming
# ============================================================================


@pytest.mark.integration
class TestBasicGlycosideNaming:
    """Simple O-glycosides where the sugar matches the lookup table."""

    def test_phenyl_beta_d_glucoside(self):
        """Phenyl beta-D-glucopyranoside: functional-class form (WSD-08 flip).

        Phase 176 re-baseline: the simple phenol aglycone flips from the
        legacy substitutive ``(beta-D-glucopyranosyloxy)...`` form to the
        Blue-Book functional-class two-word form ``phenyl beta-D-glucopyranoside``
        (P-102.5.6.2.2). OPSIN-RT-verified.
        """
        # Phenol + beta-D-glucose via O-glycosidic bond at anomeric position
        name = name_compound("OC[C@H]1O[C@@H](Oc2ccccc2)[C@H](O)[C@@H](O)[C@@H]1O")
        assert name is not None
        assert name != "unknown"
        assert name == "phenyl beta-D-glucopyranoside"

    def test_methyl_alpha_d_glucoside(self):
        """Methyl alpha-D-glucopyranoside: functional-class form (WSD-08 flip)."""
        name = name_compound("OC[C@H]1O[C@H](OC)[C@H](O)[C@@H](O)[C@@H]1O")
        assert name is not None
        assert name != "unknown"
        assert name == "methyl alpha-D-glucopyranoside"

    def test_galactoside(self):
        """4-Hydroxyphenyl beta-D-galactopyranoside stays legacy (diol aglycone).

        The aglycone here is hydroquinone (a diol), which is NOT a clean
        monovalent substituent (``_alcohol_to_alkyl`` cannot produce a -yl
        prefix), so the D-10 triad fails closed to the legacy substitutive
        ``(beta-D-galactopyranosyloxy)hydroquinone`` form (zero regression).
        """
        name = name_compound(
            "OC[C@H]1O[C@@H](Oc2ccc(O)cc2)[C@H](O)[C@@H](O)[C@H]1O"
        )
        assert name is not None
        assert name != "unknown"
        assert "galactopyranosyloxy" in name.lower()

    def test_rhamnoside(self):
        """Phenyl alpha-L-rhamnopyranoside: functional-class form (WSD-08 flip).

        Phase 176: deoxy-L rhamnose is catalog-only (the structure deriver
        returns None on it), but the ``lookup_sugar`` fast-path resolves the
        full ``(alpha, L, rhamnopyranose)`` tuple, so the inversion still fires
        (descriptor source = catalog). OPSIN-RT-verified.
        """
        name = name_compound("C[C@@H]1O[C@@H](Oc2ccccc2)[C@H](O)[C@H](O)[C@H]1O")
        assert name is not None
        assert name != "unknown"
        assert name == "phenyl alpha-L-rhamnopyranoside"

    def test_ethyl_glucoside(self):
        """Ethyl beta-D-glucopyranoside: functional-class form (WSD-08 flip)."""
        name = name_compound("OC[C@H]1O[C@@H](OCC)[C@H](O)[C@@H](O)[C@@H]1O")
        assert name is not None
        assert name != "unknown"
        assert name == "ethyl beta-D-glucopyranoside"

    def test_naphthyl_glucoside(self):
        """2-Naphthyl beta-D-glucopyranoside: functional-class form (WSD-08 flip)."""
        name = name_compound(
            "OC[C@H]1O[C@@H](Oc2ccc3ccccc3c2)[C@H](O)[C@@H](O)[C@@H]1O"
        )
        assert name is not None
        assert name != "unknown"
        assert name == "2-naphthyl beta-D-glucopyranoside"


# ============================================================================
# Glycoside name format validation
# ============================================================================


@pytest.mark.integration
class TestGlycosideNameFormat:
    """Verify the assembled glycoside name format is correct."""

    def test_glycoside_functional_class_form(self):
        """Phenyl glucoside emits the functional-class two-word form (WSD-08).

        Phase 176 re-baseline: was a parenthesized substitutive prefix
        ``(beta-D-glucopyranosyloxy)...``; now the Blue-Book functional-class
        form ``phenyl beta-D-glucopyranoside`` (aglycone substituent as a
        separate preceding word, sugar -ose -> -oside). OPSIN-RT-verified.
        """
        name = name_compound("OC[C@H]1O[C@@H](Oc2ccccc2)[C@H](O)[C@@H](O)[C@@H]1O")
        assert name == "phenyl beta-D-glucopyranoside"
        # The alpha/beta + D/L descriptors are never dropped (D-11)
        assert "beta-D-glucopyranoside" in name
        # Aglycone is cited as a separate preceding substituent word
        assert name.startswith("phenyl ")

    def test_glycoside_name_length_reasonable(self):
        """Glycoside name should be > 20 chars (not a truncated fragment)."""
        name = name_compound("OC[C@H]1O[C@@H](Oc2ccccc2)[C@H](O)[C@@H](O)[C@@H]1O")
        assert len(name) > 20

    def test_glycoside_not_oxane(self):
        """Glycoside name should NOT contain 'oxan' (systematic oxane)."""
        name = name_compound("OC[C@H]1O[C@@H](Oc2ccccc2)[C@H](O)[C@@H](O)[C@@H]1O")
        assert "oxan" not in name.lower(), (
            f"Name should use retained sugar name, not systematic oxane: {name}"
        )

    def test_galactoside_has_anomer_config(self):
        """Galactoside name should include anomer and config descriptors."""
        name = name_compound(
            "OC[C@H]1O[C@@H](Oc2ccc(O)cc2)[C@H](O)[C@@H](O)[C@H]1O"
        )
        assert "beta-D-" in name

    def test_rhamnoside_has_l_config(self):
        """Rhamnoside name should include L-configuration descriptor."""
        name = name_compound("C[C@@H]1O[C@@H](Oc2ccccc2)[C@H](O)[C@H](O)[C@H]1O")
        assert "alpha-L-" in name


# ============================================================================
# Senior-aglycone PROTECT: must NOT flip to functional-class form (D-09/D-14)
# ============================================================================


@pytest.mark.integration
class TestGlycosideSeniorAglyconeProtect:
    """A glycoside whose aglycone bears a group senior to hydroxy must stay
    on the legacy substitutive ``(glycosyloxy)R`` form (P-102.5.6.1.1): the
    ``-ose`` ending is retained and the aglycone is cited as an O-substituent
    prefix. The structural seniority guard (rank >= primary_alcohol == 54)
    fails closed for ketone/acid/aldehyde aglycones (D-09)."""

    def test_aldehyde_aglycone_stays_legacy(self):
        """4-Formylphenyl beta-D-glucoside (aldehyde aglycone) keeps legacy form.

        The aglycone (4-hydroxybenzaldehyde) has a principal aldehyde group,
        senior to hydroxy, so the D-09 guard fails closed and the name retains
        the substitutive ``...glucopyranosyloxy...`` (NOT a -glucopyranoside
        functional-class form).
        """
        name = name_compound(
            "O=Cc1ccc(O[C@@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)cc1"
        )
        assert name is not None
        assert name != "unknown"
        lower = name.lower()
        assert "glycosyloxy" in lower or "pyranosyloxy" in lower
        # Must NOT have flipped to the functional-class -oside form
        assert "glucopyranoside" not in lower

    def test_aglycone_to_substituent_guard_blocks_senior_groups(self):
        """Unit-level D-09 PROTECT: _aglycone_to_substituent returns None for
        ketone / carboxylic-acid / aldehyde aglycones, and a -yl prefix for
        the in-scope hydroxy-class aglycones."""
        from orthonym.decomposition.fragment_assembly import (
            _aglycone_to_substituent,
        )
        # Senior to hydroxy -> None (keep legacy)
        assert _aglycone_to_substituent("propan-2-one", "CC(C)=O") is None
        assert _aglycone_to_substituent("acetic acid", "CC(=O)O") is None
        assert _aglycone_to_substituent("acetaldehyde", "CC=O") is None
        # Hydroxy-class or junior -> substituent prefix (flip allowed)
        assert _aglycone_to_substituent("methanol", "CO") == "methyl"
        assert _aglycone_to_substituent("ethanol", "CCO") == "ethyl"
        assert _aglycone_to_substituent("phenol", "Oc1ccccc1") == "phenyl"
        assert (
            _aglycone_to_substituent("2-aminoethanol", "NCCO") == "2-aminoethyl"
        )
        # Diol aglycone (no clean -yl prefix) -> None (D-10 fail-closed)
        assert _aglycone_to_substituent("hydroquinone", "Oc1ccc(O)cc1") is None


# ============================================================================
# Regression tests: non-glycoside molecules unchanged
# ============================================================================


@pytest.mark.integration
class TestGlycosideRegressions:
    """Ensure non-glycoside molecules are unaffected by glycoside naming."""

    def test_simple_ester_unchanged(self):
        """Ethyl acetate still produces 'ethyl acetate'."""
        name = name_compound("CC(=O)OCC")
        assert name == "ethyl acetate"

    def test_simple_amide_unchanged(self):
        """N-methylacetamide still produces expected name."""
        name = name_compound("CC(=O)NC")
        assert name == "N-methylacetamide"

    def test_benzene_unchanged(self):
        """Benzene still produces 'benzene'."""
        name = name_compound("c1ccccc1")
        assert name == "benzene"

    def test_ethanol_unchanged(self):
        """Ethanol still produces 'ethanol'."""
        name = name_compound("CCO")
        assert name == "ethanol"

    def test_acetic_acid_unchanged(self):
        """Acetic acid still produces 'acetic acid'."""
        name = name_compound("CC(=O)O")
        assert name == "acetic acid"


# ============================================================================
# Benchmark glycoside coverage
# ============================================================================


@pytest.mark.integration
class TestGlycosideBenchmarkCoverage:
    """Test against actual benchmark glycoside SMILES."""

    # Benchmark glycoside SMILES with known sugar moieties
    BENCHMARK_GLYCOSIDES = [
        # 1. Phenyl beta-D-glucoside (stereo, hexopyranose)
        "OC[C@H]1O[C@@H](Oc2ccccc2)[C@H](O)[C@@H](O)[C@@H]1O",
        # 2. Methyl alpha-D-glucoside (stereo, hexopyranose)
        "OC[C@H]1O[C@H](OC)[C@H](O)[C@@H](O)[C@@H]1O",
        # 3. 4-hydroxyphenyl beta-D-galactoside (stereo, galactose)
        "OC[C@H]1O[C@@H](Oc2ccc(O)cc2)[C@H](O)[C@@H](O)[C@H]1O",
        # 4. Phenyl alpha-L-rhamnoside (stereo, deoxy-L-sugar)
        "C[C@@H]1O[C@@H](Oc2ccccc2)[C@H](O)[C@H](O)[C@H]1O",
        # 5. Ethyl beta-D-glucoside (stereo, hexopyranose + small aglycone)
        "OC[C@H]1O[C@@H](OCC)[C@H](O)[C@@H](O)[C@@H]1O",
        # 6. Non-stereo diglucosylchromane from CI benchmark
        "OCC1OC(Oc2cc(O)c3c(c2)OC(c2ccc(O)c(OC4OC(CO)C(O)C(O)C4O)c2)"
        "C(O)C3)C(O)C(O)C1O",
    ]

    @pytest.mark.parametrize("smiles", BENCHMARK_GLYCOSIDES, ids=[
        "phenyl-beta-D-glucoside",
        "methyl-alpha-D-glucoside",
        "hydroxyphenyl-beta-D-galactoside",
        "phenyl-alpha-L-rhamnoside",
        "ethyl-beta-D-glucoside",
        "diglucosyl-chromane",
    ])
    def test_benchmark_glycoside_produces_name(self, smiles):
        """Benchmark glycoside produces a non-trivial name (not None/unknown)."""
        name = name_compound(smiles)
        assert name is not None, f"Name should not be None for {smiles}"
        assert name != "unknown", f"Name should not be 'unknown' for {smiles}"
        assert len(name) > 15, f"Name too short ({len(name)}): {name}"

    @pytest.mark.parametrize("smiles", BENCHMARK_GLYCOSIDES, ids=[
        "phenyl-beta-D-glucoside",
        "methyl-alpha-D-glucoside",
        "hydroxyphenyl-beta-D-galactoside",
        "phenyl-alpha-L-rhamnoside",
        "ethyl-beta-D-glucoside",
        "diglucosyl-chromane",
    ])
    def test_benchmark_glycoside_has_sugar_name(self, smiles):
        """Benchmark glycoside name contains a sugar-related substring.

        Phase 176: the accepted set now includes the functional-class
        ``-pyranoside``/``-furanoside`` forms (the flipped simple-aglycone
        glycosides) alongside the legacy ``...pyranosyloxy`` substitutive
        forms (gated-out / multi-sugar rows).
        """
        name = name_compound(smiles)
        sugar_substrings = [
            "pyranosyloxy", "furanosyloxy", "glycosyloxy",
            "pyranoside", "furanoside",
        ]
        has_sugar = any(s in name.lower() for s in sugar_substrings)
        assert has_sugar, (
            f"Expected sugar-related substring in name for {smiles}, got: {name}"
        )

    def test_benchmark_coverage_threshold(self):
        """At least 4 of 6 benchmark glycosides have sugar names."""
        sugar_count = 0
        sugar_substrings = [
            "pyranosyloxy", "furanosyloxy", "glycosyloxy",
            "pyranoside", "furanoside",
        ]
        for smiles in self.BENCHMARK_GLYCOSIDES:
            name = name_compound(smiles)
            if name and any(s in name.lower() for s in sugar_substrings):
                sugar_count += 1
        assert sugar_count >= 4, (
            f"Expected >= 4 glycosides with sugar names, got {sugar_count}/6"
        )
