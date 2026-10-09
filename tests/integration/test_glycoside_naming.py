"""Integration tests for end-to-end glycoside naming.

Tests the full pipeline from SMILES to glycoside name using name_compound
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
        """Phenyl β-D-glucopyranoside: functional-class form (-08 flip).

        a phase re-baseline: the simple phenol aglycone flips from the
        legacy substitutive ``(β-D-glucopyranosyloxy)...`` form to the
        Blue-Book functional-class two-word form ``phenyl β-D-glucopyranoside``
        . OPSIN-RT-verified.
        """
        # Phenol + β-D-glucose via O-glycosidic bond at anomeric position
        name = name_compound("OC[C@H]1O[C@@H](Oc2ccccc2)[C@H](O)[C@@H](O)[C@@H]1O")
        assert name is not None
        assert name != "unknown"
        assert name == "phenyl β-D-glucopyranoside"

    def test_methyl_alpha_d_glucoside(self):
        """Methyl α-D-glucopyranoside: functional-class form (WSD-08 flip)."""
        name = name_compound("OC[C@H]1O[C@H](OC)[C@H](O)[C@@H](O)[C@@H]1O")
        assert name is not None
        assert name != "unknown"
        assert name == "methyl α-D-glucopyranoside"

    def test_galactoside(self):
        """4-Hydroxyphenyl β-D-galactopyranoside: functional-class form (diol aglycone).

        The aglycone is hydroquinone, a diol: its name cannot say which hydroxy group the
        sugar sits on, and the string rule (``_alcohol_to_alkyl``) read the multiplied '-ol' as
        the divalent 'benzene-1,4-diyl', a group with two free valences that counts the
        glycosidic oxygen twice ('benzene-1,4-diyl beta-D-galactopyranoside', which OPSIN
        cannot parse). The group word is now built from the STRUCTURE of the aglycone
        (``_aglycone_polyol_group``: the oxygen that condenses with the sugar back to the parent
        is the glycosidic one, the other hydroxy group stays a 'hydroxy' prefix) and is shipped
        only after it round-trips to the parent. "Names" (the Blue Book):
        "Glycosides are named by using functional class nomenclature.... The class name is
        preceded, as a separate word, by the name of the substituent group that is part of the
        acetal or ketal function." Only a group senior to hydroxy turns a glycoside
        substitutive,:53915: "not 4-acetylphenyl β-D-glucopyranoside; a ketone is
        senior to a hydroxy compound"); a second hydroxy group is not senior to the glycosidic
        one. OPSIN 2.9.0 full-InChIKey exact. The old text asserted the legacy
        '(glycosyloxy)hydroquinone' form, which was removed (see ``_assemble_glycoside``).
        """
        name = name_compound(
            "OC[C@H]1O[C@@H](Oc2ccc(O)cc2)[C@H](O)[C@@H](O)[C@H]1O"
        )
        assert name == "4-hydroxyphenyl β-D-galactopyranoside", name

    def test_rhamnoside(self):
        """Phenyl α-L-rhamnopyranoside: functional-class form (-08 flip).

        a phase: deoxy-L rhamnose is catalog-only (the structure deriver
        returns None on it), but the ``lookup_sugar`` fast-path resolves the
        full ``(alpha, L, rhamnopyranose)`` tuple, so the inversion still fires
        (descriptor source = catalog). OPSIN-RT-verified.
        """
        name = name_compound("C[C@@H]1O[C@@H](Oc2ccccc2)[C@H](O)[C@H](O)[C@H]1O")
        assert name is not None
        assert name != "unknown"
        assert name == "phenyl α-L-rhamnopyranoside"

    def test_ethyl_glucoside(self):
        """Ethyl β-D-glucopyranoside: functional-class form (WSD-08 flip)."""
        name = name_compound("OC[C@H]1O[C@@H](OCC)[C@H](O)[C@@H](O)[C@@H]1O")
        assert name is not None
        assert name != "unknown"
        assert name == "ethyl β-D-glucopyranoside"

    def test_naphthyl_glucoside(self):
        """Naphthalen-2-yl β-D-glucopyranoside: functional-class form (WSD-08 flip)."""
        name = name_compound(
            "OC[C@H]1O[C@@H](Oc2ccc3ccccc3c2)[C@H](O)[C@@H](O)[C@@H]1O"
        )
        assert name is not None
        assert name != "unknown"
        # (the Blue Book; the naphthyl entry is:16382) retains the traditional ring prefixes "only for
        # general nomenclature": '2-naphthyl (also 1-isomer) naphthalen-2-yl (also
        # 1-isomer; preferred prefixes)' -- the preferred prefix is naphthalen-2-yl.
        assert name == "naphthalen-2-yl β-D-glucopyranoside"


# ============================================================================
# Glycoside name format validation
# ============================================================================


@pytest.mark.integration
class TestGlycosideNameFormat:
    """Verify the assembled glycoside name format is correct."""

    def test_glycoside_functional_class_form(self):
        """Phenyl glucoside emits the functional-class two-word form (-08).

        a phase re-baseline: was a parenthesized substitutive prefix
        ``(β-D-glucopyranosyloxy)...``; now the Blue-Book functional-class
        form ``phenyl β-D-glucopyranoside`` (aglycone substituent as a
        separate preceding word, sugar -ose -> -oside). OPSIN-RT-verified.
        """
        name = name_compound("OC[C@H]1O[C@@H](Oc2ccccc2)[C@H](O)[C@@H](O)[C@@H]1O")
        assert name == "phenyl β-D-glucopyranoside"
        # The alpha/beta + D/L descriptors are never dropped
        assert "β-D-glucopyranoside" in name
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
        assert "β-D-" in name

    def test_rhamnoside_has_l_config(self):
        """Rhamnoside name should include L-configuration descriptor."""
        name = name_compound("C[C@@H]1O[C@@H](Oc2ccccc2)[C@H](O)[C@H](O)[C@H]1O")
        assert "α-L-" in name


# ============================================================================
# Senior-aglycone PROTECT: must NOT flip to functional-class form (/)
# ============================================================================


@pytest.mark.integration
class TestGlycosideSeniorAglyconeProtect:
    """A glycoside whose aglycone bears a group senior to hydroxy must stay
    on the legacy substitutive ``(glycosyloxy)R`` form: the
    ``-ose`` ending is retained and the aglycone is cited as an O-substituent
    prefix. The structural seniority guard (rank >= primary_alcohol == 54)
    fails closed for ketone/acid/aldehyde aglycones ."""

    def test_aldehyde_aglycone_stays_legacy(self):
        """4-Formylphenyl β-D-glucoside (aldehyde aglycone) keeps legacy form.

        The aglycone (4-hydroxybenzaldehyde) has a principal aldehyde group,
        senior to hydroxy, so the guard fails closed and the name retains
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
        """Unit-level PROTECT: _aglycone_to_substituent returns None for
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
        # Diol aglycone (no clean -yl prefix) -> None (fail-closed)
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
        # 1. Phenyl β-D-glucoside (stereo, hexopyranose)
        "OC[C@H]1O[C@@H](Oc2ccccc2)[C@H](O)[C@@H](O)[C@@H]1O",
        # 2. Methyl α-D-glucoside (stereo, hexopyranose)
        "OC[C@H]1O[C@H](OC)[C@H](O)[C@@H](O)[C@@H]1O",
        # 3. 4-hydroxyphenyl β-D-galactoside (stereo, galactose)
        "OC[C@H]1O[C@@H](Oc2ccc(O)cc2)[C@H](O)[C@@H](O)[C@H]1O",
        # 4. Phenyl α-L-rhamnoside (stereo, deoxy-L-sugar)
        "C[C@@H]1O[C@@H](Oc2ccccc2)[C@H](O)[C@H](O)[C@H]1O",
        # 5. Ethyl β-D-glucoside (stereo, hexopyranose + small aglycone)
        "OC[C@H]1O[C@@H](OCC)[C@H](O)[C@@H](O)[C@@H]1O",
        # 6. Non-stereo diglucosylchromane from CI benchmark
        "OCC1OC(Oc2cc(O)c3c(c2)OC(c2ccc(O)c(OC4OC(CO)C(O)C(O)C4O)c2)"
        "C(O)C3)C(O)C(O)C1O",
    ]

    @pytest.mark.parametrize("smiles", BENCHMARK_GLYCOSIDES, ids=[
        "phenyl-β-D-glucoside",
        "methyl-α-D-glucoside",
        "hydroxyphenyl-β-D-galactoside",
        "phenyl-α-L-rhamnoside",
        "ethyl-β-D-glucoside",
        "diglucosyl-chromane",
    ])
    def test_benchmark_glycoside_produces_name(self, smiles):
        """Benchmark glycoside produces a non-trivial name (not None/unknown)."""
        name = name_compound(smiles)
        assert name is not None, f"Name should not be None for {smiles}"
        assert name != "unknown", f"Name should not be 'unknown' for {smiles}"
        assert len(name) > 15, f"Name too short ({len(name)}): {name}"

    @pytest.mark.parametrize("smiles", BENCHMARK_GLYCOSIDES, ids=[
        "phenyl-β-D-glucoside",
        "methyl-α-D-glucoside",
        "hydroxyphenyl-β-D-galactoside",
        "phenyl-α-L-rhamnoside",
        "ethyl-β-D-glucoside",
        "diglucosyl-chromane",
    ])
    def test_benchmark_glycoside_has_sugar_name(self, smiles, request):
        """Benchmark glycoside name contains a sugar-related substring.

        a phase: the accepted set now includes the functional-class
        ``-pyranoside``/``-furanoside`` forms (the flipped simple-aglycone
        glycosides) alongside the legacy ``...pyranosyloxy`` substitutive
        forms (gated-out / multi-sugar rows).
        """
        if smiles == self.BENCHMARK_GLYCOSIDES[5]:
            # Suite fix j6-breadth (TRIAGE g4 C5): the diglucosyl chromane has
            # stereo-UNDEFINED hexopyranosyl groups, so no D-gluco
            # glycosyl name applies; the PIN tier abstains (the systematic
            # oxan-2-yloxy PIN with a hydro-chromene parent is not built).
            # Best-effort names it RT-exact (test_j6_breadth tier contract).
            request.applymarker(pytest.mark.xfail(strict=True, reason=(
                "PIN tier abstains: needs the stereo-free oxanyloxy substituent "
                "on a 3,4-dihydro-2H-1-benzopyran parent at the PIN tier -- "
                "TODO in TRIAGE.md 'Suite fix -- j6-breadth'")))
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


# ============================================================================
# a phase (-04): systematic monosaccharide + free uronic  +
# disaccharide, end-to-end via the carbohydrate dispatch tier
# ============================================================================


@pytest.mark.integration
class TestSystematicCarbohydrateDispatch:
    """The broadened CARBOHYDRATE_LOOKUP tier (priority 1000) now routes a
    non-cataloged decorated single sugar ring to the systematic-mono engine, a
    cataloged uronic free acid to the free-acid emitter, and >=2 linked
    sugar rings to the disaccharide assembler -- all through the public
    ``name_compound``. Cataloged sugars / simple glycosides / cataloged amino
    sugars stay byte-identical (PROTECT)."""

    def test_uronic_free_acid_d10(self):
        """A free glucuronic acid emits the OPSIN-parseable free-acid form
        ``β-D-glucopyranuronic acid`` (D-10 / WSC-04 SC#1), NOT the
        unparseable ``β-D-glucuronopyranose`` (-> 'unknown')."""
        name = name_compound(
            "O=C(O)[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O"
        )
        assert name == "β-D-glucopyranuronic acid"
        assert "glucuronopyranose" not in (name or "")
        assert name != "unknown organic compound"

    def test_six_deoxy_systematic_not_oxane(self):
        """6-deoxy-glucose names systematically, NOT as a substituted oxane."""
        name = name_compound(
            "C[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O"
        )
        assert name == "6-deoxy-β-D-glucopyranose"
        assert "oxan" not in name.lower()

    def test_amino_systematic_not_oxane(self):
        """A non-cataloged amino sugar names systematically."""
        name = name_compound(
            "N[C@@H]1[C@@H](O)[C@H](O)O[C@H](CO)[C@H]1O"
        )
        assert name == "3-amino-3-deoxy-β-D-glucopyranose"
        assert "oxan" not in name.lower()

    def test_maltose_glycosylglycose(self):
        """alpha-Maltose -> glycosylglycose, ASCII (1->4) arrow;
        the DEFINED reducing-end anomer is cited (W6B-T10,."""
        name = name_compound(
            "OC[C@H]1O[C@H](O[C@H]2[C@H](O)[C@@H](O)[C@@H](O)O[C@@H]2CO)"
            "[C@H](O)[C@@H](O)[C@@H]1O"
        )
        assert name == "α-D-glucopyranosyl-(1->4)-α-D-glucopyranose"

    def test_sucrose_glycosyl_glycoside(self):
        """Sucrose (no free hemiacetal) -> glycosyl glycoside."""
        name = name_compound(
            "O([C@@H]1[C@H](O)[C@@H](O)[C@H](O)[C@H](O1)CO)"
            "[C@@]1(CO)[C@@H](O)[C@H](O)[C@H](O1)CO"
        )
        assert name == "β-D-fructofuranosyl α-D-glucopyranoside"

    # --- PROTECT: byte-identical to the pre-183 behaviour ------------------

    def test_protect_cataloged_glucose_byte_identical(self):
        """A cataloged clean sugar stays on the lookup fast-path ."""
        name = name_compound("OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O")
        assert name == "β-D-glucopyranose"

    def test_protect_simple_glycoside_byte_identical(self):
        """A Phase-176 simple glycoside (methyl β-D-glucopyranoside) is
        NOT pulled into the disaccharide engine (Pitfall 4) and stays
        byte-identical to its functional-class form."""
        name = name_compound("CO[C@@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O")
        assert name == "methyl β-D-glucopyranoside"

    def test_protect_cataloged_amino_sugar_byte_identical(self):
        """A cataloged amino sugar (β-D-glucosamine) stays on the catalog
        fast-path -- the uronic interception fires only on a uronic base."""
        name = name_compound("N[C@@H]1[C@@H](O)[C@H](O)[C@@H](CO)O[C@H]1O")
        assert name == "β-D-glucosamine"

    def test_protect_glucuronide_glycoside_byte_identical(self):
        """The Phase-182 steroid glucuronide glycoside still emits the
        ``-osiduronic acid`` head (coupling held: the free-acid emitter
        does not regress ``uronic_glycoside_head``)."""
        name = name_compound(
            "C[C@]12CC[C@H](O[C@@H]3O[C@H](C(=O)O)[C@@H](O)[C@H](O)[C@H]3O)"
            "C[C@H]1CC[C@@H]1[C@@H]2CC[C@]2(C)C(=O)CC[C@@H]12"
        )
        assert name == (
            "17-oxo-5β-androstan-3β-yl β-D-glucopyranosiduronic acid"
        )
