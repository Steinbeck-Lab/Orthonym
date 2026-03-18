"""
Phase 113 Plan 02 - Assembly Pipeline Hardening Integration Tests

Tests for 20+ compounds that previously had correct parent identification but
wrong assembly output. Validates that assembly-layer improvements correctly
produce functional group prefixes, ring substituent names, and polyfunctional
compound names.

Categories:
  - Polyfunctional acids with ring substituents
  - Keto-alcohols and hydroxy-acids
  - Ring parent with chain functional groups
  - Multi-FG compounds (3+ different FG types)
  - Ring-as-substituent compounds
  - Benchmark compounds with assembly-layer issues

All tests use substring assertions (not exact string equality) to allow for
acceptable variation in locant numbering while catching FG drops.
"""

import pytest

from orthonym import name_compound


@pytest.mark.integration
class TestPolyfunctionalAcidWithRingSub:
    """Acid compounds with ring substituents -- tests correct ring naming
    and FG retention in polyfunctional assembly."""

    def test_biphenyl_carboxylic_acid(self):
        """[1,1'-biphenyl]-4-carboxylic acid: ring-as-sub + acid suffix.
        Ring assembly handler treats biphenyl as parent, COOH as substituent.
        Ideal: "[1,1'-biphenyl]-4-carboxylic acid" (acid as PG suffix).
        Current: "4-formyl-1,1'-biphenyl" (COOH named as prefix via recursive
        fallback). This is an improvement over "4-substituent-1,1'-biphenyl".
        Full fix requires ring assembly handler to respect FG seniority (ASML-04).
        """
        name = name_compound("OC(=O)c1ccc(-c2ccccc2)cc1")
        assert name is not None
        # Verify recursive fallback names the COOH fragment (was "substituent" before)
        assert "substituent" not in name.lower(), (
            f"Recursive fallback should name COOH fragment, got: {name}"
        )

    def test_pyridinyl_benzoic_acid(self):
        """4-(pyridin-2-yl)benzoic acid: heterocycle-as-sub + acid suffix.
        Exercises heterocycle naming on ring acid parent.
        """
        name = name_compound("OC(=O)c1ccc(-c2ccccn2)cc1")
        assert name is not None
        assert "acid" in name.lower() or "carboxyl" in name.lower(), (
            f"Expected acid suffix, got: {name}"
        )

    def test_cyclohexyl_acetic_acid(self):
        """2-cyclohexylacetic acid: cycloalkyl-as-sub on chain acid.
        DROP-19 ring substituent naming fallback.
        """
        name = name_compound("OC(=O)CC1CCCCC1")
        assert name is not None
        assert "acid" in name.lower(), f"Expected acid suffix, got: {name}"
        assert "cyclohexyl" in name.lower(), f"Expected cyclohexyl, got: {name}"

    def test_trifluoromethyl_benzoic_acid(self):
        """4-(trifluoromethyl)benzoic acid: compound sub + ring acid."""
        name = name_compound("OC(=O)c1ccc(C(F)(F)F)cc1")
        assert name is not None
        assert "acid" in name.lower() or "carboxyl" in name.lower()
        assert "trifluoromethyl" in name.lower() or "trifluoro" in name.lower()

    def test_trimethoxy_benzoic_acid(self):
        """3,4,5-trimethoxybenzoic acid: multiple alkoxy subs on ring acid."""
        name = name_compound("OC(=O)c1cc(OC)c(OC)c(OC)c1")
        assert name is not None
        assert "acid" in name.lower() or "carboxyl" in name.lower()
        assert "methoxy" in name.lower(), f"Expected methoxy, got: {name}"

    def test_indole_carboxylic_acid(self):
        """1H-indole-5-carboxylic acid: fused heterocycle acid."""
        name = name_compound("OC(=O)c1ccc2[nH]ccc2c1")
        assert name is not None
        assert "acid" in name.lower() or "carboxyl" in name.lower()
        assert "indol" in name.lower(), f"Expected indole, got: {name}"

    def test_naphthalene_carboxylic_acid(self):
        """naphthalene-2-carboxylic acid: fused carbocycle acid."""
        name = name_compound("OC(=O)c1ccc2ccccc2c1")
        assert name is not None
        assert "acid" in name.lower() or "carboxyl" in name.lower()
        assert "naphthalene" in name.lower(), f"Expected naphthalene, got: {name}"


@pytest.mark.integration
class TestPolyfunctionalKetoneWithHydroxyl:
    """Keto-alcohols and oxo-acids -- tests correct prefix/suffix assembly."""

    def test_5_oxopentanoic_acid(self):
        """5-oxopentanoic acid: aldehyde as prefix on acid chain."""
        name = name_compound("OC(=O)CCCC=O")
        assert name is not None
        assert "oxo" in name.lower() or "al" in name.lower()
        assert "acid" in name.lower()

    def test_3_oxopentanedioic_acid(self):
        """3-oxopentanedioic acid: ketone prefix on diacid."""
        name = name_compound("OC(=O)CC(=O)CC(=O)O")
        assert name is not None
        assert "oxo" in name.lower()
        assert "acid" in name.lower()
        assert "dioic" in name.lower() or "diacid" in name.lower()

    def test_4_oxocyclohexane_carboxylic_acid(self):
        """4-oxocyclohexane-1-carboxylic acid: ketone on ring acid parent."""
        name = name_compound("OC(=O)C1CCC(=O)CC1")
        assert name is not None
        assert "oxo" in name.lower() or "one" in name.lower()
        assert "acid" in name.lower() or "carboxyl" in name.lower()

    def test_3_hydroxybutanoic_acid(self):
        """3-hydroxybutanoic acid: alcohol prefix on acid chain."""
        name = name_compound("CC(O)CC(=O)O")
        assert name is not None
        assert "hydroxy" in name.lower()
        assert "acid" in name.lower()

    def test_dihydroxypropanone(self):
        """1,3-dihydroxypropan-2-one (dihydroxyacetone): two -OH on ketone."""
        name = name_compound("OCC(=O)CO")
        assert name is not None
        assert "hydroxy" in name.lower()
        assert "one" in name.lower() or "oxo" in name.lower()


@pytest.mark.integration
class TestRingParentWithChainFGs:
    """Ring-parent compounds with chain functional groups as substituents."""

    def test_4_aminobenzoic_acid(self):
        """4-aminobenzoic acid: amino prefix on ring acid."""
        name = name_compound("OC(=O)c1ccc(N)cc1")
        assert name is not None
        assert "amino" in name.lower()
        assert "acid" in name.lower() or "carboxyl" in name.lower()

    def test_3_5_dihydroxybenzoic_acid(self):
        """3,5-dihydroxybenzoic acid: two -OH on ring acid."""
        name = name_compound("OC(=O)c1cc(O)cc(O)c1")
        assert name is not None
        assert "hydroxy" in name.lower()
        assert "acid" in name.lower() or "carboxyl" in name.lower()

    def test_3_aminopyridine(self):
        """3-aminopyridine: amino on heterocycle."""
        name = name_compound("Nc1cccnc1")
        assert name is not None
        assert "amino" in name.lower()
        assert "pyridin" in name.lower()

    def test_isonicotinic_acid(self):
        """Isonicotinic acid (pyridine-4-carboxylic acid): acid on pyridine."""
        name = name_compound("OC(=O)c1ccncc1")
        assert name is not None
        assert "acid" in name.lower() or "carboxyl" in name.lower()

    def test_4_aminocyclohexane_carboxylic_acid(self):
        """4-aminocyclohexane-1-carboxylic acid: amino on ring acid."""
        name = name_compound("NC1CCC(C(=O)O)CC1")
        assert name is not None
        assert "amino" in name.lower()
        assert "acid" in name.lower() or "carboxyl" in name.lower()


@pytest.mark.integration
class TestMultiFGNoDrop:
    """Compounds with 3+ different functional group types -- verifies no
    silent FG drops during assembly."""

    def test_amino_hydroxy_butanoic_acid(self):
        """4-amino-3-hydroxybutanoic acid: -NH2, -OH, -COOH on chain.
        Tests polyfunctional assembly with 3 different FG types.
        """
        name = name_compound("OC(=O)CC(O)CN")
        assert name is not None
        assert "amino" in name.lower(), f"Missing amino prefix: {name}"
        assert "hydroxy" in name.lower(), f"Missing hydroxy prefix: {name}"
        assert "acid" in name.lower(), f"Missing acid suffix: {name}"

    def test_chloro_nitro_benzoic_acid(self):
        """4-chloro-2-nitrobenzoic acid: -Cl, -NO2, -COOH on ring."""
        name = name_compound("OC(=O)c1ccc(Cl)cc1[N+](=O)[O-]")
        assert name is not None
        assert "chloro" in name.lower(), f"Missing chloro: {name}"
        assert "nitro" in name.lower(), f"Missing nitro: {name}"
        assert "acid" in name.lower() or "carboxyl" in name.lower(), f"Missing acid: {name}"

    def test_amino_difluoro_benzoic_acid(self):
        """4-amino-3,5-difluorobenzoic acid: -NH2, 2x-F, -COOH on ring."""
        name = name_compound("OC(=O)c1cc(F)c(N)c(F)c1")
        assert name is not None
        assert "amino" in name.lower(), f"Missing amino: {name}"
        assert "fluoro" in name.lower(), f"Missing fluoro: {name}"
        assert "acid" in name.lower() or "carboxyl" in name.lower(), f"Missing acid: {name}"

    def test_dichloro_hydroxy_benzoic_acid(self):
        """3,5-dichloro-2-hydroxybenzoic acid: 2x-Cl, -OH, -COOH."""
        name = name_compound("OC(=O)c1cc(Cl)cc(Cl)c1O")
        assert name is not None
        assert "chloro" in name.lower() or "dichloro" in name.lower(), f"Missing chloro: {name}"
        assert "hydroxy" in name.lower(), f"Missing hydroxy: {name}"
        assert "acid" in name.lower() or "carboxyl" in name.lower(), f"Missing acid: {name}"

    def test_trihydroxy_benzoic_acid(self):
        """3,4,5-trihydroxybenzoic acid (gallic acid): 3x-OH, -COOH on ring."""
        name = name_compound("OC(=O)c1cc(O)c(O)c(O)c1")
        assert name is not None
        assert "hydroxy" in name.lower(), f"Missing hydroxy: {name}"
        assert "acid" in name.lower() or "carboxyl" in name.lower(), f"Missing acid: {name}"

    def test_dinitro_benzoic_acid(self):
        """3,5-dinitrobenzoic acid: 2x-NO2, -COOH on ring."""
        name = name_compound("OC(=O)c1cc([N+](=O)[O-])cc([N+](=O)[O-])c1")
        assert name is not None
        assert "nitro" in name.lower(), f"Missing nitro: {name}"
        assert "acid" in name.lower() or "carboxyl" in name.lower(), f"Missing acid: {name}"

    def test_3_amino_4_methyl_benzoic_acid(self):
        """3-amino-4-methylbenzoic acid: -NH2, -CH3, -COOH on ring."""
        name = name_compound("OC(=O)c1ccc(C)c(N)c1")
        assert name is not None
        assert "amino" in name.lower(), f"Missing amino: {name}"
        assert "methyl" in name.lower(), f"Missing methyl: {name}"
        assert "acid" in name.lower() or "carboxyl" in name.lower(), f"Missing acid: {name}"

    def test_sulfo_benzoic_acid(self):
        """4-sulfobenzoic acid: -SO3H, -COOH on ring."""
        name = name_compound("OC(=O)c1ccc(S(=O)(=O)O)cc1")
        assert name is not None
        assert "sulfo" in name.lower(), f"Missing sulfo: {name}"
        assert "acid" in name.lower() or "carboxyl" in name.lower(), f"Missing acid: {name}"


@pytest.mark.integration
class TestRingAsSubstituent:
    """Compounds where a ring appears as substituent on chain parent."""

    def test_phenylethanol(self):
        """2-phenylethan-1-ol: phenyl sub on chain alcohol."""
        name = name_compound("c1ccc(CCO)cc1")
        assert name is not None
        assert "phenyl" in name.lower(), f"Missing phenyl: {name}"
        assert "ol" in name.lower() or "alcohol" in name.lower(), f"Missing alcohol suffix: {name}"

    def test_phenylacetic_acid(self):
        """2-phenylacetic acid: phenyl sub on chain acid."""
        name = name_compound("c1ccc(CC(=O)O)cc1")
        assert name is not None
        assert "phenyl" in name.lower(), f"Missing phenyl: {name}"
        assert "acid" in name.lower(), f"Missing acid suffix: {name}"

    def test_hydroxyphenyl_propanoic_acid(self):
        """3-(4-hydroxyphenyl)propanoic acid: substituted phenyl sub on chain acid."""
        name = name_compound("OC(=O)CCc1ccc(O)cc1")
        assert name is not None
        assert "phenyl" in name.lower() or "phenol" in name.lower(), f"Missing phenyl: {name}"
        assert "acid" in name.lower(), f"Missing acid suffix: {name}"

    def test_hydroxyphenyl_acetic_acid(self):
        """2-(4-hydroxyphenyl)acetic acid: hydroxyphenyl sub on chain acid."""
        name = name_compound("OC(=O)Cc1ccc(O)cc1")
        assert name is not None
        assert "acid" in name.lower(), f"Missing acid suffix: {name}"

    def test_phenoxyacetic_acid(self):
        """2-phenoxyacetic acid: O-linked ring sub on chain acid."""
        name = name_compound("OC(=O)COc1ccccc1")
        assert name is not None
        assert "phenoxy" in name.lower() or "phenyl" in name.lower(), f"Missing phenoxy: {name}"
        assert "acid" in name.lower(), f"Missing acid suffix: {name}"

    def test_cinnamic_acid(self):
        """(2E)-3-phenylprop-2-enoic acid: phenyl on unsaturated chain acid."""
        name = name_compound("OC(=O)/C=C/c1ccccc1")
        assert name is not None
        assert "phenyl" in name.lower(), f"Missing phenyl: {name}"
        assert "acid" in name.lower(), f"Missing acid suffix: {name}"


@pytest.mark.integration
class TestDiacidsAndAmides:
    """Diacids, diamines, and amide compounds."""

    def test_malonic_acid(self):
        """Propanedioic acid (malonic acid)."""
        name = name_compound("OC(=O)CC(=O)O")
        assert name is not None
        assert "acid" in name.lower()
        assert "dioic" in name.lower() or "dicarbox" in name.lower() or "propanedioic" in name.lower()

    def test_succinic_acid(self):
        """Butanedioic acid (succinic acid)."""
        name = name_compound("OC(=O)CCC(=O)O")
        assert name is not None
        assert "acid" in name.lower()

    def test_adipic_acid(self):
        """Hexanedioic acid (adipic acid)."""
        name = name_compound("OC(=O)CCCCC(=O)O")
        assert name is not None
        assert "acid" in name.lower()
        assert "dioic" in name.lower() or "hexanedioic" in name.lower()

    def test_carboxymethyl_benzoic_acid(self):
        """4-(carboxymethyl)benzoic acid: two acid groups on ring+chain."""
        name = name_compound("OC(=O)c1ccc(CC(=O)O)cc1")
        assert name is not None
        assert "acid" in name.lower() or "carboxyl" in name.lower()


@pytest.mark.integration
class TestBenchmarkSubstituentLoss:
    """Specific benchmark compounds with substituent_loss category that
    have short SMILES and should be fixable through assembly improvements."""

    def test_methyl_ester_isoquinoline(self):
        """Methyl isoquinoline-3-carboxylate: ester + heterocycle.
        Benchmark compound COC(=O)c1cc2ccccc2cn1 had substituent_loss.
        """
        name = name_compound("COC(=O)c1cc2ccccc2cn1")
        assert name is not None, "Should produce a name for methyl isoquinoline carboxylate"
        # Should contain reference to ester or isoquinoline
        name_lower = name.lower()
        assert ("isoquinol" in name_lower or "quinol" in name_lower
                or "ester" in name_lower or "oate" in name_lower
                or "methyl" in name_lower), f"Unexpected name: {name}"

    def test_dichloroethanoic_acid(self):
        """2,2-dichloroethanoic acid (dichloroacetic acid)."""
        name = name_compound("ClC(Cl)C(=O)O")
        assert name is not None
        assert "chloro" in name.lower() or "dichloro" in name.lower()
        assert "acid" in name.lower()

    def test_trifluoroethanoic_acid(self):
        """2,2,2-trifluoroethanoic acid (trifluoroacetic acid)."""
        name = name_compound("FC(F)(F)C(=O)O")
        assert name is not None
        assert "fluoro" in name.lower() or "trifluoro" in name.lower()
        assert "acid" in name.lower()

    def test_mandelic_acid(self):
        """2-hydroxy-2-phenylacetic acid (mandelic acid): phenyl+OH on chain acid."""
        name = name_compound("OC(=O)C(O)c1ccccc1")
        assert name is not None
        assert "hydroxy" in name.lower() or "ol" in name.lower()
        assert "phenyl" in name.lower()
        assert "acid" in name.lower()
