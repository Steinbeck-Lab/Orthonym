"""End-to-end tests for phosphorus compound naming."""

import pytest
from orthonym import name_compound


class TestPhosphineE2E:
    """E2E tests for phosphine (phosphane) naming (PHOSPH-01, PHOSPH-02)."""

    def test_methylphosphane(self):
        """CP -> methylphosphane (PHOSPH-01)"""
        assert name_compound("CP") == "methylphosphane"

    def test_ethylphosphane(self):
        """CCP -> ethylphosphane"""
        assert name_compound("CCP") == "ethylphosphane"

    def test_triethylphosphane(self):
        """CCP(CC)CC -> triethylphosphane"""
        assert name_compound("CCP(CC)CC") == "triethylphosphane"

    def test_dimethylphosphane(self):
        """CPC -> dimethylphosphane"""
        assert name_compound("CPC") == "dimethylphosphane"


class TestPhosphineOxideE2E:
    """E2E tests for phosphine oxide naming (PHOSPH-03)."""

    def test_trimethylphosphane_oxide(self):
        """CP(C)(C)=O -> trimethylphosphane oxide (PHOSPH-03)"""
        assert name_compound("CP(C)(C)=O") == "trimethyl-λ5-phosphanone"

    def test_triethylphosphane_oxide(self):
        """CCP(CC)(CC)=O -> triethylphosphane oxide"""
        assert name_compound("CCP(CC)(CC)=O") == "triethyl-λ5-phosphanone"


class TestPhosphonicAcidE2E:
    """E2E tests for phosphonic acid naming (PHOSPH-04)."""

    def test_ethanephosphonic_acid(self):
        """CCP(=O)(O)O -> ethylphosphonic acid ('C2H5-P(O)(OH)2 ethylphosphonic acid
        (PIN) (not ethanephosphonic acid)', the Blue Book; 9ab971737)."""
        assert name_compound("CCP(=O)(O)O") == "ethylphosphonic acid"

    def test_phenylphosphonic_acid(self):
        """c1ccccc1P(=O)(O)O -> phenylphosphonic acid"""
        assert name_compound("c1ccccc1P(=O)(O)O") == "phenylphosphonic acid"


class TestPhosphinicAcidE2E:
    """E2E tests for phosphinic acid naming (PHOSPH-05)."""

    def test_dimethylphosphinic_acid(self):
        """CP(C)(=O)O -> dimethylphosphinic acid (PHOSPH-05)"""
        assert name_compound("CP(C)(=O)O") == "dimethylphosphinic acid"

    def test_diethylphosphinic_acid(self):
        """CCP(CC)(=O)O -> diethylphosphinic acid"""
        assert name_compound("CCP(CC)(=O)O") == "diethylphosphinic acid"

    def test_ethylmethylphosphinic_acid(self):
        """CCP(C)(=O)O -> ethylmethylphosphinic acid"""
        assert name_compound("CCP(C)(=O)O") == "ethyl(methyl)phosphinic acid"


class TestPhosphateEsterE2E:
    """E2E tests for phosphate ester naming (PHOSPH-06)."""

    def test_dimethyl_phosphate(self):
        """COP(=O)(OC)O -> dimethyl hydrogen phosphate (1 free -OH cited, fix)."""
        assert name_compound("COP(=O)(OC)O") == "dimethyl hydrogen phosphate"

    def test_trimethyl_phosphate(self):
        """COP(=O)(OC)OC -> trimethyl phosphate"""
        assert name_compound("COP(=O)(OC)OC") == "trimethyl phosphate"

    def test_ethyl_phosphate(self):
        """CCOP(=O)(O)O -> ethyl dihydrogen phosphate (fix, see test_methyl_phosphate)."""
        assert name_compound("CCOP(=O)(O)O") == "ethyl dihydrogen phosphate"

    def test_diethyl_phosphate(self):
        """CCOP(=O)(OCC)O -> diethyl hydrogen phosphate (1 free -OH cited, fix)."""
        assert name_compound("CCOP(=O)(OCC)O") == "diethyl hydrogen phosphate"

    def test_triethyl_phosphate(self):
        """CCOP(=O)(OCC)OCC -> triethyl phosphate"""
        assert name_compound("CCOP(=O)(OCC)OCC") == "triethyl phosphate"


class TestPhosphorusRetainedNames:
    """E2E tests for phosphorus retained names."""

    def test_triphenylphosphane(self):
        """Triphenylphosphane retained name."""
        result = name_compound("c1ccccc1P(c2ccccc2)c3ccccc3")
        assert result == "triphenylphosphane"

    def test_triphenylphosphane_oxide(self):
        """Triphenylphosphane oxide retained name."""
        result = name_compound("O=P(c1ccccc1)(c2ccccc2)c3ccccc3")
        assert result == "triphenyl-λ5-phosphanone"


class TestPhosphorusEdgeCases:
    """Edge case tests for phosphorus naming."""

    def test_asymmetric_phosphate(self):
        """Asymmetric phosphate diester -> cites the free -OH as 'hydrogen'.

        The bare 'ethyl methyl phosphate' is the ANION (OPSIN full-InChIKey RT
        FAILS vs the neutral input P(=O)(OCC)(OC)[O-]); the neutral diester acid
        is 'ethyl methyl hydrogen phosphate' /, RT-OK).
        """
        result = name_compound("COP(=O)(OCC)O")
        assert result == "ethyl methyl hydrogen phosphate"

    def test_dipropylphosphinic_acid(self):
        """Dipropylphosphinic acid."""
        result = name_compound("CCCP(CCC)(=O)O")
        assert result == "dipropylphosphinic acid"


class TestPhosphanylPrefix:
    """E2E tests for phosphanyl prefix naming (,)."""

    def test_diphenylphosphanyl_on_benzene(self):
        """Triphenylphosphane is a retained name ."""
        result = name_compound("c1ccc(P(c2ccccc2)c3ccccc3)cc1")
        assert result == "triphenylphosphane"

    def test_diphenylphosphanyl_on_substituted_ring(self):
        """PPh2 on methylbenzene : the phosphane is the parent (P is senior to C,
        , the Blue Book), so the name is '(4-methylphenyl)di(phenyl)phosphane';
        'phenyl' is enclosed after the multiplier as in '(arsanylmethyl)di(phenyl)phosphane
        (PIN)', the Blue Book."""
        result = name_compound("Cc1ccc(P(c2ccccc2)c3ccccc3)cc1")
        assert result == "(4-methylphenyl)di(phenyl)phosphane"

    def test_phenyl_not_counted_as_alkyl(self):
        """Phenyl groups on P should not be counted as hexyl (regression)."""
        result = name_compound("c1ccc(P(c2ccccc2)c3ccccc3)cc1")
        assert "hexyl" not in result


class TestBisPhosphanyl:
    """E2E tests for bis/tris multiplied phosphanyl groups ."""

    def test_bis_diphenylphosphanyl_benzene(self):
        """Two identical PPh2 groups on benzene use the bis multiplier .

        The two phosphanes are the parent: P is senior to C,
        the Blue Book), and identical
        parents linked by a substituent group take a multiplicative name, as printed
        for '(ethane-1,2-diyl)bis(dimethylphosphane) (PIN)' (the Blue Book),
        '(dibenzo[b,d]furan-3,7-diyl)bis(phosphane) (PIN)' (:39196) and
        '(1,2-phenylene)bis(arsane) (PIN)' (:39198). Was the substitutive
        '1,2-bis(diphenylphosphanyl)benzene' (PIN class program batch 2, Task 10)."""
        result = name_compound(
            "c1ccc(P(c2ccccc2)c3ccccc3)c(P(c4ccccc4)c5ccccc5)c1"
        )
        assert result == "(1,2-phenylene)bis(diphenylphosphane)"


class TestPhosphorusOnComplexSubstrate:
    """E2E tests for phosphorus on complex substrates (,)."""

    def test_diphenylphosphinic_acid_e2e(self):
        """Diphenylphosphinic acid via e2e ."""
        assert name_compound("O=P(O)(c1ccccc1)c2ccccc2") == "diphenylphosphinic acid"


class TestPhosphiteAndPhosphonateEsterE2E:
    """ tail #16/#17: trivalent-P phosphite triesters and phosphonic-acid
    diesters were entirely unperceived (no matching FG) -> a garbage skeletal-
    replacement name or abstention. New perception routes them to the existing
    functional-class namer's phosphite / phosphonate stems."""

    def test_trimethyl_phosphite(self):
        assert name_compound("COP(OC)OC") == "trimethyl phosphite"

    def test_triethyl_phosphite(self):
        assert name_compound("CCOP(OCC)OCC") == "triethyl phosphite"

    def test_tris_dodecylsulfanyl_phosphite(self):
        # #16: sulfenyl-ester oxygens (-O-S-R). The compound owner takes tris(...).
        smi = "CCCCCCCCCCCCSOP(OSCCCCCCCCCCCC)OSCCCCCCCCCCCC"
        assert name_compound(smi) == "tris(dodecylsulfanyl) phosphite"

    def test_dimethyl_methylphosphonate(self):
        assert name_compound("COP(=O)(C)OC") == "dimethyl methylphosphonate"

    def test_diisopropyl_cyano_isocyano_ethylphosphonate(self):
        # #17: phosphonate diester senior to the nitrile + isocyanide, which
        # become cyano/isocyano prefixes on the P-C ligand.
        smi = "CC(C)OP(=O)(C(C)(C#N)[N+]#[C-])OC(C)C"
        r = name_compound(smi)
        assert r is not None and "phosphonate" in r
        assert "cyano" in r and "isocyano" in r
        # "Parentheses are used around compound... prefixes"
        # (the Blue Book, under ENCLOSING MARKS): the compound organyl
        # ligand is enclosed, as in '(naphthalen-2-yl)phosphonate (PIN)' (:44383), and
        # the two propan-2-yl owners are 'di(propan-2-yl)' (:36921). OPSIN full
        # InChIKey exact.
        assert r == "di(propan-2-yl) (1-cyano-1-isocyanoethyl)phosphonate"
