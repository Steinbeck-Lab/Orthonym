"""End-to-end tests for sulfur compound naming."""

import pytest
from orthonym import name_compound


class TestThiolE2E:
    """E2E tests for thiol naming (SULFUR-01)."""

    def test_methanethiol(self):
        """CS -> methanethiol"""
        assert name_compound("CS") == "methanethiol"

    def test_propane_1_thiol(self):
        """CCCS -> propane-1-thiol or propan-1-thiol"""
        result = name_compound("CCCS")
        # Accept both PIN forms: propan-1-thiol (no 'e' before hyphen) is preferred
        assert result in ("propane-1-thiol", "propan-1-thiol", "propanethiol")

    def test_propane_2_thiol(self):
        """CC(S)C -> propane-2-thiol or propan-2-thiol"""
        result = name_compound("CC(S)C")
        # Accept both PIN forms
        assert result in ("propane-2-thiol", "propan-2-thiol")


class TestSulfideE2E:
    """E2E tests for sulfide naming (SULFUR-02).

     (the Blue Book, section heading "Names of chalcogen analogues of ethers,
    i.e., sulfides, selenides and tellurides"): "Method (1), substitutive
    nomenclature, gives preferred IUPAC names". The functional-class "R R' sulfide"
    (method 2) is NOT the PIN; the PIN is the substitutive (R'-sulfanyl)RH. These
    assertions were migrated from the method-2 forms in the sulfanyl-vs-sulfide
    slice; each new value OPSIN-round-trips to the input.
    """

    def test_ethyl_methyl_sulfide(self):
        """CCSC -> (methylsulfanyl)ethane (PIN).

         method (1); senior parent is the longer chain ethane,
        with the shorter arm as the methylsulfanyl prefix."""
        assert name_compound("CCSC") == "(methylsulfanyl)ethane"

    def test_methyl_propyl_sulfide(self):
        """CCCSC -> 1-(methylsulfanyl)propane (PIN).

         method (1); senior parent propane, methylsulfanyl at
        C1 (locant cited for a C3+ chain,."""
        assert name_compound("CCCSC") == "1-(methylsulfanyl)propane"


class TestSulfoxideE2E:
    """E2E tests for sulfoxide naming (SULFUR-03).

     'SULFOXIDES AND SULFONES' (the Blue Book): R-SO-R' and R-SO2-R' are named
    "(1) substitutively, by prefixing the name of the acyl group R'-SO- or R'-SO2- to the
    name of the parent hydride corresponding to R; (2) by functional class nomenclature,
    using the class names 'sulfoxide' and 'sulfone'; (3) by multiplicative nomenclature,
    except where R and R' are alkyl groups" and (:28088) "Methods (1) and (3) generate
    preferred names." The functional-class names 'dimethyl sulfoxide' etc. are method (2),
    so they are not PINs (example:28094 '1-(ethanesulfinyl)butane (PIN)... butyl ethyl
    sulfoxide'; '(methanesulfinyl)methane (PIN)',:46154). These assertions were migrated
    from the method-2 forms, like the sulfide class below; each new value OPSIN-round-trips
    to the input.
    """

    def test_diethyl_sulfoxide(self):
        """CCS(=O)CC -> (ethanesulfinyl)ethane (PIN)"""
        assert name_compound("CCS(=O)CC") == "(ethanesulfinyl)ethane"

    def test_ethyl_methyl_sulfoxide(self):
        """CCS(=O)C -> (methanesulfinyl)ethane (PIN)"""
        assert name_compound("CCS(=O)C") == "(methanesulfinyl)ethane"


class TestSulfoneE2E:
    """E2E tests for sulfone naming (SULFUR-04); method (1), see TestSulfoxideE2E."""

    def test_ethyl_methyl_sulfone(self):
        """CCS(=O)(=O)C -> (methanesulfonyl)ethane (PIN)"""
        assert name_compound("CCS(=O)(=O)C") == "(methanesulfonyl)ethane"


class TestSulfonicAcidE2E:
    """E2E tests for sulfonic acid naming (SULFUR-05)."""

    def test_ethanesulfonic_acid(self):
        """CCS(=O)(=O)O -> ethanesulfonic acid"""
        assert name_compound("CCS(=O)(=O)O") == "ethanesulfonic acid"

    def test_benzenesulfonic_acid(self):
        """c1ccccc1S(=O)(=O)O -> benzenesulfonic acid"""
        assert name_compound("c1ccccc1S(=O)(=O)O") == "benzenesulfonic acid"


class TestSulfurAdditionalCompounds:
    """Additional E2E tests for sulfur compounds."""

    def test_dipropyl_sulfide(self):
        """CCCSCCC -> 1-(propylsulfanyl)propane (was 4-thiaheptane). fix a performance pass: (the Blue Book) needs four heterounits for a skeletal replacement PIN; the ether analog is "methoxyethane (PIN)" (:27745). OPSIN 2.9.0 RT: exact."""
        assert name_compound("CCCSCCC") == "1-(propylsulfanyl)propane"

    def test_dibutyl_sulfide(self):
        """CCCCSCCCC -> 1-(butylsulfanyl)butane (was 5-thianonane). fix a performance pass: (the Blue Book) needs four heterounits for a skeletal replacement PIN; the ether analog is "methoxyethane (PIN)" (:27745). OPSIN 2.9.0 RT: exact."""
        assert name_compound("CCCCSCCCC") == "1-(butylsulfanyl)butane"

    def test_dipropyl_sulfoxide(self):
        """CCCS(=O)CCC -> 1-(propanesulfinyl)propane (PIN; method (1), locant 1 as in '1-(ethanesulfinyl)butane')"""
        assert name_compound("CCCS(=O)CCC") == "1-(propanesulfinyl)propane"

    def test_dipropyl_sulfone(self):
        """CCCS(=O)(=O)CCC -> 1-(propanesulfonyl)propane (PIN; method (1))"""
        assert name_compound("CCCS(=O)(=O)CCC") == "1-(propanesulfonyl)propane"

    def test_butyl_methyl_sulfide(self):
        """CCCCSC -> 1-(methylsulfanyl)butane (was 2-thiahexane). fix a performance pass: (the Blue Book) needs four heterounits for a skeletal replacement PIN; the ether analog is "methoxyethane (PIN)" (:27745). OPSIN 2.9.0 RT: exact."""
        assert name_compound("CCCCSC") == "1-(methylsulfanyl)butane"

    def test_butyl_methyl_sulfoxide(self):
        """CCCCS(=O)C -> 1-(methanesulfinyl)butane (PIN; method (1))"""
        assert name_compound("CCCCS(=O)C") == "1-(methanesulfinyl)butane"

    def test_butyl_methyl_sulfone(self):
        """CCCCS(=O)(=O)C -> 1-(methanesulfonyl)butane (PIN; method (1))"""
        assert name_compound("CCCCS(=O)(=O)C") == "1-(methanesulfonyl)butane"
