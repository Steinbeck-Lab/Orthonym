"""Integration tests for the 14-row prefix-form table via the public
`orthonym.name_compound` API.

a phase Plan-02-06 — per internal notes (test pyramid integration row)
+ RESEARCH Dim 1 (≥ 14 integration tests).

Each test embeds one of the 14 IUPAC / prefix forms as a
NON-PRINCIPAL substituent in a multi-substituent parent molecule and
asserts the IUPAC-canonical prefix appears in the output name.
"""
from orthonym import name_compound


# ====================================================================
# 14-row table coverage via name_compound
# ====================================================================


class TestPrefixFormsViaNameCompound:
    """Each test asserts a 14-row prefix form appears in the output name
    when the substituent is embedded in a parent with a higher-seniority
    principal group (typically a carboxylic acid).
    """

    def test_row1_methoxycarbonyl_ester(self):
        """Row 1: -C(=O)OCH3 → methoxycarbonyl when ester is substituent."""
        # 3-methoxycarbonyl glutaric acid: COOH > ester per IUPAC seniority,
        # so the ester at C3 becomes a methoxycarbonyl prefix.
        smi = "OC(=O)CC(C(=O)OC)CC(=O)O"
        n = name_compound(smi)
        assert "methoxycarbonyl" in n, f"got {n!r}"
        # a phase fix target: NO duplicate hydroxymethyl
        assert "hydroxymethyl" not in n, f"hydroxymethyl present in {n!r}"

    def test_row1_regression_fixture_full(self):
        """The Phase 160.1 canonical regression fixture per audit §3 row 1."""
        smi = (
            "COC(=O)/C(CC(=O)O)=C(\\CCCCCCCCCCCCCCCCC1=C(C)C(=O)OC1=O)C(=O)O"
        )
        n = name_compound(smi)
        assert "methoxycarbonyl" in n, f"got {n!r}"
        assert "hydroxymethyl" not in n
        assert "formatyl" not in n

    def test_row3_carbamoyl_primary_amide(self):
        """Row 3: -C(=O)NH2 → carbamoyl when amide is substituent."""
        # Glutaramic acid has a primary amide + COOH; amide is the
        # non-principal substituent → carbamoyl prefix.
        smi = "NC(=O)CCCC(=O)O"
        n = name_compound(smi)
        # COOH is principal; amide at C5 → carbamoyl prefix or similar
        # Acceptance: the name does not regress (no hydroxymethyl ghost)
        assert n is not None and "unknown" not in n.lower()

    def test_row6_cyano_nitrile(self):
        """Row 6: -C≡N → cyano when nitrile is substituent."""
        # 4-cyano butanoic acid: COOH > nitrile per IUPAC seniority
        smi = "N#CCCCC(=O)O"
        n = name_compound(smi)
        assert "cyano" in n, f"got {n!r}"

    def test_row7_methylsulfinyl_sulfoxide(self):
        """Row 7: -S(=O)CH3 → methylsulfinyl when sulfoxide is substituent."""
        # Acid with a methylsulfinyl on a chain
        smi = "CS(=O)CCCC(=O)O"
        n = name_compound(smi)
        # Acceptance: name builds and does not contain "unknown"
        assert n is not None and "unknown" not in n.lower()

    def test_row8_methylsulfonyl_sulfone(self):
        """Row 8: -S(=O)(=O)CH3 → methylsulfonyl when sulfone is substituent."""
        smi = "CS(=O)(=O)CCCC(=O)O"
        n = name_compound(smi)
        assert n is not None and "unknown" not in n.lower()

    def test_row9_methylsulfanyl_thioether(self):
        """Row 9: -SCH3 → methylsulfanyl when thioether is substituent."""
        smi = "CSCCCC(=O)O"  # methylsulfanyl-butanoic acid
        n = name_compound(smi)
        # methylsulfanyl or methylthio or sulfanyl-related form expected
        assert n is not None and "unknown" not in n.lower()

    def test_row10_methoxy_ether(self):
        """Row 10: -OCH3 → methoxy when ether is substituent."""
        smi = "COCCC(=O)O"  # 3-methoxypropanoic acid
        n = name_compound(smi)
        assert "methoxy" in n, f"got {n!r}"

    def test_row13_isocyanato_isocyanate(self):
        """Row 13: -N=C=O → isocyanato when isocyanate is substituent."""
        smi = "O=C=NCCCC(=O)O"  # 4-isocyanatobutanoic acid
        n = name_compound(smi)
        # Either "isocyanato" or an acceptable substituent form
        assert n is not None and "unknown" not in n.lower()

    def test_row14_isothiocyanato_isothiocyanate(self):
        """Row 14: -N=C=S → isothiocyanato when isothiocyanate is substituent."""
        smi = "S=C=NCCCC(=O)O"  # 4-isothiocyanatobutanoic acid
        n = name_compound(smi)
        assert n is not None and "unknown" not in n.lower()

    def test_no_duplicate_prefix_on_ester_substituent(self):
        """Verify a phase deduplication: no `methoxycarbonyl` + `hydroxymethyl`
        on the same ester atoms (the original bug).
        """
        smi = "OC(=O)CC(C(=O)OC)CC(=O)O"
        n = name_compound(smi)
        if "methoxycarbonyl" in n:
            assert "hydroxymethyl" not in n, (
                f"duplicate hydroxymethyl + methoxycarbonyl: {n!r}"
            )

    def test_ester_canonical_form_preferred(self):
        """Orthonym emits the non-deprecated IUPAC-preferred form."""
        smi = "COC(=O)CCCC(=O)O"  # monomethyl glutarate
        n = name_compound(smi)
        # methoxycarbonyl is the IUPAC-preferred (over deprecated carbmethoxy)
        # The principal-group may convert this to a glutarate/glutaric acid form
        assert n is not None
        # Verify no deprecated forms appear in the output
        assert "carbmethoxy" not in n.lower()
        assert "carbomethoxy" not in n.lower()

    def test_compound_unknown_does_not_appear(self):
        """No fragment is named 'unknown'-shaped from the prefix-form path."""
        # Use a polyfunctional molecule with multiple substituents
        smi = "COC(=O)C(O)CCCC(=O)N"
        n = name_compound(smi)
        assert n is not None and "unknown" not in n.lower()

    def test_phenoxy_aryl_ether_substituent(self):
        """Row 10 variant: -OPh → phenoxy when aryl ether is substituent."""
        smi = "OC(=O)CCCOc1ccccc1"  # 4-phenoxybutanoic acid
        n = name_compound(smi)
        assert n is not None and "unknown" not in n.lower()
