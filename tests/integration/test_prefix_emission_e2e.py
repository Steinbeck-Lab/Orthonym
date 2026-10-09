"""
End-to-end integration tests for non-principal FG prefix emission.

Verifies that the full naming pipeline (perception -> classification -> assembly)
correctly emits subordinate prefixes for non-principal functional groups.
"""
import pytest
from orthonym import name_compound


class TestPrefixEmissionE2E:
    """End-to-end naming tests for non-principal functional group prefixes."""

    def test_glutamine_full_name(self):
        """The amide carbon stays in the acid chain: 'amino' + 'oxo', not a 'carbamoyl' prefix.

        NC(=O)CCCC(N)C(=O)O is homoglutamine (six carbons; glutamine is NC(=O)CCC(N)C(=O)O,
        '2,5-diamino-5-oxopentanoic acid', Table 10.4, the Blue Book).
        'Amic acids' (the Blue Book): '4-amino-4-oxobutanoic acid (PIN)
        3-carbamoylpropanoic acid' -- the preferred name takes the amide carbon into the
        principal chain, expressed as amino + oxo; 'carbamoyl' is the non-preferred form.
        """
        name = name_compound("NC(=O)CCCC(N)C(=O)O")
        assert name == "2,6-diamino-6-oxohexanoic acid", (
            f"Expected '2,6-diamino-6-oxohexanoic acid', got '{name}'"
        )

    def test_cyanopropanoic_acid_full(self):
        """Cyanoacetic acid end-to-end (N#CCC(=O)O has two chain carbons plus the cyano carbon)."""
        name = name_compound("N#CCC(=O)O")
        # 'Cyanic acid' (the Blue Book): 'NC-CH2-COOH cyanoacetic acid (PIN)'.
        # (The earlier expected value '3-cyanopropanoic acid' is N#CCCC(=O)O, one CH2 more.)
        assert name == "cyanoacetic acid", f"Expected 'cyanoacetic acid', got '{name}'"

    def test_formylbenzoic_acid_full(self):
        """4-formylbenzoic acid end-to-end."""
        name = name_compound("O=Cc1ccc(C(=O)O)cc1")
        assert name == "4-formylbenzoic acid", f"Expected '4-formylbenzoic acid', got '{name}'"

    def test_mixed_acid_amide(self):
        """Compound with both acid (principal) and amide (subordinate).

        2,6-diamino-6-oxohexanoic acid (amide carbon in the chain;.
        """
        # NC(=O)CCCC(N)C(=O)O is homoglutamine (see test_glutamine_full_name)
        name = name_compound("NC(=O)CCCC(N)C(=O)O")
        # 'Amic acids' (the Blue Book): '4-amino-4-oxobutanoic acid (PIN)
        # 3-carbamoylpropanoic acid' -- the amide stays in the chain as amino + oxo.
        assert name == "2,6-diamino-6-oxohexanoic acid", (
            f"Expected '2,6-diamino-6-oxohexanoic acid', got '{name}'"
        )

    def test_acid_nitrile_compound(self):
        """Compound with acid (principal) and nitrile (subordinate).

        N#CCCC(=O)O -> 3-cyanopropanoic acid (the nitrile carbon is part of 'cyano', not
        of the propanoic chain; '4-cyanobutanoic acid' is N#CCCCC(=O)O).
        """
        name = name_compound("N#CCCC(=O)O")
        # (the Blue Book): when a group with priority for citation as the
        # principal characteristic group is present, -CN is the prefix 'cyano', which "must
        # also be used when the -CN group is located at the end of a chain"; example
        # '3-cyanopropanoic acid (PIN)' (:34742).
        assert name == "3-cyanopropanoic acid", f"Expected '3-cyanopropanoic acid', got '{name}'"

    def test_amide_as_principal_not_affected(self):
        """When amide IS the principal group, no carbamoyl prefix should appear."""
        name = name_compound("CCC(=O)N")
        assert "carbamoyl" not in name, f"Unexpected 'carbamoyl' in '{name}'"
        assert "propanamide" == name, f"Expected 'propanamide', got '{name}'"
