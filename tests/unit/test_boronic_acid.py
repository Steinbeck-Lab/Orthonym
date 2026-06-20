"""Tests for boronic acid naming (Phase 21, Plan 04).

Boronic acids use functional class naming: "Rboronic acid"
- Simple alkyl: "methylboronic acid", "ethylboronic acid"
- Aryl: "phenylboronic acid"
- Complex R: "(4-methylphenyl)boronic acid"
"""

import pytest
from orthonym import name_compound


@pytest.mark.unit
class TestBoronicAcidNaming:
    """Test boronic acid functional class naming."""

    @pytest.mark.parametrize("smiles,expected", [
        # Simple alkyl boronic acids
        ("CB(O)O", "methylboronic acid"),
        ("CCB(O)O", "ethylboronic acid"),
        ("CCCB(O)O", "propylboronic acid"),
        ("CCCCB(O)O", "butylboronic acid"),
    ])
    def test_alkyl_boronic_acids(self, smiles, expected):
        """Simple alkyl boronic acids: Rboronic acid."""
        assert name_compound(smiles) == expected

    def test_phenylboronic_acid(self):
        """Aryl boronic acid: phenylboronic acid."""
        assert name_compound("c1ccc(cc1)B(O)O") == "phenylboronic acid"

    def test_boronic_acid_is_principal_group(self):
        """Boronic acid detected as principal group (in SENIORITY_ORDER)."""
        from orthonym.namer import Orthonym
        from rdkit import Chem

        namer = Orthonym()
        mol = Chem.MolFromSmiles("CB(O)O")
        canonical = Chem.MolToSmiles(mol, canonical=True)
        features = namer._perceive(mol, "CB(O)O", canonical)
        namer._classify(features)
        assert features.principal_group == "boronic_acid"

    def test_boronic_acid_prefix_form_in_seniority(self):
        """Boronic acid prefix is 'borono' (v22 G2 COV-02; P-68.1.4.2).

        -B(OH)2 has the *preselected* prefix 'borono' (Blue Book P-68.1.4.2 /
        P-67.1.4.2 retained); 'dihydroxyboranyl' is the non-PIN systematic
        alternative. Updated from the pre-G2 'dihydroxyboranyl' assertion.
        """
        from orthonym.rules.seniority import PREFIX_FORMS
        assert PREFIX_FORMS.get("boronic_acid") == "borono"

    def test_boronic_acid_suffix_form_in_seniority(self):
        """Boronic acid suffix forms are registered."""
        from orthonym.rules.seniority import SUFFIX_FORMS
        assert "boronic_acid" in SUFFIX_FORMS
        assert SUFFIX_FORMS["boronic_acid"] == ("boronic acid", "boronic acid")
