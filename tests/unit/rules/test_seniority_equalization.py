import pytest
from rdkit import Chem
from orthonym.perception.functional_groups import detect_functional_groups
from orthonym.rules.seniority import (
    get_principal_group, SENIORITY_ORDER, _SENIORITY_PARENT,
    SUFFIX_FORMS, PREFIX_FORMS
)


class TestSeniorityEqualization:
    """ASML-18: All alcohol/amine subtypes have equal seniority."""

    def test_seniority_parent_map_exists(self):
        """_SENIORITY_PARENT maps all alcohol and amine subtypes."""
        assert _SENIORITY_PARENT["primary_alcohol"] == "alcohol"
        assert _SENIORITY_PARENT["secondary_alcohol"] == "alcohol"
        assert _SENIORITY_PARENT["tertiary_alcohol"] == "alcohol"
        assert _SENIORITY_PARENT["phenol"] == "alcohol"
        assert _SENIORITY_PARENT["enol"] == "alcohol"
        assert _SENIORITY_PARENT["primary_amine"] == "amine"
        assert _SENIORITY_PARENT["secondary_amine"] == "amine"
        assert _SENIORITY_PARENT["tertiary_amine"] == "amine"
        assert _SENIORITY_PARENT["aromatic_amine"] == "amine"

    def test_seniority_order_intact(self):
        """Per D-09: SENIORITY_ORDER list must still contain all subtypes."""
        assert "primary_alcohol" in SENIORITY_ORDER
        assert "secondary_alcohol" in SENIORITY_ORDER
        assert "tertiary_alcohol" in SENIORITY_ORDER
        assert "primary_amine" in SENIORITY_ORDER

    def test_generic_alcohol_principal_group(self):
        """OC(F)Cl: Only generic 'alcohol' matches -> principal_group is 'alcohol'."""
        mol = Chem.MolFromSmiles("OC(F)Cl")
        fgs = detect_functional_groups(mol)
        pg_name, pg_atoms = get_principal_group(mol, fgs)
        assert pg_name is not None
        assert "alcohol" in pg_name or "ol" in (SUFFIX_FORMS.get(pg_name, (None,))[0] or "")

    def test_acid_beats_alcohol(self):
        """OCC(O)CCC(=O)O: carboxylic_acid is principal (higher seniority than alcohol)."""
        mol = Chem.MolFromSmiles("OCC(O)CCC(=O)O")
        fgs = detect_functional_groups(mol)
        pg_name, pg_atoms = get_principal_group(mol, fgs)
        assert pg_name == "carboxylic_acid"

    def test_generic_alcohol_in_suffix_forms(self):
        """Generic 'alcohol' has suffix and prefix entries."""
        assert "alcohol" in SUFFIX_FORMS
        assert SUFFIX_FORMS["alcohol"] == ("ol", "ol")
        assert "alcohol" in PREFIX_FORMS
        assert PREFIX_FORMS["alcohol"] == "hydroxy"

    def test_mixed_alcohol_subtypes_returns_first_match(self):
        """CC(O)CO: returns first alcohol subtype in seniority order."""
        mol = Chem.MolFromSmiles("CC(O)CO")
        fgs = detect_functional_groups(mol)
        pg_name, pg_atoms = get_principal_group(mol, fgs)
        # Returns specific subtype name (first in seniority order)
        assert pg_name in ("primary_alcohol", "secondary_alcohol", "alcohol")
        assert len(pg_atoms) >= 1
