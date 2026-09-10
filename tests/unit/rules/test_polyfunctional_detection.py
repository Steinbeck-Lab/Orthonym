import pytest
from rdkit import Chem
from orthonym.perception.functional_groups import detect_functional_groups
from orthonym.rules.polyfunctional import detect_polyfunctional, PARENT_CLASS_MAP


class TestPolyfunctionalNormalization:
    """: Polyfunctional detection normalizes subtypes to parent classes."""

    def test_parent_class_map_exists(self):
        """PARENT_CLASS_MAP maps all alcohol and amine subtypes."""
        assert "primary_alcohol" in PARENT_CLASS_MAP
        assert "secondary_alcohol" in PARENT_CLASS_MAP
        assert "tertiary_alcohol" in PARENT_CLASS_MAP
        assert "phenol" in PARENT_CLASS_MAP
        assert "enol" in PARENT_CLASS_MAP
        assert PARENT_CLASS_MAP["primary_alcohol"] == "alcohol"
        assert PARENT_CLASS_MAP["primary_amine"] == "amine"

    def test_dihydroxy_acid_is_polyfunctional(self):
        """OCC(O)CCC(=O)O: has alcohol + carboxylic_acid = polyfunctional."""
        mol = Chem.MolFromSmiles("OCC(O)CCC(=O)O")
        fgs = detect_functional_groups(mol)
        assert detect_polyfunctional(mol, fgs) is True

    def test_mixed_alcohol_subtypes_not_polyfunctional(self):
        """CC(O)CO: primary + secondary alcohol only -> NOT polyfunctional (same parent class)."""
        mol = Chem.MolFromSmiles("CC(O)CO")
        fgs = detect_functional_groups(mol)
        # Both are "alcohol" parent class, only 1 distinct class
        assert detect_polyfunctional(mol, fgs) is False

    def test_alcohol_plus_amine_is_polyfunctional(self):
        """NCCO: amine + alcohol = 2 distinct parent classes = polyfunctional."""
        mol = Chem.MolFromSmiles("NCCO")
        fgs = detect_functional_groups(mol)
        assert detect_polyfunctional(mol, fgs) is True

    def test_simple_acid_not_polyfunctional(self):
        """CC(=O)O: just carboxylic acid -> NOT polyfunctional."""
        mol = Chem.MolFromSmiles("CC(=O)O")
        fgs = detect_functional_groups(mol)
        assert detect_polyfunctional(mol, fgs) is False

    def test_generic_alcohol_normalized(self):
        """Generic 'alcohol' maps to itself in PARENT_CLASS_MAP."""
        assert PARENT_CLASS_MAP.get("alcohol") == "alcohol"

    def test_amine_subtypes_normalized(self):
        """All amine subtypes map to 'amine'."""
        for sub in ["primary_amine", "secondary_amine", "tertiary_amine", "aromatic_amine"]:
            assert PARENT_CLASS_MAP[sub] == "amine"
