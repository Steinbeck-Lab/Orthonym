"""Unit tests for natural product scaffold and derivative data.

Validates data integrity, SMILES canonicalization, and pattern compilation
for the natural_products data module.
"""

import pytest
from rdkit import Chem

from orthonym.data.natural_products import (
    NATURAL_PRODUCT_DERIVATIVES,
    NATURAL_PRODUCT_SCAFFOLDS,
    get_natural_product_name,
    get_scaffold_patterns,
)


# ---------------------------------------------------------------------------
# TestScaffoldData
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestScaffoldData:
    """Tests for NATURAL_PRODUCT_SCAFFOLDS dictionary."""

    def test_scaffold_count(self):
        """There should be at least 14 scaffold entries."""
        assert len(NATURAL_PRODUCT_SCAFFOLDS) >= 14

    def test_scaffold_keys_are_canonical(self):
        """Every SMILES key must match its RDKit canonical form."""
        for smiles in NATURAL_PRODUCT_SCAFFOLDS:
            mol = Chem.MolFromSmiles(smiles)
            assert mol is not None, f"Invalid SMILES: {smiles}"
            canonical = Chem.MolToSmiles(mol)
            assert canonical == smiles, (
                f"Non-canonical key for "
                f"{NATURAL_PRODUCT_SCAFFOLDS[smiles]['name']}: "
                f"got {smiles!r}, expected {canonical!r}"
            )

    def test_scaffold_required_fields(self):
        """Each scaffold entry must have 'name', 'stem', and 'class'."""
        required = {"name", "stem", "class"}
        for smiles, info in NATURAL_PRODUCT_SCAFFOLDS.items():
            missing = required - set(info.keys())
            assert not missing, (
                f"Scaffold {smiles!r} missing fields: {missing}"
            )

    def test_steroid_scaffolds_present(self):
        """Key steroid scaffolds must be present."""
        names = {v["name"] for v in NATURAL_PRODUCT_SCAFFOLDS.values()}
        expected = {"androstane", "estrane", "pregnane", "cholestane", "gonane"}
        missing = expected - names
        assert not missing, f"Missing steroid scaffolds: {missing}"

    def test_alkaloid_scaffolds_present(self):
        """Key alkaloid scaffolds must be present."""
        names = {v["name"] for v in NATURAL_PRODUCT_SCAFFOLDS.values()}
        expected = {"morphinan", "tropane", "aporphine", "ergoline"}
        missing = expected - names
        assert not missing, f"Missing alkaloid scaffolds: {missing}"

    def test_scaffold_classes(self):
        """All entries must have a known class value."""
        allowed = {"steroid", "alkaloid", "terpenoid", "misc"}
        for smiles, info in NATURAL_PRODUCT_SCAFFOLDS.items():
            assert info["class"] in allowed, (
                f"Unknown class {info['class']!r} for {info['name']}"
            )


# ---------------------------------------------------------------------------
# TestDerivativeData
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDerivativeData:
    """Tests for NATURAL_PRODUCT_DERIVATIVES dictionary."""

    def test_derivative_count(self):
        """There should be at least 15 derivative entries."""
        assert len(NATURAL_PRODUCT_DERIVATIVES) >= 15

    def test_derivative_keys_are_canonical(self):
        """Every SMILES key must match its RDKit canonical form."""
        for smiles, name in NATURAL_PRODUCT_DERIVATIVES.items():
            mol = Chem.MolFromSmiles(smiles)
            assert mol is not None, f"Invalid SMILES for {name}: {smiles}"
            canonical = Chem.MolToSmiles(mol)
            assert canonical == smiles, (
                f"Non-canonical key for {name}: "
                f"got {smiles!r}, expected {canonical!r}"
            )

    def test_cholesterol_present(self):
        """Cholesterol must be in derivatives."""
        names = set(NATURAL_PRODUCT_DERIVATIVES.values())
        assert "cholesterol" in names

    def test_morphine_present(self):
        """Morphine must be in derivatives."""
        names = set(NATURAL_PRODUCT_DERIVATIVES.values())
        assert "morphine" in names

    def test_codeine_present(self):
        """Codeine must be in derivatives."""
        names = set(NATURAL_PRODUCT_DERIVATIVES.values())
        assert "codeine" in names

    def test_camphor_present(self):
        """Camphor must be in derivatives (terpenoid coverage)."""
        names = set(NATURAL_PRODUCT_DERIVATIVES.values())
        assert "camphor" in names

    def test_beta_carotene_present(self):
        """Beta-carotene must be in derivatives (carotenoid coverage)."""
        names = set(NATURAL_PRODUCT_DERIVATIVES.values())
        assert "beta-carotene" in names

    def test_get_natural_product_name_found(self):
        """get_natural_product_name should return correct names.

        v29 Task E2: this used camphor as its "known entry". Camphor is now an
        adjudicated non-PIN (P-64.2.1.1 BlueBookV2.md:28297 makes chalcone the ONLY
        retained ketone PIN; P-64.2.1.2 :28307 is a closed general-nomenclature list
        that excludes it), so this surface correctly returns None for it -- see
        test_get_natural_product_name_camphor_is_demoted below. Switched to
        porphyrin, which is still served here, so the test keeps testing the lookup
        rather than the deny list. Camphor was the ONLY one of the 124 derivative
        rows demoted, so the surface is otherwise untouched.
        """
        porphyrin_smi = "C1=Cc2cc3ccc(cc4nc(cc5ccc(cc1n2)[nH]5)C=C4)[nH]3"
        assert get_natural_product_name(porphyrin_smi) == "porphyrin"

    def test_get_natural_product_name_camphor_is_demoted(self):
        """Camphor is withheld from the PIN lookup but NOT deleted from the table.

        The PIN path now emits the Blue Book's own rendering,
        '1,7,7-trimethylbicyclo[2.2.1]heptan-2-one' (printed verbatim at
        BlueBookV2.md:52648 as '(1R,4R)-1,7,7-trimethylbicyclo[2.2.1]heptan-2-one').
        """
        camphor_smi = "CC12CCC(CC1=O)C2(C)C"
        assert get_natural_product_name(camphor_smi) is None
        assert "camphor" in set(NATURAL_PRODUCT_DERIVATIVES.values())

    def test_get_natural_product_name_not_found(self):
        """get_natural_product_name should return None for unknown SMILES."""
        assert get_natural_product_name("CCO") is None

    def test_get_natural_product_name_cholesterol(self):
        """Cholesterol lookup returns 'cholesterol'."""
        cholesterol_smi = (
            "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC=C4"
            "C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C"
        )
        assert get_natural_product_name(cholesterol_smi) == "cholesterol"

    def test_get_natural_product_name_morphine(self):
        """Morphine lookup returns 'morphine'."""
        morphine_smi = (
            "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2"
            "[C@@H](O)C=C[C@H]3[C@H]1C5"
        )
        assert get_natural_product_name(morphine_smi) == "morphine"


# ---------------------------------------------------------------------------
# TestPatternCompilation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPatternCompilation:
    """Tests for pre-compiled scaffold RDKit Mol patterns."""

    def test_patterns_compiled(self):
        """get_scaffold_patterns() must return a non-empty dict."""
        patterns = get_scaffold_patterns()
        assert len(patterns) > 0

    def test_patterns_are_mol_objects(self):
        """Each pattern value must be an RDKit Mol object."""
        patterns = get_scaffold_patterns()
        for smiles, mol in patterns.items():
            assert isinstance(mol, Chem.rdchem.Mol), (
                f"Pattern for {smiles!r} is not a Mol: {type(mol)}"
            )

    def test_patterns_match_keys(self):
        """Number of compiled patterns equals number of scaffolds."""
        patterns = get_scaffold_patterns()
        assert len(patterns) == len(NATURAL_PRODUCT_SCAFFOLDS), (
            f"Pattern count ({len(patterns)}) != "
            f"scaffold count ({len(NATURAL_PRODUCT_SCAFFOLDS)})"
        )

    def test_patterns_can_match_substructure(self):
        """Patterns should actually work for substructure matching."""
        patterns = get_scaffold_patterns()
        # Cocaine contains the tropane scaffold
        tropane_smi = "CN1[C@@H]2CCC[C@H]1CC2"
        cocaine = Chem.MolFromSmiles(
            "COC(=O)[C@@H]1C[C@@H]2CC[C@H](C1)N2C"
        )
        if tropane_smi in patterns:
            pat = patterns[tropane_smi]
            assert cocaine.HasSubstructMatch(pat), (
                "Cocaine should match tropane scaffold"
            )
