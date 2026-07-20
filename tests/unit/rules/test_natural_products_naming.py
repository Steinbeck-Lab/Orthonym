"""Unit tests for natural product naming rules.

Tests the integration of data (derivative lookup) and perception (scaffold
detection) into the naming pipeline via rules/natural_products.py.

Test classes:
- TestExactDerivativeNaming: Exact SMILES → trivial name
- TestScaffoldNaming: Parent scaffolds → scaffold name
- TestNonNaturalProducts: Non-NP molecules → None
- TestPipelineIntegration: End-to-end via name_compound()
"""

import pytest
from rdkit import Chem

from orthonym.rules.natural_products import name_natural_product
from orthonym import name_compound


# ---------------------------------------------------------------------------
# Helper: canonical SMILES for consistency
# ---------------------------------------------------------------------------

def _mol(smiles: str):
    """Return RDKit Mol from SMILES, raising on invalid input."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Invalid SMILES: {smiles}"
    return mol


# ---------------------------------------------------------------------------
# Exact derivative SMILES (with stereochemistry, matching data module)
# ---------------------------------------------------------------------------

CHOLESTEROL_SMILES = (
    "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC=C4C"
    "[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C"
)
MORPHINE_SMILES = (
    "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5"
)
CODEINE_SMILES = (
    "COc1ccc2c3c1O[C@H]1[C@@H](O)C=C[C@H]4[C@@H](C2)N(C)CC[C@@]341"
)
HYDROCODONE_SMILES = (
    "COc1ccc2c3c1O[C@H]1C(=O)CC[C@H]4[C@@H](C2)N(C)CC[C@]314"
)
DIAMORPHINE_SMILES = (
    "CC(=O)Oc1ccc2c3c1O[C@H]1[C@@H](OC(C)=O)C=C[C@H]4"
    "[C@@H](C2)N(C)CC[C@@]341"
)
CAMPHOR_SMILES = "CC12CCC(CC1=O)C2(C)C"
LIMONENE_SMILES = "C=C(C)C1CC=C(C)CC1"


# ---------------------------------------------------------------------------
# Scaffold SMILES (parent skeletons without substituents)
# ---------------------------------------------------------------------------

ANDROSTANE_SMILES = (
    "C[C@@]12CCC[C@H]1[C@@H]1CCC3CCCC[C@]3(C)[C@H]1CC2"
)
GONANE_SMILES = (
    "C1CC[C@H]2C(C1)CC[C@H]1[C@@H]3CCC[C@H]3CC[C@@H]12"
)
ESTRANE_SMILES = (
    "C[C@@]12CCC[C@H]1[C@@H]1CCC3CCCC[C@@H]3[C@H]1CC2"
)
MORPHINAN_SMILES = (
    "c1ccc2c(c1)C[C@H]1NCC[C@@]23CCCC[C@@H]13"
)
TROPANE_SMILES = "CN1[C@@H]2CCC[C@H]1CC2"


# ===========================================================================
# Test Class 1: Exact derivative lookup
# ===========================================================================

@pytest.mark.unit
class TestExactDerivativeNaming:
    """Exact derivatives should return their trivial names."""

    def test_cholesterol(self):
        mol = _mol(CHOLESTEROL_SMILES)
        assert name_natural_product(mol) == "cholesterol"

    def test_morphine(self):
        mol = _mol(MORPHINE_SMILES)
        assert name_natural_product(mol) == "morphine"

    def test_codeine(self):
        mol = _mol(CODEINE_SMILES)
        assert name_natural_product(mol) == "codeine"

    def test_hydrocodone(self):
        mol = _mol(HYDROCODONE_SMILES)
        assert name_natural_product(mol) == "hydrocodone"

    def test_diamorphine(self):
        mol = _mol(DIAMORPHINE_SMILES)
        assert name_natural_product(mol) == "diamorphine"

    def test_camphor(self):
        mol = _mol(CAMPHOR_SMILES)
        assert name_natural_product(mol) == "camphor"

    def test_limonene(self):
        mol = _mol(LIMONENE_SMILES)
        assert name_natural_product(mol) == "limonene"


# ===========================================================================
# Test Class 2: Parent scaffold naming
# ===========================================================================

@pytest.mark.unit
class TestScaffoldNaming:
    """Parent scaffolds without substituents should return the scaffold name."""

    def test_androstane(self):
        mol = _mol(ANDROSTANE_SMILES)
        assert name_natural_product(mol) == "androstane"

    def test_gonane(self):
        mol = _mol(GONANE_SMILES)
        assert name_natural_product(mol) == "gonane"

    def test_estrane(self):
        mol = _mol(ESTRANE_SMILES)
        assert name_natural_product(mol) == "estrane"

    def test_morphinan(self):
        mol = _mol(MORPHINAN_SMILES)
        assert name_natural_product(mol) == "morphinan"

    def test_tropane(self):
        mol = _mol(TROPANE_SMILES)
        assert name_natural_product(mol) == "tropane"


# ===========================================================================
# Test Class 3: Non-natural products should return None
# ===========================================================================

@pytest.mark.unit
class TestNonNaturalProducts:
    """Non-NP molecules should return None from name_natural_product()."""

    def test_benzene_returns_none(self):
        mol = _mol("c1ccccc1")
        assert name_natural_product(mol) is None

    def test_ethanol_returns_none(self):
        mol = _mol("CCO")
        assert name_natural_product(mol) is None

    def test_cyclohexane_returns_none(self):
        mol = _mol("C1CCCCC1")
        assert name_natural_product(mol) is None

    def test_naphthalene_returns_none(self):
        mol = _mol("c1cccc2ccccc12")
        assert name_natural_product(mol) is None

    def test_none_mol_returns_none(self):
        assert name_natural_product(None) is None


# ===========================================================================
# Test Class 4: Pipeline integration via name_compound()
# ===========================================================================

@pytest.mark.unit
class TestPipelineIntegration:
    """NP detection integrates correctly with the full namer.py pipeline."""

    def test_cholesterol_via_name_compound(self):
        result = name_compound(CHOLESTEROL_SMILES)
        assert result == "cholesterol"

    def test_ethanol_unchanged(self):
        """Ethanol should still come from retained names."""
        assert name_compound("CCO") == "ethanol"

    def test_benzene_unchanged(self):
        """Benzene should still come from retained names."""
        assert name_compound("c1ccccc1") == "benzene"

    def test_butane_unchanged(self):
        """Butane should still come from systematic naming."""
        assert name_compound("CCCC") == "butane"

    def test_systematic_style_still_returns_np(self):
        """NP names are returned even with style='systematic' (no systematic PIN exists)."""
        result = name_compound(CHOLESTEROL_SMILES, style="systematic")
        assert result == "cholesterol"

    def test_morphine_via_name_compound(self):
        result = name_compound(MORPHINE_SMILES)
        assert result == "morphine"

    def test_camphor_via_name_compound(self):
        result = name_compound(CAMPHOR_SMILES)
        assert result == "camphor"


# ===========================================================================
# Test Class 5: NP hydroxyl prefix/suffix exclusivity (Phase 105-02)
# ===========================================================================

@pytest.mark.unit
class TestNPHydroxylRepresentation:
    """Verify hydroxyl appears as suffix OR prefix, never both.

    IUPAC P-35.2.1: principal group as suffix only.
    IUPAC P-59.1: non-principal groups as prefixes only.

    - Hydroxyl-only steroid: -ol suffix, NO hydroxy prefix
    - Hydroxyl+ketone steroid: hydroxy prefix + -one suffix
    - Ketone-only steroid: -one suffix, NO hydroxy prefix
    """

    def test_hydroxyl_only_steroid_uses_ol_suffix(self):
        """Hydroxyl-only steroid: should have -ol suffix and NO 'hydroxy' prefix."""
        # Cholest-5-en-3-ol (cholesterol without the retained name)
        smiles = (
            "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC=C4C"
            "[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C"
        )
        result = name_compound(smiles)
        # cholesterol is a retained name, so check directly
        assert result == "cholesterol"

    def test_hydroxyl_only_steroid_systematic(self):
        """Non-retained hydroxyl steroid should use -ol suffix, no hydroxy prefix."""
        # Androstan-3-ol (no retained name for this)
        smiles = "O[C@H]1CC[C@@]2(C)[C@H]3CC[C@@]4(C)[C@@H](CC2)CC[C@@H]4[C@@H]3CC1"
        mol = _mol(smiles)
        result = name_natural_product(mol)
        if result is not None:
            # If detected as NP: should have -ol suffix without hydroxy prefix
            assert "ol" in result, f"Expected -ol suffix in '{result}'"
            # Should not have both
            lower = result.lower()
            if lower.endswith("ol") or "-ol" in lower:
                assert "hydroxy" not in lower, (
                    f"Both 'hydroxy' prefix and '-ol' suffix found in: '{result}'"
                )

    def test_ketone_steroid_no_hydroxy(self):
        """Ketone-only steroid: -one suffix, no hydroxy prefix."""
        # Androst-4-en-3-one (no OH group)
        smiles = (
            "C[C@]12CC[C@H]3[C@@H](CCC4=CC(=O)CC[C@@]43C)[C@@H]1CCC2"
        )
        mol = _mol(smiles)
        result = name_natural_product(mol)
        if result is not None:
            lower = result.lower()
            assert "hydroxy" not in lower, (
                f"Unexpected 'hydroxy' in ketone-only steroid: '{result}'"
            )

    def test_hydroxyl_plus_ketone_steroid(self):
        """Hydroxyl+ketone steroid: hydroxy prefix + -one suffix."""
        # Testosterone: 17-hydroxyandr-4-en-3-one
        smiles = (
            "C[C@]12CC[C@H]3[C@@H](CCC4=CC(=O)CC[C@@]43C)"
            "[C@@H]1CC[C@@H]2O"
        )
        result = name_compound(smiles)
        lower = result.lower()
        # Should have hydroxy prefix AND -one suffix, but NOT -ol suffix
        assert "hydroxy" in lower, (
            f"Expected 'hydroxy' prefix for hydroxyl+ketone steroid: '{result}'"
        )
        assert "one" in lower, (
            f"Expected '-one' suffix for hydroxyl+ketone steroid: '{result}'"
        )
        # Should NOT have -ol suffix (hydroxyl is prefix when ketone present)
        # Allow 'ol' in words like 'hydroxy' but not as suffix
        name_after_last_hyphen = result.rsplit("-", 1)[-1] if "-" in result else result
        has_ol_suffix = name_after_last_hyphen.lower().startswith("ol") or result.lower().endswith("ol")
        if has_ol_suffix:
            # Only fail if it's actually a suffix, not part of another word
            # "3-ol" would be wrong, "hydroxy" containing 'ol' is fine
            import re
            assert not re.search(r'-\d*-?\w*ol\b', result.lower().replace("hydroxy", "")), (
                f"Both 'hydroxy' prefix and '-ol' suffix for same group in: '{result}'"
            )


# ===========================================================================
# Test Class 6: Expanded steroid derivatives (Phase 141)
# ===========================================================================

@pytest.mark.unit
class TestExpandedSteroidDerivatives:
    """Expanded steroid derivatives from OPSIN NP data."""

    def test_progesterone(self):
        """Progesterone should return 'progesterone' (not gonane-based name)."""
        smiles = "CC(=O)[C@H]1CC[C@@H]2[C@@H]1CC[C@H]1[C@@H]2CCC2=CC(=O)CC[C@@]21C"
        result = name_compound(smiles)
        assert result == "progesterone", f"Expected progesterone, got: {result}"

    def test_androstenedione(self):
        smiles = "C[C@]12CCC(=O)CC1CC[C@@H]1[C@@H]2CC[C@]2(C)C(=O)CC[C@@H]12"
        result = name_compound(smiles)
        assert result == "androstenedione"

    def test_campestanol(self):
        smiles = "CC(C)[C@H](C)CC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC[C@H]4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C"
        result = name_compound(smiles)
        assert result == "campestanol"

    def test_androstanediol(self):
        smiles = "C[C@]12CCC(O)C[C@@H]1CC[C@@H]1[C@@H]2CC[C@]2(C)[C@@H](O)CC[C@@H]12"
        result = name_compound(smiles)
        assert result == "androstanediol"

    def test_androstenediol(self):
        smiles = "C[C@]12CC[C@H]3[C@@H](CCC4C[C@@H](O)CC[C@@]43C)[C@@H]1CC[C@@H]2O"
        result = name_compound(smiles)
        assert result == "androstenediol"

    def test_estratetraenol(self):
        smiles = "C[C@@]12C=CC[C@H]1[C@@H]1CCc3cc(O)ccc3[C@H]1CC2"
        result = name_compound(smiles)
        assert result == "estratetraenol"

    def test_cardenolide(self):
        smiles = "C[C@]12CC[C@H]3[C@@H](CCC4CCCC[C@@]43C)[C@H]1CC[C@@H]2C1=CC(=O)OC1"
        result = name_compound(smiles)
        assert result == "cardenolide"

    def test_bufadienolide(self):
        smiles = "C[C@]12CC[C@H]3[C@@H](CCC4CCCC[C@@]43C)[C@H]1CC[C@@H]2c1ccc(=O)oc1"
        result = name_compound(smiles)
        assert result == "bufadienolide"


# ===========================================================================
# Test Class 7: Expanded alkaloid derivatives (Phase 141)
# ===========================================================================

@pytest.mark.unit
class TestExpandedAlkaloidDerivatives:
    """Expanded alkaloid derivatives from OPSIN NP data."""

    def test_lysergic_acid(self):
        smiles = "CN1C[C@H](C(=O)O)C=C2c3cccc4[nH]cc(c34)C[C@H]21"
        result = name_compound(smiles)
        assert result == "lysergic acid"

    def test_dihydrolysergic_acid(self):
        smiles = "CN1C[C@H](C(=O)O)CC2c3cccc4[nH]cc(c34)C[C@H]21"
        result = name_compound(smiles)
        assert result == "dihydrolysergic acid"

    def test_dihydromorphine(self):
        smiles = "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)CC[C@H]3[C@H]1C5"
        result = name_compound(smiles)
        assert result == "dihydromorphine"

    def test_morphinone(self):
        smiles = "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2C(=O)C=C[C@H]3[C@H]1C5"
        result = name_compound(smiles)
        assert result == "morphinone"

    def test_codeinone(self):
        smiles = "COc1ccc2c3c1O[C@H]1C(=O)C=C[C@H]4[C@@H](C2)N(C)CC[C@]314"
        result = name_compound(smiles)
        assert result == "codeinone"

    def test_dihydrocodeine(self):
        smiles = "COc1ccc2c3c1O[C@H]1[C@@H](O)CC[C@H]4[C@@H](C2)N(C)CC[C@@]341"
        result = name_compound(smiles)
        assert result == "dihydrocodeine"

    def test_lysergamide(self):
        smiles = "CN1C[C@H](C(N)=O)C=C2c3cccc4[nH]cc(c34)C[C@H]21"
        result = name_compound(smiles)
        assert result == "lysergamide"

    def test_lysergol(self):
        smiles = "CN1C[C@H](CO)C=C2c3cccc4[nH]cc(c34)C[C@H]21"
        result = name_compound(smiles)
        assert result == "lysergol"


# ===========================================================================
# Test Class 8: Expanded terpene derivatives (Phase 141)
# ===========================================================================

@pytest.mark.unit
class TestExpandedTerpeneDerivatives:
    """Expanded terpene derivatives from OPSIN NP data."""

    def test_alpha_terpinene(self):
        smiles = "CC1=CC=C(C(C)C)CC1"
        result = name_compound(smiles)
        assert result == "alpha-terpinene"

    def test_beta_terpinene(self):
        smiles = "C=C1CC=C(C(C)C)CC1"
        result = name_compound(smiles)
        assert result == "beta-terpinene"

    def test_gamma_terpinene(self):
        smiles = "CC1=CCC(C(C)C)=CC1"
        result = name_compound(smiles)
        assert result == "gamma-terpinene"

    def test_delta_terpinene(self):
        smiles = "CC1=CCC(=C(C)C)CC1"
        result = name_compound(smiles)
        assert result == "delta-terpinene"

    def test_4_terpineol(self):
        smiles = "CC1=CCC(O)(C(C)C)CC1"
        result = name_compound(smiles)
        assert result == "4-terpineol"


# ===========================================================================
# Test Class 9: New scaffold recognition (Phase 141)
# ===========================================================================

@pytest.mark.unit
class TestNewScaffoldRecognition:
    """New scaffolds added in Phase 141 are detected."""

    def test_aconitane_scaffold(self):
        """Bare aconitane scaffold should be recognized."""
        smiles = "C1C[C@H]2CN[C@@H]3[C@@H]4C[C@H]2[C@@]3(C1)[C@@H]1C[C@@H]2CC[C@H]4[C@H]1C2"
        mol = _mol(smiles)
        result = name_natural_product(mol)
        assert result is not None, "Aconitane scaffold should be detected"
        assert "aconit" in result.lower(), f"Expected aconitane name, got: {result}"

    def test_berberine_scaffold(self):
        """Bare berberine scaffold should be recognized."""
        smiles = "c1ccc2c(c1)CC1c3ccccc3CCN1C2"
        mol = _mol(smiles)
        result = name_natural_product(mol)
        assert result is not None, "Berberine scaffold should be detected"
        assert "berber" in result.lower(), f"Expected berberine name, got: {result}"

    def test_menthane_derivative(self):
        """Menthane as exact derivative should be recognized."""
        # Menthane is in derivatives (not scaffolds -- too generic for substructure)
        smiles = "CC1CCC(C(C)C)CC1"
        result = name_compound(smiles)
        assert result == "menthane", f"Expected menthane, got: {result}"


# ===========================================================================
# Test Class 10: No regression on original 26 NP derivatives (Phase 141)
# ===========================================================================

@pytest.mark.unit
class TestNoRegressionExistingNP:
    """ALL 26 original NP derivatives must still return correct names."""

    def test_cholesterol(self):
        assert name_compound(CHOLESTEROL_SMILES) == "cholesterol"

    def test_morphine(self):
        assert name_compound(MORPHINE_SMILES) == "morphine"

    def test_codeine(self):
        assert name_compound(CODEINE_SMILES) == "codeine"

    def test_diamorphine(self):
        assert name_compound(DIAMORPHINE_SMILES) == "diamorphine"

    def test_hydrocodone(self):
        assert name_compound(HYDROCODONE_SMILES) == "hydrocodone"

    def test_camphor(self):
        assert name_compound(CAMPHOR_SMILES) == "camphor"

    def test_limonene(self):
        assert name_compound(LIMONENE_SMILES) == "limonene"

    def test_alpha_pinene(self):
        assert name_compound("CC1=CCC2CC1C2(C)C") == "alpha-pinene"

    def test_beta_pinene(self):
        assert name_compound("CC1(C)C2=CCC1CC2") == "beta-pinene"

    def test_alpha_terpineol(self):
        assert name_compound("CC1=CCC(C(C)(C)O)CC1") == "alpha-terpineol"

    def test_beta_terpineol(self):
        assert name_compound("C=C(C)C1CCC(C)(O)CC1") == "beta-terpineol"

    def test_beta_carotene(self):
        smi = "CC1=C(/C=C/C(C)=C/C=C/C(C)=C/C=C/C=C(C)/C=C/C=C(C)/C=C/C2=C(C)CCC2(C)C)C(C)(C)CCC1"
        assert name_compound(smi) == "beta-carotene"

    def test_flavone(self):
        # v26 BP-4 Ph4: de-headlined (general-only per P-102.6.1.4) -> systematic PIN
        assert name_compound("O=c1cc(-c2ccccc2)oc2ccccc12") == \
            "2-phenyl-4H-1-benzopyran-4-one"

    def test_flavanone(self):
        assert name_compound("O=C1CC(c2ccccc2)Oc2ccccc21") == \
            "2-phenyl-2,3-dihydro-4H-1-benzopyran-4-one"

    def test_isoflavone(self):
        assert name_compound("O=c1c(-c2ccccc2)coc2ccccc12") == \
            "3-phenyl-4H-1-benzopyran-4-one"

    def test_chromanone(self):
        assert name_compound("O=C1CCOc2ccccc21") == \
            "2,3-dihydro-4H-1-benzopyran-4-one"

    def test_chromone(self):
        # v23 IH-01f: chromone de-headlined to the PIN (1-benzopyran is the PIN ring
        # parent per P-19(d); P-64.2.2.2.2 ketone = substitution of the 4H >CH2).
        assert name_compound("O=c1ccoc2ccccc12") == "4H-1-benzopyran-4-one"

    def test_pinane(self):
        assert name_compound("CC1CCC2CC1C2(C)C") == "pinane"

    def test_bornane(self):
        assert name_compound("CC12CCC(CC1)C2(C)C") == "bornane"

    def test_androstane(self):
        assert name_compound(ANDROSTANE_SMILES) == "androstane"

    def test_gonane(self):
        assert name_compound(GONANE_SMILES) == "gonane"

    def test_estrane(self):
        assert name_compound(ESTRANE_SMILES) == "estrane"

    def test_morphinan(self):
        assert name_compound(MORPHINAN_SMILES) == "morphinan"

    def test_tropane(self):
        assert name_compound(TROPANE_SMILES) == "tropane"
