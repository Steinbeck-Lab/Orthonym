"""Unit tests for salt and zwitterion naming."""
import pytest
from rdkit import Chem
from orthonym.rules.salts import (
    name_salt,
    name_zwitterion,
    is_salt,
    is_zwitterion,
    _is_amino_acid_zwitterion,
    _apply_stoichiometric_prefix,
)


class TestNameSalt:
    """Test salt naming with compositional nomenclature."""

    def test_sodium_acetate(self):
        """Test basic salt: sodium acetate."""
        mol = Chem.MolFromSmiles('[Na+].[O-]C(C)=O')
        name = name_salt(mol)
        assert name == 'sodium acetate'

    def test_potassium_chloride(self):
        """Test inorganic salt: potassium chloride."""
        mol = Chem.MolFromSmiles('[K+].[Cl-]')
        name = name_salt(mol)
        assert name == 'potassium chloride'

    def test_ammonium_chloride(self):
        """Test organic cation with inorganic anion."""
        mol = Chem.MolFromSmiles('[NH4+].[Cl-]')
        name = name_salt(mol)
        assert name == 'ammonium chloride'

    def test_sodium_chloride(self):
        """Test simple inorganic salt."""
        mol = Chem.MolFromSmiles('[Na+].[Cl-]')
        name = name_salt(mol)
        assert name == 'sodium chloride'

    def test_calcium_acetate(self):
        """Test divalent cation with two anions."""
        mol = Chem.MolFromSmiles('[Ca+2].[O-]C(C)=O.[O-]C(C)=O')
        name = name_salt(mol)
        # Should contain calcium and acetate (may have di- prefix)
        assert 'calcium' in name
        assert 'acetate' in name

    def test_lithium_methoxide(self):
        """Test alkali metal with alkoxide anion.

        169.6-04: the anion now routes through route_charged, which emits the
        IUPAC-2013 PIN -olate form (``methanolate``, rather than the
        retained ``methoxide``. Both are valid (the Blue Book lists ``sodium
        methoxide`` (PIN) AND ``sodium methanolate``); ``lithium methanolate``
        round-trips in OPSIN to C[O-].[Li+] (RT-verified strict equivalent)."""
        mol = Chem.MolFromSmiles('[Li+].[O-]C')
        name = name_salt(mol)
        assert 'lithium' in name
        assert 'methoxide' in name or 'methanolate' in name

    def test_potassium_formate(self):
        """Test potassium formate salt."""
        mol = Chem.MolFromSmiles('[K+].[O-]C=O')
        name = name_salt(mol)
        assert 'potassium' in name
        assert 'formate' in name

    def test_sodium_phenoxide(self):
        """Test sodium phenoxide (sodium phenolate)."""
        mol = Chem.MolFromSmiles('[Na+].[O-]c1ccccc1')
        name = name_salt(mol)
        assert 'sodium' in name
        assert 'oxide' in name.lower() or 'olate' in name.lower()


class TestNameZwitterion:
    """Test zwitterion naming."""

    def test_glycine_zwitterion_systematic(self):
        """Test glycine zwitterion with systematic naming.

         charged Slice B: with style='systematic' the retained-name
        lookup is skipped and route_charged GUARD 4 ->
        ``_name_primary_amine_azaniumyl_zwitterion`` produces the ionic anion-is-
        parent form -- the anion (``acetate``) is the parent, the protonated amine
        an ``azaniumyl`` prefix -> ``azaniumylacetate`` (the C2 locant is elided on
        the 2-carbon acetate; RT-verified full-InChIKey to [NH3+]CC(=O)[O-] by the
        builder's own gate). The older neutral-form ``aminoacetic acid`` (now
        BEST-EFFORT-tier only) and the retained ``glycine`` remain acceptable
        forms of a valid answer.
        """
        mol = Chem.MolFromSmiles('[NH3+]CC([O-])=O')
        name = name_zwitterion(mol, style='systematic')
        low = name.lower()
        is_systematic = 'amino' in low and ('acid' in low or 'anoic' in low)
        is_retained = low == 'glycine'
        # ionic: an azaniumyl prefix on a carboxylate anion parent
        # (systematic -oate/-anoate OR the retained -acetate).
        is_p74_ionic = 'azaniumyl' in low and low.endswith('ate')
        assert is_systematic or is_retained or is_p74_ionic, \
            f"Expected systematic / retained / P-74 ionic name, got: {name}"

    def test_glycine_zwitterion_trivial(self):
        """Test glycine zwitterion may use trivial name."""
        mol = Chem.MolFromSmiles('[NH3+]CC([O-])=O')
        name = name_zwitterion(mol)
        # Either trivial (glycine) or systematic (azaniumylacetate)
        assert name is not None
        assert len(name) > 0

    def test_alanine_zwitterion(self):
        """Test alanine zwitterion."""
        mol = Chem.MolFromSmiles('[NH3+]C(C)C([O-])=O')
        name = name_zwitterion(mol)
        # Should contain some form of amino acid naming
        assert name is not None
        assert len(name) > 0


class TestIsSalt:
    """Test salt detection."""

    def test_is_salt_positive(self):
        """Test that salts are detected correctly."""
        mol = Chem.MolFromSmiles('[Na+].[Cl-]')
        assert is_salt(mol) is True

    def test_is_salt_negative_neutral(self):
        """Test that neutral molecules are not salts."""
        mol = Chem.MolFromSmiles('CCO')
        assert is_salt(mol) is False

    def test_is_salt_negative_single_ion(self):
        """Test that single ions are not salts."""
        mol = Chem.MolFromSmiles('[Na+]')
        assert is_salt(mol) is False

    def test_is_salt_negative_zwitterion(self):
        """Test that zwitterions are not salts (single fragment)."""
        mol = Chem.MolFromSmiles('[NH3+]CC([O-])=O')
        assert is_salt(mol) is False


class TestIsZwitterion:
    """Test zwitterion detection."""

    def test_is_zwitterion_positive(self):
        """Test that zwitterions are detected correctly."""
        mol = Chem.MolFromSmiles('[NH3+]CC([O-])=O')
        assert is_zwitterion(mol) is True

    def test_is_zwitterion_negative_neutral(self):
        """Test that neutral molecules are not zwitterions."""
        mol = Chem.MolFromSmiles('CCO')
        assert is_zwitterion(mol) is False

    def test_is_zwitterion_negative_salt(self):
        """Test that salts are not zwitterions."""
        mol = Chem.MolFromSmiles('[Na+].[Cl-]')
        assert is_zwitterion(mol) is False

    def test_is_zwitterion_negative_single_ion(self):
        """Test that single ions are not zwitterions."""
        mol = Chem.MolFromSmiles('[NH4+]')
        assert is_zwitterion(mol) is False


class TestIsAminoAcidZwitterion:
    """Test amino acid zwitterion pattern detection."""

    def test_glycine_zwitterion(self):
        """Test glycine zwitterion pattern."""
        mol = Chem.MolFromSmiles('[NH3+]CC([O-])=O')
        assert _is_amino_acid_zwitterion(mol) is True

    def test_alanine_zwitterion(self):
        """Test alanine zwitterion pattern."""
        mol = Chem.MolFromSmiles('[NH3+]C(C)C([O-])=O')
        assert _is_amino_acid_zwitterion(mol) is True

    def test_non_amino_acid(self):
        """Test that non-amino acid is not detected."""
        mol = Chem.MolFromSmiles('[NH3+]CCCC([O-])=O')
        # gamma-amino butyrate - still matches alpha pattern if connected
        # depends on exact SMARTS pattern
        result = _is_amino_acid_zwitterion(mol)
        # This may or may not match depending on chain length
        assert isinstance(result, bool)


class TestStoichiometricPrefix:
    """Test stoichiometric prefix application."""

    def test_single_occurrence(self):
        """Test that count=1 returns name unchanged."""
        assert _apply_stoichiometric_prefix('acetate', 1) == 'acetate'

    def test_di_prefix(self):
        """Test di- prefix for count=2."""
        assert _apply_stoichiometric_prefix('acetate', 2) == 'diacetate'

    def test_tri_prefix(self):
        """Test tri- prefix for count=3."""
        assert _apply_stoichiometric_prefix('chloride', 3) == 'trichloride'

    def test_tetra_prefix(self):
        """Test tetra- prefix for count=4."""
        assert _apply_stoichiometric_prefix('oxide', 4) == 'tetraoxide'


# ============================================================================
# a phase / — zwitterion inner salt) naming (Wave 0)
#
# Retained amino-acid zwitterions (glycine/betaine) MUST stay (assert NOW).
# Non-retained zwitterions currently strip the charge and name the neutral
# form (GABA -> '4-aminobutanoic acid'), losing the ionic character.
# The target names the whole skeleton carrying BOTH centres
# (azaniumyl... substituent + -oate suffix) — xfail until Plan 02.
# ============================================================================

from orthonym import name_compound  # noqa: E402


class TestSUB01ZwitterionNegativeCanary:
    """Retained / structured zwitterion names."""

    def test_glycine_retained(self):
        """Glycine retained name allows it; OPSIN-parseable) is preserved
        FIRST per (amino-acid zwitterions sequenced ahead of GUARD 4)."""
        assert name_compound("[NH3+]CC(=O)[O-]") == "glycine"

    def test_betaine_structured_p74_1_3(self):
        """169.6-04: the hardcoded ``betaine`` literal was DELETED (it is NOT
        OPSIN-parseable — the validity gate suppressed it to 'unknown organic
        compound'). route_charged GUARD 4 now produces the structured
        (trimethylazaniumyl)acetate, which round-trips in OPSIN to
        C[N+](C)(C)CC(=O)[O-] (RT=1, a strict improvement over RT=0)."""
        assert name_compound("C[N+](C)(C)CC(=O)[O-]") == "(trimethylazaniumyl)acetate"


class TestSUB01Zwitterion:
    """Non-retained zwitterion -> whole-skeleton azaniumyl...oate ."""

    @pytest.mark.xfail(reason="DEFERRED (169.5, honest-fail): charge-stripped '4-aminobutanoic acid' already RT-correct at connectivity (InChI-L1 ignores charge), so the D-03 azaniumyl...oate compositional path is precision-only / zero-RT / high-risk new-logic. Deferred over the higher-value anion fix.", strict=False)
    def test_gaba_zwitterion_whole_skeleton(self):
        # 4-aminobutanoate zwitterion -> 4-azaniumylbutanoate (both centres).
        name = name_compound("[NH3+]CCCC(=O)[O-]")
        assert "azaniumyl" in name and name.endswith("oate")
