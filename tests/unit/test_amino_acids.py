"""Tests for amino acid naming (POLY-07).

Tests both trivial names for standard amino acids and systematic names
for non-standard amino acids.
"""
import pytest
from orthonym import name_compound


class TestStandardAminoAcids:
    """Test trivial names for standard amino acids."""

    def test_glycine(self):
        """Glycine is achiral, simplest amino acid."""
        assert name_compound("NCC(=O)O") == "glycine"

    def test_alanine(self):
        """Alanine: 2-aminopropanoic acid."""
        assert name_compound("CC(N)C(=O)O") == "alanine"

    def test_valine(self):
        """Valine: 2-amino-3-methylbutanoic acid."""
        assert name_compound("CC(C)C(N)C(=O)O") == "valine"

    def test_leucine(self):
        """Leucine: 2-amino-4-methylpentanoic acid."""
        assert name_compound("CC(C)CC(N)C(=O)O") == "leucine"

    def test_isoleucine(self):
        """Isoleucine: 2-amino-3-methylpentanoic acid."""
        assert name_compound("CCC(C)C(N)C(=O)O") == "isoleucine"

    def test_serine(self):
        """Serine: 2-amino-3-hydroxypropanoic acid."""
        result = name_compound("NC(CO)C(=O)O")
        assert result == "serine"

    def test_cysteine(self):
        """Cysteine: 2-amino-3-mercaptopropanoic acid."""
        result = name_compound("NC(CS)C(=O)O")
        assert result == "cysteine"

    def test_methionine(self):
        """Methionine: 2-amino-4-(methylthio)butanoic acid."""
        result = name_compound("CSCC(N)C(=O)O")
        assert result == "methionine"

    def test_phenylalanine(self):
        """Phenylalanine: 2-amino-3-phenylpropanoic acid."""
        result = name_compound("NC(Cc1ccccc1)C(=O)O")
        assert result == "phenylalanine"

    def test_proline(self):
        """Proline: imino acid, cyclic amino acid."""
        result = name_compound("OC(=O)C1CCCN1")
        assert result == "proline"

    def test_threonine(self):
        """Threonine: 2-amino-3-hydroxybutanoic acid."""
        result = name_compound("CC(O)C(N)C(=O)O")
        assert result == "threonine"

    def test_tyrosine(self):
        """Tyrosine: 2-amino-3-(4-hydroxyphenyl)propanoic acid."""
        result = name_compound("NC(Cc1ccc(O)cc1)C(=O)O")
        assert result == "tyrosine"

    def test_tryptophan(self):
        """Tryptophan: contains indole ring."""
        result = name_compound("NC(Cc1c[nH]c2ccccc12)C(=O)O")
        assert result == "tryptophan"


class TestAcidicAminoAcids:
    """Test acidic amino acids and their amides."""

    def test_aspartic_acid(self):
        """Aspartic acid: 2-aminobutanedioic acid."""
        result = name_compound("NC(CC(=O)O)C(=O)O")
        assert result == "aspartic acid"

    def test_glutamic_acid(self):
        """Glutamic acid: 2-aminopentanedioic acid."""
        result = name_compound("NC(CCC(=O)O)C(=O)O")
        assert result == "glutamic acid"

    def test_asparagine(self):
        """Asparagine: 2-amino-3-carbamoylpropanoic acid."""
        result = name_compound("NC(CC(N)=O)C(=O)O")
        assert result == "asparagine"

    def test_glutamine(self):
        """Glutamine: 2-amino-4-carbamoylbutanoic acid."""
        result = name_compound("NC(CCC(N)=O)C(=O)O")
        assert result == "glutamine"


class TestBasicAminoAcids:
    """Test basic amino acids."""

    def test_lysine(self):
        """Lysine: 2,6-diaminohexanoic acid."""
        result = name_compound("NCCCCC(N)C(=O)O")
        assert result == "lysine"

    def test_arginine(self):
        """Arginine: contains guanidino group."""
        result = name_compound("NC(CCCNC(N)=N)C(=O)O")
        assert result == "arginine"

    def test_histidine(self):
        """Histidine: contains imidazole ring."""
        result = name_compound("NC(Cc1cnc[nH]1)C(=O)O")
        assert result == "histidine"


class TestNonStandardAminoAcids:
    """Test non-standard amino acids."""

    def test_sarcosine(self):
        """Sarcosine: N-methylglycine."""
        result = name_compound("CNCC(=O)O")
        assert result == "sarcosine"

    def test_ornithine(self):
        """Ornithine: 2,5-diaminopentanoic acid."""
        result = name_compound("NCCCC(N)C(=O)O")
        assert result == "ornithine"

    def test_gaba(self):
        """GABA: 4-aminobutanoic acid (not alpha-amino acid)."""
        result = name_compound("NCCCC(=O)O")
        # GABA is in our NON_STANDARD dict as "4-aminobutanoic acid"
        assert result == "4-aminobutanoic acid"


class TestAminoAcidDetection:
    """Test amino acid detection logic."""

    def test_not_amino_acid_acid_only(self):
        """Simple carboxylic acid is not an amino acid."""
        from rdkit import Chem
        from orthonym.rules.amino_acids import detect_amino_acid

        # Simple acid - no amino group
        mol = Chem.MolFromSmiles("CC(=O)O")
        assert detect_amino_acid(mol) is False

    def test_not_amino_acid_amine_only(self):
        """Simple amine is not an amino acid."""
        from rdkit import Chem
        from orthonym.rules.amino_acids import detect_amino_acid

        # Simple amine - no acid
        mol = Chem.MolFromSmiles("CCN")
        assert detect_amino_acid(mol) is False

    def test_is_amino_acid_glycine(self):
        """Glycine is detected as amino acid."""
        from rdkit import Chem
        from orthonym.rules.amino_acids import detect_amino_acid

        mol = Chem.MolFromSmiles("NCC(=O)O")
        assert detect_amino_acid(mol) is True

    def test_is_amino_acid_alanine(self):
        """Alanine is detected as amino acid."""
        from rdkit import Chem
        from orthonym.rules.amino_acids import detect_amino_acid

        mol = Chem.MolFromSmiles("CC(N)C(=O)O")
        assert detect_amino_acid(mol) is True

    def test_proline_detected(self):
        """Proline (cyclic) is detected as amino acid."""
        from rdkit import Chem
        from orthonym.rules.amino_acids import detect_amino_acid

        mol = Chem.MolFromSmiles("OC(=O)C1CCCN1")
        assert detect_amino_acid(mol) is True


class TestSystematicAminoAcidNaming:
    """Test systematic naming for amino acids not in lookup."""

    def test_2_aminopropanoic_acid_systematic(self):
        """Test systematic naming function directly."""
        from orthonym.rules.amino_acids import _name_amino_acid_systematic
        from rdkit import Chem

        mol = Chem.MolFromSmiles("CC(N)C(=O)O")
        systematic = _name_amino_acid_systematic(mol)
        assert systematic == "2-aminopropanoic acid"

    def test_2_aminobutanoic_acid_systematic(self):
        """Test 4-carbon amino acid systematic."""
        from orthonym.rules.amino_acids import _name_amino_acid_systematic
        from rdkit import Chem

        mol = Chem.MolFromSmiles("CCC(N)C(=O)O")
        systematic = _name_amino_acid_systematic(mol)
        assert systematic == "2-aminobutanoic acid"


class TestAminoAcidDataModule:
    """Test the amino acid data module functions."""

    def test_get_amino_acid_name_glycine(self):
        """get_amino_acid_name returns glycine."""
        from orthonym.data.amino_acids import get_amino_acid_name
        assert get_amino_acid_name("NCC(=O)O") == "glycine"

    def test_get_amino_acid_name_unknown(self):
        """get_amino_acid_name returns None for unknown."""
        from orthonym.data.amino_acids import get_amino_acid_name
        assert get_amino_acid_name("CCCCC") is None

    def test_is_standard_amino_acid(self):
        """is_standard_amino_acid works correctly."""
        from orthonym.data.amino_acids import is_standard_amino_acid
        assert is_standard_amino_acid("NCC(=O)O") is True
        assert is_standard_amino_acid("CC(N)C(=O)O") is True
        assert is_standard_amino_acid("CCCCC") is False


class TestNSubstitutedAminoAcidDetection:
    """Test N-substituted amino acid detection."""

    def test_sarcosine_is_n_substituted(self):
        """Sarcosine (N-methylglycine) should be detected as N-substituted."""
        from rdkit import Chem
        from orthonym.rules.amino_acids import is_n_substituted_amino_acid

        mol = Chem.MolFromSmiles("CNCC(=O)O")
        assert is_n_substituted_amino_acid(mol) is True

    def test_glycine_not_n_substituted(self):
        """Glycine is not N-substituted."""
        from rdkit import Chem
        from orthonym.rules.amino_acids import is_n_substituted_amino_acid

        mol = Chem.MolFromSmiles("NCC(=O)O")
        assert is_n_substituted_amino_acid(mol) is False


class TestIntegrationWithNameCompound:
    """Test that name_compound correctly handles amino acids."""

    def test_amino_acid_before_polyfunctional(self):
        """Amino acid naming should take precedence over polyfunctional."""
        # Glycine would be "2-aminoacetic acid" via polyfunctional
        # but should return "glycine" via trivial name
        assert name_compound("NCC(=O)O") == "glycine"

    def test_amino_acid_through_full_pipeline(self):
        """Multiple amino acids work through the full pipeline."""
        assert name_compound("NCC(=O)O") == "glycine"
        assert name_compound("CC(N)C(=O)O") == "alanine"
        assert name_compound("NC(Cc1ccccc1)C(=O)O") == "phenylalanine"

    def test_non_amino_acid_not_affected(self):
        """Regular compounds still work correctly."""
        # Simple acid
        assert name_compound("CC(=O)O") == "acetic acid"
        # Simple amine. v22 Phase B (DD1 Fix 4 / H5): 'ethylamine' is a
        # general-nomenclature functional-class name; the PIN is the substitutive
        # 'ethanamine' (P-62.2.1.2; ethane locant elided per P-14.3.4.4).
        assert name_compound("CCN") == "ethanamine"


class TestOPSINSimpleGroupAminoAcids:
    """Test OPSIN simpleGroup amino acid integration (141-02)."""

    def test_opsin_simplegroup_amino_acids(self):
        """10 representative simpleGroup entries are lookupable by canonical SMILES."""
        from orthonym.data.amino_acids import get_amino_acid_name
        from rdkit import Chem

        # 10 representative simpleGroup entries from OPSIN_AMINO_ACIDS
        test_cases = [
            ("CC(C)(CO)[C@@H](O)C(=O)NCCC(=O)NCCS", "pantetheine"),
            ("CC(C)(CO)[C@@H](O)C(=O)NCCCO", "pantothenol"),
            ("CC(C)C[C@H](N)[C@@H](O)CC(=O)O", "statine"),
            ("CC(NC(C)C(=O)O)C(=O)O", "alanopine"),
            ("CC(NCC(=O)O)C(=O)O", "strombine"),
            ("CCC(N)C(=O)O", "butyrine"),
            ("CN(CC(=O)O)C(=N)N", "creatine"),
            ("CN[C@@H](Cc1c[nH]c2ccccc12)C(=O)O", "abrine"),
            ("CSCCCN", "methioninamine"),
            ("NCCS(=O)(=O)O", "taurine"),
        ]

        for smiles, expected_name in test_cases:
            can_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)
            result = get_amino_acid_name(can_smiles)
            assert result == expected_name, (
                f"Expected {expected_name!r} for {can_smiles}, got {result!r}"
            )

    def test_existing_amino_acids_preserved(self):
        """All 20 proteinogenic amino acids still return correct names."""
        from orthonym.data.amino_acids import get_amino_acid_name

        proteinogenic = {
            "NCC(=O)O": "glycine",
            "CC(N)C(=O)O": "alanine",
            "CC(C)C(N)C(=O)O": "valine",
            "CC(C)CC(N)C(=O)O": "leucine",
            "CCC(C)C(N)C(=O)O": "isoleucine",
            "NC(CO)C(=O)O": "serine",
            "CC(O)C(N)C(=O)O": "threonine",
            "NC(CS)C(=O)O": "cysteine",
            "CSCC(N)C(=O)O": "methionine",
            "NC(CC(=O)O)C(=O)O": "aspartic acid",
            "NC(CCC(=O)O)C(=O)O": "glutamic acid",
            "NC(CC(N)=O)C(=O)O": "asparagine",
            "NC(CCC(N)=O)C(=O)O": "glutamine",
            "NCCCCC(N)C(=O)O": "lysine",
            "NC(CCCNC(N)=N)C(=O)O": "arginine",
            "NC(Cc1cnc[nH]1)C(=O)O": "histidine",
            "NC(Cc1ccccc1)C(=O)O": "phenylalanine",
            "NC(Cc1ccc(O)cc1)C(=O)O": "tyrosine",
            "NC(Cc1c[nH]c2ccccc12)C(=O)O": "tryptophan",
            "OC(=O)C1CCCN1": "proline",
        }
        for smiles, expected_name in proteinogenic.items():
            result = get_amino_acid_name(smiles)
            assert result == expected_name, (
                f"Expected {expected_name!r} for {smiles}, got {result!r}"
            )

    def test_amino_acid_count_expanded(self):
        """Total amino acid entries should be >= 90 after OPSIN integration."""
        from orthonym.data.amino_acids import STANDARD_AMINO_ACIDS, NON_STANDARD_AMINO_ACIDS

        total = len(STANDARD_AMINO_ACIDS) + len(NON_STANDARD_AMINO_ACIDS)
        assert total >= 90, f"Expected >= 90 amino acid entries, got {total}"

    def test_acyl_names_expanded(self):
        """AMINO_ACID_ACYL_NAMES should have >= 50 entries after expansion."""
        from orthonym.data.amino_acids import AMINO_ACID_ACYL_NAMES

        assert len(AMINO_ACID_ACYL_NAMES) >= 50, (
            f"Expected >= 50 acyl name entries, got {len(AMINO_ACID_ACYL_NAMES)}"
        )

    def test_acyl_name_for_new_entry(self):
        """New amino acids should have acyl name entries."""
        from orthonym.data.amino_acids import get_amino_acid_acyl_name

        # abrine -> abrinyl (standard -ine -> -inyl pattern doesn't apply,
        # but -ine -> -yl should work)
        result = get_amino_acid_acyl_name("abrine")
        assert result is not None, "Expected acyl name for 'abrine'"

    def test_abrine_in_data(self):
        """abrine should be present in expanded amino acid data."""
        from orthonym.data.amino_acids import NON_STANDARD_AMINO_ACIDS

        # Check that 'abrine' appears as a value in NON_STANDARD_AMINO_ACIDS
        assert "abrine" in NON_STANDARD_AMINO_ACIDS.values(), (
            "Expected 'abrine' in NON_STANDARD_AMINO_ACIDS values"
        )


class TestExpandedAminoAcidPipeline:
    """Test expanded amino acids through full naming pipeline (141-02 Task 2)."""

    def test_name_compound_new_amino_acid(self):
        """OPSIN simpleGroup amino acid is reachable via name_compound()."""
        # abrine is a new OPSIN simpleGroup entry
        result = name_compound("CN[C@@H](Cc1c[nH]c2ccccc12)C(=O)O")
        assert result is not None
        assert "abrine" in result, f"Expected 'abrine' in result, got: {result}"

    def test_name_compound_creatine(self):
        """Creatine (non-alpha amino acid) found via SMILES lookup."""
        result = name_compound("CN(CC(=O)O)C(=N)N")
        assert result is not None
        assert "creatine" in result.lower(), f"Expected 'creatine', got: {result}"

    def test_name_compound_taurine(self):
        """Taurine (sulfonic acid amino) found via SMILES lookup."""
        result = name_compound("NCCS(=O)(=O)O")
        assert result is not None
        assert "taurine" in result.lower(), f"Expected 'taurine', got: {result}"

    def test_pin_mode_returns_systematic(self):
        """systematic mode bypasses trivial name lookup for amino acids."""
        # Alanine in systematic mode should NOT return "alanine"
        result = name_compound("CC(N)C(=O)O", style="systematic")
        assert result is not None
        assert "amino" in result.lower(), (
            f"Expected systematic name with 'amino', got: {result}"
        )
        assert result != "alanine", (
            f"systematic mode should not return trivial name 'alanine'"
        )

    def test_pin_mode_new_amino_acid(self):
        """systematic mode produces systematic name for new amino acid entries."""
        # butyrine in systematic mode should return "2-aminobutanoic acid"
        result = name_compound("CCC(N)C(=O)O", style="systematic")
        assert result is not None
        assert result != "butyrine", (
            f"systematic mode should not return trivial name 'butyrine'"
        )
        assert "amino" in result.lower(), (
            f"Expected systematic name with 'amino', got: {result}"
        )

    def test_default_mode_returns_trivial(self):
        """Default (pin) mode returns trivial names for new amino acid entries."""
        result = name_compound("CCC(N)C(=O)O")
        assert result == "butyrine", (
            f"Expected trivial name 'butyrine', got: {result}"
        )

    def test_peptide_acyl_name_expansion(self):
        """Expanded acyl names are available for peptide naming support."""
        from orthonym.data.amino_acids import get_amino_acid_acyl_name

        # New entries should have acyl names
        assert get_amino_acid_acyl_name("creatine") is not None
        assert get_amino_acid_acyl_name("taurine") is not None
        assert get_amino_acid_acyl_name("abrine") is not None
        # Existing entries still work
        assert get_amino_acid_acyl_name("glycine") == "glycyl"
        assert get_amino_acid_acyl_name("proline") == "prolyl"

    def test_all_proteinogenic_via_name_compound(self):
        """All 20 proteinogenic amino acids work through full pipeline."""
        proteinogenic = [
            ("NCC(=O)O", "glycine"),
            ("CC(N)C(=O)O", "alanine"),
            ("CC(C)C(N)C(=O)O", "valine"),
            ("CC(C)CC(N)C(=O)O", "leucine"),
            ("CCC(C)C(N)C(=O)O", "isoleucine"),
            ("NC(CO)C(=O)O", "serine"),
            ("CC(O)C(N)C(=O)O", "threonine"),
            ("NC(CS)C(=O)O", "cysteine"),
            ("CSCC(N)C(=O)O", "methionine"),
            ("NC(CC(=O)O)C(=O)O", "aspartic acid"),
            ("NC(CCC(=O)O)C(=O)O", "glutamic acid"),
            ("NC(CC(N)=O)C(=O)O", "asparagine"),
            ("NC(CCC(N)=O)C(=O)O", "glutamine"),
            ("NCCCCC(N)C(=O)O", "lysine"),
            ("NC(CCCNC(N)=N)C(=O)O", "arginine"),
            ("NC(Cc1cnc[nH]1)C(=O)O", "histidine"),
            ("NC(Cc1ccccc1)C(=O)O", "phenylalanine"),
            ("NC(Cc1ccc(O)cc1)C(=O)O", "tyrosine"),
            ("NC(Cc1c[nH]c2ccccc12)C(=O)O", "tryptophan"),
            ("OC(=O)C1CCCN1", "proline"),
        ]
        for smiles, expected_name in proteinogenic:
            result = name_compound(smiles)
            assert result == expected_name, (
                f"Expected {expected_name!r} for {smiles}, got {result!r}"
            )
