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

    @pytest.mark.opsin_gate
    def test_sarcosine(self):
        """Sarcosine (N-methylglycine) is named systematically.

        ⚠ THIS TEST REQUIRES THE OPSIN GATE, and that is a finding, not a
        formality. With the gate OFF (the suite-wide default) this molecule names
        to '2-aminopropanoic acid' -- which is ALANINE, a DIFFERENT MOLECULE. The
        shipped name is correct only because SELF-01 vetoes that candidate:

            gate OFF -> '2-aminopropanoic acid'    <- a different molecule
            gate ON  -> '(methylamino)acetic acid' <- SELF-01 suppressed the above

        That is the same shape as the trap documented in tests/conftest.py:313, and
        an instance of the measured class where the 0-wrong margin is the GATE
        rather than the producers. Gating sarcosine did not create the bad
        candidate; it removed the trivial-name short-circuit that used to hide it
        (the contributor guide invariant 9 -- removing a wrong output can unmask a worse
        generator). Asserting the gate-off value here would encode a wrong molecule
        as the expectation, so the marker is the correct resolution and the
        underlying producer defect is recorded for its own task.

        v29 Task E2: was ``== "sarcosine"``. 'sarcosine' occurs NOWHERE in the Blue
        Book (grep validated against known positives -- 'glycine' 26 hits,
        'norvaline'/'norleucine' found), and it is absent from both retained tables,
        10.4 and 10.5 (BlueBookV2.md:54186-:54245). P-103.1.1.3 "Systematic
        substitutive names" (:54247), sentence :54251, governs: "When not denoted by
        a retained name, amino acids receive systematic substitutive names
        constructed by applying the principles, rules and conventions of
        substitutive nomenclature." The Blue Book gives the precedent at :54253 --
        norvaline and norleucine take systematic names and "The names 'norvaline'
        and 'norleucine' are not recommended."

        On the UNLOCANTED form: acetic acid is a retained PIN parent that may be
        substituted -- P-65.1.1.1 "Retained names as preferred IUPAC names"
        (:29715): "Only the following five carboxylic acids retained names and are
        also preferred IUPAC names. All can be functionalized, but only acetic acid,
        benzoic acid, and oxamic acid can be substituted according to P-15.1.8.2.1".
        The same rule prints its own substituted-acetic-acid example WITHOUT
        locants: "H2N-CO-COOH  oxamic acid (PIN)  amino(oxo)acetic acid".
        """
        result = name_compound("CNCC(=O)O")
        assert result == "(methylamino)acetic acid"

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
        """simpleGroup entries are lookupable by canonical SMILES.

        v29 Task E2: 3 of the original 10 rows (statine, abrine, taurine) are now
        adjudicated non-PINs and are withheld from this PIN-path lookup, so they
        moved to the companion assertion below. The 7 that remain still prove the
        simpleGroup integration is reached, which is what this test exists for.

        Why those 3 are withheld -- P-103.1.1.3 "Systematic substitutive names"
        (BlueBookV2.md:54247), decisive sentence :54251: "When not denoted by a
        retained name, amino acids receive systematic substitutive names constructed
        by applying the principles, rules and conventions of substitutive
        nomenclature." The retained sets are Tables 10.4/10.5 (:54186-:54245) and
        none of the three is in them. The Blue Book states the precedent itself at
        :54253: norvaline and norleucine are likewise non-retained, take systematic
        names, and "The names 'norvaline' and 'norleucine' are not recommended."
        """
        from orthonym.data.amino_acids import get_amino_acid_name
        from rdkit import Chem

        # simpleGroup entries from OPSIN_AMINO_ACIDS that remain PIN-path names
        test_cases = [
            ("CC(C)(CO)[C@@H](O)C(=O)NCCC(=O)NCCS", "pantetheine"),
            ("CC(C)(CO)[C@@H](O)C(=O)NCCCO", "pantothenol"),
            ("CC(NC(C)C(=O)O)C(=O)O", "alanopine"),
            ("CC(NCC(=O)O)C(=O)O", "strombine"),
            ("CCC(N)C(=O)O", "butyrine"),
            ("CN(CC(=O)O)C(=N)N", "creatine"),
            ("CSCCCN", "methioninamine"),
        ]

        for smiles, expected_name in test_cases:
            can_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)
            result = get_amino_acid_name(can_smiles)
            assert result == expected_name, (
                f"Expected {expected_name!r} for {can_smiles}, got {result!r}"
            )

    def test_gated_simplegroup_amino_acids_are_demoted_not_deleted(self):
        """The 3 withheld rows leave the PIN lookup but stay in the general dict.

        This is the demote-not-delete contract every existing deny row already
        follows; it is asserted here so a future deletion cannot pass silently.
        """
        from orthonym.data.amino_acids import (
            GENERAL_ONLY_AMINO_ACIDS, get_amino_acid_name,
        )
        from rdkit import Chem

        for smiles, name in [
            ("CC(C)C[C@H](N)[C@@H](O)CC(=O)O", "statine"),
            ("CN[C@@H](Cc1c[nH]c2ccccc12)C(=O)O", "abrine"),
            ("NCCS(=O)(=O)O", "taurine"),
        ]:
            can = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)
            assert get_amino_acid_name(can) is None, f"{name!r} still on the PIN path"
            assert name in set(GENERAL_ONLY_AMINO_ACIDS.values()), (
                f"{name!r} was deleted rather than demoted"
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
        """abrine should still be present in the expanded amino acid data.

        v29 Task E2: abrine is an adjudicated non-PIN (P-103.1.1.3,
        BlueBookV2.md:54247/:54251 -- amino acids not in the retained Tables
        10.4/10.5 take systematic substitutive names), so it moved OUT of
        NON_STANDARD_AMINO_ACIDS and INTO GENERAL_ONLY_AMINO_ACIDS. The point of
        this test -- that the row was imported and is not lost -- is preserved by
        asserting the demotion target instead.
        """
        from orthonym.data.amino_acids import GENERAL_ONLY_AMINO_ACIDS

        assert "abrine" in GENERAL_ONLY_AMINO_ACIDS.values(), (
            "Expected 'abrine' in GENERAL_ONLY_AMINO_ACIDS values"
        )


class TestExpandedAminoAcidPipeline:
    """Test expanded amino acids through full naming pipeline (141-02 Task 2)."""

    def test_name_compound_new_amino_acid(self):
        """OPSIN simpleGroup amino acid is reachable via name_compound().

        v29 Task E2: was ``assert "abrine" in result``. abrine is an adjudicated
        non-PIN under P-103.1.1.3 (BlueBookV2.md:54247), decisive sentence :54251 --
        "When not denoted by a retained name, amino acids receive systematic
        substitutive names constructed by applying the principles, rules and
        conventions of substitutive nomenclature." It is in neither retained Table
        10.4 nor 10.5. The molecule is still named, and the systematic name RECOVERS
        the (2S) stereodescriptor that the trivial name silently dropped.
        """
        result = name_compound("CN[C@@H](Cc1c[nH]c2ccccc12)C(=O)O")
        assert result == "(2S)-3-(1H-indol-3-yl)-2-(methylamino)propanoic acid"

    def test_name_compound_creatine(self):
        """Creatine (non-alpha amino acid) found via SMILES lookup."""
        result = name_compound("CN(CC(=O)O)C(=N)N")
        assert result is not None
        assert "creatine" in result.lower(), f"Expected 'creatine', got: {result}"

    def test_name_compound_taurine(self):
        """Taurine (sulfonic acid amino) is named systematically.

        v29 Task E2: was ``assert "taurine" in result.lower()``. 'taurine' does not
        occur anywhere in the Blue Book, and P-103.1.1.3 (BlueBookV2.md:54247),
        sentence :54251, sends amino acids that are not in the retained Tables
        10.4/10.5 to systematic substitutive names. The molecule is still named --
        this is a demotion, not a coverage loss.
        """
        result = name_compound("NCCS(=O)(=O)O")
        assert result == "2-aminoethane-1-sulfonic acid"

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
