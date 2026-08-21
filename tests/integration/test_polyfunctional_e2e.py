"""
Phase 4 Integration Tests - Polyfunctional Compounds

Verifies all POLY requirements and Phase 4 success criteria.
"""
import pytest
from orthonym import name_compound


class TestPOLY01_MultipleFunctionalGroups:
    """POLY-01: Name compounds with multiple FGs using seniority."""

    def test_hydroxy_acid_seniority(self):
        """Acid > alcohol, so acid is suffix, alcohol is prefix."""
        result = name_compound("OCC(=O)O")
        assert "hydroxy" in result and "acid" in result

    def test_keto_acid_seniority(self):
        """Acid > ketone, so ketone becomes 'oxo' prefix."""
        result = name_compound("CC(=O)C(=O)O")
        assert "oxo" in result and "acid" in result

    def test_amino_acid_seniority(self):
        """Acid > amine. Returns glycine (trivial) or 2-aminoacetic acid."""
        result = name_compound("NCC(=O)O")
        # Returns glycine (trivial) or 2-aminoacetic acid (systematic)
        assert result in ["glycine", "2-aminoacetic acid"]

    def test_cyano_acid_seniority(self):
        """Acid > nitrile, so nitrile becomes 'cyano' prefix."""
        result = name_compound("N#CCC(=O)O")
        assert "cyano" in result and "acid" in result


class TestPOLY02_Esters:
    """POLY-02: Name esters."""

    def test_methyl_acetate(self):
        """Simple ester: methyl acetate."""
        result = name_compound("COC(C)=O")
        assert "methyl" in result and ("acetate" in result or "ethanoate" in result)

    def test_ethyl_acetate(self):
        """Ethyl acetate - common solvent."""
        result = name_compound("CCOC(C)=O")
        assert "ethyl" in result and ("acetate" in result or "ethanoate" in result)

    def test_methyl_propanoate(self):
        """3-carbon acid ester."""
        result = name_compound("COC(=O)CC")
        assert result == "methyl propanoate"

    def test_methyl_acetate_alternate_smiles(self):
        """Same ester, different SMILES notation."""
        result = name_compound("CC(=O)OC")
        assert "methyl" in result and ("acetate" in result or "ethanoate" in result)


class TestPOLY03_Amides:
    """POLY-03: Name amides."""

    def test_acetamide(self):
        """Primary amide from acetic acid."""
        result = name_compound("CC(=O)N")
        assert result == "acetamide"

    def test_n_methylacetamide(self):
        """Secondary amide: N-methylacetamide."""
        result = name_compound("CNC(C)=O")
        assert result == "N-methylacetamide"

    def test_nn_dimethylformamide(self):
        """DMF - tertiary amide."""
        result = name_compound("CN(C)C=O")
        assert "dimethylformamide" in result.lower()

    def test_propanamide(self):
        """Primary amide from propanoic acid."""
        result = name_compound("CCC(=O)N")
        assert result == "propanamide"


class TestPOLY04_Nitriles:
    """POLY-04: Name nitriles."""

    def test_acetonitrile(self):
        """2-carbon nitrile (common solvent)."""
        result = name_compound("CC#N")
        assert result == "acetonitrile"

    def test_propanenitrile(self):
        """3-carbon chain nitrile."""
        result = name_compound("CCC#N")
        assert result == "propanenitrile"

    def test_cyclohexanecarbonitrile(self):
        """Ring-attached nitrile uses -carbonitrile suffix."""
        result = name_compound("C1CCCCC1C#N")
        assert result == "cyclohexanecarbonitrile"

    def test_butanenitrile(self):
        """4-carbon chain nitrile."""
        result = name_compound("CCCC#N")
        assert result == "butanenitrile"


class TestPOLY05_Ethers:
    """POLY-05: Name ethers (substitutive naming)."""

    def test_methoxyethane(self):
        """Simple ether: methyl group as methoxy."""
        result = name_compound("COCC")
        assert result == "methoxyethane"

    def test_ethoxyethane(self):
        """Symmetric ether named as ethoxy."""
        result = name_compound("CCOCC")
        assert result == "ethoxyethane"

    def test_methoxypropanoic_acid(self):
        """Ether + acid: acid is principal, ether is prefix."""
        result = name_compound("COCCC(=O)O")
        assert "methoxy" in result and "acid" in result


class TestPOLY06_HydroxyAcids:
    """POLY-06: Name hydroxy acids."""

    def test_2_hydroxyacetic_acid(self):
        """Glycolic acid - systematic name (ethanoic is also valid IUPAC)."""
        result = name_compound("OCC(=O)O")
        # Both "2-hydroxyacetic acid" and "2-hydroxyethanoic acid" are valid IUPAC
        assert result in ["2-hydroxyacetic acid", "2-hydroxyethanoic acid"]

    def test_3_hydroxypropanoic_acid(self):
        """Hydroxyl at position 3."""
        result = name_compound("OCCC(=O)O")
        assert result == "3-hydroxypropanoic acid"

    def test_2_hydroxypropanoic_acid(self):
        """Lactic acid - systematic name."""
        result = name_compound("CC(O)C(=O)O")
        assert result == "2-hydroxypropanoic acid"


class TestPOLY07_AminoAcids:
    """POLY-07: Name amino acids."""

    def test_glycine(self):
        """Simplest amino acid."""
        assert name_compound("NCC(=O)O") == "glycine"

    def test_alanine(self):
        """3-carbon amino acid, UNDEFINED stereo -> systematic.

        v33 Phase-1 stereo honesty: bare 'alanine' implies L, so an
        undefined-stereo input declines it (P-101.2.6/P-103.1.3.1)."""
        assert name_compound("CC(N)C(=O)O") == "2-aminopropanoic acid"

    def test_phenylalanine(self):
        """Aromatic amino acid, UNDEFINED stereo -> systematic (same rule)."""
        assert (name_compound("NC(Cc1ccccc1)C(=O)O")
                == "2-amino-3-phenylpropanoic acid")

    def test_valine(self):
        """Branched aliphatic amino acid, UNDEFINED stereo -> systematic."""
        assert name_compound("CC(C)C(N)C(=O)O") == "2-amino-3-methylbutanoic acid"


class TestPOLY08_KetoAcids:
    """POLY-08: Name keto acids."""

    def test_2_oxopropanoic_acid(self):
        """Pyruvic acid - systematic name."""
        result = name_compound("CC(=O)C(=O)O")
        assert result == "2-oxopropanoic acid"

    def test_3_oxobutanoic_acid(self):
        """Acetoacetic acid - systematic name."""
        result = name_compound("CC(=O)CC(=O)O")
        assert result == "3-oxobutanoic acid"


class TestPhase4SuccessCriteria:
    """Verify Phase 4 success criteria from roadmap."""

    def test_criterion_1_hydroxy_acid(self):
        """User can input 'OCC(=O)O' and receive a hydroxy acid name."""
        result = name_compound("OCC(=O)O")
        # Both "2-hydroxyacetic acid" and "2-hydroxyethanoic acid" are valid
        assert result in ["2-hydroxyacetic acid", "2-hydroxyethanoic acid"]

    def test_criterion_2_ester(self):
        """User can input 'CC(=O)OC' and receive 'methyl acetate'."""
        result = name_compound("CC(=O)OC")
        assert "methyl" in result and ("acetate" in result or "ethanoate" in result)

    def test_criterion_3_amide(self):
        """User can input 'CC(=O)N' and receive 'acetamide'."""
        assert name_compound("CC(=O)N") == "acetamide"

    def test_criterion_4_amino_acid(self):
        """User can input 'NCC(=O)O' and receive 'glycine' or '2-aminoacetic acid'."""
        result = name_compound("NCC(=O)O")
        assert result in ["glycine", "2-aminoacetic acid"]

    def test_criterion_5_principal_group(self):
        """Principal group is always highest-seniority when multiple present."""
        # In hydroxy acid, acid is principal (higher than alcohol)
        result = name_compound("OCC(=O)O")
        assert "acid" in result  # Suffix is acid
        assert "hydroxy" in result  # Alcohol is prefix


class TestPolyfunctionalEdgeCases:
    """Test edge cases in polyfunctional naming."""

    def test_multiple_same_fg(self):
        """Diol - IUPAC 2013 prefers retained name 'ethylene glycol'.

        'ethylene glycol' is the IUPAC 2013 preferred retained name.
        'ethane-1,2-diol' is the systematic equivalent.
        """
        result = name_compound("OCCO")
        # IUPAC 2013: retained name "ethylene glycol" is preferred
        # Systematic "ethane-1,2-diol" contains "diol"
        assert result == "ethylene glycol" or "diol" in result or "hydroxy" in result

    def test_three_different_fgs(self):
        """Hydroxy + amino + acid: this is serine (amino acid trivial name)."""
        result = name_compound("OCC(N)C(=O)O")
        # OCC(N)C(=O)O is serine - matches amino acid lookup
        # Either "serine" (trivial) or systematic with "acid" suffix
        assert result == "serine" or "acid" in result

    def test_ether_never_principal(self):
        """Ether + alcohol: alcohol is principal, ether is prefix."""
        result = name_compound("COCCO")
        # 2-methoxyethanol or similar
        assert "methoxy" in result


class TestNoRegressions:
    """Verify previous phases still work."""

    def test_simple_alkane(self):
        """Phase 1: Basic alkane naming."""
        assert name_compound("CCCC") == "butane"

    def test_simple_alcohol(self):
        """Phase 1: Functional group naming."""
        assert name_compound("CCCO") == "propan-1-ol"

    def test_benzene(self):
        """Phase 2: Aromatic ring naming."""
        assert name_compound("c1ccccc1") == "benzene"

    def test_pyridine(self):
        """Phase 3: Heterocycle naming."""
        assert name_compound("c1ccncc1") == "pyridine"

    def test_cyclohexane(self):
        """Phase 2: Cycloalkane naming."""
        assert name_compound("C1CCCCC1") == "cyclohexane"

    def test_methylcyclohexane(self):
        """Phase 2: Substituted cycloalkane."""
        result = name_compound("CC1CCCCC1")
        assert result == "methylcyclohexane"

    def test_toluene(self):
        """Phase 2: Substituted benzene (retained)."""
        assert name_compound("Cc1ccccc1") == "toluene"

    def test_furan(self):
        """Phase 3: Aromatic heterocycle."""
        assert name_compound("c1ccoc1") == "furan"

    def test_morpholine(self):
        """Phase 3: Saturated heterocycle."""
        assert name_compound("C1COCCN1") == "morpholine"


class TestPolyfunctionalNamingVariations:
    """Test various polyfunctional naming patterns."""

    def test_amino_ketone(self):
        """Amino + ketone: ketone is principal."""
        result = name_compound("NCC(=O)C")
        # Should have 'amino' prefix and 'one' suffix
        assert "amino" in result

    def test_cyano_alcohol(self):
        """Cyano + alcohol: nitrile is higher seniority than alcohol."""
        result = name_compound("N#CCCO")
        # Nitrile > alcohol in IUPAC seniority, so nitrile is suffix
        # Result: 3-hydroxypropanenitrile (hydroxy prefix, nitrile suffix)
        assert "hydroxy" in result and "nitrile" in result

    def test_dicarboxylic_acid(self):
        """Two carboxylic acids: dioic acid suffix."""
        result = name_compound("OC(=O)CC(=O)O")
        # Should be propanedioic acid (malonic acid is retained)
        assert "acid" in result


class TestRetainedNamesPriority:
    """Test that retained names take precedence."""

    def test_glycine_over_aminoacetic(self):
        """Glycine is preferred over 2-aminoacetic acid."""
        assert name_compound("NCC(=O)O") == "glycine"

    def test_alanine_over_aminopropanoic(self):
        """UNDEFINED-stereo alanine names systematically.

        The pre-v33 assertion (retained 'alanine' preferred) is inverted by the
        Phase-1 stereo-honesty change: bare 'alanine' implies L, so an
        undefined-stereo input must decline it and use the systematic name."""
        assert name_compound("CC(N)C(=O)O") == "2-aminopropanoic acid"

    def test_acetamide_over_ethanamide(self):
        """Acetamide is a retained name."""
        assert name_compound("CC(=O)N") == "acetamide"

    def test_acetonitrile_retained(self):
        """Acetonitrile is a retained name."""
        assert name_compound("CC#N") == "acetonitrile"
