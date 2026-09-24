"""
a phase: Data Table Completeness - Comprehensive Test Suite.

Tests all 25 new data entries added in a phase:
- 4 simple retained names (biphenyl, acetylene, anisole, caprolactam)
- 8 nucleosides (adenosine, guanosine, cytidine, thymidine, uridine + 3 deoxy forms)
- 5 disaccharides (sucrose, maltose, lactose, cellobiose, trehalose)
- 1 sialic acid (N-acetylneuraminic acid)
- 6 amino sugars (glucosamine, galactosamine, mannosamine alpha/beta pairs)
- 1 bicyclo entry (decalin)

Includes OPSIN round-trip validation where supported.
"""

import os
import shutil
import subprocess

import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.data.retained_names import RETAINED_NAMES, get_retained_name
from orthonym.data.sugar_names import (
    ALL_SUGAR_NAMES,
    AMINO_SUGAR_NAMES,
    lookup_sugar,
)
from orthonym.data.bicyclo_systems import BICYCLO_RETAINED_NAMES
from tests.support.jars import jar_or_none

# ---------------------------------------------------------------------------
# OPSIN setup
# ---------------------------------------------------------------------------

JAVA_AVAILABLE = shutil.which("java") is not None
OPSIN_JAR = jar_or_none()
OPSIN_AVAILABLE = JAVA_AVAILABLE and OPSIN_JAR is not None


def _opsin_parse(name: str) -> str:
    """Parse IUPAC name to SMILES using OPSIN CLI."""
    try:
        result = subprocess.run(
            ["java", "-jar", OPSIN_JAR, "-osmi"],
            input=name,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


# ============================================================================
# Test 1: Simple Retained Names (4 entries)
# ============================================================================

class TestSimpleRetainedNames:
    """Tests for biphenyl, acetylene, anisole, caprolactam."""

    @pytest.mark.unit
    @pytest.mark.parametrize("name", ["biphenyl", "acetylene", "anisole", "caprolactam"])
    def test_name_in_dict(self, name):
        """Each simple retained name should be in RETAINED_NAMES values."""
        assert name in RETAINED_NAMES.values(), f"'{name}' not in RETAINED_NAMES"

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,expected", [
        ("c1ccc(-c2ccccc2)cc1", "biphenyl"),
        ("C#C", "acetylene"),
        ("COc1ccccc1", "anisole"),
        ("O=C1CCCCCN1", "caprolactam"),
    ], ids=["biphenyl", "acetylene", "anisole", "caprolactam"])
    def test_canonical_smiles_key(self, smiles, expected):
        """Canonical SMILES key should map to the correct retained name."""
        canon = Chem.CanonSmiles(smiles)
        assert canon in RETAINED_NAMES, f"Canonical SMILES '{canon}' not in RETAINED_NAMES"
        assert RETAINED_NAMES[canon] == expected

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,expected", [
        # F-T9/DD6: 'biphenyl'/'anisole' are general-only; the PIN headline is
        # the systematic 1,1'-biphenyl / methoxybenzene. acetylene stays (hc_override).
        ("c1ccc(-c2ccccc2)cc1", "1,1'-biphenyl"),
        ("C#C", "acetylene"),
        ("COc1ccccc1", "anisole"),  # a review RISK 7: bare anisole IS the PIN (the Blue Book / the Blue Book)
        #: same treatment as biphenyl/anisole above -- 'caprolactam'
        # is general-only (0 the Blue Book hits) and the PIN headline is the
        # systematic pseudoketone, (the Blue Book), printed `azepan-2-one
        # (PIN)` at the Blue Book. The raw hand-curated dict still carries the trivial
        # name, which is why the two tests above are unchanged.
        ("O=C1CCCCCN1", "azepan-2-one"),
    ], ids=["biphenyl-e2e", "acetylene-e2e", "anisole-e2e", "caprolactam-e2e"])
    def test_name_compound_e2e(self, smiles, expected):
        """name_compound should return the PIN headline (retained PIN or systematic)."""
        result = name_compound(smiles)
        assert result == expected, f"Expected '{expected}', got '{result}'"


# ============================================================================
# Test 2: Nucleoside Retained Names (8 entries)
# ============================================================================

NUCLEOSIDE_ENTRIES = [
    ("Nc1ncnc2c1ncn2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O", "adenosine"),
    ("Nc1nc2c(ncn2[C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=O)[nH]1", "guanosine"),
    ("Nc1ccn([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=O)n1", "cytidine"),
    ("Cc1cn([C@H]2C[C@H](O)[C@@H](CO)O2)c(=O)[nH]c1=O", "thymidine"),
    ("O=c1ccn([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=O)[nH]1", "uridine"),
    # a phase: renamed to the 2'-deoxy PIN (the prime is required).
    ("Nc1ncnc2c1ncn2[C@H]1C[C@H](O)[C@@H](CO)O1", "2'-deoxyadenosine"),
    ("Nc1nc2c(ncn2[C@H]2C[C@H](O)[C@@H](CO)O2)c(=O)[nH]1", "2'-deoxyguanosine"),
    ("Nc1ccn([C@H]2C[C@H](O)[C@@H](CO)O2)c(=O)n1", "2'-deoxycytidine"),
    ("O=c1ccn([C@H]2C[C@H](O)[C@@H](CO)O2)c(=O)[nH]1", "2'-deoxyuridine"),
]


class TestNucleosideRetainedNames:
    """Tests for 8 nucleoside entries in RETAINED_NAMES."""

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,expected", NUCLEOSIDE_ENTRIES,
                             ids=[e[1] for e in NUCLEOSIDE_ENTRIES])
    def test_nucleoside_in_dict(self, smiles, expected):
        """Nucleoside should be in RETAINED_NAMES with correct canonical key."""
        canon = Chem.CanonSmiles(smiles)
        assert canon == smiles, f"SMILES not canonical: {smiles} -> {canon}"
        assert canon in RETAINED_NAMES
        assert RETAINED_NAMES[canon] == expected

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,expected", NUCLEOSIDE_ENTRIES,
                             ids=[f"{e[1]}-e2e" for e in NUCLEOSIDE_ENTRIES])
    def test_nucleoside_e2e(self, smiles, expected):
        """name_compound should return the retained name for nucleosides."""
        result = name_compound(smiles)
        assert result == expected, f"Expected '{expected}', got '{result}'"


# ============================================================================
# Test 3: Disaccharide Retained Names (5 entries)
# ============================================================================

DISACCHARIDE_ENTRIES = [
    ("OC[C@@H]1O[C@@](CO)(O[C@H]2[C@H](O)[C@@H](O)[C@@H](O)O[C@@H]2CO)[C@@H](O)[C@H]1O", "sucrose"),
    ("OC[C@H]1O[C@@H](O[C@H]2[C@H](O)[C@@H](O)C(O)O[C@@H]2CO)[C@H](O)[C@@H](O)[C@@H]1O", "maltose"),
    ("OC[C@H]1O[C@H](O[C@H]2[C@H](O)[C@@H](O)C(O)O[C@@H]2CO)[C@H](O)[C@@H](O)[C@H]1O", "lactose"),
    ("OC[C@H]1O[C@@H](O[C@@H]2[C@@H](O)[C@H](O)[C@@H](O)O[C@@H]2CO)[C@H](O)[C@@H](O)[C@@H]1O", "cellobiose"),
    ("OC[C@H]1O[C@H](O[C@H]2O[C@H](CO)[C@H](O)[C@H](O)[C@H]2O)[C@H](O)[C@H](O)[C@H]1O", "trehalose"),
]


class TestDisaccharideRetainedNames:
    """Tests for 5 disaccharide entries in RETAINED_NAMES."""

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,expected", DISACCHARIDE_ENTRIES,
                             ids=[e[1] for e in DISACCHARIDE_ENTRIES])
    def test_disaccharide_in_dict(self, smiles, expected):
        """Disaccharide should be in RETAINED_NAMES with correct canonical key."""
        canon = Chem.CanonSmiles(smiles)
        assert canon == smiles, f"SMILES not canonical: {smiles} -> {canon}"
        assert canon in RETAINED_NAMES
        assert RETAINED_NAMES[canon] == expected

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,expected", DISACCHARIDE_ENTRIES,
                             ids=[f"{e[1]}-canonical" for e in DISACCHARIDE_ENTRIES])
    def test_disaccharide_canonical_key(self, smiles, expected):
        """Disaccharide SMILES key must be RDKit canonical."""
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, f"Invalid SMILES: {smiles}"
        canon = Chem.MolToSmiles(mol)
        assert canon == smiles, f"Key not canonical: {smiles} -> {canon}"


# ============================================================================
# Test 4: Sialic Acid (N-acetylneuraminic acid)
# ============================================================================

class TestSialicAcidRetainedName:
    """Tests for N-acetylneuraminic acid in RETAINED_NAMES."""

    SIALIC_SMILES = "CC(=O)N[C@H]1[C@H]([C@H](O)[C@H](O)CO)OC(O)(C(=O)O)C[C@@H]1O"

    @pytest.mark.unit
    def test_sialic_acid_in_dict(self):
        """N-acetylneuraminic acid should be in RETAINED_NAMES."""
        assert "N-acetylneuraminic acid" in RETAINED_NAMES.values()

    @pytest.mark.unit
    def test_sialic_acid_canonical_key(self):
        """Sialic acid SMILES key must be RDKit canonical."""
        canon = Chem.CanonSmiles(self.SIALIC_SMILES)
        assert canon == self.SIALIC_SMILES
        assert canon in RETAINED_NAMES
        assert RETAINED_NAMES[canon] == "N-acetylneuraminic acid"

    @pytest.mark.unit
    def test_sialic_acid_e2e(self):
        """name_compound should return N-acetylneuraminic acid."""
        result = name_compound(self.SIALIC_SMILES)
        assert result == "N-acetylneuraminic acid"


# ============================================================================
# Test 5: Amino Sugar Names (6 entries)
# ============================================================================

AMINO_SUGAR_ENTRIES = [
    ("N[C@@H]1[C@@H](O)[C@H](O)[C@@H](CO)O[C@@H]1O", "α", "D", "glucosamine"),
    ("N[C@@H]1[C@@H](O)[C@H](O)[C@@H](CO)O[C@H]1O", "β", "D", "glucosamine"),
    ("N[C@@H]1[C@@H](O)[C@@H](O)[C@@H](CO)O[C@@H]1O", "α", "D", "galactosamine"),
    ("N[C@@H]1[C@@H](O)[C@@H](O)[C@@H](CO)O[C@H]1O", "β", "D", "galactosamine"),
    ("N[C@H]1[C@@H](O)[C@H](O)[C@@H](CO)O[C@@H]1O", "α", "D", "mannosamine"),
    ("N[C@H]1[C@@H](O)[C@H](O)[C@@H](CO)O[C@H]1O", "β", "D", "mannosamine"),
]


class TestAminoSugarNames:
    """Tests for 6 amino sugar entries in AMINO_SUGAR_NAMES."""

    @pytest.mark.unit
    def test_amino_sugar_count(self):
        """AMINO_SUGAR_NAMES should contain exactly 6 entries."""
        assert len(AMINO_SUGAR_NAMES) == 6

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,anomer,config,name", AMINO_SUGAR_ENTRIES,
                             ids=[f"{e[1]}-{e[3]}" for e in AMINO_SUGAR_ENTRIES])
    def test_amino_sugar_in_dict(self, smiles, anomer, config, name):
        """Amino sugar should be in AMINO_SUGAR_NAMES with correct tuple."""
        canon = Chem.CanonSmiles(smiles)
        assert canon == smiles, f"SMILES not canonical: {smiles} -> {canon}"
        assert smiles in AMINO_SUGAR_NAMES
        assert AMINO_SUGAR_NAMES[smiles] == (anomer, config, name)

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,anomer,config,name", AMINO_SUGAR_ENTRIES,
                             ids=[f"{e[1]}-{e[3]}-all" for e in AMINO_SUGAR_ENTRIES])
    def test_amino_sugar_in_all_sugar_names(self, smiles, anomer, config, name):
        """Amino sugars should be merged into ALL_SUGAR_NAMES."""
        assert smiles in ALL_SUGAR_NAMES
        assert ALL_SUGAR_NAMES[smiles] == (anomer, config, name)

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,anomer,config,name", AMINO_SUGAR_ENTRIES,
                             ids=[f"{e[1]}-{e[3]}-lookup" for e in AMINO_SUGAR_ENTRIES])
    def test_amino_sugar_lookup(self, smiles, anomer, config, name):
        """lookup_sugar should return correct tuple for amino sugars."""
        result = lookup_sugar(smiles)
        assert result is not None, f"lookup_sugar returned None for {smiles}"
        assert result == (anomer, config, name)


# ============================================================================
# Test 6: Decalin in Bicyclo Systems
# ============================================================================

class TestDecalinBicyclo:
    """Tests for decalin entry in BICYCLO_RETAINED_NAMES."""

    DECALIN_SMILES = "C1CCC2CCCCC2C1"

    @pytest.mark.unit
    def test_decalin_in_dict(self):
        """Decalin should be in BICYCLO_RETAINED_NAMES."""
        assert "decalin" in BICYCLO_RETAINED_NAMES.values()

    @pytest.mark.unit
    def test_decalin_canonical_key(self):
        """Decalin SMILES key must be RDKit canonical."""
        canon = Chem.CanonSmiles(self.DECALIN_SMILES)
        assert canon == self.DECALIN_SMILES
        assert canon in BICYCLO_RETAINED_NAMES
        assert BICYCLO_RETAINED_NAMES[canon] == "decalin"


# ============================================================================
# Test 7: OPSIN Round-Trip Validation
# ============================================================================

@pytest.mark.skipif(
    not OPSIN_AVAILABLE,
    reason="Java or OPSIN JAR not available for round-trip tests",
)
class TestOPSINRoundTrip:
    """OPSIN round-trip tests for new data entries."""

    @pytest.mark.integration
    @pytest.mark.parametrize("name", ["biphenyl", "acetylene", "anisole", "caprolactam"],
                             ids=["opsin-biphenyl", "opsin-acetylene", "opsin-anisole", "opsin-caprolactam"])
    def test_simple_retained_opsin_parse(self, name):
        """OPSIN should parse simple retained names to valid SMILES."""
        opsin_smi = _opsin_parse(name)
        assert opsin_smi, f"OPSIN could not parse '{name}'"
        mol = Chem.MolFromSmiles(opsin_smi)
        assert mol is not None, f"OPSIN produced invalid SMILES for '{name}': {opsin_smi}"

    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,name", NUCLEOSIDE_ENTRIES,
                             ids=[f"opsin-{e[1]}" for e in NUCLEOSIDE_ENTRIES])
    def test_nucleoside_opsin_parse(self, smiles, name):
        """OPSIN should parse nucleoside names to valid SMILES."""
        opsin_smi = _opsin_parse(name)
        assert opsin_smi, f"OPSIN could not parse '{name}'"
        mol = Chem.MolFromSmiles(opsin_smi)
        assert mol is not None, f"OPSIN produced invalid SMILES for '{name}': {opsin_smi}"

    @pytest.mark.integration
    @pytest.mark.parametrize("name", [
        pytest.param("sucrose", marks=pytest.mark.xfail(
            reason="OPSIN produces incorrect structure for sucrose (6+6 instead of 5+6)")),
        pytest.param("maltose", marks=pytest.mark.xfail(
            reason="OPSIN cannot parse 'maltose'")),
        "lactose",
        pytest.param("cellobiose", marks=pytest.mark.xfail(
            reason="OPSIN cannot parse 'cellobiose'")),
        pytest.param("trehalose", marks=pytest.mark.xfail(
            reason="OPSIN cannot parse 'trehalose'")),
    ], ids=["opsin-sucrose", "opsin-maltose", "opsin-lactose", "opsin-cellobiose", "opsin-trehalose"])
    def test_disaccharide_opsin_parse(self, name):
        """OPSIN disaccharide parsing (most are expected to fail)."""
        opsin_smi = _opsin_parse(name)
        assert opsin_smi, f"OPSIN could not parse '{name}'"
        mol = Chem.MolFromSmiles(opsin_smi)
        assert mol is not None

    @pytest.mark.integration
    @pytest.mark.parametrize("amino_sugar_name", [
        "glucosamine", "galactosamine", "mannosamine",
    ], ids=["opsin-glucosamine", "opsin-galactosamine", "opsin-mannosamine"])
    def test_amino_sugar_opsin_parse(self, amino_sugar_name):
        """OPSIN should parse amino sugar names."""
        opsin_smi = _opsin_parse(amino_sugar_name)
        assert opsin_smi, f"OPSIN could not parse '{amino_sugar_name}'"
        mol = Chem.MolFromSmiles(opsin_smi)
        assert mol is not None

    @pytest.mark.integration
    def test_sialic_acid_opsin_parse(self):
        """OPSIN parsing of N-acetylneuraminic acid."""
        opsin_smi = _opsin_parse("N-acetylneuraminic acid")
        assert opsin_smi, "OPSIN could not parse 'N-acetylneuraminic acid'"

    @pytest.mark.integration
    def test_decalin_opsin_parse(self):
        """OPSIN should parse 'decalin' to valid SMILES."""
        opsin_smi = _opsin_parse("decalin")
        assert opsin_smi, "OPSIN could not parse 'decalin'"
        mol = Chem.MolFromSmiles(opsin_smi)
        assert mol is not None


# ============================================================================
# Test 8: SMILES Validity (all new entries produce valid mol objects)
# ============================================================================

ALL_NEW_SMILES = (
    # Simple retained names
    [e[0] for e in [
        ("c1ccc(-c2ccccc2)cc1", "biphenyl"),
        ("C#C", "acetylene"),
        ("COc1ccccc1", "anisole"),
        ("O=C1CCCCCN1", "caprolactam"),
    ]]
    # Nucleosides
    + [e[0] for e in NUCLEOSIDE_ENTRIES]
    # Disaccharides
    + [e[0] for e in DISACCHARIDE_ENTRIES]
    # Sialic acid
    + ["CC(=O)N[C@H]1[C@H]([C@H](O)[C@H](O)CO)OC(O)(C(=O)O)C[C@@H]1O"]
    # Amino sugars
    + [e[0] for e in AMINO_SUGAR_ENTRIES]
    # Decalin
    + ["C1CCC2CCCCC2C1"]
)


class TestSMILESValidity:
    """Verify all new SMILES keys produce valid RDKit mol objects."""

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles", ALL_NEW_SMILES,
                             ids=[f"valid-{i}" for i in range(len(ALL_NEW_SMILES))])
    def test_smiles_produces_valid_mol(self, smiles):
        """Each new SMILES key should produce a valid RDKit mol object."""
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, f"Invalid SMILES: {smiles}"

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles", ALL_NEW_SMILES,
                             ids=[f"canonical-{i}" for i in range(len(ALL_NEW_SMILES))])
    def test_smiles_is_canonical(self, smiles):
        """Each new SMILES key must be RDKit canonical."""
        canon = Chem.CanonSmiles(smiles)
        assert smiles == canon, f"Non-canonical key: {smiles} -> {canon}"


# ============================================================================
# Test 9: Steroid Vocabulary Verification
# ============================================================================

class TestSteroidVocabulary:
    """Verify: all 6 steroid stems present in NATURAL_PRODUCT_SCAFFOLDS."""

    @pytest.mark.unit
    @pytest.mark.parametrize("steroid_name", [
        "gonane", "estrane", "androstane", "pregnane", "cholane", "cholestane",
    ])
    def test_steroid_in_scaffolds(self, steroid_name):
        """Each steroid stem should appear in NATURAL_PRODUCT_SCAFFOLDS values."""
        from orthonym.data.natural_products import NATURAL_PRODUCT_SCAFFOLDS
        all_names = [v.get("name", "") for v in NATURAL_PRODUCT_SCAFFOLDS.values()]
        assert steroid_name in all_names, f"Missing steroid: {steroid_name}"
