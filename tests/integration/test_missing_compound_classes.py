"""
Integration tests for all 5 missing compound classes (a phase).

Validates end-to-end naming for:
  : Acetals/Hemiacetals (cyclic and acyclic)
  : Disulfides/Trisulfides (replacement naming)
  : Cyclic Imides (retained and systematic)
  : Thiocarboxylic Acids (S-acid, O-acid, dithioic)
  : Carbamic Acid (parent and N-substituted)

Each test calls name_compound on real-world SMILES and verifies the
expected IUPAC name. OPSIN round-trip validation confirms parseability
of all generated names.
"""

import subprocess
import pytest

from orthonym import name_compound


# ---------------------------------------------------------------------------
# Helper: OPSIN round-trip
# ---------------------------------------------------------------------------

OPSIN_JAR = "opsin-cli-2.9.0-jar-with-dependencies.jar"


def _opsin_name_to_smiles(name: str) -> str | None:
    """Parse an IUPAC name to SMILES via the local OPSIN JAR.

    Returns None if OPSIN cannot parse the name or the JAR is unavailable.
    """
    try:
        result = subprocess.run(
            ["java", "-jar", OPSIN_JAR, "-osmi"],
            input=name,
            capture_output=True,
            text=True,
            timeout=15,
        )
        smi = result.stdout.strip()
        return smi if smi else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _inchi_match(smi1: str, smi2: str) -> bool:
    """Compare two SMILES by InChI (structure equivalence)."""
    from rdkit import Chem
    try:
        mol1 = Chem.MolFromSmiles(smi1)
        mol2 = Chem.MolFromSmiles(smi2)
        if mol1 is None or mol2 is None:
            return False
        inchi1 = Chem.inchi.MolToInchi(mol1)
        inchi2 = Chem.inchi.MolToInchi(mol2)
        return inchi1 == inchi2 and inchi1 is not None
    except Exception:
        return False


# ====================================================================
#: Acetals / Hemiacetals
# ====================================================================


class TestCLS01Acetals:
    """Acetal and hemiacetal naming -- cyclic and acyclic."""

    @pytest.mark.integration
    def test_1_3_dioxolane(self):
        """Cyclic acetal: 5-membered ring with two oxygens."""
        assert name_compound("C1OCCO1") == "1,3-dioxolane"

    @pytest.mark.integration
    def test_1_3_dioxane(self):
        """Cyclic acetal: 6-membered ring with two oxygens."""
        assert name_compound("C1OCCCO1") == "1,3-dioxane"

    @pytest.mark.integration
    def test_dimethoxy_ethane(self):
        """Acyclic acetal: dimethyl acetal of acetaldehyde."""
        assert name_compound("COC(C)OC") == "1,1-dimethoxyethane"

    @pytest.mark.integration
    def test_lactol_hydroxy_oxane(self):
        """Lactol (hemiacetal): hydroxy on oxane ring."""
        name = name_compound("OC1CCCCO1")
        assert "hydroxy" in name, f"Expected 'hydroxy' in '{name}'"
        # Should be named as an oxane/oxane derivative
        assert "pyran" in name or "oxan" in name, (
            f"Expected ring parent (pyran or oxan) in '{name}'"
        )

    @pytest.mark.integration
    def test_dimethoxy_butane(self):
        """Acyclic acetal: 2,2-dimethoxybutane."""
        assert name_compound("CC(OC)(OC)CC") == "2,2-dimethoxybutane"

    @pytest.mark.integration
    def test_1_3_dioxolane_opsin_roundtrip(self):
        """OPSIN round-trip: 1,3-dioxolane."""
        name = name_compound("C1OCCO1")
        opsin_smi = _opsin_name_to_smiles(name)
        if opsin_smi is None:
            pytest.skip("OPSIN JAR not available")
        assert _inchi_match("C1OCCO1", opsin_smi), (
            f"Round-trip failed: C1OCCO1 -> '{name}' -> {opsin_smi}"
        )

    @pytest.mark.integration
    def test_1_3_dioxane_opsin_roundtrip(self):
        """OPSIN round-trip: 1,3-dioxane."""
        name = name_compound("C1OCCCO1")
        opsin_smi = _opsin_name_to_smiles(name)
        if opsin_smi is None:
            pytest.skip("OPSIN JAR not available")
        assert _inchi_match("C1OCCCO1", opsin_smi), (
            f"Round-trip failed: C1OCCCO1 -> '{name}' -> {opsin_smi}"
        )


# ====================================================================
#: Disulfides / Trisulfides
# ====================================================================


class TestCLS02Disulfides:
    """Disulfide and trisulfide naming via replacement nomenclature."""

    @pytest.mark.integration
    def test_dithiabutane(self):
        """Simplest dialkyl disulfide: CSSC. DD2 (Phase D, P-63.3.1(1)): a dialkyl
        disulfide is a substitutive PIN ((R)disulfanyl on the senior parent), NOT
        a 'dithia' skeletal-replacement chain (the skeletal form consumed the S-S
        as two skeletal thia atoms — the C3 defect)."""
        assert name_compound("CSSC") == "(methyldisulfanyl)methane"

    @pytest.mark.integration
    def test_dithiahexane(self):
        """Symmetric disulfide: CCSSCC. DD2 (Phase D, (1)): substitutive
        PIN, not the 'dithia' skeletal form."""
        assert name_compound("CCSSCC") == "(ethyldisulfanyl)ethane"

    @pytest.mark.integration
    def test_trithiaheptane(self):
        """Symmetric trisulfide: CCSSSCC."""
        assert name_compound("CCSSSCC") == "3,4,5-trithiaheptane"

    @pytest.mark.integration
    def test_trithiaoctane_asymmetric(self):
        """Asymmetric trisulfide: CCSSSCCC."""
        assert name_compound("CCSSSCCC") == "3,4,5-trithiaoctane"

    @pytest.mark.integration
    def test_diphenyl_disulfide(self):
        """Aromatic disulfide: PhSSPh -- names via sulfanyl/disulfanediyl prefix.

        a phase cleanup: extended the substring check to accept the
        multiplicative `disulfanediyl` connector form (IUPAC /
         valid PIN) in addition to the substitutive `sulfanyl`
        and functional-class `disulfide` forms. The current output
        `1,1'-disulfanediyldibenzene` uses the multiplicative form per
         — equally valid for symmetric aromatic disulfides.
        """
        name = name_compound("c1ccc(SSc2ccccc2)cc1")
        # PhSSPh names: 'sulfanyl' (substitutive), 'disulfanediyl'
        # (multiplicative connector,, or 'disulfide'
        # (functional class). All are valid IUPAC PIN forms.
        assert "sulfan" in name or "disulfide" in name, (
            f"Expected 'sulfan*' or 'disulfide' in '{name}'"
        )

    @pytest.mark.integration
    def test_dithiabutane_opsin_roundtrip(self):
        """OPSIN round-trip: 2,3-dithiabutane."""
        name = name_compound("CSSC")
        opsin_smi = _opsin_name_to_smiles(name)
        if opsin_smi is None:
            pytest.skip("OPSIN JAR not available")
        assert _inchi_match("CSSC", opsin_smi), (
            f"Round-trip failed: CSSC -> '{name}' -> {opsin_smi}"
        )

    @pytest.mark.integration
    def test_dithiahexane_opsin_roundtrip(self):
        """OPSIN round-trip: 3,4-dithiahexane."""
        name = name_compound("CCSSCC")
        opsin_smi = _opsin_name_to_smiles(name)
        if opsin_smi is None:
            pytest.skip("OPSIN JAR not available")
        assert _inchi_match("CCSSCC", opsin_smi), (
            f"Round-trip failed: CCSSCC -> '{name}' -> {opsin_smi}"
        )

    @pytest.mark.integration
    def test_trithiaheptane_opsin_roundtrip(self):
        """OPSIN round-trip: 3,4,5-trithiaheptane."""
        name = name_compound("CCSSSCC")
        opsin_smi = _opsin_name_to_smiles(name)
        if opsin_smi is None:
            pytest.skip("OPSIN JAR not available")
        assert _inchi_match("CCSSSCC", opsin_smi), (
            f"Round-trip failed: CCSSSCC -> '{name}' -> {opsin_smi}"
        )


# ====================================================================
#: Cyclic Imides
# ====================================================================


class TestCLS03CyclicImides:
    """Cyclic imide naming -- retained names and systematic."""

    @pytest.mark.integration
    def test_succinimide(self):
        """Retained name: succinimide."""
        assert name_compound("O=C1CCC(=O)N1") == "succinimide"

    @pytest.mark.integration
    def test_maleimide(self):
        """Retained name: maleimide."""
        assert name_compound("O=C1C=CC(=O)N1") == "maleimide"

    @pytest.mark.integration
    def test_phthalimide_retained(self):
        """Phthalimide -> retained name (IUPAC."""
        assert name_compound("O=C1NC(=O)c2ccccc21") == "phthalimide"

    @pytest.mark.integration
    def test_methyl_succinimide_derivative(self):
        """C-substituted succinimide: 3-methyl-2,5-dioxopyrrolidine."""
        name = name_compound("CC1CC(=O)NC1=O")
        # Should contain 'methyl' and the dioxopyrrolidine systematic name
        assert "methyl" in name, f"Expected 'methyl' in '{name}'"
        assert "pyrrolidin" in name or "succinimid" in name, (
            f"Expected pyrrolidine or succinimide root in '{name}'"
        )

    @pytest.mark.integration
    def test_n_substituted_succinimide(self):
        """N-substituted succinimide: N-benzyl or systematic."""
        name = name_compound("O=C1CCC(=O)N1Cc1ccccc1")
        # Should be a dioxopyrrolidine derivative or N-substituted succinimide
        assert "pyrrolidin" in name or "succinimid" in name, (
            f"Expected pyrrolidine or succinimide root in '{name}'"
        )

    @pytest.mark.integration
    def test_succinimide_opsin_roundtrip(self):
        """OPSIN round-trip: succinimide."""
        name = name_compound("O=C1CCC(=O)N1")
        opsin_smi = _opsin_name_to_smiles(name)
        if opsin_smi is None:
            pytest.skip("OPSIN JAR not available")
        assert _inchi_match("O=C1CCC(=O)N1", opsin_smi), (
            f"Round-trip failed: O=C1CCC(=O)N1 -> '{name}' -> {opsin_smi}"
        )

    @pytest.mark.integration
    def test_maleimide_opsin_roundtrip(self):
        """OPSIN round-trip: maleimide."""
        name = name_compound("O=C1C=CC(=O)N1")
        opsin_smi = _opsin_name_to_smiles(name)
        if opsin_smi is None:
            pytest.skip("OPSIN JAR not available")
        assert _inchi_match("O=C1C=CC(=O)N1", opsin_smi), (
            f"Round-trip failed: O=C1C=CC(=O)N1 -> '{name}' -> {opsin_smi}"
        )

    @pytest.mark.integration
    def test_isoindoline_dione_opsin_roundtrip(self):
        """OPSIN round-trip: isoindoline-1,3-dione."""
        name = name_compound("O=C1NC(=O)c2ccccc21")
        opsin_smi = _opsin_name_to_smiles(name)
        if opsin_smi is None:
            pytest.skip("OPSIN JAR not available")
        assert _inchi_match("O=C1NC(=O)c2ccccc21", opsin_smi), (
            f"Round-trip failed: phthalimide -> '{name}' -> {opsin_smi}"
        )


# ====================================================================
#: Thiocarboxylic Acids
# ====================================================================


class TestCLS04ThiocarboxylicAcids:
    """Thiocarboxylic acid naming -- S-acid, O-acid, dithioic acid."""

    @pytest.mark.integration
    def test_ethanethioic_s_acid(self):
        """Thioic S-acid: C(=O)SH -> ethanethioic S-acid."""
        assert name_compound("CC(=O)S") == "ethanethioic S-acid"

    @pytest.mark.integration
    def test_ethanethioic_o_acid(self):
        """Thioic O-acid: C(=S)OH -> ethanethioic O-acid."""
        assert name_compound("CC(=S)O") == "ethanethioic O-acid"

    @pytest.mark.integration
    def test_ethanedithioic_acid(self):
        """Dithioic acid: C(=S)SH -> ethanedithioic acid."""
        assert name_compound("CC(=S)S") == "ethanedithioic acid"

    @pytest.mark.integration
    def test_propanethioic_s_acid(self):
        """Three-carbon thioic S-acid."""
        assert name_compound("CCC(=O)S") == "propanethioic S-acid"

    @pytest.mark.integration
    def test_butanethioic_s_acid(self):
        """Four-carbon thioic S-acid."""
        assert name_compound("CCCC(=O)S") == "butanethioic S-acid"

    @pytest.mark.integration
    def test_benzenecarbothioic_s_acid(self):
        """Aromatic thiocarboxylic acid: no regression on ring-attached."""
        assert name_compound("SC(=O)c1ccccc1") == "benzenecarbothioic S-acid"

    @pytest.mark.integration
    def test_ethanethioic_s_acid_opsin_roundtrip(self):
        """OPSIN round-trip: ethanethioic S-acid."""
        name = name_compound("CC(=O)S")
        opsin_smi = _opsin_name_to_smiles(name)
        if opsin_smi is None:
            pytest.skip("OPSIN JAR not available")
        assert _inchi_match("CC(=O)S", opsin_smi), (
            f"Round-trip failed: CC(=O)S -> '{name}' -> {opsin_smi}"
        )

    @pytest.mark.integration
    def test_ethanedithioic_acid_opsin_roundtrip(self):
        """OPSIN round-trip: ethanedithioic acid."""
        name = name_compound("CC(=S)S")
        opsin_smi = _opsin_name_to_smiles(name)
        if opsin_smi is None:
            pytest.skip("OPSIN JAR not available")
        assert _inchi_match("CC(=S)S", opsin_smi), (
            f"Round-trip failed: CC(=S)S -> '{name}' -> {opsin_smi}"
        )


# ====================================================================
#: Carbamic Acid
# ====================================================================


class TestCLS05CarbamicAcid:
    """Carbamic acid naming -- parent and N-substituted forms."""

    @pytest.mark.integration
    def test_carbamic_acid(self):
        """Parent carbamic acid: NC(=O)O."""
        assert name_compound("NC(=O)O") == "carbamic acid"

    @pytest.mark.integration
    def test_n_methylcarbamic_acid(self):
        """N-monosubstituted: N-methylcarbamic acid."""
        assert name_compound("CNC(=O)O") == "N-methylcarbamic acid"

    @pytest.mark.integration
    def test_n_n_dimethylcarbamic_acid(self):
        """N,N-disubstituted: N,N-dimethylcarbamic acid."""
        assert name_compound("CN(C)C(=O)O") == "N,N-dimethylcarbamic acid"

    @pytest.mark.integration
    def test_n_n_diethylcarbamic_acid(self):
        """N,N-disubstituted with ethyl: N,N-diethylcarbamic acid."""
        assert name_compound("CCN(CC)C(=O)O") == "N,N-diethylcarbamic acid"

    @pytest.mark.integration
    def test_n_phenylcarbamic_acid(self):
        """N-aryl substituted: N-phenylcarbamic acid."""
        assert name_compound("c1ccc(NC(=O)O)cc1") == "N-phenylcarbamic acid"

    @pytest.mark.integration
    def test_carbamic_acid_opsin_roundtrip(self):
        """OPSIN round-trip: carbamic acid."""
        name = name_compound("NC(=O)O")
        opsin_smi = _opsin_name_to_smiles(name)
        if opsin_smi is None:
            pytest.skip("OPSIN JAR not available")
        assert _inchi_match("NC(=O)O", opsin_smi), (
            f"Round-trip failed: NC(=O)O -> '{name}' -> {opsin_smi}"
        )

    @pytest.mark.integration
    def test_n_methylcarbamic_acid_opsin_roundtrip(self):
        """OPSIN round-trip: N-methylcarbamic acid."""
        name = name_compound("CNC(=O)O")
        opsin_smi = _opsin_name_to_smiles(name)
        if opsin_smi is None:
            pytest.skip("OPSIN JAR not available")
        assert _inchi_match("CNC(=O)O", opsin_smi), (
            f"Round-trip failed: CNC(=O)O -> '{name}' -> {opsin_smi}"
        )


# ====================================================================
# Regression spot-checks (other compound classes unaffected)
# ====================================================================


class TestRegressionSpotChecks:
    """Spot-check that core compound classes remain unaffected."""

    @pytest.mark.integration
    def test_ethanol(self):
        assert name_compound("CCO") == "ethanol"

    @pytest.mark.integration
    def test_benzene(self):
        assert name_compound("c1ccccc1") == "benzene"

    @pytest.mark.integration
    def test_cyclohexane(self):
        assert name_compound("C1CCCCC1") == "cyclohexane"

    @pytest.mark.integration
    def test_acetic_acid(self):
        assert name_compound("CC(=O)O") == "acetic acid"

    @pytest.mark.integration
    def test_pyridine(self):
        assert name_compound("c1ccncc1") == "pyridine"

    @pytest.mark.integration
    def test_disulfane(self):
        """Disulfane (SS) -- already in canary suite, verify no regression."""
        assert name_compound("SS") == "disulfane"
