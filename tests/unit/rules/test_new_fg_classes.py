"""Tests for 6 new functional group classes added in Phase 109 (DATA-03, DATA-04, DATA-05).

Each new FG class requires:
  1. SMARTS pattern in FUNCTIONAL_GROUP_SMARTS (functional_groups.py)
  2. Collision resolution rules in _resolve_fg_collisions (functional_groups.py)
  3. Seniority position in SENIORITY_ORDER (seniority.py)
  4. Suffix form in SUFFIX_FORMS (seniority.py)
  5. Prefix form in PREFIX_FORMS (seniority.py)
"""
import pytest
from rdkit import Chem
from orthonym.perception.functional_groups import detect_functional_groups
from orthonym.rules.seniority import SENIORITY_ORDER, SUFFIX_FORMS, PREFIX_FORMS


# Helper
def _detect(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Invalid SMILES: {smiles}"
    return detect_functional_groups(mol)


# ---------------------------------------------------------------------------
# DATA-03: Amidine (IUPAC P-66.4.1)
# ---------------------------------------------------------------------------
class TestAmidine:
    """Amidine: R-C(=NH)-NH2, suffix -imidamide/-carboximidamide, prefix amidino."""

    def test_acetamidine_detected(self):
        """Acetamidine CC(=N)N should be detected as amidine."""
        result = _detect("CC(=N)N")
        assert "amidine" in result

    def test_guanidine_suppresses_amidine(self):
        """Guanidine NC(=N)N should NOT be detected as amidine (guanidine is more specific)."""
        result = _detect("NC(=N)N")
        assert "amidine" not in result
        assert "guanidine" in result

    def test_amidine_suppresses_imine(self):
        """Acetamidine CC(=N)N should NOT have imine match (amidine is more specific)."""
        result = _detect("CC(=N)N")
        assert "amidine" in result
        assert "imine" not in result

    def test_amidine_seniority_position(self):
        """Amidine should be between imide and nitrile in SENIORITY_ORDER."""
        idx_imide = SENIORITY_ORDER.index("imide")
        idx_amidine = SENIORITY_ORDER.index("amidine")
        idx_nitrile = SENIORITY_ORDER.index("nitrile")
        assert idx_imide < idx_amidine < idx_nitrile

    def test_amidine_suffix_forms(self):
        """Amidine suffix: chain=imidamide, ring=carboximidamide."""
        assert SUFFIX_FORMS["amidine"] == ("imidamide", "carboximidamide")

    def test_amidine_prefix_form(self):
        """Amidine prefix: amidino."""
        assert PREFIX_FORMS["amidine"] == "amidino"


# ---------------------------------------------------------------------------
# DATA-04: Acid Iodide (IUPAC P-65.5.1)
# ---------------------------------------------------------------------------
class TestAcidIodide:
    """Acid iodide: R-C(=O)-I, parallel to acid_chloride/bromide/fluoride."""

    def test_acetyl_iodide_detected(self):
        """Acetyl iodide CC(=O)I should be detected as acid_iodide."""
        result = _detect("CC(=O)I")
        assert "acid_iodide" in result

    def test_acid_iodide_suppresses_aldehyde(self):
        """Acetyl iodide CC(=O)I should NOT be detected as aldehyde."""
        result = _detect("CC(=O)I")
        assert "aldehyde" not in result

    def test_acid_iodide_seniority(self):
        """Acid iodide should be between acid_fluoride and primary_amide."""
        idx_fluoride = SENIORITY_ORDER.index("acid_fluoride")
        idx_iodide = SENIORITY_ORDER.index("acid_iodide")
        idx_amide = SENIORITY_ORDER.index("primary_amide")
        assert idx_fluoride < idx_iodide < idx_amide

    def test_acid_iodide_suffix(self):
        """Acid iodide suffix: chain=oyl iodide, ring=carbonyl iodide."""
        assert SUFFIX_FORMS["acid_iodide"] == ("oyl iodide", "carbonyl iodide")

    def test_acid_iodide_prefix(self):
        """Acid iodide prefix: iodocarbonyl."""
        assert PREFIX_FORMS["acid_iodide"] == "iodocarbonyl"


# ---------------------------------------------------------------------------
# DATA-05a: Diazo (IUPAC P-61.5)
# ---------------------------------------------------------------------------
class TestDiazo:
    """Diazo: R=N+=N-, prefix-only."""

    def test_diazomethane_detected(self):
        """Diazomethane [CH2]=[N+]=[N-] should be detected as diazo."""
        result = _detect("[CH2]=[N+]=[N-]")
        assert "diazo" in result

    def test_diazo_not_azo(self):
        """Diazomethane should NOT be detected as azo (different bonding)."""
        result = _detect("[CH2]=[N+]=[N-]")
        assert "azo" not in result

    def test_diazo_prefix_only(self):
        """Diazo is prefix-only: SUFFIX_FORMS should be None."""
        assert SUFFIX_FORMS["diazo"] is None

    def test_diazo_prefix(self):
        """Diazo prefix: diazo."""
        assert PREFIX_FORMS["diazo"] == "diazo"


# ---------------------------------------------------------------------------
# DATA-05b: Disulfide (IUPAC P-63.6.2)
# ---------------------------------------------------------------------------
class TestDisulfide:
    """Disulfide: R-S-S-R, prefix-only."""

    def test_dimethyl_disulfide_detected(self):
        """Dimethyl disulfide CSSC should be detected as disulfide."""
        result = _detect("CSSC")
        assert "disulfide" in result

    def test_disulfide_not_thioether(self):
        """Dimethyl disulfide CSSC should NOT be detected as thioether."""
        result = _detect("CSSC")
        assert "thioether" not in result

    def test_disulfide_prefix_only(self):
        """Disulfide is prefix-only: SUFFIX_FORMS should be None."""
        assert SUFFIX_FORMS["disulfide"] is None

    def test_disulfide_prefix(self):
        """Disulfide prefix: disulfanediyl."""
        assert PREFIX_FORMS["disulfide"] == "disulfanediyl"


# ---------------------------------------------------------------------------
# DATA-05c: Hydrazine (IUPAC P-62.4)
# ---------------------------------------------------------------------------
class TestHydrazine:
    """Hydrazine FG: R-NH-NH2, prefix-only."""

    def test_methylhydrazine_detected(self):
        """Methylhydrazine CNN should be detected as hydrazine_fg."""
        result = _detect("CNN")
        assert "hydrazine_fg" in result

    def test_hydrazine_suppresses_primary_amine(self):
        """Methylhydrazine CNN: primary_amine should NOT overlap with hydrazine_fg atoms."""
        result = _detect("CNN")
        assert "hydrazine_fg" in result
        if "primary_amine" in result:
            hydrazine_atoms = set()
            for match in result["hydrazine_fg"]:
                hydrazine_atoms.update(match)
            for match in result.get("primary_amine", []):
                assert not any(atom in hydrazine_atoms for atom in match), \
                    "primary_amine should not overlap with hydrazine_fg atoms"

    def test_hydrazide_not_hydrazine(self):
        """Acetohydrazide CC(=O)NN: hydrazine_fg should NOT match."""
        result = _detect("CC(=O)NN")
        assert "hydrazine_fg" not in result

    def test_hydrazine_prefix_only(self):
        """Hydrazine FG is prefix-only: SUFFIX_FORMS should be None."""
        assert SUFFIX_FORMS["hydrazine_fg"] is None

    def test_hydrazine_prefix(self):
        """Hydrazine FG prefix: hydrazinyl."""
        assert PREFIX_FORMS["hydrazine_fg"] == "hydrazinyl"


# ---------------------------------------------------------------------------
# DATA-05d: Sulfenic Acid (IUPAC P-65.3.1.4)
# ---------------------------------------------------------------------------
class TestSulfenicAcid:
    """Sulfenic acid: R-S-OH, between sulfinic and phosphonic acid in seniority."""

    def test_methanesulfenic_acid_detected(self):
        """DD2 (Phase D, P-56.2): R-S-OH is now perceived as ``so_thioperoxol`` and
        named ``methane-SO-thioperoxol`` (PIN), NOT the Blue-Book-retired
        ``methanesulfenic acid``. ``so_thioperoxol`` suppresses ``sulfenic_acid``
        on overlap (collision resolver)."""
        result = _detect("CSO")
        assert "so_thioperoxol" in result
        assert "sulfenic_acid" not in result

    def test_sulfenic_not_thiol(self):
        """Methanesulfenic acid CSO should NOT be detected as thiol."""
        result = _detect("CSO")
        assert "thiol" not in result

    def test_sulfenic_seniority(self):
        """Sulfenic acid should be between sulfinic_acid and phosphonic_acid."""
        idx_sulfinic = SENIORITY_ORDER.index("sulfinic_acid")
        idx_sulfenic = SENIORITY_ORDER.index("sulfenic_acid")
        idx_phosphonic = SENIORITY_ORDER.index("phosphonic_acid")
        assert idx_sulfinic < idx_sulfenic < idx_phosphonic

    def test_sulfenic_suffix(self):
        """Sulfenic acid suffix: (sulfenic acid, sulfenic acid)."""
        assert SUFFIX_FORMS["sulfenic_acid"] == ("sulfenic acid", "sulfenic acid")

    def test_sulfenic_prefix(self):
        """Sulfenic acid prefix: sulfeno."""
        assert PREFIX_FORMS["sulfenic_acid"] == "sulfeno"
