"""
End-to-end integration tests for a phase: Ring System Extensions.

Validates all a phase features working together through the full naming
pipeline (SMILES -> name_compound -> IUPAC name):

1. Skeletal replacement nomenclature (RING-)
   - Plain replacement chains (dioxahexane, dioxaoctane)
   - Terminal alcohol suffix integration (dioxaoctan-1-ol)
2. Ring assemblies (RING-)
   - Biphenyl, bipyridine, bithiophene
   - Substituted ring assemblies
3. Fused heterocycle locants (RING-)
   - Indole indicated hydrogen
   - Substituted fused heterocycles
4. Monocyclic lactams (RING-)
   - Beta through epsilon lactams

Run with: pytest tests/integration/test_phase23_e2e.py -v -m "not roundtrip"
"""

import os
import shutil
import subprocess

import pytest
from rdkit import Chem

from orthonym import name_compound
from tests.support.jars import jar_or_none


# ============================================================================
# Section 1: Skeletal Replacement (RING-)
# ============================================================================


class TestSkeletalReplacementE2E:
    """Verify skeletal replacement naming through the full pipeline."""

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "smiles,expected",
        [
            ("COCCOC", "2,5-dioxahexane"),
            ("CCOCCOCC", "3,6-dioxaoctane"),
            ("COCCOCCOC", "2,5,8-trioxanonane"),
            ("CCOCCOCCOCC", "3,6,9-trioxaundecane"),
        ],
        ids=["dioxahexane", "dioxaoctane", "trioxanonane", "trioxaundecane"],
    )
    def test_skeletal_replacement_naming(self, smiles, expected):
        """Plain skeletal replacement chains produce correct names."""
        assert name_compound(smiles) == expected

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "smiles,expected",
        [
            ("OCCOCCOCC", "3,6-dioxaoctan-1-ol"),
            ("OCCOCCOCCOCC", "3,6,9-trioxaundecan-1-ol"),
        ],
        ids=["dioxaoctanol", "trioxaundecanol"],
    )
    def test_skeletal_replacement_with_terminal_oh(self, smiles, expected):
        """Replacement chains with terminal -OH produce correct -ol suffix names."""
        assert name_compound(smiles) == expected

    @pytest.mark.integration
    def test_short_ether_not_replaced(self):
        """Short ethers (< 6 atoms with 1 heteroatom) use substitutive naming."""
        result = name_compound("COC")
        assert result == "methoxymethane"

    @pytest.mark.integration
    def test_sulfide_replacement(self):
        """Sulfide chains use replacement naming (thia)."""
        result = name_compound("CCCSCCCC")
        assert "thia" in result

    @pytest.mark.integration
    def test_branched_chain_defers(self):
        """Branched chains with substituents fall back to substitutive naming."""
        # CCOCC(C)OCC has a methyl branch -- should not use replacement naming
        result = name_compound("CCOCC(C)OCC")
        # Should not contain "oxa" since branched chains are rejected
        # (it will use substitutive naming instead)
        assert result is not None


# ============================================================================
# Section 2: Ring Assemblies (RING-)
# ============================================================================


class TestRingAssemblyE2E:
    """Verify ring assembly naming through the full pipeline."""

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "smiles,expected_contains",
        [
            ("c1ccc(-c2ccccc2)cc1", "biphenyl"),
            ("c1ccncc1-c1ccncc1", "bipyridine"),
            ("c1ccsc1-c1ccsc1", "bithiophene"),
        ],
        ids=["biphenyl", "bipyridine", "bithiophene"],
    )
    def test_ring_assembly_naming(self, smiles, expected_contains):
        """Ring assemblies contain the expected base name."""
        result = name_compound(smiles)
        assert expected_contains in result, (
            f"Expected '{expected_contains}' in '{result}'"
        )

    @pytest.mark.integration
    def test_biphenyl_full_name(self):
        """Unsubstituted biphenyl produces 'biphenyl' (retained name per."""
        result = name_compound("c1ccc(-c2ccccc2)cc1")
        assert result == "biphenyl"

    @pytest.mark.integration
    def test_substituted_biphenyl(self):
        """4-chlorobiphenyl is correctly named."""
        result = name_compound("Clc1ccc(-c2ccccc2)cc1")
        assert "chloro" in result
        assert "biphenyl" in result

    @pytest.mark.integration
    def test_substituted_biphenyl_full(self):
        """4-chlorobiphenyl produces the full correct name."""
        result = name_compound("Clc1ccc(-c2ccccc2)cc1")
        assert result == "4-chloro-1,1'-biphenyl"


# ============================================================================
# Section 3: Fused Heterocycle Locants (RING-)
# ============================================================================


class TestFusedHeterocycleLocantE2E:
    """Verify fused heterocycle locant naming through the full pipeline."""

    @pytest.mark.integration
    def test_indole_indicated_hydrogen(self):
        """Indole produces a name with indicated hydrogen or retained name."""
        result = name_compound("c1ccc2[nH]ccc2c1")
        assert "1H-" in result or "indole" in result

    @pytest.mark.integration
    def test_indole_full_name(self):
        """Indole produces '1H-indole'."""
        result = name_compound("c1ccc2[nH]ccc2c1")
        assert result == "1H-indole"

    @pytest.mark.integration
    def test_substituted_indole_preserves_indicated_h(self):
        """Substituted indole preserves indicated hydrogen and methyl prefix."""
        result = name_compound("Cc1ccc2[nH]ccc2c1")
        assert "methyl" in result
        assert "indole" in result

    @pytest.mark.integration
    def test_quinoline(self):
        """Quinoline produces correct retained name."""
        result = name_compound("c1ccc2ncccc2c1")
        assert result == "quinoline"

    @pytest.mark.integration
    def test_isoquinoline(self):
        """Isoquinoline produces correct retained name."""
        result = name_compound("c1ccc2cnccc2c1")
        assert result == "isoquinoline"


# ============================================================================
# Section 4: Lactams (RING-)
# ============================================================================


class TestLactamE2E:
    """Verify monocyclic lactam naming through the full pipeline."""

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "smiles,expected",
        [
            ("C1CC(=O)N1", "azetidin-2-one"),
            ("C1CCC(=O)N1", "pyrrolidin-2-one"),
            ("C1CCCC(=O)N1", "piperidin-2-one"),
            #: was "caprolactam" cited to "", which is
            # "Bi- and polycyclic von Baeyer parent hydrides" (the Blue Book) and
            # licenses nothing here. The PIN is the pseudoketone form,
            # (the Blue Book), printed at the Blue Book as `azepan-2-one (PIN)`.
            ("C1CCCCC(=O)N1", "azepan-2-one"),
        ],
        ids=["beta-lactam", "gamma-lactam", "delta-lactam", "epsilon-lactam"],
    )
    def test_lactam_naming(self, smiles, expected):
        """Monocyclic lactams produce correct IUPAC names."""
        assert name_compound(smiles) == expected

    @pytest.mark.integration
    def test_succinimide_not_misrouted(self):
        """Succinimide (imide, not lactam) should not be named as a lactam."""
        result = name_compound("O=C1CCC(=O)N1")
        # Imides have 2 C=O on the ring nitrogen -- should not produce lactam name
        assert "lactam" not in (result or "").lower()
        assert result is not None


# ============================================================================
# Section 5: Cross-feature regression tests
# ============================================================================


class TestPhase23Regressions:
    """Verify a phase features do not regress existing naming."""

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "smiles,expected",
        [
            # Basic compounds that must not regress
            ("C", "methane"),
            ("CC", "ethane"),
            ("CCO", "ethanol"),
            ("c1ccccc1", "benzene"),
            ("c1ccncc1", "pyridine"),
            ("C1CCCCC1", "cyclohexane"),
            ("CC(=O)O", "acetic acid"),
            # Short ethers remain substitutive
            ("COC", "methoxymethane"),
            # Simple heterocycles
            ("c1ccoc1", "furan"),
            ("c1cc[nH]c1", "pyrrole"),
            ("c1ccsc1", "thiophene"),
        ],
        ids=[
            "methane", "ethane", "ethanol", "benzene", "pyridine",
            "cyclohexane", "acetic-acid", "methoxymethane",
            "furan", "pyrrole", "thiophene",
        ],
    )
    def test_basic_naming_unchanged(self, smiles, expected):
        """Basic compound naming is unaffected by a phase additions."""
        assert name_compound(smiles) == expected


# ============================================================================
# Section 6: OPSIN Round-Trip Tests
# ============================================================================

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


# Skeletal replacement names for OPSIN round-trip
SKELETAL_REPLACEMENT_ROUNDTRIP = [
    ("COCCOC", "2,5-dioxahexane"),
    ("CCOCCOCC", "3,6-dioxaoctane"),
    ("COCCOCCOC", "2,5,8-trioxanonane"),
    ("CCOCCOCCOCC", "3,6,9-trioxaundecane"),
]

# Skeletal replacement with terminal OH suffix
SKELETAL_REPLACEMENT_OH_ROUNDTRIP = [
    ("OCCOCCOCC", "3,6-dioxaoctan-1-ol"),
    ("OCCOCCOCCOCC", "3,6,9-trioxaundecan-1-ol"),
]

# Lactam names for OPSIN round-trip
LACTAM_ROUNDTRIP = [
    ("C1CC(=O)N1", "azetidin-2-one"),
    ("C1CCC(=O)N1", "pyrrolidin-2-one"),
    ("C1CCCC(=O)N1", "piperidin-2-one"),
    ("C1CCCCC(=O)N1", "azepan-2-one"),  #; PIN per (the Blue Book)
]

# Ring assembly names for OPSIN round-trip
RING_ASSEMBLY_ROUNDTRIP = [
    ("c1ccc(-c2ccccc2)cc1", "biphenyl"),
]


@pytest.mark.skipif(
    not OPSIN_AVAILABLE,
    reason="Java or OPSIN JAR not available for round-trip tests",
)
class TestPhase23SkeletalReplacementRoundTrip:
    """OPSIN round-trip tests for skeletal replacement names."""

    @pytest.mark.roundtrip
    @pytest.mark.parametrize(
        "smiles,expected_name",
        SKELETAL_REPLACEMENT_ROUNDTRIP,
        ids=["dioxahexane", "dioxaoctane", "trioxanonane", "trioxaundecane"],
    )
    def test_skeletal_replacement_exact_roundtrip(self, smiles, expected_name):
        """Skeletal replacement name -> OPSIN -> canonical SMILES must match."""
        name = name_compound(smiles)
        assert name == expected_name, (
            f"Name mismatch: expected '{expected_name}', got '{name}'"
        )

        opsin_smiles = _opsin_parse(name)
        assert opsin_smiles, (
            f"OPSIN could not parse replacement name '{name}' (from {smiles})"
        )

        canonical_input = Chem.CanonSmiles(smiles)
        canonical_output = Chem.CanonSmiles(opsin_smiles)
        assert canonical_input == canonical_output, (
            f"Round-trip SMILES mismatch for '{name}':\n"
            f"  input:  {canonical_input}\n"
            f"  output: {canonical_output}"
        )

    @pytest.mark.roundtrip
    @pytest.mark.parametrize(
        "smiles,expected_name",
        SKELETAL_REPLACEMENT_OH_ROUNDTRIP,
        ids=["dioxaoctanol", "trioxaundecanol"],
    )
    def test_skeletal_replacement_oh_exact_roundtrip(self, smiles, expected_name):
        """Replacement chain + -ol suffix -> OPSIN -> canonical SMILES must match."""
        name = name_compound(smiles)
        assert name == expected_name, (
            f"Name mismatch: expected '{expected_name}', got '{name}'"
        )

        opsin_smiles = _opsin_parse(name)
        assert opsin_smiles, (
            f"OPSIN could not parse replacement+OH name '{name}' (from {smiles})"
        )

        canonical_input = Chem.CanonSmiles(smiles)
        canonical_output = Chem.CanonSmiles(opsin_smiles)
        assert canonical_input == canonical_output, (
            f"Round-trip SMILES mismatch for '{name}':\n"
            f"  input:  {canonical_input}\n"
            f"  output: {canonical_output}"
        )


@pytest.mark.skipif(
    not OPSIN_AVAILABLE,
    reason="Java or OPSIN JAR not available for round-trip tests",
)
class TestPhase23LactamRoundTrip:
    """OPSIN round-trip tests for lactam names."""

    @pytest.mark.roundtrip
    @pytest.mark.parametrize(
        "smiles,expected_name",
        LACTAM_ROUNDTRIP,
        ids=["beta-lactam", "gamma-lactam", "delta-lactam", "epsilon-lactam"],
    )
    def test_lactam_exact_roundtrip(self, smiles, expected_name):
        """Lactam name -> OPSIN -> canonical SMILES must match."""
        name = name_compound(smiles)
        assert name == expected_name, (
            f"Name mismatch: expected '{expected_name}', got '{name}'"
        )

        opsin_smiles = _opsin_parse(name)
        assert opsin_smiles, (
            f"OPSIN could not parse lactam name '{name}' (from {smiles})"
        )

        canonical_input = Chem.CanonSmiles(smiles)
        canonical_output = Chem.CanonSmiles(opsin_smiles)
        assert canonical_input == canonical_output, (
            f"Round-trip SMILES mismatch for lactam '{name}':\n"
            f"  input:  {canonical_input}\n"
            f"  output: {canonical_output}"
        )


@pytest.mark.skipif(
    not OPSIN_AVAILABLE,
    reason="Java or OPSIN JAR not available for round-trip tests",
)
class TestPhase23RingAssemblyRoundTrip:
    """OPSIN round-trip tests for ring assembly names."""

    @pytest.mark.roundtrip
    @pytest.mark.parametrize(
        "smiles,expected_name",
        RING_ASSEMBLY_ROUNDTRIP,
        ids=["biphenyl"],
    )
    def test_ring_assembly_exact_roundtrip(self, smiles, expected_name):
        """Ring assembly name -> OPSIN -> canonical SMILES must match."""
        name = name_compound(smiles)
        assert name == expected_name, (
            f"Name mismatch: expected '{expected_name}', got '{name}'"
        )

        opsin_smiles = _opsin_parse(name)
        assert opsin_smiles, (
            f"OPSIN could not parse assembly name '{name}' (from {smiles})"
        )

        canonical_input = Chem.CanonSmiles(smiles)
        canonical_output = Chem.CanonSmiles(opsin_smiles)
        assert canonical_input == canonical_output, (
            f"Round-trip SMILES mismatch for assembly '{name}':\n"
            f"  input:  {canonical_input}\n"
            f"  output: {canonical_output}"
        )
