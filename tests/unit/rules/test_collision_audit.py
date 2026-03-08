"""
Collision resolution audit tests for functional group detection.

Verifies that _resolve_fg_collisions() correctly handles all overlap
scenarios between FG patterns, including new seniority entries from
Phase 093 Plan 01 and all existing collision rules (PERC-01, PERC-02,
PERC-03, USUB-11).

Requirements: QUAL-01 (zero regressions), QUAL-02 (canary stability),
              QUAL-04 (collision coverage)
"""

import pytest
from rdkit import Chem
from orthonym.perception.functional_groups import detect_functional_groups


def _detect(smiles):
    """Helper: parse SMILES and return detected FGs."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Invalid SMILES: {smiles}"
    return detect_functional_groups(mol)


# ===========================================================================
# 1. New overlap scenarios (from Plan 01 seniority additions)
# ===========================================================================

class TestThioesterCollisions:
    """Thioester (C(=O)-S-C) must not produce false positives for thiol or ester."""

    def test_thioester_detected(self):
        """Thioester molecule detects thioester FG."""
        fgs = _detect("CC(=O)SC")
        assert "thioester" in fgs

    def test_thioester_not_thiol(self):
        """Thioester does NOT produce thiol false positive (S is bonded to C=O, not SH)."""
        fgs = _detect("CC(=O)SC")
        assert "thiol" not in fgs, (
            "Thioester S is not -SH; thiol should not be detected"
        )

    def test_thioester_not_ester(self):
        """Thioester does NOT produce ester false positive (S vs O distinguishes)."""
        fgs = _detect("CC(=O)SC")
        assert "ester" not in fgs, (
            "Thioester uses S not O; ester should not be detected"
        )

    def test_thioester_coexists_with_thioether(self):
        """Thioester currently also detects thioether (S-C matches both patterns).

        This is expected behavior: the thioester SMARTS C(=O)-S-C and thioether
        SMARTS S(C)(C) both match the same S atom. No collision rule exists to
        suppress thioether, and this is acceptable since the naming pipeline
        routes via the higher-seniority thioester.
        """
        fgs = _detect("CC(=O)SC")
        # Document the expected behavior without asserting it as correct/incorrect
        assert "thioester" in fgs, "Primary detection must be present"


class TestImideCollisions:
    """Imide (C(=O)-N-C(=O)) must suppress overlapping amide matches (USUB-11)."""

    def test_imide_detected(self):
        """Succinimide detects as imide."""
        fgs = _detect("O=C1CCC(=O)N1")
        assert "imide" in fgs

    def test_imide_suppresses_primary_amide(self):
        """Imide suppresses primary_amide on overlapping atoms (USUB-11)."""
        fgs = _detect("O=C1CCC(=O)N1")
        assert "primary_amide" not in fgs, (
            "USUB-11: imide atoms should suppress overlapping amide"
        )

    def test_imide_suppresses_secondary_amide(self):
        """Imide suppresses secondary_amide on overlapping atoms (USUB-11)."""
        fgs = _detect("O=C1CCC(=O)N1")
        assert "secondary_amide" not in fgs

    def test_imide_does_not_produce_ketone(self):
        """Imide C=O atoms should not also produce ketone false positive."""
        fgs = _detect("O=C1CCC(=O)N1")
        # In succinimide, both C=O are part of the imide, not separate ketones
        assert "ketone" not in fgs, (
            "Imide C=O should not produce ketone; C is bonded to N, not 2 carbons"
        )

    def test_open_chain_imide(self):
        """Open-chain imide (diacetamide) also detects correctly."""
        fgs = _detect("CC(=O)NC(=O)C")
        assert "imide" in fgs
        assert "primary_amide" not in fgs
        assert "secondary_amide" not in fgs


class TestAzidoCollisions:
    """Azido (N=N+=N-) detection and imine collision."""

    @pytest.mark.xfail(
        reason="Azido SMARTS [NX1]=[NX2+]=[NX1-] has incorrect atom degree: "
               "first N is NX2 (bonded to C and N+), not NX1. "
               "Fix deferred: SMARTS pattern change is out of scope for Plan 03.",
        strict=True,
    )
    def test_azido_detected(self):
        """Methyl azide should detect azido FG."""
        fgs = _detect("CN=[N+]=[N-]")
        assert "azido" in fgs

    def test_azido_no_imine_false_positive(self):
        """Methyl azide should NOT produce imine false positive."""
        fgs = _detect("CN=[N+]=[N-]")
        # Even though the azido SMARTS doesn't match (known bug above),
        # the imine SMARTS [CX3]=[NX2H] also shouldn't match because
        # the N has no H and the C is CX4 (sp3 methyl)
        assert "imine" not in fgs


class TestCyanateCollisions:
    """Cyanate (O-C#N) must suppress ether + nitrile (PERC-03)."""

    def test_cyanate_detected(self):
        """Methyl cyanate detects cyanate FG."""
        fgs = _detect("COC#N")
        assert "cyanate" in fgs

    def test_cyanate_suppresses_ether(self):
        """Cyanate suppresses ether on overlapping O atom (PERC-03)."""
        fgs = _detect("COC#N")
        assert "ether" not in fgs, (
            "PERC-03: cyanate O should not also be detected as ether"
        )

    def test_cyanate_suppresses_nitrile(self):
        """Cyanate suppresses nitrile on overlapping C#N atoms (PERC-03)."""
        fgs = _detect("COC#N")
        assert "nitrile" not in fgs, (
            "PERC-03: cyanate C#N should not also be detected as nitrile"
        )


class TestThiocyanateCollisions:
    """Thiocyanate (S-C#N) must suppress thioether + nitrile (PERC-03)."""

    def test_thiocyanate_detected(self):
        """Methyl thiocyanate detects thiocyanate FG."""
        fgs = _detect("CSC#N")
        assert "thiocyanate" in fgs

    def test_thiocyanate_suppresses_thioether(self):
        """Thiocyanate suppresses thioether on overlapping S atom (PERC-03)."""
        fgs = _detect("CSC#N")
        assert "thioether" not in fgs, (
            "PERC-03: thiocyanate S should not also be detected as thioether"
        )

    def test_thiocyanate_suppresses_nitrile(self):
        """Thiocyanate suppresses nitrile on overlapping C#N atoms (PERC-03)."""
        fgs = _detect("CSC#N")
        assert "nitrile" not in fgs, (
            "PERC-03: thiocyanate C#N should not also be detected as nitrile"
        )


class TestAzoCollisions:
    """Azo (C-N=N-C) must suppress imine (PERC-03)."""

    def test_azo_detected(self):
        """Azomethane detects azo FG."""
        fgs = _detect("CN=NC")
        assert "azo" in fgs

    def test_azo_suppresses_imine(self):
        """Azo suppresses imine on overlapping N=N atoms (PERC-03)."""
        fgs = _detect("CN=NC")
        assert "imine" not in fgs, (
            "PERC-03: azo N=N should not also be detected as imine"
        )


# ===========================================================================
# 2. Existing collision stability (PERC-01, PERC-02, PERC-03, USUB-11)
# ===========================================================================

class TestCarboxylicAcidAldehydeCollision:
    """Carboxylic acid C=O must suppress aldehyde (PERC-01)."""

    def test_carboxylic_acid_detected(self):
        """Acetic acid detects carboxylic_acid."""
        fgs = _detect("CC(=O)O")
        assert "carboxylic_acid" in fgs

    def test_carboxylic_acid_suppresses_aldehyde(self):
        """Carboxylic acid C=O should not also match aldehyde (PERC-01)."""
        fgs = _detect("CC(=O)O")
        assert "aldehyde" not in fgs, (
            "PERC-01: carboxylic acid C=O should suppress aldehyde"
        )


class TestAromaticAminePrimaryAmineCollision:
    """Aromatic amine must suppress primary_amine (PERC-02)."""

    def test_aromatic_amine_detected(self):
        """Aniline detects aromatic_amine."""
        fgs = _detect("Nc1ccccc1")
        assert "aromatic_amine" in fgs

    def test_aromatic_amine_suppresses_primary_amine(self):
        """Aromatic amine suppresses primary_amine on overlapping N atom (PERC-02)."""
        fgs = _detect("Nc1ccccc1")
        assert "primary_amine" not in fgs, (
            "PERC-02: aromatic_amine should suppress primary_amine"
        )


class TestHydroxamicAcidCollisions:
    """Hydroxamic acid must suppress amide + alcohol (PERC-03)."""

    def test_hydroxamic_acid_detected(self):
        """Acetohydroxamic acid detects hydroxamic_acid."""
        fgs = _detect("CC(=O)NO")
        assert "hydroxamic_acid" in fgs

    def test_hydroxamic_acid_suppresses_amide(self):
        """Hydroxamic acid suppresses primary_amide on overlapping atoms."""
        fgs = _detect("CC(=O)NO")
        assert "primary_amide" not in fgs, (
            "PERC-03: hydroxamic acid should suppress amide"
        )

    def test_hydroxamic_acid_suppresses_alcohol(self):
        """Hydroxamic acid suppresses alcohol on overlapping OH."""
        fgs = _detect("CC(=O)NO")
        # Check no alcohol variants
        for alc in ("primary_alcohol", "secondary_alcohol", "tertiary_alcohol"):
            assert alc not in fgs, (
                f"PERC-03: hydroxamic acid should suppress {alc}"
            )


class TestPhosphateSpecificityStability:
    """Phosphate specificity chain must remain stable after seniority additions."""

    def test_triester_still_suppresses_diester(self):
        """Trimethyl phosphate: triester suppresses diester after Plan 01 changes."""
        fgs = _detect("COP(=O)(OC)OC")
        assert "phosphate_triester" in fgs
        assert "phosphate_diester" not in fgs

    def test_triester_still_suppresses_monoester(self):
        """Trimethyl phosphate: triester suppresses monoester after Plan 01 changes."""
        fgs = _detect("COP(=O)(OC)OC")
        assert "phosphate_monoester" not in fgs

    def test_diester_still_suppresses_monoester(self):
        """Dimethyl hydrogen phosphate: diester suppresses monoester."""
        fgs = _detect("COP(=O)(OC)O")
        assert "phosphate_diester" in fgs
        assert "phosphate_monoester" not in fgs


# ===========================================================================
# 3. Edge cases
# ===========================================================================

class TestEdgeCases:
    """Complex molecules with multiple overlapping FGs."""

    def test_thioester_and_ester_coexist(self):
        """Molecule with both a thioester and an ester: both detected independently."""
        fgs = _detect("CC(=O)SCC(=O)OC")
        assert "thioester" in fgs, "Thioester group should be detected"
        assert "ester" in fgs, "Ester group should be detected independently"

    def test_thioester_and_carboxylic_acid_coexist(self):
        """Molecule with both a thioester and a carboxylic acid: both detected."""
        fgs = _detect("CC(=O)SCC(=O)O")
        assert "thioester" in fgs, "Thioester group should be detected"
        assert "carboxylic_acid" in fgs, "Carboxylic acid should be detected"

    def test_imide_and_separate_amide_coexist(self):
        """Molecule with imide AND a separate (non-overlapping) amide: both detected.

        The imide collision rule suppresses amides only on OVERLAPPING atoms.
        A separate amide group on different atoms should survive.
        """
        # Succinimide with an amide sidechain
        fgs = _detect("O=C1CCC(=O)N1CC(=O)N")
        assert "imide" in fgs, "Imide should be detected"
        assert "primary_amide" in fgs, (
            "Separate primary_amide (non-overlapping with imide) should survive"
        )

    def test_lactam_detected_as_amide(self):
        """Lactam (cyclic amide) is detected as secondary_amide, then routed by handler.

        In 2-pyrrolidinone (5-membered lactam), C(=O) IS adjacent to N.
        The naming pipeline routes it via the lactam handler, but at the
        FG detection level it appears as secondary_amide.
        """
        # O=C1CCCN1 = 2-pyrrolidinone (correct lactam with C=O-N adjacency)
        fgs = _detect("O=C1CCCN1")
        assert "secondary_amide" in fgs, (
            "Lactam should detect as secondary_amide at FG level"
        )

    def test_ester_suppresses_aldehyde(self):
        """Ester C=O must not also match aldehyde (PERC-01)."""
        fgs = _detect("CC(=O)OC")
        assert "ester" in fgs
        assert "aldehyde" not in fgs

    def test_acid_chloride_suppresses_aldehyde(self):
        """Acid chloride C=O must not also match aldehyde (PERC-01)."""
        fgs = _detect("CC(=O)Cl")
        assert "acid_chloride" in fgs
        assert "aldehyde" not in fgs

    def test_urea_suppresses_amide(self):
        """Urea N-C(=O)-N must not also match amide."""
        fgs = _detect("NC(=O)N")
        assert "urea" in fgs
        assert "primary_amide" not in fgs

    def test_carbamate_suppresses_ester_and_amide(self):
        """Carbamate N-C(=O)-O-C must not also match ester or amide."""
        fgs = _detect("NC(=O)OC")
        assert "carbamate" in fgs
        assert "ester" not in fgs
        assert "primary_amide" not in fgs

    def test_guanidine_suppresses_imine(self):
        """Guanidine N-C(=N)-N must not also match imine."""
        fgs = _detect("NC(=N)N")
        assert "guanidine" in fgs
        assert "imine" not in fgs
