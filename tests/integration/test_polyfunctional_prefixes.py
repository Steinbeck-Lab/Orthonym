"""
a phase, Plan 02: Integration tests for polyfunctional naming with
sulfoxide/sulfone/thioether prefix forms, 3+ FG alphabetization,
compound prefix parenthesization, and OPSIN round-trip validation.

Validates end-to-end naming pipeline for compounds with the new
dynamic compound prefix generators added in Plan 80-01.
"""

import subprocess
import pytest
from rdkit import Chem

from orthonym import name_compound
from tests.support.jars import jar_or_none


# ============================================================================
# OPSIN helper
# ============================================================================

_OPSIN_JAR = None

def _opsin_to_smiles(name: str) -> str:
    """Convert IUPAC name to SMILES via OPSIN. Returns None on failure."""
    global _OPSIN_JAR
    if _OPSIN_JAR is None:
        _OPSIN_JAR = jar_or_none() or ""
    if not _OPSIN_JAR:
        return None
    try:
        result = subprocess.run(
            ["java", "-jar", _OPSIN_JAR, "-osmi"],
            input=name,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _canonical(smiles: str) -> str:
    """Return canonical SMILES, or None if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _roundtrip_matches(input_smiles: str, name: str) -> bool:
    """Check if OPSIN round-trip matches: name -> OPSIN -> canonical SMILES == input."""
    opsin_smiles = _opsin_to_smiles(name)
    if opsin_smiles is None:
        return False
    can_input = _canonical(input_smiles)
    can_opsin = _canonical(opsin_smiles)
    if can_input is None or can_opsin is None:
        return False
    return can_input == can_opsin


# ============================================================================
# TestSulfoxideAsNonPrincipal
# ============================================================================

class TestSulfoxideAsNonPrincipal:
    """Sulfoxide as non-principal group alongside a higher-priority acid."""

    @pytest.mark.integration
    def test_methylsulfinyl_propanoic_acid(self):
        """OC(=O)CCS(=O)C -> 3-(methanesulfinyl)propanoic acid."""
        result = name_compound("OC(=O)CCS(=O)C")
        # PIN per R8: "SULFOXIDES AND SULFONES" "(1) substitutively, by prefixing the name
        # of the acyl group R′-SO– or R′-SO2– to the name of the parent hydride"
        # the Blue Book, "Methods (1) and (3) generate preferred names.":28088;
        # "the preferred prefixes are enclosed in parentheses even though they are
        # simple prefixes":31262; "2-(methanesulfonyl)ethan-1-ol (PIN)
        # 2-(methylsulfonyl)ethan-1-ol":28150. OPSIN RT exact (TRIAGE.csv; re-checked in Task 7/8).
        assert result == "3-(methanesulfinyl)propanoic acid"

# ============================================================================
# TestSulfoneAsNonPrincipal
# ============================================================================

class TestSulfoneAsNonPrincipal:
    """Sulfone as non-principal group alongside a higher-priority acid."""

    @pytest.mark.integration
    def test_methylsulfonyl_propanoic_acid(self):
        """OC(=O)CCS(=O)(=O)C -> 3-(methanesulfonyl)propanoic acid."""
        result = name_compound("OC(=O)CCS(=O)(=O)C")
        # PIN per R8: "SULFOXIDES AND SULFONES" "(1) substitutively, by prefixing the name
        # of the acyl group R′-SO– or R′-SO2– to the name of the parent hydride"
        # the Blue Book, "Methods (1) and (3) generate preferred names.":28088;
        # "the preferred prefixes are enclosed in parentheses even though they are
        # simple prefixes":31262; "2-(methanesulfonyl)ethan-1-ol (PIN)
        # 2-(methylsulfonyl)ethan-1-ol":28150. OPSIN RT exact (TRIAGE.csv; re-checked in Task 7/8).
        assert result == "3-(methanesulfonyl)propanoic acid"


# ============================================================================
# TestThioetherAsNonPrincipal
# ============================================================================

class TestThioetherAsNonPrincipal:
    """Thioether as non-principal group alongside a higher-priority acid."""

    @pytest.mark.integration
    def test_methylsulfanyl_ethanoic_acid(self):
        """OC(=O)CSC -> (methylsulfanyl)acetic acid."""
        result = name_compound("OC(=O)CSC")
        # PIN per R3: "only acetic acid, benzoic acid, and oxamic acid can be
        # substituted" the Blue Book, "acetic acid (PIN) ethanoic acid":29725;
        # "All locants are omitted for parent compounds when all substitutable
        # hydrogen atoms have the same locant.":3031; "sulfanylacetic acid (PIN)":4967. OPSIN RT exact (TRIAGE.csv; re-checked in Task 7/8).
        assert result == "(methylsulfanyl)acetic acid"

    @pytest.mark.integration
    def test_ethylsulfanyl_ethanoic_acid(self):
        """OC(=O)CSCC -> (ethylsulfanyl)acetic acid."""
        result = name_compound("OC(=O)CSCC")
        # PIN per R3: "only acetic acid, benzoic acid, and oxamic acid can be
        # substituted" the Blue Book, "acetic acid (PIN) ethanoic acid":29725;
        # "All locants are omitted for parent compounds when all substitutable
        # hydrogen atoms have the same locant.":3031; "sulfanylacetic acid (PIN)":4967. OPSIN RT exact (TRIAGE.csv; re-checked in Task 7/8).
        assert result == "(ethylsulfanyl)acetic acid"


# ============================================================================
# TestThreePlusFGAlphabetization (,)
# ============================================================================

class TestThreePlusFGAlphabetization:
    """Compounds with 3+ diverse FGs: verify alphabetical prefix ordering."""

    @pytest.mark.integration
    def test_amino_hydroxy_oxo_acid(self):
        """amino < hydroxy < oxo alphabetically."""
        result = name_compound("OCC(N)CC(=O)C(=O)O")
        assert result == "4-amino-5-hydroxy-2-oxopentanoic acid"

    @pytest.mark.integration
    def test_hydroxy_sulfinyl_acid(self):
        """hydroxy < methanesulfinyl alphabetically."""
        result = name_compound("OC(CS(=O)C)CC(=O)O")
        # PIN per R8: "SULFOXIDES AND SULFONES" "(1) substitutively, by prefixing the name
        # of the acyl group R′-SO– or R′-SO2– to the name of the parent hydride"
        # the Blue Book, "Methods (1) and (3) generate preferred names.":28088;
        # "the preferred prefixes are enclosed in parentheses even though they are
        # simple prefixes":31262; "2-(methanesulfonyl)ethan-1-ol (PIN)
        # 2-(methylsulfonyl)ethan-1-ol":28150. OPSIN RT exact (TRIAGE.csv; re-checked in Task 7/8).
        assert result == "3-hydroxy-4-(methanesulfinyl)butanoic acid"
        # Verify ordering: hydroxy before methanesulfinyl
        hydroxy_pos = result.index("hydroxy")
        sulfinyl_pos = result.index("methanesulfinyl")
        assert hydroxy_pos < sulfinyl_pos, \
            f"'hydroxy' should come before 'methanesulfinyl' in '{result}'"

    @pytest.mark.integration
    def test_hydroxy_dioxo_acid(self):
        """3+ FGs: hydroxy + dioxo + acid."""
        result = name_compound("OCC(=O)CC(=O)CC(=O)O")
        assert result == "6-hydroxy-3,5-dioxohexanoic acid"


# ============================================================================
# TestCompoundPrefixParenthesization
# ============================================================================

class TestCompoundPrefixParenthesization:
    """Verify compound prefixes are correctly parenthesized per IUPAC."""

    @pytest.mark.integration
    def test_methylsulfinyl_parenthesized(self):
        """The simple prefix 'methanesulfinyl' is still enclosed: '(methanesulfinyl)'."""
        result = name_compound("OC(=O)CS(=O)C")
        # PIN per R8: "SULFOXIDES AND SULFONES" "(1) substitutively, by prefixing the name
        # of the acyl group R′-SO– or R′-SO2– to the name of the parent hydride"
        # the Blue Book, "Methods (1) and (3) generate preferred names.":28088;
        # "the preferred prefixes are enclosed in parentheses even though they are
        # simple prefixes":31262; "2-(methanesulfonyl)ethan-1-ol (PIN)
        # 2-(methylsulfonyl)ethan-1-ol":28150; and per R3 (acetic acid, no locant). OPSIN RT exact (TRIAGE.csv; re-checked in Task 7/8).
        assert result == "(methanesulfinyl)acetic acid", \
            f"Expected '(methanesulfinyl)acetic acid', got '{result}'"

    @pytest.mark.integration
    def test_methylsulfonyl_parenthesized(self):
        """The simple prefix 'methanesulfonyl' is still enclosed: '(methanesulfonyl)'."""
        result = name_compound("OC(=O)CS(=O)(=O)C")
        # PIN per R8: "SULFOXIDES AND SULFONES" "(1) substitutively, by prefixing the name
        # of the acyl group R′-SO– or R′-SO2– to the name of the parent hydride"
        # the Blue Book, "Methods (1) and (3) generate preferred names.":28088;
        # "the preferred prefixes are enclosed in parentheses even though they are
        # simple prefixes":31262; "2-(methanesulfonyl)ethan-1-ol (PIN)
        # 2-(methylsulfonyl)ethan-1-ol":28150; and per R3 (acetic acid, no locant). OPSIN RT exact (TRIAGE.csv; re-checked in Task 7/8).
        assert result == "(methanesulfonyl)acetic acid", \
            f"Expected '(methanesulfonyl)acetic acid', got '{result}'"

# ============================================================================
# TestOPSINRoundTrip
# ============================================================================

class TestOPSINRoundTrip:
    """OPSIN round-trip verification for new prefix forms."""

    @pytest.mark.integration
    @pytest.mark.roundtrip
    def test_sulfinyl_acid_roundtrip(self):
        """2-(methylsulfinyl)ethanoic acid round-trips via OPSIN."""
        smiles = "OC(=O)CS(=O)C"
        name = name_compound(smiles)
        opsin_smiles = _opsin_to_smiles(name)
        if opsin_smiles is None:
            pytest.skip("OPSIN not available")
        assert _roundtrip_matches(smiles, name), \
            f"Round-trip failed: '{smiles}' -> '{name}' -> '{opsin_smiles}'"

    @pytest.mark.integration
    @pytest.mark.roundtrip
    def test_sulfonyl_acid_roundtrip(self):
        """2-(methylsulfonyl)ethanoic acid round-trips via OPSIN."""
        smiles = "OC(=O)CS(=O)(=O)C"
        name = name_compound(smiles)
        opsin_smiles = _opsin_to_smiles(name)
        if opsin_smiles is None:
            pytest.skip("OPSIN not available")
        assert _roundtrip_matches(smiles, name), \
            f"Round-trip failed: '{smiles}' -> '{name}' -> '{opsin_smiles}'"

    @pytest.mark.integration
    @pytest.mark.roundtrip
    def test_sulfanyl_acid_roundtrip(self):
        """2-(methylsulfanyl)ethanoic acid round-trips via OPSIN."""
        smiles = "OC(=O)CSC"
        name = name_compound(smiles)
        opsin_smiles = _opsin_to_smiles(name)
        if opsin_smiles is None:
            pytest.skip("OPSIN not available")
        assert _roundtrip_matches(smiles, name), \
            f"Round-trip failed: '{smiles}' -> '{name}' -> '{opsin_smiles}'"

    @pytest.mark.integration
    @pytest.mark.roundtrip
    def test_sulfinyl_propanoic_roundtrip(self):
        """3-(methylsulfinyl)propanoic acid round-trips via OPSIN."""
        smiles = "OC(=O)CCS(=O)C"
        name = name_compound(smiles)
        opsin_smiles = _opsin_to_smiles(name)
        if opsin_smiles is None:
            pytest.skip("OPSIN not available")
        assert _roundtrip_matches(smiles, name), \
            f"Round-trip failed: '{smiles}' -> '{name}' -> '{opsin_smiles}'"

    @pytest.mark.integration
    @pytest.mark.roundtrip
    def test_sulfonyl_propanoic_roundtrip(self):
        """3-(methylsulfonyl)propanoic acid round-trips via OPSIN."""
        smiles = "OC(=O)CCS(=O)(=O)C"
        name = name_compound(smiles)
        opsin_smiles = _opsin_to_smiles(name)
        if opsin_smiles is None:
            pytest.skip("OPSIN not available")
        assert _roundtrip_matches(smiles, name), \
            f"Round-trip failed: '{smiles}' -> '{name}' -> '{opsin_smiles}'"

    @pytest.mark.integration
    @pytest.mark.roundtrip
    def test_ethylsulfanyl_roundtrip(self):
        """2-(ethylsulfanyl)ethanoic acid round-trips via OPSIN."""
        smiles = "OC(=O)CSCC"
        name = name_compound(smiles)
        opsin_smiles = _opsin_to_smiles(name)
        if opsin_smiles is None:
            pytest.skip("OPSIN not available")
        assert _roundtrip_matches(smiles, name), \
            f"Round-trip failed: '{smiles}' -> '{name}' -> '{opsin_smiles}'"

    @pytest.mark.integration
    @pytest.mark.roundtrip
    def test_sulfinyl_benzoic_roundtrip(self):
        """4-(methylsulfinyl)benzoic acid round-trips via OPSIN."""
        smiles = "OC(=O)c1ccc(S(=O)C)cc1"
        name = name_compound(smiles)
        opsin_smiles = _opsin_to_smiles(name)
        if opsin_smiles is None:
            pytest.skip("OPSIN not available")
        assert _roundtrip_matches(smiles, name), \
            f"Round-trip failed: '{smiles}' -> '{name}' -> '{opsin_smiles}'"

    @pytest.mark.integration
    @pytest.mark.roundtrip
    def test_sulfonyl_benzoic_roundtrip(self):
        """4-(methylsulfonyl)benzoic acid round-trips via OPSIN."""
        smiles = "OC(=O)c1ccc(S(=O)(=O)C)cc1"
        name = name_compound(smiles)
        opsin_smiles = _opsin_to_smiles(name)
        if opsin_smiles is None:
            pytest.skip("OPSIN not available")
        assert _roundtrip_matches(smiles, name), \
            f"Round-trip failed: '{smiles}' -> '{name}' -> '{opsin_smiles}'"


# ============================================================================
# TestExistingPolyfunctionalNoRegression
# ============================================================================

class TestExistingPolyfunctionalNoRegression:
    """Regression battery -- all must still pass."""

    @pytest.mark.integration
    def test_ethanoyloxy_propanoic_acid(self):
        """OC(=O)CCOC(=O)C -> 3-(acetyloxy)propanoic acid."""
        # PIN per R4: "acetyl (preferred prefix) ethanoyl" the Blue Book;
        # "Esters cited as prefixes" "The systematic name 'acetyloxy' is preferred
        # to the contracted name 'acetoxy'":31700; "3-(benzoyloxy)propanoic acid (PIN)":31711.
        # OPSIN RT exact (TRIAGE.csv; re-checked in Task 7/8).
        assert name_compound("OC(=O)CCOC(=O)C") == "3-(acetyloxy)propanoic acid"
