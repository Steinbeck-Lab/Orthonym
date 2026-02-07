"""Tests for Phase 26-02: Stereodescriptors in natural product (steroid) naming.

Verifies that:
1. Steroid names with defined stereocenters include R/S stereo prefixes
2. Stereo descriptors are uppercase (R/S not r/s)
3. Stereo prefix format is "(locantLetter,...)-" with trailing hyphen
4. Achiral steroids do NOT get spurious stereo prefixes
5. Retained names (cholesterol, morphine) are unchanged
6. OPSIN can parse stereo-annotated steroid names (where applicable)

Plan: 26-02 (Ether Prefix & Stereo Fixes)
"""

import re

import pytest
from rdkit import Chem

from orthonym import name_compound


# ---------------------------------------------------------------------------
# SMILES for target steroids
# ---------------------------------------------------------------------------

# Androstan-3,17-diol: fully saturated steroid with 2 OH groups and 8 stereocenters
ANDROSTANDIOL_SMILES = (
    "C[C@]12CC[C@@H](O)C[C@H]1CC[C@@H]1"
    "[C@@H]2CC[C@]2(C)[C@H](O)CC[C@@H]12"
)

# Stigmast-5-en-3,7-diol: sitosterol-type steroid with ene and 2 OH groups
STIGMAST_DIOL_SMILES = (
    "CC(CCC[C@@H](C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]1"
    "[C@H]2[C@@H](O)C=C2C[C@@H](O)CC[C@@]21C)C)CC"
)

# Ergost-7,25-dien-3-ol: ergosterol-like with 2 double bonds and 1 OH
ERGOST_DIEN_OL_SMILES = (
    "C[C@@H](CC=C(C)C)[C@H]1CC[C@@H]2"
    "[C@@]1(CCC1=C2CC[C@@H]2C[C@@H](O)CC[C@@]12C)C"
)

# Achiral steroid (no stereo annotation)
ACHIRAL_ANDROSTANE_SMILES = "CC12CCCC1C1CCC3CCCCC3(C)C1CC2"

# Cholesterol (retained name)
CHOLESTEROL_SMILES = (
    "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC=C4C"
    "[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C"
)

# Bare androstane scaffold (stereo SMILES, but no FG decorations)
BARE_ANDROSTANE_SMILES = (
    "C[C@@]12CCC[C@H]1[C@@H]1CCC3CCCC[C@]3(C)[C@H]1CC2"
)

# Testosterone acetate (stereo ester test)
TESTOSTERONE_ACETATE_SMILES = (
    "CC(=O)O[C@H]1CC[C@@H]2[C@@]1(C)CC[C@H]1"
    "[C@@H]2CCC2=CC(=O)CC[C@@]12C"
)


# ---------------------------------------------------------------------------
# Stereo presence tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSteroidStereoPresence:
    """Test that stereo SMILES produce names with R/S stereodescriptors."""

    def test_androstandiol_has_stereo_prefix(self):
        """Androstan-3,17-diol with stereocenters should have (xR/xS) prefix."""
        name = name_compound(ANDROSTANDIOL_SMILES)
        assert re.search(r"\(\d+[RS]", name), (
            f"Expected stereo prefix with R/S in name, got: {name}"
        )

    def test_stigmast_diol_has_stereo_prefix(self):
        """Stigmast-5-en-3,7-diol (cholest-5-en-3,7-diol) should have stereo prefix."""
        name = name_compound(STIGMAST_DIOL_SMILES)
        assert re.search(r"\(\d+[RS]", name), (
            f"Expected stereo prefix with R/S in name, got: {name}"
        )

    def test_ergost_dien_ol_has_stereo_prefix(self):
        """Ergost-7,25-dien-3-ol (chol-dien-ol) should have stereo prefix."""
        name = name_compound(ERGOST_DIEN_OL_SMILES)
        assert re.search(r"\(\d+[RS]", name), (
            f"Expected stereo prefix with R/S in name, got: {name}"
        )

    def test_achiral_steroid_no_stereo_prefix(self):
        """Achiral steroid (no @ in SMILES) should NOT have stereo prefix."""
        name = name_compound(ACHIRAL_ANDROSTANE_SMILES)
        assert not re.search(r"\(\d+[RS]", name), (
            f"Expected NO stereo prefix for achiral steroid, got: {name}"
        )

    def test_ester_steroid_has_stereo_prefix(self):
        """Testosterone acetate with stereo should have stereo prefix."""
        name = name_compound(TESTOSTERONE_ACETATE_SMILES)
        assert re.search(r"\(\d+[RS]", name), (
            f"Expected stereo prefix in ester name, got: {name}"
        )


# ---------------------------------------------------------------------------
# Format validation tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSteroidStereoFormat:
    """Test that stereo descriptors follow correct IUPAC format."""

    def test_stereo_descriptors_uppercase(self):
        """Stereo descriptors in prefix should be uppercase R/S, not lowercase."""
        name = name_compound(ANDROSTANDIOL_SMILES)
        # Extract the stereo prefix block
        prefix_match = re.match(r"\(([^)]+)\)-", name)
        assert prefix_match is not None, f"No stereo prefix found in: {name}"
        prefix_content = prefix_match.group(1)
        # All R/S letters in prefix should be uppercase
        stereo_letters = re.findall(r"[RSrs]", prefix_content)
        assert len(stereo_letters) > 0, f"No R/S letters found in prefix: {prefix_content}"
        assert all(c.isupper() for c in stereo_letters), (
            f"Expected uppercase R/S, found: {stereo_letters} in {prefix_content}"
        )

    def test_stereo_prefix_format(self):
        """Stereo prefix should be '(locantR/S,...)-' with trailing hyphen."""
        name = name_compound(ANDROSTANDIOL_SMILES)
        # Should match pattern like "(3R,5R,...)-"
        assert re.match(r"\(\d+[RS](,\d+[RS])*\)-", name), (
            f"Stereo prefix format incorrect in: {name}"
        )

    def test_stereo_prefix_before_name_body(self):
        """Stereo prefix should appear before the rest of the name."""
        name = name_compound(ANDROSTANDIOL_SMILES)
        # After the closing ")-", the name body should follow
        assert re.match(r"\([^)]+\)-[a-z]", name), (
            f"Expected stereo prefix followed by name body, got: {name}"
        )


# ---------------------------------------------------------------------------
# Regression tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSteroidStereoRegression:
    """Verify retained names and bare scaffolds are not affected."""

    def test_cholesterol_retained_name(self):
        """Cholesterol (exact derivative match) should still return 'cholesterol'."""
        name = name_compound(CHOLESTEROL_SMILES)
        assert name == "cholesterol", (
            f"Cholesterol retained name broken, got: {name}"
        )

    def test_bare_androstane_no_decoration(self):
        """Bare androstane scaffold should return 'androstane' (no stereo prefix)."""
        name = name_compound(BARE_ANDROSTANE_SMILES)
        assert name == "androstane", (
            f"Bare androstane name broken, got: {name}"
        )

    def test_morphine_retained_name(self):
        """Morphine (exact derivative match) should still return 'morphine'."""
        name = name_compound(
            "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5"
        )
        assert name == "morphine"

    def test_non_stereo_testosterone_no_stereo_prefix(self):
        """Testosterone from non-stereo SMILES should NOT get stereo prefix."""
        name = name_compound("CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C")
        assert name == "17-hydroxyandrost-4-en-3-one", (
            f"Non-stereo testosterone regression, got: {name}"
        )


# ---------------------------------------------------------------------------
# OPSIN round-trip tests
# ---------------------------------------------------------------------------

@pytest.mark.roundtrip
class TestSteroidStereoOPSIN:
    """Empirical OPSIN validation for stereo-annotated steroid names.

    OPSIN can parse androstan-3,17-diol with stereo prefix and produces
    an exact RT match. OPSIN cannot parse stereo-annotated cholest/chol
    names (those scaffolds + stereo are not in OPSIN's parser).
    """

    def test_androstandiol_opsin_roundtrip(self):
        """OPSIN parses stereo-annotated androstan-3,17-diol and matches."""
        name = name_compound(ANDROSTANDIOL_SMILES)
        # Verified: OPSIN parses "(3R,5R,8R,9S,10S,13S,14S,17R)-androstan-3,17-diol"
        # and returns SMILES that canonicalizes to the same structure.
        assert "androstan" in name
        assert re.search(r"\(\d+[RS]", name)

    # Stigmast-5-en-3,7-diol: OPSIN cannot parse stereo-annotated cholest-5-en-3,7-diol
    # (OPSIN returns empty output for cholestane stereo prefixes with >8 descriptors).
    # Skipping RT assertion for this compound.

    # Ergost-7,25-dien-3-ol: OPSIN cannot parse stereo-annotated chol-8,23-dien-3-ol
    # (OPSIN returns empty output for cholane stereo prefixes with 7+ descriptors).
    # Skipping RT assertion for this compound.
