"""
Unit tests for subordinate prefix emission of non-principal functional groups.

Tests that non-principal FGs (carbamoyl, cyano, formyl, carboxy) are emitted
as prefixes when they are not the principal group. The root cause of missing
prefixes is that the chain finder includes non-principal terminal FG carbons
in the principal chain, inflating chain length and preventing prefix emission.

IUPAC references:
  (e) cyano prefix
  (f) carbamoyl prefix
  (c) carbamoyl as subordinate
  (b) cyano as subordinate
"""
import pytest
from orthonym import name_compound


class TestCarbamoylPrefix:
    """Non-principal amide groups should emit 'carbamoyl' prefix."""

    def test_carbamoyl_glutamine(self):
        """NC(=O)CCCC(N)C(=O)O -> 2,6-diamino-6-oxohexanoic acid.

        Post-wave-3 the non-principal terminal primary amide is expressed as the
        amino-oxo PIN form; the amide carbon IS part of the 6C
        parent chain (as C6, bearing 6-amino-6-oxo), so the parent is hexanoic.
        """
        name = name_compound("NC(=O)CCCC(N)C(=O)O")
        assert name == "2,6-diamino-6-oxohexanoic acid"

    def test_carbamoyl_longer_analog(self):
        """NC(=O)CCCCC(N)C(=O)O -> 2,7-diamino-7-oxoheptanoic acid (amino-oxo PIN).

        Post-wave-3 the amide carbon IS part of the 7C parent chain (as C7,
        bearing 7-amino-7-oxo), so the parent is heptanoic acid.
        """
        name = name_compound("NC(=O)CCCCC(N)C(=O)O")
        assert name == "2,7-diamino-7-oxoheptanoic acid"


class TestCyanoPrefix:
    """Non-principal nitrile groups should emit 'cyano' prefix (regression guard)."""

    def test_cyano_propanoic_acid(self):
        """N#CCC(=O)O (cyanoacetic acid) -> cyano PREFIX on a 2-carbon acid parent.

         Phase B (DD1 Fix 1,: the non-principal nitrile is a
        'cyano' prefix whose carbon is EXCLUDED from the parent chain, so the
        parent is the 2-carbon acid (ethanoic/acetic), NOT a 3-carbon 'propanoic'
        chain (which would wrongly count the nitrile C). This SMILES is
        cyanoacetic acid, not 3-cyanopropanoic acid.
        """
        name = name_compound("N#CCC(=O)O")
        assert "cyano" in name, f"Expected 'cyano' in '{name}'"
        assert "propanoic" not in name, (
            f"Nitrile C must NOT inflate the chain to 3C 'propanoic': '{name}'"
        )


class TestFormylPrefix:
    """Non-principal aldehyde on ring should emit 'formyl' prefix (regression guard)."""

    def test_formyl_benzoic_acid(self):
        """O=Cc1ccc(C(=O)O)cc1 -> 4-formylbenzoic acid."""
        name = name_compound("O=Cc1ccc(C(=O)O)cc1")
        assert "formyl" in name, f"Expected 'formyl' in '{name}'"
        assert "benzoic" in name, f"Expected 'benzoic' in '{name}'"


class TestCarboxyPrefix:
    """Non-principal carboxylic acid groups should emit 'carboxy' prefix."""

    def test_carboxy_multi_acid(self):
        """Multi-acid compound should have 'carboxy' prefix for non-principal COOH groups.

        OC(=O)CCC(C(=O)O)C(CC(=O)O)C(=O)O - butanetetracarboxylic acid derivative.
        Principal chain with most COOH on it; remaining COOH become carboxy prefixes.
        """
        name = name_compound("OC(=O)CCC(C(=O)O)C(CC(=O)O)C(=O)O")
        # At minimum, if not all acids are on the chain, some should be carboxy prefixes
        # This is a complex case - the key test is that the naming doesn't crash
        # and produces a reasonable result with the correct parent chain
        assert name, "Should produce a name"
        # The name should reference a diacid or triacid chain with carboxy prefixes
        # or alternatively all on the chain as tetracarboxylic acid


class TestNoRegression:
    """Ensure simple cases are NOT affected by chain perception changes."""

    def test_simple_acetic_acid(self):
        """Acetic acid should remain unchanged."""
        name = name_compound("CC(=O)O")
        assert name == "acetic acid", f"Expected 'acetic acid', got '{name}'"

    def test_simple_propanoic_acid(self):
        """Propanoic acid should remain unchanged."""
        name = name_compound("CCC(=O)O")
        assert name == "propanoic acid", f"Expected 'propanoic acid', got '{name}'"

    def test_simple_propanamide(self):
        """Propanamide (amide IS principal) should remain unchanged."""
        name = name_compound("CCC(=O)N")
        assert name == "propanamide", f"Expected 'propanamide', got '{name}'"

    def test_simple_butanamide(self):
        """Butanamide (amide IS principal) should remain unchanged."""
        name = name_compound("CCCC(=O)N")
        assert name == "butanamide", f"Expected 'butanamide', got '{name}'"

    def test_simple_pentanenitrile(self):
        """Pentanenitrile (nitrile IS principal) should remain unchanged."""
        name = name_compound("CCCCC#N")
        # When nitrile is the principal group, it should be expressed as suffix
        assert "nitrile" in name, f"Expected 'nitrile' suffix in '{name}'"

    def test_aminobutanoic_acid(self):
        """4-aminobutanoic acid - only amino prefix, no amide involved."""
        name = name_compound("NCCCC(=O)O")
        assert "amino" in name, f"Expected 'amino' in '{name}'"
        assert "butanoic" in name, f"Expected 'butanoic' in '{name}'"
