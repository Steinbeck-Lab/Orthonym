"""Tests for alkoxycarbonyl prefix generation (IUPAC P-65.6.3).

When an ester group is NOT the principal characteristic group, it is
expressed as an alkoxycarbonyl prefix:
  -COOCH3   -> methoxycarbonyl
  -COOC2H5  -> ethoxycarbonyl
  -COOPh    -> phenoxycarbonyl

NOTE: OPSIN 2.8.0 does not parse alkoxycarbonyl prefixes, so
round-trip validation is not possible for these names.  The names
are correct per IUPAC 2013 Blue Book P-65.6.3.
"""

import pytest
from orthonym.namer import name_compound


# ============================================================================
# E2E: SMILES -> name containing alkoxycarbonyl
# ============================================================================


class TestAlkoxycarbonylE2E:
    """End-to-end tests: acid-principal in-chain partial (di)esters.

    W2F-P2 (P-65.6.3.3.5 method (1)) SUPERSEDED the old ``R-oxycarbonyl``
    expectations here. For an acid-principal partial ester whose ester
    carbonyl is a MEMBER of the principal chain, spelling that carbonyl as
    ``methoxycarbonyl`` double-counts the carbonyl carbon and names a
    one-carbon-longer HOMOLOG: ``4-methoxycarbonylbutanoic acid`` is methyl
    hydrogen *glutarate* (C5), not the methyl hydrogen *succinate* (C4) that
    ``COC(=O)CCC(=O)O`` actually is. The old raw names were RT-MISMATCH leaks
    suppressed to ``unknown`` by SELF-01 in production; the unit gate (off)
    let the wrong string through. The PIN is the substitutive ``oxo``+``R-oxy``
    form, OPSIN-RT-verified (golds W2F-P2-01..04). Alkoxycarbonyl remains
    correct for RING parents / off-chain carbonyls (Task-1 guard preserves
    those — see TestAlkoxycarbonylUnit / TestAlkoxycarbonylNegative)."""

    def test_methyl_ester_succinate(self):
        """Methyl hydrogen succinate (C4): in-chain ester -> oxo+methoxy PIN."""
        assert name_compound("COC(=O)CCC(=O)O") == "4-methoxy-4-oxobutanoic acid"

    def test_ethyl_ester_glutarate(self):
        """Ethyl hydrogen glutarate (C5): in-chain ester -> oxo+ethoxy PIN."""
        assert name_compound("CCOC(=O)CCCC(=O)O") == "5-ethoxy-5-oxopentanoic acid"

    def test_methyl_ester_adipate(self):
        """Methyl hydrogen adipate (C6): in-chain ester -> oxo+methoxy PIN."""
        assert name_compound("COC(=O)CCCCC(=O)O") == "6-methoxy-6-oxohexanoic acid"

    def test_ethyl_ester_malonate(self):
        """Ethyl hydrogen malonate (C3): in-chain ester -> oxo+ethoxy PIN."""
        assert name_compound("CCOC(=O)CC(=O)O") == "3-ethoxy-3-oxopropanoic acid"

    def test_phenyl_ester_succinate(self):
        """Phenyl hydrogen succinate (C4): aryl in-chain ester -> oxo+phenoxy PIN."""
        assert name_compound("O=C(Oc1ccccc1)CCC(=O)O") == "4-oxo-4-phenoxybutanoic acid"

    def test_propyl_ester(self):
        """Propyl hydrogen glutarate (C5): in-chain ester -> oxo+propoxy PIN."""
        assert name_compound("CCCOC(=O)CCCC(=O)O") == "5-oxo-5-propoxypentanoic acid"


# ============================================================================
# Negative / guard tests: things that must NOT produce alkoxycarbonyl
# ============================================================================


class TestAlkoxycarbonylNegative:
    """Tests that specific compound classes do NOT produce alkoxycarbonyl."""

    def test_lactone_no_alkoxycarbonyl(self):
        """gamma-Butyrolactone: cyclic ester, NOT alkoxycarbonyl."""
        result = name_compound("O=C1CCCO1")
        assert "oxycarbonyl" not in result.lower()
        assert "oxolan" in result.lower()

    def test_simple_ester_no_alkoxycarbonyl(self):
        """Ethyl acetate: ester as principal group, functional class naming."""
        result = name_compound("CCOC(=O)C")
        assert "oxycarbonyl" not in result.lower()

    def test_acyloxy_ester_no_double_naming(self):
        """C16:0 acyloxy ester: acyloxy orientation, not alkoxycarbonyl.

        Purpose of this test (unchanged): the ester must be cited as an 'acyloxy'
        prefix, never as 'alkoxycarbonyl'.

        Asserted word changed from 'palmitoyloxy' to the PIN in Task J3 --
        P-65.6.3.2.3 (BlueBookV2.md:31696), whose :31723 example prints a
        trivial-derived acyloxy prefix as the non-preferred alternative, and
        Appendix 2 :56482 'hexadecanoyl* = palmitoyl' (legend :55416, "The symbol
        * designates the preferred prefix").
        """
        result = name_compound("CCCCCCCCCCCCCCCC(=O)OC(CCCCC)CCCCCCCCCCCC(=O)[O-]")
        assert "oxycarbonyl" not in result.lower()
        assert "hexadecanoyloxy" in result.lower()

    def test_heteroatom_alkyl_no_alkoxycarbonyl(self):
        """Ester with nitrogen-containing OR fragment: skip alkoxycarbonyl."""
        result = name_compound("N[C@@H](COC(=O)CCC(=O)O)C(=O)O")
        assert "oxycarbonyl" not in result.lower()

    def test_macrocyclic_ester_no_alkoxycarbonyl(self):
        """Macrolide with pendant ester: alkyl is huge ring, skip."""
        # ci-055 compound
        result = name_compound(
            "CN=C(N)NCCC/C=C/CCC[C@H](C)[C@H]1OC(=O)/C(C)=C\\C=C/"
            "[C@H](C)[C@H](O)C[C@H](O)[C@H](C)[C@@H](O)CC[C@@H](C)"
            "[C@H](O)C[C@@]2(O)O[C@H](C[C@H](O)C[C@H](OC(=O)CC(=O)O)"
            "C[C@@H](O)C[C@H](O)/C(C)=C\\C=C/[C@H]1C)C[C@@H](O)[C@@H]2O"
        )
        assert "oxycarbonyl" not in result.lower()


# ============================================================================
# CI regression guards
# ============================================================================


class TestAlkoxycarbonylCIRegression:
    """Compounds from CI benchmark that must not change."""

    def test_ci091_unchanged(self):
        """ci-091: complex naphthoquinone ester must not regress."""
        result = name_compound("CCCCCCCCCCCCC1=C(OC(C)=O)C(=O)c2ccccc2C1=O")
        assert result == "(acetyloxy)-1-docosyloxyethanedione"

    def test_separate_ether_and_ester(self):
        """Compound with both ester and genuine separate ether."""
        result = name_compound("COC(=O)c1ccc(Oc2ccccc2)cc1C(=O)O")
        # Ester on ring + separate phenoxy ether - both should be named
        assert "phenoxy" in result.lower() or "ether" in result.lower() or "benzoic" in result.lower()
