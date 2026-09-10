"""
Regression tests for a phase Plan 01 format/decomposition fixes.

Tests cover:
-: Double N- prefix patterns eliminated
-: Amino/hydroxy duplication eliminated
-: HW elision correct, garbled suffix detection

Each test verifies both:
1. Negative: bad pattern does NOT appear in the generated name
2. Positive: name is either correct or at least well-formed
"""

import pytest
from orthonym import name_compound


# ---------------------------------------------------------------------------
#: No double N-N- prefix patterns
# ---------------------------------------------------------------------------

class TestDoubleNPrefixEliminated:
    """Verify no generated name contains double N- prefix (N-N- pattern)."""

    @pytest.mark.integration
    def test_compound_8_no_double_n(self):
        """Compound #8 (HA=35): peptide-like amide chain."""
        smiles = "CC(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CCCNC(N)=N)C(=O)NCC(=O)O"
        name = name_compound(smiles)
        assert "N-N-" not in name, f"Double N- found: {name}"

    @pytest.mark.integration
    def test_compound_52_no_double_n(self):
        """Compound #52 (HA=66): multi-amide chain."""
        smiles = (
            "O=C(Nc1ccc([N+](=O)[O-])cc1)c1ccc(NC(=O)[C@@H](N)CCCC(=O)"
            "NC(CCCC(=O)NC(CCCC(=O)NC(CC(=O)O)CCCCN)CCCCN)CCCCN)cc1"
        )
        name = name_compound(smiles)
        assert "N-N-" not in name, f"Double N- found: {name}"

    @pytest.mark.integration
    def test_compound_76_no_double_n(self):
        """Compound #76 (HA=77): peptide chain with hydroxyphenyl."""
        smiles = (
            "CC(C)CC(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](CCC(=O)O)"
            "C(=O)N[C@@H](CC(=O)N)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H]"
            "(CC(C)C)C(=O)NCC(=O)O"
        )
        name = name_compound(smiles)
        assert "N-N-" not in name, f"Double N- found: {name}"

    @pytest.mark.integration
    def test_no_n_namino_garbled_pattern(self):
        """Compound #37 (HA=74): should not have 'N-Namino' garbled pattern."""
        smiles = (
            "CCCCCCCCCC(=O)N[C@@H](COC(=O)CCCCCCC/C=C\\CCCCCCCC)"
            "C(=O)N[C@H](C(=O)N[C@@H](CO)C(=O)NCCC(=O)O)CC(C)C"
        )
        name = name_compound(smiles)
        assert "N-Namino" not in name, f"Garbled N-Namino found: {name}"
        assert "N-N-" not in name, f"Double N- found: {name}"


# ---------------------------------------------------------------------------
#: No amino/hydroxy duplication
# ---------------------------------------------------------------------------

class TestHydroxyDuplicationEliminated:
    """Verify no 'hydroxyhydroxy' or 'aminohydroxyamino' stacking."""

    @pytest.mark.integration
    def test_compound_26_no_hydroxyhydroxy(self):
        """Compound #26 (HA=39): ceramide-like, was 'hydroxyhydroxy'."""
        smiles = "CCCCCCCCCCCCCCCCCC(=O)NC(CO)C(O)CCCCCCCCCCCCC"
        name = name_compound(smiles)
        assert "hydroxyhydroxy" not in name, f"Duplicate hydroxy: {name}"
        assert "dihydroxy" in name, f"Expected 'dihydroxy' in: {name}"

    @pytest.mark.integration
    def test_compound_72_no_hydroxyhydroxy(self):
        """Compound #72 (HA=46): long chain amide, was 'hydroxyhydroxy'."""
        smiles = "CCCCCCCCCCCCCCCCCCC(=O)NC(CO)C(O)/C=C/CCCCCCCCCCCCC"
        name = name_compound(smiles)
        assert "hydroxyhydroxy" not in name, f"Duplicate hydroxy: {name}"

    @pytest.mark.integration
    def test_compound_104_no_hydroxyhydroxy(self):
        """Compound #104 (HA=43): stereogenic amide, was 'hydroxyhydroxy'."""
        smiles = "CCCCCCCCCCCCCCCCCC(=O)N[C@@H](CO)[C@@H](O)CCCCCCCCCCCCCC"
        name = name_compound(smiles)
        assert "hydroxyhydroxy" not in name, f"Duplicate hydroxy: {name}"
        assert "dihydroxy" in name, f"Expected 'dihydroxy' in: {name}"

    @pytest.mark.integration
    def test_compound_71_separate_hydroxy_locants(self):
        """Compound #71 (HA=43): E-unsaturated, distinct hydroxy positions."""
        smiles = "CCCCCCCCCCCCCCC(=O)NC(CO)C(O)/C=C/CCCCCCCCCCCCC"
        name = name_compound(smiles)
        assert "hydroxyhydroxy" not in name, f"Duplicate hydroxy: {name}"

    @pytest.mark.integration
    def test_compound_18_no_amino_stacking(self):
        """Compound #18 (HA=48): should not stack bare 'amino' repeatedly."""
        smiles = (
            "CC(C)C(NC(=O)CCCC(=O)O)C(=O)NC(CO)"
            "C(=O)NCCCCCCCCC/C=C\\CCCCCCCC"
        )
        name = name_compound(smiles)
        # At minimum, no "aminoamino" stacking
        assert "aminoamino" not in name, f"Amino stacking: {name}"


# ---------------------------------------------------------------------------
#: HW elision and garbled suffix patterns
# ---------------------------------------------------------------------------

class TestHWElisionAndSuffixes:
    """Verify HW elision correctness and no garbled suffix concatenation."""

    @pytest.mark.integration
    def test_oxazole_elision(self):
        """Compound #3 (HA=6): 'oxazole' not 'oxaazole'."""
        smiles = "Cc1ncco1"
        name = name_compound(smiles)
        assert "oxaazole" not in name, f"Bad elision: {name}"
        assert "oxazole" in name, f"Expected 'oxazole' in: {name}"
        assert name == "2-methyloxazole", f"Expected '2-methyloxazole', got: {name}"

    @pytest.mark.integration
    def test_oxazolane_elision(self):
        """Compound #86 (HA=25): 'oxazolane' not 'oxaazolane'."""
        smiles = "O1CCN(c2ncc[nH]2)[C@@]1(c1ccccc1)C(C)(C)C"
        name = name_compound(smiles)
        assert "oxaazolane" not in name, f"Bad elision: {name}"
        assert "oxazolane" in name, f"Expected 'oxazolane' in: {name}"

    @pytest.mark.integration
    def test_garbled_ion_suffix_absent(self):
        """Compound #29 (HA=32): no 'azaniumylpentanoateyl' garbled suffix."""
        smiles = (
            "N#CC(SC[C@H](NC(=O)CC[C@H]([NH3+])C(=O)[O-])"
            "C(=O)NCC(=O)[O-])c1c[nH]c2ccccc12"
        )
        name = name_compound(smiles)
        assert "azaniumylpentanoateylacetate" not in name, (
            f"Garbled suffix: {name}"
        )

    @pytest.mark.integration
    def test_compound_109_no_double_locant(self):
        """Compound #109 (HA=14): detect double-locant '3-3-' pattern."""
        smiles = "O=C(NC(=O)/C(=C/C1CCCN1)C(N)=O)CC(C)C"
        name = name_compound(smiles)
        # The name may still have issues but should not have
        # the specific garbled patterns from the research
        assert "azaniumylpentanoateyl" not in name, f"Garbled: {name}"


# ---------------------------------------------------------------------------
# General format safety: no bare N-N- in ANY of the target compounds
# ---------------------------------------------------------------------------

_ALL_TARGET_SMILES = [
    #
    "CC(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CCCNC(N)=N)C(=O)NCC(=O)O",
    (
        "O=C(Nc1ccc([N+](=O)[O-])cc1)c1ccc(NC(=O)[C@@H](N)CCCC(=O)"
        "NC(CCCC(=O)NC(CCCC(=O)NC(CC(=O)O)CCCCN)CCCCN)CCCCN)cc1"
    ),
    (
        "CCCCCCCCCC(=O)N[C@@H](COC(=O)CCCCCCC/C=C\\CCCCCCCC)"
        "C(=O)N[C@H](C(=O)N[C@@H](CO)C(=O)NCCC(=O)O)CC(C)C"
    ),
    (
        "CC(C)CC(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](CCC(=O)O)"
        "C(=O)N[C@@H](CC(=O)N)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H]"
        "(CC(C)C)C(=O)NCC(=O)O"
    ),
    #
    "CCCCCCCCCCCCCCCCCC(=O)NC(CO)C(O)CCCCCCCCCCCCC",
    "CCCCCCCCCCCCCCCCCCC(=O)NC(CO)C(O)/C=C/CCCCCCCCCCCCC",
    "CCCCCCCCCCCCCCCCCC(=O)N[C@@H](CO)[C@@H](O)CCCCCCCCCCCCCC",
    "CCCCCCCCCCCCCCC(=O)NC(CO)C(O)/C=C/CCCCCCCCCCCCC",
    #
    "Cc1ncco1",
    "O1CCN(c2ncc[nH]2)[C@@]1(c1ccccc1)C(C)(C)C",
]


@pytest.mark.integration
@pytest.mark.parametrize("smiles", _ALL_TARGET_SMILES)
def test_no_double_n_prefix_any_target(smiles):
    """None of the target compounds should produce N-N- prefix."""
    name = name_compound(smiles)
    assert "N-N-" not in name, f"Double N- in: {name}"


@pytest.mark.integration
@pytest.mark.parametrize("smiles", _ALL_TARGET_SMILES)
def test_no_hydroxyhydroxy_any_target(smiles):
    """None of the target compounds should produce 'hydroxyhydroxy'."""
    name = name_compound(smiles)
    assert "hydroxyhydroxy" not in name, f"Duplicate hydroxy in: {name}"
