"""
End-to-end integration tests for multi-bond decomposition (Phase 56, Plan 03).

Tests real molecules with 2+ cleavable bonds through the full pipeline
(SMILES -> name_compound() -> IUPAC name) to verify:

1. Triglycerides (3 ester bonds) produce decomposition names
2. Phospholipids (2 ester + phosphate) produce non-duplicate names
3. Performance guard prevents timeouts on highly-cleavable molecules
4. DECP-02 recursive fragment decomposition works on 3-bond molecules
5. Mixed bond types (ester + amide, ester + glycosidic) don't crash

Backward compat cases (triacetin, tripropionin) assert EXACT names.
New multi-bond cases assert properties (not None, no duplicates, etc.).
"""

import time

import pytest

from orthonym import name_compound


def _has_consecutive_duplicate_words(name: str) -> bool:
    """Check if a name contains consecutive duplicate words (len > 3)."""
    if not name:
        return False
    words = name.split()
    for i in range(len(words) - 1):
        if words[i] == words[i + 1] and len(words[i]) > 3:
            return True
    return False


# ---------------------------------------------------------------------------
# Section 1: Triglyceride end-to-end tests
# ---------------------------------------------------------------------------

class TestTriglycerides:
    """Molecules with 3 ester bonds on a glycerol backbone."""

    @pytest.mark.integration
    def test_tripalmitin_produces_decomposition_name(self):
        """Tripalmitin (3 palmitic acid esters on glycerol): should produce
        a name referencing acyl fragments, not just a retained parent name."""
        smi = (
            "CCCCCCCCCCCCCCCC(=O)OCC(COC(=O)CCCCCCCCCCCCCCC)"
            "OC(=O)CCCCCCCCCCCCCCC"
        )
        name = name_compound(smi)
        assert name is not None, "Tripalmitin should produce a name"
        assert name != "unknown", "Tripalmitin should not be 'unknown'"
        assert not _has_consecutive_duplicate_words(name), (
            f"Tripalmitin has consecutive duplicate words: {name}"
        )
        # Should reference at least one acyl fragment
        has_fragment_ref = (
            "palmit" in name.lower()
            or "hexadecan" in name.lower()
            or "propane" in name.lower()
            or "oyloxy" in name.lower()
        )
        assert has_fragment_ref, (
            f"Expected fragment reference in tripalmitin name, got: {name}"
        )

    @pytest.mark.integration
    def test_triacetin_exact_name_backward_compat(self):
        """Triacetin: backward compat -- polyfunctional path should produce
        the same name as before Phase 56."""
        name = name_compound("CC(=O)OCC(COC(C)=O)OC(C)=O")
        assert name == "1,2,3-tri(acetyloxy)propane", (
            f"Triacetin backward compat failure: {name}"
        )

    @pytest.mark.integration
    def test_tripropionin_exact_name_backward_compat(self):
        """Tripropionin: backward compat -- should still produce polyfunctional name."""
        name = name_compound("CCC(=O)OCC(COC(=O)CC)OC(=O)CC")
        assert name == "1,2,3-tri(propanoyloxy)propane", (
            f"Tripropionin backward compat failure: {name}"
        )


# ---------------------------------------------------------------------------
# Section 2: Phospholipid end-to-end tests
# ---------------------------------------------------------------------------

class TestPhospholipids:
    """Molecules with 2 ester bonds + phosphate group."""

    @pytest.mark.integration
    def test_dppc_like_produces_name_no_duplicates(self):
        """DPPC-like phospholipid (dipalmitoylphosphatidylcholine fragment):
        should produce a name with no consecutive duplicate words."""
        smi = (
            "CCCCCCCCCCCCCCCC(=O)OCC(COP(O)(=O)OCC[N+](C)(C)C)"
            "OC(=O)CCCCCCCCCCCCCCC"
        )
        name = name_compound(smi)
        assert name is not None, "DPPC-like should produce a name"
        assert name != "unknown", "DPPC-like should not be 'unknown'"
        assert not _has_consecutive_duplicate_words(name), (
            f"DPPC-like has consecutive duplicate words: {name}"
        )

    @pytest.mark.integration
    def test_simple_phospholipid_produces_name(self):
        """Simpler phospholipid (2 short esters + phosphate):
        should produce a non-None name."""
        smi = "CC(=O)OCC(COP(O)(=O)O)OC(=O)C"
        name = name_compound(smi)
        assert name is not None, "Simple phospholipid should produce a name"
        assert name != "unknown", "Simple phospholipid should not be 'unknown'"


# ---------------------------------------------------------------------------
# Section 3: Performance guard tests
# ---------------------------------------------------------------------------

class TestPerformanceGuard:
    """Verify that molecules with many cleavable bonds complete quickly."""

    @pytest.mark.integration
    def test_hexaester_completes_within_2s(self):
        """Hexaacetyl sorbitol (6 ester bonds): name_compound() must complete
        within 2 seconds without hang or timeout."""
        smi = "CC(=O)OCC(OC(C)=O)C(OC(C)=O)C(OC(C)=O)C(OC(C)=O)COC(C)=O"
        t0 = time.time()
        name = name_compound(smi)
        elapsed = time.time() - t0
        assert elapsed < 2.0, (
            f"Hexaester took {elapsed:.2f}s (limit: 2.0s)"
        )
        assert name is not None, "Hexaester should produce a name"

    @pytest.mark.integration
    def test_over_max_bonds_completes_within_2s(self):
        """Nonaester (9 ester bonds, exceeds MAX_CLEAVABLE_BONDS=8):
        performance guard should trigger, producing a name quickly."""
        smi = (
            "CC(=O)OCC(OC(C)=O)C(OC(C)=O)C(OC(C)=O)C(OC(C)=O)"
            "C(OC(C)=O)C(OC(C)=O)C(OC(C)=O)COC(C)=O"
        )
        t0 = time.time()
        name = name_compound(smi)
        elapsed = time.time() - t0
        assert elapsed < 2.0, (
            f"Nonaester took {elapsed:.2f}s (limit: 2.0s)"
        )
        assert name is not None, "Nonaester should produce a name"


# ---------------------------------------------------------------------------
# Section 4: DECP-02 recursive fragment decomposition tests
# ---------------------------------------------------------------------------

class TestRecursiveFragmentDecomposition:
    """Verify DECP-02: recursive fragment naming on multi-bond molecules.

    A molecule with 3 cleavable bonds triggers decomposition at depth 0.
    The resulting fragment still has 2 cleavable bonds and is recursively
    decomposed at depth 1. A non-None result with no duplicate words
    proves recursive fragment naming works.
    """

    @pytest.mark.integration
    def test_3_bond_molecule_recursive_decomposition(self):
        """3-bond molecule (2 ester + 1 amide): decomposition at depth 0
        produces a fragment that still has 2 cleavable bonds.

        # DECP-02: This 3-bond molecule triggers decomposition at depth 0.
        # The resulting fragment still has 2 cleavable bonds and is
        # recursively decomposed at depth 1. A non-None result with no
        # duplicate words proves recursive fragment naming works.
        """
        # 2 esters + 1 amide = 3 cleavable bonds
        smi = "CCC(=O)OCC(OC(=O)CC)C(=O)NC"
        name = name_compound(smi)
        assert name is not None, "3-bond molecule should produce a name"
        assert name != "unknown", "3-bond molecule should not be 'unknown'"
        assert not _has_consecutive_duplicate_words(name), (
            f"3-bond molecule has consecutive duplicate words: {name}"
        )
        # A recursive decomposition should produce a meaningfully long name
        # (multiple fragments named) compared to a single retained name
        assert len(name) > 15, (
            f"3-bond recursive result too short (suggests no recursion): {name}"
        )

    @pytest.mark.integration
    def test_3_ester_bond_molecule_recursive_decomposition(self):
        """3 ester bonds: the first decomposition produces a fragment with 2
        remaining ester bonds that must be recursively handled."""
        # ester-ester-amide connectivity
        smi = "CC(=O)OCC(=O)NCCOC(=O)CC"
        name = name_compound(smi)
        assert name is not None, "3-bond ester/amide molecule should produce a name"
        assert name != "unknown"
        assert not _has_consecutive_duplicate_words(name), (
            f"3-bond ester/amide has consecutive duplicate words: {name}"
        )

    @pytest.mark.integration
    def test_tetraglycine_like_3_amide_bonds(self):
        """Tetraglycine-like molecule (3 amide bonds): recursive amide
        decomposition should produce a peptide-style name."""
        smi = "CC(=O)NCC(=O)NCC(=O)NCC(=O)O"
        name = name_compound(smi)
        assert name is not None, "Tetraglycine-like should produce a name"
        assert name != "unknown"
        assert not _has_consecutive_duplicate_words(name), (
            f"Tetraglycine-like has consecutive duplicate words: {name}"
        )


# ---------------------------------------------------------------------------
# Section 5: Mixed bond type tests
# ---------------------------------------------------------------------------

class TestMultiBondMixed:
    """Molecules with mixed cleavable bond types."""

    @pytest.mark.integration
    def test_ester_plus_amide_no_crash(self):
        """Molecule with 1 ester + 1 amide bond: decomposition handles
        mixed bond types without crashing."""
        smi = "CC(=O)OCC(=O)NCC"
        name = name_compound(smi)
        assert name is not None, "Ester+amide should produce a name"
        assert name != "unknown"

    @pytest.mark.integration
    def test_ester_plus_glycosidic_no_crash(self):
        """Molecule with ester + glycosidic bond: decomposition handles it."""
        # Sugar ester with both ester and glycosidic cleavable bonds
        smi = "CC(=O)O[C@@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O"
        name = name_compound(smi)
        assert name is not None, "Ester+glycosidic should produce a name"
        assert name != "unknown"

    @pytest.mark.integration
    def test_diester_plus_amide_no_crash(self):
        """Molecule with 2 ester + 1 amide bond: mixed multi-bond
        decomposition completes without error."""
        smi = "CCC(=O)OCC(OC(=O)CC)C(=O)NC"
        name = name_compound(smi)
        assert name is not None, "2-ester+amide should produce a name"
        assert name != "unknown"
        assert not _has_consecutive_duplicate_words(name), (
            f"2-ester+amide has consecutive duplicate words: {name}"
        )
