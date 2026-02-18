"""
Phase 62 small molecule accuracy sprint - regression tests.

Each fixed compound has an individual test ensuring the naming fix persists.
Tests are organized by fix category and validate both name correctness
and structural keywords.

Phase 62 Plan 01 (Wave 1): EASY compound fixes
"""

import pytest

from orthonym import name_compound


# === EASY FIX REGRESSION TESTS ===


class TestOxazoleElision:
    """Compound #3: HW prefix 'a' elision fix (oxaazole -> oxazole)."""

    def test_compound_3_contains_oxazole(self):
        """CC1=NCCO1 must produce name with 'oxazole' not 'oxaazole'."""
        name = name_compound("CC1=NCCO1")
        assert "oxazole" in name, f"Expected 'oxazole' in name, got: {name}"
        assert "oxaazole" not in name, f"'oxaazole' still present in: {name}"

    def test_compound_3_has_methyl(self):
        """CC1=NCCO1 must include the methyl substituent."""
        name = name_compound("CC1=NCCO1")
        assert "methyl" in name, f"Expected 'methyl' in name, got: {name}"


class TestAminiumCation:
    """Compound #2: Protonated amine -> aminium suffix (P-73.1.2)."""

    def test_compound_2_aminium_suffix(self):
        """CC(C)(C)[NH3+] must use aminium suffix, not ammonium."""
        name = name_compound("CC(C)(C)[NH3+]")
        assert "aminium" in name, f"Expected 'aminium' in name, got: {name}"
        assert "ammonium" not in name, f"'ammonium' should not appear in: {name}"

    def test_compound_2_branched_parent(self):
        """CC(C)(C)[NH3+] parent must be methylpropan (branched), not butyl."""
        name = name_compound("CC(C)(C)[NH3+]")
        assert "methylpropan" in name, f"Expected 'methylpropan' in name, got: {name}"
        assert "butyl" not in name.lower(), f"'butyl' (linear) should not appear in: {name}"


class TestCarboxylateAnion:
    """Compounds #17, #18, #25, #31: -oate suffix for [O-] carboxylates (P-72.2.1)."""

    def test_compound_17_dioate(self):
        """O=C([O-])CC=CC(=O)C(=O)[O-] must use -oate suffix."""
        name = name_compound("O=C([O-])CC=CC(=O)C(=O)[O-]")
        assert "oate" in name, f"Expected 'oate' in name, got: {name}"
        assert "oic acid" not in name, f"'oic acid' wrong for deprotonated: {name}"

    def test_compound_18_dioate_with_stereo(self):
        """O=C([O-])C(=O)C[C@H](O)C(=O)[O-] must use -oate with stereo."""
        name = name_compound("O=C([O-])C(=O)C[C@H](O)C(=O)[O-]")
        assert "oate" in name, f"Expected 'oate' in name, got: {name}"
        assert "oic acid" not in name, f"'oic acid' wrong for deprotonated: {name}"

    def test_compound_25_mono_oate(self):
        """C[C@H](CC(=O)[O-])OC(=O)C[C@@H](C)O must use -oate."""
        name = name_compound("C[C@H](CC(=O)[O-])OC(=O)C[C@@H](C)O")
        assert "oate" in name, f"Expected 'oate' in name, got: {name}"

    def test_compound_31_dioate(self):
        """Galactarate: must use -oate suffix."""
        name = name_compound(
            "O=C([O-])[C@@H](O)[C@H](O)[C@H](O)[C@@H](O)C(=O)[O-]"
        )
        assert "oate" in name, f"Expected 'oate' in name, got: {name}"
        assert "oic acid" not in name, f"'oic acid' wrong for deprotonated: {name}"


class TestBranchedNSubstituent:
    """Compound #7: Branched N-substituent naming (propan-2-yl not propyl)."""

    def test_compound_7_not_propyl(self):
        """CCN(C(C)C)C(C)C must NOT use bare 'propyl' for branched group."""
        name = name_compound("CCN(C(C)C)C(C)C")
        # The N-substituent should be isopropyl or propan-2-yl, never bare propyl
        # Allow both "isopropyl" (acceptable IUPAC) and "propan-2-yl" (preferred PIN)
        assert "N-propyl" not in name, f"'N-propyl' wrong for branched: {name}"

    def test_compound_7_has_ethyl(self):
        """CCN(C(C)C)C(C)C must include ethyl substituent."""
        name = name_compound("CCN(C(C)C)C(C)C")
        assert "ethyl" in name, f"Expected 'ethyl' in name, got: {name}"


# === INORGANIC UNKNOWN DESCRIPTIVE MESSAGE TESTS ===


class TestInorganicDescriptiveMessages:
    """Inorganic compounds must produce descriptive messages, not bare 'unknown'."""

    def test_wildcard_boron(self):
        """*B(*)* must indicate wildcard atoms."""
        name = name_compound("*B(*)*")
        assert "not supported" in name, f"Expected descriptive message, got: {name}"
        assert name != "unknown", "Must not be bare 'unknown'"

    def test_ytterbium_trichloride(self):
        """[Cl-].[Cl-].[Cl-].[Yb+3] must identify ytterbium."""
        name = name_compound("[Cl-].[Cl-].[Cl-].[Yb+3]")
        assert "ytterbium" in name, f"Expected 'ytterbium' in name, got: {name}"
        assert "not supported" in name, f"Expected 'not supported', got: {name}"

    def test_antimony_compound(self):
        """[O]=[Sb]([O-])([O-])[OH] must identify antimony."""
        name = name_compound("[O]=[Sb]([O-])([O-])[OH]")
        assert "antimony" in name, f"Expected 'antimony' in name, got: {name}"

    def test_mercury_tetraiodide(self):
        """[I][Hg-2]([I])([I])[I] must identify mercury."""
        name = name_compound("[I][Hg-2]([I])([I])[I]")
        assert "mercury" in name, f"Expected 'mercury' in name, got: {name}"

    def test_gold_pentafluoride(self):
        """[F][Au]([F])([F])([F])[F] must identify gold."""
        name = name_compound("[F][Au]([F])([F])([F])[F]")
        assert "gold" in name, f"Expected 'gold' in name, got: {name}"

    def test_nickel_sulfate(self):
        """[Ni+2].[O-]S(=O)(=O)[O-] must identify nickel."""
        name = name_compound("[Ni+2].[O-]S(=O)(=O)[O-]")
        assert "nickel" in name, f"Expected 'nickel' in name, got: {name}"


# === ORGANIC UNKNOWN ELIMINATION TESTS ===


class TestOrganicUnknownElimination:
    """Organic compounds must not produce bare 'unknown'."""

    def test_dinitrogen_anion_not_bare_unknown(self):
        """[N-2][NH-] must not return bare 'unknown'."""
        name = name_compound("[N-2][NH-]")
        assert name != "unknown", f"Organic compound must not be bare 'unknown'"

    def test_phosphonate_aa_not_bare_unknown(self):
        """C[C@@H]([NH3+])P(=O)([O-])[O-] must not return bare 'unknown'."""
        name = name_compound("C[C@@H]([NH3+])P(=O)([O-])[O-]")
        assert name != "unknown", f"Organic compound must not be bare 'unknown'"
