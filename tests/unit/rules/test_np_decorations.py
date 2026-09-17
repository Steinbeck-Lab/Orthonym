"""Tests for natural product decoration enumeration.

Tests that steroid derivatives with functional group decorations
(hydroxyl, ketone, double bonds) produce correct systematic names
using the scaffold stem + decoration suffixes/prefixes.

Plan 15-07: NP Decoration Enumeration

Test coverage:
1. Cholesterol (exact match still works)
2. Testosterone (17-hydroxy + 3-one + 4-ene)
3. Progesterone (pregnane with 3,20-dione + 4-ene)
4. Androst-4-ene-3,17-dione (two ketones + ene)
5. Estradiol (estrane with 3,17-diol)
6. Cholestane bare scaffold (no decorations)
7. Camphor (exact derivative match, not steroid enumeration)
8. Saturated 3-hydroxyandrostane (androstan-3-ol)
"""

import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.rules.natural_products import name_natural_product


def _mol(smiles: str):
    """Return RDKit Mol from SMILES, raising on invalid input."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Invalid SMILES: {smiles}"
    return mol


# ---------------------------------------------------------------------------
# Test Class: NP Decoration Enumeration
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestNPDecorationEnumeration:
    """Test that steroid decorations (-OH, =O, C=C) are correctly enumerated."""

    def test_cholesterol_exact_match(self):
        """Cholesterol with correct stereochemistry should still use exact match."""
        smiles = (
            "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC=C4C"
            "[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C"
        )
        result = name_compound(smiles)
        assert result == "cholesterol"

    def test_testosterone_decoration(self):
        """Testosterone-configured input: 17β-hydroxyandrost-4-en-3-one.

        Has -OH at C-17 (prefix when ketone present), =O at C-3 (suffix),
        and C=C between C-4 and C-5 (ene suffix).

         a phase (C2a stereo honesty, fix a performance pass): re-keyed from a flat
        (stereo-undefined) input to a fully stereo-defined one -- the flat form
        is now correctly declined by name_natural_product's steroid stereo-honesty
        guard (it fabricated the natural ring configuration). RT-full verified
        (OPSIN round-trip, byte-identical name).
        """
        smiles = "C[C@]12CC[C@H]3[C@@H](CCC4=CC(=O)CC[C@]34C)[C@@H]1CC[C@@H]2O"
        result = name_compound(smiles)
        assert result == "17β-hydroxyandrost-4-en-3-one", f"Got '{result}'"

    def test_progesterone_decoration(self):
        """Progesterone-configured input: pregn-4-ene-3,20-dione.

        Has two ketone groups (C-3 and C-20) and one double bond (C-4,5).

         a phase (C2a stereo honesty, fix a performance pass): re-keyed from a flat
        (stereo-undefined) input to a fully stereo-defined one (a valid
        diastereomer distinct from the exact-derivative 'progesterone' lookup
        entry, so this test still exercises scaffold-based decoration
        enumeration rather than the Step-2 exact match). RT-full verified.
        """
        smiles = "CC(=O)[C@H]1CC[C@H]2[C@@H]3CCC4=CC(=O)CC[C@]4(C)[C@H]3CC[C@@]12C"
        result = name_compound(smiles)
        assert result == "(8S,9S,10R,13R,14S,17S)-pregn-4-ene-3,20-dione", f"Got '{result}'"

    def test_androstanedione_decoration(self):
        """Androst-4-ene-3,17-dione: two ketones + one double bond.

         a phase (C2a stereo honesty, fix a performance pass): re-keyed from a flat
        (stereo-undefined) input to a fully stereo-defined one (natural ring
        config; C-4=C-5 is an enone double bond so C-5 is not a stereocentre
        here). RT-full verified. Also corrects a stale expected value: the
        W5-A2 elision fix retains the terminal 'e' before the multiplied
        '-dione' ('...ene-3,17-dione', not the old '...en-3,17-dione').
        """
        smiles = "C[C@@]12C(CC[C@H]1[C@@H]1CCC3=CC(CC[C@]3(C)[C@H]1CC2)=O)=O"
        result = name_compound(smiles)
        assert result == "androst-4-ene-3,17-dione", f"Got '{result}'"

    def test_estradiol_decoration(self):
        """Estradiol-configured input: estrane with two -OH groups.

        Aromatic ring A is now detected via Kekulized copy (a phase-02),
        producing ene locants for the aromatic C=C bonds.

         a phase (C2a stereo honesty, fix a performance pass): re-keyed from a flat
        (stereo-undefined) input to a fully stereo-defined one (matches the
        W5-A2 gold row re-assertion). RT-full verified.
        """
        smiles = "C[C@@]12[C@H](CC[C@H]1[C@@H]1CCC=3C=C(C=CC3[C@H]1CC2)O)O"
        result = name_compound(smiles)
        assert result == "(8R,9S,13S,14S,17S)-estra-1,3,5(10)-triene-3,17-diol", (
            f"Got '{result}'"
        )

    def test_cholestane_bare_scaffold(self):
        """Bare cholestane (no decorations) should return 'cholestane'."""
        smiles = (
            "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CCC4CCCC"
            "[C@]4(C)[C@H]3CC[C@]12C"
        )
        result = name_compound(smiles)
        assert result == "cholestane"

    def test_camphor_not_steroid_enumeration(self):
        """Camphor must not be reached by steroid decoration enumeration.

        : renamed from test_camphor_exact_match and the expected value
        changed from "camphor". Camphor is now an adjudicated non-PIN -- it is a
        ketone, and (the Blue Book) makes chalcone "the only
        retained name as a preferred IUPAC name", while (:28307) is a
        closed general-nomenclature list that excludes it. The point this test
        guards is unchanged: the name comes from the von Baeyer builder, NOT from
        steroid enumeration. The value is the Blue Book's own form at:52648.
        """
        result = name_compound("CC12CCC(CC1=O)C2(C)C")
        assert result == "1,7,7-trimethylbicyclo[2.2.1]heptan-2-one"

    def test_saturated_hydroxy_androstane(self):
        """3-Hydroxyandrostane (saturated), stereo-defined -> (3S,8R,9S,10R,13S,14S)-androstan-3-ol.

        Saturated steroid with single -OH and no ketone uses -ol suffix.

         a phase (C2a stereo honesty, fix a performance pass): re-keyed from a flat
        (stereo-undefined) input -- the flat form is now correctly declined
        (it fabricated the natural ring configuration + the new C-3 stereo-
        centre). RT-full verified.
        """
        smiles = "C[C@@]12CCC[C@H]1[C@H]1CCC3C[C@H](CC[C@@]3(C)[C@H]1CC2)O"
        result = name_compound(smiles)
        assert result == "(3S,8R,9S,10R,13S,14S)-androstan-3-ol", f"Got '{result}'"


@pytest.mark.unit
class TestNPDecorationEdgeCases:
    """Edge cases for NP decoration enumeration."""

    def test_no_decorations_returns_scaffold_name(self):
        """Steroid scaffold match with only H substituents returns scaffold name."""
        # Androstane (exact scaffold)
        smiles = "C[C@@]12CCC[C@H]1[C@@H]1CCC3CCCC[C@]3(C)[C@H]1CC2"
        result = name_compound(smiles)
        assert result == "androstane"

    def test_non_steroid_scaffold_returns_scaffold_name(self):
        """Non-steroid NP scaffolds return plain scaffold name (no decoration enum)."""
        # Tropane
        smiles = "CN1[C@@H]2CCC[C@H]1CC2"
        result = name_compound(smiles)
        assert result == "tropane"

    def test_ketone_only_saturated(self):
        """Saturated steroid with only ketone decoration.

         a phase (C2a stereo honesty, fix a performance pass): re-keyed from a flat
        (non-stereo) input to a fully stereo-defined one -- flat is now
        correctly declined (fabricated the natural ring configuration).
        RT-full verified.
        """
        # 3-Oxoandrostane (androstane with =O at C-3, no unsaturation)
        smiles = "C[C@@]12CCC[C@H]1[C@H]1CCC3CC(CC[C@@]3(C)[C@H]1CC2)=O"
        result = name_compound(smiles)
        assert result == "(8R,9S,10R,13S,14S)-androstan-3-one", f"Got '{result}'"

    def test_double_bond_only(self):
        """Steroid with only a double bond modification, no FG substituents."""
        # Cholest-5-ene (cholestane with one C=C, no -OH or =O)
        # Use non-stereo to avoid exact match
        smiles = "CC(C)CCCC(C)C1CCC2C3CC=C4CCCCC4(C)C3CCC12C"
        result = name_compound(smiles)
        assert "en" in result, f"Expected 'en' suffix in '{result}'"

    def test_multiple_hydroxyls(self):
        """Steroid with multiple -OH groups.

         a phase (C2a stereo honesty, fix a performance pass): re-keyed from a flat
        (stereo-undefined) input -- flat is now correctly declined (fabricated
        the natural ring configuration + both new stereocentres). RT-full
        verified.
        """
        # 3,17-Dihydroxyandrostane (no unsaturation)
        smiles = "C[C@@]12[C@H](CC[C@H]1[C@@H]1CCC3C[C@H](CC[C@@]3(C)[C@H]1CC2)O)O"
        result = name_compound(smiles)
        assert result == "(3S,8R,9S,10R,13S,14S,17S)-androstane-3,17-diol", (
            f"Got '{result}'"
        )

    def test_decoration_does_not_break_exact_match(self):
        """Molecules with exact derivative entries still use exact match."""
        # Morphine should still be "morphine"
        smiles = "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5"
        assert name_compound(smiles) == "morphine"

    def test_name_natural_product_none_input(self):
        """name_natural_product(None) should return None."""
        assert name_natural_product(None) is None
