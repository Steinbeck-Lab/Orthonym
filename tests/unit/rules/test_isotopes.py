"""Isotopic substitution decorator (Wave-2 P2 Tasks 1-4).

BB P-82.2.1 (BlueBookV2.md:43718): the nuclide symbol(s) in parentheses,
preceded by any necessary locants, are inserted before the isotopically
substituted part; a preceding locant takes a hyphen after the parenthesis;
polysubstitution count is a right subscript. P-45.4.1/.4.2/.4.3
(BlueBookV2.md:22212-22232): lowest locants to modified positions, then to
higher atomic number, then to higher mass number.
"""
import pytest
from rdkit import Chem

from orthonym.rules.isotopes import (
    has_isotopes,
    strip_isotopes,
)


class TestHasIsotopes:
    @pytest.mark.parametrize("smiles,expected", [
        ("[2H]C([2H])([2H])CO", True),   # trideuterio-ethanol
        ("[14CH3]CO", True),             # 14C ethanol
        ("[13CH3]CO", True),
        ("[3H]C([3H])([3H])CO", True),   # tritium
        ("CCO", False),                  # unlabeled — must be inert
        ("c1ccccc1", False),
        ("[Na+].[O-]C(=O)C", False),     # charged, unlabeled
    ])
    def test_detection(self, smiles, expected):
        mol = Chem.MolFromSmiles(smiles)
        assert has_isotopes(mol) is expected

    def test_none_mol_is_false(self):
        assert has_isotopes(None) is False


class TestStripIsotopes:
    def test_strip_returns_unlabeled_copy_and_map(self):
        mol = Chem.MolFromSmiles("[14CH3]CO")
        stripped, label_map = strip_isotopes(mol)
        # every isotope cleared on the copy
        assert all(a.GetIsotope() == 0 for a in stripped.GetAtoms())
        # original untouched (copy semantics)
        assert any(a.GetIsotope() != 0 for a in mol.GetAtoms())
        # canonical skeleton is plain ethanol
        assert Chem.MolToSmiles(stripped) == Chem.MolToSmiles(Chem.MolFromSmiles("CCO"))
        # the one 14C atom is recorded with its mass number
        assert list(label_map.values()) == [14]

    def test_multi_label_map(self):
        mol = Chem.MolFromSmiles("[2H]C([2H])([2H])CO")
        stripped, label_map = strip_isotopes(mol)
        assert sorted(label_map.values()) == [2, 2, 2]
        assert Chem.MolToSmiles(stripped) == Chem.MolToSmiles(Chem.MolFromSmiles("CCO"))


from orthonym.rules.isotopes import (
    nuclide_symbol,
    format_isotope_descriptor,
)


class TestNuclideSymbol:
    @pytest.mark.parametrize("mass,element,out", [
        (14, "C", "14C"),
        (2, "H", "2H"),
        (3, "H", "3H"),
        (13, "C", "13C"),
        (18, "O", "18O"),
        (12, "C", "12C"),
        (81, "Br", "81Br"),
    ])
    def test_symbol(self, mass, element, out):
        assert nuclide_symbol(mass, element) == out


class TestFormatIsotopeDescriptor:
    def test_trideuterio_with_locant(self):
        # (2,2,2-2H3)  three 2H at locant 2 -> single grouped token, count 3
        # BB P-84 (BlueBookV2.md:44506): (2,2,2-2H3)ethan-1-ol
        groups = [(2, 2, "H", 3)]
        assert format_isotope_descriptor(groups) == "(2,2,2-2H3)"

    def test_single_14c_with_locant(self):
        # BB:43740 (2-13C); here (2-14C)
        assert format_isotope_descriptor([(2, 14, "C", 1)]) == "(2-14C1)"

    def test_deuterio_no_locant_single_position_ring_substituent(self):
        # BB:43730 (2H3)methoxybenzene — descriptor at front, count 3, no locant
        assert format_isotope_descriptor([(None, 2, "H", 3)]) == "(2H3)"

    def test_12c_methane_no_locant(self):
        # BB:43724 trichloro(12C)methane — single position; Orthonym emits the
        # count-subscript form (12C1) which OPSIN also parses.
        assert format_isotope_descriptor([(None, 12, "C", 1)]) == "(12C1)"

    def test_deuterio_methane_count_one(self):
        # BB:43726 (2H1)methane — count subscript kept even for count 1
        assert format_isotope_descriptor([(None, 2, "H", 1)]) == "(2H1)"

    def test_two_nuclides_same_place_alphabetical_then_mass(self):
        # P-82.2.1: cited alphabetically by element, then by mass number.
        # elements alphabetical C < H, so 13C first.
        groups = [(1, 2, "H", 1), (1, 13, "C", 1)]
        assert format_isotope_descriptor(groups) == "(1-13C1,1-2H1)"


class TestDecorateEndToEnd:
    """Full name_compound path through the isotope decorator (P-45.4 + P-82.2.1).

    Every expected_pin below was OPSIN round-trip verified against its SMILES
    on 2026-07-09 (see plan header table).
    """

    @pytest.mark.parametrize("smiles,expected", [
        ("[14CH3]CO", "(2-14C1)ethan-1-ol"),          # P-45.4.1 lowest locant -> CH3 = C2
        ("[13CH3]CO", "(2-13C1)ethan-1-ol"),
        ("[2H]C([2H])([2H])CO", "(2,2,2-2H3)ethan-1-ol"),  # P-84 verbatim
        ("[2H]C([2H])([2H])Oc1ccccc1", "(2H3)methoxybenzene"),  # BB:43730
        ("[12CH](Cl)(Cl)Cl", "trichloro(12C1)methane"),         # BB:43724 (count form)
    ])
    def test_expected_pin(self, smiles, expected):
        from orthonym.namer import name_compound
        got = name_compound(smiles, style="systematic")
        assert got == expected, f"{smiles}: got {got!r} want {expected!r}"

    def test_unlabeled_is_untouched(self):
        from orthonym.namer import name_compound
        assert name_compound("CCO", style="systematic") == "ethan-1-ol"
        assert name_compound("CCO") == "ethanol"

    def test_unmappable_label_fails_closed(self):
        # A label on an atom the oracle cannot place (no round-tripping
        # candidate) must NOT emit a wrong labeled name — decorator returns
        # None and the pipeline yields the (unlabeled) skeleton best-effort.
        from orthonym.rules.isotopes import decorate_isotopic_name
        from orthonym.namer import Orthonym
        n = Orthonym(style="systematic")
        # Deliberately exotic: a labeled atom with no OPSIN-expressible locus.
        out = decorate_isotopic_name("[36Cl]", "systematic", n)
        assert out is None or "36Cl" not in out or "chlorane" in out  # never a wrong labeled PIN


class TestNuclideTieBreaks:
    def test_tritium_round_trips_via_count_form(self):
        # P-45.4.3 / oracle: [3H] parses only as (nH1); the count form does.
        from orthonym.namer import name_compound
        got = name_compound("[3H]C([3H])([3H])CO", style="systematic")
        assert got == "(2,2,2-3H3)ethan-1-ol", f"got {got!r}"

    def test_p4543_higher_mass_lower_locant_key(self):
        # 14C preferred at the lower locant over 13C (BB:22232).
        from orthonym.rules.isotopes import _p4542_p4543_key
        # two candidate placements of the SAME structure differing only in which
        # nuclide sits at the lower locant; higher mass -> lower locant wins.
        hi_at_2 = [(2, 14, "C", 1), (4, 13, "C", 1)]
        hi_at_4 = [(4, 14, "C", 1), (2, 13, "C", 1)]
        assert _p4542_p4543_key(hi_at_2) < _p4542_p4543_key(hi_at_4)

    def test_p4542_higher_z_lower_locant_key(self):
        # 18O (Z=8) preferred at the lower locant over 13C (Z=6) (BB:22224).
        from orthonym.rules.isotopes import _p4542_p4543_key
        o_low = [(1, 18, "O", 1), (2, 13, "C", 1)]
        c_low = [(1, 13, "C", 1), (2, 18, "O", 1)]
        assert _p4542_p4543_key(o_low) < _p4542_p4543_key(c_low)


class TestDeMultiplicationP4542:
    """P-82.2.2.1 / P-45.4.1 — one labeled copy among identical substituent prefixes.

    Target OPSIN-verified in  §Item-2 §A/§D.
    """
    def test_evidence_target(self):
        from orthonym.namer import name_compound
        got = name_compound("[13CH3]OCCOC", style="pin")
        assert got == "1-(13C1)methoxy-2-methoxyethane", f"got {got!r}"

    def test_ambiguity_tiebreak_lowest_labeled_locant(self):
        # both 1- and 2- forms RT (symmetric); P-45.4.1 picks the labeled copy at locant 1
        from orthonym.namer import name_compound
        got = name_compound("[13CH3]OCCOC", style="pin")
        assert got.startswith("1-(13C1)methoxy"), got

    @pytest.mark.parametrize("smiles,expected", [
        # PROTECT: existing shipped isotope golds unaffected (skeletons carry no
        # LEADING locanted simple multiplier -> de-mult branch inert)
        ("[14CH3]CO",               "(2-14C1)ethan-1-ol"),
        ("[13CH3]CO",               "(2-13C1)ethan-1-ol"),
        ("[2H]C([2H])([2H])CO",     "(2,2,2-2H3)ethan-1-ol"),
        ("[2H]C([2H])([2H])Oc1ccccc1", "(2H3)methoxybenzene"),
        ("[12CH](Cl)(Cl)Cl",        "trichloro(12C1)methane"),  # 'tri' but NO leading locant -> inert
        ("[13CH3]OC(C)=O",          "(13C1)methyl acetate"),    # front descriptor path
        ("CC[18OH]",                "(18O1)ethan-1-ol"),        # parent-front, unaffected
    ])
    def test_existing_isotope_placements_unaffected(self, smiles, expected):
        from orthonym.namer import name_compound
        assert name_compound(smiles, style="systematic") == expected \
            or name_compound(smiles, style="pin") == expected


class TestRefuseOnUndecoratablePolicy:
    def test_mixed_two_label_construct_refuses(self):
        # mixed 18O/13C two-copy: two DIFFERENT (mass,el) groups -> de-mult inert
        # (single-group only) and the combined single-token candidate does NOT RT
        # -> decorator None -> REFUSE, never the label-dropping '1,2-dimethoxyethane'.
        from orthonym.namer import name_compound
        assert name_compound("C[18O]CCO[13CH3]", style="pin") == "unknown organic compound"

    def test_labeled_molecule_never_drops_label(self):
        from orthonym.namer import name_compound
        out = name_compound("C[18O]CCO[13CH3]", style="pin")
        assert "dimethoxyethane" not in out   # the unlabeled skeleton is NOT emitted

    def test_unlabeled_corpus_byte_identical(self):
        # the has_isotopes gate keeps the new policy inert on unlabeled input
        from orthonym.namer import name_compound
        assert name_compound("CCO") == "ethanol"
        assert name_compound("CCO", style="systematic") == "ethan-1-ol"

    def test_decoratable_labels_still_ship(self):
        # Task 7 target + shipped golds still emit their labeled PIN (policy only
        # fires on decorator None)
        from orthonym.namer import name_compound
        assert name_compound("[13CH3]OCCOC", style="pin") == "1-(13C1)methoxy-2-methoxyethane"
        assert name_compound("[14CH3]CO", style="systematic") == "(2-14C1)ethan-1-ol"
