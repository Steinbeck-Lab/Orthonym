"""Isotopic substitution decorator (Wave-2 P2 Tasks 1-4).

BB (the Blue Book): the nuclide symbol(s) in parentheses,
preceded by any necessary locants, are inserted before the isotopically
substituted part; a preceding locant takes a hyphen after the parenthesis;
polysubstitution count is a right subscript. /.4.2/.4.3
(the Blue Book-22232): lowest locants to modified positions, then to
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
    """ (the Blue Book-43720): the count subscript is shown "when
    polysubstitution at a single position is possible". Groups carry an optional
    5th field ``max_at_pos`` (the polysubstitution quantity); the subscript is
    emitted iff ``count > 1`` OR ``max_at_pos > 1``. A missing 5th field (legacy
    4-tuple) keeps the subscript unconditionally (the pre-FIX-A fallback the
    placement search relies on for OPSIN-unparseable omissions).
    """

    def test_trideuterio_with_locant(self):
        # (2,2,2-2H3) three 2H at locant 2 -> single grouped token, count 3
        # BB (the Blue Book): (2,2,2-2H3)ethan-1-ol. count > 1 -> shown.
        groups = [(2, 2, "H", 3, 3)]
        assert format_isotope_descriptor(groups) == "(2,2,2-2H3)"

    def test_single_14c_drops_subscript(self):
        # the Blue Book (2-13C); here (2-14C). A carbon position holds one carbon
        # (max_at_pos 1, count 1) -> subscript OMITTED (FIX-A)..
        assert format_isotope_descriptor([(2, 14, "C", 1, 1)]) == "(2-14C)"

    def test_deuterio_no_locant_single_position_ring_substituent(self):
        # the Blue Book (2H3)methoxybenzene — descriptor at front, count 3, no locant.
        assert format_isotope_descriptor([(None, 2, "H", 3, 3)]) == "(2H3)"

    def test_12c_methane_drops_subscript(self):
        # the Blue Book trichloro(12C)methane — a carbon position (max_at_pos 1,
        # count 1) omits the subscript (FIX-A)..
        assert format_isotope_descriptor([(None, 12, "C", 1, 1)]) == "(12C)"

    def test_deuterio_methane_count_one_KEEPS_subscript(self):
        # the Blue Book (2H1)methane — REGRESSION PIN. The methane carbon can bear 4 H
        # (max_at_pos 4), so polysubstitution at that position IS possible and the
        # subscript is KEPT even at count 1.. Mutating the predicate to
        # unconditionally drop would break this.
        assert format_isotope_descriptor([(None, 2, "H", 1, 4)]) == "(2H1)"

    def test_deuterio_on_single_H_position_drops_subscript(self):
        # A D on a CH (1 substitutable H, max_at_pos 1) omits the subscript:
        # (2S)-(2-2H)butan-2-ol..
        assert format_isotope_descriptor([(2, 2, "H", 1, 1)]) == "(2-2H)"

    def test_legacy_four_tuple_keeps_subscript(self):
        # No max_at_pos supplied -> the placement-search fallback form, subscript
        # kept unconditionally (used when the omitted spelling fails OPSIN RT).
        assert format_isotope_descriptor([(2, 14, "C", 1)]) == "(2-14C1)"

    def test_force_show_overrides_predicate(self):
        # force_show=True restores the always-emit form regardless of max_at_pos.
        assert format_isotope_descriptor([(2, 14, "C", 1, 1)], force_show=True) \
            == "(2-14C1)"

    def test_two_nuclides_same_place_alphabetical_then_mass(self):
        #: cited alphabetically by element, then by mass number.
        # elements alphabetical C < H, so 13C first. 13C drops (carbon position);
        # the D at max_at_pos 1 also drops -> (1-13C,1-2H).
        groups = [(1, 2, "H", 1, 1), (1, 13, "C", 1, 1)]
        assert format_isotope_descriptor(groups) == "(1-13C,1-2H)"

    def test_fixf_bracket_mode_is_opt_in(self):
        # FIX-F: the specifically-labelled bracket form is OPT-IN. The
        # default path stays parenthetical; bracket=True swaps the outer
        # enclosing marks. All ch83 rows are a documented ceiling under the default.
        assert format_isotope_descriptor([(None, 2, "H", 1, 4)]) == "(2H1)"
        assert format_isotope_descriptor([(None, 2, "H", 1, 4)], bracket=True) == "[2H1]"
        assert format_isotope_descriptor([(None, 13, "C", 1, 1)], bracket=True) == "[13C]"


class TestDecorateEndToEnd:
    """Full name_compound path through the isotope decorator +.

    Every expected_pin below was OPSIN round-trip verified against its SMILES
    on 2026-07-09 (see plan header table).
    """

    @pytest.mark.parametrize("smiles,expected", [
        # FIX-A: a heavy-atom label at a single position omits the count subscript
        #. Verbatim BB PINs.
        ("[14CH3]CO", "(2-14C)ethan-1-ol"),           # lowest locant -> CH3 = C2
        ("[13CH3]CO", "(2-13C)ethan-1-ol"),           # the Blue Book
        ("[2H]C([2H])([2H])CO", "(2,2,2-2H3)ethan-1-ol"),  # verbatim, count 3 keeps
        ("[2H]C([2H])([2H])Oc1ccccc1", "(2H3)methoxybenzene"),  # the Blue Book, count 3 keeps
        ("[12CH](Cl)(Cl)Cl", "trichloro(12C)methane"),         # the Blue Book
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
        # / oracle: [3H] parses only as (nH1); the count form does.
        from orthonym.namer import name_compound
        got = name_compound("[3H]C([3H])([3H])CO", style="systematic")
        assert got == "(2,2,2-3H3)ethan-1-ol", f"got {got!r}"

    def test_p4543_higher_mass_lower_locant_key(self):
        # 14C preferred at the lower locant over 13C (the Blue Book).
        from orthonym.rules.isotopes import _p4542_p4543_key
        # two candidate placements of the SAME structure differing only in which
        # nuclide sits at the lower locant; higher mass -> lower locant wins.
        hi_at_2 = [(2, 14, "C", 1), (4, 13, "C", 1)]
        hi_at_4 = [(4, 14, "C", 1), (2, 13, "C", 1)]
        assert _p4542_p4543_key(hi_at_2) < _p4542_p4543_key(hi_at_4)

    def test_p4542_higher_z_lower_locant_key(self):
        # 18O (Z=8) preferred at the lower locant over 13C (Z=6) (the Blue Book).
        from orthonym.rules.isotopes import _p4542_p4543_key
        o_low = [(1, 18, "O", 1), (2, 13, "C", 1)]
        c_low = [(1, 13, "C", 1), (2, 18, "O", 1)]
        assert _p4542_p4543_key(o_low) < _p4542_p4543_key(c_low)


class TestDeMultiplicationP4542:
    """ / — one labeled copy among identical substituent prefixes.

    Target OPSIN-verified in internal notes §Item-2 §A/§D.
    """
    def test_evidence_target(self):
        from orthonym.namer import name_compound
        got = name_compound("[13CH3]OCCOC", style="pin")
        assert got == "1-(13C)methoxy-2-methoxyethane", f"got {got!r}"

    def test_ambiguity_tiebreak_lowest_labeled_locant(self):
        # both 1- and 2- forms RT (symmetric); picks the labeled copy at locant 1
        from orthonym.namer import name_compound
        got = name_compound("[13CH3]OCCOC", style="pin")
        assert got.startswith("1-(13C)methoxy"), got

    def test_mixed_two_nuclide_demux_p4542(self):
        # W2F-P5-11: TWO DIFFERENT nuclides (18O and 13C), one per identical
        # methoxy copy. The parent (1,2-dimethoxyethane) is symmetric, so BOTH
        # numbering directions round-trip in OPSIN — the oracle cannot choose.
        # governs: the HIGHER atomic number gets the LOWER locant, so
        # 18O (Z=8) > 13C (Z=6) => 18O-methoxy at locant 1. Gold OPSIN-RT verified.
        from orthonym.namer import name_compound
        got = name_compound("C[18O]CCO[13CH3]", style="pin")
        assert got == "1-(18O)methoxy-2-(13C)methoxyethane", f"got {got!r}"

    @pytest.mark.parametrize("smiles,expected", [
        # PROTECT: existing shipped isotope golds unaffected by the de-mult branch
        # (skeletons carry no LEADING locanted simple multiplier -> branch inert).
        # Heavy-atom labels omit the count subscript (FIX-A,; count>1
        # deuterio groups keep it.
        ("[14CH3]CO",               "(2-14C)ethan-1-ol"),
        ("[13CH3]CO",               "(2-13C)ethan-1-ol"),
        ("[2H]C([2H])([2H])CO",     "(2,2,2-2H3)ethan-1-ol"),
        ("[2H]C([2H])([2H])Oc1ccccc1", "(2H3)methoxybenzene"),
        ("[12CH](Cl)(Cl)Cl",        "trichloro(12C)methane"),  # 'tri' but NO leading locant -> inert
        ("[13CH3]OC(C)=O",          "(13C)methyl acetate"),    # front descriptor path
        # A locant-free O nuclide on the ``-ol`` suffix goes immediately before the
        # suffix, and the unlabelled name's locant-free spelling stays when the label
        # needs no locant: (the Blue Book) 'ethan(2H)ol (PIN) (as in
        # ethanol)' (:44184); (:43718) '1-(aminomethyl)cyclopentan-1-(18O)ol
        # (PIN)' (:43744). Gold W2F-P5-P4 follows. OPSIN 2.9.0 reads it back to CC[18OH].
        ("CC[18OH]",                "ethan(18O)ol"),           # gold W2F-P5-P4
    ])
    def test_existing_isotope_placements_unaffected(self, smiles, expected):
        from orthonym.namer import name_compound
        assert name_compound(smiles, style="systematic") == expected \
            or name_compound(smiles, style="pin") == expected


class TestDeMultScopeGuards:
    """Fail-closed scope guards in _decorate_demultiplied (OPSIN-free: each case
    returns None BEFORE the oracle loop, so these are fast and deterministic).

    These lock the PIN-safe / cost-safe boundaries hardened after code review:
    repeated locants, repeated nuclides, >1 bare copy, and combinatorial blow-up
    all fail closed rather than emit a non-PIN separated name or exhaust the
    JVM-per-candidate oracle.
    """
    def _call(self, skeleton, keys):
        from orthonym.rules.isotopes import _decorate_demultiplied
        # keys entries are ((mass, el), count[, max_at_pos]); pad legacy 2-tuples
        # to the 3-tuple shape the enriched pipeline expects (max_at_pos is
        # irrelevant here -- every guard fires before the descriptor is built).
        keys = [k if len(k) == 3 else (k[0], k[1], 1) for k in keys]
        # original/stripped are untouched when a guard fires before the oracle.
        return _decorate_demultiplied(skeleton, keys, None, None)

    def test_not_a_demult_form_returns_none(self):
        assert self._call("ethan-1-ol", [((13, "C"), 1)]) is None

    def test_repeated_locants_gem_copies_fail_closed(self):
        # 1,1,2-...: a locant-keyed copy map cannot tell the two locant-1 copies
        # apart -> fail closed (never a mis-labeled candidate).
        assert self._call("1,1,2-trimethylbenzene", [((13, "C"), 1)]) is None

    def test_repeated_nuclide_across_copies_fail_closed(self):
        # both copies 13C: identical copies must stay grouped under a multiplier
        # (a different name shape), not be separated -> fail closed.
        assert self._call("1,2-dimethylbenzene", [((13, "C"), 2)]) is None

    def test_more_than_one_bare_copy_fail_closed(self):
        # one 13C among THREE methyl copies: the two unmodified copies must be
        # re-multiplied (3,5-dimethyl), not exploded -> out of scope, fail closed.
        assert self._call("1,2,3-trimethylbenzene", [((13, "C"), 1)]) is None

    def test_combinatorial_blowup_capped(self):
        # hexa with 5 distinct nuclides = C(6,5)*5! = 720 candidates -> reject
        # rather than spend minutes of fresh-JVM OPSIN calls.
        keys = [((13, "C"), 1), ((14, "C"), 1), ((15, "N"), 1),
                ((17, "O"), 1), ((18, "O"), 1)]
        assert self._call("1,2,3,4,5,6-hexamethylbenzene", keys) is None


class TestRefuseOnUndecoratablePolicy:
    def test_mixed_two_label_now_decorated(self):
        # HISTORY: the mixed 18O/13C two-copy case used to fail closed (v1 de-mult
        # was single-group only). The multi-nuclide de-mult extension now names it;
        # it must emit the PIN, NEVER the label-dropping unlabeled skeleton.
        from orthonym.namer import name_compound
        assert name_compound("C[18O]CCO[13CH3]", style="pin") == \
            "1-(18O)methoxy-2-(13C)methoxyethane"

    def test_labeled_molecule_never_drops_label(self):
        from orthonym.namer import name_compound
        out = name_compound("C[18O]CCO[13CH3]", style="pin")
        # the unlabeled skeleton '1,2-dimethoxyethane' must NOT be emitted, and
        # BOTH nuclide labels must appear in the name.
        assert "dimethoxyethane" not in out
        assert "18O" in out and "13C" in out

    def test_unlabeled_corpus_byte_identical(self):
        # the has_isotopes gate keeps the new policy inert on unlabeled input
        from orthonym.namer import name_compound
        assert name_compound("CCO") == "ethanol"
        assert name_compound("CCO", style="systematic") == "ethan-1-ol"

    def test_decoratable_labels_still_ship(self):
        # Task 7 target + shipped golds still emit their labeled PIN (policy only
        # fires on decorator None)
        from orthonym.namer import name_compound
        assert name_compound("[13CH3]OCCOC", style="pin") == "1-(13C)methoxy-2-methoxyethane"
        assert name_compound("[14CH3]CO", style="systematic") == "(2-14C)ethan-1-ol"


class TestFixACountSubscriptOmission:
    """FIX-A, the Blue Book-43720): the count subscript is shown "when
    polysubstitution at a single position is possible". A heavy-atom label is a
    single skeleton position (one atom of that element), and a D on a
    single-hydrogen carrier has one occupiable H, so both OMIT the subscript; a D
    on a multi-hydrogen carrier and any count>1 group KEEP it. Every expected
    string below is the verbatim Blue-Book PIN from the conformance oracle
    (benchmarks/bb_conformance/bb_measure_rows.jsonl), OPSIN-2.9.0 round-trip
    verified.
    """

    @pytest.mark.parametrize("smiles,expected", [
        ("Cl[12CH](Cl)Cl",              "trichloro(12C)methane"),          # 82.2.1
        ("[13CH3]CO",                   "(2-13C)ethan-1-ol"),              # 82.2.1
        ("Cc1cccnc1[13CH3]",            "2-(13C)methyl-3-methylpyridine"), # 82.2.2.1
        ("c1ccc2c(c1)CC[15NH]2",        "2,3-dihydro(15N)-1H-indole"),     # 82.2.3
        ("CCC(=O)O[14CH2]C",            "(1-14C)ethyl propanoate"),        # 82.2.4
        ("CCOC(=O)[14CH2]C",            "ethyl (2-14C)propanoate"),        # 82.2.4
        ("O=[14CH][O-].[Na+]",          "sodium (14C)formate"),            # 82.2.4
        ("[2H][C@@](C)(O)CC",           "(2S)-(2-2H)butan-2-ol"),          # 82.4.2, D on 1-H C
        ("[2H][C@](C)(O)[C@@H](C)Cl",   "(2R,3R)-3-chloro(2-2H)butan-2-ol"),  # 82.4.2
    ])
    def test_heavy_and_single_h_positions_omit_subscript(self, smiles, expected):
        from orthonym.namer import name_compound
        got = name_compound(smiles, style="systematic")
        assert got == expected, f"{smiles}: got {got!r} want {expected!r}"

    @pytest.mark.parametrize("smiles,expected", [
        ("[2H]C",                       "(2H1)methane"),        # methane C bears 4 H
        ("[2H]CCO",                     "(2-2H1)ethan-1-ol"),   # CH3 bears 3 H
        ("[2H]C([2H])([2H])CO",         "(2,2,2-2H3)ethan-1-ol"),  # count 3
        ("[2H]c1c([2H])c([2H])c([2H])c([2H])c1[2H]", "(2H6)benzene"),  # count 6
    ])
    def test_multi_h_or_polysubstituted_KEEP_subscript(self, smiles, expected):
        # REGRESSION PIN: a position that can bear >1 H (or a count>1 group) keeps
        # the subscript. Mutating the predicate to unconditionally drop breaks
        # these..
        from orthonym.namer import name_compound
        got = name_compound(smiles, style="systematic")
        assert got == expected, f"{smiles}: got {got!r} want {expected!r}"

    def test_trifluoroethane_keeps_subscript_on_ch3(self):
        # REGRESSION PIN: the D sits on the CH3 (C2 of 1,1,1-trifluoroethane, 3 H)
        # so the subscript is KEPT. a phase Task 2 ALSO restored the
        # position locant `2-` -- C2 is a distinguishable position of the parent
        # hydride, not a symmetric orbit, so requires the locant. The
        # pre-fix `(2H1)` (no locant) was the Sub-pattern-A defect this task fixes;
        # the full PIN is now emitted.
        from orthonym.namer import name_compound
        got = name_compound("[2H]CC(F)(F)F", style="systematic")
        assert got == "1,1,1-trifluoro(2-2H1)ethane", f"got {got!r}"

    def test_fixb_amide_nitrogen_letter_locant(self):
        # FIX-B, the Blue Book): a D on the amide nitrogen takes the letter
        # locant N, which no integer locant can express -- (N-2H1)acetamide. The
        # amide N bears 2 H, so the count subscript is KEPT. The skeleton
        # names via the retained 'acetamide' stem. Row 82.2.5 (was ABSTAIN).
        from orthonym.namer import Orthonym
        n = Orthonym(style="systematic", general_fallback=True,
                      general_fallback_unverified=True, allow_aromatic_general=True)
        assert n.name_tiered("[2H]NC(C)=O").get("name") == "(N-2H1)acetamide"

    @pytest.mark.parametrize("smiles,expected", [
        # FIX-D, BB:~43855): a centre that is chiral ONLY because a nuclide
        # breaks a local symmetry. The stripped skeleton is achiral, so the
        # stereodescriptor is cited first, ahead of the isotope descriptor. Every
        # candidate is FULL-InChIKey (stereo + isotope) round-trip gated.
        ("[2H][C@H](C)O",   "(1R)-(1-2H1)ethan-1-ol"),   # 82.4.1 / 92.3
        ("[2H]C[C@@H](C)O", "(2R)-(1-2H1)propan-2-ol"),  # 14.4 (neighbour-D induced)
    ])
    def test_fixd_isotope_induced_stereocentre(self, smiles, expected):
        from orthonym.namer import Orthonym
        n = Orthonym(style="systematic", general_fallback=True,
                      general_fallback_unverified=True, allow_aromatic_general=True)
        got = n.name_tiered(smiles).get("name")
        assert got == expected, f"{smiles}: got {got!r} want {expected!r}"

    @pytest.mark.parametrize("smiles,expected", [
        # FIX-D + de-multiplication (task-10-ISO): the induced
        # centre sits on a DE-MULTIPLIED parent -- 1,3-diiodopropan-2-ol whose two
        # iodomethyl arms are made distinct by one heavy-iodine nuclide, which is
        # precisely what makes C2 a stereocentre. The single-descriptor placement
        # cannot SPLIT `diiodo`, so `_decorate_isotope_stereo` takes its
        # constitution base from the demux helpers stereo-blind (connectivity
        # only) and then prefixes the stereodescriptor; the FINAL name is re-gated
        # on the full stereo+isotope InChIKey, so 0-wrong is preserved.
        #
        # (the Blue Book, decisive sentence: "the stereodescriptors are cited
        # first"; PIN example the Blue Book `(1R)-(1-2H1)ethan-1-ol`). The
        # substituted-compound descriptor is PARENTHESISED per
        # (the Blue Book "the nuclide symbol(s) enclosed in parentheses"), NOT the
        # labelled-compound square bracket of (the Blue Book) -- the Blue Book
        # prints `(2R)-1-(131I)iodo-3-iodopropan-2-ol (PIN)` for this exact
        # constitution.
        ("O[C@H](CI)C[131I]",  "(2R)-1-(131I)iodo-3-iodopropan-2-ol"),   # def 14.4, the Blue Book
        # def 92.3 gold was corrected `[125I]`->`(125I)`: the Blue Book (a CIP
        # illustration) prints the labelled-compound bracket, inconsistent with
        # and with the Blue Book for the same constitution.
        ("O[C@@H](CI)C[125I]", "(2S)-1-(125I)iodo-3-iodopropan-2-ol"),   # def 92.3, the Blue Book
    ])
    def test_fixd_demultiplied_induced_stereocentre(self, smiles, expected):
        from orthonym.namer import Orthonym
        n = Orthonym(style="systematic", general_fallback=True,
                      general_fallback_unverified=True, allow_aromatic_general=True)
        got = n.name_tiered(smiles).get("name")
        assert got == expected, f"{smiles}: got {got!r} want {expected!r}"

    def test_fixd_only_fires_on_induced_centres_never_wrong(self):
        # A REAL (non-isotope) stereocentre is handled by the normal stereo path,
        # not FIX-D; and an unlabelled achiral molecule must be untouched. Guards
        # that FIX-D neither regresses nor emits a wrong stereodescriptor.
        from orthonym.namer import name_compound
        # butan-2-ol is chiral WITHOUT the D (CH3 != CH2CH3): normal path.
        assert name_compound("[2H][C@@](C)(O)CC", style="systematic") == \
            "(2S)-(2-2H)butan-2-ol"

    def test_multiposition_glycine_falls_back_when_omission_unparseable(self):
        # An O-bound single D before an -oic acid suffix: the omitted (2H) spelling
        # is not parseable by OPSIN 2.9.0, so the placement search falls back to the
        # forced (2H1) form rather than abstaining -- right molecule, non-preferred
        # spelling, never silence. The NH2 group (count 2) keeps its (2H2) subscript;
        # the alpha CH2 keeps BOTH subscript and locants -> (2,2-2H2), restored by
        # a phase Task 2: C2 is a distinguishable position of the
        # ethanoic-acid parent, exactly like (2,2,2-2H3)ethan-1-ol). Both the
        # locanted and omitted spellings round-trip to the same molecule (verified).
        #
        # + the Blue Book ((2R)-1-(131I)iodo-3-iodopropan-2-ol): the descriptor is
        # inserted directly before the PART it labels (locant -> descriptor -> affix),
        # so the amino-N deuteriums are cited 2-(2H2)amino, NOT the front-detached
        # (2H2)2-amino (the same defect fixed for 5-(81Br)bromo). Both forms round-trip
        # to the identical per-D glycine (verified: same InChIKey); this is the
        # BB-conformant spelling.
        #
        # a performance pass: every hydrogen of per-D glycine is labelled, so the compound is
        # "completely isotopically substituted" and cites no locant,
        # (the Blue Book, heading ' Omission of locants'; '(2H6)
        # benzene (PIN)':44200): '(2H5)glycine'. No position has to be identified,
        # so the switch to a systematic parent (:44251) does not apply.
        # The multi-position fallback is exercised by the glycine with one amino
        # hydrogen left; its forced '(2H1)' is below the PIN,:43720).
        from orthonym.namer import Orthonym
        n = Orthonym(style="pin", general_fallback=True,
                      general_fallback_unverified=True, allow_aromatic_general=True)
        got = n.name_tiered("[2H]OC(=O)C([2H])([2H])N([2H])[2H]").get("name")
        assert got == "(2H5)glycine", f"got {got!r}"
        res = n.name_tiered("[2H]OC(=O)C([2H])([2H])N[2H]")
        assert res.get("name") == "2-(2H1)amino(2,2-2H2)ethan(2H1)oic acid", res
        assert res.get("tier") != "pin_verified" and not res.get("is_pin"), res


class TestIsotopeDescriptorPlacementAndNestedBracket:
    """ + the Blue Book ((2R)-1-(131I)iodo-3-iodopropan-2-ol): a LOCANTED
    isotope descriptor is inserted directly before the affix it modifies
    (locant -> descriptor -> affix), NOT detached at the front of the name; and
     (the Blue Book): a substituent that received an inner isotope
    descriptor steps its enclosing marks up ->.

    Multi-position witnesses (the Blue Book) and (the Blue Book) exercise
    both fixes at once: the parent 81Br gets `5-(81Br)bromo` (placement) and the
    81Br-bearing propyl substituent gets `[...]` (nested bracket)."""

    @pytest.mark.roundtrip
    @pytest.mark.parametrize("smiles,expected", [
        # (the Blue Book): parent 5-(81Br)bromo + [1-(81Br)bromopropyl].
        ("CCC([81Br])C(CC(=O)O)C(C)C(C)[81Br]",
         "5-(81Br)bromo-3-[1-(81Br)bromopropyl]-4-methylhexanoic acid"),
        # (the Blue Book): parent 4-(81Br)bromo + [2-(81Br)bromopropyl].
        ("CCC([81Br])C(CC(=O)O)CC(C)[81Br]",
         "4-(81Br)bromo-3-[2-(81Br)bromopropyl]hexanoic acid"),
    ])
    def test_placement_and_nested_bracket(self, smiles, expected):
        from orthonym.namer import name_compound
        assert name_compound(smiles, style="pin") == expected

    def test_front_placement_preserved_for_parent_scope_descriptor(self):
        # Invariant 9: the placement fix must NOT move a parent-scope descriptor
        # whose front placement IS correct -- (2H3)methanol / (2H1)benzene sit
        # directly before an alphabetic parent, so they stay at the front.
        from orthonym.namer import name_compound
        assert name_compound("[2H]C([2H])([2H])O", style="systematic") == "(2H3)methanol"
        assert name_compound("[2H]c1ccccc1", style="systematic") == "(2H1)benzene"


class TestUniformMultiplierBracketStepUp:
    """ + (the Blue Book; example:7492) -- a UNIFORM multiplied
    substituent, every copy carrying the IDENTICAL nuclide descriptor, keeps the
    multiplier and steps the enclosing marks up ->:
        1,2-di[(13C)methyl]benzene (PIN,

    NOT the whole-molecule combined descriptor 1,2-di(1,1-13C2)methylbenzene the
    single-descriptor enumeration used to build. _decorate_demultiplied declines
    the repeated-identical-nuclide case by design; _decorate_uniform_multiplier
    builds the grouped shape it leaves.
    """

    @pytest.mark.parametrize("smiles", [
        "[13CH3]c1ccccc1[13CH3]",          # aromatic SMILES
        "[13CH3]C1=C(C=CC=C1)[13CH3]",     # Kekulé SMILES (gold-set form)
    ])
    def test_di_13C_methyl_benzene(self, smiles):
        from orthonym.namer import name_compound
        assert name_compound(smiles, style="pin") == "1,2-di[(13C)methyl]benzene"
        assert name_compound(smiles, style="systematic") == "1,2-di[(13C)methyl]benzene"

    def test_uniform_branch_direct_non_hydrogen(self):
        # The new branch builds the bracket-stepped form directly (OPSIN-RT gated).
        from rdkit import Chem
        from orthonym.rules.isotopes import _decorate_uniform_multiplier
        original = Chem.MolFromSmiles("[13CH3]c1ccccc1[13CH3]")
        keys = [((13, "C"), 2, 1)]  # one uniform 13C, one atom per copy, maxpos 1
        got = _decorate_uniform_multiplier(
            "1,2-dimethylbenzene", keys, original, None)
        assert got == "1,2-di[(13C)methyl]benzene", f"got {got!r}"

    def test_hazard_deuterium_scoped_out_returns_none(self):
        # ⚠ MANDATORY hazard guard: apply_enclosing_marks routes a leading (2H)
        # through _INDICATED_H_RE and would return ((2H)methyl) (parens, no step-up)
        # instead of [(2H)methyl]; OPSIN-RT is blind to that bracket-level error. The
        # branch used to be scoped to NON-hydrogen nuclides for that reason; it now steps
        # the marks up with a stand-in descriptor (leads L7 / 17b: tests/unit/rules/
        # test_leads_l7_17b.py pins 'tetra[(2H3)methyl]silane' and the stand-in itself).
        # With no molecule to read a candidate back against (``original`` None) it places
        # nothing, whichever the nuclide.
        from orthonym.rules.isotopes import _decorate_uniform_multiplier
        assert _decorate_uniform_multiplier(
            "1,2-dimethylbenzene", [((2, "H"), 2, 1)], None, None) is None
        assert _decorate_uniform_multiplier(
            "1,2-dimethylbenzene", [((3, "H"), 2, 1)], None, None) is None

    def test_hazard_deuterium_input_never_ships_wrong_bracket(self):
        # End-to-end: the deuterium analog (two CD3) must NOT emit a wrong-parens
        # complex prefix. The branch declines (el == 'H') and control falls through
        # to the enumeration path -- an RT-valid or abstained name, never ((2H.
        from orthonym.namer import name_compound
        got = name_compound(
            "[2H]C([2H])([2H])c1ccccc1C([2H])([2H])[2H]", style="pin") or ""
        assert "((2H" not in got, f"wrong bracket shipped: {got!r}"
        assert "((3H" not in got, f"wrong bracket shipped: {got!r}"

    def test_single_labeled_copy_not_uniform_returns_none(self):
        # count 1 among two copies is the single-group de-mult shape
        # (_decorate_demultiplied), not the uniform shape -> fail closed here.
        from orthonym.rules.isotopes import _decorate_uniform_multiplier
        assert _decorate_uniform_multiplier(
            "1,2-dimethylbenzene", [((13, "C"), 1, 1)], None, None) is None

    def test_non_multiplier_skeleton_returns_none(self):
        from orthonym.rules.isotopes import _decorate_uniform_multiplier
        assert _decorate_uniform_multiplier(
            "ethan-1-ol", [((13, "C"), 1, 1)], None, None) is None

    def test_invariant9_unlabeled_and_single_prefix_unchanged(self):
        # Invariant 9: the demux path (no isotope) and a single (13C)methyl prefix
        # (count 1) are untouched by the new branch.
        from orthonym.namer import name_compound
        assert name_compound("Cc1ccccc1C", style="pin") == "1,2-dimethylbenzene"
        assert name_compound("Cc1cccnc1[13CH3]", style="pin") == \
            "2-(13C)methyl-3-methylpyridine"
