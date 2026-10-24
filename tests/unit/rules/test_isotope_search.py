"""The pruning of the isotope placement search (``rules.isotope_search``).

The search builds a candidate name for every pair of an insertion offset and a locant and reads each
back with OPSIN. OPSIN 2.9.0 refuses most pairs in two ways that say where the rest are refused: the
grammar has no place for a descriptor at the offset (``Parser.parse``: "... is unparsable due to the
following being uninterpretable"), and the part the descriptor modifies has no atom of the locant
(``Fragment.getAtomByLocantOrThrow``: "Could not find the atom with locant N."), a part being numbered
1..n. The search leaves out the pairs those refusals cover; it must give the name it gave when it asked
for all of them. Some fragments of OPSIN's data are not numbered 1..n (``TestGapsInOpsinNumbering``), and the
rules of the module hold for the pinned 2.9.0 release only.
"""
import pytest
from rdkit import Chem

from orthonym import jvm_bridge
from orthonym.rules import isotope_search, isotopes
from orthonym.rules.isotope_search import (
    GAP_WINDOW, LocantReach, direct_refusal_kind, fixed_h_inchi, integer_locant_reach, pinned_opsin)
from orthonym.validation.opsin_roundtrip import opsin_parse

pytestmark = pytest.mark.unit


def _locanted(tok):
    return f"({tok}-2H1)"


class _AsksEverything:
    """The search before the pruning: it never knows an offset or a locant OPSIN cannot read."""

    def __init__(self, *args, **kwargs):
        pass

    def beyond(self, *args, **kwargs):
        return False

    def learn(self, *args, **kwargs):
        return None


class TestRefusalKind:
    @pytest.mark.parametrize("message,kind", [
        ("me(1-2H1)thyl-benzene is unparsable due to the following being uninterpretable: "
         "me(1-2H1)thyl-benzene The following was not parseable: me(1-2H1)thyl-benzene", "grammar"),
        ("Could not find the atom with locant 7.", "locant"),
        ("Could not find the atom with locant N5.", "locant"),
        ("Could not find the atom with locant 7'.", "locant"),
        ("Could not find the atom with locant 3 for use in polycyclic spiro system", "other"),
        ("Atom is in unphysical valency state! Element: C valency: 6", "other"),
        ("Failed to assign all double bonds! (Check that indicated hydrogens have been specified)", "other"),
        ("(2H1)x could be parsed but OPSIN was unsure of the meaning of the words.", "other"),
        ("", "other"),
        (None, "other"),
    ])
    def test_message_kind(self, message, kind):
        assert jvm_bridge.refusal_kind(message) == kind

    @pytest.mark.parametrize("name,kind", [
        ("me(1-2H1)thyl-benzene", "grammar"),          # no tokenisation: a descriptor in the middle of a word
        ("(7-2H1)methylbenzene", "locant"),            # methyl has the locant 1 only
        ("(1-2H1)methylbenzene", None),                # read
        ("(2H3)methylbenzene", None),
    ])
    def test_opsin_refusal_kind(self, name, kind):
        assert opsin_parse(name) is None or kind is None
        assert jvm_bridge.opsin_refusal_kind(name, False) == kind

    def test_a_name_with_a_twin_spelling_is_not_classified(self):
        """The read-back asks the twin spelling when OPSIN refuses the name (``validation.twin_spellings``),
        so OPSIN's refusal of the name as written does not say the candidate cannot be read."""
        assert direct_refusal_kind("(1-2H1)1,4-dihydro-1,4-sulfanonaphthalene") is None
        assert direct_refusal_kind("(7-2H1)methylbenzene") == "locant"

    @pytest.mark.parametrize("message,locant", [
        ("Could not find the atom with locant 7.", "7"),
        ("Could not find the atom with locant 7'.", "7'"),
        ("Could not find the atom with locant N5.", "N5"),
        ("Could not find the atom with locant 3 for use in polycyclic spiro system", None),
        ("Atom is in unphysical valency state! Element: C valency: 6", None),
        (None, None),
    ])
    def test_the_locant_of_a_refusal(self, message, locant):
        assert jvm_bridge.refusal_locant(message) == locant

    @pytest.mark.parametrize("name,refusal", [
        ("(7-2H1)methylbenzene", ("locant", "7")),
        ("(7'-2H1)1,1'-biphenyl", ("locant", "7'")),
        ("me(1-2H1)thyl-benzene", ("grammar", None)),
        ("(1-2H1)methylbenzene", None),
    ])
    def test_opsin_names_the_locant_it_did_not_find(self, name, refusal):
        assert jvm_bridge.opsin_refusal(name, False) == refusal

    def test_a_refusal_of_another_locant_than_the_one_asked_is_not_classified(self):
        """(1,9-2H2)methylbenzene is refused for its 9, not for the 1 the search asked."""
        assert direct_refusal_kind("(1,9-2H2)methylbenzene") == "locant"
        assert direct_refusal_kind("(1,9-2H2)methylbenzene", 1) is None
        assert direct_refusal_kind("(1,9-2H2)methylbenzene", 9) == "locant"

    def test_the_jar_of_the_read_back_must_be_the_jar_of_the_jvm(self, monkeypatch):
        assert jvm_bridge.opsin_refusal_kind("(7-2H1)methylbenzene", False) == "locant"
        monkeypatch.setattr(jvm_bridge, "_same_jar", lambda jar: False)
        assert jvm_bridge.opsin_refusal_kind("(7-2H1)methylbenzene", False) is None
        assert jvm_bridge.opsin_refusal("me(1-2H1)thyl-benzene", False) is None


class TestPinnedRelease:
    """The rules are drawn from OPSIN 2.9.0: no other OPSIN prunes anything."""

    @pytest.fixture(autouse=True)
    def _fresh(self):
        isotope_search._pinned_release.cache_clear()
        yield
        isotope_search._pinned_release.cache_clear()

    def test_the_release_jar_is_pinned(self):
        assert pinned_opsin()

    @pytest.mark.parametrize("found", [
        {"label": "main", "commit": "abc", "path": "/jars/opsin-main.jar"},
        {"label": "custom", "commit": None, "path": "/jars/mine.jar"},
        {"label": "2.9.0", "commit": None, "path": None},
    ])
    def test_another_opsin_prunes_nothing(self, monkeypatch, found):
        from orthonym import jars
        monkeypatch.setattr(jars, "active_opsin", lambda *a, **k: found)
        assert not pinned_opsin()
        assert direct_refusal_kind("(7-2H1)methylbenzene") is None
        reach = LocantReach("methylbenzene", 40, _locanted)
        reach.learn(2, 1, "me" + _locanted(1) + "thylbenzene")
        assert not reach.dead and not reach.absent
        assert integer_locant_reach("methylbenzene", 6, _locanted, 40) == (False, None)

    def test_a_missing_jar_prunes_nothing(self, monkeypatch):
        from orthonym import jars

        def missing(*args, **kwargs):
            raise jars.JarUnavailable("none")

        monkeypatch.setattr(jars, "active_opsin", missing)
        assert not pinned_opsin()


class TestIntegerLocantReach:
    def test_the_numbering_of_the_part_a_descriptor_modifies(self):
        skel = "methylbenzene"
        assert integer_locant_reach(skel, 0, _locanted, 40) == (False, 1)    # (n-2H1)methyl: methyl has C1
        assert integer_locant_reach(skel, 6, _locanted, 40) == (False, 6)    # (n-2H1)benzene: C1..C6
        assert integer_locant_reach(skel, 2, _locanted, 40) == (True, 0)     # me(n-2H1)thyl: no place for a descriptor

    def test_the_limit_bounds_the_series(self):
        assert integer_locant_reach("methylbenzene", 6, _locanted, 3) == (False, 3)

    def test_the_primed_series_of_a_ring_assembly(self):
        skel = "1,1'-biphenyl"
        assert integer_locant_reach(skel, 0, _locanted, 40, "'") == (False, 6)
        assert integer_locant_reach(skel, 0, _locanted, 40) == (False, 6)


def _search(reach, skel, off, locants, descriptor_of=_locanted):
    """What the placement search does at one offset: ask ``reach`` before it builds a candidate, tell it
    about every candidate OPSIN does not read. The locants it built are returned."""
    built = []
    for locant in locants:
        if reach.beyond(off, locant):
            continue
        built.append(locant)
        candidate = skel[:off] + descriptor_of(locant) + skel[off:]
        if not isotopes._opsin_parse(candidate):
            reach.learn(off, locant, candidate)
    return built


class TestLocantReach:
    @staticmethod
    def _reach(skel, limit=40, descriptor_of=_locanted):
        return LocantReach(skel, limit, descriptor_of)

    def test_no_locant_and_a_bare_letter_are_never_beyond_reach(self):
        reach = self._reach("methylbenzene")
        assert [reach.beyond(2, locant) for locant in (None, "N", "O")] == [False, False, False]

    def test_an_offset_where_the_grammar_has_no_place_is_dead_for_every_locant(self):
        reach = self._reach("methylbenzene")
        offset = 2
        candidate = "me" + _locanted(1) + "thylbenzene"
        reach.learn(offset, 1, candidate)
        assert reach.beyond(offset, 1) and reach.beyond(offset, 5) and reach.beyond(offset, "N3")

    def test_the_first_locant_a_part_lacks_is_refused_in_the_next_spelling(self):
        reach = self._reach("methylbenzene")
        reach.learn(0, 2, _locanted(2) + "methylbenzene")
        assert not reach.beyond(0, 1)
        assert reach.beyond(0, 2)
        assert not reach.beyond(0, 3)          # nothing is known above it until the window has been asked

    def test_the_locants_above_are_refused_once_the_window_and_the_limit_were(self):
        """methylbenzene at offset 0 modifies 'methyl', which has the locant 1 only."""
        reach = self._reach("methylbenzene", limit=60)
        built = _search(reach, "methylbenzene", 0, range(1, 61))
        assert built == list(range(1, 2 + GAP_WINDOW + 1))       # 1 read, 2.. refused, the window asked, the rest not
        assert reach.caps == {0: 1}
        assert reach.beyond(0, 60) and not reach.beyond(0, 1)

    def test_another_offset_is_another_part(self):
        reach = self._reach("methylbenzene")
        _search(reach, "methylbenzene", 0, range(1, 41))
        assert reach.caps == {0: 1}
        assert not reach.beyond(6, 5)

    def test_a_search_that_ends_within_the_window_asks_nothing_more(self):
        reach = self._reach("methylbenzene", limit=60)
        built = _search(reach, "methylbenzene", 0, range(1, 6))
        assert built == [1, 2, 3, 4, 5] and not reach.caps

    def test_only_the_first_spelling_teaches(self):
        reach = self._reach("methylbenzene")
        reach.learn(0, 2, _locanted(2) + "methylbenzene", 1)
        assert not reach.beyond(0, 2) and not reach.absent

    def test_a_candidate_that_is_read_teaches_nothing(self):
        reach = self._reach("methylbenzene")
        reach.learn(0, 1, _locanted(1) + "methylbenzene")
        assert not reach.dead and not reach.absent and not reach.caps

    def test_a_refusal_of_another_kind_teaches_nothing(self):
        """A locant OPSIN finds on an atom that cannot take the descriptor is refused by the structure
        builder for the valency, and a larger locant may still be read."""
        reach = LocantReach("methylbenzene", 40, lambda tok: f"({tok}-2H4)")
        reach.learn(0, 1, "(1-2H4)methylbenzene")
        assert not reach.dead and not reach.absent and not reach.caps

    def test_a_refusal_of_another_locant_than_the_one_asked_teaches_nothing(self):
        reach = self._reach("methylbenzene")
        reach.learn(0, 1, "(1,9-2H2)methylbenzene")
        assert not reach.absent and not reach.dead

    def test_a_numbered_letter_reads_the_integers_of_its_offset_first(self):
        reach = self._reach("methylbenzene")
        assert reach.beyond(0, "N7") and not reach.beyond(0, "N1")
        assert reach.caps[0] == 1
        assert reach.beyond(2, "O1") and 2 in reach.dead

    def test_a_locant_read_above_the_first_absent_one_leaves_the_offset_uncapped(self):
        reach = self._reach("methylbenzene", limit=60)
        reach.learn(0, 2, _locanted(2) + "methylbenzene")
        reach.learn(0, 3, "(3-2H4)methylbenzene")       # refused for its valency: the locant 3 is found
        assert not reach.beyond(0, 60) and 0 in reach.gapped and not reach.caps


class TestGapsInOpsinNumbering:
    """The integer locants of a fragment are not always 1..n in OPSIN 2.9.0's data. The facts the module
    docstring states, and the search over them."""

    @pytest.mark.parametrize("parent,read,not_read", [
        ("gorgostane", (33, 34), (29, 30, 31, 32)),     # the side chain has 33 and 34, no 29-32
        ("rotenane", (21,), (13, 14, 20)),              # the "2^1" after 12
        ("aurone", (21,), (12, 20)),
    ])
    def test_the_facts(self, parent, read, not_read):
        from orthonym.validation.opsin_roundtrip import opsin_parse
        assert all(opsin_parse(_locanted(n) + parent) for n in read)
        assert not any(opsin_parse(_locanted(n) + parent) for n in not_read)

    @pytest.mark.parametrize("parent,read", [
        ("gorgostane", (33, 34)),
        ("rotenane", (21,)),
        ("aurone", (21,)),
    ])
    def test_the_search_leaves_no_locant_OPSIN_reads_out(self, parent, read):
        reach = LocantReach(parent, 45, _locanted)
        built = _search(reach, parent, 0, range(1, 46))
        assert all(n in built for n in read)
        assert 0 not in reach.caps
        assert integer_locant_reach(parent, 0, _locanted, 45) == (False, None)

    def test_arginines_far_locant_is_the_exception_the_window_does_not_see(self):
        """Arginine reads (99-13C) (its guanidino carbon) and not (6-13C): the search of a molecule of 99 heavy
        atoms or more caps arginine at 5 and does not build it. That loses a name and never makes a wrong one,
        and no search reaches it today (module docstring); the pin stays here so the next such parent is
        added to this list."""
        from orthonym.validation.opsin_roundtrip import opsin_parse

        def carbon(tok):
            return f"({tok}-13C)"

        assert opsin_parse(carbon(99) + "arginine") and not opsin_parse(carbon(6) + "arginine")
        reach = LocantReach("arginine", 120, carbon)
        built = _search(reach, "arginine", 0, range(1, 121), carbon)
        assert 99 not in built and reach.caps == {0: 5}


class TestSearchIsUnchanged:
    """The name the pruned search gives is the name of the search that asks for every pair."""

    @pytest.mark.parametrize("smiles", [
        "[2H]c1ccccc1C",                                    # a ring position: an integer locant
        "[13CH3]CCO",                                       # a heavy nuclide on a chain
        "[2H]NC(C)CN",                                      # a numbered letter locant (N1, N2)
        "[2H]C1CCC([2H])CC1",                               # one nuclide at two positions (distinct multi-locant)
        "[2H]c1ccc([2H])cc1-c1ccc([2H])cc1",                # the primed locants of a ring assembly
        "CC(=O)Nc1ccc([2H])cc1.[2H]C([2H])([2H])Oc1ccccc1",  # two parts of a mixture
    ])
    def test_name(self, smiles, monkeypatch):
        from orthonym import Orthonym
        engine = Orthonym(general_fallback=True, general_fallback_unverified=True, allow_aromatic_general=True)
        with monkeypatch.context() as full:
            full.setattr(isotopes, "LocantReach", _AsksEverything)
            full.setattr(isotopes, "integer_locant_reach", lambda *a, **k: (False, None))
            expected = isotopes.decorate_isotopic_name(smiles, engine.style, engine)
        assert expected
        assert isotopes.decorate_isotopic_name(smiles, engine.style, engine) == expected

    def test_it_asks_for_fewer_pairs(self, monkeypatch):
        """A name of several parts: the offsets in the middle of its words, and the locants no part has, are
        not built."""
        from orthonym import Orthonym
        engine = Orthonym(general_fallback=True, general_fallback_unverified=True, allow_aromatic_general=True)
        smiles = "[2H]C([2H])([2H])c1ccc(cc1)C(=O)Nc1ccc(cc1)C(F)(F)F"
        asked = []
        real = isotopes._isotope_round_trips

        def counting(candidate, original, stereo_blind=False):
            asked.append(candidate)
            return real(candidate, original, stereo_blind=stereo_blind)

        monkeypatch.setattr(isotopes, "_isotope_round_trips", counting)
        name = isotopes.decorate_isotopic_name(smiles, engine.style, engine)
        pruned = len(asked)
        asked.clear()
        monkeypatch.setattr(isotopes, "LocantReach", _AsksEverything)
        monkeypatch.setattr(isotopes, "integer_locant_reach", lambda *a, **k: (False, None))
        assert isotopes.decorate_isotopic_name(smiles, engine.style, engine) == name
        assert pruned <= len(asked)


class TestFixedHInchiMemo:
    def test_same_string_as_inchi(self):
        from rdkit.Chem import inchi
        mol = Chem.MolFromSmiles("[2H]OC(=O)CN")
        assert fixed_h_inchi(mol) == inchi.MolToInchi(mol, options="/FixedH")

    def test_one_structure_is_computed_once(self, monkeypatch):
        from rdkit.Chem import inchi
        calls = []
        real = inchi.MolToInchi

        def counting(mol, **kwargs):
            calls.append(Chem.MolToSmiles(mol))
            return real(mol, **kwargs)

        monkeypatch.setattr(inchi, "MolToInchi", counting)
        isotope_search._FIXED_H_INCHI.clear()
        first = fixed_h_inchi(Chem.MolFromSmiles("OC(=O)CN[2H]"))
        again = fixed_h_inchi(Chem.MolFromSmiles("[2H]NCC(=O)O"))     # the same structure, written otherwise
        other = fixed_h_inchi(Chem.MolFromSmiles("[2H]OC(=O)CN"))      # the D on the other exchangeable hydrogen
        assert first == again != other
        assert len(calls) == 2

    def test_a_radical_electron_is_part_of_the_key(self):
        """'[CH2]CC(=O)O' with and without a radical electron is one canonical SMILES and two InChIs
        (the one without has the hydrogen of a closed shell)."""
        from rdkit.Chem import inchi
        radical = Chem.MolFromSmiles("[CH2]CC(=O)O")
        closed = Chem.MolFromSmiles("[CH2]CC(=O)O", sanitize=False)
        closed.UpdatePropertyCache(strict=False)
        assert Chem.MolToSmiles(radical) == Chem.MolToSmiles(closed)
        assert inchi.MolToInchi(radical, options="/FixedH") != inchi.MolToInchi(closed, options="/FixedH")
        isotope_search._FIXED_H_INCHI.clear()
        assert fixed_h_inchi(radical) == inchi.MolToInchi(radical, options="/FixedH")
        assert fixed_h_inchi(closed) == inchi.MolToInchi(closed, options="/FixedH")

    def test_the_fixed_h_layer_still_tells_the_exchangeable_positions_apart(self):
        a = Chem.MolFromSmiles("[2H]OC(=O)CN")
        b = Chem.MolFromSmiles("OC(=O)CN[2H]")
        assert not isotopes._same_fixed_h_inchi(a, b)
        assert isotopes._same_fixed_h_inchi(a, Chem.MolFromSmiles("[2H]OC(=O)CN"))
