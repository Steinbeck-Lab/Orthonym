""" at PARENT scope -- Phase C Task 5b.

Governing rule chain, verbatim from ``the Blue Book Blue Book`` with headings.

```` "Citation of locants" (``:2869``) is the **DENY-DEFAULT** --

    "In preferred IUPAC names, if any locants are essential for defining the structure
     of the parent structure or of a unit of structure as defined by its appropriate
     enclosing marks, then all locants must be cited for the parent structure or that
     structural unit."

```` (``:3007``), under ```` "Omission of locants", grants the
licence --

    "All locants are omitted in compounds or substituent groups in which all
     substitutable positions are completely substituted or modified, for example, by
     hydro, in the same way. Except for hydrogen atoms attached to chalcogen atoms,
     such as in acids, alcohols, and to the carbon atoms of formyl groups (aldehydes),
     all hydrogen atoms are considered substitutable."

and its counter-clause ``:3009`` is the tripwire --

    "In case of partial substitution or modification, all numerical prefixes must be
     indicated. The prefix 'per-' is no longer recommended."

THE VERBATIM WITNESS is ``:3017`` ``heptafluorobutanoic acid (PIN)``. The arithmetic:
butanoic acid ``CH3-CH2-CH2-COOH`` has C2(2H) + C3(2H) + C4(3H) = **7** substitutable
hydrogens -- C1 has none of its own and its ``-OH`` hydrogen sits on a CHALCOGEN, which
``:3007`` excludes -- so seven fluorines exhaust the set, uniformly, and every locant
goes.

★ THE SHARPEST BOUNDARY PAIR IN THE PHASE, and it needs no special case::

    F3C-CF2-COOH -> pentafluoropropanoic acid OMITS
    F3C-CF2-CO-NH2 -> 2,2,3,3,3-pentafluoropropanamide KEEPS

Identical fluorination. The acid omits because its only remaining hydrogen is on
oxygen and is carved out. Propanamide's substitutable set is C2(2H) + C3(3H) **+ the
amide N-H(2H)**; an amide N-H is neither a chalcogen H nor a formyl H, so it counts, it
is unsubstituted, the substitution is *partial*, and ``:3009`` restores every locant.
That an amide N-H is substitutable is proven independently by ``:2889``
``N1,N3-dimethylpropanediamide (PIN)``. A wiring that moves the amide row has
implemented "fluorines everywhere => drop locants", not.

THE CLASS IS OPEN. ``:3009`` retires the ``per-`` contraction, which *was* exactly a
closed-list mechanism, and the 2013 recommendations replaced it with counting. ``:3017``
is the only fully-substituted acyclic PARENT printed in the Blue Book, so
``pentafluoropropanoic acid``, ``nonafluoropentanoic acid``,
``pentachloropropanoic acid``, ``octafluoropropane`` and ``hexafluoropentanedioic
acid`` below are DERIVED from the predicate, not verbatim rows. A table keyed on the
spelling would be wrong on its complement by construction.

MEASURED CODE PATH (call-trace validated on 2 known positives -- ``chloropropanedioic
acid`` and ``chlorobutanedioic acid``, whose sibling licence is wired at the
same site -- and 2 known negatives, ``ethanol`` and ``benzene``, which record ZERO
calls at EVERY candidate site):
  * LIVE: ``handlers/_handler_shared._assemble_fragments``, 1 call, productive (its
    return value IS the whole emitted name), reached from
    ``handlers/general_acyclic.name_general_acyclic``;
  * REFUTED as off-path: ``composer._format_prefix_groups`` -- never called for any
    row in this class (it is the substituent-scope joiner);
  * NESTED INSIDE the live site, so not a separate wiring point:
    ``composition_primitives._join_prefixes``.

★ THE CRITICAL GUARD IS ALSO OFF-PATH, which is why it is additionally asserted at the
PREDICATE level below: ``FC(F)(F)C(F)(F)C(=O)N`` records **zero** calls at every
instrumented site, because ``_is_general_acyclic`` defers a single-group primary amide
to composer's inline amide branch. An end-to-end assertion alone would therefore be
green for a reason that has nothing to do with the licence.

Invariant 11: removing a locant can unmask something worse -- in that happened four
times, and in Task 5a dropping these very locants LOST THE ENCLOSING MARKS
(``heptafluoropropylbenzene`` for ``(heptafluoropropyl)benzene``). Every check below
asserts the FULL emitted name, never merely that a locant vanished.
"""
import dataclasses

import pytest
from rdkit import Chem

from orthonym.assembly.composer import NameFragment
from orthonym.assembly.fragment_naming import _get_visited
from orthonym.assembly.handlers._handler_shared import (
    _l5_prefix_locants_omitted,
    _naming_call_produces_a_name_component,
)
from orthonym.assembly.locant_omission import (
    forced_locant_scope,
    isotopic_naming_scope,
    substitutable_h_count,
    substitutable_positions,
)
from orthonym.namer import Orthonym


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


# --------------------------------------------------------------------------- #
# A features stand-in for the PREDICATE-level checks. #
# --------------------------------------------------------------------------- #
# Needed because the critical amide guard is OFF-PATH end-to-end (measured: zero
# calls), so an end-to-end assertion cannot exercise the licence on it. The stub
# carries exactly the five attributes the predicate reads, and it is VALIDATED against
# a known positive (`test_stub_harness_is_valid_on_a_known_positive`) -- a stub that
# denied everything would make every negative below vacuously green, which is the
# `feedback_harness_that_reports_success` shape.
@dataclasses.dataclass
class _FeaturesStub:
    mol: object
    principal_chain: tuple
    principal_group_atoms: tuple = ()
    oriented_ring: object = None
    principal_ring: object = None


def _fragments(stem, suffix_text, suffix_locants, prefix_text, prefix_locants,
               *, count=1, parent_locants=((), ())):
    frags = [NameFragment(text=stem, locants=parent_locants, fragment_type="parent")]
    if suffix_text is not None:
        frags.append(NameFragment(text=suffix_text, locants=tuple(suffix_locants),
                                  fragment_type="suffix"))
    frags.append(NameFragment(text=prefix_text, locants=tuple(prefix_locants),
                              fragment_type="prefix", count=count))
    return frags


# The two stub scopes used throughout. Atom indices are read off the SMILES ordering
# and asserted below, never taken on trust.
_ACID = "OC(=O)C(F)(F)C(F)(F)F"        # O0 C1 O2 C3 F4 F5 C6 F7 F8 F9
_AMIDE = "FC(F)(F)C(F)(F)C(=O)N"       # F0 C1 F2 F3 C4 F5 F6 C7 O8 N9


def _acid_stub():
    mol = Chem.MolFromSmiles(_ACID)
    return _FeaturesStub(mol=mol, principal_chain=(1, 3, 6),
                         principal_group_atoms=((1, 2, 0),))


def _amide_stub():
    mol = Chem.MolFromSmiles(_AMIDE)
    return _FeaturesStub(mol=mol, principal_chain=(7, 4, 1),
                         principal_group_atoms=((7, 8, 9),))


def test_stub_atom_indices_are_what_the_tests_assume():
    """The stubs' hardcoded indices are DERIVED here, so a RDKit ordering change
    breaks this test rather than silently re-pointing every stub below."""
    acid = Chem.MolFromSmiles(_ACID)
    assert [a.GetSymbol() for a in acid.GetAtoms()] == [
        "O", "C", "O", "C", "F", "F", "C", "F", "F", "F"]
    amide = Chem.MolFromSmiles(_AMIDE)
    assert [a.GetSymbol() for a in amide.GetAtoms()] == [
        "F", "C", "F", "F", "C", "F", "F", "C", "O", "N"]


def test_stub_harness_is_valid_on_a_known_positive():
    """★ VALIDATE THE HARNESS BEFORE BELIEVING ANY NEGATIVE IT PRODUCES.

    The stub must make the licence FIRE on the acid; only then does its answer of
    False on the amide mean something.
    """
    assert _l5_prefix_locants_omitted(
        _acid_stub(),
        _fragments("prop", "oic acid", (), "pentafluoro", (2, 2, 3, 3, 3)),
    ) is True


# --------------------------------------------------------------------------- #
# 1. The licence fires -- whole emitted name asserted #
# --------------------------------------------------------------------------- #
class TestLicensedOmission:
    def test_the_verbatim_pin_witness(self, namer):
        """★ ``:3017`` ``heptafluorobutanoic acid (PIN)``, byte-identical.

        7 of 7 substitutable H (C2 2H + C3 2H + C4 3H); the acid O-H is excluded by
        ``:3007``'s chalcogen carve-out and C1 has no H of its own.
        """
        assert namer.name("OC(=O)C(F)(F)C(F)(F)C(F)(F)F") == "heptafluorobutanoic acid"

    def test_one_carbon_shorter_is_the_same_licence(self, namer):
        """5 of 5 (C2 2H + C3 3H). Derived; no verbatim Blue Book row."""
        assert namer.name("OC(=O)C(F)(F)C(F)(F)F") == "pentafluoropropanoic acid"

    @pytest.mark.parametrize("smiles,expected", [
        # The halogen is not part of the rule -- only "in the same way" is.
        ("OC(=O)C(Cl)(Cl)C(Cl)(Cl)Cl", "pentachloropropanoic acid"),
        ("OC(=O)C(Br)(Br)C(Br)(Br)Br", "pentabromopropanoic acid"),
        # Chain length is not part of the rule either.
        ("OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F", "nonafluoropentanoic acid"),
        # A DIacid: both -OH are chalcogen H, so C2/C3/C4 are the whole set (6 H).
        ("OC(=O)C(F)(F)C(F)(F)C(F)(F)C(=O)O", "hexafluoropentanedioic acid"),
        # ``:3007``'s SECOND carve-out: "and to the carbon atoms of formyl groups
        # (aldehydes)". The formyl C-H does not count, so C2+C3 is the whole set.
        ("FC(F)(F)C(F)(F)C=O", "pentafluoropropanal"),
        # acetaldehyde is the retained parent with substitution allowed,
        # the Blue Book; 'phenoxyacetaldehyde (PIN)':35076), all its hydrogen on one
        # carbon, so the locants are omitted ('trifluoroacetaldehyde', as 'trifluoroacetic acid')
        ("FC(F)(F)C=O", "trifluoroacetaldehyde"),
        # A nitrile carbon carries no hydrogen at all.
        ("FC(F)(F)C(F)(F)C(F)(F)C#N", "heptafluorobutanenitrile"),
        ("FC(F)(F)C(F)(F)C#N", "pentafluoropropanenitrile"),
        # No suffix at all -- a bare parent hydride.
        ("FC(F)(F)C(F)(F)F", "hexafluoroethane"),
        ("ClC(Cl)(Cl)C(Cl)(Cl)Cl", "hexachloroethane"),
        ("FC(F)(F)C(F)(F)C(F)(F)F", "octafluoropropane"),
        ("FC(F)(F)C(F)(F)C(F)(F)C(F)(F)F", "decafluorobutane"),
    ])
    def test_the_open_class_derived_members(self, namer, smiles, expected):
        """The predicate entails these; the Blue Book prints none of them.

        A finite table keyed on ``heptafluoro`` would be wrong on every row here.
        """
        assert namer.name(smiles) == expected


# --------------------------------------------------------------------------- #
# 2. ★ THE CRITICAL GUARD -- the amide keeps #
# --------------------------------------------------------------------------- #
class TestAmideNitrogenHydrogenIsSubstitutable:
    def test_pentafluoropropanamide_keeps_every_locant(self, namer):
        """★ Identical fluorination to ``pentafluoropropanoic acid``, opposite answer.

        5 of 7 substitutable H replaced (the two amide N-H are not), so ``:3009``
        applies: "In case of partial substitution or modification, all numerical
        prefixes must be indicated."
        """
        assert namer.name("FC(F)(F)C(F)(F)C(=O)N") == "2,2,3,3,3-pentafluoropropanamide"

    def test_and_at_the_predicate_level_because_the_row_is_OFF_PATH(self):
        """The end-to-end assertion above is green for a reason unrelated to the
        licence -- a measured ZERO calls at every join site. So the licence is put the
        question directly, with the same scope the acid stub uses (which the harness
        check above proves DOES fire)."""
        assert _l5_prefix_locants_omitted(
            _amide_stub(),
            _fragments("prop", "amide", (), "pentafluoro", (2, 2, 3, 3, 3)),
        ) is False

    def test_the_arithmetic_that_separates_them(self):
        """The whole boundary in two numbers, read off ``:3007`` directly."""
        acid = Chem.MolFromSmiles("CCC(=O)O")           # propanoic acid
        amide = Chem.MolFromSmiles("CCC(=O)N")          # propanamide
        assert sum(substitutable_h_count(acid, i)
                   for i in substitutable_positions(acid)) == 5
        assert sum(substitutable_h_count(amide, i)
                   for i in substitutable_positions(amide)) == 7
        #...and the reason is exactly one atom: the acid's O-H is on a chalcogen.
        assert substitutable_h_count(acid, 4) == 0      # the -OH oxygen
        assert substitutable_h_count(amide, 4) == 2     # the amide nitrogen

    @pytest.mark.parametrize("smiles,expected", [
        ("NC(=O)C(F)(F)C(F)(F)C(F)(F)F", "2,2,3,3,4,4,4-heptafluorobutanamide"),
        ("CNC(=O)C(F)(F)C(F)(F)F", "2,2,3,3,3-pentafluoro-N-methylpropanamide"),
        # ``:2889`` proves the N-H substitutable; this row must stay put.
        ("CNC(=O)CC(=O)NC", "N1,N3-dimethylpropanediamide"),
        # ⚠ urea is the ONE member here that OMITS: its four N-H are a single orbit
        # (one kind of substitutable H), so (:2943 `methylurea (PIN)`)
        # fires -- unlike the diamides above, whose C-H + N-H give two kinds.
        ("CNC(=O)N", "methylurea"),
    ])
    def test_the_whole_amide_family_keeps(self, namer, smiles, expected):
        assert namer.name(smiles) == expected

    @pytest.mark.parametrize("smiles", [
        # ★ The EXACT SMILES of gold row D8-AMIDINE-LOCANT in
        # benchmarks/the gold set/packs/characteristic_groups.json -- asserted verbatim so
        # the gate row and this suite cannot drift apart.
        "FC(F)(C(F)(F)F)C(=N)N",
        #...and a second spelling of the same molecule, so the assertion is about the
        # structure and not about one SMILES traversal.
        "NC(=N)C(F)(F)C(F)(F)F",
    ])
    def test_the_gold_row_amidine_keeps(self, namer, smiles):
        """Gold row ``D8-AMIDINE-LOCANT``: 5 of 8 substitutable H, because
        the three amidine N-H are unsubstituted => partial => ``:3009`` retains."""
        assert namer.name(smiles) == "2,2,3,3,3-pentafluoropropanimidamide"


# --------------------------------------------------------------------------- #
# 3. PARTIAL substitution -- ``:3009`` restores every locant #
# --------------------------------------------------------------------------- #
class TestPartialSubstitutionRetains:
    @pytest.mark.parametrize("smiles,expected,why", [
        ("OC(=O)CC(F)(F)F", "3,3,3-trifluoropropanoic acid",
         "C2 keeps both its hydrogens"),
        ("OC(=O)C(F)C(F)(F)F", "2,3,3,3-tetrafluoropropanoic acid",
         "C2 keeps 1 of 2 -- partial AT A SINGLE POSITION still denies"),
        ("FC(F)C(F)F", "1,1,2,2-tetrafluoroethane",
         "every position partial, none complete"),
        ("OCC(F)(F)F", "2,2,2-trifluoroethan-1-ol", "C1 keeps both hydrogens"),
    ])
    def test_partial(self, namer, smiles, expected, why):
        assert namer.name(smiles) == expected, why

    def test_the_blue_books_own_parent_scope_negative(self, namer):
        """``:3019`` ``CF3-CF2-CH2-OH -> 2,2,3,3,3-pentafluoropropan-1-ol (PIN)``.

        Printed WITH its locants two lines after ``heptafluorobutanoic acid``, and for
        one reason only: C1 is a ``-CH2-OH``, so the propane skeleton is not completely
        substituted.
        """
        assert namer.name("OCC(F)(F)C(F)(F)F") == "2,2,3,3,3-pentafluoropropan-1-ol"
        assert namer.name("FC(F)(F)C(F)(F)CO") == "2,2,3,3,3-pentafluoropropan-1-ol"

    def test_the_46359_near_miss_shape(self, namer):
        """``:46359``'s ``(1,1,1,3,3,3-hexafluoropropan-2-yl)`` shape as a parent: the
        MIDDLE carbon keeps its hydrogen, so a predicate that checked only the terminal
        carbons would wrongly fire."""
        assert namer.name("OC(C(F)(F)F)C(F)(F)F") == \
            "1,1,1,3,3,3-hexafluoropropan-2-ol"

    def test_the_41664_minimal_pair(self, namer):
        """``:41664`` ``Cl2CH-CH2-S+ -> (2,2-dichloroethyl)sulfanylium (PIN)`` --
        2 of 5 replaced on the same ethyl skeleton."""
        assert namer.name("ClC(Cl)CS") == "2,2-dichloroethane-1-thiol"

    def test_the_3529_pervasive_negative(self, namer):
        """``:3529`` ``6-(1-chloroethyl)-5-(2-chloroethyl)-1H-indole (PIN)`` -- 1 of 5
        replaced, inside a real complex PIN."""
        assert namer.name("CC(Cl)c1cc2cc[nH]c2cc1") == "5-(1-chloroethyl)-1H-indole"


# --------------------------------------------------------------------------- #
# 4. UNIFORMITY -- "in the same way" #
# --------------------------------------------------------------------------- #
class TestUniformity:
    def test_the_29619_failure_mode(self, namer):
        """``:29619`` is the Blue Book CITING to explain a NEGATIVE:
        ``...-pentadecafluorooctan-1-one (PIN, the locants for the fluoro substituents
        are required, see ``. Every carbon there has zero hydrogens, but C1
        is substituted by something that is NOT fluorine.

        Same shape here: the chain is exhausted, but not "in the same way".
        """
        assert namer.name("OC(=O)C(F)(F)C(F)(F)C(F)(F)Cl") == \
            "4-chloro-2,2,3,3,4,4-hexafluorobutanoic acid"

    def test_two_kinds_at_one_position(self, namer):
        assert namer.name("OC(=O)C(Cl)(F)C(F)(F)F") == \
            "2-chloro-2,3,3,3-tetrafluoropropanoic acid"

    def test_a_non_halogen_breaking_the_uniformity_is_also_caught(self):
        """Predicate level, because the marshalling must include EVERY prefix and not
        only halogens -- the reason ``:29619`` is detectable at all."""
        stub = _acid_stub()
        frags = _fragments("prop", "oic acid", (), "pentafluoro", (2, 2, 3, 3, 3))
        assert _l5_prefix_locants_omitted(stub, frags) is True
        # Same molecule, same scope, but the name now cites a SECOND kind at C2.
        frags2 = list(frags) + [NameFragment(text="methyl", locants=(2,),
                                             fragment_type="prefix")]
        assert _l5_prefix_locants_omitted(stub, frags2) is False


# --------------------------------------------------------------------------- #
# 5. -- another cited locant in the scope restores them all #
# --------------------------------------------------------------------------- #
class TestOtherCitedLocantsRestoreEverything:
    def test_a_cited_suffix_locant_keeps_a_COMPLETELY_substituted_chain(self, namer):
        """★ The sharpest witness for the suffix-locant clause, because the chain here
        IS completely and uniformly fluorinated -- 7 of 7 -- and the locants stay
        anyway.

        ``propan-1-ol``'s ``1`` is essential (it distinguishes propan-2-ol), so
        's *"then all locants must be cited"* restores the prefix locants.
        """
        assert namer.name("FC(F)(F)C(F)(F)C(F)(F)O") == \
            "1,1,2,2,3,3,3-heptafluoropropan-1-ol"

    def test_propan_2_one_is_one_of_the_four_hardcoded_keepers(self, namer):
        """``:3005``: "As an exception the locant is not omitted from propan-2-one,
        butan-2-one, prop-2-enoic acid and prop-2-ynoic acid although unambiguous
        without a locant." That cited ``2`` is in the same scope."""
        assert namer.name("FC(F)(F)C(=O)C(F)(F)F") == \
            "1,1,1,3,3,3-hexafluoropropan-2-one"

    @pytest.mark.parametrize("smiles,expected", [
        ("FC(F)(F)C(F)(F)S", "1,1,2,2,2-pentafluoroethane-1-thiol"),
        # Also partial (the amine N-H count), so it keeps for two reasons.
        ("FC(F)(F)C(F)(F)N", "1,1,2,2,2-pentafluoroethan-1-amine"),
    ])
    def test_other_cited_suffix_locants(self, namer, smiles, expected):
        assert namer.name(smiles) == expected

    def test_an_unsaturation_locant_in_the_parent_keeps(self, namer):
        """RECORDED BOUNDARY, taken on the deny-by-default side. Ethene's own name
        cites no locant (``(d)`` omits it for unsubstituted dinuclear
        alkenes), so the elided ``tetrafluoroethene`` is arguably licensed -- but the
        Blue Book prints no example either way, and the parent fragment DOES carry an
        unsaturation locant, so is applied."""
        assert namer.name("FC(F)=C(F)F") == "1,1,2,2-tetrafluoroethene"

    def test_a_stereodescriptor_in_the_scope_denies(self):
        stub = _acid_stub()
        frags = _fragments("prop", "oic acid", (), "pentafluoro", (2, 2, 3, 3, 3))
        assert _l5_prefix_locants_omitted(stub, frags) is True
        frags2 = list(frags) + [NameFragment(text="(2R)-", fragment_type="stereo")]
        assert _l5_prefix_locants_omitted(stub, frags2) is False

    def test_a_parent_stem_that_is_not_purely_alphabetic_denies(self):
        stub = _acid_stub()
        frags = _fragments("1H-prop", "oic acid", (), "pentafluoro", (2, 2, 3, 3, 3))
        assert _l5_prefix_locants_omitted(stub, frags) is False

    def test_a_prefix_that_already_names_its_own_locants_denies(self):
        stub = _acid_stub()
        frags = _fragments("prop", "oic acid", (), "2,2,3,3,3-pentafluoro",
                           (2, 2, 3, 3, 3))
        assert _l5_prefix_locants_omitted(stub, frags) is False


# --------------------------------------------------------------------------- #
# 6. ★ The two AMBIENT scopes #
# --------------------------------------------------------------------------- #
class TestAmbientScopes:
    def test_an_isotopic_label_keeps_every_locant_end_to_end(self, namer):
        """★ **** (``:44180``): "In preferred IUPAC names, locants are
        omitted if no locants are necessary in unmodified names. However, if isotopic
        modification requires a locant to specify its position, then all locants must
        be specified and none are omitted."

        The two propanoic-acid carbons C2/C3 are inequivalent, so the ``(2-13C)``
        position must be stated -- and then every locant in the scope must be.
        (The count subscript is omitted per, FIX-A: a carbon position
        holds one carbon.)
        """
        assert namer.name("FC(F)(F)[13C](F)(F)C(=O)O") == \
            "2,2,3,3,3-pentafluoro(2-13C)propanoic acid"

    def test_the_WEAKER_declaration_is_the_one_that_fires(self):
        """★ MEASURED, and it is why ``locants_are_forced`` alone is insufficient.

        ``rules/isotopes.py`` strips the labels before naming and enters
        ``forced_locant_scope`` only CONDITIONALLY, so at the live site the labelled
        acid above presents with ``locants_are_forced == False`` and every
        ``GetIsotope`` reading 0 -- BOTH structural signals negative. Only the
        unconditional ``isotopic_naming_scope`` denies. Asserted here directly so a
        future refactor that drops the weaker consult fails.
        """
        stub, frags = _acid_stub(), _fragments(
            "prop", "oic acid", (), "pentafluoro", (2, 2, 3, 3, 3))
        assert all(a.GetIsotope() == 0 for a in stub.mol.GetAtoms())
        assert _l5_prefix_locants_omitted(stub, frags) is True
        with isotopic_naming_scope("isotope"):
            assert _l5_prefix_locants_omitted(stub, frags) is False

    def test_the_stronger_declaration_also_denies(self):
        stub, frags = _acid_stub(), _fragments(
            "prop", "oic acid", (), "pentafluoro", (2, 2, 3, 3, 3))
        with forced_locant_scope("test"):
            assert _l5_prefix_locants_omitted(stub, frags) is False

    @pytest.mark.parametrize("smiles,expected", [
        # Task 5a's isotope witness, substituent scope -- must stay fixed.
        ("FC(F)(F)[13C](F)(F)C1CCCCC1",
         "[1,1,2,2,2-pentafluoro(1-13C)ethyl]cyclohexane"),
        # (``:44202``) ``(2H6)benzene (PIN)``: when every candidate
        # position is ONE orbit no locant is needed, so these keep their omission.
        # The count subscript is omitted per (FIX-A: a carbon position
        # holds one carbon); the locant behaviour under test is unchanged.
        ("Cc1c(C)c(C)c(C)c(C)[13c]1C", "hexamethyl(13C)benzene"),
        ("Oc1c(O)c(O)c(O)c(O)[13c]1O", "(13C)benzenehexol"),
        ("OC[13CH3]", "(2-13C)ethan-1-ol"),
        ("OC1CCCC[13CH2]1", "(2-13C)cyclohexan-1-ol"),
    ])
    def test_the_isotope_rows_already_shipped_do_not_move(self, namer, smiles, expected):
        assert namer.name(smiles) == expected


# --------------------------------------------------------------------------- #
# 6b. ★ The scope boundary must BE the molecule (the:29619 regression) #
# --------------------------------------------------------------------------- #
class TestTheScopeBoundaryMustBeTheWholeMolecule:
    """★ INVARIANT 11 FIRED HERE during implementation, and this is the whole story.

    ```` scopes citation to a unit *"as defined by its appropriate enclosing
    marks"*. For ``F(CF2)7-CO-N(piperidine)`` the decomposition engine cuts the acyl
    bond, CAPS the fragment as the free acid, and asks for a whole-molecule name of
    ``pentadecafluorooctanoic acid`` -- for which the licence genuinely
    fires, 15 of 15 -- then rewrites ``oic acid`` -> ``oyl`` and splices it in. The
    resulting scope cites ``N``, an essential letter locant, so every fluoro locant
    must be cited: the general-nomenclature form the engine emits today is
    ``N-(2,2,...,8,8,8-pentadecafluorooctanoyl)piperidine`` (0-wrong, OPSIN-RT-valid).

    That form is CORRECT but NON-PIN. The PIN is the substituted-acyl PSEUDOKETONE
    ``2,2,...,8,8,8-pentadecafluoro-1-(piperidin-1-yl)octan-1-one`` /
    ; verified OPSIN-RT-MATCH to the input InChIKey
    ``an InChIKey``). The Blue Book's OWN example for this rule
    confirms the pseudoketone CONSTRUCTION: ``:29619`` names the sibling molecule
    ``1-[4-(3,4-dihydroisoquinoline-2(1H)-carbonyl)piperidin-1-yl]-2,2,...,8,8,8-
    pentadecafluorooctan-1-one (PIN, the locants for the fluoro substituents are
    required, see `` -- a DIFFERENT decorated molecule but
    likewise an ``octan-1-one`` pseudoketone whose fluoro locants are required. It
    is therefore NOT authority for the old ``N-...oylpiperidine`` spelling, which is
    not re-asserted here.

    The engine does not yet BUILD the substituted-acyl pseudoketone, so the assertion
    below pins the PIN as ``xfail(strict=True)`` -- a canary that XPASSes (and so trips
    the strict marker, flagging its own removal) the moment the pseudoketone build
    lands.
    """

    _29619 = ("FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(=O)N1CCCCC1")

    @pytest.mark.xfail(strict=True, reason=(
        "pseudoketone is the PIN per P-66.1.3/P-64.1.2.1; engine emits a correct "
        "general-nomenclature N-acyl form (0-wrong, RT-valid) but does not yet build "
        "the substituted-acyl pseudoketone -- canary XPASSes when the pseudoketone "
        "build lands"))
    def test_the_blue_books_own_negative_for_this_very_rule(self, namer):
        # PIN = the substituted-acyl pseudoketone /; verified
        # OPSIN-RT-MATCH to the input an InChIKey.
        assert namer.name(self._29619) == (
            "2,2,3,3,4,4,5,5,6,6,7,7,8,8,8-pentadecafluoro-1-(piperidin-1-yl)octan-1-one")

    def test_the_signal_is_false_for_a_whole_molecule_naming(self, namer):
        assert _naming_call_produces_a_name_component() is False
        assert namer.name("OC(=O)C(F)(F)C(F)(F)C(F)(F)F") == \
            "heptafluorobutanoic acid"
        assert _naming_call_produces_a_name_component() is False

    def test_the_signal_is_true_while_a_fragment_naming_is_in_progress(self):
        """The signal is ``fragment_naming``'s visited-SMILES set, which that module
        populates before delegating to ``name_compound`` and discards in a
        ``finally``. Simulated here rather than inferred."""
        visited = _get_visited()
        assert not visited
        visited.add("O=C(O)CC")
        try:
            assert _naming_call_produces_a_name_component() is True
        finally:
            visited.discard("O=C(O)CC")
        assert _naming_call_produces_a_name_component() is False

    def test_the_licence_declines_inside_a_fragment_naming(self):
        """The mutation-killing form: the SAME scope that fires at top level must be
        refused while the result is destined to become a name component."""
        stub, frags = _acid_stub(), _fragments(
            "prop", "oic acid", (), "pentafluoro", (2, 2, 3, 3, 3))
        assert _l5_prefix_locants_omitted(stub, frags) is True
        visited = _get_visited()
        visited.add("O=C(O)C(F)(F)C(F)(F)F")
        try:
            assert _l5_prefix_locants_omitted(stub, frags) is False
        finally:
            visited.discard("O=C(O)C(F)(F)C(F)(F)F")


# --------------------------------------------------------------------------- #
# 7. Scope preconditions re-established inside the predicate #
# --------------------------------------------------------------------------- #
class TestScopePreconditions:
    def test_a_ring_parent_is_denied_and_the_in_handler_control_holds(self, namer):
        """``cyclohexanecarboxylic acid`` reaches the SAME handler (measured) and is
        correct today, so it is the in-handler control. It arrives with an EMPTY
        ``principal_chain``, so the acyclic precondition denies it by construction."""
        assert namer.name("OC(=O)C1CCCCC1") == "cyclohexanecarboxylic acid"
        mol = Chem.MolFromSmiles("OC(=O)C1CCCCC1")
        stub = _FeaturesStub(mol=mol, principal_chain=(),
                             principal_group_atoms=((1, 2, 0),))
        assert _l5_prefix_locants_omitted(
            stub, _fragments("cyclohex", "carboxylic acid", (), "hexafluoro",
                             (1, 1, 2, 2, 3, 3))) is False

    def test_an_oriented_ring_denies_even_with_a_chain(self):
        mol = Chem.MolFromSmiles(_ACID)
        stub = _FeaturesStub(mol=mol, principal_chain=(1, 3, 6),
                             principal_group_atoms=((1, 2, 0),),
                             oriented_ring=[1, 3, 6])
        assert _l5_prefix_locants_omitted(
            stub, _fragments("prop", "oic acid", (), "pentafluoro",
                             (2, 2, 3, 3, 3))) is False

    def test_a_wrong_locant_to_atom_map_DENIES_rather_than_mismeasures(self):
        """★ The mapping is a MEASUREMENT: for each chain atom the predicate counts the
        bonds leaving the parent compound and requires that to equal the count the name
        cites at that locant. Feed it a REVERSED chain and it must refuse."""
        mol = Chem.MolFromSmiles(_ACID)
        reversed_stub = _FeaturesStub(mol=mol, principal_chain=(6, 3, 1),
                                      principal_group_atoms=((1, 2, 0),))
        assert _l5_prefix_locants_omitted(
            reversed_stub, _fragments("prop", "oic acid", (), "pentafluoro",
                                      (2, 2, 3, 3, 3))) is False

    def test_a_locant_out_of_chain_range_denies(self):
        assert _l5_prefix_locants_omitted(
            _acid_stub(), _fragments("prop", "oic acid", (), "pentafluoro",
                                     (2, 2, 3, 3, 9))) is False

    def test_a_letter_locant_denies(self):
        assert _l5_prefix_locants_omitted(
            _acid_stub(), _fragments("prop", "oic acid", (), "pentafluoro",
                                     (2, 2, 3, 3, "N"))) is False

    def test_a_count_bearing_prefix_denies_so_the_multiplier_cannot_be_lost(self):
        """``_assemble_fragments`` ignores ``count`` entirely, so a prefix relying on
        it for its multiplier would emit ``fluoropropanoic acid`` for five fluorines
        once the locants went. Invariant 11."""
        assert _l5_prefix_locants_omitted(
            _acid_stub(), _fragments("fluoro", None, (), "fluoro",
                                     (2, 2, 3, 3, 3), count=5)) is False

    def test_no_prefix_at_all_denies(self):
        assert _l5_prefix_locants_omitted(
            _acid_stub(),
            [NameFragment(text="prop", locants=((), ()), fragment_type="parent"),
             NameFragment(text="oic acid", fragment_type="suffix")]) is False

    def test_empty_inputs_deny(self):
        assert _l5_prefix_locants_omitted(None, []) is False
        assert _l5_prefix_locants_omitted(_acid_stub(), []) is False

    def test_a_heteroatom_chain_is_out_of_scope(self, namer):
        """The all-carbon precondition is a deliberate SCOPE NARROWING whose mutation
        SURVIVES because it is unreachable: no heteroatom-chain parent reaches this
        handler at all. Recorded as a measurement so that the day one does, the boundary
        is revisited rather than silently crossed -- ``hexafluorodisilane`` WOULD be
        licensed by, but the hydrogen count would have to be re-derived for a
        chain whose atoms are not all tetravalent.

        ⚠ The end-to-end half asserts ``is_refusal_sentinel``, NOT a sentinel STRING:
        the exact wording differs between the OPSIN-enabled and OPSIN-skipped runs
        (``unknown organic compound`` vs ``unknown``), and pinning either makes the test
        pass or fail for a reason unrelated to this rule.
        """
        from orthonym.errors import is_refusal_sentinel
        for smiles in ("F[Si](F)(F)[Si](F)(F)F", "FN(F)N(F)F", "F[P](F)[P](F)F"):
            assert is_refusal_sentinel(namer.name(smiles)), smiles
        #...and the predicate itself refuses a silicon chain, so the narrowing is
        # asserted directly rather than resting on the molecule being unnameable.
        mol = Chem.MolFromSmiles("F[Si](F)(F)[Si](F)(F)F")
        assert [a.GetSymbol() for a in mol.GetAtoms()][1] == "Si"
        stub = _FeaturesStub(mol=mol, principal_chain=(1, 5))
        assert _l5_prefix_locants_omitted(
            stub, [NameFragment(text="disil", locants=((), ()),
                                fragment_type="parent"),
                   NameFragment(text="hexafluoro", locants=(1, 1, 1, 2, 2, 2),
                                fragment_type="prefix")]) is False

    def test_a_mixed_kind_position_denies_on_EVERY_hash_seed(self):
        """★ The seed-dependence the M5 mutation exposed, pinned as a property.

        C2 of ``2-chloro-2,3,3,3-tetrafluoropropanoic acid`` carries TWO kinds and its
        cited count (2) equals its substitutable-H count (2) -- so an arbitrary
        ``next(iter(set))`` pick of ``tetrafluoro`` would have presented the rule with
        one uniform kind and elided a required locant, on some hash seeds only. The
        composite ``"|".join(sorted(...))`` key makes the refusal seed-independent.

        Asserted through the predicate, in both prefix orders, so neither the explicit
        two-kinds check nor the composite key can be removed without a failure.
        """
        stub = _acid_stub()
        chloro = NameFragment(text="chloro", locants=(2,), fragment_type="prefix")
        tetra = NameFragment(text="tetrafluoro", locants=(2, 3, 3, 3),
                             fragment_type="prefix")
        parent = NameFragment(text="prop", locants=((), ()), fragment_type="parent")
        suffix = NameFragment(text="oic acid", locants=(), fragment_type="suffix")
        for order in ([chloro, tetra], [tetra, chloro]):
            assert _l5_prefix_locants_omitted(
                stub, [parent, suffix, *order]) is False, order


# --------------------------------------------------------------------------- #
# 8. Invariant 11 -- what is EMITTED, not merely that a locant vanished #
# --------------------------------------------------------------------------- #
class TestNothingWorseIsUnmasked:
    def test_the_multiplier_morpheme_survives_the_elision(self, namer):
        """The count of substituents is the one thing the locants also encoded. Each
        name below must still state it."""
        for smiles, mult in [
            ("OC(=O)C(F)(F)C(F)(F)C(F)(F)F", "hepta"),
            ("OC(=O)C(F)(F)C(F)(F)F", "penta"),
            ("FC(F)(F)C(F)(F)C(F)(F)C(F)(F)F", "deca"),
            ("OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F", "nona"),
        ]:
            name = namer.name(smiles)
            assert name.startswith(mult), (smiles, name)
            assert not any(ch.isdigit() for ch in name), (smiles, name)

    def test_no_atom_is_dropped_and_no_morpheme_fabricated(self, namer):
        """The elided names must still account for every fluorine, and must not gain a
        stray hyphen, parenthesis or ``per-`` contraction (``:3009`` retires it)."""
        for smiles, expected in [
            ("OC(=O)C(F)(F)C(F)(F)C(F)(F)F", "heptafluorobutanoic acid"),
            ("FC(F)(F)C(F)(F)F", "hexafluoroethane"),
            ("FC(F)(F)C(F)(F)C=O", "pentafluoropropanal"),
        ]:
            name = namer.name(smiles)
            assert name == expected
            assert "per" not in name.split("fluoro")[0]
            assert "(" not in name and ")" not in name and "-" not in name

    def test_the_two_renderers_agree_on_the_licensed_rows(self, namer):
        """★ The licence is applied by rebuilding the prefix FRAGMENTS, not by a
        print-time flag, so the legacy assembler and the name-tree serializer (which
        is the production composition site for this handler -- ``general_acyclic`` is
        the sole member of ``SERIALIZER_PRODUCTION_CLASSES``) cannot disagree. A flag
        threaded into only one of them makes
        ``composer._serializer_flip_or_name`` silently fall back for exactly the rows
        the licence touched."""
        import orthonym.assembly.composer as _CO
        from orthonym.assembly.name_tree_to_string import name_tree_to_string
        from orthonym.namer import Orthonym as _OS

        seen = []
        orig = _CO._serializer_flip_or_name

        def probe(result, style):
            tree = getattr(result, "tree", None)
            ser = None
            if tree is not None and getattr(tree, "class_id", "") == "general_acyclic":
                ser = name_tree_to_string(tree, style=style)
            seen.append((result.name, ser))
            return orig(result, style)

        _CO._serializer_flip_or_name = probe
        try:
            local = _OS()
            for smiles, expected in [
                ("OC(=O)C(F)(F)C(F)(F)C(F)(F)F", "heptafluorobutanoic acid"),
                ("OC(=O)C(F)(F)C(F)(F)F", "pentafluoropropanoic acid"),
                ("FC(F)(F)C(F)(F)F", "hexafluoroethane"),
            ]:
                seen.clear()
                assert local.name(smiles) == expected
                assert seen, f"{smiles}: the serializer seam was never reached"
                assert any(legacy == ser == expected for legacy, ser in seen), \
                    (smiles, seen)
        finally:
            _CO._serializer_flip_or_name = orig


# --------------------------------------------------------------------------- #
# 9. Tripwires -- every name Phase C has already shipped #
# --------------------------------------------------------------------------- #
class TestPhaseCTripwires:
    @pytest.mark.parametrize("smiles,expected", [
        # Task 5a, substituent scope (must stay fixed, WITH its enclosing marks)
        ("Clc1ccccc1C(F)(F)C(F)(F)F", "1-chloro-2-(pentafluoroethyl)benzene"),
        ("FC(F)(F)C(F)(F)C1CCCCC1", "(pentafluoroethyl)cyclohexane"),
        ("FC(F)(F)C(F)(F)C(F)(F)c1ccccc1", "(heptafluoropropyl)benzene"),
        # L3 -- the sibling licence, wired at the SAME site
        ("OC(=O)c1cnccn1", "pyrazinecarboxylic acid"),
        ("OC(=O)C(Cl)C(=O)O", "chloropropanedioic acid"),
        ("OC(=O)CC(Cl)C(=O)O", "chlorobutanedioic acid"),
        # L1 -- terminal suffix locants
        ("OC(=O)CCC(=O)O", "butanedioic acid"),
        # L6 -- a DIFFERENT licence; ``trifluoroacetic acid`` must not be re-routed
        ("OC(=O)C(F)(F)F", "trifluoroacetic acid"),
        ("ClCC(=O)O", "chloroacetic acid"),
        # ring / benzene scopes -- untouched by this task
        ("Oc1c(Cl)cccc1C", "2-chloro-6-methylphenol"),
        ("Oc1c(O)c(O)c(O)c(O)c1Cl", "6-chlorobenzene-1,2,3,4,5-pentol"),
        ("Cc1c(C)c(C)c(C)c(C)c1Cl", "1-chloro-2,3,4,5,6-pentamethylbenzene"),
        ("Oc1c(O)c(O)c(O)c(O)c1O", "benzenehexol"),
        ("Cc1c(C)c(C)c(C)c(C)c1C", "hexamethylbenzene"),
        ("Fc1c(F)c(F)c(F)c(F)c1F", "hexafluorobenzene"),
        ("Clc1c(Cl)c(Cl)c(Cl)c(Cl)c1Cl", "hexachlorobenzene"),
        ("OC1CCCCC1", "cyclohexanol"),
        ("OC(=O)C1CCCCC1", "cyclohexanecarboxylic acid"),
        ("[O-]C(=O)CN", "glycinate"),
        ("CCO", "ethanol"),
        # Blue Book worked examples that were ALREADY correct
        ("C1CCC2CCCCC2C1", "decahydronaphthalene"),
        ("FN(F)C(=O)N(F)F", "tetrafluorourea"),
    ])
    def test_unchanged(self, namer, smiles, expected):
        assert namer.name(smiles) == expected
