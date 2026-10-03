"""Two-component carbocyclic fusion names for the parents no table holds
(``bridged_fused_pin.fusion_names``): the book's own names of the class as the spelling gold,
one test per rule step, and the numbering taken from OPSIN's reading of the name.

 (the Blue Book) a parent component and prefixed attached components;
 (:11968) [n]annulene parent components "starts at n = 7"; (b)
(:12163) "a component containing the greater number of rings", (c) (:12234) "A component
containing the larger ring at the first point of difference when comparing rings in order of
decreasing size"; (:11911) the parent sides lettered "beginning with a for the side
numbered '1,2'", "the letter as early in the alphabet as possible"; (:12000) the
monocyclic prefixes; (:13970) locants omitted; Note (:11909) no elision;
 (:23710) two rings of five or more members."""
import pytest
from rdkit import Chem

from orthonym.rules.bridged_fused_pin import fusion_names, parents


def _skeleton(smiles):
    return Chem.MolFromSmiles(parents._key_of(Chem.MolFromSmiles(smiles)))


#: every two-ring or three-ring carbocyclic two-component name the book prints as a PIN or as
#: the fused ring system of a PIN (the rules study's 212-system table), on OPSIN's structure
BOOK_NAMES = [
    ("C1C=CC=C2C1=CC=CC=C2", "benzo[7]annulene"),                      #:17032
    ("C1=CC=CC2=C1C=CC=CC=C2", "benzo[8]annulene"),                    #:19564
    ("C1C=CC=C2C1=CC=CC=CC=C2", "benzo[9]annulene"),                   #:14661
    ("C1C=CC=C2C1=CC=CC=CC=CC=CC=C2", "benzo[13]annulene"),            #:23875
    ("C1=CC=CC2=C1C=CC=CC=CC=CC=CC=C2", "benzo[14]annulene"),          #:23839
    ("C1C=CC2=C1C=CC=CC=C2", "cyclopenta[8]annulene"),                 #:13974
    ("C1=CC2C1=CC=1C=CC=CC21", "cyclobuta[a]indene"),                  #:14267
    ("C1=CC=C2C1=CC=1C=CC=CC21", "cyclopenta[a]indene"),               #:19829
    ("C1C=CC=2C1=C1C=CC=CC1=CC2", "cyclopenta[a]naphthalene"),         #:24822
    ("C1=CC=C2C1=CC1=CC=CC=C1C2", "cyclopenta[b]naphthalene"),         #:24748
    ("C1=C2CC=3C(=C2C=C1)C=CC3", "cyclopenta[a]pentalene"),            #:24790
]


@pytest.mark.parametrize("smiles,name", BOOK_NAMES)
def test_the_book_names_of_the_class_are_rebuilt_by_the_rules(smiles, name):
    assert fusion_names._rule_name(_skeleton(smiles)) == name


@pytest.mark.parametrize("smiles,name", [
    ("C1CCC2CC3CCCCC3C2CC1", "benzo[a]azulene"),          # (c): azulene (7,5) > indene (6,5)
    ("C1CCC2CCC3CCCCC3C2CC1", "cyclohepta[a]naphthalene"),  # (b): naphthalene > [7]annulene
    ("C1CCC2CCC3CCCC3CC2CC1", "cyclopenta[b]heptalene"),  # (c): heptalene (7,7) > azulene (7,5)
    ("C1CCC2CC3CCCC3CC2CC1", "cyclohepta[f]indene"),      # indene; 7-6 is no component
    ("C1CCC2CCCC3CCC3C2CC1", "cyclobuta[a]heptalene"),
    ("C1CC2CC3CC3C2C1", "cyclopropa[a]pentalene"),
])
def test_three_rings_the_two_ring_component_with_the_larger_ring_is_the_parent(smiles, name):
    assert fusion_names.two_component_name(_skeleton(smiles)) == name


def test_the_letter_is_as_early_as_possible():
    # (:11911): a seven-membered ring on naphthalene side 1,2 is 'a' (angular), on
    # side 2,3 'b' (linear); side 3,4 would be 'c', the same side read from the other end
    assert fusion_names.two_component_name(_skeleton("C1CCC2CCC3CCCCC3C2CC1")) == "cyclohepta[a]naphthalene"
    assert fusion_names.two_component_name(_skeleton("C1CCC2CC3CCCCC3CC2CC1")) == "cyclohepta[b]naphthalene"


@pytest.mark.parametrize("smiles,name", [
    ("C1CCCC2CCCC2CCC1", "cyclopenta[9]annulene"),
    ("C1CCCCC2CCCCC2CCC1", "benzo[10]annulene"),
    ("C1CCCCC2CCCC2CCCC1", "cyclopenta[11]annulene"),
    ("C1CCCCC2CCCCCC2C1", "cyclohepta[8]annulene"),
])
def test_two_rings_the_larger_ring_is_the_parent_and_no_letter_is_cited(smiles, name):
    # (c) and (:13970) "made of two monocyclic hydrocarbons"
    assert fusion_names.two_component_name(_skeleton(smiles)) == name


@pytest.mark.parametrize("smiles,why", [
    ("c1ccc2cc3ccccc3cc2c1", "anthracene: a retained name"),
    ("c1ccc2c(c1)ccc1ccccc12", "phenanthrene: a retained name"),
    ("c1ccc2c(c1)Cc1ccccc1-2", "fluorene: a retained name"),
    ("C1=CC2=CC3=CC=CC3=CC2=C1", "s-indacene: a retained name"),
    ("C1c2ccccc2C=Cc2ccccc12", "5H-dibenzo[a,d][7]annulene: three components, no table name"),
    ("c1ccc2cc3cc4ccccc4cc3cc2c1", "four rings"),
    ("c1cc2ccc3cccc4ccc(c1)c2c34", "pyrene: peri-fusion"),
    ("c1ccc2ncccc2c1", "a heteroatom"),
    ("C1CCCCC2CCCCCCC2C1", "two equal rings: octalene, a polyalene"),
    ("C1=CC2=CC=CC=CC2=C1", "azulene: a retained name"),
    ("C1CC2CCC2C1", "P-52.2.4.1: one ring of five or more members"),
])
def test_outside_the_class_is_declined(smiles, why):
    assert fusion_names.two_component_name(_skeleton(smiles)) is None, why


@pytest.mark.parametrize("smiles,why", [
    ("C1c2ccccc2C=Cc2ccccc12", "6-7-6 linear: dibenzo[a,d][7]annulene"),
    ("C1C=Cc2ccccc2-c2ccccc12", "6-7-6 angular: dibenzo[a,c][7]annulene"),
    ("C1=Cc2ccccc2C=Cc2ccccc21", "6-8-6: dibenzo[a,e][8]annulene"),
    ("C1=CC=C2C=C3C=CC=CC=C3C=C2C=C1", "7-6-7: benzo[1,2:4,5]di[7]annulene"),
])
def test_three_components_are_declined_by_the_rules(smiles, why):
    # a middle ring that forms no two-ring hydrocarbon component, the
    # polyalenes) with either outer ring: the name has three components, outside
    # the class; the rule layer declines it whether or not a table holds the skeleton
    assert fusion_names._rule_name(_skeleton(smiles)) is None, why


def test_the_prefixes_follow_p25_3_2_2_1():
    # 'benzo' for six; otherwise the saturated monocycle less 'ne', no upper ring-size limit
    assert fusion_names._prefix(6) == "benzo"
    assert [fusion_names._prefix(n) for n in (3, 4, 5, 7, 8, 9, 11)] == [
        "cyclopropa", "cyclobuta", "cyclopenta", "cyclohepta", "cycloocta", "cyclonona", "cycloundeca"]


@pytest.mark.opsin_gate
def test_the_numbering_is_opsins_reading_of_the_name():
    # 'cyclohepta[a]naphthalene' has an odd number of atoms: OPSIN reads it with an indicated
    # hydrogen; the parent source maps OPSIN's $_AV locants onto the skeleton
    key = _skeleton("C1CCC2CCC3CCCCC3C2CC1")
    struct = fusion_names.opsin_structure("cyclohepta[a]naphthalene", key.GetNumAtoms())
    assert struct is not None
    assert sorted(loc for loc in struct[1] if not loc.isdigit()) == ["11a", "11b", "4a", "6a"]
    got = parents._parent_for_key(Chem.MolToSmiles(key))
    assert got is not None and got[0] == "cyclohepta[a]naphthalene" and got[2] == "fusion_name+opsin"


@pytest.fixture
def _opsin_reads_nothing(monkeypatch):
    # OPSIN unavailable to name construction (reduced mode ORTHONYM_ALLOW_REDUCED=1 without the
    # jar, or a name OPSIN rejects): the helper ``opsin_structure`` calls returns None
    from orthonym.validation import opsin_roundtrip
    monkeypatch.setattr(opsin_roundtrip, "extended_smiles_or_unavailable",
                        lambda name, jar_version="2.9.0": None)
    fusion_names.opsin_structure.cache_clear()
    parents._parent_for_key.cache_clear()
    yield
    fusion_names.opsin_structure.cache_clear()
    parents._parent_for_key.cache_clear()


def test_without_opsins_reading_the_producer_parent_declines(_opsin_reads_nothing):
    # fail closed: the rules still give the name, but without OPSIN's structure there is no
    # numbering, so the parent source returns no parent (and the builder names nothing)
    key = _skeleton("C1CCC2CCC3CCCCC3C2CC1")
    assert fusion_names.two_component_name(key) == "cyclohepta[a]naphthalene"
    assert fusion_names.opsin_structure("cyclohepta[a]naphthalene", key.GetNumAtoms()) is None
    assert parents._parent_for_key(Chem.MolToSmiles(key)) is None


def test_a_transient_opsin_failure_declines_and_is_not_remembered(monkeypatch):
    # a call OPSIN could not answer (no in-process path and the subprocess timed out) is a
    # state of the process, not of the name: the parent source declines this time, and
    # neither ``opsin_structure``'s nor ``_parent_for_key``'s cache keeps the failure
    from orthonym.validation import opsin_roundtrip

    def unavailable(name, jar_version="2.9.0"):
        raise opsin_roundtrip.OpsinUnavailable(name)

    key = _skeleton("C1CCC2CCC3CCCCC3C2CC1")
    fusion_names.opsin_structure.cache_clear()
    parents._parent_for_key.cache_clear()
    try:
        monkeypatch.setattr(opsin_roundtrip, "extended_smiles_or_unavailable", unavailable)
        assert parents.fused_parent(key, set(range(key.GetNumAtoms()))) is None
        with pytest.raises(opsin_roundtrip.OpsinUnavailable):
            parents._parent_for_key(Chem.MolToSmiles(key))
        monkeypatch.undo()
        got = parents._parent_for_key(Chem.MolToSmiles(key))
        assert got is not None and got[0] == "cyclohepta[a]naphthalene" and got[2] == "fusion_name+opsin"
        assert parents.fused_parent(key, set(range(key.GetNumAtoms()))) is not None
    finally:
        fusion_names.opsin_structure.cache_clear()
        parents._parent_for_key.cache_clear()


def _counting(monkeypatch, answer):
    from orthonym.validation import opsin_roundtrip
    calls = []

    def ask(name, jar_version="2.9.0"):
        calls.append(name)
        return answer(name)
    monkeypatch.setattr(opsin_roundtrip, "extended_smiles_or_unavailable", ask)
    fusion_names.opsin_structure.cache_clear()
    return calls


def test_an_unreadable_name_costs_at_most_three_opsin_calls(monkeypatch):
    # the bare name, '1H-' and '2H-' (``_IH_FORMS``); before, every position up to the atom
    # count was tried, n + 1 calls, each a Java start on the subprocess path
    calls = _counting(monkeypatch, lambda name: None)
    try:
        assert fusion_names.opsin_structure("cyclodeca[q]nonsense", 21) is None
        assert calls == ["cyclodeca[q]nonsense", "1H-cyclodeca[q]nonsense", "2H-cyclodeca[q]nonsense"]
    finally:
        fusion_names.opsin_structure.cache_clear()


@pytest.mark.opsin_gate
def test_the_first_form_opsin_reads_decides(monkeypatch):
    from orthonym.validation import opsin_roundtrip
    real = opsin_roundtrip.extended_smiles_or_unavailable
    calls = _counting(monkeypatch, real)
    try:
        # OPSIN reads the bare odd-atom name itself (its own indicated hydrogen): one call
        assert fusion_names.opsin_structure("cyclohepta[a]naphthalene", 15) is not None
        assert calls == ["cyclohepta[a]naphthalene"]
        # OPSIN rejects the bare name and reads the '1H-' form: two calls
        del calls[:]
        got = fusion_names.opsin_structure("cyclopenta[a]azulene", 13)
        assert got is not None and len(got[1]) == 13
        assert calls == ["cyclopenta[a]azulene", "1H-cyclopenta[a]azulene"]
        # a read form with another atom count: an indicated hydrogen moves no skeleton atom,
        # so no later form can match; one call, no structure
        del calls[:]
        assert fusion_names.opsin_structure("benzo[a]azulene", 20) is None
        assert calls == ["benzo[a]azulene"]
    finally:
        fusion_names.opsin_structure.cache_clear()
