""" with: the isotope names of radicals, and of molecules OPSIN reads as a tautomer.

``rules.isotopes._isotope_round_trips`` accepts a name only when OPSIN reads it as the input with its
isotopes (``rules.isotope_compare``).

Cited rules (the Blue Book, with their section headings):

* GENERAL METHODOLOGY (:40400): "Radicals are named by modifying a parent hydride name to signal the
  subtraction or addition of one or more hydrogen atoms"; "The suffixes 'yl' (-H.), 'ylidene' (-2H.), 'ylidyne'
  (-3H.) denote the removal of hydrogen atoms". A name with a free valence does not parse without OPSIN's
  ``-r``, which the read-back gives only for an input that has radical electrons
  (``validation/read_back_policy.py``).
* (heading 'Guanidine and its derivatives',:34248): "Hydrocarbyl derivatives are named as
  substituted guanidines. When the position of the double bond is unknown, the preferred IUPAC name uses a
  minimum number of primes." ('Substituted amidines',:34375): "Because of tautomerism, the
  locants N, N', etc. are used when only one substituent is present on each imidamide group." No rule makes
  the name of another kind of tautomer (an amide and its imidic acid, an azole NH) neutral, so a name read as
  one of those is not accepted as the name of the input.
*: the descriptor stands "before the part of the compound that is isotopically substituted"
  ('(2H3)methoxy' of a methoxysulfinyl radical).
* /: a descriptor cites the position it needs; a nuclide on a carbon or a heavy atom is
  fixed by the key whichever tautomer the name is read as.
"""
import pytest
from rdkit import Chem

from orthonym.rules import isotope_compare as C
from orthonym.rules import isotopes as I
from tests.support.rt_assert import name_best_effort, name_is_rt_exact

pytestmark = pytest.mark.unit

METHOXYSULFINYL = "[2H]C([2H])([2H])O[S]=O"


# -- the reading of a radical (OPSIN) ----------------------------------------------------------------------


def test_a_name_with_a_free_valence_is_read_with_the_radical_flag_only_for_a_radical():
    mol = Chem.MolFromSmiles(METHOXYSULFINYL)
    name = "[(2H3)methoxy]sulfinyl"
    assert not I._isotope_round_trips(name, mol)              # as a direct use: no -r, no reading
    token = C.RADICAL_READING.set(True)
    try:
        assert I._isotope_round_trips(name, mol)
        # the reading has to carry the radical centre: the closed-shell input is not the radical
        assert not I._isotope_round_trips(name, Chem.MolFromSmiles("[2H]C([2H])([2H])OS=O"))
    finally:
        C.RADICAL_READING.reset(token)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    (METHOXYSULFINYL, "[(2H3)methoxysulfinyl]"),
    ("[2H]NC1=C(C=C(C(=C1)NC2=NC=NC(=C2)C3=CC=CC=C3)[B])C",
     "{4-(2H1)amino-5-methyl-2-[(6-phenylpyrimidin-4-yl)amino]phenyl}boranylidene"),
    ("[2H]C1=C[CH-]OC1[S]", "[(5-2H)-2-oxacyclopent-4-en-3-id-1-yl]sulfanyl"),
])
def test_the_isotope_name_of_a_radical_reads_back_as_the_radical(smiles, expected):
    res = name_best_effort(smiles)
    assert res["name"] == expected
    assert name_is_rt_exact(res["name"], smiles)


# -- a bare metal atom of a mixture: radical electrons that are no chemical radical ---------------------------


@pytest.mark.parametrize("smiles", [
    "CCO.[99Tc]", "CCO.[64Cu]", "CCO.[57Co]", "CCO.[23Na]", "[99Tc]", "C([C@H]([C@H]([C@@H]([C@@H](CO)O)O)O)O)O.[99Tc]"])
def test_a_bare_metal_atom_has_radical_electrons_and_no_chemical_radical(smiles):
    """RDKit gives a bare metal atom radical electrons; ``validation.radical_identity`` does not count them
    : they are no radical of a molecule), so its verdict for such an input is "n/a", which is no
    contradiction: the full key and the isotope layer decide, as they did before the radical reading."""
    from orthonym.validation.radical_identity import radical_identity_verdict
    mol = Chem.MolFromSmiles(smiles)
    assert any(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
    assert radical_identity_verdict(Chem.MolToSmiles(mol), Chem.MolToSmiles(mol)) == "n/a"
    assert C.radical_centres_agree(mol, Chem.MolToSmiles(mol))


def test_a_reading_that_contradicts_the_radical_centres_is_refused():
    radical = Chem.MolFromSmiles("[2H]C([2H])([2H])O[S]=O")
    assert C.radical_centres_agree(radical, "[2H]C([2H])([2H])O[S]=O")
    assert not C.radical_centres_agree(radical, "[2H]C([2H])([2H])OS=O")                  # a closed-shell reading
    assert not C.radical_centres_agree(Chem.MolFromSmiles("[2H]C([2H])([2H])OS=O"), "[2H]C([2H])([2H])O[S]=O")


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    ("CCO.[99Tc]", "ethan-1-ol—(99Tc)technetium (1/1)"),
    ("C([C@H]([C@H]([C@@H]([C@@H](CO)O)O)O)O)O.[99Tc]",
     "(2R,3R,4R,5R)-hexane-1,2,3,4,5,6-hexol—(99Tc)technetium (1/1)"),
])
def test_a_labelled_metal_atom_of_a_mixture_is_named(smiles, expected):
    """The names main ships for a labelled bare metal atom beside a molecule (best-effort tier)."""
    res = name_best_effort(smiles)
    assert res["name"] == expected
    assert name_is_rt_exact(res["name"], smiles)


# -- a tautomer of the input (OPSIN) -----------------------------------------------------------------------

#: an N-methylguanidine: the name does not say which nitrogen carries the double bond
GUANIDINE = "[11CH3]N(c1ccccc1)C(N)=Nc1ccccc1"
GUANIDINE_NAME = "N'-phenyl-N-[(11C)methyl]-N-phenylguanidine"


def test_a_tautomer_is_the_input_only_in_the_pass_that_accepts_it():
    mol = Chem.MolFromSmiles(GUANIDINE)
    assert not I._isotope_round_trips(GUANIDINE_NAME, mol)    # outside a decorator call
    with C.tautomer_passes(mol) as state:
        assert state["possible"]                              # the guanidine has a hydrogen that moves
        assert not I._isotope_round_trips(GUANIDINE_NAME, mol)
        assert (state["accept"], state["refused"]) == (False, True)   # the first pass says it refused one
        state["accept"] = True
        assert I._isotope_round_trips(GUANIDINE_NAME, mol)


def test_an_input_without_a_moving_hydrogen_has_no_tautomer_to_judge():
    toluene = Chem.MolFromSmiles("[2H]C([2H])([2H])c1ccccc1")
    with C.tautomer_passes(toluene) as state:
        assert not state["possible"]
        assert not C.tautomer_accepted(Chem.MolFromSmiles("[2H]C([2H])([2H])c1ccccc1"), toluene)
        assert not state["refused"]


@pytest.mark.parametrize("a,b,tautomers", [
    ("[11CH3]N(c1ccccc1)C(N)=Nc1ccccc1", "[11CH3]N(c1ccccc1)C(=N)Nc1ccccc1", True),
    ("[11CH3]N(c1ccccc1)C(N)=Nc1ccccc1", "[11CH3]N(c1ccccc1)C(N)=Nc1ccccc1", False),   # one drawing
    ("[2H]N=C(NC)c1ccccc1", "N=C(N(C)[2H])c1ccccc1", False),   # the deuterium on another nitrogen
    ("[2H]N=C(N)c1ccccc1", "N=C(N[2H])c1ccccc1", True),     # the two nitrogens of an amidine are one place
    ("C1=CC=CC=C1", "C1=CC=CC=C1", False),
])
def test_tautomer_of(a, b, tautomers):
    assert C.tautomer_of(Chem.MolFromSmiles(a), Chem.MolFromSmiles(b)) is tautomers


@pytest.mark.opsin_gate
def test_the_guanidine_of_a_11c_label_is_named():
    smiles = "CSC1=CC(=C(C=C1)Cl)N=C(N)N([11CH3])C2=CC(=CC=C2)SC"
    res = name_best_effort(smiles)
    assert res["name"] == ("N'-[2-chloro-5-(methylsulfanyl)phenyl]-N-(11C)methyl-"
                           "N-[3-(methylsulfanyl)phenyl]guanidine")
    assert name_is_rt_exact(res["name"], smiles)


@pytest.mark.parametrize("a,b,one_group", [
    ("[11CH3]N(c1ccccc1)C(N)=Nc1ccccc1", "[11CH3]N(c1ccccc1)C(=N)Nc1ccccc1", True),     # guanidine
    ("CN=C(N)c1ccccc1", "CNC(=N)c1ccccc1", True),                                      # amidine
    ("[2H]CC(=O)NC", "[2H]CC(O)=NC", False),                                           # amide, imidic acid
    ("Cc1cc[nH]n1", "Cc1ccn[nH]1", False),                                             # a pyrazole NH
    ("Cc1c[nH]cn1", "Cc1cnc[nH]1", False),                                             # an imidazole NH (aromatic)
])
def test_only_the_tautomers_of_a_guanidine_or_an_amidine_are_one_group(a, b, one_group):
    pa, pb = Chem.MolFromSmiles(a), Chem.MolFromSmiles(b)
    assert C.tautomer_of(pa, pb)                               # all of them are tautomers
    assert C.in_one_amidine_group(pa, pb) is one_group


def test_a_tautomer_outside_those_classes_is_not_accepted_and_does_not_ask_for_another_pass():
    amide = Chem.MolFromSmiles("[2H]CC(=O)NC")
    with C.tautomer_passes(amide) as state:
        state["accept"] = True
        assert not C.tautomer_accepted(Chem.MolFromSmiles("[2H]CC(O)=NC"), amide)
        assert not state["refused"]
