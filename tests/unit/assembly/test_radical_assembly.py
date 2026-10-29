"""Assemblies of identical parent radicals: radical centres in separate identical parts of one molecule.

The radical floor names several monovalent centres as the free valences of ONE parent, a ring system or an acyclic
chain that holds them all "CHOICE OF PARENT RADICAL", the Blue Book). A molecule whose centres sit in
different parts of the structure has no such parent. ``## **** ASSEMBLIES OF PARENT RADICALS``
(the Blue Book): "Polyradicals with radical centers identically derived from the same parent hydride or the
same characteristic group (except for polyacyl or polyamide radicals described in and, respectively)
but located in different parts of the structure are named, if possible, according to the principles for nomenclature of
assemblies of identical units linked by multivalent substituents (see." The book's examples are
'(naphthalene-2,6-diyl)bis(disulfanyl) (PIN)', '(2,4-dimethylpentane-2,4-diyl)bis(oxyl) (PIN)' and
'(cyclobutane-1,3-diyl)bis(peroxyl) (PIN)'; the producers of the PIN paths name the ones they can, and
``assembly/radical_assembly.py`` is an offer of the radical last resort (``assembly/radical_offer.py``) for the rest:
``(<linker>-diyl)bis(<unit>)``, systematic_verified, made only after every name of the floor's routes
(``radical_floor.radical_floor_names``) and shipped only when OPSIN ``-r`` reads it back to the input's full InChIKey and
radical graph.
"""
import pytest
from rdkit import Chem

from orthonym.assembly.radical_assembly import split_identical_units, unit_splits
from orthonym.validation.radical_identity import radical_identity_verdict
from tests.support.rt_assert import _independent_parse, name_best_effort, name_is_rt_exact

pytestmark = pytest.mark.opsin_gate


def _assert_named(smiles, expected):
    res = name_best_effort(smiles)
    assert res["name"] == expected, res
    assert res["tier"] == "systematic_verified" and not res["is_pin"], res
    assert name_is_rt_exact(res["name"], smiles), res
    parsed = _independent_parse(res["name"], smiles)
    assert parsed is not None and radical_identity_verdict(smiles, parsed) == "ok", (res, parsed)


# --- the forms of the unit -----------------------------------------------------------------------------------------

ROWS = [
    # the radical of the part, the bond to the linker closed by a hydrogen on the atom of the centre; the linker is spelled by
    # the book's name of its ring 'benzene', 'imidazole' with its indicated hydrogen, 'thiophene')
    ("C1C(=NC(=N1)[O])[O]", "(4H-imidazole-2,5-diyl)bis(oxyl)"),
    ("C[Si](C)c1ccccc1[Si](C)C", "(benzene-1,2-diyl)bis(dimethylsilyl)"),
    ("CC1=CC(=C(C=C1C)[NH])[NH]", "(4,5-dimethylbenzene-1,2-diyl)bis(azanyl)"),
    ("[B](C)C1CCCC(CCC1)[B]C", "(cyclooctane-1,5-diyl)bis(methylboranyl)"),
    # three units on one linker
    ("CC[Si](CC)C1=C(SC(=C1Br)[Si](CC)CC)[Si](CC)CC", "(4-bromothiophene-2,3,5-triyl)tris(diethylsilyl)"),
    # a bare centre bonded to a ring that carries the bond to the linker: '(group-diyl)radical', the first locant of the
    # group the atom bonded to the linker; the linker cites the one-atom divalent group after its host
    ("CC1(CC(CC(N1[O])(C)C)OC(=O)CCCC(=O)OC2CC(N(C(C2)(C)C)[O])(C)C)C",
     "[1,5-dioxopentane-1,5-diylbis(oxy)]bis[(2,2,6,6-tetramethylpiperidine-4,1-diyl)oxyl]"),
    ("[CH2]C1=CC=C(C=C1)CCCC2=CC=C(C=C2)[CH2]", "(propane-1,3-diyl)bis[(benzene-4,1-diyl)methyl]"),
    # units bonded to the linker through one-atom divalent groups, which the linker cites after its host, without enclosing
    # marks of their own ('benzene-1,4-diylbis(oxy)',
    ("C[Si](C)OC1=CC=C(C=C1)O[Si](C)C", "[benzene-1,4-diylbis(oxy)]bis(dimethylsilyl)"),
    ("C[Si](C)CC1CCC(CC1)C[Si](C)C", "[cyclohexane-1,4-diylbis(methylene)]bis(dimethylsilyl)"),
    # the mechanical spelling of a group of the unit (the retry of the floor) leaves the linker in the book's name of its ring
    ("CC1=CC(=CC(=C1)OC[Si](C)C)OC[Si](C)C", "[5-methylbenzene-1,3-diylbis(oxy)]bis(1,1-dimethyl-1-silaethan-1-yl)"),
    # the spacer is tried before the linker as one chain with its oxygens: '[ethane-1,2-diylbis(oxy)]', never '1,4-dioxabutane-1,4-diyl'
    ("[O]N1C(C)(C)CC(CC1(C)C)OCCOC1CC(C)(C)N([O])C(C)(C)C1",
     "[ethane-1,2-diylbis(oxy)]bis[(2,2,6,6-tetramethylpiperidine-4,1-diyl)oxyl]"),
]


@pytest.mark.parametrize("smiles,expected", ROWS)
def test_identical_radical_units_are_named_as_an_assembly(smiles, expected):
    _assert_named(smiles, expected)


# --- what the split takes and what it leaves ------------------------------------------------------------------------

def test_the_units_are_the_same_rooted_structure_and_the_linker_holds_no_centre():
    mol = Chem.MolFromSmiles("C1C(=NC(=N1)[O])[O]")
    split = split_identical_units(mol)
    assert split is not None and len(split.units) == 2
    assert {len(u.atoms) for u in split.units} == {1}
    assert all(mol.GetAtomWithIdx(a).GetNumRadicalElectrons() == 0 for a in split.linker)
    assert {u.anchor for u in split.units} <= split.linker and len({u.anchor for u in split.units}) == 2


@pytest.mark.parametrize("smiles", [
    # two units that differ (methyl, ethyl): the book names identical units only
    "[B](C)C1CCC1[B]CC",
    # the centres are in one ring system: one parent carries them, no assembly
    "[CH]1[CH][CH][CH][C]1C",
    # a divalent centre is cited by the lambda convention, not by an assembly
    "[B]c1ccccc1[B]",
    # a charged atom is not a unit of this kind
    "[B-]CC",
    # two units on one atom of the linker: no divalent group of the floor carries both
    "C([Si](C)C)[Si](C)C",
])
def test_a_molecule_that_is_not_an_assembly_of_identical_units_has_no_split(smiles):
    assert split_identical_units(Chem.MolFromSmiles(smiles)) is None


def test_a_unit_may_take_in_the_atoms_between_its_centre_and_the_linker():
    """The unit is the smallest part first and then the parts that hold it: the same molecule is cut at each bridge on the
    way to the linker, so a linker that no parent holds at the first cut is tried at the next."""
    mol = Chem.MolFromSmiles("CC1(CC(CC(N1[O])(C)C)OC(=O)CCCC(=O)OC2CC(N(C(C2)(C)C)[O])(C)C)C")
    sizes = [len(split.units[0].atoms) for split in unit_splits(mol)]
    assert sizes == sorted(sizes) and sizes[0] == 1 and len(sizes) > 1


# --- the order of the offers ---------------------------------------------------------------------------------------

def test_the_assembly_names_come_after_every_name_of_the_routes(monkeypatch):
    """A name that ships today never changes: the routes of the floor give every name they have before the assembly of
    parent radicals is tried, so a molecule that one of those names reads back for keeps it."""
    from orthonym.assembly import radical_floor
    from orthonym.assembly.radical_floor import _further_names, radical_floor_names
    from orthonym.validation.read_back_policy import radical_last_resort, reading_for
    mol = Chem.MolFromSmiles("C1C(=NC(=N1)[O])[O]")
    monkeypatch.setattr(radical_floor, "_route_names", lambda m, leaf: iter(["a name of the routes"]))
    with radical_last_resort(), reading_for(mol):
        every = list(radical_floor_names(mol))
        further = list(dict.fromkeys(_further_names(mol)))
    assert further and every == ["a name of the routes"] + further
    pair = Chem.MolFromSmiles("[B]c1ccccc1[B]")
    with radical_last_resort(), reading_for(pair):
        assert list(_further_names(pair)) == []     # two lambda centres: the routes name it, no assembly is offered


def test_the_assembly_is_not_offered_outside_the_radical_last_resort():
    from orthonym.assembly.radical_assembly import radical_assembly_names
    from orthonym.assembly.universal_substituent import name_universal_polyradical
    mol = Chem.MolFromSmiles("C[Si](C)c1ccccc1[Si](C)C")
    assert name_universal_polyradical(mol) is None
    assert list(radical_assembly_names(mol)) == []


# --- the unit named without an edit of a name -----------------------------------------------------------------------

@pytest.mark.parametrize("smiles,expected", [
    ("[O]C(C)(C)CC(C)(C)[O]", ["oxyl", "oxidanyl"]),
    ("[NH]c1ccccc1[NH]", ["azanyl"]),
    ("C[N]c1ccc([N]C)cc1", ["methylazanyl"]),
    ("C[Si](C)c1ccccc1[Si](C)C", ["dimethylsilyl"]),
    ("C[B]c1ccc([B]C)cc1", ["methylboranyl"]),
    ("[S]c1ccc([S])cc1", ["sulfanyl"]),
    ("[PH]c1ccccc1[PH]", ["phosphanyl"]),
    ("[O]OC1CC(C1)O[O]", ["oxyl"]),                       # the unit 'O' alone; 'peroxyl' is the unit of the larger part
])
def test_the_radical_of_a_one_atom_parent_hydride_is_built_from_its_element(smiles, expected):
    """ "Monovalent radicals" (the Blue Book): the radical of the parent hydride of one atom is its name with the
    final 'e' elided ('azanyl', 'boranyl', 'silyl'), every group on the atom a prefix. The names come from the element
    (``rules.radicals._parent_hydride_radical_name``) and the floor's prefixes, never from the floor's prefix form of the group
    ('hydroxy', 'amino') edited into the radical's."""
    from orthonym.assembly.radical_assembly import _parent_radical_names
    from orthonym.validation.read_back_policy import radical_last_resort
    mol = Chem.MolFromSmiles(smiles)
    unit = split_identical_units(mol).units[0]
    (centre,) = unit.centres
    with radical_last_resort():
        names = _parent_radical_names(mol, unit.atoms, unit.root, centre, True)
    assert [n for n in names if n in expected] == expected, names


def test_the_radical_of_a_two_atom_parent_hydride_is_the_peroxyl_of_the_book():
    """'s '(cyclobutane-1,3-diyl)bis(peroxyl)' and '(naphthalene-2,6-diyl)bis(disulfanyl)': the unit O-O. (HSS.)"""
    from orthonym.assembly.radical_assembly import _parent_radical_names
    from orthonym.validation.read_back_policy import radical_last_resort
    mol = Chem.MolFromSmiles("[O]OC1CC(C1)O[O]")
    unit = next(u for split in unit_splits(mol) for u in split.units[:1] if len(u.atoms) == 2)
    (centre,) = unit.centres
    with radical_last_resort():
        assert _parent_radical_names(mol, unit.atoms, unit.root, centre, True)[:2] == ["peroxyl", "dioxidanyl"]


@pytest.mark.parametrize("unit,count,expected", [
    ("oxyl", 2, ["bis(oxyl)"]),                       # 'dioxyl' would read as the O-O of a peroxide
    ("azanyl", 2, ["bis(azanyl)"]),
    ("silyl", 3, ["tris(silyl)"]),                    # 'trisilyl' would read as Si-Si-Si
    ("phosphanyl", 2, ["bis(phosphanyl)"]),
    ("disulfanyl", 2, ["bis(disulfanyl)"]),           # HSS, the book's example
    ("methyl", 2, ["dimethyl", "bis(methyl)"]),       # '(cyclopropane-1,2-diyl)dimethyl (PIN)',
    ("dimethylsilyl", 2, ["bis(dimethylsilyl)"]),
])
def test_a_parent_hydride_radical_takes_bis_and_never_di(unit, count, expected):
    from orthonym.assembly.radical_assembly import _multiplied_units
    assert _multiplied_units(unit, count) == expected


def test_the_book_spelling_of_a_ring_with_free_valences_is_the_assembly_s_alone():
    """The floor's own names of a polyradical on a ring ship today and keep the mechanical spelling; only the assembly asks for the
    book's ('benzene-1,2-diyl', the Blue Book) through ``universal_substituent._BOOK_RING_FREE_VALENCES``."""
    from orthonym.assembly.universal_substituent import _BOOK_RING_FREE_VALENCES, name_universal_polyradical
    from orthonym.validation.read_back_policy import radical_last_resort
    mol = Chem.MolFromSmiles("[c]1[c]cccc1")
    with radical_last_resort():
        assert name_universal_polyradical(mol).name == "cyclohexa-1,3,5-triene-1,2-diyl"
        token = _BOOK_RING_FREE_VALENCES.set(True)
        try:
            assert name_universal_polyradical(mol).name == "benzene-1,2-diyl"
        finally:
            _BOOK_RING_FREE_VALENCES.reset(token)


def test_a_class_that_fails_to_build_a_name_is_logged_and_gives_up_closed(monkeypatch, caplog):
    """The further classes fail closed (the names before them stand) and say why through the module logger."""
    import logging

    from orthonym.assembly import radical_assembly, radical_floor
    from orthonym.validation.read_back_policy import radical_last_resort

    def boom(mol, **_kwargs):
        raise RuntimeError("no name")

    monkeypatch.setattr(radical_assembly, "radical_assembly_names", boom)
    mol = Chem.MolFromSmiles("C1C(=NC(=N1)[O])[O]")
    with caplog.at_level(logging.DEBUG, logger="orthonym.assembly.radical_floor"), radical_last_resort():
        assert list(radical_floor._further_names(mol)) == []
    assert any("gave up" in record.getMessage() and record.exc_info for record in caplog.records)
