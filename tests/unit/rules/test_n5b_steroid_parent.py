"""The steroid parent of the fused carbocycle builder (lane N5b step 1).

'cyclopenta[a]phenanthrene' is the fused parent of the steroid skeleton:
(the Blue Book) names a fused system "by prefixing to the name of a component ring or ring
system (the parent component) designations of the other component(s)"; (b) (:12163)
"a component containing the greater number of rings" makes phenanthrene the parent component;
 (:12493) "Anthracene, phenanthrene, acridine, carbazole, xanthene and its chalcogen
analogues, purine, and cyclopenta[a]phenanthrene are exceptions; traditional numberings are
retained", which numbers the fusion carbon atoms 5, 8, 9, 10, 13 and 14 like the others."""
from rdkit import Chem

from orthonym.rules.bridged_fused_pin import hydro, parents, selection
from orthonym.rules.bridged_fused_pin.derived_parents import DERIVED_PARENTS

STEROID = "C1CCC2C(C1)CCC1C2CCC2CCCC12"
NAME = "cyclopenta[a]phenanthrene"


def _fused(smiles):
    mol = Chem.MolFromSmiles(smiles)
    atoms = {a.GetIdx() for a in mol.GetAtoms()}
    return mol, atoms, parents.fused_parent(mol, atoms)


def test_the_entry_is_a_pin_parent_with_a_written_derivation():
    entry = DERIVED_PARENTS[NAME]
    assert parents.is_pin_parent_name(NAME)
    assert NAME in parents.TRADITIONAL_NUMBERING
    assert 12493 in {line for _, line in entry.derivation}
    assert all(isinstance(line, int) for _, line in entry.derivation)


def test_fused_parent_names_and_numbers_the_steroid_skeleton():
    _mol, _atoms, got = _fused(STEROID)
    assert got is not None and got.name == NAME and got.source == "derived_table"
    for numbering in got.numberings:
        assert sorted(numbering.values()) == list(range(1, 18))


def test_the_rings_of_the_steroid_numbering():
    mol, _atoms, got = _fused(STEROID)
    at = {loc: a for a, loc in got.numberings[0].items()}
    rings = {frozenset(r) for r in mol.GetRingInfo().AtomRings()}
    for ring in ((1, 2, 3, 4, 5, 10), (5, 6, 7, 8, 9, 10), (8, 9, 11, 12, 13, 14),
                 (13, 14, 15, 16, 17)):
        assert frozenset(at[x] for x in ring) in rings, ring


def test_only_the_listed_parents_number_fusion_carbons_with_plain_numbers():
    mol, _atoms, got = _fused("C1CCC2CCCCC2C1")
    assert {"4a", "8a"} <= set(got.numberings[0].values())
    key = Chem.MolFromSmiles(parents._key_of(mol))
    plain = {i: i + 1 for i in range(key.GetNumAtoms())}
    assert parents._locant_types_ok(key, plain) is False
    assert parents._locant_types_ok(key, plain, traditional=True) is True


def test_hydro_state_marks_the_fusion_carbons_by_structure():
    mol, atoms, _got = _fused(STEROID)
    state = hydro.hydro_state(mol, selection.Split(frozenset(atoms), (), (), NAME, 0, frozenset()))
    expect = {a.GetIdx() for a in mol.GetAtoms()
              if sum(1 for n in a.GetNeighbors() if n.GetIdx() in atoms) >= 3}
    assert len(expect) == 6 and state.fusion == frozenset(expect)
