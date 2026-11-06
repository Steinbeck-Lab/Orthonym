"""``perception.substructure.same_graph_matches``: the matches of the plain call, for a mixture drawn as one graph.

RDKit's matcher takes the atoms of the query in the query's own order. A query that is a mixture of many components
(one-atom and two-atom components among them) and a target of the same graph in another atom order made it let a
one-atom component take a carbon of a two-atom component and find the misfit only after the last component: the call
did not return in 30 s and cannot be interrupted (PubChem rows of the 2026-10-30 filtered run, NATIVE_HANG, in
``assembly.locant_omission.skeleton_label_maps``). The helper compares the atom and bond count of each component, which
cuts only branches that hold no complete match, so the list it returns is the list of the plain call."""
import random
import subprocess
import sys
import textwrap

import pytest
from rdkit import Chem, RDLogger

from orthonym.perception.substructure import same_graph_matches

pytestmark = pytest.mark.unit

#: the pair that hung the plain call, written in the atom order of the molecules that the isotope search matched:
#: the query lists its one-atom components first, the target has two of them after its last hydrogen
HANG_QUERY = "C.C.C.C.C.C.CC.CC.CC.CC.CC.CC.CC.CC.CC.CC.CCCC.[HH].[H][H]"
HANG_TARGET = "C.C.C.C.CC.CC.CC.CC.CC.CC.CC.CC.CC.CC.CCCC.[HH].C.C.[H][H]"

PIECES = ["C", "CC", "CCC", "C1CC1", "N", "O", "[Na+]", "[Cl-]", "C=C", "c1ccccc1", "CC(=O)O", "C[C@H](F)Cl",
          "C[C@@H](F)Cl", "[HH]", "[H][H]", "NC", "C#N"]


def _plain(target, query, use_chirality):
    return list(target.GetSubstructMatches(query, uniquify=False, maxMatches=64, useChirality=use_chirality))


def _is_isomorphism(target, query, match):
    """Every query bond is a target bond of the same order, on distinct atoms: with equal counts, an isomorphism."""
    if len(set(match)) != query.GetNumAtoms():
        return False
    for bond in query.GetBonds():
        other = target.GetBondBetweenAtoms(match[bond.GetBeginAtomIdx()], match[bond.GetEndAtomIdx()])
        if other is None or other.GetBondType() != bond.GetBondType():
            return False
    return True


def test_a_mixture_the_plain_call_does_not_return_on_is_matched():
    """The pair is run in a child process: were the helper to fall back to the plain call, the test would end at the
    time-out and fail, not hang the suite."""
    code = textwrap.dedent(f"""
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog('rdApp.*')
        from orthonym.perception.substructure import same_graph_matches
        query, target = Chem.MolFromSmiles({HANG_QUERY!r}), Chem.MolFromSmiles({HANG_TARGET!r})
        found = same_graph_matches(target, query, use_chirality=True)
        ok = len(found) == 64 and len(set(found)) == 64 and all(
            len(set(m)) == query.GetNumAtoms() for m in found)
        print('OK' if ok else 'BAD')
    """)
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert done.stdout.strip().endswith("OK"), done.stdout + done.stderr


def test_the_matches_of_the_hang_pair_are_isomorphisms():
    RDLogger.DisableLog("rdApp.*")
    query, target = Chem.MolFromSmiles(HANG_QUERY), Chem.MolFromSmiles(HANG_TARGET)
    found = same_graph_matches(target, query, use_chirality=True)
    assert len(found) == 64
    assert all(_is_isomorphism(target, query, m) for m in found)


@pytest.mark.parametrize("seed", range(6))
def test_the_list_is_the_list_of_the_plain_call(seed):
    """Small mixtures in two random atom orders, with and without chirality: the same matches in the same order
    (the plain call returns at once on these; 386 of 1,500 such pairs were capped at 64 and equal too)."""
    RDLogger.DisableLog("rdApp.*")
    rng = random.Random(seed)
    for _ in range(25):
        mol = Chem.MolFromSmiles(".".join(rng.choice(PIECES) for _ in range(rng.randint(2, 5))))
        assert mol is not None

        def shuffled():
            order = list(range(mol.GetNumAtoms()))
            rng.shuffle(order)
            return Chem.RenumberAtoms(mol, order)

        target, query = shuffled(), shuffled()
        for use_chirality in (False, True):
            assert same_graph_matches(target, query, use_chirality=use_chirality) == _plain(target, query,
                                                                                            use_chirality)


def test_a_connected_query_and_a_pair_of_other_sizes_are_matched_by_the_plain_call():
    benzene = Chem.MolFromSmiles("c1ccccc1")
    assert same_graph_matches(benzene, benzene, max_matches=64) == _plain(benzene, benzene, False)
    ethane, mixture = Chem.MolFromSmiles("CC"), Chem.MolFromSmiles("CC.CC.C")
    assert same_graph_matches(mixture, ethane) == _plain(mixture, ethane, False)
    assert same_graph_matches(ethane, mixture) == _plain(ethane, mixture, False) == []


def test_the_graphs_of_the_caller_are_not_changed():
    query, target = Chem.MolFromSmiles("C.CC.CC"), Chem.MolFromSmiles("CC.C.CC")
    smiles = [Chem.MolToSmiles(m, canonical=False) for m in (query, target)]
    same_graph_matches(target, query)
    assert [Chem.MolToSmiles(m, canonical=False) for m in (query, target)] == smiles
    assert not any(atom.HasProp("_orthonym_component_shape") for m in (query, target) for atom in m.GetAtoms())


def test_ordered_matches_of_the_hang_pair_returns(monkeypatch):
    """``ordered_matches`` renumbers the query into its canonical order and did not return on the pair either (the
    frames of ``isotope_scopes`` and the tautomer reading of ``isotope_compare`` of the census rows)."""
    code = textwrap.dedent(f"""
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog('rdApp.*')
        from orthonym.perception.substructure import ordered_matches
        query, target = Chem.MolFromSmiles({HANG_QUERY!r}), Chem.MolFromSmiles({HANG_TARGET!r})
        found = ordered_matches(target, query, use_chirality=True, uniquify=False, max_matches=64)
        print('OK' if len(found) == 64 else 'BAD')
    """)
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert done.stdout.strip().endswith("OK"), done.stdout + done.stderr


@pytest.mark.parametrize("seed", range(4))
def test_ordered_matches_gives_what_it_gave(seed, monkeypatch):
    """The shape tags cut no complete match: the matches of ``ordered_matches`` are the matches of the renumbered
    plain call, in its order, whether the pair is tagged or not."""
    from orthonym.perception import substructure
    RDLogger.DisableLog("rdApp.*")
    rng = random.Random(100 + seed)
    for _ in range(25):
        mol = Chem.MolFromSmiles(".".join(rng.choice(PIECES) for _ in range(rng.randint(2, 5))))
        order = list(range(mol.GetNumAtoms()))
        rng.shuffle(order)
        target, query = Chem.RenumberAtoms(mol, order), mol
        for uniquify in (False, True):
            tagged = substructure.ordered_matches(target, query, uniquify=uniquify, max_matches=64)
            with monkeypatch.context() as m:
                m.setattr(substructure, "same_graph_pair", lambda *a, **k: None)
                plain = substructure.ordered_matches(target, query, uniquify=uniquify, max_matches=64)
            assert tagged == plain


#: a cyclodextrin of seven glucose units, six of them with a (tert-butyl)dimethylsilyl ether, the rest methyl ethers
#: (142 atoms; the target is the input of a PubChem row, the query its name read by OPSIN, in OPSIN's atom order): the
#: matcher did not return on the first 100 matches in 40 s, in the query's own order and in its canonical order
CYCLODEXTRIN_TARGET = 'CO[C@@H]1[C@@H](OC)[C@H]2O[C@H]3[C@H](OC)[C@@H](OC)[C@@H](O[C@H]4[C@H](OC)[C@@H](OCCO)[C@@H](O[C@H]5[C@H](OC)[C@H](OC)[C@@H](O[C@H]6[C@@H](OC)[C@H](OC)[C@@H](O[C@H]7[C@H](OC)[C@@H](OC)[C@@H](O[C@H]8[C@H](OC)[C@@H](OC)[C@@H](O[C@@H]1[C@H](CO[Si](C)(C)C(C)(C)C)O2)O[C@H]8CO[Si](C)(C)C(C)(C)C)O[C@@H]7CO[Si](C)(C)C(C)(C)C)O[C@@H]6CO[Si](C)(C)C(C)(C)C)O[C@@H]5CO[Si](C)(C)C(C)(C)C)O[C@H]4CO[Si](C)(C)C(C)(C)C)O[C@H]3CO[Si](C)(C)C(C)(C)C'
CYCLODEXTRIN_QUERY = 'C(C)(C)(C)[Si](OCC1C2OC3OC(CO[Si](C)(C)C(C)(C)C)C(OC4OC(CO[Si](C)(C)C(C)(C)C)C(OC5OC(CO[Si](C)(C)C(C)(C)C)C(OC6OC(CO[Si](C)(C)C(C)(C)C)C(OC7OC(CO[Si](C)(C)C(C)(C)C)C(OC8OC(CO[Si](C)(C)C(C)(C)C)C(OC(O1)C(OC)C2OC)C(OC)C8OCCO)C(OC)C7OC)C(OC)C6OC)C(OC)C5OC)C(OC)C4OC)C(OC)C3OC)(C)C'


def test_a_connected_pair_of_a_symmetric_molecule_is_matched_through_the_symmetry_classes():
    """The pair runs in a child process: were the helper to fall back to the plain call, the test would end at the
    time-out and fail, not hang the suite (the symmetry classes of ``substructure._connected_pair``)."""
    code = textwrap.dedent(f"""
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog('rdApp.*')
        from orthonym.perception.substructure import ordered_matches, same_graph_matches
        target, query = Chem.MolFromSmiles({CYCLODEXTRIN_TARGET!r}), Chem.MolFromSmiles({CYCLODEXTRIN_QUERY!r})
        found = ordered_matches(target, query, uniquify=False, max_matches=1000)
        again = same_graph_matches(target, query, max_matches=64)
        n = query.GetNumAtoms()
        ok = len(found) == 1000 and len(again) == 64 and all(len(set(m)) == n for m in found + again)
        print('OK' if ok else 'BAD')
    """)
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert done.stdout.strip().endswith("OK"), done.stdout + done.stderr


CONNECTED = ["CC(C)(C)C", "C1CCCCC1", "c1ccccc1", "CC(C)(C)[Si](C)(C)OCC(O)CO[Si](C)(C)C(C)(C)C", "C[C@H](O)C(=O)O",
             "OC1C(O)C(O)C(O)C(O)C1O", "C1CC2CCC1CC2", "CCOCCOCCOCC", "N#CC(C)(C)N=NC(C)(C)C#N", "FC(F)(F)c1ccc(cc1)C(F)(F)F"]


@pytest.mark.parametrize("seed", range(4))
def test_the_symmetry_classes_cut_no_match_of_a_connected_pair(seed):
    """Two random atom orders of a symmetric molecule, with and without chirality: the matches of the tagged pair are
    the matches of the plain call, in its order (the classes cut only branches that hold no complete match)."""
    from orthonym.perception import substructure
    RDLogger.DisableLog("rdApp.*")
    rng = random.Random(200 + seed)
    for smiles in CONNECTED:
        mol = Chem.MolFromSmiles(smiles)

        def shuffled():
            order = list(range(mol.GetNumAtoms()))
            rng.shuffle(order)
            return Chem.RenumberAtoms(mol, order)

        target, query = shuffled(), shuffled()
        assert substructure._connected_pair(target, query) is not None
        for use_chirality in (False, True):
            assert same_graph_matches(target, query, use_chirality=use_chirality, max_matches=500) == list(
                target.GetSubstructMatches(query, uniquify=False, maxMatches=500, useChirality=use_chirality))


@pytest.mark.parametrize("seed", range(3))
def test_the_symmetry_class_of_an_atom_does_not_depend_on_the_atom_order(seed):
    """The pruning is sound only if the atoms a match pairs have one class in both graphs: the classes are the same
    function of the graph whatever its atom order, whatever the charges, bond orders or hydrogens of the drawing."""
    from orthonym.perception.substructure import _symmetry_classes
    RDLogger.DisableLog("rdApp.*")
    rng = random.Random(300 + seed)
    for smiles in CONNECTED + ["C[N+](C)(C)CC([O-])=O", "[2H]C([2H])([2H])O", "c1ccc2ccccc2c1"]:
        mol = Chem.MolFromSmiles(smiles)
        order = list(range(mol.GetNumAtoms()))
        rng.shuffle(order)
        moved = Chem.RenumberAtoms(mol, order)
        base, after = _symmetry_classes(mol), _symmetry_classes(moved)
        assert all(after[new] == base[old] for new, old in enumerate(order)), smiles
    # a different drawing of one graph (a Kekule form, an explicit hydrogen) gives the same classes
    kekule = Chem.MolFromSmiles("C1=CC=CC=C1")
    assert _symmetry_classes(kekule) == _symmetry_classes(Chem.MolFromSmiles("c1ccccc1"))


def test_a_dummy_atom_or_a_query_atom_is_matched_by_the_plain_call():
    from orthonym.perception.substructure import _connected_pair
    assert _connected_pair(Chem.MolFromSmiles("CC"), Chem.MolFromSmiles("C*")) is None
    assert _connected_pair(Chem.MolFromSmiles("CC"), Chem.MolFromSmarts("[#6][#6]")) is None


#: a PubChem row of 177 atoms (a polyether of fluorescein-like units): the target is the input, the query a name OPSIN
#: reads as another molecule, with one double bond fewer and two hydrogens more, in OPSIN's atom order
BOND_ORDER_TARGET = 'O=C(Nc1ccc(-c2c3ccc(=O)cc-3oc3cc(O)ccc23)c(C(=O)OC(O)C(C(O)O)C(C(C(O)O)C(O)O)C(C(C(C(O)O)C(O)O)C(C(O)O)C(O)O)C(C(C(C(C(O)O)C(O)O)C(C(O)O)C(O)O)C(C(C(O)O)C(O)O)C(C(O)O)C(O)O)(C(C(C(C(O)O)C(O)O)C(C(O)O)C(O)O)C(C(C(O)O)C(O)O)C(C(O)O)C(O)O)C(C(C(C(O)O)C(O)O)C(C(O)O)C(O)O)C(C(C(O)O)C(O)O)C(C(O)O)C(O)O)c1)c1cc(NC=S)ccc1-c1c2ccc(=O)cc-2oc2cc(O)ccc12'
BOND_ORDER_QUERY = 'OC(C(C(O)O)C(C(C(C(C(O)O)C(O)O)C(C(O)O)C(O)O)C(C(C(C(C(O)OC(=O)c1c(-c2c3ccc(O)cc3oc3cc(=O)ccc2-3)ccc(NC(=O)c2c(-c3c4ccc(O)cc4oc4cc(=O)ccc3-4)ccc(NCS)c2)c1)C(O)O)C(C(O)O)C(O)O)C(C(C(O)O)C(O)O)C(C(O)O)C(O)O)(C(C(C(C(O)O)C(O)O)C(C(O)O)C(O)O)C(C(C(O)O)C(O)O)C(C(O)O)C(O)O)C(C(C(C(O)O)C(O)O)C(C(O)O)C(O)O)C(C(C(O)O)C(O)O)C(C(O)O)C(O)O)C(C(O)O)C(O)O)O'


def test_a_pair_that_differs_by_a_bond_order_holds_no_match_and_returns():
    """A name that OPSIN reads as another molecule is a query of the input's size that no match fits; the matcher
    backtracked through the symmetry of its equivalent atoms (48 of them in one class) and did not return in 40 s (the
    call cannot be interrupted). The local environments of the two graphs differ, so the pair is tagged to end at
    the first atom. A child process: the plain call would end at the time-out."""
    from orthonym.perception.substructure import _local_environments, ordered_matches
    code = textwrap.dedent(f"""
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog('rdApp.*')
        from orthonym.perception.substructure import ordered_matches
        target, query = Chem.MolFromSmiles({BOND_ORDER_TARGET!r}), Chem.MolFromSmiles({BOND_ORDER_QUERY!r})
        print('OK' if not ordered_matches(target, query, uniquify=False, max_matches=1000) else 'BAD')
    """)
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert done.stdout.strip().endswith("OK"), done.stdout + done.stderr
    ethane, ethene = Chem.MolFromSmiles("CC"), Chem.MolFromSmiles("C=C")
    assert _local_environments(ethane) != _local_environments(ethene)
    assert ordered_matches(ethane, ethene) == [] == ordered_matches(ethene, ethane)


@pytest.mark.parametrize("seed", range(3))
def test_the_local_environments_cut_no_match_of_a_pair_that_matches(seed):
    """Aromatic or Kekule, charged or not: the pairs RDKit's matcher pairs keep their environments, so the tagged call
    returns the plain call's matches; a pair it does not match gives none from both."""
    RDLogger.DisableLog("rdApp.*")
    rng = random.Random(400 + seed)
    smiles = ["c1ccccc1C(=O)O", "CC(=O)[O-]", "C[N+](C)(C)C", "O=C1CCCCC1", "c1ccc2ccccc2c1", "CC(C)(C)OC(=O)N"]
    for one in smiles:
        mol = Chem.MolFromSmiles(one)
        kek = Chem.Mol(mol)
        Chem.Kekulize(kek, clearAromaticFlags=True)
        for target_graph, query_graph in ((mol, mol), (mol, kek), (kek, mol)):
            order = list(range(mol.GetNumAtoms()))
            rng.shuffle(order)
            target, query = Chem.RenumberAtoms(target_graph, order), query_graph
            assert same_graph_matches(target, query, max_matches=200) == list(
                target.GetSubstructMatches(query, uniquify=False, maxMatches=200))
