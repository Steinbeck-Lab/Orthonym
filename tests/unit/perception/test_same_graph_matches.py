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
