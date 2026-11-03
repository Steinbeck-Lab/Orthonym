"""The main ring of a ring system of 16 or more rings, found as a longest cycle (``rules/main_ring_cycles.py``).

 "Selection of the main ring" (``the Blue Book Blue Book``): "The main ring of a polycyclic
hydrocarbon ring system is selected so as to include as many skeletal atoms of the structure as possible";
 "Selection of the main bridge" (``:9603``): "The main bridge is selected so as to be as large as
possible"; "Selection of the main bridge and secondary bridges", rule (``:9661``):
"The main ring must be divided as symmetrically as possible by the main bridge".

The path search lists the paths between every pair of bridgeheads and runs the budget out on a system of
21 rings (a hundred times the budget does not name it either). These tests pin:

* the cycles listed are the longest simple cycles of the system, against an independent enumeration;
* with the search forced on small systems, the main ring has the size the path search finds, and every
  candidate the path search keeps is among the candidates here (with the preference for a 2+-atom main
  bridge on and off);
* a set of ring atoms that is not one biconnected ring system is refused, not searched without a bound;
* the 21-ring system of boron, oxygen and carbon gets its descriptor in seconds, the same in every
  spelling, and a 20- and an 18-ring member of its class keep theirs;
* the numberings of a 16-ring cage are ranked over all its decompositions (the cap of systems of 16
  rings or more);
* no system below 16 rings reaches the search.

No JVM is needed.
"""
import random

import pytest
from rdkit import Chem

from orthonym.rules import main_ring_cycles as mrc
from orthonym.rules import polycyclic as pc
from orthonym.rules import vonbaeyer_universal as vbu
from orthonym.rules.polycyclic import VonBaeyerAnalyzer
from orthonym.errors import OrthonymLimitError
from tests.support.large_polycycles import ladder, spellings
from tests.unit.breadth.test_unnamed_l7_vonbaeyer import GOLD_VON_BAEYER, _ring_systems
from tests.unit.rules.test_main_ring_levels import _polyhex, _random_cage

pytestmark = pytest.mark.unit


@pytest.fixture
def free_budget(monkeypatch):
    """The searches charge nothing: they are compared, not timed."""
    monkeypatch.setattr(pc, "spend_perf_work", lambda n=1: None)


def _small_systems():
    mols = [Chem.MolFromSmiles(s) for s in GOLD_VON_BAEYER]
    mols += [_random_cage(random.Random(seed)) for seed in range(120)]
    mols += [_polyhex(random.Random(seed), random.Random(seed).randint(3, 9)) for seed in range(30)]
    return mols


def _edges(ring):
    return frozenset(frozenset((ring[i], ring[(i + 1) % len(ring)])) for i in range(len(ring)))


def _all_simple_cycles(nbrs):
    """Every simple cycle, by brute force (independent of the code under test)."""
    out = set()

    def rec(s, v, path, seen):
        for w in nbrs[v]:
            if w == s and len(path) >= 3:
                out.add(_edges(path))
            elif w > s and w not in seen:
                seen.add(w)
                path.append(w)
                rec(s, w, path, seen)
                path.pop()
                seen.discard(w)

    for s in nbrs:
        rec(s, s, [s], {s})
    return out


def test_the_cycles_listed_are_the_longest_simple_cycles(free_budget):
    checked = 0
    for mol in _small_systems():
        for ring_atoms, _bridgeheads in _ring_systems(VonBaeyerAnalyzer(), mol):
            nbrs = {a: [n.GetIdx() for n in mol.GetAtomWithIdx(a).GetNeighbors() if n.GetIdx() in ring_atoms]
                    for a in ring_atoms}
            cycles = _all_simple_cycles(nbrs)
            longest = max(len(c) for c in cycles)
            length, listed, complete = mrc.longest_cycles(nbrs, sorted(ring_atoms), lambda n: None)
            assert complete and length == longest
            assert [len(c) for c in listed] == [length] * len(listed)
            assert {_edges(c) for c in listed} == {c for c in cycles if len(c) == longest}
            assert len({_edges(c) for c in listed}) == len(listed)      # each cycle once
            checked += 1
    assert checked > 150


@pytest.mark.parametrize("allow_multiatom_bridge", [True, False])
def test_every_candidate_of_the_path_search_is_a_candidate_here(allow_multiatom_bridge, free_budget, monkeypatch):
    checked = 0
    for mol in _small_systems():
        analyzer = VonBaeyerAnalyzer()
        analyzer._allow_multiatom_bridge = allow_multiatom_bridge
        for ring_atoms, bridgeheads in _ring_systems(analyzer, mol):
            monkeypatch.setattr(mrc, "CYCLE_FIRST_MIN_RINGS", 10 ** 9)
            old_ring, _old_pair, old_tied = analyzer._find_main_ring_impl(
                mol, ring_atoms, bridgeheads, return_candidates=True)
            monkeypatch.setattr(mrc, "CYCLE_FIRST_MIN_RINGS", 0)
            try:
                new_ring, _new_pair, new_tied = analyzer._find_main_ring_impl(
                    mol, ring_atoms, bridgeheads, return_candidates=True)
            except OrthonymLimitError:
                # a spiro union of ring systems: not one biconnected system (next test)
                continue
            assert len(new_ring) == len(old_ring)
            new = {(_edges(r), frozenset(b)) for r, b in new_tied}
            assert {(_edges(r), frozenset(b)) for r, b in old_tied} <= new
            checked += 1
    assert checked > 150


def test_a_ring_set_that_is_not_one_biconnected_system_is_refused(free_budget, monkeypatch):
    monkeypatch.setattr(mrc, "CYCLE_FIRST_MIN_RINGS", 0)
    mol = Chem.MolFromSmiles("CC1CCC2(C1)CC21CCC(C)(C)C1")       # two rings joined at one atom
    analyzer = VonBaeyerAnalyzer()
    ring_atoms = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    bridgeheads = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing() and a.GetDegree() > 2}
    with pytest.raises(OrthonymLimitError):
        analyzer._find_main_ring_impl(mol, ring_atoms, bridgeheads, return_candidates=True)


CLASS_DESCRIPTORS = {
    "R05_core_21rings": (
        "henicosacyclo[30.28.1.1^4,20.1^7,12.1^10,50.1^40,44.0^2,57.0^3,26.0^5,56.0^6,18.0^8,55.0^9,52."
        "0^28,61.0^34,60.0^36,41.0^42,59.0^43,53.0^46,51.0^54,58.0^16,65.0^24,64]"),
    "core20b_a_20rings": (
        "icosacyclo[40.18.1.1^9,56.1^36,40.1^50,54.0^2,39.0^3,19.0^4,59.0^5,18.0^6,57.0^7,16.0^8,13."
        "0^17,22.0^20,38.0^21,30.0^23,28.0^32,37.0^46,61.0^48,60.0^58,63]"),
    "R04_corannulene+4unit_BH": (
        "octadecacyclo[26.25.1.1^4,16.1^32,36.1^40,44.0^2,7.0^3,22.0^5,13.0^6,10.0^8,52.0^9,49.0^11,47."
        "0^24,54.0^30,53.0^38,50.0^20,57.0^48,55.0^51,56]"),
}


@pytest.mark.parametrize("label", sorted(CLASS_DESCRIPTORS))
def test_the_class_members_get_their_descriptor_in_every_spelling(label):
    """The 21-ring member ran the budget out in the path search; its main bicycle [30.28.1] is the
    longest cycle of 60 of the 65 atoms with a 1-atom main bridge dividing it 30/28. Each name built
    on these descriptors read back to the input's full standard InChIKey through OPSIN with ring
    counts above 20 added to its von Baeyer multipliers."""
    smiles = ladder()[label]
    for spelling in [smiles] + list(spellings(smiles, 2)):
        mol = Chem.MolFromSmiles(spelling)
        analyzer = VonBaeyerAnalyzer()
        with vbu.lifted_ring_ceiling_scope():
            found = [analyzer.analyze(mol, ring_atoms).descriptor_string
                     for ring_atoms, _bridgeheads in _ring_systems(analyzer, mol)]
        assert found == [CLASS_DESCRIPTORS[label]]


def test_no_system_below_sixteen_rings_reaches_the_search(monkeypatch):
    """The ceilings of the PIN tier and of the first best-effort attempt lie below the threshold, so
    their systems keep the path search and its result; a 15-ring polyhex does not call the search."""
    assert mrc.CYCLE_FIRST_MIN_RINGS > vbu.BEST_EFFORT_MAX_CAGE_RINGS >= vbu.MAX_CAGE_RINGS
    calls = []
    monkeypatch.setattr(mrc, "main_bicycles", lambda *a, **k: calls.append(1) or [])
    monkeypatch.setattr(pc, "spend_perf_work", lambda n=1: None)
    mol = _polyhex(random.Random(5), 15)
    analyzer = VonBaeyerAnalyzer()
    for ring_atoms, bridgeheads in _ring_systems(analyzer, mol):
        analyzer._find_main_ring_impl(mol, ring_atoms, bridgeheads, return_candidates=True)
    assert calls == []


def test_the_numbering_of_a_many_ring_cage_is_ranked_over_all_its_decompositions():
    """A 16-ring cage of 25 atoms has 28 tied decompositions (113 numberings) here, where the path
    search kept 14. Under the cap of 64 the numberings went unranked and a decomposition chosen by the
    canonical ranks alone spelled '0^1,6.0^2,23...'; the cap of systems of 16 rings or more ranks them
    all, and (the Blue Book, "The superscript locants for the secondary bridges must
    be as low as possible when considered as a set in ascending numerical order") gives '0^1,14.0^2,5...',
    the descriptor of the path search."""
    mol = Chem.MolFromSmiles("CC1C2C3C4CC5C46C37C28C1C9C81C72C63C5C4C33C22C11C9CC1C2C3C4")
    analyzer = VonBaeyerAnalyzer()
    with vbu.lifted_ring_ceiling_scope():
        found = [analyzer.analyze(mol, ring_atoms) for ring_atoms, _bridgeheads in _ring_systems(analyzer, mol)]
    assert [d.descriptor_string for d in found] == [
        "hexadecacyclo[11.11.1.0^1,14.0^2,5.0^2,23.0^3,10.0^3,25.0^4,7.0^4,9.0^5,22.0^6,21.0^11,25.0^15,24."
        "0^17,24.0^18,23.0^19,22]"]
    assert found[0].numbering_certified is not False



def test_a_cut_list_of_cycles_does_not_follow_the_neighbour_order(monkeypatch):
    """When ``MAX_CYCLES`` cuts the list, the cycles kept are the first of a DFS that takes atoms and
    neighbours in the order of ``order`` (canonical ranks in the analyzer), never in the order the
    input listed its bonds in, which a pure cage keeps through ``Chem.RenumberAtoms``
    (dodecahedrane: 20 longest cycles, 3 kept)."""
    monkeypatch.setattr(mrc, "MAX_CYCLES", 3)
    mol = Chem.MolFromSmiles(ladder()["D01_dodecahedrane"])
    nbrs = {a.GetIdx(): [n.GetIdx() for n in a.GetNeighbors()] for a in mol.GetAtoms()}
    order = sorted(nbrs)
    kept = set()
    for seed in range(12):
        rng = random.Random(seed)
        shuffled = {a: rng.sample(ns, len(ns)) for a, ns in nbrs.items()}
        length, cycles, complete = mrc.longest_cycles(shuffled, order, lambda n: None)
        assert (length, len(cycles), complete) == (20, 3, False)
        kept.add(tuple(cycles))
    assert len(kept) == 1
