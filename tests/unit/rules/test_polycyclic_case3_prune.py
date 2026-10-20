"""The pruned Case-3 pair loop of the von Baeyer main-ring search is exact.

``polycyclic._consider_multiatom_bridges`` used to charge one perf-budget unit per (i, j)
pair of bh1->bh2 paths and reject almost every pair afterwards; it now skips the pairs that
cannot reach ``_MainRingBest.consider`` or cannot change what ``best`` ends up holding (see its
docstring for the proof of each bound). What the change must not do is move any result: the
final ``score``, ``ring``, ``bh_pair`` and the ``tied`` list IN ITS ORDER (``tied`` feeds the
locant-based numbering choice of ``_choose_lowest_locant_numbering``).

``_reference_loop`` below is the loop as it stood before the change, verbatim. It is run beside
the pruned one

* on random ring systems with random incumbents (none, one candidate, a tie, a ring larger than
  any pair can reach), and
* on the main-ring search of real ring systems: the four hang witnesses of
  ``test_m25_workbudget.py`` and three unnamed PubChem rows whose Case-3 loop was the top
  spender of the 12M budget.

No JVM is needed.
"""
import random

import pytest
from rdkit import Chem

from orthonym.rules import polycyclic as pc

pytestmark = pytest.mark.unit


def _reference_loop(all_paths, spend, best, bh1, bh2, room=None):
    """``_consider_multiatom_bridges`` at feb066505, before the prune (``room`` unused)."""
    npaths = len(all_paths)
    masks, plens = pc._interior_masks(all_paths)
    bridge_order = sorted((k for k in range(npaths) if plens[k] >= 2), key=lambda k: -plens[k])
    for i in range(npaths):
        mask_i = masks[i]
        len_i = plens[i]
        for j in range(i + 1, npaths):
            spend()
            mask_j = masks[j]
            if mask_i & mask_j:
                continue
            len_j = plens[j]
            ring_size = len_i + len_j + 2
            if best.score is not None and ring_size < best.score[0][0]:
                continue
            used = mask_i | mask_j
            best_mb_len = -1
            for k in bridge_order:
                if k != i and k != j and not (masks[k] & used):
                    best_mb_len = plens[k]
                    break
            if best_mb_len < 2:
                continue
            p2 = all_paths[j]
            best.consider(all_paths[i] + p2[1:-1][::-1], bh1, bh2, len_i, len_j, best_mb_len)


def _snapshot(best):
    return (best.score, best.ring, best.bh_pair, [(list(r), b) for r, b in best.tied])


# ---------------------------------------------------------------------------
# random ring systems
# ---------------------------------------------------------------------------

def _random_graph(rng):
    """A connected random graph of 8-14 carbons with 3-6 more edges than a tree."""
    n = rng.randint(8, 14)
    mol = Chem.RWMol()
    for _ in range(n):
        mol.AddAtom(Chem.Atom(6))
    edges = set()
    for v in range(1, n):
        u = rng.randrange(v)
        edges.add((u, v))
    extra = rng.randint(3, 6)
    while extra:
        u, v = sorted(rng.sample(range(n), 2))
        if (u, v) not in edges:
            edges.add((u, v))
            extra -= 1
    for u, v in sorted(edges):
        mol.AddBond(u, v, Chem.BondType.SINGLE)
    return mol, n


def _seeded_best(rank_key, seed_candidates):
    best = pc._MainRingBest(rank_key)
    for ring, bh1, bh2, l1, l2, mb in seed_candidates:
        best.consider(ring, bh1, bh2, l1, l2, mb)
    return best


def _run(loop, all_paths, rank_key, seeds, bh1, bh2, room):
    best = _seeded_best(rank_key, seeds)
    spent = [0]

    def spend():
        spent[0] += 1

    loop(all_paths, spend, best, bh1, bh2, room)
    return _snapshot(best), spent[0]


def test_pruned_loop_matches_reference_on_random_ring_systems():
    rng = random.Random(20261020)
    stats = {"runs": 0, "offered": 0, "early_return": 0, "fewer_pairs": 0, "tied_gt1": 0}
    for _ in range(150):
        mol, n = _random_graph(rng)
        ranks = rng.sample(range(n), n)

        def rank_key(atoms, ranks=ranks):
            return tuple(sorted(ranks[a] for a in atoms))

        allowed = set(range(n))
        for bh1, bh2 in (tuple(sorted(rng.sample(range(n), 2))) for _ in range(2)):
            all_paths = pc._find_all_simple_paths_impl(mol, bh1, bh2, allowed)
            if len(all_paths) < 3:
                continue
            room = len(allowed - {bh1, bh2})
            ring_size_cap = n + 2

            def seed(size, mb, rng_local):
                ring = [rng_local.randrange(n) for _ in range(size)]
                half = (size - 2) // 2
                return (ring, bh1, bh2, half, size - 2 - half, mb)

            seed_sets = [
                [],                                                       # nothing yet
                [seed(rng.randint(3, room - 2), rng.randint(0, 1), rng)],  # Case 1/2 incumbent
                [seed(s, 0, rng) for s in (rng.randint(3, room - 2),) * 2],  # a tie already
                [seed(ring_size_cap, 1, rng)],                            # above any pair
                [seed(rng.randint(room - 4, room - 1), rng.randint(0, 1), rng)],  # near the top
            ]
            for seeds in seed_sets:
                ref, n_ref = _run(_reference_loop, all_paths, rank_key, seeds, bh1, bh2, room)
                new, n_new = _run(pc._consider_multiatom_bridges, all_paths, rank_key, seeds,
                                  bh1, bh2, room)
                assert new == ref, (bh1, bh2, seeds)
                assert n_new <= n_ref
                stats["runs"] += 1
                stats["offered"] += new != _snapshot(_seeded_best(rank_key, seeds))
                stats["early_return"] += n_new == 0 and n_ref > 0
                stats["fewer_pairs"] += n_new < n_ref
                stats["tied_gt1"] += len(new[3]) > 1
    # the comparison must not be vacuous: candidates were offered, ties were built, the early
    # return fired, and the window skipped pairs
    assert stats["runs"] > 200
    assert stats["offered"] > 50, stats
    assert stats["tied_gt1"] > 20, stats
    assert stats["early_return"] > 10, stats
    assert stats["fewer_pairs"] > 50, stats


def test_pruned_loop_matches_reference_on_random_path_families():
    """Path families built directly (interiors drawn from a pool of ``room`` atoms, so the
    bound holds as it does for a real ring system), many more disjoint triples and exact ties
    than a random graph gives, including families that partition the pool into three paths."""
    rng = random.Random(1020)
    stats = {"runs": 0, "tied_gt1": 0, "at_room": 0, "fewer_pairs": 0}
    for _ in range(300):
        room = rng.randint(6, 13)
        pool = list(range(2, 2 + room))
        ranks = dict(zip(range(2 + room), rng.sample(range(2 + room), 2 + room)))

        def rank_key(atoms, ranks=ranks):
            return tuple(sorted(ranks[a] for a in atoms))

        interiors = []
        if rng.random() < 0.5:                      # three paths that use every pool atom
            shuffled = rng.sample(pool, room)
            cuts = sorted(rng.sample(range(2, room - 1), 2)) if room >= 6 else [2, 4]
            interiors += [shuffled[:cuts[0]], shuffled[cuts[0]:cuts[1]], shuffled[cuts[1]:]]
        for _ in range(rng.randint(3, 9)):
            interiors.append(rng.sample(pool, rng.randint(0, min(5, room))))
        all_paths = []
        for interior in rng.sample(interiors, len(interiors)):
            path = [0, *interior, 1]
            if path not in all_paths:
                all_paths.append(path)
        if len(all_paths) < 3:
            continue
        for size in (None, rng.randint(3, room), room, room - 1, room + 1):
            seeds = []
            if size is not None:
                half = (size - 2) // 2
                seeds = [([rng.choice(pool) for _ in range(size)], 0, 1, half, size - 2 - half, 1)]
            ref, n_ref = _run(_reference_loop, all_paths, rank_key, seeds, 0, 1, room)
            new, n_new = _run(pc._consider_multiatom_bridges, all_paths, rank_key, seeds,
                              0, 1, room)
            assert new == ref, (all_paths, seeds)
            assert n_new <= n_ref
            stats["runs"] += 1
            stats["tied_gt1"] += len(new[3]) > 2
            stats["at_room"] += new[0] is not None and new[0][0][0] == room
            stats["fewer_pairs"] += n_new < n_ref
    assert stats["runs"] > 500
    assert stats["tied_gt1"] > 30, stats
    assert stats["at_room"] > 30, stats
    assert stats["fewer_pairs"] > 200, stats


def test_bound_of_the_window_is_the_ring_atoms_outside_the_bridgeheads():
    """Two paths of 2 interior atoms plus a third of 2 need 6 free atoms: the ring of the first
    two (6 atoms) is offered at room 6 and unreachable at room 5 (such a call cannot occur
    for a real ring system, whose paths all lie inside it; here it shows where the bound sits)."""
    paths = [[0, 2, 3, 1], [0, 4, 5, 1], [0, 6, 7, 1]]
    ranks = {a: a for a in range(8)}

    def rank_key(atoms):
        return tuple(sorted(ranks[a] for a in atoms))

    def run(loop, room):
        best = pc._MainRingBest(rank_key)
        loop(paths, lambda: None, best, 0, 1, room)
        return _snapshot(best)

    assert run(pc._consider_multiatom_bridges, 6) == run(_reference_loop, 6)
    assert run(pc._consider_multiatom_bridges, 6)[0] is not None
    assert len(run(pc._consider_multiatom_bridges, 6)[3]) == 3   # all three pairs tie
    assert run(pc._consider_multiatom_bridges, 5)[0] is None     # ring 6 > room 5


# ---------------------------------------------------------------------------
# real ring systems
# ---------------------------------------------------------------------------

_NI_CORRIN = ("C[C@@]1(CC(=O)[O-])C2=CC3=[N+]4C(=Cc5c(CC(=O)[O-])c(CCC(=O)[O-])c6"
              "[n]5[Ni-2]45[N]2C(=CC2=[N+]5C(=C6)C(CCC(=O)[O-])=C2CC(=O)[O-])"
              "[C@H]1CCC(=O)[O-])[C@@](C)(CC(=O)[O-])[C@@H]3CCC(=O)[O-]")
_COBYRINATE = ("C/C1=C2/[N]([Co+])[C@H]([C@H](CC(=O)O)[C@@]2(C)CCC(=O)O)[C@]2(C)"
               "N=C(/C(C)=C3\\N=C(/C=C4\\N=C1[C@@H](CCC(=O)O)C4(C)C)"
               "[C@@H](CCC(=O)O)[C@]3(C)CC(=O)O)[C@@H](CCC(=O)O)[C@]2(C)CC(=O)O")
_FE_CORRIN = ("C[C@@]1(CC(=O)O)C2=CC3=[N+]4C(=Cc5c(CCC(=O)O)c(CC(=O)O)c6[n]5"
              "[Fe-2]45[N]2C(=CC2=[N+]5C(=C6)[C@@H](CCC(=O)O)[C@]2(C)CC(=O)O)"
              "[C@H]1CCC(=O)O)C(CCC(=O)O)=C3CC(=O)O")
_CHLOROPHYLL = ("C=CC1=C(C)C2=Cc3c(C=C)c(C)c4[n]3[Mg-2]35[n]6c(c(C)c7c6=C(C6=[N+]3"
                "C(=C4)[C@@H](C)[C@@H]6CCC(=O)OC/C=C(\\C)CCC[C@H](C)CCC[C@H](C)"
                "CCCC(C)C)[C-](C(=O)OC)C7=O)=CC1=[N+]25")
# PubChem rows that abstain at the 12M budget with the unpruned loop (the Case-3 pair loop is
# 59-93% of their units) and name with the pruned one
_CID152863978 = ("CN1C2=NC3=NC(=NC4=NC(=NC(=N2)N34)N(C)C)N(C5=NC6=NC(=NC7=NC(=NC(=N5)N67)N(C)C)"
                 "N(C8=NC9=NC(=NC2=NC1=NC(=N8)N29)N(C)C)C1=NC2=NC(=NC3=NC(=NC(=N1)N23)N(C)C)"
                 "N(C)C)C")
_CID169057336 = ("CC(C)(C)C1=CC2=C(C(=C1)C3=CC4=C(C=C3)N(C5=CC=CC=C54)C6=CC=CC=C6)N7C8=C2C9=CC="
                 "CC=C9C=C8C1=CC2=C(C3=CC(=CC4=C3N2C2=C4C3=CC=CC=C3C=C2)C(C)(C)C)N=C17")
_CID162480633 = ("B12C3=CC(=CC4=C3N(C5=CC6=CC=CC=C6C=C54)C7=CC(=CC(=C71)N8C9=CC1=CC=CC=C1C=C9C1="
                 "C8C2=CC(=C1)C)C1=CC=C(C=C1)C(C)(C)C)C")


def _ring_systems(mol):
    """Fused ring systems (rings that share an atom), largest first."""
    systems = []
    for ring in mol.GetRingInfo().AtomRings():
        merged = set(ring)
        for system in [s for s in systems if s & merged]:
            systems.remove(system)
            merged |= system
        systems.append(merged)
    return sorted(systems, key=lambda s: -len(s))


def _main_ring_search(smiles, loop, monkeypatch):
    """``_find_main_ring_impl`` (with the tied candidates) of every ring system of ``smiles``
    with the given Case-3 loop, and the budget units the search charged."""
    mol = Chem.MolFromSmiles(smiles)
    analyzer = pc.VonBaeyerAnalyzer()
    spent = [0]

    def spend(n=1):
        spent[0] += n

    monkeypatch.setattr(pc, "_consider_multiatom_bridges", loop)
    monkeypatch.setattr(pc, "spend_perf_work", spend)
    results = []
    for ring_atoms in _ring_systems(mol):
        bridgeheads = analyzer._find_all_bridgeheads(mol, ring_atoms)
        if len(bridgeheads) < 2:
            continue
        results.append(analyzer._find_main_ring_impl(mol, ring_atoms, bridgeheads,
                                                     return_candidates=True))
    return results, spent[0]


def _assert_same_search(smiles, monkeypatch, fewer_units):
    new, n_new = _main_ring_search(smiles, pc._consider_multiatom_bridges, monkeypatch)
    ref, n_ref = _main_ring_search(smiles, _reference_loop, monkeypatch)
    assert new and new == ref
    assert n_new <= n_ref
    if fewer_units:
        assert n_new < n_ref, (n_new, n_ref)


@pytest.mark.parametrize("smiles", [_NI_CORRIN, _COBYRINATE, _FE_CORRIN, _CHLOROPHYLL],
                         ids=["Ni-corrin", "cobyrinate", "Fe-corrin", "chlorophyll"])
def test_witness_ring_systems_keep_their_main_ring_and_tied_list(smiles, monkeypatch):
    _assert_same_search(smiles, monkeypatch, fewer_units=smiles != _COBYRINATE)


@pytest.mark.parametrize("smiles", [_CID152863978, _CID169057336, _CID162480633],
                         ids=["CID152863978", "CID169057336", "CID162480633"])
def test_unnamed_row_ring_systems_keep_their_main_ring_and_tied_list(smiles, monkeypatch):
    _assert_same_search(smiles, monkeypatch, fewer_units=True)
