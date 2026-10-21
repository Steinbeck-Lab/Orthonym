"""UN L7b part A: the von Baeyer main-ring search without the cubic pair scan.

``VonBaeyerAnalyzer._find_main_ring_impl`` (``rules/polycyclic.py``) scored every pair (i, j) of
bridgehead paths and, for each disjoint pair, scanned all paths again for the main bridge.
It now keeps path interiors as int bitmasks, finds a pair's main bridge in one pass over the
candidates sorted once by length, and builds the canonical-rank tie key only for a candidate that
ties or beats the incumbent on ring size, main bridge and balance,,
. The change is for speed only, so these tests pin what must NOT move:

* the result (main ring, bridgehead pair, every oriented tied candidate in order) equals the
  algorithm as it was, kept below as ``_reference_find_main_ring_impl`` (tests only);
* the perf-budget units charged (``spend_perf_work``: one per (i, j) pair and one per DFS
  expansion) are equal, so the deterministic budget decides every abstention as before --
  except in Case 3, which UN lane P prunes (``_consider_multiatom_bridges`` charges one unit per
  VISITED pair, never more than the loop it replaces; the result is unchanged, see
  ``test_polycyclic_case3_prune.py``): there the pruned search charges at most as many units, and
  exactly as many when the main-bridge preference is off;
* the full name of three cluster-A rows (UN L7b, 20 rows whose solo time was dominated by this
  search) is the name the engine gave before the change.
"""

import random
from itertools import combinations
from typing import List, Set, Tuple

import pytest
from rdkit import Chem

from orthonym.assembly import fragment_naming as fn
from orthonym.rules import polycyclic as pc
from orthonym.rules.polycyclic import VonBaeyerAnalyzer

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# The algorithm as it was before part A (verbatim; the only edits are the module
# prefix on the two helpers it calls and this function's name and docstring).
# ---------------------------------------------------------------------------

def _reference_find_main_ring_impl(
    self,
    mol,
    ring_atoms: Set[int],
    bridgeheads: Set[int],
    return_candidates: bool = False,
) -> Tuple[List[int], Tuple[int, int]]:
    """The main-ring search exactly as it was before UN L7b part A (kept only here, as the reference)."""
    # Build adjacency
    adj = {}
    for idx in ring_atoms:
        atom = mol.GetAtomWithIdx(idx)
        adj[idx] = set(n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() in ring_atoms)

    adj_list = {idx: [n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors()]
                for idx in ring_atoms}          # R2(c): GetNeighbors order, built once

    # Spelling-invariant tie-break key: RDKit canonical atom ranks are
    # identical for every SMILES spelling of the same molecule. breakTies=
    # True forces a TOTAL order so that even symmetry-equivalent atoms in a
    # cage (homocubane etc.) get a stable, reproducible ordering -- without
    # it the path-enumeration (GetNeighbors) order would break rank ties
    # non-deterministically. This makes the main-ring selection fully
    # order-independent (determinism).
    try:
        _ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=True))
    except Exception:
        _ranks = list(range(mol.GetNumAtoms()))

    def _rank_key(atom_indices):
        return tuple(sorted(_ranks[i] for i in atom_indices))

    best_ring = None
    best_bh_pair = None
    # Score tuple, compared with ">" (higher wins). Faithful to the
    # -> -> cascade:
    # (ring_size, #: main ring includes max skeletal atoms
    # main_bridge_len, #: main bridge as large as possible
    # balance, #: main ring divided as symmetrically as possible
    # tie_key) # deterministic canon-rank tie-break (lowest ranks win)
    best_score = None
    # Every decomposition tied at the top of the STRUCTURAL part of the score
    # (ring_size, main_bridge_len, balance). These all yield the same von
    # Baeyer descriptor, so does not distinguish them -- the choice
    # between them is exactly the "choice for numbering" that and
    # govern. Retained so ``_choose_lowest_locant_numbering`` can rank
    # them on locants instead of on the arbitrary canonical-rank backstop.
    tied_candidates: List[Tuple[List[int], Tuple[int, int]]] = []

    def _consider(ring, bh1, bh2, branch1_len, branch2_len, main_bridge_len):
        nonlocal best_score, best_ring, best_bh_pair
        balance = min(branch1_len, branch2_len)
        tie_key = tuple(-r for r in _rank_key(ring))
        structural = (len(ring), main_bridge_len, balance)
        score = (structural, tie_key)
        if best_score is None or structural > best_score[0]:
            tied_candidates.clear()
        if best_score is None or structural >= best_score[0]:
            tied_candidates.append((ring, (bh1, bh2)))
        if best_score is None or score > best_score:
            best_score = score
            best_ring = ring
            best_bh_pair = (bh1, bh2)

    def _best_disjoint_pair(allowed, exclude_direct):
        """Largest, then most symmetric, pair of interior-disjoint bh1->bh2
        paths through ``allowed``. ``exclude_direct`` drops the trivial
        direct edge so a 0-atom main bridge keeps the rest as the main ring.
        Deterministic: ties broken by the lowest canon-rank tuple.
        """
        all_paths = pc._find_all_simple_paths(mol, bh1, bh2, allowed, adj=adj_list)
        if exclude_direct:
            all_paths = [p for p in all_paths if len(p) >= 3]
        if len(all_paths) < 2:
            return None
        best_local = None
        best_pair = None
        for i, p1 in enumerate(all_paths):
            p1_interior = set(p1[1:-1])
            for p2 in all_paths[i + 1:]:
                pc.spend_perf_work()  # M2.5: charge the O(paths^2) pairing loop
                p2_interior = set(p2[1:-1])
                if p1_interior & p2_interior:
                    continue
                ring_size = len(p1) + len(p2) - 2  # shared endpoints
                balance = min(len(p1) - 2, len(p2) - 2)
                key = (ring_size, balance,
                       tuple(-r for r in _rank_key(set(p1) | set(p2))))
                if best_local is None or key > best_local:
                    best_local = key
                    best_pair = (p1, p2)
        return best_pair

    for bh1, bh2 in combinations(sorted(bridgeheads), 2):
        # Find common neighbors (potential 1-atom main bridge atoms)
        common_neighbors = adj.get(bh1, set()) & adj.get(bh2, set())

        # Also consider direct connection (0-atom main bridge)
        has_direct_bond = bh2 in adj.get(bh1, set())

        # Case 1: Direct bond between bridgeheads (0-atom main bridge).
        # The main ring is the largest pair of interior-disjoint paths that
        # do NOT use the direct edge; the direct bond is the main bridge.
        # (Previously this greedily grabbed a single longest path, which on
        # a dense cage consumed every atom and left no second branch -- so a
        # 0-atom-bridge main ring was never even generated, e.g. prismane.)
        if has_direct_bond:
            pair = _best_disjoint_pair(ring_atoms, exclude_direct=True)
            if pair:
                p1, p2 = pair
                ring = p1 + p2[1:-1][::-1]
                _consider(ring, bh1, bh2, len(p1) - 2, len(p2) - 2, 0)

        # Case 2: 1-atom main bridge via a common neighbor.
        for bridge_atom in sorted(common_neighbors):
            pair = _best_disjoint_pair(ring_atoms - {bridge_atom},
                                       exclude_direct=False)
            if pair:
                p1, p2 = pair
                ring = p1 + p2[1:-1][::-1]
                _consider(ring, bh1, bh2, len(p1) - 2, len(p2) - 2, 1)

        # Case 3 /: a 2+-atom main bridge. Cases 1
        # and 2 only ever offer a 0- or 1-atom main bridge, so a bridgehead
        # pair joined to the rest of the cage by a longer third path never had
        # its true, larger main bridge scored -- the ``main_bridge_len`` slot
        # of the score tuple could hold only 0 or 1, and Blue Book PINs such
        # as tricyclo[9.3.3.1^1,11] (the Blue Book) came out with a
        # non-preferred main bicycle. The three primary segments of the main
        # bicycle are three mutually interior-disjoint bh1->bh2 paths: the two
        # longest form the main ring, largest ring), and the longest
        # of the remaining disjoint paths is the main bridge, largest
        # bridge). The (ring_size, main_bridge_len, balance) score then ranks
        # this faithfully against Cases 1/2, and among equal main bridges
        # prefers the more symmetric main-ring division. Cases
        # 1/2 are kept: this is a strict ADD, skipped whenever the largest
        # available main bridge is only 0 or 1 atom.
        #
        # Case 3 is preference-only and can, on a cage whose secondary
        # bridges cannot be numbered consistently under the new main bicycle,
        # yield a descriptor that fails the legality audit. ``analyze`` runs
        # the cascade with this enabled first and, only if the result is
        # illegal, re-runs it disabled -- so the preferred main bridge can
        # never COST a legal (if non-preferred) name (accurate-or-degrade,
        # never a regression).
        if not getattr(self, "_allow_multiatom_bridge", True):
            continue
        all_paths = pc._find_all_simple_paths(mol, bh1, bh2, ring_atoms, adj=adj_list)
        interiors = [set(p[1:-1]) for p in all_paths]
        npaths = len(all_paths)
        for i in range(npaths):
            p1 = all_paths[i]
            int_i = interiors[i]
            for j in range(i + 1, npaths):
                pc.spend_perf_work()  # M2.5: charge the O(paths^2) pairing loop
                int_j = interiors[j]
                if int_i & int_j:
                    continue
                used = int_i | int_j
                # Largest remaining interior-disjoint path is the main bridge.
                best_mb_len = -1
                for k in range(npaths):
                    if k == i or k == j or interiors[k] & used:
                        continue
                    if len(interiors[k]) > best_mb_len:
                        best_mb_len = len(interiors[k])
                if best_mb_len < 2:
                    # 0/1-atom main bridges are already covered by Cases 1/2.
                    continue
                p2 = all_paths[j]
                ring = p1 + p2[1:-1][::-1]
                _consider(ring, bh1, bh2, len(p1) - 2, len(p2) - 2, best_mb_len)

    # Fallback: use largest ring method (for bicyclo or unusual cases)
    if best_ring is None:
        best_ring, best_bh_pair = self._find_main_ring_fallback(
            mol, ring_atoms, bridgeheads
        )

    # Reorder main ring so that bh1 is first and the longer path to bh2 comes first
    if best_ring and best_bh_pair:
        best_ring = self._orient_main_ring(best_ring, best_bh_pair)

    if return_candidates:
        oriented = []
        seen_c = set()
        for ring, bhp in tied_candidates:
            ring = self._orient_main_ring(ring, bhp)
            k = (tuple(ring), bhp)
            if k not in seen_c:
                seen_c.add(k)
                oriented.append((ring, bhp))
        return best_ring, best_bh_pair, oriented

    return best_ring, best_bh_pair


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def units(monkeypatch):
    """Counts the perf-budget units charged inside ``polycyclic`` (DFS expansions and pair loops).

    The counter replaces ``spend_perf_work`` in the module namespace; both implementations look it
    up there at call time. It never touches the real budget."""
    spent = [0]

    def counting(n=1):
        spent[0] += n

    monkeypatch.setattr(pc, "spend_perf_work", counting)
    return spent


def _ring_systems(an, mol):
    """``(ring_atoms, bridgeheads)`` of every ring system of ``mol`` with two or more bridgeheads."""
    comps: List[Set[int]] = []
    for ring in mol.GetRingInfo().AtomRings():
        merged = set(ring)
        for comp in [c for c in comps if c & merged]:
            comps.remove(comp)
            merged |= comp
        comps.append(merged)
    systems = []
    for comp in sorted(comps, key=min):
        bridgeheads = an._find_all_bridgeheads(mol, comp)
        if len(bridgeheads) >= 2:
            systems.append((comp, bridgeheads))
    return systems


def _same_on(mol, ring_atoms, bridgeheads, units, allow_multiatom_bridge=True, return_candidates=True):
    """Run the reference and the new search on one system; assert equal result and units not above the reference's
    (equal when the main-bridge preference, hence Case 3, is off)."""
    an = VonBaeyerAnalyzer()
    an._allow_multiatom_bridge = allow_multiatom_bridge
    units[0] = 0
    ref = _reference_find_main_ring_impl(an, mol, ring_atoms, bridgeheads, return_candidates=return_candidates)
    ref_units = units[0]
    units[0] = 0
    new = an._find_main_ring_impl(mol, ring_atoms, bridgeheads, return_candidates=return_candidates)
    new_units = units[0]
    assert new == ref
    if allow_multiatom_bridge:
        # Case 3 is pruned (UN lane P): one unit per visited pair, never more than the reference
        assert new_units <= ref_units
    else:
        assert new_units == ref_units        # Case 3 does not run: Cases 1/2 and the DFS are as they were
    return ref, ref_units


# ---------------------------------------------------------------------------
# (a) identity: von Baeyer molecules of the Blue Book gold set and a few cages
# ---------------------------------------------------------------------------

GOLD_VON_BAEYER = [
    'O=C1CC2CCC1C2',
    'O=C(O)C1CC2CCC1C2',
    'N#CC1CC2CCC1C2',
    'O=CC1CC2CCC1C2',
    'NC1C2CC3CC(C2)CC1C3',
    'CC1(C)C2CCC1(C)C(=O)C2',
    'C1CC2C=CC1C2',
    'CN1[C@@H]2CC[C@H]1C[C@H](C2)OC(=O)C(CO)c1ccccc1',
    'OC(=O)CCC1CC2CCC1C=C2',
    'C1CN2CCC1CC2',
    'C1CC2CCC1CN2',
    'C12C[SH2]CC(CC1)C2',
    'C12C[SeH2]CC(CC1)C2',
    'C1CC2CCC1C2',
    'O=C(O)CCC1CC2CCC1C2',
    '[SiH2]1[SiH2][SiH]2[SiH2][SiH2][SiH]1[SiH2]2',
    '[GeH2]1[GeH2][GeH]2[GeH2][GeH2][GeH]1[GeH2]2',
    '[SnH2]1[SnH2][SnH]2[SnH2][SnH2][SnH]1[SnH2]2',
    '[PbH2]1[PbH2][PbH]2[PbH2][PbH2][PbH]1[PbH2]2',
    'B1BB2BBB1B2',
    '[AsH]1[AsH][As]2[AsH][AsH][As]1[AsH]2',
    '[SbH]1[SbH][Sb]2[SbH][SbH][Sb]1[SbH]2',
    '[BiH]1[BiH][Bi]2[BiH][BiH][Bi]1[BiH]2',
    '[Te]1[Te][TeH]2[Te][Te][TeH]1[Te]2',
    'C12C3C1C1C2C31',
    'C1C2CC3C1C3C2',
    'O1CC2CCC1CC2',
    'C12COCC(CC1)C2',
    'O1CC2CCC1O2',
    'N1CC2CCCC1C2',
    'C1C2CC3CC1C23',
    'O1[SiH2]O[SiH]2O[SiH2]O[SiH]1O2',
    'C123CCCCCCCCCC(CCC1)(CCC2)C3',
    'C12C3CCC(C(C4CCC1CC4)CC2)CC3',
    'C12CC34CCCC(CC(CCCC1)CCC2)(CC3)C4',
    'C12C3C4CCC(CC(CCC1)CCC2)(CC4)C3',
    'C12C3C4CCC(CC(CCC1)CCC2)(C4)CC3',
    'C12C3C4CCCC5CCCC(C(C6CCCC(CCC1)CC26)C3)C4C5',
    'C12CC3CCCCC4CCCCC(CC(CCC5CC5CC1)CCCC2)CC(C4)C3',
    'C1C2CC3CC1CC(C2)C3',
    'C12C3C4C1C5C2C3C45',
    'C1CC2CCC1CC2',
    'C12C3C4C1C1C2C3C41',
    'C1CC2CC3CC2CC13',
    'C1C2C3CC4C1C2C34',
]


@pytest.mark.parametrize("smiles", GOLD_VON_BAEYER)
def test_main_ring_search_is_identical_on_gold_von_baeyer_molecules(smiles, units):
    """/.4/.6.2.1: same main ring, bridgeheads, oriented tied candidates and units.

    Both settings of the main-bridge preference (``_allow_multiatom_bridge``: ``analyze`` re-runs the
    cascade with it off when the preferred main bicycle cannot be numbered) and both return shapes."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    an = VonBaeyerAnalyzer()
    systems = _ring_systems(an, mol)
    assert systems, smiles
    for ring_atoms, bridgeheads in systems:
        (_, _, tied), _ = _same_on(mol, ring_atoms, bridgeheads, units, True, True)
        assert tied
        # Preference off: a bicyclo[x.y.z] with no 0/1-atom main bridge has no tied candidate and
        # takes the fallback; the two implementations must still agree.
        _same_on(mol, ring_atoms, bridgeheads, units, False, True)
        _same_on(mol, ring_atoms, bridgeheads, units, True, False)


def _random_cage(rng):
    """A random connected skeleton (spanning tree plus 2-5 extra edges, degree <= 4), with
    heteroatoms on low-degree atoms so that canonical-rank ties and non-ties both occur."""
    n = rng.randint(8, 13)
    edges = set()
    degree = [0] * n
    for v in range(1, n):
        choices = [u for u in range(v) if degree[u] < 4]
        u = rng.choice(choices)
        edges.add((u, v))
        degree[u] += 1
        degree[v] += 1
    extra = rng.randint(2, 5)
    for _ in range(60):
        if extra == 0:
            break
        u, v = sorted(rng.sample(range(n), 2))
        if (u, v) in edges or degree[u] >= 4 or degree[v] >= 4:
            continue
        edges.add((u, v))
        degree[u] += 1
        degree[v] += 1
        extra -= 1
    rw = Chem.RWMol()
    for v in range(n):
        symbols = ["C", "N", "O", "S"] if degree[v] <= 2 else (["C", "N"] if degree[v] == 3 else ["C"])
        rw.AddAtom(Chem.Atom(rng.choice(symbols)))
    for u, v in sorted(edges):
        rw.AddBond(u, v, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return mol


@pytest.mark.parametrize("seed", range(80))
def test_main_ring_search_is_identical_on_random_cage_skeletons(seed, units):
    """Random polycyclic skeletons: equal results (ties and their order) and equal units."""
    mol = _random_cage(random.Random(seed))
    an = VonBaeyerAnalyzer()
    for ring_atoms, bridgeheads in _ring_systems(an, mol):
        for allow in (True, False):
            _same_on(mol, ring_atoms, bridgeheads, units, allow, True)


# ---------------------------------------------------------------------------
# (a) identity of the two pair loops on random path sets (what the cages above rarely vary:
# several disjoint main-bridge candidates of different lengths, ties of every kind)
# ---------------------------------------------------------------------------

class _ReferenceBest:
    """The incumbent bookkeeping as it was: the rank key is computed for every candidate."""

    def __init__(self, rank_key):
        self.rank_key = rank_key
        self.score = None
        self.ring = None
        self.bh_pair = None
        self.tied = []

    def consider(self, ring, bh1, bh2, branch1_len, branch2_len, main_bridge_len):
        balance = min(branch1_len, branch2_len)
        tie_key = tuple(-r for r in self.rank_key(ring))
        structural = (len(ring), main_bridge_len, balance)
        score = (structural, tie_key)
        if self.score is None or structural > self.score[0]:
            self.tied.clear()
        if self.score is None or structural >= self.score[0]:
            self.tied.append((ring, (bh1, bh2)))
        if self.score is None or score > self.score:
            self.score = score
            self.ring = ring
            self.bh_pair = (bh1, bh2)


def _reference_case3(all_paths, spend, best, bh1, bh2):
    """Case 3 of the main-ring search, the triple loop as it was."""
    interiors = [set(p[1:-1]) for p in all_paths]
    npaths = len(all_paths)
    for i in range(npaths):
        p1 = all_paths[i]
        int_i = interiors[i]
        for j in range(i + 1, npaths):
            spend()
            int_j = interiors[j]
            if int_i & int_j:
                continue
            used = int_i | int_j
            best_mb_len = -1
            for k in range(npaths):
                if k == i or k == j or interiors[k] & used:
                    continue
                if len(interiors[k]) > best_mb_len:
                    best_mb_len = len(interiors[k])
            if best_mb_len < 2:
                continue
            p2 = all_paths[j]
            ring = p1 + p2[1:-1][::-1]
            best.consider(ring, bh1, bh2, len(p1) - 2, len(p2) - 2, best_mb_len)


def _reference_best_disjoint_pair(all_paths, spend, rank_key):
    """The pair search of Cases 1 and 2, as it was."""
    best_local = None
    best_pair = None
    for i, p1 in enumerate(all_paths):
        p1_interior = set(p1[1:-1])
        for p2 in all_paths[i + 1:]:
            spend()
            p2_interior = set(p2[1:-1])
            if p1_interior & p2_interior:
                continue
            ring_size = len(p1) + len(p2) - 2
            balance = min(len(p1) - 2, len(p2) - 2)
            key = (ring_size, balance, tuple(-r for r in rank_key(set(p1) | set(p2))))
            if best_local is None or key > best_local:
                best_local = key
                best_pair = (p1, p2)
    return best_pair


def _random_paths(rng):
    """Random bh1(0)->bh2(1) paths with interiors drawn from a small universe, so that interiors
    overlap, repeat and have equal sizes; plus a random total rank order over the atoms."""
    universe = rng.randint(6, 13)
    paths = []
    for _ in range(rng.randint(2, 15)):
        interior = rng.sample(range(2, universe), rng.randint(0, min(6, universe - 2)))
        paths.append([0] + interior + [1])
    ranks = list(range(16))                       # also covers the seeded incumbent rings below
    rng.shuffle(ranks)
    return paths, (lambda atoms: tuple(sorted(ranks[a] for a in atoms)))


@pytest.mark.parametrize("seed", range(300))
def test_case3_pair_scan_equals_the_triple_loop_on_random_path_sets(seed):
    """Same incumbent, same tied candidates in the same order; units: the triple loop charges one per
    (i, j) pair, the pruned scan one per visited pair (never more). ``room`` is the number of distinct
    interior atoms of the path set, the tightest bound a real call can have (its ring atoms outside
    the bridgeheads are a superset)."""
    rng = random.Random(seed)
    paths, rank_key = _random_paths(rng)
    ref_best, new_best = _ReferenceBest(rank_key), pc._MainRingBest(rank_key)
    for _ in range(rng.randint(0, 2)):                    # an incumbent from Cases 1/2
        ring = list(range(rng.randint(4, 12)))
        args = (ring, 0, 1, rng.randint(0, 4), rng.randint(0, 4), rng.randint(0, 3))
        ref_best.consider(*args)
        new_best.consider(*args)
    spent = {"ref": 0, "new": 0}
    _reference_case3([list(p) for p in paths], lambda: spent.__setitem__("ref", spent["ref"] + 1), ref_best, 0, 1)
    room = len({a for p in paths for a in p[1:-1]})
    pc._consider_multiatom_bridges([list(p) for p in paths], lambda: spent.__setitem__("new", spent["new"] + 1),
                                   new_best, 0, 1, room)
    assert spent["ref"] == len(paths) * (len(paths) - 1) // 2
    assert spent["new"] <= spent["ref"]
    assert (new_best.score, new_best.ring, new_best.bh_pair, new_best.tied) == \
        (ref_best.score, ref_best.ring, ref_best.bh_pair, ref_best.tied)


@pytest.mark.parametrize("seed", range(300))
def test_best_disjoint_pair_equals_the_pair_loop_on_random_path_sets(seed):
    rng = random.Random(10_000 + seed)
    paths, rank_key = _random_paths(rng)
    spent = {"ref": 0, "new": 0}
    ref = _reference_best_disjoint_pair([list(p) for p in paths],
                                        lambda: spent.__setitem__("ref", spent["ref"] + 1), rank_key)
    new = pc._best_disjoint_path_pair([list(p) for p in paths],
                                      lambda: spent.__setitem__("new", spent["new"] + 1), rank_key)
    assert new == ref
    assert spent["new"] == spent["ref"] == len(paths) * (len(paths) - 1) // 2


# ---------------------------------------------------------------------------
# (a) identity: the 20 UN L7b cluster-A rows (PubChem CIDs; whole-molecule ring systems)
# ---------------------------------------------------------------------------

CLUSTER_A = {
    '157508500': 'C1=CC=C2C(=C1)C3=C4N2C5=C6C7=C(C=C5)C8=CC=CC=C8N7C9=CC=CC=[N+]9C61C4=C(C=C3)OC2=CC=CC=[N+]12',
    '11815786': 'COC[C@@H]1[C@@H]2[C@@H]([C@H]([C@H](O1)O[C@@H]3[C@H](O[C@@H]([C@@H]([C@H]3O)OC)O[C@@H]4[C@H](O[C@@H]([C@@H]([C@H]4O)OC)O[C@@H]5[C@H](O[C@@H]([C@@H]([C@H]5O)OC)O[C@@H]6[C@H](O[C@@H]([C@@H]([C@H]6O)OC)O[C@@H]7[C@H](O[C@@H]([C@@H]([C@H]7O)OC)O[C@@H]8[C@H](O[C@@H]([C@@H]([C@H]8O)OC)O[C@@H]9[C@H](O[C@H](O2)[C@@H]([C@H]9O)OC)COC)COC)COC)COC)COC)COC)COC)OC)O',
    '140889996': 'C(C1[C@H]([C@H]2C([C@@H](O1)O[C@H]3[C@@H](C([C@@H](O[C@H]4[C@@H](C([C@@H](O[C@H]5[C@@H](C([C@@H](O[C@H]6[C@@H](C([C@@H](O[C@H]7[C@@H](C([C@H](O[C@H]8[C@@H](C([C@@H](O[C@H]9[C@@H](C([C@@H](O2)OC9CO)O)O)OC8CO)O)O)OC7CO)O)O)OC6CO)O)O)OC5CO)O)O)OC4CO)O)O)OC3CO)O)O)O)O)O',
    '71546558': 'CN1C=C2C=C1C(=O)NC[C@@H](CC(=O)NC3=CN(C(=N3)C(=O)NC4=CN(C(=C4)C(=O)NC5=CN(C(=C5)C(=O)NC6=CN(C(=C6)C(=O)NC[C@@H](CC(=O)NC7=CN(C(=N7)C(=O)NC8=CN(C(=C8)C(=O)NC9=CN(C(=C9)C(=O)N2)C)C)C)NC(=O)C1=CC=CC=C1)C)C)C)C)N',
    '10307808': 'COC[C@@H]1[C@@H]2[C@@H]([C@H]([C@H](O1)O[C@@H]3[C@H](O[C@@H]([C@@H]([C@H]3OC)OC)O[C@@H]4[C@H](O[C@@H]([C@@H]([C@H]4OC)OC)O[C@@H]5[C@H](O[C@@H]([C@@H]([C@H]5OC)OC)O[C@@H]6[C@H](O[C@@H]([C@@H]([C@H]6OC)OC)O[C@@H]7[C@H](O[C@@H]([C@@H]([C@H]7OC)OC)O[C@@H]8[C@H](O[C@@H]([C@@H]([C@H]8OC)OC)O[C@@H]9[C@H](O[C@H](O2)[C@@H]([C@H]9OC)OC)COC)COC)COC)COC)COC)COC)COC)OC)OC',
    '136159023': 'CC1=CC2=C(C(=C1)C=NC3CCCC(C3)NC(=O)C4=C5C(=CC(=C4)C)C(=O)NC6CCCCC6N=CC7=CC(=CC(=C7O)C=NC8CCCC(C8)NC(=O)C9=CC(=CC(=C9OCCOCCOCCO5)C(=O)NC1CCCC(C1)N=C2)C)C)O',
    '156664006': 'C([C@@H]1[C@@H]2[C@@H]([C@H](C(O1)O[C@@H]3[C@H](OC([C@@H]([C@H]3O)O)O[C@@H]4[C@H](OC([C@@H]([C@H]4O)O)O[C@@H]5[C@H](OC([C@@H](C5O)O)O[C@@H]6[C@H](OC([C@@H]([C@H]6O)O)OC7[C@H](OC([C@@H]([C@H]7O)O)O[C@@H]8[C@H](OC([C@@H]([C@H]8O)O)O[C@@H]9[C@H](OC(O2)[C@@H]([C@H]9O)O)CO)CO)CO)CO)CO)CO)CO)O)O)O',
    '161612552': 'CCOC(=O)OC1=CC=C(C=C1)C[C@H]2C(=O)N3C[C@@H]([C@@H]([C@H]3C4=NC(=CS4)C5=NC(=CS5)C6=C(C=CC(=N6)C7=NC(=CS7)NC(=O)OC8CCC(CC8)C(=O)OC)C9=NC(=CS9)C(=O)N[C@H](C1=NC(=C(S1)C)C(=O)C[C@H](C1=NC(=CS1)C(=O)N2)[C@@H](C1=CC=CC=C1)O)CC(=O)N)C)O',
    '143830930': 'CC(C)C1C2=NC3=C(O2)C45C(NC6=C(C=CC=C64)C7=C8C(=CC=C7)N(C=C8C9=CN=C3O9)P(=O)(OCC2=CC=CC=C2)OCC2=CC=CC=C2)OC2=C5C=C(C[C@@H](C(=O)N1)NC(=O)[C@H](C(C)C)O)C=C2',
    '17757540': 'CC(C)(C)OC(=O)NC(CCCN)C(=O)NC[C@@H]1[C@@H]2[C@@H]([C@H]([C@H](O1)O[C@@H]3[C@H](O[C@@H]([C@@H]([C@H]3O)O)O[C@@H]4[C@H](O[C@@H]([C@@H]([C@H]4O)O)O[C@@H]5[C@H](O[C@H]6[C@@H]([C@H]5O)O[C@@H]7[C@H]([C@H](O[C@@H]([C@H]7O6)CNC(=O)C(CCCN)NC(=O)OC(C)(C)C)O[C@@H]8[C@H](O[C@@H]([C@@H]([C@H]8O)O)O[C@@H]9[C@H](O[C@H](O2)[C@@H]([C@H]9O)O)CNC(=O)C(CCCN)NC(=O)OC(C)(C)C)CNC(=O)C(CCCN)NC(=O)OC(C)(C)C)O)CNC(=O)C(CCCN)NC(=O)OC(C)(C)C)CNC(=O)C(CCCN)NC(=O)OC(C)(C)C)CNC(=O)C(CCCN)NC(=O)OC(C)(C)C)O)O',
    '152882583': 'C1=COC(=C1)COC(=O)CC[Si]23O[Si]4(O[Si]5(O[Si](O2)(O[Si]6(O[Si](O3)(O[Si](O4)(O[Si](O5)(O6)CCC(=O)OCC7=CC=CO7)CCC(=O)OCC8=CC=CO8)CCC(=O)OCC9=CC=CO9)CCC(=O)OCC1=CC=CO1)CCC(=O)OCC1=CC=CO1)CCC(=O)OCC1=CC=CO1)CCC(=O)OCC1=CC=CO1',
    '5282405': 'C=CC[N@@+]12[C@@H]3[C@]4(C5=CC=CC=C5N/6C4/C(=C\\N7C8/C(=C6)/[C@@H]9/C(=C\\CO)/C[N@+]4([C@H]([C@]8(C5=CC=CC=C75)CC4)C9)CC=C)/[C@@H](C3)/C(=C\\CO)/C1)CC2',
    '2733545': 'CC(COCC1[C@@H]2[C@@H](C([C@H](O1)O[C@H]3[C@@H](C([C@@H](O[C@H]4[C@@H](C([C@@H](O[C@H]5[C@@H](C([C@@H](O[C@H]6[C@@H](C([C@@H](O[C@H]7[C@@H](C([C@@H](O[C@H]8[C@@H](C([C@@H](O[C@H]9[C@@H](C([C@@H](O2)OC9CO)O)O)OC8CO)O)O)OC7COCC(C)O)O)O)OC6COCC(C)O)O)O)OC5CO)O)O)OC4COCC(C)O)O)O)OC3COCC(C)O)O)O)O)O)O',
    '168434342': 'C(CSC[C@@H]1[C@@H]2[C@@H]([C@H]([C@H](O1)O[C@@H]3[C@H](O[C@@H]([C@@H]([C@H]3O)O)O[C@@H]4[C@H](O[C@@H]([C@@H]([C@H]4O)O)O[C@@H]5[C@H](O[C@@H]([C@@H]([C@H]5O)O)O[C@@H]6[C@H](O[C@@H]([C@@H]([C@H]6O)O)O[C@@H]7[C@H](O[C@@H]([C@@H]([C@H]7O)O)O[C@@H]8[C@H](O[C@@H]([C@@H]([C@H]8O)O)O[C@@H]9[C@H](O[C@H](O2)[C@@H]([C@H]9O)O)CSCCC(=O)O)CSCCC(=O)O)CSCCC(=O)O)CBr)CSCCC(=O)O)CSCCC(=O)O)CSCCC(=O)O)O)O)C(=O)O',
    '136010150': 'CC1=C(C2=NC1=CC3=C(C(=C(N3)C4=C(C(=C(N4)C5=C(C(=C(N5)C6=NC(=CC7=C(C(=C(N7)C8=C(C(=C(N8)C9=C(C(=C2N9)C)C)C)C)C)C)C(=C6C)C)C)C)C)C)C)C)C',
    '124082853': 'C([C@@H]1[C@@H]2[C@@H]([C@H](C(O1)O[C@@H]3[C@H](OC([C@@H]([C@H]3O)O)OC4[C@H](OC([C@@H]([C@H]4O)O)OC5[C@H](OC([C@@H]([C@H]5O)O)OC6[C@H](OC([C@@H]([C@H]6O)O)OC7[C@H](OC([C@@H]([C@H]7O)O)OC8[C@H](OC([C@@H]([C@H]8O)O)O[C@@H]9[C@H](OC(O2)[C@@H]([C@H]9O)O)CO)CO)CO)CO)CO)CO)CO)O)O)O',
    '14376902': 'CC(=O)OC1C2C(OC(C1OC(=O)C)OC3C(OC(C(C3OC(=O)C)OC(=O)C)OC4C(OC(C(C4OC(=O)C)OC(=O)C)OC5C(OC(C(C5OC(=O)C)OC(=O)C)OC6C(OC(C(C6OC(=O)C)OC(=O)C)OC7C(OC(C(C7OC(=O)C)OC(=O)C)OC8C(OC(C(C8OC(=O)C)OC(=O)C)OC9C(OC(O2)C(C9OC(=O)C)OC(=O)C)CO[Si](C)(C)C(C)(C)C)CO[Si](C)(C)C(C)(C)C)CO[Si](C)(C)C(C)(C)C)CO[Si](C)(C)C(C)(C)C)CO[Si](C)(C)C(C)(C)C)CO[Si](C)(C)C(C)(C)C)CO[Si](C)(C)C(C)(C)C)CO[Si](C)(C)C(C)(C)C',
    '72998600': 'C(C1C2C(C(C(O1)OC3C(OC(C(C3O)O)OC4C(OC(C(C4O)O)OC5C(OC(C(C5O)O)OC6C(OC(C(C6O)O)OC7C(OC(C(C7O)O)OC8C(OC(C(C8O)O)OC9C(OC(O2)C(C9O)O)CO)CO)CO)CO)CO)CO)CO)O)N)O',
    '175674727': 'C(CSC[C@@H]1[C@@H]2[C@@H]([C@H]([C@H](O1)O[C@@H]3[C@H](O[C@@H]([C@@H]([C@H]3O)O)O[C@@H]4[C@H](O[C@@H]([C@@H]([C@H]4O)O)O[C@@H]5[C@H](O[C@@H]([C@@H]([C@H]5O)O)O[C@@H]6[C@H](O[C@@H]([C@@H]([C@H]6O)O)O[C@@H]7[C@H](O[C@@H]([C@@H]([C@H]7O)O)O[C@@H]8[C@H](O[C@@H]([C@@H]([C@H]8O)O)O[C@@H]9[C@H](O[C@H](O2)[C@@H]([C@H]9O)O)CSCCC(=O)O)CSCCC(=O)O)CSCCC(=O)O)CSSC[C@@H]1[C@@H]2[C@@H]([C@H]([C@H](O1)O[C@@H]1[C@H](O[C@@H]([C@@H]([C@H]1O)O)O[C@@H]1[C@H](O[C@@H]([C@@H]([C@H]1O)O)O[C@@H]1[C@H](O[C@@H]([C@@H]([C@H]1O)O)O[C@@H]1[C@H](O[C@@H]([C@@H]([C@H]1O)O)O[C@@H]1[C@H](O[C@@H]([C@@H]([C@H]1O)O)O[C@@H]1[C@H](O[C@@H]([C@@H]([C@H]1O)O)O[C@@H]1[C@H](O[C@H](O2)[C@@H]([C@H]1O)O)CSCCC(=O)O)CSCCC(=O)O)CSCCC(=O)O)CSCCC(=O)O)CSCCC(=O)O)CSCCC(=O)O)CSCCC(=O)O)O)O)CSCCC(=O)O)CSCCC(=O)O)CSCCC(=O)O)O)O)C(=O)O',
    '135850218': 'C(CC1=NNN=N1)CSC[C@@H]2[C@@H]3[C@@H]([C@H]([C@H](O2)O[C@@H]4[C@H](O[C@@H]([C@@H]([C@H]4O)O)O[C@@H]5[C@H](O[C@@H]([C@@H]([C@@H]5O)O)O[C@@H]6[C@@H](O[C@@H]([C@H]([C@H]6O)O)O[C@@H]7[C@@H](O[C@@H]([C@H]([C@H]7O)O)O[C@@H]8[C@H](O[C@@H]([C@@H]([C@H]8O)O)O[C@@H]9[C@H](O[C@@H]([C@H]([C@H]9O)O)O[C@@H]1[C@H](O[C@H](O3)[C@@H]([C@H]1O)O)CSCCCC1=NNN=N1)CSCCCC1=NNN=N1)CSCCCC1=NNN=N1)CSCCCC1=NNN=N1)CSCCCC1=NNN=N1)CSCCCC1=NNN=N1)CSCCCC1=NNN=N1)O)O',
}


@pytest.mark.parametrize("cid", list(CLUSTER_A))
def test_main_ring_search_is_identical_on_cluster_a_rows(cid, units):
    mol = Chem.MolFromSmiles(CLUSTER_A[cid])
    assert mol is not None
    an = VonBaeyerAnalyzer()
    systems = _ring_systems(an, mol)
    assert systems
    for ring_atoms, bridgeheads in systems:
        (ring, bh_pair, tied), ref_units = _same_on(mol, ring_atoms, bridgeheads, units)
        assert ring and bh_pair and tied
        assert ref_units > 1_000      # a search that really exercises the pair loops


# ---------------------------------------------------------------------------
# (a) the budget decides the abstentions at the units the search charges
# ---------------------------------------------------------------------------

def test_budget_trips_at_the_units_the_pruned_search_charges(monkeypatch):
    """With the real ``spend_perf_work`` armed, each search finishes with the same result when its own
    unit count fits the budget and raises ``PerfBudgetExceeded`` (the abstention) when it does not.
    The pruned search charges fewer units than the reference (UN lane P, Case 3), so a budget between
    the two counts is one the reference abstains at and the pruned search finishes inside, with the
    reference's result."""
    smiles = "C12C3C4CCC(CC(CCC1)CCC2)(CC4)C3"           # tetracyclo[6.3.3.2^3,6.1^2,6]heptadecane
    mol = Chem.MolFromSmiles(smiles)
    an = VonBaeyerAnalyzer()
    (ring_atoms, bridgeheads), = _ring_systems(an, mol)
    spent = [0]
    monkeypatch.setattr(pc, "spend_perf_work", lambda n=1: spent.__setitem__(0, spent[0] + n))
    _reference_find_main_ring_impl(an, mol, ring_atoms, bridgeheads, return_candidates=True)
    total_ref = spent[0]
    spent[0] = 0
    an._find_main_ring_impl(mol, ring_atoms, bridgeheads, return_candidates=True)
    total_new = spent[0]
    assert 100 < total_new < total_ref
    monkeypatch.undo()

    def run(impl, budget):
        fn._fragment_guard.perf_budget = budget
        try:
            result = impl(mol, ring_atoms, bridgeheads, return_candidates=True)
            return result, fn._fragment_guard.perf_budget
        except fn.PerfBudgetExceeded:
            return "exceeded", fn._fragment_guard.perf_budget
        finally:
            fn._fragment_guard.perf_budget = None

    def reference(*args, **kwargs):
        return _reference_find_main_ring_impl(an, *args, **kwargs)

    old_result, old_left = run(reference, total_ref + 1)
    assert old_left == 1 and old_result != "exceeded"
    assert run(reference, total_ref)[0] == "exceeded"
    new_result, new_left = run(an._find_main_ring_impl, total_new + 1)
    assert new_left == 1 and new_result == old_result
    assert run(an._find_main_ring_impl, total_new)[0] == "exceeded"
    for budget in (total_new // 2, 1):
        assert run(an._find_main_ring_impl, budget)[0] == "exceeded"
        assert run(reference, budget)[0] == "exceeded"
    # between the two counts: the reference abstains, the pruned search names
    assert run(reference, total_new + 1)[0] == "exceeded"
    assert run(an._find_main_ring_impl, total_new + 1)[0] == old_result


# ---------------------------------------------------------------------------
# (b) the full name of three cluster-A rows
# ---------------------------------------------------------------------------

# cid: (SMILES, the name before part A). 143830930 is the re-check's name (rows74.tsv column 6) except for
# its phosphorus substituent (below); for 157508500 and 136159023 the von Baeyer numbering of the re-check
# was superseded by earlier engine changes, so the name is the one the engine gave at the base commit of
# this lane.
#
# 143830930, substituent of the ring system: "1-(benzyloxy)-1-oxo-3-phenyl-2-oxa-1-phosphapropyl" until UN
# lane B2, "bis(benzyloxy)phosphoryl" since. The first was the engine's name only because the main-ring search
# of the cage (above) spent the whole 12M perf budget (19.3M units, one PerfBudgetExceeded) and the substituent
# was then named by the fallback; the search now charges 1.2M units and the budget does not trip.
# Measured (VERIFIED): the base commit names the second form with ORTHONYM_PERF_OP_BUDGET=120000000, and with
# ORTHONYM_PERF_BUDGET_MEASURE=1 (nothing trips); OPSIN reads both names back to the input's full standard
# InChIKey, so the choice is the preferred form's: "Skeletal replacement ('a') nomenclature in acyclic
# chains", (the Blue Book) "...must be used to generate preferred IUPAC names for acyclic
# structures when four or more heterounits are present in an unbranched chain" -- the chain here has two (O, P)
# -- and "Fundamental acyl groups for substituents derived from mononuclear noncarbon oxoacids"
# (the Blue Book: 'phosphoryl', -P(O)<, "preselected prefixes"; its example in a PIN,:36335,
# "3-[(dimethoxyphosphoryl)sulfanyl]propanoic acid"). The old name is not wrong, only not the preferred form.
PINNED_NAMES = {
    '157508500': (
        'C1=CC=C2C(=C1)C3=C4N2C5=C6C7=C(C=C5)C8=CC=CC=C8N7C9=CC=CC=[N+]9C61C4=C(C=C3)OC2=CC=CC=[N+]12',
        '30-oxa-2,8,19,36-tetraazaundecacyclo[17.17.2.2^15,18.0^1,17.0^2,7.0^8,16.0^9,14.0^20,25.0^26,38.0^29,37.0^31,36]tetraconta-2,4,6,9,11,13,15(40),16,18(39),20,22,24,26,28,31,33,35,37-octadecaene-2,36-diium',
    ),
    '136159023': (
        'CC1=CC2=C(C(=C1)C=NC3CCCC(C3)NC(=O)C4=C5C(=CC(=C4)C)C(=O)NC6CCCCC6N=CC7=CC(=CC(=C7O)C=NC8CCCC(C8)NC(=O)C9=CC(=CC(=C9OCCOCCOCCO5)C(=O)NC1CCCC(C1)N=C2)C)C)O',
        '71,73-dihydroxy-13,28,42,56-tetramethyl-2,25,31,53-tetraoxo-59,62,65,68-tetraoxa-3,9,17,24,32,38,46,52-octaazadecacyclo[28.27.12.1^4,8.1^11,15.1^33,37.1^40,44.1^47,51.0^18,23.0^26,69.0^54,58]tetraheptaconta-1(57),9,11(73),12,14,16,26,28,30(69),38,40,42,44(71),45,54(58),55-hexadecaene',
    ),
    '143830930': (
        'CC(C)C1C2=NC3=C(O2)C45C(NC6=C(C=CC=C64)C7=C8C(=CC=C7)N(C=C8C9=CN=C3O9)P(=O)(OCC2=CC=CC=C2)OCC2=CC=CC=C2)OC2=C5C=C(C[C@@H](C(=O)N1)NC(=O)[C@H](C(C)C)O)C=C2',
        '(13S)-34-[bis(benzyloxy)phosphoryl]-13-[(2S)-2-hydroxy-3-methylbutanamido]-12-oxo-10-(propan-2-yl)-8,37,40-trioxa-4,11,22,34,39-pentaazadecacyclo[27.6.1.1^2,5.1^6,9.1^15,19.1^18,21.0^7,20.0^20,24.0^23,28.0^33,36]tetraconta-1(35),2,4,6,9(39),15(38),16,18,23(28),24,26,29,31,33(36)-tetradecaene',
    ),
}


@pytest.mark.integration
@pytest.mark.opsin_gate
@pytest.mark.parametrize("cid", list(PINNED_NAMES))
def test_cluster_a_row_keeps_its_name(cid):
    from orthonym.namer import Orthonym
    from tests.support.jars import jar_or_skip

    jar_or_skip("opsin")
    smiles, expected = PINNED_NAMES[cid]
    namer = Orthonym(general_fallback=True, general_fallback_unverified=True, allow_aromatic_general=True)
    namer.name_tiered("CCO")      # as the run tool does: the JVM is up before the molecule is named
    result = namer.name_tiered(smiles)
    assert result["name"] == expected
    assert result["tier"] == "systematic_verified"
