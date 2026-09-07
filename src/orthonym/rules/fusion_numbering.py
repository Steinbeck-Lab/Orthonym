"""Deterministic peripheral numbering of fused polycyclic ring systems.

Given an oriented integer hex-lattice embedding (from ``fusion_orientation``),
this module assigns IUPAC locants to the peripheral skeletal atoms following
P-25.3.3.1: start from the uppermost ring (tie -> furthest right), begin at the
most-counterclockwise non-fusion atom, and walk CLOCKWISE assigning integers to
non-fusion atoms (and fusion heteroatoms), giving each fusion carbon the number
of the preceding non-fusion atom modified by a Roman letter ('a', 'b', ...).

The lowest-locant cascade P-25.3.3.1.2 is then applied across all surviving
(orientation, start) candidates using the shared comparison primitives in
``rules/locants.py`` (``compare_locant_sets`` / ``_compare_heteroatom_seniority``
via ``compare_numbering``) — the comparison logic is NEVER reimplemented here.

Fixed-numbering exceptions (P-25.1.1 / P-31.1.4.3.4)
----------------------------------------------------
``anthracene`` and ``phenanthrene`` are retained names with FIXED, traditional
numbering ("special numbering" — Blue Book P-25.1.1 entries (11) and (12); also
P-31.1.4.3.4: "in purine, anthracene, and phenanthrene, this numbering must be
used").  Their traditional numbers do NOT follow the systematic P-25.3.3
peripheral walk (anthracene numbers its meso carbons 9,10 last; phenanthrene
uses 4a,4b/8a,10a fusion labels).  A correct engine must encode these two
exceptions, so ``compute_fused_numbering`` recognises their exact ring graph and
returns the fixed numbering mapped onto the input atoms.  Every other all-6
cata-fused carbocyclic system (naphthalene, tetracene, pentacene, chrysene,
triphenylene, ...) is numbered systematically by the peripheral walk.

Stage 1 scope
-------------
``compute_fused_numbering`` returns a map ONLY for all-six-membered, ortho-
(cata-)fused, carbocyclic ring systems with no interior (peri-fusion) atom.
Otherwise it returns ``None`` and the caller falls back to its existing path.

Locant shape: ``int`` for ordinary atoms, ``(int, 'a')`` tuples for fusion
carbons — the existing ``rules.locants._Locant`` type, so downstream consumers
(``compare_locant_sets``, the cascade) are unchanged.

Source: IUPAC 2013 Blue Book P-25.3.3 (BlueBookV2/BlueBookV2.md ~line 12501);
        P-25.1.1 / P-31.1.4.3.4 (anthracene/phenanthrene fixed numbering).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from rdkit import Chem

from .fusion_orientation import (
    best_orientations,
    classify_ring_system,
)
from .locants import _Locant, compare_locant_sets, compare_numbering

# ---------------------------------------------------------------------------
# Fixed-numbering retained ring systems (P-25.1.1 / P-31.1.4.3.4).
#
# anthracene and phenanthrene have traditional "special" numbering that the
# systematic P-25.3.3 peripheral walk does NOT reproduce.  Each reference is a
# SMILES whose atoms are listed in molecule order with their FIXED IUPAC locant
# (string form), taken verbatim from OPSIN's extended-SMILES ``$_AV`` output
# (``opsin -o extendedsmi``) — the authoritative source already used to populate
# data/polycyclic_data.py.  The recognizer maps these onto an input ring system
# by substructure isomorphism.
# ---------------------------------------------------------------------------
_FIXED_NUMBERING_SYSTEMS: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    (
        'anthracene',
        'C1=CC=CC2=CC3=CC=CC=C3C=C12',
        ('1', '2', '3', '4', '4a', '10', '10a', '5', '6', '7', '8', '8a',
         '9', '9a'),
    ),
    (
        'phenanthrene',
        'C1=CC=CC=2C3=CC=CC=C3C=CC12',
        ('1', '2', '3', '4', '4a', '4b', '5', '6', '7', '8', '8a', '9',
         '10', '10a'),
    ),
    # ----- anthracene-type heterocycles with retained "special numbering" -----
    # Blue Book Table 2.8 lists acridine (entry 4, "special numbering") and the
    # xanthene chalcogen family (entry 22: xanthene/thioxanthene/selenoxanthene/
    # telluroxanthene, all "special numbering"; PIN of each is the 9H-isomer).
    # Like anthracene/phenanthrene their two central-ring meso atoms take the
    # HIGHEST locants 9 and 10, which the systematic P-25.3.3 peripheral walk
    # does NOT reproduce (the walk numbers a meso atom mid-sequence).  Reference
    # SMILES + locants are OPSIN ``-o extendedsmi $_AV`` output (authoritative).
    (
        'acridine',
        'C1=CC=CC2=NC3=CC=CC=C3C=C12',
        ('1', '2', '3', '4', '4a', '10', '10a', '5', '6', '7', '8', '8a',
         '9', '9a'),
    ),
    (
        '9H-xanthene',
        'C1=CC=CC=2OC3=CC=CC=C3CC12',
        ('1', '2', '3', '4', '4a', '10', '10a', '5', '6', '7', '8', '8a',
         '9', '9a'),
    ),
    (
        '9H-thioxanthene',
        'C1=CC=CC=2SC3=CC=CC=C3CC12',
        ('1', '2', '3', '4', '4a', '10', '10a', '5', '6', '7', '8', '8a',
         '9', '9a'),
    ),
    (
        '9H-selenoxanthene',
        'C1=CC=CC=2[Se]C3=CC=CC=C3CC12',
        ('1', '2', '3', '4', '4a', '10', '10a', '5', '6', '7', '8', '8a',
         '9', '9a'),
    ),
    # ----- (5,6)/(5,6,6) heterocycles with retained "special numbering" -----
    # Blue Book Table 2.8: purine (entry 16, "special numbering" — also listed
    # with anthracene/phenanthrene at P-14.4 (a) as fixed) and carbazole (entry
    # 6, "special numbering"); both number all skeletal atoms without the
    # systematic peripheral walk's choice (purine: fusion C at 4,5 not 4a/9a;
    # carbazole: NH at 9, fusion 4a/4b/8a/9a).  beta-carboline (the retained
    # name for 9H-pyrido[3,4-b]indole) uses carbazole's exact numbering pattern.
    # SMILES + locants are OPSIN ``-o extendedsmi $_AV`` (authoritative).
    (
        '7H-purine',
        'N1=CN=C2NC=NC2=C1',
        ('1', '2', '3', '4', '9', '8', '7', '5', '6'),
    ),
    (
        '9H-carbazole',
        'C1=CC=CC=2C3=CC=CC=C3NC12',
        ('1', '2', '3', '4', '4a', '4b', '5', '6', '7', '8', '8a', '9', '9a'),
    ),
    (
        '9H-beta-carboline',
        'C1=NC=CC=2C3=CC=CC=C3NC12',
        ('1', '2', '3', '4', '4a', '4b', '5', '6', '7', '8', '8a', '9', '9a'),
    ),
    # 9H-fluorene — the carbocyclic carbazole-shape (CH2 at 9, no heteroatom to
    # drive the lowest-locant cascade) is a retained PAH with the SAME special
    # numbering; the systematic walk/scorer mis-selects it (P-25.1.1 retained).
    (
        '9H-fluorene',
        'C1=CC=CC=2C3=CC=CC=C3CC12',
        ('1', '2', '3', '4', '4a', '4b', '5', '6', '7', '8', '8a', '9', '9a'),
    ),
)


def _parse_locant(text: str) -> _Locant:
    """'4a' -> (4, 'a'); '4' -> 4 (matching the ``_Locant`` int|tuple shape)."""
    import re as _re
    m = _re.match(r'^(\d+)([a-z]*)$', text)
    if m is None:  # pragma: no cover - reference strings are well-formed
        raise ValueError(f'bad locant {text!r}')
    base = int(m.group(1))
    suffix = m.group(2)
    return (base, suffix) if suffix else base


def _try_fixed_numbering(
    mol: Chem.Mol,
    ring_atoms: Set[int],
) -> Optional[Dict[int, _Locant]]:
    """Return the fixed traditional numbering if the ring system is a known
    special case (anthracene / phenanthrene); otherwise ``None``.

    Uses an exact substructure (graph) match restricted to ``ring_atoms`` so a
    decorated anthracene still resolves to anthracene's fixed numbering.  The
    match must cover ALL and ONLY the ring atoms (a same-size bijection) so a
    larger fused system (e.g. tetracene) does not spuriously match the
    anthracene fragment.
    """
    for _name, smi, locant_strs in _FIXED_NUMBERING_SYSTEMS:
        ref = Chem.MolFromSmiles(smi)
        if ref is None or ref.GetNumAtoms() != len(ring_atoms):
            continue
        ref_locants = [_parse_locant(s) for s in locant_strs]
        # uniquify=False so symmetric automorphisms are all available; any one
        # of them yields an equivalent (byte-identical after relabelling) map —
        # we take the first whose image is exactly the ring-atom set.
        for match in mol.GetSubstructMatches(ref, uniquify=False):
            if set(match) == ring_atoms:
                return {match[i]: ref_locants[i] for i in range(len(match))}
    return None


def _fusion_atoms(graph: Dict[int, Dict[str, object]]) -> Set[int]:
    """Atoms shared by 2+ rings (fusion atoms) in the ring-fusion graph."""
    membership: Dict[int, int] = {}
    for node in graph.values():
        for a in node['atoms']:  # type: ignore[union-attr]
            membership[a] = membership.get(a, 0) + 1
    return {a for a, c in membership.items() if c >= 2}


def _peripheral_cycle(
    mol: Chem.Mol,
    ring_atoms: Set[int],
) -> Optional[List[int]]:
    """Return the peripheral cycle of the fused system in connectivity order.

    The periphery of a cata-fused system is the unique cycle formed by every
    ring bond that bounds exactly ONE ring face (an exterior bond).  Interior
    (fusion) bonds bound two faces and are excluded.  For a cata-fused (no
    interior atom) system every skeletal atom lies on this cycle.

    Returns the atoms in cyclic order, or ``None`` if a single closed periphery
    cannot be formed (e.g. a peri-fused system slipped through).
    """
    ri = mol.GetRingInfo()
    # Count, for each ring bond, how many SSSR rings contain it.
    bond_face_count: Dict[Tuple[int, int], int] = {}
    for ring in ri.AtomRings():
        if not (set(ring) <= ring_atoms):
            continue
        n = len(ring)
        for i in range(n):
            a, b = ring[i], ring[(i + 1) % n]
            if mol.GetBondBetweenAtoms(a, b) is None:
                continue
            key = (a, b) if a < b else (b, a)
            bond_face_count[key] = bond_face_count.get(key, 0) + 1

    # Peripheral bonds bound exactly one face.
    peripheral_adj: Dict[int, List[int]] = {a: [] for a in ring_atoms}
    for (a, b), c in bond_face_count.items():
        if c == 1:
            peripheral_adj[a].append(b)
            peripheral_adj[b].append(a)

    # In a cata-fused system every atom must have exactly two peripheral
    # neighbours (the periphery is a single simple cycle through all atoms).
    if any(len(peripheral_adj[a]) != 2 for a in ring_atoms):
        return None

    # Walk the single cycle deterministically from the lowest atom index.
    start = min(ring_atoms)
    order = [start]
    prev = None
    cur = start
    while True:
        nbrs = peripheral_adj[cur]
        nxt = None
        for cand in sorted(nbrs):
            if cand != prev:
                nxt = cand
                break
        if nxt is None:
            return None
        if nxt == start:
            break
        order.append(nxt)
        prev, cur = cur, nxt
        if len(order) > len(ring_atoms):
            return None
    if len(order) != len(ring_atoms):
        return None
    return order


def _signed_area2(coords: List[Tuple[int, int]]) -> int:
    """Twice the signed area of a closed polygon (shoelace), in lattice units.

    Positive = counterclockwise in the lattice coordinate frame where +y is up.
    (The constant sqrt(3)/... real-y factor only scales the magnitude; the SIGN
    is preserved, which is all we use.)
    """
    n = len(coords)
    s = 0
    for i in range(n):
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return s


def _start_ring_key(
    graph: Dict[int, Dict[str, object]],
    coords: Dict[int, Tuple[int, int]],
) -> int:
    """Pick the start ring per P-25.3.3.1.1: uppermost, tie -> furthest right.

    Ring vertical position uses 6*true_y (= sum of atom y over the 6 atoms);
    horizontal uses 6*true_x.  Deterministic final tie-break by node key (the
    ring's minimum atom index) — only reached for genuinely coincident centers,
    which cannot happen for distinct rings, so it is a safety net.
    """
    best_key: Optional[int] = None
    best_metric: Optional[Tuple[int, int, int]] = None
    for key in sorted(graph):
        atoms = graph[key]['atoms']  # type: ignore[index]
        sy = sum(coords[a][1] for a in atoms)
        sx = sum(coords[a][0] for a in atoms)
        # Uppermost = max sy; tie -> furthest right = max sx.  Use negatives so
        # min() picks the winner; final tie-break by key (ascending).
        metric = (-sy, -sx, key)
        if best_metric is None or metric < best_metric:
            best_metric = metric
            best_key = key
    assert best_key is not None
    return best_key


def _clockwise_direction(
    coords: Dict[int, Tuple[int, int]],
    periphery: List[int],
) -> int:
    """Return +1 or -1: the ``periphery``-index step that walks CLOCKWISE.

    In the +y-up lattice frame a counterclockwise polygon has positive signed
    area, so the clockwise traversal is the -index direction (and vice-versa).
    """
    area2 = _signed_area2([coords[a] for a in periphery])
    return -1 if area2 > 0 else 1


def _candidate_starts(
    mol: Chem.Mol,
    coords: Dict[int, Tuple[int, int]],
    periphery: List[int],
    start_ring_atoms: Set[int],
    fusion_atoms: Set[int],
) -> List[Tuple[int, int]]:
    """Enumerate (start_atom, direction) candidates per P-25.3.3.1.1.

    P-25.3.3.1.1: "Numbering starts from the non-fused atom MOST COUNTERCLOCKWISE
    in the [uppermost, then rightmost] ring and proceeds in a CLOCKWISE direction."

    Exact geometric realisation (validated against OPSIN ``-o extendedsmi``
    numbering for every all-6 cata-fused PAH — linear acenes, angular polyaphenes,
    branched triphenylene, the chrysene / anthracene-fused families): the
    most-counterclockwise non-fusion atom is the non-fusion atom of the start ring
    at the COUNTERCLOCKWISE END of its peripheral non-fusion arc, i.e. the one
    whose counterclockwise peripheral neighbour is a fusion atom.  Walking
    CLOCKWISE from there enters the start ring's non-fusion arc first, giving it
    the lowest numbers.  (The earlier "uppermost non-fusion atom" proxy only
    coincided with this for linear acenes and was wrong for every angular or
    branched system — it minimised fusion-carbon locants one position too early.)

    For a terminal or angular start ring the non-fusion atoms form a single
    contiguous arc, so there is exactly ONE such atom — deterministic, no cascade
    needed.  A start ring with two separate non-fusion arcs (a linear-middle ring
    selected as uppermost-rightmost) yields more than one candidate; all are
    returned and the P-25.3.3.1.2 lowest-locant cascade in
    ``compute_fused_numbering`` discriminates (with the canonical-rank tie-break
    making symmetric alternatives deterministic).  If the start ring has no
    non-fusion atom at all, no candidate is returned (fail-closed; the rarer
    "advance to the next ring clockwise" case is outside this engine's scope).

    Returns ``(start_atom, clockwise_direction)`` pairs.
    """
    clockwise_dir = _clockwise_direction(coords, periphery)
    n = len(periphery)
    pos = {a: i for i, a in enumerate(periphery)}
    # The counterclockwise end of a non-fusion arc: a non-fusion start-ring atom
    # whose counterclockwise peripheral neighbour (index step -clockwise_dir) is a
    # fusion atom.
    candidates = [
        a for a in start_ring_atoms
        if a not in fusion_atoms
        and periphery[(pos[a] - clockwise_dir) % n] in fusion_atoms
    ]
    return [(a, clockwise_dir) for a in sorted(candidates)]


def _assign_from_start(
    mol: Chem.Mol,
    periphery: List[int],
    start_atom: int,
    direction: int,
    fusion_atoms: Set[int],
) -> Dict[int, _Locant]:
    """Assign locants walking the periphery clockwise from ``start_atom``.

    P-25.3.3.1.1: assign integers to non-fusion atoms AND fusion heteroatoms;
    each fusion CARBON is given the number of the immediately preceding
    non-fusion (numbered) atom, modified by 'a'/'b'/'c'/...  (Stage 1 is
    carbocyclic, so fusion heteroatoms do not occur, but the heteroatom branch
    is written for forward-compatibility and is harmless here.)
    """
    n = len(periphery)
    pos = {a: i for i, a in enumerate(periphery)}
    start_i = pos[start_atom]

    locants: Dict[int, _Locant] = {}
    counter = 0          # last assigned integer locant
    letter_ord = 0       # next Roman-letter suffix index for fusion carbons
    last_int: Optional[int] = None

    for step in range(n):
        atom_idx = periphery[(start_i + step * direction) % n]
        atom = mol.GetAtomWithIdx(atom_idx)
        is_fusion = atom_idx in fusion_atoms
        is_carbon = atom.GetAtomicNum() == 6
        if is_fusion and is_carbon:
            # Fusion carbon -> previous int + letter.
            if last_int is None:
                # A fusion carbon cannot legally be the first atom (start atom is
                # always a non-fusion atom); guard defensively.
                return {}
            suffix = chr(ord('a') + letter_ord)
            locants[atom_idx] = (last_int, suffix)
            letter_ord += 1
        else:
            # Non-fusion atom OR fusion heteroatom -> next integer.
            counter += 1
            locants[atom_idx] = counter
            last_int = counter
            letter_ord = 0
    return locants


def _heteroatom_pairs(
    mol: Chem.Mol,
    locants: Dict[int, _Locant],
) -> List[Tuple[_Locant, str]]:
    """(locant, element) pairs for ring heteroatoms (for the cascade tier)."""
    pairs: List[Tuple[_Locant, str]] = []
    for idx, loc in locants.items():
        sym = mol.GetAtomWithIdx(idx).GetSymbol()
        if sym != 'C':
            pairs.append((loc, sym))
    return pairs


def _fusion_letters(locants: Dict[int, _Locant]) -> List[_Locant]:
    """The fusion-carbon locant tuples (e.g. (4,'a'),(8,'a')) for tier (c)."""
    return [loc for loc in locants.values() if isinstance(loc, tuple)]


def _indicated_h_locants(
    mol: Chem.Mol,
    locants: Dict[int, _Locant],
) -> List[_Locant]:
    """Locants of indicated-hydrogen atoms, for the P-14.4(b) cascade tier.

    Restricted to AROMATIC heteroatoms bearing H (NH-type — 1H-indole,
    1H-benzimidazole): these are the genuine indicated-H of a mancude ring
    system.  A fully saturated ring (``-idine``, e.g. pyrrolizidine — every
    ring atom is an sp3 CH/CH2) has NO indicated hydrogen, and dihydro/sp3
    saturation is a HYDRO feature (a separate, lower cascade tier), so sp3 ring
    atoms are deliberately NOT counted here.  sp3 indicated-H mancude parents
    that the systematic walk cannot reproduce (9H-xanthene, purine, carbazole)
    are handled by ``_FIXED_NUMBERING_SYSTEMS`` instead.
    """
    out: List[_Locant] = []
    for idx, loc in locants.items():
        atom = mol.GetAtomWithIdx(idx)
        if (atom.GetTotalNumHs() >= 1 and atom.GetIsAromatic()
                and atom.GetAtomicNum() != 6):
            out.append(loc)
    return out


def compute_fused_numbering(
    mol: Chem.Mol,
    ring_system_atoms,
) -> Optional[Dict[int, _Locant]]:
    """Deterministic IUPAC fused-ring numbering for an all-6 cata-fused PAH.

    Returns ``{atom_idx -> locant}`` (``int`` or ``(int, 'a')`` tuple) when the
    ring system is all-six-membered, ortho-(cata-)fused, carbocyclic, and has no
    interior (peri-fusion) atom.  Returns ``None`` otherwise (the caller then
    keeps its existing numbering path — fail-closed, never a regression).

    The numbering is selected by the IUPAC P-25.3.3 cascade over every
    (surviving-orientation, start-atom) candidate, compared via the shared
    ``rules.locants`` primitives so the choice is fully deterministic and
    coordinate-free.

    Args:
        mol: RDKit Mol.
        ring_system_atoms: iterable of atom indices forming the fused ring
            system (the ring atoms of the parent — substituents excluded).

    Source: IUPAC 2013 Blue Book P-25.3.3.1 / P-25.3.3.1.2.
    """
    ring_atoms = set(ring_system_atoms)
    if len(ring_atoms) < 6:
        return None

    # v23 13B(a) S2a: heteroatoms in the rings are admitted (quinoline/acridine/
    # phenazine/pteridine/... families).  S2b: mixed 5/6-membered rings are
    # admitted too (indole/benzofuran/carbazole/...).  The embedding (hex for
    # all-6, regular-polygon for mixed) + peripheral walk are element- and
    # size-agnostic; the heteroatom- then indicated-H-lowest-locant cascade
    # (P-25.3.3.1.2 / P-14.4) is applied below.  We require ortho-(cata-)fused +
    # connected; ``best_orientations`` fail-closes on non-embeddable systems
    # (7-/8-membered rings = S2b.3, peri-fusion = S4), so this is a superset gate
    # that never regresses the all-6 path.
    info = classify_ring_system(mol, ring_atoms)
    if not (info['cata_fused'] and info['connected']):
        return None
    graph = info['graph']  # type: ignore[assignment]

    # This is a FUSION engine: require at least two ortho-fused rings.  A lone
    # ring (e.g. benzene) is not a fused system and is handled elsewhere.
    if len(graph) < 2:
        return None

    # P-25.1.1 / P-31.1.4.3.4 fixed-numbering exceptions (anthracene,
    # phenanthrene) — their traditional numbering is NOT the systematic walk.
    fixed = _try_fixed_numbering(mol, ring_atoms)
    if fixed is not None:
        return fixed

    periphery = _peripheral_cycle(mol, ring_atoms)
    if periphery is None or len(periphery) != len(ring_atoms):
        return None

    fusion_atoms = _fusion_atoms(graph)  # type: ignore[arg-type]

    orientations = best_orientations(mol, ring_atoms)
    if not orientations:
        return None

    # Canonical atom ranks (SMILES-order-INDEPENDENT) for the final symmetry
    # tie-break.  Two numberings tying through the whole P-25.3.3.1.2 cascade are
    # genuinely equivalent (a molecular automorphism relates them); to return ONE
    # deterministic representative regardless of input SMILES order we then
    # prefer the candidate whose locant->canonical-rank assignment is
    # lexicographically smallest.  This is a graph-canonical decision, never a
    # set/dict iteration-order one.
    canon_rank = list(Chem.CanonicalRankAtoms(mol, breakTies=False))

    def _canon_signature(cand: Dict[int, _Locant]) -> Tuple:
        # Order atoms by their assigned locant, emit the atom's canonical rank.
        items = sorted(
            cand.items(),
            key=lambda kv: (kv[1][0], kv[1][1]) if isinstance(kv[1], tuple)
            else (kv[1], ''),
        )
        return tuple(canon_rank[a] for a, _ in items)

    # Enumerate every (orientation, start) candidate numbering, then select the
    # IUPAC-preferred one via the shared lowest-locant cascade.
    best_map: Optional[Dict[int, _Locant]] = None
    best_het: Optional[List[Tuple[_Locant, str]]] = None
    best_fus: Optional[List[_Locant]] = None
    best_ih: Optional[List[_Locant]] = None
    best_sig: Optional[Tuple] = None
    for coords in orientations:
        start_ring = _start_ring_key(graph, coords)  # type: ignore[arg-type]
        start_ring_atoms = set(graph[start_ring]['atoms'])  # type: ignore[index]
        for start_atom, direction in _candidate_starts(
            mol, coords, periphery, start_ring_atoms, fusion_atoms,
        ):
            cand = _assign_from_start(
                mol, periphery, start_atom, direction, fusion_atoms,
            )
            if not cand:
                continue
            # Cascade comparison terms:
            #   P-25.3.3.1.2(a)/(b): heteroatom set then element seniority;
            #   P-25.3.3.1.2(c): low locants to fusion carbons;
            #   P-14.4(b): low locants to indicated hydrogen (1H-indole etc.).
            het = _heteroatom_pairs(mol, cand)
            fus = _fusion_letters(cand)
            ih = _indicated_h_locants(mol, cand)
            if best_map is None:
                best_map, best_het, best_fus, best_ih = cand, het, fus, ih
                best_sig = _canon_signature(cand)
                continue
            # Tier (a)+(b): heteroatoms (via the shared comparator).
            cmp = compare_numbering(
                {'heteroatoms': het}, {'heteroatoms': best_het},
            )
            if cmp == 0:
                # Tier (c): low locants to fusion carbons (the (int,'a') set).
                cmp = compare_locant_sets(fus, best_fus)
            if cmp == 0:
                # Tier (b, P-14.4): low locants to indicated hydrogen.
                cmp = compare_locant_sets(ih, best_ih)
            if cmp == 0:
                # Genuine symmetry tie -> canonical-rank representative.
                sig = _canon_signature(cand)
                if sig < best_sig:  # type: ignore[operator]
                    best_map, best_het, best_fus, best_ih, best_sig = (
                        cand, het, fus, ih, sig)
                continue
            if cmp < 0:
                best_map, best_het, best_fus, best_ih = cand, het, fus, ih
                best_sig = _canon_signature(cand)

    return best_map
