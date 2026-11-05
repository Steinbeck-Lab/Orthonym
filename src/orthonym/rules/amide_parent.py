"""The principal amide of a molecule that holds two carboxamide groups on different chains.

 "SENIORITY ORDER FOR CLASSES" (the Blue Book), Table 4.1, has one class of amides: the primary, secondary and
tertiary amide have the one rank, so no amide is the principal group because of the number of hydrogens on its nitrogen.
``SENIORITY_ORDER`` lists the three subtypes in a row (primary first), which made the primary amide the parent of
'CCC(=O)N(C)C(C)C(N)=O' ('2-(N-methylpropanamido)propanamide', a name that cites the better parent as a prefix). Which
amide is the parent is a choice between parent structures of the one class, by the criteria in this order:

0. "The senior parent structure has the maximum number of substituents corresponding to the principal
   characteristic group (suffix)" (:18875): the parent that holds more carboxamide groups outranks one that holds fewer,
   whatever the rings are ('5-(pyridine-4-carboxamido)benzene-1,3-dicarboxamide': the benzene with two groups outranks the
   pyridine with one). The groups of one ring system are counted together, as the suffix 'dicarboxamide' counts them; a
   chain holds one (``comparable_amide_units`` does not compare two groups of one chain).
1. "Systems composed of rings and chains" (:19336) (1) "Within the same class, a ring or ring system has
   seniority over a chain". Of two ring carboxamides the one on the senior ring system is the parent:
   "SENIORITY ORDER ONLY FOR RINGS AND RING SYSTEMS" (:19402), (heterocycle, nitrogen, the senior
   heteroatom, the number of rings, the skeletal atoms,...) and the criteria of that decide between rings
   (``rules.ring_selection.ring_system_score``). Two ring carboxamides that tie on the ring system are compared by
   (criterion 4), as two acyl chains are.
2. "The principal chain has the greater number of skeletal atoms" (:20938). The chains are the acyl chains
   through acyclic carbon atoms, so (skeletal heteroatoms) does not arise.
3. "If the criteria of through, where applicable, do not effect a choice of a senior parent
   structure, the following criteria are applied successively until there are no alternatives remaining" (:21016), as
   far as they can differ between two carbon chains whose first atom is the carbonyl carbon: (a) "the greater number of
   multiple bonds", (b) "the greater number of double bonds", (j) "the lower locant(s) for
   endings or prefixes that express changes in the level of hydrogenation, i.e., for 'ene' and 'yne' endings"
   ; "lower locants are assigned first to the endings as a set without regard to type and
   then to 'ene' endings"), (k) "one or more isotopically modified atoms":.1 the greater number of
   modified atoms,.2 the greater number of nuclides of higher atomic number,.3 of higher mass number,.4 the lowest
   locants,.5 the lower locants for the nuclides of higher atomic number,.6 for those of higher mass number). Criteria
   (c) to (g) and (i) need a heteroatom, a ring or a free valence and (h) ties (the carbonyl carbon is C-1 of every chain).
   (l) "The senior ring, ring system, or principal chain has one or more stereogenic centers [criterion (l)
   in ]" (:21529) is read as its rules and examples show it. "'cis/trans' Isomerism, the 'E/Z'
   convention": "When there is a choice between parent structures that differ only by 'Z' and 'E' configurations, the
   senior parent structure contains the greater number of double bonds with 'Z' configuration" ((Z)-cyclooctene over
   (E)-cyclooctene, (2Z)- over (2E)-2-methylbut-2-enoic acid, (4Z)- over (4E)-hex-4-enenitrile); the rule for chirality
   centres begins "When there is a choice between parent structures differing only by the configurations of the
   chirality centers" (:21541). Every example compares parents of one constitution, so (l) does not make a stereogenic
   chain senior to a non-stereogenic one of another constitution (ruling of the lead, 2026-11-05); example (5) is
   of a kind with them: "3-[5-(3-amino-2-methyl-3-oxopropyl)-2-methylphenyl]-N-methylpropanamide (PIN) [not 2-methyl-3-{4-
   methyl-3-[3-(methylamino)-3-oxopropyl]phenyl}propanamide; the locant set 'N,3' in the PIN is lower than '2,3']" compares
   two propanamide parents of one constitution by their prefixes, as the examples of (l) compare parents that differ in
   configuration only. It is no key here: two chains that tie up to (k) and differ in configuration only tie.
   The reading is a ruling, not the book's own words ("the senior ring, ring system, or principal chain has one or more
   stereogenic centers"), so a name whose parent chose over a chain that (l) read literally would make senior (two
   chains that tie up to (k), the one that leaves junior holds a stereogenic centre or double bond and the parent holds
   none) is labelled below the PIN (:func:`stereo_criterion_could_differ`).
4. "CRITERIA RELATED TO NUMBER AND LOCATION OF SUBSTITUENT GROUPS" (:21596): "the senior parent
   structure that has the maximum number of substituents cited as prefixes (other than 'hydro/dehydro')" (:21604;
   example (4), the N-substituents counted: 'N,N,2-trimethyl-3-{...}propanamide (PIN)'), then "the lower locant
   or set of locants for substituents cited as prefixes" (:21698), whose example (1) reads the locant 'N' as lower than a
   numeral ('the locant set N,2 in the PIN is lower than 2,5'; "Lowest set of locants" (:3189): "Italic
   capital and lower-case letter locants are lower than Greek letter locants, which, in turn, are lower than numerals").
   The two criteria rank the amides of two ring systems that tie in the same way: 'N-(4-carbamoylphenyl)benzamide'
   (the locant N) is the PIN of 'c1ccc(cc1)C(=O)Nc1ccc(cc1)C(N)=O', not '4-benzamidobenzamide' (the locant 4), and
   '4-benzamido-N-methylbenzamide' (two prefixes) is the PIN of its N-methyl derivative, not 'N-[4-(methylcarbamoyl)phenyl]-
   benzamide' (one). A ring cites its prefixes at the locants of its numbering (``_monocycle_prefix_locants``, for a
   monocycle; the number of prefixes and of N terms alone decide for any other ring system). The groups of the ring system
   that are carboxamides count as the suffix, every other substituent as a prefix.

5. Two amides that tie on all of those (``tied_amide_parents``: the same number of prefixes at the same locants, and not a
   symmetric copy of one another) are told apart by the names of their prefixes, which the key of the structures does not
   carry: "the lower locant set for substituent groups in order of citation in the name" (:21791) and "the
   name that is earlier in alphanumerical order" (:22234), ``rules.amide_tie_names`` (the decomposition engine applies it to
   the amide splits, ``decomposition.amide_ties``).

``senior_carboxamide_group`` re-selects the principal group among the amide subtypes by those criteria. A key that ties
keeps the head of ``SENIORITY_ORDER`` (the choice stays as it was where the rules above do not decide). Molecules whose
amide groups lie on one chain (a diamide) are not touched: the suffix 'diamide' counts them together.
"""
from typing import Dict, Iterable, List, Optional, Tuple

#: the amide subtypes of the one class, class of amides), in the order of ``SENIORITY_ORDER``
CARBOXAMIDE_CLASSES = ("primary_amide", "secondary_amide", "tertiary_amide")

#: the first term of the rank of a ring parent: a ring outranks every chain (1))
_RING_RANK = 0

#: locant terms of the lowest-set comparison of: 'N' is lower than every numeral
_N_TERM = (0, 0)

#: the positions of the amide nitrogen and of the carbonyl oxygen in the locant comparison of the isotopic criteria
#: (an italic capital is lower than a numeral,, and N comes before O)
_N_POSITION, _O_POSITION = -2, -1


def _is_carbonyl(mol, idx: int) -> bool:
    return any(nb.GetAtomicNum() == 8 and mol.GetBondBetweenAtoms(idx, nb.GetIdx()).GetBondTypeAsDouble() == 2.0
               for nb in mol.GetAtomWithIdx(idx).GetNeighbors())


def _acyl_count(mol, n_idx: int) -> int:
    """The carbon neighbours of the nitrogen ``n_idx`` that carry a carbonyl oxygen."""
    return sum(1 for nb in mol.GetAtomWithIdx(n_idx).GetNeighbors()
               if nb.GetAtomicNum() == 6 and _is_carbonyl(mol, nb.GetIdx()))


def _acyclic_carbons(mol, start: int) -> set:
    """``start`` and the acyclic carbon atoms reached from it through acyclic carbon atoms: the chain system of the acyl."""
    seen, stack = {start}, [start]
    while stack:
        for nb in mol.GetAtomWithIdx(stack.pop()).GetNeighbors():
            i = nb.GetIdx()
            if i not in seen and nb.GetAtomicNum() == 6 and not nb.IsInRing():
                seen.add(i)
                stack.append(i)
    return seen


def _acyl_paths(mol, carbonyl_c: int) -> List[Tuple[int, ...]]:
    """Every path from the carbonyl carbon to a leaf of its acyclic carbon chain system: the candidate chains of the acyl."""
    paths: List[Tuple[int, ...]] = []

    def walk(path):
        nxt = [nb.GetIdx() for nb in mol.GetAtomWithIdx(path[-1]).GetNeighbors()
               if nb.GetAtomicNum() == 6 and not nb.IsInRing() and nb.GetIdx() not in path]
        if not nxt:
            paths.append(path)
        for i in nxt:
            walk(path + (i,))
    walk((carbonyl_c,))
    return paths


def _label_map(mol) -> Dict[int, List[Tuple[int, int]]]:
    """``{heavy atom: [(atomic number, mass number),...]}``: the isotopic modifications of the atom itself and of its
    isotopic hydrogens, read from ``mol`` when it carries them, else from the labelled molecule whose isotope-stripped
    skeleton ``mol`` is (``assembly.locant_omission.skeleton_label_maps``: the decorator names the skeleton, so no label
    is on its atoms). A symmetric skeleton fits the labelled molecule more than once; the first fit is read, the
    choice among its symmetric amides is then the labelled one's. ``{}`` for a molecule without labels."""
    direct: Dict[int, List[Tuple[int, int]]] = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        found = ([(atom.GetAtomicNum(), atom.GetIsotope())] if atom.GetIsotope() else []) + [
            (1, nb.GetIsotope()) for nb in atom.GetNeighbors() if nb.GetAtomicNum() == 1 and nb.GetIsotope()]
        if found:
            direct[atom.GetIdx()] = found
    if direct:
        return direct
    try:
        from ..assembly.locant_omission import skeleton_label_maps
        maps = skeleton_label_maps(mol)
    except Exception:  # noqa: BLE001 - no declared labelled molecule: no labels
        return {}
    return dict(maps[0]) if maps else {}


def _isotope_key(parent_positions: Dict[int, int], labels: Dict[int, List[Tuple[int, int]]]) -> tuple:
    """Criterion (k) of to.6) for the parent atoms ``parent_positions`` (atom -> locant
    position); the smaller key is the senior parent. The modifications of an atom are cited at its position, those
    of its hydrogens too."""
    found = [(pos, z, m) for atom, pos in parent_positions.items() for z, m in labels.get(atom, ())]
    if not found:
        return (0, (), (), (), (), ())
    by_z = sorted(found, key=lambda t: (-t[1], -t[2]))
    return (-len(found),                                           #.1 the greater number of modified atoms or groups
            tuple(-z for _, z, _ in by_z),                          #.2 the greater number of nuclides of higher atomic number
            tuple(-m for _, _, m in by_z),                          #.3 of higher mass number
            tuple(sorted(pos for pos, _, _ in found)),              #.4 the lowest locants
            tuple(sorted((-z, pos) for pos, z, _ in found)),        #.5 the lower locants for the higher atomic number
            tuple(sorted((-z, -m, pos) for pos, z, m in found)))    #.6 and for the higher mass number


def _path_key(mol, path: Tuple[int, ...], nitrogen: Optional[int], oxygen: Optional[int] = None,
              labels: Optional[Dict[int, List[Tuple[int, int]]]] = None) -> tuple:
    """``(-skeletal atoms, -multiple bonds, -double bonds, ene/yne locants, ene locants, isotopes, -prefixes, prefix
    locants)``: the smaller key is the senior chain, (a), (b), (j), (k),,.
    The carbonyl carbon is C-1; every heavy neighbour of a chain atom off the chain, and every heavy substituent of the
    amide nitrogen, is one substituent cited as a prefix; its locant is the position on the chain, or N."""
    on_path = set(path)
    multiple, double = [], []
    for pos, (a, b) in enumerate(zip(path, path[1:]), start=1):
        order = mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble()
        if order > 1.0:
            multiple.append(pos)
            if order == 2.0:
                double.append(pos)
    terms = []
    for pos, a in enumerate(path, start=1):
        for nb in mol.GetAtomWithIdx(a).GetNeighbors():
            if nb.GetAtomicNum() > 1 and nb.GetIdx() not in on_path and pos > 1:
                terms.append((1, pos))
    if nitrogen is not None:
        for nb in mol.GetAtomWithIdx(nitrogen).GetNeighbors():
            if nb.GetAtomicNum() > 1 and nb.GetIdx() != path[0]:
                terms.append(_N_TERM)
    parent = {a: pos for pos, a in enumerate(path, start=1)}
    if nitrogen is not None:
        parent[nitrogen] = _N_POSITION
    if oxygen is not None:
        parent[oxygen] = _O_POSITION
    labels = _label_map(mol) if labels is None else labels
    return ((-len(path), -len(multiple), -len(double), tuple(multiple), tuple(double))
            + _isotope_key(parent, labels) + (-len(terms), tuple(sorted(terms))))


def _monocycle_prefix_locants(mol, ring: set, suffix_atoms: List[int], prefix_atoms: List[int]) -> Optional[Tuple[int, ...]]:
    """The locants of the prefixes ``prefix_atoms`` (the ring atom of each, once per substituent) in the numbering of the
    monocycle ``ring`` that gives: the heteroatoms as a set, then in the order of their seniority, the
    indicated hydrogen of an aromatic ring, the principal characteristic groups (``suffix_atoms``, the ring atom of each)
    and then the prefixes, each as the lowest set. None for a ring that is not one plain cycle, or whose double bonds
    are neither all of the mancude ring nor absent (the hydro prefixes and the 'ene' endings come between the suffixes
    and the prefixes there and are not read here)."""
    cycles = [r for r in mol.GetRingInfo().AtomRings() if set(r) == set(ring)]
    if len(cycles) != 1:
        return None
    order = cycles[0]
    n = len(order)
    aromatic = all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in order)
    saturated = all(not mol.GetAtomWithIdx(a).GetIsAromatic() for a in order) and all(
        mol.GetBondBetweenAtoms(order[i], order[(i + 1) % n]).GetBondTypeAsDouble() == 1.0 for i in range(n))
    if not (aromatic or saturated):
        return None
    from .ring_replacement import HETEROATOM_PREFIXES
    het = [a for a in order if mol.GetAtomWithIdx(a).GetAtomicNum() != 6]
    hydrogen = [a for a in het if aromatic and mol.GetAtomWithIdx(a).GetTotalNumHs() > 0]
    best = None
    for start in range(n):
        for step in (1, -1):
            place = {order[(start + step * i) % n]: i + 1 for i in range(n)}
            seniority = tuple(place[a] for a in sorted(
                het, key=lambda a: (HETEROATOM_PREFIXES.get(mol.GetAtomWithIdx(a).GetSymbol(), ('', 99))[1], place[a])))
            prefixes = tuple(sorted(place[a] for a in prefix_atoms))
            key = (tuple(sorted(place[a] for a in het)), seniority, tuple(sorted(place[a] for a in hydrogen)),
                   tuple(sorted(place[a] for a in suffix_atoms)), prefixes)
            if best is None or key < best:
                best = key
    return best[-1]


def _ring_prefix_terms(mol, system: set, suffix_carbons: List[int]) -> tuple:
    """``(-prefixes, -N terms, ring locants)`` of the ring carboxamide parent on the ring system ``system`` that holds
    the carboxamide groups ``suffix_carbons`` (the carbonyl carbons): "the senior parent structure that has the
    maximum number of substituents cited as prefixes" (the Blue Book) counts every substituent of the ring, other
    than the groups of the suffix, and every substituent of the amide nitrogens (cited with the locant N); (:21698)
    compares the locants, 'N' lower than a numeral (example (1), 'the locant set N,2 in the PIN is lower than 2,5'). The
    locants of the N terms come first, so the number of N terms decides before the numerals do, and the numerals are
    the lowest set of the numbering of ``_monocycle_prefix_locants`` (```` for a ring that is not a monocycle: the
    two parents of a tie are then judged by what the count and the N terms say)."""
    suffix = set(suffix_carbons)
    n_terms = sum(1 for c in suffix_carbons for nb in mol.GetAtomWithIdx(c).GetNeighbors() if nb.GetAtomicNum() == 7
                  for sub in nb.GetNeighbors() if sub.GetAtomicNum() > 1 and sub.GetIdx() != c)
    bearing_suffix = [a for c in suffix_carbons for a in (nb.GetIdx() for nb in mol.GetAtomWithIdx(c).GetNeighbors())
                      if a in system]
    prefix_atoms = [a for a in sorted(system) for nb in mol.GetAtomWithIdx(a).GetNeighbors()
                    if nb.GetAtomicNum() > 1 and nb.GetIdx() not in system and nb.GetIdx() not in suffix]
    numerals = _monocycle_prefix_locants(mol, system, bearing_suffix, prefix_atoms) if is_lone_ring(mol, system) else None
    return (-(len(prefix_atoms) + n_terms), -n_terms, numerals or ())


def is_lone_ring(mol, system) -> bool:
    """True when the ring system ``system`` is one ring (not fused, spiro or bridged)."""
    rings = [set(r) for r in mol.GetRingInfo().AtomRings() if set(r) <= set(system)]
    return len(rings) == 1 and rings[0] == set(system)


def amide_parent_key(mol, carbonyl_c: int, labels: Optional[Dict[int, List[Tuple[int, int]]]] = None,
                     carbonyls: Optional[Iterable[int]] = None) -> tuple:
    """The rank of the amide whose carbonyl carbon is ``carbonyl_c`` as a parent structure: the smaller is the senior.

    ``(-groups, 0, score of the ring system, terms)`` for the amide of a ring carboxamide, where ``groups`` is the
    number of carboxamide groups on its ring system, the maximum number of principal characteristic groups,
    comes before the seniority of the rings: the benzene of 'benzene-1,3-dicarboxamide' outranks a pyridine with one),
    then (1) and between the ring systems, then (see the module docstring); else
    ``(-1, 1)`` and the key of the best acyl chain by the criteria of the module docstring (a chain holds one amide
    group of the comparison). The chain runs through acyclic carbon atoms only. ``labels`` is:func:`_label_map` of
    ``mol`` where the caller has it; ``carbonyls`` are the carbonyl carbons of the amides that can be parents
    (:func:`comparable_amide_units`), found from ``mol`` when the caller has not got them."""
    atom = mol.GetAtomWithIdx(carbonyl_c)
    ring_neighbours = [nb.GetIdx() for nb in atom.GetNeighbors() if nb.GetAtomicNum() == 6 and nb.IsInRing()]
    if ring_neighbours:
        from ..perception.rings import get_ring_systems
        from .ring_selection import ring_system_score
        system = set(next(sy for sy in get_ring_systems(mol) if ring_neighbours[0] in sy))
        if carbonyls is None:
            carbonyls = {m[0] for _, m in comparable_amide_units(mol) or ()}
        on_system = sorted({c for c in carbonyls if any(nb.GetIdx() in system for nb in mol.GetAtomWithIdx(c).GetNeighbors())}
                           | {carbonyl_c})
        return (-len(on_system), _RING_RANK, ring_system_score(mol, system)) + _ring_prefix_terms(mol, system, on_system)
    nitrogen = next((nb.GetIdx() for nb in atom.GetNeighbors() if nb.GetAtomicNum() == 7), None)
    oxygen = next((nb.GetIdx() for nb in atom.GetNeighbors() if nb.GetAtomicNum() == 8
                   and mol.GetBondBetweenAtoms(carbonyl_c, nb.GetIdx()).GetBondTypeAsDouble() == 2.0), None)
    labels = _label_map(mol) if labels is None else labels
    return (-1, 1) + min(_path_key(mol, p, nitrogen, oxygen, labels) for p in _acyl_paths(mol, carbonyl_c))


def _is_carboxamide_unit(mol, match) -> bool:
    """A match ``(C, O, N,...)`` that is the carboxamide of a carbon chain or ring and may be a parent structure: the
    carbonyl carbon holds a carbon (or a hydrogen, a formamide) and is not in a ring with its nitrogen (a lactam is a
    pseudoketone, "Cyclic anhydrides, esters and amides are named as pseudoketones", the Blue Book), and its
    nitrogen is not a ring atom (an acyl on a heterocyclic nitrogen is a pseudoketone too,: ``is_hidden_amide``).
    The carbonyl carbon of a urea or a carbamate has no carbon neighbour and is another class."""
    from .pseudoketones import is_hidden_amide
    if len(match) < 3:
        return False
    bond = mol.GetBondBetweenAtoms(match[0], match[2])
    if bond is None or bond.IsInRing() or is_hidden_amide(mol, match) is not None:
        return False
    carbon = mol.GetAtomWithIdx(match[0])
    return carbon.GetTotalNumHs() > 0 or any(nb.GetAtomicNum() == 6 for nb in carbon.GetNeighbors())


def _chain_units(mol, functional_groups: Dict[str, List[tuple]]) -> Dict[str, List[tuple]]:
    """The carboxamide matches of each subtype that may be a parent structure (:func:`_is_carboxamide_unit`)."""
    units: Dict[str, List[tuple]] = {}
    for cls in CARBOXAMIDE_CLASSES:
        found = [m for m in functional_groups.get(cls) or [] if _is_carboxamide_unit(mol, m)]
        if found:
            units[cls] = found
    return units


def comparable_amide_units(mol, functional_groups: Optional[Dict[str, List[tuple]]] = None
                           ) -> Optional[List[Tuple[str, tuple]]]:
    """``[(subtype, match),...]``: the carboxamide groups of ``mol`` that are candidate parent structures of one
    choice, or None when ``mol`` has no choice to make or the choice is not one between parents: fewer than two amides,
    two amide groups on one chain (one diamide, not two parents) or a nitrogen acylated twice (an imide: which acyl
    group is the parent is not asked here)."""
    if functional_groups is None:
        from ..perception.functional_groups import detect_functional_groups
        functional_groups = detect_functional_groups(mol)
    units = _chain_units(mol, functional_groups)
    flat = [(cls, m) for cls, ms in units.items() for m in ms]
    if len(flat) < 2:
        return None
    carbonyls = [m[0] for _, m in flat]
    for c in carbonyls:
        if any(o != c and o in _acyclic_carbons(mol, c) for o in carbonyls):
            return None
    if any(_acyl_count(mol, m[2]) > 1 for _, m in flat):
        return None
    return flat


def senior_carboxamide_group(mol, functional_groups: Dict[str, List[tuple]], head: Tuple[Optional[str], List[tuple]]
                             ) -> Tuple[Optional[str], List[tuple]]:
    """``head`` (the principal group of ``SENIORITY_ORDER``) or, when it is a carboxamide and the molecule holds amide
    groups on chains of their own, the amide group (its subtype and its match) whose chain is the senior parent
    structure (see the module docstring). The matches of the answer are those of the senior rank only: a junior amide of
    the same subtype is cited inside the name of the N-substituent that holds it, not as a second suffix. Unchanged when
    the amides tie, lie on one chain, or when a nitrogen is acylated twice."""
    group = head[0]
    if group not in CARBOXAMIDE_CLASSES:
        return head
    try:
        flat = comparable_amide_units(mol, functional_groups)
        if flat is None or not set(head[1]) <= {m for c, m in flat if c == group}:   # a lactam among the head's matches
            return head
        labels = _label_map(mol)
        carbonyls = {m[0] for _, m in flat}
        keyed = [(amide_parent_key(mol, m[0], labels, carbonyls), CARBOXAMIDE_CLASSES.index(cls), cls, m) for cls, m in flat]
        best_key, best_rank = min((k, r) for k, r, _, _ in keyed)
        matches = [m for k, r, _, m in keyed if (k, r) == (best_key, best_rank)]
        best = CARBOXAMIDE_CLASSES[best_rank]
        # an equal rank across the amides of the head's subtype is a tie: the head, as it was
        if best == group and set(matches) == set(head[1]):
            return head
        return best, matches
    except Exception:  # noqa: BLE001 - not decidable: the head of SENIORITY_ORDER stays
        return head


def parent_amide_is_junior(mol, parent_carbonyls) -> bool:
    """True when the amide that a name would take as its parent is strictly junior (:func:`amide_parent_key`) to another
    carboxamide of the molecule: the name would cite the senior amide as a prefix, which is not the PIN.
    ``parent_carbonyls`` holds the carbonyl carbons of the amides the name may take as its parent (the best of them is the
    parent). False when it holds none, when no other amide exists, when the amides are not a choice between parents
    (:func:`comparable_amide_units`), and when the keys tie or the choice cannot be decided."""
    try:
        flat = comparable_amide_units(mol)
        if flat is None:
            return False
        labels = _label_map(mol)
        carbonyls = {m[0] for _, m in flat}
        parent_set = set(parent_carbonyls)
        inside = [amide_parent_key(mol, m[0], labels, carbonyls) for _, m in flat if m[0] in parent_set]
        outside = [amide_parent_key(mol, m[0], labels, carbonyls) for _, m in flat if m[0] not in parent_set]
        return bool(inside and outside and min(outside) < min(inside))
    except Exception:  # noqa: BLE001 - not decidable: the parent stands
        return False


def tied_amide_parents(mol, parent_atoms) -> Optional[List[Tuple[int, bool]]]:
    """``[(carbonyl carbon, is the name's own),...]``, one amide for each class of atoms that the symmetry of the molecule
    maps onto each other, when the amide that a name takes as its parent ranks equal (:func:`amide_parent_key`) with another
    carboxamide of the molecule that is not a symmetric copy of it; None otherwise. The Blue Book then decides between the
    parents by "the lower locant set for substituent groups in order of citation in the name" (:21791) and
    "the name that is earlier in alphanumerical order" (:22234), criteria of the names that the key of the structures does
    not carry (``rules.amide_tie_names``).

    ``parent_atoms`` are the atoms of the part of the molecule that the name takes its parent from, as for
    :func:`parent_amide_is_junior`: the best amide inside them is the parent. None where that function is False for want of
    a choice (fewer than two amides, one chain, an imide), where the parent is strictly senior or junior, and where the
    parent holds no carboxamide."""
    try:
        flat = comparable_amide_units(mol)
        if flat is None:
            return None
        labels = _label_map(mol)
        carbonyls = {m[0] for _, m in flat}
        keys = {c: amide_parent_key(mol, c, labels, carbonyls) for c in carbonyls}
        parent = set(parent_atoms)
        inside = [k for c, k in keys.items() if c in parent]
        outside = [k for c, k in keys.items() if c not in parent]
        if not inside or not outside or min(outside) < min(inside):
            return None
        best = min(inside)
        from rdkit import Chem
        ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
        classes: Dict[int, Tuple[int, bool]] = {}
        for c in sorted(c for c, k in keys.items() if k == best):
            own = c in parent
            if ranks[c] not in classes or (own and not classes[ranks[c]][1]):
                classes[ranks[c]] = (c, own)
        return list(classes.values()) if len(classes) > 1 else None
    except Exception:  # noqa: BLE001 - not decidable: no tie is asserted
        return None


#: the part of:func:`amide_parent_key` of a chain amide that and (a), (b), (j), (k) decide: ``(-1, 1)``, the
#: chain length, the multiple and double bonds and their locants (5 terms), the isotope key (6 terms); the terms follow
_KEY_THROUGH_CRITERION_K = 13


def _best_acyl_path(mol, carbonyl_c: int, labels) -> Tuple[int, ...]:
    atom = mol.GetAtomWithIdx(carbonyl_c)
    nitrogen = next((nb.GetIdx() for nb in atom.GetNeighbors() if nb.GetAtomicNum() == 7), None)
    oxygen = next((nb.GetIdx() for nb in atom.GetNeighbors() if nb.GetAtomicNum() == 8
                   and mol.GetBondBetweenAtoms(carbonyl_c, nb.GetIdx()).GetBondTypeAsDouble() == 2.0), None)
    return min(_acyl_paths(mol, carbonyl_c), key=lambda p: _path_key(mol, p, nitrogen, oxygen, labels))


def _chain_is_stereogenic(mol, path: Tuple[int, ...], atoms: set, bonds: set) -> bool:
    """The chain ``path`` holds a stereogenic centre or a stereogenic double bond whose configuration the drawing gives
    (``Chem.FindPotentialStereo`` of the molecule, the specified ones: ``atoms`` and ``bonds`` of its answer)."""
    if any(a in atoms for a in path):
        return True
    return any(mol.GetBondBetweenAtoms(a, b).GetIdx() in bonds for a, b in zip(path, path[1:]))


def stereo_criterion_could_differ(mol, parent_carbonyls) -> bool:
    """True when the amide that a name takes as its parent (the best of ``parent_carbonyls``, the carbonyl carbons) was chosen
    over a chain that criterion (l) of read literally would make senior: another carboxamide chain that ties with the
    parent up to criterion (k) (:data:`_KEY_THROUGH_CRITERION_K`) and holds a stereogenic centre or double bond, the parent's
    holding none. "The senior ring, ring system, or principal chain has one or more stereogenic centers [criterion
    (l) in ]" (the Blue Book) is read here as its examples show it (the module docstring), so picked the
    parent; the two readings differ for exactly this molecule, and the name is not shown to be the preferred one. A centre or
    double bond counts when the drawing gives its configuration: the criterion compares the parents of names that cite the
    descriptors ("When there is a choice between parent structures differing only by the configurations of the chirality
    centers",:21541), and a centre the input leaves open is cited by no name, so the plain alanyl chain of the rows
    keeps their label. False for ring amides, where the key is no chain's, and where the choice cannot be decided."""
    try:
        flat = comparable_amide_units(mol)
        if flat is None:
            return False
        labels = _label_map(mol)
        carbonyls = {m[0] for _, m in flat}
        parent_set = set(parent_carbonyls)
        keys = {c: amide_parent_key(mol, c, labels, carbonyls) for c in carbonyls}
        inside = {c: k for c, k in keys.items() if c in parent_set}
        outside = {c: k for c, k in keys.items() if c not in parent_set}
        if not inside or not outside:
            return False
        best = min(inside, key=lambda c: (inside[c], c))
        tied = [c for c, k in outside.items() if len(k) > 1 and k[1] == 1 and inside[best][1] == 1
                and k[:_KEY_THROUGH_CRITERION_K] == inside[best][:_KEY_THROUGH_CRITERION_K]]
        if not tied:
            return False
        from rdkit import Chem
        stereo = [i for i in Chem.FindPotentialStereo(mol) if i.specified == Chem.StereoSpecified.Specified]
        atoms = {i.centeredOn for i in stereo if i.type == Chem.StereoType.Atom_Tetrahedral}
        bonds = {i.centeredOn for i in stereo if i.type == Chem.StereoType.Bond_Double}
        if _chain_is_stereogenic(mol, _best_acyl_path(mol, best, labels), atoms, bonds):
            return False
        return any(_chain_is_stereogenic(mol, _best_acyl_path(mol, c, labels), atoms, bonds) for c in tied)
    except Exception:  # noqa: BLE001 - not decidable: no difference is asserted
        return False


STEREO_RULE = "P-44.4.1.12"
STEREO_DETAIL = ("criterion (l), the senior chain has one or more stereogenic centres, read literally makes another amide "
                 "the parent; the parent chosen by P-45.2 is not shown to be the preferred one")


def spelled_outside_n_substituents(mol, amide_atoms, prefixes):
    """``prefixes`` without those whose atoms all lie in an N-substituent of the amide ``amide_atoms``: the name of the
    N-substituent ('N-(1-amino-1-oxopropan-2-yl)') spells the groups it holds, so a prefix of the parent for one of them
    ('carbamoyl') would cite the same group a second time."""
    from .amides import get_n_substituents
    inside = set()
    for sub in get_n_substituents(mol, amide_atoms) or []:
        inside.update(sub.get("atoms") or ())
    if not inside:
        return prefixes
    return [p for p in prefixes if not (getattr(p, "atoms", None) and set(p.atoms) <= inside)]
