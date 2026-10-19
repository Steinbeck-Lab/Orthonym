"""The component name of a carbocyclic monocycle in a spiro system, with its double bonds.

One namer for every spiro builder that cites a carbocyclic ring as a component
(``spiro[cyclohexane-1,1'-indene]``): a saturated ring is ``cyclo<stem>ane``; a ring
with double bonds cites them with the ring's own locants, ``cyclohex-2-ene``,
``cyclopent-3-ene``, ``cyclohexa-2,4-diene``.

Governing rules (the Blue Book Blue Book):

* 'Monospiro ring systems with different ring components at least one being a
  polycyclic ring and at least one ring component requiring the use of skeletal
  replacement ('a') nomenclature' (:10272), Explanation of the second example (:10291):
  "The spiro atom of the monocyclic hydrocarbon component is given preference for low
  locant." The spiro atom of the ring is locant 1.
* 'Spiro compounds composed of unsaturated rings', (:16715): "Low
  locants are assigned to double bonds in accordance with the fixed numbering of the spiro
  compound." The direction round the ring gives the double bonds the lowest locants.
* 'Unbranched polyspiro ring systems with different ring components one being a
  polycyclic ring system' (:10705), (:10719): "If there is a choice of locants, the
  lowest set of locants for all spiro atoms is selected when compared in increasing numerical
  order and, if there is still a choice, in order of their citation in the name". A ring that
  carries two spiro atoms (the middle ring of a polyspiro system) numbers the junction toward the
  earlier-cited component 1 (the caller passes it as ``spiro_center``), gives the other junction
  the lowest locant the ring allows, and only then looks at the double bonds.
* (:16497): "when the endings 'ene' and 'yne' are preceded by a multiplying
  prefix and a locant the letter 'a' is inserted": ``cyclohexa-2,4-diene``.
* NUMBERING (:3219), criteria (c) 'principal characteristic groups and free valences
  (suffixes)' (:3256), (e) 'saturation/unsaturation' (:3288, "low locants are given to
  hydro/dehydro prefixes... and 'ene' and 'yne' endings") and (f) (:3301) "detachable
  alphabetized prefixes, all considered together in a series of increasing numerical order". The
  direction takes them in that order after the spiro junctions. The ring is numbered before any
  suffix is chosen, so the callers of today pass no ``suffix_atoms``; a direction that puts a
  double bond lower than a ketone locant is still a correct name of the structure, never a
  different one.

The Blue Book's own spelling for a monocyclic component with a double bond is the 'ene'
ending after the closing bracket, 'Spiro compounds' (:17042), example:17052:
'4,5-dihydro-3H-spiro[[1]benzofuran-2,1'-cyclohexan]-2'-ene (PIN)'. OPSIN 2.9.0 has no
grammar for it (it reads ``spiro[indene-1,1'-cyclohex-2-ene]``), so the name built here is
the spelling of the component itself, not the PIN; the callers label it below the PIN.
"""
from typing import Dict, Optional, Set, Tuple

from rdkit import Chem

from ..data.chain_names import get_chain_prefix as _get_chain_prefix
from ..perception.rings import get_spiro_atoms


def name_carbocyclic_monocycle_component(
    mol, component_atoms: Set[int], spiro_center: int, *,
    other_spiro_atoms: Tuple[int, ...] = (), substituent_tiebreak: bool = False,
    suffix_atoms: Tuple[int, ...] = (),
) -> Optional[Tuple[str, Dict[int, int]]]:
    """Name a single CARBOCYCLIC ring component of a spiro system, numbering it
    with locant 1 at the spiro atom (the side-ring convention). Returns
    ``(cyclo<N>ane / cyclo<stem>a-<enes>-diene / -ene, {orig_idx: locant})`` or
    None for a heteroatom ring (fail-closed).

     tail #7: a ring bearing C=C (the spiro-cyclohexadienone side of a
    spiro-quinone alkaloid) is now rendered as the mancude ``cyclohexa-2,5-diene``
    component. The double-bond locants are cited UNPRIMED here; the caller's spiro
    assembler is responsible for the primed/unprimed placement of the
    OTHER (non-first-cited) component and RT-verifies the whole name . The
    numbering walks from the spiro atom (locant 1) in the direction giving the
    C=C set the lowest locants; a saturated ring is byte-identical
    to the old output (no ene infix).

    ``other_spiro_atoms`` (a ring that carries more than one spiro atom,: the
    direction first gives those atoms the lowest locants; ``suffix_atoms`` (the ring atoms that
    carry the principal characteristic group or a free valence, (c)) come next, then the
    double bonds (e)). ``substituent_tiebreak``: when the double bonds tie (a ring unsaturated
    symmetrically about the spiro atom, ``cyclopent-3-ene``, ``cyclohexa-2,5-diene``),
    take the direction that gives the ring atoms bearing an exocyclic substituent the
    lowest locants as a set (f)); otherwise a tie keeps the first direction, as it
    always did."""
    from .spiro import _extract_subfragment
    extracted = _extract_subfragment(mol, component_atoms)
    if extracted is None:
        return None
    frag, orig_to_frag = extracted
    frag_to_orig = {v: k for k, v in orig_to_frag.items()}
    rings = frag.GetRingInfo().AtomRings()
    if len(rings) != 1:
        return None
    ring = list(rings[0])
    for fi in ring:  # carbocyclic only
        if frag.GetAtomWithIdx(fi).GetAtomicNum() != 6:
            return None
    ring_set = set(ring)
    # Collect ring C=C bonds (frozenset pairs); any ring triple/aromatic bond is
    # out of this v1 scope (fail-closed -> a richer namer or abstain).
    double_pairs: Set[frozenset] = set()
    for bond in frag.GetBonds():
        if (bond.GetBeginAtomIdx() in ring_set
                and bond.GetEndAtomIdx() in ring_set):
            bt = bond.GetBondType()
            if bt == Chem.BondType.DOUBLE:
                double_pairs.add(frozenset(
                    (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
            elif bt != Chem.BondType.SINGLE:
                return None  # aromatic / triple ring bond -> out of scope
    fadj: Dict[int, list] = {a: [] for a in ring}
    for bond in frag.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in ring_set and j in ring_set:
            fadj[i].append(j)
            fadj[j].append(i)
    start = orig_to_frag[spiro_center]
    n = len(ring)
    stem = _get_chain_prefix(n)
    best = None  # ((spiro_key, suffix_key, ene_locant_tuple, subst_key), path)
    for first in sorted(fadj[start]):
        path = [start, first]
        visited = {start, first}
        cur = first
        while len(path) < n:
            nxt = next((nb for nb in fadj[cur] if nb not in visited), None)
            if nxt is None:
                break
            path.append(nxt)
            visited.add(nxt)
            cur = nxt
        if len(path) != n:
            continue
        pos = {a: i + 1 for i, a in enumerate(path)}
        # each ring C=C -> its lower endpoint locant; a bond closing n->1 is the
        # ring-closure double bond and would be cited at locant n.
        ene_locs = []
        for pr in double_pairs:
            a, b = tuple(pr)
            la, lb = pos[a], pos[b]
            # consecutive-in-ring endpoints: cite the lower, unless it is the
            # 1<->n closure bond (locant n).
            ene_locs.append(min(la, lb) if abs(la - lb) == 1 else n)
        key = tuple(sorted(ene_locs))
        spiro_key = tuple(sorted(pos[orig_to_frag[a]] for a in other_spiro_atoms))
        suffix_key = tuple(sorted(pos[orig_to_frag[a]] for a in suffix_atoms if a in orig_to_frag))
        subst_key = ()
        if substituent_tiebreak:
            subst_key = tuple(sorted(
                pos[orig_to_frag[a]] for a in component_atoms
                if a != spiro_center and a not in other_spiro_atoms
                and any(nb.GetIdx() not in component_atoms
                        for nb in mol.GetAtomWithIdx(a).GetNeighbors())))
        full_key = (spiro_key, suffix_key, key, subst_key)
        if best is None or full_key < best[0]:
            best = (full_key, path)
    if best is None:
        return None
    (_spiro_key, _suffix_key, ene_key, _subst_key), path = best
    a2l = {frag_to_orig[a]: i + 1 for i, a in enumerate(path)}
    if not double_pairs:
        return f"cyclo{stem}ane", a2l
    # mancude component: 'cyclohexa-2,5-diene' etc. Multiplied ene infix uses the
    # euphonic 'a' (di/tri-ene); a single C=C is 'cyclohex-2-ene'.
    from ..assembly.naming_utils import SIMPLE_MULTIPLIERS
    loc_str = ",".join(str(x) for x in ene_key)
    if len(ene_key) == 1:
        return f"cyclo{stem}-{loc_str}-ene", a2l
    mult = SIMPLE_MULTIPLIERS.get(len(ene_key))
    if mult is None:
        return None
    return f"cyclo{stem}a-{loc_str}-{mult}ene", a2l


def name_unsaturated_carbocyclic_side_ring(
    mol, side_ring, spiro_center: Optional[int] = None,
) -> Optional[Tuple[str, Dict[int, int]]]:
    """The component name of a carbocyclic spiro SIDE ring that has a double bond, with the
    double-bond locants in the ring's own numbering (``cyclohex-2-ene``,
    ``cyclohexa-2,4-diene``), for the builders that name a side ring on its own
    (``rules/spiro.py::_name_side_ring``).

    ``spiro_center`` is the ring's spiro atom when the caller knows it (a masked spiro atom
    is not in ``get_spiro_atoms``): it is locant 1. A ring with two spiro atoms (the middle ring
    of a polyspiro chain) is numbered from the one the caller passes, the junction toward the
    earlier-cited component,:10719); without one the choice would come from RDKit's
    ring order, so the ring is not named here. Returns None too when the ring has no known spiro
    atom or when ``name_carbocyclic_monocycle_component`` declines (a triple or aromatic ring
    bond): the caller then keeps its older name, which a read-back rejects."""
    ring_set = set(side_ring)
    spiro_set = set(get_spiro_atoms(mol))
    if spiro_center is not None:
        spiro_set.add(spiro_center)
    in_ring = [a for a in side_ring if a in spiro_set]
    if not in_ring or (len(in_ring) > 1 and spiro_center not in ring_set):
        return None
    start = spiro_center if spiro_center in ring_set else in_ring[0]
    others = tuple(sorted(a for a in in_ring if a != start))
    return name_carbocyclic_monocycle_component(
        mol, ring_set, start, other_spiro_atoms=others, substituent_tiebreak=True)
