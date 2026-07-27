"""Organyl substituent-prefix naming for the parent-hydride / oxoacid namers.

Single source of truth for the question "is this substituent an organyl that can
be cited as a detachable prefix, and if so what is its prefix name?".  Used by
the P-67 oxoacids (phosphonic / arsonic / stibonic …), the P-68 mononuclear
hydrides (arsane / stibane / silane), the P-21.2.2 chalcogen chains, the P-68.3
polyazanes and their siblings.

Two functions, deliberately not interchangeable:

``organyl_prefix_name``
    The NAMING primitive.  Delegates to the audited shared chokepoint
    ``assembly.substituent_enumerator.name_substituent`` — the same 5-tier
    cascade every ``general_engine`` locus uses — behind a strict acceptance
    filter, so a fragment the general machinery can already name honestly is no
    longer refused by a private approximation of it.

``is_simple_unbranched_organyl``
    A narrow ROUTING predicate.  True only for the small class the retired
    private walker accepted (unbranched alkyl attached at a terminus, or
    phenyl/naphthyl).  Kept for the one call site that uses the guard to choose
    *which producer* handles a fragment rather than to name it: widening a
    routing test silently changes which handler claims a molecule.

There is deliberately no third, deprecated entry point.  ``pure_organyl_prefix_name``
existed only as a shim while the nine families migrated (v29 Phase 3); all 23 of
its call sites now use one of the two functions above, so it was deleted rather
than left with zero callers -- a zero-caller shim is how the next session acquires
new ones.  ``_narrow_walk_name`` stays private and backs only the predicate.

What this module used to be, and why it changed
-----------------------------------------------
It used to wrap ``rules.phosphorus._characterize_substituent``, which follows
ONLY the carbon skeleton and therefore (a) silently DROPS a hanging heteroatom
(2-hydroxyethyl -> "ethyl"), (b) cannot tell propan-1-yl from propan-2-yl
(isopropyl -> "propyl"), and (c) miscounts a ring or benzyl as a linear alkyl
(cyclohexyl -> "hexyl", benzyl -> "heptyl").  A purity walk here rejected every
one of those, so the guard was correct — but the four refusals were artefacts of
the walker, not of nomenclature, and they fail-closed nine families at once.
The chokepoint has no such artefacts (it spells ``benzyl``, ``cyclohexylmethyl``,
``propan-2-yl``, ``naphthalen-1-yl``), so the walker is retired in its favour.

The bounded element set (and the P-41 reason for it)
----------------------------------------------------
``organyl_prefix_name`` accepts a fragment built only from **carbon, hydrogen and
halogen**, and fail-closes on every other element.  This is a nomenclature
constraint, not a walker artefact:

* P-41 lets the *principal characteristic group*, not the hub, decide the parent.
  A heteroatom in the organyl may carry a suffixable characteristic group that
  outranks the hub's, in which case citing the organyl as a mere prefix names a
  non-preferred structure — ``HO-CH2CH2-NH-NH2`` is ``2-hydrazinylethan-1-ol``,
  not ``(2-hydroxyethyl)hydrazine``, and ``HOOC-CH2-P(=O)(OH)2`` puts the
  carboxylic acid (senior inside P-41 class 7) in the suffix.
* A hydrocarbon fragment has no characteristic group at all, and the halogens
  are cited *only* as prefixes (P-59, Table 28), so neither can ever displace the
  hub's principal group.  Widening to them is therefore safe at every call site
  without a per-family seniority argument.

Extending the class to heteroatom organyls needs that per-family seniority
comparison; until it exists those fragments keep failing closed.
"""
import logging
from typing import List, Optional

from rdkit import Chem

logger = logging.getLogger(__name__)

# Halogens: P-59 Table 28 cites them only as prefixes, so a halogen can never
# demand a suffix and can never outrank the hub's principal characteristic group.
_PREFIX_ONLY_ELEMENTS = frozenset({'F', 'Cl', 'Br', 'I', 'At'})

# Rejected chokepoint answers.  A space means a multi-word functional-class name
# leaked out where a single prefix TOKEN is required; the refusal sentinels
# themselves are recognised by the shared ``errors.is_refusal_sentinel``.
#
# v29 P3B: this module's own ``('unknown', 'not supported')`` tuple was the
# FULLEST of the private copies of the refusal predicate — which is exactly why it
# had to go: being the most complete copy made it the one most likely to be
# mistaken for the definition. The sentinel families are now named in one place.


def _fragment_atoms(mol, start_idx: int, exclude_idx: int) -> Optional[List[int]]:
    """The connected subtree rooted at ``start_idx`` with ``exclude_idx`` as its
    only boundary, or None when the fragment leaves the C/H/halogen class.

    Also fail-closes on a charged or radical atom: an ion and a radical are
    SENIOR classes in P-41, so such a fragment is not a plain detachable prefix.

    Pure: no mol mutation.
    """
    subtree: List[int] = []
    seen = {exclude_idx}
    stack = [start_idx]
    while stack:
        i = stack.pop()
        if i in seen:
            continue
        seen.add(i)
        atom = mol.GetAtomWithIdx(i)
        sym = atom.GetSymbol()
        if sym != 'C' and sym not in _PREFIX_ONLY_ELEMENTS:
            return None
        if atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() != 0:
            return None
        subtree.append(i)
        for nbr in atom.GetNeighbors():
            j = nbr.GetIdx()
            if j == exclude_idx or nbr.GetSymbol() == 'H':
                continue
            stack.append(j)
    return subtree or None


def organyl_prefix_name(mol, start_idx: int, exclude_idx: int) -> Optional[str]:
    """Return the detachable prefix name for the organyl rooted at ``start_idx``
    and bonded to ``exclude_idx``, or None (fail-closed).

    ``start_idx`` is FRAGMENT-side and ``exclude_idx`` is its parent-side
    neighbour — the dominant ``attach_idx`` convention of the shared chokepoint.
    The convention is verified rather than trusted (the codebase holds one
    documented violation of it, ``SubstituentInfo.attach_mol_idx``), so a call
    whose two indices are not bonded fail-closes with a warning instead of
    naming some unrelated subtree.

    Pure: no mol mutation.
    """
    n_atoms = mol.GetNumAtoms()
    if not (0 <= start_idx < n_atoms) or not (0 <= exclude_idx < n_atoms):
        return None
    if mol.GetBondBetweenAtoms(start_idx, exclude_idx) is None:
        logger.warning(
            "organyl_prefix_name: atoms %d and %d are not bonded; the caller "
            "must pass the FRAGMENT-side atom first and its parent-side "
            "neighbour second", start_idx, exclude_idx)
        return None
    # An ORGANYL is carbon-attached by definition (P-29.2): a halogen is allowed
    # inside the fragment but never as the attachment atom, so a terminal halide
    # keeps going to its caller's own halogen-prefix handling.
    if mol.GetAtomWithIdx(start_idx).GetSymbol() != 'C':
        return None

    frag = _fragment_atoms(mol, start_idx, exclude_idx)
    if frag is None:
        return None
    if start_idx not in frag:                    # unreachable by construction
        return None

    from ..assembly.substituent_enumerator import name_substituent
    name = name_substituent(mol, frag, start_idx, allow_mancude=False)

    # Strict acceptance filter: only a single prefix TOKEN is usable here.
    from ..errors import is_refusal_sentinel
    if is_refusal_sentinel(name):
        return None
    if ' ' in name:
        return None
    return name


def _narrow_walk_name(mol, start_idx: int, exclude_idx: int) -> Optional[str]:
    """The retired private walker, kept ONLY to back the narrow routing predicate.

    Returns the prefix name iff the substituent is a pure-hydrocarbon unbranched
    alkyl attached at a chain terminus, or a phenyl/naphthyl aryl, else None.
    Every guard here narrows; see the module docstring for why the class it
    accepts is smaller than the one nomenclature allows.
    """
    from .phosphorus import _characterize_substituent

    subtree = []
    seen = {exclude_idx}
    stack = [start_idx]
    while stack:
        i = stack.pop()
        if i in seen:
            continue
        seen.add(i)
        atom = mol.GetAtomWithIdx(i)
        if atom.GetSymbol() != 'C':          # only a carbon skeleton is allowed
            return None
        subtree.append(i)
        for nbr in atom.GetNeighbors():
            j = nbr.GetIdx()
            if j == exclude_idx or nbr.GetSymbol() == 'H':
                continue
            stack.append(j)

    if not subtree:
        return None

    if not all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in subtree):
        subtree_set = set(subtree)

        def _sub_carbons(idx):
            return [n for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                    if n.GetIdx() in subtree_set]

        # The attachment atom must be a chain TERMINUS (<=1 in-subtree carbon):
        # _characterize_substituent emits the n-alkyl name, so an INTERNAL
        # attachment (isopropyl / sec-butyl) would be mislabelled.
        if len(_sub_carbons(start_idx)) > 1:
            return None
        for i in subtree:
            atom = mol.GetAtomWithIdx(i)
            # a ring in a non-fully-aromatic subtree (cyclohexyl, benzyl) is
            # miscounted as a linear alkyl -> fail-closed.
            if atom.IsInRing():
                return None
            if len(_sub_carbons(i)) > 2:     # branch point -> fail-closed
                return None
            for bond in atom.GetBonds():
                if (bond.GetBondType() != Chem.BondType.SINGLE
                        and not bond.GetIsAromatic()):
                    return None              # vinyl / alkynyl -> mislabel risk
    result = _characterize_substituent(mol, start_idx, {exclude_idx})
    return result[1] if result is not None else None


def is_simple_unbranched_organyl(mol, start_idx: int, exclude_idx: int) -> bool:
    """True iff the substituent is an unbranched alkyl attached at a terminus, or
    a phenyl/naphthyl aryl — the narrow class the retired walker accepted.

    A ROUTING predicate, not a namer.  Used where the answer selects a different
    producer (the contracted alkoxy-prefix builder), whose behaviour on
    ring-bearing, branched or unsaturated input is not established; widening the
    test there would change which producer claims a molecule rather than fix a
    refusal.
    """
    return _narrow_walk_name(mol, start_idx, exclude_idx) is not None


__all__ = [
    "organyl_prefix_name",
    "is_simple_unbranched_organyl",
]
