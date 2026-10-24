"""The numbering criteria of a von Baeyer cage that follow the detachable prefixes' locant set.

``VonBaeyerAnalyzer._locant_criteria_key`` ranks the numberings that leaves open on
 and on (c) principal groups, (e) unsaturation and (f) the locant set of the
detachable prefixes. When those tie, the Blue Book has four more criteria, (g) to (j), and an engine that
stops at (f) leaves the choice to the input atom order, so one molecule gets one name per SMILES
spelling (``3-bromo-5-chloro...`` or ``5-bromo-3-chloro...`` for one cage).

 NUMBERING (the Blue Book), after (f) "detachable alphabetized prefixes, all considered
together in a series of increasing numerical order" (:3300):

* (g) (:3307) "lowest locants for the substituent cited first as a prefix in the name", with
  ``1-methyl-4-nitronaphthalene (PIN) (not 4-methyl-1-nitronaphthalene)``. The prefixes are cited in
  alphanumerical order, ``alpha_sort_key``); after the first, the next cited is compared
  in turn.
* (h) (:3319) "When a choice is needed between the same skeletal atom in different valence states, the
  one in a nonstandard valence state is assigned the lower locant... the one in the higher valence
  state" (the atoms that carry a cited lambda, ``ring_replacement.vb_lambda_for_atom``).
* (i) (:3340) isotopically modified atoms take the lowest locants considered together as a set, then the
  higher atomic number, then the higher mass number.
* (j) (:3346) "the lower locant is assigned to CIP stereodescriptors Z, R, M, and r (pseudoasymmetry)
  that are preferred to E, S, P, and s" (``naming_utils.cip_locant_rank_key``).

The Blue Book decides nothing beyond (j); the canonical atom ranks that come last only stop the atom
order from deciding. Numberings still tied after (g) to (j) that are related by an automorphism of the
whole structure give one name; when they are not, a criterion that is not encoded here (the citation
order of the components of a spiro system, say) tells them apart, and the first keeps the choice the
analysis made.

(j) reads the stereodescriptors of the branches as well as those of the cage: ``1-[(2R)-butan-2-yl]-3-[(2S)-butan-2-yl]benzene
(PIN)`` (:3399, an example of (j)) takes the R prefix to the lower locant. A descriptor of a branch is keyed at the
locant of the cage atom that carries the branch, then by its place in the name of the branch (``(2R,3S)-...`` cites
2R before 3S), after the descriptors of the cage atom itself.

Cost. Naming a branch is the full substituent cascade, which can itself analyse a cage (a branch that
holds a carbazole, say), so the criteria are applied lazily: (g) needs names only when the tied numberings
place DIFFERENT branches differently, which a name-free signature of each branch (its canonical SMILES, the
order of the bond to the cage and, for a branch that holds stereo, its isomeric SMILES) tells first; the CIP
labels are computed only for numberings that still tie.

What the Blue Book would decide but this module cannot. A branch of more than ``MAX_NAMED_BRANCH_ATOMS`` heavy
atoms is not named, so (g) (and the descriptors of a branch at (j)) are unknown for that tie; a branch that loops
back into the cage (a spiro component) has no prefix name at all; CIP labels may be unavailable. When such a
criterion is unknown and the tied numberings are not related by an automorphism of the whole structure, the
numbering is chosen by the canonical atom order of the analysis, which does not depend on the input atom order,
and the descriptor is marked ``numbering_certified = False``: the name is not a preferred name (the
caller records it as a non-PIN fragment) because the Blue Book may prefer the other numbering.
"""
import logging
from typing import Dict, List, Optional, Set, Tuple

from rdkit import Chem

logger = logging.getLogger(__name__)

#: a branch larger than this is not named for (g): the cascade that names it is the cost the
#: tie-break must not multiply, and a tie that only (g) could break and that holds such a branch
#: is left to (h) to (j), the canonical atom order of the analysis, and a label that says it is not
#: a preferred name.
MAX_NAMED_BRANCH_ATOMS = 30


def _stereo_signature(mol, branch, idx: int) -> str:
    """The stereo of a branch as the canonical isomeric SMILES of the branch with its cage atom ``idx`` as a
    dummy atom, or ``''`` when the branch holds none. The cage atom must be in the fragment: a chiral first
    atom and the double bond next to the cage have the cage atom as a neighbour, and a SMILES of the branch
    alone leaves both unwritten. A double bond between ``idx`` and the branch is not in the string (its
    stereo atoms are on the cage); it is the cage's, and (j) reads it there."""
    holds = any(mol.GetAtomWithIdx(a).GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in branch)
    if not holds:
        holds = any(b.GetStereo() != Chem.BondStereo.STEREONONE and b.GetBeginAtomIdx() in branch
                    and b.GetEndAtomIdx() in branch for b in mol.GetBonds())
    if not holds:
        return ""
    marked = Chem.RWMol(mol)
    cage = marked.GetAtomWithIdx(idx)
    cage.SetAtomicNum(0)
    cage.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    cage.SetNoImplicit(True)
    cage.SetNumExplicitHs(0)
    cage.SetFormalCharge(0)
    cage.SetIsotope(0)
    cage.SetIsAromatic(False)
    return Chem.MolFragmentToSmiles(marked, atomsToUse=sorted(set(branch) | {idx}), rootedAtAtom=idx,
                                    isomericSmiles=True, canonical=True)


def cage_branches(mol, ring_atoms: Set[int], suffix_pairs: Set[Tuple[int, int]],
                  substituent_atoms: Optional[Set[int]] = None):
    """The detachable prefixes of the cage: ``{cage atom: [(signature, branch atoms, first atom)]}``.

    A branch is a prefix unless its first atom is the principal characteristic group itself (a
    ``(cage atom, first atom)`` pair of ``suffix_pairs``), or lies outside ``substituent_atoms`` (the cage
    is named as a substituent and that is its parent). ``signature`` identifies the branch without
    naming it: ``(order of the bond to the cage, canonical SMILES, stereo)``, so equal signatures are
    equal branches and two branches that differ only by stereo are two. ``None`` when a branch loops back
    into the ring or a second bond joins it to its cage atom, so that the caller never ranks a partial
    set of prefixes.
    """
    branches: Dict[int, list] = {}
    for idx in sorted(ring_atoms):
        for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
            first = neighbor.GetIdx()
            if first in ring_atoms or neighbor.GetAtomicNum() <= 1 or (idx, first) in suffix_pairs:
                continue
            if substituent_atoms is not None and first not in substituent_atoms:
                continue
            branch = {first}
            stack = [first]
            while stack:
                current = stack.pop()
                for nxt in mol.GetAtomWithIdx(current).GetNeighbors():
                    k = nxt.GetIdx()
                    if k in ring_atoms:
                        if k != idx or current != first:
                            return None      # a bridge back into the ring, or a second bond to this atom
                        continue
                    if k not in branch:
                        branch.add(k)
                        stack.append(k)
            order = mol.GetBondBetweenAtoms(idx, first).GetBondTypeAsDouble()
            signature = (order,
                         Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(branch), rootedAtAtom=first,
                                                  isomericSmiles=False, canonical=True),
                         _stereo_signature(mol, branch, idx))
            branches.setdefault(idx, []).append((signature, frozenset(branch), first))
    return branches


def aromatic_view(mol):
    """``mol`` with its aromaticity perceived again: the cage is analysed on a kekulized copy, in which a
    phenyl branch is a cyclohexatrienyl one and would be cited under ``c`` instead of ``p``."""
    view = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(view)
    except Exception:  # noqa: BLE001 -- the branches are then judged as given
        return mol
    return view


def _branch_names(mol, branches, only_stereo: bool = False) -> Optional[Dict[tuple, str]]:
    """``{signature: the prefix name, stereodescriptors included}``; ``None`` when one cannot be had.

    ``only_stereo`` names just the branches that hold stereo (the descriptors of a branch at (j))."""
    try:
        from ..assembly.substituent_enumerator import name_substituent_for_ordering
    except Exception:  # noqa: BLE001 -- a missing helper means the criterion is unknown, not a crash
        return None
    names: Dict[tuple, str] = {}
    for entries in branches.values():
        for signature, atoms, first in entries:
            if signature in names or (only_stereo and not signature[2]):
                continue
            if len(atoms) > MAX_NAMED_BRANCH_ATOMS:
                return None
            name = name_substituent_for_ordering(mol, sorted(atoms), first)
            if not name:
                return None
            names[signature] = name
    return names


def _name_keys(mol, branches) -> Optional[Dict[tuple, str]]:
    """``{signature: alphanumerical sort key of the prefix name}``; ``None`` when one cannot be had."""
    names = _branch_names(mol, branches)
    if names is None:
        return None
    try:
        from ..assembly.naming_utils import alpha_sort_key
    except Exception:  # noqa: BLE001
        return None
    return {signature: alpha_sort_key(name) for signature, name in names.items()}


def _numbered_structure(mol, desc) -> str:
    """The canonical SMILES of ``mol`` with each numbered atom marked by its locant: equal for two numberings
    exactly when they differ by an automorphism of the whole structure."""
    try:
        marked = Chem.RWMol(mol)
        for idx, locant in desc.numbering.items():
            marked.GetAtomWithIdx(idx).SetAtomMapNum(locant)
        return Chem.MolToSmiles(marked)
    except Exception:  # noqa: BLE001 -- not comparable: the numberings are then taken as different
        return repr(sorted(desc.numbering.items()))


def choose_numbering(mol, ring_atoms: Set[int], candidates: list, suffix_pairs: Set[Tuple[int, int]],
                     substituent_atoms: Optional[Set[int]] = None):
    """The descriptor among the tied ``candidates`` that (g), (h), (i), then (j) prefer.

    Each candidate has a ``numbering`` (atom index to locant). Candidates with one numbering are one.
    ``candidates`` come in the order of the analysis, which the caller makes independent of the input atom
    order; the first of the numberings that nothing ranks keeps that choice. The chosen descriptor has
    ``numbering_certified = False`` when a criterion the Blue Book ranks could not be evaluated for a tie
    between numberings that are not an automorphism apart (see the module docstring).
    """
    distinct: Dict[tuple, object] = {}
    for desc in candidates:
        distinct.setdefault(tuple(sorted(desc.numbering.items())), desc)
    pool: List = list(distinct.values())
    if len(pool) < 2:
        return pool[0] if pool else None
    tied = list(pool)
    unknown: List[str] = []         # the criteria that could not be evaluated for this tie

    def chosen(desc):
        if unknown and len({_numbered_structure(mol, d) for d in tied}) > 1:
            desc.numbering_certified = False
            logger.debug("cage numbering chosen without %s: not a preferred name", unknown)
        return desc

    # (g): the locants of the prefixes in citation order, the branches named as the molecule is written
    view = aromatic_view(mol)
    branches = cage_branches(view, ring_atoms, suffix_pairs, substituent_atoms)
    if branches is None:
        unknown.append("g")
    elif branches:
        def placed(desc):
            return tuple(sorted((signature, desc.numbering[idx])
                                for idx, entries in branches.items() if idx in desc.numbering
                                for signature, _atoms, _first in entries))
        if len({placed(d) for d in pool}) > 1:      # the tied numberings place different branches differently
            name_keys = _name_keys(view, branches)
            if name_keys is None:
                unknown.append("g")
            else:
                def g_key(desc):
                    return tuple(sorted((name_keys[signature], desc.numbering[idx])
                                        for idx, entries in branches.items() if idx in desc.numbering
                                        for signature, _atoms, _first in entries))
                keys = [g_key(d) for d in pool]
                best = min(keys)
                pool = [d for d, k in zip(pool, keys) if k == best]
                if len(pool) == 1:
                    return chosen(pool[0])

    # (h) nonstandard valence states, then (i) isotopic modification
    from .ring_replacement import vb_lambda_for_atom

    def h_key(desc):
        lam = sorted((desc.numbering[i], vb_lambda_for_atom(mol, i)) for i in ring_atoms
                     if i in desc.numbering and vb_lambda_for_atom(mol, i) is not None)
        return tuple(loc for loc, _ in lam), tuple(-v for _, v in lam)

    def i_key(desc):
        marked = []
        for i in ring_atoms:
            if i not in desc.numbering:
                continue
            atom = mol.GetAtomWithIdx(i)
            for modified in [atom] + [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]:
                if modified.GetIsotope():
                    marked.append((desc.numbering[i], -modified.GetAtomicNum(), -modified.GetIsotope()))
        marked.sort()
        return tuple(loc for loc, _, _ in marked), tuple((z, m) for _, z, m in marked)

    for key in (h_key, i_key):
        keys = [key(d) for d in pool]
        best = min(keys)
        pool = [d for d, k in zip(pool, keys) if k == best]
    if len(pool) == 1:
        return chosen(pool[0])

    # (j): CIP stereodescriptors at the lowest locants
    if any(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in mol.GetAtoms()) or any(
            b.GetStereo() != Chem.BondStereo.STEREONONE for b in mol.GetBonds()):
        try:
            from ..assembly.naming_utils import cip_descriptor_codes, cip_locant_rank_key
            from ..perception.stereo import assign_stereochemistry
            assign_stereochemistry(mol)
            # the descriptors the name cites at cage locants: each stereogenic ring atom, each ring
            # double bond (at its lower locant) and each exocyclic double bond (at the cage atom it
            # leaves, the 3Z of a 3-ylidene)
            bonds = []
            for bond in mol.GetBonds():
                a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if bond.GetBondTypeAsDouble() > 1.0 and bond.HasProp('_CIPCode') and (a in ring_atoms or b in ring_atoms):
                    bonds.append((a, b))
            # the descriptors a branch cites, in the order its name cites them: a branch that holds stereo is
            # named, and a cage with such a branch the Blue Book would rank by it is left unknown when the
            # name cannot be had
            branch_codes: Dict[tuple, tuple] = {}
            if branches:
                names = _branch_names(view, branches, only_stereo=True)
                if names is None:
                    unknown.append("j")
                else:
                    branch_codes = {signature: cip_descriptor_codes(name) for signature, name in names.items()}

            def j_key(desc):
                numbering = desc.numbering
                # (locant, 0) is the cage's own descriptor, (locant, k) the k-th of a branch on that atom
                items = [((numbering[i], 0), mol.GetAtomWithIdx(i).GetProp('_CIPCode'))
                         for i in numbering if i in ring_atoms and mol.GetAtomWithIdx(i).HasProp('_CIPCode')]
                for a, b in bonds:
                    cited = [numbering[i] for i in (a, b) if i in ring_atoms and i in numbering]
                    if cited:
                        items.append(((min(cited), 0), mol.GetBondBetweenAtoms(a, b).GetProp('_CIPCode')))
                for idx, entries in (branches or {}).items():
                    if idx not in numbering:
                        continue
                    for signature, _atoms, _first in entries:
                        for k, code in enumerate(branch_codes.get(signature, ()), start=1):
                            items.append(((numbering[idx], k), code))
                return cip_locant_rank_key(items)
            keys = [j_key(d) for d in pool]
            best = min(keys)
            pool = [d for d, k in zip(pool, keys) if k == best]
        except Exception:  # noqa: BLE001 -- no labels: (j) is unknown
            logger.debug("CIP labels unavailable for the numbering tie-break", exc_info=True)
            unknown.append("j")
    if len(pool) == 1:
        return chosen(pool[0])

    # nothing the Blue Book ranks is left. Numberings that number the whole structure alike (an automorphism
    # apart) give one name, and the canonical ranks pick one of them so that the atom order cannot decide.
    # Numberings that still number it differently are told apart by a criterion that is not encoded here
    # (the citation order of the components of a spiro system, say): the first keeps the choice the analysis
    # made, which is the choice that was made before this module ranked anything.
    if len({_numbered_structure(mol, d) for d in pool}) > 1:
        return chosen(pool[0])
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=True))
    return chosen(min(pool, key=lambda d: tuple(ranks[i] for i in sorted(d.numbering, key=d.numbering.get))))
