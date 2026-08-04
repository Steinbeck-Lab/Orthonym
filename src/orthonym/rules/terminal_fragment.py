"""v30 Phase B: the TERMINAL FRAGMENT namer -- the complete-by-construction
fallback that stands where a declining producer used to DROP a substituent.

Why this module exists
----------------------
Phase A measured that 132 of the 162 largest-class abstentions (81.5%) are silent
ATOM DROPS, median -8 heavy atoms, and that they are multi-blocked at the code
site: mean 2.69 distinct ``DROP-*`` codes per row, best single-site fix 10/132
(8%). All 17 ``substituent_skip`` sites ``continue`` when a narrow producer
declines a fragment, which removes that fragment's atoms from the name; because
selection is ``first_applicable`` at pool size 1, each skip is terminal and
SELF-01 then correctly rejects the atom-short name.

So the lever is not a fix list. It is one fallback that always accounts for every
atom -- the generalisation of the doctrine ``rules/terminal_ring.py`` already
states: *a table miss degrades to an UGLIER name instead of a refusal*
(the contributor guide invariant 1, T4 clause).

Contract
--------
``terminal_fragment_name`` returns a substituent prefix token accounting for
EVERY atom in ``frag_atoms``, or ``None``. It returns ``None`` -- never a partial
name -- when the fragment is out of the implemented scope, when an element has no
admitted morpheme (Table 1.5 is CLOSED, so fail-closed is the only sound design),
or when the result fails its audit. It never raises: 17 call sites depend on it
degrading rather than crashing.

This is the T4 best-effort tier. The backbone choice is deterministic, NOT the
P-44 seniority cascade: a deterministic-and-complete name beats a
PIN-optimal-or-absent one here, and tier labels express preference, never
correctness.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Set

from rdkit import Chem

from ..assembly.naming_utils import alpha_sort_key, enclose_if_compound, get_alkyl_name
from .ring_replacement import build_replacement_prefix

logger = logging.getLogger(__name__)

#: Mirrors ``terminal_ring.MAX_CAGE_ATOMS``'s intent: a bound past which a
#: systematic name is neither useful nor cheap to audit.
MAX_FRAGMENT_ATOMS = 40


@dataclass(frozen=True)
class TerminalFragmentName:
    """One audited terminal fragment name.

    ``name``       the emitted substituent prefix token ('2-oxabutyl').
    ``numbering``  atom idx -> backbone locant, the SAME map the name was
                   spelled from (never re-derived), so a consumer can place
                   further locants consistently.
    ``basis``      which generator produced it: 'chain' (this task), later
                   'ring' or 'composite'.
    ``atoms``      every fragment atom the name accounts for. The completeness
                   invariant is ``atoms == frozenset(frag_atoms)``; it is
                   asserted before returning, because an atom-short name is
                   exactly the defect this module removes.
    """

    name: str
    numbering: Dict[int, int]
    basis: str
    atoms: FrozenSet[int]


def _canonical_ranks(mol) -> Sequence[int]:
    """Spelling-independent atom ordering, for deterministic tie-breaks.

    Atom-index order varies with the SMILES spelling, and ordering by it is how a
    latent P-14.4(g) tie became a live nondeterminism once before (v29 Phase 6).
    """
    return list(Chem.CanonicalRankAtoms(mol, breakTies=True))


def _backbone_from(mol, atoms: Set[int], start: int,
                   ranks: Sequence[int]) -> List[int]:
    """The deepest simple path from ``start`` inside ``atoms``.

    ``atoms`` is acyclic here, so the induced subgraph rooted at ``start`` is a
    tree and the deepest root-to-leaf path is the backbone. Ties are broken on
    the path's canonical-rank sequence, never on atom index.
    """
    best_path: List[int] = [start]
    best_key = (1, tuple([ranks[start]]))

    def walk(cur: int, path: List[int], seen: Set[int]) -> None:
        nonlocal best_path, best_key
        children = [nb.GetIdx() for nb in mol.GetAtomWithIdx(cur).GetNeighbors()
                    if nb.GetIdx() in atoms and nb.GetIdx() not in seen]
        if not children:
            key = (len(path), tuple(ranks[i] for i in path))
            if key > best_key:
                best_key, best_path = key, list(path)
            return
        for j in sorted(children, key=lambda k: ranks[k]):
            seen.add(j)
            path.append(j)
            walk(j, path, seen)
            path.pop()
            seen.discard(j)

    walk(start, [start], {start})
    return best_path


def _is_acyclic(mol, atoms: Set[int]) -> bool:
    ri = mol.GetRingInfo()
    return not any(ri.NumAtomRings(a) > 0 for a in atoms)


def _branches_off(mol, frag: Set[int], backbone: Sequence[int]):
    """(backbone_locant, branch_attachment_atom, branch_atom_set) per branch.

    A branch is a connected component of ``frag - backbone`` together with the
    backbone atom it hangs from. Collected by BFS so a branch of any depth comes
    out whole -- a depth cap here would silently truncate atoms.
    """
    on_backbone = set(backbone)
    locant_of = {a: i + 1 for i, a in enumerate(backbone)}
    out = []
    claimed: Set[int] = set()
    for b in backbone:
        for nb in mol.GetAtomWithIdx(b).GetNeighbors():
            j = nb.GetIdx()
            if j in on_backbone or j not in frag or j in claimed:
                continue
            comp, stack = set(), [j]
            while stack:
                cur = stack.pop()
                if cur in comp:
                    continue
                comp.add(cur)
                for nb2 in mol.GetAtomWithIdx(cur).GetNeighbors():
                    k = nb2.GetIdx()
                    if k in frag and k not in on_backbone and k not in comp:
                        stack.append(k)
            claimed |= comp
            out.append((locant_of[b], j, comp))
    return out


def _assemble_prefixes(tokens) -> str:
    """'2-methyl', '2,2-dimethyl', '1-ethyl-2-methyl' from (key, locant, name).

    Identical tokens collapse onto one multiplied prefix with their locants
    ascending; distinct tokens are ordered by the project's P-14.5 key, which
    already strips multiplying prefixes, Greek letters and CIP descriptors.

    Multiplier table selection is P-16.3.3(b)/P-16.3.5(a) (BlueBookV2.md:4857):
    a compound/complex branch token -- already wrapped in P-16.5.1.1 marks by
    the caller (``enclose_if_compound``), so it opens with '(', '[' or '{' --
    is multiplied with the DERIVED table (``bis``/``tris``/...), never the
    basic ``di``/``tri`` table reserved for simple prefixes. Reusing
    ``get_bracket_depth`` here keeps the compound test the SAME decision
    ``enclose_if_compound`` already made, rather than re-deriving it.

    NO trailing hyphen. Hyphens separate one prefix from the NEXT prefix, never a
    prefix from the stem it qualifies: the substituent group CC(C)C- is
    ``2-methylpropyl`` (isobutyl), not ``2-methyl-propyl``. Two prefixes still
    read ``1-ethyl-2-methylpropyl`` because the join supplies the internal
    hyphen. An earlier draft appended '-' here and produced the malformed form on
    all four branched cases.
    """
    from ..assembly.naming_utils import COMPLEX_MULTIPLIERS, SIMPLE_MULTIPLIERS, get_bracket_depth

    grouped: Dict[str, List[int]] = {}
    keys: Dict[str, str] = {}
    for key, locant, tok in tokens:
        grouped.setdefault(tok, []).append(locant)
        keys[tok] = key
    parts = []
    for tok in sorted(grouped, key=lambda t: keys[t]):
        locs = sorted(grouped[tok])
        if len(locs) > 1:
            table = COMPLEX_MULTIPLIERS if get_bracket_depth(tok) > 0 else SIMPLE_MULTIPLIERS
            mult = table.get(len(locs), str(len(locs)))
        else:
            mult = ""
        parts.append(f"{','.join(str(x) for x in locs)}-{mult}{tok}")
    return "-".join(parts)


def _join_prefix_block(block: str, stem: str) -> str:
    """Concatenate a substituent-prefix block onto the stem it qualifies.

    A hyphen goes in IFF the stem begins with a LOCANT, because a locant is
    always separated from preceding alphabetic text:

        '2-methyl'  + 'propyl'      -> '2-methylpropyl'     (isobutyl)
        '3-methyl'  + '2-oxabutyl'  -> '3-methyl-2-oxabutyl'

    Both malformed directions were produced while getting this right: always
    appending a hyphen gave '2-methyl-propyl', and never appending one gave
    '3-methyl2-oxabutyl', which runs a letter straight into a digit.
    """
    if not block:
        return stem
    return f"{block}-{stem}" if stem[:1].isdigit() else f"{block}{stem}"


def terminal_fragment_name(
    mol, frag_atoms, attach_idx: int,
) -> Optional[TerminalFragmentName]:
    """Name ``frag_atoms`` as a substituent prefix attached at ``attach_idx``.

    Returns ``None`` rather than a partial name for anything out of scope. Never
    raises.
    """
    try:
        return _terminal_fragment_name(mol, frag_atoms, attach_idx)
    except Exception:                                   # noqa: BLE001
        # 17 call sites: degrade, never crash. Logged so a bug is visible in the
        # census instead of looking like an honest decline.
        logger.exception("terminal_fragment: unexpected error; refusing")
        return None


def _terminal_fragment_name(
    mol, frag_atoms, attach_idx: int,
) -> Optional[TerminalFragmentName]:
    if mol is None or not frag_atoms:
        return None
    frag = set(frag_atoms)
    if attach_idx not in frag:
        return None
    if len(frag) > MAX_FRAGMENT_ATOMS:
        logger.info("terminal_fragment: %d atoms > MAX_FRAGMENT_ATOMS; refuse",
                    len(frag))
        return None
    if any(a >= mol.GetNumAtoms() for a in frag):
        return None
    # SCOPE, mirroring terminal_ring.py:582: a charged skeletal atom is P-73,
    # not replacement nomenclature, and naming it neutral denotes a different
    # species.
    if any(mol.GetAtomWithIdx(a).GetFormalCharge() != 0 for a in frag):
        return None
    if not _is_acyclic(mol, frag):
        return None                      # Task 4 adds ring systems

    ranks = _canonical_ranks(mol)
    backbone = _backbone_from(mol, frag, attach_idx, ranks)
    branches = _branches_off(mol, frag, backbone)

    # Saturated only in this task: an unspelled multiple bond would denote a
    # different molecule, so refuse rather than emit the saturated name.
    for i in range(len(backbone) - 1):
        b = mol.GetBondBetweenAtoms(backbone[i], backbone[i + 1])
        if b is None or b.GetBondType() != Chem.BondType.SINGLE:
            return None                  # Task 3 adds unsaturation

    numbering = {a: i + 1 for i, a in enumerate(backbone)}
    rp = build_replacement_prefix(mol, numbering, set(backbone))
    if rp.unexpressed:
        logger.info(
            "terminal_fragment: %d backbone atom(s) have no admitted morpheme; "
            "refuse (Table 1.5 is closed)", len(rp.unexpressed))
        return None

    # The stem counts EVERY skeletal atom, carbon and heteroatom alike -- that is
    # what makes replacement nomenclature complete (2-oxabutyl has 4 skeletal
    # atoms: C-O-C-C). Verified against OPSIN.
    stem = get_alkyl_name(len(backbone))
    name = f"{rp.prefix}{stem}" if rp.prefix else stem

    accounted = set(backbone)
    prefix_tokens: List[tuple] = []          # (alpha_key, locant, token)
    for locant, battach, comp in branches:
        sub = _terminal_fragment_name(mol, comp, battach)
        if sub is None:
            # Refuse the WHOLE fragment. Emitting the backbone alone would drop
            # this branch's atoms -- the exact defect this module removes.
            logger.info("terminal_fragment: branch at locant %d unnameable; "
                        "refusing the whole fragment", locant)
            return None
        accounted |= set(sub.atoms)
        # P-16.5.1.1 (BlueBookV2.md:7232): "Parentheses (round brackets) ...
        # are used around compound (P-29.1.2) and complex (P-29.1.3) prefixes"
        # -- a recursively-named branch must be enclosed BEFORE it is spliced
        # in as a token, or the assembled name reads as an unparseable run-on
        # (e.g. '5-1-methylethylnonyl' instead of '5-(1-methylethyl)nonyl').
        # The sort key is computed on the pre-enclosure name; alpha_sort_key
        # strips enclosing marks itself, so this is not an approximation.
        token = enclose_if_compound(sub.name)
        prefix_tokens.append((alpha_sort_key(sub.name), locant, token))

    if prefix_tokens:
        name = _join_prefix_block(_assemble_prefixes(prefix_tokens), name)

    result = TerminalFragmentName(name=name, numbering=numbering,
                                  basis="chain", atoms=frozenset(accounted))
    if result.atoms != frozenset(frag):
        logger.error("terminal_fragment: completeness invariant violated "
                     "(%d named of %d); refuse",
                     len(result.atoms), len(frag))
        return None
    return result
