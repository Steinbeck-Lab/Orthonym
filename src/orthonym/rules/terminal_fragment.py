""" Phase B: the TERMINAL FRAGMENT namer -- the complete-by-construction
fallback that stands where a declining producer used to DROP a substituent.

Why this module exists
----------------------
Phase A measured that 132 of the 162 largest-class abstentions (81.5%) are silent
ATOM DROPS, median -8 heavy atoms, and that they are multi-blocked at the code
site: mean 2.69 distinct producer-refusal codes per row, best single-site fix 10/132
(8%). All 17 ``substituent_skip`` sites ``continue`` when a narrow producer
declines a fragment, which removes that fragment's atoms from the name; because
selection is ``first_applicable`` at pool size 1, each skip is terminal and
 then correctly rejects the atom-short name.

So the lever is not a fix list. It is one fallback that always accounts for every
atom -- the generalisation of the doctrine ``rules/terminal_ring.py`` already
states: *a table miss degrades to an UGLIER name instead of a refusal*
(the contributor guide a project rule, clause).

Contract
--------
``terminal_fragment_name`` returns a substituent prefix token accounting for
EVERY atom in ``frag_atoms``, or ``None``. It returns ``None`` -- never a partial
name -- when the fragment is out of the implemented scope, when an element has no
admitted morpheme (Table 1.5 is CLOSED, so fail-closed is the only sound design),
or when the result fails its audit. It never raises: 17 call sites depend on it
degrading rather than crashing.

This is the best-effort tier. The backbone choice is deterministic, NOT the
 seniority cascade: a deterministic-and-complete name beats a
PIN-optimal-or-absent one here, and tier labels express preference, never
correctness.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Set

from rdkit import Chem

from ..assembly.book_prefixes import (
    CHALCOGEN_HEAD,
    HETERO_ROOT_VALENCE,
    MONONUCLEAR_HEAD,
    a_chain_licensed,
    hetero_roots_composable,
    forms_enabled,
)
from ..assembly.naming_utils import alpha_sort_key, enclose_if_compound, get_alkyl_name
from .ring_replacement import build_replacement_prefix

# Monovalent halogens (F/Cl/Br/I/At) are NEVER skeletal replacement-chain atoms
# / skeletal replacement uses only C and the 'a'-elements). A
# halogen is a terminal substituent expressed as a fluoro/chloro/... prefix, so
# the backbone walk must not step onto one -- doing so miscounts the chain length
# and drops halogens (``-O-CH2-CF3`` -> ``3,3-difluoro-1-oxabutan-1-yl``, a
# 4-atom/2-F WRONG constitution for the real 3-atom/3-F chain). Wave F.
_HALOGEN_ATOMIC_NUMS = frozenset({9, 17, 35, 53, 85})

logger = logging.getLogger(__name__)

#: Mirrors ``terminal_ring.MAX_CAGE_ATOMS``'s intent: a bound past which a
#: systematic name is neither useful nor cheap to audit.
MAX_FRAGMENT_ATOMS = 40


@dataclass(frozen=True)
class TerminalFragmentName:
    """One audited terminal fragment name.

    ``name`` the emitted substituent prefix token ('2-oxabutyl').
    ``numbering`` atom idx -> backbone locant, the SAME map the name was
                   spelled from (never re-derived), so a consumer can place
                   further locants consistently.
    ``basis`` which generator produced it: 'chain' (this task), later
                   'ring' or 'composite'.
    ``atoms`` every fragment atom the name accounts for. The completeness
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
    latent (g) tie became a live nondeterminism once before (a phase).
    """
    return list(Chem.CanonicalRankAtoms(mol, breakTies=True))


def _backbone_from(mol, atoms: Set[int], start: int,
                   ranks: Sequence[int], stop_at_ring: bool = False,
                   carbon_only: bool = False) -> List[int]:
    """The deepest simple path from ``start`` inside ``atoms``.

    ``atoms`` is acyclic here, so the induced subgraph rooted at ``start`` is a
    tree and the deepest root-to-leaf path is the backbone. Ties are broken on
    the path's canonical-rank sequence, never on atom index.

    ``stop_at_ring`` (ring-branch lever): when the fragment carries ring
    systems but the attachment is acyclic, the backbone is the ACYCLIC chain and
    each ring system is a decoration. Ring atoms are then not valid backbone
    steps -- the walk stops at the ring boundary and ``_branches_off`` collects
    the whole ring system as one decoration component.
    """
    best_path: List[int] = [start]
    best_key = (1, tuple([ranks[start]]))

    def walk(cur: int, path: List[int], seen: Set[int]) -> None:
        nonlocal best_path, best_key
        children = [nb.GetIdx() for nb in mol.GetAtomWithIdx(cur).GetNeighbors()
                    if nb.GetIdx() in atoms and nb.GetIdx() not in seen
                    # A monovalent halogen is a substituent, never a skeletal
                    # backbone step (Wave F: fixes the trifluoroethoxy
                    # miscount -- see _HALOGEN_ATOMIC_NUMS).
                    and nb.GetAtomicNum() not in _HALOGEN_ATOMIC_NUMS
                    and not (stop_at_ring
                             and mol.GetAtomWithIdx(nb.GetIdx()).IsInRing())
                    # roadmap N5c: a substitutive backbone walks carbon atoms only
                    and not (carbon_only and nb.GetAtomicNum() != 6)]
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


def _has_defined_stereo(mol, frag: Set[int]) -> bool:
    """True if any atom or wholly-internal bond of ``frag`` carries DEFINED stereo.

    ⚠ **Do not relax this to raise coverage.** This module emits no
    stereodescriptors at all, and -- the gate that delivers 0-wrong --
    suppresses only on a *verified constitutional* mismatch
    (``namer.py:920-924``). A stereocentre is not a constitutional difference, so
    an achiral token emitted for a stereocentred fragment ships a DIFFERENT
    COMPOUND with every gate green. Nothing downstream catches it.

    Keys on DEFINED stereo, not on the presence of a double bond: an undefined
    double bond is ordinary Task 3 unsaturation and must still name, or the guard
    would swallow that whole feature.

    Only stereo INSIDE the fragment matters. A centre elsewhere in the molecule
    belongs to whatever names that part; blocking on it would refuse fragments
    this module handles correctly.

    Split (internal-C=C lever) into an ATOM half and a BOND half. A defined
    R/S centre is still never expressible here, but a defined BACKBONE C=C
    configuration is now emitted as a leading ``(nE)/(nZ)`` block on the ACYCLIC
    path, so only the atom half unconditionally refuses; the bond half still
    guards the composite/ring path, which emits no stereodescriptor.
    """
    return _has_defined_atom_stereo(mol, frag) or _has_defined_bond_stereo(mol, frag)


def _has_defined_atom_stereo(mol, frag: Set[int]) -> bool:
    """True if any atom of ``frag`` carries a DEFINED chiral tag. This module
    emits no R/S descriptor, so such a fragment always refuses (0-wrong)."""
    return any(
        mol.GetAtomWithIdx(idx).GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        for idx in frag)


def _has_defined_bond_stereo(mol, frag: Set[int]) -> bool:
    """True if any wholly-internal bond of ``frag`` carries a DEFINED
    configuration (E/Z / cis-trans)."""
    return any(
        bond.GetStereo() != Chem.BondStereo.STEREONONE
        and bond.GetBeginAtomIdx() in frag and bond.GetEndAtomIdx() in frag
        for bond in mol.GetBonds())


def _single_heteroatom_branch_prefix(mol, comp: Set[int]) -> Optional[str]:
    """The standard prefix for a branch that is ONE heteroatom, or None.

    ⚠ Without this, the branch loops below recurse ``_terminal_fragment_name`` on a
    lone ``=O`` and get back a one-atom oxa-replacement chain, ``1-oxamethyl`` --
    which is a **wrong molecule**, not merely an ugly name. Asked what it denotes,
    OPSIN answers:

        (1-oxamethyl)benzene -> OC1=CC=CC=C1 i.e. PHENOL
        (2-(1-oxamethyl)butyl)benzene -> OC(CC1=CC=CC=C1)CC i.e. an ALCOHOL

    So ``1-oxamethyl`` denotes ``-OH``: emitting it for ``=O`` loses the double bond
    *and* the carbon, turning a ketone into an alcohol. Measured, the construction was
    wrong wherever it appeared and right nowhere -- while the same oxygen placed in the
    BACKBONE (``3-oxaprop-2-en-1-yl`` for ``-CH2CHO``) round-trips exactly. Chain
    handling was sound; only branch handling was not.

    The vocabulary is REUSED, not re-tabled: ``_descriptive_fallback`` already maps a
    single heteroatom to its standard prefix and already discriminates ``oxo`` from
    ``hydroxy`` by hydrogen count (``substituent_enumerator.py:2101``), which is exactly
    the distinction needed. Imported lazily because ``substituent_enumerator`` imports
    THIS module, so a module-level import would be a cycle.

    Returns None for anything that is not a single non-carbon atom, so a carbon branch
    (``methyl``, ``ethyl``, a compound branch) still goes through the recursion that
    handles it correctly.
    """
    if len(comp) != 1:
        return None
    idx = next(iter(comp))
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetAtomicNum() == 6:
        return None
    # A charged or isotopically-labelled branch atom is out of the prefix
    # vocabulary's scope; let the recursion refuse it rather than guess.
    if atom.GetFormalCharge() != 0 or atom.GetIsotope():
        return None
    try:
        from ..assembly.substituent_enumerator import _descriptive_fallback
        prefix = _descriptive_fallback(mol, {idx}, idx)
    except Exception:  # pragma: no cover - never break naming for a prefix lookup
        return None
    if not isinstance(prefix, str) or not prefix.strip():
        return None
    # The cascade placeholder is a refusal, not a name (errors.py:223).
    if prefix.strip() == 'substituent':
        return None
    return prefix


def _branches_off(mol, frag: Set[int], skeleton: Sequence[int],
                  locant_of: Dict[int, int]):
    """(locant, branch_attachment_atom, branch_atom_set) per branch.

    A branch is a connected component of ``frag - skeleton`` together with the
    skeleton atom it hangs from. Collected by BFS so a branch of any depth comes
    out whole -- a depth cap here would silently truncate atoms.

    ``locant_of`` is SUPPLIED, never derived from list position. For a chain it is
    position-based; for a ring system it is the ring's own numbering, which skips
    ``4 -> 4a -> 5``, so a position-derived locant would silently disagree with
    the numbering the ring name was actually spelled from -- the same defect class
    as `internal notes`.
    """
    on_backbone = set(skeleton)
    backbone = skeleton
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
    ascending; distinct tokens are ordered by the project's key, which
    already strips multiplying prefixes, Greek letters and CIP descriptors.

    Multiplier table selection is (b)/(a) (the Blue Book):
    a compound/complex branch token -- already wrapped in marks by
    the caller (``enclose_if_compound``), so it opens with '(', '[' or '{' --
    is multiplied through the shared primitive
    (``naming_utils.multiplied_component``): ``bis``/``tris`` for a substituted
    prefix, the basic ``di``/``tri`` for a simple one with locants
    ('di(propan-2-yl)', (a)).

    NO trailing hyphen. Hyphens separate one prefix from the NEXT prefix, never a
    prefix from the stem it qualifies: the substituent group CC(C)C- is
    ``2-methylpropyl`` (isobutyl), not ``2-methyl-propyl``. Two prefixes still
    read ``1-ethyl-2-methylpropyl`` because the join supplies the internal
    hyphen. An earlier draft appended '-' here and produced the malformed form on
    all four branched cases.
    """
    grouped: Dict[str, List[int]] = {}
    keys: Dict[str, str] = {}
    for key, locant, tok in tokens:
        grouped.setdefault(tok, []).append(locant)
        keys[tok] = key
    parts = []
    from .fused_forms import locant_key
    for tok in sorted(grouped, key=lambda t: keys[t]):
        locs = sorted(grouped[tok], key=locant_key)
        from ..assembly.naming_utils import multiplied_component as _mc
        parts.append(f"{','.join(str(x) for x in locs)}-{_mc(len(locs), tok, tok)}")
    return "-".join(parts)


def _chain_stem_with_unsaturation(n: int, ene: List[int],
                                  yne: List[int]) -> Optional[str]:
    """'butyl' / 'but-3-en-1-yl' / 'penta-1,3-dien-1-yl' / 'hexa-1,3-dien-5-yn-1-yl'.

    : 'ene' always precedes 'yne', with elision of the final 'e' of
    'ene'. The attachment locant is cited explicitly ('-1-yl') whenever an
    unsaturation locant is present -- OPSIN-verified: '(2-oxabut-3-en-1-yl)benzene'
    parses, and the bare '...enyl' contraction is not used here because the
    explicit form is unambiguous for every chain length.

    Multiplied bonds: when TWO OR MORE double (or triple) bonds share the
    backbone, the suffix takes its own multiplying prefix ('di'/'tri'/...,
    SIMPLE_MULTIPLIERS -- the SAME table ``_assemble_prefixes`` uses, never a
    second copy) immediately before that suffix, and the STEM gains a single
    terminal linking 'a' -- once, regardless of how many of {ene, yne} are
    multiplied: 'penta-1,3-dien-1-yl', 'hexa-1,3-dien-5-yn-1-yl' (the 'dien' is
    multiplied, the 'yn' is not, and 'a' still appears exactly once, at the
    stem). A single bond of a type never gets a multiplier or the 'a':
    'pent-1-en-3-yn-1-yl' is unchanged by this rule (neither suffix is
    multiplied). All required strings OPSIN-verified 2026-08-04 by wrapping in
    a parent, e.g. '(hexa-1,3-dien-5-yn-1-yl)benzene' -> C#CC=CC=Cc1ccccc1.

    Returns ``None`` -- never an unparseable string -- if a bond-type count has
    no entry in the multiplier table (Table unspellable past 'icosa'/20); the
    caller must refuse the whole fragment rather than emit a name denoting
    nothing.
    """
    from ..assembly.naming_utils import SIMPLE_MULTIPLIERS
    from ..data.chain_names import get_chain_prefix  # 'but' for 4

    if not ene and not yne:
        return get_alkyl_name(n)
    stem = get_chain_prefix(n)

    def _suffix_part(locants: List[int], suffix: str) -> Optional[tuple]:
        """(needs_linking_a, '-<locants>-<mult><suffix>') for one bond type.

        ``None`` if a real 2+ multiplicity has no admitted multiplier.
        """
        if not locants:
            return (False, "")
        locs = sorted(locants)
        if len(locs) == 1:
            return (False, f"-{locs[0]}-{suffix}")
        mult = SIMPLE_MULTIPLIERS.get(len(locs))
        if mult is None:
            return None
        return (True, "-" + ",".join(str(x) for x in locs) + "-" + mult + suffix)

    ene_part = _suffix_part(ene, "en")
    if ene_part is None:
        return None
    yne_part = _suffix_part(yne, "yn")
    if yne_part is None:
        return None

    needs_a = ene_part[0] or yne_part[0]
    return stem + ("a" if needs_a else "") + ene_part[1] + yne_part[1] + "-1-yl"


def _book_chain_name(mol, backbone: Sequence[int], replacement_prefix: str,
                     stem: str, raw_names: Sequence[str],
                     unit_cites_locant: bool) -> Optional[str]:
    """The book's spelling of a decorated carbon backbone, or ``None`` to keep the
    locant-for-every-prefix assembly (roadmap N5f).

    * a one-carbon backbone is a methyl group: no locants (a),
      the Blue Book), the second and further prefixes enclosed,
      :7272): 'trifluoromethyl', '(cyclopropylamino)methyl';
    * every substitutable position of the backbone carries the same prefix: no
      locants,:3007): 'pentafluoroethyl'.

    ``unit_cites_locant`` is the caller's fact that ``stem`` itself cites a locant
    (an ene/yne locant), passed through to:func:`positions_alike_prefix`."""
    from ..assembly.book_prefixes import (
        book_forms_enabled, methyl_group_name, positions_alike_prefix)
    if not book_forms_enabled() or replacement_prefix or not raw_names:
        return None
    if any(mol.GetAtomWithIdx(a).GetAtomicNum() != 6 for a in backbone):
        return None
    if len(backbone) == 1:
        return methyl_group_name(raw_names, 1)
    alike = positions_alike_prefix(mol, backbone, raw_names,
                                   unit_cites_locant=unit_cites_locant)
    if alike is None:
        return None
    return _join_prefix_block(alike, stem)


def _join_prefix_block(block: str, stem: str) -> str:
    """Concatenate a substituent-prefix block onto the stem it qualifies.

    A hyphen goes in IFF the stem begins with a LOCANT, because a locant is
    always separated from preceding alphabetic text:

        '2-methyl' + 'propyl' -> '2-methylpropyl' (isobutyl)
        '3-methyl' + '2-oxabutyl' -> '3-methyl-2-oxabutyl'

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
    # Breadth Job 1: never a PIN-tier producer (see ``in_pin_promotion``).
    from .pin_vocabulary import in_pin_promotion
    if in_pin_promotion():
        return None
    try:
        return _same_reach_name(mol, frag_atoms, attach_idx)
    except Exception:                                   # noqa: BLE001
        # 17 call sites: degrade, never crash. Logged so a bug is visible in the
        # census instead of looking like an honest decline.
        logger.exception("terminal_fragment: unexpected error; refusing")
        return None


def _same_reach_name(mol, frag_atoms, attach_idx: int) -> Optional[TerminalFragmentName]:
    """Roadmap N5, same reach: the book spellings re-spell a fragment this writer names
    with its mechanical spellings; they never make it name a fragment it declines.

    ``name_substituent`` calls this writer after the ring composer and before the chain
    composer (``_recursive_chain_fragment_substituent_name``), which names a decorated
    chain with the recursive prefixes ('propan-2-yl', '1H-1,2,4-triazol-...' numbered by
    . A ring this writer used to decline in its replacement spelling (an aromatic
    triazole) but now names by its Hantzsch-Widman name would let it take a fragment
    the chain composer names better: measured on 'Cc1nnc(C(C)C)n1C1CC2CCC(C1)N2CC[C@H]
    (NC(=O)C1CCC(F)(F)CC1)c1ccc(O)cc1' at the best-effort tier, '5-methyl-3-(1-methyl
    ethyl)-4H-1,2,4-triazol-4-yl' (g), the Blue Book, wants methyl at 3)
    in place of the chain composer's '3-methyl-5-(propan-2-yl)-4H-1,2,4-triazol-4-yl'.
    So the fragment is named mechanically first (one more run of this writer, not one
    per level: the nested calls run inside it); when that declines, this writer
    declines as before."""
    from ..assembly.book_prefixes import book_forms_enabled, mechanical_forms
    if not book_forms_enabled():
        return _terminal_fragment_name(mol, frag_atoms, attach_idx)
    with mechanical_forms():
        mech = _terminal_fragment_name(mol, frag_atoms, attach_idx)
    if mech is None:
        return None
    book = _terminal_fragment_name(mol, frag_atoms, attach_idx)
    return book if book is not None else mech


def _ring_system_of(mol, frag: Set[int], seed: int) -> Set[int]:
    """The connected ring SYSTEM containing ``seed``, restricted to ``frag``.

    Union of every SSSR ring sharing an atom with the growing set, so ortho-fused,
    bridged and spiro systems come out as ONE system -- the same grouping
    ``metrics/breadth.ring_systems`` uses. Rings joined only by a bond (biphenyl)
    stay separate, which is correct: they are two systems, and the second is a
    decoration of the first.
    """
    rings = [set(r) & frag for r in mol.GetRingInfo().AtomRings()]
    system = {seed}
    changed = True
    while changed:
        changed = False
        for r in rings:
            if r & system and not r <= system:
                system |= r
                changed = True
    return system


def _composite_fragment_name(
    mol, frag: Set[int], attach_idx: int,
) -> Optional[TerminalFragmentName]:
    """A fragment containing at least one ring system.

    The ring system carrying the attachment is named by
    ``terminal_ring.terminal_ring_name`` -- reused unchanged, so skeletal
    replacement, lambda, ring multiple-bond locants and its own reconstruction
    audit all apply -- and every remaining branch recurses through
    ``_terminal_fragment_name``.

    Refuses whole if the ring refuses or ANY decoration refuses: emitting the ring
    alone would drop the decoration's atoms, which is the defect this module
    exists to remove.

    Scope, stated: when the attachment atom is NOT itself a ring atom (a chain
    leading to a ring), this declines rather than half-naming. Handling it needs a
    chain backbone that terminates in a ring system, which is a separate build.
    """
    from .terminal_ring import terminal_ring_name

    ri = mol.GetRingInfo()
    if ri.NumAtomRings(attach_idx) < 1:
        logger.info("terminal_fragment: attachment is acyclic but the fragment "
                    "carries a ring; out of implemented scope, refuse")
        return None

    core = _ring_system_of(mol, frag, attach_idx)
    tr = terminal_ring_name(mol, sorted(core), attach_idx)
    if tr is None:
        logger.info("terminal_fragment: terminal_ring declined the %d-atom ring "
                    "system; refuse", len(core))
        return None

    accounted = set(core)
    prefix_tokens: List[tuple] = []
    raw_names: List[str] = []
    branch_comps: List[Set[int]] = []
    # The RING's own numbering is authoritative -- it skips 4 -> 4a -> 5, so a
    # position-derived locant would disagree with the name it was spelled from.
    for locant, battach, comp in _branches_off(mol, frag, sorted(core),
                                               tr.numbering):
        branch_comps.append(set(comp))
        _het = _single_heteroatom_branch_prefix(mol, comp)
        if _het is not None:
            accounted |= set(comp)
            prefix_tokens.append((alpha_sort_key(_het), locant,
                                  enclose_if_compound(_het)))
            raw_names.append(_het)
            continue
        # 0-WRONG guard (mirrors the chain loop): a ring decoration joined by a
        # NON-single bond (exocyclic =CH2 / =C<, an ylidene) recurses to a `-yl`
        # token that asserts a single bond -- wrong constitution. Fail closed.
        # (§A2 follow-on converts this to the {ring}ylidene form; until then, refuse.)
        _cn = next((n.GetIdx() for n in mol.GetAtomWithIdx(battach).GetNeighbors()
                    if n.GetIdx() in core), None)
        if (_cn is not None and mol.GetBondBetweenAtoms(battach, _cn)
                .GetBondType() != Chem.BondType.SINGLE):
            logger.info("terminal_fragment: ring decoration at locant %s is "
                        "joined by a non-single bond (ylidene); refuse", locant)
            return None
        sub = _terminal_fragment_name(mol, comp, battach)
        if sub is None:
            logger.info("terminal_fragment: ring decoration at locant %s "
                        "unnameable; refusing the whole fragment", locant)
            return None
        accounted |= set(sub.atoms)
        token = enclose_if_compound(sub.name)
        prefix_tokens.append((alpha_sort_key(sub.name), locant, token))
        raw_names.append(sub.name)

    # --- ring-system stereo -> one leading (nR,nZ,...) block (§A) ----------
    # Cite the RING SYSTEM's own atom R/S + ring-bond E/Z with the RING numbering
    # (tr.numbering). Branch-internal stereo is ALREADY inside each branch token
    # (its recursion emitted it). A defined centre/geometry that is neither on the
    # ring system nor inside a placed branch cannot be cited here -> refuse
    # (0-wrong: never drop a centre). 's C6 stereo layer verifies the
    # emitted descriptor downstream, so a wrong CIP/locant abstains, never ships.
    descriptor = ""
    if _has_defined_atom_stereo(mol, frag) or _has_defined_bond_stereo(mol, frag):
        from ..perception.stereo import assign_stereochemistry
        assign_stereochemistry(mol)  # populate _CIPCode on atoms AND bonds
        _bits: List[tuple] = []
        for a in frag:
            atom = mol.GetAtomWithIdx(a)
            if not atom.HasProp('_CIPCode'):
                continue
            if a in core:
                cip = atom.GetProp('_CIPCode')
                if cip not in ('R', 'S') or a not in tr.numbering:
                    return None  # r/s/M/P or a ring atom with no locant -> refuse
                _bits.append((tr.numbering[a], cip))
            elif any(a in bc for bc in branch_comps):
                continue  # inside a branch -> its recursion cited it
            else:
                return None  # a centre on neither ring nor branch -> refuse
        for b in mol.GetBonds():
            if b.GetStereo() == Chem.BondStereo.STEREONONE:
                continue
            bi, bj = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if bi not in frag or bj not in frag:
                continue
            if bi in core and bj in core:
                if (not b.HasProp('_CIPCode') or b.GetProp('_CIPCode') not in ('E', 'Z')
                        or bi not in tr.numbering or bj not in tr.numbering):
                    return None
                _bits.append((min(tr.numbering[bi], tr.numbering[bj]),
                              b.GetProp('_CIPCode')))
            elif any(bi in bc and bj in bc for bc in branch_comps):
                continue  # inside a branch -> its recursion cited it
            else:
                return None  # ring-to-branch / unplaceable geometry -> refuse
        if _bits:
            # a fusion numbering has lettered locants ('4a'): number, then letters
            # (``fused_forms.locant_key``; the same order for plain int locants)
            from .fused_forms import locant_key
            _bits.sort(key=lambda bit: (locant_key(bit[0]), bit[1]))
            descriptor = "(" + ",".join(f"{lc}{c}" for lc, c in _bits) + ")-"

    name = tr.name
    if prefix_tokens:
        # (the Blue Book): every substitutable ring position carries
        # the same prefix -> no locants ('pentafluorophenyl')
        alike = None
        if tr.basis in ('monocycle_book', 'fused_book'):
            from ..assembly.book_prefixes import positions_alike_prefix
            alike = positions_alike_prefix(mol, sorted(core), raw_names,
                                           unit_cites_locant=tr.cites_locant)
        name = _join_prefix_block(
            alike if alike is not None else _assemble_prefixes(prefix_tokens), name)
    if descriptor:
        name = descriptor + name

    if accounted != frag:
        logger.error("terminal_fragment: composite completeness violated "
                     "(%d of %d); refuse", len(accounted), len(frag))
        return None
    return TerminalFragmentName(
        name=name, numbering=dict(tr.numbering),
        basis="composite" if prefix_tokens else "ring",
        atoms=frozenset(accounted))


def _book_hetero_fragment(mol, frag: Set[int], attach_idx: int
                          ) -> Optional[TerminalFragmentName]:
    """A fragment attached through a heteroatom, composed from its parts (roadmap N5c):

    * -O-R 'R-oxy' with the retained contractions the Blue Book,
       :27667: 'methoxy', '2-chloroethoxy', 'phenoxy', '(propan-2-yl)oxy');
      -O-CO-R the acyloxy prefix:31698);
    * -S-R '(R)sulfanyl', -Se-R '(R)selanyl', -Te-R '(R)tellanyl':27649);
    * -N< 'amino' with its substituents, the second enclosed,;
      'methylamino', 'methyl(phenyl)amino', 'acetylamino').

    The parts are named by this module's own recursion (an acyl part by
    ``book_prefixes.acyl_group_prefix``). ``None`` for any other shape."""
    from rdkit import Chem as _Chem

    from ..assembly.book_prefixes import acyl_group_prefix
    from ..assembly.substituent_enumerator import (
        alkoxy_prefix_from_substituent, cite_organyl_in_composed_prefix)
    root = mol.GetAtomWithIdx(attach_idx)
    sym = root.GetSymbol()
    if (sym not in HETERO_ROOT_VALENCE or root.IsInRing()
            or root.GetFormalCharge() or root.GetIsotope()):
        return None
    if any(b.GetBondType() != _Chem.BondType.SINGLE for b in root.GetBonds()):
        return None
    if root.GetTotalValence() != HETERO_ROOT_VALENCE[sym]:
        return None
    parents = [nb.GetIdx() for nb in root.GetNeighbors() if nb.GetIdx() not in frag]
    if len(parents) != 1:
        return None
    parts = []
    claimed = {attach_idx}
    for nb in root.GetNeighbors():
        j = nb.GetIdx()
        if j not in frag or j in claimed:
            continue
        comp, stack = set(), [j]
        while stack:
            cur = stack.pop()
            if cur in comp or cur == attach_idx:
                continue
            comp.add(cur)
            for nb2 in mol.GetAtomWithIdx(cur).GetNeighbors():
                k = nb2.GetIdx()
                if k in frag and k not in comp and k != attach_idx:
                    stack.append(k)
        claimed |= comp
        parts.append((j, comp))
    if not parts or (sym in CHALCOGEN_HEAD and len(parts) != 1):
        return None
    if claimed != frag:
        return None

    def name_of(p_attach, p_atoms):
        if len(p_atoms) == 1:
            lone = _single_heteroatom_branch_prefix(mol, p_atoms)
            if lone is not None:
                return lone, False
        acyl = acyl_group_prefix(
            mol, p_attach, attach_idx, p_atoms,
            lambda atoms, r: (lambda t: t.name if t is not None else None)(
                _terminal_fragment_name(mol, set(atoms), r)))
        if acyl is not None:
            return acyl, True
        sub = _terminal_fragment_name(mol, set(p_atoms), p_attach)
        return (sub.name, False) if sub is not None else (None, False)

    names = []
    for p_attach, p_atoms in parts:
        nm, is_acyl = name_of(p_attach, p_atoms)
        if not nm:
            return None
        names.append((nm, is_acyl))
    if sym in MONONUCLEAR_HEAD:
        # a mononuclear parent hydride: its prefixes without locants, the second and
        # further enclosed:7272): 'trimethylsilyl',
        # '*tert*-butyldi(methyl)silyl', 'dimethylphosphanyl'
        from ..assembly.composer import _assemble_decorated_amino_prefix
        token = _assemble_decorated_amino_prefix(
            [(n, False) for n, _ in names], enclose=False, head=MONONUCLEAR_HEAD[sym])
    elif sym == "N":
        token = None
        if len(names) == 1 and names[0][1] and root.GetTotalNumHs() == 1:
            # -NH-CO-R: the amido prefix, method (1) (the Blue Book
            # "Method (1) generates preferred IUPAC names"), from the verified acid
            from ..assembly.substituent_naming import acyl_amido_prefix_from_branch
            try:
                token = acyl_amido_prefix_from_branch(
                    mol, attach_idx, parts[0][0], frag, verify_acid=True)
            except Exception:  # noqa: BLE001 - a producer that raises is a decline
                token = None
        if not token:
            from ..assembly.composer import _assemble_decorated_amino_prefix
            token = _assemble_decorated_amino_prefix(
                [(n, False) for n, _ in names], enclose=False, head="amino")
    else:
        nm, is_acyl = names[0]
        if sym == "O" and not is_acyl:
            token = alkoxy_prefix_from_substituent(nm)
        else:
            cited = cite_organyl_in_composed_prefix(nm)
            token = f"{cited}{CHALCOGEN_HEAD[sym]}" if cited else None
            if token and is_acyl:
                # Lane L2: an acyl-oxy / acyl-sulfanyl prefix is compound,
                # the Blue Book; '2-(acetyloxy)ethane-1-sulfonic acid (PIN)':31713)
                from ..assembly.prefix_derivation import built
                token = built(token, substituted=True, enclosed=True)
    if not token:
        return None
    from ..assembly.book_prefixes import note_book_form
    note_book_form()
    return TerminalFragmentName(name=token, numbering={attach_idx: 1}, basis="hetero",
                                atoms=frozenset(frag))


def _group_part_names(mol, shape):
    """The names of the parts of an N/O group (``assembly.hetero_group_prefixes``), each by
    this module's own recursion; ``None`` when a part is unnameable. A lone heteroatom takes
    its standard prefix ('hydroxy', never the 'a' chain '1-oxamethyl'); a part joined by a
    double bond is a one-carbon ylidene ('diaminomethylidene', ``methyl_group_name``) with
    its geometry cited at the front when defined."""
    names = []
    for part in shape.parts:
        if part.order == 2:
            name = _ylidene_part_name(mol, part)
            if name is None:
                return None
            names.append(name)
            continue
        lone = _single_heteroatom_branch_prefix(mol, set(part.atoms))
        if lone is not None:
            names.append(lone)
            continue
        sub = _terminal_fragment_name(mol, set(part.atoms), part.attach)
        if sub is None:
            return None
        names.append(sub.name)
    return names


def _ylidene_part_name(mol, part) -> Optional[str]:
    """'(E)-phenylmethylidene' for the part of an N=C group: a carbon with single-bonded
    branches (one-carbon parent, (a) the Blue Book)."""
    from ..assembly.book_prefixes import methyl_group_name
    atoms = set(part.atoms)
    c = part.attach
    if mol.GetAtomWithIdx(c).GetSymbol() != "C" or mol.GetAtomWithIdx(c).IsInRing():
        return None
    branch_names = []
    claimed = {c}
    for nb in mol.GetAtomWithIdx(c).GetNeighbors():
        j = nb.GetIdx()
        if j not in atoms or j in claimed:
            continue
        if mol.GetBondBetweenAtoms(c, j).GetBondType() != Chem.BondType.SINGLE:
            return None
        comp, stack = set(), [j]
        while stack:
            cur = stack.pop()
            if cur in comp or cur == c:
                continue
            comp.add(cur)
            for nb2 in mol.GetAtomWithIdx(cur).GetNeighbors():
                k = nb2.GetIdx()
                if k in atoms and k not in comp and k != c:
                    stack.append(k)
        claimed |= comp
        if all(mol.GetAtomWithIdx(a).GetAtomicNum() == 6 for a in comp) and _is_acyclic(mol, comp):
            return None  # an alkylidene ('ethylidene'), not 'methylmethylidene': not built here
        lone = _single_heteroatom_branch_prefix(mol, comp)
        if lone is not None:
            branch_names.append(lone)
            continue
        sub = _terminal_fragment_name(mol, comp, j)
        if sub is None:
            return None
        branch_names.append(sub.name)
    if claimed != atoms:
        return None
    name = methyl_group_name(branch_names, 2) if branch_names else "methylidene"
    if name is None:
        return None
    bond = mol.GetBondBetweenAtoms(part.host, c)
    if bond.GetStereo() != Chem.BondStereo.STEREONONE:
        from ..perception.stereo import assign_stereochemistry
        assign_stereochemistry(mol)
        ez = bond.GetProp('_CIPCode') if bond.HasProp('_CIPCode') else None
        if ez not in ('E', 'Z'):
            return None
        name = f"(1{ez})-{name}"
    return name


def _book_group_fragment(mol, frag: Set[int], attach_idx: int, late: bool = False
                         ) -> Optional[TerminalFragmentName]:
    """A fragment that is a diazenyl, hydrazinyl or peroxy group attached by a single bond
    ('phenyldiazenyl', '2-methylhydrazinyl', '(tert-butyl)peroxy', 'hydroperoxy'),
    composed from its parts (roadmap item 12a; the Blue Book,
    :27858,:27944), not spelled as the 'a' chain '2-phenyl-1,2-diazaeth-1-en-1-yl'
    that (:6465) does not allow. The N=N geometry of a diazenyl group is cited
    at the front, '(1E)-'; ``None`` for any other shape."""
    from ..assembly.hetero_group_prefixes import LATE_KINDS, compose_group, group_shape
    if not forms_enabled("chain"):
        return None
    parents = [nb.GetIdx() for nb in mol.GetAtomWithIdx(attach_idx).GetNeighbors()
               if nb.GetIdx() not in frag and nb.GetAtomicNum() > 1]
    if len(parents) != 1:
        return None
    if mol.GetBondBetweenAtoms(attach_idx, parents[0]).GetBondType() != Chem.BondType.SINGLE:
        return None
    shape = group_shape(mol, attach_idx, frozenset(frag), parents[0], unstereo_iminyl=True)
    if shape is None or shape.kind not in ("diazenyl", "hydrazinyl", "peroxy", "hydroperoxy",
                                           "iminyl", "nitroso", "aminooxy"):
        return None
    if (shape.kind in LATE_KINDS) != late:
        return None
    names = _group_part_names(mol, shape)
    if names is None:
        return None
    token = compose_group(shape, names)
    if not token:
        return None
    if shape.stereo_bond is not None:
        bond = mol.GetBondBetweenAtoms(*shape.stereo_bond)
        if bond.GetStereo() != Chem.BondStereo.STEREONONE:
            from ..perception.stereo import assign_stereochemistry
            assign_stereochemistry(mol)
            ez = bond.GetProp('_CIPCode') if bond.HasProp('_CIPCode') else None
            if ez not in ('E', 'Z'):
                return None
            from ..assembly.prefix_derivation import carried
            token = carried(f"(1{ez})-{token}", like=token)
    from ..assembly.book_prefixes import note_book_form
    note_book_form("chain")
    return TerminalFragmentName(name=token, numbering={attach_idx: 1}, basis="hetero",
                                atoms=frozenset(frag))


def _book_ylidene_branch(mol, battach: int, comp: Set[int], backbone_atom: int
                         ) -> Optional[str]:
    """The prefix of a branch joined to the backbone by a double bond that is an imino or
    hydrazinylidene group ('hydroxyimino', '(methoxyimino)', 'dimethylhydrazinylidene';
     the Blue Book, Changes from the 1979 edition 7(j):1703), or
    ``None``. This module has no other -ylidene constructor."""
    from ..assembly.hetero_group_prefixes import compose_group, group_shape
    if not forms_enabled("chain"):
        return None
    shape = group_shape(mol, battach, frozenset(comp), backbone_atom)
    if shape is None or shape.kind not in ("imino", "hydrazinylidene"):
        return None
    names = _group_part_names(mol, shape)
    if names is None:
        return None
    token = compose_group(shape, names)
    if token:
        from ..assembly.book_prefixes import note_book_form
        note_book_form("chain")
    return token


def _ylidene_geometry(mol, bond, numbering, branches) -> bool:
    """True when ``bond`` joins a backbone atom to the root of a branch that is an imino or
    hydrazinylidene group (``_book_ylidene_branch``)."""
    for _loc, battach, comp in branches:
        ends = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        if battach in ends and (ends[0] in numbering or ends[1] in numbering):
            other = ends[0] if ends[1] == battach else ends[1]
            if other not in numbering:
                return False
            from ..assembly.hetero_group_prefixes import group_shape
            shape = group_shape(mol, battach, frozenset(comp), other)
            return shape is not None and shape.kind in ("imino", "hydrazinylidene")
    return False


def _book_acyl_fragment(mol, frag: Set[int], attach_idx: int
                        ) -> Optional[TerminalFragmentName]:
    """The acyl-group name of a fragment attached through a carbonyl carbon
    (``assembly.book_prefixes.acyl_group_prefix``), its parts named by this module's
    own recursion; ``None`` when the fragment is not such a group."""
    from ..assembly.book_prefixes import acyl_group_prefix, book_forms_enabled
    if not book_forms_enabled():
        return None
    atom = mol.GetAtomWithIdx(attach_idx)
    if atom.GetSymbol() != "C" or atom.IsInRing():
        return None
    parents = [nb.GetIdx() for nb in atom.GetNeighbors() if nb.GetIdx() not in frag]
    if len(parents) != 1:
        return None

    def part(atoms, root):
        # a lone heteroatom takes its standard prefix: '(hydroxy)carbamoyl', never the 'a'
        # chain '(1-oxamethyl)carbamoyl' (``_single_heteroatom_branch_prefix``)
        lone = _single_heteroatom_branch_prefix(mol, set(atoms)) if forms_enabled("chain") else None
        if lone is not None:
            return lone
        sub = _terminal_fragment_name(mol, set(atoms), root)
        return sub.name if sub is not None else None

    token = acyl_group_prefix(mol, attach_idx, parents[0], frag, part)
    if token is None:
        return None
    return TerminalFragmentName(name=token, numbering={attach_idx: 1}, basis="acyl",
                                atoms=frozenset(frag))


def _book_oxoacid_fragment(mol, frag: Set[int], attach_idx: int
                           ) -> Optional[TerminalFragmentName]:
    """A P or S oxoacid group (sulfo, phosphono, '[ethoxy(hydroxy)phosphoryl]oxy',...) is one
    prefix read from the structure, the Blue Book;,:36484),
    never an 'a' chain that threads its central atom,:6465); the kind 'ps'."""
    from ..assembly.book_prefixes import forms_enabled
    from .oxoacid_group_prefix import (
        is_oxoacid_centre, is_oxoacid_linker, note_ps_form, oxoacid_group_prefix_ex,
        ps_tier_ctx)
    if not (ps_tier_ctx.get() and forms_enabled("ps")):
        return None     # wider tiers only: PIN-path callers reach this writer too
    if not (is_oxoacid_centre(mol, attach_idx) or is_oxoacid_linker(mol, attach_idx)):
        return None
    token, preferred = oxoacid_group_prefix_ex(mol, frag, attach_idx)
    if token is None:
        return None
    note_ps_form(token, preferred)
    return TerminalFragmentName(name=token, numbering={attach_idx: 1}, basis="chain",
                                atoms=frozenset(frag))


def _book_retained_fragment(mol, frag: Set[int], attach_idx: int
                            ) -> Optional[TerminalFragmentName]:
    """'benzyl' or 'tert-butyl' for an unsubstituted group on a single bond,
    the Blue Book-:24414), decided from the atoms by
    ``book_prefixes.retained_group_prefix``; ``None`` for any other fragment. This writer
    spells single-bond prefixes only."""
    from ..assembly.book_prefixes import note_book_form, retained_group_prefix
    parents = [nb.GetIdx() for nb in mol.GetAtomWithIdx(attach_idx).GetNeighbors()
               if nb.GetIdx() not in frag and nb.GetAtomicNum() > 1]
    if len(parents) != 1:
        return None
    if mol.GetBondBetweenAtoms(attach_idx, parents[0]).GetBondType() != Chem.BondType.SINGLE:
        return None
    token = retained_group_prefix(mol, frag, attach_idx, 1)
    if token is None:
        return None
    note_book_form()
    return TerminalFragmentName(name=token, numbering={attach_idx: 1}, basis="chain",
                                atoms=frozenset(frag))


def _terminal_fragment_name(
    mol, frag_atoms, attach_idx: int,
) -> Optional[TerminalFragmentName]:
    """The book's spelling first (roadmap N5c: a carbon backbone, heteroatom-rooted
    branches composed as 'R-oxy', '(R)sulfanyl', '(R)amino'), then -- when that
    declines, e.g. on an imine C=N the carbon walk cannot place -- the earlier
    replacement-chain spelling."""
    from ..assembly.book_prefixes import book_forms_enabled
    if book_forms_enabled():
        res = _terminal_fragment_name_impl(mol, frag_atoms, attach_idx, book=True)
        if res is not None:
            return res
    return _terminal_fragment_name_impl(mol, frag_atoms, attach_idx, book=False)


def _terminal_fragment_name_impl(
    mol, frag_atoms, attach_idx: int, book: bool,
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
    if book:
        oxoacid = _book_oxoacid_fragment(mol, frag, attach_idx)
        if oxoacid is not None:
            return oxoacid
    # SCOPE, mirroring terminal_ring.py:582: a charged skeletal atom is,
    # not replacement nomenclature, and naming it neutral denotes a different
    # species.
    if any(mol.GetAtomWithIdx(a).GetFormalCharge() != 0 for a in frag):
        return None
    # Stereochemistry. The COMPOSITE path (ring parent, below) emits NO
    # stereodescriptor, so it still refuses any defined stereo -- dropping a
    # centre/geometry there names a different molecule cannot catch. The
    # ACYCLIC path below now emits BOTH backbone C=C geometry (nE/nZ,
    # internal-C=C) AND backbone R/S centres (B3), so a defined atom
    # stereocentre no longer forces a blanket refusal there -- it is cited in the
    # leading (...) block, and any centre it cannot place (off-backbone) still
    # refuses. Every emitted descriptor is RT-verified by downstream.
    if mol.GetRingInfo().NumAtomRings(attach_idx) > 0:
        # Attachment is ON a ring system -> the ring is the parent (composite
        # path). §A: that path now cites the ring system's own R/S + E/Z in a
        # leading (...) block from the ring numbering, and refuses INTERNALLY any
        # centre it cannot place -- so the former blanket stereo refusal is gone.
        # 's C6 RegistrationHash stereo layer verifies every emitted
        # descriptor downstream, so a wrong CIP/locant abstains, never ships.
        return _composite_fragment_name(mol, frag, attach_idx)

    # Attachment is ACYCLIC -> the chain path names it. ring-branch lever:
    # when the fragment ALSO carries ring system(s) -- a chain leading to a ring,
    # the case _composite_fragment_name explicitly declines ("attachment is
    # acyclic but the fragment carries a ring; out of implemented scope") -- the
    # backbone STOPS at the ring boundary and each ring system falls out as a
    # decoration recursed through _composite_fragment_name. The ring is spelled
    # by terminal_ring (REPLACEMENT nomenclature), so -C(=O)-Ph emits the
    # ugly-but-RT-correct `1-(cyclohexa-1,3,5-trien-1-yl)-2-oxaeth-1-en-1-yl`
    # and -CH2CH2-cyclohexyl emits `2-(cyclohexan-1-yl)ethyl` (a project rule's T4
    # clause; retained ring names are a follow-on). The pure-acyclic case (no
    # ring) is stop_at_ring=False -> byte-identical.
    # Roadmap N5e: an acyl group attached through its carbonyl carbon takes its acyl
    # name ('carbamoyl', '(R)carbamoyl', 'azetidine-1-carbonyl', 'methoxycarbonyl',
    # 'benzoyl'; (2) the Blue Book), never an 'a' chain that writes the
    # carbonyl oxygen as a chain end ('2-oxaeth-1-en-1-yl', '1-oxo-2-azaethyl').
    acyl = _book_acyl_fragment(mol, frag, attach_idx)
    if acyl is not None:
        return acyl
    # a chain the book allows as an 'a' chain the Blue Book) stays one
    licensed_a = False
    if book and any(mol.GetAtomWithIdx(a).GetAtomicNum() != 6 for a in frag):
        _bb = _backbone_from(mol, frag, attach_idx, _canonical_ranks(mol),
                             stop_at_ring=not _is_acyclic(mol, frag))
        _het = [a for a in _bb if mol.GetAtomWithIdx(a).GetAtomicNum() != 6]
        # (:6465) ends and (:23348) four or more hetero atoms, at least one C
        licensed_a = bool(_het) and (a_chain_licensed(mol, _bb) or (
            len(_het) >= 4 and len(_het) < len(_bb)
            and all(mol.GetAtomWithIdx(e).GetSymbol() in ("C", "P", "As", "Sb", "Bi", "Si", "Ge",
                                                          "Sn", "Pb", "B", "Al", "Ga", "In", "Tl")
                    for e in (_bb[0], _bb[-1]))))
    _licensed_backbone = licensed_a
    if book:
        group = None if licensed_a else _book_group_fragment(mol, frag, attach_idx)
        if group is not None:
            return group
        hetero = _book_hetero_fragment(mol, frag, attach_idx)
        if hetero is not None:
            return hetero
        late_group = (None if licensed_a
                      else _book_group_fragment(mol, frag, attach_idx, late=True))
        if late_group is not None:
            return late_group
        retained = _book_retained_fragment(mol, frag, attach_idx)
        if retained is not None:
            return retained

    _frag_has_ring = not _is_acyclic(mol, frag)
    ranks = _canonical_ranks(mol)
    backbone = _backbone_from(mol, frag, attach_idx, ranks,
                              stop_at_ring=_frag_has_ring)
    if (book and mol.GetAtomWithIdx(attach_idx).GetAtomicNum() == 6
            and not a_chain_licensed(mol, backbone)
            and hetero_roots_composable(mol, backbone, group_leaves=forms_enabled('chain'))):
        # Roadmap N5c: (the Blue Book) and (:23348) allow an
        # 'a' chain only with C (or P, As, Sb, Bi, Si, Ge, Sn, Pb, B, Al, Ga, In, Tl)
        # ends, at least one carbon atom and four or more heterounits; otherwise the
        # backbone is the carbon chain and every heteroatom roots a substituent prefix.
        backbone = _backbone_from(mol, frag, attach_idx, ranks,
                                  stop_at_ring=_frag_has_ring, carbon_only=True)
        from ..assembly.book_prefixes import note_book_form
        note_book_form()
    numbering_for_branches = {a: i + 1 for i, a in enumerate(backbone)}
    branches = _branches_off(mol, frag, backbone, numbering_for_branches)

    # Backbone unsaturation: every bond order must have an admitted morpheme,
    # or the fragment refuses -- spelling an unrepresentable bond as single
    # would denote a DIFFERENT molecule, the exact defect this module exists
    # to remove.
    ene, yne = [], []
    for i in range(len(backbone) - 1):
        b = mol.GetBondBetweenAtoms(backbone[i], backbone[i + 1])
        if b is None:
            return None
        bt = b.GetBondType()
        if bt == Chem.BondType.SINGLE:
            continue
        if bt == Chem.BondType.DOUBLE:
            ene.append(i + 1)            # cite the LOWER locant
        elif bt == Chem.BondType.TRIPLE:
            yne.append(i + 1)
        else:
            # AROMATIC or DATIVE on an acyclic backbone: no morpheme, and
            # spelling it single would denote a different molecule.
            return None

    numbering = {a: i + 1 for i, a in enumerate(backbone)}

    # --- backbone stereo -> one leading (nR,nZ,...) block ------------------
    # BOTH atom R/S centres (B3) and backbone C=C geometry (internal-C=C)
    # are cited, merged and locant-ordered. Locant = the atom's backbone locant
    # (for a C=C, the lower endpoint,. A centre/geometry on the
    # backbone is cited here; one wholly inside a branch is cited by that
    # branch's own recursion (which routes back through this function); anything
    # that can be placed on NEITHER fails closed. 's isomeric round-trip
    # verifies the emitted descriptor, so a wrong CIP/locant abstains, never ships.
    descriptor = ""
    _stereo_bonds = [
        b for b in mol.GetBonds()
        if b.GetStereo() != Chem.BondStereo.STEREONONE
        and b.GetBeginAtomIdx() in frag and b.GetEndAtomIdx() in frag]
    if _has_defined_atom_stereo(mol, frag) or _stereo_bonds:
        _bb_bond_ids = set()
        for i in range(len(backbone) - 1):
            _bb = mol.GetBondBetweenAtoms(backbone[i], backbone[i + 1])
            if _bb is not None:
                _bb_bond_ids.add(_bb.GetIdx())
        _branch_sets = [set(comp) for _, _, comp in branches]
        from ..perception.stereo import assign_stereochemistry
        assign_stereochemistry(mol)  # populate _CIPCode on atoms AND bonds
        _bits = []  # (locant, letter) for atom R/S and bond E/Z together
        # atom R/S centres
        for a in frag:
            atom = mol.GetAtomWithIdx(a)
            if not atom.HasProp('_CIPCode'):
                continue
            cip = atom.GetProp('_CIPCode')
            if a in numbering:
                if cip not in ('R', 'S'):
                    return None  # r/s/M/P etc. not placeable on this path
                _bits.append((numbering[a], cip))
            elif any(a in s for s in _branch_sets):
                continue  # inside a branch -> its recursion cites it
            else:
                return None  # off-backbone centre we cannot place -> refuse
        # backbone C=C geometry
        for b in _stereo_bonds:
            if b.GetIdx() in _bb_bond_ids:
                if not b.HasProp('_CIPCode'):
                    return None
                _ez = b.GetProp('_CIPCode')
                if _ez not in ('E', 'Z'):
                    return None
                _bits.append((min(numbering[b.GetBeginAtomIdx()],
                                  numbering[b.GetEndAtomIdx()]), _ez))
            elif any(b.GetBeginAtomIdx() in s and b.GetEndAtomIdx() in s
                     for s in _branch_sets):
                continue  # inside a branch -> its own recursion emits the E/Z
            elif book and not _licensed_backbone and forms_enabled('chain') and _ylidene_geometry(mol, b, numbering, branches):
                # roadmap item 12a: the =N- group's bond to the backbone is cited with the
                # backbone atom's locant, (1) method (a) (the Blue Book)
                if not b.HasProp('_CIPCode') or b.GetProp('_CIPCode') not in ('E', 'Z'):
                    return None
                _bb_end = b.GetBeginAtomIdx() if b.GetBeginAtomIdx() in numbering \
                    else b.GetEndAtomIdx()
                _bits.append((numbering[_bb_end], b.GetProp('_CIPCode')))
            else:
                return None  # off-backbone geometry we cannot place -> refuse
        if _bits:
            _bits.sort()
            descriptor = "(" + ",".join(f"{lc}{c}" for lc, c in _bits) + ")-"

    rp = build_replacement_prefix(mol, numbering, set(backbone))
    if rp.unexpressed:
        logger.info(
            "terminal_fragment: %d backbone atom(s) have no admitted morpheme; "
            "refuse (Table 1.5 is closed)", len(rp.unexpressed))
        return None

    # The stem counts EVERY skeletal atom, carbon and heteroatom alike -- that is
    # what makes replacement nomenclature complete (2-oxabutyl has 4 skeletal
    # atoms: C-O-C-C). Verified against OPSIN.
    stem = _chain_stem_with_unsaturation(len(backbone), ene, yne)
    if stem is None:
        logger.info(
            "terminal_fragment: %d ene / %d yne locants have no admitted "
            "multiplier; refuse rather than emit an unparseable name",
            len(ene), len(yne))
        return None
    if book and not rp.prefix and len(backbone) == 2 and ene == [1] and not yne:
        # (the Blue Book): one place for the double bond, so no
        # ene locant; the free-valence locant only with a branch prefix
        # ('ethenyl':24547, '2-chloroethen-1-yl':3003)
        stem = "ethen-1-yl" if branches else "ethenyl"
    name = f"{rp.prefix}{stem}" if rp.prefix else stem
    if rp.prefix and not a_chain_licensed(mol, backbone):
        # (the Blue Book), (:23348): an 'a' chain the book does
        # not allow is valid but never part of a PIN; the block and the stem root are in
        # every name built on this chain
        from ..data.chain_names import get_chain_prefix
        from ..metrics.provenance import record_non_pin_fragment
        record_non_pin_fragment(f"{rp.prefix}{get_chain_prefix(len(backbone))}")

    accounted = set(backbone)
    _backbone_set = set(backbone)
    prefix_tokens: List[tuple] = []          # (alpha_key, locant, token)
    raw_names: List[str] = []                # the bare prefix name of each branch
    for locant, battach, comp in branches:
        # A lone heteroatom branch takes its STANDARD prefix. Recursing on it
        # yields `1-oxamethyl` for `=O`, which OPSIN reads as `-OH` -- a wrong
        # molecule. See _single_heteroatom_branch_prefix. (`=O` is spelled `oxo`
        # here, which correctly encodes the double bond, so the ylidene guard
        # below must not see it -- hence it runs only for the recursed branches.)
        _het = _single_heteroatom_branch_prefix(mol, comp)
        if _het is not None:
            accounted |= set(comp)
            prefix_tokens.append((alpha_sort_key(_het), locant,
                                  enclose_if_compound(_het)))
            raw_names.append(_het)
            continue
        # 0-WRONG guard: a branch joined to the backbone by a NON-single bond is a
        # =/# -attached substituent (methylidene, cyclohexylidene,...). Recursing
        # it yields a `-yl` (single free valence) token that ASSERTS a single bond
        # -- a wrong CONSTITUTION (`-CH=C<ring` named `...(cyclohexan-1-yl)...`).
        # This module has no -ylidene/-ylidyne constructor, so fail closed. Fixes
        # the ring-ylidene class this lever newly reaches AND the pre-existing
        # acyclic methylidene one (a review ring-review R1).
        _bbn = next((n.GetIdx() for n in mol.GetAtomWithIdx(battach).GetNeighbors()
                     if n.GetIdx() in _backbone_set), None)
        if (book and not _licensed_backbone and _bbn is not None
                and mol.GetBondBetweenAtoms(battach, _bbn).GetBondType() == Chem.BondType.DOUBLE):
            _yl = _book_ylidene_branch(mol, battach, comp, _bbn)
            if _yl is not None:
                accounted |= set(comp)
                prefix_tokens.append((alpha_sort_key(_yl), locant, enclose_if_compound(_yl)))
                raw_names.append(_yl)
                continue
        if (_bbn is not None and mol.GetBondBetweenAtoms(battach, _bbn)
                .GetBondType() != Chem.BondType.SINGLE):
            logger.info("terminal_fragment: branch at locant %s is joined by a "
                        "non-single bond (ylidene); no -ylidene constructor, "
                        "refuse", locant)
            return None
        sub = _terminal_fragment_name(mol, comp, battach)
        if sub is None:
            # Refuse the WHOLE fragment. Emitting the backbone alone would drop
            # this branch's atoms -- the exact defect this module removes.
            logger.info("terminal_fragment: branch at locant %d unnameable; "
                        "refusing the whole fragment", locant)
            return None
        accounted |= set(sub.atoms)
        # (the Blue Book): "Parentheses (round brackets)...
        # are used around compound and complex prefixes"
        # -- a recursively-named branch must be enclosed BEFORE it is spliced
        # in as a token, or the assembled name reads as an unparseable run-on
        # (e.g. '5-1-methylethylnonyl' instead of '5-(1-methylethyl)nonyl').
        # The sort key is computed on the pre-enclosure name; alpha_sort_key
        # strips enclosing marks itself, so this is not an approximation.
        token = enclose_if_compound(sub.name)
        prefix_tokens.append((alpha_sort_key(sub.name), locant, token))
        raw_names.append(sub.name)

    if prefix_tokens:
        book = _book_chain_name(mol, backbone, rp.prefix, name, raw_names,
                                unit_cites_locant=bool(ene or yne))
        name = book if book is not None else _join_prefix_block(
            _assemble_prefixes(prefix_tokens), name)

    result = TerminalFragmentName(name=descriptor + name, numbering=numbering,
                                  basis="chain", atoms=frozenset(accounted))
    if result.atoms != frozenset(frag):
        logger.error("terminal_fragment: completeness invariant violated "
                     "(%d named of %d); refuse",
                     len(result.atoms), len(frag))
        return None
    return result
