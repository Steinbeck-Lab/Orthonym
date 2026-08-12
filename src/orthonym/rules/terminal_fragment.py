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
                   ranks: Sequence[int], stop_at_ring: bool = False) -> List[int]:
    """The deepest simple path from ``start`` inside ``atoms``.

    ``atoms`` is acyclic here, so the induced subgraph rooted at ``start`` is a
    tree and the deepest root-to-leaf path is the backbone. Ties are broken on
    the path's canonical-rank sequence, never on atom index.

    ``stop_at_ring`` (v30 ring-branch lever): when the fragment carries ring
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
                    and not (stop_at_ring
                             and mol.GetAtomWithIdx(nb.GetIdx()).IsInRing())]
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
    stereodescriptors at all, and SELF-01 -- the gate that delivers 0-wrong --
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

    Split (v30 internal-C=C lever) into an ATOM half and a BOND half. A defined
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

        (1-oxamethyl)benzene           -> OC1=CC=CC=C1        i.e. PHENOL
        (2-(1-oxamethyl)butyl)benzene  -> OC(CC1=CC=CC=C1)CC  i.e. an ALCOHOL

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
    as ``.
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


def _chain_stem_with_unsaturation(n: int, ene: List[int],
                                  yne: List[int]) -> Optional[str]:
    """'butyl' / 'but-3-en-1-yl' / 'penta-1,3-dien-1-yl' / 'hexa-1,3-dien-5-yn-1-yl'.

    P-31.1.1.1: 'ene' always precedes 'yne', with elision of the final 'e' of
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
    from ..data.chain_names import get_chain_prefix    # 'but' for 4

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
    ``terminal_ring.terminal_ring_name`` -- reused unchanged, so P-22.2.3 skeletal
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
    # The RING's own numbering is authoritative -- it skips 4 -> 4a -> 5, so a
    # position-derived locant would disagree with the name it was spelled from.
    for locant, battach, comp in _branches_off(mol, frag, sorted(core),
                                               tr.numbering):
        _het = _single_heteroatom_branch_prefix(mol, comp)
        if _het is not None:
            accounted |= set(comp)
            prefix_tokens.append((alpha_sort_key(_het), locant,
                                  enclose_if_compound(_het)))
            continue
        # 0-WRONG guard (mirrors the chain loop): a ring decoration joined by a
        # NON-single bond (exocyclic =CH2 / =C<, an ylidene) recurses to a `-yl`
        # token that asserts a single bond -- wrong constitution. Fail closed.
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

    name = tr.name
    if prefix_tokens:
        name = _join_prefix_block(_assemble_prefixes(prefix_tokens), name)

    if accounted != frag:
        logger.error("terminal_fragment: composite completeness violated "
                     "(%d of %d); refuse", len(accounted), len(frag))
        return None
    return TerminalFragmentName(
        name=name, numbering=dict(tr.numbering),
        basis="composite" if prefix_tokens else "ring",
        atoms=frozenset(accounted))


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
    # Stereochemistry (v30 internal-C=C lever). A defined R/S CENTRE is never
    # expressible here (no R/S descriptor is emitted), so it always refuses --
    # SELF-01 is constitution-only and cannot catch a dropped centre. A defined
    # BACKBONE C=C configuration, by contrast, IS emitted as a leading
    # (nE)/(nZ) block on the acyclic path below, so it no longer forces a refusal.
    if _has_defined_atom_stereo(mol, frag):
        logger.info("terminal_fragment: fragment carries a defined stereocentre "
                    "and this module emits no R/S descriptor; refuse")
        return None

    if mol.GetRingInfo().NumAtomRings(attach_idx) > 0:
        # Attachment is ON a ring system -> the ring is the parent (composite
        # path). It emits no stereodescriptor, so a defined ring / ring-adjacent
        # double-bond configuration would be dropped (a wrong molecule SELF-01
        # cannot catch) -- keep refusing it here.
        if _has_defined_bond_stereo(mol, frag):
            logger.info("terminal_fragment: composite fragment carries defined "
                        "bond stereo; refuse (no stereodescriptor on this path)")
            return None
        return _composite_fragment_name(mol, frag, attach_idx)

    # Attachment is ACYCLIC -> the chain path names it. v30 ring-branch lever:
    # when the fragment ALSO carries ring system(s) -- a chain leading to a ring,
    # the case _composite_fragment_name explicitly declines ("attachment is
    # acyclic but the fragment carries a ring; out of implemented scope") -- the
    # backbone STOPS at the ring boundary and each ring system falls out as a
    # decoration recursed through _composite_fragment_name. The ring is spelled
    # by terminal_ring (REPLACEMENT nomenclature), so -C(=O)-Ph emits the
    # ugly-but-RT-correct `1-(cyclohexa-1,3,5-trien-1-yl)-2-oxaeth-1-en-1-yl`
    # and -CH2CH2-cyclohexyl emits `2-(cyclohexan-1-yl)ethyl` (invariant 1's T4
    # clause; retained ring names are a follow-on). The pure-acyclic case (no
    # ring) is stop_at_ring=False -> byte-identical.
    _frag_has_ring = not _is_acyclic(mol, frag)
    ranks = _canonical_ranks(mol)
    backbone = _backbone_from(mol, frag, attach_idx, ranks,
                              stop_at_ring=_frag_has_ring)
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
            ene.append(i + 1)            # cite the LOWER locant (P-31.1.1.1)
        elif bt == Chem.BondType.TRIPLE:
            yne.append(i + 1)
        else:
            # AROMATIC or DATIVE on an acyclic backbone: no morpheme, and
            # spelling it single would denote a different molecule.
            return None

    numbering = {a: i + 1 for i, a in enumerate(backbone)}

    # --- backbone C=C geometry -> leading (nE)/(nZ) block (v30 internal-C=C) ---
    # A defined-stereo backbone double bond is now expressed rather than refused
    # (the atom-stereo guard above already refused any R/S centre). Mirrors
    # assembly.substituent_naming._unsaturated_substituent_name: E/Z from the
    # bond CIP code, locant = the lower backbone locant, merged + locant-ordered.
    # Any in-fragment bond stereo that is neither a backbone bond of THIS chain
    # nor wholly inside a branch (named by the recursion below) fails closed.
    descriptor = ""
    _stereo_bonds = [
        b for b in mol.GetBonds()
        if b.GetStereo() != Chem.BondStereo.STEREONONE
        and b.GetBeginAtomIdx() in frag and b.GetEndAtomIdx() in frag]
    if _stereo_bonds:
        _bb_bond_ids = set()
        for i in range(len(backbone) - 1):
            _bb = mol.GetBondBetweenAtoms(backbone[i], backbone[i + 1])
            if _bb is not None:
                _bb_bond_ids.add(_bb.GetIdx())
        _branch_sets = [set(comp) for _, _, comp in branches]
        from ..perception.stereo import assign_stereochemistry
        if any(not b.HasProp('_CIPCode') for b in _stereo_bonds):
            assign_stereochemistry(mol)
        _ez_bits = []
        for b in _stereo_bonds:
            if b.GetIdx() in _bb_bond_ids:
                if not b.HasProp('_CIPCode'):
                    return None
                _ez = b.GetProp('_CIPCode')
                if _ez not in ('E', 'Z'):
                    return None
                _ez_bits.append((min(numbering[b.GetBeginAtomIdx()],
                                     numbering[b.GetEndAtomIdx()]), _ez))
            elif any(b.GetBeginAtomIdx() in s and b.GetEndAtomIdx() in s
                     for s in _branch_sets):
                continue  # inside a branch -> its own recursion emits the E/Z
            else:
                return None  # off-backbone geometry we cannot place -> refuse
        if _ez_bits:
            _ez_bits.sort()
            descriptor = "(" + ",".join(f"{lc}{ez}" for lc, ez in _ez_bits) + ")-"

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
    name = f"{rp.prefix}{stem}" if rp.prefix else stem

    accounted = set(backbone)
    _backbone_set = set(backbone)
    prefix_tokens: List[tuple] = []          # (alpha_key, locant, token)
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
            continue
        # 0-WRONG guard: a branch joined to the backbone by a NON-single bond is a
        # =/# -attached substituent (methylidene, cyclohexylidene, ...). Recursing
        # it yields a `-yl` (single free valence) token that ASSERTS a single bond
        # -- a wrong CONSTITUTION (`-CH=C<ring` named `...(cyclohexan-1-yl)...`).
        # This module has no -ylidene/-ylidyne constructor, so fail closed. Fixes
        # the ring-ylidene class this lever newly reaches AND the pre-existing
        # acyclic methylidene one (fable ring-review R1).
        _bbn = next((n.GetIdx() for n in mol.GetAtomWithIdx(battach).GetNeighbors()
                     if n.GetIdx() in _backbone_set), None)
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

    result = TerminalFragmentName(name=descriptor + name, numbering=numbering,
                                  basis="chain", atoms=frozenset(accounted))
    if result.atoms != frozenset(frag):
        logger.error("terminal_fragment: completeness invariant violated "
                     "(%d named of %d); refuse",
                     len(result.atoms), len(frag))
        return None
    return result
