"""Systematic substituted-purine naming retained purine parent; fixed
purine numbering; indicated H derived from the graph; / amine suffix).

Names a SUBSTITUTED purine ring system -- the adenine/hypoxanthine/purine
skeleton carrying ring-N substituents and/or exocyclic characteristic groups --
as a whole-molecule parent (`9-methyl-9H-purin-6-amine`) and (Task 3) as a
`-yl` substituent (`6-amino-9H-purin-9-yl`). The bare bases are named here too
('9H-purin-6-amine'): 'adenine', 'guanine' and 'hypoxanthine' do not occur in
the Blue Book (0 hits) and are not retained heterocycle names; the ring
system is purine ("the PIN is 7H-purine", the Blue Book).

Modelled on rules/purine_oxo.py: an atom-mapped skeleton SMARTS whose map
numbers ARE the fixed IUPAC purine locants, per-structure indicated H derived
from the actually-saturated ring nitrogen (NOT hardcoded, so it can never
contradict the input tautomer on round-trip), fail-closed substituent
collection, and reuse of the existing fused-heterocycle assembler.

Accuracy-first, fail-closed: returns None (caller falls through; the /
OPSIN gate is the final backstop) unless the whole purine ring system is
accounted for and every substituent is identifiable.
"""
from collections import defaultdict
from typing import Optional

from rdkit import Chem

# Bare purine skeleton, IUPAC locants encoded as atom-map numbers.
# 6-ring: N1-C2-N3-C4... C5-C6-N1 5-ring: C4-N9-C8-N7-C5 (fused C4-C5)
# ``~`` (any bond) makes the match Kekule/aromatic independent. The carbon
# skeleton is asymmetric (N1 neighbours the degree-2 C6; N3 neighbours the
# degree-3 fusion C4), so the match orientation -- hence the numbering -- is
# unique. VERIFIED: single match assigning purine's fixed numbering on adenine,
# 9-methyladenine, 6-chloropurine and N9-substituted adenine.
_CORE_SMARTS = "[#7:1]1~[#6:2]~[#7:3]~[#6:4]2~[#7:9]~[#6:8]~[#7:7]~[#6:5]2~[#6:6]1"
_PURINE_CORE = Chem.MolFromSmarts(_CORE_SMARTS)
_MAP_NUMS = [a.GetAtomMapNum() for a in _PURINE_CORE.GetAtoms()]


def _purine_atom_mapping(match) -> dict:
    """mol-atom-index -> IUPAC purine locant, from a substructure match tuple."""
    return {midx: _MAP_NUMS[pidx] for pidx, midx in enumerate(match)}


def _core_is_own_ring_system(mol, core_atoms, get_ring_systems) -> bool:
    covered = set()
    for rs in get_ring_systems(mol):
        rs = set(rs)
        if rs & core_atoms:
            covered |= rs
    return covered == set(core_atoms)


def _purine_indicated_h(mol, loc_to_idx, core_atoms) -> Optional[int]:
    """Locant (7 or 9) of the saturated five-ring nitrogen -- the indicated-H
    site -- derived per structure. Reuses purine_oxo's proven logic (exactly one
    of N7/N9 must be pyrrole-type; else None)."""
    from .purine_oxo import _saturated_five_ring_locant
    return _saturated_five_ring_locant(mol, loc_to_idx, core_atoms)


def name_substituted_purine(mol) -> Optional[str]:
    """Systematic PIN for a SUBSTITUTED purine, else None."""
    if mol is None or _PURINE_CORE is None:
        return None
    matches = mol.GetSubstructMatches(_PURINE_CORE, uniquify=False)
    if not matches:
        return None

    from ..perception.rings import get_ring_systems
    from .fused_rings import (
        _assemble_fused_heterocycle_name,
        get_fused_heterocycle_substituents,
    )

    for match in matches:
        atom_mapping = _purine_atom_mapping(match)
        core_atoms = set(match)
        ring_atoms = set(atom_mapping)

        # The matched bicyclic must be its own complete ring system (nothing
        # else fused): an extra fused ring is a different, larger parent.
        if not _core_is_own_ring_system(mol, ring_atoms, get_ring_systems):
            continue

        loc_to_idx = {loc: midx for midx, loc in atom_mapping.items()}
        sat = _purine_indicated_h(mol, loc_to_idx, core_atoms)
        if sat is None:
            continue

        subs = get_fused_heterocycle_substituents(mol, atom_mapping)

        # ``get_fused_heterocycle_substituents`` silently ``continue``s past any
        # exocyclic branch it cannot identify, so its output can be MISSING a
        # substituent -- i.e. denote a DIFFERENT molecule. Fail closed at the
        # source rather than trust an incomplete collection -- UNLESS best-effort
        # is on, in which case retry any unidentified/incomplete branch with the
        # general recursive substituent namer (Defect B): this is what lets a
        # giant arm -- e.g. a nucleotide's ribose-diphosphate-pantetheine chain on
        # a purine N9, as in acetyl-CoA -- be named as an ordinary ring substituent
        # instead of silently dropping the whole molecule to a malformed von
        # Baeyer fallback. PIN/default byte-identical: `_best_effort_augment_purine_subs`
        # is a no-op whenever the standard collection already succeeded, and it
        # fails closed (returns None, same as the old bare check) whenever
        # best-effort is off or a branch is still unnameable even by the general
        # recursion.
        subs = _best_effort_augment_purine_subs(mol, atom_mapping, core_atoms, subs)
        if subs is None:
            continue  # a substituent the shared collector -- and, in best-effort
                      # mode, the general recursive namer -- would still silently
                      # omit -> decline rather than name a different molecule

        if subs.get('oxo_substituents'):
            continue  # Tier 1 is non-oxo; oxo purines (hypoxanthine/guanine/
                      # xanthine family) defer to purine_oxo.py. Prevents the
                      # amino+oxo silent-oxo-drop wrong-molecule name.

        # Every purine with at least one exocyclic group is named here, the bare
        # 6-amine included ('7H-purin-6-amine', '9H-purin-6-amine'): 'adenine' is
        # not a Blue Book name (0 hits) and the ring system is purine,
        # "the PIN is 7H-purine", the Blue Book). The unsubstituted parent
        # has no exocyclic group and stays with the catalogue entry '7H-purine'.
        amino = subs.get('amino_substituents') or []
        substituted = bool(
            subs.get('c_substituents') or subs.get('n_substituents')
            or subs.get('other') or subs.get('suffix_groups') or amino
        )
        if not substituted:
            continue

        parent = f"{sat}H-purine"
        return _assemble_fused_heterocycle_name(mol, parent, subs, atom_mapping)

    return None


def _branch_subtree(mol, root_idx: int, core_atoms) -> set:
    """All atoms reachable from `root_idx` (INCLUSIVE) without re-entering
    `core_atoms` -- the full extent of the one exocyclic branch rooted at
    `root_idx`. Used to name a whole branch as a unit even when the shared
    identifier only recognised (or partially recognised) part of it."""
    from collections import deque
    visited = {root_idx}
    queue = deque([root_idx])
    while queue:
        idx = queue.popleft()
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in core_atoms or nidx in visited:
                continue
            visited.add(nidx)
            queue.append(nidx)
    return visited


def _best_effort_augment_purine_subs(mol, atom_mapping, core_atoms, subs):
    """Best-effort completeness rescue for the purine substituent collection.

    `get_fused_heterocycle_substituents` silently drops any exocyclic branch
    `_identify_fused_substituent` cannot fully identify, so callers guard with
    `_exocyclic_atoms_accounted` and fail closed. That is correct for the PIN/
    default tier -- an unidentifiable branch must never be silently omitted --
    but it also means a purine bearing one GIANT arm (e.g. a nucleotide's
    ribose-diphosphate-pantetheine chain on N9, as in acetyl-CoA) can never be
    named at all: the narrow identifier's ring/alkyl/functional-group producers
    were never built to recognise a whole nucleotide tail.

    Under `best_effort_ctx`, retry every branch the standard collector left
    unaccounted (absent OR only partially covered) with the general recursive
    substituent namer (`name_ring_system_substituent`, which itself falls back
    to the plain-chain cascade for a ring-free branch), and inject a
    successfully-named branch into `subs['c_substituents']` at its ring locant
    -- the same bucket a ring-atom substituent with a numeric locant already
    uses (see `get_fused_heterocycle_substituents`'s ring-N-locant comment).

    Returns:
      - `subs` UNCHANGED when the standard collection already accounted for
        every exocyclic atom (byte-identical no-op -- this is the ONLY case
        reached when `best_effort_ctx` is off, since the caller's own
        `_exocyclic_atoms_accounted` check already gated it identically).
      - an AUGMENTED copy of `subs` when best-effort naming closed every gap.
      - `None` (fail closed) when `best_effort_ctx` is off and the standard
        collection was incomplete, or when a branch remains unnameable even
        by the general recursion.
    """
    from .fused_rings import _exocyclic_atoms_accounted, _identify_fused_substituent

    if _exocyclic_atoms_accounted(mol, core_atoms):
        return subs

    from ..metrics.provenance import best_effort_ctx
    if not best_effort_ctx.get():
        return None  # PIN/default: fail closed, exactly as before this fix

    from ..errors import is_refusal_sentinel
    from .ring_substituents import name_ring_system_substituent

    exocyclic = {
        a.GetIdx() for a in mol.GetAtoms()
        if a.GetIdx() not in core_atoms and a.GetAtomicNum() > 1
    }
    subs = dict(subs)
    subs['c_substituents'] = {
        name: list(locants)
        for name, locants in (subs.get('c_substituents') or {}).items()
    }
    accounted: set = set()

    for core_atom_idx, locant in atom_mapping.items():
        core_atom = mol.GetAtomWithIdx(core_atom_idx)
        for neighbor in core_atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx in core_atoms:
                continue

            branch = _branch_subtree(mol, nbr_idx, core_atoms)
            heavy_branch = {
                a for a in branch if mol.GetAtomWithIdx(a).GetAtomicNum() > 1
            }
            if not heavy_branch:
                continue  # nothing heavy on this branch (shouldn't happen)

            sub_info = _identify_fused_substituent(mol, nbr_idx, core_atoms)
            info_atoms = set()
            if sub_info is not None:
                info_atoms = {
                    a for a in sub_info.get('atoms', [])
                    if mol.GetAtomWithIdx(a).GetAtomicNum() > 1
                }
            if sub_info is not None and info_atoms == heavy_branch:
                # Already fully named by the standard identifier -- leave the
                # bucket `get_fused_heterocycle_substituents` already filled.
                accounted |= info_atoms
                continue

            # Unidentified, or only PARTIALLY identified (would otherwise be
            # a silent atom drop naming a different molecule) -- best-effort
            # retry the WHOLE branch as one ring-or-chain substituent.
            name = name_ring_system_substituent(
                mol, sorted(branch), nbr_idx, allow_mancude=True)
            # `is_refusal_sentinel` catches the bare 'substituent' placeholder
            # AND its DECORATED forms (e.g. 'N-substituentformamide',
            # 'N-substituenthydroxyphosphonooxy...') -- a bare equality/space
            # check misses those (errors.py's documented 310/10000-row leak
            # class), so a leaked sentinel could otherwise be welded into the
            # assembled purine name.
            if not name or is_refusal_sentinel(name):
                return None  # still unnameable / sentinel leak -> fail closed

            locants_here = subs['c_substituents'].setdefault(name, [])
            if locant not in locants_here:
                locants_here.append(locant)
            accounted |= heavy_branch

    if accounted != exocyclic:
        return None  # a branch remains unaccounted even after best-effort

    for name in subs['c_substituents']:
        subs['c_substituents'][name].sort()

    return subs


def _external_subtree(mol, attach_idx, core_atoms) -> set:
    """All atoms reachable from `attach_idx` without re-entering
    `core_atoms`, excluding `attach_idx` itself -- the parent-side structure
    this substituent attaches to. Empty if `attach_idx` has no exocyclic
    neighbour (an unattached/free ring, not expected from a real fragment)."""
    from collections import deque
    start = [n.GetIdx() for n in mol.GetAtomWithIdx(attach_idx).GetNeighbors()
             if n.GetIdx() not in core_atoms]
    visited = set(start)
    queue = deque(start)
    while queue:
        idx = queue.popleft()
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in core_atoms or nidx in visited:
                continue
            visited.add(nidx)
            queue.append(nidx)
    return visited


def name_purine_substituent(mol, frag_atoms, attach_idx) -> Optional[str]:
    """Name a purine ring-system fragment as a `-yl` substituent rooted at
    `attach_idx` (a ring atom). Returns `<prefixes>-<indH>H-purin-<loc>-yl` or
    None. C6-amino renders as the `6-amino` PREFIX (purine is a substituent,
    not the parent). Indicated H derived from the graph (tautomer-safe).

    Accuracy-first, fail-closed: declines an oxo-bearing fragment (Tier 1
    substituent scope is non-oxo, the same boundary as
    ``name_substituted_purine``) and any detachable-suffix decoration
    (carboxylic acid etc.) elsewhere on the ring, which this simple `-yl`
    builder cannot combine correctly.
    """
    if mol is None or _PURINE_CORE is None:
        return None
    frag_set = set(frag_atoms)
    matches = mol.GetSubstructMatches(_PURINE_CORE, uniquify=False)
    if not matches:
        return None

    from ..perception.rings import get_ring_systems
    from .fused_rings import (
        _assemble_fused_heterocycle_name,
        get_fused_heterocycle_substituents,
    )

    for match in matches:
        if set(match) != frag_set:
            continue  # the fragment must be exactly the purine ring system
        atom_mapping = _purine_atom_mapping(match)
        core_atoms = set(match)
        if attach_idx not in atom_mapping:
            continue

        if not _core_is_own_ring_system(mol, core_atoms, get_ring_systems):
            continue  # something else is fused on -> a different, larger parent

        attach_loc = atom_mapping[attach_idx]
        loc_to_idx = {loc: midx for midx, loc in atom_mapping.items()}
        sat = _purine_indicated_h(mol, loc_to_idx, core_atoms)
        if sat is None:
            continue

        # Mask off everything past `attach_idx` -- that is the PARENT this
        # substituent attaches to, not a ring decoration. Left visible, the
        # shared collector walks straight past it and misidentifies it as
        # one (e.g. an acetic-acid tail read as a `9-(carboxymethyl)` ring
        # substituent). Rather than mutate `mol` (a bond cut there breaks
        # re-Kekulization of the whole fused ring -- measured), fold the
        # entire externally-reachable subtree into an EXTENDED core-atoms
        # set passed only to the collector: every atom in it is already
        # "core" from the collector's point of view, so its neighbours
        # inside the subtree are never visited and the walk never reaches
        # past the attachment point. The real (narrow) `atom_mapping` --
        # ring atoms only -- still goes to the assembler, so no placeholder
        # locant ever leaks into the assembled name.
        external_subtree = _external_subtree(mol, attach_idx, core_atoms)
        if not external_subtree:
            return None  # attach atom has no exocyclic free valence (e.g. a
                         # fusion carbon C4/C5) -> not a real -yl attachment
        extended_mapping = dict(atom_mapping)
        for idx in external_subtree:
            extended_mapping[idx] = idx  # placeholder; never emitted (no
                                          # exocyclic neighbour of these
                                          # atoms lies outside extended_mapping)

        subs = get_fused_heterocycle_substituents(mol, extended_mapping)

        # Same source-level completeness guard as the parent path (best-effort
        # augmented, see `name_substituted_purine` / `_best_effort_augment_purine_subs`):
        # fail closed on any exocyclic branch the shared collector -- and, in
        # best-effort mode, the general recursive namer -- still can't identify.
        # `atom_mapping` (the real, narrow ring-locant mapping) says WHICH ring
        # positions to inspect; `set(extended_mapping)` (which also covers the
        # masked parent-side subtree) is the exclusion boundary, exactly as the
        # bare `_exocyclic_atoms_accounted` call above used to receive.
        subs = _best_effort_augment_purine_subs(
            mol, atom_mapping, set(extended_mapping), subs)
        if subs is None:
            continue

        if subs.get('oxo_substituents'):
            continue  # Tier 1 substituent scope is non-oxo (same boundary
                       # as name_substituted_purine)
        if subs.get('suffix_groups'):
            continue  # a detachable-suffix decoration elsewhere on the ring
                       # can't be combined with -yl output by this builder

        # C6-amino must be a PREFIX here (substituent context): demote the
        # amine SUFFIX to an `amino` PREFIX at its locant.
        amino_locants = subs.get('amino_substituents') or []
        subs = dict(subs)
        subs['amino_substituents'] = []
        other = list(subs.get('other') or [])
        for loc in amino_locants:
            other.append({'name': 'amino', 'locant': loc})
        subs['other'] = other

        # Assemble prefixes on the bare parent hydride, then attach `-<loc>-yl`.
        parent = f"{sat}H-purine"
        base = _assemble_fused_heterocycle_name(mol, parent, subs, atom_mapping)
        # The three guards above (oxo/suffix declined, amino demoted to a
        # prefix) guarantee the assembler never applies a suffix here, so
        # `base` always ends in the bare parent hydride "...purine". If a
        # future guard edit ever lets a suffix leak through anyway, fail
        # closed here rather than string-surger a truncated name.
        if not base or not base.endswith("purine"):
            return None  # a suffix leaked (guard drift) -> fail closed, never string-surger
        # base is e.g. "6-amino-9H-purine"; convert to "6-amino-9H-purin-9-yl".
        stem = base[:-1] if base.endswith('e') else base
        return f"{stem}-{attach_loc}-yl"

    return None


# --- Mono-6-oxo purines (hypoxanthine / guanine family) ----------------------
#
# Same purine skeleton, but C6 carries a fixed exocyclic =O (pinned in the
# SMARTS below), and C2 is left UNCONSTRAINED so the same match covers both
# hypoxanthine (C2 = plain ring CH) and guanine (C2 = C-NH2). ``~`` (any bond)
# keeps the match Kekule/aromatic independent; the connectivity is asymmetric
# for the same reason as `_CORE_SMARTS` above, so the orientation -- hence the
# numbering -- is unique.
_OXO_CORE_SMARTS = (
    "[#7:1]1~[#6:2]~[#7:3]~[#6:4]2~[#7:9]~[#6:8]~[#7:7]~[#6:5]2~[#6:6]1=O"
)
_OXO_CORE = Chem.MolFromSmarts(_OXO_CORE_SMARTS)
_OXO_MAP_NUMS = [a.GetAtomMapNum() for a in _OXO_CORE.GetAtoms()]

# The added/indicated-H form is {six},{sat}-dihydro-6H, where {sat} (7 or 9) is
# the locant of the SATURATED five-membered-ring nitrogen -- derived per
# structure by `_purine_indicated_h`, exactly as in `purine_oxo.py` -- and {six}
# (1 or 3) that of the saturated six-membered-ring nitrogen
# (`_saturated_six_ring_locant`): the N3-H tautomer is
# '2-amino-3,7-dihydro-6H-purin-6-one', not the 1,7-dihydro tautomer. A C2-amino
# substituent (guanine family), when present, is placed as an ordinary
# `2-amino` PREFIX (never passed through `amino_substituents`): the shared
# `_build_fused_suffix` treats amino as the SUFFIX and DROPS the oxo whenever
# both are present -- a wrong-molecule defect for exactly this family, which
# is why this engine builds the name from a constructive template instead of
# reusing that suffix logic.
_OXO_PARENT_TEMPLATE = "{six},{sat}-dihydro-6H-purin-6-one"


def _saturated_six_ring_locant(mol, loc_to_idx, core_atoms) -> Optional[int]:
    """Return 1 or 3 -- the locant of the saturated (H- or substituent-bearing)
    six-membered-ring nitrogen next to the C6=O. Exactly one of N1/N3 must
    qualify; otherwise None, so the match declines (the name would denote
    another tautomer)."""
    qualifying = []
    for loc in (1, 3):
        idx = loc_to_idx.get(loc)
        if idx is None:
            return None
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetTotalNumHs() >= 1 or any(
                nb.GetIdx() not in core_atoms for nb in atom.GetNeighbors()):
            qualifying.append(loc)
    return qualifying[0] if len(qualifying) == 1 else None

# Only plain hydrocarbon, halogen and bare (-NH2) amino substituents are named
# here; every other substituent type declines (fail-closed) so an unverifiable
# name never leaks. 'amino' is handled separately below (its shared `type` is
# 'functional', ambiguous with hydroxy/alkoxy/etc., which must NOT be accepted).
_OXO_ACCEPTED_TYPES = frozenset({"alkyl", "halogen"})


def _oxo_purine_atom_mapping(match) -> dict:
    """mol-atom-index -> IUPAC purine locant, from a substructure match tuple.

    Filters out the unmapped C6=O oxygen (map number 0), mirroring
    `purine_oxo._CORE`'s identical filter.
    """
    return {
        midx: _OXO_MAP_NUMS[pidx]
        for pidx, midx in enumerate(match)
        if _OXO_MAP_NUMS[pidx]
    }


def _collect_oxo_purine_substituents(mol, atom_mapping, core_atoms, identify_fn, unsat_fn):
    """Gather ring substituents keyed by purine locant, or None (fail-closed)
    if any substituent is unidentifiable or outside the alkyl/halogen/bare-
    amino scope. Mirrors `purine_oxo._collect_substituents`."""
    c_substituents = defaultdict(list)
    other = []

    for midx, locant in atom_mapping.items():
        atom = mol.GetAtomWithIdx(midx)
        for nb in atom.GetNeighbors():
            nidx = nb.GetIdx()
            if nidx in core_atoms:
                continue  # includes the pinned C6=O oxygen

            info = identify_fn(mol, nidx, core_atoms)
            if info is None:
                return None

            sub_type = info.get("type")
            sub_name = info.get("name")

            if sub_type == "functional" and sub_name == "amino":
                # A bare exocyclic -NH2 (guanine's C2-amino, or any other
                # ring position carrying one): always a PREFIX here, never
                # the shared suffix path (see module note above).
                other.append({"name": "amino", "locant": locant})
                continue

            if sub_type not in _OXO_ACCEPTED_TYPES:
                return None  # e.g. a second ring =O (dione/trione), or any
                             # other functional group -> out of scope, decline

            # Mirrors purine_oxo's guard: the shared substituent namer
            # mis-numbers branched aliphatic-unsaturated substituents, so
            # decline rather than trust an unverifiable name.
            if unsat_fn(mol, info.get("atoms") or []):
                return None

            if sub_type == "halogen":
                other.append({"name": sub_name, "locant": locant})
            else:  # alkyl
                c_substituents[sub_name].append(locant)

    for name in c_substituents:
        c_substituents[name].sort()

    return {
        "n_substituents": {},
        "c_substituents": dict(c_substituents),
        "oxo_substituents": [],
        "amino_substituents": [],
        "suffix_groups": {},
        "other": other,
    }


def name_oxo_purine(mol) -> Optional[str]:
    """Systematic PIN for a substituted mono-6-oxo purine -- the hypoxanthine
    or guanine family -- else None.

    The bare parents are named too ('1,7-dihydro-6H-purin-6-one',
    '2-amino-1,9-dihydro-6H-purin-6-one'): 'hypoxanthine' and 'guanine' do not
    occur in the Blue Book (0 hits) and are not retained heterocycle
    names ("the PIN is 7H-purine", the Blue Book).

    Declines (fail-closed):
      - a 2,6-dione (purine_oxo.py's job) or any extra ring oxo (8-oxo /
        trione) -- the pinned single C6=O in the SMARTS means a second ring
        =O is picked up as an unaccepted 'oxo' substituent by the shared
        identifier, so this declines those structurally, defense-in-depth
        alongside the caller's ordering (purine_oxo / xanthine run first);
      - any substituent the shared identifier can't type as plain
        alkyl/halogen/bare-amino (fail-closed).

    This engine owns the -oxo case (hypoxanthine/guanine family) only.
    The purine-2,6-DIONE case (xanthine/caffeine family) lives in
    ``rules/purine_oxo.py::name_purine_26_dione``.
    """
    if mol is None or _OXO_CORE is None:
        return None
    matches = mol.GetSubstructMatches(_OXO_CORE, uniquify=False)
    if not matches:
        return None

    from ..perception.rings import get_ring_systems
    from .fused_rings import (
        _assemble_fused_heterocycle_name,
        _identify_fused_substituent,
    )
    from .purine_oxo import _has_aliphatic_unsaturation

    for match in matches:
        core_atoms = set(match)
        atom_mapping = _oxo_purine_atom_mapping(match)
        ring_atoms = set(atom_mapping)

        # The matched bicyclic must be its own complete ring system; an extra
        # fused ring means a different (larger) parent -> decline.
        if not _core_is_own_ring_system(mol, ring_atoms, get_ring_systems):
            continue

        loc_to_idx = {loc: midx for midx, loc in atom_mapping.items()}
        sat = _purine_indicated_h(mol, loc_to_idx, core_atoms)
        if sat is None:
            continue
        six = _saturated_six_ring_locant(mol, loc_to_idx, core_atoms)
        if six is None:
            continue

        subs = _collect_oxo_purine_substituents(
            mol, atom_mapping, core_atoms,
            _identify_fused_substituent, _has_aliphatic_unsaturation,
        )
        if subs is None:
            continue  # an out-of-scope / unidentifiable substituent

        parent = _OXO_PARENT_TEMPLATE.format(six=six, sat=sat)
        return _assemble_fused_heterocycle_name(mol, parent, subs, atom_mapping)

    return None
