"""Phase 160 shared handler helpers (DECOMP-01 + CONTEXT D-08).

Substrate commit 02-00: lazy re-export wrappers around the canonical
implementations in ``composer.py``. Per CONTEXT D-24 incremental-migration
discipline, composer.py STILL OWNS:

* ``_name_iso_x_cyanate`` (composer.py:2204; 27 LOC) — shared between
  isocyanate (commit 02-04) and isothiocyanate (commit 02-05) handlers.
* ``_name_r_group`` (composer.py:2233; 217 LOC) — shared between urea
  (commit 02-08), guanidine (commit 02-09), and the general_acyclic
  catch-all (Plan-03 commit 03-09).

The substrate ships THIS module so handler files can write the
forward-looking import path::

    from ._handler_shared import name_iso_x_cyanate
    from ._handler_shared import name_r_group

while internally the symbols delegate (via lazy import inside each
function body) to composer.py. When Plan-03 commit 03-10 lands, the
function BODIES move here verbatim and composer.py's
``_name_iso_x_cyanate`` + ``_name_r_group`` definitions delete. The
re-export shape ensures handler files do NOT need to change import paths
at thinning time — only this delegation layer flips.

Per CONTEXT D-08 catch-all helper convention: shared logic between
multiple handlers MUST live in this module (not duplicated across
handlers/). The "≥ 2 handler" threshold is per CONTEXT D-03 +
160-AUDIT-DECOMP.md § 1.2 in-file-handler-body inventory.

Anti-pattern hygiene:
- AP-160-02 banned: do NOT group two handlers into one extraction commit
  because they share a body. The handlers are separate files; the SHARED
  helper lives HERE; each handler imports the helper but keeps its own
  file + own atomic commit.
- AP-160-12 banned: handler logic in shim (must be 1-3-line wrapper).
  This module's helpers ARE the multi-line logic; handlers import them.

References:
- composer.py:2204-2230 (``_name_iso_x_cyanate``) — verbatim source.
- composer.py:2233-2449 (``_name_r_group``) — verbatim source.
- 160-PATTERNS.md § "Common Conventions" + line 458.
- 160-CONTEXT.md D-08 + D-24 — catch-all helper convention + migration.

PHASE 160.2 EXTENSION (Plan-02-01; CONTEXT D-02 + D-03):
Six NEW public functions lifted verbatim from composer.py per the
two-step "helpers-first" extraction:
  - _generate_chain_parent  (composer.py:3785-3818, 33 LOC)
  - _generate_ring_parent   (composer.py:3821-3893, 72 LOC)
  - _generate_suffix        (composer.py:3934-4076, 142 LOC)
  - _generate_prefixes      (composer.py:4079-4282, 203 LOC)
  - _generate_stereodescriptors (composer.py:6634-6704, 70 LOC)
  - _assemble_fragments     (composer.py:6799-6954, 155 LOC)

Each function body is COPIED VERBATIM from composer.py per Phase 145.1 D-09
mechanical-translation discipline. composer.py keeps thin re-export shims
(``from .handlers._handler_shared import _generate_chain_parent`` at module
top) so all in-file call sites in _name_oxime_or_hydrazone,
_assemble_amide_name, _assemble_amine_name, _assemble_ring_with_ester_prefixes,
_assemble_complex_ring_name remain byte-identical per CONTEXT D-13
forbidden-boundary preservation.

Per CONTEXT D-09 honest-fail-on-data: any byte-identical canary regression
at commit 02-01 reverts the commit; remediation lands in a follow-up.

Module-level dependencies resolve via LAZY imports inside each function
body to avoid the circular dependency
``_handler_shared.py -> composer.py -> _handler_shared.py`` that the
composer-side shim re-export creates at module-load time. composer.py
imports this module at its top; this module deferring composer.py imports
to first call breaks the cycle while preserving byte-identical behavior.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from ..naming_utils import (
    BRANCH_HANDLED_FGS,
    SIMPLE_MULTIPLIERS,
    alpha_sort_key,
    format_suffix_with_locants,
    get_multiplier_prefix,
    should_omit_locant_one,
)
from ...data.chain_names import get_chain_prefix
from ...rules.locants import get_bond_locants, get_functional_group_locants
from ...rules.seniority import get_prefix, get_suffix


logger = logging.getLogger(__name__)


def name_iso_x_cyanate(
    features: Any, fg_key: str, suffix_word: str,
) -> Optional[str]:
    """Common implementation for isocyanate and isothiocyanate naming.

    Lazy delegate to ``composer.py:_name_iso_x_cyanate`` (composer.py:2204).
    Per CONTEXT D-24 + PATTERNS § 5 first-wave guidance, composer.py owns
    the canonical body at this commit; this wrapper provides the
    forward-looking import path ``handlers._handler_shared.name_iso_x_cyanate``
    for handler files that want stable paths now.

    SMARTS pattern: ``[#6][NX2]=[CX2]=[OX1]`` (isocyanate) or
    ``[#6][NX2]=[CX2]=[SX1]`` (isothiocyanate). Match tuple:
    ``(R_carbon, N, C, O/S)``.

    Args:
        features: MolecularFeatures object.
        fg_key: Either ``'isocyanate'`` or ``'isothiocyanate'``.
        suffix_word: The functional-class suffix word
            (``'isocyanate'`` / ``'isothiocyanate'``).

    Returns:
        Functional class name like ``'methyl isocyanate'``, or None.

    See Also:
        composer.py:_name_iso_x_cyanate — canonical implementation.
        composer.py:_name_isocyanate / _name_isothiocyanate — single-line
            callers; both move to handlers/{isocyanate,isothiocyanate}.py.
    """
    # Lazy import per PATTERNS § Lazy Import.
    from ..composer import _name_iso_x_cyanate
    return _name_iso_x_cyanate(features, fg_key, suffix_word)


def name_r_group(
    mol: Any, start_idx: int, exclude_atoms: set,
) -> Optional[str]:
    """Name an R group (substituent fragment) starting from start_idx.

    Lazy delegate to ``composer.py:_name_r_group`` (composer.py:2233).
    Per CONTEXT D-24 + PATTERNS § 5 first-wave guidance.

    Args:
        mol: RDKit Mol object.
        start_idx: Atom index of the R-group attachment point.
        exclude_atoms: Set of atom indices to exclude from the R-group walk
            (e.g., the functional group atoms already named).

    Returns:
        Substituent name as IUPAC P-29 substituent prefix (e.g., ``'methyl'``,
        ``'phenyl'``, ``'4-chlorophenyl'``), or None on failure.

    See Also:
        composer.py:_name_r_group — canonical implementation; consumed by
            urea, guanidine, isocyanate, isothiocyanate, and the
            general_acyclic catch-all.
    """
    # Lazy import per PATTERNS § Lazy Import.
    from ..composer import _name_r_group
    return _name_r_group(mol, start_idx, exclude_atoms)


def cached_is_complex_ring_system(features: Any) -> bool:
    """WR-02: per-features memoization of composer._is_complex_ring_system.

    The SMARTS-based complex-ring check is heavy; predicates that call it
    inside the dispatch loop violate the spirit of CONTEXT D-25 (predicates
    are pure read-only over already-perceived state). Cache the result on
    the features object as a private attribute so partial_sat / polycyclic /
    ring_ester predicates share a single SMARTS evaluation per features
    instance instead of running it three times per dispatch.

    The cache is per-features-instance state owned by features itself; the
    predicate remains pure with respect to shared/global state.

    Args:
        features: MolecularFeatures-like object with a ``mol`` attribute.

    Returns:
        True iff the molecule is a complex ring system per the SMARTS check.
    """
    cached = getattr(features, "_cached_complex_ring_system_result", None)
    if cached is not None:
        return cached
    from ..composer import _is_complex_ring_system
    result = bool(_is_complex_ring_system(features.mol))
    try:
        features._cached_complex_ring_system_result = result
    except (AttributeError, TypeError):
        # Frozen / immutable features: fall back to per-call computation.
        pass
    return result


# =============================================================================
# Phase 160.2 Plan-02-01: 6 _generate_* helpers lifted from composer.py
# verbatim per CONTEXT D-02 + D-03 + Phase 145.1 D-09 mechanical-translation
# discipline. Each function body is byte-identical to its composer.py
# counterpart prior to this commit. Composer-side symbols (NameFragment,
# TERMINAL_GROUPS, _generate_alkyl_prefixes, etc.) resolve via lazy imports
# inside each function body to avoid the import cycle that the
# composer.py shim re-export creates at module-load time.
# =============================================================================


def _generate_chain_parent(features: Any) -> "NameFragment":
    """Generate parent name for acyclic chains.

    Returns a NameFragment where:
    - text: stem + unsaturation_base (e.g., "but", "prop")
    - locants: tuple of (double_bond_locants, triple_bond_locants)
    """
    from ..composer import NameFragment

    chain_length = len(features.principal_chain)

    # Get chain prefix (stem)
    stem = get_chain_prefix(chain_length)

    # Get bond locants if we have atom_to_locant mapping
    double_locants = []
    triple_locants = []
    if features.atom_to_locant:
        double_locants = get_bond_locants(
            features.principal_chain,
            features.double_bonds,
            features.atom_to_locant
        )
        triple_locants = get_bond_locants(
            features.principal_chain,
            features.triple_bonds,
            features.atom_to_locant
        )

    # Store stem as text, bond locants as locants tuple
    # We'll use a nested tuple: ((double_locants), (triple_locants))
    return NameFragment(
        text=stem,
        locants=(tuple(double_locants), tuple(triple_locants)),
        fragment_type="parent"
    )


def _generate_ring_parent(features: Any) -> "NameFragment":
    """
    Generate parent name for cyclic compounds.

    For cycloalkanes: "cyclo" + chain prefix + "an" (e.g., "cyclohexan")
    The final 'e' is added during assembly if no suffix follows.

    For cycloalkenes: "cyclo" + chain prefix + double bond info
    - Mono-cycloalkenes: cyclo + stem + "en" (cyclohexene) - no locant
    - Cycloalkadienes: cyclo + stem + "a" + locants + "dien" (cyclohexa-1,3-diene)

    For other ring types, returns placeholder for now (to be implemented
    in subsequent plans).

    Args:
        features: MolecularFeatures object with ring_type and principal_ring

    Returns:
        NameFragment with parent text and bond locants
    """
    from ..composer import NameFragment

    ring_type = getattr(features, 'ring_type', None)
    principal_ring = getattr(features, 'principal_ring', None)

    if not principal_ring:
        # Fallback: no ring identified -- return empty parent to avoid
        # generating garbled 'cycloane' (cyclo + ane with no stem)
        return NameFragment(text="", fragment_type="parent")

    ring_size = len(principal_ring)

    # Guard: ring_size must produce a valid stem; otherwise return empty parent
    try:
        stem = get_chain_prefix(ring_size)
    except (ValueError, KeyError):
        # Invalid ring size (0, negative, etc.) -- return empty parent
        return NameFragment(text="", fragment_type="parent")

    # Guard: stem must be non-empty to avoid generating 'cycloane'
    if not stem:
        return NameFragment(text="", fragment_type="parent")

    if ring_type == 'cycloalkane':
        # Cycloalkane naming: cyclo + stem + an (e.g., cyclohexan)
        # Return stem - the "ane" will be added in assembly
        return NameFragment(
            text=f"cyclo{stem}",
            locants=((), ()),  # No bond locants for saturated rings
            fragment_type="parent"
        )

    elif ring_type == 'cycloalkene':
        # Get double bond locants from features
        ring_double_bond_locants = getattr(features, 'ring_double_bond_locants', [])

        return NameFragment(
            text=f"cyclo{stem}",
            locants=(tuple(ring_double_bond_locants), ()),  # (double_bond_locants, triple_bond_locants)
            fragment_type="parent"
        )

    elif ring_type == 'aromatic':
        # Aromatic ring type not handled by cyclic naming -- return empty
        # parent to signal that this ring needs a specialized handler
        # (benzene retained name, fused ring dictionary, etc.)
        return NameFragment(text="", fragment_type="parent")

    elif ring_type and ring_type.startswith('heterocyclic'):
        # Heterocyclic ring type not handled by cyclic naming -- return
        # empty parent to signal need for specialized handler
        return NameFragment(text="", fragment_type="parent")

    # Unknown ring type -- return empty parent rather than garbled 'cyclo'
    return NameFragment(text="", fragment_type="parent")


def _generate_suffix(features: Any) -> Optional["NameFragment"]:
    """Generate suffix fragment for principal group.

    Handles locant assignment for the principal functional group:
    - Terminal groups (carboxylic acid, aldehyde): locant always 1, omitted from name
    - Non-terminal groups (alcohol, ketone): locant MUST be included in PIN style
    - Multiple instances: include multiplier (di, tri) and all locants
    """
    from ..composer import NameFragment, TERMINAL_GROUPS

    fg_name = features.principal_group
    if not fg_name:
        return None

    # Correct is_ring: use chain suffix when chain is parent or principal chain exists,
    # ring suffix only when ring is parent and PG is on a ring carbon.
    if getattr(features, 'chain_is_parent', False):
        is_ring = False
    elif features.principal_chain:
        is_ring = False
    else:
        is_ring = features.is_cyclic
    suffix_text = get_suffix(fg_name, is_ring=is_ring)

    if not suffix_text:
        return None

    # Get locants for functional group positions on the chain
    locants = ()
    fg_count = len(features.principal_group_atoms) if features.principal_group_atoms else 1

    # Validate suffix count against parent capacity:
    # The number of suffix groups cannot exceed the number of atoms in the parent
    # structure (chain length or ring size). E.g., ethane (2C) cannot have tetraol.
    if features.principal_chain:
        max_capacity = len(features.principal_chain)
    elif getattr(features, 'oriented_ring', None):
        max_capacity = len(features.oriented_ring)
    else:
        max_capacity = fg_count  # no constraint if we can't determine parent size

    if fg_count > max_capacity:
        fg_count = max_capacity

    if features.principal_chain and features.atom_to_locant and features.principal_group_atoms:
        fg_locants = get_functional_group_locants(
            features.principal_chain,
            features.principal_group_atoms,
            features.atom_to_locant,
            mol=features.mol
        )

        # Deduplicate locants (overlapping SMARTS can produce duplicates)
        fg_locants = sorted(set(fg_locants))

        # Further validate: count should match unique locants when locants exist
        if fg_locants:
            fg_count = len(fg_locants)

        # Terminal groups: locant is implicitly 1, do NOT include in name
        if should_omit_locant_one(context="suffix", fg_type=fg_name):
            locants = ()
        else:
            # For non-terminal groups (alcohol, ketone), include locants
            locants = tuple(fg_locants)

    elif (not features.principal_chain
          and getattr(features, 'oriented_ring', None)
          and features.principal_group_atoms
          and fg_name not in TERMINAL_GROUPS):
        # Ring compounds: build idx_to_locant from oriented_ring and compute
        # suffix locants.  This mirrors the logic in _get_fg_locants() but
        # uses get_functional_group_locants for consistency.
        oriented_ring = features.oriented_ring
        ring_idx_to_locant = {
            atom_idx: pos + 1
            for pos, atom_idx in enumerate(oriented_ring)
        }
        # Map FG atoms to ring locants
        fg_locants = []
        mol = features.mol
        for match in features.principal_group_atoms:
            found = False
            match_set = set(match)
            ring_set_local = set(ring_idx_to_locant.keys())
            # First pass: prefer a ring C bonded to a non-ring atom in
            # the same FG match (the characteristic heteroatom, e.g.,
            # the C in C=O for ketone, C in C-OH for alcohol).
            for atom_idx in match:
                if atom_idx in ring_idx_to_locant:
                    atom = mol.GetAtomWithIdx(atom_idx)
                    if atom.GetSymbol() == 'C':
                        # Check if bonded to a non-ring FG atom
                        has_fg_hetero = any(
                            nbr.GetIdx() in match_set and nbr.GetIdx() not in ring_set_local
                            for nbr in atom.GetNeighbors()
                        )
                        if has_fg_hetero:
                            fg_locants.append(ring_idx_to_locant[atom_idx])
                            found = True
                            break
            if not found:
                # Second pass: any ring C in the match
                for atom_idx in match:
                    if atom_idx in ring_idx_to_locant:
                        atom = mol.GetAtomWithIdx(atom_idx)
                        if atom.GetSymbol() == 'C':
                            fg_locants.append(ring_idx_to_locant[atom_idx])
                            found = True
                            break
            if not found:
                # Fallback: any atom in match on the ring
                for atom_idx in match:
                    if atom_idx in ring_idx_to_locant:
                        fg_locants.append(ring_idx_to_locant[atom_idx])
                        found = True
                        break
            if not found:
                # Try neighbors
                for atom_idx in match:
                    atom = mol.GetAtomWithIdx(atom_idx)
                    for nbr in atom.GetNeighbors():
                        if nbr.GetIdx() in ring_idx_to_locant:
                            fg_locants.append(ring_idx_to_locant[nbr.GetIdx()])
                            found = True
                            break
                    if found:
                        break

        fg_locants = sorted(set(fg_locants))
        if fg_locants:
            fg_count = len(fg_locants)
            locants = tuple(fg_locants)

    # Final safety: reconcile multiplier count with actual locants
    if locants:
        from ...rules.locant_validation import reconcile_multiplier_count
        fg_count = reconcile_multiplier_count(fg_count, list(locants))

    return NameFragment(
        text=suffix_text,
        locants=locants,
        fragment_type="suffix",
        count=fg_count,
    )


def _generate_prefixes(features: Any) -> List["NameFragment"]:
    """
    Generate prefix fragments for alkyl substituents and non-principal groups.

    For alkane/cycloalkane naming, this extracts alkyl substituents from
    features.substituents (for chains) or features.ring_substituents (for rings),
    groups them by name (methyl, ethyl, etc.), and formats with locants and
    multiplicative prefixes.

    When chain_is_parent=True (ring-chain compounds where chain won parent selection),
    rings become substituents and are named as prefixes (phenyl, cyclohexyl, etc.).
    """
    from ..composer import (
        NameFragment,
        _generate_alkyl_prefixes,
        _generate_ring_alkyl_prefixes,
        _generate_ring_substituent_prefixes,
        _get_fg_locants,
        _merge_duplicate_prefixes,
    )

    prefixes = []

    # --- Handle ring-as-substituent prefixes when chain is parent ---
    if getattr(features, 'chain_is_parent', False):
        ring_sub_prefixes = _generate_ring_substituent_prefixes(features)
        prefixes.extend(ring_sub_prefixes)

    # --- Handle alkyl substituents from features.substituents (chains) ---
    if features.substituents and features.mol:
        alkyl_prefixes = _generate_alkyl_prefixes(features)
        prefixes.extend(alkyl_prefixes)

    # --- Handle ring substituents from features.ring_substituents ---
    # Skip when chain_is_parent: ring is a substituent of the chain, not the parent.
    # Ring substituent data was populated for ring-as-substituent naming but should
    # not be used for prefix generation on the chain parent (Phase 139 ARCH-01).
    ring_substituents = getattr(features, 'ring_substituents', None)
    oriented_ring = getattr(features, 'oriented_ring', None)
    # DROP-07 fix: track FG atoms handled by ring alkyl prefixes to prevent
    # double-emission in the global FG loop below
    handled_ring_fg_atoms = frozenset()
    if ring_substituents and features.mol and oriented_ring and not getattr(features, 'chain_is_parent', False):
        ring_prefixes, handled_ring_fg_atoms = _generate_ring_alkyl_prefixes(features)
        prefixes.extend(ring_prefixes)

    # --- Handle non-principal functional groups as prefixes ---
    # When chain_is_parent, skip FGs on ring atoms (already in ring substituent name).
    # Also include non-chain atoms reachable from ring atoms: inner substituents on
    # fused heterocycles (CF3, NO2, CN, etc.) are named as part of the ring compound
    # prefix and must NOT leak as FG prefixes on the parent chain.
    ring_atom_set = set()
    if getattr(features, 'chain_is_parent', False):
        chain_set_for_expand = set(features.principal_chain)
        for rg in getattr(features, 'ring_substituents_as_groups', []):
            ring_atom_set.update(rg)
        # BFS expand: include all atoms reachable from ring atoms that are not
        # on the principal chain (captures inner substituents like CF3, NO2, CN)
        expand_queue = list(ring_atom_set)
        while expand_queue:
            aidx = expand_queue.pop()
            for nbr in features.mol.GetAtomWithIdx(aidx).GetNeighbors():
                nidx = nbr.GetIdx()
                if nidx not in ring_atom_set and nidx not in chain_set_for_expand:
                    ring_atom_set.add(nidx)
                    expand_queue.append(nidx)

    # BUG-B: Collect all substituent branch atom indices.
    # FG matches located entirely on a branch are handled by substituent naming,
    # so we skip them here to avoid double-counting (e.g., standalone "hydroxy"
    # when the branch is already named "(hydroxymethyl)").
    branch_atoms = set()
    if features.substituents:
        for _pos, sub_list in features.substituents.items():
            for sub_atoms in sub_list:
                branch_atoms.update(sub_atoms)
    # Also collect ring substituent branch atoms -- the enumerator now names
    # compound substituents (trifluoromethyl, etc.) on rings, so their FG atoms
    # should not be double-counted as standalone FG prefixes.
    ring_substituents = getattr(features, 'ring_substituents', None)
    if ring_substituents:
        for _ring_idx, sub_list in ring_substituents.items():
            for sub_atoms in sub_list:
                branch_atoms.update(sub_atoms)

    # DEF-4 (P-14.3.4, Phase 171 BBR-ASM): the locant-1 omission decision must use
    # the MOLECULE-WIDE substituent count, not the per-FG-type count. Collect FG
    # prefix specs here, then emit them after the loop once the total is known
    # (composer.py:5044-5066 parity). Without this, a C1 substituent elides its
    # locant whenever its own type-count is 1 even though another substituent exists
    # (e.g. FCCCl -> '1-chloro-2-fluoroethane', not 'chloro-2-fluoroethane').
    fg_prefix_specs = []  # list of (prefix_text, fg_locants)

    for fg_name, matches in features.functional_groups.items():
        if fg_name == features.principal_group:
            continue

        # Unsaturation indicators are NOT functional groups (IUPAC P-31.1).
        # They are handled as -ene/-yne infixes by _build_unsaturation_infix(),
        # not as prefixes.  Skip to avoid false DROP-16 noise.
        if fg_name in ('alkene', 'alkyne'):
            continue

        # Filter out FGs on ring atoms when chain is parent
        if ring_atom_set:
            filtered = []
            for match in matches:
                # Skip FG if ANY atom in the match is part of the ring
                # substituent fragment (includes inner substituents like CF3)
                on_ring = any(a in ring_atom_set for a in match)
                if not on_ring:
                    filtered.append(match)
            matches = filtered

        # DROP-07 fix: skip FG matches already handled as ring substituents
        # by _generate_ring_alkyl_prefixes() to prevent double-emission.
        # FG SMARTS matches include anchor atoms (e.g., C-F match = (C_idx, F_idx)),
        # so check if ANY atom in the match is in the handled set.
        if handled_ring_fg_atoms:
            matches = [m for m in matches if not any(a in handled_ring_fg_atoms for a in m)]

        # BUG-B: Skip simple FG matches on small substituent branches (<=3 carbons)
        # that get named as compound substituents (hydroxymethyl, aminomethyl, etc.)
        # Uses shared BRANCH_HANDLED_FGS from naming_utils (unified in Phase 113).
        if branch_atoms and fg_name in BRANCH_HANDLED_FGS:
            filtered_branch = []
            for match in matches:
                if not all(a in branch_atoms for a in match):
                    filtered_branch.append(match)
                    continue
                # Check if FG is on a small branch with carbons
                # Search both chain substituents and ring substituents
                on_small_branch = False
                # Chain substituents
                if features.substituents:
                    for _pos, sub_list in features.substituents.items():
                        for sub_atoms in sub_list:
                            sub_set = set(sub_atoms)
                            if all(a in sub_set for a in match):
                                c_count = sum(1 for a in sub_atoms
                                              if features.mol.GetAtomWithIdx(a).GetSymbol() == 'C')
                                if 1 <= c_count <= 3:
                                    on_small_branch = True
                                    break
                        if on_small_branch:
                            break
                # Ring substituents (for enumerator-handled compound substituents)
                if not on_small_branch and ring_substituents:
                    for _ring_idx, sub_list in ring_substituents.items():
                        for sub_atoms in sub_list:
                            sub_set = set(sub_atoms)
                            if all(a in sub_set for a in match):
                                c_count = sum(1 for a in sub_atoms
                                              if features.mol.GetAtomWithIdx(a).GetSymbol() == 'C')
                                if 1 <= c_count <= 3:
                                    on_small_branch = True
                                    break
                        if on_small_branch:
                            break
                if not on_small_branch:
                    filtered_branch.append(match)
            original_count = len(matches)
            matches = filtered_branch
            if not matches and original_count > 0:
                # DROP-17: BUG-B removed all matches. Per-branch containment
                # check ensures each match is genuinely on a small branch whose
                # compound name (e.g., "chloromethyl") already includes the FG.
                # Do NOT blindly restore -- that would cause double-emission.
                # Only log for diagnostic purposes.
                logger.debug(
                    "DROP-17 all_filtered: fg_name=%s original_count=%d (branch naming handles these)",
                    fg_name, original_count,
                )

        prefix_text = get_prefix(fg_name)
        if prefix_text and matches:
            count = len(matches)

            # Compute locants by mapping FG anchor atoms to chain positions
            fg_locants = _get_fg_locants(features, fg_name, matches)

            # Reconcile multiplier count with actual locants found
            if fg_locants:
                from ...rules.locant_validation import reconcile_multiplier_count
                count = reconcile_multiplier_count(count, fg_locants)

            if count > 1:
                # Add multiplier
                multiplier = SIMPLE_MULTIPLIERS.get(count, str(count))
                prefix_text = f"{multiplier}{prefix_text}"

            # DEF-4: defer the locant-1 omission decision to the post-loop block,
            # which knows the molecule-wide substituent count.
            fg_prefix_specs.append((prefix_text, fg_locants))
        elif not prefix_text and matches:
            # By-design: FGs using functional class naming (ether, sulfoxide, etc.)
            # don't have prefix forms — handled by specialized naming paths
            logger.debug(
                "DROP-16 substituent_skip: reason=no_fg_prefix_form fg_name=%s match_count=%d",
                fg_name, len(matches),
            )

    # DEF-4 (P-14.3.4, Phase 171 BBR-ASM): emit the collected FG prefixes using a
    # MOLECULE-WIDE substituent count for the locant-1 omission decision. 'prefixes'
    # already holds the alkyl/ring substituents; count their attachment positions
    # plus the FG positions. The omission only fires for a genuinely single-
    # substituent parent (total == 1); for >= 2 substituents every locant is cited
    # (mirrors composer.py:5044-5066; gold FCCCl -> '1-chloro-2-fluoroethane').
    chain_len = len(getattr(features, 'principal_chain', []))
    existing_sub_positions = sum(len(p.locants) if p.locants else 1 for p in prefixes)
    fg_sub_positions = sum(len(locs) if locs else 1 for _txt, locs in fg_prefix_specs)
    total_substituents = existing_sub_positions + fg_sub_positions
    for prefix_text, fg_locants in fg_prefix_specs:
        # NOTE: do NOT add an outer `and total_substituents == 1` guard. The
        # is_monosubstituted arg already encodes the single-substituent condition,
        # and should_omit_locant_one Rule 1 (chain_length == 1) must still omit for
        # methane regardless of substituent count (e.g. CBr4 -> 'tetrabromomethane',
        # NOT '1,1,1,1-tetrabromomethane').
        omit_locants = should_omit_locant_one(
            context="prefix",
            chain_length=chain_len,
            is_monosubstituted=(total_substituents == 1 and fg_locants == [1]
                                and features.principal_group is None),
        )
        prefixes.append(NameFragment(
            text=prefix_text,
            locants=tuple(sorted(fg_locants)) if (fg_locants and not omit_locants) else (),
            fragment_type="prefix"
        ))

    # --- Merge duplicate prefix names ---
    # If the same base prefix name appears multiple times (from different sources),
    # merge them into a single entry with combined count and appropriate multiplier.
    # This prevents stacking like "dihydroxyhydroxy" -> should be "trihydroxy".
    prefixes = _merge_duplicate_prefixes(prefixes)

    return prefixes


def _generate_stereodescriptors(features: Any, atom_to_locant_override: Optional[Dict[int, int]] = None) -> Optional["NameFragment"]:
    """
    Generate stereodescriptor prefix using correct IUPAC locants.

    Uses the stereochemistry rules module to collect R/S and E/Z descriptors
    based on the atom_to_locant mapping (from chain/ring orientation).

    For different compound types:
    - Acyclic: uses features.atom_to_locant (from principal chain orientation)
    - Heterocycles: uses features.heterocycle_atom_to_locant
    - Cycloalkanes/cycloalkenes: builds from features.oriented_ring

    When atom_to_locant_override is provided, it takes priority over all
    features.* fields. This allows early-return handlers to pass the correct
    locant map for the structure they are naming (which may differ from
    features.principal_ring).

    Args:
        features: MolecularFeatures object with stereocenters and/or double_bond_stereo
        atom_to_locant_override: Optional explicit locant map. When provided,
            bypasses the features.* priority chain entirely.

    Returns:
        NameFragment with stereodescriptor prefix like "(2R)-" or "(2E,3R)-",
        or None if no stereodescriptors.
    """
    from ..composer import NameFragment
    from ...rules.stereochemistry import collect_stereodescriptors, format_stereodescriptor_string

    # Need either stereocenters or double_bond_stereo
    if not features.stereocenters and not getattr(features, 'double_bond_stereo', None):
        return None

    mol = features.mol

    # When an explicit override is provided, use it directly instead of
    # reading from features.* fields. This allows early-return handlers
    # to pass the correct locant map for the structure they are naming
    # (which may differ from features.principal_ring).
    if atom_to_locant_override:
        atom_to_locant = atom_to_locant_override
    else:
        # Existing priority chain (unchanged):
        atom_to_locant = features.atom_to_locant

        # For heterocycles, use ring-specific mapping
        if getattr(features, 'heterocycle_atom_to_locant', None):
            atom_to_locant = features.heterocycle_atom_to_locant
        elif getattr(features, 'oriented_ring', None):
            # Build mapping from oriented_ring for cycloalkanes/cycloalkenes
            # oriented_ring is a list of atom indices in ring order starting at position 1
            # This creates ring_atom_to_locant: {atom_idx: ring_locant} where locants are 1-indexed
            atom_to_locant = {idx: pos + 1 for pos, idx in enumerate(features.oriented_ring)}

    if not atom_to_locant:
        return None

    # Collect stereodescriptors with proper locants.
    # Enable near-parent E/Z detection for top-level naming only:
    # substituent E/Z bonds one hop from the parent ring/chain should
    # be included in the stereo block per IUPAC P-93.5.2.
    descriptors = collect_stereodescriptors(
        mol, atom_to_locant, include_near_parent_ez=True
    )

    if not descriptors:
        return None

    # Format as "(2R,3S)-" etc
    text = format_stereodescriptor_string(descriptors)

    return NameFragment(text=text, fragment_type="stereo")


def _assemble_fragments(fragments: List["NameFragment"], style: str) -> str:
    """
    Assemble fragments into final name string.

    Order: stereo + prefixes (alphabetized) + parent + suffix

    For functional group compounds, uses PIN-style infix locants:
    - propan-1-ol (not propanol or 1-propanol)
    - butan-2-one (not butanone or 2-butanone)

    For unsaturated hydrocarbons, includes bond locants:
    - but-1-ene (not butene or 1-butene)
    - pent-1-en-4-yne (enyne)
    """
    from ..composer import (
        NameFragment,
        _build_hydrocarbon_name,
        _build_unsaturation_infix,
        _estimate_parent_size_from_name,
        _join_prefix_to_name,
        _join_prefixes,
    )

    stereo = ""
    prefixes = []
    parent_frag = None
    suffix_frag = None

    for frag in fragments:
        if frag.fragment_type == "stereo":
            stereo = frag.text
        elif frag.fragment_type == "prefix":
            prefixes.append(frag)
        elif frag.fragment_type == "parent":
            parent_frag = frag
        elif frag.fragment_type == "suffix":
            suffix_frag = frag

    # ----------------------------------------------------------------
    # Detect suffix-prefix locant collisions on ring systems.
    # A collision occurs when a suffix locant (e.g., ketone at position 3)
    # and a prefix locant (e.g., methyl at position 3) share the same
    # numeric value.  OPSIN interprets this as both groups on the same
    # carbon, producing an unphysical valency.
    # Resolution: remove the colliding prefix locant (suffix has priority
    # per IUPAC P-14.7).  Only applies to ring parents.
    # ----------------------------------------------------------------
    if suffix_frag and suffix_frag.locants and prefixes:
        # Determine if parent is a ring (collision only matters for rings)
        parent_text = parent_frag.text if parent_frag else ""
        is_ring_parent = any(
            kw in parent_text.lower()
            for kw in ('cyclo', 'benz', 'pyrid', 'pyrrol', 'furan',
                       'thiophen', 'imidazol', 'naphthal', 'indol',
                       'quinol', 'pyrimid', 'pyrazin', 'oxazol',
                       'thiazol', 'triazol', 'morpholin', 'piperidin',
                       'pyrrolidin', 'aziridin', 'oxiran', 'thiiran',
                       'oxetan', 'azetidin', 'thietan')
        )
        if is_ring_parent:
            from ...rules.locant_validation import detect_locant_collisions

            suffix_locants_list = list(suffix_frag.locants)
            prefix_locant_groups = [
                list(p.locants) for p in prefixes if p.locants
            ]
            parent_size = _estimate_parent_size_from_name(parent_text)

            if suffix_locants_list and prefix_locant_groups:
                collisions = detect_locant_collisions(
                    suffix_locants_list,
                    prefix_locant_groups,
                    parent_type="ring",
                    parent_size=parent_size,
                )
                if collisions:
                    import logging
                    _log = logging.getLogger(__name__)
                    collision_set = set(loc for _, loc in collisions)
                    adjusted = []
                    for pf in prefixes:
                        if pf.locants:
                            new_locants = tuple(
                                l for l in pf.locants if l not in collision_set
                            )
                            if new_locants != pf.locants:
                                _log.debug(
                                    "Collision resolved: removed prefix locant(s) %s "
                                    "for '%s' (suffix has priority per IUPAC P-14.7)",
                                    set(pf.locants) - set(new_locants),
                                    pf.text,
                                )
                                pf = NameFragment(
                                    text=pf.text,
                                    locants=new_locants,
                                    fragment_type=pf.fragment_type,
                                    count=len(new_locants) if new_locants else pf.count,
                                )
                        adjusted.append(pf)
                    prefixes = adjusted

    # Alkyl prefixes are already sorted by _generate_alkyl_prefixes.
    # For non-alkyl prefixes added later, sort all together.
    prefixes.sort(key=lambda f: alpha_sort_key(f.text))

    # Build prefix strings with locants
    # IUPAC rule: hyphens separate locants from names, and are needed
    # between prefixes when one ends with a letter and the next starts with a digit
    # NOTE: Some prefixes already have locants baked in (ring substituent prefixes
    # like "4-phenyl"). Only add locants to those that don't already have them.
    import re
    prefix_texts = []
    for f in prefixes:
        text = f.text
        already_has_locant = bool(re.match(r'^\d', text))
        if f.locants and not already_has_locant:
            loc_str = ",".join(str(l) for l in f.locants)
            prefix_texts.append(f"{loc_str}-{text}")
        else:
            prefix_texts.append(text)
    prefix_str = _join_prefixes(prefix_texts)

    # Extract parent info: stem is in text, bond locants in locants
    stem = parent_frag.text if parent_frag else ""
    double_locants = []
    triple_locants = []
    if parent_frag and parent_frag.locants:
        # locants is a tuple of (double_bond_locants, triple_bond_locants)
        double_locants = list(parent_frag.locants[0]) if parent_frag.locants[0] else []
        triple_locants = list(parent_frag.locants[1]) if len(parent_frag.locants) > 1 and parent_frag.locants[1] else []

    # Handle suffix attachment using PIN-style formatting
    if suffix_frag and suffix_frag.text:
        suffix_text = suffix_frag.text
        suffix_locants = list(suffix_frag.locants) if suffix_frag.locants else []

        # Determine multiplier for multiple functional groups
        # Use suffix_frag.count (set by _generate_suffix) which includes terminal
        # groups like diacids where locants are omitted but multiplier is needed
        count = max(len(suffix_locants), getattr(suffix_frag, 'count', 1))
        multiplier = get_multiplier_prefix(count, suffix_text) if count > 1 else ""

        # Build unsaturation infix with locants for compounds with functional groups
        unsaturation_infix = _build_unsaturation_infix(double_locants, triple_locants)

        name = format_suffix_with_locants(
            stem,
            unsaturation_infix,
            suffix_text,
            suffix_locants,
            multiplier
        )
    else:
        # No suffix = hydrocarbon, build name with unsaturation.
        # WSD-06 (NUM-01): a substituted cycloalkene must keep its ring ene-locant
        # (`3-bromocyclohex-1-ene`), so the bond-locant is omittable ONLY when the
        # ring is unsubstituted (no substituent prefixes). Acyclic / non-cyclo stems
        # pass None to preserve the existing count-proxy behavior.
        _ring_bond_omittable = (not prefix_str) if stem.startswith("cyclo") else None
        name = _build_hydrocarbon_name(
            stem, double_locants, triple_locants,
            ring_bond_locant_omittable=_ring_bond_omittable,
        )

    # Add prefixes with proper hyphenation at boundary
    if prefix_str:
        name = _join_prefix_to_name(prefix_str, name)

    # Add stereodescriptors at the very start
    if stereo:
        name = f"{stereo}{name}"

    return name


__all__ = [
    "name_iso_x_cyanate",
    "name_r_group",
    "cached_is_complex_ring_system",
    "_generate_chain_parent",
    "_generate_ring_parent",
    "_generate_suffix",
    "_generate_prefixes",
    "_generate_stereodescriptors",
    "_assemble_fragments",
]
