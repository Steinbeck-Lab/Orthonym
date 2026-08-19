"""Systematic substituted-purine naming (P-25 retained purine parent; fixed
purine numbering; indicated H derived from the graph; P-62/P-66 amine suffix).

Names a SUBSTITUTED purine ring system -- the adenine/hypoxanthine/purine
skeleton carrying ring-N substituents and/or exocyclic characteristic groups --
as a whole-molecule parent (`9-methyl-9H-purin-6-amine`) and (Task 3) as a
`-yl` substituent (`6-amino-9H-purin-9-yl`). Declines the bare retained bases
(adenine/guanine/hypoxanthine) for their standard tautomers, which keep their
retained names via another path.

Modelled on rules/purine_oxo.py: an atom-mapped skeleton SMARTS whose map
numbers ARE the fixed IUPAC purine locants, per-structure indicated H derived
from the actually-saturated ring nitrogen (NOT hardcoded, so it can never
contradict the input tautomer on round-trip), fail-closed substituent
collection, and reuse of the existing fused-heterocycle assembler.

Accuracy-first, fail-closed: returns None (caller falls through; the SELF-01 /
OPSIN gate is the final backstop) unless the whole purine ring system is
accounted for and every substituent is identifiable.
"""
from typing import Optional

from rdkit import Chem

# Bare purine skeleton, IUPAC locants encoded as atom-map numbers.
#   6-ring: N1-C2-N3-C4 ... C5-C6-N1   5-ring: C4-N9-C8-N7-C5 (fused C4-C5)
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

    from .fused_rings import (
        get_fused_heterocycle_substituents,
        _assemble_fused_heterocycle_name,
        _exocyclic_atoms_accounted,
    )
    from ..perception.rings import get_ring_systems

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
        # source rather than trust an incomplete collection.
        if not _exocyclic_atoms_accounted(mol, core_atoms):
            continue  # a substituent the shared collector would silently omit
                      # -> decline rather than name a different molecule

        if subs.get('oxo_substituents'):
            continue  # Tier 1 is non-oxo; oxo purines (hypoxanthine/guanine/
                      # xanthine family) defer to purine_oxo.py. Prevents the
                      # amino+oxo silent-oxo-drop wrong-molecule name.

        # Fire ONLY for a genuinely substituted purine. A bare purine base whose
        # only exocyclic group is the retained-defining C6 amino/oxo (adenine /
        # hypoxanthine) keeps its retained name -> decline here. "Substituted"
        # means at least one ring-position substituent beyond that: a C/N-alkyl
        # or aryl prefix, a halogen, a suffix group, or an N-substituent.
        substituted = bool(
            subs.get('c_substituents') or subs.get('n_substituents')
            or subs.get('other') or subs.get('suffix_groups')
        )
        if not substituted:
            continue

        parent = f"{sat}H-purine"
        return _assemble_fused_heterocycle_name(mol, parent, subs, atom_mapping)

    return None


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

    from .fused_rings import (
        get_fused_heterocycle_substituents,
        _assemble_fused_heterocycle_name,
        _exocyclic_atoms_accounted,
    )
    from ..perception.rings import get_ring_systems

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
        extended_mapping = dict(atom_mapping)
        for idx in external_subtree:
            extended_mapping[idx] = idx  # placeholder; never emitted (no
                                          # exocyclic neighbour of these
                                          # atoms lies outside extended_mapping)

        subs = get_fused_heterocycle_substituents(mol, extended_mapping)

        # Same source-level completeness guard as the parent path: fail
        # closed on any exocyclic branch the shared collector can't identify.
        if not _exocyclic_atoms_accounted(mol, set(extended_mapping)):
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
        # base is e.g. "6-amino-9H-purine"; convert to "6-amino-9H-purin-9-yl".
        stem = base[:-1] if base.endswith('e') else base
        return f"{stem}-{attach_loc}-yl"

    return None
