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
        if not subs:
            continue

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
