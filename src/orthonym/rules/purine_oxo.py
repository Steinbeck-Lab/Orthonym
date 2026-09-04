"""Substituted purine-2,6-dione naming (P-25.2.1 retained purine parent; fixed
purine numbering P-31.1.4.3.4; P-66 dione suffix).

The purine-2,6-dione parent (the xanthine skeleton) is named systematically as
``3,{X}-dihydro-1H-purine-2,6-dione``, where {X} (7 or 9) is the locant of the
saturated five-membered-ring nitrogen -- derived per-structure, NOT hardcoded:
the common methylxanthines (caffeine, theobromine, theophylline, paraxanthine,
the mono-methylxanthines) are the 3,7- form, but an N9-substituted isomer is the
3,9- form. A substituted member's PIN is the substituent prefixes placed at their
fixed purine locants on that parent. All forms were verified by OPSIN round-trip
on 2026-07-17 (named family + novel 1-ethyl-3,7-dimethyl / 8-bromocaffeine /
8-alkyl / N9-substituted patterns).

This engine reads the locants from an atom-mapped substructure match (the purine
numbering is unambiguous because the skeleton is asymmetric) and reuses the
fused-heterocycle substituent identifier/assembler.

Accuracy-first, fail-closed: it returns ``None`` -- so the caller falls through
to another handler and a wrong name is never emitted -- unless every substituent
is a hydrocarbyl (alkyl/aryl, as typed 'alkyl' by the shared identifier) or a
halogen on a bare purine-2,6-dione ring system whose imidazole has exactly one
saturated N. Functional/heteroatom substituents, a C8=O (purine-2,6,8-trione,
i.e. the uric-acid family), extra fused rings, and the unsubstituted parent all
decline here (the unsubstituted parent keeps its retained name via another path).

This module owns the purine-2,6-DIONE (xanthine/caffeine family) only. The
MONO-6-oxo case (hypoxanthine/guanine) lives in ``rules/purine.py::name_oxo_purine``.
"""
from collections import defaultdict
from typing import Optional

from rdkit import Chem

# purine-2,6-dione core with IUPAC locants encoded as atom-map numbers.
#   6-ring: N1-C2(=O)-N3-C4 ... C5-C6(=O)-N1
#   5-ring: C4-N9-C8-N7-C5      (fused across the shared C4-C5 bond)
# ``~`` (any bond) makes the match independent of Kekule/aromatic perception;
# the two ring carbonyls are pinned by explicit ``=O``. The connectivity is
# asymmetric (N1 is the only ring N between two carbonyls; C2 the only ring C
# between two N; N7 sits on the C5/C6 side), so the match orientation -- hence
# the numbering -- is unique.
_CORE_SMARTS = "[#7:1]1~[#6:2](=O)~[#7:3]~[#6:4]2~[#7:9]~[#6:8]~[#7:7]~[#6:5]2~[#6:6]1=O"
_CORE = Chem.MolFromSmarts(_CORE_SMARTS)
_MAP_NUMS = [a.GetAtomMapNum() for a in _CORE.GetAtoms()]

# The purine-2,6-dione added/indicated-H form is 3,{X}-dihydro-1H, where {X} is
# the locant (7 or 9) of the SATURATED five-membered-ring nitrogen -- N1/N3 are
# always saturated (amide-type, between the carbonyls); exactly one of N7/N9 is
# the pyrrole-type (H/substituent-bearing) N and the other is the pyridine-type
# =N-. {X} is derived per-structure (NOT hardcoded 7): an N9-substituted isomer
# is 3,9-dihydro-1H, not 3,7-.
_PARENT_TEMPLATE = "3,{sat}-dihydro-1H-purine-2,6-dione"
# Only plain hydrocarbon and halogen substituents are named here; every other
# substituent type declines (fail-closed) so an unverifiable name never leaks.
_ACCEPTED_TYPES = frozenset({"alkyl", "halogen"})


def name_purine_26_dione(mol) -> Optional[str]:
    """Return the systematic PIN for a substituted purine-2,6-dione, else None."""
    if mol is None:
        return None
    matches = mol.GetSubstructMatches(_CORE, uniquify=False)
    if not matches:
        return None

    # Lazy import: fused_rings imports data modules, not this one, so importing
    # its helpers here (call time) avoids a rules<->rules import cycle at load.
    from ..perception.rings import get_ring_systems
    from .fused_rings import (
        _assemble_fused_heterocycle_name,
        _identify_fused_substituent,
    )

    for match in matches:
        # ``match`` also covers the two carbonyl =O atoms (unmapped in the
        # SMARTS): exclude all of them from substituent search, but test the
        # ring-system guard against the 9 mapped RING atoms only.
        core_atoms = set(match)
        atom_mapping = {
            midx: _MAP_NUMS[pidx]
            for pidx, midx in enumerate(match)
            if _MAP_NUMS[pidx]
        }
        ring_atoms = set(atom_mapping)

        # The matched bicyclic must be its own complete ring system; any extra
        # ring fused to it means a different (larger) parent -> decline.
        if not _core_is_own_ring_system(mol, ring_atoms, get_ring_systems):
            continue

        # Locant of the saturated five-membered-ring N (7 or 9), or None if the
        # imidazole saturation is not a clean single-pyrrole-N pattern.
        loc_to_idx = {loc: midx for midx, loc in atom_mapping.items()}
        sat = _saturated_five_ring_locant(mol, loc_to_idx, core_atoms)
        if sat is None:
            continue

        subs = _collect_substituents(
            mol, atom_mapping, core_atoms, _identify_fused_substituent
        )
        if subs is None:
            continue  # an out-of-scope / unidentifiable substituent
        if not subs["c_substituents"] and not subs["other"]:
            continue  # unsubstituted parent -> defer to the retained-name path

        parent = _PARENT_TEMPLATE.format(sat=sat)
        return _assemble_fused_heterocycle_name(mol, parent, subs, atom_mapping)

    return None


def _saturated_five_ring_locant(mol, loc_to_idx, core_atoms) -> Optional[int]:
    """Return 7 or 9 -- the locant of the saturated (pyrrole-type, H- or
    substituent-bearing) five-membered-ring nitrogen. Exactly one of N7/N9 must
    qualify (the other is the pyridine-type =N-); otherwise return None so the
    match declines (an imidazole with both or neither saturated is not a plain
    purine-2,6-dione)."""
    qualifying = []
    for loc in (7, 9):
        idx = loc_to_idx.get(loc)
        if idx is None:
            return None
        atom = mol.GetAtomWithIdx(idx)
        has_h = atom.GetTotalNumHs() >= 1
        has_exocyclic = any(
            nb.GetIdx() not in core_atoms for nb in atom.GetNeighbors()
        )
        if has_h or has_exocyclic:
            qualifying.append(loc)
    return qualifying[0] if len(qualifying) == 1 else None


def _has_aliphatic_unsaturation(mol, atoms) -> bool:
    """True if any bond WITHIN the substituent fragment is a non-aromatic double
    or triple bond -- the signature the shared substituent namer mis-handles.
    Aromatic substituents (phenyl etc.) use aromatic bonds and are not flagged."""
    aset = set(atoms)
    for a in atoms:
        for bond in mol.GetAtomWithIdx(a).GetBonds():
            if bond.GetOtherAtomIdx(a) not in aset:
                continue
            if bond.GetIsAromatic():
                continue
            if bond.GetBondTypeAsDouble() >= 2.0:
                return True
    return False


def _core_is_own_ring_system(mol, core_atoms, get_ring_systems) -> bool:
    """True iff the purine core is exactly its fused ring system (nothing else
    fused to it)."""
    covered = set()
    for rs in get_ring_systems(mol):
        rs = set(rs)
        if rs & core_atoms:
            covered |= rs
    return covered == core_atoms


def _collect_substituents(mol, atom_mapping, core_atoms, identify_fn):
    """Gather ring substituents keyed by purine locant, or None (fail-closed) if
    any substituent is unidentifiable or outside the alkyl/halogen scope."""
    c_substituents = defaultdict(list)
    other = []

    for midx, locant in atom_mapping.items():
        atom = mol.GetAtomWithIdx(midx)
        for nb in atom.GetNeighbors():
            nidx = nb.GetIdx()
            # ``core_atoms`` already contains the two parent -2,6-dione carbonyl
            # oxygens (they are part of the SMARTS match: [#6:2](=O) / ...=O), so
            # they are skipped here. Any OTHER exocyclic =O -- e.g. the C8=O of a
            # purine-2,6,8-trione (uric acid family) -- is NOT in the pattern and
            # deliberately falls through to identify_fn below, which types it as
            # 'oxo'. 'oxo' is not an accepted type, so the whole match declines:
            # a trione must never be named as a dione (dropping the 8-oxo would
            # name a different molecule).
            if nidx in core_atoms:
                continue

            info = identify_fn(mol, nidx, core_atoms)
            if info is None or info.get("type") not in _ACCEPTED_TYPES:
                return None

            # The shared substituent namer mis-numbers BRANCHED aliphatic-
            # unsaturated substituents (e.g. prenyl -> "2-methylbut-2-enyl", a
            # different molecule). We cannot verify a substituent name without
            # OPSIN (which fails open with no Java), so -- accuracy-first -- the
            # engine declines ANY substituent carrying an aliphatic (non-aromatic)
            # C=C/C#C rather than trust an unverifiable name. Saturated alkyl,
            # aryl (aromatic bonds), cycloalkyl and halogen are unaffected.
            if _has_aliphatic_unsaturation(mol, info.get("atoms") or []):
                return None

            if info["type"] == "halogen":
                other.append({"name": info["name"], "locant": locant})
            else:  # alkyl
                c_substituents[info["name"]].append(locant)

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
