"""Phase 151 D-04: Von Baeyer naming for tetracyclic and pentacyclic systems.

Replaces the routing helpers at tricyclo.py:556 (_generate_higher_polycyclo_
descriptor) and tricyclo.py:636 (_name_higher_polycyclo_system) with a
dedicated module that exposes the cascade-step-6 supplier Phase 147
reserved as a stub. bicyclo.py and tricyclo.py REMAIN authoritative for
their proven 2-ring and 3-ring cases per CONTEXT D-04.

Audit verdict (151-AUDIT-A.md): THIN_WRAPPER. The existing
src/orthonym/rules/polycyclic.py::VonBaeyerAnalyzer (2,537 LOC, runs
VB-1..VB-7) produces structurally and chemically correct VB-4/VB-5
descriptors for every Blue Book example audited. Round-trip via OPSIN
yields InChI L1 == input on every named in-scope row. Coverage invariant
(cascade-step-6 gate) holds without modification.

This module therefore wraps VonBaeyerAnalyzer rather than re-implementing.

IUPAC Reference: Blue Book 2013 P-23.3 (tricyclic and higher), P-23.2.5
(numbering cascade), P-25.2 (heteroatom 'a'-prefix replacement).

Source:
 - .planning/phases/151-*/151-CONTEXT.md (D-04, D-05, D-06, D-07, D-08, D-12, D-21).
 - .planning/phases/151-*/151-RESEARCH.md §"Existing Code Audit", §"Risk Landmines".
 - .planning/phases/151-*/151-AUDIT-A.md verdict THIN_WRAPPER.
 - .planning/references/AUTONOM-1990-insights.md §3 (symmetric-ring locant
   generation), §4 (hybrid dictionary + algorithmic).

Naming-collision note (CONTEXT D-06 deviation; RESEARCH Q-01):
 The cascade-step-6 supplier here is named ``get_higher_polycyclo_iupac_locants``
 — distinct from ``polycyclics.py:386::get_polycyclic_iupac_locants(mol,
 pah_name)`` which serves cataloged PAHs by name lookup. The two
 functions have different signatures and different responsibilities.
 The literal CONTEXT D-06 identifier (`get_polycyclic_iupac_locants`)
 was renamed here to prevent an obscure import-shadowing bug.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple, Union

from rdkit import Chem

logger = logging.getLogger(__name__)

# D-07 reuse — single canonical comparator across the project (no parallel
# locant ranking ever introduced). This import is kept even though the
# THIN_WRAPPER verdict means we don't currently invoke it directly: the
# unit-test suite (test_polycyclic_von_baeyer.py::TestNoParallelComparator)
# verifies its presence as a structural lock per CONTEXT D-07 / D-20.
# Future v19 P-23.2.5(a) tiebreak refinement will use it.
from .locants import compare_locant_sets  # noqa: F401

# Engine — VonBaeyerAnalyzer (audit verdict THIN_WRAPPER per 151-AUDIT-A.md).
from .polycyclic import VonBaeyerAnalyzer

# D-04 anti-canary lock — bicyclo / tricyclo input must NOT be re-routed
# through the new module; bicyclo.py and tricyclo.py keep authority on
# their 99 + 13 canary blast-radius compounds (RESEARCH "Risk Landmines").
from .polycyclic_bridged import classify_bridged_system

# D-08 retained-name passthrough (AUTONOM §4 hybrid pattern).
# adamantane / twistane stay served by tricyclo.get_retained_tricyclo_name
# regardless of cycle-rank. Plan 151-01 keeps this delegation explicit.
from .tricyclo import (
    _generate_heteroatom_prefix,  # D-12 hetero-prefix reuse
    _get_alkane_name,
    get_retained_tricyclo_name,
)

# v29 Task S — the Java-free legality floor: the emitted descriptor STRING must
# rebuild the input's cage. Shared with ``vonbaeyer_universal`` so the PIN-side
# wrapper and the general-engine cage analyzer apply ONE proof, not two.
from .vonbaeyer_universal import audit_von_baeyer_descriptor

# Type alias: locants are int OR (int, str) per Phase 147 D-01 tuple-aware
# format used for superscripted bridge locants like 8a.
_Locant = Union[int, Tuple[int, str]]


# ============================================================================
# Classification predicate (D-04)
# ============================================================================


def is_higher_polycyclo(mol) -> bool:
    """Return True iff *mol* is a Von Baeyer ≥4-ring system handled here.

    Predicate (per CONTEXT D-04 + RESEARCH "Risk Landmines"):
      - mol is not None
      - ≥4 SSSR rings
      - single connected ring system (perception.rings.get_ring_systems)
      - ≥2 bridgeheads (atoms in ≥2 rings)
      - no spiro atoms (spiro routes via spiro.py — Plan 151-02)
      - no aromatic ring atoms (PAHs route via polycyclics.py)
      - detect_natural_product(mol) is None (steroids/alkaloids stay
        retained — natural-product backbone short-circuit)
      - classify_bridged_system(mol) NOT in ('bicyclo', 'tricyclo')
        (D-04 anti-canary — bicyclo.py / tricyclo.py retain authority)

    Returns False on any failed gate. Defensive: returns False on None
    input rather than raising.
    """
    if mol is None:
        return False
    ri = mol.GetRingInfo()
    if ri.NumRings() < 4:
        return False
    # Lazy imports to avoid circular-import risk during module load.
    from ..perception.rings import get_ring_systems, get_spiro_atoms

    if get_spiro_atoms(mol):
        return False
    rsys = get_ring_systems(mol, include_spiro=False)
    if len(rsys) != 1:
        return False
    # Bridgehead = ring atom appearing in ≥2 of the SSSR rings.
    atom_rings = ri.AtomRings()
    bh_count = 0
    for i in range(mol.GetNumAtoms()):
        if sum(1 for r in atom_rings if i in r) >= 2:
            bh_count += 1
    if bh_count < 2:
        return False
    if any(mol.GetAtomWithIdx(a).GetIsAromatic() for a in rsys[0]):
        return False
    from ..perception.natural_products import detect_natural_product

    if detect_natural_product(mol) is not None:
        return False
    cls = classify_bridged_system(mol)
    if cls in ("bicyclo", "tricyclo"):
        return False
    return True


# ============================================================================
# Naming entry point (D-08 retained-name FIRST, algorithmic SECOND)
# ============================================================================


def name_higher_polycyclo(mol) -> Optional[str]:
    """Generate the IUPAC name for a Von Baeyer ≥4-ring system.

    Algorithm (CONTEXT D-08, AUTONOM §4):
      1. Retained-name dictionary lookup FIRST. tricyclo.get_retained_
         tricyclo_name is consulted on the canonical SMILES regardless
         of ring count — adamantane / twistane stay retained.
      2. If no retained name AND is_higher_polycyclo(mol), delegate to
         VonBaeyerAnalyzer.analyze for descriptor + numbering. Append
         the parent alkane name. If heteroatoms are present, build the
         'a'-prefix via tricyclo._generate_heteroatom_prefix using the
         analyzer's numbering map.
      3. Otherwise return None (defer to bicyclo.py / tricyclo.py /
         polycyclics.py).
    """
    if mol is None:
        return None
    canonical = Chem.MolToSmiles(mol, canonical=True)
    retained = get_retained_tricyclo_name(canonical)
    if retained is not None:
        return retained
    if not is_higher_polycyclo(mol):
        return None
    try:
        ri = mol.GetRingInfo()
        ring_atoms: set[int] = set()
        for r in ri.AtomRings():
            ring_atoms.update(r)
        analyzer = VonBaeyerAnalyzer()
        result = analyzer.analyze(mol, ring_atoms)
        descriptor = result.descriptor_string
        total_atoms = result.total_atoms
        numbering = result.numbering
        # Legality floor (v29 Task S): the descriptor STRING must rebuild this
        # exact cage, or a name like ``tetracyclo[5.1.1.2^3,6]dodecane`` (11
        # bracketed atoms, ``dodecane`` = 12) gets built here. Both OPSIN gates
        # are documented FAIL-OPEN with no Java, so this must not rely on them.
        #
        # v29 Task S2 also wired this proof into ``analyze`` itself, which now
        # publishes the same verdict as ``result.legality``. This call is kept
        # rather than replaced by the flag: it is the guard this module's own
        # tests exercise, it re-proves against the ring-atom set THIS function
        # chose, and it is idempotent. (The rule the atom count comes from is
        # P-23.2.6.1.4 at ``:9651``; this comment cited P-23.2.6.1.1, which is
        # the ring-count-WORD rule at ``:9645``.)
        if not audit_von_baeyer_descriptor(
                mol, ring_atoms, numbering, descriptor):
            logger.info(
                "name_higher_polycyclo: descriptor %s does not rebuild the "
                "cage; refuse", descriptor)
            return None
        parent_name = _get_alkane_name(total_atoms)

        # Heteroatom 'a'-prefix per D-12 (delegate to existing helper that
        # already handles VB locants, multiplier prefixes, and IUPAC
        # priority order O > S > Se > N > P > Si > B).
        heteroatoms = [
            (idx, mol.GetAtomWithIdx(idx).GetSymbol())
            for idx in ring_atoms
            if mol.GetAtomWithIdx(idx).GetSymbol() != "C"
        ]
        if heteroatoms:
            prefix = _generate_heteroatom_prefix(
                mol, heteroatoms, ring_atoms, numbering=numbering
            )
            if prefix is None:
                # A skeletal atom has no Table-1.5 replacement prefix (or no
                # locant). Interpolating None here would literally spell
                # "Nonetetracyclo[...]"; emitting the bare descriptor would name
                # a cage whose stem counts an atom no morpheme spells. Refuse.
                return None
            return f"{prefix}{descriptor}{parent_name}"
        return f"{descriptor}{parent_name}"
    except (ValueError, KeyError, IndexError) as e:
        # Phase 151-04 WR-10: narrowed exception clause per CLAUDE.md /
        # .claude/skills/fix-methodology.md. Real bugs (AttributeError,
        # TypeError) propagate so they surface during testing instead
        # of being silently masked.
        logger.debug(
            "name_higher_polycyclo declined %s: %s",
            Chem.MolToSmiles(mol, canonical=True), e,
        )
        return None


# ============================================================================
# Cascade-step-6 supplier (D-06 + D-21; Phase 147 SC-7 lock)
# ============================================================================


def get_higher_polycyclo_iupac_locants(mol) -> Optional[Dict[int, _Locant]]:
    """Cascade-step-6 IUPAC locant supplier for VB ≥4-ring systems.

    Returns Optional[Dict[atom_idx -> int | (int, str)]] covering ALL
    ring atoms, or None if the predicate fails / coverage is partial.
    Partial maps are NEVER returned — Phase 147 SC-7 / Pitfall 7 gate.

    Naming note (CONTEXT D-06 deviation; RESEARCH Q-01):
      Distinct from polycyclics.py::get_polycyclic_iupac_locants which
      serves cataloged PAHs by ``pah_name`` lookup. This function is the
      Phase 147 cascade-step-6 supplier for non-cataloged ≥4-ring
      bridged systems where retained-name passthrough did not match.
    """
    if not is_higher_polycyclo(mol):
        return None
    try:
        ri = mol.GetRingInfo()
        ring_atoms: set[int] = set()
        for r in ri.AtomRings():
            ring_atoms.update(r)
        analyzer = VonBaeyerAnalyzer()
        result = analyzer.analyze(mol, ring_atoms)
        numbering = result.numbering or {}
        if not numbering:
            return None
        # Pitfall 7 lock: full coverage required.
        if not (set(numbering.keys()) >= ring_atoms):
            return None
        # Same legality floor as ``name_higher_polycyclo``: these locants are
        # only meaningful if the descriptor they belong to actually describes
        # this cage. A numbering handed out against a bridge-dropping descriptor
        # would place every downstream substituent locant on the wrong atom.
        if not audit_von_baeyer_descriptor(
                mol, ring_atoms, numbering, result.descriptor_string):
            logger.info(
                "get_higher_polycyclo_iupac_locants: descriptor %s does not "
                "rebuild the cage; refuse", result.descriptor_string)
            return None
        # Filter to ring atoms only (analyzer may include non-ring entries
        # for some downstream uses; the cascade gate cares about ring set).
        return {idx: numbering[idx] for idx in ring_atoms if idx in numbering}
    except (ValueError, KeyError, IndexError) as e:
        # Phase 151-04 WR-10: narrowed exception clause per CLAUDE.md /
        # .claude/skills/fix-methodology.md.
        logger.debug(
            "get_higher_polycyclo_iupac_locants declined %s: %s",
            Chem.MolToSmiles(mol, canonical=True), e,
        )
        return None


__all__ = [
    "get_higher_polycyclo_iupac_locants",
    "is_higher_polycyclo",
    "name_higher_polycyclo",
]
