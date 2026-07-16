"""P-102.7 disaccharide / oligosaccharide whole-structure assembler (Phase 183, WSC-04, D-02).

A NEW module that reasons over the WHOLE multi-ring sugar structure to emit the
IUPAC 2013 carbohydrate-specific name forms (P-102.7) the generic decomposition
path cannot produce:

* **glycosylglycose** (P-102.7.1.2) when a unit's anomeric carbon bears a *free
  hemiacetal* -OH (the reducing end) -> reducing unit is the glycose parent
  (``-ose``), the other unit(s) are glycosyl substituents (``-osyl``):
  ``alpha-D-glucopyranosyl-(1->4)-D-glucopyranose`` (maltose);
* **glycosyl glycoside** (P-102.7.1.1) when NO unit has a free hemiacetal (both
  anomeric carbons are in the glycosidic linkage, e.g. sucrose): parent chosen by
  P-102.4, cited ``-oside``: ``beta-D-fructofuranosyl alpha-D-glucopyranoside``.

The assembler (D-02) does NOT route through ``decomposition/`` (which emits the
substitutive ``bis(glycosyloxy)...`` P-68 form, structurally wrong for sugars). It
detects the units + the inter-unit glycosidic bond directly with a RING-RESTRICTED
SMARTS over the whole structure (Pitfall 4 — the aglycone carbon must be in a SECOND
recognized sugar ring, so a single monoglycoside (Phase 176) is NOT pulled in),
splits each unit with the proven ``conjugate_controller._extract_capped_sugar``
primitive (FragmentOnBonds + restore the anomeric -OH; a ``rules/`` sibling, not a
``decomposition/`` import), names each per its OWN ring via the
``lookup_sugar`` -> ``recognize_sugar_skeleton`` -> ``name_monosaccharide_systematic``
cascade (D-05/D-09 — never inherit an anomer across units), derives the directional
``(c->c')`` ASCII linkage locants by ring-walking from each unit's anomeric carbon
(D-07), enforces the no-silent-drop completeness invariant over the ORIGINAL mol's
heavy atoms reconciling the single shared bridging O (D-12, Pitfall 7), and gates the
assembled name on an OPSIN round-trip fallback (D-13, fail-OPEN when Java/OPSIN
absent) before shipping.

Root-cause-only (the contributor guide): no postprocessor, no regex / string surgery on any
derived base, no per-molecule hardcode. Every gate failure returns ``None`` so the
molecule cascade-continues to the existing pipeline (fail-closed, D-11). The input
mol is never mutated (the unit split copies via the reused capper).

Blue Book: P-102.7 (@54101), P-102.7.1.1 glycosyl glycoside (@54109),
P-102.7.1.2 glycosylglycose (@54119), P-102.4 parent choice (@52888),
P-102.6.1 glycosyl groups -ose -> -osyl (@53894).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

from rdkit import Chem

logger = logging.getLogger(__name__)

# Ring-restricted glycosidic-bond SMARTS (RESEARCH §3, Pitfall 4): an acetal anomeric
# ring carbon -- ring-O, then the bridging glycosidic O (NOT in a ring) -- a SECOND
# ring carbon (the aglycone). ``[#6;R]`` keeps the aglycone in a ring so a monoglycoside
# (whose aglycone need not be a ring) is not detected; we further require the second
# ring to recognize as a sugar (the both-rings-are-sugars gate).
_GLYCOSIDIC_SMARTS = "[CX4;R]([OX2;R])[OX2;!R][#6;R]"

# ASCII descriptors + ASCII arrow throughout (Pitfall 6 — pin_strict_eval.normalize
# does not transliterate Greek or the Unicode arrow).
_ARROW = "->"


# --------------------------------------------------------------------------- #
# Per-unit recognition (D-05 / D-09 cascade)
# --------------------------------------------------------------------------- #
def _recognize_unit(mol, detach_carbon: int, bridging_o_idx: int) -> Optional[Tuple[str, str, str]]:
    """Recognize a single sugar unit from its OWN ring (never inherited, D-09).

    Isolates the unit by cleaving the bond from ``detach_carbon`` (the ring carbon
    bonded to the inter-unit bridging O) to ``bridging_o_idx`` with the proven
    ``_extract_capped_sugar`` primitive (FragmentOnBonds + restore the -OH on that
    carbon), then runs the catalog-first cascade:
    ``lookup_sugar`` -> ``recognize_sugar_skeleton`` -> ``name_monosaccharide_systematic``.

    For the glycosyl (non-reducing) unit ``detach_carbon`` is its anomeric carbon;
    for the reducing (glycose) unit it is the attachment carbon (e.g. maltose C4) --
    capping there isolates the ring and restores its free -OH, while the unit's OWN
    anomeric -OH (already free) stays intact, so recognition reads the reducing
    anomer from that unit's own ring.

    Returns ``(anomer, config, base)`` (the systematic path is folded into a
    base-only tuple) or ``None`` (fail-closed) for an unrecognized unit.
    """
    from orthonym.rules.conjugate_controller import _extract_capped_sugar
    from orthonym.data.sugar_names import (
        lookup_sugar,
        name_monosaccharide_systematic,
        recognize_sugar_skeleton,
    )

    canon = _extract_capped_sugar(mol, detach_carbon, bridging_o_idx)
    if not canon:
        return None
    sugar_mol = Chem.MolFromSmiles(canon)
    if sugar_mol is None:
        return None

    # Catalog FIRST (D-05 — keeps cataloged sugars byte-identical).
    tup = lookup_sugar(Chem.MolToSmiles(sugar_mol))
    if tup is not None:
        return tup
    tup = recognize_sugar_skeleton(sugar_mol)
    if tup is not None:
        return tup
    # Decorated (deoxy/amino/uronic) unit via the Plan-01 systematic engine: it
    # returns a full name string; carry it as a base-only tuple (no separate
    # anomer/config -- the systematic name already embeds them).
    systematic = name_monosaccharide_systematic(sugar_mol)
    if systematic:
        return ("", "", systematic)
    return None


# --------------------------------------------------------------------------- #
# Ring numbering -> IUPAC carbohydrate locants (D-07, Open Q2)
# --------------------------------------------------------------------------- #
def _ring_carbon_locants(mol, ring: Tuple[int, ...], ring_oxygen: int,
                         anomeric_idx: int) -> Dict[int, int]:
    """Number a sugar ring's carbons C1.. by IUPAC carbohydrate convention.

    Anomeric carbon = C1; walk the ring away from the ring-O (C2, C3, ...); the
    exocyclic ``CH2OH``/``COOH`` carbon attached to the last ring carbon = the next
    locant (C6 on a hexopyranose). Returns ``{atom_idx: locant}`` for the ring
    carbons plus the exocyclic terminal carbon, used to read the attachment locant.
    """
    ringset = set(ring)
    adjacency = {
        idx: [
            n.GetIdx()
            for n in mol.GetAtomWithIdx(idx).GetNeighbors()
            if n.GetIdx() in ringset
        ]
        for idx in ringset
    }
    non_oxygen = [x for x in adjacency[anomeric_idx] if x != ring_oxygen]
    if len(non_oxygen) != 1:
        return {}
    order = [anomeric_idx]
    prev, cur = anomeric_idx, non_oxygen[0]
    while cur != anomeric_idx:
        order.append(cur)
        nxt = [x for x in adjacency[cur] if x != prev]
        if not nxt:
            break
        prev, cur = cur, nxt[0]

    locants: Dict[int, int] = {}
    n = 1
    last_ring_carbon = None
    for idx in order:
        if idx == ring_oxygen:
            continue
        locants[idx] = n
        last_ring_carbon = idx
        n += 1
    # The exocyclic terminal carbon (CH2OH / COOH) on the last ring carbon = next locant.
    if last_ring_carbon is not None:
        for nbr in mol.GetAtomWithIdx(last_ring_carbon).GetNeighbors():
            if nbr.GetIdx() in ringset or nbr.GetAtomicNum() != 6:
                continue
            # Must be a carbon bearing an O (the CH2OH/COOH tail), not a bare methyl.
            if any(nn.GetAtomicNum() == 8 for nn in nbr.GetNeighbors()):
                locants[nbr.GetIdx()] = n
                break
    return locants


def _anomeric_locant(mol, ring: Tuple[int, ...], ring_oxygen: int,
                     anomeric_idx: int) -> int:
    """The glycosyl unit's anomeric locant ``c``: 1 for an aldose, 2 for a ketose.

    A KETOSE anomeric carbon (e.g. fructose C2) bears an EXOCYCLIC carbon — the C1
    CH2OH — in addition to its ring-carbon neighbour; an aldose anomeric carbon
    (C1) has no exocyclic carbon (only -OH/-OR / the glycosidic O). Derived
    structurally by the presence of an exocyclic carbon (review W6B fix: the prior
    ring-carbon count returned 1 for a keto-furanosyl/-pyranosyl donor, which bears
    only ONE ring carbon)."""
    ringset = set(ring)
    exo_carbon_nbrs = [
        n.GetIdx()
        for n in mol.GetAtomWithIdx(anomeric_idx).GetNeighbors()
        if n.GetIdx() not in ringset and n.GetAtomicNum() == 6
    ]
    return 2 if exo_carbon_nbrs else 1


# --------------------------------------------------------------------------- #
# Unit / glycosidic-bond classification (whole structure, D-02 / D-06)
# --------------------------------------------------------------------------- #
def _ring_of(mol, atom_idx: int) -> Optional[Tuple[int, ...]]:
    """The single SSSR ring containing ``atom_idx`` (None if 0 or >1)."""
    rings = [r for r in mol.GetRingInfo().AtomRings() if atom_idx in r]
    if len(rings) != 1:
        return None
    return rings[0]


def _ring_oxygen(mol, ring: Tuple[int, ...]) -> Optional[int]:
    ros = [i for i in ring if mol.GetAtomWithIdx(i).GetSymbol() == "O"]
    return ros[0] if len(ros) == 1 else None


def _unit_atoms(mol, ring: Tuple[int, ...], bridging_os: Set[int]) -> Set[int]:
    """Heavy atoms belonging to a unit: its ring + exocyclic substituents.

    BFS out from each ring atom, NOT crossing a bridging O (the shared inter-unit
    linker) and NOT entering another ring. Collects the ring atoms and their
    pendant substituent atoms (hydroxyls, the CH2OH tail, etc.) so the completeness
    invariant (D-12) can reconcile every heavy atom.
    """
    ringset = set(ring)
    claimed: Set[int] = set(ringset)
    stack: List[int] = []
    for idx in ringset:
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            n = nbr.GetIdx()
            if n in ringset or n in bridging_os:
                continue
            stack.append(n)
    while stack:
        idx = stack.pop()
        if idx in claimed or idx in bridging_os:
            continue
        atom = mol.GetAtomWithIdx(idx)
        # Do not absorb a second ring (that is another unit).
        if atom.IsInRing():
            continue
        claimed.add(idx)
        for nbr in atom.GetNeighbors():
            n = nbr.GetIdx()
            if n in claimed or n in bridging_os:
                continue
            stack.append(n)
    return claimed


def _classify_units(mol) -> Optional[Dict]:
    """Detect a LINEAR chain of >=2 recognized sugar units + decide the name shape.

    Implements the perception half of the assembler (D-02/D-05/D-06/D-09):

    1. Detect inter-unit glycosidic bonds with the RING-RESTRICTED SMARTS over the
       WHOLE structure (Pitfall 4); require BOTH ring carbons to live in a recognized
       sugar ring (the both-rings-are-sugars gate -- a monoglycoside is excluded).
    2. Recognize each unit per its OWN ring (D-09, never inherited).
    3. Decide the shape (D-06): a unit whose anomeric carbon carries a FREE hemiacetal
       -OH (exocyclic O with an H, degree 1, NOT the bridging O) is the reducing end ->
       ``glycosylglycose``; none free (all anomeric C linked) -> ``glycoside``;
       neither (O-substituted anomeric, no free/linked) -> None (fail-closed).

    Returns an ordered dict describing the chain (units, the bridging O set, the
    glycosidic links with their atoms, the shape, and the consumed heavy-atom set for
    Task-2's completeness check), or ``None`` for any out-of-scope topology
    (single ring / branched / C-glycoside / polymeric / unrecognized unit).
    """
    if mol is None:
        return None
    ri = mol.GetRingInfo()
    if ri.NumRings() < 2:
        return None  # a single ring is a monosaccharide / monoglycoside, not a disaccharide

    smarts = Chem.MolFromSmarts(_GLYCOSIDIC_SMARTS)
    if smarts is None:
        return None
    matches = mol.GetSubstructMatches(smarts)
    if not matches:
        return None

    # Collect candidate inter-unit links: (anomeric_c, ring_o, bridging_o, aglycone_c).
    # The SMARTS is symmetric for a ketose-anomeric linkage (sucrose finds both
    # directions); dedup by the bridging-O atom so each physical bond is one link.
    links_by_bridge: Dict[int, Tuple[int, int, int, int]] = {}
    for anomeric_c, ring_o, bridging_o, aglycone_c in matches:
        # Both carbons must sit in a recognized sugar ring (the both-rings gate).
        ring_a = _ring_of(mol, anomeric_c)
        ring_b = _ring_of(mol, aglycone_c)
        if ring_a is None or ring_b is None or set(ring_a) == set(ring_b):
            continue
        links_by_bridge.setdefault(bridging_o, (anomeric_c, ring_o, bridging_o, aglycone_c))

    if not links_by_bridge:
        return None
    # Disaccharide / linear trisaccharide only: 1 or 2 inter-unit bonds.
    if len(links_by_bridge) > 2:
        return None

    bridging_os: Set[int] = set(links_by_bridge.keys())

    # Gather the distinct sugar rings touched by the links.
    ring_set_by_key: Dict[frozenset, Tuple[int, ...]] = {}
    for anomeric_c, ring_o, bridging_o, aglycone_c in links_by_bridge.values():
        for c in (anomeric_c, aglycone_c):
            r = _ring_of(mol, c)
            if r is not None:
                ring_set_by_key[frozenset(r)] = r
    rings = list(ring_set_by_key.values())
    if len(rings) < 2:
        return None

    # Each ring must have exactly one ring oxygen and recognize as a sugar.
    unit_records: List[Dict] = []
    for ring in rings:
        ro = _ring_oxygen(mol, ring)
        if ro is None:
            return None
        # The ring's anomeric carbon: ring C bonded to ring-O AND an exocyclic O.
        ringset = set(ring)
        anomeric_idx = None
        anomeric_exo_o = None
        for idx in ring:
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetSymbol() != "C":
                continue
            if not any(n.GetIdx() == ro for n in atom.GetNeighbors()):
                continue
            exo_os = [
                n.GetIdx()
                for n in atom.GetNeighbors()
                if n.GetIdx() not in ringset and n.GetSymbol() == "O"
            ]
            if len(exo_os) == 1:
                anomeric_idx = idx
                anomeric_exo_o = exo_os[0]
                break
        if anomeric_idx is None:
            return None

        # Reducing iff the anomeric exocyclic O is a FREE hemiacetal -OH
        # (degree-1, an H, NOT a bridging O). Otherwise it is linked.
        exo_atom = mol.GetAtomWithIdx(anomeric_exo_o)
        is_bridging = anomeric_exo_o in bridging_os
        is_free_oh = (
            (not is_bridging)
            and exo_atom.GetDegree() == 1
            and exo_atom.GetTotalNumHs() >= 1
        )
        if (not is_bridging) and (not is_free_oh):
            # O-substituted anomeric (acylated/methylated): neither free nor the
            # inter-unit bond -> out of scope (D-06, Open Q1 strict).
            return None

        # Find the ring carbon that bears an inter-unit bridging O (the linker we cap
        # to isolate this unit). For the glycosyl unit this is its anomeric carbon;
        # for the reducing unit it is the attachment carbon (a non-anomeric ring C).
        detach_carbon = None
        detach_bridging_o = None
        for idx in ring:
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetSymbol() != "C":
                continue
            for nbr in atom.GetNeighbors():
                if nbr.GetIdx() in bridging_os:
                    detach_carbon = idx
                    detach_bridging_o = nbr.GetIdx()
                    break
            if detach_carbon is not None:
                break
        if detach_carbon is None:
            return None

        # Recognize the unit per its OWN ring (D-09, never inherited).
        tup = _recognize_unit(mol, detach_carbon, detach_bridging_o)
        if tup is None:
            return None

        unit_records.append({
            "detach_carbon": detach_carbon,
            "ring": ring,
            "ring_oxygen": ro,
            "anomeric_idx": anomeric_idx,
            "anomeric_exo_o": anomeric_exo_o,
            "is_reducing": is_free_oh,
            "tuple": tup,
        })

    n_reducing = sum(1 for u in unit_records if u["is_reducing"])
    if n_reducing > 1:
        return None  # >1 free anomeric -OH -> not a single linear reducing chain
    shape = "glycosylglycose" if n_reducing == 1 else "glycoside"

    # Heavy-atom accounting for the completeness invariant (D-12): each unit's atoms
    # + the shared bridging O(s). Track over the ORIGINAL mol's atom set.
    consumed: Set[int] = set(bridging_os)
    for u in unit_records:
        u_atoms = _unit_atoms(mol, u["ring"], bridging_os)
        u["atoms"] = u_atoms
        consumed |= u_atoms

    return {
        "units": unit_records,
        "links": list(links_by_bridge.values()),
        "bridging_os": bridging_os,
        "shape": shape,
        "consumed": consumed,
    }


# --------------------------------------------------------------------------- #
# RT-fallback gate (D-13, mirror natural_products._alpha_beta_rt_ok)
# --------------------------------------------------------------------------- #
def _sugar_name_rt_ok(mol, name: str, relax_atoms: Optional[Set[int]] = None) -> bool:
    """Return True iff ``name`` OPSIN-round-trips to ``mol`` (fail-OPEN, D-13).

    Fail-OPEN: when OPSIN/Java is unavailable the round-trip cannot be checked, so we
    ship the name (the SUB-03 validity-gate posture). When OPSIN IS available, a name
    that does not round-trip is rejected so the caller falls back to ``None`` ->
    existing pipeline (zero RT regression).

    ``relax_atoms`` (the reducing-end anomeric carbon, when its descriptor is omitted
    per D-09) have their chirality cleared on a copy of ``mol`` before the InChI
    comparison, so an honestly-unspecified reducing anomer still RT-validates the rest
    of the structure (the name asserts "anomer unspecified", which must compare against
    the structure with that one centre relaxed -- never invent an alpha/beta).
    """
    try:
        from ..validation.opsin_roundtrip import (
            opsin_roundtrip_check, _find_opsin_jar, _java_available,
        )
        if _find_opsin_jar() is None or not _java_available():
            return True  # fail-OPEN
        target = mol
        if relax_atoms:
            rw = Chem.RWMol(mol)
            for idx in relax_atoms:
                rw.GetAtomWithIdx(idx).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            target = rw.GetMol()
            Chem.SanitizeMol(target)
        smi = Chem.MolToSmiles(target)
        return bool(opsin_roundtrip_check(smi, name).get("passed"))
    except Exception:
        return True


def _count_sugar_rings(mol) -> int:
    """Count 5-/6-membered rings with exactly one ring oxygen (a pyranose/furanose
    ring proxy). A disaccharide has 2; a trisaccharide+ has 3+. Used to fail closed
    on >=3-unit saccharides, which the binary assembler mis-parses (v23 Phase 5)."""
    n = 0
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) in (5, 6) and sum(
            1 for i in ring if mol.GetAtomWithIdx(i).GetSymbol() == "O"
        ) == 1:
            n += 1
    return n


# --------------------------------------------------------------------------- #
# Public entry point (D-02)
# --------------------------------------------------------------------------- #
def _name_disaccharide_binary(mol) -> Optional[str]:
    """Name a BINARY disaccharide (2 units, 1 link) by the P-102.7 form, else
    ``None``.  This is the proven binary assembler; the public
    :func:`name_disaccharide` wraps it and falls back to the general
    :func:`name_linear_oligosaccharide` for 3+ unit chains and the 1->6 links the
    ring-restricted SMARTS here misses.

    Composes the glycosylglycose / glycosyl-glycoside form with ASCII ``(c->c')``
    linkage locants and per-unit anomers (none invented for an unspecified reducing
    end, D-09), enforces the completeness invariant (D-12), and gates on the RT
    fallback (D-13) before shipping. Returns ``None`` for any out-of-scope topology
    (branched / non-linear / C-glycoside / polymeric / unrecognized unit / dropped
    atom / RT-fail) so the molecule cascade-continues.
    """
    # v23 Phase 5: trisaccharide+ are out of scope (deferred to v24). The binary
    # assembler only validates the disaccharide (gold + RT gate), and its
    # unit-parser can mis-read a 3rd sugar ring as a glycosyloxy substituent rather
    # than a separate unit — bypassing the len(units)!=2 guard below and emitting a
    # MALFORMED hybrid. Fail closed here on >=3 sugar rings so this assembler never
    # ships a trisaccharide name. (Disaccharides have exactly 2 sugar rings, so this
    # is a no-op for them — sucrose/maltose/etc. are unaffected.) NOTE: raffinose's
    # OWN production output is already 'unknown' — its malformed name comes from a
    # DIFFERENT path (the heterocycle handler's sugar-substituent naming), which the
    # SUB-03 validity gate suppresses; that path over-claiming pure oligosaccharides
    # is a separate pre-existing issue (shared substituent layer, v24 carbohydrate
    # scope), not addressed by hardening THIS assembler.
    if _count_sugar_rings(mol) >= 3:
        return None
    info = _classify_units(mol)
    if info is None:
        return None

    units: List[Dict] = info["units"]
    shape: str = info["shape"]

    # --- Completeness invariant (D-12, Pitfall 7): every ORIGINAL heavy atom must be
    # consumed by exactly one unit or be a shared bridging O. ---
    all_heavy = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1}
    if info["consumed"] != all_heavy:
        return None

    # Only the binary disaccharide is fully supported here (the linear walk below
    # composes n>=2 generically, but the gold + RT gate binds the disaccharide).
    if len(units) != 2:
        return None
    if len(info["links"]) != 1:
        return None

    # Identify the glycosyl (substituent) unit and the parent unit.
    if shape == "glycosylglycose":
        # Reducing end is FORCED parent (D-06); the other unit is the glycosyl.
        parent = next(u for u in units if u["is_reducing"])
        glycosyl = next(u for u in units if not u["is_reducing"])
    else:
        # No free hemiacetal -> glycoside; parent by P-102.4 (D-08).
        parent, glycosyl = _choose_glycoside_parent(mol, units)
        if parent is None:
            return None

    # --- Derive the (c->c') linkage locants (D-07). c = glycosyl anomeric locant
    # (1 aldose / 2 ketose); c' = the parent carbon bearing the glycosidic O. ---
    link = info["links"][0]
    anomeric_c, ring_o, bridging_o, aglycone_c = link
    # Orient the link so anomeric_c is on the GLYCOSYL unit and aglycone_c on the parent.
    if anomeric_c in set(glycosyl["ring"]):
        gly_anomeric, par_attach = anomeric_c, aglycone_c
    else:
        gly_anomeric, par_attach = aglycone_c, anomeric_c

    c = _anomeric_locant(mol, glycosyl["ring"], glycosyl["ring_oxygen"], gly_anomeric)

    # --- Assemble per shape ---
    gly_anomer, gly_config, gly_base = glycosyl["tuple"]
    par_anomer, par_config, par_base = parent["tuple"]

    glycosyl_str = _glycosyl_term(gly_anomer, gly_config, gly_base)
    if glycosyl_str is None:
        return None

    relax: Set[int] = set()

    if shape == "glycosylglycose":
        c_prime = _attachment_locant(mol, parent, par_attach)
        if c_prime is None:
            return None
        # P-102.7.1.2: the reducing-end anomer MUST be cited when it is DEFINED
        # (beta-maltose -> ...-(1->4)-beta-D-glucopyranose).  W6B-T10 root-cause
        # fix: previously the anomer was always dropped (D-09) even for a defined
        # reducing centre, which is only correct for an UNSPECIFIED (mutarotating)
        # anomer.  When the reducing anomeric atom has no defined chirality, keep
        # the omit-and-relax path (the name does not assert the anomer).
        parent_anomeric = parent["anomeric_idx"]
        anomeric_defined = (
            mol.GetAtomWithIdx(parent_anomeric).GetChiralTag()
            != Chem.ChiralType.CHI_UNSPECIFIED
        )
        if anomeric_defined and par_anomer and par_config:
            parent_str = f"{par_anomer}-{par_config}-{par_base}"
        else:
            parent_str = f"{par_config}-{par_base}" if par_config else par_base
            relax.add(parent_anomeric)
        name = f"{glycosyl_str}-({c}{_ARROW}{c_prime})-{parent_str}"
    else:
        parent_str = _glycoside_head(par_anomer, par_config, par_base)
        if parent_str is None:
            return None
        name = f"{glycosyl_str} {parent_str}"

    # --- RT-fallback gate (D-13) ---
    if not _sugar_name_rt_ok(mol, name, relax_atoms=relax or None):
        return None
    return name


def name_disaccharide(mol) -> Optional[str]:
    """Public entry (D-02): name a linear di/oligo-saccharide by the P-102.7 form.

    Tries the proven binary disaccharide assembler first (byte-identical to the
    prior behaviour); if it declines, falls back to the general linear
    oligosaccharide namer (W6B-T11) which handles 3+ unit reducing chains and the
    1->6 links the binary ring-restricted SMARTS misses.  ``None`` (fail-closed)
    for any out-of-scope topology.
    """
    result = _name_disaccharide_binary(mol)
    if result is not None:
        return result
    return name_linear_oligosaccharide(mol)


def _extract_unit_capped(mol, ringset: Set[int],
                         bond_pairs: List[Tuple[int, int]]) -> Optional[str]:
    """Isolate ONE sugar unit by capping ALL its inter-unit bonds (each a
    ``(unit_carbon, bridging_o)`` pair) and restoring the -OH on every unit-side
    carbon.  Returns the canonical SMILES of the isolated unit, or ``None``.
    Multi-bond generalization of :func:`_extract_capped_sugar` (a middle unit of a
    chain carries two bridging bonds)."""
    if not bond_pairs:
        return None
    bond_ids = []
    for c, o in bond_pairs:
        b = mol.GetBondBetweenAtoms(c, o)
        if b is None:
            return None
        bond_ids.append(b.GetIdx())
    try:
        frag = Chem.FragmentOnBonds(mol, bond_ids, addDummies=True)
        mapping: List = []
        frags = Chem.GetMolFrags(
            frag, asMols=True, sanitizeFrags=False, fragsMolAtomMapping=mapping
        )
    except Exception:
        return None
    sample = next(iter(ringset))
    unit = None
    for fm, mp in zip(frags, mapping):
        if sample in mp:
            unit = fm
            break
    if unit is None:
        return None
    rw = Chem.RWMol(unit)
    for at in rw.GetAtoms():
        if at.GetAtomicNum() == 0:  # a dummy from a cleaved bridging bond -> -OH
            at.SetAtomicNum(8)
            at.SetNoImplicit(False)
            at.SetNumExplicitHs(1)
    try:
        Chem.SanitizeMol(rw)
        clean = Chem.RemoveHs(rw)
        return Chem.CanonSmiles(Chem.MolToSmiles(clean))
    except Exception:
        return None


def _oligo_topology(mol) -> Optional[Dict]:
    """Structural (NO-OPSIN) detection of a LINEAR reducing oligosaccharide chain.

    Returns ``{units, links, order, parent_ui, bridging_os}`` for a chain of >=2
    glycosidically-linked sugar rings with a unique free-hemiacetal reducing end
    and no branching, else ``None``.  Shared by :func:`name_linear_oligosaccharide`
    (which adds recognition + assembly + RT) and :func:`_has_oligo_chain` (the cheap
    dispatch predicate).  Recognition and RT are NOT done here."""
    if mol is None:
        return None
    from collections import Counter
    ri = mol.GetRingInfo()

    # 1. Sugar rings (5/6-membered, exactly one ring O, rest carbons) + anomeric C.
    units: List[Dict] = []
    for ring in ri.AtomRings():
        if len(ring) not in (5, 6):
            continue
        if sum(1 for i in ring if mol.GetAtomWithIdx(i).GetSymbol() == "O") != 1:
            continue
        if any(mol.GetAtomWithIdx(i).GetSymbol() not in ("C", "O") for i in ring):
            continue
        ro = _ring_oxygen(mol, ring)
        if ro is None:
            return None
        ringset = set(ring)
        anomeric_idx = anomeric_exo_o = None
        for idx in ring:
            a = mol.GetAtomWithIdx(idx)
            if a.GetSymbol() != "C" or not any(n.GetIdx() == ro for n in a.GetNeighbors()):
                continue
            exo_os = [n.GetIdx() for n in a.GetNeighbors()
                      if n.GetIdx() not in ringset and n.GetSymbol() == "O"]
            if len(exo_os) == 1:
                anomeric_idx, anomeric_exo_o = idx, exo_os[0]
                break
        if anomeric_idx is None:
            return None
        units.append({"ring": ring, "ringset": ringset, "ring_oxygen": ro,
                      "anomeric_idx": anomeric_idx, "anomeric_exo_o": anomeric_exo_o})
    if len(units) < 2:
        return None

    # Map every unit carbon (ring + exocyclic) -> unit index (for acceptor lookup).
    carbon_to_unit: Dict[int, int] = {}
    for ui, u in enumerate(units):
        for idx in u["ring"]:
            if mol.GetAtomWithIdx(idx).GetSymbol() == "C":
                carbon_to_unit[idx] = ui
            for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
                if nbr.GetIdx() not in u["ringset"] and nbr.GetAtomicNum() == 6:
                    carbon_to_unit.setdefault(nbr.GetIdx(), ui)

    # 2. Glycosidic links: each unit's anomeric exocyclic O, if a bridging ether O
    #    (no H, 2 carbons) whose OTHER carbon maps to a DIFFERENT sugar unit.
    links: List[Tuple[int, int, int, int, int]] = []  # donor,acc,anom_c,bridge_o,acc_c
    bridging_os: Set[int] = set()
    for ui, u in enumerate(units):
        o_idx = u["anomeric_exo_o"]
        o_atom = mol.GetAtomWithIdx(o_idx)
        if o_atom.GetTotalNumHs() > 0:
            continue  # free -OH (reducing-end candidate), not a bridge
        c_nbrs = [n.GetIdx() for n in o_atom.GetNeighbors() if n.GetAtomicNum() == 6]
        if len(c_nbrs) != 2:
            continue
        acceptor_c = next((c for c in c_nbrs if c != u["anomeric_idx"]), None)
        if acceptor_c is None:
            continue
        acc_ui = carbon_to_unit.get(acceptor_c)
        if acc_ui is None or acc_ui == ui:
            continue  # acceptor must belong to another sugar unit
        links.append((ui, acc_ui, u["anomeric_idx"], o_idx, acceptor_c))
        bridging_os.add(o_idx)
    if not links:
        return None

    # 3. Reducing parent = the unique non-donor unit whose anomeric is a free -OH.
    donors = {l[0] for l in links}
    non_donors = [ui for ui in range(len(units)) if ui not in donors]
    if len(non_donors) != 1:
        return None
    parent_ui = non_donors[0]
    if mol.GetAtomWithIdx(units[parent_ui]["anomeric_exo_o"]).GetTotalNumHs() == 0:
        return None  # parent anomeric linked -> non-reducing glycoside (deferred)
    if any(v > 1 for v in Counter(l[1] for l in links).values()):
        return None  # a unit accepts >1 glycosyl -> BRANCHED

    # 4. Order the linear chain: walk parent <- donor <- ... (must cover every unit).
    donor_of: Dict[int, Tuple] = {}
    for l in links:
        if l[1] in donor_of:
            return None  # >1 donor onto one acceptor -> branched
        donor_of[l[1]] = l
    order = [parent_ui]
    seen = {parent_ui}
    cur = parent_ui
    while cur in donor_of:
        donor_ui = donor_of[cur][0]
        if donor_ui in seen:
            return None
        order.append(donor_ui)
        seen.add(donor_ui)
        cur = donor_ui
    if len(order) != len(units):
        return None  # disconnected / not a single linear chain

    return {"units": units, "links": links, "order": order,
            "parent_ui": parent_ui, "bridging_os": bridging_os}


def _has_oligo_chain(mol) -> bool:
    """Cheap NO-OPSIN dispatch precondition for :func:`name_linear_oligosaccharide`."""
    return _oligo_topology(mol) is not None


def name_linear_oligosaccharide(mol) -> Optional[str]:
    """Name a LINEAR reducing oligosaccharide (>=2 units) by P-102.7.2.2:
    ``glycosyl-(1->c')-[glycosyl-(1->c')-]n-glycose`` (maltotriose ->
    ``alpha-D-glucopyranosyl-(1->4)-alpha-D-glucopyranosyl-(1->4)-D-glucopyranose``;
    isomaltose 1->6).  Fail-closed (``None``) on: a BRANCHED chain (a unit accepts
    >1 glycosyl), a NON-reducing chain (no free-hemiacetal parent -> P-102.7.2.1
    glycoside, deferred), >1 reducing unit, an unrecognized unit, a dropped atom,
    or an RT-fail.  Each unit is recognized from its OWN isolated ring (D-09); the
    whole name is OPSIN-RT gated (D-13)."""
    topo = _oligo_topology(mol)
    if topo is None:
        return None
    units = topo["units"]
    links = topo["links"]
    order = topo["order"]
    parent_ui = topo["parent_ui"]
    bridging_os = topo["bridging_os"]

    # 5. Isolate + recognize each unit (D-09), collecting its bridging bonds.
    unit_bridge_bonds: Dict[int, List[Tuple[int, int]]] = {ui: [] for ui in range(len(units))}
    for donor_ui, acc_ui, anom_c, o_idx, acc_c in links:
        unit_bridge_bonds[donor_ui].append((anom_c, o_idx))
        unit_bridge_bonds[acc_ui].append((acc_c, o_idx))
    from orthonym.data.sugar_names import (
        lookup_sugar, recognize_sugar_skeleton, name_monosaccharide_systematic,
    )
    tuples: Dict[int, Tuple[str, str, str]] = {}
    for ui, u in enumerate(units):
        canon = _extract_unit_capped(mol, u["ringset"], unit_bridge_bonds[ui])
        if canon is None:
            return None
        sm = Chem.MolFromSmiles(canon)
        if sm is None:
            return None
        tup = lookup_sugar(Chem.MolToSmiles(sm)) or recognize_sugar_skeleton(sm)
        if tup is None:
            systematic = name_monosaccharide_systematic(sm)
            tup = ("", "", systematic) if systematic else None
        if tup is None:
            return None
        tuples[ui] = tup

    # 6. Assemble in name order (terminal donor ... -> reducing parent).
    relax: Set[int] = set()
    parts: List[str] = []
    for ui in reversed(order):
        if ui == parent_ui:
            pa, pc, pb = tuples[ui]
            anom_defined = (mol.GetAtomWithIdx(units[ui]["anomeric_idx"]).GetChiralTag()
                            != Chem.ChiralType.CHI_UNSPECIFIED)
            if anom_defined and pa and pc:
                parent_str = f"{pa}-{pc}-{pb}"
            else:
                parent_str = f"{pc}-{pb}" if pc else pb
                relax.add(units[ui]["anomeric_idx"])
            parts.append(parent_str)
        else:
            ga, gc, gb = tuples[ui]
            gterm = _glycosyl_term(ga, gc, gb)
            if gterm is None:
                return None
            link = next(x for x in links if x[0] == ui)
            _d, acc_ui, anom_c, _o, acc_c = link
            acc_u = units[acc_ui]
            acc_locs = _ring_carbon_locants(
                mol, acc_u["ring"], acc_u["ring_oxygen"], acc_u["anomeric_idx"]
            )
            c_prime = acc_locs.get(acc_c)
            if c_prime is None:
                return None
            c = _anomeric_locant(mol, units[ui]["ring"], units[ui]["ring_oxygen"], anom_c)
            parts.append(f"{gterm}-({c}{_ARROW}{c_prime})")
    name = "-".join(parts)

    # 7. Completeness invariant (D-12) + RT gate (D-13).
    all_heavy = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1}
    consumed: Set[int] = set(bridging_os)
    for u in units:
        consumed |= _unit_atoms(mol, u["ring"], bridging_os)
    if consumed != all_heavy:
        return None
    if not _sugar_name_rt_ok(mol, name, relax_atoms=relax or None):
        return None
    return name


def _glycosyl_term(anomer: str, config: str, base: str) -> Optional[str]:
    """Build a glycosyl substituent term ``{anomer}-{config}-{base}yl`` (-ose -> -osyl).

    Uses the structural ``-ose`` -> ``-osyl`` slice; never string-surgery on a derived
    base beyond the canonical suffix transform.
    """
    if not base:
        return None
    if base.endswith("ose"):
        stem = base[:-1] + "yl"  # -ose -> -osyl
    else:
        stem = base + "yl"
    if anomer and config:
        return f"{anomer}-{config}-{stem}"
    return stem


def _glycoside_head(anomer: str, config: str, base: str) -> Optional[str]:
    """Build the parent glycoside head ``...oside`` (uronic via the explicit map)."""
    from orthonym.data.sugar_names import (
        sugar_to_glycoside_class_name,
        uronic_glycoside_head,
    )
    if not base:
        return None
    if "urono" in base:
        return uronic_glycoside_head(anomer, config, base)
    return sugar_to_glycoside_class_name(anomer, config, base)


def _attachment_locant(mol, parent: Dict, attach_carbon: int) -> Optional[int]:
    """The parent carbon's IUPAC locant (c') derived by ring-walking from C1."""
    locants = _ring_carbon_locants(
        mol, parent["ring"], parent["ring_oxygen"], parent["anomeric_idx"]
    )
    return locants.get(attach_carbon)


def _unit_carbon_count(mol, unit: Dict) -> int:
    """Total carbons of a unit (ring + claimed exocyclic substituent atoms, D-08 step b)."""
    return sum(
        1 for i in unit["atoms"] if mol.GetAtomWithIdx(i).GetAtomicNum() == 6
    )


def _choose_glycoside_parent(mol, units: List[Dict]) -> Tuple[Optional[Dict], Optional[Dict]]:
    """P-102.4 parent choice for the no-free-hemiacetal glycoside case (D-08).

    Cascade (linear di only here): (b) greatest carbon count -> parent; (c) on a tie,
    alphabetical by config-prefix / trivial stem -- the EARLIER stem is cited first as
    the glycosyl substituent, so the LATER stem is the parent (``-oside``)
    (``fructo`` < ``gluco`` -> glucose is the parent of sucrose); then D before L,
    then alpha before beta. Returns ``(parent, glycosyl)``.
    """
    a, b = units[0], units[1]
    ca, cb = _unit_carbon_count(mol, a), _unit_carbon_count(mol, b)
    if ca != cb:
        parent, glycosyl = (a, b) if ca > cb else (b, a)
        return parent, glycosyl

    # Carbon-count tie -> alphabetical stem (the later stem is the parent -oside).
    key_a = (a["tuple"][2], a["tuple"][1], a["tuple"][0])
    key_b = (b["tuple"][2], b["tuple"][1], b["tuple"][0])
    if key_a <= key_b:
        glycosyl, parent = a, b
    else:
        glycosyl, parent = b, a
    return parent, glycosyl
