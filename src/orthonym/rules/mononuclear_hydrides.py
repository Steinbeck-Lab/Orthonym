"""Mononuclear parent-hydride namer (P-68 / P-21.1 / P-31.1.4.2 λ-convention).

(v23 Phase 10 adds a sibling ``name_dinuclear_hydride`` for the two-atom
Group-14/Group-15 catenated hydride ``germylstibane`` family — P-69.5.3 — at
the bottom of this module; it reuses the structural guards but is a distinct
entry point with its own dispatch slot.)

Names a single non-carbon "hub" atom — a Group-15 pnictogen (P/As/Sb/Bi), a
chalcogen (S/Se/Te) or iodine — as a *substitutive parent hydride* (P-68.3),
optionally bearing the λ-convention when the hub valence is non-standard::

    FS(F)(F)(F)(F)F   -> hexafluoro-lambda6-sulfane    (SF6, λ6 nonstandard)
    FS(F)(F)F         -> tetrafluoro-lambda4-sulfane    (SF4, λ4)
    FP(F)(F)(F)F      -> pentafluoro-lambda5-phosphane  (PF5, λ5)
    FI(F)(F)(F)F      -> pentafluoro-lambda5-iodane      (IF5, λ5)
    ClP(Cl)Cl         -> trichlorophosphane              (PCl3, standard valence)
    FS(F)             -> difluorosulfane                 (SF2, standard valence)
    F[Si](F)(F)F      -> tetrafluorosilane               (SiF4, Group-14, std valence)
    Cl[Ge](Cl)(Cl)Cl  -> tetrachlorogermane              (GeCl4, Group-14)
    Cl[As](Cl)Cl      -> trichloroarsane                 (AsCl3, Group-15)
    C[As](C)C         -> trimethylarsane                 (organyl As, Group-15)
    c1ccccc1[As]...   -> triphenylarsane
    [AsH3]            -> arsane                           (bare parent hydride)

Every emitted name round-trips through OPSIN 2.9.0 to the input structure (the
λ-convention parses; ``lambda`` ASCII spelling mirrors the spiro gold
``4lambda4-thiaspiro[3.5]nonane``).

SCOPE (fail-closed, accuracy-first):
  * Two substituent regimes, never mixed (a mixed halo+organyl hub fails both
    guards and cascades onward — fail-closed):
      - **All-halogen** hub (any element in ``_HUB_STEMS``, hub degree >= 2):
        the halogen-prefix block (no enclosing marks — halogens are simple
        substituents, P-16.3.3) + optional λ. Covers SF6/SF4/PF5/IF5 (λ) AND the
        standard-valence PCl3/PF3/SF2/AsCl3/SbBr3/BiCl3 (no λ).
      - **Organyl / bare** hub restricted to **Group-15 As/Sb/Bi only**
        (``_ORGANYL_HUBS``): every substituent a pure unbranched-alkyl or
        phenyl/naphthyl organyl, named via the mononuclear-parent-hydride
        enclosing-mark rule (P-16.5.1.3). Covers trimethylarsane / triphenylarsane
        and the bare arsane/stibane/bismuthane.
  * P / S / Se / Te / I / Si / Ge take only the all-halogen regime here.
    Carbon-substituted Si/Ge (tetramethylsilane) stay with the P-69 organometallic
    hydride-parent namer; carbon-substituted
    P (trimethylphosphane) stays with ``rules.phosphorus.name_phosphine`` (no
    double-claim); carbon-substituted chalcogens are sulfides/sulfanes named
    elsewhere; the organyl regime is reserved for the Group-15 metals As/Sb/Bi,
    which otherwise route to the P-69 organometallic handler and are mis-opened to
    a carbon chain (``C[As](C)C`` -> ``(methylmethyl)methane``).
  * Cl/Br are never hubs (RDKit refuses to construct their hypervalent forms);
    only iodine builds among the halogens, and interhalogens (ICl, degree 1) are
    excluded by the hub-degree>=2 guard.

This is a graph/atom classifier (NOT a SMARTS broadening — per the
feedback_smarts_and_seniority lesson): every guard narrows, never widens.
Oxoacids (hub with O neighbours), oxo-halides (SOCl2), organo-substituted
chalcogens, di-/poly-nuclear hydrides (diphosphane), ions and rings all fail a
guard and cascade onward — zero false positives.
"""

from typing import List, Optional

from rdkit import Chem

from ..assembly.naming_utils import get_multiplier_prefix
from .lambda_convention import nonstandard_bonding_number
from .phosphorus import _build_substituent_string
from .substituent_purity import organyl_prefix_name
from ..metrics.provenance import best_effort_ctx

# Hub element -> parent-hydride stem (P-68 / P-21.1 substitutive parent hydrides).
_HUB_STEMS = {
    'S': 'sulfane',
    'Se': 'selane',
    'Te': 'tellane',
    'P': 'phosphane',
    'As': 'arsane',
    'Sb': 'stibane',
    'Bi': 'bismuthane',
    'I': 'iodane',
    # v23 Phase 8: Group-14 Si/Ge for the ALL-HALOGEN regime only (P-68.2.1.1 +
    # P-67.1.2.5.2: "halides of silicic acid are substitutive names") — SiF4 ->
    # tetrafluorosilane, GeCl4 -> tetrachlorogermane. Kept OUT of _ORGANYL_HUBS so
    # the carbon-substituted forms (tetramethylsilane) stay with the P-69
    # organometallic hydride-parent namer; only the all-halogen tetrahalides are
    # claimed here (a mixed methyltrifluorosilane fails both guards -> cascades).
    'Si': 'silane',
    'Ge': 'germane',
    # Wave-2 completion (P-21.1.1.1): Sn/Pb complete the Group-14 column for the
    # bare-hydride regime ([SnH4] -> stannane) and the all-halogen regime
    # (SnCl4 -> tetrachlorostannane). Same OUT-of-_ORGANYL_HUBS reasoning as
    # Si/Ge: carbon-substituted forms stay with the P-69 organometallic namer.
    'Sn': 'stannane',
    'Pb': 'plumbane',
    # Wave-3 (P-68.1.1.1 / P-69.1 / P-61.3.2.1): Group-13 B/Ga/In/Tl parent
    # hydrides ([BH3] -> borane, C[Ga](C)C -> trimethylgallane). Stems mirror
    # ions.py::_GROUP13_UIDE_STEMS. B standard bonding number is 3, so BH3 ->
    # borane (no lambda). Al is DELIBERATELY excluded (it stays on the
    # organometallics Branch B to protect the trimethylaluminum canary; see
    # _ORGANYL_HUBS below). B is kept OUT of _ORGANYL_HUBS so a boron oxoacid
    # (methylboronic acid, boronic acid) is never claimed here — it fails the
    # halogen/organyl guards on its O neighbours and cascades to the boron-acid
    # handler; only bare BH3 (bare path) and the mixed halo+organyl regime
    # (dichloro(methyl)borane) reach a boron name here. Ga/In/Tl ARE organyl
    # hubs (trimethylgallane / dimethylindigane), directly parallel to As/Sb/Bi.
    'B': 'borane',
    'Ga': 'gallane',
    'In': 'indigane',
    'Tl': 'thallane',
}

# Hubs that additionally accept ORGANYL / bare substituents (no pre-existing
# substitutive namer; the P-69 organometallic handler mangles them). Restricted
# to the Group-15 metals As/Sb/Bi + the Group-13 metals Ga/In/Tl (P-69.1
# substitutive parent-hydride names: C[Ga](C)C -> trimethylgallane, C[In]C ->
# dimethylindigane — the trivalent organyl branch is valence-agnostic, so these
# parallel the working As/Sb/Bi hubs exactly). P stays with name_phosphine,
# chalcogens/iodine with their halide-only / mixed regimes. B and Al are
# EXCLUDED: B protects the boron-oxoacid handler (bare BH3 uses the bare-hydride
# path; mixed CH3-BCl2 uses the mixed regime below), and Al stays on the
# organometallic Branch B to protect the trimethylaluminum canary (ORG-T3-10).
_ORGANYL_HUBS = frozenset({'As', 'Sb', 'Bi', 'Ga', 'In', 'Tl'})

# Halogen substituent prefixes (cited alphanumerically, P-14.5.2; the
# multiplying prefix di/tri/... does NOT count for ordering).
_HALO_PREFIX = {'F': 'fluoro', 'Cl': 'chloro', 'Br': 'bromo', 'I': 'iodo'}
_HALOGENS = frozenset(_HALO_PREFIX)

# Hubs that accept the MIXED halogen + organyl regime (P-61.3.2.1 / P-67.1.2.5.2
# / P-68.5.1): a hub bearing BOTH terminal halogen(s) AND pure-organyl group(s).
# Group-13 (B/Ga/In/Tl) + Group-14 (Si/Ge/Sn/Pb) at standard valence, plus the
# halogen hub I ONLY when hypervalent (lambda != None) so CH3-I stays a
# halo-alkane. BB P-61.3.2.1 sanctions halogen prefixes on exactly this element
# set. Al is excluded (not a hub); P/As/Sb/Bi mixed halo+organyl forms cascade
# (fail-closed) — they are not in the BB P-61.3.2.1 substitutive-halide list.
_MIXED_HALO_ORGANYL_HUBS = frozenset({'B', 'Ga', 'In', 'Tl',
                                      'Si', 'Ge', 'Sn', 'Pb', 'I'})


# === v23 Phase 10 (P-69.5.3): di-nuclear Group-14 / Group-15 catenated hydride ===
# A Group-14 atom bonded to a senior Group-15 parent hydride. Per P-41 the
# Group-15 element outranks Group-14, so it is the PARENT hydride and the
# Group-14 element is the -yl substituent prefix:
#   [GeH3][SbH2] -> germylstibane    [SiH3][AsH2] -> silylarsane
# Only As/Sb/Bi parents (P stays with rules.phosphorus.name_phosphine — no
# double-claim; N is not a metal in this family). Only the Group-14 substituent
# prefixes silyl/germyl/stannyl/plumbyl.
_GROUP14_SUBST_YL = {'Si': 'silyl', 'Ge': 'germyl', 'Sn': 'stannyl', 'Pb': 'plumbyl'}
_GROUP15_HYDRIDE_PARENT = {'As': 'arsane', 'Sb': 'stibane', 'Bi': 'bismuthane'}


def _find_unique_hub(mol):
    """Return the unique nameable hub atom, or None.

    A nameable hub is the SOLE heavy atom whose element is in ``_HUB_STEMS``
    (so diphosphane PH2-PH2 = two P hubs -> None; an all-halogen molecule with no
    hub -> None)."""
    hubs = [a for a in mol.GetAtoms()
            if a.GetSymbol() != 'H' and a.GetSymbol() in _HUB_STEMS]
    return hubs[0] if len(hubs) == 1 else None


# P-44.1.2 class seniority for the senior-atom parent choice (N>P>As>...>Si>C).
_GROUP15_CLASS_RANK = {'N': 0, 'P': 1, 'As': 2, 'Sb': 3, 'Bi': 4,
                       'Si': 5, 'Ge': 6, 'Sn': 7, 'Pb': 8, 'C': 20}


def _find_senior_phosphane_silyl_hub(mol):
    """Return the phosphane P atom for a senior-atom-parent molecule of the
    shape ``(organyl/H)_n P (SiH3)_m`` (m>=1), else None.

    P-44.1.2: P is senior to Si, so P is the parent hydride and each SiH3 is a
    'silyl' substituent. Restricted (fail-closed) to: a single P; every other
    heavy atom is either part of a pure-hydrocarbyl organyl or a bare SiH3
    directly on the P; at least one silyl present (else the C-only phosphane
    stays with name_phosphine)."""
    p_atoms = [a for a in mol.GetAtoms() if a.GetSymbol() == 'P']
    if len(p_atoms) != 1:
        return None
    p = p_atoms[0]
    if p.IsInRing():
        return None
    # No hub-eligible atom more senior than P may be present (N-P etc. handled
    # elsewhere); a Si is allowed only as a bare-silyl substituent (checked in
    # _classify_phosphane_subs). Any Ge/Sn/Pb/other-metal hub -> fail closed.
    for a in mol.GetAtoms():
        sym = a.GetSymbol()
        if sym in ('H', 'C', 'P', 'Si'):
            continue
        return None
    return p


def _classify_halogens(mol, hub) -> Optional[dict]:
    """Return ``{halogen_symbol: count}`` iff EVERY non-hub heavy atom is a
    terminal halogen single-bonded to the hub, else None."""
    hub_idx = hub.GetIdx()
    halo_counts: dict = {}
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'H' or atom.GetIdx() == hub_idx:
            continue
        symbol = atom.GetSymbol()
        if symbol not in _HALOGENS or atom.GetDegree() != 1:
            return None
        neighbors = atom.GetNeighbors()
        if len(neighbors) != 1 or neighbors[0].GetIdx() != hub_idx:
            return None
        bond = mol.GetBondBetweenAtoms(atom.GetIdx(), hub_idx)
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            return None
        halo_counts[symbol] = halo_counts.get(symbol, 0) + 1
    # The hub's neighbours are exactly the counted halogens (no other element).
    if not halo_counts or hub.GetDegree() != sum(halo_counts.values()):
        return None
    return halo_counts


def _classify_organyls(mol, hub) -> Optional[List[str]]:
    """Return the list of organyl prefix names (possibly empty for a bare hydride)
    iff EVERY heavy hub-neighbour is a valid pure-hydrocarbyl organyl AND no stray
    heteroatom exists, else None.

    A halogen neighbour (mixed halo+organyl) -> None (fail-closed; the all-halogen
    regime is handled separately)."""
    hub_idx = hub.GetIdx()
    names: List[str] = []
    for nbr in hub.GetNeighbors():
        if nbr.GetSymbol() == 'H':
            continue
        if nbr.GetSymbol() in _HALOGENS:
            return None  # mixed halo+organyl -> fail-closed
        name = organyl_prefix_name(mol, nbr.GetIdx(), hub_idx)
        if name is None:
            return None
        names.append(name)
    # Defensive: every heavy atom is the hub or a carbon.
    # v29 P3-FIX Item 9: this used to say the per-neighbour walk "rejects any
    # heteroatom inside a substituent". It does NOT -- `organyl_prefix_name`
    # admits `_PREFIX_ONLY_ELEMENTS = {F, Cl, Br, I, At}` inside the fragment
    # (P-59 Table 28 cites halogens only as prefixes, so one can never demand a
    # suffix). It is THIS all-carbon scan, not the walk, that keeps a halogenated
    # organyl out of this regime, so the halogenated class is unreachable here at
    # both revisions -- a pre-existing coverage limit, deliberately left closed
    # rather than a property of the guard above.
    if any(a.GetSymbol() != 'C' and a.GetIdx() != hub_idx
           for a in mol.GetAtoms() if a.GetSymbol() != 'H'):
        return None
    return names


def _classify_mixed_halo_organyl(mol, hub) -> Optional[List[str]]:
    """Return the combined list of substituent prefix names (halogen prefixes +
    organyl prefixes) for a hub bearing BOTH at least one terminal halogen AND at
    least one pure-organyl group, else None (fail-closed).

    The pure-halogen regime (``_classify_halogens``) and the pure-organyl regime
    (``_classify_organyls``) handle their own cases; this one fires only for the
    genuinely MIXED hub (P-61.3.2.1 / P-67.1.2.5.2 / P-68.5.1), e.g. CH3-SiCl3 ->
    trichloro(methyl)silane, CH3-BCl2 -> dichloro(methyl)borane, CH3-ICl2 ->
    dichloro(methyl)-lambda3-iodane. A stray heteroatom (O/N on a boron oxoacid),
    a non-terminal halide, or an impure organyl -> None."""
    hub_idx = hub.GetIdx()
    names: List[str] = []
    n_halo = 0
    n_org = 0
    halo_nbr_idxs = set()
    for nbr in hub.GetNeighbors():
        sym = nbr.GetSymbol()
        if sym == 'H':
            continue
        if sym in _HALOGENS:
            if nbr.GetDegree() != 1:
                return None
            bond = mol.GetBondBetweenAtoms(nbr.GetIdx(), hub_idx)
            if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
                return None
            names.append(_HALO_PREFIX[sym])
            halo_nbr_idxs.add(nbr.GetIdx())
            n_halo += 1
        elif sym == 'C':
            name = organyl_prefix_name(mol, nbr.GetIdx(), hub_idx)
            if name is None:
                return None
            names.append(name)
            n_org += 1
        else:
            return None  # stray heteroatom (boron oxoacid O, amino N, ...)
    if n_halo == 0 or n_org == 0:
        return None  # not mixed -> a dedicated pure regime handles it
    # Full coverage: every heavy atom is the hub, a counted terminal halogen, or a
    # carbon inside a verified organyl.
    # v29 P3-FIX Item 9: the parenthetical used to claim `organyl_prefix_name`
    # "rejected any internal heteroatom". It does not -- it admits F/Cl/Br/I/At
    # inside the fragment. The all-carbon requirement below is what actually
    # holds, and it is what makes the claim true of THIS function rather than of
    # the primitive it calls.
    for a in mol.GetAtoms():
        if a.GetSymbol() == 'H':
            continue
        if a.GetIdx() == hub_idx or a.GetIdx() in halo_nbr_idxs or a.GetSymbol() == 'C':
            continue
        return None
    return names


def _build_mixed_substituent_string(names: List[str]) -> Optional[str]:
    """Compose the halogen+organyl prefix block per P-16.3.3 / P-16.5.1.3, exactly
    as the Blue Book cites the substitutive halides of B/Si/... parent hydrides:
    sort ALL substituent base-names alphanumerically; the FIRST unique group takes
    no enclosing marks, each SUBSEQUENT unique group is enclosed in parentheses
    with its multiplying prefix OUTSIDE the marks.

        ['chloro','chloro','chloro','methyl'] -> 'trichloro(methyl)'   (BB 35754)
        ['chloro','chloro','methyl']          -> 'dichloro(methyl)'    (BB 39668)
        ['chloro','methyl','methyl']          -> 'chlorodi(methyl)'    (BB 25866)
        ['bromo','chloro','phenyl']           -> 'bromo(chloro)(phenyl)' (BB 35750)

    Returns None if a multiplicity exceeds the supported multiplier table."""
    from collections import Counter

    from ..assembly.naming_utils import (apply_enclosing_marks,
                                         enclose_if_compound,
                                         multiplied_component,
                                         prefix_citation_sort_key)
    # v29 P3-CLOSEOUT Item A: the arity BOUND stays local (fail closed beyond
    # it), but the multiplier WORD now comes from the shared primitive, which
    # knows P-16.3.5(a). This table could only ever say `di`/`tri`, so a
    # SUBSTITUTED prefix here emitted `tri(2-methylpropyl)arsane` where the
    # Blue Book requires `tris(...)`.
    _SUPPORTED_COUNTS = frozenset(range(1, 9))
    counts = Counter(names)
    parts = []
    # v29 P3: the organyl guard feeding this is the shared chokepoint, so a prefix
    # may now carry a locant, a retained italicized prefix, or its own marks. Raw
    # `sorted()` keyed `tert-butyl` on its 't'; P-14.5.2/P-14.5.4 keys on the
    # letters ('butyl'), which is also what decides whether the compound prefix
    # lands in the unmarked FIRST slot.
    for i, name in enumerate(sorted(counts, key=prefix_citation_sort_key)):
        if counts[name] not in _SUPPORTED_COUNTS:
            return None
        if i == 0:
            # First unique: P-16.5.1.3.1 withholds only the marks that SEPARATE
            # the groups, so a compound prefix still takes its own P-16.5.1.1 marks
            # (`(cyclohexylmethyl)di(methyl)silane`), and the P-16.3.3(b)/P-16.2.4.1(d) italicized
            # carve-out keeps `di-tert-butyl` hyphenated rather than `ditert-`.
            marked = enclose_if_compound(name)
            parts.append(multiplied_component(counts[name], name, marked))
        else:
            # Subsequent: marks always, multiplier OUTSIDE them (BB 25866
            # 'chlorodi(methyl)'), escalating ( -> [ over an inner pair.
            parts.append(multiplied_component(
                counts[name], name, apply_enclosing_marks(name, -1)))
    return ''.join(parts)


def _assemble(prefix_block: str, lam: Optional[int], stem: str) -> str:
    """Compose ``<prefix><stem>`` (standard valence) or
    ``<prefix>-lambda<n>-<stem>`` (non-standard valence, P-31.1.4.2)."""
    if lam is None:
        return f"{prefix_block}{stem}"
    if not prefix_block:
        return f"lambda{lam}-{stem}"
    return f"{prefix_block}-lambda{lam}-{stem}"


_PNICTOGEN_YL_STEM = {'As': 'arsanyl', 'Sb': 'stibanyl', 'Bi': 'bismuthanyl'}


def name_arsanyl_substituent(mol, frag_atoms, attach_idx: int) -> Optional[str]:
    """P-67.1.5.1 / P-68.3: a pnictogen-rooted substituent (As/Sb/Bi) named on the
    parent hydride arsane/stibane/bismuthane -> ``{prefixes}{arsanyl|stibanyl|bismuthanyl}``.

        -As(OH)2  -> dihydroxyarsanyl  (P-67.1.5.1: -COOH is senior to -As(OH)2,
                                        so the arsonic acid is cited as a prefix)
        -AsH2     -> arsanyl
        -As(CH3)2 -> dimethylarsanyl
        -Sb(C6H5)2 -> diphenylstibanyl  (P-68.3.2.3.2.2: stibanyl preselected prefix)
        -BiH2     -> bismuthanyl

    Named for the historical As-only origin; now general over the whole Group-15
    (pnictogen) family As/Sb/Bi (Bi added for symmetry; all share the '-anyl' stem).
    Fail-closed (returns ``None``) for a non-pnictogen attachment, a charge / radical,
    any multiple bond on the pnictogen, or a substituent that is neither a terminal
    ``-OH`` nor a simple unbranched-alkyl / phenyl-naphthyl organyl. Pure: no mol
    mutation."""
    if attach_idx is None:
        return None
    frag_set = set(frag_atoms)
    if attach_idx not in frag_set:
        return None
    a = mol.GetAtomWithIdx(attach_idx)
    _stem = _PNICTOGEN_YL_STEM.get(a.GetSymbol())
    if (_stem is None or a.GetFormalCharge() != 0
            or a.GetNumRadicalElectrons() != 0):
        return None
    from .substituent_purity import organyl_prefix_name
    from .phosphorus import _build_substituent_string
    prefixes = []
    for b in a.GetBonds():
        nb = b.GetOtherAtom(a)
        if nb.GetIdx() not in frag_set:
            continue                                  # the parent attachment
        if b.GetBondType() != Chem.BondType.SINGLE:
            return None
        sym = nb.GetSymbol()
        if (sym == 'O' and nb.GetFormalCharge() == 0 and nb.GetDegree() == 1
                and nb.GetTotalNumHs() == 1):
            prefixes.append('hydroxy')
        elif sym == 'C':
            nm = organyl_prefix_name(mol, nb.GetIdx(), attach_idx)
            if nm is None:
                return None
            prefixes.append(nm)
        else:
            return None
    if not prefixes:
        return _stem                                  # bare -AsH2/-SbH2/-BiH2
    return f"{_build_substituent_string(prefixes)}{_stem}"


def name_mononuclear_hydride(mol) -> Optional[str]:
    """Return the substitutive PIN for a mononuclear parent hydride (P-68 /
    P-31.1.4.2), else ``None`` (fail-closed cascade-continuation).

    Pure: no mol mutation, no global state.
    """
    if mol is None:
        return None

    # Single neutral fragment, no radicals.
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() != 0:
            return None

    hub = _find_unique_hub(mol)
    if hub is None:
        # P-44.1.2: multiple hub-eligible atoms -> the SENIOR element (class
        # order N>P>As>...>Si>...>C) is the parent hydride; the rest are
        # substituents. Restricted to a phosphane P parent bearing a bare silyl
        # (C[PH][SiH3] -> methyl(silyl)phosphane); handled by the dedicated
        # branch below (fail-closed for any other senior-atom shape).
        senior = _find_senior_phosphane_silyl_hub(mol)
        if senior is None:
            return None
        hub = senior
    # The hub itself must NOT be a ring member (an arsenin/thiophene/arsole ring
    # parent is a heterocycle, not a mononuclear parent hydride) — but aromatic
    # ring SUBSTITUENTS (triphenylarsane) are fine: the per-substituent purity
    # guard handles those, so we check the hub specifically, not the whole mol.
    if hub.IsInRing():
        return None
    hub_idx = hub.GetIdx()
    stem = _HUB_STEMS[hub.GetSymbol()]
    lam = nonstandard_bonding_number(mol, hub_idx)

    # --- Bare parent hydride (P-21.1.1.1 / P-52.1.1): the hub is the ONLY
    #     heavy atom (single-fragment guard above makes zero heavy neighbours
    #     equivalent), saturated with hydrogen. [SiH4] -> silane, [SH2] ->
    #     sulfane, and the lambda-convention hypervalent forms (P-21.1.2):
    #     [PH5] -> lambda5-phosphane, [SH4] -> lambda4-sulfane,
    #     [IH3] -> lambda3-iodane. ---
    if not any(n.GetSymbol() != 'H' for n in hub.GetNeighbors()):
        return _assemble("", lam, stem)

    # --- All-halogen regime (every hub element; hub degree >= 2 excludes the
    #     diatomic interhalogens ICl/IBr). Halogens are simple substituents:
    #     alphanumerical concatenation with NO enclosing marks (P-16.5.1.1). ---
    halo_counts = _classify_halogens(mol, hub)
    if halo_counts is not None and hub.GetDegree() >= 2:
        halo_parts = []
        for symbol in sorted(halo_counts, key=lambda s: _HALO_PREFIX[s]):
            halo_name = _HALO_PREFIX[symbol]
            multiplier = get_multiplier_prefix(halo_counts[symbol], halo_name)
            halo_parts.append(f"{multiplier}{halo_name}")
        return _assemble(''.join(halo_parts), lam, stem)

    # --- Organyl / bare regime (Group-15 As/Sb/Bi + Group-13 Ga/In/Tl). ---
    if hub.GetSymbol() in _ORGANYL_HUBS:
        organyls = _classify_organyls(mol, hub)
        if organyls is not None:
            prefix_block = _build_substituent_string(organyls) if organyls else ""
            return _assemble(prefix_block, lam, stem)

    # --- Mixed halogen + organyl regime (P-61.3.2.1 / P-67.1.2.5.2 / P-68.5.1):
    #     Group-13 (B/Ga/In/Tl) + Group-14 (Si/Ge/Sn/Pb) at any valence, plus a
    #     hypervalent iodine hub (lambda != None so CH3-I stays a halo-alkane).
    #     CH3-SiCl3 -> trichloro(methyl)silane, CH3-BCl2 -> dichloro(methyl)borane,
    #     CH3-ICl2 -> dichloro(methyl)-lambda3-iodane. ---
    if hub.GetSymbol() in _MIXED_HALO_ORGANYL_HUBS and not (
            hub.GetSymbol() in _HALOGENS and lam is None):
        mixed = _classify_mixed_halo_organyl(mol, hub)
        if mixed is not None:
            prefix_block = _build_mixed_substituent_string(mixed)
            if prefix_block is not None:
                return _assemble(prefix_block, lam, stem)

    # --- P-44.1.2: a phosphane hub bearing a SILYL (SiH3) substituent (with
    #     optional pure-hydrocarbyl organyls). P > Si in the seniority of
    #     classes, so P is the parent; the SiH3 is the 'silyl' substituent
    #     (name_phosphine only counts C neighbours and would silently DROP the
    #     silyl). Only fires when >=1 silyl is present, so plain C-only
    #     phosphanes stay with name_phosphine (no double-path). ---
    if hub.GetSymbol() == 'P':
        subs = _classify_phosphane_subs(mol, hub)
        if subs is not None and 'silyl' in subs:
            prefix_block = _build_substituent_string(subs)
            return _assemble(prefix_block, lam, stem)
        # v30 T4 degrade floor (best-effort ONLY): a pure-organyl phosphane whose
        # substituent is a RING (tricyclododecylphosphane) is never reached by
        # name_phosphine -- the ring routing bypasses it -- and would otherwise
        # abstain (measured DEGRADE row). _classify_phosphane_subs already names the
        # ring organyls, so emit the substitutive phosphane. Gated on best_effort_ctx
        # so the PIN path stays byte-identical (name_phosphine still owns acyclic
        # phosphanes on the PIN cascade; a phosphane-with-ring-substituent PIN
        # spelling is a separate refinement). 0-wrong: SELF-01 backstops the emission.
        if subs is not None and best_effort_ctx.get():
            return _assemble(_build_substituent_string(subs), lam, stem)

    return None


def _classify_phosphane_subs(mol, hub) -> Optional[List[str]]:
    """Classify EVERY heavy hub-neighbour of a phosphane P as either a pure
    hydrocarbyl organyl (methyl/ethyl/phenyl...) or a bare silyl (-SiH3), else
    None. Returns the list of substituent prefix names. Fail-closed: any other
    heteroatom substituent, a substituted silyl, or an impure organyl -> None."""
    hub_idx = hub.GetIdx()
    names: List[str] = []
    for nbr in hub.GetNeighbors():
        if nbr.GetSymbol() == 'H':
            continue
        if nbr.GetSymbol() == 'Si':
            # bare silyl only: SiH3 attached solely to the hub (degree 1)
            if nbr.GetDegree() != 1 or nbr.GetTotalNumHs() != 3:
                return None
            if nonstandard_bonding_number(mol, nbr.GetIdx()) is not None:
                return None
            names.append('silyl')
            continue
        name = organyl_prefix_name(mol, nbr.GetIdx(), hub_idx)
        if name is None:
            return None
        names.append(name)
    return names or None


_HETERONE_STEMS = {'Si': 'silanone', 'Ge': 'germanone',
                   'P': 'phosphanone', 'As': 'arsanone'}

# P-61.6 "HETERONES" (BlueBookV2.md:25977) enumerates the heterone class as
# exactly four groups, mapped 1:1 onto four preselected prefixes:
#   "Compounds containing the -PO, -PO2, -AsO or -AsO2 are called heterones
#    (see P-64.1.2.2, P-64.4). In the presence of a more senior characteristic
#    group they are described by the compound prefixes oxophosphanyl,
#    dioxo-lambda5-phosphanyl, oxoarsanyl, and dioxo-lambda5-arsanyl."
# The two -XO2 members are DIONES. P-64.1.2.2 "Heterones" (:28281) supplies the
# suffix: "Heterones are compounds having an oxygen atom formally doubly bonded
# to a heteroatom ... They are named in the same way as ketones" -- so two oxo
# groups on one parent take the multiplied '-dione', and the hub's bonding
# number of 5 supplies the lambda descriptor (P-31.1.4.2). Worked (PIN)
# examples, both re-opened at write time:
#   CH3-PO2   methyl-lambda5-phosphanedione (PIN)   :28287, under P-64.1.2.2
#   C6H5-PO2  phenyl-lambda5-phosphanedione (PIN)   :25983, under P-61.6
#
# Membership of THIS dict is the element gate for the dione, and it is
# deliberately narrower than _HETERONE_STEMS:
#   * Si/Ge are absent -- P-61.6 does not place them in the class, and RDKit
#     rejects a two-oxo Si/Ge hub outright (valence), so there is no molecule.
#   * 'arsanedione' is not printed verbatim in the Blue Book, but it is fully
#     DETERMINED by the two rules above: P-61.6 declares -AsO2 a heterone in the
#     same sentence as -PO2 and prints its preselected prefix
#     dioxo-lambda5-arsanyl (:25977, and the prefix table :55895 with structure
#     O2As-), while the 'arsane' stem, its lambda5 form and the mononuclear
#     '-one'/'-dione' suffix are each printed elsewhere (arsanone PIN :25985,
#     trimethyl-lambda5-arsanone :43057, phosphanedione PIN x2). Every morpheme
#     is printed and the elision follows phosphane+dione exactly.
#   * The CHALCOGEN analogue (R-PS2, R3P=S) is deliberately NOT built. The Blue
#     Book prints no worked example of a thione suffix on a phosphane stem, so
#     its spelling would have to be invented; it stays fail-closed via the
#     terminal-oxygen check below. See 
_HETERONE_DIONE_STEMS = {'P': 'phosphanedione', 'As': 'arsanedione'}


def name_heterone(mol) -> Optional[str]:
    """P-64.1.2.2 / P-64.4.1 heterone parents (Wave-2 completion C):
    (CH3)2Si=O -> dimethylsilanone (BB verbatim), CH3SiH=O -> methylsilanone,
    R3P=O -> lambda5-phosphanone forms. Chalcogens are EXCLUDED (sulfoxides /
    sulfones are P-63.6 compulsory-prefix exceptions per P-64.4), as is N.

    Covers BOTH oxo counts of the P-61.6 heterone class -- the mono-oxo -PO/-AsO
    ('-one') and the DIONE -PO2/-AsO2 ('-dione'), CH3-PO2 ->
    methyl-lambda5-phosphanedione (PIN, BB :28287). The oxo count selects the
    stem dict; see _HETERONE_DIONE_STEMS for the derivation and for why Si/Ge
    and the P=S chalcogen analogue stay fail-closed.

    Fail-closed; pure."""
    if mol is None:
        return None
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() != 0:
            return None

    hubs = [a for a in mol.GetAtoms() if a.GetSymbol() in _HETERONE_STEMS]
    if len(hubs) != 1:
        return None
    hub = hubs[0]
    if hub.IsInRing():
        return None
    doubles = [b for b in hub.GetBonds()
               if b.GetBondType() == Chem.BondType.DOUBLE]
    if not 1 <= len(doubles) <= 2:
        return None
    oxos = [b.GetOtherAtom(hub) for b in doubles]
    # Every doubly-bonded partner must be a TERMINAL oxygen. This is the single
    # check that keeps =S, =N and =C partners out, so the out-of-scope P=S
    # chalcogen heterone and the phosphanimines stay fail-closed, and a bridging
    # (degree-2) oxygen -- e.g. the P-O-P of a metaphosphate -- is rejected.
    if any(o.GetSymbol() != 'O' or o.GetDegree() != 1 for o in oxos):
        return None
    if len(oxos) == 2:
        # -PO2 / -AsO2: P-61.6 + P-64.1.2.2, see _HETERONE_DIONE_STEMS.
        stem = _HETERONE_DIONE_STEMS.get(hub.GetSymbol())
        if stem is None:
            return None
    else:
        stem = _HETERONE_STEMS[hub.GetSymbol()]
    oxo_idxs = {o.GetIdx() for o in oxos}
    from .substituent_purity import organyl_prefix_name
    prefixes = []
    for nb in hub.GetNeighbors():
        if nb.GetIdx() in oxo_idxs:
            continue
        name = organyl_prefix_name(mol, nb.GetIdx(), hub.GetIdx())
        if name is None:
            return None
        prefixes.append(name)
    from .lambda_convention import nonstandard_bonding_number
    lam = nonstandard_bonding_number(mol, hub.GetIdx())
    # P-16.5.1.3 mononuclear enclosing marks (W5-C): 2+ DIFFERENT substituents on a
    # mononuclear hub are cited alphanumerically with the first unmarked and the rest
    # in enclosing marks -> 'methyl(phenyl)(propyl)' (NOT the OPSIN-ambiguous
    # 'methylphenylpropyl', where 'phenylpropyl' reads as one compound substituent).
    # Identical substituents keep the plain multiplied form (trimethyl / triphenyl).
    if len(prefixes) > 4:
        return None
    from .phosphorus import _build_substituent_string
    prefix_str = _build_substituent_string(prefixes)
    return _assemble(prefix_str, lam, stem)


_HETEROIMINE_STEMS = {'P': 'phosphan', 'As': 'arsan', 'Si': 'silan'}


def name_heteroimine(mol) -> Optional[str]:
    """P-62.3.1.3 (BB 26568): X=NH, X a mononuclear-hydride heteroatom hub
    -> '<organyls><stem>imine' (CH3-P=NH -> 1-methylphosphanimine).

    Mirrors ``name_heterone`` with =NH in place of =O. Fail-closed: single
    neutral fragment, no rings, exactly one terminal =NH on the hub, only
    pure-hydrocarbyl organyl co-substituents, at most one organyl (multi-
    organyl N/locant machinery not built here). Pure — no mol mutation."""
    if mol is None:
        return None
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    if mol.GetRingInfo().NumRings() > 0:
        return None
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() != 0:
            return None
    # Exactly one terminal =N (degree-1, double-bonded to the hub): the =NH
    # imine group. An N-substituted =N-R (degree 2) fails this guard.
    imine_n = [a for a in mol.GetAtoms()
               if a.GetAtomicNum() == 7 and a.GetDegree() == 1
               and a.GetBonds()[0].GetBondType() == Chem.BondType.DOUBLE]
    if len(imine_n) != 1:
        return None
    hub = imine_n[0].GetNeighbors()[0]
    stem = _HETEROIMINE_STEMS.get(hub.GetSymbol())
    if stem is None:
        return None
    if hub.IsInRing():
        return None
    # Remaining hub neighbours must be pure organyls (else fail-closed).
    organyls: List[str] = []
    for nb in hub.GetNeighbors():
        if nb.GetIdx() == imine_n[0].GetIdx():
            continue
        if nb.GetSymbol() == 'H':
            continue
        name = organyl_prefix_name(mol, nb.GetIdx(), hub.GetIdx())
        if name is None:
            return None
        organyls.append(name)
    # No stray heteroatom beyond the hub + the imine N.
    if any(a.GetSymbol() not in ('C', 'H')
           and a.GetIdx() not in (hub.GetIdx(), imine_n[0].GetIdx())
           for a in mol.GetAtoms()):
        return None
    if not organyls:
        return f"{stem}imine"
    if len(organyls) != 1:
        return None      # multi-organyl locant assembly not built here
    # The organyl follows a locant + hyphen, so a compound prefix takes its
    # P-16.5.1.1 marks (`1-(propan-2-yl)phosphanimine`) while the retained
    # italicized prefix stays bare (P-16.3.3(b)/P-16.2.4.1(d)).
    from ..assembly.naming_utils import enclose_if_compound
    return f"1-{enclose_if_compound(organyls[0])}{stem}imine"


_PHOSPHANIMINE_STEMS = {'P': 'phosphan', 'As': 'arsan', 'Sb': 'stiban'}
_PHOSPHANIMINE_MULT = {1: '', 2: 'di', 3: 'tri', 4: 'tetra'}


def name_lambda5_phosphanimine(mol) -> Optional[str]:
    """P-74.2.1.5 phosphine imide, named substitutively (method 3 = PIN) as a
    lambda5-phosphanimine (a heterimine on a lambda5 P/As/Sb hub)::

        (C6H5)3P=N-CH2CH3 -> N-ethyl-P,P,P-triphenyl-lambda5-phosphanimine (BB 43076)

    The hub X (P/As/Sb) carries exactly one imine ``=N`` and, at the non-standard
    (lambda5) bonding number, three single-bonded organyls; the imine N may bear
    one organyl (the 'N-' substituent) or an H. Organyls on the hub take the
    italic element-symbol locant (P,P,P); the N-substituent takes 'N'. Only the
    lambda5 (five-bond hub) case is owned here — the standard-valence heteroimine
    (CH3-P=NH -> 1-methylphosphanimine) stays with ``name_heteroimine``.
    Fail-closed (None); pure — no mol mutation."""
    if mol is None:
        return None
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() != 0:
            return None
    hubs = [a for a in mol.GetAtoms() if a.GetSymbol() in _PHOSPHANIMINE_STEMS]
    if len(hubs) != 1:
        return None
    hub = hubs[0]
    if hub.IsInRing():
        return None
    from .lambda_convention import nonstandard_bonding_number
    if nonstandard_bonding_number(mol, hub.GetIdx()) != 5:
        return None  # only lambda5 R3X=N-R owned here (lambda3 -> name_heteroimine)
    # Exactly one imine =N on the hub (degree 1 =NH or degree 2 =N-R).
    imine_ns = [b.GetOtherAtom(hub) for b in hub.GetBonds()
                if b.GetBondType() == Chem.BondType.DOUBLE
                and b.GetOtherAtom(hub).GetSymbol() == 'N']
    if len(imine_ns) != 1:
        return None
    n_atom = imine_ns[0]
    # The imine N must carry ONLY the =X plus at most one single-bonded R (+ H).
    n_heavy = [nb for nb in n_atom.GetNeighbors() if nb.GetIdx() != hub.GetIdx()
               and nb.GetSymbol() != 'H']
    if len(n_heavy) > 1:
        return None
    if any(b.GetBondType() != Chem.BondType.SINGLE
           for b in n_atom.GetBonds()
           if b.GetOtherAtom(n_atom).GetIdx() != hub.GetIdx()):
        return None  # the N's non-hub bonds must be single (no cumulene)

    # Hub organyls (every hub neighbour except the imine N): cited with 'P' locants.
    hub_subs = []
    for nb in hub.GetNeighbors():
        if nb.GetIdx() == n_atom.GetIdx() or nb.GetSymbol() == 'H':
            continue
        name = organyl_prefix_name(mol, nb.GetIdx(), hub.GetIdx())
        if name is None:
            return None
        hub_subs.append(name)
    # The imine-N substituent (if any): cited with the 'N' locant.
    n_sub = None
    if n_heavy:
        n_sub = organyl_prefix_name(mol, n_heavy[0].GetIdx(), n_atom.GetIdx())
        if n_sub is None:
            return None

    from ..assembly.naming_utils import (alpha_sort_key, enclose_if_compound,
                                         multiplier_needs_hyphen)
    locant = 'P' if hub.GetSymbol() == 'P' else hub.GetSymbol()[0].upper()
    from collections import Counter

    def _block(name, locants):
        count = len(locants)
        mult = _PHOSPHANIMINE_MULT.get(count)
        if mult is None:
            return None
        loc_str = ','.join(locants)
        # `enclose_if_compound` replaces the raw `f"({name})"`: it ESCALATES over
        # an inner pair (P-16.5.4.1) and carries the P-16.3.4 carve-out, so
        # `tert-butyl` is cited bare and keeps its hyphen under a multiplier.
        marked = enclose_if_compound(name)
        if mult and marked == name and multiplier_needs_hyphen(name):
            body = f"{mult}-{marked}"
        else:
            body = f"{mult}{marked}"
        return f"{loc_str}-{body}"

    prefix_items = []  # (alpha_key, block_text)
    for name, count in Counter(hub_subs).items():
        blk = _block(name, [locant] * count)
        if blk is None:
            return None
        prefix_items.append((alpha_sort_key(name), blk))
    if n_sub is not None:
        blk = _block(n_sub, ['N'])
        if blk is None:
            return None
        prefix_items.append((alpha_sort_key(n_sub), blk))
    prefix_items.sort(key=lambda t: t[0])
    prefix_block = '-'.join(b for _, b in prefix_items)
    return _assemble(prefix_block, 5, f"{_PHOSPHANIMINE_STEMS[hub.GetSymbol()]}imine")


_LAMBDA_IMINE_OXIDE_HUBS = {'S': 'sulfane', 'Se': 'selane', 'Te': 'tellane'}
# Suffix multipliers for the -one / -imine char-group suffix (P-16.3.4).
_IO_SUFFIX_MULT = {1: '', 2: 'di', 3: 'tri', 4: 'tetr'}
# Prefix multipliers (di/tri) for substituent + imino prefixes.
_IO_PREFIX_MULT = {1: '', 2: 'di', 3: 'tri', 4: 'tetra'}


def _apply_hydride_suffix(stem: str, suffix: str) -> str:
    """Attach a suffix to a parent-hydride stem, eliding the stem's terminal 'e'
    before a vowel-initial suffix (P-16.7.1(a)): sulfane+one -> sulfanone,
    sulfane+imine -> sulfanimine, but sulfane+dione -> sulfanedione (consonant)."""
    if suffix and suffix[0] in 'aeiouy' and stem.endswith('e'):
        return stem[:-1] + suffix
    return stem + suffix


def _compose_io_prefixes(items) -> Optional[str]:
    """Build the alphanumerically-ordered detachable-prefix block for the
    lambda-sulfane imine/oxide family, per P-16.3.3 (a compound/complex prefix
    such as ``methylimino`` is ALWAYS enclosed) + P-16.5.1.3 (for a mononuclear
    parent hydride the FIRST cited group takes no marks and each subsequent group
    is enclosed).

    ``items`` is a list of ``(name, is_complex)`` tuples. Returns the composed
    string, or None if a multiplicity exceeds the supported table.

        [('methylimino', True)]                       -> '(methylimino)'
        [('methyl',False),('methyl',False),('phenylimino',True)]
                                                       -> 'dimethyl(phenylimino)'
        [('phenyl',False),('phenyl',False)]           -> 'diphenyl'
        [('ethyl',False),('methyl',False)]            -> 'ethyl(methyl)'
    """
    from collections import Counter

    from ..assembly.naming_utils import (apply_enclosing_marks,
                                         enclose_if_compound,
                                         multiplier_needs_hyphen,
                                         prefix_citation_sort_key)
    names = [n for n, _ in items]
    complex_names = {n for n, c in items if c}
    counts = Counter(names)
    parts = []
    # v29 P3: the S-organyls now come from the shared chokepoint, so a prefix here
    # may carry a locant or a retained italicized prefix. Two consequences:
    #   * order by P-14.5.2/P-14.5.4, not raw string order;
    #   * the multiplier goes OUTSIDE the marks. BB 7272 (P-16.5.1.3.1) verbatim:
    #     "When the simple substituent groups are accompanied by multiplicative
    #     prefixes such as 'di' and 'tri', the multiplicative prefixes are NOT
    #     included in the parentheses" -- BB 16286 spells the resulting shape,
    #     `tert-butyldi(methyl)phosphane` (PIN). The old `f"({mult}{name})"` put
    #     them inside and produced `imino(dimethyl)` and, once widened, the
    #     ambiguous `imino(dipropan-2-yl)`.
    for i, name in enumerate(sorted(counts, key=prefix_citation_sort_key)):
        mult = _IO_PREFIX_MULT.get(counts[name])
        if mult is None:
            return None
        marked = enclose_if_compound(name)
        if marked == name and ((i > 0) or (name in complex_names)):
            marked = apply_enclosing_marks(name, -1)
        if mult and marked == name and multiplier_needs_hyphen(name):
            parts.append(f"{mult}-{marked}")          # P-16.3.3(b)/P-16.2.4.1(d) di-tert-butyl
        else:
            parts.append(f"{mult}{marked}")
    return ''.join(parts)


def name_lambda_sulfane_imine_oxide(mol) -> Optional[str]:
    """P-68.4.3.3 / .4 / .5 / .6 / .7 / .8: the mononuclear lambda-sulfane
    imine/oxide family — a single non-ring chalcogen hub E in {S, Se, Te} of
    non-standard valence (lambda4 / lambda6) bearing any combination of

      * single-bonded organyl groups (S-substituents),
      * terminal ``=O`` (oxo), and
      * ``=N-H`` / ``=N-R`` imine groups,

    with AT LEAST ONE imine (a pure sulfoxide/sulfone with no imine stays with
    the P-63.6 sulfur handler). Seniority (P-41): ``=O`` outranks ``=N``, so when
    ANY oxo is present it is the principal characteristic group (``-one``/
    ``-dione`` suffix) and every imine becomes an ``(R-imino)`` prefix; when NO
    oxo is present the imines are the suffix (``-imine``/``-diimine``/
    ``-triimine``)::

        (C2H5)2S=N-C6H5      -> S,S-diethyl-N-phenyl-lambda4-sulfanimine  (.3)
        CH3-N=S(=O)2         -> (methylimino)-lambda6-sulfanedione        (.4)
        (C6H5)2S(=NH)2       -> diphenyl-lambda6-sulfanediimine           (.5)
        (CH3)2S(=O)=N-C6H5   -> dimethyl(phenylimino)-lambda6-sulfanone   (.6)
        CH3-N=S=N-CH2CH3     -> ethyl(methyl)-lambda4-sulfanediimine      (.7)
        CH3-N=S(=NCH3)=NC6H5 -> dimethyl(phenyl)-lambda6-sulfanetriimine  (.8)

    Italic element locants (``S,S-`` / ``N-``) are cited ONLY when substituents
    sit on BOTH the hub AND an imine nitrogen (P-68.4.3.3 example); when all
    substituents live on one kind of atom the italic locants are unnecessary and
    omitted (the other five examples). Every emitted name round-trips through
    OPSIN 2.9.0.

    SCOPE (fail-closed, accuracy-first — a graph classifier, NOT a SMARTS
    broadening): exactly one S/Se/Te hub, not in a ring, neutral, non-radical,
    single fragment, hub valence non-standard, >=1 imine, and every hub
    neighbour is a pure-hydrocarbyl organyl (single bond), a terminal =O, or an
    =N with at most one pure-hydrocarbyl substituent. A non-hydrocarbyl N/S
    substituent (which would create a SENIOR parent, P-68.4.3.3 note), a
    single-bonded heteroatom on the hub, a standard-valence hub, or the
    both-substituted-with-multiple-imine-N combination all fail a guard and
    cascade onward — zero false positives. Pure: no mol mutation.
    """
    if mol is None:
        return None
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() != 0:
            return None

    hubs = [a for a in mol.GetAtoms() if a.GetSymbol() in _LAMBDA_IMINE_OXIDE_HUBS]
    if len(hubs) != 1:
        return None
    hub = hubs[0]
    if hub.IsInRing():
        return None
    hub_idx = hub.GetIdx()
    lam = nonstandard_bonding_number(mol, hub_idx)
    if lam is None:
        return None  # standard-valence chalcogen (a sulfide/sulfoxide) — decline

    oxo_atoms = []       # terminal =O
    imine_ns = []        # =N (=NH or =N-R)
    s_organyls = []      # single-bonded organyl prefix names on the hub
    for b in hub.GetBonds():
        nbr = b.GetOtherAtom(hub)
        sym = nbr.GetSymbol()
        bt = b.GetBondType()
        if bt == Chem.BondType.DOUBLE:
            if sym == 'O':
                if nbr.GetDegree() != 1 or nbr.GetTotalNumHs() != 0:
                    return None
                oxo_atoms.append(nbr)
            elif sym == 'N':
                imine_ns.append(nbr)
            else:
                return None  # =C (sulfine) / =S etc. — not this family
        elif bt == Chem.BondType.SINGLE:
            if sym != 'C':
                return None  # single-bonded heteroatom on hub — not this family
            name = organyl_prefix_name(mol, nbr.GetIdx(), hub_idx)
            if name is None:
                return None
            s_organyls.append(name)
        else:
            return None  # triple / aromatic bond on hub

    if not imine_ns:
        return None  # pure sulfoxide/sulfone — P-63.6 sulfur handler owns it

    # Validate each imine N: double-bonded to the hub only, at most one pure-
    # hydrocarbyl substituent (a non-hydrocarbyl N-substituent would create a
    # senior parent — P-68.4.3.3 note — so fail closed).
    n_substituents = []
    for n in imine_ns:
        if n.GetFormalCharge() != 0 or n.IsInRing():
            return None
        for nb in n.GetBonds():
            if nb.GetOtherAtom(n).GetIdx() == hub_idx:
                continue
            if nb.GetBondType() != Chem.BondType.SINGLE:
                return None  # N=X (X != hub) — cumulated/azo — not this family
        heavy_other = [x for x in n.GetNeighbors()
                       if x.GetIdx() != hub_idx and x.GetSymbol() != 'H']
        if not heavy_other:
            if n.GetDegree() != 1:
                return None  # =N- with no organyl but degree>1 — malformed
            continue         # =NH
        if len(heavy_other) != 1 or heavy_other[0].GetSymbol() != 'C':
            return None
        name = organyl_prefix_name(mol, heavy_other[0].GetIdx(), n.GetIdx())
        if name is None:
            return None
        n_substituents.append(name)

    # Full coverage: every heavy atom is the hub, a counted oxo O, an imine N, or
    # a carbon inside a verified organyl (any stray heteroatom -> fail closed).
    allowed = {hub_idx}
    allowed |= {a.GetIdx() for a in oxo_atoms}
    allowed |= {a.GetIdx() for a in imine_ns}
    for a in mol.GetAtoms():
        if a.GetSymbol() in ('H', 'C') or a.GetIdx() in allowed:
            continue
        return None

    stem = _LAMBDA_IMINE_OXIDE_HUBS[hub.GetSymbol()]
    o = len(oxo_atoms)
    i = len(imine_ns)

    if o >= 1:
        # P-41: =O is the principal characteristic group -> -one suffix; every
        # imine becomes an (R-imino) prefix. S-organyls are plain prefixes.
        suffix_mult = _IO_SUFFIX_MULT.get(o)
        if suffix_mult is None:
            return None
        suffix = f"{suffix_mult}one"
        items = [(nm, False) for nm in s_organyls]
        for nm in n_substituents:
            items.append((f"{nm}imino", True))
        # An unsubstituted =NH alongside oxo would be a plain 'imino' prefix.
        for _ in range(i - len(n_substituents)):
            items.append(('imino', False))
        prefix_block = _compose_io_prefixes(items)
        if prefix_block is None:
            return None
        return _assemble(prefix_block, lam, _apply_hydride_suffix(stem, suffix))

    # o == 0, i >= 1: the imines are the principal characteristic group.
    suffix_mult = _IO_SUFFIX_MULT.get(i)
    if suffix_mult is None:
        return None
    suffix = f"{suffix_mult}imine"
    stem_suffix = _apply_hydride_suffix(stem, suffix)

    both_substituted = bool(s_organyls) and bool(n_substituents)
    if both_substituted:
        # Element locants needed to distinguish hub- from N-substituents
        # (P-68.4.3.3). Priming for >1 substituted imine N is not built here —
        # fail closed (accuracy-first) rather than emit an unverified primed name.
        if len(n_substituents) > 1:
            return None
        from collections import Counter
        loc_by_name: dict = {}
        for nm in s_organyls:
            loc_by_name.setdefault(nm, []).append('S')
        for nm in n_substituents:
            loc_by_name.setdefault(nm, []).append('N')
        parts = []
        for nm in sorted(loc_by_name):
            locs = loc_by_name[nm]
            # italic element locants ordered N < S (alphabetical).
            locs_sorted = sorted(locs)
            mult = _IO_PREFIX_MULT.get(len(locs))
            if mult is None:
                return None
            parts.append(f"{','.join(locs_sorted)}-{mult}{nm}")
        prefix_block = '-'.join(parts)
        return _assemble(prefix_block, lam, stem_suffix)

    # All substituents on one kind of atom (or none): plain alphanumeric prefixes.
    items = [(nm, False) for nm in s_organyls] + [(nm, False) for nm in n_substituents]
    prefix_block = _compose_io_prefixes(items) if items else ''
    if prefix_block is None:
        return None
    return _assemble(prefix_block, lam, stem_suffix)


def name_sulfine(mol) -> Optional[str]:
    """P-64.4.2 / P-74.2.2.1.8 acyclic thiocarbonyl S-oxides (Wave-2 completion C;
    W4-I4): CH3-CH2-CH=S=O -> propylidene-lambda4-sulfanone (BB verbatim). Accepts
    BOTH the hypervalent form ``CH3CH2CH=S=O`` (neutral S, two S=X double bonds)
    AND the equivalent charge-separated dipolar form ``CH3CH2CH=[S+]-[O-]``
    (BB P-74.2.2.1.8, same InChI): the S bears one double bond to an unbranched
    all-carbon chain and one bond (double =O, or single -O with the O carrying the
    negative charge) to a terminal oxygen. Ring S-oxides belong to
    ring_chalcogen_oxide; thials (no O) keep their FG path. Fail-closed; pure."""
    if mol is None:
        return None
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    sulfurs = [a for a in mol.GetAtoms() if a.GetSymbol() == 'S']
    if len(sulfurs) != 1:
        return None
    s = sulfurs[0]
    if s.IsInRing() or s.GetDegree() != 2 or s.GetTotalNumHs() != 0:
        return None
    # The sulfur hub is neutral (hypervalent =O) or +1 (charge-separated dipole).
    if s.GetFormalCharge() not in (0, 1) or s.GetNumRadicalElectrons() != 0:
        return None
    oxo = c = None
    for b in s.GetBonds():
        other = b.GetOtherAtom(s)
        bt = b.GetBondType()
        if other.GetSymbol() == 'O' and other.GetDegree() == 1:
            # Terminal oxide: hypervalent (=O, O charge 0) or dipole (-O-, O charge -1).
            oc = other.GetFormalCharge()
            if bt == Chem.BondType.DOUBLE and oc == 0:
                oxo = other
            elif bt == Chem.BondType.SINGLE and oc == -1:
                oxo = other
            else:
                return None
        elif other.GetSymbol() == 'C' and bt == Chem.BondType.DOUBLE:
            c = other
        else:
            return None
    if oxo is None or c is None:
        return None
    # Only the S/O dipole may carry charge, and it must be net-neutral (S+ + O- = 0);
    # every other atom is neutral and non-radical (fail-closed).
    if s.GetFormalCharge() + oxo.GetFormalCharge() != 0:
        return None
    for atom in mol.GetAtoms():
        if atom.GetIdx() in (s.GetIdx(), oxo.GetIdx()):
            continue
        if atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() != 0:
            return None
    from ..assembly.naming_utils import unbranched_alkylidene_name
    ylidene = unbranched_alkylidene_name(mol, c.GetIdx(), s.GetIdx())
    if ylidene is None:
        return None
    # Full coverage: nothing outside chain + S + O (the chain walker already
    # refused branches/heteroatoms, so the atom count closes the guard).
    if mol.GetNumHeavyAtoms() != 2 + sum(
            1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6):
        return None
    return f"{ylidene}-lambda4-sulfanone"


def name_dinuclear_hydride(mol) -> Optional[str]:
    """Return the substitutive PIN for a two-atom Group-14/Group-15 catenated
    parent hydride (P-69.5.3 two-class-2-metal substitutive), else ``None``
    (fail-closed cascade-continuation).

    A single bond joins EXACTLY one Group-14 atom (Si/Ge/Sn/Pb) and one
    Group-15 atom (As/Sb/Bi), each otherwise saturated with hydrogen (no
    carbon, no halide, no other heavy atom). Per P-41 the Group-15 element is
    senior, so it is the PARENT hydride (arsane/stibane/bismuthane) and the
    Group-14 element is the substituent prefix (silyl/germyl/stannyl/plumbyl)::

        [GeH3][SbH2] -> germylstibane       (Sb senior -> stibane parent)
        [SiH3][AsH2] -> silylarsane
        [PbH3][BiH2] -> plumbylbismuthane

    Every emitted name round-trips through OPSIN 2.9.0 to the input structure.

    SCOPE (fail-closed, accuracy-first): exactly one Group-14 + one Group-15
    hub, single-bonded, H-saturated, neutral, non-radical, single fragment,
    neither hub in a ring. Homo-dinuclear (Si-Si disilane, Sb-Bi), substituted
    (hexamethyldisilane), >2 hubs, any carbon/halide/stray heteroatom, P/N
    parents, ions and rings each fail a guard and cascade onward — zero false
    positives. (P is excluded so phosphane stays with name_phosphine.)

    Pure: no mol mutation, no global state.
    """
    if mol is None:
        return None
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() != 0:
            return None

    # P-41 cls 21 / P-44.1.2: the neutral 2-heteroatom H2P-NH2 parent. N is the
    # senior skeletal class, so the parent is a phosphane skeleton carrying the
    # senior N as the '-amine' terminal -> 'phosphanamine' (a(ba)n preselected,
    # P-21.2.3.1). Guard: exactly one P + one N, single-bonded, both H-saturated
    # (PH2-NH2), neutral, acyclic, no other heavy atom. Fail-closed otherwise.
    heavy = [a for a in mol.GetAtoms() if a.GetSymbol() != 'H']
    if len(heavy) == 2:
        syms = sorted(a.GetSymbol() for a in heavy)
        if syms == ['N', 'P']:
            p = next(a for a in heavy if a.GetSymbol() == 'P')
            n = next(a for a in heavy if a.GetSymbol() == 'N')
            if (not p.IsInRing() and not n.IsInRing()
                    and p.GetTotalNumHs() == 2 and n.GetTotalNumHs() == 2
                    and nonstandard_bonding_number(mol, p.GetIdx()) is None):
                bond = mol.GetBondBetweenAtoms(p.GetIdx(), n.GetIdx())
                if bond is not None and bond.GetBondType() == Chem.BondType.SINGLE:
                    return "phosphanamine"

    # Every heavy atom must be one of the two hub elements (a carbon, halide or
    # stray heteroatom -> not a bare catenated hydride -> fail-closed).
    g14 = []
    g15 = []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym == 'H':
            continue
        if sym in _GROUP14_SUBST_YL:
            g14.append(atom)
        elif sym in _GROUP15_HYDRIDE_PARENT:
            g15.append(atom)
        else:
            return None
    if len(g14) != 1 or len(g15) != 1:
        return None
    sub_atom, parent_atom = g14[0], g15[0]

    # A metallacycle (ring-member hub) is a different class (P-69.4) — decline.
    if sub_atom.IsInRing() or parent_atom.IsInRing():
        return None

    # The two hubs are joined by the catenation bond, which must be single
    # (a double bond would be a -ylidene / -diyl, out of scope). With exactly
    # two heavy atoms in one fragment, this bond is their only connection.
    bond = mol.GetBondBetweenAtoms(sub_atom.GetIdx(), parent_atom.GetIdx())
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return None

    return (f"{_GROUP14_SUBST_YL[sub_atom.GetSymbol()]}"
            f"{_GROUP15_HYDRIDE_PARENT[parent_atom.GetSymbol()]}")


__all__ = ["name_mononuclear_hydride", "name_dinuclear_hydride",
           "name_lambda_sulfane_imine_oxide"]
