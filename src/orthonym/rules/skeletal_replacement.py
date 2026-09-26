"""
Skeletal replacement ("a") nomenclature for chains with embedded heteroatoms.

Implements IUPAC 2013 replacement nomenclature where heteroatoms
embedded in a carbon chain backbone are named using replacement terms
(oxa, aza, thia, etc.) rather than substitutive prefixes (methoxy, amino, etc.).

Examples: the 'a' name is the PIN only with >= 4 heterounits):
    COCCOCCOCCOC -> 2,5,8,11-tetraoxadodecane (4 O; BB:27762)
    OCCOCCOCCOCCOCC -> 3,6,9,12-tetraoxatetradecan-1-ol (4 O + -ol suffix)
    COCCOCCOC -> None (3 O: the substitutive PIN is
                        1-methoxy-2-(2-methoxyethoxy)ethane, BB:27756)
    OCCOCCOCC -> None (2 O: 2-(2-ethoxyethoxy)ethan-1-ol)
    CCNCCC -> None (amine gate; substitutive N-ethylpropan-1-amine)

Scope: Chain-only (acyclic). Rings <= 10 atoms are handled by
Hantzsch-Widman naming in the heterocycles module.

References:
    IUPAC 2013 Blue Book, (Replacement nomenclature)
    IUPAC 2013 Blue Book, (Order of citation of replacement terms)
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from rdkit import Chem

from ..assembly.naming_utils import SIMPLE_MULTIPLIERS
from ..data.chain_names import get_chain_prefix

# a phase : shared / λ-convention. A non-standard-valence
# embedded chain heteroatom cites its bonding number after the locant
# (``...lambda<n>...``); standard valences emit the bare locant (byte-identical).
from .lambda_convention import format_lambda_token, nonstandard_bonding_number

# ============================================================================
# Replacement term table (IUPAC, Table 2.3)
# ============================================================================

REPLACEMENT_TERMS: Dict[str, str] = {
    'O': 'oxa',
    'S': 'thia',
    'Se': 'selena',
    'Te': 'tellura',
    'N': 'aza',
    'P': 'phospha',
    'As': 'arsa',
    'Sb': 'stiba',
    'Bi': 'bisma',
    'Si': 'sila',
    'Ge': 'germa',
    'Sn': 'stanna',
    'Pb': 'plumba',
    'B': 'bora',
}

#: replacement ('a') prefixes are cited in the name in the element
# seniority order of / Table 2.4 — NOT in ascending-locant order.
# BB verbatim: "3-phospha-2,5,7-trisilaoctane" (phospha cited first although
# sila holds the lower locant 2); "8-thia-2,4,6-trisiladecane".
_A_CITATION_ORDER = ['F', 'Cl', 'Br', 'I', 'O', 'S', 'Se', 'Te', 'N', 'P',
                     'As', 'Sb', 'Bi', 'Si', 'Ge', 'Sn', 'Pb', 'B',
                     'Al', 'Ga', 'In', 'Tl']
_A_CITATION_INDEX = {el: i for i, el in enumerate(_A_CITATION_ORDER)}

# /: a heterochain may be TERMINATED by C or one of
# P, As, Sb, Bi, Si, Ge, Sn, Pb, B, Al, Ga, In, Tl (BB example:
# 2-oxa-4-thia-1,5-disilapentane). Al/Ga/In/Tl carry no entry in
# REPLACEMENT_TERMS, so they fail closed at the terminator check.
_ALLOWED_HETERO_TERMINATORS = {'P', 'As', 'Sb', 'Bi',
                               'Si', 'Ge', 'Sn', 'Pb', 'B'}


# heterounits (the Blue Book): "A heterounit is a set of
# heteroatoms having a name of its own such as, -SS-, disulfanediyl;
# -SiH2-O-SiH2-, disiloxane-1,3-diyl; -SOS-, dithioxanediyl (not -OSiH2O- nor
# -OSO- that correspond to three consecutive units 'oxysilanediyloxy' and
# 'oxysulfanediyloxy', respectively)." A same-element pair (-OO-, -SS-, -NN-,
# -SiSi-,...) is one unit; the two named triads are one unit each.
_NAMED_HETEROUNIT_TRIADS = {('Si', 'O', 'Si'), ('S', 'O', 'S')}


def _heterounits_in_run(run: List[str]) -> Optional[int]:
    """Heterounits in one maximal run of consecutive chain heteroatoms, or None
    when the run is not a countable sequence of units (fail closed)."""
    if len(run) == 1:
        return 1
    if len(run) == 2:
        # -XX- has a name of its own (disulfanediyl, dioxidanediyl, diazanediyl,
        # disilanediyl); two different atoms are two units (2-oxa-4-thia-1,5-
        # disilapentane (PIN),:23436: Si-O is silyl + oxy).
        return 1 if run[0] == run[1] else 2
    if len(run) == 3:
        if tuple(run) in _NAMED_HETEROUNIT_TRIADS:
            return 1
        if len(set(run)) == 1:
            # "trisulfane, HS-S-SH, is a parent hydride and is not allowed to be
            # a heterounit" (:23385): no 'a' name for this chain.
            return None
        if run[0] == run[2]:
            return 3  # -OSiH2O-, -OSO-: "three consecutive units" (:23350)
    return None


def _count_chain_heterounits(mol, backbone: List[int]) -> Optional[int]:
    """Heterounits in the unbranched chain ``backbone``, or None
    when a heteroatom run cannot be counted. 0 for an all-carbon chain."""
    syms = [mol.GetAtomWithIdx(i).GetSymbol() for i in backbone]
    units = 0
    run: List[str] = []
    for sym in syms + ['C']:
        if sym != 'C':
            run.append(sym)
            continue
        if run:
            n = _heterounits_in_run(run)
            if n is None:
                return None
            units += n
            run = []
    return units


def _general_tier_active() -> bool:
    """True on the valid / complete / best-effort tiers (the published
    ``general_fallback`` flag), False on the PIN tier. A tier read never breaks
    naming."""
    try:
        from ..metrics.provenance import general_fallback_ctx
        return bool(general_fallback_ctx.get())
    except Exception:  # noqa: BLE001
        return False


def _qualifies_for_pin_skeletal_replacement(
    backbone: List[int], mol,
) -> Tuple[bool, str]:
    """: is the skeletal replacement ('a') name the PIN of this chain?

    Blue Book 2013, ``## **** Skeletal replacement ('a') nomenclature
    in acyclic chains``, (the Blue Book): "Skeletal
    replacement ('a') nomenclature rather than substitutive or multiplicative
    names must be used to generate preferred IUPAC names for acyclic
    structures when four or more heterounits are present in a unbranched chain
    containing at least one carbon atom and when none of the heteroatoms
    constitute all or part of the principal characteristic group of the
    compound." The Blue Book's own boundary rows: "(1) 1-methoxy-2-(2-
    methoxyethoxy)ethane (PIN)" for three ether O (:27756) against "(4)
    2,5,8,11-tetraoxadodecane (PIN)" for four (:27762); "13-amino-N-(2-{[2-
    ({2-[(2-aminoethyl)amino]ethyl}amino)ethyl]amino}ethyl)-2,5,8,11-
    tetraazatridecanamide (PIN... since only three heteroatoms are present in
    the N-substituent group, it must be named substitutively)" (:23419-23423); "[not
    3,4-dioxa-1,2,6,7-tetrasilaheptane; four hetero units are required...]"
    (:6365). Below four heterounits the substitutive namers give the PIN.

    (The labels this function used to return -- '>=3-mixed-kind',
    'single-hetero-long-chain', 'two-hetero-substitutive-equivalent' -- had no
    Blue Book basis and shipped '2,5,8-trioxanonane' / '3,6-dioxaoctan-1-ol' /
    '4-thiaheptane' at the PIN tier.)

    Returns ``(True, ">=4-heterounits")`` or ``(False, reason)`` with reason in
    {"no-heteroatoms", "fewer-than-4-heterounits", "no-carbon",
    "heterounits-not-countable"}.
    """
    syms = [mol.GetAtomWithIdx(i).GetSymbol() for i in backbone]
    if 'C' not in syms:
        return (False, "no-carbon")
    units = _count_chain_heterounits(mol, backbone)
    if units is None:
        return (False, "heterounits-not-countable")
    if units == 0:
        return (False, "no-heteroatoms")
    if units >= 4:
        return (True, ">=4-heterounits")
    return (False, "fewer-than-4-heterounits")


# IUPAC: Order of citation for replacement terms
# When different heteroatom groups have the same lowest locant,
# alphabetical order of the replacement term breaks the tie.
# The seniority order from the IUPAC table (high to low):
# O > S > Se > Te > N > P > As > Si > Ge > Sn > Pb > B
# But citation order in the name is by ascending locant, then alphabetical.

# Functional groups that take priority over replacement naming.
# If ANY of these SMARTS match, do NOT use skeletal replacement.
_PRIORITY_FG_SMARTS = [
    '[CX3](=O)[OX2H1]',    # Carboxylic acid
    '[CX3](=O)[OX1-]',     # Carboxylate
    '[CX3H1](=O)',          # Aldehyde
    '[CX3](=O)[#6]',        # Ketone (C=O bonded to two carbons)
    '[CX3](=O)[OX2][#6]',  # Ester
    '[CX3](=O)[NX3]',      # Amide
    '[CX3](=O)[FX1,ClX1,BrX1,IX1]',  # Acid halide
    '[C]#[N]',             # Nitrile
    '[NX3][CX3](=[NX1])',  # Amidine
    '[SX2H]',              # Thiol
    '[NX2]=[CX2]=[OX1]',  # Isocyanate (N=C=O)
    '[NX2]=[CX2]=[SX1]',  # Isothiocyanate (N=C=S)
]

#: trivalent N bonded only to carbons → substitutive naming preferred
# over skeletal ("aza") replacement for ACYCLIC carbon-chain amines.
# Applied only in the acyclic path (Gate 2c); cyclic large-ring aza-replacement
# is governed by and must not be blocked here.
# The !$([NX3]~[!#6]) exclusion ensures N–N bonds (polyazane) and N–O/N–S bonds
# (hydroxylamine, sulfonamide) are NOT blocked; aromatic N (!a) also excluded.
_ACYCLIC_AMINE_SMARTS = '[NX3;!a;!$([NX3]~[!#6])]'
_ACYCLIC_AMINE_PATTERN = Chem.MolFromSmarts(_ACYCLIC_AMINE_SMARTS)

# Pre-compile the SMARTS patterns
_PRIORITY_FG_PATTERNS = []
for sma in _PRIORITY_FG_SMARTS:
    pat = Chem.MolFromSmarts(sma)
    if pat is not None:
        _PRIORITY_FG_PATTERNS.append(pat)


#: carboxylic-acid suffix integration on a fixed-numbered heterochain
# (3,6,9,12-tetraoxatetradecanedioic acid; 3,6,9,12-tetraoxapentadecan-15-oic
# acid, — the acid carbon takes locant 15 because the heteroatoms own
# the numbering,.
_ACID_PATTERN = Chem.MolFromSmarts('[CX3](=[OX1])[OX2H1]')


def _detect_terminal_acid_groups(mol: Chem.Mol) -> Optional[List[Dict]]:
    """Detect -C(=O)OH groups eligible for suffix integration.

    Returns ```` when the molecule carries no carboxyl at all (classic path),
    a list of ``{'c', 'oxo', 'oh'}`` dicts for 1-2 clean chain-terminal acid
    carbons, or ``None`` when a carboxyl exists but is not of that clean shape
    (geminal diacid carbon, formic-type, charged, >2 acids) — the caller fails
    closed exactly as Gate 2 always did for acids.
    """
    matches = mol.GetSubstructMatches(_ACID_PATTERN)
    if not matches:
        return []
    groups: List[Dict] = []
    used_c = set()
    for c, oxo, oh in matches:
        if c in used_c:
            return None  # two carboxyls on one carbon (carbonic-type)
        used_c.add(c)
        c_atom = mol.GetAtomWithIdx(c)
        if c_atom.GetFormalCharge() != 0:
            return None
        chain_nbrs = [n for n in c_atom.GetNeighbors()
                      if n.GetIdx() not in (oxo, oh)]
        if len(chain_nbrs) != 1:
            return None  # formic acid / exotic substitution on the acid carbon
        for o_idx in (oxo, oh):
            o_atom = mol.GetAtomWithIdx(o_idx)
            if o_atom.GetFormalCharge() != 0 or o_atom.GetDegree() != 1:
                return None
        groups.append({'c': c, 'oxo': oxo, 'oh': oh})
    if len(groups) > 2:
        return None
    return groups


def _strict_heterounit_chain_ok(mol: Chem.Mol, backbone: List[int]) -> bool:
    """ strict qualification for the NEW classes (suffix-bearing,
    heteroatom-terminated, or substituted heterochains): the replacement name
    is the PIN only when FOUR OR MORE heterounits sit in the unbranched chain
    together with at least one carbon.

    Conservative unit accounting: every isolated heteroatom is one unit; a
    same-element adjacent pair (–SS–/–OO–/–NN–) is a catenated-hydride unit
    class handled elsewhere (peroxide/disulfide/polyazane) → fail closed; a
    heteroatom run of 3+ is a parent hydride in its own right (BB: trisulfane
    "is not allowed to be a heterounit") → fail closed. Mixed-element adjacent
    pairs (Si-O, S-Si — 2-oxa-4-thia-1,5-disilapentane) each count singly.
    """
    syms = [mol.GetAtomWithIdx(i).GetSymbol() for i in backbone]
    if 'C' not in syms:
        return False
    hetero_count = 0
    run_len = 0
    prev_sym = None
    for idx, sym in zip(backbone, syms):
        if sym == 'C':
            run_len = 0
            prev_sym = None
            continue
        if sym not in REPLACEMENT_TERMS:
            return False
        if mol.GetAtomWithIdx(idx).GetFormalCharge() != 0:
            return False
        hetero_count += 1
        run_len += 1
        if run_len >= 3:
            return False
        if prev_sym == sym:
            return False
        prev_sym = sym
    return hetero_count >= 4


def _collect_simple_alkyl_substituents(
    mol: Chem.Mol, backbone_set: set, allowed_extra: set,
) -> Optional[List[Tuple[int, str]]]:
    """Group every off-backbone heavy atom into substituent branches and name
    each one, or return None (fail-closed) on anything but a simple unbranched
    saturated all-carbon alkyl bonded to exactly one backbone atom.

    : substituent locants follow the FIXED heterochain numbering
    (5,5-dimethyl-2,5λ4,8,11-tetrathiadodecane). Returns a list of
    ``(backbone_atom_idx, prefix_name)`` — locants are assigned by the caller
    after orientation.
    """
    num_atoms = mol.GetNumAtoms()
    off = [i for i in range(num_atoms)
           if i not in backbone_set and i not in allowed_extra]
    if not off:
        return []
    off_set = set(off)
    seen: set = set()
    subs: List[Tuple[int, str]] = []
    for start in off:
        if start in seen:
            continue
        comp = [start]
        seen.add(start)
        qi = 0
        while qi < len(comp):
            a = comp[qi]
            qi += 1
            for nb in mol.GetAtomWithIdx(a).GetNeighbors():
                j = nb.GetIdx()
                if j in off_set and j not in seen:
                    seen.add(j)
                    comp.append(j)
        attach = set()
        for a in comp:
            for nb in mol.GetAtomWithIdx(a).GetNeighbors():
                j = nb.GetIdx()
                if j in backbone_set:
                    attach.add((j, a))
                elif j in allowed_extra:
                    return None  # branch touching a suffix oxygen — malformed
        if len(attach) != 1:
            return None  # multiply-attached / detached branch
        b_idx, root = next(iter(attach))
        comp_set = set(comp)
        for a in comp:
            atom = mol.GetAtomWithIdx(a)
            if (atom.GetAtomicNum() != 6 or atom.GetFormalCharge() != 0
                    or atom.GetNumRadicalElectrons() != 0):
                return None
            for b in atom.GetBonds():
                if b.GetBondType() != Chem.BondType.SINGLE:
                    return None
        # Unbranched: a single walk from the attachment root must cover it all.
        prev, cur, length = None, root, 1
        while True:
            nxts = [nb.GetIdx() for nb in mol.GetAtomWithIdx(cur).GetNeighbors()
                    if nb.GetIdx() in comp_set and nb.GetIdx() != prev]
            if len(nxts) > 1:
                return None
            if not nxts:
                break
            prev, cur = cur, nxts[0]
            length += 1
        if length != len(comp):
            return None
        prefix = get_chain_prefix(length)
        if not prefix:
            return None
        subs.append((b_idx, f'{prefix}yl'))
    return subs


def _format_substituent_prefix(placed: List[Tuple[int, str]]) -> str:
    """``[(5, 'methyl'), (5, 'methyl')]`` -> ``'5,5-dimethyl'``; distinct
    prefixes are alphabetised (multiplying prefixes ignored,."""
    by_name: Dict[str, List[int]] = defaultdict(list)
    for loc, name in placed:
        by_name[name].append(loc)
    parts = []
    for name in sorted(by_name):
        locs = sorted(by_name[name])
        mult = SIMPLE_MULTIPLIERS.get(len(locs), '') if len(locs) > 1 else ''
        parts.append(f"{','.join(str(l) for l in locs)}-{mult}{name}")
    return '-'.join(parts)


def try_skeletal_replacement_name(mol: Chem.Mol) -> Optional[str]:
    """Try to name a molecule using skeletal replacement nomenclature.

    Returns an IUPAC replacement name if the molecule is an acyclic chain
    with embedded heteroatoms suitable for replacement naming. Returns None
    if the molecule does not qualify (cyclic, has priority functional groups,
    too few heteroatoms, etc.).

    Supports terminal alcohol (-OH) suffix integration:
        OCCOCCOCC -> 3,6-dioxaoctan-1-ol

    Args:
        mol: RDKit molecule object (already parsed from SMILES).

    Returns:
        Replacement name string (e.g., '2,5-dioxahexane') or None.
    """
    if mol is None:
        return None

    # ----------------------------------------------------------------
    # Gate 1: Rings gate (IUPAC /
    # Chain replacement requires no rings, EXCEPT for large heterocyclic
    # rings (>= 7 members with heteroatoms) where skeletal replacement
    # naming may be simpler than substitutive naming.
    # Small rings (<= 6 members) are handled by Hantzsch-Widman naming.
    # ----------------------------------------------------------------
    ring_info = mol.GetRingInfo()
    if ring_info.NumRings() > 0:
        # Check if ALL rings are large (>= 7) and contain heteroatoms
        all_rings_large_hetero = True
        for ring in ring_info.AtomRings():
            if len(ring) < 7:
                all_rings_large_hetero = False
                break
            # Check ring has at least one heteroatom
            has_hetero = any(
                mol.GetAtomWithIdx(idx).GetSymbol() in REPLACEMENT_TERMS
                for idx in ring
            )
            if not has_hetero:
                all_rings_large_hetero = False
                break
        if not all_rings_large_hetero:
            return None
        # "Heteromonocyclic hydrides named by skeletal replacement
        # ('a') nomenclature" opens by fixing this boundary, and it is RING
        # SIZE ALONE -- saturation plays no part:
        #
        # "Mancude and saturated heteromonocyclic compounds with up to and
        # including ten ring members are named by the extended
        # Hantzsch-Widman system (see. For monocyclic rings with
        # eleven and more ring members, skeletal replacement ('a')
        # nomenclature (see is used for the fully saturated or
        # fully unsaturated compounds ([n]annulenes)."
        #
        # Confirmed independently by ("...for heteromonocyclic
        # compounds having more than ten ring atoms") and by
        # ("Preferred IUPAC names for heteromonocyclic rings with no more than
        # ten ring members are Hantzsch-Widman names").
        #
        # So every single heteromonocycle of ten atoms or fewer goes back to
        # the HW namer, which owns both the mancude parent (1H-azepine,
        # azocine, 2H-oxocine) and the partially saturated forms, expressed as
        # hydro prefixes on that parent per (2,3-dihydro-1H-azepine,
        # 4,5,6,7-tetrahydro-1,4-thiazepine). Only rings of ELEVEN or more
        # fall through to cyclic replacement.
        #
        # This test used to also require every ring bond to be single, which
        # let unsaturated 7- and 8-membered rings reach replacement naming and
        # emit non-PIN forms such as "1-azacyclohepta-2,4,6-triene" for
        # 1H-azepine. Unsaturated 9- and 10-rings escaped only by accident,
        # via the separate has_aromatic gate in _try_cyclic_replacement_name.
        #
        # Redirecting the 7-10 rings here left a residue of producer-level
        # abstentions. puts lambda rings on the Hantzsch-Widman side
        # of this same boundary ("1H-1<lambda>4-thiepine (PIN)" at:9496,
        # against the 14-membered "1-oxa-4<lambda>4-thiacyclotetradecane (PIN)"
        # at:9486, which is replacement), so the replacement names those rings
        # used to get were not PINs and withdrawing them was right.
        #
        # ⚠ The figures once recorded here -- "82 abstain, split into two
        # unbuilt gaps of 78 + 4" -- were WRONG, and wrong in a way worth
        # keeping visible. They came from an enumeration of 1,274 bare
        # heteromonocycles; the real enumeration is 27,687 (sizes 3-14 x
        # N/O/S/O+N/S+N/N+N at EVERY heteroatom position x EVERY independent
        # edge set of the ring), a ~22x superset. Measured against that:
        #
        # * the residue was 707 abstentions, not 82, and it spanned sizes
        # 3-10 (2/4/13/24/53/96/185/330) -- so it was never a boundary
        # effect at all. This edit only made sizes 7-10 VISIBLE; sizes 3-6
        # were abstaining before it and were untouched by it;
        # * there was no "4-row second gap". Those 9-membered 2-N rings were
        # never producer abstentions -- they EMITTED a malformed name that
        # only the OPSIN validity gate suppressed downstream. Counting a
        # gate-suppressed emission as a producer abstention merged a
        # coverage gap with a correctness bug. Fixed separately in
        # _aromatizable_hydro_name.
        #
        # All 707 are now named: the mancude lambda parents whose indicated
        # hydrogen sits on a CARBON (3H-1<lambda>4-thiophene (PIN),:9171) and
        # the genuine hydro forms (3,4,5,6-tetrahydro-1<lambda>4,2-thiazin-1-ol
        # (PIN),:33292). See _name_lambda_heteromonocycle.
        #
        # Real-corpus impact of the redirect itself was measured as zero at the
        # time (0 router diffs over pubchem_2000 + chebi_5000) -- inherited,
        # not re-run here.
        if len(ring_info.AtomRings()) == 1:
            if len(ring_info.AtomRings()[0]) <= 10:
                return None
        # For large heterocyclic rings (>= 11, or unsaturated/multi-heteroatom
        # 7-10-rings), try cyclic replacement naming per IUPAC.
        # Keep the "no priority FGs" gate.
        for pat in _PRIORITY_FG_PATTERNS:
            if mol.HasSubstructMatch(pat):
                return None
        return _try_cyclic_replacement_name(mol, ring_info)

    # ----------------------------------------------------------------
    # acid-suffix integration: a clean chain-terminal -C(=O)OH
    # (1 or 2 of them) is expressed as the -oic/-dioic acid suffix on the
    # fixed-numbered heterochain instead of tripping Gate 2.  = no acid
    # (classic path, byte-identical); None = malformed acid (fail closed,
    # exactly what Gate 2 did for every acid before this branch existed).
    # ----------------------------------------------------------------
    acid_groups = _detect_terminal_acid_groups(mol)
    if acid_groups is None:
        return None
    acid_mode = bool(acid_groups)
    acid_atoms: set = set()
    acid_oxygens: set = set()
    for _g in acid_groups:
        acid_atoms.update((_g['c'], _g['oxo'], _g['oh']))
        acid_oxygens.update((_g['oxo'], _g['oh']))

    # ----------------------------------------------------------------
    # Gate 2: No priority functional groups. In acid mode the acid's own
    # carbonyl matches the acid/ketone/ester/aldehyde patterns — matches
    # confined to the acid groups are the suffix itself; any match touching
    # atoms OUTSIDE them is a genuine second FG and still rejects.
    # ----------------------------------------------------------------
    for pat in _PRIORITY_FG_PATTERNS:
        if not acid_mode:
            if mol.HasSubstructMatch(pat):
                return None
        else:
            for match in mol.GetSubstructMatches(pat):
                if acid_atoms.isdisjoint(match):
                    return None

    # ----------------------------------------------------------------
    # Gate 2c: acyclic carbon-chain amines use substitutive naming.
    # If the molecule contains a trivalent N bonded ONLY to carbons (secondary
    # or tertiary amine on an all-carbon backbone), skeletal ("aza") replacement
    # is NOT the PIN — the substitutive handler produces N-alkyl-alkan-1-amine.
    # Applies to the ACYCLIC path only (cyclic large-ring aza-replacement is
    # governed by and is checked separately above via
    # _try_cyclic_replacement_name, which is never reached by this gate).
    # N–N bonds (polyazane: NNN → triazane), N–O (hydroxylamine), N–S
    # (sulfonamide) and aromatic N are NOT blocked (they carry !$([NX3]~[!#6])
    # and !a guards in the pattern).
    # ----------------------------------------------------------------
    if (_ACYCLIC_AMINE_PATTERN is not None
            and mol.HasSubstructMatch(_ACYCLIC_AMINE_PATTERN)):
        return None

    # Gate 2c-bis (W3-P15,: an N,O-disubstituted hydroxylamine
    # (R-NH-O-R') is an O-substituted AMINE; the BB note  EXPLICITLY
    # forbids skeletal ('a') replacement for it (an 'a' chain cannot terminate
    # on oxygen and the amine characteristic group would be lost). Decline so the
    # substitutive amine namer (handlers.hydroxylamine._name_no_disub_hydroxylamine)
    # produces the PIN (CNOC -> N-methoxymethanamine) instead of '2-oxa-3-azabutane'.
    from ..assembly.handlers.hydroxylamine import no_disub_hydroxylamine_core
    if no_disub_hydroxylamine_core(mol) is not None:
        return None

    # ----------------------------------------------------------------
    # Gate 2b (functional-group perception fix/, a phase): no prefix-only characteristic-group
    # atoms. Azide / diazo / nitroso / nitrite / nitro / N-oxide heteroatoms are
    # characteristic groups / /, NOT chain skeletal atoms —
    # skeletal replacement must not walk them into an aza/oxa chain (e.g.
    # CN=[N+]=[N-] -> wrong '2,3-diazabutane'; should be 'azidomethane' via the
    # substitutive azido prefix). This is the STRUCTURAL gate (internal notes):
    # derived from perception's own FG matches, NOT an extension of the per-FG
    # _PRIORITY_FG_SMARTS blocklist above.
    # ----------------------------------------------------------------
    from ..perception.functional_groups import get_chain_excluded_atoms
    if get_chain_excluded_atoms(mol):
        return None

    # ----------------------------------------------------------------
    # Gate 3: Check terminal functional groups
    # Terminal OH is allowed (suffix integration). Other terminal FGs
    # (NH2, SH) cause fallback to substitutive naming.
    # ----------------------------------------------------------------
    # In acid mode the -oic suffix owns the chain ends: -ol integration is
    # disabled (a coexisting hydroxyl leaves an unaccounted oxygen and fails
    # the atom-coverage gate below — never a lossy name).
    terminal_oh_info = None if acid_mode else _detect_terminal_oh(mol)
    has_other_terminal_fg = _has_terminal_functional_group(mol, exclude_oh=True)

    if has_other_terminal_fg:
        return None

    # If terminal OH detected, check that it's only OH (no NH2/SH combo)
    # and that there are still enough embedded heteroatoms for replacement naming
    has_terminal_oh = terminal_oh_info is not None

    # ----------------------------------------------------------------
    # Find the longest chain backbone including heteroatoms. The acid oxygens
    # are suffix atoms, not skeletal atoms — excise them so the acid CARBON
    # terminates the chain.
    # ----------------------------------------------------------------
    backbone = _find_replacement_chain(mol, exclude_atoms=acid_oxygens)
    if backbone is None:
        return None

    # ----------------------------------------------------------------
    # For terminal OH: strip the OH oxygen from the backbone.
    # The terminal O-H is NOT a chain atom -- it's a functional suffix.
    # The chain consists only of C and embedded heteroatoms.
    # OCCOCCOCC backbone: O-C-C-O-C-C-O-C-C -> strip terminal O
    # -> chain = C-C-O-C-C-O-C-C (8 atoms = octane)
    # ----------------------------------------------------------------
    if has_terminal_oh:
        oh_idx = terminal_oh_info['oh_idx']
        if backbone[0] == oh_idx:
            backbone = backbone[1:]
        elif backbone[-1] == oh_idx:
            backbone = backbone[:-1]
        # After stripping, verify backbone is still valid
        if backbone is None or len(backbone) < 3:
            return None

    # ----------------------------------------------------------------
    # Acid carbons must be the chain terminals (a mid-chain carboxyl would be
    # a -carboxylic acid on a branch point, a different class — fail closed).
    # ----------------------------------------------------------------
    acid_carbons = {g['c'] for g in acid_groups}
    if acid_mode:
        ends = {backbone[0], backbone[-1]}
        if not acid_carbons <= ends:
            return None
        if len(acid_groups) == 2 and ends != acid_carbons:
            return None

    # ----------------------------------------------------------------
    # Gate 3b / terminator rule): the chain must be
    # terminated by C or by P/As/Sb/Bi/Si/Ge/Sn/Pb/B (BB verbatim example:
    # 2-oxa-4-thia-1,5-disilapentane). An allowed heteroatom terminator is
    # admitted ONLY into the strict >=4-heterounit class validated below —
    # everything else (terminal F/Cl, λ-bearing terminals, Al/Ga/In/Tl with
    # no 'a' term, O/S/N terminals) fails closed rather than being silently
    # counted as a carbon of the alkane stem (the old structure-loss hazard:
    # trisiloxane [SiH3]O[SiH2]O[SiH3] -> '2,4-dioxa-3-silapentane' RTs to
    # dimethoxysilane, a DIFFERENT molecule — carbon-less chains still exit
    # here to the catenated-hydride namer).
    # ----------------------------------------------------------------
    hetero_terminal_mode = False
    for end_pos in (0, len(backbone) - 1):
        end_sym = mol.GetAtomWithIdx(backbone[end_pos]).GetSymbol()
        if end_sym == 'C':
            continue
        if end_sym not in _ALLOWED_HETERO_TERMINATORS:
            return None
        if end_sym not in REPLACEMENT_TERMS:
            return None
        if acid_mode or has_terminal_oh:
            return None
        # λ-bearing terminal parents territory) are not built here.
        if nonstandard_bonding_number(mol, backbone[end_pos]) is not None:
            return None
        hetero_terminal_mode = True

    # ----------------------------------------------------------------
    # Gate 4: atom coverage. Every heavy atom must be a backbone atom, a
    # suffix oxygen (-ol / -oic acid), or part of a SIMPLE unbranched alkyl
    # substituent named as a prefix on the fixed numbering:
    # 5,5-dimethyl-2,5λ4,8,11-tetrathiadodecane). Anything else fails closed.
    # ----------------------------------------------------------------
    backbone_set = set(backbone)
    suffix_extra = set(acid_oxygens)
    if has_terminal_oh:
        suffix_extra.add(terminal_oh_info['oh_idx'])
    substituents: List[Tuple[int, str]] = []
    if len(backbone_set) + len(suffix_extra) != mol.GetNumAtoms():
        substituents = _collect_simple_alkyl_substituents(
            mol, backbone_set, suffix_extra)
        if substituents is None:
            return None
        if has_terminal_oh:
            return None  # substituted -ol heterochains not built — fail closed
    substituted_mode = bool(substituents)

    # ----------------------------------------------------------------
    # Gate 5: Check heteroatom count and chain length thresholds
    # ----------------------------------------------------------------
    embedded_heteroatoms = []
    for i, atom_idx in enumerate(backbone):
        atom = mol.GetAtomWithIdx(atom_idx)
        symbol = atom.GetSymbol()
        if symbol in REPLACEMENT_TERMS and i > 0 and i < len(backbone) - 1:
            embedded_heteroatoms.append((i, symbol))

    if len(embedded_heteroatoms) == 0 and not hetero_terminal_mode:
        return None

    # ----------------------------------------------------------------
    # PIN qualification, (the Blue Book): the 'a' name is the
    # PIN only with FOUR OR MORE heterounits in the unbranched chain, at least
    # one chain carbon, and no heteroatom in the principal characteristic
    # group; below that the substitutive namers own the PIN ('1-methoxy-2-(2-
    # methoxyethoxy)ethane (PIN)',:27756; '1,2-dimethoxyethane (PIN)',:27754).
    # This one gate replaces the former R4 (1-2 ether O) and DD5 (2 mixed
    # chalcogens) carve-outs, which were narrower cases of the same rule.
    # * special modes (acid suffix, heteroatom-terminated, alkyl-substituted):
    # the conservative _strict_heterounit_chain_ok (same-element pairs fail
    # closed there);
    # * the classic path (-ol suffix or none): _qualifies_for_pin_skeletal_
    # replacement, which counts heterounits as:23350 defines them.
    # ----------------------------------------------------------------
    _special_mode = acid_mode or hetero_terminal_mode or substituted_mode
    if _special_mode and not _strict_heterounit_chain_ok(mol, backbone):
        return None
    if not _special_mode:
        qualifies, _rationale = _qualifies_for_pin_skeletal_replacement(
            backbone, mol)
        if not qualifies:
            # General-tier fallback, never the PIN: a chain whose heteroatom has a
            # non-standard bonding number ('9λ2-stannaheptadecane' for
            # CCCCCCCC[Sn]CCCCCCCC) has a substitutive PIN on a lambda parent
            # hydride ("SnH2 λ2-stannane (preselected name, see ",
            # the Blue Book: 'dioctyl-λ2-stannane'), which this engine does
            # not build (it refuses the tin compound). So on the general tiers
            # only, the RT-exact 'a' name is kept and marked general
            # (record_general_ring_prefix -> not is_pin); the PIN tier declines.
            if not (_rationale == "fewer-than-4-heterounits"
                    and _general_tier_active()
                    and any(nonstandard_bonding_number(mol, i) is not None
                            for i in backbone
                            if mol.GetAtomWithIdx(i).GetSymbol() != 'C')):
                return None
            from ..metrics.provenance import record_general_ring_prefix
            record_general_ring_prefix()

    # ----------------------------------------------------------------
    # Number the chain: for -ol suffix, the OH end gets locant 1.
    # For plain replacement chains, give lowest locants to heteroatoms.
    # ----------------------------------------------------------------
    if has_terminal_oh:
        backbone = _orient_oh_end_first(
            backbone, mol, terminal_oh_info['carbon_idx']
        )
    else:
        backbone = _orient_for_lowest_locants(
            backbone, mol,
            suffix_atoms=acid_carbons if acid_mode else None,
            substituent_atoms=(
                {b for b, _ in substituents} if substituents else None),
        )

    # ----------------------------------------------------------------
    #: double/triple bonds get locants per the FIXED heterochain
    # numbering (2,4,6,8-tetrasiladec-9-ene). The scan fails closed on any
    # multiple bond that is not a clean C=C / C#C between consecutive backbone
    # atoms — WITHOUT it the builder emitted the SATURATED stem for an
    # unsaturated chain (structure loss, caught only by the downstream RT
    # gate). Suffix + ene integration interplay) is not built
    # yet — fail closed rather than guess the composite numbering.
    # ----------------------------------------------------------------
    unsat = _backbone_unsaturation(mol, backbone, ignore_atoms=acid_oxygens)
    if unsat is None:
        return None
    ene_locants, yne_locants = unsat
    if (ene_locants or yne_locants) and (has_terminal_oh or acid_mode):
        return None

    # Rebuild heteroatom positions after reorientation. A non-standard-valence
    # embedded heteroatom carries the λ-convention; standard valences
    # (every ordinary oxa/aza/thia chain) record no λ -> byte-identical output.
    # Terminal positions are included: whitelisted heteroatom terminators
    # (Gate 3b) take their own 'a' prefix ("1,5-disila..."), and no classic
    # path can reach here with a heteroatom terminal.
    heteroatom_positions = []
    lambda_by_locant: Dict[int, int] = {}
    for i, atom_idx in enumerate(backbone):
        atom = mol.GetAtomWithIdx(atom_idx)
        symbol = atom.GetSymbol()
        if symbol in REPLACEMENT_TERMS:
            # Locants are 1-based
            locant = i + 1
            heteroatom_positions.append((locant, symbol))
            lam = nonstandard_bonding_number(mol, atom_idx)
            if lam is not None:
                lambda_by_locant[locant] = lam

    if not heteroatom_positions:
        return None

    # ----------------------------------------------------------------
    # Determine the suffix: terminal -ol, or the acid forms on
    # the fixed heterochain numbering (the acid carbon can hold the HIGH
    # locant — 3,6,9,12-tetraoxapentadecan-15-oic acid,.
    # ----------------------------------------------------------------
    suffix = None
    if has_terminal_oh:
        # The OH-bearing carbon should be at position 0 (locant 1) after orient
        suffix = ('ol', 1)
    elif acid_mode:
        if len(acid_groups) == 2:
            suffix = ('dioic acid', None)
        else:
            pos_by_idx = {idx: i for i, idx in enumerate(backbone)}
            suffix = ('oic acid',
                      pos_by_idx[next(iter(acid_carbons))] + 1)

    # ----------------------------------------------------------------
    # Build the replacement name (+ substituent prefixes on the fixed
    # numbering,
    # ----------------------------------------------------------------
    name = _build_replacement_name(
        len(backbone), heteroatom_positions, suffix=suffix,
        lambda_by_locant=lambda_by_locant,
        ene_locants=ene_locants, yne_locants=yne_locants,
    )
    if name is not None and substituents:
        pos_by_idx = {idx: i for i, idx in enumerate(backbone)}
        placed = [(pos_by_idx[b] + 1, s) for b, s in substituents]
        name = f'{_format_substituent_prefix(placed)}-{name}'
    return name


def _detect_terminal_oh(mol: Chem.Mol) -> Optional[Dict]:
    """Detect a terminal alcohol (-OH) suitable for replacement name suffix.

    A terminal OH is an oxygen with 1 H, bonded to exactly 1 heavy neighbor
    (a carbon), where that carbon is at the end of the chain (degree <= 2
    in the heavy-atom graph, meaning it has at most one other heavy neighbor).

    Args:
        mol: RDKit molecule object.

    Returns:
        Dict with 'oh_idx' (oxygen atom index) and 'carbon_idx' (bearing
        carbon index), or None if no suitable terminal OH found.
    """
    terminal_ohs = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != 'O':
            continue
        if atom.GetTotalNumHs() < 1:
            continue
        heavy_neighbors = [n for n in atom.GetNeighbors() if n.GetSymbol() != 'H']
        if len(heavy_neighbors) != 1:
            continue
        carbon = heavy_neighbors[0]
        if carbon.GetSymbol() != 'C':
            continue
        # Check that the carbon is a chain terminal (degree 1 or 2 in heavy graph)
        carbon_heavy_nbrs = [n for n in carbon.GetNeighbors() if n.GetSymbol() != 'H']
        # The carbon should have at most 2 heavy neighbors: the OH oxygen + one chain atom
        if len(carbon_heavy_nbrs) <= 2:
            terminal_ohs.append({
                'oh_idx': atom.GetIdx(),
                'carbon_idx': carbon.GetIdx(),
            })

    # Only support single terminal OH for now
    if len(terminal_ohs) == 1:
        return terminal_ohs[0]

    return None


# a phase.A: terminal -amine / -thiol support DEFERRED to.
# internal notes-A.md corpus tally:
# acyclic-terminal-amine candidates: 275 mined, 2 eligible (no priority FG)
# acyclic-terminal-thiol candidates: 25 mined, 0 eligible (no priority FG)
# Threshold per internal notes is 5 corpus compounds per FG; both below
# threshold => follow-ups -D05-amine / -D05-thiol.
# Effective true-positive count is 0 cpd benefit because the 2 amine
# candidates also carry phosphate priority FGs that gate-2 already rejects;
# extending gate-3 with `_detect_terminal_amine` / `_detect_terminal_thiol`
# does not unblock any RT failures.
# Source: 154-internal notes; internal notes-A.md


def _has_terminal_functional_group(
    mol: Chem.Mol, exclude_oh: bool = False
) -> bool:
    """Check if molecule has terminal functional groups (OH, NH2, SH, etc.).

    Terminal means an atom at degree 1 (or H-bearing heteroatom at chain end)
    that would normally take a functional group suffix.

    Args:
        mol: RDKit molecule object.
        exclude_oh: If True, ignore terminal OH groups (for suffix integration).

    Returns:
        True if terminal functional groups are present.
    """
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        # Skip carbons and hydrogens
        if symbol in ('C', 'H'):
            continue

        if symbol not in REPLACEMENT_TERMS:
            continue

        # Check if this heteroatom is terminal (has H atoms indicating
        # a functional group: -OH, -NH2, -SH)
        num_h = atom.GetTotalNumHs()
        heavy_neighbors = [n for n in atom.GetNeighbors() if n.GetSymbol() != 'H']

        if symbol == 'O' and num_h >= 1:
            if exclude_oh:
                continue  # Skip OH when checking for non-OH terminal FGs
            return True
        if symbol == 'N' and num_h >= 2 and len(heavy_neighbors) <= 1:
            # Terminal NH2 (primary amine at chain end)
            return True
        if symbol == 'S' and num_h >= 1 and len(heavy_neighbors) <= 1:
            # Terminal SH (thiol). An H-bearing S with TWO heavy neighbours is
            # an EMBEDDED nonstandard-valence skeletal atom (λ4-SH2 / λ6-SH4,
            # — 2,5λ4,8,11-tetrathiadodecane), not a thiol; genuine
            # thiols are already rejected by Gate 2's [SX2H] pattern.
            return True

    return False


# DD2 Fix A.1 (Phase D, /: chalcogens whose mutual single bond is a
# peroxide / disulfide / thioperoxol linkage (-O-O-, -S-S-, -Se-Se-, -Te-Te-, and the
# mixed -S-O- / -O-S- of the thioperoxol family). The set is the veto candidate list
# for _find_replacement_chain, which since fix a performance pass vetoes only the MIXED
# pairs: a same-element pair is one heterounit and the 'a' PIN is built
# across it when four or more heterounits are present ('2,4,5,8,11-pentathiadodecane
# (PIN)', the Blue Book); below four the classic path declines on the count.
# A terminal -OOH still fails the terminator gate (CCCCOO -> 'butane-1-peroxol').
_CHALCOGEN_ATOMIC_NUMS = frozenset({8, 16, 34, 52})  # O, S, Se, Te


def _dichalcogen_bond_set(mol: Chem.Mol) -> set:
    """Return the set of {a, b} index frozensets for every divalent
    chalcogen-chalcogen single bond in *mol* (peroxide / disulfide / thioperoxol
    linkages, /.

    A bond qualifies when BOTH endpoints are divalent chalcogens (O/S/Se/Te,
    no double/triple/aromatic bond, neutral, the ``-X-`` ether-oxidation state)
    joined by a single bond. This is a STRUCTURAL graph property derived from
    the molecule itself (internal notes pattern), NOT a per-FG SMARTS blocklist —
    so it precisely forbids only the O-O/S-S traversal (leaving an unrelated
    C-O-C ether elsewhere in the chain walkable) and uniformly covers the Se/Te
    analogues and the terminal ``-SSH``/``-OOH`` cases that the carbon-flanked
    ``peroxide``/``disulfide`` SMARTS do not perceive.
    """
    bonds: set = set()

    def _is_divalent_chalcogen(atom) -> bool:
        if atom.GetAtomicNum() not in _CHALCOGEN_ATOMIC_NUMS:
            return False
        if atom.GetFormalCharge() != 0 or atom.GetIsAromatic():
            return False
        # All bonds on a peroxide/disulfide-type chalcogen are single bonds
        # (excludes sulfinyl/sulfonyl/carbonyl-adjacent higher-valent S/Se).
        for b in atom.GetBonds():
            if b.GetBondType() != Chem.BondType.SINGLE:
                return False
        return True

    def _chalcogen_neighbour_count(atom) -> int:
        return sum(
            1 for n in atom.GetNeighbors()
            if n.GetAtomicNum() in _CHALCOGEN_ATOMIC_NUMS
        )

    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        a = bond.GetBeginAtom()
        b = bond.GetEndAtom()
        if not (_is_divalent_chalcogen(a) and _is_divalent_chalcogen(b)):
            continue
        # Scope to a 2-chalcogen linkage (peroxide / disulfide / thioperoxol):
        # each endpoint must have EXACTLY ONE chalcogen neighbour (the other). A
        # 3+ chalcogen chain (-S-S-S-, polysulfide) has an interior chalcogen with
        # two chalcogen neighbours; those remain skeletal ('trithia...') — DD2
        # covers only the 2-chalcogen peroxide/disulfide/thioperoxol class, and
        # over-vetoing polysulfides would strip their established skeletal names.
        if _chalcogen_neighbour_count(a) == 1 and _chalcogen_neighbour_count(b) == 1:
            bonds.add(frozenset((a.GetIdx(), b.GetIdx())))
    return bonds


def _skeletal_atoms_all_expressible(mol: Chem.Mol, atoms: List[int]) -> bool:
    """Is every non-carbon skeletal atom in ``atoms`` spellable as an 'a' prefix?

    The Table-1.5 replacement set is CLOSED, so an element outside
    ``REPLACEMENT_TERMS`` has no morpheme at all. This module's collect loops all
    filter on ``symbol in REPLACEMENT_TERMS``, which SKIPS such an atom -- while
    ``chain_length``/``ring_size`` keep counting it, so the parent stem renames it
    as a CARBON. That is a wrong structure, not a coverage gap:

        ``CC[Tl]CCSCC`` -> ``3-thiaoctane`` (C6STl named as C7S)
        ``CCS[Zn]SCC`` -> ``3,5-dithiaheptane``
        ``C1CCOCCOCC[Tl]CCOCC1`` -> ``1,4,10-trioxacyclopentadecane``
        ``CC[Tl]CC[Tl]CCSCC`` -> ``3-thiaundecane`` (TWO atoms absorbed)

    ``_find_replacement_chain`` builds its adjacency over every heavy atom with no
    element filter, so the off-table atom enters the backbone freely, and the
    terminator gate only inspects the two chain ENDS. Gating here --
    on the whole skeleton, at the point the skeleton is chosen -- is what makes the
    refusal total rather than end-relative.

    Note this is reachable only in the configuration where cannot run (no
    OPSIN jar / gate disabled), because the round trip otherwise suppresses these
    to an honest ``"<element> compound (not supported)"``. That is precisely the
    supported fail-OPEN mode, so it needs a Java-free source-level refusal.
    """
    for idx in atoms:
        symbol = mol.GetAtomWithIdx(idx).GetSymbol()
        if symbol != 'C' and symbol not in REPLACEMENT_TERMS:
            return False
    return True


def _find_replacement_chain(
    mol: Chem.Mol, exclude_atoms: Optional[set] = None,
) -> Optional[List[int]]:
    """Find the longest chain backbone including heteroatoms.

    For acyclic molecules, finds the longest simple path between
    terminal atoms (degree 1 in heavy-atom graph).

    Args:
        mol: RDKit molecule object.
        exclude_atoms: atom indices removed from the graph before the walk
            (the acid-suffix oxygens — they are suffix atoms, not skeletal
            atoms, and would otherwise win the diameter as leaves).

    Returns:
        List of atom indices forming the backbone, or None if no valid
        backbone found.
    """
    exclude_atoms = exclude_atoms or set()
    # Build adjacency list for heavy atoms only
    num_atoms = mol.GetNumAtoms()
    if num_atoms - len(exclude_atoms) < 3:
        return None

    # DD2 Fix A.1, narrowed to the MIXED thioperoxide bond (-O-S-, -S-Se-,...).
    # A SAME-element pair (-OO-, -SS-, -SeSe-, -TeTe-) is walked:
    # (the Blue Book, '## Skeletal replacement ('a') nomenclature
    # in acyclic chains') names it ONE heterounit -- "A heterounit is a set of
    # heteroatoms having a name of its own such as, -SS-, disulfanediyl" -- and
    # method (4) (:27861,:27866) builds the 'a' PIN straight across it:
    # "(4) 2,4,5,8,11-pentathiadodecane (PIN)" (:27894), "(3) 2,4,5,8-tetrathia-
    # 11-selenadodecane (PIN)" (:27927). The heterounit count then decides:
    # below four the classic path declines and the substitutive (R)disulfanyl /
    # (R)peroxy name is the PIN, "(1) (methyldisulfanyl)methane (PIN)" (:27874).
    # A terminal -OOH / -SSH still ends the chain on a chalcogen and fails the
    # terminator gate (Gate 3b), so 'butane-1-peroxol' keeps its suffix path.
    # The mixed pair stays unwalked (unchanged): (:27900) gives no 'a'
    # example across an -O-S- bond to fix its heterounit count, so fail closed.
    _veto_bonds = {
        b for b in _dichalcogen_bond_set(mol)
        if len({mol.GetAtomWithIdx(i).GetAtomicNum() for i in b}) == 2
    }

    adj: Dict[int, List[int]] = defaultdict(list)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        if a1 in exclude_atoms or a2 in exclude_atoms:
            continue
        if _veto_bonds and frozenset((a1, a2)) in _veto_bonds:
            continue
        adj[a1].append(a2)
        adj[a2].append(a1)

    # Find terminal atoms (degree 1 in heavy-atom graph)
    terminals = [idx for idx in range(num_atoms)
                 if idx not in exclude_atoms and len(adj[idx]) == 1]

    if len(terminals) < 2:
        # No clear chain endpoints -- not a chain molecule
        return None

    # For acyclic molecules, find the longest path using BFS from each terminal.
    # In a tree (acyclic graph), the longest path can be found by:
    # 1. BFS from any node to find the farthest node
    # 2. BFS from that farthest node to find the actual longest path
    # But since we need the actual path, we use DFS enumeration between terminal pairs.

    # Optimization: For trees, use double-BFS to find diameter endpoints
    # Step 1: BFS from first terminal to find farthest node
    farthest, _ = _bfs_farthest(adj, terminals[0], num_atoms)
    # Step 2: BFS from farthest to find the other end of the diameter
    other_end, _ = _bfs_farthest(adj, farthest, num_atoms)

    # Now find the actual path between farthest and other_end using BFS
    path = _find_path_bfs(adj, farthest, other_end, num_atoms)

    if path is None or len(path) < 3:
        return None

    # TOTALITY: refuse a backbone carrying a skeletal atom no 'a' prefix spells,
    # rather than letting the collect loops skip it into the carbon stem.
    if not _skeletal_atoms_all_expressible(mol, path):
        return None

    return path


def _bfs_farthest(
    adj: Dict[int, List[int]], start: int, num_atoms: int
) -> Tuple[int, int]:
    """BFS from start, return (farthest_node, distance).

    Args:
        adj: Adjacency list.
        start: Starting atom index.
        num_atoms: Total number of atoms.

    Returns:
        Tuple of (farthest atom index, distance to it).
    """
    visited = [False] * num_atoms
    visited[start] = True
    queue = [(start, 0)]
    farthest = start
    max_dist = 0

    head = 0
    while head < len(queue):
        node, dist = queue[head]
        head += 1
        if dist > max_dist:
            max_dist = dist
            farthest = node
        for neighbor in adj[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append((neighbor, dist + 1))

    return farthest, max_dist


def _find_path_bfs(
    adj: Dict[int, List[int]], start: int, end: int, num_atoms: int
) -> Optional[List[int]]:
    """Find the path between start and end using BFS (for trees, this is unique).

    Args:
        adj: Adjacency list.
        start: Starting atom index.
        end: Ending atom index.
        num_atoms: Total number of atoms.

    Returns:
        List of atom indices from start to end, or None.
    """
    visited = [False] * num_atoms
    parent = [-1] * num_atoms
    visited[start] = True
    queue = [start]

    head = 0
    while head < len(queue):
        node = queue[head]
        head += 1
        if node == end:
            # Reconstruct path
            path = []
            current = end
            while current != -1:
                path.append(current)
                current = parent[current]
            path.reverse()
            return path
        for neighbor in adj[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                parent[neighbor] = node
                queue.append(neighbor)

    return None


def _orient_oh_end_first(
    backbone: List[int], mol: Chem.Mol, carbon_idx: int
) -> List[int]:
    """Orient backbone so the carbon bearing the terminal OH gets locant 1.

    For replacement chains with terminal -ol suffix, the principal group
    (OH) must receive the lowest possible locant per IUPAC.

    The OH oxygen has already been stripped from the backbone. This function
    ensures the carbon that was bonded to the OH is at position 0 (locant 1).

    Args:
        backbone: List of atom indices forming the backbone (OH oxygen excluded).
        mol: RDKit molecule object.
        carbon_idx: Atom index of the carbon bearing the -OH group.

    Returns:
        Reoriented backbone list with OH-bearing carbon at position 0.
    """
    if backbone[0] == carbon_idx:
        return backbone
    elif backbone[-1] == carbon_idx:
        return list(reversed(backbone))
    else:
        # Carbon not at either end -- shouldn't happen for unbranched chain
        # Fall back to lowest heteroatom locants
        return _orient_for_lowest_locants(backbone, mol)


def _backbone_unsaturation(
    mol: Chem.Mol, backbone: List[int],
    ignore_atoms: Optional[set] = None,
) -> Optional[Tuple[List[int], List[int]]]:
    """Return ``(ene_locants, yne_locants)`` for the backbone's multiple bonds
    under the given orientation, or ``None`` (fail-closed).

    : double/triple bonds take locants from the FIXED heterochain
    numbering. Fail-closed conditions (any -> None, never a lossy name):
      * an aromatic/exotic bond order anywhere in the molecule;
      * a multiple bond not between two CONSECUTIVE backbone atoms;
      * a multiple bond involving a heteroatom (C=N / S=O etc. are
        characteristic groups or λ-territory, not chain ene/yne).
    A fully saturated chain returns ``(, )`` — byte-identical downstream.
    """
    pos = {idx: i for i, idx in enumerate(backbone)}
    ene: List[int] = []
    yne: List[int] = []
    ignore_atoms = ignore_atoms or set()
    for bond in mol.GetBonds():
        bond_type = bond.GetBondType()
        if bond_type == Chem.BondType.SINGLE:
            continue
        # Suffix oxygens (the acid C=O) are accounted for by the suffix
        # itself, not as chain unsaturation.
        if (bond.GetBeginAtomIdx() in ignore_atoms
                or bond.GetEndAtomIdx() in ignore_atoms):
            continue
        if bond_type not in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE):
            return None
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a1 not in pos or a2 not in pos or abs(pos[a1] - pos[a2]) != 1:
            return None
        if (bond.GetBeginAtom().GetAtomicNum() != 6
                or bond.GetEndAtom().GetAtomicNum() != 6):
            return None
        locant = min(pos[a1], pos[a2]) + 1
        (ene if bond_type == Chem.BondType.DOUBLE else yne).append(locant)
    return sorted(ene), sorted(yne)


def _lambda_orientation_key(
    chain: List[int], mol: Chem.Mol,
) -> List[Tuple[int, int]]:
    """P-21.2.4.1/2 orientation key: sorted ``(locant, -bonding_number)`` for
    every nonstandard-valence backbone atom. Lexicographic minimum implements
    both tiers — lowest locants to the λ set, and (P-21.2.4.2) the HIGHER
    bonding number at the lower locant on a positional tie (λ6 before λ4)."""
    key: List[Tuple[int, int]] = []
    for i, idx in enumerate(chain):
        lam = nonstandard_bonding_number(mol, idx)
        if lam is not None:
            key.append((i + 1, -lam))
    return sorted(key)


def _orient_for_lowest_locants(
    backbone: List[int], mol: Chem.Mol,
    suffix_atoms: Optional[set] = None,
    substituent_atoms: Optional[set] = None,
) -> List[int]:
    """Orient the backbone chain to give lowest locants to heteroatoms.

    Tries both directions and picks the one whose heteroatom numbering is
    preferred by the shared ``compare_numbering`` comparator (DD4 / E1):

      1. lowest heteroatom locant SET, kind-agnostic; then
      2. on a positional tie, the lowest locant to the element highest in the
         element-seniority order — e.g. ``COCSC`` (positions
         ``{2,4}`` either way) gives O the locant 2 over S, so both ``COCSC``
         and ``CSCOC`` deterministically yield ``2-oxa-4-thiapentane``.

    The old code broke a positional tie by ``return forward`` (input-order
    dependent — the H2 non-determinism bug); the element-seniority tier is now
    a real total order, so genuine ties only occur for true symmetry.

    Args:
        backbone: List of atom indices forming the backbone.
        mol: RDKit molecule object.

    Returns:
        Reoriented backbone list.
    """
    from .locants import compare_numbering

    forward = backbone
    reverse = list(reversed(backbone))

    forward_pairs = _get_heteroatom_pairs(forward, mol)
    reverse_pairs = _get_heteroatom_pairs(reverse, mol)

    decision = compare_numbering(
        {'heteroatoms': forward_pairs},
        {'heteroatoms': reverse_pairs},
    )
    if decision == 1:
        return reverse
    if decision == 0:
        # Heteroatom tie — apply the fixed-numbering tie-break cascade:
        # λ /2) -> suffix -> ene/yne
        # / -> substituent prefixes (f)).
        fwd_lam = _lambda_orientation_key(forward, mol)
        rev_lam = _lambda_orientation_key(reverse, mol)
        if fwd_lam != rev_lam:
            return forward if fwd_lam < rev_lam else reverse

        if suffix_atoms:
            fwd_suf = sorted(i + 1 for i, idx in enumerate(forward)
                             if idx in suffix_atoms)
            rev_suf = sorted(i + 1 for i, idx in enumerate(reverse)
                             if idx in suffix_atoms)
            if fwd_suf != rev_suf:
                return forward if fwd_suf < rev_suf else reverse

        fwd_unsat = _backbone_unsaturation(mol, forward)
        rev_unsat = _backbone_unsaturation(mol, reverse)
        if fwd_unsat is not None and rev_unsat is not None:
            fwd_key = (sorted(fwd_unsat[0] + fwd_unsat[1]), fwd_unsat[0])
            rev_key = (sorted(rev_unsat[0] + rev_unsat[1]), rev_unsat[0])
            if rev_key < fwd_key:
                return reverse
            if fwd_key < rev_key:
                return forward

        if substituent_atoms:
            fwd_sub = sorted(i + 1 for i, idx in enumerate(forward)
                             if idx in substituent_atoms)
            rev_sub = sorted(i + 1 for i, idx in enumerate(reverse)
                             if idx in substituent_atoms)
            if fwd_sub != rev_sub:
                return forward if fwd_sub < rev_sub else reverse
    # decision <= 0: forward preferred OR a genuine symmetry tie (either
    # orientation is correct and byte-identical) — return forward.
    return forward


def _get_heteroatom_pairs(
    backbone: List[int], mol: Chem.Mol
) -> List[Tuple[int, str]]:
    """Get ``(locant, element_symbol)`` pairs for embedded skeletal heteroatoms.

    Position-AND-element (DD4): the element is needed for the element-seniority
    numbering tie-break, which the previous positions-only helper discarded.

    Args:
        backbone: List of atom indices.
        mol: RDKit molecule object.

    Returns:
        List of (1-based locant, element symbol) for embedded heteroatoms,
        sorted by locant.
    """
    pairs: List[Tuple[int, str]] = []
    for i, atom_idx in enumerate(backbone):
        # Terminal positions are included: a whitelisted heteroatom terminator
        # (Gate 3b, carries its own 'a' prefix and must steer the
        # numbering ("2-oxa-4-thia-1,5-disilapentane"). No classic-path chain
        # reaches numbering with a heteroatom terminal (Gates 3/3b refuse).
        atom = mol.GetAtomWithIdx(atom_idx)
        symbol = atom.GetSymbol()
        if symbol in REPLACEMENT_TERMS:
            pairs.append((i + 1, symbol))  # 1-based
    return sorted(pairs)


def _get_heteroatom_locants(backbone: List[int], mol: Chem.Mol) -> List[int]:
    """Get sorted positional locants of embedded heteroatoms in a backbone.

    Retained as the positional-only primitive (kind-agnostic); the
    element-aware ``_get_heteroatom_pairs`` is preferred for numbering
    decisions.

    Args:
        backbone: List of atom indices.
        mol: RDKit molecule object.

    Returns:
        Sorted list of 1-based locants for embedded heteroatoms.
    """
    return [loc for loc, _ in _get_heteroatom_pairs(backbone, mol)]


def _build_replacement_name(
    chain_length: int,
    heteroatom_positions: List[Tuple[int, str]],
    suffix: Optional[Tuple[str, int]] = None,
    lambda_by_locant: Optional[Dict[int, int]] = None,
    ene_locants: Optional[List[int]] = None,
    yne_locants: Optional[List[int]] = None,
) -> str:
    """Build the skeletal replacement name from chain length and heteroatom info.

    Follows IUPAC: replacement terms are cited in ascending
    locant order. When different elements share the same lowest locant
    (rare), alphabetical order of the replacement term breaks the tie.

    Args:
        chain_length: Total number of atoms in the backbone.
        heteroatom_positions: List of (locant, element_symbol) tuples.
        suffix: Optional (suffix_name, locant) tuple for terminal FG,
                e.g., ('ol', 1) for terminal alcohol.
        lambda_by_locant: Optional {locant: bonding_number} for embedded
                heteroatoms whose valence is non-standard /.
                The λ is cited after the locant (``2lambda4-thia...``); absent /
                standard locants emit the bare number (byte-identical default).

    Returns:
        Complete replacement name string.
    """
    lambda_by_locant = lambda_by_locant or {}
    ene_locants = ene_locants or []
    yne_locants = yne_locants or []
    # Get the chain prefix (hex, oct, non, etc.)
    chain_prefix = get_chain_prefix(chain_length)

    # Group heteroatoms by element symbol
    element_groups: Dict[str, List[int]] = defaultdict(list)
    for locant, symbol in heteroatom_positions:
        element_groups[symbol].append(locant)

    # Sort each group's locants
    for symbol in element_groups:
        element_groups[symbol].sort()

    # /: cite the 'a' prefixes in the ELEMENT SENIORITY
    # order of (O > S > Se > Te > N > P >... > Si >...), NOT by
    # ascending locant. BB verbatim: "3-phospha-2,5,7-trisilaoctane" (phospha
    # cited first although sila holds locant 2); "8-thia-2,4,6-trisiladecane".
    sorted_groups = sorted(
        element_groups.items(),
        key=lambda item: _A_CITATION_INDEX.get(item[0], 99)
    )

    # Build replacement term parts
    parts = []
    for symbol, locants in sorted_groups:
        term = REPLACEMENT_TERMS[symbol]
        locant_str = ','.join(
            format_lambda_token(loc, lambda_by_locant.get(loc)) for loc in locants
        )
        count = len(locants)

        if count == 1:
            multiplier = ''
        elif count in SIMPLE_MULTIPLIERS:
            multiplier = SIMPLE_MULTIPLIERS[count]
        else:
            # Fallback for very large counts (unlikely for replacement)
            multiplier = SIMPLE_MULTIPLIERS.get(count, f'{count}')

        parts.append(f'{locant_str}-{multiplier}{term}')

    # Join parts with hyphens
    replacement_prefix = '-'.join(parts)

    if suffix is not None:
        # Build name with functional group suffix
        # e.g., "3,6-dioxaoctan-1-ol"
        suffix_name, suffix_locant = suffix
        # acid suffixes on the fixed heterochain numbering:
        # "3,6,9,12-tetraoxatetradecanedioic acid" (terminal diacid — no
        # locants needed) / "3,6,9,12-tetraoxapentadecan-15-oic acid"
        # — the heteroatoms own the low locants).
        if suffix_name == 'dioic acid':
            return f'{replacement_prefix}{chain_prefix}anedioic acid'
        if suffix_name == 'oic acid':
            return (f'{replacement_prefix}{chain_prefix}an'
                    f'-{suffix_locant}-oic acid')
        # Vowel elision: remove terminal 'e' before suffix starting with vowel
        # "octane" -> "octan" before "-1-ol"
        stem = f'{chain_prefix}an'
        if suffix_name.startswith(('a', 'e', 'i', 'o', 'u', 'y')):
            # "an" already drops the 'e' from "ane"
            pass
        else:
            stem = f'{chain_prefix}ane'
        return f'{replacement_prefix}{stem}-{suffix_locant}-{suffix_name}'

    # ene/yne endings on the fixed heterochain numbering:
    # 2,4,6,8-tetrasiladec-9-ene /...deca-2,4-diene /...dec-1-en-9-yne.
    # Standard elision: the multiplied form keeps the connecting 'a'
    # (octa-2,6-diene); 'ene' drops its final 'e' before '-N-yne'.
    if ene_locants or yne_locants:
        stem = chain_prefix
        if ene_locants:
            multiplier = (SIMPLE_MULTIPLIERS[len(ene_locants)]
                          if len(ene_locants) > 1 else '')
            if multiplier:
                stem += 'a'
            locs = ','.join(str(loc) for loc in ene_locants)
            stem += f'-{locs}-{multiplier}en'
            if not yne_locants:
                stem += 'e'
        if yne_locants:
            multiplier = (SIMPLE_MULTIPLIERS[len(yne_locants)]
                          if len(yne_locants) > 1 else '')
            if multiplier and not ene_locants:
                stem += 'a'
            locs = ','.join(str(loc) for loc in yne_locants)
            stem += f'-{locs}-{multiplier}yne'
        return f'{replacement_prefix}{stem}'

    # Plain replacement name: "3,6-dioxaoctane"
    return f'{replacement_prefix}{chain_prefix}ane'


def _try_cyclic_replacement_name(mol: Chem.Mol, ring_info) -> Optional[str]:
    """Try cyclic skeletal replacement naming for large heterocyclic rings.

    Per IUPAC, large heterocyclic rings (>= 7 members) can use
    replacement nomenclature (oxa-/aza-/thia- prefixes on cycloalkane parent).
    Example: 1,4-dioxacyclononane for a 9-membered ring with 2 oxygens.

    Only applies when:
    - Single ring with >= 7 members
    - Ring contains heteroatoms from REPLACEMENT_TERMS
    - No substituents off the ring (all heavy atoms are ring atoms)
    - No priority functional groups (checked before calling this function)

    Args:
        mol: RDKit molecule object.
        ring_info: RDKit RingInfo object.

    Returns:
        Replacement name string or None if not applicable.
    """
    rings = ring_info.AtomRings()
    if len(rings) != 1:
        return None  # Only handle single-ring molecules for now

    ring = rings[0]
    ring_size = len(ring)

    # Gate: all heavy atoms must be in the ring (no substituents)
    if mol.GetNumAtoms() != ring_size:
        return None

    # Collect heteroatom positions in the ring
    ring_atoms = list(ring)
    # TOTALITY: same rule as the acyclic backbone. The loop below filters on
    # ``symbol in REPLACEMENT_TERMS``, so an off-table ring atom would be skipped
    # while ``ring_size`` still counted it into the cycloalkane stem.
    if not _skeletal_atoms_all_expressible(mol, ring_atoms):
        return None
    heteroatoms = []
    for i, atom_idx in enumerate(ring_atoms):
        atom = mol.GetAtomWithIdx(atom_idx)
        symbol = atom.GetSymbol()
        if symbol in REPLACEMENT_TERMS:
            heteroatoms.append((i, symbol))

    if not heteroatoms:
        return None

    # Check bond types: detect saturated vs unsaturated vs aromatic
    all_single = True
    has_aromatic = False
    for i in range(ring_size):
        a1 = ring_atoms[i]
        a2 = ring_atoms[(i + 1) % ring_size]
        bond = mol.GetBondBetweenAtoms(a1, a2)
        if bond:
            btype = bond.GetBondTypeAsDouble()
            if btype == 1.5:
                has_aromatic = True
                break
            elif btype != 1.0:
                all_single = False

    # Aromatic rings use Hantzsch-Widman or retained names, not replacement
    if has_aromatic:
        return None

    # Orient the ring to give lowest locants to heteroatoms.
    # Try all rotations and both directions; pick the one giving the
    # lowest heteroatom locant set at first point of difference.
    # For unsaturated rings, double bond locants serve as tiebreaker
    # after heteroatom locants (per IUPAC.
    best_key = None
    best_positions = None
    best_ordered = None

    for start in range(ring_size):
        for direction in [1, -1]:
            # Build the ordered ring from this starting point
            ordered = []
            for step in range(ring_size):
                idx = (start + step * direction) % ring_size
                ordered.append(ring_atoms[idx])

            # Compute heteroatom locants (1-based)
            positions = []
            for i, atom_idx in enumerate(ordered):
                atom = mol.GetAtomWithIdx(atom_idx)
                symbol = atom.GetSymbol()
                if symbol in REPLACEMENT_TERMS:
                    positions.append((i + 1, symbol))

            hetero_locant_set = sorted(pos[0] for pos in positions)

            # Per-element locant lists, in 'a'-prefix seniority order, compared
            # lexicographically — the "then, if necessary, according to the
            # order of seniority" criterion of (O before N in
            # 1,4,10,13-tetraoxa-7,16-diazacyclooctadecane, NOT 1,10-diaza-...).
            by_element: Dict[str, List[int]] = defaultdict(list)
            for loc, sym in positions:
                by_element[sym].append(loc)
            seniority_key = [
                sorted(by_element[sym])
                for sym in sorted(by_element,
                                  key=lambda s: _A_CITATION_INDEX.get(s, 99))
            ]

            # (section "Heteromonocyclic hydrides named by
            # skeletal replacement ('a') nomenclature", subsection
            # "Numbering"): "the locant '1' is given to the heteroatom first
            # cited in the order of seniority... The direction of numbering is
            # then chosen to give lower locants to the heteroatoms as a set...
            # and then, if necessary, according to the order of seniority."
            # So locant '1' to the MOST-SENIOR heteroatom present is the PRIMARY
            # criterion (senior to low-locants-as-a-set). This is the RING rule;
            # the low-locants-as-a-set-first rule is the CHAIN rule
            # and must NOT be applied here. Encoded as: minimise the lowest
            # locant borne by the most-senior heteroatom present, so the winner
            # necessarily carries that atom at locant 1.
            senior_element = min(
                by_element, key=lambda s: _A_CITATION_INDEX.get(s, 99)
            )
            senior_first_locant = min(by_element[senior_element])

            # Compute double bond locants for tiebreaker
            db_locants = []
            if not all_single:
                for i in range(ring_size):
                    a1 = ordered[i]
                    a2 = ordered[(i + 1) % ring_size]
                    bond = mol.GetBondBetweenAtoms(a1, a2)
                    if bond and bond.GetBondTypeAsDouble() == 2.0:
                        # Locant of double bond = lower-numbered atom (1-based)
                        db_locants.append(i + 1)
                db_locants.sort()

            # numbering order: (1) locant '1' to the senior
            # heteroatom; (2) low locants to the heteroatoms as a set; (3)
            # seniority of the 'a' prefixes; then (4) low locants to the
            # unsaturated sites ("Low locants are assigned first to the
            # heteroatoms and then to unsaturated sites").
            comparison_key = (
                senior_first_locant, hetero_locant_set, seniority_key,
                db_locants,
            )

            if best_key is None or comparison_key < best_key:
                best_key = comparison_key
                best_positions = positions
                best_ordered = ordered

    if not best_positions:
        return None

    # Build the cyclic replacement name using cyclo- prefix
    chain_prefix = get_chain_prefix(ring_size)

    # Group heteroatoms by element symbol
    element_groups: Dict[str, List[int]] = defaultdict(list)
    for locant, symbol in best_positions:
        element_groups[symbol].append(locant)

    for symbol in element_groups:
        element_groups[symbol].sort()

    # /: cite the 'a' prefixes in the ELEMENT SENIORITY
    # order of (O > S > Se > Te > N > P >... > Si >...), NOT by
    # ascending locant. BB verbatim: "3-phospha-2,5,7-trisilaoctane" (phospha
    # cited first although sila holds locant 2); "8-thia-2,4,6-trisiladecane".
    sorted_groups = sorted(
        element_groups.items(),
        key=lambda item: _A_CITATION_INDEX.get(item[0], 99)
    )

    total_hetero = sum(len(locs) for _, locs in sorted_groups)
    #: a single ring heteroatom is assigned locant '1', which is
    # OMITTED from the name (unless an indicated-hydrogen locant is present).
    # The omission applies ONLY to the SATURATED ring (all_single): when the
    # ring carries unsaturation (ene/yne locants are cited), the heteroatom
    # locant '1' is retained as the reference for those locants
    # (1-azacyclopentadeca-2,4,6,8,10,12,14-heptaene, NOT azacyclopentadeca-...).
    elide_single = total_hetero == 1 and all_single

    # "Homogeneous heteromonocyclic parent hydrides" (:8848) via
    # (:3007): "All locants are omitted in compounds... in which
    # all substitutable positions are completely substituted or modified...
    # in the same way." When EVERY ring skeletal atom is the SAME single
    # replacement element (a homogeneous ring, not merely one heteroatom) and
    # the ring is fully saturated (no unsaturation locant to anchor), the
    # whole locant set is omitted: "dodecasilacyclododecane (preselected
    # name)" (:8882), no locants at all despite 12 silicon atoms. A mixed
    # ring (more than one element present) is NOT "the same way" and keeps
    # full locants -- confirmed on the O+Si mixed-ring a trace positive.
    omit_all_locants = (
        all_single and len(sorted_groups) == 1 and total_hetero == ring_size
    )
    omit_locants = elide_single or omit_all_locants

    parts = []
    for symbol, locants in sorted_groups:
        term = REPLACEMENT_TERMS[symbol]
        locant_str = '' if omit_locants else ','.join(str(loc) for loc in locants)
        count = len(locants)

        if count == 1:
            multiplier = ''
        elif count in SIMPLE_MULTIPLIERS:
            multiplier = SIMPLE_MULTIPLIERS[count]
        else:
            multiplier = SIMPLE_MULTIPLIERS.get(count, f'{count}')

        sep = '' if omit_locants else '-'
        parts.append(f'{locant_str}{sep}{multiplier}{term}')

    replacement_prefix = '-'.join(parts)

    if all_single:
        return f'{replacement_prefix}cyclo{chain_prefix}ane'

    # Unsaturated large heterocyclic rings: build name with -ene/-adiene/-atriene
    # Compute double bond locants from the best orientation
    db_locants = []
    for i in range(ring_size):
        a1 = best_ordered[i]
        a2 = best_ordered[(i + 1) % ring_size]
        bond = mol.GetBondBetweenAtoms(a1, a2)
        if bond and bond.GetBondTypeAsDouble() == 2.0:
            db_locants.append(i + 1)
    db_locants.sort()

    if not db_locants:
        # No double bonds found despite all_single being False (shouldn't happen)
        return f'{replacement_prefix}cyclo{chain_prefix}ane'

    return _build_unsaturated_cyclic_name(
        replacement_prefix, chain_prefix, db_locants
    )


def _build_unsaturated_cyclic_name(
    replacement_prefix: str,
    chain_prefix: str,
    double_bond_locants: List[int],
) -> str:
    """Build cyclic replacement name with unsaturation suffix.

    Constructs names like '1-oxacyclohept-4,5-diene' from the replacement
    prefix, chain size prefix, and double bond locant positions.

    Handles vowel elision: when chain_prefix ends in 'a' and the suffix
    starts with a vowel, the trailing 'a' is dropped (e.g., 'octa' + 'ene'
    becomes 'octene', not 'octaene'). Since get_chain_prefix returns
    stems without trailing 'a' (e.g., 'oct', 'dec'), and we build the
    suffix directly, no special elision is needed for most cases.

    For single double bond: '-{locant}-ene'
    For 2 double bonds: '-{loc1},{loc2}-diene' (with linking 'a')
    For 3 double bonds: '-{loc1},{loc2},{loc3}-triene' (with linking 'a')

    Args:
        replacement_prefix: Heteroatom prefix (e.g., '1-oxa').
        chain_prefix: Ring size prefix from get_chain_prefix (e.g., 'hept').
        double_bond_locants: Sorted list of 1-based locant positions for
            double bonds.

    Returns:
        Complete unsaturated cyclic replacement name string.
    """
    count = len(double_bond_locants)
    locant_str = ','.join(str(loc) for loc in double_bond_locants)

    if count == 1:
        suffix = 'ene'
    else:
        # For multiple double bonds: multiplier + 'ene'
        # 2 DB = "diene", 3 DB = "triene", 4 DB = "tetraene", etc.
        if count in SIMPLE_MULTIPLIERS:
            multiplier = SIMPLE_MULTIPLIERS[count]
        else:
            multiplier = str(count)
        suffix = f'{multiplier}ene'

    # Build stem: cyclo + chain_prefix
    # get_chain_prefix returns e.g., 'hept', 'oct', 'dec' (no trailing 'a')
    # IUPAC convention for unsaturation:
    # - Single ene: stem without linking vowel -> cyclohept-2-ene
    # - Multiple ene: stem with linking 'a' -> cyclohepta-2,4-diene
    # The linking 'a' goes on the stem when the suffix starts with a
    # consonant (d in diene, t in triene).
    stem = chain_prefix
    if count > 1:
        # Add linking vowel 'a' to chain prefix for multi-ene
        # "hept" -> "hepta", "oct" -> "octa", "dec" -> "deca"
        stem = chain_prefix + 'a'

    return f'{replacement_prefix}cyclo{stem}-{locant_str}-{suffix}'
