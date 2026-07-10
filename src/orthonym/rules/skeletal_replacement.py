"""
Skeletal replacement ("a") nomenclature for chains with embedded heteroatoms.

Implements IUPAC 2013 P-15.4 replacement nomenclature where heteroatoms
embedded in a carbon chain backbone are named using replacement terms
(oxa, aza, thia, etc.) rather than substitutive prefixes (methoxy, amino, etc.).

Examples:
    COCCOCCOC -> 2,5,8-trioxanonane   (3 O: skeletal replacement)
    CCNCCC  -> 3-azahexane             (N: amine gate blocks; substitutive N-ethylpropan-1-amine)
    COCCOC  -> None (2 O, no -ol: R4 routes substitutive -> 1,2-dimethoxyethane)
    OCCOCCOCC -> 3,6-dioxaoctan-1-ol   (2 O with terminal -ol: skeletal + suffix)

Scope: Chain-only (acyclic). Rings <= 10 atoms are handled by
Hantzsch-Widman naming in the heterocycles module.

References:
    IUPAC 2013 Blue Book, P-15.4 (Replacement nomenclature)
    IUPAC 2013 Blue Book, P-15.4.3.1 (Order of citation of replacement terms)
"""

from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from rdkit import Chem

from ..data.chain_names import get_chain_prefix
from ..assembly.naming_utils import SIMPLE_MULTIPLIERS
# Phase 6 (v23): shared P-21.2.4 / P-31.1.4.2 λ-convention. A non-standard-valence
# embedded chain heteroatom cites its bonding number after the locant
# (``...lambda<n>...``); standard valences emit the bare locant (byte-identical).
from .lambda_convention import nonstandard_bonding_number, format_lambda_token


# ============================================================================
# Replacement term table (IUPAC P-15.4, Table 2.3)
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

# P-15.4.3.1: replacement ('a') prefixes are cited in the name in the element
# seniority order of P-15.4.1.2 / Table 2.4 — NOT in ascending-locant order.
# BB verbatim: "3-phospha-2,5,7-trisilaoctane" (phospha cited first although
# sila holds the lower locant 2); "8-thia-2,4,6-trisiladecane".
_A_CITATION_ORDER = ['F', 'Cl', 'Br', 'I', 'O', 'S', 'Se', 'Te', 'N', 'P',
                     'As', 'Sb', 'Bi', 'Si', 'Ge', 'Sn', 'Pb', 'B',
                     'Al', 'Ga', 'In', 'Tl']
_A_CITATION_INDEX = {el: i for i, el in enumerate(_A_CITATION_ORDER)}

# P-15.4.3.1 / P-51.4.1.4: a heterochain may be TERMINATED by C or one of
# P, As, Sb, Bi, Si, Ge, Sn, Pb, B, Al, Ga, In, Tl (BB example:
# 2-oxa-4-thia-1,5-disilapentane). Al/Ga/In/Tl carry no entry in
# REPLACEMENT_TERMS, so they fail closed at the terminator check.
_ALLOWED_HETERO_TERMINATORS = {'P', 'As', 'Sb', 'Bi',
                               'Si', 'Ge', 'Sn', 'Pb', 'B'}


def _qualifies_for_pin_skeletal_replacement(
    backbone: List[int], mol,
) -> Tuple[bool, str]:
    """Phase 154.A D-03: lock PIN trigger to strict IUPAC P-15.4.1.2.

    Three accept branches per IUPAC Blue Book P-15.4.1.2:
      (a) >= 4 same-kind embedded heteroatoms in the chain backbone
      (b) >= 3 mixed-kind embedded heteroatoms (>= 2 distinct elements)
      (c) substitutive expression would require >= 5 prefix units
          ("undue complexity"; conservative threshold for v18 -- equivalent
          to >= 5 embedded heteroatoms total regardless of kind diversity)

    Falls through to the legacy gate-5 semantics: a single embedded heteroatom
    in a backbone of length < 6 is REJECTED (substitutive form preferred:
    methoxymethane / ethoxyethane / etc.). A single embedded heteroatom in a
    backbone of length >= 6 is ACCEPTED (substitutive form becomes awkward).

    Two-heteroatom cases that do not trip branches (a/b/c) are ACCEPTED
    (preserves the legacy "diether and similar" behavior for compounds like
    3,6-dioxaoctan-1-ol).

    Args:
        backbone: ordered list of atom indices forming the principal chain
                  (output of _find_replacement_chain).
        mol: RDKit Mol object.

    Returns:
        (True, branch_label) where branch_label in
            {">=4-same-kind", ">=3-mixed-kind", "undue-complexity",
             "single-hetero-long-chain", "two-hetero-substitutive-equivalent"}
        (False, reason) where reason in
            {"no-heteroatoms", "single-hetero-short-chain"}.

    Source: IUPAC Blue Book 2013 P-15.4.1.2.
    Source: 154-CONTEXT.md D-03; 154-AUDIT-A.md gap inventory; 154-RESEARCH.md §3.2.
    """
    from collections import Counter
    embedded = []
    for i, atom_idx in enumerate(backbone):
        if i == 0 or i == len(backbone) - 1:
            continue  # skip terminal positions
        symbol = mol.GetAtomWithIdx(atom_idx).GetSymbol()
        if symbol in REPLACEMENT_TERMS:
            embedded.append(symbol)

    if not embedded:
        return (False, "no-heteroatoms")

    counts = Counter(embedded)
    same_kind_max = max(counts.values())
    distinct_kinds = len(counts)
    total = sum(counts.values())

    # Branch (a): >= 4 same-kind heteroatoms
    if same_kind_max >= 4:
        return (True, ">=4-same-kind")

    # Branch (b): >= 3 mixed-kind heteroatoms
    if distinct_kinds >= 2 and total >= 3:
        return (True, ">=3-mixed-kind")

    # Branch (c): "undue complexity" -- conservative >= 5 total threshold
    if total >= 5:
        return (True, "undue-complexity")

    # Fall-through: single heteroatom -- preserve legacy gate-5 semantics.
    if total == 1 and len(backbone) < 6:
        return (False, "single-hetero-short-chain")

    # Single heteroatom in a long chain (>= 6): accept (legacy gate-5 behavior).
    if total == 1:
        return (True, "single-hetero-long-chain")

    # Two heteroatoms not covered by branches (a/b/c): accept (preserves
    # current behavior for diethers, etc. -- e.g., 3,6-dioxaoctan-1-ol).
    return (True, "two-hetero-substitutive-equivalent")


# IUPAC P-15.4.3.1: Order of citation for replacement terms
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

# P-62.2.2: trivalent N bonded only to carbons → substitutive naming preferred
# over skeletal ("aza") replacement for ACYCLIC carbon-chain amines.
# Applied only in the acyclic path (Gate 2c); cyclic large-ring aza-replacement
# is governed by P-22.1.3 and must not be blocked here.
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


# P-51.4.1.3: carboxylic-acid suffix integration on a fixed-numbered heterochain
# (3,6,9,12-tetraoxatetradecanedioic acid; 3,6,9,12-tetraoxapentadecan-15-oic
# acid, P-59.2.2 — the acid carbon takes locant 15 because the heteroatoms own
# the numbering, P-51.4.1.2).
_ACID_PATTERN = Chem.MolFromSmarts('[CX3](=[OX1])[OX2H1]')


def _detect_terminal_acid_groups(mol: Chem.Mol) -> Optional[List[Dict]]:
    """Detect -C(=O)OH groups eligible for P-51.4.1.3 suffix integration.

    Returns ``[]`` when the molecule carries no carboxyl at all (classic path),
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
    """P-51.4.1.1 strict qualification for the NEW classes (suffix-bearing,
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

    P-15.4.3.2.2: substituent locants follow the FIXED heterochain numbering
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
    prefixes are alphabetised (multiplying prefixes ignored, P-14.5.2)."""
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
    # Gate 1: Rings gate (IUPAC P-15.4 / P-22.1.3)
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
        # P-22.2.2.1: saturated heterocyclic rings of size 3-10 use
        # Hantzsch-Widman naming (oxepane/azepane, and mixed-heteroatom PINs
        # such as 1,4-oxazepane / 1,4-diazepane / 1,4-thiazepane), NOT cyclic
        # skeletal ("aza"/"oxa") replacement. Route ALL saturated 7-10 single-
        # ring heterocycles back to the HW namer here (return None) regardless
        # of how many heteroatoms they carry — the HW builder emits the correct
        # collected-locant PIN. Only UNSATURATED single-rings (mancude/partial)
        # and rings > 10 keep cyclic replacement (P-22.1.3); those fall through.
        if len(ring_info.AtomRings()) == 1:
            _ring = ring_info.AtomRings()[0]
            _rsize = len(_ring)
            if 7 <= _rsize <= 10:
                # Check saturation: any double/triple ring bond?
                _all_single = all(
                    mol.GetBondBetweenAtoms(_ring[i], _ring[(i + 1) % _rsize])
                    .GetBondTypeAsDouble() == 1.0
                    for i in range(_rsize)
                )
                if _all_single:
                    # HW namer owns saturated 7-10 rings: oxepane, azepane,
                    # 1,4-oxazepane, 1,4-diazepane, 1,4-thiazepane, ...
                    return None
        # For large heterocyclic rings (>= 11, or unsaturated/multi-heteroatom
        # 7-10-rings), try cyclic replacement naming per IUPAC P-22.1.3.
        # Keep the "no priority FGs" gate.
        for pat in _PRIORITY_FG_PATTERNS:
            if mol.HasSubstructMatch(pat):
                return None
        return _try_cyclic_replacement_name(mol, ring_info)

    # ----------------------------------------------------------------
    # P-51.4.1.3 acid-suffix integration: a clean chain-terminal -C(=O)OH
    # (1 or 2 of them) is expressed as the -oic/-dioic acid suffix on the
    # fixed-numbered heterochain instead of tripping Gate 2. [] = no acid
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
    # Gate 2c (P-62.2.2): acyclic carbon-chain amines use substitutive naming.
    # If the molecule contains a trivalent N bonded ONLY to carbons (secondary
    # or tertiary amine on an all-carbon backbone), skeletal ("aza") replacement
    # is NOT the PIN — the substitutive handler produces N-alkyl-alkan-1-amine.
    # Applies to the ACYCLIC path only (cyclic large-ring aza-replacement is
    # governed by P-22.1.3 and is checked separately above via
    # _try_cyclic_replacement_name, which is never reached by this gate).
    # N–N bonds (polyazane: NNN → triazane), N–O (hydroxylamine), N–S
    # (sulfonamide) and aromatic N are NOT blocked (they carry !$([NX3]~[!#6])
    # and !a guards in the pattern).
    # ----------------------------------------------------------------
    if (_ACYCLIC_AMINE_PATTERN is not None
            and mol.HasSubstructMatch(_ACYCLIC_AMINE_PATTERN)):
        return None

    # ----------------------------------------------------------------
    # Gate 2b (BBR-PERC/DEF-3, Phase 169.7): no prefix-only characteristic-group
    # atoms. Azide / diazo / nitroso / nitrite / nitro / N-oxide heteroatoms are
    # characteristic groups (P-59 / P-65.5 / P-61), NOT chain skeletal atoms —
    # skeletal replacement must not walk them into an aza/oxa chain (e.g.
    # CN=[N+]=[N-] -> wrong '2,3-diazabutane'; should be 'azidomethane' via the
    # substitutive azido prefix). This is the STRUCTURAL gate (CONTEXT D-04):
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
    # terminates the chain (P-51.4.1.3).
    # ----------------------------------------------------------------
    backbone = _find_replacement_chain(mol, exclude_atoms=acid_oxygens)
    if backbone is None:
        return None

    # ----------------------------------------------------------------
    # For terminal OH: strip the OH oxygen from the backbone.
    # The terminal O-H is NOT a chain atom -- it's a functional suffix.
    # The chain consists only of C and embedded heteroatoms.
    # OCCOCCOCC backbone: O-C-C-O-C-C-O-C-C -> strip terminal O
    #   -> chain = C-C-O-C-C-O-C-C (8 atoms = octane)
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
    # Gate 3b (P-15.4.3.1 / P-51.4.1.4 terminator rule): the chain must be
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
        # λ-bearing terminal parents (P-45.3.2 territory) are not built here.
        if nonstandard_bonding_number(mol, backbone[end_pos]) is not None:
            return None
        hetero_terminal_mode = True

    # ----------------------------------------------------------------
    # Gate 4: atom coverage. Every heavy atom must be a backbone atom, a
    # suffix oxygen (-ol / -oic acid), or part of a SIMPLE unbranched alkyl
    # substituent named as a prefix on the fixed numbering (P-15.4.3.2.2:
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
    # R4 / P-12.1 / P-63.2.4: simple O-ether chains are named substitutively
    # (alkoxy prefix), NOT by skeletal 'oxa' replacement.
    #
    # A single embedded O (no terminal-OH suffix) is always a plain ether ->
    # the substitutive namer produces '1-ethoxypropane', '1-ethoxybutane', etc.
    # (The chain-length threshold for "single-hetero-long-chain" in
    # _qualifies_for_pin_skeletal_replacement is a legacy P-15.4 gate for
    # non-O heteroatoms; O-ethers are explicitly substitutive per P-63.2.4.)
    #
    # Similarly, exactly 2 embedded O-ethers with no terminal-OH -> substitutive
    # ('1,2-dimethoxyethane', '1,2-diethoxyethane').  A 2-O chain that DOES
    # carry a principal characteristic group (terminal -ol) keeps replacement
    # ('3,6-dioxaoctan-1-ol') because the suffix anchors the replacement parent.
    #
    # Non-O single heteroatom (thia, aza, sila, ...) and >= 3 O-ethers keep
    # the existing skeletal-replacement path.
    # ----------------------------------------------------------------
    # NEW-class qualification (P-51.4.1.1): a suffix-bearing, heteroatom-
    # terminated, or substituted heterochain is the PIN class ONLY with >=4
    # heterounits alongside at least one chain carbon — anything short of
    # that belongs to the substitutive namers and fails closed here. The
    # classic unsubstituted embedded-heteroatom path below keeps its
    # established gates byte-identically.
    # ----------------------------------------------------------------
    _special_mode = acid_mode or hetero_terminal_mode or substituted_mode
    if _special_mode and not _strict_heterounit_chain_ok(mol, backbone):
        return None

    if not has_terminal_oh and not _special_mode:
        _emb_elements = [sym for _, sym in embedded_heteroatoms]
        _emb_O_count = _emb_elements.count('O')
        _all_O = (_emb_O_count == len(_emb_elements))
        _has_carbon = any(
            mol.GetAtomWithIdx(a).GetAtomicNum() == 6 for a in backbone
        )
        if _has_carbon and _all_O and _emb_O_count in (1, 2):
            return None

    # ----------------------------------------------------------------
    # DD5 SEN-02 / Fix C (P-41 Table 4.1 cls 40 > 41/42; P-15.4.3.2.2): a carbon
    # skeleton is SENIOR to ether/sulfide. For a plain acyclic chain with EXACTLY
    # TWO embedded heteroatoms, both divalent chalcogen ether/sulfide links
    # (O/S/Se/Te), a carbon present to be the parent, and NO terminal-OH suffix
    # integration, the substitutive carbon-parent name is the PIN
    # (COCSC -> methoxy(methylsulfanyl)methane; glyme COCCOC -> 1,2-dimethoxyethane),
    # NOT skeletal replacement.
    #
    # SCOPED to exactly-2 MIXED chalcogens (>= 2 DISTINCT elements), carbon-bearing:
    #   - the carbon-over-ether/sulfide PIN is unambiguous for a MIXED chain
    #     (COCSC O+S -> methoxy(methylsulfanyl)methane);
    #   - 1 heteroatom keeps single-hetero-long-chain (4-thiaheptane);
    #   - >= 3 keeps skeletal (2,5,8-trioxanonane / triglyme);
    #   - a non-chalcogen replacement driver (N/P/Si/Ge/... -> 2-oxa-4-azapentane,
    #     silyl cages) keeps skeletal;
    #   - a carbon-less chain ([O-]SS[O-]) has no carbon parent -> keeps skeletal;
    #   - terminal-OH polyether-ol (3,6,9-trioxadecan-1-ol, has_terminal_oh) exempt.
    #   - homogeneous 1-O or 2-O chain (ether/diether) handled by R4 block above.
    if not has_terminal_oh and not _special_mode and len(embedded_heteroatoms) == 2:
        _CHALCOGEN_LINK = {'O', 'S', 'Se', 'Te'}
        _elements = {sym for _, sym in embedded_heteroatoms}
        _has_carbon = any(
            mol.GetAtomWithIdx(a).GetAtomicNum() == 6 for a in backbone
        )
        if (_has_carbon and len(_elements) >= 2
                and _elements <= _CHALCOGEN_LINK):
            return None

    # ----------------------------------------------------------------
    # Gate 5 (Phase 154.A D-03): strict IUPAC P-15.4.1.2 PIN trigger.
    # Replaces the legacy single-hetero chain-len < 6 reject with explicit
    # branch labels. Rationale string is for debug logging + 154-AUDIT-A.md
    # evidence trail.
    # ----------------------------------------------------------------
    if not _special_mode:
        qualifies, _rationale = _qualifies_for_pin_skeletal_replacement(
            backbone, mol)
        if not qualifies:
            return None
    # NOTE: _rationale (">=4-same-kind", ">=3-mixed-kind", "undue-complexity",
    # "single-hetero-long-chain", "two-hetero-substitutive-equivalent") is
    # currently unused but available for debug logging via:
    # logger.debug("skeletal_replacement: trigger_branch=%s smiles=%s",
    #              _rationale, Chem.MolToSmiles(mol))

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
    # P-15.4.3.2.4: double/triple bonds get locants per the FIXED heterochain
    # numbering (2,4,6,8-tetrasiladec-9-ene). The scan fails closed on any
    # multiple bond that is not a clean C=C / C#C between consecutive backbone
    # atoms — WITHOUT it the builder emitted the SATURATED stem for an
    # unsaturated chain (structure loss, caught only by the downstream RT
    # gate). Suffix + ene integration (P-15.4.3.2.3 interplay) is not built
    # yet — fail closed rather than guess the composite numbering.
    # ----------------------------------------------------------------
    unsat = _backbone_unsaturation(mol, backbone, ignore_atoms=acid_oxygens)
    if unsat is None:
        return None
    ene_locants, yne_locants = unsat
    if (ene_locants or yne_locants) and (has_terminal_oh or acid_mode):
        return None

    # Rebuild heteroatom positions after reorientation. A non-standard-valence
    # embedded heteroatom carries the λ-convention (P-21.2.4); standard valences
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
    # Determine the suffix: terminal -ol, or the P-51.4.1.3 acid forms on
    # the fixed heterochain numbering (the acid carbon can hold the HIGH
    # locant — 3,6,9,12-tetraoxapentadecan-15-oic acid, P-59.2.2).
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
    # numbering, P-15.4.3.2.2)
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


# Phase 154.A D-05: terminal -amine / -thiol support DEFERRED to v19.
# 154-AUDIT-A.md §4 corpus tally:
#   acyclic-terminal-amine candidates: 275 mined, 2 eligible (no priority FG)
#   acyclic-terminal-thiol candidates: 25 mined, 0 eligible (no priority FG)
# Threshold per CONTEXT D-05 is 5 corpus compounds per FG; both below
# threshold => v19 follow-ups IM-154-D05-amine / IM-154-D05-thiol.
# Effective true-positive count is 0 cpd benefit because the 2 amine
# candidates also carry phosphate priority FGs that gate-2 already rejects;
# extending gate-3 with `_detect_terminal_amine` / `_detect_terminal_thiol`
# does not unblock any v18 RT failures.
# Source: 154-CONTEXT.md D-05; 154-AUDIT-A.md §4.


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
            # P-21.2.4 — 2,5λ4,8,11-tetrathiadodecane), not a thiol; genuine
            # thiols are already rejected by Gate 2's [SX2H] pattern.
            return True

    return False


# DD2 Fix A.1 (Phase D, P-63.3/P-63.4): chalcogens whose mutual single bond is a
# peroxide / disulfide / thioperoxol linkage (-O-O-, -S-S-, -Se-Se-, -Te-Te-, and the
# mixed -S-O- / -O-S- of the thioperoxol family). Such a bond is a characteristic
# group (named substitutively or with the -peroxol/-thioperoxol suffix, P-63.3/P-63.4),
# NEVER two adjacent skeletal `oxa`/`thia` replacement atoms — so skeletal replacement
# must not walk it (e.g. CCCCOO -> wrong '2-oxahexane'; PIN 'butane-1-peroxol').
_CHALCOGEN_ATOMIC_NUMS = frozenset({8, 16, 34, 52})  # O, S, Se, Te


def _dichalcogen_bond_set(mol: Chem.Mol) -> set:
    """Return the set of {a, b} index frozensets for every divalent
    chalcogen-chalcogen single bond in *mol* (peroxide / disulfide / thioperoxol
    linkages, P-63.3 / P-63.4).

    A bond qualifies when BOTH endpoints are divalent chalcogens (O/S/Se/Te,
    no double/triple/aromatic bond, neutral, the ``-X-`` ether-oxidation state)
    joined by a single bond. This is a STRUCTURAL graph property derived from
    the molecule itself (CONTEXT D-04 pattern), NOT a per-FG SMARTS blocklist —
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

    # DD2 Fix A.1: never traverse a peroxide/disulfide/thioperoxol
    # chalcogen-chalcogen bond — it is a characteristic group, not a skeletal
    # `oxa`/`thia` linkage. Omitting it from the adjacency stops the longest
    # skeletal chain at the first chalcogen, leaving the group intact for the
    # substitutive / -peroxol suffix path.
    _veto_bonds = _dichalcogen_bond_set(mol)

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
    (OH) must receive the lowest possible locant per IUPAC P-14.7.

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

    P-15.4.3.2.4: double/triple bonds take locants from the FIXED heterochain
    numbering. Fail-closed conditions (any -> None, never a lossy name):
      * an aromatic/exotic bond order anywhere in the molecule;
      * a multiple bond not between two CONSECUTIVE backbone atoms;
      * a multiple bond involving a heteroatom (C=N / S=O etc. are
        characteristic groups or λ-territory, not chain ene/yne).
    A fully saturated chain returns ``([], [])`` — byte-identical downstream.
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
    preferred by the shared ``compare_numbering`` comparator (DD4 / v22 E1):

      1. lowest heteroatom locant SET, kind-agnostic (P-15.4.3.2.1); then
      2. on a positional tie, the lowest locant to the element highest in the
         element-seniority order (P-15.4.1.2) — e.g. ``COCSC`` (positions
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
        # λ (P-21.2.4.1/2) -> suffix (P-15.4.3.2.3) -> ene/yne
        # (P-15.4.3.2.4 / P-31.1.2.2.2) -> substituent prefixes (P-14.4(f)).
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
        # (Gate 3b, P-51.4.1.4) carries its own 'a' prefix and must steer the
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

    Follows IUPAC P-15.4.3.1: replacement terms are cited in ascending
    locant order. When different elements share the same lowest locant
    (rare), alphabetical order of the replacement term breaks the tie.

    Args:
        chain_length: Total number of atoms in the backbone.
        heteroatom_positions: List of (locant, element_symbol) tuples.
        suffix: Optional (suffix_name, locant) tuple for terminal FG,
                e.g., ('ol', 1) for terminal alcohol.
        lambda_by_locant: Optional {locant: bonding_number} for embedded
                heteroatoms whose valence is non-standard (P-21.2.4 / P-31.1.4.2).
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

    # P-15.4.3.1 / P-22.2.3: cite the 'a' prefixes in the ELEMENT SENIORITY
    # order of P-15.4.1.2 (O > S > Se > Te > N > P > ... > Si > ...), NOT by
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
        # P-51.4.1.3 acid suffixes on the fixed heterochain numbering:
        # "3,6,9,12-tetraoxatetradecanedioic acid" (terminal diacid — no
        # locants needed) / "3,6,9,12-tetraoxapentadecan-15-oic acid"
        # (P-59.2.2 — the heteroatoms own the low locants).
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

    # P-15.4.3.2.4 ene/yne endings on the fixed heterochain numbering:
    # 2,4,6,8-tetrasiladec-9-ene / ...deca-2,4-diene / ...dec-1-en-9-yne.
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

    Per IUPAC P-22.1.3, large heterocyclic rings (>= 7 members) can use
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
    # after heteroatom locants (per IUPAC P-31.1.3.4).
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

            # P-22.2.3 (mirrors P-15.4.3.2.1): on a positional tie, low
            # locants go to the element cited first in the seniority order
            # (O before N — 1,4,10,13-tetraoxa-7,16-diazacyclooctadecane,
            # NOT 1,10-diaza-...). Key = per-element locant lists in
            # seniority order, compared lexicographically.
            by_element: Dict[str, List[int]] = defaultdict(list)
            for loc, sym in positions:
                by_element[sym].append(loc)
            seniority_key = [
                sorted(by_element[sym])
                for sym in sorted(by_element,
                                  key=lambda s: _A_CITATION_INDEX.get(s, 99))
            ]

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

            # Comparison key: heteroatom locant set, then element seniority,
            # then DB locants
            comparison_key = (hetero_locant_set, seniority_key, db_locants)

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

    # P-15.4.3.1 / P-22.2.3: cite the 'a' prefixes in the ELEMENT SENIORITY
    # order of P-15.4.1.2 (O > S > Se > Te > N > P > ... > Si > ...), NOT by
    # ascending locant. BB verbatim: "3-phospha-2,5,7-trisilaoctane" (phospha
    # cited first although sila holds locant 2); "8-thia-2,4,6-trisiladecane".
    sorted_groups = sorted(
        element_groups.items(),
        key=lambda item: _A_CITATION_INDEX.get(item[0], 99)
    )

    total_hetero = sum(len(locs) for _, locs in sorted_groups)
    # P-22.2.3.2.1: a single ring heteroatom is assigned locant '1', which is
    # OMITTED from the name (unless an indicated-hydrogen locant is present).
    # The omission applies ONLY to the SATURATED ring (all_single): when the
    # ring carries unsaturation (ene/yne locants are cited), the heteroatom
    # locant '1' is retained as the reference for those locants
    # (1-azacyclopentadeca-2,4,6,8,10,12,14-heptaene, NOT azacyclopentadeca-...).
    elide_single = total_hetero == 1 and all_single

    parts = []
    for symbol, locants in sorted_groups:
        term = REPLACEMENT_TERMS[symbol]
        locant_str = '' if elide_single else ','.join(str(loc) for loc in locants)
        count = len(locants)

        if count == 1:
            multiplier = ''
        elif count in SIMPLE_MULTIPLIERS:
            multiplier = SIMPLE_MULTIPLIERS[count]
        else:
            multiplier = SIMPLE_MULTIPLIERS.get(count, f'{count}')

        sep = '' if elide_single else '-'
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
    becomes 'octene', not 'octaene'). Since get_chain_prefix() returns
    stems without trailing 'a' (e.g., 'oct', 'dec'), and we build the
    suffix directly, no special elision is needed for most cases.

    For single double bond: '-{locant}-ene'
    For 2 double bonds: '-{loc1},{loc2}-diene'  (with linking 'a')
    For 3 double bonds: '-{loc1},{loc2},{loc3}-triene' (with linking 'a')

    Args:
        replacement_prefix: Heteroatom prefix (e.g., '1-oxa').
        chain_prefix: Ring size prefix from get_chain_prefix() (e.g., 'hept').
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
    # get_chain_prefix() returns e.g., 'hept', 'oct', 'dec' (no trailing 'a')
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
