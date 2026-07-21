"""v25 G1: general substitutive chain namer with atom->token bindings.

NEW code (design: , G1).
Unlike the legacy composer path, every emission carries a TokenBinding
partition over ALL heavy atoms, so the E1 certificate
(validation/e1_certificate.py) can verify no atom was silently dropped --
Java-free. Output is OPT-IN (namer general_fallback flag); it re-enters the
existing moat (>15-HA gate, P10 vetoes, SELF-01 OPSIN-RT) downstream.

G1 scope: neutral, single-fragment, chain-parented molecules with an
all-carbon parent chain of length >= 2 and a suffix from the supported set
(or none). Ring parents -> G2 (general ring fallback). Charged -> G3.
Anything outside scope REFUSES (returns None): fail-closed, never partial.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from rdkit import Chem

logger = logging.getLogger(__name__)

# Suffix support keyed by the SUFFIX STRING (robust to fg-name spelling):
# 'terminal' suffixes sit on chain terminus C1 (locants omitted / dioic);
# 'locant' suffixes carry explicit locants.
_SUPPORTED_SUFFIX_STYLES = {
    'oic acid': 'terminal', 'al': 'terminal', 'nitrile': 'terminal',
    'amide': 'terminal',
    'one': 'locant', 'ol': 'locant', 'amine': 'locant', 'thiol': 'locant',
}

_MULT_SIMPLE = {2: 'di', 3: 'tri', 4: 'tetra', 5: 'penta', 6: 'hexa',
                7: 'hepta', 8: 'octa', 9: 'nona', 10: 'deca'}
_MULT_COMPLEX = {2: 'bis', 3: 'tris', 4: 'tetrakis', 5: 'pentakis',
                 6: 'hexakis'}
# Complex prefix (needs bis/tris + enclosure): contains locants, hyphens,
# brackets, or an internal multiplying prefix.
_COMPLEX_PREFIX_RE = re.compile(r"[0-9\-\(\)\[\]]")


@dataclass(frozen=True)
class TokenBinding:
    """Atoms expressed by one emitted name token."""
    atom_ids: Tuple[int, ...]
    token: str
    role: str  # 'parent' | 'prefix' | 'suffix'


@dataclass(frozen=True)
class GeneralEngineResult:
    name: str
    bindings: Tuple[TokenBinding, ...]


def _refuse(reason: str) -> None:
    logger.info("general_engine refused: %s", reason)
    return None


def _common_refusal(mol, allow_charged: bool = False) -> Optional[str]:
    """Engine-wide scope refusals shared by the chain and ring paths.

    v26 P5: ``allow_charged`` (set only under ``complete`` /
    ``allow_aromatic_general``) lifts the net-charge refusal so the charged
    general path can emit a ``-ylium``/``-ide``/``-uide``/``-ium`` suffix on the
    numbered parent (``_charge_suffix_text`` + fail-closed). The multi-fragment,
    radical (radical-cation) and isotope refusals STAY even when charge is
    allowed -- those remain out of scope for ``complete``.
    """
    if mol is None:
        return "no mol"
    if not allow_charged and Chem.GetFormalCharge(mol) != 0:
        return "net charge (G3 scope)"
    if len(Chem.GetMolFrags(mol)) > 1:
        return "multi-fragment (G3 scope)"
    if any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        return "radical"
    if any(a.GetIsotope() for a in mol.GetAtoms()):
        return "isotope"
    return None


def _charge_suffix_text(mol, atom_to_locant) -> Optional[str]:
    """v26 P5 (BB P-73 cations / P-74 anions): the charge-suffix string for
    skeletal charge(s) on the ALREADY-NUMBERED general parent --
    ``-1-ium`` / ``-2-ylium`` / ``-1-ide`` / ``-1-uide`` (or the multiplied
    ``-1,4-diium`` / ``-1,2-diylium`` / ``-1,4-diide`` forms).

    The locant is the ACTUAL charged atom's parent locant, so the emitted
    descriptor is structurally faithful on its own (SELF-01 is only the
    backstop, which fails OPEN without Java). Reuses the proven ion perception
    (``get_ion_sites``) + classifiers (``classify_cation`` / ``classify_anion``)
    rather than reinventing charge typing.

    FAIL CLOSED (return None -> the caller abstains, never a wrong/neutral name)
    on any charge that cannot be faithfully expressed as a suffix on THIS parent:
      * a charge on a SUBSTITUENT atom (not in the parent numbering);
      * an FG-anchored anion (alkoxide/thiolate/carboxylate/sulfonate/...) whose
        charge sits on an off-parent oxygen -- the PIN charged path owns those;
      * diazonium / acylium cations -- the PIN path owns those;
      * a multiply-charged single atom, mixed sign centres, mixed suffix kinds,
        an internal-only (P-59 nitro/azide/N-oxide/diazo) charge, or a
        multiplicity beyond the simple table.
    """
    from ..perception.ions import get_ion_sites
    from ..rules.ions import classify_cation, classify_anion

    sites = get_ion_sites(mol)  # excludes internal P-59 charges
    anions = list(sites.get('anions') or [])
    cations = list(sites.get('cations') or [])
    if bool(anions) == bool(cations):
        # neither (internal-only net charge) or BOTH (mixed-sign) -> out of scope
        return None
    charged = anions or cations
    negative = bool(anions)
    parent_atoms = set(atom_to_locant)

    per_locant_base: List[Tuple[int, str]] = []
    for site in charged:
        idx = site['atom_idx']
        if idx not in parent_atoms:
            return None  # charge on a substituent -> not a parent suffix
        if abs(int(site.get('charge', 0))) != 1:
            return None  # multiply-charged single atom -> tight scope
        if negative:
            acls = classify_anion(mol, site)
            if acls in ('carbanion', 'heteroatom_hydride_anion'):
                base = 'ide'          # P-72.2.2.1: loss of H+ from a skeletal atom
            elif acls == 'uide_anion':
                base = 'uide'         # P-72.3: hydride ADDED to a skeletal atom
            else:
                return None           # FG anion -> PIN path owns it
        else:
            ccls = classify_cation(mol, site)
            if ccls == 'ylium':
                base = 'ylium'        # P-73.2.2.1.1: loss of H- from a skeletal C
            elif ccls in ('aminium', 'onium', 'quaternary'):
                base = 'ium'          # P-73.1: protonated / substituted skeletal heteroatom
            else:
                return None           # diazonium / acylium -> PIN path owns it
        per_locant_base.append((atom_to_locant[idx], base))

    bases = {b for _, b in per_locant_base}
    if len(bases) != 1:
        return None                   # mixed suffix kinds on one parent
    base = next(iter(bases))
    locants = sorted(loc for loc, _ in per_locant_base)
    n = len(locants)
    if n == 1:
        mult = ''
    else:
        mult = _MULT_SIMPLE.get(n)
        if mult is None:
            return None
    return '-' + ','.join(map(str, locants)) + '-' + mult + base


def _append_charge_suffix(name: str, mol, atom_to_locant,
                          has_fg_suffix: bool) -> Optional[str]:
    """Splice the P5 charge suffix onto an assembled parent ``name`` (pre-stereo).

    Elides a single trailing parent 'e' only when the charge-suffix text
    begins with a vowel (``cyclohexane``->``cyclohexan-1-ide``,
    ``1-methylpyridine``->``1-methylpyridin-1-ium``, ``...pentaene``->
    ``...pentaen-4-ium``): the single-charge bases (``-ium``/``-ylium``/
    ``-ide``/``-uide``) start with i/y/u. The MULTIPLIED forms
    (``-1,4-diium``/``-1,4-diide``/...) begin with the consonant of the
    multiplier (di/tri/...), so the terminal 'e' must be RETAINED
    (P-16.3.3), e.g. ``1,4-diazine``->``1,4-diazine-1,4-diium`` (NOT
    ``diazin-1,4-diium``). Returns the charged name, or None to FAIL CLOSED
    (a charge that is not expressible, or a co-occurring FG suffix -- the
    cumulative FG+charge construction is out of P5 scope)."""
    if has_fg_suffix:
        return None  # FG suffix + skeletal charge (cumulative) -> out of P5 scope
    cs = _charge_suffix_text(mol, atom_to_locant)
    if cs is None:
        return None
    first_alpha = next((c for c in cs if c.isalpha()), '')
    stem = name[:-1] if name.endswith('e') and first_alpha in 'aeiouy' else name
    return stem + cs


def name_general_chain(
    mol, features, allow_charged: bool = False, allow_mancude: bool = False,
) -> Optional[GeneralEngineResult]:
    """Name a chain-parented molecule with a full atom->token partition.

    Returns None on ANY condition outside the verified G1 scope.

    v26 P5: ``allow_charged`` (only under ``complete``) lifts the net-charge
    refusal and emits a ``-ide``/``-ylium``/``-ium``/``-uide`` suffix on the
    numbered chain parent when the charge sits on a chain skeletal atom
    (fail-closed otherwise).

    v27 P1: ``allow_mancude`` (complete/best-effort tier only) lets a multi-ring
    cage SUBSTITUENT on the chain be named via the universal von-Baeyer engine
    (parent<->substituent symmetry). Default False -> PIN path byte-identical.
    """
    reason = _common_refusal(mol, allow_charged=allow_charged)
    if reason:
        return _refuse(reason)

    ring_atoms = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    chain = list(getattr(features, 'principal_chain', None) or ())
    if ring_atoms and not getattr(features, 'chain_is_parent', False):
        return _refuse("ring parent (ring path owns it)")
    if len(chain) < 2:
        return _refuse("chain too short")
    if any(mol.GetAtomWithIdx(i).GetSymbol() != 'C' for i in chain):
        return _refuse("hetero parent chain (replacement nomenclature)")

    part = _partition(mol, features, chain)
    if part is None:
        return None
    return _assemble(mol, features, chain, part, allow_charged=allow_charged,
                     allow_mancude=allow_mancude)


def _partition(mol, features, chain) -> Optional[dict]:
    """Split heavy atoms into chain / suffix / substituent fragments.

    Suffix atoms (PG-match atoms off the chain) are folded into the
    blocked set BEFORE substituent discovery so a suffix oxygen is never
    double-expressed as a 'hydroxy'/'oxo' prefix.
    """
    from .substituent_enumerator import discover_substituents
    from ..rules.seniority import get_suffix

    chain_set = set(chain)
    pg = getattr(features, 'principal_group', None)
    pg_matches = list(getattr(features, 'principal_group_atoms', None) or [])

    atom_to_locant = dict(getattr(features, 'atom_to_locant', None)
                          or {a: i + 1 for i, a in enumerate(chain)})
    if set(chain) - set(atom_to_locant):
        return _refuse("chain atom missing from atom_to_locant")

    suffix_core = None
    suffix_atoms: set = set()
    pg_chain_locants: List[int] = []
    if pg:
        suffix_core = get_suffix(pg, is_ring=False)
        if suffix_core not in _SUPPORTED_SUFFIX_STYLES:
            return _refuse(f"unsupported suffix for pg={pg!r}")
        seen = set()
        for match in pg_matches:
            key = tuple(sorted(match))
            if key in seen:
                continue
            seen.add(key)
            on_chain = [i for i in match if i in chain_set]
            if not on_chain:
                return _refuse("PG instance not on parent chain")
            suffix_atoms.update(i for i in match
                                if i not in chain_set
                                and mol.GetAtomWithIdx(i).GetAtomicNum() > 1)
            pg_chain_locants.append(min(atom_to_locant[i] for i in on_chain))

    try:
        subs = discover_substituents(
            mol, chain_set | suffix_atoms, parent_type='chain',
            principal_chain=chain, atom_to_locant=atom_to_locant)
    except AssertionError as e:
        return _refuse(f"partition incomplete: {e}")

    return {
        'suffix_core': suffix_core,
        'suffix_atoms': frozenset(suffix_atoms),
        'pg_locants': sorted(pg_chain_locants),
        'substituents': subs,
        'atom_to_locant': atom_to_locant,
    }


def _alpha_key(prefix: str) -> str:
    """Alphabetization key: letters only; sec-/tert- excluded, iso/neo/cyclo
    included (IUPAC P-14.5.2)."""
    p = prefix
    for skip in ('tert-', 'sec-'):
        if p.startswith(skip):
            p = p[len(skip):]
    return re.sub(r"[^a-z]", "", p.lower())


def _mult_prefix(n: int, name: str) -> Optional[str]:
    """'2,2-' + this -> 'dimethyl' / 'bis(2-chloroethyl)'. None if n too big."""
    if n == 1:
        return f"({name})" if _COMPLEX_PREFIX_RE.search(name) else name
    table = _MULT_COMPLEX if _COMPLEX_PREFIX_RE.search(name) else _MULT_SIMPLE
    if n not in table:
        return None
    return (f"{table[n]}({name})" if table is _MULT_COMPLEX
            else f"{table[n]}{name}")


def _stereo_prefix(mol, atom_to_locant) -> str:
    """Parent-scope stereodescriptor block from STRUCTURE (rdCIPLabeler via
    collect_stereodescriptors' idempotent guard). '' when achiral."""
    from ..rules.stereochemistry import (
        collect_stereodescriptors, format_stereodescriptor_string,
    )
    return format_stereodescriptor_string(
        collect_stereodescriptors(mol, atom_to_locant))


def _stem_block(mol, chain, atom_to_locant) -> Optional[Tuple[str, str]]:
    """(parent_token, hydride_block) e.g. ('but', 'but-2-ene') or
    ('hex', 'hexane'). None on an unsupported bond pattern."""
    from ..data.chain_names import get_chain_prefix

    base = get_chain_prefix(len(chain))
    if not base:
        return None
    ene, yne = [], []
    pos = {a: atom_to_locant[a] for a in chain}
    for i in range(len(chain) - 1):
        bond = mol.GetBondBetweenAtoms(chain[i], chain[i + 1])
        if bond is None:
            return None
        order = bond.GetBondTypeAsDouble()
        loc = min(pos[chain[i]], pos[chain[i + 1]])
        if order == 2.0:
            ene.append(loc)
        elif order == 3.0:
            yne.append(loc)
        elif order != 1.0:
            return None  # aromatic/dative chain bond: out of scope
    ene.sort()
    yne.sort()
    if not ene and not yne:
        return base, base + "ane"
    block = base
    if ene:
        mult = _MULT_SIMPLE.get(len(ene), '') if len(ene) > 1 else ''
        if len(ene) > 1:
            block += 'a'
        block += '-' + ','.join(map(str, ene)) + '-' + mult + 'ene'
    if yne:
        if ene:
            block = block[:-1]  # 'ene' -> 'en' before '-N-yne'
        mult = _MULT_SIMPLE.get(len(yne), '') if len(yne) > 1 else ''
        if len(yne) > 1 and not ene:
            block += 'a'
        block += '-' + ','.join(map(str, yne)) + '-' + mult + 'yne'
    return base, block


def _suffix_block(style: str, core: str, locants: List[int]) -> Optional[str]:
    """Suffix text WITHOUT the elision decision: '-1-ol', '-1,2-diol',
    'oic acid', 'dial'. None if unsupported multiplicity/placement."""
    n = len(locants)
    if n == 0:
        return None
    if style == 'terminal':
        if n == 1:
            if locants != [1]:
                return None  # terminal suffix must sit on C1
            return core
        if n == 2:
            return 'di' + core  # e.g. 'dioic acid', 'dial' (locants implicit)
        return None
    mult = '' if n == 1 else _MULT_SIMPLE.get(n)
    if n > 1 and mult is None:
        return None
    return '-' + ','.join(map(str, sorted(locants))) + '-' + (mult or '') + core


def _assemble(mol, features, chain, part,
              allow_charged: bool = False,
              allow_mancude: bool = False) -> Optional[GeneralEngineResult]:
    from .substituent_enumerator import name_substituent

    atom_to_locant = part['atom_to_locant']

    stem = _stem_block(mol, chain, atom_to_locant)
    if stem is None:
        return _refuse("unsupported chain bond pattern")
    parent_token, hydride = stem

    # --- substituent prefixes (recursion reuse) ---
    groups: Dict[str, List[int]] = {}
    frag_bindings: List[TokenBinding] = []
    for sub in part['substituents']:
        frag = set(sub.frag_atoms)
        attach_nbrs = [n.GetIdx() for n in
                       mol.GetAtomWithIdx(sub.attach_mol_idx).GetNeighbors()
                       if n.GetIdx() in frag]
        if not attach_nbrs:
            return _refuse("substituent without chain attachment")
        # v27 P1: complete-tier cage substituent recursion (see name_general_ring).
        prefix = name_substituent(mol, frag, attach_nbrs[0],
                                  allow_mancude=allow_mancude)
        if not prefix or prefix == 'substituent':
            return _refuse("branch unnameable (tier-5 fallback)")
        groups.setdefault(prefix, []).append(sub.locant)
        frag_bindings.append(TokenBinding(tuple(sorted(frag)), prefix,
                                          'prefix'))

    prefix_parts = []
    for prefix in sorted(groups, key=_alpha_key):
        locs = sorted(groups[prefix])
        text = _mult_prefix(len(locs), prefix)
        if text is None:
            return _refuse("multiplicity beyond table")
        prefix_parts.append(','.join(map(str, locs)) + '-' + text)

    # --- suffix ---
    bindings = frag_bindings
    suffix_text = ''
    if part['suffix_core']:
        style = _SUPPORTED_SUFFIX_STYLES[part['suffix_core']]
        suffix_text = _suffix_block(style, part['suffix_core'],
                                    part['pg_locants'])
        if suffix_text is None:
            return _refuse("unsupported suffix placement/multiplicity")
        if part['suffix_atoms']:
            bindings.append(TokenBinding(tuple(sorted(part['suffix_atoms'])),
                                         part['suffix_core'], 'suffix'))

    # --- elision: 'ane' + vowel-initial suffix -> 'an' + suffix ---
    body = hydride
    if suffix_text:
        first_alpha = next((c for c in suffix_text if c.isalpha()), '')
        if body.endswith('e') and first_alpha in 'aeiouy':
            body = body[:-1]
        body += suffix_text

    # Prefix text abuts the stem directly ('3-ethyl-2,2-dimethylhexane').
    name = ('-'.join(prefix_parts) + body) if prefix_parts else body
    # v26 P5: charge suffix on a chain skeletal atom (fail closed otherwise).
    if allow_charged and Chem.GetFormalCharge(mol) != 0:
        name = _append_charge_suffix(
            name, mol, {a: atom_to_locant[a] for a in chain},
            has_fg_suffix=bool(part['suffix_core']))
        if name is None:
            return _refuse("charge not expressible as a chain-parent suffix")
    # v25 G4: parent-scope stereo from structure (substituent-internal
    # stereo is already handled inside name_substituent's stereo route).
    name = _stereo_prefix(
        mol, {a: atom_to_locant[a] for a in chain}) + name
    bindings.append(TokenBinding(tuple(chain), parent_token, 'parent'))
    return GeneralEngineResult(name=name, bindings=tuple(bindings))


# Ring suffix forms (get_suffix(pg, is_ring=True) values) -- all carry locants
# on a ring parent; 'appended' matches _build_parent_with_unsaturation types.
#
# v27 P2: widened for the high-enrichment linker/suffix groups that previously
# forced the ring engine to abstain. All are gated on the engine tier
# (allow_aromatic_general) via name_general_ring/_monocycle, so the PIN default
# is byte-identical; each emission is SELF-01-verified downstream.
#   * 'sulfonamide' / 'carboximidamide' (amidine) attach directly to the ring
#     carbon (like -carboxamide) -> 'appended' (P-65.3.1 / P-66.4.1).
#   * 'imine' is the aza-'-one' (P-66.3) -> 'inline'; it shares the ketone
#     valence guard (see the suffix_core in ('one','imine') check below).
_RING_SUFFIX_STYLES = {
    'ol': 'inline', 'one': 'inline', 'amine': 'inline', 'thiol': 'inline',
    'imine': 'inline',
    'carboxylic acid': 'appended', 'carbaldehyde': 'appended',
    'carbonitrile': 'appended', 'carboxamide': 'appended',
    'sulfonamide': 'appended', 'carboximidamide': 'appended',
    # v27 P2 (P-65.6.3.2.1): ester ring PIN is the functional-class TWO-WORD
    # `<R-yl> <ring>carboxylate`. The parent block is built with the 'appended'
    # carboxylate suffix on the acid core; the alcoholic `R-yl ` word is
    # prepended by name_general_ring (see _extract_ring_ester). Only the clean
    # acyclic mono-ester is built here; lactones / aryl / poly-esters fail closed.
    'carboxylate': 'appended',
}


def _extract_ring_ester(mol, pg_matches, ring_set, allow_mancude):
    """v27 P2 (P-65.6.3.2.1): decompose a ring carboxylic-acid ESTER for the
    functional-class two-word PIN ``<R-yl> <ring>carboxylate``.

    Returns ``(r_word, r_frag_atoms, acid_core_atoms, ring_attach_atom)`` for
    the single clean RING-ACID mono-ester with an ACYCLIC alcohol, or ``None``
    (fail closed) for any of: more than one ester (polyester), a reverse/aryl
    ester (the ring is on the alcohol side, so the acid C is not bonded to the
    ring), a lactone / ring-bearing R, or an R the substituent namer declines.
    Never guesses the acid side.
    """
    from .substituent_enumerator import name_substituent

    seen: set = set()
    matches = []
    for m in pg_matches:
        k = tuple(sorted(m))
        if k in seen:
            continue
        seen.add(k)
        matches.append(set(m))
    if len(matches) != 1:
        return None  # mono-ester only; polyester fails closed (Phase 4+)
    match = matches[0]

    # Identify the acid core: a match C bearing one =O and one single-bond O.
    carbonyl_c = carbonyl_o = ester_o = None
    for i in match:
        a = mol.GetAtomWithIdx(i)
        if a.GetSymbol() != 'C':
            continue
        dbl_o = single_o = None
        for b in a.GetBonds():
            o = b.GetOtherAtom(a)
            if o.GetSymbol() != 'O':
                continue
            if b.GetBondType() == Chem.BondType.DOUBLE:
                dbl_o = o.GetIdx()
            elif b.GetBondType() == Chem.BondType.SINGLE:
                single_o = o.GetIdx()
        if dbl_o is not None and single_o is not None:
            carbonyl_c, carbonyl_o, ester_o = i, dbl_o, single_o
            break
    if carbonyl_c is None:
        return None

    # The acid carbon must attach to the ring (else the ring is the alcohol side
    # -> reverse/aryl ester, deferred). Exactly one ring neighbour.
    ring_nbrs = [n.GetIdx()
                 for n in mol.GetAtomWithIdx(carbonyl_c).GetNeighbors()
                 if n.GetIdx() in ring_set]
    if len(ring_nbrs) != 1:
        return None
    ring_attach = ring_nbrs[0]

    # R (alcoholic component): the ester-O neighbour that is NOT the acid C.
    r_starts = [n.GetIdx()
                for n in mol.GetAtomWithIdx(ester_o).GetNeighbors()
                if n.GetIdx() != carbonyl_c]
    if len(r_starts) != 1:
        return None
    r_start = r_starts[0]
    if mol.GetAtomWithIdx(r_start).IsInRing():
        return None  # lactone / aryl ester -> fail closed (acyclic R only)

    core = {carbonyl_c, carbonyl_o, ester_o}
    r_frag: set = set()
    stack = [r_start]
    while stack:
        x = stack.pop()
        if x in r_frag or x in core:
            continue
        r_frag.add(x)
        for n in mol.GetAtomWithIdx(x).GetNeighbors():
            if n.GetIdx() not in r_frag and n.GetIdx() not in core:
                stack.append(n.GetIdx())
    if not r_frag or (r_frag & ring_set):
        return None

    r_word = name_substituent(mol, r_frag, r_start, allow_mancude=allow_mancude)
    if not r_word or r_word == 'substituent':
        return None
    return r_word, r_frag, core, ring_attach


def name_general_ring(
    mol, features, allow_aromatic_general: bool = False,
    allow_charged: bool = False,
) -> Optional[GeneralEngineResult]:
    """v25 G2: universal von-Baeyer ring-parent path (opt-in engine only).

    v26 P0: ``allow_aromatic_general`` is threaded to
    ``analyze_cage_universal(..., allow_mancude=...)`` (plumbing only; the
    mancude refusal there still fires unconditionally until P2).

    v26 P5: ``allow_charged`` (only under ``complete``) lifts the net-charge
    refusal and emits a charge suffix on a von-Baeyer cage skeletal atom
    (``...pentaen-4-ium``); fail-closed otherwise.
    """
    from ..rules.vonbaeyer_universal import analyze_cage_universal
    from ..rules.polycyclic import _build_parent_with_unsaturation
    from ..rules.ring_selection import select_principal_ring_system
    from ..rules.seniority import get_suffix
    from .substituent_enumerator import discover_substituents, name_substituent

    reason = _common_refusal(mol, allow_charged=allow_charged)
    if reason:
        return _refuse(reason)
    ring_atoms = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    if not ring_atoms:
        return _refuse("acyclic (chain path owns it)")
    if getattr(features, 'chain_is_parent', False):
        return _refuse("chain parent (chain path owns it)")

    ring_systems = list(getattr(features, 'ring_systems', None) or [])
    if ring_systems:
        senior = select_principal_ring_system(mol, ring_systems)
        cage_seed = set(senior) if senior else None
    else:
        cage_seed = None

    cage = analyze_cage_universal(
        mol, cage_atoms=cage_seed, allow_mancude=allow_aromatic_general)
    if cage is None:
        return _refuse("cage unanalyzable (monocycle/spiro/caps/kekulize)")

    cage_set = set(cage.cage_atoms)
    atom_to_locant = dict(cage.atom_to_locant)

    # --- suffix (ring forms; every PG instance must touch the cage) ---
    pg = getattr(features, 'principal_group', None)
    pg_matches = list(getattr(features, 'principal_group_atoms', None) or [])
    suffix_core = None
    suffix_atoms: set = set()
    pg_locants: List[int] = []
    ester_r_word = None          # v27 P2: alcoholic `R-yl ` word (functional class)
    ester_r_frag: set = set()    # v27 P2: R-alkyl atoms held out of discovery
    if pg:
        suffix_core = get_suffix(pg, is_ring=True)
        if suffix_core not in _RING_SUFFIX_STYLES:
            return _refuse(f"unsupported ring suffix for pg={pg!r}")
        if suffix_core == 'carboxylate':
            # v27 P2 (P-65.6.3.2.1): ring ester -> functional-class two-word.
            # The acid core (C=O, ester O) becomes the appended 'carboxylate'
            # suffix on the ring; the alcoholic R is prepended as a word.
            er = _extract_ring_ester(
                mol, pg_matches, cage_set, allow_aromatic_general)
            if er is None:
                return _refuse("ester not a clean acyclic ring-acid mono-ester")
            ester_r_word, ester_r_frag, _acid_core, _ring_attach = er
            suffix_atoms.update(_acid_core)
            pg_locants.append(atom_to_locant[_ring_attach])
        else:
            seen = set()
            for match in pg_matches:
                key = tuple(sorted(match))
                if key in seen:
                    continue
                seen.add(key)
                on_cage = [i for i in match if i in cage_set]
                if on_cage:
                    loc = min(atom_to_locant[i] for i in on_cage)
                else:
                    # appended suffix (e.g. -carboxylic acid): match sits fully
                    # off-cage; locant = the cage neighbor of any match atom.
                    nbrs = [n.GetIdx()
                            for i in match
                            for n in mol.GetAtomWithIdx(i).GetNeighbors()
                            if n.GetIdx() in cage_set]
                    if not nbrs:
                        return _refuse("PG instance not attached to cage")
                    loc = min(atom_to_locant[n] for n in nbrs)
                pg_locants.append(loc)
                suffix_atoms.update(i for i in match
                                    if i not in cage_set
                                    and mol.GetAtomWithIdx(i).GetAtomicNum() > 1)

    # v25 G5-A defense-in-depth: a ring '-one' locant must never coincide with a
    # ring double-bond locant -- that carbon would be both =ring and =O (the
    # 5-bond-carbon / caffeine class). v26 P2 lifted the aromatic-cage refusal in
    # analyze_cage_universal behind ``allow_mancude`` (--emit-tier complete), so
    # mancude cages NOW reach this guard as well as saturated isolated-ene cages.
    # The guard keys on ``double_bond_pairs`` (populated for BOTH the kekulized
    # mancude polyene and the isolated ene), so it holds for both classes -- do
    # not weaken it on the stale "mancude refused upstream" premise. Fail-closed.
    if suffix_core == 'one':
        # both endpoints of every ring double bond are termini a =O cannot share
        _ene_termini = {loc for pair in cage.unsaturation.get('double_bond_pairs', ())
                        for loc in pair}
        if _ene_termini & set(pg_locants):
            return _refuse("ring ketone locant coincides with ring double bond "
                           "(valence)")

    # --- substituent prefixes (generic ordered-atom discovery + recursion) ---
    ordered_cage = sorted(cage_set, key=lambda i: atom_to_locant[i])
    try:
        subs = discover_substituents(
            mol, cage_set | suffix_atoms | ester_r_frag, parent_type='chain',
            principal_chain=ordered_cage, atom_to_locant=atom_to_locant)
    except AssertionError as e:
        return _refuse(f"partition incomplete: {e}")

    groups: Dict[str, List[int]] = {}
    frag_bindings: List[TokenBinding] = []
    for sub in subs:
        frag = set(sub.frag_atoms)
        attach_nbrs = [n.GetIdx() for n in
                       mol.GetAtomWithIdx(sub.attach_mol_idx).GetNeighbors()
                       if n.GetIdx() in frag]
        if not attach_nbrs:
            return _refuse("substituent without cage attachment")
        # v27 P1: under the complete/best-effort tier (allow_aromatic_general),
        # a multi-ring cage substituent is named via the universal von-Baeyer
        # engine (parent<->substituent symmetry). PIN default (flag off) is
        # byte-identical — name_substituent's allow_mancude defaults False.
        prefix = name_substituent(mol, frag, attach_nbrs[0],
                                  allow_mancude=allow_aromatic_general)
        if not prefix or prefix == 'substituent':
            return _refuse("branch unnameable (tier-5 fallback)")
        groups.setdefault(prefix, []).append(sub.locant)
        frag_bindings.append(TokenBinding(tuple(sorted(frag)), prefix,
                                          'prefix'))

    prefix_parts = []
    for prefix in sorted(groups, key=_alpha_key):
        locs = sorted(groups[prefix])
        text = _mult_prefix(len(locs), prefix)
        if text is None:
            return _refuse("multiplicity beyond table")
        prefix_parts.append(','.join(map(str, locs)) + '-' + text)

    # --- parent block: hetero-prefix + descriptor + parent(+ene)(+suffix) ---
    fg_suffix = None
    bindings = frag_bindings
    if suffix_core:
        fg_suffix = {'suffix': suffix_core, 'locants': sorted(pg_locants),
                     'type': _RING_SUFFIX_STYLES[suffix_core]}
        if suffix_atoms:
            bindings.append(TokenBinding(tuple(sorted(suffix_atoms)),
                                         suffix_core, 'suffix'))
    parent_block = _build_parent_with_unsaturation(
        cage.total_atoms, cage.unsaturation, fg_suffix=fg_suffix)

    # Hyphen glue (mirrors polycyclic.py:2830): a substituent prefix keeps its
    # joining '-' only before a locant-initial hetero prefix.
    core = cage.hetero_prefix + cage.descriptor + parent_block
    if prefix_parts:
        joined = '-'.join(prefix_parts)
        head = cage.hetero_prefix or cage.descriptor
        name = joined + ('-' if head[:1].isdigit() else '') + core
    else:
        name = core

    # v26 P5: charge suffix on a cage skeletal atom (fail closed otherwise).
    if allow_charged and Chem.GetFormalCharge(mol) != 0:
        name = _append_charge_suffix(name, mol, atom_to_locant,
                                     has_fg_suffix=bool(suffix_core))
        if name is None:
            return _refuse("charge not expressible as a cage-parent suffix")

    # v25 G4: parent-scope stereo from structure (VB locants).
    name = _stereo_prefix(mol, atom_to_locant) + name

    # v27 P2 (P-65.6.3.2.1): prepend the alcoholic component as a separate word
    # -> `ethyl <ring>carboxylate` (functional-class ester two-word PIN).
    if ester_r_word is not None:
        name = ester_r_word + ' ' + name
        bindings.append(TokenBinding(tuple(sorted(ester_r_frag)),
                                     ester_r_word, 'prefix'))

    bindings.append(TokenBinding(tuple(cage.cage_atoms), cage.descriptor,
                                 'parent'))
    return GeneralEngineResult(name=name, bindings=tuple(bindings))


def _ring_suffix_text(core: str, locants: List[int]) -> Optional[str]:
    """Ring suffix WITHOUT the parent-elision decision: '-3-ol', '-2,5-diol',
    '-2-carboxylic acid', '-1,3-dicarboxylic acid'. None on unsupported
    multiplicity. The parent-'e' elision is applied by the caller from the
    first alphabetic char of this string (mirrors chain ``_assemble``)."""
    n = len(locants)
    if n == 0:
        return None
    mult = '' if n == 1 else _MULT_SIMPLE.get(n)
    if n > 1 and mult is None:
        return None
    return '-' + ','.join(map(str, sorted(locants))) + '-' + (mult or '') + core


def _bond_ring_locant(la: int, lb: int, n: int) -> int:
    """Ring-bond locant for a bond between ring positions ``la`` and ``lb``
    (1..n): the lower endpoint, except the wraparound bond {n, 1} which is
    cited as ``n`` (P-31.1.4.3 lowest-locant numbering already handled by the
    orienter; this only formats the chosen orientation)."""
    if {la, lb} == {1, n}:
        return n
    return min(la, lb)


def _orient_carbocycle(mol, ring_order, sub_positions, pg_ring_atoms):
    """Number an all-carbon monocycle by lowest locants to, in order:
    principal group -> ring unsaturation -> substituents -> canonical rank
    (P-14.4 / P-31.1.4). Returns (oriented_ring, atom_to_locant).

    The canonical-rank final tier is a deterministic symmetry-breaker; among
    orientations that tie on every IUPAC criterion the choice is round-trip
    equivalent (SELF-01 is the downstream authority for the emitted string)."""
    ring_list = list(ring_order)
    n = len(ring_list)
    canon = list(Chem.CanonicalRankAtoms(mol, breakTies=True))
    best_key = None
    best_oriented = None
    for start in range(n):
        for direction in (1, -1):
            oriented = [ring_list[(start + direction * k) % n] for k in range(n)]
            loc = {a: i + 1 for i, a in enumerate(oriented)}
            pg = sorted(loc[i] for i in pg_ring_atoms if i in loc)
            uns = []
            for i in range(n):
                a, b = oriented[i], oriented[(i + 1) % n]
                bond = mol.GetBondBetweenAtoms(a, b)
                if bond is not None and bond.GetBondTypeAsDouble() in (2.0, 3.0):
                    uns.append(_bond_ring_locant(loc[a], loc[b], n))
            uns.sort()
            sub = sorted(loc[i] for i in sub_positions if i in loc)
            key = (pg, uns, sub, [canon[a] for a in oriented])
            if best_key is None or key < best_key:
                best_key = key
                best_oriented = oriented
    return best_oriented, {a: i + 1 for i, a in enumerate(best_oriented)}


def _carbocycle_parent_name(mol, oriented, atom_to_locant) -> Optional[str]:
    """Parent name for an all-carbon monocycle: 'benzene' (6-membered
    aromatic), 'cyclohexane' (saturated), or 'cyclohex-1-ene' style
    (unsaturated). None -> fail closed (aromatic non-6, unnameable stem)."""
    from ..data.chain_names import get_chain_prefix
    n = len(oriented)
    ring_set = set(oriented)
    if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in oriented):
        return 'benzene' if n == 6 else None
    base = get_chain_prefix(n)
    if not base:
        return None
    ene, yne = [], []
    for i in range(n):
        a, b = oriented[i], oriented[(i + 1) % n]
        bond = mol.GetBondBetweenAtoms(a, b)
        if bond is None:
            return None
        order = bond.GetBondTypeAsDouble()
        if order == 1.0:
            continue
        loc = _bond_ring_locant(atom_to_locant[a], atom_to_locant[b], n)
        if order == 2.0:
            ene.append(loc)
        elif order == 3.0:
            yne.append(loc)
        else:
            return None  # aromatic/dative bond in a non-aromatic ring: bail
    ene.sort()
    yne.sort()
    if not ene and not yne:
        return 'cyclo' + base + 'ane'
    stem = 'cyclo' + base
    if ene:
        mult = _MULT_SIMPLE.get(len(ene), '') if len(ene) > 1 else ''
        if len(ene) > 1:
            stem += 'a'
        stem += '-' + ','.join(map(str, ene)) + '-' + mult + 'ene'
    if yne:
        if ene:
            stem = stem[:-1]  # 'ene' -> 'en' before '-N-yne'
        mult = _MULT_SIMPLE.get(len(yne), '') if len(yne) > 1 else ''
        if len(yne) > 1 and not ene:
            stem += 'a'
        stem += '-' + ','.join(map(str, yne)) + '-' + mult + 'yne'
    return stem


def name_general_monocycle(
    mol, features, allow_aromatic_general: bool = False,
    allow_charged: bool = False,
) -> Optional[GeneralEngineResult]:
    """v26 P1: general LONE-monocycle ring-parent path (opt-in engine only).

    Names a molecule whose SENIOR ring system is a single (non-fused) ring
    -- benzene, pyridine, thiophene, imidazole, ... -- with its substituents
    coming from the never-None universal recursion
    (``substituent_enumerator.name_substituent``) instead of the default
    composer's finite per-class vocabulary. That is the root-cause fix for
    "bare ring names, substituted form abstains."

    The parent ring name + numbering come from the EXISTING lowest-locant
    machinery (``name_heterocycle`` + ``orient_heterocycle_with_substituents``
    for heterocycles; ``benzene``/cycloalkane for all-carbon). Fail-closed
    (returns None) on anything outside this scope; the E1 atom-partition and
    SELF-01 OPSIN round-trip are the downstream authorities.

    Gated behind ``allow_aromatic_general`` -- inert (returns None) when the
    flag is False, so the PIN/default path stays byte-identical.
    """
    if not allow_aromatic_general:
        return None

    from ..rules.ring_selection import select_principal_ring_system
    from ..rules.seniority import get_suffix
    from ..rules.heterocycles import (
        name_heterocycle, orient_heterocycle_with_substituents,
    )
    from .substituent_enumerator import discover_substituents, name_substituent

    reason = _common_refusal(mol, allow_charged=allow_charged)
    if reason:
        return _refuse(reason)
    ring_info = mol.GetRingInfo()
    if ring_info.NumRings() == 0:
        return _refuse("acyclic (chain path owns it)")
    if getattr(features, 'chain_is_parent', False):
        return _refuse("chain parent (chain path owns it)")

    # --- identify the parent monocycle: the SENIOR ring system must be a
    #     single, non-fused ring (substituents MAY carry their own rings; the
    #     universal recursion names them). ---
    ring_systems = list(getattr(features, 'ring_systems', None) or [])
    if ring_systems:
        senior = select_principal_ring_system(mol, ring_systems)
        senior_set = set(senior) if senior else None
    else:
        senior_set = None
    atom_rings = [tuple(r) for r in ring_info.AtomRings()]
    parent_ring = None
    if senior_set is not None:
        subrings = [r for r in atom_rings if set(r) <= senior_set]
        if len(subrings) == 1 and set(subrings[0]) == senior_set:
            parent_ring = subrings[0]
    elif len(atom_rings) == 1:
        parent_ring = atom_rings[0]
    if parent_ring is None:
        return _refuse("no lone monocycle parent (fused/cage or ambiguous)")

    ring_set = set(parent_ring)

    # --- suffix (ring forms; mirror name_general_ring) ---
    pg = getattr(features, 'principal_group', None)
    pg_matches = list(getattr(features, 'principal_group_atoms', None) or [])
    suffix_core = None
    suffix_atoms: set = set()
    pg_locants: List[int] = []
    pg_ring_atoms: set = set()
    if pg:
        suffix_core = get_suffix(pg, is_ring=True)
        if suffix_core not in _RING_SUFFIX_STYLES:
            return _refuse(f"unsupported ring suffix for pg={pg!r}")

    # --- numbering (heteroatom/pg/substituent lowest-locant) + parent name ---
    sub_positions: set = set()  # filled after suffix_atoms known; provisional below
    # First pass: collect off-ring pg (suffix) atoms so they are excluded from
    # the substituent set that drives numbering.
    if pg:
        seen = set()
        for match in pg_matches:
            key = tuple(sorted(match))
            if key in seen:
                continue
            seen.add(key)
            suffix_atoms.update(i for i in match
                                if i not in ring_set
                                and mol.GetAtomWithIdx(i).GetAtomicNum() > 1)
            on_ring = [i for i in match if i in ring_set]
            pg_ring_atoms.update(on_ring)
    for i in ring_set:
        for nb in mol.GetAtomWithIdx(i).GetNeighbors():
            j = nb.GetIdx()
            if (j not in ring_set and j not in suffix_atoms
                    and nb.GetAtomicNum() > 1):
                sub_positions.add(i)
                break

    has_hetero = any(mol.GetAtomWithIdx(i).GetSymbol() != 'C' for i in ring_set)
    if has_hetero:
        oriented, atom_to_locant = orient_heterocycle_with_substituents(
            mol, parent_ring, sub_positions, pg_ring_atoms)
        parent_name = name_heterocycle(mol, parent_ring)
    else:
        oriented, atom_to_locant = _orient_carbocycle(
            mol, parent_ring, sub_positions, pg_ring_atoms)
        parent_name = _carbocycle_parent_name(mol, oriented, atom_to_locant)
    if not parent_name or 'unknown' in parent_name.lower():
        return _refuse("monocycle parent name underivable")
    if any(i not in atom_to_locant for i in ring_set):
        return _refuse("ring atom missing from numbering")

    # --- pg locants under the chosen numbering (mirror name_general_ring) ---
    if pg:
        seen = set()
        for match in pg_matches:
            key = tuple(sorted(match))
            if key in seen:
                continue
            seen.add(key)
            on_ring = [i for i in match if i in ring_set]
            if on_ring:
                loc = min(atom_to_locant[i] for i in on_ring)
            else:
                nbrs = [n.GetIdx()
                        for i in match
                        for n in mol.GetAtomWithIdx(i).GetNeighbors()
                        if n.GetIdx() in ring_set]
                if not nbrs:
                    return _refuse("PG instance not attached to ring")
                loc = min(atom_to_locant[n] for n in nbrs)
            pg_locants.append(loc)

    # A '-one' on an aromatic ring carbon needs added-hydrogen machinery this
    # path does not build (pyridin-2(1H)-one); fail closed rather than emit an
    # aromatic-carbon-with-=O impossibility.
    if suffix_core == 'one' and any(
            mol.GetAtomWithIdx(i).GetIsAromatic()
            for m in pg_matches for i in m if i in ring_set):
        return _refuse("ring ketone on aromatic carbon (added-H not built)")

    # --- substituent prefixes (universal recursion; mirror name_general_ring) ---
    ordered_ring = sorted(ring_set, key=lambda i: atom_to_locant[i])
    try:
        subs = discover_substituents(
            mol, ring_set | suffix_atoms, parent_type='chain',
            principal_chain=ordered_ring, atom_to_locant=atom_to_locant)
    except AssertionError as e:
        return _refuse(f"partition incomplete: {e}")

    groups: Dict[str, List[int]] = {}
    frag_bindings: List[TokenBinding] = []
    for sub in subs:
        frag = set(sub.frag_atoms)
        attach_nbrs = [n.GetIdx() for n in
                       mol.GetAtomWithIdx(sub.attach_mol_idx).GetNeighbors()
                       if n.GetIdx() in frag]
        if not attach_nbrs:
            return _refuse("substituent without ring attachment")
        if len(attach_nbrs) != 1:
            return _refuse(
                "substituent attaches to parent ring at >1 point "
                "(spiro/fused/bridge)")
        # v27 P1: complete-tier cage substituent recursion (see name_general_ring).
        prefix = name_substituent(mol, frag, attach_nbrs[0],
                                  allow_mancude=allow_aromatic_general)
        if not prefix or prefix == 'substituent':
            return _refuse("branch unnameable (tier-5 fallback)")
        groups.setdefault(prefix, []).append(sub.locant)
        frag_bindings.append(TokenBinding(tuple(sorted(frag)), prefix,
                                          'prefix'))

    prefix_parts = []
    for prefix in sorted(groups, key=_alpha_key):
        locs = sorted(groups[prefix])
        text = _mult_prefix(len(locs), prefix)
        if text is None:
            return _refuse("multiplicity beyond table")
        prefix_parts.append(','.join(map(str, locs)) + '-' + text)

    # --- suffix text + parent-'e' elision ---
    bindings = frag_bindings
    core = parent_name
    if suffix_core:
        suffix_text = _ring_suffix_text(suffix_core, pg_locants)
        if suffix_text is None:
            return _refuse("unsupported suffix multiplicity/placement")
        first_alpha = next((c for c in suffix_text if c.isalpha()), '')
        if core.endswith('e') and first_alpha in 'aeiouy':
            core = core[:-1]
        core = core + suffix_text
        if suffix_atoms:
            bindings.append(TokenBinding(tuple(sorted(suffix_atoms)),
                                         suffix_core, 'suffix'))

    # --- glue prefixes + core (mirror name_general_ring hyphen rule) ---
    if prefix_parts:
        joined = '-'.join(prefix_parts)
        name = joined + ('-' if core[:1].isdigit() else '') + core
    else:
        name = core

    # v26 P5: charge suffix on a ring skeletal atom (fail closed otherwise).
    if allow_charged and Chem.GetFormalCharge(mol) != 0:
        name = _append_charge_suffix(name, mol, atom_to_locant,
                                     has_fg_suffix=bool(suffix_core))
        if name is None:
            return _refuse("charge not expressible as a monocycle-parent suffix")

    # v25 G4: parent-scope stereo from structure (ring locants).
    name = _stereo_prefix(mol, atom_to_locant) + name

    # E1 parent binding: a stem guaranteed to survive suffix elision.
    parent_token = parent_name[:-1] if parent_name.endswith('e') else parent_name
    bindings.append(TokenBinding(tuple(sorted(ring_set)), parent_token, 'parent'))
    return GeneralEngineResult(name=name, bindings=tuple(bindings))


def name_general(
    mol, features, allow_aromatic_general: bool = False,
) -> Optional[GeneralEngineResult]:
    """v25 engine dispatcher: chain parent -> G1 path, ring parent -> G2 path.

    v26 P0/P1: ``allow_aromatic_general`` widens the ring producer -- it is
    threaded to ``name_general_ring`` -> ``analyze_cage_universal`` (aromatic
    cages) and enables the general lone-monocycle path
    (``name_general_monocycle``). Default False -> byte-identical to pre-P0.

    v26 P5: net charge is lifted ONLY under ``complete`` (``allow_charged`` ==
    ``allow_aromatic_general``); the charge becomes a ``-ylium``/``-ide``/
    ``-uide``/``-ium`` suffix on the numbered parent (fail-closed). Under
    ``valid`` / the PIN default (``allow_aromatic_general`` False) charge is
    still refused -> byte-identical to pre-P5.
    """
    allow_charged = allow_aromatic_general
    ring_atoms = any(a.IsInRing() for a in mol.GetAtoms()) if mol else False
    if not ring_atoms or getattr(features, 'chain_is_parent', False):
        return name_general_chain(mol, features, allow_charged=allow_charged,
                                  allow_mancude=allow_aromatic_general)
    cage_result = name_general_ring(
        mol, features, allow_aromatic_general=allow_aromatic_general,
        allow_charged=allow_charged)
    if cage_result is not None:
        return cage_result
    return name_general_monocycle(
        mol, features, allow_aromatic_general=allow_aromatic_general,
        allow_charged=allow_charged)
