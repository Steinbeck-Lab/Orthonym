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


def _common_refusal(mol) -> Optional[str]:
    """Engine-wide scope refusals shared by the chain and ring paths."""
    if mol is None:
        return "no mol"
    if Chem.GetFormalCharge(mol) != 0:
        return "net charge (G3 scope)"
    if len(Chem.GetMolFrags(mol)) > 1:
        return "multi-fragment (G3 scope)"
    if any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        return "radical"
    if any(a.GetIsotope() for a in mol.GetAtoms()):
        return "isotope"
    return None


def name_general_chain(mol, features) -> Optional[GeneralEngineResult]:
    """Name a chain-parented molecule with a full atom->token partition.

    Returns None on ANY condition outside the verified G1 scope.
    """
    reason = _common_refusal(mol)
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
    return _assemble(mol, features, chain, part)


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


def _assemble(mol, features, chain, part) -> Optional[GeneralEngineResult]:
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
        prefix = name_substituent(mol, frag, attach_nbrs[0])
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
    # v25 G4: parent-scope stereo from structure (substituent-internal
    # stereo is already handled inside name_substituent's stereo route).
    name = _stereo_prefix(
        mol, {a: atom_to_locant[a] for a in chain}) + name
    bindings.append(TokenBinding(tuple(chain), parent_token, 'parent'))
    return GeneralEngineResult(name=name, bindings=tuple(bindings))


# Ring suffix forms (get_suffix(pg, is_ring=True) values) -- all carry locants
# on a ring parent; 'appended' matches _build_parent_with_unsaturation types.
_RING_SUFFIX_STYLES = {
    'ol': 'inline', 'one': 'inline', 'amine': 'inline', 'thiol': 'inline',
    'carboxylic acid': 'appended', 'carbaldehyde': 'appended',
    'carbonitrile': 'appended', 'carboxamide': 'appended',
}


def name_general_ring(mol, features) -> Optional[GeneralEngineResult]:
    """v25 G2: universal von-Baeyer ring-parent path (opt-in engine only)."""
    from ..rules.vonbaeyer_universal import analyze_cage_universal
    from ..rules.polycyclic import _build_parent_with_unsaturation
    from ..rules.ring_selection import select_principal_ring_system
    from ..rules.seniority import get_suffix
    from .substituent_enumerator import discover_substituents, name_substituent

    reason = _common_refusal(mol)
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

    cage = analyze_cage_universal(mol, cage_atoms=cage_seed)
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
    if pg:
        suffix_core = get_suffix(pg, is_ring=True)
        if suffix_core not in _RING_SUFFIX_STYLES:
            return _refuse(f"unsupported ring suffix for pg={pg!r}")
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
    # 5-bond-carbon / caffeine class). Mancude cages are already refused upstream
    # (analyze_cage_universal), so this only guards a residual saturated
    # isolated-ene cage that also carries a ring ketone. Fail-closed.
    if suffix_core == 'one':
        _ene = set(cage.unsaturation.get('double_bonds', ()))
        if _ene & set(pg_locants):
            return _refuse("ring ketone locant coincides with ring double bond "
                           "(valence)")

    # --- substituent prefixes (generic ordered-atom discovery + recursion) ---
    ordered_cage = sorted(cage_set, key=lambda i: atom_to_locant[i])
    try:
        subs = discover_substituents(
            mol, cage_set | suffix_atoms, parent_type='chain',
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
        prefix = name_substituent(mol, frag, attach_nbrs[0])
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

    # v25 G4: parent-scope stereo from structure (VB locants).
    name = _stereo_prefix(mol, atom_to_locant) + name

    bindings.append(TokenBinding(tuple(cage.cage_atoms), cage.descriptor,
                                 'parent'))
    return GeneralEngineResult(name=name, bindings=tuple(bindings))


def name_general(mol, features) -> Optional[GeneralEngineResult]:
    """v25 engine dispatcher: chain parent -> G1 path, ring parent -> G2 path."""
    ring_atoms = any(a.IsInRing() for a in mol.GetAtoms()) if mol else False
    if not ring_atoms or getattr(features, 'chain_is_parent', False):
        return name_general_chain(mol, features)
    return name_general_ring(mol, features)
