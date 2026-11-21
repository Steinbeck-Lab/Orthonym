"""N-substituted carboxamides on a fused ring system (2026-11-20).

 (the Blue Book: "The suffix 'carboxamide' is always used to name amides with the -CO-NH2 group
attached to a ring, ring system") and (:32774, N-Substitution): an amide R-CO-NHR' or R-CO-NR'R" on a
ring is named with the suffix 'carboxamide' and its nitrogen substituents cited as prefixes with the locant N
('N,N-dimethylbenzamide'). The fused-ring producer (``rules.polycyclics``) matched only -CONH2, so an N-substituted
amide on naphthalene, quinoline, indole or indazole got no PIN ('N-methylnaphthalene-2-carboxamide' was shipped as
'N-methylbicyclo[4.4.0]deca-1(10),2,4,6,8-pentaene-3-carboxamide'). The rule here is the heterocycle producer's
(``rules.heterocycles``: the same SMARTS, the hydrazide and hydroxamic shapes left out, and a fail-closed count of the
nitrogen's branches against the names ``benzene._detect_n_substituents`` gives)."""
from typing import Dict, List, Optional, Set

from ..perception.smarts_cache import compiled as _compiled_smarts

_N_AMIDE_SMARTS = ('[CX3](=O)[NX3;H1;+0][#6]', '[CX3](=O)[NX3;H0;+0]([#6])[#6]')


def n_substituted_carboxamide(mol, start_idx: int, core_atoms: Set[int], chain_atoms: List[int],
                               several: bool = False) -> Optional[Dict]:
    """The suffix record of an N-substituted carboxamide whose carbon ``start_idx`` is joined to the fused core
    ``core_atoms``: ``{'name': 'carboxamide',..., 'n_substituents': [...]}``, or None (another shape, or a nitrogen
    branch the namer cannot name: the whole record declines rather than drop a substituent). ``several``: the caller
    cites each amide's N groups with its own N-locant (``rules.benzene_polyamide.polycyclic_polyamide_prefixes``), so
    two or more carboxamides of the core may carry N groups."""
    if not _one_n_group_owner(mol, core_atoms, several):
        return None
    for smarts in _N_AMIDE_SMARTS:
        pattern = _compiled_smarts(smarts)
        for match in (mol.GetSubstructMatches(pattern) if pattern is not None else ()):
            if match[0] != start_idx:
                continue
            n_atom = mol.GetAtomWithIdx(match[2])
            if any(nb.GetSymbol() in ('N', 'O') and nb.GetIdx() != start_idx
                   and mol.GetBondBetweenAtoms(n_atom.GetIdx(), nb.GetIdx()).GetBondTypeAsDouble() == 1.0
                   for nb in n_atom.GetNeighbors()):
                return None          # a hydrazide or a hydroxamic acid: not this suffix
            if any(nb.GetIdx() in core_atoms for nb in n_atom.GetNeighbors()):
                return None          # a lactam of the core: the ring itself
            from .benzene import _detect_n_substituents
            n_subs = _detect_n_substituents(mol, match, set(core_atoms))
            expected = sum(1 for nb in n_atom.GetNeighbors() if nb.GetAtomicNum() > 1 and nb.GetIdx() != start_idx)
            if not n_subs or len(n_subs) != expected:
                _ask_pin_tier()
                return None
            return {'name': 'carboxamide', 'atoms': chain_atoms, 'is_suffix': True,
                    'suffix_type': 'n_substituted_amide', 'n_substituents': list(n_subs)}
    return None


def cites_n_prefixes(suffix_groups: Dict[str, List[int]]) -> bool:
    """True when the italic-N prefixes of the name belong to its suffix: a promoted amine, or exactly one carboxamide
    or sulfonamide (two with N groups would need N and N' locants, which this producer does not write)."""
    return bool(suffix_groups.get('amine')) or any(len(suffix_groups.get(k) or ()) == 1 for k in ('carboxamide', 'sulfonamide'))


def _one_n_group_owner(mol, core_atoms: Set[int], several: bool = False) -> bool:
    """True when exactly one N-substituted carboxamide sits on the core and no other exocyclic nitrogen of the core
    carries a carbon group: the italic-N prefixes then belong to that amide alone (with two, N and N' locants or a
    multiplicative name would be needed). With ``several``, every acyl group of the molecule is a carboxamide on the
    core and at least one carries N groups: each amide's N groups take its N-locant."""
    core = set(core_atoms)
    rings = [set(r) for r in mol.GetRingInfo().AtomRings() if set(r) <= core]
    if not any(len(a & b) >= 2 for i, a in enumerate(rings) for b in rings[i + 1:]):
        return False               # not a fused core (a lone benzene is the amide producers' parent)
    if any(a.GetFormalCharge() for a in mol.GetAtoms()):
        return False               # an ion or a salt: an anion is senior and the ion producers name it
    from ..assembly.strict_ring_prefix import principal_group
    if principal_group(mol)[0] not in ('secondary_amide', 'tertiary_amide') + (('primary_amide',) if several else ()):
        return False               # a senior group (a sulfonic acid of a neutralised salt, an acid, an ester) holds the suffix
    acyl = _compiled_smarts('[CX3](=[O,S])[O,N,S,F,Cl,Br,I]')
    acyl_carbons = {m[0] for m in mol.GetSubstructMatches(acyl)} if acyl is not None else set()
    if several and len(acyl_carbons) >= 2:
        return _all_acyls_are_core_carboxamides(mol, core, acyl_carbons)
    if len(acyl_carbons) != 1:
        return False               # another acid, ester, amide, anhydride or acyl halide: the core may not be the parent
    owners = 0
    for smarts in _N_AMIDE_SMARTS:
        pattern = _compiled_smarts(smarts)
        for match in (mol.GetSubstructMatches(pattern) if pattern is not None else ()):
            if any(nb.GetIdx() in core for nb in mol.GetAtomWithIdx(match[0]).GetNeighbors()):
                owners += 1
    for idx in core:
        for nb in mol.GetAtomWithIdx(idx).GetNeighbors():
            if (nb.GetIdx() not in core and nb.GetSymbol() == 'N' and not nb.GetIsAromatic()
                    and any(x.GetSymbol() == 'C' and x.GetIdx() != idx for x in nb.GetNeighbors())
                    and not any(x.GetSymbol() == 'C' and x.GetIdx() != idx and any(
                        b.GetBondTypeAsDouble() == 2.0 and b.GetOtherAtom(x).GetSymbol() == 'O' for b in x.GetBonds())
                        for x in nb.GetNeighbors())):
                return False       # an N-substituted amine of the core: its N prefixes would mix with the amide's
    return owners == 1


def _all_acyls_are_core_carboxamides(mol, core: Set[int], acyl_carbons: Set[int]) -> bool:
    """True when two or more acyl carbons exist, each a carboxamide carbon (C(=O)N, the N not in a ring, not joined to
    N or O) bonded to the core, and at least one amide N carries a group; no other exocyclic N of the core carries
    carbon (its N prefixes would mix with the amides')."""
    amide = _compiled_smarts('[CX3](=[OX1])[NX3;!R;!$(N[N,O])]')
    amide_carbons = {m[0]: m[2] for m in mol.GetSubstructMatches(amide)} if amide is not None else {}
    if len(acyl_carbons) < 2 or not acyl_carbons <= set(amide_carbons):
        return False
    if not all(any(nb.GetIdx() in core for nb in mol.GetAtomWithIdx(c).GetNeighbors()) for c in acyl_carbons):
        return False
    if not any(nb.GetAtomicNum() > 1 and nb.GetIdx() != c for c in acyl_carbons
               for nb in mol.GetAtomWithIdx(amide_carbons[c]).GetNeighbors()):
        return False
    return not any(nb.GetIdx() not in core and nb.GetSymbol() == 'N' and nb.GetIdx() not in amide_carbons.values()
                   for idx in core for nb in mol.GetAtomWithIdx(idx).GetNeighbors())


def fused_heterocycle_n_amide(mol, start_idx: int, excluded: Set[int], chain_atoms: List[int]) -> Optional[Dict]:
    """:func:`n_substituted_carboxamide` in the record form of ``rules.fused_rings`` (a 'suffix' branch whose N groups
    join the producer's italic-N table)."""
    if _oxo_or_amino_on(mol, excluded):
        return None     # the producer's oxo branch and its amine N table do not take an amide's N groups: as before
    found = n_substituted_carboxamide(mol, start_idx, excluded, chain_atoms, several=True)
    if found is None:
        return None
    prefix = carbamoyl_prefix(found['n_substituents'])
    if prefix is None:
        return None
    _ask_pin_tier()     # the wider tiers may name the molecule by another route first; the PIN tier's PIN wins
    return {'name': prefix, 'atoms': chain_atoms, 'functional_group': 'amide', 'type': 'suffix',
            'suffix_name': 'carboxamide', 'n_substituents': found['n_substituents']}


def fused_n_sources(mol, substituents: Dict, atom_mapping: Dict) -> Optional[List]:
    """The italic-N prefix sources ``('n', name, locants)`` of ``rules.fused_rings._assemble_fused_heterocycle_name``:
    a bare 'N' per group, or, for two or more ring carboxamides of which a nitrogen carries a group, the N-locant of
    each group's own nitrogen, the Blue Book: 'N1-methylbutanediamide (PIN)'). None (decline)
    when a symmetry of the ring system that keeps every other locant would give the nitrogens lower locants;
    the producer does not renumber for them)."""
    counts = substituents.get('n_substituents') or {}
    by_locant = {loc: names for loc, names in (substituents.get('n_by_locant') or {}).items() if names}
    amides = (substituents.get('suffix_groups') or {}).get('carboxamide') or []
    if len(amides) < 2 or not by_locant:
        return [('n', name, ['N'] * count) for name, count in counts.items()]
    if sum(len(v) for v in by_locant.values()) != sum(counts.values()):
        return None                     # an N group of another nitrogen: not one amide's
    from .benzene_polyamide import _symmetry_images
    from ..assembly.naming_utils import alpha_sort_key
    ring = {a: loc for a, loc in atom_mapping.items() if mol.GetAtomWithIdx(a).IsInRing()}
    atom_of = {loc: a for a, loc in ring.items()}
    fixed = [list(v) for v in (substituents.get('suffix_groups') or {}).values()]
    fixed += [list(v) for v in (substituents.get('c_substituents') or {}).values()]
    fixed += [[o.get('locant')] for o in substituents.get('other') or ()]
    fixed += [list(substituents.get('oxo_substituents') or ()), list(substituents.get('amino_substituents') or ())]
    if any(loc not in atom_of for locs in fixed for loc in locs) or any(loc not in atom_of for loc in by_locant):
        return None

    def order(loc):
        text = str(loc)
        digits = ''.join(ch for ch in text if ch.isdigit())
        return (int(digits or 0), text)

    def key(image):
        moved = {image[atom_of[loc]]: names for loc, names in by_locant.items()}
        all_n = sorted(order(loc) for loc, names in moved.items() for _ in names)
        first = tuple(tuple(sorted(order(loc) for loc, names in moved.items() if nm in names))
                      for nm in sorted({n for names in moved.values() for n in names}, key=alpha_sort_key))
        return (all_n, first)

    here = key(dict(ring))
    for image in _symmetry_images(mol, ring):
        keeps = all(sorted(map(order, (image[atom_of[loc]] for loc in locs))) == sorted(map(order, locs))
                    for locs in fixed)
        if keeps and key(image) < here:
            return None
    sources: Dict[str, List[str]] = {}
    for loc in sorted(by_locant, key=order):
        for name in by_locant[loc]:
            sources.setdefault(name, []).append(f"N{loc}")
    return [('n', name, locs) for name, locs in sources.items()]


def carbamoyl_prefix(n_substituents: List[str]) -> Optional[str]:
    """The prefix form of the amide when the ring is itself a substituent ('methylcarbamoyl', 'dimethylcarbamoyl',
    'benzyl(methyl)carbamoyl', '(propan-2-yl)carbamoyl'):, the N groups as prefixes of 'carbamoyl', the first
    cited bare unless compound and the next in parentheses; None for more than two groups."""
    from ..assembly.naming_utils import alpha_sort_key, enclose_if_compound
    names = sorted(n_substituents, key=alpha_sort_key)
    if len(names) == 1:
        return f"{enclose_if_compound(names[0])}carbamoyl"
    if len(names) == 2 and names[0] == names[1]:
        from ..assembly.naming_utils import multiplied_component
        return f"{multiplied_component(2, names[0], enclose_if_compound(names[0]))}carbamoyl"
    if len(names) == 2:
        return f"{enclose_if_compound(names[0])}({names[1]})carbamoyl"
    return None


def _ask_pin_tier() -> None:
    """An N-substituted carboxamide whose N group this rule cannot name (a ring group with a stereocentre is left to
    the amide producers, ``benzene._detect_n_substituents``): the fused producer of a wider tier then cites the amide as a
    'carbamoyl' prefix, a lower class than the amide suffix, while the PIN tier may name it through the
    decomposition ('N-[(1S)-1-phenylethyl]naphthalene-2-carboxamide'). The PIN tier is asked, for the whole input only
    (``assembly.pin_first.note_pin_tier_asked``; a part of a mixture is a nested call)."""
    from ..assembly.fragment_naming import name_scope_depth
    if name_scope_depth() <= 1:
        from ..assembly.pin_first import note_pin_tier_asked
        note_pin_tier_asked()


def _oxo_or_amino_on(mol, core_atoms: Set[int]) -> bool:
    """True when a ring atom of the fused core carries an exocyclic =O, =S or =N, or an amino nitrogen that is no amide
    nitrogen: ``rules.fused_rings`` names such a core through its oxo branch or its amine suffix, which do not take the
    N groups of an amide (measured 2026-11-20: '4-(dimethylcarbamoyl)isoquinolin-1-amine' and a pyrido[1,2-a]pyrimidin-4-
    one carboxamide were left to a von Baeyer name)."""
    core = set(core_atoms)
    for idx in core:
        atom = mol.GetAtomWithIdx(idx)
        for bond in atom.GetBonds():
            nb = bond.GetOtherAtom(atom)
            if nb.GetIdx() in core:
                continue
            if bond.GetBondTypeAsDouble() == 2.0 and nb.GetSymbol() in ('O', 'S', 'N'):
                return True
            if (nb.GetSymbol() == 'N' and nb.GetFormalCharge() == 0 and not nb.GetIsAromatic()
                    and not any(x.GetSymbol() == 'C' and any(b.GetBondTypeAsDouble() == 2.0 and b.GetOtherAtom(x).GetSymbol()
                                                             in ('O', 'S') for b in x.GetBonds())
                                for x in nb.GetNeighbors() if x.GetIdx() != idx)):
                return True
    return False

