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


def n_substituted_carboxamide(mol, start_idx: int, core_atoms: Set[int], chain_atoms: List[int]) -> Optional[Dict]:
    """The suffix record of an N-substituted carboxamide whose carbon ``start_idx`` is joined to the fused core
    ``core_atoms``: ``{'name': 'carboxamide',..., 'n_substituents': [...]}``, or None (another shape, or a nitrogen
    branch the namer cannot name: the whole record declines rather than drop a substituent)."""
    if not _one_n_group_owner(mol, core_atoms):
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
    (two carboxamides with N groups would need N and N' locants, which this producer does not write)."""
    return bool(suffix_groups.get('amine')) or len(suffix_groups.get('carboxamide') or ()) == 1


def _one_n_group_owner(mol, core_atoms: Set[int]) -> bool:
    """True when exactly one N-substituted carboxamide sits on the core and no other exocyclic nitrogen of the core
    carries a carbon group: the italic-N prefixes then belong to that amide alone (with two, N and N' locants or a
    multiplicative name would be needed)."""
    core = set(core_atoms)
    rings = [set(r) for r in mol.GetRingInfo().AtomRings() if set(r) <= core]
    if not any(len(a & b) >= 2 for i, a in enumerate(rings) for b in rings[i + 1:]):
        return False               # not a fused core (a lone benzene is the amide producers' parent)
    if any(a.GetFormalCharge() for a in mol.GetAtoms()):
        return False               # an ion or a salt: an anion is senior and the ion producers name it
    from ..assembly.strict_ring_prefix import principal_group
    if principal_group(mol)[0] not in ('secondary_amide', 'tertiary_amide'):
        return False               # a senior group (a sulfonic acid of a neutralised salt, an acid, an ester) holds the suffix
    acyl = _compiled_smarts('[CX3](=[O,S])[O,N,S,F,Cl,Br,I]')
    if acyl is not None and len({m[0] for m in mol.GetSubstructMatches(acyl)}) != 1:
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


def fused_heterocycle_n_amide(mol, start_idx: int, excluded: Set[int], chain_atoms: List[int]) -> Optional[Dict]:
    """:func:`n_substituted_carboxamide` in the record form of ``rules.fused_rings`` (a 'suffix' branch whose N groups
    join the producer's italic-N table)."""
    if _oxo_or_amino_on(mol, excluded):
        return None     # the producer's oxo branch and its amine N table do not take an amide's N groups: as before
    found = n_substituted_carboxamide(mol, start_idx, excluded, chain_atoms)
    if found is None:
        return None
    prefix = carbamoyl_prefix(found['n_substituents'])
    if prefix is None:
        return None
    _ask_pin_tier()     # the wider tiers may name the molecule by another route first; the PIN tier's PIN wins
    return {'name': prefix, 'atoms': chain_atoms, 'functional_group': 'amide', 'type': 'suffix',
            'suffix_name': 'carboxamide', 'n_substituents': found['n_substituents']}


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

