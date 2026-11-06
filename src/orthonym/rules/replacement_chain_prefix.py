"""The skeletal replacement ('a') prefix of an acyclic chain substituent with four or more heterounits.

 (the Blue Book): "Skeletal replacement ('a') nomenclature rather than substitutive or multiplicative
names must be used to generate preferred IUPAC names for acyclic structures when four or more heterounits are present
in a unbranched chain";: the numbering of such a parent hydride is fixed by its heteroatoms, and the free
valence follows it: '2,5,8,11-tetraoxatridec-12-en-13-yl (preferred prefix) (not 3,6,9,12-tetraoxatridec-1-en-1-yl;
the suffix '-yl' is added to the parent hydride name 2,5,8,11-tetraoxatridecane)' (:24543).

Scope (fail closed): one unbranched saturated acyclic chain, carbon at both ends, whose other atoms are separate
(never adjacent) neutral divalent O, S, Se or Te, at least four of them, the free valence on one of its carbons,
which is bonded to a ring atom or a heteroatom (on an acyclic carbon the chain belongs to the parent,.
"""
from typing import List, Optional

from rdkit import Chem

#: a name that cites an 'a' prefix more often than OPSIN has a count word for has no reading, so it is never built here
#: and the chain is named as the other routes name it
from ..opsin_capabilities import MAX_READABLE_COUNT as _MAX_READABLE_COUNT

#: 'a' prefixes in the order of (O > S > Se > Te), each cited with its locants
_A_PREFIX = (('O', 'oxa'), ('S', 'thia'), ('Se', 'selena'), ('Te', 'tellura'))


def replacement_chain_prefix(mol, sub_atoms: List[int], attach_idx: int) -> Optional[str]:
    """'2,5,8,11-tetraoxatridecan-13-yl' for the chain ``sub_atoms`` attached at carbon ``attach_idx``, or None."""
    from ..data.chain_names import get_chain_name, multiplying_prefix
    from ..data.locant_text import render_locants
    atoms = set(sub_atoms)
    if len(atoms) < 9 or attach_idx not in atoms:
        return None
    for i in atoms:
        a = mol.GetAtomWithIdx(i)
        if (a.IsInRing() or a.GetFormalCharge() or a.GetNumRadicalElectrons() or a.GetIsotope()
                or a.GetSymbol() not in ('C', 'O', 'S', 'Se', 'Te')
                or a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED):
            return None
        if a.GetSymbol() != 'C' and (a.GetTotalNumHs() or a.GetDegree() != 2):
            return None
        if any(b.GetBondType() != Chem.BondType.SINGLE for b in a.GetBonds()):
            return None
        inner = [nb.GetIdx() for nb in a.GetNeighbors() if nb.GetIdx() in atoms]
        outer = [nb.GetIdx() for nb in a.GetNeighbors() if nb.GetIdx() not in atoms and nb.GetAtomicNum() > 1]
        if len(inner) > 2 or (outer and i != attach_idx) or len(outer) > 1:
            return None
        # on an acyclic carbon of the parent the 'a' chain would itself be the longer heteroatom chain
        if outer and mol.GetAtomWithIdx(outer[0]).GetSymbol() == 'C' and not mol.GetAtomWithIdx(outer[0]).IsInRing():
            return None
    if not _parent_outranks_the_chain(mol, atoms):
        return None
    ends = [i for i in atoms if sum(1 for nb in mol.GetAtomWithIdx(i).GetNeighbors() if nb.GetIdx() in atoms) == 1]
    if len(ends) != 2 or any(mol.GetAtomWithIdx(e).GetSymbol() != 'C' for e in ends):
        return None
    path, prev = [ends[0]], None
    while len(path) < len(atoms):
        nxt = [nb.GetIdx() for nb in mol.GetAtomWithIdx(path[-1]).GetNeighbors()
               if nb.GetIdx() in atoms and nb.GetIdx() != prev]
        if len(nxt) != 1:
            return None
        prev = path[-1]
        path.append(nxt[0])
    symbols = [mol.GetAtomWithIdx(i).GetSymbol() for i in path]
    hetero = [k for k, s in enumerate(symbols) if s != 'C']
    if len(hetero) < 4 or any(b - a == 1 for a, b in zip(hetero, hetero[1:])):
        return None
    counts = {el: symbols.count(el) for el, _p in _A_PREFIX}
    if any(n > _MAX_READABLE_COUNT for n in counts.values()):
        return None
    best = None
    for order in (path, path[::-1]):
        loc = {idx: k + 1 for k, idx in enumerate(order)}
        het = sorted(loc[i] for i in order if mol.GetAtomWithIdx(i).GetSymbol() != 'C')
        by_el = tuple(sorted(loc[i] for i in order if mol.GetAtomWithIdx(i).GetSymbol() == el) for el, _p in _A_PREFIX)
        key = (het, by_el, loc[attach_idx])
        if best is None or key < best[0]:
            best = (key, loc)
    _key, loc = best
    parts = []
    for el, word in _A_PREFIX:
        locs = sorted(loc[i] for i in path if mol.GetAtomWithIdx(i).GetSymbol() == el)
        if locs:
            mult = multiplying_prefix(len(locs)) if len(locs) > 1 else ''
            parts.append(f"{render_locants(locs)}-{mult}{word}")
    stem = get_chain_name(len(path))
    if not stem or not stem.endswith('ane'):
        return None
    name = f"{'-'.join(parts)}{stem[:-1]}-{loc[attach_idx]}-yl"
    if attach_idx in (path[0], path[-1]) and any(
            counts[el] + more > _MAX_READABLE_COUNT for el, more in _chain_beyond(mol, atoms, attach_idx).items()):
        # gives the whole unbranched chain, this group and the atoms it goes on with, one 'a' name, which
        # has a count word OPSIN cannot read: the name shipped is a valid one of the structure and is not the PIN
        from ..metrics.provenance import record_non_pin_fragment
        record_non_pin_fragment(name)
    return name


def _chain_beyond(mol, atoms, attach_idx: int) -> dict:
    """``{element: count}`` of the neutral divalent O, S, Se, Te atoms of the unbranched acyclic chain that goes on from
    carbon ``attach_idx`` outside ``atoms`` (``-O-CH2-CH2-O-...``). The chain ends in carbon, the Blue Book
    :23430), so heteroatoms after its last carbon, the oxygen bonded to a ring or a terminal -OH, are no unit of it; the
    walk ends at a ring, a branch or any other atom."""
    outer = [nb.GetIdx() for nb in mol.GetAtomWithIdx(attach_idx).GetNeighbors()
             if nb.GetIdx() not in atoms and nb.GetAtomicNum() > 1]
    seq, prev, here = [], attach_idx, outer[0] if len(outer) == 1 else None
    while here is not None and here not in atoms:
        a = mol.GetAtomWithIdx(here)
        onward = [nb.GetIdx() for nb in a.GetNeighbors() if nb.GetIdx() != prev and nb.GetAtomicNum() > 1]
        if (a.IsInRing() or a.GetFormalCharge() or a.GetSymbol() not in ('C', 'O', 'S', 'Se', 'Te') or len(onward) > 1
                or any(b.GetBondType() != Chem.BondType.SINGLE for b in a.GetBonds())):
            break
        seq.append(a.GetSymbol())
        prev, here = here, onward[0] if onward else None
    while seq and seq[-1] != 'C':
        seq.pop()
    return {el: seq.count(el) for el, _p in _A_PREFIX}


#: (the Blue Book) "N > P > As > Sb > Bi > Si > Ge > Sn > Pb > B > Al > Ga > In > Tl > O > S > Se > Te > C"
_SENIOR_ATOM = ('N', 'P', 'As', 'Sb', 'Bi', 'Si', 'Ge', 'Sn', 'Pb', 'B', 'Al', 'Ga', 'In', 'Tl', 'O', 'S', 'Se', 'Te',
                'C')


def _parent_outranks_the_chain(mol, chain) -> bool:
    """True when the rest of ``mol`` is the parent before the chain: a principal characteristic group outside it
    , or a ring that holds an atom at least as senior as the chain's most senior heteroatom
    (:18917) "The senior parent structure, whether cyclic or acyclic, has the senior atom", a ring before a chain on a
    tie,. Else the 'a' chain is the parent ('16-cyclohexyl-2,5,8,11,14-pentaoxahexadecane')."""
    from ..perception.functional_groups import detect_functional_groups
    from .seniority import get_principal_group
    try:
        _pg, pg_atoms = get_principal_group(mol, detect_functional_groups(mol))
    except Exception:  # noqa: BLE001 -- no principal group read: the ring test below decides
        pg_atoms = None
    hetero = {i for i in chain if mol.GetAtomWithIdx(i).GetSymbol() != 'C'}
    if pg_atoms and not any(set(m) & hetero for m in pg_atoms):
        return True             # the parent carries the principal group (an ester's alkyl group shares its carbon)
    rank = {el: k for k, el in enumerate(_SENIOR_ATOM)}
    chain_best = min(rank.get(mol.GetAtomWithIdx(i).GetSymbol(), 99) for i in chain)
    ring_best = min((rank.get(a.GetSymbol(), 99) for a in mol.GetAtoms() if a.IsInRing() and a.GetIdx() not in chain),
                    default=99)
    return ring_best <= chain_best
