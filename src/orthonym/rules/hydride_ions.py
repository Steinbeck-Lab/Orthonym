"""Ions and radicals of homogeneous parent hydrides radicals, anions, cations, radical ions).

A species whose every heavy atom is one element E (a parent-hydride element of on one unbranched acyclic
chain is named on the parent hydride of that chain ('sulfane', 'trisulfane', 'hydrazine', 'disilane') with the
cumulative suffixes that say which hydrogen atoms or hydrons were added or removed at each atom:

- 'ium' (+H+), 'ylium' (-H-), 'ide' (-H+), 'uide' (+H-):,,,;
- 'yl' (-H•):.

 "Ionic centers in the same parent structure" cites them in the order 'ium', 'ylium', 'ide', 'uide', and
 "Radical ions derived from parent hydrides" puts the radical suffixes after the ionic ones
('fluoraniumyl', 'boranuidyl', 'ethan-2-id-1-yl', 'methyliumyl'). Two 'ylium' suffixes are 'bis(ylium)'
: 'diylium' could describe a cation radical). A mononuclear hydride with a nonstandard bonding number
carries its lambda descriptor ('λ4-sulfanyl',. Group-14 mononuclear hydrides drop 'ane' before 'yl'
, 'germyl'); other parents keep it, 'sulfanyl', 'boranyl').

Every name is carbon-free, so a preselected name at most: the caller's read-back decides whether it ships.
Radical centres are read from the electron count (odd: one radical, even: lone pairs), as the book names H2N+
'azanylium' and HO+ 'oxidanylium'; an atom that needs two radical centres ('azanylidene') is left to the
radical producer. Fails closed (None) on a branch, a ring, a second element, a multiple bond other than the N=N of
diazene, a neutral species with no ionic or radical centre, a lone atom, a metal that data/metal_scope.py does not
admit to PARENT_HYDRIDE_IONS, or an atom whose hydrogen count no suffix set explains.
"""
from itertools import product
from typing import Dict, List, Optional, Tuple

from rdkit import Chem

#: parent hydride names, Table 2.1); 'hydrazine' for H2N-NH2
_PARENT = {'B': 'borane', 'Al': 'alumane', 'Ga': 'gallane', 'In': 'indigane', 'Tl': 'thallane',
           'Si': 'silane', 'Ge': 'germane', 'Sn': 'stannane', 'Pb': 'plumbane',
           'N': 'azane', 'P': 'phosphane', 'As': 'arsane', 'Sb': 'stibane', 'Bi': 'bismuthane',
           'O': 'oxidane', 'S': 'sulfane', 'Se': 'selane', 'Te': 'tellane', 'Po': 'polane',
           'F': 'fluorane', 'Cl': 'chlorane', 'Br': 'bromane', 'I': 'iodane', 'At': 'astatane'}
_STANDARD = {**dict.fromkeys(('B', 'Al', 'Ga', 'In', 'Tl', 'N', 'P', 'As', 'Sb', 'Bi'), 3),
             **dict.fromkeys(('Si', 'Ge', 'Sn', 'Pb'), 4),
             **dict.fromkeys(('O', 'S', 'Se', 'Te', 'Po'), 2),
             **dict.fromkeys(('F', 'Cl', 'Br', 'I', 'At'), 1)}
#: nonstandard bonding numbers a mononuclear parent may take
_LAMBDA = {'S': (4, 6), 'Se': (4, 6), 'Te': (4, 6), 'Po': (4, 6), 'P': (5,), 'As': (5,), 'Sb': (5,), 'Bi': (5,),
           'Cl': (3, 5, 7), 'Br': (3, 5, 7), 'I': (3, 5, 7), 'At': (3, 5, 7)}
_GROUP14 = frozenset(('Si', 'Ge', 'Sn', 'Pb'))
#: the metals among the parent-hydride elements: named here only where data/metal_scope.py admits them to
#: PARENT_HYDRIDE_IONS
_METALS = frozenset(('Al', 'Ga', 'In', 'Tl', 'Sn', 'Pb', 'Bi', 'Po'))
#: suffixes in citation order, then: ionic first, the radical last
_ORDER = ('ium', 'ylium', 'ide', 'uide', 'yl')
_CATION, _ANION = ('ium', 'ylium'), ('ide', 'uide')


def _atom_ops(total: int, charge: int, radicals: int, bonding: int) -> Optional[Dict[str, int]]:
    """The suffixes that turn a parent atom of bonding number ``bonding`` into one with ``total`` bonds (hydrogen
    and skeleton), ``charge`` and ``radicals``: removals only, or one kind of addition carrying the charge plus a
    radical; the fewest suffixes. None when no such set exists."""
    best = None
    for ium, uide, ylium, ide in product(range(4), repeat=4):
        ops = {'ium': ium, 'uide': uide, 'ylium': ylium, 'ide': ide, 'yl': radicals}
        if ium + uide - ylium - ide - radicals != total - bonding or ium - uide + ylium - ide != charge:
            continue
        if (ium and (uide or ylium or ide)) or (uide and (ylium or ide)) or (ylium and ide):
            continue
        if best is None or sum(ops.values()) < sum(best.values()):
            best = ops
    return best


def _chain(mol) -> Optional[List[int]]:
    """The heavy atoms of ``mol`` in chain order when they form one unbranched acyclic chain of one element."""
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    if not heavy or len({mol.GetAtomWithIdx(i).GetSymbol() for i in heavy}) != 1:
        return None
    if mol.GetRingInfo().NumRings() or len(Chem.GetMolFrags(mol)) != 1:
        return None
    deg = {i: sum(1 for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetAtomicNum() > 1) for i in heavy}
    if any(d > 2 for d in deg.values()):
        return None
    if len(heavy) == 1:
        return heavy
    order = [next(i for i in heavy if deg[i] == 1)]
    while len(order) < len(heavy):
        nxt = [n.GetIdx() for n in mol.GetAtomWithIdx(order[-1]).GetNeighbors()
               if n.GetAtomicNum() > 1 and n.GetIdx() not in order]
        if len(nxt) != 1:
            return None
        order.append(nxt[0])
    return order


def _parent_name(element: str, n: int, double: bool) -> Optional[str]:
    """The chain's parent hydride: 'sulfane', 'trisulfane', 'hydrazine', 'diazene',."""
    if double:
        return 'diazene' if (element, n) == ('N', 2) else None
    if n == 1:
        return _PARENT[element]
    if (element, n) == ('N', 2):
        return 'hydrazine'
    from ..data.multipliers import multiplier_table
    mult = multiplier_table(10).get(n)
    return f"{mult}{_PARENT[element]}" if mult else None


def _numbering_key(ops_by_position: List[Dict[str, int]]) -> Tuple:
    """: lower locants to all suffix centres, then to radical, anionic and cationic centres: radicals >
    anions > cations); among anions then to 'uide' "'Ide' and 'uide' centers in the same parent hydride":
    "first to the anionic centers regardless of the kind and then to '-uide' anionic centers"), among cations then to
    'ylium': "then to the 'ylium' centers")."""
    def locs(kinds):
        return sorted(p + 1 for p, ops in enumerate(ops_by_position) for k in kinds for _ in range(ops[k]))
    return (locs(_ORDER), locs(('yl',)), locs(_ANION), locs(('uide',)), locs(_CATION), locs(('ylium',)))


def _multiplied(kind: str, count: int) -> str:
    if count == 1:
        return kind
    from ..data.multipliers import multiplier_table
    if kind == 'ylium':
        return {2: 'bis', 3: 'tris', 4: 'tetrakis'}[count] + '(ylium)'
    return multiplier_table(10)[count] + kind


def _join(name: str, suffix: str) -> str:
    """Elide a final 'e' of ``name`` before a suffix that starts with a vowel or 'y' ('sulfan' + 'yl')."""
    if name.endswith('e') and suffix[:1] in 'aeiouy':
        return name[:-1] + suffix
    return name + suffix


def name_hydride_ion(mol) -> Optional[str]:
    """The name of a homogeneous parent-hydride chain with ionic and/or radical centres; else None."""
    if mol is None:
        return None
    order = _chain(mol)
    if order is None:
        return None
    element = mol.GetAtomWithIdx(order[0]).GetSymbol()
    if element not in _PARENT:
        return None
    if element in _METALS:
        from ..data.metal_scope import PARENT_HYDRIDE_IONS, has_mechanism
        if not has_mechanism(element, PARENT_HYDRIDE_IONS):
            return None                                 # a metal is admitted one at a time (data/metal_scope.py)
    if len(order) == 1 and mol.GetAtomWithIdx(order[0]).GetTotalNumHs() == 0:
        return None                                     # a lone atom or ion: rules/metal_atoms.lone_atom_name
    double = False
    for b in mol.GetBonds():
        if b.GetBondType() == Chem.BondType.DOUBLE:
            double = True
        elif b.GetBondType() != Chem.BondType.SINGLE:
            return None
    if double and (element, len(order)) != ('N', 2):
        return None
    parent = _parent_name(element, len(order), double)
    if parent is None:
        return None
    ops_list, lam = [], None
    outer = Chem.GetPeriodicTable().GetNOuterElecs(mol.GetAtomWithIdx(order[0]).GetAtomicNum())
    for i in order:
        atom = mol.GetAtomWithIdx(i)
        total = int(sum(b.GetBondTypeAsDouble() for b in atom.GetBonds())) + atom.GetTotalNumHs()
        # The radical centre is read from the electron count, not from the drawing's radical flags (RDKit gives the
        # AlH3 radical cation three): an odd number of nonbonding electrons is one radical, an even number lone pairs.
        radicals = (outer - atom.GetFormalCharge() - total) % 2
        bondings = (_STANDARD[element],) + (_LAMBDA.get(element, ()) if len(order) == 1 else ())
        ops = None
        for bonding in bondings:
            ops = _atom_ops(total, atom.GetFormalCharge(), radicals, bonding)
            if ops is not None:
                lam = bonding if bonding != _STANDARD[element] else None
                break
        if ops is None:
            return None
        ops_list.append(ops)
    if len(order) > 1 and _numbering_key(ops_list[::-1]) < _numbering_key(ops_list):
        ops_list = ops_list[::-1]
    from .lambda_convention import LAMBDA
    head = f"{LAMBDA}{lam}-" if lam else ""
    counts = {k: sum(ops[k] for ops in ops_list) for k in _ORDER}
    if not any(counts.values()):
        return None                                     # a neutral hydride is the mononuclear/chain handlers' job
    first = next(k for k in _ORDER if counts[k])
    # "The locant '1' is omitted:" (b) "in monosubstituted homogeneous chains consisting of only two
    # identical atoms" ('hydrazinyl', not 'hydrazin-1-yl'): one suffix centre on a two-atom chain carries no locant
    if len(order) == 1 or (len(order) == 2 and sum(counts.values()) == 1):
        if (len(order) == 1 and element in _GROUP14 and first in ('ylium', 'yl') and counts[first] == 1
                and counts['ium'] == 0):
            name = parent[:-3]                          #: 'silane' -> 'sil' + 'yl'/'ylium'
        else:
            name = parent
        for kind in _ORDER:
            if counts[kind]:
                name = _join(name, _multiplied(kind, counts[kind]))
        return head + name
    name = parent
    for kind in _ORDER:
        if counts[kind]:
            locants = ",".join(str(p + 1) for p, ops in enumerate(ops_list) for _ in range(ops[kind]))
            word = _multiplied(kind, counts[kind])
            name = (name[:-1] if name.endswith('e') and word[0] in 'aeiouy' else name) + f"-{locants}-{word}"
    return head + name


__all__ = ["name_hydride_ion"]
