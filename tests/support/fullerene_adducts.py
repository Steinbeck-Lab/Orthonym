"""Build (C60-Ih)[5,6]fullerene adducts from the template of ``data.fullerene_c60``.

``adduct(fragment)``: the cage with the atoms ``a`` and ``b`` (default 1 and 9, the ends of a [6,6] bond) saturated,
a Kekule structure of the other 58 cage atoms, and the fragment bonded to the two cage atoms. The fragment is a
SMILES whose two attachment atoms are the dummy atoms ``[1*]`` (bonded to locant ``a``) and ``[2*]`` (locant ``b``):
one fragment holding both is a fused ring ('[1*]C[2*]' is the cyclopropane of 3'H-cyclopropa[1,9]); two fragments
joined by a dot are two exocyclic groups ('[1*]O.[2*]O', two hydroxy groups), and a dummy with no neighbour is a
hydrogen atom ('[1*].[2*]').
"""
from rdkit import Chem

from orthonym.data.fullerene_c60 import C60_NEIGHBOURS, neighbour_pairs

#: [60]fullerene as the aromatic SMILES the PubChem rows use
C60_SMILES = ("c12c3c4c5c1c1c6c7c2c2c8c3c3c9c4c4c%10c5c5c1c1c6c6c%11c7c2c2c7c8c3c3c8c9c4c4c9c%10c5c5c1c1c6c6c%11c2c2c7"
              "c3c3c8c4c4c9c5c1c1c6c2c3c41")


def _perfect_matching(free):
    """A perfect matching of the graph ``{atom: neighbours}`` as a list of pairs, or None."""
    if not free:
        return []
    v = min(free, key=lambda x: len(free[x]))
    for w in sorted(free[v]):
        rest = {x: ns - {v, w} for x, ns in free.items() if x not in (v, w)}
        found = _perfect_matching(rest)
        if found is not None:
            return [(v, w)] + found
    return None


def adduct(fragment: str, a: int = 1, b: int = 9) -> str:
    """The canonical SMILES of the C60 adduct of ``fragment`` on the cage bond ``a``-``b``."""
    free = {x: set(ns) - {a, b} for x, ns in C60_NEIGHBOURS.items() if x not in (a, b)}
    doubles = {frozenset(pair) for pair in _perfect_matching(free)}
    rw = Chem.RWMol()
    for _ in range(60):
        rw.AddAtom(Chem.Atom(6))
    for i, j in neighbour_pairs():
        rw.AddBond(i - 1, j - 1, Chem.BondType.DOUBLE if frozenset((i, j)) in doubles else Chem.BondType.SINGLE)
    first = rw.GetNumAtoms()
    rw.InsertMol(Chem.MolFromSmiles(fragment))
    links = []
    for atom in list(rw.GetAtoms()):
        if atom.GetIdx() >= first and atom.GetAtomicNum() == 0 and atom.GetIsotope() in (1, 2):
            neighbours = atom.GetNeighbors()
            links.append((atom.GetIsotope(), atom.GetIdx(), neighbours[0].GetIdx() if neighbours else None))
    for isotope, _dummy, neighbour in links:
        if neighbour is not None:
            rw.AddBond(neighbour, (a if isotope == 1 else b) - 1, Chem.BondType.SINGLE)
    for _isotope, dummy, _neighbour in sorted(links, key=lambda link: -link[1]):
        rw.RemoveAtom(dummy)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(mol)
