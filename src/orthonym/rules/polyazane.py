"""Polyazane parent-hydride family (P-68.3.1.1 / P-68.3.1.3 / P-21.2.2).

Acyclic chains of nitrogen atoms joined by N-N bonds:

    saturated    : NN -> hydrazine, NNN -> triazane, NNNN -> tetraazane, ...
    unsaturated  : N=N -> diazene, N=NN -> triaz-1-ene, N=NNN -> tetraaz-1-ene
    azo (P-68.3.1.3.2): R-N=N-R -> 1,2-dimethyldiazene / 1,2-diphenyldiazene
    substituted hydrazine: CNN -> methylhydrazine

``hydrazine`` and ``diazene`` are retained / preselected PINs (NOT "diazane" /
"diimide"); the longer members are systematic ``<multiplier>azane`` / ``…az-n-ene``
(P-21.2.2: no elision of the multiplier vowel -> "tetraazane").

Graph classifier (NOT SMARTS — feedback_smarts_and_seniority). Fail-closed: an
amine (no N-N bond), a diamine (N-C-C-N), hydroxylamine (N-O), a hydrazone
(C=N-N: an ylidene substituent), an azide (N=N=N: two cumulated double bonds), or
any charged / ring / radical species fails a guard and cascades onward.
"""
from typing import Dict, List, Optional, Tuple

from rdkit import Chem

from .substituent_purity import pure_organyl_prefix_name

# Saturated homogeneous N-chain PINs (P-21.2.2). n=2 is the retained "hydrazine".
_SAT_MULT = {3: 'tri', 4: 'tetra', 5: 'penta', 6: 'hexa', 7: 'hepta', 8: 'octa'}
_SUB_MULTIPLIER = {1: '', 2: 'di', 3: 'tri', 4: 'tetra'}


def _saturated_parent(n: int) -> Optional[str]:
    if n == 2:
        return "hydrazine"
    mult = _SAT_MULT.get(n)
    return f"{mult}azane" if mult else None


def _ene_parent(n: int, locant: int) -> Optional[str]:
    if n == 2:
        return "diazene"                      # retained; the only double-bond site
    sat = _saturated_parent(n)
    if sat is None:
        return None
    return f"{sat[:-3]}-{locant}-ene"         # strip 'ane' -> 'triaz' -> 'triaz-1-ene'


def _nitrogen_chain(mol) -> Optional[List[int]]:
    """Return the N atom indices ordered as a simple N-N path, else None."""
    nitrogens = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == 'N']
    if len(nitrogens) < 2:
        return None
    nset = set(nitrogens)
    adj: Dict[int, List[int]] = {i: [] for i in nitrogens}
    for i in nitrogens:
        for nbr in mol.GetAtomWithIdx(i).GetNeighbors():
            j = nbr.GetIdx()
            if j in nset:
                adj[i].append(j)
    if any(len(adj[i]) > 2 for i in nitrogens):
        return None                            # branched N -> not a simple chain
    endpoints = [i for i in nitrogens if len(adj[i]) == 1]
    if len(endpoints) != 2:
        return None                            # ring (0) or disconnected/forked
    order = [endpoints[0]]
    prev, cur = -1, endpoints[0]
    while True:
        nxts = [j for j in adj[cur] if j != prev]
        if not nxts:
            break
        prev, cur = cur, nxts[0]
        order.append(cur)
    return order if len(order) == len(nitrogens) else None


def _chain_double_bond(mol, chain: List[int]) -> Optional[int]:
    """Return the 0-based index ``i`` of the single N=N double bond between
    chain[i] and chain[i+1], 0 if the chain is fully saturated, or None if there
    is a triple bond or more than one double bond (azide etc.)."""
    n_double = 0
    pos = -1
    for i in range(len(chain) - 1):
        bond = mol.GetBondBetweenAtoms(chain[i], chain[i + 1])
        bt = bond.GetBondType()
        if bt == Chem.BondType.TRIPLE:
            return None
        if bt == Chem.BondType.DOUBLE:
            n_double += 1
            pos = i
    if n_double > 1:
        return None
    return pos                                 # -1 if saturated


def _collect_substituents(mol, chain: List[int]
                          ) -> Optional[List[Tuple[int, str]]]:
    """Return ``(0-based chain position, organyl name)`` for every substituent on
    the N-chain, or None if any substituent is non-organyl or doubly bonded
    (a hydrazone/azine ylidene)."""
    chain_set = set(chain)
    subs: List[Tuple[int, str]] = []
    for pos, idx in enumerate(chain):
        atom = mol.GetAtomWithIdx(idx)
        for nbr in atom.GetNeighbors():
            if nbr.GetIdx() in chain_set or nbr.GetSymbol() == 'H':
                continue
            bond = mol.GetBondBetweenAtoms(idx, nbr.GetIdx())
            if bond.GetBondType() != Chem.BondType.SINGLE:
                return None                    # =CR2 ylidene (hydrazone) -> fail-closed
            name = pure_organyl_prefix_name(mol, nbr.GetIdx(), idx)
            if name is None:
                return None
            subs.append((pos, name))
    return subs


def _format_n2_substituents(subs: List[Tuple[int, str]]) -> str:
    """Cite substituents on a 2-N parent (hydrazine/diazene), choosing the
    orientation giving the lowest locant set.

    A SINGLE substituent omits the locant (P-14.3.4.2 — its position on the
    symmetric 2-N parent is unambiguous): ``methylhydrazine`` / ``phenylhydrazine``
    (BB PINs, P-68.3.1.2), NOT ``1-methylhydrazine``. Two or more substituents
    are located to distinguish 1,1- from 1,2- (``1,2-dimethyldiazene``)."""
    if len(subs) == 1:
        return subs[0][1]
    def locants(flip):
        return sorted((1 - p if flip else p) + 1 for p, _ in subs)
    flip = locants(True) < locants(False)
    placed = [((1 - p if flip else p) + 1, name) for p, name in subs]
    by_name: Dict[str, List[int]] = {}
    for loc, name in placed:
        by_name.setdefault(name, []).append(loc)
    parts = []
    for name in sorted(by_name):
        locs = sorted(by_name[name])
        mult = _SUB_MULTIPLIER.get(len(locs), '')
        parts.append((min(locs), f"{','.join(map(str, locs))}-{mult}{name}"))
    parts.sort()
    return ''.join(p[1] for p in parts)


def name_polyazane(mol) -> Optional[str]:
    """Return the PIN for a polyazane-family parent hydride, else None
    (fail-closed cascade-continuation). Pure: no mol mutation."""
    if mol is None:
        return None
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() != 0:
            return None

    chain = _nitrogen_chain(mol)
    if chain is None:
        return None
    # The N-chain itself must not be a ring member (a pyrazole / indazole N-N is a
    # heterocycle, not a polyazane) — but aromatic ring SUBSTITUENTS are fine
    # (1,2-diphenyldiazene): the per-substituent purity guard handles those.
    if any(mol.GetAtomWithIdx(i).IsInRing() for i in chain):
        return None
    # Every heavy atom is an N of the chain or a carbon (organyl). Any other
    # heteroatom (O -> hydroxylamine/oxime, etc.) -> decline.
    if any(a.GetSymbol() not in ('N', 'C')
           for a in mol.GetAtoms() if a.GetSymbol() != 'H'):
        return None

    dbpos = _chain_double_bond(mol, chain)
    if dbpos is None:
        return None
    n = len(chain)
    subs = _collect_substituents(mol, chain)
    if subs is None:
        return None

    saturated = dbpos < 0
    if saturated:
        parent = _saturated_parent(n)
    else:
        # lowest-locant numbering for the single double bond.
        loc = min(dbpos + 1, n - dbpos - 1)
        parent = _ene_parent(n, loc)
    if parent is None:
        return None

    if not subs:
        return parent
    # Substituted members handled only for the 2-N parents (hydrazine/diazene);
    # longer substituted polyazanes need full N-locant rules -> fail-closed.
    if n != 2:
        return None
    return f"{_format_n2_substituents(subs)}{parent}"


__all__ = ["name_polyazane"]
