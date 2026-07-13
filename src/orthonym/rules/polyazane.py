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


def _format_substituted_polyazane(n: int, dbpos: int,
                                  subs: List[Tuple[int, str]]) -> Optional[str]:
    """P-68.3.1.4 / P-31.1.4: substituted polyazane (>=3 N) or polyazene PIN.

    Numbers the homogeneous N-chain in the direction giving (1) the lowest
    locant to the skeletal double bond (the 'ene'), then (2) the lowest locants
    to the detachable substituent prefixes; cites organyl substituents with
    N-chain locants. Returns the full substituted name, or None (fail-closed).

        Ph-N=N-NH-Ph  ->  1,3-diphenyltriaz-1-ene   (P-68.3.1.4.2)
        CH3-NH-NH-NH2 ->  1-methyltriazane          (P-68.3.1.4.1)

    A locant-bearing / substituted substituent name is enclosed in parentheses
    and multiplied with the SIMPLE multiplier ('di', not 'bis' — BB verbatim
    '1,3-di(naphthalen-2-yl)triaz-1-ene')."""
    from ..assembly.naming_utils import is_complex_substituent, alpha_sort_key

    def _analyse(rev: bool):
        def loc(p: int) -> int:
            return (n - p) if rev else (p + 1)
        ene = None if dbpos < 0 else min(loc(dbpos), loc(dbpos + 1))
        sub_locs = sorted(loc(p) for p, _ in subs)
        return ene, sub_locs, loc

    fwd, rev = _analyse(False), _analyse(True)

    def _key(d):
        ene, sub_locs, _ = d
        return (ene if ene is not None else 0, sub_locs)

    ene, _sub_locs, loc = min((fwd, rev), key=_key)

    if dbpos < 0:
        parent = _saturated_parent(n)
    else:
        parent = _ene_parent(n, ene)
    if parent is None:
        return None

    by_name: Dict[str, List[int]] = {}
    for p, name in subs:
        by_name.setdefault(name, []).append(loc(p))
    # Cite prefixes in ALPHABETICAL order (P-14.5.2), each with its locant set;
    # join with a hyphen between a letter and a following locant digit.
    parts: List[str] = []
    for name in sorted(by_name, key=alpha_sort_key):
        locs = sorted(by_name[name])
        mult = _SUB_MULTIPLIER.get(len(locs))
        if mult is None:
            return None
        enclosed = f"({name})" if is_complex_substituent(name) else name
        parts.append(f"{','.join(map(str, locs))}-{mult}{enclosed}")
    from ..assembly.substituent_naming import _joined_prefix_parts
    return ''.join(_joined_prefix_parts(parts)) + parent


def _name_azane_carboxylic_acid(mol) -> Optional[str]:
    """P-58.3.2 (BB 24896): H2N-(NH)k-COOH -> '<azane>-1-carboxylic acid'.

    Peels a single terminal bare carboxyl -C(=O)OH bonded to a chain-terminal N,
    verifies the remainder is a PURE homogeneous saturated N-chain of length
    >= 3, and returns '<parent>-1-carboxylic acid' (the acid-bearing N is locant
    1 per P-31.1.4, lowest locant to the suffix). Returns None (fail-closed) for
    the 2-N member (retained hydrazinecarboxylic acid path), any other
    decoration, an interior attachment, unsaturation, ring/charge, or a
    non-nameable parent. Pure: no mol mutation of the input.
    """
    # Locate exactly ONE carboxyl carbon: =O, -OH, exactly one N neighbour,
    # degree 3 (no other heavy neighbour).
    carboxyl_c = terminal_n = None
    n_carboxyls = 0
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != 'C':
            continue
        nbrs = atom.GetNeighbors()
        o_double = [n for n in nbrs if n.GetSymbol() == 'O'
                    and mol.GetBondBetweenAtoms(
                        atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 2.0]
        o_single = [n for n in nbrs if n.GetSymbol() == 'O'
                    and mol.GetBondBetweenAtoms(
                        atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 1.0
                    and n.GetTotalNumHs() >= 1]
        n_nbrs = [n for n in nbrs if n.GetSymbol() == 'N']
        if (len(o_double) == 1 and len(o_single) == 1 and len(n_nbrs) == 1
                and atom.GetDegree() == 3):
            n_carboxyls += 1
            carboxyl_c = atom.GetIdx()
            terminal_n = n_nbrs[0].GetIdx()
    if carboxyl_c is None or n_carboxyls != 1:
        return None

    # Build a fragment view WITHOUT the carboxyl C and its two O's.
    rw = Chem.RWMol(mol)
    to_del = {carboxyl_c}
    for n in mol.GetAtomWithIdx(carboxyl_c).GetNeighbors():
        if n.GetSymbol() == 'O':
            to_del.add(n.GetIdx())
    for idx in sorted(to_del, reverse=True):
        rw.RemoveAtom(idx)
    frag = rw.GetMol()
    try:
        Chem.SanitizeMol(frag)
    except Exception:
        return None
    if len(Chem.GetMolFrags(frag)) != 1:
        return None
    # The remaining fragment must be a PURE homogeneous saturated N-chain: every
    # heavy atom is an N of a simple N-N path, none in a ring, no double bond.
    chain = _nitrogen_chain(frag)
    if chain is None:
        return None
    if any(a.GetSymbol() != 'N' for a in frag.GetAtoms() if a.GetSymbol() != 'H'):
        return None
    if any(frag.GetAtomWithIdx(i).IsInRing() for i in chain):
        return None
    if _chain_double_bond(frag, chain) != -1:
        return None  # unsaturated / triple -> not a saturated azane parent
    n = len(chain)
    if n < 3:
        return None  # 2-N keeps retained 'hydrazinecarboxylic acid' via cascade
    # The carboxyl-bearing N must be a chain terminus (endpoint of the N-N path)
    # so the acid takes locant 1. terminal_n is an index in the ORIGINAL mol; in
    # the peeled frag the N indices shift only for atoms after the deletions —
    # re-verify chain-endpoint status structurally in the ORIGINAL mol instead.
    orig_n = mol.GetAtomWithIdx(terminal_n)
    orig_n_neighbors = [nb for nb in orig_n.GetNeighbors()
                        if nb.GetSymbol() == 'N']
    if len(orig_n_neighbors) != 1:
        return None  # acid-bearing N is interior, not a terminus -> decline
    parent = _saturated_parent(n)
    if parent is None:
        return None
    return f"{parent}-1-carboxylic acid"


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

    # P-58.3.2 (BB 24896): a homogeneous heteroatom (N-N-N…) chain may be BROKEN
    # to express a senior characteristic group. H2N-(NH)k-COOH -> the carboxylic
    # acid is senior to the carbonic-acid derivative, so the N-chain is the
    # 'azane' parent and -COOH is the '-carboxylic acid' suffix at the N bearing
    # it: tetraazane-1-carboxylic acid (PIN). Peel a terminal bare -C(=O)OH on a
    # chain-terminal N when the remaining fragment is a PURE >=3-N homogeneous
    # azane chain. Gated to n>=3: the 2-N acid (NNC(=O)O) keeps its retained
    # 'hydrazinecarboxylic acid' PIN via the existing cascade (do not hijack).
    _peeled = _name_azane_carboxylic_acid(mol)
    if _peeled is not None:
        return _peeled

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
    # The 2-N parents (hydrazine/diazene) keep the special single-substituent
    # locant-omission rule (methylhydrazine, not 1-methylhydrazine, P-14.3.4.2).
    if n == 2:
        return f"{_format_n2_substituents(subs)}{parent}"
    # W3-P15 (P-68.3.1.4.2 / P-31.1.4): substituted polyazanes/polyazenes with
    # >=3 N are numbered for lowest ene-then-substituent locants and cite organyl
    # substituents with N-chain locants (1,3-diphenyltriaz-1-ene). Fail-closed.
    return _format_substituted_polyazane(n, dbpos, subs)


__all__ = ["name_polyazane"]
