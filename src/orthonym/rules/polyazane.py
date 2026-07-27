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

from .substituent_purity import organyl_prefix_name

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
            name = organyl_prefix_name(mol, nbr.GetIdx(), idx)
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
    from ..assembly.naming_utils import (enclose_if_compound,
                                         multiplied_component,
                                         prefix_citation_sort_key)
    if len(subs) == 1:
        return enclose_if_compound(subs[0][1])
    # Lowest locant set, then the P-14.4 (g) tie-break (BB 3307): "lowest locants
    # for the substituent cited first as a prefix in the name" -- BB 29956
    # `1-hydroxy-3-oxopropane-1,2,3-tricarboxylic acid` (PIN). On the symmetric
    # 2-N parent the set is {1,2} either way, so without the tie-break the
    # orientation fell through to RDKit atom order.
    def orientation_key(flip):
        placed_ = [((1 - p if flip else p) + 1, name) for p, name in subs]
        grouped: Dict[str, List[int]] = {}
        for loc, name in placed_:
            grouped.setdefault(name, []).append(loc)
        return (sorted(loc for loc, _ in placed_),
                [sorted(grouped[nm])
                 for nm in sorted(grouped, key=prefix_citation_sort_key)])
    flip = orientation_key(True) < orientation_key(False)
    placed = [((1 - p if flip else p) + 1, name) for p, name in subs]
    by_name: Dict[str, List[int]] = {}
    for loc, name in placed:
        by_name.setdefault(name, []).append(loc)
    # P-14.5.2 citation order; P-16.3.3 marks; P-16.3.4 italicized carve-out; and
    # a HYPHEN between segments -- the old code joined with '' and shipped
    # `1-ethyl2-methylhydrazine` (BB 21649 `3-ethyl-2-methylhexane` (PIN)).
    parts = []
    for name in sorted(by_name, key=prefix_citation_sort_key):
        locs = sorted(by_name[name])
        # v29 P3-CLOSEOUT Item A: multiplier word from the shared primitive
        # (P-16.3.5(a) bis/tris for a SUBSTITUTED prefix); the local table
        # could only say di/tri, so `1,2-di(cyclohexylmethyl)hydrazine`
        # shipped where `1,2-bis(...)` is required.
        if len(locs) not in _SUB_MULTIPLIER:
            return None
        marked = enclose_if_compound(name)
        loc_str = ','.join(map(str, locs))
        parts.append(
            f"{loc_str}-{multiplied_component(len(locs), name, marked)}")
    return '-'.join(parts)


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
    from ..assembly.naming_utils import (alpha_sort_key, enclose_if_compound,
                                         multiplied_component,
                                         prefix_citation_sort_key)

    def _analyse(rev: bool):
        def loc(p: int) -> int:
            return (n - p) if rev else (p + 1)
        ene = None if dbpos < 0 else min(loc(dbpos), loc(dbpos + 1))
        sub_locs = sorted(loc(p) for p, _ in subs)
        # P-14.4 (g) (BB 3307) tie-break: once the 'ene' locant and the
        # substituent locant SET have tied, the lowest locant goes to the prefix
        # cited first alphanumerically -- otherwise the direction fell through to
        # atom order (BB 29956 is the verbatim witness).
        grouped: Dict[str, List[int]] = {}
        for p, name in subs:
            grouped.setdefault(name, []).append(loc(p))
        tie = [sorted(grouped[nm])
               for nm in sorted(grouped, key=prefix_citation_sort_key)]
        return ene, sub_locs, loc, tie

    fwd, rev = _analyse(False), _analyse(True)

    def _key(d):
        ene, sub_locs, _, tie = d
        return (ene if ene is not None else 0, sub_locs, tie)

    ene, _sub_locs, loc, _tie = min((fwd, rev), key=_key)

    if dbpos < 0:
        parent = _saturated_parent(n)
    else:
        parent = _ene_parent(n, ene)
    if parent is None:
        return None

    by_name: Dict[str, List[int]] = {}
    for p, name in subs:
        by_name.setdefault(name, []).append(loc(p))
    # Cite prefixes in P-14.5 alphanumerical order, each with its locant set;
    # join with a hyphen between a letter and a following locant digit.
    #
    # v29 P3-FIX Item 8: this used `alpha_sort_key`, while the ORIENTATION
    # tie-break 25 lines up (`_analyse`) uses `prefix_citation_sort_key`. Two
    # different orders in one code path: the direction was chosen to give the
    # lowest locant to the prefix cited first under one rule, and then the
    # prefixes were cited under another. `alpha_sort_key` is letters-only, so
    # `2-methylbutyl` and `3-methylbutyl` tied and the citation fell through to
    # dict insertion = RDKit atom order, which re-spelling then flipped
    # (`1-(2-methylbutyl)-3-(3-methylbutyl)triazane` vs
    # `3-(3-methylbutyl)-1-(2-methylbutyl)triazane` for ONE molecule).
    # `**P-14.5.4**` (`BlueBookV2.md:3517`) is the rule that separates them, and
    # it must be the SAME key on both sides or the name contradicts the locants
    # that were assigned to justify it.
    parts: List[str] = []
    for name in sorted(by_name, key=prefix_citation_sort_key):
        locs = sorted(by_name[name])
        # v29 P3-CLOSEOUT Item A: arity bound stays local (fail closed); the
        # multiplier WORD comes from the shared primitive so a SUBSTITUTED
        # prefix here can take bis/tris (P-16.3.5(a)).
        if len(locs) not in _SUB_MULTIPLIER:
            return None
        # `enclose_if_compound` ESCALATES the marks over an inner pair
        # (P-16.5.4.1) and carries the italicized carve-out, so `tert-butyl`
        # stays bare; `multiplied_component` then keeps its P-16.2.4.1(d) hyphen
        # ('1,2-di-tert-butyl', never '1,2-ditert-butyl').
        enclosed = enclose_if_compound(name)
        parts.append(
            f"{','.join(map(str, locs))}-"
            f"{multiplied_component(len(locs), name, enclosed)}")
    from ..assembly.substituent_naming import _joined_prefix_parts
    return ''.join(_joined_prefix_parts(parts)) + parent


def _name_azoxy(mol) -> Optional[str]:
    """P-68.3.1.3.3.1 azoxy compound R-N=N(O)-R' -> '{diazene base} oxide'
    (method (1) = PIN). The N-oxide of an azo compound is a zwitterion
    ([O-]-[N+]=N-): one chain nitrogen is a degree-3 [N+] bearing an -O(-), the
    other a degree-2 neutral =N; each nitrogen bears one organyl group.

        C6H5-N=N(O)-C6H5  ->  diphenyldiazene oxide   (BB 38857)

    SCOPE (fail-closed -> None): only the SYMMETRIC diaryl/dialkyl case
    (R == R'), which per the BB example omits the oxide locant
    ('diphenyldiazene oxide', not '...1-oxide'). The unsymmetric case needs the
    NNO/ONN oxide-locant machinery (P-68.3.1.3.3.1 method (2)) and is left
    unbuilt (returns None) rather than emitting a locant-ambiguous name. Also
    fail-closed for a non-organyl R, a ring N, or any extra charge. Pure."""
    from rdkit import Chem
    if sum(a.GetFormalCharge() for a in mol.GetAtoms()) != 0:
        return None
    n_plus = [a for a in mol.GetAtoms()
              if a.GetSymbol() == 'N' and a.GetFormalCharge() == 1]
    if len(n_plus) != 1:
        return None
    na = n_plus[0]
    if na.IsInRing() or na.GetDegree() != 3 or na.GetNumRadicalElectrons() != 0:
        return None
    o_minus = nb = c_a = None
    for b in na.GetBonds():
        other = b.GetOtherAtom(na)
        bt = b.GetBondType()
        if (other.GetSymbol() == 'O' and other.GetDegree() == 1
                and other.GetFormalCharge() == -1
                and bt == Chem.BondType.SINGLE):
            o_minus = other
        elif other.GetSymbol() == 'N' and bt == Chem.BondType.DOUBLE:
            nb = other
        elif other.GetSymbol() == 'C' and bt == Chem.BondType.SINGLE:
            c_a = other
        else:
            return None
    if o_minus is None or nb is None or c_a is None:
        return None
    # The other nitrogen: neutral, acyclic, degree 2 (=na + one organyl C).
    if (nb.GetFormalCharge() != 0 or nb.IsInRing() or nb.GetDegree() != 2
            or nb.GetNumRadicalElectrons() != 0):
        return None
    c_b = None
    for b in nb.GetBonds():
        other = b.GetOtherAtom(nb)
        if other.GetIdx() == na.GetIdx():
            continue
        if other.GetSymbol() == 'C' and b.GetBondType() == Chem.BondType.SINGLE:
            c_b = other
        else:
            return None
    if c_b is None:
        return None
    # No charge anywhere but the N+/O- zwitterion pair.
    if any(a.GetFormalCharge() != 0 for a in mol.GetAtoms()
           if a.GetIdx() not in (na.GetIdx(), o_minus.GetIdx())):
        return None
    ra = organyl_prefix_name(mol, c_a.GetIdx(), na.GetIdx())
    rb = organyl_prefix_name(mol, c_b.GetIdx(), nb.GetIdx())
    if ra is None or rb is None:
        return None
    if ra != rb:
        return None                          # unsymmetric -> NNO/ONN machinery
    from ..assembly.naming_utils import (enclose_if_compound,
                                         multiplier_needs_hyphen)
    enclosed = enclose_if_compound(ra)
    if enclosed == ra and multiplier_needs_hyphen(ra):
        return f"di-{enclosed}diazene oxide"      # P-16.3.4 di-tert-butyl...
    return f"di{enclosed}diazene oxide"


def _formazan_terminal_n(mol, inner_n, central_c, bond_type):
    """Return the terminal-N neighbour of a formazan inner nitrogen ``inner_n``
    (the neighbour that is NOT ``central_c``, joined by ``bond_type`` and being
    an acyclic N), or None."""
    from rdkit import Chem
    other = None
    for nb in inner_n.GetNeighbors():
        if nb.GetIdx() == central_c.GetIdx():
            continue
        bt = mol.GetBondBetweenAtoms(inner_n.GetIdx(), nb.GetIdx()).GetBondType()
        if nb.GetSymbol() == 'N' and bt == bond_type and not nb.IsInRing():
            if other is not None:
                return None
            other = nb
        else:
            return None
    return other


def _formazan_substituent(mol, atom_idx, skeleton):
    """The single organyl substituent name on a formazan skeleton atom, or None
    (bare), or False (fail-closed: >1 substituent, a non-single bond, or a
    non-organyl / non-nameable group)."""
    from rdkit import Chem
    names = []
    for nb in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
        if nb.GetIdx() in skeleton or nb.GetAtomicNum() <= 1:
            continue
        bond = mol.GetBondBetweenAtoms(atom_idx, nb.GetIdx())
        if bond.GetBondType() != Chem.BondType.SINGLE or nb.GetSymbol() != 'C':
            return False
        nm = organyl_prefix_name(mol, nb.GetIdx(), atom_idx)
        if nm is None:
            return False
        names.append(nm)
    if len(names) > 1:
        return False
    return names[0] if names else None


def name_formazan(mol) -> Optional[str]:
    """P-68.3.1.3.5: substituted formazan named on the retained parent hydride
    'formazan' (H2N-N=CH-N=NH) with its special fixed numbering — N1 the
    diazenyl-terminal N (C3-N2=N1), C3 the central carbon, N5 the
    hydrazinyl-terminal N (C3=N4-N5). Organyl substituents on N1/C3/N5 are cited
    by those locants:

        Ph-NH-N=CH-N=N-Ph  ->  1,5-diphenylformazan   (BB 38938)

    Fail-closed (returns None): the UNSUBSTITUTED parent (kept on the retained
    RETAINED_NAME table), any charge/radical, a ring skeleton atom, a
    non-organyl substituent, a substituent on an inner N, or any shape not
    matching the N=N-C(=N-N) formazan skeleton. Pure: no mol mutation."""
    from rdkit import Chem
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        return None
    for a in mol.GetAtoms():
        if a.GetFormalCharge() != 0 or a.GetNumRadicalElectrons() != 0:
            return None
    for c in mol.GetAtoms():
        if c.GetSymbol() != 'C' or c.IsInRing():
            continue
        n_dbl, n_sgl, bad = [], [], False
        for b in c.GetBonds():
            o = b.GetOtherAtom(c)
            if o.GetSymbol() != 'N':
                continue
            bt = b.GetBondType()
            if bt == Chem.BondType.DOUBLE:
                n_dbl.append(o)
            elif bt == Chem.BondType.SINGLE:
                n_sgl.append(o)
            else:
                bad = True
        if bad or len(n_dbl) != 1 or len(n_sgl) != 1:
            continue
        n4, n2 = n_dbl[0], n_sgl[0]        # C3=N4 (hydrazinyl), C3-N2 (diazenyl)
        if (n4.IsInRing() or n2.IsInRing() or n4.GetDegree() != 2
                or n2.GetDegree() != 2 or n4.GetTotalNumHs() != 0
                or n2.GetTotalNumHs() != 0):
            continue
        n5 = _formazan_terminal_n(mol, n4, c, Chem.BondType.SINGLE)   # N4-N5
        n1 = _formazan_terminal_n(mol, n2, c, Chem.BondType.DOUBLE)   # N2=N1
        if n1 is None or n5 is None:
            continue
        skeleton = {c.GetIdx(), n1.GetIdx(), n2.GetIdx(), n4.GetIdx(), n5.GetIdx()}
        if len(skeleton) != 5:
            continue
        s1 = _formazan_substituent(mol, n1.GetIdx(), skeleton)
        s3 = _formazan_substituent(mol, c.GetIdx(), skeleton)
        s5 = _formazan_substituent(mol, n5.GetIdx(), skeleton)
        if s1 is False or s3 is False or s5 is False:
            continue
        placed = [(loc, nm) for loc, nm in ((1, s1), (3, s3), (5, s5)) if nm]
        if not placed:
            return None                    # bare formazan -> retained table
        # v29 P3-FIX Item 5: this site was WIDENED by the Phase 3 organyl
        # migration but its composer was left on pre-migration raw code, so the
        # newly-admitted compound prefixes were mis-spelled three ways at once:
        # a local `_SUB_MULTIPLIER` table that knows neither the P-16.3.5(a)
        # bis/tris rule nor the P-16.2.4.1(d) hyphen; a bare `f"({nm})"` instead
        # of the escalating `enclose_if_compound` (P-16.5.4.1.5 requires brackets
        # once parentheses are already used inside -- `grep -c 'bis((' ` over the
        # Blue Book is 0 against 7 hits for `bis[(...)...]`); and `alpha_sort_key`,
        # which ties on identical letters. It shipped `1,5-ditert-butylformazan`
        # and `1,5-di((3-methylphenyl)methyl)formazan`, both OPSIN-clean so
        # SELF-01 passed them.
        #
        # `formazan` is a retained PIN "fully substitutable by suffixes and
        # prefixes" (`BlueBookV2.md:16521`), and `## **P-68.3.1.3.5.1** Derivatives
        # of formazan` (`:38930`) gives `1,3-diphenylformazan (PIN)`, so the
        # locant+multiplier+marks block is an ordinary substituent prefix block
        # and belongs to the canonical builder rather than to a local copy.
        # `format_substituent_prefix` is documented as THE one correct
        # substituent-prefix + enclosing-mark reference; the sibling
        # `_name_diazene_oxide` 100 lines up already does it this way.
        from ..assembly.naming_utils import (format_substituent_prefix,
                                             prefix_citation_sort_key)
        from ..assembly.substituent_naming import _joined_prefix_parts
        by_name: Dict[str, List[int]] = {}
        for loc, nm in placed:
            by_name.setdefault(nm, []).append(loc)
        parts: List[str] = []
        for nm in sorted(by_name, key=prefix_citation_sort_key):
            locs = sorted(by_name[nm])
            if len(locs) not in _SUB_MULTIPLIER:
                return None            # keep the supported-multiplicity bound
            parts.append(format_substituent_prefix(nm, locs, len(locs)))
        return ''.join(_joined_prefix_parts(parts)) + 'formazan'
    return None


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
    # W3-P15 (P-68.3.1.3.3): azoxy R-N=N(O)-R' is the zwitterionic N-oxide of a
    # diazene ([N+]=N with an [O-]); it must be recognised BEFORE the general
    # neutral-only charge guard below, which would otherwise decline it. Method
    # (1) gives the PIN ('diphenyldiazene oxide'). Fail-closed -> the guard.
    _azoxy = _name_azoxy(mol)
    if _azoxy is not None:
        return _azoxy
    # W3-P15 (P-68.3.1.3.5): a SUBSTITUTED formazan (R-N=N-C(R')=N-NH-R'') is
    # named on the retained parent 'formazan' with its special numbering. The
    # unsubstituted parent keeps its RETAINED_NAME table entry (name_formazan
    # returns None for it). Detected here (the N-N chain is broken by the central
    # C, so _nitrogen_chain below never sees it).
    _formazan = name_formazan(mol)
    if _formazan is not None:
        return _formazan
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
