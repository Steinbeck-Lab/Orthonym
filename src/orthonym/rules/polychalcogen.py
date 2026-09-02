"""Homogeneous chalcogen-chain parent hydrides (P-21.2.2 / P-68.3).

A chain of *n* IDENTICAL chalcogen atoms (O / S / Se / Te) singly bonded
end-to-end and terminated by H (potential terminal -OH/-SH functionality is
ignored, P-21.2.2) is a preselected parent hydride named ``<multiplier><stem>``::

    OO -> dioxidane (H2O2) OOO -> trioxidane (H2O3)
    SS -> disulfane (H2S2) SSSS -> tetrasulfane (H2S4)
    [SeH][SeH] -> diselane [TeH][TeH] -> ditellane

When the two TERMINAL chalcogens instead bear a simple organyl group, the chain
is named substitutively with locants — but ONLY for chains of **>=3** chalcogens::

    CSSS -> 1-methyltrisulfane CSSSC -> 1,3-dimethyltrisulfane

A 1-chalcogen "chain" (CSC) is a sulfide and a 2-chalcogen one (CSSC) a
disulfide — both named by sulfanyl-ether nomenclature elsewhere, NOT as a
``-sulfane`` (Blue Book compound-class index 36: "polysulfanes … but not
disulfides or sulfides"). The bare H-terminated chains carry no such ambiguity
and are admitted from n>=2.

Graph classifier (NOT SMARTS — feedback_smarts_and_seniority): every guard
narrows. An oxoacid (chalcogen with =O), a sulfoxide/sulfone, a hetero-chain
(O-S), an internal-substituted (non-standard-valence) chalcogen, a ring, an ion
or a radical all fail a guard and cascade onward — zero false positives.
"""
from typing import List, Optional, Tuple

from rdkit import Chem

from ..assembly.naming_utils import apply_vowel_elision
from .lambda_convention import format_lambda_token, nonstandard_bonding_number
from .substituent_purity import organyl_prefix_name

# Chalcogen element -> parent-hydride stem (P-21.1 / P-21.2.2).
_CHALCOGEN_STEMS = {'O': 'oxidane', 'S': 'sulfane', 'Se': 'selane', 'Te': 'tellane'}

# Basic multiplying prefixes (Table 1.4). NO elision of the terminal vowel
# (P-21.2.2): di+oxidane -> "dioxidane", tetra+oxidane -> "tetraoxidane".
_MULTIPLIER = {
    2: 'di', 3: 'tri', 4: 'tetra', 5: 'penta', 6: 'hexa', 7: 'hepta', 8: 'octa',
}
# Substituent multiplying prefixes (used WITH locants, normal elision rules).
_SUB_MULTIPLIER = {1: '', 2: 'di', 3: 'tri', 4: 'tetra'}


def _chalcogen_chain(mol, element: str) -> Optional[List[int]]:
    """Return the chalcogen atom indices ordered as a simple linear path, or None
    if the chalcogens of ``element`` do not form one (branch, ring, fork)."""
    chal = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == element]
    chal_set = set(chal)
    # Each chalcogen's chalcogen-neighbours via SINGLE bonds; build adjacency.
    adj = {i: [] for i in chal}
    for i in chal:
        atom = mol.GetAtomWithIdx(i)
        for nbr in atom.GetNeighbors():
            j = nbr.GetIdx()
            if j in chal_set:
                bond = mol.GetBondBetweenAtoms(i, j)
                if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
                    return None
                adj[i].append(j)
    # Linear path: exactly two endpoints (degree 1), the rest degree 2.
    degrees = {i: len(adj[i]) for i in chal}
    if any(d > 2 for d in degrees.values()):
        return None
    endpoints = [i for i, d in degrees.items() if d == 1]
    if len(chal) == 1:
        return chal                       # single chalcogen (sulfide hub) handled by caller scope
    if len(endpoints) != 2:
        return None                       # ring (no endpoint) or fork
    # Walk from one endpoint to the other.
    order = [endpoints[0]]
    prev = -1
    cur = endpoints[0]
    while True:
        nxts = [j for j in adj[cur] if j != prev]
        if not nxts:
            break
        prev, cur = cur, nxts[0]
        order.append(cur)
    if len(order) != len(chal):
        return None
    return order


def _terminal_substituents(mol, chain: List[int]
                           ) -> Optional[List[Tuple[int, str]]]:
    """Validate every chalcogen's non-chain environment and return the list of
    ``(chain_index_0_based, organyl_name)`` substituents, or None on any violation.

    Internal chalcogens must carry only H (a substituent there is non-standard
    valence). Terminal chalcogens carry exactly one H OR one pure organyl."""
    chain_set = set(chain)
    subs: List[Tuple[int, str]] = []
    for pos, idx in enumerate(chain):
        atom = mol.GetAtomWithIdx(idx)
        heavy_nonchain = [n for n in atom.GetNeighbors()
                          if n.GetIdx() not in chain_set and n.GetSymbol() != 'H']
        is_terminal = pos in (0, len(chain) - 1)
        if not heavy_nonchain:
            continue
        if not is_terminal or len(heavy_nonchain) != 1:
            return None                   # internal substituent / >1 organyl
        name = organyl_prefix_name(mol, heavy_nonchain[0].GetIdx(), idx)
        if name is None:
            return None
        subs.append((pos, name))
    return subs


def _format_substituents(subs: List[Tuple[int, str]], n: int) -> str:
    """Build the locant + alphabetised multiplied substituent prefix string,
    choosing the chain orientation that gives the lowest locant set."""
    # Two orientations: position p or (n-1-p). Pick the lower locant multiset,
    # then break a TIE by P-14.4 (g) (BB 3307): "lowest locants for the
    # substituent cited first as a prefix in the name" -- BB 29956
    # `1-hydroxy-3-oxopropane-1,2,3-tricarboxylic acid` (PIN) [not
    # `3-hydroxy-1-oxo...`; lowest locants are attributed to prefixes that are
    # cited first, see P-14.4 (g)].
    #
    # Without the tie-break the orientation fell through to RDKit atom order, so
    # `1-ethyl-3-methyltrisulfane` and `3-ethyl-1-methyltrisulfane` were both
    # reachable for the same compound depending on how its SMILES was written.
    def orientation_key(flip):
        placed = [((n - 1 - p if flip else p) + 1, name) for p, name in subs]
        grouped: dict = {}
        for loc, name in placed:
            grouped.setdefault(name, []).append(loc)
        return (sorted(loc for loc, _ in placed),
                [sorted(grouped[nm]) for nm in _cited_order(grouped)])
    flip = orientation_key(True) < orientation_key(False)
    placed = [((n - 1 - p if flip else p) + 1, name) for p, name in subs]
    # Group identical substituents -> "1,3-dimethyl"; cite in P-14.5.2 order.
    by_name: dict = {}
    for loc, name in placed:
        by_name.setdefault(name, []).append(loc)
    return _cite_locanted_prefixes(by_name)


def _l4_omits_locants(mol, chain: List[int],
                      subs: List[Tuple[int, str]]) -> bool:
    """§**P-14.3.4.4** (BB 2953) for this chain: may the locants be omitted?

    ``:39335`` prints ``CH3-S-S-SH methyltrisulfane (PIN)`` and ``:39339``
    ``CH3-S-S-S-CH3 dimethyltrisulfane (PIN)``, both inside §**P-68.4.1.1**
    *"Compounds with three or more contiguous identical chalcogen atoms are treated
    as parent hydrides in substitutive nomenclature"* -- i.e. this producer's own
    example block, every row of which is locant-free. The licence is
    ``P-14.3.4.4``: *"Locants are omitted when no isomer can be generated by moving
    suffixes and/or prefixes (if any) from their position to another or by
    interchanging them between two different positions."*

    NOT ``P-14.3.4.3``/``.5``/``.6``: those all route through
    ``substitutable_positions()``, which applies ``:3007``'s exclusion of hydrogens
    on chalcogen atoms -- and every hydrogen a polysulfane has is on a sulfur, so
    that set is EMPTY here and those three licences deny by construction. See the
    predicate's docstring; the exclusion is correct and must not be loosened.

    ★ The decision is delegated whole to ``assembly.locant_omission``, the one place
    the P-14.3.4 licences live; this function only supplies the parent-atom
    boundary (the chalcogen chain) and the THIRD of the three checks
    ``ARCH-a-licence-can-be-evaluated-on-the-wrong-molecule.md`` requires -- the
    fragment-boundary observation, which lives in ``handlers/_handler_shared.py`` so
    that ``locant_omission`` stays a pure leaf. It matters here and is not
    theoretical: ``:23385`` prints
    ``1,1'-(ethane-1,2-diyl)bis(3-methyltrisulfane) (PIN)``, where the trisulfane
    component KEEPS its ``3-`` because a multiplicative name always cites
    (``P-14.3.3``). Fail-closed on every import or read failure.
    """
    try:
        from ..assembly.handlers._handler_shared import \
            locant_scope_is_a_name_component
        from ..assembly.locant_omission import l4_no_isomer_by_relocation
    except Exception:                          # noqa: BLE001 -- deny-by-default
        return False
    try:
        if locant_scope_is_a_name_component():
            return False
        return bool(l4_no_isomer_by_relocation(
            mol,
            chain,
            prefix_locants=[pos + 1 for pos, _ in subs],
            suffix_locants=[],
            stereo_text="",
            has_indicated_h=False,
            has_isotope=any(a.GetIsotope() for a in mol.GetAtoms()),
        ))
    except Exception:                          # noqa: BLE001 -- deny-by-default
        return False


def _cite_unlocanted_prefixes(subs: List[Tuple[int, str]]) -> Optional[str]:
    """The prefix block with NO locants, per §**P-16.5.1.3.1** (BB 7272).

        "For mononuclear parent hydrides with two or more substituents the first
         cited substituent never has enclosing marks unless it includes a locant.
         The second and further substituents are each enclosed with parentheses
         even for simple substituents. When the simple substituent groups are
         accompanied by multiplicative prefixes such as 'di' and 'tri', the
         multiplicative prefixes are not included in the parentheses."

    Its scope sentence says *mononuclear*, but §**P-16.5.1.3.2** (BB 7304) extends
    the same convention to any parent whose substituents carry no locants --
    *"Simple substituents of a parent where only one possible position can be
    substituted do not require locants. Enclosing marks are used with second and
    subsequent simple substituents (see P-16.5.1.3.1)"* -- with
    ``bromo(chloro)acetic acid (PIN)`` among its examples, acetic acid being no more
    mononuclear than a trisulfane. And this family supplies the direct witness:
    ``:39341C6H5-Se-Se-Se-CH3 methyl(phenyl)triselane (PIN)``.

    So ``ethyl`` + ``methyl`` on a trisulfane is ``ethyl(methyl)trisulfane``, never
    ``ethylmethyltrisulfane``; one kind stays bare, ``dimethyltrisulfane``
    (``:39339``). Sibling of:func:`_cite_locanted_prefixes`, sharing its P-14.5.2
    citation order and its shared multiplier/enclosure primitives; returns None
    (fail closed, caller keeps the locants) when the arity is off the table.
    """
    from ..assembly.naming_utils import (enclose_if_compound,
                                         multiplied_component)
    counts: dict = {}
    for _pos, name in subs:
        counts[name] = counts.get(name, 0) + 1
    parts = []
    for rank, name in enumerate(_cited_order(counts)):
        count = counts[name]
        if count not in _SUB_MULTIPLIER:
            return None
        marked = enclose_if_compound(name)
        if rank > 0 and marked == name:
            # P-16.5.1.3.1: second and further, even when SIMPLE, are enclosed.
            # A compound prefix already carries its P-16.5.1.1 marks above.
            marked = f"({name})"
        parts.append(multiplied_component(count, name, marked))
    # No locants, so no locant hyphens: the tokens abut, exactly as BB 7272 prints
    # `chloro(methyl)silane` and BB 39341 `methyl(phenyl)triselane`.
    return ''.join(parts)


def _cited_order(by_name: dict) -> List[str]:
    """The prefix names of ``by_name`` in P-14.5.2/P-14.5.4 citation order."""
    from ..assembly.naming_utils import prefix_citation_sort_key
    return sorted(by_name, key=prefix_citation_sort_key)


def _cite_locanted_prefixes(by_name: dict) -> str:
    """Join ``{prefix_name: [locants]}`` into one locanted prefix block.

    THE single implementation for this module's two prefix blocks (the chain
    parent and the polysulfoxide/sulfone), each of which used to open-code it.

    * P-14.5.2/P-14.5.4 citation order via ``prefix_citation_sort_key``. The old
      code cited in ascending FIRST-LOCANT order with the comment "matches alpha
      here", which held only for the letters-only class the retired narrow walker
      could return. BB 21649 ``3-ethyl-2-methylhexane`` (PIN) shows the order is
      alphanumerical even when its locants then descend.
    * Segments are joined by a HYPHEN. The old code joined with '' and emitted
      ``1-ethyl3-methyltrisulfane``; BB 21649's ``3-ethyl-2-methylhexane`` (PIN)
      is the witness for the separator.
    * P-16.5.1.1 marks for a compound prefix and the P-16.3.3(b)/P-16.2.4.1(d) italicized carve-out
      come from the shared primitives, so ``propan-2-yl`` is cited
      ``1,3-di(propan-2-yl)`` (BB 25719 ``1,4-di(propan-2-yl)cyclohexane`` (PIN):
      SIMPLE multiplier OUTSIDE the marks) and ``tert-butyl`` keeps its hyphen.
    """
    from ..assembly.naming_utils import (enclose_if_compound,
                                         multiplied_component)
    parts = []
    for name in _cited_order(by_name):
        locs = sorted(by_name[name])
        # P3-CLOSEOUT Item A: the multiplier WORD comes from the shared
        # primitive, which knows P-16.3.5(a); the local `_SUB_MULTIPLIER` table
        # could only ever say di/tri, so a SUBSTITUTED prefix on a polysulfane
        # could not take bis/tris. The arity bound stays local (fail closed).
        if len(locs) not in _SUB_MULTIPLIER:
            return None
        marked = enclose_if_compound(name)
        loc_str = ','.join(str(l) for l in locs)
        parts.append(
            f"{loc_str}-{multiplied_component(len(locs), name, marked)}")
    return '-'.join(parts)


# P-63.1.2 / P-63.4 chalcogen functional-group suffixes for the P-68.4.2.2/.3
# heterogeneous-chalcogen parent+suffix layer. Suffix seniority follows the
# element order O > S > Se > Te (-ol senior to -thiol senior to...).
_CHALCOGEN_SUFFIX_NAME = {'O': 'ol', 'S': 'thiol', 'Se': 'selenol', 'Te': 'tellurol'}
_SUFFIX_SENIORITY = {'O': 0, 'S': 1, 'Se': 2, 'Te': 3}


def _all_chalcogen_chain(mol) -> Optional[List[int]]:
    """Return every chalcogen atom (O/S/Se/Te, ANY element) ordered as a single
    unbranched single-bonded path, or None (branch, ring, fork, multi-component,
    double bond). Heterogeneous sibling of:func:`_chalcogen_chain`."""
    chal = [a.GetIdx() for a in mol.GetAtoms()
            if a.GetSymbol() in _CHALCOGEN_STEMS]
    if len(chal) < 2:
        return None
    cset = set(chal)
    adj: dict = {i: [] for i in chal}
    for i in chal:
        for nbr in mol.GetAtomWithIdx(i).GetNeighbors():
            j = nbr.GetIdx()
            if j in cset:
                bond = mol.GetBondBetweenAtoms(i, j)
                if bond.GetBondType() != Chem.BondType.SINGLE:
                    return None
                adj[i].append(j)
    if any(len(adj[i]) > 2 for i in chal):
        return None
    endpoints = [i for i in chal if len(adj[i]) == 1]
    if len(endpoints) != 2:
        return None
    order = [endpoints[0]]
    prev, cur = -1, endpoints[0]
    while True:
        nxts = [j for j in adj[cur] if j != prev]
        if not nxts:
            break
        prev, cur = cur, nxts[0]
        order.append(cur)
    return order if len(order) == len(chal) else None


def _name_chalcogen_chain_with_suffix(mol) -> Optional[str]:
    """P-68.4.2.2/.3 (BB 39427/39433/39441): a HETEROGENEOUS contiguous chalcogen
    chain named on a homogeneous chalcogen PARENT hydride carrying a functional
    ``-ol``/``-thiol``/``-selenol``/``-tellurol`` suffix (the terminal ``-XH`` on a
    DIFFERENT chalcogen)::

        HS-OH -> sulfanol (parent sulfane; oxidane is never a parent)
        CH3-SS-OH -> methyldisulfanol (parent disulfane, -ol, methyl prefix)

    The senior functional group (-OH > -SH > -SeH > -TeH) is expressed as the
    suffix; the remaining atoms are the parent hydride. A run of 2-3 identical
    chalcogens must not be broken (P-68.4.2.2), so the removable suffix atom is a
    single terminal chalcogen (element differs from its neighbour) and the
    remainder must be one homogeneous run.

    Fires ONLY when the parent run length >= 2 (an unbreakable run forces the
    parent-hydride name, e.g. disulfane) OR the molecule has NO carbon (a pure
    inorganic pair, e.g. HS-OH). Otherwise a P-63 carbon-parent name wins
    (CH3-S-OH -> methane-*SO*-thioperoxol) and this declines. Fail-closed; pure.
    """
    if mol is None:
        return None
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    if mol.GetRingInfo().NumRings() > 0:
        return None
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() != 0:
            return None
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'H':
            continue
        if atom.GetSymbol() not in _CHALCOGEN_STEMS and atom.GetSymbol() != 'C':
            return None                       # stray heteroatom -> not this class

    chain = _all_chalcogen_chain(mol)
    if chain is None or len(chain) < 2:
        return None
    syms = [mol.GetAtomWithIdx(i).GetSymbol() for i in chain]
    if len(set(syms)) < 2:
        return None                           # homogeneous -> the parent-hydride path

    chain_set = set(chain)
    # Suffix candidates: an endpoint whose element differs from its single chain
    # neighbour (a removable len-1 run) that bears >=1 H and no organyl.
    candidates = []
    for endpt, neigh in ((0, 1), (len(chain) - 1, len(chain) - 2)):
        if syms[endpt] == syms[neigh]:
            continue                          # part of a run -> not removable
        atom = mol.GetAtomWithIdx(chain[endpt])
        if atom.GetTotalNumHs() < 1:
            continue                          # bears organyl not -H -> not a suffix
        if any(n.GetIdx() not in chain_set and n.GetSymbol() != 'H'
               for n in atom.GetNeighbors()):
            continue                          # organyl on the suffix atom -> skip
        candidates.append(endpt)
    if not candidates:
        return None
    # Express the SENIOR suffix (-ol > -thiol >...); junior chalcogen -> parent.
    suffix_pos = min(candidates, key=lambda p: _SUFFIX_SENIORITY[syms[p]])
    suffix_elem = syms[suffix_pos]

    remaining = [chain[i] for i in range(len(chain)) if i != suffix_pos]
    rem_syms = {mol.GetAtomWithIdx(i).GetSymbol() for i in remaining}
    if len(rem_syms) != 1:
        return None                           # parent not a homogeneous run
    parent_elem = next(iter(rem_syms))
    parent_len = len(remaining)

    has_carbon = any(a.GetSymbol() == 'C' for a in mol.GetAtoms())
    if not (parent_len >= 2 or not has_carbon):
        return None                           # P-63 carbon-parent name wins -> defer

    # Organyls: only on the parent-run TERMINI (at most one; internal atoms H-only).
    subs = []
    for pos, idx in enumerate(remaining):
        atom = mol.GetAtomWithIdx(idx)
        heavy_nonchain = [n for n in atom.GetNeighbors()
                          if n.GetIdx() not in chain_set and n.GetSymbol() != 'H']
        is_terminal = pos in (0, len(remaining) - 1)
        if not heavy_nonchain:
            continue
        if not is_terminal or len(heavy_nonchain) != 1:
            return None
        name = organyl_prefix_name(mol, heavy_nonchain[0].GetIdx(), idx)
        if name is None:
            return None
        subs.append(name)
    if len(subs) > 1:
        return None

    if parent_len == 1:
        parent = _CHALCOGEN_STEMS[parent_elem]           # sulfane
    elif parent_len in _MULTIPLIER:
        parent = f"{_MULTIPLIER[parent_len]}{_CHALCOGEN_STEMS[parent_elem]}"  # disulfane
    else:
        return None
    core = apply_vowel_elision(parent, _CHALCOGEN_SUFFIX_NAME[suffix_elem])
    if not subs:
        return core
    # The single organyl is cited with no locant (its terminal position on the
    # symmetric parent is unambiguous), so a compound prefix takes its P-16.3.3
    # marks here or it would run straight into the parent stem.
    from ..assembly.naming_utils import enclose_if_compound
    return f"{enclose_if_compound(subs[0])}{core}"


def name_chalcogen_chain(mol) -> Optional[str]:
    """Return the PIN for a homogeneous chalcogen-chain parent hydride (P-21.2.2),
    else None (fail-closed cascade-continuation). Pure: no mol mutation."""
    if mol is None:
        return None
    # W3-P14 (P-68.4.2.2/.3): a HETEROGENEOUS chalcogen chain named as a
    # homogeneous parent hydride + a functional -ol/-thiol suffix
    # (methyldisulfanol / sulfanol). Runs first; declines (None) for the
    # homogeneous chains handled by the parent-hydride logic below.
    _suffixed = _name_chalcogen_chain_with_suffix(mol)
    if _suffixed is not None:
        return _suffixed
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    if mol.GetRingInfo().NumRings() > 0:
        return None
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() != 0:
            return None

    # Exactly one chalcogen element present (homogeneous chain).
    chalcogens = {a.GetSymbol() for a in mol.GetAtoms()
                  if a.GetSymbol() in _CHALCOGEN_STEMS}
    if len(chalcogens) != 1:
        return None
    element = next(iter(chalcogens))

    # Every heavy atom is either the chalcogen or a carbon (organyl). Any other
    # heteroatom (N, P, halogen, a second chalcogen) -> decline.
    if any(a.GetSymbol() not in (element, 'C')
           for a in mol.GetAtoms() if a.GetSymbol() != 'H'):
        return None

    chain = _chalcogen_chain(mol, element)
    if chain is None or len(chain) < 2:
        return None
    n = len(chain)

    # P-21.2.4: an INTERNAL chalcogen may carry a nonstandard bonding number
    # filled entirely by H (2λ6,5λ4-hexasulfane, 2λ4-trisulfane — both BB
    # verbatim preselected names). A λ terminal, an organyl/heteroatom on the
    # λ atom, or a valence the shared table cannot certify fails closed.
    lam_by_pos = {}
    for pos, idx in enumerate(chain):
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetTotalValence() == 2:
            continue
        if pos in (0, len(chain) - 1):
            return None
        # _terminal_substituents below already refuses internal heavy
        # substituents; the extra valences here are H by construction.
        lam = nonstandard_bonding_number(mol, idx)
        if lam is None:
            return None
        lam_by_pos[pos] = lam

    subs = _terminal_substituents(mol, chain)
    if subs is None:
        return None

    stem = _CHALCOGEN_STEMS[element]
    base = f"{_MULTIPLIER[n]}{stem}" if n in _MULTIPLIER else None
    if base is None:
        return None  # chain too long for the basic multiplier table

    if lam_by_pos:
        if subs:
            return None  # substituted λ-chains not built — fail closed
        # P-21.2.4.1: low locants to the λ set; P-21.2.4.2: on a positional
        # tie the HIGHER bonding number takes the lower locant (λ6 before λ4).
        # sorted (locant, -λ) keys implement both tiers lexicographically.
        fwd = sorted((p + 1, -l) for p, l in lam_by_pos.items())
        rev = sorted((n - p, -l) for p, l in lam_by_pos.items())
        chosen = min(fwd, rev)
        prefix = ','.join(format_lambda_token(loc, -neg) for loc, neg in chosen)
        return f"{prefix}-{base}"

    if not subs:
        return base
    # Carbon-substituted chains are admitted only for n>=3 (avoid sulfide/disulfide).
    if n < 3:
        return None
    # P-14.3.4.4 (BB 2953): every row of this producer's own Blue Book example
    # block (P-68.4.1.1, BB 39333-39343) is locant-free, because a homogeneous
    # chalcogen chain's internal atoms carry NO hydrogen -- the two termini are the
    # only placements, and they are one orbit. Positively licensed per molecule,
    # never a locant-stripping pass; on any doubt the locanted form below ships.
    if _l4_omits_locants(mol, chain, subs):
        unlocanted = _cite_unlocanted_prefixes(subs)
        if unlocanted:
            return f"{unlocanted}{base}"
    return f"{_format_substituents(subs, n)}{base}"


# Chain-element stems for the polysulfoxide/sulfone family (P-68.4.3.2). Oxygen
# is EXCLUDED as a chain element — an O-O chain is a peroxide, not a chain of
# oxo-bearing chalcogens; here O appears only as the terminal =O (oxo) group.
_POLYSO_CHAIN_STEMS = {'S': 'sulfane', 'Se': 'selane', 'Te': 'tellane'}
_ONE_SUFFIX_MULT = {1: '', 2: 'di', 3: 'tri', 4: 'tetr', 5: 'penta', 6: 'hexa'}


def name_polysulfoxide_sulfone(mol) -> Optional[str]:
    """P-68.4.3.2 di-/polysulfoxides, polysulfones and their Se/Te analogues:
    a chain of >=2 IDENTICAL chalcogen atoms E in {S, Se, Te} (each of
    non-standard valence lambda4 / lambda6) singly bonded end-to-end, every chain
    atom bearing at least one terminal ``=O`` (oxo) and optionally organyl
    groups. Named substitutively by adding the ``-one`` suffix to the
    lambda-<multiplier><stem> parent hydride (method (1), the PIN)::

        CH3-S(=O)-S(=O)-CH3 -> 1,2-dimethyl-1lambda4,2lambda4-disulfane-1,2-dione
        CH3CH2-SO2-SO2-CH3 -> 1-ethyl-2-methyl-1lambda6,2lambda6-disulfane-1,1,2,2-tetrone

    Each lambda4 centre contributes one oxo (one ``one`` at its locant); each
    lambda6 centre contributes two. The molecule orientation is chosen for lowest
    locants to the lambda set (higher bonding number first on a tie), then to the
    oxo suffix, then to the substituents.

    SCOPE (fail-closed, accuracy-first — a graph classifier, NOT a SMARTS
    broadening): exactly one chain element (S/Se/Te), a simple linear chain of
    >=2 of them, every chain atom lambda4/lambda6 with >=1 oxo, every non-chain
    non-oxo neighbour a pure-hydrocarbyl organyl, neutral, non-radical, single
    fragment, no ring. A single-S sulfoxide/sulfone (n=1 -> P-63.6 sulfur
    handler), a bare polysulfane (no oxo -> P-21.2.2 chalcogen-chain), a
    hetero-chain, a stray heteroatom, an ion, a radical or a ring all fail a
    guard and cascade onward. Pure: no mol mutation.
    """
    if mol is None:
        return None
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    if mol.GetRingInfo().NumRings() > 0:
        return None
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() != 0:
            return None

    # Exactly one chain element among S/Se/Te (O excluded — it is the oxo group).
    chain_elems = {a.GetSymbol() for a in mol.GetAtoms()
                   if a.GetSymbol() in _POLYSO_CHAIN_STEMS}
    if len(chain_elems) != 1:
        return None
    element = next(iter(chain_elems))

    # Every heavy atom is the chain element, an oxo O, or an organyl carbon.
    if any(a.GetSymbol() not in (element, 'O', 'C')
           for a in mol.GetAtoms() if a.GetSymbol() != 'H'):
        return None

    chain = _chalcogen_chain(mol, element)
    if chain is None or len(chain) < 2:
        return None
    n = len(chain)
    chain_set = set(chain)

    lam_by_pos: dict = {}
    oxo_by_pos: dict = {}
    sub_by_pos: dict = {}
    oxo_idxs = set()
    for pos, idx in enumerate(chain):
        atom = mol.GetAtomWithIdx(idx)
        lam = nonstandard_bonding_number(mol, idx)
        if lam not in (4, 6):
            return None
        n_oxo = 0
        organyls: List[str] = []
        for b in atom.GetBonds():
            nbr = b.GetOtherAtom(atom)
            if nbr.GetIdx() in chain_set:
                if b.GetBondType() != Chem.BondType.SINGLE:
                    return None  # chain bond must be single
                continue
            sym = nbr.GetSymbol()
            bt = b.GetBondType()
            if (sym == 'O' and bt == Chem.BondType.DOUBLE
                    and nbr.GetDegree() == 1 and nbr.GetTotalNumHs() == 0):
                n_oxo += 1
                oxo_idxs.add(nbr.GetIdx())
            elif sym == 'C' and bt == Chem.BondType.SINGLE:
                name = organyl_prefix_name(mol, nbr.GetIdx(), idx)
                if name is None:
                    return None
                organyls.append(name)
            else:
                return None
        if n_oxo < 1:
            return None  # a bare -S- in the chain -> not a polysulfoxide/sulfone
        lam_by_pos[pos] = lam
        oxo_by_pos[pos] = n_oxo
        sub_by_pos[pos] = organyls

    # Coverage: every heavy atom is a chain atom, a counted oxo O, or an organyl C.
    for a in mol.GetAtoms():
        if a.GetSymbol() in ('H', 'C') or a.GetIdx() in chain_set \
                or a.GetIdx() in oxo_idxs:
            continue
        return None

    def locant(pos, flip):
        return (n - 1 - pos) + 1 if flip else pos + 1

    def _sub_tie(flip):
        """P-14.4 (g) (BB 3307) tie-break: lowest locants to the prefix cited
        FIRST alphanumerically, once the lambda, oxo and substituent locant sets
        have all tied. Without it the orientation fell through to atom order."""
        grouped: dict = {}
        for p, names in sub_by_pos.items():
            for nm in names:
                grouped.setdefault(nm, []).append(locant(p, flip))
        return [sorted(grouped[nm]) for nm in _cited_order(grouped)]

    def descriptor(flip):
        lam_key = sorted((locant(p, flip), -lam_by_pos[p]) for p in lam_by_pos)
        oxo_key = sorted(locant(p, flip)
                         for p in oxo_by_pos for _ in range(oxo_by_pos[p]))
        sub_key = sorted(locant(p, flip)
                         for p in sub_by_pos for _ in sub_by_pos[p])
        return (lam_key, oxo_key, sub_key, _sub_tie(flip))

    flip = descriptor(True) < descriptor(False)

    stem = _POLYSO_CHAIN_STEMS[element]
    base = f"{_MULTIPLIER[n]}{stem}" if n in _MULTIPLIER else None
    if base is None:
        return None

    # lambda block: cite each chain atom's locant + lambda in ascending locant.
    lam_tokens = [format_lambda_token(locant(p, flip), lam_by_pos[p])
                  for p in sorted(lam_by_pos, key=lambda p: locant(p, flip))]
    lam_block = ','.join(lam_tokens)

    # oxo (-one) suffix: one 'one' per oxo, cited with its locant.
    oxo_locs = sorted(locant(p, flip)
                      for p in oxo_by_pos for _ in range(oxo_by_pos[p]))
    total_oxo = len(oxo_locs)
    one_mult = _ONE_SUFFIX_MULT.get(total_oxo)
    if one_mult is None:
        return None
    oxo_suffix = f"-{','.join(str(l) for l in oxo_locs)}-{one_mult}one"

    # substituent block (alphanumeric by name, each with ascending locants).
    by_name: dict = {}
    for pos, names in sub_by_pos.items():
        for nm in names:
            by_name.setdefault(nm, []).append(locant(pos, flip))
    if any(_SUB_MULTIPLIER.get(len(locs)) is None for locs in by_name.values()):
        return None                       # more substituents than the table covers
    sub_block = _cite_locanted_prefixes(by_name)

    core = f"{lam_block}-{base}{oxo_suffix}"
    return f"{sub_block}-{core}" if sub_block else core


__all__ = ["name_chalcogen_chain", "name_polysulfoxide_sulfone"]
