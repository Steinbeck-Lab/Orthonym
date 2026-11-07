"""The acyl prefix of a saturated chain substituent that is a carbonyl group and carries another
detachable group: '2-hydroxypropanoyl', not '2-hydroxy-1-oxopropyl'.

``substituent_naming._name_polyfunctional_acyclic_substituent_impl`` numbers a chain from its free
valence and cites every group on it as a detachable prefix, the oxo on the free-valence carbon
included ('1-oxo'). That spelling is the name of the group by substitution on a ``-yl`` prefix; the
Blue Book gives the acyl group the name of the acid and puts the other groups on it.

 Acyl groups derived from carboxylic acids with names retained only for general
nomenclature, (the Blue Book): "substitution on acyl groups is identical to that of
acids... Preferred IUPAC names are systematic substitutive names", with 'propanoyl (preferred prefix)',
'1-oxopropyl' the alternative that is not preferred (:30562). Substituted acyl prefixes the book prints as
preferred: '3-chloropropanoyl' (N-acetyl-N-(3-chloropropanoyl)benzamide (PIN),:33117),
'2-methylidenebutanoyl' ([2,3-dichloro-4-(2-methylidenebutanoyl)phenoxy]acetic acid (PIN),:31400),
'oxoacetyl' (:30516) and, for the two-carbon acyl, 'chloroacetyl' ((chloroacetyl)oxyl (PIN),:40694); the
locant of the one position a substituent can take on an acetyl group is omitted, (:3031) "All
locants are omitted for parent compounds when all substitutable hydrogen atoms have the same locant".
"""
from typing import Dict, List, Optional


def is_acyl_prefix(name: str) -> bool:
    """True when ``name`` is an acyl name ``acyl_prefix_text`` spells ('2-oxopropanoyl', 'chloroacetyl'),
    not the '1-oxopropyl' form that cites the oxo of the free-valence carbon as a prefix."""
    return name.endswith(("anoyl", "acetyl"))


def acyl_prefix_text(mol, attach_idx: int, sub_atoms, prefix_on: Dict[int, List[str]],
                     groups: Dict[str, List[int]], backbone) -> Optional[str]:
    """The acyl name of a linear saturated chain (``backbone``, numbered from ``attach_idx``) whose
    free-valence carbon carries an oxo and nothing else, with the other groups of ``groups`` (prefix ->
    locants counted from the free valence) on it; None when the chain is not such a group. A two-carbon
    acyl takes one kind of prefix (acetyl has one position and the locant is omitted); two different
    prefixes on the one carbon need the enclosing marks of, and the caller keeps its
    spelling. An acyl group on a nitrogen is the acylamino prefix of an amide: Substituents of the
    types -NH-CO-R and -NH-SO2-R (the Blue Book) names it by the 'amido' prefix (method (1),
    '(4-acetamido-3-methylphenyl)arsonic acid (PIN)',:33010) or by 'acylamino' (method (2)); that
    choice is not made here and the caller keeps its spelling."""
    n_carbons = len(backbone)
    if n_carbons < 2 or prefix_on.get(attach_idx) != ["oxo"]:
        return None
    if any(nb.GetAtomicNum() == 7 for nb in mol.GetAtomWithIdx(attach_idx).GetNeighbors()
           if nb.GetIdx() not in sub_atoms):
        return None
    rest = {p: [loc for loc in locs if not (p == "oxo" and loc == 1)] for p, locs in groups.items()}
    rest = {p: locs for p, locs in rest.items() if locs}
    if not rest or (n_carbons == 2 and len(rest) != 1):
        return None
    from ..data.chain_names import get_chain_prefix
    from .naming_utils import alpha_sort_key
    from .substituent_naming import _joined_prefix_parts, _located_prefix_part
    parts = [_located_prefix_part(p, rest[p], cite_locants=n_carbons > 2)
             for p in sorted(rest, key=alpha_sort_key)]
    return "".join(_joined_prefix_parts(parts)) + (
        "acetyl" if n_carbons == 2 else f"{get_chain_prefix(n_carbons)}anoyl")


def is_acylsulfanyl(mol, sub_atoms, attach_idx) -> bool:
    """True for a thioester group -S-CO-R attached at its sulfur, R hydrogen or a carbon-only
    group: (the Blue Book) "an ester group is indicated by preferred prefixes
    [...] such as 'acylsulfanyl' for the group -S-CO-R", '1-[(acetylsulfanyl)peroxy]propan-1-one
    (PIN)' (:39499). The fragment holds the sulfur, one oxo oxygen and carbon atoms only, with no
    charge and no radical."""
    from rdkit import Chem
    try:
        sub_set = set(sub_atoms)
        s = mol.GetAtomWithIdx(attach_idx)
        if attach_idx not in sub_set or s.GetSymbol() != 'S' or s.GetTotalNumHs():
            return False
        inner = [x for x in s.GetNeighbors() if x.GetIdx() in sub_set]
        if len(inner) != 1 or inner[0].GetSymbol() != 'C':
            return False
        oxo = [x for x in inner[0].GetNeighbors() if x.GetSymbol() == 'O' and x.GetDegree() == 1
               and mol.GetBondBetweenAtoms(x.GetIdx(), inner[0].GetIdx()).GetBondType()
               == Chem.BondType.DOUBLE]
        if len(oxo) != 1:
            return False
        for idx in sub_set:
            a = mol.GetAtomWithIdx(idx)
            if a.GetFormalCharge() or a.GetNumRadicalElectrons():
                return False
            if idx not in (attach_idx, oxo[0].GetIdx()) and a.GetSymbol() != 'C':
                return False
        return True
    except Exception:
        return False



def is_carbamate_amino_branch(mol, sub_atoms, attach_idx) -> bool:
    """True for a chain branch attached at an acyclic carbon (or at the nitrogen of the group itself) whose only
    heteroatoms are one carbamate ester group -NH-CO-O-R on it (Boc, Cbz, Fmoc, alkoxycarbonyl): the '[(alkoxycarbonyl)amino]' prefix of
    (the Blue Book), '(benzyloxy)carbonyl (preferred prefix)' (:18116), on an alkyl chain
    ('2-[(tert-butoxycarbonyl)amino]ethyl'). The chain and the nitrogen are acyclic; R may hold rings. No charge, radical
    or isotope."""
    from rdkit import Chem
    try:
        sub_set = set(sub_atoms)
        a0 = mol.GetAtomWithIdx(attach_idx)
        if attach_idx not in sub_set or a0.GetSymbol() not in ('C', 'N') or a0.IsInRing():
            return False      # attached at a chain carbon, or the group itself at its nitrogen
        ns = [i for i in sub_set if mol.GetAtomWithIdx(i).GetSymbol() == 'N']
        if len(ns) != 1:
            return False
        n = mol.GetAtomWithIdx(ns[0])
        if n.IsInRing() or n.GetTotalNumHs() != 1 or n.GetDegree() != 2:
            return False
        carbonyl = [x for x in n.GetNeighbors() if any(
            o.GetSymbol() == 'O' and o.GetDegree() == 1
            and mol.GetBondBetweenAtoms(o.GetIdx(), x.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
            for o in x.GetNeighbors())]
        if len(carbonyl) != 1:
            return False
        c = carbonyl[0]
        ester_o = [o for o in c.GetNeighbors() if o.GetSymbol() == 'O'
                   and mol.GetBondBetweenAtoms(o.GetIdx(), c.GetIdx()).GetBondType() == Chem.BondType.SINGLE]
        if len(ester_o) != 1 or ester_o[0].GetDegree() != 2 or c.IsInRing():
            return False
        oxo = next(o.GetIdx() for o in c.GetNeighbors() if o.GetIdx() != ester_o[0].GetIdx() and o.GetSymbol() == 'O')
        # the chain side: the carbons reached from the attachment without crossing the nitrogen, all acyclic
        chain, stack = set(), [attach_idx]
        while stack:
            x = stack.pop()
            if x in chain or x == ns[0] or x not in sub_set:
                continue
            chain.add(x)
            stack.extend(nb.GetIdx() for nb in mol.GetAtomWithIdx(x).GetNeighbors())
        if any(mol.GetAtomWithIdx(x).IsInRing() or mol.GetAtomWithIdx(x).GetSymbol() != 'C' for x in chain):
            return False
        allowed = {ns[0], c.GetIdx(), oxo, ester_o[0].GetIdx()}
        for idx in sub_set:
            a = mol.GetAtomWithIdx(idx)
            if a.GetFormalCharge() or a.GetNumRadicalElectrons() or a.GetIsotope():
                return False
            if idx not in allowed and a.GetSymbol() != 'C':
                return False
        return True
    except Exception:
        return False

def is_carboacyl_head(mol, frag_atoms, attach_idx) -> bool:
    """True when the attachment atom is the carbonyl carbon of a chain carboacyl branch R-CO- (R an acyclic carbon):
    a carbon outside every ring that carries one terminal =O and exactly one other carbon of the fragment, which is
    not a ring atom. A ring carbon there is an aroyl or cycloalkanecarbonyl head, which the located-FG writer
    already spells as the acyl name ('benzoyl', 'cyclohexanecarbonyl')."""
    from rdkit import Chem
    try:
        sub_set = set(frag_atoms)
        head = mol.GetAtomWithIdx(attach_idx)
        if attach_idx not in sub_set or head.GetSymbol() != 'C' or head.IsInRing():
            return False
        oxo = [b for b in head.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE
               and b.GetOtherAtom(head).GetSymbol() == 'O' and b.GetOtherAtom(head).GetDegree() == 1]
        chain = [x for x in head.GetNeighbors() if x.GetIdx() in sub_set and x.GetSymbol() == 'C']
        return len(oxo) == 1 and len(chain) == 1 and not chain[0].IsInRing()
    except Exception:
        return False


def _head_on_nitrogen(mol, sub_atoms, attach_idx) -> bool:
    """True when the carbonyl carbon at ``attach_idx`` is bound to a nitrogen outside the fragment (an acyl on N)."""
    sub_set = set(sub_atoms)
    return any(nb.GetAtomicNum() == 7 and nb.GetIdx() not in sub_set for nb in mol.GetAtomWithIdx(attach_idx).GetNeighbors())


def tier_fg_engages(mol, sub_atoms, attach_idx, best_effort, pin) -> bool:
    """Whether Tier FG (the located FG assembly) names the fragment: best-effort or the PIN carve-outs, from an
    attachment atom of the fragment. A ring-bearing chain carboacyl branch is not assembled as '2-R-1-oxoethyl': the
    cascade declines it and the caller spells the acyl group "Definitions and name formation",
    the Blue Book: "Compound substitutive names for acyclic acyl groups, such as '1-oxopropyl'... are included
    for use in general nomenclature"; the acyl name is the preferred prefix, "CH3-CO– acetyl (preferred
    prefix) ethanoyl 1-oxoethyl",:30442). The head of an acyl on a nitrogen is not declined: that group is the amido
    prefix of (``n_acyl_amido``), which is built before this one is asked; where it cannot name the group
    (the PIN tier's amide handler names no amide of a three-carbon N-substituent beside a ring-bearing acyl) nothing spells the acyl, and it floats as 'N-<acyl>' on a
    parent that has no nitrogen suffix Position of locants, the Blue Book), so the general '[1-oxo-2-R-ethyl](R')amino' name stands."""
    if not (best_effort or pin) or attach_idx is None or attach_idx not in set(sub_atoms):
        return False
    try:
        return not (best_effort and is_carboacyl_head(mol, sub_atoms, attach_idx)
                    and any(mol.GetAtomWithIdx(a).IsInRing() for a in sub_atoms)
                    and not _head_on_nitrogen(mol, sub_atoms, attach_idx))
    except Exception:
        return True


def one_carbon_acyl_prefix(mol, attach_idx: int, sub_set) -> Optional[str]:
    """Roadmap N5e: a one-carbon chain carrying =O and one more group is an acyl group (2),
    the Blue Book) -- 'carbamoyl', '(R)carbamoyl', '<ring>-1-carbonyl', 'methoxycarbonyl', 'benzoyl' -- never
    '{[R]amino}(oxo)methyl': (:32920) "method (2) ['carbamoyl'] for rings and ring systems"
    ,:32922-:32928). The book's prefix (``book_prefixes.acyl_group_prefix``), or None."""
    from .book_prefixes import acyl_group_prefix, book_forms_enabled
    from .substituent_naming import name_substituent_fragment
    if not book_forms_enabled():
        return None
    parents = [nb.GetIdx() for nb in mol.GetAtomWithIdx(attach_idx).GetNeighbors() if nb.GetIdx() not in sub_set]
    if len(parents) != 1:
        return None

    def _part(atoms, root):
        boundary = [nb.GetIdx() for nb in mol.GetAtomWithIdx(root).GetNeighbors() if nb.GetIdx() not in atoms]
        try:
            nm = name_substituent_fragment(mol, sorted(atoms), root, boundary)
        except Exception:  # noqa: BLE001
            return None
        return None if (not nm or nm == 'substituent') else nm
    return acyl_group_prefix(mol, attach_idx, parents[0], set(sub_set), _part)


def ring_acyl_branch_prefix(mol, sub_atoms, attach_idx: int) -> Optional[str]:
    """The acyl prefix of a branch whose head is an acyclic carbon bearing =O ('benzoyl', '3-(trifluoromethyl)benzoyl',
    'phenylcarbamoyl'; ``strict_ring_prefix.acyl_branch_head``), through:func:`one_carbon_acyl_prefix`,
    ; None for any other branch."""
    from .strict_ring_prefix import acyl_branch_head
    head = acyl_branch_head(mol, sub_atoms, attach_idx)
    return None if head is None else one_carbon_acyl_prefix(mol, head, set(sub_atoms) | {head})


def book_acyl_name(mol, sub_atoms, attach_idx: int) -> Optional[str]:
    """The ring chokepoint's acyl prefix of a branch whose carbonyl carries a ring directly or through one N or O
    ('benzoyl', '2,3-dihydro-1H-indole-1-carbonyl', 'phenylcarbamoyl', 'phenoxycarbonyl'), by
    :func:`ring_acyl_branch_prefix`; None for any other branch (a chain acyl group keeps the chokepoint's own name)
    and for the writer's general form 'R(oxo)methyl'. PIN tier only (``at_pin_tier``): at the wider tiers the
    chokepoint's own name stands, as the parent and spelling choices that follow are tuned to it (the writer's name
    there moved ten best-effort names, some to worse spellings, on the eval sets)."""
    from ..rules.pin_vocabulary import at_pin_tier
    from .strict_ring_prefix import acyl_branch_head
    if not at_pin_tier():
        return None
    sub_set = set(sub_atoms)
    head = acyl_branch_head(mol, sub_set, attach_idx)
    if head is None:
        return None
    nbrs = [n for n in mol.GetAtomWithIdx(head).GetNeighbors() if n.GetIdx() in sub_set and n.GetAtomicNum() != 8
            or n.GetIdx() in sub_set and mol.GetBondBetweenAtoms(head, n.GetIdx()).GetBondTypeAsDouble() == 1]
    if not any(n.IsInRing() or (n.GetAtomicNum() in (7, 8) and any(
            m.IsInRing() for m in n.GetNeighbors() if m.GetIdx() != head)) for n in nbrs):
        return None
    name = ring_acyl_branch_prefix(mol, sub_set, attach_idx)
    return None if name is None or name.rstrip(')]}').endswith('(oxo)methyl') else name


def _free_valence_at_one(mol, attach_idx: int, name: str) -> bool:
    """True when ``name`` is an alkyl-type prefix ('1-carboxy-2-methylpropyl', no free-valence locant) of an acyclic
    group attached at ``attach_idx``: (1) (the Blue Book) the free valence of such a prefix is at
    position 1, so a stereocentre at the attachment has the locant 1, which (:44624) cites
    ('[(1R)-1-chloropropyl]benzene (PIN)',:44668)."""
    import re
    if mol.GetAtomWithIdx(attach_idx).IsInRing() or mol.GetAtomWithIdx(attach_idx).GetSymbol() != 'C':
        return False
    return bool(re.search(r"(?:meth|eth|prop|but|pent|hex|hept|oct|non|dec)yl$", name)) and not re.search(
        r"-\d+-yl$", name) and re.match(r"1-", name.lstrip("([{")) is not None


def demoted_n_carbamoyl(suffix_name, locants, n_substituents, n_hydroxy, prefix_forms):
    """The prefix of a demoted ring carboxamide whose nitrogen carries substituents (or the N-hydroxy of a hydroxamic
    acid), as the benzene handler builds it: (the Blue Book) 'carbamoyl' for rings, the
    N-substituents without the italic N ('5-methyl-2-[methyl(phenyl)carbamoyl]benzoic acid (PIN)',:32957). One
    group only; None for any other N-substituted suffix (the caller declines); a group whose nitrogen carries nothing
    takes its plain prefix form from ``prefix_forms``."""
    if not n_substituents.get(suffix_name) and suffix_name not in n_hydroxy:
        return prefix_forms.get(suffix_name)
    if suffix_name != 'carboxamide' or len(locants) != 1:
        return None
    from ..rules.benzene import _build_n_substituted_sulfamoyl_prefix
    names = list(n_substituents.get(suffix_name) or []) + (['hydroxy'] if suffix_name in n_hydroxy else [])
    return _build_n_substituted_sulfamoyl_prefix(names, 'carbamoyl')
