"""The ester of a sulfonic or sulfinic acid cited as a prefix "Esters cited as prefixes").

the Blue Book: "When, in an ester with the general structure R-CO-O-R′ or R-S(O)x-O-R′, another group is present
that has priority for citation as the principal group... an ester group is indicated by prefixes as 'acyloxy' for the
group R-CO-O–, and 'alkoxy...oxo', '(alkyloxy)...oxo'...". The acyl group of a sulfonic acid is its acid-stem prefix,
'benzenesulfonyl (preferred prefix)',:31266), enclosed in parentheses,:1754): R-SO2-O- on a
parent is '(methanesulfonyl)oxy', R-O-SO2- is '(methoxysulfonyl)' ('[2-(methoxysulfonyl)phenyl]phosphonic acid (PIN)',
:36540). One home for the group, read by the FG prefix dispatcher (``substituent_prefix_forms``), the branch namers
(``substituent_enumerator``), the fragment walk (``substituent_naming``) and the benzene handler (``rules.benzene``).
Kept out of those files so they stay within their size budget."""
from typing import Dict, Optional, Set

from rdkit import Chem

from .naming_utils import enclose_if_compound


def _forms():
    from . import substituent_prefix_forms
    return substituent_prefix_forms


def get_sulfur_acid_ester_prefix(
    mol,
    atoms: tuple,
    principal_chain=None,
    oxide_kind: str = "sulfonyl",
) -> Optional[str]:
    """The prefix of an ester of a sulfonic or sulfinic acid that is not the principal group "Esters
    cited as prefixes", the Blue Book: "an ester group is indicated by prefixes as 'acyloxy' for the group
    R-CO-O–, and 'alkoxy...oxo', '(alkyloxy)...oxo'... ", for R-CO-O-R′ and R-S(O)x-O-R′ alike).

    ``atoms`` is the match ``(S, O, [O,] O_ester, C)`` of ``sulfonic_ester`` / ``sulfinic_ester``. When the carbon
    on the ester oxygen is on the parent, the prefix is the acyloxy form, the acyl group of the acid in parentheses
    , ':31266 C6H5-SO2– benzenesulfonyl (preferred prefix)';,:1754 "Names such as
    cyclohexanecarbonyl and benzenesulfonyl are enclosed in parentheses"): '(methanesulfonyl)oxy',
    '(4-methylbenzene-1-sulfonyl)oxy'. When the carbon on the sulfur is on the parent, it is the
    '(alkyloxy)sulfonyl' form, as 'methyl 4-(phenoxysulfinothioyl)naphthalene-1-carboxylate (PIN)' (:31719):
    'methoxysulfonyl'. None when neither side, or both, is on ``principal_chain`` or a side cannot be named."""
    chain_set: Set[int] = set(principal_chain) if principal_chain else set()
    forms = _forms()
    if len(atoms) < 4:
        return None
    s_idx, o_idx, r_idx = atoms[0], atoms[-2], atoms[-1]
    s_carbons = [n.GetIdx() for n in mol.GetAtomWithIdx(s_idx).GetNeighbors() if n.GetSymbol() == "C"]
    if len(s_carbons) != 1:
        return None
    acid_c = s_carbons[0]
    if (r_idx in chain_set) == (acid_c in chain_set):
        return None
    # a configuration on the sulfur or in the group ('{(S)-[(non-2-yn-1-yl)oxy]sulfinyl}') is cited by the stereo-aware
    # prefix producers; this one spells constitution only, so it leaves such a group to them
    group = forms._collect_fragment_atoms(mol, s_idx, {o_idx} if r_idx in chain_set else {acid_c})
    if any(mol.GetAtomWithIdx(i).GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for i in group) or any(
            b.GetStereo() != Chem.BondStereo.STEREONONE and b.GetBeginAtomIdx() in group and b.GetEndAtomIdx() in group
            for b in mol.GetBonds()):
        return None
    if r_idx in chain_set:
        # the acyl group depends on the acid side alone; naming its acid is a nested naming call, so it is done once per
        # naming scope (``assembly.memo``)
        from .memo import cache_or_compute
        side = sorted(forms._collect_fragment_atoms(mol, s_idx, {o_idx}))
        acyl = cache_or_compute(
            "sulfur_acid_ester_acyl", (Chem.MolFragmentToSmiles(mol, side, isomericSmiles=True), oxide_kind),
            lambda: forms._acid_stem_oxide_prefix(mol, acid_c, s_idx, oxide_kind, parent_idx=o_idx))
        return f"{enclose_if_compound(acyl)}oxy" if acyl else None
    alkoxy = forms.get_alkoxy_prefix(mol, (o_idx, s_idx, r_idx), [s_idx])
    return f"{enclose_if_compound(alkoxy)}{oxide_kind}" if alkoxy else None


def sulfur_acid_ester_sulfur_prefix(mol, s_idx: int, parent_idx: int, frag_atoms) -> Optional[str]:
    """The '(alkoxysulfonyl)' / '(alkoxysulfinyl)' prefix (:func:`get_sulfur_acid_ester_prefix`) of the branch
    ``frag_atoms`` that hangs from the carbon ``parent_idx`` by its sulfur ``s_idx``: -S(=O)x-O-R with R a carbon
    ('[2-(methoxysulfonyl)phenyl]phosphonic acid (PIN)', the Blue Book). None for any other shape."""
    frag = set(frag_atoms)
    s_atom = mol.GetAtomWithIdx(s_idx)
    if (s_atom.GetSymbol() != "S" or s_atom.GetFormalCharge() or s_atom.IsInRing()
            or mol.GetAtomWithIdx(parent_idx).GetSymbol() != "C"):
        return None
    oxo, ester_o = [], []
    for nb in s_atom.GetNeighbors():
        i = nb.GetIdx()
        if i == parent_idx:
            continue
        order = mol.GetBondBetweenAtoms(s_idx, i).GetBondTypeAsDouble()
        if nb.GetSymbol() == "O" and nb.GetDegree() == 1 and order == 2.0 and not nb.GetFormalCharge():
            oxo.append(i)
        elif nb.GetSymbol() == "O" and nb.GetDegree() == 2 and order == 1.0 and not nb.IsInRing():
            ester_o.append(i)
        else:
            return None
    if len(ester_o) != 1 or len(oxo) not in (1, 2):
        return None
    r = [n.GetIdx() for n in mol.GetAtomWithIdx(ester_o[0]).GetNeighbors() if n.GetIdx() != s_idx]
    if len(r) != 1 or mol.GetAtomWithIdx(r[0]).GetSymbol() != "C":
        return None
    r_side = _forms()._collect_fragment_atoms(mol, r[0], {ester_o[0]})
    if not ({s_idx, ester_o[0], *oxo} | r_side) <= frag:
        return None
    return get_sulfur_acid_ester_prefix(mol, (s_idx, *oxo, ester_o[0], r[0]), [parent_idx],
                                        oxide_kind="sulfonyl" if len(oxo) == 2 else "sulfinyl")


def sulfur_acid_oxy_prefix(mol, o_idx: int, parent_idx: int, frag_atoms) -> Optional[str]:
    """The '(R-sulfonyl)oxy' / '(R-sulfinyl)oxy' prefix (:func:`get_sulfur_acid_ester_prefix`) of the branch
    ``frag_atoms`` that hangs from ``parent_idx`` by its ester oxygen ``o_idx``: R-S(=O)(=O)-O- or R-S(=O)-O- with one
    carbon R on the sulfur, nothing else on it, every atom but the parent in the branch. None for any other shape."""
    frag = set(frag_atoms)
    o_atom = mol.GetAtomWithIdx(o_idx)
    if o_atom.GetSymbol() != "O" or o_atom.GetFormalCharge() or o_atom.IsInRing() or o_atom.GetDegree() != 2:
        return None
    s_side = [n for n in o_atom.GetNeighbors() if n.GetIdx() != parent_idx]
    if len(s_side) != 1 or s_side[0].GetSymbol() != "S" or s_side[0].GetFormalCharge() or s_side[0].IsInRing():
        return None
    s_atom = s_side[0]
    oxo, carbons = [], []
    for nb in s_atom.GetNeighbors():
        if nb.GetIdx() == o_idx:
            continue
        bond = mol.GetBondBetweenAtoms(s_atom.GetIdx(), nb.GetIdx())
        if nb.GetSymbol() == "O" and nb.GetDegree() == 1 and bond.GetBondTypeAsDouble() == 2.0 and not nb.GetFormalCharge():
            oxo.append(nb.GetIdx())
        elif nb.GetSymbol() == "C" and bond.GetBondTypeAsDouble() == 1.0:
            carbons.append(nb.GetIdx())
        else:
            return None
    if len(carbons) != 1 or len(oxo) not in (1, 2) or not ({s_atom.GetIdx(), carbons[0], *oxo} <= frag):
        return None
    match = (s_atom.GetIdx(), *oxo, o_idx, parent_idx)
    return get_sulfur_acid_ester_prefix(mol, match, [parent_idx],
                                        oxide_kind="sulfonyl" if len(oxo) == 2 else "sulfinyl")


def consume_sulfur_acid_oxy(mol, sub_set, consumed, add_prefix) -> bool:
    """The interior sibling of:func:`sulfur_acid_oxy_prefix` in a substituent fragment: an ester oxygen between a
    carbon of the fragment's skeleton and the sulfur of R-S(=O)x- is the prefix '(methanesulfonyl)oxy' on that carbon
    ('{[(methanesulfonyl)oxy]methyl}',,:31696). Adds each prefix with ``add_prefix(host, token)`` and
    marks the ester O and the whole acid side ``consumed``. False when one token would repeat (the caller cannot spell
    'bis'), so the fragment declines."""
    seen: Dict[str, int] = {}
    for idx in sorted(sub_set):
        a = mol.GetAtomWithIdx(idx)
        if idx in consumed or a.GetSymbol() != "O" or a.GetDegree() != 2 or a.IsInRing():
            continue
        nbrs = [n.GetIdx() for n in a.GetNeighbors()]
        hosts = [n for n in nbrs if mol.GetAtomWithIdx(n).GetSymbol() == "C"]
        sulfur = [n for n in nbrs if mol.GetAtomWithIdx(n).GetSymbol() == "S"]
        if (len(hosts) != 1 or len(sulfur) != 1 or not set(nbrs) <= sub_set or hosts[0] in consumed
                or mol.GetAtomWithIdx(hosts[0]).IsInRing()):   # a ring host: the ring branch names the group
            continue
        acid_side = _forms()._collect_fragment_atoms(mol, sulfur[0], {idx})
        if not acid_side <= sub_set or hosts[0] in acid_side or acid_side & consumed:
            continue
        token = sulfur_acid_oxy_prefix(mol, idx, hosts[0], acid_side | {idx})
        if not token:
            continue
        consumed.add(idx)
        consumed.update(acid_side)
        token = enclose_if_compound(token)
        add_prefix(hosts[0], token)
        seen[token] = seen.get(token, 0) + 1
    return all(n < 2 for n in seen.values())


def sulfur_acid_branch_prefix(mol, frag_atoms, attach_idx: int) -> Optional[str]:
    """The prefix of the branch ``frag_atoms`` joined to its parent by ``attach_idx`` when it is the ester of a sulfonic
    or sulfinic acid: the ester O ('(methanesulfonyl)oxy') or the S ('(methoxysulfonyl)') bonded to one parent atom."""
    frag = set(frag_atoms)
    if attach_idx is None or attach_idx not in frag:
        return None
    root = mol.GetAtomWithIdx(attach_idx)
    parent = [n.GetIdx() for n in root.GetNeighbors() if n.GetIdx() not in frag]
    if len(parent) != 1:
        return None
    if root.GetSymbol() == "O":
        return sulfur_acid_oxy_prefix(mol, attach_idx, parent[0], frag)
    if root.GetSymbol() == "S":
        return sulfur_acid_ester_sulfur_prefix(mol, attach_idx, parent[0], frag)
    return None


def ring_sulfur_acid_oxy(mol, o_idx: int, ring_atoms, sub_atoms) -> Optional[dict]:
    """The benzene handler's group record of the ester oxygen ``o_idx`` on a ring atom of ``ring_atoms`` that joins the
    sulfur of R-S(=O)x- ('[(methanesulfonyl)oxy]'), or None. The bare 'sulfonyloxy' has no R group: another molecule."""
    ring_side = [n.GetIdx() for n in mol.GetAtomWithIdx(o_idx).GetNeighbors() if n.GetIdx() in ring_atoms]
    token = sulfur_acid_oxy_prefix(mol, o_idx, ring_side[0], set(sub_atoms)) if len(ring_side) == 1 else None
    if not token:
        return None
    return {'name': enclose_if_compound(token), 'atoms': sub_atoms, 'is_complex': True}


def is_sulfur_acid_ester_centre(mol, idx: int) -> bool:
    """True when the atom ``idx`` is the sulfur of an ester of a sulfonic or sulfinic acid, R-S(=O)x-O-R' with R and R'
    carbons, R' no acyl carbon (an anhydride is not this group): the group this module names, which the generic
    oxoacid-group rules leave to it."""
    a = mol.GetAtomWithIdx(idx)
    if a.GetSymbol() != "S" or a.GetFormalCharge():
        return False
    kinds = sorted("=O" if n.GetSymbol() == "O" and n.GetDegree() == 1 else n.GetSymbol() if n.GetSymbol() == "C"
                   else "-O" if n.GetSymbol() == "O" and n.GetDegree() == 2 and all(
                       x.GetIdx() == idx or (x.GetSymbol() == "C" and not any(
                           b.GetBondTypeAsDouble() == 2.0 and b.GetOtherAtom(x).GetSymbol() in ("O", "S", "N")
                           for b in x.GetBonds())) for x in n.GetNeighbors())
                   else "?" for n in a.GetNeighbors())
    return kinds in (["-O", "=O", "=O", "C"], ["-O", "=O", "C"])
