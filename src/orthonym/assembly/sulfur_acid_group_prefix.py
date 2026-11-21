"""The prefix of a sulfur acid group cited as a substituent: 'sulfo', 'sulfino', 'sulfamoyl', 'chlorosulfonyl'.

Appendix 2 (preferred prefixes marked *): 'sulfo* (when unmodified)' (the Blue Book), 'sulfino* (when
unmodified)' (:56832), 'aminosulfonyl = sulfamoyl* = sulfuramidoyl' (:55489,, 'phenylsulfamoyl*'
(:55494), 'aminosulfinyl* (not sulfinamoyl)' (:55485); (:36472,:36482): '-SO2-Cl chlorosulfonyl (preselected
prefix)', '-S(=O)-Cl chlorosulfinyl (preselected prefix)'. The shared substituent namers had no case for a fragment
rooted at such a sulfur and returned their 'substituent' sentinel, so a producer that names its prefixes through
them dropped the group ('propanoyl chloride' for 3-sulfamoylpropanoyl chloride, rejected as another molecule).
"""
from typing import Iterable, Optional

#: the halogen prefix of a sulfonyl or sulfinyl halide group
_HALO = {"F": "fluoro", "Cl": "chloro", "Br": "bromo", "I": "iodo"}


def sulfur_acid_group_prefix(mol, frag_atoms: Iterable[int], attach_idx: int) -> Optional[str]:
    """The prefix of ``frag_atoms`` when it is one sulfur ``attach_idx`` (two or one doubly bonded O) with one OH,
    NH2, N-substituted amino (sulfonyl only) or halogen group, joined to one atom outside the fragment; None for any
    other shape (a charge, a nuclide label, a ring sulfur or nitrogen, a branch the namer cannot name)."""
    frag = set(frag_atoms)
    if attach_idx not in frag:
        return None
    sulfur = mol.GetAtomWithIdx(attach_idx)
    if sulfur.GetSymbol() != "S" or sulfur.GetTotalNumHs() or sulfur.IsInRing():
        return None
    if any(mol.GetAtomWithIdx(i).GetIsotope() or mol.GetAtomWithIdx(i).GetFormalCharge() for i in frag):
        return None
    if sum(1 for nb in sulfur.GetNeighbors() if nb.GetIdx() not in frag) != 1:
        return None
    oxo, rest = 0, []
    for bond in sulfur.GetBonds():
        other = bond.GetOtherAtom(sulfur)
        if other.GetIdx() not in frag:
            continue
        if bond.GetBondTypeAsDouble() == 2 and other.GetSymbol() == "O" and other.GetDegree() == 1:
            oxo += 1
        elif bond.GetBondTypeAsDouble() == 1:
            rest.append(other)
        else:
            return None
    if oxo not in (1, 2) or len(rest) != 1:
        return None
    from .acyl_halide_prefix import holds_principal_group
    if holds_principal_group(mol, attach_idx):
        return None                                  # the principal group is the suffix, never this prefix
    group, sym = rest[0], rest[0].GetSymbol()
    sulfonyl = oxo == 2
    if sym in _HALO and group.GetDegree() == 1 and len(frag) == oxo + 2:
        return f"{_HALO[sym]}{'sulfonyl' if sulfonyl else 'sulfinyl'}"
    if sym == "O" and group.GetDegree() == 1 and len(frag) == oxo + 2:
        return "sulfo" if sulfonyl else "sulfino"
    if sym != "N" or group.IsInRing() or any(b.GetBondTypeAsDouble() != 1 for b in group.GetBonds()):
        return None
    if group.GetDegree() == 1 and len(frag) == oxo + 2:
        return "sulfamoyl" if sulfonyl else "aminosulfinyl"
    if not sulfonyl:
        return None
    return _n_substituted_sulfamoyl(mol, frag, attach_idx, group.GetIdx(), oxo)


def _n_substituted_sulfamoyl(mol, frag, s_idx, n_idx, oxo) -> Optional[str]:
    """'(methylsulfamoyl)' for an -SO2-NHR or -SO2-NR2 fragment, each R named by the shared namer; None otherwise."""
    from .substituent_enumerator import name_substituent
    from ..rules.benzene import _build_n_substituted_sulfamoyl_prefix
    names, covered = [], {s_idx, n_idx}
    for branch in mol.GetAtomWithIdx(n_idx).GetNeighbors():
        if branch.GetIdx() == s_idx:
            continue
        comp, stack = set(), [branch.GetIdx()]
        while stack:
            x = stack.pop()
            if x in comp or x in (s_idx, n_idx):
                continue
            if x not in frag:
                return None
            comp.add(x)
            stack.extend(nb.GetIdx() for nb in mol.GetAtomWithIdx(x).GetNeighbors())
        name = name_substituent(mol, sorted(comp), branch.GetIdx())
        if not name or name == "substituent":
            return None
        names.append(name)
        covered |= comp
    if len(covered) + oxo != len(frag):
        return None                                  # a ring through the nitrogen, or atoms the branches do not reach
    return _build_n_substituted_sulfamoyl_prefix(names)
