"""Acyl anhydrides joined by two chalcogen atoms other than -O-O-: 'acetic dithioperoxyanhydride'.

 (the Blue Book): "Related anhydrides, in which the junction between two acyl groups is of the type
-SS-, -OS-, -SSe-, etc. are named as 'dithioperoxyanhydrides', 'thioperoxyanhydrides', 'selenothioperoxyanhydrides'";
'CH3-CO-SS-CO-CH3 acetic dithioperoxyanhydride (PIN)' (:32354). Each acyl side is named as its acid, as for the
peroxyanhydride (``anhydrides._name_peroxy_anhydride``); the two acids in alphanumerical order:32253).

Only the junctions of two atoms of one element (-SS-, -SeSe-, -TeTe-) are named: OPSIN 2.9.0, the read-back of every
name, reads no mixed one ('acetic thioperoxyanhydride'), and between two acids a mixed one would not say which acid
holds which atom. The bridge is folded into the 'anhydride' class only when no senior group is present
(``perception.functional_groups.SUBSTITUTIVE_ANHYDRIDE_BRIDGES``): beside a senior group the substitutive read stands.
"""
from typing import Optional

from rdkit import Chem

#: -C(=X)-E-E-C(=X)- with E sulfur, selenium or tellurium (the -O-O- peroxyanhydride has its own writer)
JUNCTION_SMARTS = "[CX3;$([CX3]=[O,S,Se,Te]),$([CX3](=[NX2H1])[NX3])](=[O,S,Se,Te,NX2])[$([SX2]-[SX2]),$([SeX2]-[SeX2]),$([TeX2]-[TeX2])]-[SX2,SeX2,TeX2][CX3;$([CX3]=[O,S,Se,Te]),$([CX3](=[NX2H1])[NX3])]=[O,S,Se,Te,NX2]"
_JUNCTION = Chem.MolFromSmarts(JUNCTION_SMARTS)
#: the chalcogen prefixes of the junction, in alphanumerical order ('selenothio')
_PREFIX = {"S": "thio", "Se": "seleno", "Te": "telluro"}


def junction_word(e1: str, e2: str) -> Optional[str]:
    """'dithioperoxy' for S/S, 'thioperoxy' for O/S, 'selenothioperoxy' for S/Se."""
    pair = sorted((e1, e2))
    if pair == ["O", "O"]:
        return None
    if pair[0] == pair[1]:
        return "di" + _PREFIX[pair[0]] + "peroxy"
    return "".join(sorted(_PREFIX[e] for e in pair if e != "O")) + "peroxy"


def _carbamimidic_component(mol, c: int, imino: int, junction: int):
    """('N-ethylcarbamimidic', True) for an H2N-C(=NH)- side whose amino nitrogen carries simple substituents, the acid
    word of 'N-ethylcarbamimidic N-methylcarbamimidic dithioperoxyanhydride (PIN)', the Blue Book);
    (None, False) for any other side."""
    from ..assembly.naming_utils import alpha_sort_key
    from ..assembly.substituent_enumerator import name_substituent
    from .anhydrides import _bfs_side_atoms
    im = mol.GetAtomWithIdx(imino)
    others = [nb for nb in mol.GetAtomWithIdx(c).GetNeighbors() if nb.GetIdx() not in (imino, junction)]
    if im.GetDegree() != 1 or im.GetTotalNumHs() != 1 or len(others) != 1 or others[0].GetSymbol() != "N":
        return None, False
    amino = others[0]
    if amino.GetFormalCharge() or amino.GetIsAromatic() or amino.IsInRing():
        return None, False
    subs = []
    for nb in amino.GetNeighbors():
        if nb.GetIdx() == c:
            continue
        atoms = _bfs_side_atoms(mol, nb.GetIdx(), {amino.GetIdx()})
        if c in atoms or nb.GetSymbol() != "C":
            return None, False
        token = name_substituent(mol, sorted(atoms), nb.GetIdx())
        if not token or token == "substituent":
            return None, False
        subs.append(token)
    if not subs:
        return "carbamimidic", False
    # a compound substituent takes enclosing marks and 'bis': 'N-(2-chloroethyl)', 'N,N-bis(2-chloroethyl)'
    if len(subs) == 2 and subs[0] == subs[1]:
        return (f"N,N-di{subs[0]}" if subs[0].isalpha() else f"N,N-bis({subs[0]})") + "carbamimidic", True
    subs.sort(key=alpha_sort_key)
    return "-".join(f"N-{t}" if t.isalpha() else f"N-({t})" for t in subs) + "carbamimidic", True


def name_chalcogen_peroxy_anhydride(mol) -> Optional[str]:
    """'acetic dithioperoxyanhydride' for an acyclic -C(=X)-E-E'-C(=X)- core, else None."""
    from .anhydrides import _bfs_side_atoms, _name_acyl_acid_component, _senior_anhydride_link
    from ..assembly.naming_utils import alpha_sort_key_total
    matches = mol.GetSubstructMatches(_JUNCTION, uniquify=True)
    if len(matches) != 1:
        return None
    c1, x1, e1, e2, c2, x2 = matches[0]
    if c2 in _bfs_side_atoms(mol, c1, {e1, e2}):
        return None          # a cyclic junction: two sides of one ring are not two acids
    if mol.GetAtomWithIdx(e1).GetSymbol() != mol.GetAtomWithIdx(e2).GetSymbol():
        return None
    # 'Chalcogen analogues of di- and polyanhydrides' (the Blue Book): "the usual order of seniority
    # established for chalcogen atoms, O > S > Se > Te, is used to choose the senior anhydride... Other anhydride
    # linkages are named substitutively" ('acetic 5-(acetylsulfanyl)-5-oxopentanoic anhydride (PIN)')
    if _senior_anhydride_link(mol, mol.GetAtomWithIdx(e1).GetSymbol()):
        return None
    word = junction_word(mol.GetAtomWithIdx(e1).GetSymbol(), mol.GetAtomWithIdx(e2).GetSymbol())
    if word is None:
        return None
    acids = []
    for c, x, e in ((c1, x1, e1), (c2, x2, e2)):
        if mol.GetAtomWithIdx(x).GetSymbol() == "N":
            name, substituted = _carbamimidic_component(mol, c, x, e)
        else:
            name, substituted = _name_acyl_acid_component(mol, c, x, e)
            hetero = [nb for nb in mol.GetAtomWithIdx(c).GetNeighbors() if nb.GetIdx() not in (x, e) and nb.GetSymbol() != "C"]
            if hetero and (hetero[0].GetSymbol() != "N" or (name and "carbam" not in name)):
                # a carbamic or carbonic side spelt on formic acid ('(diethylamino)methanethioic', 'ethoxymethanethioic'),
                # "Substitution of formic acid is not recommended":30756): no name rather than a non-PIN word
                return None
            if hetero:
                substituted = hetero[0].GetDegree() > 1    # the carbamic N is the acid's own atom: 'carbamothioic' is bare
        if not name:
            return None
        acids.append((name, substituted))
    if acids[0][0] == acids[1][0]:
        if not acids[0][1]:
            return f"{acids[0][0]} {word}anhydride"
        open_, close = ("[", "]") if "(" in acids[0][0] else ("(", ")")    # nesting order
        return f"bis{open_}{acids[0][0]}{close} {word}anhydride"
    names = sorted((a[0] for a in acids), key=alpha_sort_key_total)
    return f"{names[0]} {names[1]} {word}anhydride"


#: R-SO2-E-C(=X)-R' and R-S(=O)-E-C(=X)-R' with E sulfur, selenium or tellurium (the -O- bridge is
#: ``anhydrides._name_sulfur_mixed_anhydride``)
SULFONYL_THIOACYL_SMARTS = "[$([SX4]([#6])(=O)=O),$([SX3]([#6])=O)][SX2,SeX2,TeX2][CX3;$([CX3][#6]),$([CH1])]=[O,S,Se]"
_SULFONYL_THIOACYL = Chem.MolFromSmarts(SULFONYL_THIOACYL_SMARTS)


def name_sulfonyl_acyl_chalcogenoanhydride(mol) -> Optional[str]:
    """'benzenecarbothioic ethanesulfonic thioanhydride' for R-SO2-S-C(=S)-R', else None.

     'Thioanhydrides and other chalcogen analogues' (the Blue Book): 'benzenecarbothioic ethanesulfonic
    thioanhydride (PIN)' (:32296); the class word is fixed by the bridge atom, each side is named as
    its acid, the two acids in alphanumerical order:32253)."""
    from .anhydrides import _BRIDGE_CLASS_WORD, _bfs_side_atoms, _name_acyl_acid_component
    from .esters import _extract_fragment_smiles
    from ..assembly.fragment_naming import name_fragment_recursively
    from ..assembly.naming_utils import alpha_sort_key_total
    matches = mol.GetSubstructMatches(_SULFONYL_THIOACYL, uniquify=True)
    if len(matches) != 1:
        return None
    s, bridge, c, x = matches[0]
    if c in _bfs_side_atoms(mol, s, {bridge}) or mol.GetAtomWithIdx(s).GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
        return None
    keep = _bfs_side_atoms(mol, s, {bridge})
    smi = _extract_fragment_smiles(mol, set(keep), s, cap_element=8)
    sulfur_acid = name_fragment_recursively(smi) if smi else None
    if not sulfur_acid or sulfur_acid.count(" acid") != 1 or not sulfur_acid.endswith(" acid") or " " in sulfur_acid[:-5]:
        return None
    acyl, _substituted = _name_acyl_acid_component(mol, c, x, bridge)
    if not acyl or " " in acyl:
        return None
    word = _BRIDGE_CLASS_WORD[mol.GetAtomWithIdx(bridge).GetSymbol()]
    names = sorted((sulfur_acid[:-5].strip(), acyl), key=alpha_sort_key_total)
    return f"{names[0]} {names[1]} {word}"
