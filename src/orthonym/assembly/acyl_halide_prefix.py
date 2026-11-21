"""The prefix of an acyl halide or pseudohalide group on a ring or a heteroatom: 'carbonochloridoyl'.

 'Acyl halides and pseudohalides as substituent groups' (the Blue Book): "When another group is
present that has priority for citation as principal group... an acyl halide or pseudohalide is expressed: (1) by
a prefix formed from the name of the acid, for example, 'carbonochloridoyl';... (3) at the end of an acyclic carbon
chain by a prefix denoting the halide or pseudohalide group and the prefix 'oxo'... Method (1) leads to preferred
IUPAC names when the suffix '-carboxylic acid' is used to name the corresponding acid; method (3) generates
preferred IUPAC names for acyclic carbon chains." Examples: '2-carbonochloridoylbenzoic acid (PIN)',
'2-carbonocyanidoyl-5-methylbenzoyl chloride (PIN)', '2-carbonocyanidothioylbenzoyl chloride (PIN)',
'3-(carbonobromidoyloxy)-3-oxopropanoic acid (PIN)'. The general substituent writer spelled the group
'chloro(oxo)methyl', which the PIN tier refuses, so a ring acyl halide beside an ester had no PIN
('methyl 4-[chloro(oxo)methyl]benzoate'). A group on a chain carbon is left to method (3).
"""
from typing import Iterable, Optional

#: the class word of each halide or pseudohalide in the acid name,
_WORD = {"F": "fluorid", "Cl": "chlorid", "Br": "bromid", "I": "iodid"}
#: the chalcogen on the acyl carbon: 'carbonochloridoyl', 'carbonochloridothioyl' examples)
_CHALCOGEN = {"O": "oyl", "S": "othioyl"}


def carbono_acyl_prefix(mol, frag_atoms: Iterable[int], attach_idx: int) -> Optional[str]:
    """'carbonochloridoyl' when ``frag_atoms`` is one acyl carbon ``attach_idx`` with a doubly bonded O or S and one
    F, Cl, Br, I or cyano group, joined to a ring atom or a heteroatom outside the fragment; None for any other shape
    (a chain carbon outside, a charge, a nuclide label, more atoms)."""
    frag = set(frag_atoms)
    if attach_idx not in frag:
        return None
    carbon = mol.GetAtomWithIdx(attach_idx)
    if carbon.GetSymbol() != "C" or carbon.GetTotalNumHs() or carbon.GetFormalCharge():
        return None
    if any(mol.GetAtomWithIdx(i).GetIsotope() or mol.GetAtomWithIdx(i).GetFormalCharge() for i in frag):
        return None
    outside = [nb for nb in carbon.GetNeighbors() if nb.GetIdx() not in frag]
    if len(outside) != 1 or (outside[0].GetSymbol() == "C" and not outside[0].IsInRing()):
        return None                                  # on a chain carbon: method (3), 'chloro...oxo'
    ending = word = None
    used = {attach_idx}
    for bond in carbon.GetBonds():
        other = bond.GetOtherAtom(carbon)
        if other.GetIdx() not in frag:
            continue
        used.add(other.GetIdx())
        order = bond.GetBondTypeAsDouble()
        if order == 2 and other.GetSymbol() in _CHALCOGEN and other.GetDegree() == 1:
            ending = _CHALCOGEN[other.GetSymbol()]
        elif order == 1 and other.GetSymbol() in _WORD and other.GetDegree() == 1:
            word = _WORD[other.GetSymbol()]
        elif order == 1 and other.GetSymbol() == "C" and other.GetDegree() == 2:
            nitrogen = [b.GetOtherAtom(other) for b in other.GetBonds() if b.GetBondTypeAsDouble() == 3]
            if len(nitrogen) == 1 and nitrogen[0].GetSymbol() == "N" and nitrogen[0].GetDegree() == 1:
                word = "cyanid"
                used.add(nitrogen[0].GetIdx())
    if ending is None or word is None or used != frag or holds_principal_group(mol, attach_idx):
        return None
    return f"carbono{word}{ending}"


_PRINCIPAL_MEMO: list = [None, None]


def holds_principal_group(mol, atom_idx: int) -> bool:
    """True when ``atom_idx`` is an atom of a match of the principal characteristic group of ``mol``. Such a group
    is the suffix, the Blue Book: a functional suffix represents "the principal characteristic group"),
    so these prefix writers leave it to the producers that weigh parents with it ('4-ethoxy-N-[(pyridin-2-yl)methyl]
    naphthalene-1-sulfonamide', never '1-ethoxy-4-{[(pyridin-2-yl)methyl]sulfamoyl}naphthalene')."""
    if _PRINCIPAL_MEMO[0] is not mol:
        from rdkit import rdBase
        from ..perception.functional_groups import detect_functional_groups
        from ..rules.polyfunctional import get_principal_group
        _quiet = rdBase.BlockLogs()                  # fragment molecules make RDKit warn; the answer is unchanged
        try:
            groups = detect_functional_groups(mol)
            found = get_principal_group(mol, groups)
            found = found[0] if isinstance(found, tuple) else found
            # no principal group perceived: the group spelled here may be the one perception missed
            # ('1-(chlorosulfonyl)-1H-indole' for 1H-indole-1-sulfonyl chloride), so no prefix
            atoms = {a for m in groups.get(found, ()) for a in m} if found else None
        except Exception:                            # undecided: no prefix, as before these writers existed
            atoms = None
        finally:
            del _quiet
        _PRINCIPAL_MEMO[:] = [mol, atoms]
    return _PRINCIPAL_MEMO[1] is None or atom_idx in _PRINCIPAL_MEMO[1]
