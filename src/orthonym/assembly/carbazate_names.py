"""The ester of hydrazinecarboxylic acid, R-O-C(=O)-NH-NH2 (lane/l17 item 3 fix round).

 (the Blue Book, its Note and its example:30682): the replacement of the hydrogen of formic acid by -NH-NH2
gives 'hydrazinecarboxylic acid (PIN)' (not carbazic acid), a carboxylic acid on the parent hydride hydrazine.
 (:31663): "All preferred IUPAC names for esters are named by functional class nomenclature", so the ester of it
is 'R hydrazinecarboxylate', with no locant for the one hydrazine position the acid group can take. The carbamic-acid builder
of ``composer._name_carbamate`` would spell the unit 'R aminocarbamate' (its N-substituent is an amino group), a name of the
wrong class, so the carbazate is built here and the builder is not asked for it.
"""
from __future__ import annotations

from typing import Any, Optional

from rdkit import Chem


def free_hydrazine_nitrogen(mol, n_idx: int, c_idx: int) -> Optional[int]:
    """The terminal nitrogen of the NH2 of a hydrazinecarboxylic acid unit R-O-C(=O)-NH-NH2, when the carbamate nitrogen
    ``n_idx`` (bonded to the carbonyl carbon ``c_idx``) is that unit's NH; None otherwise.

    Both nitrogens are neutral and not ring atoms, the nitrogen next to the carbonyl bears one hydrogen, the terminal one
    two, the bond between them is single, and neither is isotopically labelled (a substituted hydrazine has locants,
    which this name has not; a labelled molecule is named through its unlabelled form)."""
    n_atom = mol.GetAtomWithIdx(n_idx)
    if (n_atom.GetSymbol() != 'N' or n_atom.GetFormalCharge() != 0 or n_atom.GetIsotope()
            or n_atom.GetNumRadicalElectrons() or n_atom.IsInRing() or n_atom.GetTotalNumHs() != 1
            or n_atom.GetDegree() != 2):
        return None
    for nb in n_atom.GetNeighbors():
        if nb.GetIdx() == c_idx:
            continue
        if (nb.GetSymbol() == 'N' and nb.GetFormalCharge() == 0 and not nb.GetIsotope()
                and not nb.GetNumRadicalElectrons() and not nb.IsInRing() and nb.GetTotalNumHs() == 2
                and nb.GetDegree() == 1
                and mol.GetBondBetweenAtoms(n_idx, nb.GetIdx()).GetBondType() == Chem.BondType.SINGLE):
            return nb.GetIdx()
    return None


def hydrazinecarboxylate_name(features: Any) -> Optional[str]:
    """'R hydrazinecarboxylate' for the molecule's one carbamate when it is the ester of hydrazinecarboxylic acid; ''
    when it is, and R cannot be named (the caller declines, never the 'aminocarbamate' spelling); None when the
    carbamate is not that ester (the carbamic-acid builder names it)."""
    mol = getattr(features, 'mol', None)
    matches = (getattr(features, 'functional_groups', None) or {}).get('carbamate') or []
    if mol is None or not matches or len(matches[0]) < 4:
        return None
    n_idx, c_idx = matches[0][0], matches[0][1]
    if free_hydrazine_nitrogen(mol, n_idx, c_idx) is None:
        return substituted_hydrazinecarboxylate_name(mol, n_idx, c_idx)
    # the ether oxygen and the R on it: the single-bonded oxygen of the carbonyl carbon (what the carbamate builder walks)
    c_atom = mol.GetAtomWithIdx(c_idx)
    core = {n_idx, c_idx}
    r_start = None
    for nb in c_atom.GetNeighbors():
        if nb.GetSymbol() != 'O':
            continue
        core.add(nb.GetIdx())
        if mol.GetBondBetweenAtoms(c_idx, nb.GetIdx()).GetBondType() == Chem.BondType.SINGLE:
            r_start = next((o.GetIdx() for o in nb.GetNeighbors() if o.GetIdx() != c_idx), None)
    if r_start is None:
        return ''
    from .composer import _name_r_group
    r_name = _name_r_group(mol, r_start, exclude_atoms=core)
    return f"{r_name} hydrazinecarboxylate" if r_name else ''


def _plain_nitrogen(atom) -> bool:
    return (atom.GetSymbol() == 'N' and atom.GetFormalCharge() == 0 and not atom.GetIsotope()
            and not atom.GetNumRadicalElectrons() and not atom.IsInRing())


#: a free acid group (class 7, senior to the class-9 ester,:18165): an OH or O- on an atom bearing =O/=S
_FREE_ACID = Chem.MolFromSmarts("[$(*=[O,S])][OX2H1,OX1-]")


def _hydrazine_unit_ok(mol, n1: int, n2: int, c_idx: int) -> bool:
    """The hydrazine of a hydrazinecarboxylate: each nitrogen bound by single bonds to carbon (or hydrogen) only, apart
    from the N-N bond and the carbonyl. An N=O ('nitrous amide', the Blue Book), an N=N or a third nitrogen (a
    triazane is the longer parent, 'tetraazane-1-carboxylic acid (PIN)',:24910) is no unit of this name."""
    for idx, other in ((n1, n2), (n2, n1)):
        for b in mol.GetAtomWithIdx(idx).GetBonds():
            nb = b.GetOtherAtom(mol.GetAtomWithIdx(idx))
            if nb.GetIdx() in (other, c_idx):
                continue
            if nb.GetSymbol() != 'C' or b.GetBondType() != Chem.BondType.SINGLE:
                return False
    return True


def substituted_hydrazinecarboxylate_name(mol, n_idx: int, c_idx: int) -> Optional[str]:
    """'R 2-methylhydrazine-1-carboxylate' for the ester R-O-C(=O)-N(R1)-N(R2)R3 of a substituted hydrazinecarboxylic
    acid; None when the carbamate nitrogen ``n_idx`` is not bonded to a second acyclic nitrogen (the carbamic-acid
    builder names it); '' when it is and a part cannot be named (the caller declines).

    The hydrazine nitrogens are numbered from the carbonyl (1, then 2) and the substituents cited with those locants, as
    the book's carboxamides of the same acid: '2-(propan-2-ylidene)hydrazine-1-carboxamide (PIN)' (the Blue Book),
    'N,1-dimethylhydrazine-1-carboxamide (PIN)' (:38637); the substituents are named and ordered by the helpers of
    ``rules.inorganic_acids.name_hydrazinecarboxamide``,."""
    n1 = mol.GetAtomWithIdx(n_idx)
    if not _plain_nitrogen(n1):
        return None
    n2 = [nb for nb in n1.GetNeighbors() if nb.GetIdx() != c_idx and _plain_nitrogen(nb)
          and mol.GetBondBetweenAtoms(n_idx, nb.GetIdx()).GetBondType() == Chem.BondType.SINGLE]
    if len(n2) != 1 or not _hydrazine_unit_ok(mol, n_idx, n2[0].GetIdx(), c_idx):
        return None
    if mol.HasSubstructMatch(_FREE_ACID):
        return ''          # a free acid is the parent; the ester is a prefix of it
    c_atom = mol.GetAtomWithIdx(c_idx)
    carbonyl_o = ester_o = r_start = None
    for nb in c_atom.GetNeighbors():
        if nb.GetSymbol() != 'O':
            continue
        if mol.GetBondBetweenAtoms(c_idx, nb.GetIdx()).GetBondType() == Chem.BondType.DOUBLE:
            carbonyl_o = nb.GetIdx()
        else:
            ester_o = nb.GetIdx()
            r_start = next((o.GetIdx() for o in nb.GetNeighbors() if o.GetIdx() != c_idx), None)
    if carbonyl_o is None or r_start is None:
        return ''
    # the R side of the ester: every atom reached from R without crossing the ester oxygen
    r_side, stack = {r_start}, [r_start]
    while stack:
        for nb in mol.GetAtomWithIdx(stack.pop()).GetNeighbors():
            j = nb.GetIdx()
            if j != ester_o and j not in r_side:
                r_side.add(j)
                stack.append(j)
    if n_idx in r_side:
        return ''          # the ester closes a ring through the hydrazine: no two parts
    from ..rules.inorganic_acids import _hzc_assemble_prefix, _hzc_unit_substituents
    centre = {'C': c_idx, 'O': carbonyl_o, 'N': ester_o, 'N1': n_idx, 'N2': n2[0].GetIdx()}
    subs = _hzc_unit_substituents(mol, centre, frozenset(r_side))
    if subs is None:
        return ''
    from .composer import _name_r_group
    r_name = _name_r_group(mol, r_start, exclude_atoms={ester_o, c_idx, carbonyl_o, n_idx})
    if not r_name:
        return ''
    if not subs:
        return f"{r_name} hydrazinecarboxylate"
    return f"{r_name} {_hzc_assemble_prefix(subs)}hydrazine-1-carboxylate"


def hydrazine_second_nitrogen(mol, n_idx: int, c_idx: int) -> Optional[int]:
    """The second nitrogen of a hydrazinecarboxylic acid unit R-O-C(=O)-N(R1)-N(R2)R3 whose first nitrogen is ``n_idx``:
    the free NH2 (:func:`free_hydrazine_nitrogen`) or the one plain acyclic nitrogen single-bonded to a plain acyclic
    ``n_idx`` (:func:`substituted_hydrazinecarboxylate_name`); None otherwise."""
    free = free_hydrazine_nitrogen(mol, n_idx, c_idx)
    if free is not None:
        return free
    if not _plain_nitrogen(mol.GetAtomWithIdx(n_idx)):
        return None
    n2 = [nb.GetIdx() for nb in mol.GetAtomWithIdx(n_idx).GetNeighbors() if nb.GetIdx() != c_idx
          and _plain_nitrogen(nb) and mol.GetBondBetweenAtoms(n_idx, nb.GetIdx()).GetBondType() == Chem.BondType.SINGLE]
    return n2[0] if len(n2) == 1 and _hydrazine_unit_ok(mol, n_idx, n2[0], c_idx) else None
