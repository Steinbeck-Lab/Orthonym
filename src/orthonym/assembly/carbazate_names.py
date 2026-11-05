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
        return None
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
