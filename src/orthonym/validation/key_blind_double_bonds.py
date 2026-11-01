"""Double-bond configurations the standard InChIKey cannot see: does a name's OPSIN parse carry the input's E/Z?

The standard InChI leaves out the configuration of some double bonds whose atoms take part in a mobile-hydrogen
group (an amidine C=N, for example), so the two configurations share one key:

    C/C(N)=N/C vs C/C(N)=N\\C ((Z)- and (E)-N'-methylethanimidamide)

and a name with the wrong descriptor passes every gate that accepts a name on full-key equality alone, as it would
pass with the wrong hydrogens of a hypervalent halogen (``key_blind_hydrogens``) or the wrong site of a hydron
(``protonation_identity``).

The rule asks only what the key cannot: when the input draws a configured double bond whose configuration its own
InChI leaves out, and the OPSIN parse is the same constitution drawn the same way (equal canonical SMILES without
stereo), the two canonical isomeric SMILES must be equal too. A parse drawn as another tautomer, or a molecule with
no such bond, is never rejected by it ('n/a'), so a correct name is never lost to a drawing difference. A parse
that carries no configuration on those bonds is an omission ('omission'), not another molecule: the name keeps the
label it had, as for any stereodescriptor a name leaves out; only the opposite configuration ('mismatch') refuses it.
"""
from typing import Optional

from rdkit import Chem


def _key_blind_bonds(mol) -> list:
    """The indices of the configured double bonds of ``mol`` its standard InChI does not encode (each bond tested
    alone: the InChI with the bond's configuration cleared equals the InChI of the molecule)."""
    from rdkit.Chem.inchi import MolToInchi
    configured = [b.GetIdx() for b in mol.GetBonds()
                  if b.GetBondType() == Chem.BondType.DOUBLE and b.GetStereo() != Chem.BondStereo.STEREONONE]
    if not configured:
        return []
    full = MolToInchi(mol)
    return [idx for idx in configured if MolToInchi(_cleared(mol, [idx])) == full]


def _cleared(mol, bond_indices):
    """A copy of ``mol`` with the configuration of the double bonds ``bond_indices`` removed."""
    cleared = Chem.Mol(mol)
    for idx in bond_indices:
        bond = cleared.GetBondWithIdx(idx)
        bond.SetStereo(Chem.BondStereo.STEREONONE)
        ends = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        for b in cleared.GetBonds():
            if b.GetBondDir() in (Chem.BondDir.ENDUPRIGHT, Chem.BondDir.ENDDOWNRIGHT) and (
                    b.GetBeginAtomIdx() in ends or b.GetEndAtomIdx() in ends):
                b.SetBondDir(Chem.BondDir.NONE)
    return cleared


def key_blind_double_bond_verdict(smiles: Optional[str], opsin_smiles: Optional[str]) -> str:
    """'mismatch' when the input draws a configured double bond its InChI leaves out and the OPSIN parse, drawn as
    the same constitution, has another configuration; 'omission' when the parse leaves those configurations out;
    'match' when it has the same; 'n/a' otherwise."""
    if not smiles or not opsin_smiles:
        return "n/a"
    mol_in = Chem.MolFromSmiles(smiles)
    mol_out = Chem.MolFromSmiles(opsin_smiles)
    blind = _key_blind_bonds(mol_in) if mol_in is not None and mol_out is not None else []
    if not blind:
        return "n/a"
    if Chem.MolToSmiles(mol_in, isomericSmiles=False) != Chem.MolToSmiles(mol_out, isomericSmiles=False):
        return "n/a"
    out = Chem.MolToSmiles(mol_out)
    if out == Chem.MolToSmiles(mol_in):
        return "match"
    return "omission" if out == Chem.MolToSmiles(_cleared(mol_in, blind)) else "mismatch"
