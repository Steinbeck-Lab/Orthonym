"""Atom-by-atom comparison of a drawing with the SMILES a name reads back to, for drawings whose standard InChIKey
cannot see a hydrogen.

The standard InChIKey does not see the hydrogens of an over-valent halogen: ``C[Cl-]1CCCCC1`` and ``C[ClH-]1CCCCC1``
have one key, and so do ``C1CC[Cl]CC1`` and ``C1CC[ClH]CC1``. A name of such a drawing that reads back to the key can
still hold one hydrogen more or less on the halogen.:func:`depiction_matches` compares the two graphs with every atom
labelled by its element, isotope, formal charge and hydrogen count, so that difference is seen.

Hydrogen counts follow the OpenSMILES rules: a bracket atom has the hydrogens written in it; an atom outside brackets
takes the hydrogens of the lowest normal valence at or above its bond count, and a halogen's only normal valence is 1,
so an unbracketed halogen with two or more bonds has none. RDKit gives an unbracketed iodine with two bonds one
hydrogen (its valence list holds 3), and OPSIN writes the λ3-iodanyl radical unbracketed ('1λ3-iodinan-1-yl' ->
``I1CCCCC1``), so RDKit's count would accept an iodine '-yl' name for an ``[IH]`` drawing. Radical electrons are not
compared: both sides take them from the same rule, so they add no evidence (the hydrogens, charge and bonds do).
"""
from typing import Optional

from rdkit import Chem

_HALOGENS = frozenset((9, 17, 35, 53))
_RELAXED = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES


def _relaxed(smiles: str) -> Optional[Chem.Mol]:
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    try:
        mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(mol, _RELAXED)
    except Exception:  # noqa: BLE001 -- a SMILES that fails another step has no depiction to compare
        return None
    return mol


def _hydrogens(atom: Chem.Atom) -> int:
    """The hydrogen count of ``atom`` by the OpenSMILES rules (see the module docstring)."""
    if atom.GetNoImplicit() or atom.GetAtomicNum() not in _HALOGENS:
        return atom.GetTotalNumHs()
    bonds = sum(b.GetValenceContrib(atom) for b in atom.GetBonds()) + atom.GetNumExplicitHs()
    return atom.GetNumExplicitHs() + (1 if bonds == 0 else 0)


def depiction_key(mol: Chem.Mol, halogens_only: bool = False) -> Optional[str]:
    """A canonical SMILES of ``mol`` with every heavy atom written in brackets with its OpenSMILES hydrogen count,
    formal charge and isotope; None when RDKit cannot write it. With ``halogens_only`` only the halogens keep their
    hydrogen count and charge, every other atom is written bare and every bond single: the graph and the halogens'
    hydrogens, for a caller that has the full InChIKey for the rest (a nitro group drawn ``N(=O)=O`` or
    ``[N+](=O)[O-]`` is then one depiction)."""
    work = Chem.RWMol(mol)
    for atom, orig in zip(work.GetAtoms(), mol.GetAtoms()):
        keep = not halogens_only or orig.GetAtomicNum() in _HALOGENS
        atom.SetNumExplicitHs(_hydrogens(orig) if keep else 0)
        atom.SetNoImplicit(True)
        atom.SetNumRadicalElectrons(0)
        if not keep:
            atom.SetFormalCharge(0)
        if halogens_only:
            atom.SetIsAromatic(False)
    if halogens_only:
        for bond in work.GetBonds():
            bond.SetBondType(Chem.BondType.SINGLE)
            bond.SetIsAromatic(False)
    try:
        work.UpdatePropertyCache(strict=False)
        return Chem.MolToSmiles(work)
    except Exception:  # noqa: BLE001
        return None


def depiction_matches(input_mol: Chem.Mol, read_smiles: str, halogens_only: bool = False) -> bool:
    """True when ``read_smiles`` (a name's read-back) is the drawing ``input_mol`` atom by atom: one graph with the same
    element, isotope, formal charge and hydrogen count on every atom (both read without RDKit's valence check); with
    ``halogens_only``, the same graph and the same hydrogen count and charge on every halogen (:func:`depiction_key`)."""
    read = _relaxed(read_smiles) if read_smiles else None
    if read is None or input_mol is None or read.GetNumAtoms() != input_mol.GetNumAtoms():
        return False
    left, right = depiction_key(input_mol, halogens_only), depiction_key(read, halogens_only)
    return left is not None and left == right
