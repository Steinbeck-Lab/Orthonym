"""The carve-outs of the validity gate that ship a name OPSIN misreads, bound to the input they were built for
(``namer._final_opsin_validity_gate``)."""
from typing import Optional

from rdkit import Chem


def polyacid_carveout_bound(name: str, smiles: Optional[str], opsin_smiles: Optional[str]) -> bool:
    """The polyacid carve-out ships ``name`` only when its producer builds exactly this name for this input
    (``inorganic_acids.name_polyacid_derivative``, which reads the structure) and OPSIN's reading, though wrong in
    its locants, is an isomer of the input (same molecular formula: no atom lost or added). A name that only
    matches the carve-out's pattern proves nothing about the structure."""
    if not smiles or not opsin_smiles:
        return False
    try:
        from rdkit.Chem.rdMolDescriptors import CalcMolFormula
        from ..rules.inorganic_acids import name_polyacid_derivative
        mol, read = Chem.MolFromSmiles(smiles), Chem.MolFromSmiles(opsin_smiles)
        return (mol is not None and read is not None and name_polyacid_derivative(mol) == name
                and CalcMolFormula(mol) == CalcMolFormula(read))
    except Exception:
        return False
