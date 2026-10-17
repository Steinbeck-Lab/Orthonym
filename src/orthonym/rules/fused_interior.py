"""The default tier's names of the fused parents the Blue Book prints with an interior-atom locant.

``data.fused_interior_pins`` holds the printed parents ('1,3a^1,4,9-tetraazaphenalene',,
the Blue Book,:14491). A molecule whose whole ring system is one of them, with exocyclic
substituents only, is named on that parent: the atoms take the numbering of the entry (the parents have
no symmetry that leaves their heteroatoms in place, so the numbering is fixed by the entry) and the
substituents are named by the producers of the fused catalogue. Only the default tier reads the table
(``validation.twin_spellings.pin_spellings_on``); a wider tier keeps the name it has, because OPSIN 2.9.0
has no grammar for the superscript locant and the wider tier ships only names OPSIN reads. The validity
gate reads the name through its twin spelling (rule ``interior_phenalene``), which decides whether the
structure is the input's.
"""
from typing import Dict, Optional, Tuple, Union

from rdkit import Chem

_PATTERNS: Dict[str, Chem.Mol] = {}


def _interior_core(mol) -> Optional[Tuple[str, Dict[int, Union[int, str]]]]:
    """``(parent name, {atom index: locant})`` when the ring skeleton of ``mol`` is exactly one
    printed parent, else None."""
    from ..data.fused_heterocycles import ring_skeleton_key
    from ..data.fused_interior_pins import INTERIOR_PIN_PARENTS
    skeleton = ring_skeleton_key(mol)
    for smiles, entry in INTERIOR_PIN_PARENTS.items():
        pattern = _PATTERNS.setdefault(smiles, Chem.MolFromSmiles(smiles))
        if skeleton != Chem.MolToSmiles(pattern):
            continue
        match = mol.GetSubstructMatch(pattern)
        if len(match) != pattern.GetNumAtoms():
            return None
        return entry["name"], dict(zip(match, entry["locants"]))
    return None


def interior_pin_name(mol) -> Optional[Tuple[str, None]]:
    """``(name, None)`` for a molecule named on a printed interior-atom parent at the default tier
    (the shape of ``get_fused_heterocycle_name``, whose slot it takes), else None."""
    from ..validation.twin_spellings import pin_spellings_on
    if mol is None or not pin_spellings_on():
        return None
    core = _interior_core(mol)
    if core is None:
        return None
    name, mapping = core
    if mol.GetNumHeavyAtoms() == len(mapping):
        return name, None
    from .fused_rings import (
        _assemble_fused_heterocycle_name, _exocyclic_atoms_accounted, get_fused_heterocycle_substituents)
    if not _exocyclic_atoms_accounted(mol, set(mapping)):
        return None
    substituents = get_fused_heterocycle_substituents(mol, mapping)
    named = _assemble_fused_heterocycle_name(mol, name, substituents, mapping) if substituents else None
    return (named, None) if named else None
