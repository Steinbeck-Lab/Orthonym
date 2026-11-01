"""The preselected name of a von Baeyer or spiro system made of one noncarbon element.

 "Heterocyclic noncarbon von Baeyer and spiro compounds" (the Blue Book),
(:23614): "Preselected names for von Baeyer compounds and for spiro compounds having only monocyclic
components and consisting entirely of heteroatoms of the same kind are names formed by citing the
appropriate prefix, such as 'bicyclo', 'spiro', etc., and descriptor enclosed within square brackets
followed by a numerical prefix for the total number of heteroatoms and the name of the mononuclear
parent hydride; alternative names for use in general nomenclature are formed by skeletal replacement
('a') nomenclature (see and ". The example is 'bicyclo[4.2.1]nonasilane (preselected
name)' beside 'nonasilabicyclo[4.2.1]nonane' (:23618; HOMOGENEOUS HETEROCYCLIC VON BAEYER PARENT
HYDRIDES,:9797, example:9810). The 'a' name is therefore not the preselected one, and
``ring_replacement.ReplacementPrefix.preselected_only`` makes every PIN producer refuse it.

This module builds the preselected name for the cage that is the whole structure (no group on it, no
charge, no isotope, no stereo, standard bonding numbers, one single-bonded skeleton of one element). A
cage with a group on it has no such name here and keeps the 'a' name of the wider tiers. Its twin spelling,
the 'a' name OPSIN reads (``validation.preselected_cage_twin``), carries the read-back.

Boron is left out: (:7992) and (:8848) name the homogeneous chains and rings "other than
boron hydrides", whose names are those of (:37105, 'triborane(5)'). The chalcogens are left
out because their bridgeheads have a nonstandard bonding number.
"""
from typing import Dict, Optional, Tuple

from rdkit import Chem

from ..data.chain_names import multiplying_prefix

#: element -> (stem of the mononuclear parent hydride without 'ane', standard bonding number): 'silane',
#: 'germane', 'stannane', 'plumbane', 'azane', 'phosphane', 'arsane', 'stibane', 'bismuthane',
#: Table 2.1,:7913)
HYDRIDE_STEMS: Dict[str, Tuple[str, int]] = {
    'Si': ('sil', 4), 'Ge': ('germ', 4), 'Sn': ('stann', 4), 'Pb': ('plumb', 4),
    'N': ('az', 3), 'P': ('phosph', 3), 'As': ('ars', 3), 'Sb': ('stib', 3), 'Bi': ('bismuth', 3),
}


def is_bare_single_element_cage(mol) -> Optional[str]:
    """The element symbol when ``mol`` is one connected skeleton of a single element of
    :data:`HYDRIDE_STEMS`, every bond single, every atom at its standard bonding number with the hydrogens
    that fill it, and no charge, isotope, radical or stereo; else None."""
    if mol is None or mol.GetNumAtoms() < 4 or len(Chem.GetMolFrags(mol)) != 1:
        return None
    symbols = {a.GetSymbol() for a in mol.GetAtoms()}
    if len(symbols) != 1:
        return None
    symbol = next(iter(symbols))
    if symbol not in HYDRIDE_STEMS:
        return None
    standard = HYDRIDE_STEMS[symbol][1]
    for atom in mol.GetAtoms():
        if (atom.GetFormalCharge() or atom.GetIsotope() or atom.GetNumRadicalElectrons()
                or atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
                or atom.GetDegree() + atom.GetTotalNumHs() != standard):
            return None
    if any(b.GetBondType() != Chem.BondType.SINGLE or b.GetStereo() != Chem.BondStereo.STEREONONE
           for b in mol.GetBonds()):
        return None
    return symbol


def preselected_cage_name(mol, descriptor: str) -> Optional[str]:
    """``bicyclo[4.2.1]nonasilane``: the descriptor, the multiplying prefix of the number of skeletal atoms
    and the mononuclear parent hydride,:23614), or None when ``mol`` is not exactly one cage
    of one element."""
    symbol = is_bare_single_element_cage(mol)
    if symbol is None or not descriptor:
        return None
    return f"{descriptor}{multiplying_prefix(mol.GetNumAtoms())}{HYDRIDE_STEMS[symbol][0]}ane"


def preselected_cage_result(mol, ring_type: str):
    """``(name, ring_atoms, atom_to_locant, True)`` for the dispatch of ``composer._assemble_complex_ring_name``
    when ``mol`` is exactly one von Baeyer ('bicyclo', 'polycyclic-bridged') or spiro cage of one element;
    else None. The numbering is that of the hydrocarbon (a),:9803: "as described for bi- and
    polycyclic hydrocarbons in "), and every atom of the skeleton is the same, so it fixes nothing."""
    if ring_type not in ('bicyclo', 'polycyclic-bridged', 'spiro') or is_bare_single_element_cage(mol) is None:
        return None
    ring_atoms = {i for ring in mol.GetRingInfo().AtomRings() for i in ring}
    if len(ring_atoms) != mol.GetNumAtoms():
        return None
    if ring_type == 'spiro':
        from .spiro import generate_spiro_descriptor
        descriptor, numbering = generate_spiro_descriptor(mol), {}
    else:
        from .polycyclic import VonBaeyerAnalyzer
        try:
            found = VonBaeyerAnalyzer().analyze(mol, ring_atoms)
        except Exception:  # an unanalysable cage keeps the refusal of the strict path
            return None
        descriptor, numbering = found.descriptor_string, found.numbering
    name = preselected_cage_name(mol, descriptor)
    return None if name is None else (name, ring_atoms, numbering, True)
