"""Isotopically substituted compound names (IUPAC P-82.2.1 + P-45.4).

Fail-closed decorator layer. Orthonym's core pipeline strips isotope labels
(RDKit perception ignores GetIsotope for skeleton naming), so a labeled mol
would otherwise emit the UNLABELED name — a wrong PIN. This module runs as an
EARLY branch in Orthonym.name(): it strips the labels, names the skeleton in
systematic style (locanted parents), then re-derives the isotopic descriptor
by an INTERNAL ORACLE (OPSIN-parse a candidate name; compare rdkit canonical
SMILES — with isotopes — to the original). If no candidate round-trips, it
returns None and the label is never mis-placed.

BB P-82.2.1 (BlueBookV2.md:43718): the nuclide symbol(s) in parentheses,
preceded by any necessary locant(s), are inserted before the isotopically
substituted part; polysubstitution count is a right subscript to the symbol.
BB P-45.4.1/.4.2/.4.3 (BlueBookV2.md:22212-22232): lowest locants to modified
positions; then to higher atomic number; then to higher mass number.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from rdkit import Chem


def has_isotopes(mol: Optional[Chem.Mol]) -> bool:
    """True iff any atom of ``mol`` carries a non-zero isotope label."""
    if mol is None:
        return False
    return any(a.GetIsotope() != 0 for a in mol.GetAtoms())


def strip_isotopes(mol: Chem.Mol) -> Tuple[Chem.Mol, Dict[int, int]]:
    """Return an isotope-cleared COPY of ``mol`` and a {atom_idx: mass} map.

    Atom indices are preserved by RWMol copy, so the returned map indexes the
    stripped mol AND the original identically. Used to name the skeleton and
    later to place isotopic descriptors at the correct positions.
    """
    copy = Chem.Mol(mol)
    label_map: Dict[int, int] = {}
    for atom in copy.GetAtoms():
        iso = atom.GetIsotope()
        if iso != 0:
            label_map[atom.GetIdx()] = iso
            atom.SetIsotope(0)
    # Deuterium/tritium ([2H]/[3H]) are EXPLICIT hydrogen atoms; clearing their
    # isotope leaves plain explicit [H] atoms, which change the canonical SMILES
    # ([H]C([H])([H])CO != CCO) and could perturb skeleton perception. Collapse
    # them back into the implicit H count so the skeleton is the true unlabeled
    # parent. RemoveHs is a no-op for heavy-atom labels (14C etc). label_map
    # keys index the ORIGINAL mol (element symbols are read there in Task 3), so
    # the index shift RemoveHs introduces on the returned copy is irrelevant —
    # the copy is used only for its canonical skeleton SMILES + heavy-atom count.
    try:
        copy = Chem.RemoveHs(copy)
    except Exception:
        pass
    return copy, label_map


def decorate_isotopic_name(smiles: str, style: str, namer) -> Optional[str]:
    """Entry point for Orthonym.name() when the mol carries isotopes.

    Task-1 stub: strips + names the skeleton, but returns None for any real
    label so the name() branch is inert until Task 3 fills in the oracle. This
    keeps HEAD byte-identical for every LABELED input until the oracle lands,
    and byte-identical for UNLABELED inputs forever (has_isotopes gates entry).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not has_isotopes(mol):
        return None
    # Task 3 completes this. Stub fails closed.
    return None
