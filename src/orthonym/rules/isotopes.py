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


# ---------------------------------------------------------------------------
# P-82.2.1 — isotopic descriptor formatting
# ---------------------------------------------------------------------------


def nuclide_symbol(mass: int, element: str) -> str:
    """Nuclide symbol with the mass number as a leading integer (ASCII form).

    Orthonym emits ASCII (no <sup>); OPSIN accepts the leading-digit form
    (probe: (2-14C)ethan-1-ol -> C([14CH3])O). P-82.2.1.
    """
    return f"{mass}{element}"


def format_isotope_descriptor(groups) -> str:
    """Build the parenthesized isotopic descriptor from grouped labels.

    ``groups`` = list of (locant, mass, element, count):
      locant None  -> no leading locant (single-position parent / front
                      descriptor, e.g. (2H3)methoxybenzene, (12C1)methane).
      locant int   -> the count locants are repeated then hyphen-joined to the
                      nuclide, e.g. (2,2,2-2H3), (2-14C1).
    Multiple groups at (possibly) the same place are cited alphabetically by
    element then by mass number (P-82.2.1 / P-82.3); groups are comma-joined.
    The count subscript is ALWAYS emitted (P-82.2.1 line 43720 + tritium
    parseability, Task 4).
    """
    def _one(locant, mass, element, count):
        sym = nuclide_symbol(mass, element)
        if locant is None:
            return f"{sym}{count}"
        # BB: the locant is repeated once per substituted atom at that position
        # for a grouped multi-count token (2,2,2-2H3); for count 1 a single
        # locant + hyphen (2-14C1).
        loc_part = ",".join(str(locant) for _ in range(count))
        return f"{loc_part}-{sym}{count}"

    # P-82.2.1: alphabetical by element symbol, then by mass number, then locant.
    ordered = sorted(groups, key=lambda g: (g[2], g[1], g[0] if g[0] is not None else -1))
    inner = ",".join(_one(*g) for g in ordered)
    return f"({inner})"
