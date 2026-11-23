"""The oxidation number of the metal of an adduct component drawn with covalent ligands: 'oxomolybdenum(II)'.

A reader balances the charges of the components of an adduct name itself: OPSIN gives a neutral metal that carries
no oxidation number the charge of a typical oxidation state when the other components are ions ('pyridine-2,6-
dicarboxylic acid—hydroxy(4-nitrophenyl)azanide—oxomolybdenum (1/1/1)' reads [Mo+]), and keeps the metal of a
component that cites its oxidation number (the Roman numeral after the element name, the Stock number of inorganic
nomenclature) at that number less the bonds drawn to it. The oxidation number of a metal bonded only to ligands
counted as anions (chloro and alkyl one each, oxo and imido two, nitrido three) is the sum of the orders of those
bonds, so a component name with that number reads back to the metal as drawn: 'oxomolybdenum(II)' to O=[Mo]
(OPSIN StructureBuilder.getMetalsThatCanBeImplicitlyCations leaves a metal with an oxidation number uncharged).

The number is computed by that rule, never fitted to the read-back: a carbonyl or isocyanide ligand is neutral and
counts zero (a cyanide one), and a component whose metal holds a metal-carbon or metal-silicon double or triple bond
(a carbene, carbyne or silylidene, whose formal count is a convention), another metal, a dative or aromatic bond, a
charged or radical neighbour, or two neighbours bonded to each other (an eta drawing) gets no number. An adduct name is no PIN, the Blue Book), so the spelling is offered at the
best-effort tier only, after the name without it fails, and ships only when OPSIN reads the whole name back to the
drawn structure (``rules.adducts._parse_reproduces_depiction``). Only the metals that ``data/metal_scope.py`` admits
to ``ZERO_VALENT`` (an oxidation number cited in an adduct component name).
"""
from typing import List, Optional, Tuple

from rdkit import Chem

from .metal_atoms import METAL_ELEMENT_NAMES

_ROMAN = {1: 'I', 2: 'II', 3: 'III', 4: 'IV', 5: 'V', 6: 'VI', 7: 'VII', 8: 'VIII'}
_ORDER = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3}
#: ligand atoms whose bond to the metal counts by its order (halide, oxo, sulfido, imido, nitrido, phosphido, hydrido)
_X_LIGAND_ATOMS = frozenset({'H', 'N', 'P', 'As', 'O', 'S', 'Se', 'Te', 'F', 'Cl', 'Br', 'I'})
#: atoms of a group 14 or 13 ligand that count one on a single bond (alkyl, silyl, germyl, stannyl, boryl)
_SIGMA_LIGAND_ATOMS = frozenset({'C', 'Si', 'Ge', 'Sn', 'B'})


def stock_number(frag) -> Optional[Tuple[str, int]]:
    """(metal symbol, oxidation number) of the one metal atom of ``frag`` by the rule above, or None."""
    from ..data.metal_scope import ZERO_VALENT, has_mechanism
    metals = [a for a in frag.GetAtoms() if a.GetSymbol() in METAL_ELEMENT_NAMES
              and a.GetSymbol() not in _SIGMA_LIGAND_ATOMS | _X_LIGAND_ATOMS]
    if len(metals) != 1 or frag.GetNumAtoms() < 2:
        return None
    metal = metals[0]
    if (metal.GetFormalCharge() or metal.GetNumRadicalElectrons() or not metal.GetDegree()
            or not has_mechanism(metal.GetSymbol(), ZERO_VALENT)):
        return None
    neighbours = [a.GetIdx() for a in metal.GetNeighbors()]
    if any(frag.GetAtomWithIdx(i).GetNumRadicalElectrons() for i in neighbours) or any(
            frag.GetBondBetweenAtoms(i, j) is not None for k, i in enumerate(neighbours) for j in neighbours[k + 1:]):
        return None  # a radical ligand atom, or two bonded ligand atoms (an eta drawing, a three-membered ring)
    total = metal.GetTotalNumHs()
    for bond in metal.GetBonds():
        order, other = _ORDER.get(bond.GetBondType()), bond.GetOtherAtom(metal)
        if order is None or other.GetFormalCharge():
            return None
        if other.GetSymbol() == 'C' and _neutral_carbon_ligand(other, metal):
            continue
        if other.GetSymbol() in _X_LIGAND_ATOMS or (other.GetSymbol() in _SIGMA_LIGAND_ATOMS and order == 1):
            total += order
            continue
        return None
    return (metal.GetSymbol(), total) if total in _ROMAN else None


def _neutral_carbon_ligand(carbon, metal) -> bool:
    """A carbonyl (M-C#O, M=C=O) or isocyanide (M-C#N-R) carbon: a neutral ligand that counts zero."""
    others = [b for b in carbon.GetBonds() if b.GetOtherAtom(carbon).GetIdx() != metal.GetIdx()]
    if carbon.GetTotalNumHs() or len(others) != 1:
        return False
    far = others[0].GetOtherAtom(carbon)
    if far.GetSymbol() == 'O':
        return others[0].GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE)
    # an isocyanide nitrogen holds its group; a terminal one is a cyanide ligand, counted as an anion
    return far.GetSymbol() == 'N' and others[0].GetBondType() == Chem.BondType.TRIPLE and far.GetDegree() == 2


def stock_numbered(ordered, named, frag_mols) -> Optional[List[Tuple[str, int]]]:
    """``named`` with each component of one covalently bonded metal cited with its oxidation number
    ('oxomolybdenum(II)'), or None when no component takes one."""
    out, changed = [], False
    for (smi, _count), (name, count) in zip(ordered, named):
        try:
            found = stock_number(frag_mols[smi])
        except (RuntimeError, ValueError):  # an atom RDKit cannot count the hydrogens of: no number
            found = None
        if found is not None:
            element = METAL_ELEMENT_NAMES[found[0]]
            if name.endswith(element) and name != element:
                name, changed = f"{name}({_ROMAN[found[1]]})", True
        out.append((name, count))
    return out if changed else None
