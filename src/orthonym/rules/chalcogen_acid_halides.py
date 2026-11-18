"""Halides of the mononuclear oxoacids of sulfur, selenium and tellurium.

 "Acid halides and pseudohalides" (the Blue Book): "preferred IUPAC names of acid halides and
pseudohalides are formed by adding the class name(s) of a halide or pseudohalide to that of the acid. Exceptionally,
in accordance with tradition and the recommended nomenclature of inorganic compounds (ref. 12), halides and
pseudohalides with identical atoms or groups and derived from phosphoric acid, sulfuric acid, selenic acid and
telluric acid are named by adding the class name(s) to the acyl group name 'phosphoryl', 'sulfuryl',... 'selenonyl'
and 'telluronyl', and not to the name of the acid itself."

So SOCl2 is 'sulfurous dichloride' (the acid is sulfurous acid, not one of the four), SO2ClF 'sulfuric chloride
fluoride' (the halides are not identical) and SO2Br2 'sulfuryl dibromide', SeO2F2 'selenonyl difluoride' (the
exception; SO2Cl2 and SO2F2 are named first by ``inorganic_acids._INORGANIC_ACID_DERIVATIVES``). The class words are
cited in alphanumerical order and identical ones are multiplied (``functional_replacement.build_p_frn_halide_name``). A compound without carbon gets no PIN; the name is the
preselected one. The S-O bond of a charge-separated drawing ('[O-][S+](Cl)Cl') is the oxo group of the same compound;
the whole name still ships only through the gates that compare it with the drawing.
"""
from typing import Optional

from rdkit import Chem

#: element -> (acid stem with one oxo group, acid stem with two, acyl word of the -ic acid with identical halides)
_STEMS = {"S": ("sulfurous", "sulfuric", "sulfuryl"),
          "Se": ("selenous", "selenic", "selenonyl"),
          "Te": ("tellurous", "telluric", "telluronyl")}

_HALIDE_WORD = {"F": "fluoride", "Cl": "chloride", "Br": "bromide", "I": "iodide"}


def _oxo(bond, centre) -> bool:
    """True for the oxo group of ``centre``: a double bond to a neutral oxygen, or the single bond of the
    charge-separated drawing ([S+]-[O-], the oxygen bonded to nothing else)."""
    o = bond.GetOtherAtom(centre)
    if o.GetSymbol() != "O" or o.GetDegree() != 1 or o.GetTotalNumHs():
        return False
    if bond.GetBondType() == Chem.BondType.DOUBLE:
        return o.GetFormalCharge() == 0
    return bond.GetBondType() == Chem.BondType.SINGLE and o.GetFormalCharge() == -1


def name_chalcogen_oxoacid_halide(mol) -> Optional[str]:
    """'sulfurous dichloride' for O=S(Cl)Cl, 'sulfuric chloride fluoride' for O=S(=O)(F)Cl, 'selenous dichloride' for
    O=[Se](Cl)Cl; None for anything else: more than one fragment, a net charge, an atom besides the centre, its oxo
    oxygens and two halogen atoms, a hydrogen on the centre."""
    if mol is None or len(Chem.GetMolFrags(mol)) != 1 or Chem.GetFormalCharge(mol) != 0:
        return None
    from ..opsin_capabilities import label_free_names_withheld
    if label_free_names_withheld():
        return None  # the isotope decorator's second pass: '(18O)sulfurous dichloride' reads unlabelled
    centres = [a for a in mol.GetAtoms() if a.GetSymbol() in _STEMS]
    if len(centres) != 1:
        return None
    centre = centres[0]
    if centre.GetTotalNumHs() or centre.GetNumRadicalElectrons():
        return None
    oxo, halides = 0, {}
    for b in centre.GetBonds():
        o = b.GetOtherAtom(centre)
        if _oxo(b, centre):
            oxo += 1
        elif (b.GetBondType() == Chem.BondType.SINGLE and o.GetSymbol() in _HALIDE_WORD and o.GetDegree() == 1
              and o.GetFormalCharge() == 0 and not o.GetIsotope()):
            word = _HALIDE_WORD[o.GetSymbol()]
            halides[word] = halides.get(word, 0) + 1
        else:
            return None
    if oxo not in (1, 2) or sum(halides.values()) != 2 or mol.GetNumAtoms() != 1 + oxo + 2:
        return None
    if centre.GetFormalCharge() != (1 if any(a.GetFormalCharge() == -1 for a in mol.GetAtoms()) else 0):
        return None
    ous, ic, acyl = _STEMS[centre.GetSymbol()]
    from .functional_replacement import build_acyl_halide_name, build_p_frn_halide_name
    if oxo == 2 and len(halides) == 1:  # the acyl-word exception: 'sulfuryl dibromide', 'selenonyl difluoride'
        (word, count), = halides.items()
        return build_acyl_halide_name(acyl, word, count)
    return build_p_frn_halide_name(ous if oxo == 1 else ic, halides)
