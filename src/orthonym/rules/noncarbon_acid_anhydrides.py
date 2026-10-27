"""Mixed anhydrides of an acid named by a suffix with a monobasic noncarbon oxoacid of P, As, Sb or B.

 "Pseudoesters" (the Blue Book): when 'E', in R-CO-O-E, is B, Al, In, Ga, Tl, Si, Ge, Sn, Pb,
N (cyclic), P, As, Sb or Bi, "the pseudoester is named as a traditional ester, unless other names must be
selected in accordance with the seniority order of classes, in decreasing order: salts > acids > anhydrides >
esters. For anhydrides, see." ('trimethylsilyl acetate (PIN)': silanol is no acid; 'phosphinan-1-yl
benzoate (PIN)': the ring phosphorus is no acid centre; 'acetic phosphinous anhydride (PIN)',:32064.)
 "Mixed anhydrides" (:32253), "Mixed anhydrides with carbonic acid, cyanic acid, and inorganic acids
are named as anhydrides": 'benzoic phosphinous anhydride (PIN)' (:32280). "Anhydrides of mononuclear
noncarbon oxoacids" (:35982): "Neutral anhydrides formed between acids named by suffixes and mononuclear
noncarbon oxoacids... are named in the same way as described for anhydrides derived from carboxylic and the sulfur
acids... The names of the acids are cited in alphabetical order followed by the class name 'anhydride'...; such
names lead to preferred IUPAC names": 'acetic dimethylborinic anhydride (PIN)', 'acetic dimethylarsinic anhydride
(PIN)' (:35992).

The E-OH acid is monobasic when E bears, besides the bridging oxygen, only hydrogen, carbon and at most one
terminal =O (phosphinous, arsinous, stibinous, borinic; phosphinic, arsinic, stibinic). A polybasic acid
('boric', 'phosphoric') takes a numerical prefix of the anhydride linkages,:32283) and is left to the
producers that name it. The acid is named by the general pipeline, and the anhydride is built only when that name
is the name of an acid.
"""
import re
from typing import Optional

from rdkit import Chem

_E_ELEMENTS = frozenset({"P", "As", "Sb", "B"})
_ACID = " acid"
_ACYL_OXYGEN = Chem.MolFromSmarts("[CX3,SX4](=[OX1])[OX2]")


def _free_acid_of_the_noncarbon_side(mol, ester_o: int, e_idx: int) -> Optional[str]:
    """The SMILES of E-OH: the molecule with the bond O-E cut, the atoms of E's side kept, and a hydroxyl on E."""
    rw = Chem.RWMol(mol)
    rw.RemoveBond(ester_o, e_idx)
    new_o = rw.AddAtom(Chem.Atom(8))
    rw.AddBond(e_idx, new_o, Chem.BondType.SINGLE)
    try:
        cut = rw.GetMol()
        Chem.SanitizeMol(cut)
    except (ValueError, RuntimeError):
        return None
    side = next((frag for frag in Chem.GetMolFrags(cut) if e_idx in frag), None)
    if side is None or ester_o in side:
        return None          # the two sides are one ring system: no two acids
    return Chem.MolFragmentToSmiles(cut, atomsToUse=list(side), canonical=True, isomericSmiles=True)


def _borinic_acid_name(mol, ester_o: int, e_idx: int) -> Optional[str]:
    """'dimethylborinic acid' for the boron of R2B-O-(acyl): two carbon ligands and the bridging oxygen on the boron.

     (the Blue Book): "The names boric acid, boronic acid, and borinic acid are preselected retained
    names for the compounds B(OH)3, HB(OH)2, and H2B(OH), respectively. They are preferred IUPAC names for deriving
    the names of the corresponding salts, esters, or anhydrides; or when the hydrogen atom(s) attached to the boron
    atom is (are) substituted, for example methylboronic acid for CH3-B(OH)2"; 'dimethylborinothioic acid (PIN)'
    for (CH3)2B(SH) (:37325). The organyl groups are cited as the substituents of a mononuclear centre,
    (:7272): 'dimethyl', 'ethyl(methyl)'."""
    from ..assembly.prefix_marks import mononuclear_substituent_block
    from .substituent_purity import organyl_prefix_name
    ligands = [nb.GetIdx() for nb in mol.GetAtomWithIdx(e_idx).GetNeighbors() if nb.GetIdx() != ester_o]
    if len(ligands) != 2 or any(mol.GetAtomWithIdx(i).GetSymbol() != "C" for i in ligands):
        return None
    names = [organyl_prefix_name(mol, i, e_idx) for i in ligands]
    block = None if None in names else mononuclear_substituent_block(names, max_count=3)
    return f"{block}borinic acid" if block else None


def mixed_noncarbon_anhydride(mol, ester_o: int, e_idx: int, acid_name: str) -> Optional[str]:
    """'<acid> <acid> anhydride' for R-CO-O-E (or R-SO2-O-E...) with E the acid centre of a monobasic noncarbon
    oxoacid, else None. ``acid_name`` is the name of the acyl side as a free acid ('acetic acid')."""
    e = mol.GetAtomWithIdx(e_idx)
    if (e.GetSymbol() not in _E_ELEMENTS or e.IsInRing() or e.GetFormalCharge() != 0
            or e.GetNumRadicalElectrons() != 0 or e.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED):
        return None
    oxo = 0
    for nb in e.GetNeighbors():
        if nb.GetIdx() == ester_o:
            continue
        bond = mol.GetBondBetweenAtoms(e_idx, nb.GetIdx())
        if nb.GetSymbol() == "O" and nb.GetDegree() == 1 and bond.GetBondTypeAsDouble() == 2.0:
            oxo += 1
        elif not (nb.GetSymbol() == "C" and bond.GetBondTypeAsDouble() == 1.0):
            return None      # a second oxygen, a halogen, nitrogen: not the monobasic acids of this rule
    if oxo > 1 or not acid_name.endswith(_ACID):
        return None
    # one acyl-oxygen group in the molecule: a dibasic acyl side ('oxalic acid' with both groups joined to E) is no
    # '<acid> <acid> anhydride' of this rule and keeps the ester name
    if len(mol.GetSubstructMatches(_ACYL_OXYGEN)) != 1:
        return None
    acid_smiles = _free_acid_of_the_noncarbon_side(mol, ester_o, e_idx)
    if acid_smiles is None:
        return None
    from ..assembly.fragment_naming import name_fragment_recursively
    from ..metrics.provenance import get_provenance, name_carries_non_pin_part, restore_provenance
    before = get_provenance()
    try:
        other = name_fragment_recursively(acid_smiles)
        not_a_pin = name_carries_non_pin_part(get_provenance(), other)
    except Exception:
        return None
    finally:
        restore_provenance(before)     # the label of the call is the name that ships, not a probe of its acid
    if e.GetSymbol() == "B" and not (other or "").endswith(_ACID):
        other, not_a_pin = _borinic_acid_name(mol, ester_o, e_idx), False
    if (not other or not_a_pin or not other.endswith(_ACID) or " " in other[:-len(_ACID)]):
        return None
    from ..assembly.naming_utils import cip_descriptor_rank_key
    # (:32253): "citing in alphabetical order the names of the two acids"; an acid name is a whole name,
    # and (:3477) "The name of a prefix... is considered to begin with the first letter of its complete
    # name": the 'di' of 'dimethylphosphinous' is part of the name that is alphabetized ('dimethylphosphinous formic
    # anhydride'), not a multiplier of the order key
    words = sorted([acid_name[:-len(_ACID)], other[:-len(_ACID)]],
                   key=lambda w: (re.sub("[^a-z]", "", w.lower()), cip_descriptor_rank_key(w)))
    return f"{words[0]} {words[1]} anhydride"
