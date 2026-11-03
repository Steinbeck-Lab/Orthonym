"""The retained alkoxide names, substituted: '4-chlorophenoxide', '2-chloroethoxide',.

``## SALTS OF HYDROXY COMPOUNDS, HYDROPEROXY COMPOUNDS AND THEIR CHALCOGEN ANALOGUES`` (the Blue Book),
 (:28180): "The traditional names methoxide, ethoxide, propoxide, butoxide, phenoxide, and aminoxide... are
retained as preferred IUPAC names and may be substituted in the same way as the corresponding alcohols. The
traditional name tert-butoxide... is also retained as a preferred IUPAC name but cannot be substituted. The
traditional name isopropoxide... is retained for general nomenclature but cannot be substituted." Examples: 'sodium
propoxide (PIN) sodium propan-1-olate', 'lithium phenoxide (PIN) lithium phenolate' (:28186,:28190);
"Anions derived from hydroxy compounds" says the same (:41017).

 (:28204): "Partial salts of polyols and their chalcogen analogues are named substitutively on the basis of the
corresponding anion": 'sodium 3-hydroxypropan-1-olate (PIN)' (:28208), so a molecule with another hydroxy-class
group (O, S, Se, Te; an acid OH and an enol included) keeps the 'olate' name. A group senior to OH (an acid, an ester)
makes another parent than the alcohol, so the senior-group route keeps 'olate' too.

The substitutive 'olate' name of the anion is built first; its parent word is the one the alcohol takes
('4-chlorophenol', '2-chloroethan-1-ol'), so the retained word replaces it with the same substituents and locants:
'phenolate' -> 'phenoxide', 'methanolate', 'ethan-1-olate', 'propan-1-olate', 'butan-1-olate' -> 'methoxide',
'ethoxide', 'propoxide', 'butoxide'. The structure decides that the parent word names the right atoms (an acyclic
saturated carbon for the chains, a carbon of a benzene ring that is in no other ring for phenol), and the caller reads
the name back by its full InChIKey.
"""
from __future__ import annotations

from typing import Optional

from rdkit import Chem

#: the 'olate' word of the alcohol parent and the retained word that replaces it,:28180)
_CHAIN_WORDS = (("methanolate", "methoxide"), ("ethan-1-olate", "ethoxide"), ("propan-1-olate", "propoxide"),
                ("butan-1-olate", "butoxide"))
_HYDROXY_CLASS = Chem.MolFromSmarts("[$([O,S,Se,Te;X2;H1][#6]),$([O,S,Se,Te;X1;-1][#6])]")


def _centre(mol: Chem.Mol) -> Optional[Chem.Atom]:
    """The carbon of the one O- of a mono-anion that holds no other hydroxy-class group, or None."""
    # a nitro group's N+ and O- are no anionic centre of the compound: '2-nitrophenoxide')
    if Chem.GetFormalCharge(mol) != -1 or any(
            a.GetFormalCharge() > 0 and not (a.GetSymbol() == "N" and any(
                n.GetSymbol() == "O" and n.GetFormalCharge() == -1 for n in a.GetNeighbors()))
            for a in mol.GetAtoms()):
        return None
    oxygens = [a for a in mol.GetAtoms() if a.GetSymbol() == "O" and a.GetFormalCharge() == -1
               and all(n.GetSymbol() == "C" for n in a.GetNeighbors())]
    if len(oxygens) != 1 or oxygens[0].GetDegree() != 1 or oxygens[0].GetIsotope():
        return None
    if len(mol.GetSubstructMatches(_HYDROXY_CLASS)) != 1:
        return None
    carbon = oxygens[0].GetNeighbors()[0]
    return carbon if carbon.GetSymbol() == "C" else None


def _benzene_carbon(mol: Chem.Mol, carbon: Chem.Atom) -> bool:
    ri = mol.GetRingInfo()
    if ri.NumAtomRings(carbon.GetIdx()) != 1:
        return False
    ring = next(r for r in ri.AtomRings() if carbon.GetIdx() in r)
    return len(ring) == 6 and all(
        mol.GetAtomWithIdx(i).GetSymbol() == "C" and mol.GetAtomWithIdx(i).GetIsAromatic()
        and ri.NumAtomRings(i) == 1 for i in ring)


def retained_alkoxide_form(mol: Optional[Chem.Mol], olate_name: str) -> Optional[str]:
    """The retained-word spelling of the 'olate' name ``olate_name`` of the anion ``mol``, or None when the parent
    word is not one of the retained alcohols or the structure does not fit it (see the module text)."""
    if mol is None or not olate_name:
        return None
    carbon = _centre(mol)
    if carbon is None:
        return None
    if olate_name.endswith("phenolate") and olate_name != "phenolate":
        if _benzene_carbon(mol, carbon):
            return olate_name[:-len("phenolate")] + "phenoxide"
        return None
    if carbon.IsInRing() or carbon.GetIsAromatic():
        return None
    for word, retained in _CHAIN_WORDS:
        head = olate_name[:-len(word)]
        if olate_name.endswith(word) and head and not head.endswith("cyclo"):
            return head + retained
    return None


def retained_alkoxide_or(mol: Optional[Chem.Mol], name: str) -> str:
    """``retained_alkoxide_form`` of ``name`` when it reads back to ``mol`` by its full InChIKey, else ``name``."""
    retained = retained_alkoxide_form(mol, name)
    if retained:
        from .charged_router import _full_inchikey_rt_ok
        if _full_inchikey_rt_ok(mol, retained):
            return retained
    return name
