"""Acyl halides on a ring named from the acid they come from: '4-methylbenzoyl chloride'.

 'Acid halides' (the Blue Book): "Acyl halides are named by citing the name of the corresponding acyl
group followed by the class name(s) of the halide(s)"; (:30624) gives the acyl group of an acid named
with the suffix 'carboxylic acid' ('-carbonyl') and the '-ic acid' -> '-yl' acyl groups ('benzoyl').
'2-carbonocyanidoyl-5-methylbenzoyl chloride (PIN)' (:31543) is a substituted one. The ring acid halide producer
wrote only 'benzoyl chloride' or 'cyclohexanecarbonyl chloride' and so named no substituted ring: the substituents
fell to the general engine ('1-[chloro(oxo)methyl]-4-methylbenzene', no PIN). Here the acid with every acyl halide
group drawn as a carboxy group is named by the engine, and its suffix is changed to the acyl form
(``lipids._acid_to_acyl``, which keeps a non-PIN record of the acid name on the acyl). Acid halides rank below acids,
anhydrides and esters, so the input has no carboxy group of its own and every carboxy group of the acid is one
of its acyl halide groups; the multiplied suffix must count all of them, or the name is not built.
"""
import re
from typing import List, Optional, Tuple

from rdkit import Chem

#: the halide word of each halogen
_WORD = {"F": "fluoride", "Cl": "chloride", "Br": "bromide", "I": "iodide"}
#: the multiplied acid suffix of a name and its multiplier ('-1,4-dicarboxylic acid')
_MULTIPLIED_ACID = re.compile(r"-[\d,]+-(di|tri|tetra)(?:carboxylic|oic) acid$")
_ACID = re.compile(r"\bcarboxy\b|carboxylic acid|oic acid|ic acid$")


def acyl_halide_from_acid(mol, combined: List[Tuple[tuple, str]]) -> Optional[str]:
    """'4-methylbenzoyl chloride' for ``mol`` whose acyl halide groups ``combined`` (``((C, O, X), symbol)`` each) carry
    one halogen and sit on rings; None for any other case (mixed halogens, a group off a ring, a carboxy group of the
    input's own, an acid name whose suffix does not count every group)."""
    if not combined or len({sym for _m, sym in combined}) != 1 or len(combined) > 4:
        return None
    if any(mol.GetAtomWithIdx(m[2]).GetIsotope() for m, _s in combined):
        return None                                  # a labelled halogen: the acid would lose the label
    if mol.HasSubstructMatch(_CARBOXY):
        return None                                  # an acid is senior; another producer names it
    for match, _sym in combined:
        carbons = [nb for nb in mol.GetAtomWithIdx(match[0]).GetNeighbors()
                   if nb.GetIdx() not in match and nb.GetAtomicNum() == 6]
        if len(carbons) != 1 or not carbons[0].IsInRing():
            return None
    acid = _acid_smiles(mol, [m[2] for m, _s in combined])
    if acid is None:
        return None
    from .lipids import _acid_to_acyl, _name_fragment
    acid_name = _name_fragment(acid)
    if not acid_name or not acid_name.endswith(" acid"):
        return None
    found = _MULTIPLIED_ACID.search(acid_name)
    count = {"di": 2, "tri": 3, "tetra": 4}[found.group(1)] if found else 1
    if count != len(combined) or len(_ACID.findall(acid_name)) != 1:
        return None                                  # a 'carboxy' prefix or another acid word is a group not converted
    acyl = _acid_to_acyl(acid_name)
    if not acyl:
        return None
    from ..assembly.naming_utils import get_multiplier_prefix
    word = _WORD[combined[0][1]]
    return f"{acyl} {get_multiplier_prefix(count, word)}{word}"


_CARBOXY = Chem.MolFromSmarts("[CX3](=O)[OX2H1,OX1-]")


def _acid_smiles(mol, halogen_atoms: List[int]) -> Optional[str]:
    """SMILES of ``mol`` with each atom of ``halogen_atoms`` replaced by a hydroxy oxygen, or None."""
    rw = Chem.RWMol(mol)
    for idx in halogen_atoms:
        atom = rw.GetAtomWithIdx(idx)
        atom.SetAtomicNum(8)
        atom.SetFormalCharge(0)
        atom.SetNoImplicit(False)
        atom.SetNumExplicitHs(0)
    try:
        Chem.SanitizeMol(rw)
    except Exception:
        return None
    return Chem.MolToSmiles(rw)
