"""A zwitterion drawing named as the neutral compound.

 "Ionic centers in the same parent structure" (the Blue Book, sentence:42419): "For nomenclature
purposes, zwitterionic compounds having the ionic centers in the same parent structure are not considered as
neutral compounds." "Anionic and cationic centers on different parent structures" (:42462) names such a
compound by citing the cationic part as a prefix to the anionic parent ('(N,N-dimethylmethanaminiumyl)acetate
(PIN)'). A name of the neutral compound for a structure drawn as a zwitterion ('2-aminoethanoic acid' for
``[NH3+]CC([O-])=O``) is the name of another drawing of the compound, and the standard InChIKey, which
normalises the hydron positions, cannot tell the two drawings apart: the round trip passes. The name is a valid
systematic name of the compound; the drawing as written is the structure to name, and its charges are not in
the name.

:func:`named_as_neutral` decides from the structure and the name alone: a single connected component with true
zwitterionic character (``perception.ions._has_true_zwitterion_character``: separated positive and negative
sites that are no nitro, azide, N-oxide or similar group) whose name cites no cationic centre. Every name of the
drawing cites its cationic centre in a morpheme that contains 'ium' ('azaniumyl', 'pyridin-1-ium', 'ammonium'),
'onia' ('azonia...') or 'onio' (the 1979 prefixes 'ammonio', 'sulfonio').
"""
from __future__ import annotations

import re

from rdkit import Chem

#: the morphemes that cite a cationic centre in a name
_CATION_MORPHEME = re.compile(r"ium|onia|onio")


def named_as_neutral(smiles: str, name: str) -> bool:
    """True when ``smiles`` is drawn as a zwitterion and ``name`` cites no cationic centre."""
    if not smiles or not name:
        return False
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        return False
    from ..perception.ions import _has_true_zwitterion_character
    return _has_true_zwitterion_character(mol) and _CATION_MORPHEME.search(name) is None
