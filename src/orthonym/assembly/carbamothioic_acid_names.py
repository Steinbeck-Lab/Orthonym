"""The free N-substituted thio acids of carbamic acid, R2N-C(=S)-SH, R2N-C(=S)-OH, R2N-C(=O)-SH: 'diethylcarbamodithioic acid'.

 (the Blue Book): "Functional replacement in carbonic acid and carbamic acid names by -OO-, -S-, -Se-, and
-Te- is expressed by the infixes 'peroxo', 'thio', 'seleno', and 'telluro'", "Tautomeric groups in mixed chalcocarbonic acids
... are distinguished by prefixing italic element symbols, such as 'S' or 'O', respectively, to the term 'acid'":
'H2N-CS-OH carbamothioic O-acid (PIN)' (:30774), 'H2N-CO-SH carbamothioic S-acid (PIN)' (:30826); (:30756):
"Substitution of formic acid is not recommended for generation of these names", so '(diethylamino)methane(dithioic acid)' is
not the PIN. The N-substituents are cited in front of the acid word with no italic N, as for 'dimethylcarbamic acid (PIN)'
 :30762) and 'diethylcarbamothioic di(propan-2-yl)stibinous thioanhydride (PIN)':35996).

Perception reads these acids as 'dithioic_acid', 'thioic_O_acid' or 'thioic_S_acid' on a carbon that bears an amine nitrogen;
the carbamic acid handler (``handlers.carbamic_acid``) names them here when that acid is the principal group. The italic locant
is written at the PIN tier only (``rules.inorganic_acids._with_acid_locant``): OPSIN 2.9.0 reads no 'O-acid' word, and a wider
tier ships the readable word, which the spelling check ``validation.spelling.checks_carbamothioic`` labels below the PIN.
"""
from __future__ import annotations

from typing import Any, Optional

from rdkit import Chem

#: the perception keys of a thio acid, with the acid word of its carbamic analogue
THIO_ACID_STEMS = {'dithioic_acid': 'carbamodithioic acid', 'thioic_O_acid': 'carbamothioic acid',
                   'thioic_S_acid': 'carbamothioic acid'}
#: the nitrogen of a carbamic acid (perception's 'carbamic_acid' SMARTS): neutral, acyclic, not N-N or N-O
_CARBAMIC_N = Chem.MolFromSmarts('[NX3;+0;!R;!$([NX3][NX3]);!$([NX3]~[OX1]);!$([NX3][OX2])]')


def carbamothioic_unit(features: Any) -> Optional[tuple]:
    """``(stem, n_idx, c_idx, core atoms)`` when the principal group of ``features`` is the one thio acid of the molecule and
    its carbon bears a carbamic nitrogen, else None."""
    pg = getattr(features, 'principal_group', None)
    if pg not in THIO_ACID_STEMS:
        return None
    matches = (getattr(features, 'functional_groups', None) or {}).get(pg, [])
    if len(matches) != 1:
        return None
    mol = features.mol
    c_idx = matches[0][0]
    c = mol.GetAtomWithIdx(c_idx)
    if c.IsInRing() or c.GetDegree() != 3 or c.GetFormalCharge() or c.GetIsotope():
        return None
    n_atoms = [nb.GetIdx() for nb in c.GetNeighbors() if nb.GetIdx() not in matches[0]]
    allowed = {m[0] for m in mol.GetSubstructMatches(_CARBAMIC_N)}
    if len(n_atoms) != 1 or n_atoms[0] not in allowed:
        return None
    core = set(matches[0]) | {n_atoms[0]}
    # (the Blue Book-:18174): carboxylic acids and their chalcogen analogues ('7a') outrank a carbonic
    # acid ('7b'), and this module ranks no other acid against it: beside any other acid group it names nothing
    for key, found in (getattr(features, 'functional_groups', None) or {}).items():
        if 'acid' in key and any(not set(m) <= core for m in found):
            return None
    return THIO_ACID_STEMS[pg], n_atoms[0], c_idx, core


def carbamothioic_acid_name(features: Any) -> Optional[str]:
    """'diethylcarbamodithioic acid', 'methylcarbamothioic O-acid' (the locant at the PIN tier), else None."""
    unit = carbamothioic_unit(features)
    if unit is None:
        return None
    stem, n_idx, c_idx, core = unit
    from .composer import _carbamic_n_substituted_name, _named_n_substituents
    from ..rules.inorganic_acids import _with_acid_locant
    n_subs = _named_n_substituents(features.mol, n_idx, c_idx, core)
    if n_subs is None:
        return None
    name = _carbamic_n_substituted_name(n_subs, stem) if n_subs else stem
    return _with_acid_locant(features.mol, name)
