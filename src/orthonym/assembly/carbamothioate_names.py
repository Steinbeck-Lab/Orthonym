"""The esters of carbamothioic and carbamodithioic acid, R-O-C(=S)-NR'2, R-S-C(=O)-NR'2 and R-S-C(=S)-NR'2 (lane/l17 item 3).

 (the Blue Book): "Functional replacement in carbonic acid and carbamic acid names by -OO-, -S-, -Se-, and
-Te- is expressed by the infixes 'peroxo', 'thio', 'seleno', and 'telluro'", "Contrary to Rules through,
functional replacement nomenclature is applied to the retained name 'carbamic acid'": 'carbamothioic O-acid (PIN)' for
H2N-CS-OH (:30774) and 'carbamothioic S-acid (PIN)' for H2N-CO-SH (:30826); "Substitution of formic acid is not recommended
for generation of these names",:30756), so '(methylamino)methanethioate' is no name of these esters.
 "Esters of carbonic acid, cyanic acid, and polycarbonic acids modified by functional replacement" (:32023):
"Names of acids modified by functional replacement are used to generate preferred IUPAC names of corresponding esters. Element
symbols O, S, etc. and locants are used to designate the location of organyl groups" ('O,S-dimethyl carbonodithioate (PIN)').
So the ester is 'O-R carbamothioate' (the C=S one), 'S-R carbamothioate' (the C=O one) and, with no tautomer to tell, 'R
carbamodithioate'; the substituents of the nitrogen are cited in front of the acid word with no italic N, as for 'carbamate'
,:30766: '2-hydroxypropyl (2-aminoethyl)carbamate (PIN)'). Carbamic acid is a carbonic acid (class 7b,, so
the ester ranks as the carbamate does (``handlers.carbamate``).

The carbamate builder of ``composer._name_carbamate`` names the O-esters of carbamic acid; this one names the sulfur analogues
with the same R-group and N-substituent walkers. No Se/Te form is built: the unit is not recognised, so the molecule goes to the
producers that follow, as before.
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

from rdkit import Chem

#: (kind, SMARTS) with the match order (N, C, doubly bonded chalcogen, ester chalcogen, R's first atom). Neither the carbon, the
#: ester chalcogen nor the nitrogen is a ring atom (a ring N is a ring carbothioate, a cyclic unit a heterocycle:, and the
#: nitrogen bears no other heteroatom (a hydrazine, a hydroxylamine, an N-acylated one is another parent).
_UNIT_PATTERNS = (
    ('O', '[NX3;!R;!$(N[#7,#8])][CX3;!R](=[SX1])[OX2;!R][#6]'),
    ('S', '[NX3;!R;!$(N[#7,#8])][CX3;!R](=[OX1])[SX2;!R][#6]'),
    ('SS', '[NX3;!R;!$(N[#7,#8])][CX3;!R](=[SX1])[SX2;!R][#6]'),
)
_COMPILED = {}


def _pattern(kind: str, smarts: str):
    if kind not in _COMPILED:
        _COMPILED[kind] = Chem.MolFromSmarts(smarts)
    return _COMPILED[kind]


def thiocarbamate_units(mol) -> List[Tuple[int, int, str, int, int]]:
    """The thiocarbamate ester units of ``mol`` as ``(n_idx, c_idx, kind, x_idx, r_idx)``: ``kind`` 'O' (R-O-C(=S)-N), 'S'
    (R-S-C(=O)-N) or 'SS' (R-S-C(=S)-N), ``x_idx`` the ester chalcogen, ``r_idx`` the first atom of R. Neutral, unlabelled atoms
    only (a labelled or charged unit is another molecule's business)."""
    units = []
    if not any(a.GetAtomicNum() == 16 for a in mol.GetAtoms()):
        return units            # every unit holds a sulfur: the O-ester of an oxo acid is the 'carbamate' group of perception
    for kind, smarts in _UNIT_PATTERNS:
        for m in mol.GetSubstructMatches(_pattern(kind, smarts)):
            n_idx, c_idx, _, x_idx, r_idx = m
            if any(mol.GetAtomWithIdx(i).GetFormalCharge() or mol.GetAtomWithIdx(i).GetIsotope()
                   or mol.GetAtomWithIdx(i).GetNumRadicalElectrons() for i in m[:4]):
                continue
            units.append((n_idx, c_idx, kind, x_idx, r_idx))
    return units


def carbamothioate_name(features: Any, unit: Tuple[int, int, str, int, int]) -> str:
    """'O-R carbamothioate', 'S-R carbamothioate' or 'R carbamodithioate' (with the N-substituents in front of the acid
    word) for the thiocarbamate ester ``unit``; '' when R or an N-substituent cannot be named (the caller declines)."""
    mol = features.mol
    n_idx, c_idx, kind, x_idx, r_idx = unit
    from .composer import _carbamic_n_substituted_name, _name_r_group, _named_n_substituents
    from .naming_utils import enclose_if_compound
    core = {n_idx, c_idx, x_idx}
    core.update(nb.GetIdx() for nb in mol.GetAtomWithIdx(c_idx).GetNeighbors()
                if mol.GetBondBetweenAtoms(c_idx, nb.GetIdx()).GetBondType() == Chem.BondType.DOUBLE)
    r_name = _name_r_group(mol, r_idx, exclude_atoms=core)
    if not r_name:
        return ''
    n_subs = _named_n_substituents(mol, n_idx, c_idx, core)
    if n_subs is None:
        return ''
    stem = 'carbamodithioate' if kind == 'SS' else 'carbamothioate'
    acid_word = _carbamic_n_substituted_name(n_subs, stem) if n_subs else stem
    organyl = r_name if kind == 'SS' else f"{kind}-{enclose_if_compound(r_name)}"
    return f"{organyl} {acid_word}"
