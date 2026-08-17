"""Acid-ester anion producer (IUPAC-2013 P-72.2.2.2.1.2).

The anion of an acid-ester of a P or S oxoacid, where the ester owner R is a
plain organyl: ``dodecyl phosphate``, ``dodecyl hydrogen phosphate``,
``diethyl phosphate``, ``dodecyl sulfate``. The protonation word is derived IN
PLACE from the surviving free ``-OH`` count; the ``[O-]`` carry the charge and
are not counted as hydrogens (D-04 — never neutralize-then-rename). Fail-closed
(``None``) off the clean single-centre ester-anion shape so a wrong molecule is
never emitted; the caller RT-gates any returned name.
"""
from typing import List, Optional

from rdkit import Chem

# The number of PROTONATED terminal acidic oxygens -> the sulfate word (D-04).
_SULFATE_WORD = {1: "hydrogen sulfate", 0: "sulfate"}


def name_sulfate_ester_anion(mol, sulfur_idx: int) -> Optional[str]:
    """Name the anion of a sulfate acid-ester ``R-O-S(=O)(=O)-[O-]`` -> ``{R}yl sulfate``.

    Requires: neutral non-ring S; exactly two ``=O``; exactly one ``-O-C`` ester
    owner; exactly one terminal acidic O with at least one ``[O-]``; NO S-C bond
    (that is a sulfonate — a different, already-handled class); the molecule's
    only charges are the terminal ``[O-]``; complete atom coverage. Otherwise
    ``None`` (honest fail).
    """
    from .phosphorus import _p_ester_owner_group
    if mol is None:
        return None
    s = mol.GetAtomWithIdx(sulfur_idx)
    if s.GetSymbol() != 'S' or s.GetFormalCharge() != 0 or s.IsInRing():
        return None

    accounted = {sulfur_idx}
    anion_oxygens: set = set()
    ester_oxygens: List[int] = []
    oh_count = 0
    anion_count = 0
    dbl_oxo = 0
    for b in s.GetBonds():
        nb = b.GetOtherAtom(s)
        bt = b.GetBondType()
        sym = nb.GetSymbol()
        if bt == Chem.BondType.DOUBLE and sym == 'O':
            dbl_oxo += 1
            accounted.add(nb.GetIdx())
            continue
        if bt != Chem.BondType.SINGLE:
            return None                         # S=C / S=S / thio -> defer
        if sym != 'O':
            return None                         # S-C (sulfonate) / S-N -> defer
        others = [x for x in nb.GetNeighbors() if x.GetIdx() != sulfur_idx]
        if not others and nb.GetFormalCharge() < 0:
            anion_count += 1
            anion_oxygens.add(nb.GetIdx())
            accounted.add(nb.GetIdx())
        elif not others and nb.GetTotalNumHs() >= 1 and nb.GetFormalCharge() == 0:
            oh_count += 1
            accounted.add(nb.GetIdx())
        elif len(others) == 1 and others[0].GetSymbol() == 'C' \
                and nb.GetFormalCharge() == 0:
            ester_oxygens.append(nb.GetIdx())
            accounted.add(nb.GetIdx())
        else:
            return None                         # S-O-S bridge / charged owner O -> defer

    if dbl_oxo != 2 or len(ester_oxygens) != 1 or anion_count < 1:
        return None
    # 0-wrong: ONLY the counted terminal [O-] may carry charge. A net-sum check
    # would pass a charge-separated zwitterion whose remote +/- cancel; scan
    # per-atom so any other charged centre fails closed before the owner namer.
    for a in mol.GetAtoms():
        if a.GetFormalCharge() != 0 and a.GetIdx() not in anion_oxygens:
            return None

    got = _p_ester_owner_group(mol, ester_oxygens[0], sulfur_idx)
    if got is None:
        return None
    owner, frag = got
    accounted |= frag
    if accounted != set(range(mol.GetNumAtoms())):
        return None
    word = _SULFATE_WORD.get(oh_count)
    if word is None:
        return None
    return f"{owner} {word}"


def name_acid_ester_anion(mol) -> Optional[str]:
    """Find the single qualifying P/S acid-ester centre and name its anion.

    Tries each non-ring P then each non-ring S; each builder's coverage audit
    guarantees a returned name accounts for the WHOLE molecule, so a
    poly-phosphate / multi-centre species fails closed (``None`` -> the caller
    falls through). ``None`` when nothing qualifies.
    """
    if mol is None:
        return None
    from .phosphorus import name_phosphate_ester_anion
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym == 'P' and not atom.IsInRing():
            nm = name_phosphate_ester_anion(mol, atom.GetIdx())
            if nm:
                return nm
        elif sym == 'S' and not atom.IsInRing():
            nm = name_sulfate_ester_anion(mol, atom.GetIdx())
            if nm:
                return nm
    return None
