"""Prefix forms and principal-group rules of the benzene-ring writer (``rules.benzene``) that do not
come from the groups of the ring alone: they depend on the molecule's principal group or are
spelled by a retained prefix of the Blue Book."""
from typing import Optional


def hydroxy_is_principal(principal_group: Optional[str]) -> bool:
    """True when the molecule's principal group is a hydroxy or thiol group: a ketone or an
    ester of the ring cannot be the principal group of this naming, so it is a prefix
    ('4-acetylphenol', '4-(methoxycarbonyl)phenol'). That is the case when the group is
    the centre of an anion, which 'SENIORITY ORDER FOR CLASSES' (the Blue Book,
    Table 4.1) ranks above acids and every class below them; in a neutral molecule the
    ketone or ester would have been the principal group."""
    if not principal_group:
        return False
    from .seniority import get_suffix
    return get_suffix(principal_group, is_ring=True) in ('ol', 'thiol')


def pnictogen_oxoacid_prefix(mol, sub_atoms, start_idx: int) -> Optional[str]:
    """'arsono' or 'stibono' for a carbon-free ``-As(=O)(OH)2`` or ``-Sb(=O)(OH)2`` substituent.

     Retained names for substituent groups derived from mononuclear noncarbon
    oxoacids (the Blue Book): '–As(O)(OH)2 arsono (preselected prefix)' and '–Sb(O)(OH)2
    stibono (preselected prefix)' (:36030-36031), as '–P(O)(OH)2 phosphono (preselected prefix)'
    (:36029); the acids themselves, arsonic acid HAs(O)(OH)2 and stibonic acid HSb(O)(OH)2, are
    preselected names (:35413,:35435). ``seniority.FG_PREFIXES`` holds the prefixes. ``None``
    for any other shape (an organyl or a hydrogen on the central atom, an ester oxygen, a charge)."""
    centre = mol.GetAtomWithIdx(start_idx)
    prefix = {'As': 'arsono', 'Sb': 'stibono'}.get(centre.GetSymbol())
    if prefix is None or centre.GetFormalCharge() != 0 or centre.GetTotalNumHs() != 0 or len(sub_atoms) != 4:
        return None
    oxygens = [mol.GetAtomWithIdx(a) for a in sub_atoms if a != start_idx]
    order = {o.GetIdx(): mol.GetBondBetweenAtoms(start_idx, o.GetIdx()).GetBondTypeAsDouble() for o in oxygens}
    doubly = [o for o in oxygens if order[o.GetIdx()] == 2.0]
    singly = [o for o in oxygens if order[o.GetIdx()] == 1.0]
    if len(doubly) != 1 or len(singly) != 2 or doubly[0].GetSymbol() != 'O':
        return None
    if any(o.GetSymbol() != 'O' or o.GetFormalCharge() != 0 or o.GetTotalNumHs() != 1 for o in singly):
        return None
    return prefix
