"""The zwitterion of a phosphoric diester: ``R-O-P(=O)(O-)-O-R'`` with the cation in one of the groups.

 "Acid esters of organic acids" (the Blue Book): "Preferred IUPAC names of acid esters of
inorganic acids as discussed in are formed by the method of 'hydrogen salts'"; the anion of a diester is the
name of the diester without 'hydrogen' ('diethyl phosphate'), which ``phosphorus.name_phosphate_ester_anion`` builds for
a molecule whose only charge is the anion. "Anionic and cationic centers on different parent structures"
(:42462): the anion is the parent and the cationic part is cited as a prefix to it, here inside the name of the group that
holds it: '3-(icosanoyloxy)-2-hydroxypropyl 2-(trimethylazaniumyl)ethyl phosphate', the phosphatidylcholines and
sphingomyelins "Phosphatidylcholines",:55164).

The groups are the ester words of the functional class name, each named by the substituent namer with its cation in
place (``phosphorus._p_ester_owner_group``) and cited in alphanumerical order with the multipliers of
``phosphorus._assemble_p_owner_text``,. The label stays below the PIN: the book's PIN of a quaternary
ammonium prefix is '-aminiumyl',:42296; '(N,N-dimethylmethanaminiumyl)acetate (PIN)',:42473), which OPSIN reads
in no version, so the name with the 'azaniumyl' prefix is a general name and the substituent namer records it as such.
A decline returns ''; the atom coverage is checked here and the whole name is read back to the full InChIKey here too
(``charged_router._full_inchikey_rt_ok``: charges and protonation site), since the route that calls the builder
(``_name_ester_anion_zwitterion``) ships what it returns.
"""
from rdkit import Chem


def phosphodiester_zwitterion_name(mol, cation_idx: int, central_idx: int, ester_oxygens) -> str:
    """'<group> <group> phosphate' of a phosphoric diester whose one terminal oxygen is the anion and whose two ester
    groups are carbon groups, the cation in exactly one of them, or ''."""
    from .phosphorus import _assemble_p_owner_text, _p_ester_owner_group
    central = mol.GetAtomWithIdx(central_idx)
    cation = mol.GetAtomWithIdx(cation_idx)
    if central.GetSymbol() != 'P' or len(ester_oxygens) != 2:
        return ''
    if cation.GetSymbol() != 'N' or cation.GetTotalNumHs():
        return ''  # a quaternary ammonium group: the owner namer encloses the prefix of a protonated amine ('(azaniumyl)')
    accounted = {central_idx, *ester_oxygens}
    double_oxo = anions = 0
    for bond in central.GetBonds():
        other = bond.GetOtherAtom(central)
        if other.GetIdx() in ester_oxygens:
            continue
        if other.GetSymbol() != 'O' or other.GetDegree() != 1:
            return ''
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            double_oxo += 1
        elif bond.GetBondType() == Chem.BondType.SINGLE and other.GetFormalCharge() == -1 and not other.GetTotalNumHs():
            anions += 1
        else:
            return ''  # a free hydroxy group: the diester is no anion of its own
        accounted.add(other.GetIdx())
    if double_oxo != 1 or anions != 1:
        return ''
    tokens, with_cation = [], 0
    for oxygen in ester_oxygens:
        got = _p_ester_owner_group(mol, oxygen, central_idx)
        if not got:
            return ''
        token, atoms = got
        if not token or token == 'substituent' or not isinstance(token, str):
            return ''
        tokens.append(token)
        with_cation += cation_idx in atoms
        accounted |= atoms
    if with_cation != 1 or accounted != set(range(mol.GetNumAtoms())):
        return ''  # the cation is in neither group or in both, or an atom is in no group
    from .charged_router import _full_inchikey_rt_ok
    name = f"{_assemble_p_owner_text(tokens)} phosphate"
    return name if _full_inchikey_rt_ok(mol, name) else ''  # the name reads back to the full key, charges included
