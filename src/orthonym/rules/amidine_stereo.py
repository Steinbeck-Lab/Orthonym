"""The configuration of the C=N double bond of an acyclic amidine, cited in front of its name.

 (the Blue Book): a PIN cites each stereogenic unit, and the C=N of an amidine is one, though
the standard InChIKey does not encode it (its hydrogen is mobile): '(1Z)-N'-methylethanimidamide'. The descriptor
takes the locant of the amidine carbon, C-1 of the '-imidamide' chain, as '(1E)-N-methylethanimine' does.
"""
from rdkit import Chem

# An acyclic amidine carbon on an aromatic ring atom: C(=N)(-N) with the C=N acyclic.
_RING_AMIDINE_C = Chem.MolFromSmarts('[CX3;!R](=,:[NX2;!R])(-[NX3;!R])[a]')

def amidine_cn_descriptor(mol, c_idx: int) -> str:
    """'(1Z)-' or '(1E)-' when the C=N of the amidine carbon ``c_idx`` is configured and is the only stereogenic
    unit of ``mol``; '' otherwise (the name is then left as it was)."""
    from ..perception.stereo import assign_stereochemistry
    m = Chem.Mol(mol)
    assign_stereochemistry(m)
    if any(a.HasProp('_CIPCode') for a in m.GetAtoms()):
        return ''
    configured = [b for b in m.GetBonds() if b.GetBondTypeAsDouble() == 2.0 and b.HasProp('_CIPCode')]
    if len(configured) != 1:
        return ''
    b = configured[0]
    if c_idx not in (b.GetBeginAtomIdx(), b.GetEndAtomIdx()) or b.GetOtherAtom(m.GetAtomWithIdx(c_idx)).GetSymbol() != 'N':
        return ''
    code = b.GetProp('_CIPCode')
    return f"(1{code})-" if code in ('E', 'Z') else ''


def amidine_prefix_descriptor(mol) -> str:
    """'(Z)-' or '(E)-' for the configured C=N of a demoted ring amidine ('carbamimidoyl') when that bond is the only
    stereogenic unit of ``mol`` and the molecule has only that one ring amidine; '' otherwise. The prefix cites its own
    descriptor at its front (b) for a substituent group,:48277): '4-[(Z)-N'-methylcarbamimidoyl]benzoic
    acid'. The caller gives every carbamimidoyl prefix the same form, so with two amidines (or a configured oxime or
    imine instead) no descriptor is given."""
    from ..perception.stereo import assign_stereochemistry
    amidines = {t[0] for t in mol.GetSubstructMatches(_RING_AMIDINE_C)}
    if len(amidines) != 1:
        return ''
    m = Chem.Mol(mol)
    assign_stereochemistry(m)
    if any(a.HasProp('_CIPCode') for a in m.GetAtoms()):
        return ''
    configured = [b for b in m.GetBonds() if b.GetBondTypeAsDouble() == 2.0 and b.HasProp('_CIPCode')]
    if len(configured) != 1:
        return ''
    b = configured[0]
    c, n = b.GetBeginAtom(), b.GetEndAtom()
    if c.GetSymbol() != 'C':
        c, n = n, c
    if (c.GetSymbol(), n.GetSymbol()) != ('C', 'N') or c.GetIdx() not in amidines:
        return ''
    code = b.GetProp('_CIPCode')
    return f"({code})-" if code in ('E', 'Z') else ''
