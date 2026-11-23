"""The anion R2C=[N-] of an imine named as the azanide of its ylidene group: '(propan-2-ylidene)azanide'.

 "Anionic centers on other characteristic groups" (the Blue Book): "Anionic centers generated
formally by the removal of hydrons from atoms of characteristic groups other than those considered in,
, and are named on the basis of the corresponding anionic parent hydrides": 'acetylazanide
(PIN)' (:41079). The anion of an imine whose carbon holds groups that the one-carbon and the ketene iminide cannot
cite (an amidine, a carbon chain, a ring) is the azanide, NH2(-), with the group of the carbon as its '-ylidene'
prefix, a substituent group of two free valences. The preferred name of the anion of an imine is its
'iminide',:41047), so the name is recorded as not the PIN; it is read back to the full InChIKey by
the caller's gate before it is returned.

A chalcogen double bonded to the nitrogen (O=[N-]) is the '-ylidene' group of one atom ('oxoazanide', as
'hydroxyazanide (preselected name)' at:41081 names the group of one oxygen atom), and the cumulated X=C=[N-] the
group of one carbon atom that holds it ('(oxomethylidene)azanide', '(sulfanylidenemethylidene)azanide'): the
prefixes 'oxo', 'sulfanylidene', 'selanylidene', 'tellanylidene' of =X "Retained traditional prefixes": '=O oxo (preselected prefix)' at
:17965; "Chalcogen analogues of aldehydes", Table 6.4: 'sulfanylidene (preferred prefix)', 'selanylidene', 'tellanylidene' at:35033;
'oxomethylidene' at:24926).
"""
from rdkit import Chem

#: The '-ylidene' prefix of a chalcogen atom double bonded to its parent.
_CHALCOGEN_YLIDENE = {'O': 'oxo', 'S': 'sulfanylidene', 'Se': 'selanylidene', 'Te': 'tellanylidene'}


def emit_ylidene_azanide(mol, anion_idx: int) -> str:
    """'(R-ylidene)azanide' for R=[N-] with a group R that has an '-ylidene' prefix, else ''."""
    from ..assembly.naming_utils import apply_enclosing_marks
    from .group_prefixes import first_name_that_reads_back, group_prefix_candidates
    n = mol.GetAtomWithIdx(anion_idx)
    if (n.GetSymbol() != 'N' or n.GetFormalCharge() != -1 or n.GetTotalNumHs() or n.GetDegree() != 1
            or n.GetIsAromatic() or n.IsInRing()):
        return ''
    attach = n.GetNeighbors()[0]
    if mol.GetBondBetweenAtoms(anion_idx, attach.GetIdx()).GetBondType() != Chem.BondType.DOUBLE:
        return ''
    own = _one_or_two_atom_group(attach, anion_idx)
    if own and _naming_the_ionic_drawing_of_a_bonded_metal():
        return ''
    if own:
        # a simple prefix ('oxo', as 'hydroxyazanide') takes no enclosing marks, a compound one does
        return first_name_that_reads_back(mol, (f"{own if own in _CHALCOGEN_YLIDENE.values() else '(' + own + ')'}azanide",))
    if attach.GetSymbol() != 'C':
        return ''
    rw = Chem.RWMol(mol)
    rw.RemoveBond(anion_idx, attach.GetIdx())
    group = next((list(c) for c in Chem.GetMolFrags(rw.GetMol(), asMols=False, sanitizeFrags=False)
                  if attach.GetIdx() in c), None)
    if not group or anion_idx in group:
        return ''
    prefixes = group_prefix_candidates(mol, group, attach.GetIdx(), 'ylidene', [anion_idx])
    return first_name_that_reads_back(mol, (f"{apply_enclosing_marks(p, -1)}azanide" for p in prefixes))


def _one_or_two_atom_group(attach, anion_idx: int) -> str:
    """The '-ylidene' prefix of the group =X (a chalcogen atom, 'oxo') or =C=X (one carbon atom that holds only the
    chalcogen, 'oxomethylidene'), else ''."""
    if attach.GetFormalCharge() or attach.GetNumRadicalElectrons() or attach.GetIsotope():
        return ''
    if attach.GetSymbol() in _CHALCOGEN_YLIDENE:
        return _CHALCOGEN_YLIDENE[attach.GetSymbol()] if attach.GetDegree() == 1 and not attach.GetTotalNumHs() else ''
    if attach.GetSymbol() != 'C' or attach.GetDegree() != 2 or attach.GetTotalNumHs():
        return ''
    far = next(a for a in attach.GetNeighbors() if a.GetIdx() != anion_idx)
    bond = attach.GetOwningMol().GetBondBetweenAtoms(attach.GetIdx(), far.GetIdx())
    if (far.GetSymbol() in _CHALCOGEN_YLIDENE and far.GetDegree() == 1 and not far.GetTotalNumHs()
            and not far.GetFormalCharge() and not far.GetNumRadicalElectrons() and not far.GetIsotope()
            and bond.GetBondType() == Chem.BondType.DOUBLE):
        return _CHALCOGEN_YLIDENE[far.GetSymbol()] + 'methylidene'
    return ''


def _naming_the_ionic_drawing_of_a_bonded_metal() -> bool:
    """True while the metal offer names another drawing of the input, its components included
    (``assembly.metal_offer.INSIDE_METAL_REWRITE``): 'S=C=N[Zn]N=C=S' keeps its name as drawn ('diisothiocyanatozinc'),
    which the offer would give up for the equally labelled name of the ions it splits off. An anion drawn as an ion
    ('C(=[N-])=S.[K+]') is named in the run of the drawing as written."""
    from ..assembly.metal_offer import INSIDE_METAL_REWRITE
    return INSIDE_METAL_REWRITE.get()
