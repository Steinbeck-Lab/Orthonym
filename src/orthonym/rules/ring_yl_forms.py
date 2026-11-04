"""Two forms of a ring-yl substituent prefix that the composers of ``rules.ring_substituents`` build.

*:func:`phenoxy_form`: "Retained names" (the Blue Book): 'phenoxy' is a retained contraction, "fully
  substitutable": '(4-chlorophenoxy)methyl', not '[(4-chlorophenyl)oxy]methyl'.
*:func:`free_valence_stand_in`: the numbering of an unsaturated von Baeyer ring-yl gives the free valence lower locants
  than the 'ene' endings, (c) "principal characteristic groups and free valences (suffixes)" before
  (e), the 'ene' and 'yne' endings): '4,6,6-trimethylbicyclo[3.1.1]hept-3-en-2-yl', not
  '2,6,6-trimethylbicyclo[3.1.1]hept-2-en-4-yl'.
"""
from typing import Optional, Set, Tuple

from rdkit import Chem


def phenoxy_form(ring_name: str) -> Optional[str]:
    """'phenoxy' for 'phenyl', the enclosed contraction for a substituted phenyl ('(4-chlorophenoxy)' for
    '4-chlorophenyl'), else None. A biphenylyl reaches the composers as '[1,1'-biphenyl]-4-yl'
    (``substituent_enumerator.alkoxy_prefix_from_substituent``), never as '...phenyl'."""
    if ring_name == 'phenyl':
        return 'phenoxy'
    if ring_name.endswith('phenyl'):
        from ..assembly.naming_utils import apply_enclosing_marks
        return apply_enclosing_marks(ring_name[:-2] + 'oxy', -1)
    return None


def free_valence_stand_in(core, attach: int) -> Tuple[object, Optional[Set[int]]]:
    """``(mol, substituent_atoms)`` for the von Baeyer analysis of the ring-only ``core`` whose free valence is on
    ``attach``: for a core with a ring double or triple bond, ``core`` with a carbon bonded at ``attach`` that stands
    for the parent, and every atom but it as the substituent group (``vonbaeyer_universal.analyze_cage_universal``), so
    the analysis ranks the free valence with the suffixes; else ``(core, None)``, numbered as before. A core with an
    aromatic ring is left as before too: the analysis chooses its 'ene' locants among its Kekule structures."""
    bond_types = {b.GetBondType() for b in core.GetBonds()}
    if (not bond_types & {Chem.BondType.DOUBLE, Chem.BondType.TRIPLE}
            or Chem.BondType.AROMATIC in bond_types):
        return core, None
    try:
        rw = Chem.RWMol(core)
        stand_in = rw.AddAtom(Chem.Atom(6))
        rw.AddBond(attach, stand_in, Chem.BondType.SINGLE)
        with_parent = rw.GetMol()
        Chem.SanitizeMol(with_parent)
    except Exception:  # noqa: BLE001 - no stand-in: numbered as before
        return core, None
    return with_parent, set(range(core.GetNumAtoms()))
