"""W3-P15 azine handler — substitutive 'ylidene' derivatives of hydrazine
(P-68.3.1.2.3).

An azine has the general structure R2C=N-N=CR2 (BB P-68.3.1.2.3.1) or the
unsymmetrical R2C=N-N=CR'2 (P-68.3.1.2.3.2). Preferred IUPAC names are formed
SUBSTITUTIVELY as 'ylidene' derivatives of the parent hydride hydrazine, NOT by
functional-class nomenclature ('acetone azine'):

    (CH3)2C=N-N=C(CH3)2   -> di(propan-2-ylidene)hydrazine   (symmetric, P-...3.1)
    CH3CH2C(CH3)=N-N=C6H10 -> (butan-2-ylidene)(cyclohexylidene)hydrazine  (P-...3.2)

The two C=N carbons carry the ylidene fragments. This handler perceives the
acyclic C=N-N=C motif on the molecule graph (NOT via a global-table SMARTS — the
motif also appears ring-internally in kekulised diazines, which are NOT azines),
BFS-names each ylidene arm through the shared substituent pipeline, and combines
them: identical arms -> 'di(<X>ylidene)hydrazine'; distinct arms -> alphanumeric
'(<A>ylidene)(<B>ylidene)hydrazine'.

Fail-closed (returns None -> the molecule keeps its cascade fallback) off this
exact shape: any charge/radical, a ring-internal C=N-N=C, an N bearing anything
but its =C and the N-N bond, a non-organyl arm, or any atom the two arms + the
two nitrogens do not fully account for.

IUPAC cite: P-68.3.1.2.3.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional, Set, Tuple

from ..name_tree import NameTreeNode, NamingResult

logger = logging.getLogger(__name__)


def _find_azine_core(mol) -> Optional[Tuple[int, int, int, int]]:
    """Return ``(c1, n1, n2, c2)`` for the single acyclic C=N-N=C azine core, or
    None. ``n1``/``n2`` are the two nitrogens (N-N single bond); ``c1``/``c2`` the
    carbons double-bonded to them. Fail-closed for >1 core or any decorated N."""
    from rdkit import Chem

    cores: List[Tuple[int, int, int, int]] = []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        if a.GetSymbol() != "N" or b.GetSymbol() != "N":
            continue
        if a.IsInRing() or b.IsInRing():
            continue
        # Each N must be neutral, degree 2, and double-bonded to exactly one C.
        n1, n2 = a, b
        c_ends = []
        ok = True
        for n, other in ((n1, n2), (n2, n1)):
            if n.GetFormalCharge() != 0 or n.GetNumRadicalElectrons() != 0:
                ok = False
                break
            if n.GetDegree() != 2:
                ok = False
                break
            dbl_c = None
            for nb in n.GetNeighbors():
                if nb.GetIdx() == other.GetIdx():
                    continue
                bt = mol.GetBondBetweenAtoms(n.GetIdx(), nb.GetIdx()).GetBondType()
                if bt == Chem.BondType.DOUBLE and nb.GetSymbol() == "C":
                    dbl_c = nb.GetIdx()
                else:
                    ok = False
            if dbl_c is None:
                ok = False
                break
            c_ends.append(dbl_c)
        if ok and len(c_ends) == 2:
            cores.append((c_ends[0], n1.GetIdx(), n2.GetIdx(), c_ends[1]))
    if len(cores) != 1:
        return None
    return cores[0]


def _is_azine(features: Any) -> bool:
    """Fire when the molecule is a bare azine (R2C=N-N=CR2) with NO senior
    principal characteristic group. Pure read-only graph inspection."""
    if getattr(features, "principal_group", None) is not None:
        return False
    mol = getattr(features, "mol", None)
    if mol is None:
        return False
    from rdkit import Chem
    if len(Chem.GetMolFrags(mol)) != 1:
        return False
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() != 0:
            return False
    return _find_azine_core(mol) is not None


def _ylidene_arm(mol, c_idx: int, n_idx: int) -> Optional[Tuple[str, Set[int]]]:
    """BFS the ylidene fragment rooted at ``c_idx`` (never crossing the imino
    ``n_idx``) and name it via the substituent pipeline, which emits the
    P-29.2 '-ylidene' directly from the C=N bond order.
    Return ``(ylidene_name, fragment_atom_set)`` or None."""
    frag: Set[int] = {c_idx}
    stack = [c_idx]
    while stack:
        i = stack.pop()
        for nb in mol.GetAtomWithIdx(i).GetNeighbors():
            j = nb.GetIdx()
            if j != n_idx and j not in frag:
                frag.add(j)
                stack.append(j)
    # The carbon's only bond leaving the fragment must be the =N (double).
    c_atom = mol.GetAtomWithIdx(c_idx)
    ext = [nb.GetIdx() for nb in c_atom.GetNeighbors() if nb.GetIdx() not in frag]
    if ext != [n_idx]:
        return None
    from ..substituent_enumerator import name_ylidene_substituent
    ylidene = name_ylidene_substituent(mol, frag, c_idx)
    if ylidene is None:
        return None
    return ylidene, frag


def name_azine(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Name a bare azine as a substitutive 'ylidene' derivative of hydrazine."""
    m = features.mol
    core = _find_azine_core(m)
    if core is None:
        return None
    c1, n1, n2, c2 = core
    arm1 = _ylidene_arm(m, c1, n1)
    arm2 = _ylidene_arm(m, c2, n2)
    if arm1 is None or arm2 is None:
        return None
    (name1, frag1), (name2, frag2) = arm1, arm2
    # Every atom must be a ylidene-arm atom or one of the two core nitrogens.
    if frag1 | frag2 | {n1, n2} != {a.GetIdx() for a in m.GetAtoms()}:
        return None

    from ..naming_utils import is_complex_substituent

    def _enclose(nm: str) -> str:
        return f"({nm})" if is_complex_substituent(nm) else nm

    if name1 == name2:
        # Symmetric azine (P-68.3.1.2.3.1): 'di(<X>ylidene)hydrazine'. The BB
        # PIN uses the simple multiplier 'di' even for a locant-bearing ylidene
        # (BB 38592 'di(propan-2-ylidene)hydrazine', not 'bis(...)').
        name = f"di{_enclose(name1)}hydrazine"
    else:
        # Unsymmetric azine (P-68.3.1.2.3.2): alphanumeric citation, each
        # ylidene enclosed, no position locants (BB '(butan-2-ylidene)'
        # '(cyclohexylidene)hydrazine').
        from ..naming_utils import alpha_sort_key
        pair = sorted((name1, name2), key=alpha_sort_key)
        name = f"({pair[0]})({pair[1]})hydrazine"

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "HANDLER_COVERAGE: handler=azine coverage=NA accounted=NA/%d name=%s",
            m.GetNumHeavyAtoms(), name[:60],
        )

    from ..candidate_pool import get_current_pool
    from ..composer import _inject_stereo_if_missing

    pool = get_current_pool()
    pool.add(name, "azine", features)
    final_name = _inject_stereo_if_missing(features, pool.best().name)
    return NamingResult(
        name=final_name,
        tree=NameTreeNode(
            parent_stem="hydrazine", fragment_legacy=final_name,
            class_id="azine", iupac_section_cite="P-68.3.1.2.3",
        ),
        atom_to_locant_hint=None,
    )


__all__ = ["name_azine", "_is_azine"]
