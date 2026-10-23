"""The iminide of a one-carbon imine whose carbon holds two groups: 'methoxy(phenyl)methaniminide'.

 "Anions derived from amines and imines" (the Blue Book): the anion R2C=[N-] is named with the suffix
'iminide' on the imine, here the one-carbon parent 'methanimine' with the two groups of the carbon as prefixes. The
locant '1' is omitted on a substituted mononuclear parent hydride (a),:2893), the groups after the first in
parentheses, as in the Blue Book's 'cyclopropyl(phenyl)methanol (PIN)' (:7302). The charged route converts the 'imine'
suffix of the neutral name; when the neutral is an ester of an imidic acid (R-C(=NH)-OR', R-C(=NH)-SR': 'methyl
benzenecarboximidate', there is no 'imine' suffix to convert and the anion had no name. A carbon chain on the
carbon is part of the parent, not a group on methane, the principal chain): an unbranched alkyl group makes the
parent the alkane of one more carbon ('1-methoxyethan-1-iminide', every locant cited,,:2869), and any other
carbon chain is left unnamed. The name is read back to the full InChIKey by the caller's gate before it is returned; it
is recorded as not the PIN, the Blue Book giving no example of the anion of an imidic ester.
"""
from rdkit import Chem


def _is_carbon_chain_atom(atom) -> bool:
    """A carbon of an acyclic chain (not aromatic, not in a ring): the group it heads is part of the parent."""
    return atom.GetSymbol() == 'C' and not atom.IsInRing() and not atom.GetIsAromatic()


def _alkyl_length(mol, atoms, head: int) -> int:
    """The number of carbons of the unbranched, saturated, unsubstituted alkyl chain that ``atoms`` form from ``head``,
    0 for any other group."""
    members, path, previous, current = set(atoms), [], None, head
    while True:
        atom = mol.GetAtomWithIdx(current)
        if (atom.GetSymbol() != 'C' or atom.IsInRing() or atom.GetIsAromatic() or atom.GetFormalCharge()
                or atom.GetNumRadicalElectrons()):
            return 0
        path.append(current)
        onward = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() in members and n.GetIdx() != previous]
        if len(onward) > 1 or any(mol.GetBondBetweenAtoms(current, n).GetBondType() != Chem.BondType.SINGLE
                                  for n in onward):
            return 0
        if not onward:
            return len(path) if len(path) == len(members) else 0
        previous, current = current, onward[0]


def _chain_iminide(mol, carbon_idx: int, arms, chain_arms, components) -> str:
    """'1-X-alkan-1-iminide' for X-C(=[N-])-alkyl, the alkyl an unbranched, saturated, unsubstituted chain of n carbons
    and X the group of the other arm: the parent is the alkane of n + 1 carbons with the carbon of the iminide at
    1. '' for any other carbon chain, and for a carbon chain on both arms (a ketimine anion, named by the suffix seam)."""
    from ..assembly.naming_utils import enclose_if_compound
    from ..assembly.substituent_naming import name_substituent_fragment
    from ..data.chain_names import get_chain_name
    from ..metrics.provenance import record_non_pin_fragment
    from .charged_router import _cation_name_rt_ok
    if len(chain_arms) != 1:
        return ''
    other_nb = next(a for a in arms if a != chain_arms[0])
    length = _alkyl_length(mol, next(list(c) for c in components if chain_arms[0] in c), chain_arms[0])
    other_atoms = next(list(c) for c in components if other_nb in c)
    group = name_substituent_fragment(mol, other_atoms, other_nb, [carbon_idx]) if length else ''
    if not group:
        return ''
    name = f"1-{enclose_if_compound(group)}{get_chain_name(length + 1)[:-1]}-1-iminide"
    if not _cation_name_rt_ok(name, mol):
        return ''
    record_non_pin_fragment(name)
    return name


def emit_one_carbon_iminide(mol, anion_idx: int) -> str:
    """'X(Y)methaniminide' for R2C=[N-] with a carbon holding exactly two groups, else ''."""
    from ..assembly.naming_utils import alpha_sort_key, apply_enclosing_marks, enclose_if_compound, multiplied_component
    from ..assembly.substituent_naming import name_substituent_fragment
    from ..assembly.universal_substituent import name_universal_substituent_prefix
    from ..metrics.provenance import best_effort_ctx, record_non_pin_fragment
    from .charged_router import _cation_name_rt_ok
    n = mol.GetAtomWithIdx(anion_idx)
    if (n.GetSymbol() != 'N' or n.GetFormalCharge() != -1 or n.GetTotalNumHs() or n.GetDegree() != 1
            or n.GetIsAromatic() or n.IsInRing()):
        return ''
    carbon = n.GetNeighbors()[0]
    bond = mol.GetBondBetweenAtoms(anion_idx, carbon.GetIdx())
    arms = [a.GetIdx() for a in carbon.GetNeighbors() if a.GetIdx() != anion_idx]
    if (carbon.GetSymbol() != 'C' or carbon.GetFormalCharge() or carbon.IsInRing() or carbon.GetTotalNumHs()
            or bond.GetBondType() != Chem.BondType.DOUBLE or len(arms) != 2):
        return ''
    rw = Chem.RWMol(mol)
    for arm in arms:
        rw.RemoveBond(carbon.GetIdx(), arm)
    components = Chem.GetMolFrags(rw.GetMol(), asMols=False, sanitizeFrags=False)
    chain_arms = [arm for arm in arms if _is_carbon_chain_atom(mol.GetAtomWithIdx(arm))]
    if chain_arms:
        return _chain_iminide(mol, carbon.GetIdx(), arms, chain_arms, components)
    names = []
    for arm in arms:
        atoms = next((list(c) for c in components if arm in c), None)
        if not atoms or carbon.GetIdx() in atoms:
            return ''
        prefix = name_substituent_fragment(mol, atoms, arm, [carbon.GetIdx()])
        if not prefix and best_effort_ctx.get():
            prefix = name_universal_substituent_prefix(mol, atoms, arm)
        if not prefix:
            return ''
        names.append(prefix)
    names.sort(key=alpha_sort_key)
    if names[0] == names[1]:
        body = multiplied_component(2, names[0], enclose_if_compound(names[0]))
    else:
        body = enclose_if_compound(names[0]) + apply_enclosing_marks(names[1], -1)
    name = f"{body}methaniminide"
    if not _cation_name_rt_ok(name, mol):
        return ''
    record_non_pin_fragment(name)
    return name
