"""The prefix of a chalcogen-attached Group-14 group: ``(trimethylsilyl)oxy``, ``(trimethylgermyl)sulfanyl``.

 "Systematic names" (the Blue Book): " Substituent prefix names for R'-O- groups
are formed by concatenation, i.e., by adding the prefix 'oxy' to the substituent prefix name for the group
R'." (its example 'H3Si-O silyloxy (preselected prefix) (not siloxy)',:27647), and (:27649) "Substituent
prefixes for the substituent groups R'S-, R'Se-, and R'Te-, are formed by substitution of the groups HS-,
'sulfanyl', HSe-, 'selanyl'; and HTe-, 'tellanyl'". The substituent group R' of a silyl ether is the silyl
group, named by ``substituent_naming._name_group14_substituent``.

The generic cascade of ``substituent_naming._name_substituent_fragment_uncached`` caps a fragment with
hydrogen, names that molecule and cuts a ``yl`` from it. For -O-Si(CH3)3 the capped molecule is the silanol
'hydroxytrimethylsilane' and the cut is 'hydroxytrimethylsilyl', which puts the free valence on silicon: a
different group. The OPSIN read-back refused every such name, so every molecule with a silyl ether on a chain
that carries a principal characteristic group (``OCCO[Si](C)(C)C``) had no preferred name.
"""
from typing import Optional

from rdkit import Chem

#: Returned by:func:`group14_chalcogen_prefix` when the fragment is not this shape, so the caller goes on.
UNHANDLED = object()

#: The prefix of the chalcogen atom that carries a Group-14 group; oxygen takes 'oxy' by
#: concatenation,.
_CHALCOGEN_SUFFIX = {'S': 'sulfanyl', 'Se': 'selanyl', 'Te': 'tellanyl'}


#: What the Si or Ge centre of a prefix may hold besides carbon: a halogen is no characteristic group.
_HALOGENS = frozenset({'F', 'Cl', 'Br', 'I'})


def _carbon_parent_is_senior(mol, members, link, centre) -> bool:
    """True when the carbon side of the molecule is the parent and the Si or Ge group a prefix.

     "SENIORITY ORDER FOR PARENT STRUCTURES" (the Blue Book): "The senior parent structure has the
    maximum number of substituents corresponding to the principal characteristic group (suffix)";
    (:18917): without one, the senior parent has the senior atom, "N > P > As > Sb > Bi > Si > Ge > Sn > Pb > B > Al >
    Ga > In > Tl > O > S > Se > Te > C", so a molecule with no principal group anywhere is named on the silane or
    germane ('tert-butyldi(methyl)(oxiranylmethoxy)silane (PIN)', Si is senior to O; '[(trimethylsilyl)sulfanyl]
    benzene' is no PIN). The prefix is therefore cited only when the molecule has a principal group, every one
    stands outside the Si/Ge group (0 there against at least 1 on the carbon parent), and the centre holds nothing
    but the link, carbon and halogen: a hydroxy, amino or sulfanyl group on it would be a principal group of the
    silane or germane too, and its count would tie or beat the carbon parent."""
    from ..perception.functional_groups import detect_functional_groups
    from ..rules.seniority import get_principal_group
    name, matches = get_principal_group(mol, detect_functional_groups(mol))
    if not name or not matches or any(set(m) & members for m in matches):
        return False
    return all(n.GetIdx() == link.GetIdx() or n.GetSymbol() == 'C' or n.GetSymbol() in _HALOGENS
               for n in centre.GetNeighbors())


def group14_chalcogen_prefix(mol, sub_atoms, attach_idx):
    """The prefix of the fragment ``sub_atoms`` attached through a divalent chalcogen to Si or Ge, or UNHANDLED.

    :data:`UNHANDLED` when the fragment is not that shape, or is and this builder does not name it: the Si/Ge group
    is not a prefix by (:func:`_carbon_parent_is_senior`), a stereocentre lies in the fragment, or
    ``_name_group14_substituent`` (fail-closed: a siloxane, a ring) declines. The generic cascade then runs as it
    did before this builder, and its name still has to read back."""
    atom = mol.GetAtomWithIdx(attach_idx)
    symbol = atom.GetSymbol()
    if ((symbol != 'O' and symbol not in _CHALCOGEN_SUFFIX) or len(sub_atoms) < 2
            or atom.GetFormalCharge() != 0 or atom.GetTotalNumHs() != 0 or atom.IsInRing()):
        return UNHANDLED
    members = set(sub_atoms)
    inner = [n for n in atom.GetNeighbors() if n.GetIdx() in members]
    if len(inner) != 1 or inner[0].GetSymbol() not in ('Si', 'Ge'):
        return UNHANDLED
    if (any(mol.GetAtomWithIdx(i).GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for i in members)
            or not _carbon_parent_is_senior(mol, members, atom, inner[0])):
        return UNHANDLED
    from .substituent_naming import _name_group14_substituent
    group = _name_group14_substituent(mol, [i for i in sub_atoms if i != attach_idx], inner[0].GetIdx())
    if not group:
        return UNHANDLED
    if symbol == 'O':
        from .substituent_enumerator import alkoxy_prefix_from_substituent
        return alkoxy_prefix_from_substituent(group) or UNHANDLED
    from .naming_utils import enclose_if_compound
    return enclose_if_compound(group) + _CHALCOGEN_SUFFIX[symbol]


def ring_group14_chalcogen_substituent(mol, chalcogen_idx, ring_atoms):
    """The substituent record (``{'name', 'atoms', 'is_complex'}``) of a Group-14 group joined to a benzene
    ring through ``chalcogen_idx``, or None when the atom is not that shape or the group cannot be named."""
    from ..rules.benzene import _bfs_substituent_atoms
    members = _bfs_substituent_atoms(mol, chalcogen_idx, ring_atoms)
    prefix = group14_chalcogen_prefix(mol, members, chalcogen_idx)
    if prefix is UNHANDLED or not prefix:
        return None
    from .naming_utils import apply_enclosing_marks
    return {'name': apply_enclosing_marks(prefix, -1), 'atoms': members, 'is_complex': True}
