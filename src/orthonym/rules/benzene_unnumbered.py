"""The groups of a benzene whose retained names number nothing of them (IUPAC 2013,.

 'Location of nuclides on positions not normally denoted by locants' (the Blue Book),
: "When the nuclide is located at a position in a retained name that is not numbered a systematic
name that identifies separately the relevant atom is used for the IUPAC preferred name." Its examples
(:44253,:44255): 'benzene(13C)carbonitrile (PIN)', 'benzene(13C)carboxylic acid (PIN)', not
'(cyano-13C)benzonitrile' and '(carboxyl-13C)benzoic acid'.

The retained 'benzonitrile', 'benzoic acid', 'benzamide', 'benzohydrazide' and 'benzaldehyde' number the ring
and no position of the carbon of their group. The isotope decorator (``rules.isotopes``) names the
isotope-stripped molecule inside ``assembly.locant_omission.systematic_names_for_unnumbered_labels`` once no
placement on the retained spelling has read back, and ``rules.benzene._assemble_benzene_with_suffix`` then asks
:func:`label_on_unnumbered_group` whether a nuclide sits on that carbon (or, for an aldehyde, its oxygen or its
hydrogen) and builds the systematic name (:func:`systematic_group_name`) instead of the retained one.
"""
from typing import Dict, List

#: The suffix groups whose carbon the retained names above do not number.
UNNUMBERED_GROUP_SUFFIXES = frozenset(
    {'carbonitrile', 'carboxylic acid', 'carboxamide', 'carbohydrazide', 'carbaldehyde'})


def label_on_unnumbered_group(mol, oriented_ring, suffix, locants) -> bool:
    """True inside ``systematic_names_for_unnumbered_labels`` when a nuclide of the labelled molecule being named
    sits on the carbon of the ONE group ``suffix`` (one of ``UNNUMBERED_GROUP_SUFFIXES``) that stands at ring
    locant ``locants[0]`` of ``oriented_ring``, on the oxygen of its formyl, or on a hydrogen of either.
    Outside the scope the retained name stands."""
    if suffix not in UNNUMBERED_GROUP_SUFFIXES or len(locants) != 1:
        return False
    from ..assembly.locant_omission import label_on_unnumbered_position, unnumbered_label_scope_active
    if not unnumbered_label_scope_active():
        return False
    ring = set(oriented_ring)
    ring_atom = mol.GetAtomWithIdx(oriented_ring[locants[0] - 1])
    group = [nb for nb in ring_atom.GetNeighbors() if nb.GetIdx() not in ring and nb.GetSymbol() == 'C']
    atoms = [nb.GetIdx() for nb in group]
    if suffix == 'carbaldehyde':
        atoms += [o.GetIdx() for c in group for o in c.GetNeighbors()
                  if o.GetSymbol() == 'O' and o.GetDegree() == 1]
    return label_on_unnumbered_position(mol, atoms)


#: The retained names of a whole benzene with one such group (``data.retained_names``): 'benzonitrile', 'benzoic acid',
#: 'benzamide', 'benzohydrazide', 'benzaldehyde'
RETAINED_BENZENE_NAMES = frozenset({'benzonitrile', 'benzoic acid', 'benzamide', 'benzohydrazide', 'benzaldehyde'})


def retained_benzene_group_is_labelled(mol, name: str) -> bool:
    """True inside ``systematic_names_for_unnumbered_labels`` when ``name`` is one of ``RETAINED_BENZENE_NAMES`` and a nuclide
    of the labelled molecule being named sits on the carbon of the group of ``mol`` (or on the oxygen of its formyl, or on
    a hydrogen of either): the table of retained names then does not name it, and the benzene handler spells the systematic
    name (:func:`systematic_group_name`), as for a benzene with other substituents."""
    if name not in RETAINED_BENZENE_NAMES:
        return False
    from ..assembly.locant_omission import label_on_unnumbered_position, unnumbered_label_scope_active
    if not unnumbered_label_scope_active():
        return False
    group = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == 'C' and not a.IsInRing()
             and any(nb.GetIsAromatic() for nb in a.GetNeighbors())]
    group += [o.GetIdx() for c in group for o in mol.GetAtomWithIdx(c).GetNeighbors()
              if o.GetSymbol() == 'O' and o.GetDegree() == 1]
    return label_on_unnumbered_position(mol, group)


def systematic_group_name(prefix_groups: Dict[str, List[int]], group_locant: int, suffix: str) -> str:
    """The systematic name of a benzene with one group of ``UNNUMBERED_GROUP_SUFFIXES`` and the substituents
    ``prefix_groups``: its locant is cited beside any other substituent, as for the sulfonic acids
    ('4-aminobenzene-1-sulfonic acid',, the Blue Book: deny by default)."""
    from .benzene import _build_prefix_string_with_locants, _renumber_relative_to
    if not prefix_groups:
        return f"benzene{suffix}"
    renumbered = _renumber_relative_to(prefix_groups, group_locant)
    return f"{_build_prefix_string_with_locants(renumbered, mono_needs_locant=True)}benzene-1-{suffix}"
