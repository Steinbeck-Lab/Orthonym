"""The substituent prefix of a hydrazinyl unit and of an imine joined through nitrogen.

A fragment rooted at one of these nitrogen atoms used to be named as a free molecule
('methylhydrazine', 'methanimine') and a prefix cut from that name by the ring rule of
(``-ine`` to ``-inyl``, ``substituent_naming.parent_to_prefix``). The rule is for ring
hydrides, and both words are wrong:

* 'methylhydrazinyl' does not say which nitrogen the methyl is on. "Citation of
  locants" (the Blue Book): "if any locants are essential for defining the structure of the
  parent structure or of a unit of structure as defined by its appropriate enclosing marks, then
  all locants must be cited"; "Omission of locants" (:2953) leaves them out only where
  "no isomer can be generated". A methyl can stand on either nitrogen, so the prefix is
  '2-methylhydrazin-1-yl' ('4-(1-acetyl-2-ethylhydrazin-1-yl)benzoic acid (PIN)',
  "Substituent groups derived from hydrazides",:34042; '3-(2-carbamoylhydrazin-1-yl)propanoic
  acid (PIN)', "Semicarbazides",:38644).
* 'methaniminyl' is a radical. (:40620), Table 7.2 "Suffixes for Radicals of Amines,
  Imines and Amides" (:40622): '-iminyl' is the suffix of =N•. R-CH=N- joined through nitrogen
  is the amino prefix with its ylidene, '4-{[(4-chlorophenyl)methylidene]amino}aniline (PIN)'
   "Substitutive names for imines",:26529), '4-[(hydrazinylmethylidene)amino]benzoic
  acid (PIN)' "Prefixes" of the amidrazones,:34513), '4-[(diaminomethylidene)
  amino]butanoic acid (PIN)',:34282). R-C(=NH)- joined through carbon is an
  acyl group, 'methanimidoyl (preferred prefix)',:26540,:30468); it is not this
  unit.

The group is read by ``hetero_group_prefixes.group_shape`` and joined by ``compose_group`` from
the names the fragment namer gives its parts. The three steps (``read_n_group``,
``principal_group_in``, ``composed_group``) are shared with the diazenyl prefix
(``substituent_naming._diazenyl_prefix``). Left to the other routes, which name them as before: a hydrazinyl unit that
holds an ylidene ('ethylidenehydrazinyl',,:2959) or no substituent, a chain of three
nitrogen atoms (a polyazane, 'tetraazan-1-yl',,:39015), an N=C bond
with a defined geometry, and an ylidene carbon that carries a nitrogen group other than the two
amino groups of a guanidine (``_plain_ylidene_carbon``).
"""
from __future__ import annotations

from typing import Optional

from rdkit import Chem


def _plain_ylidene_carbon(mol, part, bond) -> bool:
    """True when the carbon of the ylidene part has single bonds only (no '-N=C=O', no
    ketenimine: those have their own prefixes) and carries carbon, an ether or thioether
    oxygen or sulfur, or the two amino groups of a guanidine ('(diaminomethylidene)amino
    (preferred prefix)',, the Blue Book). The fragment namer's names of the
    ylidene of other nitrogen groups are not all names of the group ('carbamoylidene' for
    =C(NH2)NH-NO2), so those units are left to the routes that name them."""
    carbon = mol.GetAtomWithIdx(part.attach)
    if any(b.GetBondType() != Chem.BondType.SINGLE
           for b in carbon.GetBonds() if b.GetIdx() != bond.GetIdx()):
        return False
    hetero = [nb for nb in carbon.GetNeighbors()
              if nb.GetIdx() != part.host and nb.GetSymbol() != 'C']
    amino = [nb for nb in hetero if nb.GetSymbol() == 'N' and nb.GetDegree() == 1
             and nb.GetFormalCharge() == 0 and nb.GetTotalNumHs() == 2]
    if len(amino) == 2 and len(hetero) == 2:
        return True
    return all(nb.GetSymbol() in ('O', 'S') and nb.GetDegree() == 2 and nb.GetFormalCharge() == 0
               and any(x.GetSymbol() == 'C' for x in nb.GetNeighbors()
                       if x.GetIdx() != carbon.GetIdx())
               for nb in hetero)


def read_n_group(mol, sub_atoms, attach_idx, unstereo_iminyl: bool = False):
    """``(group, parent, shape)`` of the fragment ``sub_atoms`` rooted at the neutral acyclic
    nitrogen ``attach_idx`` and joined by one single bond to the one atom ``parent`` outside it
    (``hetero_group_prefixes.group_shape`` reads the shape), or ``None``. The set-up the diazenyl
    prefix (``substituent_naming._diazenyl_prefix``) and ``n_unit_prefix`` share."""
    root = mol.GetAtomWithIdx(attach_idx)
    if root.GetSymbol() != 'N' or root.GetFormalCharge() != 0 or root.IsInRing():
        return None
    group = frozenset(sub_atoms)
    parents = [nb.GetIdx() for nb in root.GetNeighbors()
               if nb.GetIdx() not in group and nb.GetAtomicNum() > 1]
    if (len(parents) != 1
            or mol.GetBondBetweenAtoms(attach_idx, parents[0]).GetBondType() != Chem.BondType.SINGLE):
        return None
    from .hetero_group_prefixes import group_shape
    shape = group_shape(mol, attach_idx, group, parents[0], unstereo_iminyl=unstereo_iminyl)
    return None if shape is None else (group, parents[0], shape)


def principal_group_in(mol, group) -> bool:
    """True when a principal characteristic group of the molecule has an atom in ``group`` (an
    amidine, a hydrazide or an imine that is the senior class is named as the parent or by its
    own producer, not as a prefix), and when the perception cannot tell (fail closed)."""
    try:
        from ..perception.functional_groups import detect_functional_groups
        from ..rules.seniority import get_principal_group
        principal, matches = get_principal_group(mol, detect_functional_groups(mol))
    except Exception:  # noqa: BLE001 -- fail closed
        return True
    return bool(principal) and any(set(m) & group for m in (matches or ()))


def composed_group(mol, shape, attach_idx, parent_chain) -> Optional[str]:
    """The prefix ``hetero_group_prefixes.compose_group`` joins from the names the fragment namer
    gives the parts of ``shape``; ``None`` when a part has no name, which fails the whole prefix
    closed."""
    from .hetero_group_prefixes import compose_group
    from .substituent_naming import name_substituent_fragment
    outer = list(parent_chain or []) + [attach_idx]
    names = []
    for part in shape.parts:
        nm = name_substituent_fragment(mol, sorted(part.atoms), part.attach, outer + [part.host])
        if not nm or nm == 'substituent':
            return None
        names.append(nm)
    return compose_group(shape, names) or None


def n_unit_prefix(mol, sub_atoms, attach_idx, parent_chain) -> Optional[str]:
    """The prefix of the fragment ``sub_atoms`` rooted at nitrogen ``attach_idx`` and joined to a
    carbon by a single bond, when it is a substituted hydrazinyl unit ('2-methylhydrazin-1-yl')
    or an imine ('(methylideneamino)'); ``None`` for any other fragment, for a shape the module
    docstring leaves to the other routes, and when a part has no name, which fails the whole
    prefix closed."""
    read = read_n_group(mol, sub_atoms, attach_idx, unstereo_iminyl=True)
    if read is None or mol.GetAtomWithIdx(read[1]).GetSymbol() != 'C':
        return None
    group, _parent, shape = read
    if shape.kind == 'hydrazinyl':
        if not shape.parts or any(part.order == 2 for part in shape.parts):
            return None
        # a chain of three nitrogen atoms is a polyazane, 'triazan-1-yl', 'tetraazan-1-yl'
        #, the Blue Book), which precedes a hydrazine in class 21:18197);
        # a nitro or nitroso group is no chain, '2-nitrohydrazin-1-yl (preselected prefix)' (:36437)
        if any(x.GetSymbol() == 'N' and x.GetIdx() != part.host
               for part in shape.parts if mol.GetAtomWithIdx(part.attach).GetSymbol() == 'N'
               for x in mol.GetAtomWithIdx(part.attach).GetNeighbors()):
            return None
    elif shape.kind == 'iminyl':
        part = shape.parts[0]
        bond = mol.GetBondBetweenAtoms(part.host, part.attach)
        if bond.GetStereo() != Chem.BondStereo.STEREONONE or not _plain_ylidene_carbon(mol, part, bond):
            return None
    else:
        return None
    # the guard of ``_diazenyl_prefix``: the principal group is not this unit's
    if principal_group_in(mol, group):
        return None
    return composed_group(mol, shape, attach_idx, parent_chain)
