"""The parent name of a hetero monocycle that holds a ring halogen ion, for the general engine.

A ring iodanuide ``[I-]`` or iodonium ``[I+]`` centre (``lambda_convention.ring_halogen_ion``) is named from the
lambda hydride whose ring atom has one hydrogen more than the ion has bonds: the anion is that hydride less a
hydron ('ide', method (2)), the cation that hydride less a hydride ion ('ylium', "Application
of the lambda-convention with the suffix 'ylium'", the Blue Book). ``ring_ion_parent_name`` is the name of
that hydride, which the caller's charge suffix then takes the hydron or the hydride ion off.
"""
import re
from typing import Dict, Optional

from rdkit import Chem

# the indicated hydrogen at the front of a hydride name, after its 'hydro' prefix when it has one
_INDICATED_HYDROGEN = re.compile(r"^(?P<hydro>(?:[0-9][0-9a-z,']*-[a-z]*hydro-)?)(?P<loc>[0-9]+[a-z]?)H-")


def ring_halogen_hydride_view(mol, ring_set):
    """``mol`` with each ring iodanuide and iodonium centre of ``ring_set`` given its hydrogen back and its charge
    taken off: the lambda hydride the ion is named from, so the ring name is the hydride's
    ('1H-1lambda3-iodepine') and the caller's ending takes the hydron or the hydride ion off again.
    ``mol`` itself when the ring holds no such centre; atom indices are the same in the copy."""
    from ..rules.lambda_convention import ring_halogen_ion
    ions = [i for i in sorted(ring_set) if ring_halogen_ion(mol, i) is not None]
    if not ions:
        return mol
    rw = Chem.RWMol(mol)
    for i in ions:
        atom = rw.GetAtomWithIdx(i)
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(1)
        atom.SetNoImplicit(True)
    view = rw.GetMol()
    view.UpdatePropertyCache(strict=False)
    return view


def ring_ion_parent_name(mol, ring_set, parent_ring, atom_to_locant: Dict[int, object]) -> Optional[str]:
    """The ring name ``name_heterocycle`` gives the hydride view of ``mol``'s ring, less the indicated hydrogen
    that stands on the iodonium centre itself: the 'ylium' takes that very hydrogen (with its hydride ion), so
    the book names the cation of '1H-1lambda3-benziodole' '1lambda3-benziodol-1-ylium (PIN)',
    the Blue Book), as '5lambda5-quinolizin-5-ylium' (:41700) carries none. An indicated hydrogen on
    another ring atom stays ('3H-1lambda4-thiophen-1-ylium',:41692)."""
    from ..rules.heterocycles import name_heterocycle
    name = name_heterocycle(ring_halogen_hydride_view(mol, ring_set), parent_ring)
    return _drop_ylium_indicated_hydrogen(name, mol, ring_set, atom_to_locant)


def _drop_ylium_indicated_hydrogen(name, mol, ring_set, atom_to_locant):
    """``name`` less its leading indicated hydrogen when that stands on a ring iodonium centre."""
    from ..rules.lambda_convention import ring_halogen_ion
    m = _INDICATED_HYDROGEN.match(name or '')
    if m:
        cation_locants = {str(atom_to_locant[i]) for i in ring_set
                          if i in atom_to_locant and ring_halogen_ion(mol, i) == 'cation'}
        if m.group('loc') in cation_locants:
            return m.group('hydro') + name[m.end():]
    return name


def fused_ring_ion_view(mol, ring_set):
    """The lambda hydride view of ``mol`` when its only charge is ONE ring iodanuide or iodonium centre
    of the fusable ring system ``ring_set`` (``ring_halogen_ion``) and nothing else is ionic, else ``None``."""
    from ..rules.lambda_convention import ring_halogen_ion
    ions = [a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge() != 0]
    if (len(ions) != 1 or ions[0] not in ring_set or ring_halogen_ion(mol, ions[0]) is None
            or any(a.GetNumRadicalElectrons() for a in mol.GetAtoms())):
        return None
    return ring_halogen_hydride_view(mol, ring_set)


def fused_ring_ion_parent(parent: str, mol, ring_set, atom_to_locant, append_charge_suffix):
    """``(name, charge_atom_ids)`` of the fusion ``parent`` of the hydride view with the charge ending of the ion
    (``append_charge_suffix`` is the general engine's), less the indicated hydrogen the 'ylium' takes
    (``ring_ion_parent_name``); ``None`` when the ending cannot be built."""
    parent = _drop_ylium_indicated_hydrogen(parent, mol, ring_set, atom_to_locant)
    return append_charge_suffix(parent, mol, atom_to_locant, has_fg_suffix=False)
