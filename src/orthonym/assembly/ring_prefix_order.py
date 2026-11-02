"""Two decisions of the substituent cascade (``substituent_enumerator``) on a ring-bearing fragment.

* the order of the ring-prefix producers at the wider tiers (``ring_prefix_in_pin_order``);
* the ring '-ylidene' bond a chain host cites, which a ring prefix leaves out (``host_cited_ylidene_bonds``).
"""
from ..perception.molcache import bonds_of


def ring_prefix_in_pin_order(mol, frag_atoms_set, attach_idx, *, allow_mancude, pos_out):
    """The Tier 1.95 ring-system prefix. The wider tiers (``allow_mancude``) ask what the PIN tier asks first, then
    the fusion prefix of Tier 4.4, and only then their own von Baeyer cage namer: (the Blue Book) a
    fusion name, not von Baeyer, for a system of two or more rings of five or more members ('2H-benzo[7]annulen-2-yl',
    not 'bicyclo[5.4.0]undeca-...-9-yl'). ``pos_out`` is filled as by ``name_ring_system_substituent``."""
    from ..rules.ring_substituents import name_ring_system_substituent
    from .fused_ring_prefix import fused_ring_system_prefix
    name = None
    if allow_mancude:
        name = name_ring_system_substituent(
            mol, sorted(frag_atoms_set), attach_idx, allow_enumerator_fallback=False, pos_out=pos_out,
        ) or fused_ring_system_prefix(mol, frag_atoms_set, attach_idx)
    if not name:
        pos_out.clear()
        name = name_ring_system_substituent(
            mol, sorted(frag_atoms_set), attach_idx, allow_enumerator_fallback=False,
            allow_mancude=allow_mancude, pos_out=pos_out,
        )
    return name


def host_cited_ylidene_bonds(mol, pos, frag_set):
    """The double bonds from the fragment numbered by ``pos`` to an acyclic parent atom outside ``frag_set`` (a ring
    '-ylidene' group on a chain): the host cites them, (1) method (a) (the Blue Book), the PIN
    method, at the front of the name with the parent locant. Empty without ``frag_set``."""
    if frag_set is None:
        return ()
    from ..rules.stereochemistry import ylidene_bond_on_chain
    return tuple(
        b.GetIdx() for b in bonds_of(mol)
        if ((b.GetBeginAtomIdx() in pos) != (b.GetEndAtomIdx() in pos))
        and (b.GetBeginAtomIdx() not in frag_set or b.GetEndAtomIdx() not in frag_set)
        and ylidene_bond_on_chain(mol, b))
