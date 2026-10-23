"""The ring assembly a composer fragment names as a bare prefix.

 / name the bare assembly as a prefix ('[1,1'-biphenyl]-4-yl', the Blue Book). A group on
one of its rings is cited with it ('4'-chloro[1,1'-biphenyl]-4-yl',,:15560 and:7455), which the bare
prefix cannot hold: it dropped the group. A decorated assembly is named with its groups by the single-ring
pass of ``composer._generate_ring_substituent_prefixes``, whose chokepoint takes the ring system and every
group that hangs off it.
"""
from ..rules.ring_assemblies import detect_ring_assembly


def bare_ring_assembly_info(mol, ring_systems, fragment_atoms, chain_atoms):
    """``detect_ring_assembly`` of ``ring_systems``, or None when a group other than the parent chain hangs
    off an atom of the fragment (``fragment_atoms``): the bare prefix would omit it."""
    info = detect_ring_assembly(mol, ring_systems)
    if info is not None and any(
            nb.GetIdx() not in fragment_atoms and nb.GetIdx() not in chain_atoms and nb.GetAtomicNum() > 1
            for a in fragment_atoms for nb in mol.GetAtomWithIdx(a).GetNeighbors()):
        return None
    return info
