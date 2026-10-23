"""The name of a decorated ring assembly that is a substituent, labelled as what it is.

 DEFINITIONS (the Blue Book): identical rings joined by single bonds are a ring assembly and
the PIN names the assembly as one unit, '[1,1':4',1''-terphenyl]' for three rings,:50157 among
the PINs of the assembly prefixes). The substituent chokepoint names a fragment that holds an assembly
with a group on one of its rings by one ring and cites the others as substituents of it
('4-(4'-chloro[1,1'-biphenyl]-4-yl)phenyl'): a valid name, never the PIN, which is built only for two
benzene rings ('4'-chloro[1,1'-biphenyl]-4-yl', ``ring_substituents._decorated_biphenylyl``). Such a name is
recorded as not the PIN, so it is labelled below pin_verified and the default tier declines it.

A name of a decorated assembly is a compound prefix: it is marked as one the Blue Book, 'bis'
for a compound or complex substituent prefix).
"""
from typing import Optional

from ..assembly.prefix_derivation import built
from ..metrics.provenance import record_non_pin_fragment


def mark_decorated_assembly(mol, frag_atoms, name: Optional[str]) -> Optional[str]:
    """``name`` of the fragment ``frag_atoms`` as the chokepoint returns it: unchanged unless the fragment is
    a ring assembly of identical rings that carries a group, whose name is then marked as a compound prefix, and
    recorded as not the PIN when it does not cite the assembly as one unit (N rings: N-1 primes in the name)."""
    if not name:
        return name
    from ..perception.rings import get_ring_systems
    from .ring_assemblies import detect_ring_assembly
    frag = {int(a) for a in frag_atoms}
    systems = [s for s in get_ring_systems(mol) if s <= frag]
    if len(systems) < 2 or sum(len(s) for s in systems) == len(frag):
        return name  # one ring system, or the bare assembly (its prefix is the PIN)
    info = detect_ring_assembly(mol, systems)
    if info is None:
        return name
    if "'" * (info["count"] - 1) not in name:
        record_non_pin_fragment(str(name))
    return built(str(name), substituted=True)
