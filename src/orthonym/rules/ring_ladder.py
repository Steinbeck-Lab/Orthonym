"""The rungs below the fusion namers, for a ring system the fusion namers decline (lane DT5).

The composer classifies a ring system whose rings share bonds as 'ortho-fused' and tries the fusion
namers in turn (catalogued parent, fusion carbocycle, ortho-fused bicycle, bridged fused). When every one
declines, the ladder goes on here instead of ending in ``ring_system_unnameable``:

1. Fusion names are not allowed for a PIN when fewer than two rings have five or more members.
    "Five-membered ring requirement" (the Blue Book): "Fusion nomenclature gives preferred
   IUPAC names only to compounds having at least two rings of at least five or more members.... When
   fusion names are not allowed, unsaturated von Baeyer ring system names are preferred IUPAC names (see
   ." The book's rows: 'bicyclo[4.2.0]octa-1,3,5,7-tetraene (PIN)' for cyclobutabenzene,
   'bicyclo[4.1.0]hepta-1,3,5-triene (PIN)' for 1H-cyclopropabenzene. Such a skeleton (an epoxide pair on
   a macrocycle, two cyclopropane or cyclobutane rings on a six-membered ring) is named by the von Baeyer
   namer (``polycyclic.name_polycyclic_complete``), which the 'ortho-fused' route never reached.

Each name is still read back by OPSIN before it ships (the engine's own check), so a rung that names the
wrong structure is dropped like any other candidate.
"""
from rdkit import Chem


def fusion_name_not_allowed(mol) -> bool:
    """True when the largest ring system of ``mol`` is an ortho- or ortho- and peri-fused system with fewer
    than two rings of five or more members: its PIN is a von Baeyer name)."""
    from .bridged_fused_pin import selection
    system = selection.ring_system(mol)
    rings = selection.fused_rings(mol, system) if system else None
    return rings is not None and not selection._five_membered_rule(rings)


def von_baeyer_when_fusion_not_allowed(mol, features):
    """``(name, ring_atoms, atom_to_locant, substituents_included)`` of the von Baeyer PIN of a ring system
    that fusion nomenclature may not name (``fusion_name_not_allowed``), or None."""
    if not fusion_name_not_allowed(mol):
        return None
    from ..errors import OrthonymLimitError
    from .polycyclic import name_polycyclic_complete
    try:
        return name_polycyclic_complete(mol, features)
    except OrthonymLimitError:
        return None                  # a rung that declines is no candidate; the ladder ends as before


def next_rung_or_unnameable(mol, features, logger, ring_type):
    """The ladder after the fusion namers of an 'ortho-fused' ring system declined: the von Baeyer rung when
    the book allows no fusion name, else the clean refusal the composer made before."""
    from ..assembly.composer import ComplexRingResult
    got = von_baeyer_when_fusion_not_allowed(mol, features)
    if got:
        return ComplexRingResult(*got)
    from ..metrics.abstention import ring_system_unnameable
    return ring_system_unnameable(logger, f"Fused ring naming failed for {ring_type} system")
