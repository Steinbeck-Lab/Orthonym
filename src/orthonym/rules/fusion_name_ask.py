"""A von Baeyer substituent name where the ring system takes a fusion name asks the PIN tier (lane/c4c, 2026-11-20).

 "Five-membered ring requirement" (the Blue Book): "Fusion nomenclature gives preferred IUPAC names
only to compounds having at least two rings of at least five or more members... When fusion names are not allowed,
unsaturated von Baeyer ring system names are preferred IUPAC names". The universal cage namer of the wider tiers
(``rules.ring_substituents._polycyclic_substituent_name``, ``allow_mancude``) names a fused substituent with its von
Baeyer name: '1-methyl-4-{[tricyclo[9.4.0.0^3,8]pentadeca-1(15),3,5,7,11,13-hexaen-2-yl]oxy}piperidine', where the PIN
tier names the molecule '4-[(10,11-dihydro-5H-dibenzo[a,d][7]annulen-5-yl)oxy]-1-methylpiperidine'. Such a name is a
name in a lower form, and the producer says so (``assembly.pin_first.note_pin_tier_asked``): the PIN tier's verified
PIN is then taken; asking more often than needed costs time, never a name."""


def takes_a_fusion_name(sub) -> bool:
    """True for a ring system with an aromatic ring and two rings of five or more members that share a bond."""
    rings = [set(r) for r in sub.GetRingInfo().BondRings() if len(r) >= 5]
    if len(rings) < 2 or not any(a.GetIsAromatic() for a in sub.GetAtoms()):
        return False
    return any(rings[i] & rings[j] for i in range(len(rings)) for j in range(i + 1, len(rings)))


def asked_for_a_fusion_name(name: str, sub) -> str:
    """``name`` (a von Baeyer substituent name of ``sub``), asking the PIN tier when ``sub`` takes a fusion name, in the
    naming of the whole input only: a part of a mixture is named by a nested call, and the ask would make the PIN tier
    search the whole mixture, which the labelled-parts route keeps for mixtures whose parts are PINs
    (``rules.isotope_components``; end check 2026-11-20: a slow labelled mixture was searched whole again)."""
    from ..assembly.fragment_naming import name_scope_depth
    if name_scope_depth() <= 1 and takes_a_fusion_name(sub):
        from ..assembly.pin_first import note_pin_tier_asked
        note_pin_tier_asked()
    return name
