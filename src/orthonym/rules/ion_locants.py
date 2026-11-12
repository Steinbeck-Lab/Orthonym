"""Where the suffix of an ion may leave out the locant of its charged atom.

 Citation of locants (the Blue Book): "In preferred IUPAC names, if any locants are
essential for defining the structure of the parent structure or of a unit of structure as defined
by its appropriate enclosing marks, then all locants must be cited for the parent structure or
that structural unit." A cationic or anionic suffix is cited at the position of its charge
,:41366;,:41559;,:40900); the licences that leave the locant
out are (:2873) and the specific methods of (:41555).
"""
from rdkit import Chem


from .ring_ion_numbering import bare_fused_ring_locants  # noqa: F401,E402 (re-exported for ions)

def cites_a_locant(name: str) -> bool:
    """A neutral name that carries a digit numbers its own parent hydride: its indicated
    hydrogen, hydro prefixes, ring fusion or substituents. That numbering does not number the
    charged atom of the ion, so an ionic suffix glued to the name has no locant to cite."""
    return any(ch.isdigit() for ch in name)


def ring_ylium_may_be_locant_free(mol) -> bool:
    """ Specific method (the Blue Book): a hydride removed "from a terminal atom
    of a saturated unbranched acyclic hydrocarbon, a saturated monocyclic hydrocarbon, or a
    mononuclear parent hydride belonging to Group 14" gives the name of the parent hydride with
    'ane' replaced by 'ylium' ('cyclobutylium (PIN)'); the examples of the same paragraph include
    'triphenylsilylium (PIN)', so the Blue Book does not confine the method to a bare ring. The
    general method (:41559) cites the position of the centre ('furan-2-ylium (PIN)',:41571).
    ``mol`` is the whole cation. This producer glues 'ylium' to the neutral name and has no
    locant for a substituent, so it takes only the bare saturated monocycle and declines every
    other ring cation (``route_charged`` returns no name for it; another producer then cites the
    centre, '1-methylcyclobutan-1-ylium' at the best-effort tier). ASSUMED, follow-up: the
    specific-method PIN of a substituted saturated monocycle ('1-methylcyclobutylium')."""
    from .ions import _is_saturated_carbocycle

    rings = mol.GetRingInfo().AtomRings()
    return (len(rings) == 1 and len(rings[0]) == mol.GetNumHeavyAtoms()
            and _is_saturated_carbocycle(mol, rings[0]))


def _substituent_locant_is_licensed_away(ring_mol, ring_atoms) -> bool:
    """ (the Blue Book) leaves out the locant of the only substituent of a ring
    whose positions are of one kind ('methyloxirane'); with two substituents, or two kinds of
    position, the name cites it anyway."""
    from .ring_position_kinds import one_kind_of_ring_position

    ring = set(ring_atoms)
    substituents = sum(1 for i in ring for nb in ring_mol.GetAtomWithIdx(i).GetNeighbors()
                       if nb.GetIdx() not in ring and nb.GetAtomicNum() > 1)
    return substituents == 1 and one_kind_of_ring_position(ring_mol, ring_atoms)


def name_neutral_ring_for_ion(ring_mol, ring_atoms, centre_locant_is_cited: bool) -> str:
    """The name of the neutral ring that an ionic suffix is added to. When the suffix cites the
    locant of its centre, (the Blue Book, see above) cites every locant of the
    unit, so the licence that left out the locant of the only substituent of the ring is off
    ('2-methyloxirane', not 'methyloxirane', for '2-methyloxiran-2-ide'; '1-methylcyclohexane'
    for '1-methylcyclohexan-1-ide'). Only then: a switch that turns off every licence of
    the scope also reaches the substituents, which are units of their own."""
    from ..assembly.locant_omission import parent_positional_scope
    from ..namer import Orthonym

    namer = Orthonym(style='pin', _disable_opsin_validity_gate=True)
    smiles = Chem.MolToSmiles(ring_mol)
    if not (centre_locant_is_cited and _substituent_locant_is_licensed_away(ring_mol, ring_atoms)):
        return namer.name(smiles)
    with parent_positional_scope("ring ion suffix locant"):
        return namer.name(smiles)
