"""The key of one numbering of a ring among those that tie on every earlier criterion of ``### ****
NUMBERING`` (``the Blue Book``): (g), (i), (j)."""


def ring_numbering_tie_key(mol, prefix_names, atom_to_locant, oriented) -> tuple:
    """The key of one numbering of a ring among those that tie on every earlier criterion of
    ``### **** NUMBERING`` (``the Blue Book``): (g) the substituent cited first takes the lowest
    locants (``:3307``), by, which leaves R and S out so that two prefixes differing only in them
    tie; (i) the lowest locants to the isotopically modified atoms (``:3336``); then (j) the CIP
    stereodescriptors of the ring and those cited inside its prefixes, together, at the locants of the ring
    atoms they stand on or hang on (``:3346``; ``1-[(2R)-butan-2-yl]-3-[(2S)-butan-2-yl]benzene (PIN)``,
    ``:3399``). ``prefix_names`` is:func:`_ring_prefix_names`; ``oriented`` the ring atoms in locant order."""
    from ..assembly.locant_omission import isotope_locant_key
    from .heterocycles import _ring_cip_locant_key
    here = [(idx, alpha, nm) for idx, lst in prefix_names.items() if idx in atom_to_locant
            for alpha, nm in lst]
    return (tuple(sorted((alpha, atom_to_locant[idx]) for idx, alpha, _nm in here)),
            isotope_locant_key(mol, atom_to_locant),
            _ring_cip_locant_key(mol, oriented, [(atom_to_locant[idx], nm) for idx, _alpha, nm in here]))


def spiro_tie_key(mol, mapping, names, at, from_submolecule: bool, isotopic: bool) -> tuple:
    """The key of one numbering of a spiro system among those that tie on every earlier criterion: (g) by
    , (i) the lowest locants to the isotopically modified atoms (``the Blue Book``), (j) the CIP
    stereodescriptors of the ring and of its prefixes (``:3346``). ``mapping`` numbers the atoms of ``mol``;
    ``at`` the atoms of the molecule that carries the prefixes ``names`` (:func:`heterocycles._ring_prefix_names`).
    (i) and (j) are not asked of the ring-only submolecule of the general engine, which has no labels and no CIP
    labels, nor (j) of an isotopically modified ``mol``."""
    from ..assembly.locant_omission import isotope_locant_key
    from .spiro import _spiro_cip_locant_key
    here = [(idx, alpha, nm) for idx, lst in names.items() if idx in at for alpha, nm in lst]
    return (tuple(sorted((alpha, at[idx]) for idx, alpha, _nm in here)),
            () if from_submolecule else isotope_locant_key(mol, mapping),
            () if isotopic or from_submolecule else _spiro_cip_locant_key(
                mol, mapping, [(at[idx], nm) for idx, _alpha, nm in here]))
