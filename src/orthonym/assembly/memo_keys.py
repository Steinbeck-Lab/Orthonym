"""Parts of memo keys shared by several memos (``assembly.memo``)."""


def mol_graph_key(mol):
    """A complete, atom-ORDERED description of *mol*'s graph (every atom's element,
    charge, isotope, hydrogens, aromaticity and chiral tag; every bond's atoms,
    order, stereo and direction), so two molecule objects share a key only when
    atom indices mean the same atoms. Read-only: unlike ``MolToSmiles`` it sets no
    output-order property. RDKit's own pickle of the molecule (no properties), one
    C++ call: 57 us for 84 atoms where the tuples below took 1.1 ms, and the key of
    every ``name_substituent`` call holds it."""
    try:
        return mol.ToBinary()
    except Exception:  # noqa: BLE001 -- a molecule RDKit cannot pickle is described atom by atom
        pass
    atoms = tuple(
        (a.GetAtomicNum(), a.GetFormalCharge(), a.GetIsotope(), a.GetNumExplicitHs(),
         a.GetNoImplicit(), a.GetIsAromatic(), int(a.GetChiralTag()),
         a.GetNumRadicalElectrons())
        for a in mol.GetAtoms())
    bonds = tuple(
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx(), int(b.GetBondType()),
         int(b.GetStereo()), tuple(b.GetStereoAtoms()), int(b.GetBondDir()))
        for b in mol.GetBonds())
    return atoms, bonds
