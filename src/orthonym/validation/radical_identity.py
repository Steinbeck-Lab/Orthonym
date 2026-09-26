"""Radical identity: does a name's OPSIN parse carry the same radical centres as
the input structure?

The full standard InChIKey encodes neither radical electrons nor bond order, so
it cannot tell a radical from a closed-shell or differently placed twin with the
same formula and connectivity:

    [CH2][CH2] vs C=C (ethane-1,2-diyl vs ethene)
    [O]c1ccc([O])cc1 vs O=C1C=CC(=O)C=C1 (a bis(oxyl) vs p-benzoquinone)
    CN(C)[O] vs C[N+](C)[O-] (an aminoxyl vs its charge form)
    [CH2]C=CC vs C=C[CH]C (two allyl-type radical sites)

Every gate that accepts a name on full-key equality alone is therefore
radical-blind. This module is the one definition of the stricter rule, applied
whenever the input or the parse carries a chemical radical:

  1. the radical profile (element and radical-electron count of every radical
     atom) must be equal -- this also rejects a lambda-convention name whose
     parse is a radical for a closed-shell input, and the reverse;
  2. the stereo-stripped canonical isomeric SMILES must be equal -- graph
     identity over element, charge, hydrogen count, isotope, bond order and
     radical placement. Stereo is left to the caller's own stereo logic.

Radical electrons on most METAL atoms are not counted: RDKit's valence model
assigns them to bare cations such as [Pb+2] or [Sn+2] and to covalent metal
centres, and they are not chemical radicals in the sense of. The p-block
metals Al, Ga, In, Sn, Tl, Pb, Bi and Po are the exception, because names
radicals on them ('stannyl' the Blue Book, 'plumbyl':38069, 'alumanyl'
:15900) and the key cannot place them either: [SnH2][SnH2][SnH3]
(tristannan-1-yl) and [SnH3][SnH][SnH3] (tristannan-2-yl) share one InChIKey. A
radical electron on one of these counts unless the atom is drawn as a salt: a
bare ion (no neighbour, no hydrogen, charged) or an atom with no hydrogen whose
neighbours are all salt-like (N, O, F, S, Cl, Se, Br, Te, I) -- so 'tin(II)
dichloride' ([Sn+2].2[Cl-], parsed as Cl[Sn]Cl) stays "n/a". Transition metals
and groups 1/2 stay excluded (Cu+2, Mn+2 and Cl[Mg] carry RDKit radical
counts).

One named whole-molecule exemption: dioxygen, the retained name for O2
(data/retained_names.py), whose ground state is the triplet [O][O] while OPSIN
draws O=O; says radical names "do not indicate nor imply an electronic
structure or spin multiplicity" (the Blue Book).
"""
from typing import Tuple

from rdkit import Chem

# Non-metals and metalloids: a radical electron on one of these is a chemical
# radical (silyl, boranyl, arsanyl, germyl... are radicals).
_NONMETALS = frozenset({
    1, 2, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 32, 33, 34, 35, 36,
    51, 52, 53, 54, 85, 86,
})

# p-block metals whose radicals names (stannyl, plumbyl, alumanyl,...):
# Al, Ga, In, Sn, Tl, Pb, Bi, Po. Counted unless drawn as a salt (see below).
_P_BLOCK_METALS = frozenset({13, 31, 49, 50, 81, 82, 83, 84})
# Neighbours that make a hydrogen-free p-block metal a salt-like (ionic or
# covalent-salt) drawing rather than a radical centre.
_SALT_LIKE_NEIGHBOURS = frozenset({7, 8, 9, 16, 17, 34, 35, 52, 53})

# (input canonical SMILES, parse canonical SMILES) pairs accepted by name.
_EXEMPT_PAIRS = frozenset({("[O][O]", "O=O")})


def _is_radical_centre(atom) -> bool:
    """Does this atom's radical electron count as a chemical radical?"""
    if not atom.GetNumRadicalElectrons():
        return False
    z = atom.GetAtomicNum()
    if z in _NONMETALS:
        return True
    if z not in _P_BLOCK_METALS:
        return False
    if atom.GetTotalNumHs():
        return True
    if atom.GetDegree() == 0:
        return atom.GetFormalCharge() == 0   # a bare ion is a salt drawing
    return not all(nb.GetAtomicNum() in _SALT_LIKE_NEIGHBOURS
                   for nb in atom.GetNeighbors())


def radical_profile(mol) -> Tuple[Tuple[str, int], ...]:
    """Sorted (element, radical electrons) of every radical centre: each
    non-metal radical atom, and each p-block-metal radical atom not drawn as a
    salt."""
    return tuple(sorted(
        (a.GetSymbol(), a.GetNumRadicalElectrons())
        for a in mol.GetAtoms()
        if _is_radical_centre(a)))


def _stereo_free_smiles(mol) -> str:
    copy = Chem.Mol(mol)
    Chem.RemoveStereochemistry(copy)
    return Chem.MolToSmiles(copy)


def radical_identity_verdict(input_smiles: str, parsed_smiles: str) -> str:
    """``"n/a"`` when neither side carries a chemical radical (the caller's own
    checks decide), else ``"ok"`` or ``"mismatch"``. A radical input whose parse
    RDKit cannot read is a ``"mismatch"`` (fail closed)."""
    mi = Chem.MolFromSmiles(input_smiles) if input_smiles else None
    if mi is None:
        return "n/a"
    pi = radical_profile(mi)
    mo = Chem.MolFromSmiles(parsed_smiles) if parsed_smiles else None
    if mo is None:
        return "mismatch" if pi else "n/a"
    po = radical_profile(mo)
    if not pi and not po:
        return "n/a"
    if (Chem.MolToSmiles(mi), Chem.MolToSmiles(mo)) in _EXEMPT_PAIRS:
        return "ok"
    if pi != po:
        return "mismatch"
    return "ok" if _stereo_free_smiles(mi) == _stereo_free_smiles(mo) else "mismatch"
