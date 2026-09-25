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

Radical electrons on METAL atoms are not counted: RDKit's valence model assigns
them to bare cations such as [Pb+2] or [Sn+2] and to covalent metal centres, and
they are not chemical radicals in the sense of. One named whole-molecule
exemption: dioxygen, the retained name for O2 (data/retained_names.py), whose
ground state is the triplet [O][O] while OPSIN draws O=O; says radical
names "do not indicate nor imply an electronic structure or spin multiplicity"
(the Blue Book).
"""
from typing import Tuple

from rdkit import Chem

# Non-metals and metalloids: a radical electron on one of these is a chemical
# radical (silyl, boranyl, arsanyl, germyl... are radicals).
_NONMETALS = frozenset({
    1, 2, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 32, 33, 34, 35, 36,
    51, 52, 53, 54, 85, 86,
})

# (input canonical SMILES, parse canonical SMILES) pairs accepted by name.
_EXEMPT_PAIRS = frozenset({("[O][O]", "O=O")})


def radical_profile(mol) -> Tuple[Tuple[str, int], ...]:
    """Sorted (element, radical electrons) of every non-metal radical atom."""
    return tuple(sorted(
        (a.GetSymbol(), a.GetNumRadicalElectrons())
        for a in mol.GetAtoms()
        if a.GetNumRadicalElectrons() and a.GetAtomicNum() in _NONMETALS))


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
