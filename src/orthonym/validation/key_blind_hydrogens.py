"""Hydrogens the standard InChIKey cannot see: does a name's OPSIN parse carry the same hydrogens as the input?

The standard InChI drops the hydrogens of a hypervalent halogen, so two structures that differ in them share
one key and one InChI string:

    C[IH2] vs CI (methyl-lambda3-iodane vs iodomethane)
    C[IH2+]C vs C[I+]C (a lambda3 'ium' of one hydrogen pair more vs the iodonium)
    C[IH4] vs CI

``CN1CCCC1C1CCC(CC1)[IH2]`` was named '2-(4-iodocyclohexyl)-1-methylpyrrolidine' and shipped as a preferred
name, and a spiro iodonium as '3lambda3-iodaspiro[5.5]undecan-3-ium', each with the full key of the input.
Every gate that accepts a name on full-key equality alone is blind to them, as it is to radical electrons
(``radical_identity``) and to the site of a hydron (``protonation_identity``).

The class is measured, not assumed. Over every element from He to Rn, with zero to seven hydrogens and a charge
of 0, +1 and -1 on a neutral, a chain, a ring and a branched drawing, the pairs of structures that RDKit reads, that
differ in the number of hydrogens and that share a standard InChIKey are all on Cl, Br or I
(``tests/unit/validation/test_key_blind_hydrogens.py`` runs the measurement again, so an element the InChI
library comes to treat the same way is found). Sulfur, selenium, tellurium and phosphorus with extra hydrogens keep
them in the key; so do carbon and nitrogen. So the rule applies when either structure holds a halogen atom of
bonding number two or more (an onium or a hypervalent halogen, never the monovalent halide or the bare ion, whose
hydrogens the key does see), and asks the one thing the key cannot: the same number of hydrogen atoms on both
sides. The count is the whole structure's, so a drawing that moves a hydrogen from one atom to another, a salt
drawn with its proton or without, and a Kekule form of the same conjugated ring are never rejected by it; only a
name that adds or loses a hydrogen does.
"""
from rdkit import Chem

#: F, Cl, Br, I, At: the halogens. The measurement finds the key blind to Cl, Br and I; fluorine and astatine are listed
#: with them because the same hypervalent state is the one question, and a count that agrees costs nothing.
_HALOGENS = frozenset({9, 17, 35, 53, 85})
#: the symbols of those elements as a SMILES spells them ('F' also finds Fe and Fr, 'I' In and Ir: a superset)
_HALOGEN_SYMBOLS = ("F", "Cl", "Br", "I", "At")


def _has_halogen_of_higher_bonding_number(mol) -> bool:
    """A halogen atom whose bonding number (bonds and hydrogens) is two or more: the onium and the
    hypervalent states, the only ones whose hydrogens the key may drop."""
    return any(a.GetAtomicNum() in _HALOGENS and a.GetTotalValence() >= 2 for a in mol.GetAtoms())


def hydrogen_atom_count(mol) -> int:
    """The number of hydrogen atoms of ``mol``, implicit and explicit alike (the hydrogens of its formula)."""
    return sum(1 for a in Chem.AddHs(mol).GetAtoms() if a.GetAtomicNum() == 1)


def key_blind_hydrogen_verdict(input_smiles: str, parsed_smiles: str) -> str:
    """``"n/a"`` when neither structure holds a halogen of bonding number two or more (the caller's own
    checks decide), else ``"ok"`` when the two have the same number of hydrogen atoms and ``"mismatch"`` when
    they do not. An input that holds such a halogen whose parse RDKit cannot read is a ``"mismatch"``
    (fail closed)."""
    if not any(sym in text for text in (input_smiles or "", parsed_smiles or "") for sym in _HALOGEN_SYMBOLS):
        return "n/a"        # no halogen symbol in either string: no parse is needed to say so
    mi = Chem.MolFromSmiles(input_smiles) if input_smiles else None
    if mi is None:
        return "n/a"
    mo = Chem.MolFromSmiles(parsed_smiles) if parsed_smiles else None
    if mo is None:
        return "mismatch" if _has_halogen_of_higher_bonding_number(mi) else "n/a"
    if not (_has_halogen_of_higher_bonding_number(mi) or _has_halogen_of_higher_bonding_number(mo)):
        return "n/a"
    return "ok" if hydrogen_atom_count(mi) == hydrogen_atom_count(mo) else "mismatch"
