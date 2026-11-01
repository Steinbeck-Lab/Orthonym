"""Oligonucleotide structures built from nucleosides, for the tests of ``rules/oligonucleotides.py``.

``oligo`` joins the 3'-O of each unit to the 5'-O of the next through a phosphate (or a P-thio phosphate). The bonds
are made at oxygen atoms, which are no stereocentres, so every sugar keeps the configuration of its nucleoside
below (beta-D, written once per base and sugar). The names the tests expect are read back by OPSIN
(``tests.support.rt_assert.name_is_rt_exact``), which does not use this module.
"""
from rdkit import Chem

#: 'A' adenosine, 'G' guanosine, 'C' cytidine, 'U' uridine, 'I' inosine, 'X' xanthosine; 'M' is not one of the bases
RIBO = {
    "A": "Nc1ncnc2c1ncn2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O",
    "G": "Nc1nc2c(ncn2[C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=O)[nH]1",
    "C": "Nc1ccn([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=O)n1",
    "U": "O=c1ccn([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=O)[nH]1",
    "I": "O=c1[nH]cnc2c1ncn2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O",
    "X": "O=c1[nH]c(=O)c2ncn([C@@H]3O[C@H](CO)[C@@H](O)[C@H]3O)c2[nH]1",
    "M": "Cc1cn([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=O)[nH]c1=O",       # 5-methyluridine, no prefix name
}
#: the 2'-deoxy series; 'T' is thymidine
DEOXY = {
    "A": "Nc1ncnc2c1ncn2[C@H]1C[C@H](O)[C@@H](CO)O1",
    "G": "Nc1nc2c(ncn2[C@H]2C[C@H](O)[C@@H](CO)O2)c(=O)[nH]1",
    "C": "Nc1ccn([C@H]2C[C@H](O)[C@@H](CO)O2)c(=O)n1",
    "U": "O=c1ccn([C@H]2C[C@H](O)[C@@H](CO)O2)c(=O)[nH]1",
    "I": "O=c1[nH]cnc2c1ncn2[C@H]1C[C@H](O)[C@@H](CO)O1",
    "T": "Cc1cn([C@H]2C[C@H](O)[C@@H](CO)O2)c(=O)[nH]c1=O",
    "M": "Cc1cn([C@H]2C[C@H](O)[C@@H](CO)O2)c(=O)nc1N",               # 5-methyl-2'-deoxycytidine, no prefix name
}
_ENDS = Chem.MolFromSmarts("[OX2H1:5][CH2][CH]1[OX2][CH]([#7])[CH2,CH1][CH]1[OX2H1:3]")


def oligo(units, thio=(), ends=None):
    """SMILES of the chain ``units`` = [(sugar, base),...] from the first unit, sugar 'r' (ribo) or 'd' (2'-deoxy).
    ``thio``: the 0-based links that are P-thio (P=S). ``ends``: per link the sugar oxygens joined, as (end of the
    unit before, end of the next unit) with an end "3'" or "5'"; the default is 3'->5' throughout."""
    mol, last = None, None
    for k, (sugar, base) in enumerate(units):
        unit = Chem.MolFromSmiles((RIBO if sugar == "r" else DEOXY)[base])
        (o5, o3), = [(m[0], m[-1]) for m in unit.GetSubstructMatches(_ENDS)]
        if mol is None:
            mol, last = Chem.RWMol(unit), {"5'": o5, "3'": o3}
            continue
        end_prev, end_next = (ends or [("3'", "5'")] * len(units))[k - 1]
        offset = mol.GetNumAtoms()
        mol = Chem.RWMol(Chem.CombineMols(mol, unit))
        p = mol.AddAtom(Chem.Atom(15))
        dbl = mol.AddAtom(Chem.Atom(16 if (k - 1) in thio else 8))
        oh = mol.AddAtom(Chem.Atom(8))
        mol.AddBond(p, dbl, Chem.BondType.DOUBLE)
        mol.AddBond(p, oh, Chem.BondType.SINGLE)
        mol.AddBond(p, last[end_prev], Chem.BondType.SINGLE)
        mol.AddBond(p, offset + {"5'": o5, "3'": o3}[end_next], Chem.BondType.SINGLE)
        last = {"5'": offset + o5, "3'": offset + o3}
    built = mol.GetMol()
    Chem.SanitizeMol(built)
    return Chem.MolToSmiles(built)
