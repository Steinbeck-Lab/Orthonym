"""v33 glyco composer, slice 1 — NON-REDUCING oligosaccharides (3+ units).

Raffinose is the canonical witness: a non-reducing trisaccharide
(alpha-D-Gal-(1->6)-alpha-D-Glc central-linked (1<->2) to beta-D-Fru) that the
reducing-chain assembler (`_oligo_topology`) and the binary assembler
(`_count_sugar_rings>=3`) both fail closed on, so it currently emits `unknown`.

The expected name is OPSIN-RT-verified to raffinose's InChIKey (MUPFEKGTMRGPLJ).
"""
from rdkit import Chem
from orthonym.rules import oligosaccharides as O

RAFFINOSE = "OC[C@H]1O[C@H](OC[C@H]2O[C@H](O[C@]3(CO)O[C@H](CO)[C@@H](O)[C@@H]3O)[C@H](O)[C@@H](O)[C@@H]2O)[C@H](O)[C@@H](O)[C@H]1O"
EXPECTED = "alpha-D-galactopyranosyl-(1->6)-alpha-D-glucopyranosyl beta-D-fructofuranoside"


def test_raffinose_nonreducing_trisaccharide():
    mol = Chem.MolFromSmiles(RAFFINOSE)
    assert mol is not None
    name = O.name_nonreducing_oligosaccharide(mol)
    assert name == EXPECTED, f"got {name!r}"


def test_raffinose_via_public_entry():
    # name_disaccharide (the public P-102.7 entry) must now route raffinose too
    mol = Chem.MolFromSmiles(RAFFINOSE)
    assert O.name_disaccharide(mol) == EXPECTED


def test_nonreducing_declines_under_three_units():
    # the new 3+ namer must fail closed on <3 sugar units (a monosaccharide /
    # disaccharide is the single-sugar / binary assembler's job — no double-handling).
    glucose = "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O"  # 1 unit
    m = Chem.MolFromSmiles(glucose)
    assert m is not None
    assert O.name_nonreducing_oligosaccharide(m) is None
