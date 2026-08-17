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


# --- slice 2: BRANCHED reducing oligosaccharides (P-102.7.3) ---
# a unit accepting >1 glycosyl; both expected names are OPSIN-RT-verified to the input.
BRANCHED_GLUCOTRIOSE = "OC[C@H]1O[C@H](OC[C@H]2OC(O)[C@H](O)[C@@H](O)[C@@H]2O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@H](O)[C@@H](O)[C@@H]1O"
BRANCHED_GLUCOTRIOSE_NAME = "alpha-D-glucopyranosyl-(1->6)-[alpha-D-glucopyranosyl-(1->4)]-D-glucopyranose"


def test_branched_glucotriose():
    mol = Chem.MolFromSmiles(BRANCHED_GLUCOTRIOSE)
    assert mol is not None
    assert O.name_branched_oligosaccharide(mol) == BRANCHED_GLUCOTRIOSE_NAME


def test_branched_via_public_entry():
    mol = Chem.MolFromSmiles(BRANCHED_GLUCOTRIOSE)
    assert O.name_disaccharide(mol) == BRANCHED_GLUCOTRIOSE_NAME


def test_branched_lewis_type():
    # beta-D-Gal-(1->3)-[alpha-L-Fuc-(1->4)]-D-Glc (a Lewis-a core) round-trips.
    smi = "C[C@@H]1O[C@@H](O[C@H]2[C@H](O[C@@H]3O[C@H](CO)[C@H](O)[C@H](O)[C@H]3O)[C@@H](O)C(O)O[C@@H]2CO)[C@@H](O)[C@H](O)[C@@H]1O"
    mol = Chem.MolFromSmiles(smi)
    name = O.name_branched_oligosaccharide(mol)
    assert name == "alpha-L-fucopyranosyl-(1->4)-[beta-D-galactopyranosyl-(1->3)]-D-glucopyranose", name


def test_branched_declines_linear():
    # a LINEAR reducing chain (no branch point) must fall through to the linear
    # namer, not the branched one.
    maltose = "OC[C@H]1O[C@H](O[C@H]2[C@H](O)[C@@H](O)C(O)O[C@@H]2CO)[C@H](O)[C@@H](O)[C@@H]1O"
    m = Chem.MolFromSmiles(maltose)
    assert m is not None
    assert O.name_branched_oligosaccharide(m) is None


def test_dispatch_precondition_routes_extended_oligo():
    # DISPATCH INTEGRATION (closes the choke-point-off-path blind spot): the cheap
    # no-OPSIN precondition must fire for non-reducing AND branched oligosaccharides,
    # else name_tiered routes them to the general oxane engine and the composer,
    # though correct when called directly, is never reached.
    assert O._has_extended_oligo(Chem.MolFromSmiles(RAFFINOSE)) is True
    assert O._has_extended_oligo(Chem.MolFromSmiles(BRANCHED_GLUCOTRIOSE)) is True
    # a single monosaccharide / disaccharide must NOT trip it.
    assert O._has_extended_oligo(Chem.MolFromSmiles("OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O")) is False


def test_nonreducing_declines_under_three_units():
    # the new 3+ namer must fail closed on <3 sugar units (a monosaccharide /
    # disaccharide is the single-sugar / binary assembler's job — no double-handling).
    glucose = "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O"  # 1 unit
    m = Chem.MolFromSmiles(glucose)
    assert m is not None
    assert O.name_nonreducing_oligosaccharide(m) is None
