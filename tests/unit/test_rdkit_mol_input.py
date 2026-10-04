"""The public naming functions accept an RDKit ``Chem.Mol`` as well as a SMILES string.

A Mol is named through RDKit's SMILES of it: each public call turns the Mol into
``Chem.MolToSmiles(mol)`` and names that string as it names any string, after a check
that the string holds the molecule RDKit reads from the Mol (``orthonym.input_mol``).
So the oracle for every Mol here is the result of the same call on
``Chem.MolToSmiles(mol)``, or the documented ValueError where the check declines the
Mol. The caller's Mol is never changed, and a str argument runs no new code.
"""
import json
import os
import subprocess
import sys

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

import orthonym
import orthonym.input_mol as input_mol
from orthonym import Orthonym, classify_limit, name_compound, name_with_tree

pytestmark = pytest.mark.opsin_gate

BEST_EFFORT = dict(general_fallback=True, general_fallback_unverified=True,
                   allow_aromatic_general=True)

#: Every public call, by label, as a function of its argument; values that compare
#: with ``==`` (an ``OrthonymLimitError`` as its code, message and echoed input).
CALLS = {
    "name_compound": lambda a: name_compound(a),
    "name_compound_confidence": lambda a: name_compound(a, include_confidence=True),
    "name": lambda a: Orthonym().name(a),
    "name_tiered": lambda a: Orthonym().name_tiered(a),
    "name_tiered_best_effort": lambda a: Orthonym(**BEST_EFFORT).name_tiered(a),
    "name_with_confidence": lambda a: Orthonym().name_with_confidence(a),
    "name_with_tree": lambda a: name_with_tree(a),
    "Orthonym.name_with_tree": lambda a: Orthonym().name_with_tree(a),
    "classify_limit": lambda a: _limit(classify_limit(a)),
}


def _limit(found):
    return None if found is None else (found.code, found.message, found.smiles)


def _mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    return mol


def _written(mol):
    """RDKit's SMILES of ``mol``, written from a copy (writing stores a property)."""
    return Chem.MolToSmiles(Chem.Mol(mol))


def _assert_named_as_its_smiles(mol, calls=tuple(CALLS)):
    written = _written(mol)
    for label in calls:
        assert CALLS[label](mol) == CALLS[label](written), (label, written)


def _assert_declined(mol, message):
    """Every public call declines ``mol`` with the documented ValueError, whose
    message contains ``message``; ``classify_limit`` says None (a reading error)."""
    for label, call in CALLS.items():
        if label == "classify_limit":
            assert call(mol) is None
            continue
        with pytest.raises(ValueError, match=message) as raised:
            call(mol)
        assert isinstance(raised.value, input_mol.InvalidInputError), label


_ANOTHER_INCHI = "has another InChI than RDKit reads from the Mol"
_NOT_ITS_STEREO = "does not hold its stereo"
_ANOTHER_MOLECULE = "back as another molecule"
_ATROPISOMER = "has an atropisomer tag"
_NOT_TETRAHEDRAL = "has a chiral tag that is not tetrahedral"
_LONE_PAIR = "has a lone-pair stereocentre whose configuration the check cannot confirm"


# ---------------------------------------------------------------------------
# A Mol gets the result of RDKit's SMILES of it
# ---------------------------------------------------------------------------

SAME_NAME = [
    "CCO", "CC(=O)O", "CCCCCC",
    "C[C@H](N)C(=O)O", "C[C@@H](O)[C@H](N)C(=O)O", "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
    "C/C=C/C", "C/C=C\\C", "OC(=O)/C=C/C(=O)O",
    "[2H]C([2H])([2H])O", "[13CH3]C(=O)O",
    "C[NH3+]", "CC(=O)[O-]", "CC(=O)[O-].[Na+]", "C[N+](C)(C)C.[Cl-]", "CCO.O",
    "[H]OC([H])([H])C", "[H][C@](C)(N)C(=O)O",
    "c1ccncc1", "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "c1ccc2[nH]ccc2c1", "C1=CC=CC=C1O",
    "C1CC2CC1c1ccccc12",
    "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5",
    "[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1", "C[Hg]Cl",
    "CC(C)C[C@H](NC(=O)[C@@H](N)Cc1ccccc1)C(=O)O",
    "CC(C)C[C@H](NC(=O)[C@H](CC(=O)O)NC(=O)[C@@H](N)CO)C(=O)O",
    "CC1(C)[C@@H]2CC[C@@]1(C)C(=O)C2",
    "C[S@@](=O)c1ccccc1", "CN1CCC[C@H]1c1cccnc1",
    "O=[U](=O)=O", "CC*",
    "CCC(N)C(=O)O", "NCCCC(N)C(=O)O", "CSCC(N)C(=O)O",
]


@pytest.mark.parametrize("smiles", SAME_NAME)
def test_name_compound_of_a_mol_is_the_name_of_its_smiles(smiles):
    mol = _mol(smiles)
    assert name_compound(mol) == name_compound(_written(mol))


@pytest.mark.parametrize("smiles", SAME_NAME)
def test_name_tiered_row_of_a_mol_is_the_row_of_its_smiles(smiles):
    mol = _mol(smiles)
    assert Orthonym().name_tiered(mol) == Orthonym().name_tiered(_written(mol))
    assert (Orthonym(**BEST_EFFORT).name_tiered(mol)
            == Orthonym(**BEST_EFFORT).name_tiered(_written(mol)))


@pytest.mark.parametrize("smiles", ["CCO", "C[C@H](N)C(=O)O", "C/C=C/C",
                                    "CC(=O)[O-].[Na+]", "O=[U](=O)=O", "CC*",
                                    "CC(C)C[C@H](NC(=O)[C@@H](N)Cc1ccccc1)C(=O)O"])
def test_every_public_call_on_a_mol_gives_the_result_of_its_smiles(smiles):
    _assert_named_as_its_smiles(_mol(smiles))


def test_limit_record_of_a_mol_echoes_its_smiles():
    mol = _mol("O=[U](=O)=O")
    found = classify_limit(mol)
    assert found.code == "UNSUPPORTED_ELEMENT" and found.smiles == _written(mol)


@pytest.mark.parametrize("smiles", ["C[C@@H](O)[C@H](N)C(=O)O", "C/C=C\\C",
                                    "[2H]C([2H])([2H])O", "CC(=O)[O-].[Na+]",
                                    "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5"])
def test_mol_with_hydrogen_atoms_is_named_as_its_smiles(smiles):
    with_h = Chem.AddHs(_mol(smiles))
    assert "[H]" in _written(with_h)
    _assert_named_as_its_smiles(with_h, ("name_compound", "name_tiered"))


@pytest.mark.parametrize("smiles", ["C[C@H](N)C(=O)O", "CC(=O)[O-].[Na+]", "C1=CC=CC=C1O"])
def test_unsanitized_mol_is_named_as_its_smiles(smiles):
    raw = Chem.MolFromSmiles(smiles, sanitize=False)
    before = _written(raw)
    _assert_named_as_its_smiles(raw, ("name_compound", "name_tiered"))
    assert _written(raw) == before


def test_unsanitized_mol_with_double_bond_stereo_only_in_directions_is_declined():
    """``MolFromSmiles('C/C=C/C', sanitize=False)`` holds its double-bond stereo only
    as directions on its single bonds. RDKit's InChI writer reads the tags, and finds
    none; its SMILES writer reads the directions and writes E. RDKit reads two
    molecules in this Mol, so the check declines it."""
    raw = Chem.MolFromSmiles("C/C=C/C", sanitize=False)
    assert "/b" not in Chem.MolToInchi(Chem.Mol(raw))
    assert _written(raw) == "C/C=C/C"
    _assert_declined(raw, _ANOTHER_INCHI)
    # Sanitized, the reader sets the tag from the directions: E, and named so.
    _assert_named_as_its_smiles(_mol("C/C=C/C"), ("name_compound",))


def test_keyword_argument_smiles_takes_a_mol():
    mol = _mol("C[C@H](N)C(=O)O")
    name = name_compound(_written(mol))
    assert name_compound(smiles=mol) == name
    assert Orthonym().name(smiles=mol) == name
    assert Orthonym().name_tiered(smiles=mol)["name"] == name
    assert Orthonym().name_with_confidence(smiles=mol)["name"] == name
    assert Orthonym().name_with_tree(smiles=mol).name == name
    assert name_with_tree(smiles=mol).name == name
    assert classify_limit(smiles=mol) is None


# ---------------------------------------------------------------------------
# Mols that RDKit's readers make
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smiles", ["C[C@@H](O)[C@H](N)C(=O)O",
                                    "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O", "C/C=C\\CO"])
@pytest.mark.parametrize("remove_hs", [True, False])
def test_3d_mol_block_is_named_as_its_smiles(smiles, remove_hs):
    with_h = Chem.AddHs(_mol(smiles))
    assert AllChem.EmbedMolecule(with_h, randomSeed=42) == 0
    read = Chem.MolFromMolBlock(Chem.MolToMolBlock(with_h), removeHs=remove_hs)
    assert read.GetConformer().Is3D()
    _assert_named_as_its_smiles(read, ("name_compound", "name_tiered"))


@pytest.mark.parametrize("smiles", ["C[C@@H](O)[C@H](N)C(=O)O", "C/C=C\\CO"])
def test_2d_mol_block_is_named_as_its_smiles(smiles):
    mol = _mol(smiles)
    AllChem.Compute2DCoords(mol)
    read = Chem.MolFromMolBlock(Chem.MolToMolBlock(mol))
    _assert_named_as_its_smiles(read, ("name_compound", "name_tiered"))


def test_sdf_records_are_named_as_their_smiles(tmp_path):
    smiles = ["C[C@H](N)C(=O)O", "OC(=O)/C=C/C(=O)O", "c1ccncc1", "CC(=O)[O-].[Na+]"]
    path = tmp_path / "small.sdf"
    writer = Chem.SDWriter(str(path))
    for s in smiles:
        mol = _mol(s)
        AllChem.Compute2DCoords(mol)
        writer.write(mol)
    writer.close()
    mols = list(Chem.SDMolSupplier(str(path)))
    assert [name_compound(mol) for mol in mols] == [name_compound(_written(m)) for m in mols]


# A mol block or SDF record can hold text that is not (a Latin-1 title, data
# field or atom property list); RDKit reads it, and only the SMILES of the Mol is named.

_LATIN1_TEXT = "25 \xb5g/ml, 4 \xb0C"


def _latin1_record(path, smiles, where):
    mol = _mol(smiles)
    AllChem.Compute2DCoords(mol)
    lines = Chem.MolToMolBlock(mol).split("\n")
    if where == "title":
        lines[0] = _LATIN1_TEXT
        path.write_bytes("\n".join(lines).encode("latin-1"))
        return Chem.MolFromMolFile(str(path))
    if where == "data_field":
        tail = f"> <comment>\n{_LATIN1_TEXT}\n\n$$$$\n"
    else:
        tail = "> <atom.prop.note>\n" + " ".join(["\xb5"] * mol.GetNumAtoms()) + "\n\n$$$$\n"
    path.write_bytes(("\n".join(lines) + tail).encode("latin-1"))
    return next(iter(Chem.SDMolSupplier(str(path))))


@pytest.mark.parametrize("smiles,where", [
    ("C/C=C/C(=O)O", "data_field"),
    ("C/C=C/C(=O)O", "title"),
    ("C[C@H](N)C(=O)N[C@@H](C)C(=O)O", "atom_property_list"),
])
def test_mol_with_text_that_is_not_utf8_is_named_as_its_smiles(tmp_path, smiles, where):
    mol = _latin1_record(tmp_path / "record.sdf", smiles, where)
    with pytest.raises(UnicodeDecodeError):          # the Mol holds such text
        mol.GetPropsAsDict(True, True)
    assert _written(mol) == smiles
    _assert_named_as_its_smiles(mol)


# Mols made by RDKit's readers carry their own atom order, hydrogen flags and
# bond-stereo flags; each is named as its SMILES, so these never change the result.

@pytest.mark.parametrize("inchi", [
    # MolFromInchi sets explicit-H flags on every atom
    "InChI=1S/C13H12N2O2/c14-11-6-7-15-12(8-11)13(16)17-9-10-4-2-1-3-5-10/h1-8H,9H2,(H2,14,15)",
    # the reader's atom order
    "InChI=1S/C17H35N3/c1-2-3-4-5-10-17(15-18)20-13-11-19(12-14-20)16-8-6-7-9-16"
    "/h16-17H,2-15,18H2,1H3",
    "InChI=1S/C8H10N2O2/c1-5(2)8(12)6-3-9-4-7(11)10-6/h3-5H,1-2H3,(H,10,11)",
    # ethyl diazoacetate (an abstain before)
    "InChI=1S/C4H6N2O2/c1-2-8-4(7)3-6-5/h3H,2H2,1H3",
])
def test_mol_from_inchi_is_named_as_its_smiles(inchi):
    mol = Chem.MolFromInchi(inchi)
    assert all(atom.GetNoImplicit() for atom in mol.GetAtoms())
    _assert_named_as_its_smiles(mol, ("name_compound", "name_tiered"))


def test_piperazine_row_is_named_with_the_lower_locant_for_the_free_valence():
    """The Mol of the InChI (or of a SMILES written in another atom order) gets the
    name of its SMILES, with the free valence locant 1 (c)), not
    '2-(1-cyclopentylpiperazin-4-yl)octan-1-amine'."""
    expected = "2-(4-cyclopentylpiperazin-1-yl)octan-1-amine"
    renumbered = ("C-1.C1-1.C1-1.C1-1.C1-1.C-2-3.C2-2.C3-3.C2-2.C1-1.C-4-5.C-6-7.C4-4."
                  "C6-6.C-8-9.C32-2.C18-8.N9.N572.N468")
    from_inchi = Chem.MolFromInchi(
        "InChI=1S/C17H35N3/c1-2-3-4-5-10-17(15-18)20-13-11-19(12-14-20)16-8-6-7-9-16"
        "/h16-17H,2-15,18H2,1H3")
    for mol in (from_inchi, _mol(renumbered)):
        assert name_compound(mol) == name_compound(_written(mol)) == expected


def _mol_block_in_atom_order(smiles, order):
    mol = Chem.RenumberAtoms(_mol(smiles), order)
    AllChem.Compute2DCoords(mol)
    return Chem.MolFromMolBlock(Chem.MolToMolBlock(mol))


@pytest.mark.parametrize("smiles,order", [
    ("CCNC(=O)C1CCN(Cc2cnccn2)CC1", [7, 15, 5, 1, 8, 14, 10, 16, 2, 0, 6, 3, 11, 4, 17, 9, 12, 13]),
    ("O/N=C(\\C)/C(/CC)=N\\O", [3, 5, 8, 2, 7, 6, 1, 4, 0]),
])
def test_mol_block_in_another_atom_order_is_named_as_its_smiles(smiles, order):
    mol = _mol_block_in_atom_order(smiles, order)
    _assert_named_as_its_smiles(mol, ("name_compound", "name_tiered"))


def test_mol_block_with_implicit_hydrogens_on_a_charged_atom_is_named_as_its_smiles():
    mol = Chem.MolFromMolBlock(Chem.MolToMolBlock(_mol("NCC[NH3+].[Cl-]")))
    assert name_compound(mol) == name_compound(_written(mol)) == "2-aminoethan-1-aminium chloride"


@pytest.mark.parametrize("smiles", ["NCC=CC1=CN2C(=O)C[C@@H]2SC1",
                                    "COc1cccc2n[nH]c(C=C[N+](=O)[O-])c12",
                                    "Brc1ccc(C=CCc2ccc(Br)cc2)cc1"])
def test_either_double_bond_from_a_mol_block_is_named_as_its_smiles(smiles):
    """RDKit's mol block of an open stereogenic C=C draws it crossed, and RDKit's
    reader gives it STEREOANY, which no SMILES holds."""
    mol = _mol(smiles)
    AllChem.Compute2DCoords(mol)
    read = Chem.MolFromMolBlock(Chem.MolToMolBlock(mol))
    assert Chem.BondStereo.STEREOANY in [b.GetStereo() for b in read.GetBonds()]
    _assert_named_as_its_smiles(read, ("name_compound", "name_tiered"))


def test_all_bracket_smiles_mol_is_named_as_its_smiles():
    """``MolFromSmiles`` of a SMILES with every atom in brackets sets the
    explicit-H flags; the Mol is named as RDKit's SMILES of it."""
    smiles = "COC(=O)c1ccc(Cl)c(NC(=O)CCNC2CC2)c1"
    expected = "methyl 4-chloro-3-[3-(cyclopropylamino)propanamido]benzoate"
    mol = _mol(Chem.MolToSmiles(_mol(smiles), allHsExplicit=True))
    assert all(atom.GetNoImplicit() for atom in mol.GetAtoms())
    assert name_compound(mol) == name_compound(_written(mol)) == expected
    from_json = Chem.JSONToMols(Chem.MolToJSON(_mol(smiles)))[0]
    assert name_compound(from_json) == name_compound(_written(from_json)) == expected


@pytest.mark.parametrize("smiles", ["C/C=C\\NC(=N)N/C=C\\C", "C/C=C/NC(=N)N/C=C/C",
                                    "C/C=C/N/C(=N/C)N/C=C/C"])
def test_guanidine_mol_gets_the_name_of_its_smiles(smiles):
    """A guanidine Mol gets the name of its SMILES."""
    _assert_named_as_its_smiles(_mol(smiles), ("name_compound", "name", "name_tiered",
                                                "name_with_confidence", "name_with_tree"))


# ---------------------------------------------------------------------------
# Stereo set in code
# ---------------------------------------------------------------------------

def test_mol_built_in_code_without_stereo_tags_is_named_without_stereo():
    rw = Chem.RWMol()
    for symbol in ("C", "C", "C", "C", "O"):
        rw.AddAtom(Chem.Atom(symbol))
    for a, b in ((0, 1), (1, 2), (2, 3), (1, 4)):
        rw.AddBond(a, b, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    assert name_compound(mol) == name_compound(_written(mol)) == "butan-2-ol"


def test_chiral_tag_set_in_code_is_used():
    rw = Chem.RWMol()
    for symbol in ("C", "C", "C", "C", "O"):
        rw.AddAtom(Chem.Atom(symbol))
    for a, b in ((0, 1), (1, 2), (2, 3), (1, 4)):
        rw.AddBond(a, b, Chem.BondType.SINGLE)
    rw.GetAtomWithIdx(1).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
    mol = rw.GetMol()
    assert "@" in _written(mol)
    _assert_named_as_its_smiles(mol, ("name_compound", "name_tiered"))
    assert name_compound(mol) in ("(2R)-butan-2-ol", "(2S)-butan-2-ol")


def _tagged_but_2_ene(stereo):
    mol = _mol("CC=CC")
    bond = mol.GetBondWithIdx(1)
    bond.SetStereoAtoms(0, 3)
    bond.SetStereo(stereo)
    return mol


@pytest.mark.parametrize("stereo,smiles", [(Chem.BondStereo.STEREOTRANS, "C/C=C/C"),
                                           (Chem.BondStereo.STEREOCIS, "C/C=C\\C"),
                                           (Chem.BondStereo.STEREOE, "C/C=C/C"),
                                           (Chem.BondStereo.STEREOZ, "C/C=C\\C")])
def test_double_bond_tag_set_in_code_is_used(stereo, smiles):
    mol = _tagged_but_2_ene(stereo)
    assert _written(mol) == smiles
    _assert_named_as_its_smiles(mol, ("name_compound", "name_tiered"))


@pytest.mark.parametrize("smiles", ["C/C=C/C", "C/C=C\\C", "OC(=O)/C=C/C(=O)O",
                                    "C/C(F)=C(/Cl)Br"])
def test_double_bond_stereo_kept_when_bond_directions_are_cleared(smiles):
    mol = _mol(smiles)
    for bond in mol.GetBonds():
        bond.SetBondDir(Chem.BondDir.NONE)
    assert _written(mol) == _written(_mol(smiles))
    assert name_compound(mol) == name_compound(smiles)


def _with_hydrogen_stereo_atom(smiles, end, stereo):
    """The ``AddHs`` Mol of ``smiles`` whose bond 1 is tagged ``stereo`` in code, with a
    hydrogen atom as the stereo atom on its ``end`` ("begin", "end" or "both")."""
    mol = Chem.AddHs(_mol(smiles))
    bond = mol.GetBondWithIdx(1)

    def hydrogen(atom):
        return [a.GetIdx() for a in atom.GetNeighbors() if a.GetAtomicNum() == 1][0]
    first = hydrogen(bond.GetBeginAtom()) if end in ("begin", "both") else 0
    second = hydrogen(bond.GetEndAtom()) if end in ("end", "both") else 3
    bond.SetStereoAtoms(first, second)
    bond.SetStereo(getattr(Chem.BondStereo, stereo))
    return mol


@pytest.mark.parametrize("smiles,end,stereo", [
    ("CC=NO", "begin", "STEREOE"), ("CC=NO", "begin", "STEREOZ"),
    ("CC=CC(=O)O", "begin", "STEREOE"), ("CC=CC(=O)O", "end", "STEREOZ"),
    ("CC=CC(=O)O", "begin", "STEREOTRANS"), ("CC=N", "both", "STEREOZ"),
])
def test_e_z_tag_on_a_hydrogen_stereo_atom_never_gets_the_other_configuration(smiles, end,
                                                                               stereo):
    """An E/Z tag set in code on an ``AddHs`` Mol, with a hydrogen atom as a
    stereo atom. RDKit's InChI writer reads the tag (E as trans on the stereo atoms);
    RDKit's SMILES writer writes no configuration for it. The SMILES does not hold
    what RDKit's InChI reads, so the Mol is declined rather than named with the
    other configuration. Its SMILES is named without one."""
    mol = _with_hydrogen_stereo_atom(smiles, end, stereo)
    assert "/b" in Chem.MolToInchi(mol)
    assert "/" not in _written(mol) and "\\" not in _written(mol)
    _assert_declined(mol, _ANOTHER_INCHI)


def _tags_set_in_code(smiles, tags):
    """``MolFromSmiles(smiles)`` with every bond direction and tag cleared, then ``tags``
    (bond index -> (stereo atoms, BondStereo name)) set in code."""
    mol = _mol(smiles)
    for bond in mol.GetBonds():
        bond.SetBondDir(Chem.BondDir.NONE)
        bond.SetStereo(Chem.BondStereo.STEREONONE)
    for index, (atoms, stereo) in tags.items():
        bond = mol.GetBondWithIdx(index)
        bond.SetStereoAtoms(*atoms)
        bond.SetStereo(getattr(Chem.BondStereo, stereo))
    return mol


def _ctu(smiles, pairs):
    """The CXSMILES of ``smiles`` whose bonds between the atom ``pairs`` are marked
    unknown (``ctu``), read by ``Chem.MolFromSmiles``."""
    parsed = _mol(smiles)
    ctu = ",".join(str(parsed.GetBondBetweenAtoms(a, b).GetIdx()) for a, b in pairs)
    return _mol(f"{smiles} |ctu:{ctu}|")


def _crossed_in_a_mol_block(smiles, pairs):
    """The 2D mol block of ``smiles`` with bond stereo 3 (crossed, either) on the bonds
    between the atom ``pairs``, read by ``Chem.MolFromMolBlock``."""
    parsed = _mol(smiles)
    AllChem.Compute2DCoords(parsed)
    lines = Chem.MolToMolBlock(parsed).split("\n")
    count = parsed.GetNumAtoms()
    wanted = [{a + 1, b + 1} for a, b in pairs]
    for i in range(4 + count, 4 + count + parsed.GetNumBonds()):
        if {int(lines[i][0:3]), int(lines[i][3:6])} in wanted:
            lines[i] = lines[i][:9] + "  3" + lines[i][12:]
    return Chem.MolFromMolBlock("\n".join(lines))


_BRANCHED = "C/C=C/C(/C=C/C)=C/C=C/C"
_AMIDINE = "CCOC(=O)/C=C/C(N)=N/C=C/C"

#: Double bonds left open (untagged, either, ctu, crossed) next to set ones. Each Mol
#: is named as its SMILES only when that SMILES holds its stereo; else declined.
_OPEN_NEXT_TO_SET = {
    "triene_open_middle": lambda: _tags_set_in_code(
        "CC(F)=C(Cl)C=CC=C(Br)C", {2: ((0, 4), "STEREOZ"), 7: ((6, 9), "STEREOE")}),
    "octatriene_open_middle": lambda: _tags_set_in_code(
        "CC=CC=CC=CC", {1: ((0, 3), "STEREOE"), 5: ((4, 7), "STEREOZ")}),
    "octatriene_any_middle": lambda: _tags_set_in_code(
        "CC=CC=CC=CC", {1: ((0, 3), "STEREOE"), 3: ((2, 5), "STEREOANY"),
                        5: ((4, 7), "STEREOZ")}),
    "cyclooctatetraene_ZZZE": lambda: _tags_set_in_code(
        "C1=CC=CC=CC=C1", {0: ((7, 2), "STEREOZ"), 2: ((1, 4), "STEREOZ"),
                           4: ((3, 6), "STEREOZ"), 6: ((5, 0), "STEREOE")}),
    "cyclooctatetraene_EEEZ": lambda: _tags_set_in_code(
        "C1=CC=CC=CC=C1", {0: ((7, 2), "STEREOE"), 2: ((1, 4), "STEREOE"),
                           4: ((3, 6), "STEREOE"), 6: ((5, 0), "STEREOZ")}),
    "branched_ctu": lambda: _ctu(_BRANCHED, [(1, 2), (3, 7)]),
    "branched_crossed_mol_block": lambda: _crossed_in_a_mol_block(_BRANCHED, [(1, 2), (3, 7)]),
    "branched_untagged": lambda: _tags_set_in_code(
        "CC=CC(C=CC)=CC=CC", {4: ((3, 6), "STEREOTRANS"), 8: ((7, 10), "STEREOE")}),
    "amidine_ctu": lambda: _ctu(_AMIDINE, [(7, 9)]),
    "amidine_untagged": lambda: _tags_set_in_code(
        "CCOC(=O)C=CC(N)=NC=CC", {5: ((3, 7), "STEREOE"), 10: ((9, 12), "STEREOE")}),
    "imidic_acid_untagged": lambda: _tags_set_in_code(
        "CC=CC(O)=NC=CC", {1: ((0, 3), "STEREOE"), 6: ((5, 8), "STEREOE")}),
}


def _set_double_bonds(mol):
    return sum(1 for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE
               and b.GetStereo() in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ,
                                     Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS))


@pytest.mark.parametrize("case", sorted(_OPEN_NEXT_TO_SET))
def test_double_bond_left_open_next_to_set_ones_never_gets_a_configuration(case):
    """A SMILES writes a double bond's configuration as directions on its single
    bonds, and RDKit's reader sets every double bond with a direction on each side, so
    for some of these Mols every SMILES sets the open bond (an amidine C=N among them,
    which InChI does not see). Such a Mol is declined; one whose SMILES leaves the bond
    open is named as its SMILES. Neither gets a configuration the Mol leaves open."""
    mol = _OPEN_NEXT_TO_SET[case]()
    written = _written(mol)
    holds_it = (Chem.MolToInchi(_mol(written)) == Chem.MolToInchi(mol)
                and _set_double_bonds(_mol(written)) == _set_double_bonds(mol))
    if holds_it:
        _assert_named_as_its_smiles(mol, ("name_compound", "name_tiered",
                                          "name_tiered_best_effort"))
    else:
        with pytest.raises(ValueError, match=f"{_ANOTHER_INCHI}|{_NOT_ITS_STEREO}"):
            name_compound(mol)
        _assert_declined(mol, f"{_ANOTHER_INCHI}|{_NOT_ITS_STEREO}")
    assert holds_it == (case == "triene_open_middle")


def test_imine_tag_on_a_double_bond_rdkit_reads_no_configuration_for_is_declined():
    """A tag on the C=N of hepta-2,5-dien-4-imine, whose carbon carries two arms that
    are alike while their C=C bonds are open: RDKit's SMILES and InChI writers read no
    configuration from it. The check compares the tags and declines it (fail closed)."""
    mol = Chem.AddHs(_mol("CC=CC(=N)C=CC"))
    bond = mol.GetBondWithIdx(3)
    hydrogen = [a.GetIdx() for a in bond.GetEndAtom().GetNeighbors() if a.GetAtomicNum() == 1][0]
    bond.SetStereoAtoms(2, hydrogen)
    bond.SetStereo(Chem.BondStereo.STEREOE)
    assert "/" not in _written(mol)
    _assert_declined(mol, _NOT_ITS_STEREO)


_ALKENYL_CENTRE = "C/C=C/[C@H](/C=C/CC)/C=C\\C"


def _wedged(kind):
    mol = _mol(_ALKENYL_CENTRE)
    if kind == "wedge_mol_bonds":
        AllChem.Compute2DCoords(mol)
        Chem.WedgeMolBonds(mol, mol.GetConformer())
    else:
        mol.GetBondWithIdx(2).SetBondDir(getattr(Chem.BondDir, kind))
    return mol


@pytest.mark.parametrize("kind", ["wedge_mol_bonds", "BEGINDASH", "BEGINWEDGE"])
def test_wedge_on_the_single_bond_next_to_a_double_bond_is_named_as_its_smiles(kind):
    mol = _wedged(kind)
    assert Chem.MolToInchi(mol) == Chem.MolToInchi(_mol(_ALKENYL_CENTRE))
    _assert_named_as_its_smiles(mol, ("name_compound", "name_tiered"))


@pytest.mark.parametrize("smiles,bond,declined", [
    (_ALKENYL_CENTRE, 2, True),      # begin atom C2, of a double bond: unknown for InChI
    ("C[C@H](O)CC", 1, True),        # begin atom C1, a stereocentre: unknown for InChI
    ("C/C=C/C", 0, False),           # begin atom C0, neither: no change
], ids=["double_bond_atom", "stereocentre", "neither"])
def test_wavy_bond_rdkit_reads_two_ways_is_declined(smiles, bond, declined):
    """RDKit's InChI writer reads a wavy (``UNKNOWN``) direction as unknown stereo at
    the begin atom of its bond; RDKit's SMILES writer ignores it."""
    mol = _mol(smiles)
    mol.GetBondWithIdx(bond).SetBondDir(Chem.BondDir.UNKNOWN)
    if declined:
        assert Chem.MolToInchi(mol) != Chem.MolToInchi(_mol(_written(mol)))
        _assert_declined(mol, _ANOTHER_INCHI)
    else:
        _assert_named_as_its_smiles(mol, ("name_compound", "name_tiered"))


def test_e_z_tag_on_an_aromatic_bond_is_declined():
    """A Z tag on an aromatic bond of [10]annulene: RDKit's InChI writer reads it from
    the Kekule form, RDKit's SMILES writer writes none."""
    mol = _mol("C1=CC=CC=CC=CC=C1")
    bond = mol.GetBondWithIdx(0)
    bond.SetStereoAtoms(9, 2)
    bond.SetStereo(Chem.BondStereo.STEREOZ)
    _assert_declined(mol, _ANOTHER_INCHI)


# ---------------------------------------------------------------------------
# Atropisomer tags and chiral tags that are not tetrahedral
# ---------------------------------------------------------------------------

def _atropisomer_next_to_double_bond():
    """An atropisomer tag on the single bond next to a tagged double bond, with
    the directions cleared. RDKit's InChI and SMILES writers read no configuration for
    the double bond, so the Mol must not be named '(3Z)-hex-3-ene'."""
    mol = Chem.AddHs(_mol("CC/C=C\\CC"))
    rw = Chem.RWMol(mol)
    for bond in rw.GetBonds():
        bond.SetBondDir(Chem.BondDir.NONE)
    h3 = [n.GetIdx() for n in rw.GetAtomWithIdx(3).GetNeighbors() if n.GetAtomicNum() == 1][0]
    h4 = [n.GetIdx() for n in rw.GetAtomWithIdx(4).GetNeighbors() if n.GetAtomicNum() == 1][0]
    single = rw.GetBondBetweenAtoms(3, 4)
    single.SetStereoAtoms(h3, h4)
    single.SetStereo(Chem.BondStereo.STEREOATROPCCW)
    return rw.GetMol()


def _atropisomer_amidine(tag):
    mol = _mol("CCOC(=O)C=CC(N)=NC=CC")
    bond = mol.GetBondBetweenAtoms(7, 9)
    bond.SetStereoAtoms(8, 10)
    bond.SetStereo(getattr(Chem.BondStereo, tag))
    mol.GetBondBetweenAtoms(6, 7).SetBondDir(Chem.BondDir.ENDUPRIGHT)
    mol.GetBondBetweenAtoms(9, 10).SetBondDir(Chem.BondDir.ENDUPRIGHT)
    return mol


def _atropisomer_on(smiles, index, atoms):
    mol = _mol(smiles)
    bond = mol.GetBondWithIdx(index)
    if atoms:
        bond.SetStereoAtoms(*atoms)
    bond.SetStereo(Chem.BondStereo.STEREOATROPCW)
    return mol


@pytest.mark.parametrize("build", [
    _atropisomer_next_to_double_bond,
    lambda: _atropisomer_amidine("STEREOATROPCW"),
    lambda: _atropisomer_amidine("STEREOATROPCCW"),
    lambda: _atropisomer_on("CC=CCC", 1, (0, 3)),
    lambda: _atropisomer_on("OC(=O)/C=C/c1ccccc1", 0, None),
], ids=["R5_1_next_to_a_double_bond", "amidine_cw", "amidine_ccw", "alkene",
        "single_bond_no_axis"])
def test_atropisomer_tag_is_declined(build):
    """No SMILES holds an atropisomer configuration, so a Mol with an atropisomer tag
    is declined before its SMILES is written (on a single bond that is no axis, RDKit's
    own MolToSmiles raises)."""
    mol = build()
    _assert_declined(mol, _ATROPISOMER)


def test_atropisomer_next_to_double_bond_is_declined_where_its_smiles_has_none():
    mol = _atropisomer_next_to_double_bond()
    assert "/b" not in Chem.MolToInchi(mol)
    assert name_compound(_written(mol)) == "hex-3-ene"
    with pytest.raises(ValueError, match=_ATROPISOMER):
        name_compound(mol)


_SRC_DIR = os.path.dirname(os.path.dirname(orthonym.__file__))

_IN_OWN_PROCESS = """
import json, logging
logging.disable(logging.CRITICAL)
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from orthonym import Orthonym, classify_limit, name_compound, name_with_tree
BEST_EFFORT = dict(general_fallback=True, general_fallback_unverified=True,
                   allow_aromatic_general=True)
{build}
def outcome(call):
    try:
        return dict(value=call())
    except Exception as exc:
        return dict(raised=type(exc).__name__, value_error=isinstance(exc, ValueError),
                    message=" ".join(str(exc).split()))
def limit(arg):
    found = classify_limit(arg)
    return None if found is None else [found.code, found.message, found.smiles]
calls = dict(
    name_compound=lambda a: name_compound(a),
    name=lambda a: Orthonym().name(a),
    name_tiered=lambda a: Orthonym().name_tiered(a),
    name_tiered_best_effort=lambda a: Orthonym(**BEST_EFFORT).name_tiered(a),
    name_with_confidence=lambda a: Orthonym().name_with_confidence(a)["name"],
    name_with_tree=lambda a: name_with_tree(a).name,
    classify_limit=limit)
before = m.ToBinary(Chem.PropertyPickleOptions.AllProps)
result = dict(mol={{label: outcome(lambda: call(m)) for label, call in calls.items()}})
result["unchanged"] = m.ToBinary(Chem.PropertyPickleOptions.AllProps) == before
if {write}:
    written = Chem.MolToSmiles(Chem.Mol(m))
    result["written"] = written
    result["str"] = {{label: outcome(lambda: call(written)) for label, call in calls.items()}}
print("RESULT " + json.dumps(result))
"""


def _in_own_process(build, write=True):
    """Build the Mol ``m`` with the code ``build`` and make every public call on it in
    a fresh interpreter, so a crash inside RDKit fails the test instead of ending the
    pytest process. With ``write``, the same calls on RDKit's SMILES of ``m``, made in
    that interpreter after them, are returned too."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        p for p in (_SRC_DIR, os.environ.get("PYTHONPATH", "")) if p))
    proc = subprocess.run([sys.executable, "-c",
                           _IN_OWN_PROCESS.format(build=build, write=write)],
                          capture_output=True, text=True, env=env, timeout=900)
    assert proc.returncode == 0, (proc.returncode, proc.stderr[-3000:])
    lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
    assert len(lines) == 1, proc.stdout[-3000:]
    return json.loads(lines[0][len("RESULT "):])


def _declined_in_own_process(out, message):
    assert out["unchanged"]
    for label, got in out["mol"].items():
        if label == "classify_limit":
            assert got == {"value": None}, got
        else:
            assert got.get("raised") == "InvalidInputError" and got["value_error"], (label, got)
            assert message in got["message"], (label, got)


#: A Mol RDKit's own ``MolToSmiles`` crashes the process on (SIGSEGV): a CHI_OTHER
#: tag on atom 4 of a 23-atom amide, found by fuzzing, pickled.
_CHI_OTHER_CRASH = (
    "776t3gAAAAAQAAAAAwAAAAAAAAAXAAAAGQAAAIABBkAoAAAAAwQGQGgAAAADAwEGACgAAAADBAZAaAAAAAMDAQZA"
    "LAAAAAMDBAcAaAAAAAMCAQZAaAAAAAMDAQYAYAAAAAICBkBoAAAAAwMBBkAoAAAAAwQGQGgAAAADAwEGQGgAAAAD"
    "AwEGQGgAAAADAwEGQGgAAAADAwEHQCgAAAADAwZAKAAAAAMEBkBoAAAAAwMBCAAoAAAAAwIGQGgAAAADAwEGQGgA"
    "AAADAwEGQGgAAAADAwEIACgAAAADAgZAaAAAAAMDAQsVAigCAgUgBQcABwQABBRoDBQGaAwGAWgMAQNoDAMOaAwC"
    "ACAAFmgMFg1oDA0JaAwJESARDyAPCmgMCghoDAgLaAwLDGgMDBNoDAkQaAwQEmgMDgRoDBIAaAwTD2gMRAMAAAAG"
    "FAYBAw4EBhYNCRASAAYKCAsMEw8XBAAAAAAAAAAW")

_CHI_OTHER = {
    "cyclohexane": "m = Chem.MolFromSmiles('C1CCCCC1'); "
                   "m.GetAtomWithIdx(2).SetChiralTag(Chem.ChiralType.CHI_OTHER)",
    "cyclooctene_any": "m = Chem.MolFromSmiles('C1=CCCCCCC1'); "
                       "m.GetBondWithIdx(0).SetStereo(Chem.BondStereo.STEREOANY); "
                       "m.GetAtomWithIdx(6).SetChiralTag(Chem.ChiralType.CHI_OTHER)",
    "unspecified_bond": "m = Chem.MolFromSmiles('C[N@+]1(C)CC1'); "
                        "m.GetAtomWithIdx(4).SetChiralTag(Chem.ChiralType.CHI_OTHER); "
                        "m.GetBondWithIdx(0).SetBondType(Chem.BondType.UNSPECIFIED)",
    "writer_crash": f"import base64; m = Chem.Mol(base64.b64decode({_CHI_OTHER_CRASH!r}))",
}


@pytest.mark.parametrize("case", sorted(_CHI_OTHER))
def test_chi_other_tag_is_declined_without_a_crash(case):
    """A ``CHI_OTHER`` chiral tag set in code. RDKit's forced stereo assignment crashed
    the process on such Mols, and RDKit's
    own ``MolToSmiles`` crashes on the last one; the check cannot compare the tag, so
    each call declines the Mol before RDKit writes it."""
    out = _in_own_process(_CHI_OTHER[case], write=False)
    _declined_in_own_process(out, _NOT_TETRAHEDRAL)


@pytest.mark.parametrize("cxsmiles", [
    "C[C@H](N)C(=O)O |&1:1|",           # AND: racemic
    "C[C@H](N)C(=O)O |o1:1|",           # OR: one enantiomer, unknown which
    "C[C@H](O)[C@@H](C)Cl |&1:1,3|",
])
def test_and_or_stereo_group_is_declined(cxsmiles):
    """A SMILES holds no stereo group, so RDKit's SMILES of such a Mol states one
    enantiomer; the Mol is declined instead of named as that enantiomer."""
    mol = Chem.MolFromSmiles(cxsmiles)
    assert any(g.GetGroupType() != Chem.StereoGroupType.STEREO_ABSOLUTE
               for g in mol.GetStereoGroups())
    _assert_declined(mol, "stereo group")


def test_absolute_stereo_group_is_named_as_its_smiles():
    _assert_named_as_its_smiles(Chem.MolFromSmiles("C[C@H](N)C(=O)O |a:1|"))


@pytest.mark.parametrize("smiles", [
    "[Fe+3].[Cl-].[Cl-].[Cl-]",
    "[Co+2].CC(=O)[O-].CC(=O)[O-]",
    "[Cu+2].[O-]S(=O)(=O)[O-]",
])
def test_metal_salt_from_a_maestro_file_is_named_as_its_smiles(smiles):
    """RDKit's Maestro reader gives a metal ion no radical electrons, its SMILES reader
    gives the same bracket ion some; that convention does not decline the Mol."""
    import io

    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)
    text = io.StringIO()
    writer = Chem.MaeWriter(text)
    writer.write(mol)
    writer.close()
    read = next(iter(Chem.MaeMolSupplier(io.BytesIO(text.getvalue().encode()))))
    _assert_named_as_its_smiles(read)


def test_pdb_tag_on_a_centre_stereogenic_only_in_another_diastereomer():
    """C3 of (2R,4R)-pentane-2,3,4-triol is no stereocentre (it is one in (2R,4S));
    RDKit's PDB reader tags it and its SMILES writer drops the tag."""
    _assert_named_as_its_smiles(_pdb_mol("C[C@@H](O)C(O)[C@@H](C)O", False))


def _pdb_mol(smiles, keep_hs):
    from rdkit.Chem import AllChem
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    if not keep_hs:
        mol = Chem.RemoveHs(mol)
    return Chem.MolFromPDBBlock(Chem.MolToPDBBlock(mol), removeHs=not keep_hs)


@pytest.mark.parametrize("smiles,keep_hs", [
    ("CC(C)c1ccc(N(C)C(=O)CCN)cc1", False),         # isopropyl CH tagged
    ("COc1cccc2c(CCCO)coc12", True),               # CH2 groups tagged
    ("C=C[C@@H](NC(=O)OC)C(C)(C)C(F)(F)F", True),  # a real centre beside tagged ones
    ("CC(=O)N(C(C)C)C1(C(=O)O)CN2CCC1CC2", False),   # quinuclidine bridgehead tagged
    ("CC(=O)N(C(C)C)C1(C(=O)O)CN2CCC1CC2", True),
])
def test_pdb_tags_on_atoms_that_cannot_be_stereocentres_are_ignored(smiles, keep_hs):
    """RDKit's PDB reader tags atoms from 3D coordinates, also atoms that cannot be
    stereocentres; its SMILES writer drops those tags, and the check ignores them, so
    the Mol gets the result of its SMILES."""
    _assert_named_as_its_smiles(_pdb_mol(smiles, keep_hs))


@pytest.mark.parametrize("smiles", [
    "[NH3][Pt@SP1]([NH3])(Cl)Cl",      # cisplatin
    "[NH3][Pt@SP2]([NH3])(Cl)Cl",      # transplatin
    "C[Pt@SP1](Cl)(Br)C",
    "F[S@OH1](F)(F)(F)(F)Cl",          # octahedral
    "N[Co@OH1](N)(N)(N)(Cl)Cl",
    "Cl[P@TB1](Cl)(Cl)(Cl)Cl",         # trigonal bipyramidal
])
def test_square_planar_octahedral_and_trigonal_bipyramidal_tags_name_as_the_smiles(smiles):
    """A SMILES holds these configurations and no name states them, so a Mol with such
    a tag (cisplatin as RDKit reads it, for one) gets the result of its SMILES."""
    _assert_named_as_its_smiles(_mol(smiles))


@pytest.mark.parametrize("build,smiles", [
    ("m = Chem.ReplaceSubstructs(Chem.MolFromSmiles('Cl/C=C/C'), "
     "Chem.MolFromSmiles('Cl'), Chem.MolFromSmiles('F'))[0]", "CC=CF"),
    ("m = Chem.MolFromSmiles('CC=CC'); m.GetBondWithIdx(1).SetStereo(Chem.BondStereo.STEREOE)",
     "CC=CC"),
    ("m = Chem.MolFromSmiles('CC=CC(=O)O'); "
     "m.GetBondWithIdx(1).SetStereo(Chem.BondStereo.STEREOZ)", "CC=CC(=O)O"),
    ("m = Chem.MolFromSmiles('C/C=C/C=CC'); "
     "m.GetBondWithIdx(3).SetStereo(Chem.BondStereo.STEREOZ)", "CC=C/C=C/C"),
], ids=["replace_substructs", "E_but_2_ene", "Z_but_2_enoic_acid", "diene_second_bond"])
def test_double_bond_tag_without_stereo_atoms_is_named_as_its_smiles(build, smiles):
    """RDKit's editing functions (``ReplaceSubstructs`` here) leave an E/Z tag without
    its stereo atoms, which fixes no configuration; RDKit's direction step can crash
    the process on it. Each call returns the result of its SMILES."""
    out = _in_own_process(build)
    assert out["written"] == smiles
    assert out["mol"] == out["str"]
    assert out["unchanged"]


# ---------------------------------------------------------------------------
# Lone-pair stereocentres
# ---------------------------------------------------------------------------

def test_lone_pair_centre_written_first_is_named_as_the_mol_holds_it():
    """RDKit reads '[S@@](C)(=O)c1ccccc1' as the (R) sulfoxide (its CIP labeller says
    so); the string is named as the standard reading of SMILES reads it, (S). The Mol
    holds RDKit's reading, and RDKit's SMILES of it, 'C[S@@](=O)c1ccccc1', is (R) for
    both readings: the Mol is named (R)."""
    from rdkit.Chem import rdCIPLabeler
    mol = _mol("[S@@](C)(=O)c1ccccc1")
    labelled = Chem.Mol(mol)
    rdCIPLabeler.AssignCIPLabels(labelled)
    assert labelled.GetAtomWithIdx(0).GetProp("_CIPCode") == "R"
    _assert_named_as_its_smiles(mol, ("name_compound", "name_tiered"))
    assert name_compound(mol) == "(R)-(methanesulfinyl)benzene"
    assert name_compound("[S@@](C)(=O)c1ccccc1") == "(S)-(methanesulfinyl)benzene"


def _opsin_stdinchi(name):
    """OPSIN's own standard InChI of ``name`` (no RDKit reading of a SMILES)."""
    from orthonym.jvm_flags import JVM_HYGIENE_FLAGS
    from tests.support.jars import jar_or_skip
    proc = subprocess.run(["java", *JVM_HYGIENE_FLAGS, "-jar", jar_or_skip("opsin"),
                           "-ostdinchi"], input=name + "\n", capture_output=True,
                          text=True, timeout=300)
    return proc.stdout.strip()


@pytest.mark.parametrize("smiles,str_name", [
    ("CO[P@@]1OC[C@H](C)O1", "(2R,4S)-2-methoxy-4-methyl-1,3,2-dioxaphospholane"),
    ("COC(=O)[C@H]1C[N@]1Cc1cc(-c2cccc3c2OC(F)(F)O3)ccc1F", None),
], ids=["phosphorus_ring_closure", "aziridine"])
def test_lone_pair_centre_rdkit_reads_otherwise_in_its_smiles_is_declined(smiles, str_name):
    """RDKit reads a P(III) or aziridine N centre that carries a ring-closure digit
    unlike the standard reading of SMILES, and its SMILES of the Mol writes the centre
    so again. A SMILES string is named by the standard reading, so the SMILES would be
    named as the other configuration than the Mol holds: the Mol of the phosphorus row
    is (2S,4S) (its InChI is OPSIN's InChI of that name), the name of its SMILES
    (2R,4S). Declined."""
    mol = _mol(smiles)
    assert _written(mol) == smiles
    if str_name is not None:
        assert name_compound(smiles) == str_name
        assert Chem.MolToInchi(mol) != _opsin_stdinchi(str_name)
        assert Chem.MolToInchi(mol) == _opsin_stdinchi(str_name.replace("2R,4S", "2S,4S"))
    _assert_declined(mol, _LONE_PAIR)


# ---------------------------------------------------------------------------
# Other molecules RDKit reads from the Mol and from its SMILES
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smarts", ["C/C=C/C", "[CH3]/[CH]=[CH]/[CH3]", "C[C@H](F)Cl", "CC=CC"])
def test_query_mol_read_back_as_another_molecule_is_declined(smarts):
    """A query Mol (``Chem.MolFromSmarts``): RDKit's sanitization gives its query atoms
    other hydrogen counts than RDKit's reading of its SMILES."""
    _assert_declined(Chem.MolFromSmarts(smarts), _ANOTHER_MOLECULE)


def test_query_mol_read_back_as_the_same_molecule_is_named_as_its_smiles():
    mol = Chem.MolFromSmarts("O=C=O")
    _assert_named_as_its_smiles(mol, ("name_compound", "name_tiered"))
    assert name_compound(mol) == "carbon dioxide"


@pytest.mark.parametrize("smiles", ["[IH3]", "[IH5]", "[IH3]1C=CC=C1"])
def test_hypervalent_iodine_with_hydrogen_atoms_is_declined(smiles):
    """RDKit's InChI writer gives the H-kept Mol (``AddHs``) another formula
    than RDKit's reading of its SMILES, which removes the H atoms (H3I against HI for
    [IH3]). Declined; the Mol without hydrogen atoms is named as its SMILES."""
    with_h = Chem.AddHs(_mol(smiles))
    assert Chem.MolToInchi(with_h) != Chem.MolToInchi(_mol(_written(with_h)))
    _assert_declined(with_h, _ANOTHER_INCHI)
    _assert_named_as_its_smiles(_mol(smiles), ("name_compound", "name_tiered"))


def test_3d_mol_block_whose_smiles_rdkit_writes_as_another_stereoisomer_is_declined():
    """RDKit's SMILES of the 3D Mol with its
    hydrogen atoms is another stereoisomer than the Mol (RDKit's InChI of the two
    differs at one C=C); the Mol is declined, never named as that SMILES."""
    smiles = "CCN(CC)C/C1=C\\CC(C)(C)/C=C/C(=O)/C(C)=C/CC1"
    with_h = Chem.AddHs(_mol(smiles))
    assert AllChem.EmbedMolecule(with_h, randomSeed=7) == 0
    mol = Chem.MolFromMolBlock(Chem.MolToMolBlock(with_h), removeHs=False)
    assert Chem.MolToInchi(mol) == Chem.MolToInchi(_mol(smiles))
    assert Chem.MolToInchi(_mol(_written(mol))) != Chem.MolToInchi(mol)
    with pytest.raises(ValueError, match=_ANOTHER_INCHI):
        name_compound(mol)


# ``rotated``: the writer's own atom order maps the Mol onto another stereoisomer.
@pytest.mark.parametrize("smiles", ["C[C@H]1CC[C@@H](O)CC1", "C1CC[C@H]2CCCC[C@@H]2C1"])
def test_symmetric_molecule_written_as_its_symmetric_equivalent_is_named(smiles):
    """RDKit's SMILES writer can write a symmetric molecule's configurations as those
    of a symmetric equivalent (the ring read the other way round): through the order
    in which it wrote the atoms, the Mol's centres have the other configuration in the
    reading. The check matches the atoms through the molecule's symmetry."""
    mol = _mol(smiles)
    written = Chem.Mol(mol)
    key = Chem.MolToSmiles(written)
    order = input_mol._output_order(written)
    params = Chem.SmilesParserParams()
    params.removeHs = False
    reading = Chem.MolFromSmiles(key, params)
    matching = [0] * len(order)
    for position, index in enumerate(order):
        matching[index] = position
    assert (input_mol._stereo_units(mol, matching.__getitem__)
            != input_mol._stereo_units(reading, lambda index: index))
    assert input_mol.smiles_for(mol) == key
    _assert_named_as_its_smiles(mol, ("name_compound", "name_tiered"))


# ---------------------------------------------------------------------------
# The caller's Mol is not modified
# ---------------------------------------------------------------------------

def _props(obj):
    out = {}
    for key, value in obj.GetPropsAsDict(includePrivate=True, includeComputed=True).items():
        if not isinstance(value, (str, bytes, int, float, bool)):
            value = list(value)
        out[key] = value
    return out


def _snapshot(mol):
    return (_props(mol), [_props(a) for a in mol.GetAtoms()],
            [_props(b) for b in mol.GetBonds()],
            [(int(a.GetChiralTag()), a.GetNoImplicit(), a.GetNumExplicitHs())
             for a in mol.GetAtoms()],
            [(str(b.GetStereo()), list(b.GetStereoAtoms()), str(b.GetBondDir()))
             for b in mol.GetBonds()],
            mol.GetNumConformers(), mol.GetNumAtoms(), mol.GetNumBonds(),
            Chem.MolToMolBlock(Chem.Mol(mol)))


@pytest.mark.parametrize("make", ["plain", "addhs", "3d", "unsanitized", "declined"])
def test_callers_mol_is_unchanged(make):
    smiles = "C[C@@H](O)[C@H](N)C(=O)O"
    if make == "plain":
        mol = _mol(smiles)
    elif make == "addhs":
        mol = Chem.AddHs(_mol(smiles))
    elif make == "unsanitized":
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
    elif make == "declined":
        mol = _atropisomer_next_to_double_bond()
    else:
        mol = Chem.AddHs(_mol(smiles))
        assert AllChem.EmbedMolecule(mol, randomSeed=7) == 0
    mol.SetProp("_Name", "caller")
    before = _snapshot(mol)
    for call in CALLS.values():
        try:
            call(mol)
        except ValueError:
            assert make == "declined"
    assert _snapshot(mol) == before


# ---------------------------------------------------------------------------
# Input rules
# ---------------------------------------------------------------------------

_NAMING = sorted(set(CALLS) - {"classify_limit"})


@pytest.mark.parametrize("entry", _NAMING)
def test_none_raises_value_error(entry):
    with pytest.raises(ValueError, match="Invalid SMILES: None"):
        CALLS[entry](None)


def test_classify_limit_none_is_none():
    # classify_limit gives None for a SMILES RDKit cannot read ("a reading error,
    # not a scope limit"); None is that case.
    assert classify_limit(None) is None


@pytest.mark.parametrize("entry", sorted(CALLS))
@pytest.mark.parametrize("bad", [5, ["CCO"], b"CCO", 1.5])
def test_other_types_raise_type_error_naming_the_accepted_types(entry, bad):
    with pytest.raises(TypeError, match=r"SMILES string or an RDKit Chem\.Mol"):
        CALLS[entry](bad)


def test_mol_rdkit_cannot_sanitize_raises_value_error():
    pentavalent = Chem.MolFromSmiles("CC(C)(C)(C)(C)C", sanitize=False)
    _assert_declined(pentavalent, "Invalid molecule")
    assert classify_limit("CC(C)(C)(C)(C)C") is None


def test_empty_mol_behaves_like_the_empty_string():
    empty = Chem.Mol()
    assert input_mol.smiles_for(empty) == ""
    for label, call in CALLS.items():
        assert call(empty) == call(""), label


def test_str_arguments_keep_their_exceptions():
    with pytest.raises(ValueError, match="Invalid SMILES: C1CC"):
        name_compound("C1CC")
    with pytest.raises(ValueError, match="Invalid SMILES: C1CC"):
        Orthonym().name_tiered("C1CC")


def test_str_call_runs_no_new_code(monkeypatch):
    """A str argument never reaches ``orthonym.input_mol``; a Mol reaches it once per
    public call (the positive control)."""
    calls = []
    monkeypatch.setattr(input_mol, "handles", lambda arg: calls.append(("handles", arg)))
    monkeypatch.setattr(input_mol, "smiles_for", lambda arg: calls.append(("smiles_for", arg)))
    for smiles in ("C[C@H](N)C(=O)O", "CC(C)C[C@H](NC(=O)[C@@H](N)Cc1ccccc1)C(=O)O",
                   "CC(=O)[O-].[Na+]"):
        for call in CALLS.values():
            call(smiles)
    assert calls == []
    monkeypatch.undo()
    seen = []
    real = input_mol.smiles_for
    monkeypatch.setattr(input_mol, "smiles_for", lambda arg: seen.append(arg) or real(arg))
    mol = _mol("CCO")
    for call in CALLS.values():
        call(mol)
    assert len(seen) == len(CALLS) and all(arg is mol for arg in seen)


def test_package_exports_unchanged():
    assert orthonym.__all__ == ["name_compound", "name_with_tree", "Orthonym",
                                "NameTreeNode", "NamingResult", "OrthonymLimitError",
                                "classify_limit", "__version__"]


def test_python_api_guide_example():
    # guide/reference/python-api.md, "Input: a SMILES string or an RDKit molecule"
    assert name_compound(Chem.MolFromSmiles("C[C@H](O)CC")) == "(2S)-butan-2-ol"
