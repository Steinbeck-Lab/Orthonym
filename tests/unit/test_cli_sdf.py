"""``orthonym --sdf FILE`` names every record of a MOL or SDF file as an RDKit Mol is named (to-do item 6,
2026-11-10): through its proven SMILES, or its CXSMILES for an AND or OR stereo group, one output line per
record in file order. The stereo is the file's own (wedges, the coordinates of a double bond, an 'either' bond
left open); the coordinates are dropped before naming, so nothing is perceived from them again."""
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from orthonym import Orthonym
from orthonym.cli import main

pytestmark = pytest.mark.opsin_gate

BEST_EFFORT = dict(general_fallback=True, general_fallback_unverified=True, allow_aromatic_general=True)


def _block_2d(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)
    return Chem.MolToMolBlock(mol)


def _either_double_bond_block() -> str:
    """But-2-ene with its double bond marked 'either' (MDL bond stereo 3): the configuration is open."""
    mol = Chem.MolFromSmiles("C/C=C/C")
    AllChem.Compute2DCoords(mol)
    mol.GetBondWithIdx(1).SetStereo(Chem.BondStereo.STEREOANY)
    block = Chem.MolToMolBlock(mol)
    lines = block.splitlines()
    # the bond block: atoms 2-3 double; set its stereo field (columns 10-12) to 3
    for i, line in enumerate(lines):
        if line.startswith("  2  3  2"):
            lines[i] = line[:9] + "  3" + line[12:]
    return "\n".join(lines) + "\n"


def _racemic_v3000_block() -> str:
    mol = Chem.MolFromSmiles("C[C@H](N)C(=O)O |&1:1|")
    AllChem.Compute2DCoords(mol)
    Chem.WedgeMolBonds(mol, mol.GetConformer())
    return Chem.MolToV3KMolBlock(mol)


@pytest.fixture
def sdf(tmp_path):
    records = [_block_2d("C[C@@H](N)C(=O)O"), _either_double_bond_block(), _racemic_v3000_block(),
               "not a mol block\n"]
    path = tmp_path / "in.sdf"
    path.write_text("$$$$\n".join(records) + "$$$$\n")
    return path


def test_every_record_gets_its_line_in_file_order(sdf, capsys):
    status = main(["--sdf", str(sdf), "--emit-tier", "best-effort"])
    out, err = capsys.readouterr()
    lines = out.splitlines()
    assert len(lines) == 4, out
    smiles, names = zip(*(line.split("\t") if "\t" in line else (line, "") for line in lines))
    # the wedge of the 2D record: 'C[C@@H](N)C(=O)O' is L-alanine 'N[C@@H](C)C(=O)O' (2S) with N and CH3
    # swapped, one transposition, so (2R)
    assert names[0] == "(2R)-2-aminopropanoic acid"
    assert names[1] == "but-2-ene"                          # the 'either' bond stays open
    assert names[2] == "rac-(2R)-2-aminopropanoic acid"     # the V3000 AND collection
    assert smiles[3] == "<record 4>" and names[3].startswith("ERROR:")   # as an unreadable --batch line
    assert "record 4: RDKit cannot read it" in err
    assert status == 1


def test_a_record_is_named_as_its_string(sdf, capsys):
    main(["--sdf", str(sdf), "--emit-tier", "best-effort"])
    out, _ = capsys.readouterr()
    namer = Orthonym(**BEST_EFFORT)
    for line in out.splitlines():
        smiles, name = line.split("\t")
        if not smiles.startswith("<record"):
            assert namer.name_tiered(smiles)["name"] == name


def test_coordinates_of_a_double_bond_give_its_configuration(tmp_path, capsys):
    """A double bond drawn without an 'either' mark has the configuration its coordinates show (the MDL
    convention), not a configuration perceived again later."""
    path = tmp_path / "z.mol"
    path.write_text(_block_2d("C/C=C\\C"))
    main(["--sdf", str(path), "--emit-tier", "best-effort"])
    out, _ = capsys.readouterr()
    assert out.split("\t")[1].strip() == "(2Z)-but-2-ene"


def test_sdf_with_batch_is_a_usage_error(sdf, tmp_path):
    other = tmp_path / "x.smi"
    other.write_text("CCO\n")
    with pytest.raises(SystemExit) as raised:
        main(["--sdf", str(sdf), "--batch", str(other)])
    assert raised.value.code == 2


def test_missing_file_is_reported(tmp_path, capsys):
    assert main(["--sdf", str(tmp_path / "missing.sdf")]) == 1
    assert "not found" in capsys.readouterr().err


def test_a_v2000_chiral_flag_is_not_read(tmp_path, capsys):
    """Chiral flag 0 (RDKit writes it for every V2000 block): the configuration is absolute, as in the standard
    InChI; a racemate is stated by a V3000 AND collection."""
    block = _block_2d("C[C@H](O)CC")
    assert block.splitlines()[3][12:15].strip() == "0"
    path = tmp_path / "flag0.mol"
    path.write_text(block)
    main(["--sdf", str(path), "--emit-tier", "best-effort"])
    assert capsys.readouterr().out.split("\t")[1].strip() == "(2S)-butan-2-ol"


def test_a_record_with_no_atoms_gets_an_empty_line_and_a_message(tmp_path, capsys):
    path = tmp_path / "empty.sdf"
    path.write_text(Chem.MolToMolBlock(Chem.Mol()) + "$$$$\n" + _block_2d("CCO") + "$$$$\n")
    main(["--sdf", str(path)])
    out, err = capsys.readouterr()
    assert out.splitlines()[0] == "" and out.splitlines()[1] == "CCO\tethanol"
    assert "record 1: it has no atoms" in err
