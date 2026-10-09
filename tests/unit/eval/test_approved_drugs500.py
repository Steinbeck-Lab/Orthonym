"""The approved-drug benchmark curriculum (eval/curricula/approved_drugs500.json).

Format and integrity only: the file is the curriculum the drug benchmark scorer reads
(eval/drug_bench/score.py) and eval/harness.py runs with --split-file. Source, licence and
filters: eval/drug_bench/SOURCE.md. No a holdout split file is read here.
"""
import hashlib
import json
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import inchi

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[3]
SPLIT = ROOT / "eval" / "curricula" / "approved_drugs500.json"
ELEMENTS = {"H", "B", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"}


def _split():
    return json.loads(SPLIT.read_text())


def _keys(smiles):
    return {inchi.MolToInchiKey(Chem.MolFromSmiles(s)) for s in smiles}


def test_the_curriculum_has_the_harness_format():
    d = _split()
    assert d["split"] == "approved_drugs500"
    assert d["n"] == len(d["smiles"]) == len(d["meta"]) == 500
    assert d["derived"] is True and d["holdout"] is False
    assert d["sha256"] == hashlib.sha256("\n".join(d["smiles"]).encode()).hexdigest()
    assert "CC0 1.0" in d["licence"]
    assert not (ROOT / "eval" / "splits" / "approved_drugs500.json").exists()


def test_every_row_is_one_parsed_molecule_inside_the_prefilter():
    for smi in _split()["smiles"]:
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None, smi
        assert len(Chem.GetMolFrags(mol)) == 1, smi
        assert mol.GetNumHeavyAtoms() <= 100, smi
        assert {a.GetSymbol() for a in mol.GetAtoms()} <= ELEMENTS, smi
        assert any(a.GetSymbol() == "C" for a in mol.GetAtoms()), smi
        assert not any(a.GetIsotope() or a.GetNumRadicalElectrons() for a in mol.GetAtoms()), smi


def test_rows_are_distinct_molecules_and_the_meta_keys_agree():
    d = _split()
    keys = [inchi.MolToInchiKey(Chem.MolFromSmiles(s)) for s in d["smiles"]]
    assert len(set(keys)) == 500
    assert keys == [m["inchikey"] for m in d["meta"]]


def test_the_target_drug_is_in_the_curriculum():
    names = [m["name"] for m in _split()["meta"]]
    assert "NILOTINIB" in names


def test_overlap_with_the_development_sets_is_the_documented_one():
    mine = _keys(_split()["smiles"])
    overlap = {}
    for rel in ("splits/dev500.json", "splits/dev2000.json", "curricula/milestone1500.json",
                "curricula/pubchem2000_druglike.json"):
        other = set()
        for s in json.loads((ROOT / "eval" / rel).read_text())["smiles"]:
            m = Chem.MolFromSmiles(s)
            if m is not None:
                other.add(inchi.MolToInchiKey(m))
        overlap[rel] = len(mine & other)
    assert overlap == {"splits/dev500.json": 0, "splits/dev2000.json": 9,
                       "curricula/milestone1500.json": 1, "curricula/pubchem2000_druglike.json": 0}
