"""The approved-drug benchmark scorer (eval/drug_bench): correctness and name quality.

The scorer imports no engine code: the principal characteristic group is read from the
structure by the class order (Table 4.1, the Blue Book-18192), the suffix from the
name's last word, and the forms a chemist rejects on sight by the detector of
internal notes section 8 (eval/drug_bench/forms.py).
"""
import json
import sys
from pathlib import Path

import pytest
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eval"))

from drug_bench import features, forms, score  # noqa: E402

NILOTINIB = "CC1=C(C=C(C(=O)NC2=CC(=CC(=C2)C(F)(F)F)N2C=NC(=C2)C)C=C1)NC1=NC=CC(=N1)C=1C=NC=CC1"
VENADAPARIB = "C1(CC1)NCC1CN(C1)C(=O)C=1C=C(C=CC1F)CC1=NNC(C2=CC=CC=C12)=O"
SORAFENIB = "CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1"
NILOTINIB_PIN = ("4-methyl-N-[3-(4-methyl-1H-imidazol-1-yl)-5-(trifluoromethyl)phenyl]-3-"
                 "{[4-(pyridin-3-yl)pyrimidin-2-yl]amino}benzamide")
NILOTINIB_BE_MAIN = ("2-{[2-methyl-5-({[3-(4-methyl-1H-imidazol-1-yl)-5-(trifluoromethyl)phenyl]"
                     "amino}-oxomethyl)phenyl]amino}-4-(pyridin-3-yl)pyrimidine")
NILOTINIB_VALID_MAIN = ("2-[(2-methyl-5-{2-[3-(4-methyl-1,3-diazacyclopenta-2,4-dien-1-yl)-5-"
                        "(1,1,1-trifluoromethyl)cyclohexa-1,3,5-trien-1-yl]-1-oxo-2-azaethyl}"
                        "phenyl)amino]-4-(pyridin-3-yl)pyrimidine")


@pytest.mark.parametrize("smiles,kind", [
    (NILOTINIB, "amide/N-aryl amide"),        # an acyclic -CO-NH-Ar: class 11 (:18184)
    (VENADAPARIB, "ketone"),                  # N-acyl ring N: pseudoketone,:33127)
    (SORAFENIB, "amide/amide"),               # carboxamide before urea,:33386)
    ("NC(=O)c1ccccc1", "amide/amide"),
    ("CC(=O)O", "acid"),
    ("CCOC(C)=O", "ester"),
    ("CCO", "hydroxy"),
    ("CCN", "amine"),
    ("O=C1CCCCC1", "ketone"),
])
def test_the_principal_class_follows_the_book(smiles, kind):
    pc, tags, _found = features.principal_class(Chem.MolFromSmiles(smiles))
    assert features.amide_kind(pc, tags) == kind


@pytest.mark.parametrize("name,cls", [
    (NILOTINIB_PIN, "amide"),
    (NILOTINIB_BE_MAIN, "none"),
    ("propan-2-ol", "hydroxy"),
    ("ethyl acetate", "ester"),
    ("acetic acid", "acid"),
    ("N-methylpyridin-3-amine", "amine"),
    ("cyclohexan-1-one", "ketone"),
    ("2-hydroxy-N-(5-nitro-1,3-thiazol-2-yl)benzamide", "amide"),
])
def test_the_suffix_class_is_read_from_the_name(name, cls):
    assert features.name_suffix_class(name) == cls


def test_suffix_ok():
    assert features.suffix_ok("amide", "amide") is True
    assert features.suffix_ok("amide", "none") is False
    assert features.suffix_ok("amide", "abstain") is False
    assert features.suffix_ok("amide", "special_parent") is None


@pytest.mark.parametrize("name,tags", [
    (NILOTINIB_PIN, []),
    (NILOTINIB_BE_MAIN, ["OXOM"]),
    (NILOTINIB_VALID_MAIN, ["ACH", "BENZ", "MONO"]),
    ("2-{[3-(1-{3-[2-(cyclopropan-1-yl)-2-azaethyl]-1-azacyclobutan-1-yl}-2-oxaeth-1-en-1-yl)-4-"
     "fluorophenyl]methyl}-5-oxo-3,4-diazabicyclo[4.4.0]deca-1(10),2,6,8-tetraene",
     ["ACH", "ANYL", "MONO", "VBF"]),
    ("(3R,5R)-7-{2-(4-fluorophenyl)-4-[oxo-(phenylamino)methyl]-3-phenyl-5-(propan-2-yl)-1H-pyrrol-"
     "1-yl}-3,5-dihydroxyheptanoic acid", ["OXOP"]),
    ("[2-butyl-4-chloro-1-({4-[2-(2H-tetrazol-5-yl)phenyl]phenyl}methyl)-1H-imidazol-5-yl]methanol",
     ["BIPH"]),
])
def test_the_form_detector(name, tags):
    assert sorted(forms.detect(name)) == tags


def test_carrier_and_phane_screen_of_nilotinib():
    mol = Chem.MolFromSmiles(NILOTINIB)
    pc = features.principal_class(mol)[0]
    # the amide's acyl ring (the methylbenzene) is not the senior ring system: decides
    assert features.carrier_feature(mol, pc)[0] == "PCG on junior ring"
    # five ring systems on an eight-node path, but every phane parent carries 0 amides
    assert features.phane_screen(mol, pc)[:2] == (True, "S fails")


@pytest.mark.parametrize("name,n", [
    ("benzene-1,4-diamine", 2),
    ("N-(4-aminophenyl)pyrimidin-2-amine", 1),
    ("propanedioic acid", 2),
    ("propane-1,2,3-triol", 3),
    ("cyclohexane-1,2,4,5-tetrone", 4),
    ("dimethyl benzene-1,4-dicarboxylate", 2),
    ("4,4'-oxydibenzoic acid", 1),       # the multiplier belongs to the multiplicative parent
    ("pentan-1-ol", 1),
    (NILOTINIB_PIN, 1),
    ("6-(piperidin-1-yl)pyrimidine-2,4-diamine 3-oxide", 2),   # minoxidil: additive oxide
])
def test_the_suffix_multiplicity_is_read_from_the_name(name, n):
    assert features.name_suffix_multiplicity(name) == n


@pytest.mark.parametrize("smiles,cap,assembly", [
    # (:18875): the benzene ring carries both amines; cf. 'N1-(4-aminophenyl)-N4-
    # phenylbenzene-1,4-diamine (PIN)' (:26404)
    ("Nc1ccc(Nc2ncccn2)cc1", 2, False),
    (NILOTINIB, 1, False),
    # the amide's ring is one ring of a ring assembly,:15542;:49805)
    ("O=C(Nc1ccccc1)c1ccc(cc1)-c1ccccc1", 1, True),
    ("O=C(Nc1ccc(cc1)-c1ccccc1)c1ccccc1", 1, False),
    # a ring assembly is one carrier: both acids are its suffixes
    ("OC(=O)c1ccc(cc1)-c1ccc(cc1)C(O)=O", 2, True),
    # the senior kind only: a carboxylic acid before a sulfonic acid, a carboxamide before a
    # sulfonamide
    ("OC(=O)c1ccc(cc1)S(O)(=O)=O", 1, False),
    ("NC(=O)c1ccc(cc1)S(N)(=O)=O", 1, False),
    ("CC(O)CO", 0, False),                # chains are not modelled
    # thiopental: the two C=O are the ketones cited as suffixes, the C=S a 'sulfanylidene' prefix
    ("CCCC(C)C1(CC)C(=O)NC(=S)NC1=O", 2, False),
    # a pyridine and a piperidine are one ring assembly, named with hydro prefixes
    #:24153; '1,2,3,4,5,6-hexahydro-2,2'-bipyridine (PIN)',:24159): the amine's parent is
    # the assembly, "1',2',3',4',5',6'-hexahydro[2,4'-bipyridin]-5-amine"
    ("Nc1ccc(nc1)C1CCNCC1", 1, True),
    # a benzene ring and a cyclohexene ring (:17083 '2,3-dihydro-1,1'-biphenyl (PIN)')
    ("OC(=O)c1ccc(cc1)C1=CCCCC1", 1, True),
    # the exception: one benzene ring and one cyclohexane ring ('cyclohexylbenzene
    # (PIN)',:24157)
    ("OC(=O)c1ccc(cc1)C1CCCCC1", 1, False),
])
def test_the_suffix_capacity_of_one_ring_carrier(smiles, cap, assembly):
    mol = Chem.MolFromSmiles(smiles)
    pc = features.principal_class(mol)[0]
    assert features.suffix_capacity(mol, pc) == cap
    assert features.assembly_parent_expected(mol, pc) is assembly


@pytest.mark.parametrize("name,expected", [
    ("N,4-diphenylbenzamide", False),
    ("N-phenyl[1,1'-biphenyl]-4-carboxamide", True),
    ("N-([1,1'-biphenyl]-4-yl)benzamide", False),
    ("5-methoxy[1,1'-biphenyl]-2-carboxylic acid", True),
    ("[1,1'-biphenyl]-4,4'-dicarboxylic acid", True),
    ("2,2'-bipyridin-5-amine", True),
    ("1',2',3',4',5',6'-hexahydro[2,4'-bipyridin]-5-amine", True),
    ("6-(piperidin-4-yl)pyridin-3-amine", False),
    ("bicyclo[2.2.1]heptane-2-carboxylic acid", False),
])
def test_a_ring_assembly_parent_is_read_from_the_name(name, expected):
    assert features.name_cites_ring_assembly_parent(name) is expected


def test_the_count_and_parent_checks_end_to_end(tmp_path):
    split = {"split": "mini", "sha256": "x",
             "smiles": ["Nc1ccc(Nc2ncccn2)cc1", "O=C(Nc1ccccc1)c1ccc(cc1)-c1ccccc1"],
             "meta": [{"name": "AMINE"}, {"name": "AMIDE"}]}
    sp = _write(tmp_path / "split.json", split)
    old = _run(split, ["N-(4-aminophenyl)pyrimidin-2-amine", "N,4-diphenylbenzamide"],
               ["pin_verified", "pin_verified"])
    new = _run(split, ["N1-(pyrimidin-2-yl)benzene-1,4-diamine",
                       "N-phenyl[1,1'-biphenyl]-4-carboxamide"], ["pin_verified", "pin_verified"])
    rp = _write(tmp_path / "old.json", old)
    assert score.main(["--split", sp, "--run", f"pin={rp}", "--out", str(tmp_path / "o.json"),
                       "--rank"]) == 0
    base = json.loads((tmp_path / "o.json").read_text())
    T = base["tables"]["pin"]
    assert (T["suffix_ok"], T["suffix_count_miss"], T["parent_miss"], T["clean"]) == (2, 1, 1, 0)
    assert T["pin_verified_parent_miss"] == ["AMINE", "AMIDE"]
    # the two amines sit on the benzene ring, not on the senior pyrimidine first)
    assert base["rank"]["pin"]["amine | PCG on junior ring"]["sole:SUFFIX_COUNT"] == 1
    rp = _write(tmp_path / "new.json", new)
    assert score.main(["--split", sp, "--run", f"pin={rp}", "--out", str(tmp_path / "n.json"),
                       "--base", str(tmp_path / "o.json")]) == 0
    cmp = json.loads((tmp_path / "n.json").read_text())["compare"]["pin"]
    assert cmp["parent_miss_removed"] == ["AMINE: ['SUFFIX_COUNT']", "AMIDE: ['PARENT']"]


def _write(path, obj):
    path.write_text(json.dumps(obj))
    return str(path)


def _run(split, names, labels):
    rows = []
    for smi, nm, lab in zip(split["smiles"], names, labels):
        named = lab != "abstain"
        rows.append({"smiles": smi, "name": nm if named else "unknown organic compound",
                     "tier": lab, "source": "x", "limit_code": None if named else "UNNAMEABLE",
                     "outcome": "rt_exact" if named else "abstain"})
    return {"split": {"name": split["split"], "sha256": split["sha256"]}, "rows": rows}


def test_score_end_to_end(tmp_path, capsys):
    split = {"split": "mini", "sha256": "x", "smiles": [NILOTINIB, "CCO"],
             "meta": [{"name": "NILOTINIB"}, {"name": "ETHANOL"}]}
    sp = _write(tmp_path / "split.json", split)
    runs = {
        "pin": _run(split, [None, "ethanol"], ["abstain", "pin_verified"]),
        "valid": _run(split, [NILOTINIB_VALID_MAIN, "ethanol"], ["systematic_verified", "pin_verified"]),
        "complete": _run(split, [NILOTINIB_VALID_MAIN, "ethanol"], ["systematic_verified", "pin_verified"]),
        "best-effort": _run(split, [NILOTINIB_BE_MAIN, "ethanol"], ["systematic_verified", "pin_verified"]),
    }
    args = ["--split", sp, "--out", str(tmp_path / "base.json"), "--rank"]
    for t, rec in runs.items():
        args += ["--run", f"{t}={_write(tmp_path / (t + '.json'), rec)}"]
    assert score.main(args) == 0
    base = json.loads((tmp_path / "base.json").read_text())
    T = base["tables"]
    assert T["pin"]["pin_verified"] == 1 and T["pin"]["outcomes"] == {"abstain": 1, "rt_exact": 1}
    assert T["valid"]["suffix_miss"] == 1 and T["valid"]["forms_rows"]["BENZ"] == 1
    assert T["best-effort"]["forms_rows"]["OXOM"] == 1 and T["best-effort"]["clean"] == 1
    # two defects (suffix not cited, 'oxomethyl'): counted under each, sole under neither
    nil = base["rank"]["best-effort"]["amide/N-aryl amide | PCG on junior ring"]
    assert (nil["has:SUFFIX"], nil["has:OXOM"], nil.get("sole:SUFFIX", 0)) == (1, 1, 0)
    assert "-- rank best-effort" in capsys.readouterr().out
    # the lane's target: the PIN at every tier
    new = {t: _run(split, [NILOTINIB_PIN, "ethanol"], ["pin_verified", "pin_verified"]) for t in runs}
    args = ["--split", sp, "--out", str(tmp_path / "new.json"), "--base", str(tmp_path / "base.json")]
    for t, rec in new.items():
        args += ["--run", f"{t}={_write(tmp_path / ('n' + t + '.json'), rec)}"]
    assert score.main(args) == 0
    cmp = json.loads((tmp_path / "new.json").read_text())["compare"]
    assert cmp["pin"]["gained_rt_exact"] == ["NILOTINIB"]
    assert cmp["best-effort"]["suffix_gained"] == ["NILOTINIB"]
    assert cmp["best-effort"]["core_form_removed"] == ["NILOTINIB: ['OXOM']"]
    assert "suffix_lost" not in cmp["best-effort"]


def test_a_run_of_another_split_is_refused(tmp_path):
    split = {"split": "mini", "sha256": "x", "smiles": ["CCO"], "meta": [{"name": "ETHANOL"}]}
    sp = _write(tmp_path / "split.json", split)
    other = _run({"split": "other", "sha256": "y", "smiles": ["CCO"]}, ["ethanol"], ["pin_verified"])
    rp = _write(tmp_path / "pin.json", other)
    assert score.main(["--split", sp, "--run", f"pin={rp}", "--out", str(tmp_path / "o.json")]) == 2
