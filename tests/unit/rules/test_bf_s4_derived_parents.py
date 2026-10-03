"""The rule-derived parent table of the bridged fused builder
(``bridged_fused_pin.derived_parents``; user ruling D6, 2026-10-02): every entry is OPSIN
2.9.0's own structure and numbering of its name, and each step of its written
derivation holds on the entry.

'[1]benzofuro[3,2-e]isoquinoline', the parent of '4,12-methano[1]benzofuro[3,2-e]-
isoquinoline' (the 4,5-epoxymorphinans): (a)/(b) (the Blue Book,:12163)
isoquinoline is the parent component; (:13437) 1-benzofuran is one attached unit;
 (:11907,:11911) '[1]benzofuro', side e (a peripheral side of isoquinoline),
attached locants '3,2'; the fusion atom 12b carries four ring bonds, as naphthalene 8a in
'naphtho[1,8a-b]azirine (PIN)',:14011); (:12501) the peripheral
numbering."""
import re
import subprocess

import pytest
from rdkit import Chem

import orthonym.rules.bridged_fused_pin as bfp
from orthonym.rules.bridged_fused_pin import parents, selection
from orthonym.rules.bridged_fused_pin.derived_parents import DERIVED_PARENTS
from orthonym.rules.bridged_fused_pin.parent_table import BB_PARENTS
from orthonym.validation.opsin_roundtrip import _find_opsin_jar

_AV = re.compile(r"^(\S+)\s+\|\$_AV:(.*)\$\|\s*$")
MORPHINE_PARENT = "[1]benzofuro[3,2-e]isoquinoline"


@pytest.fixture(scope="module")
def opsin_lines():
    jar = _find_opsin_jar("2.9.0")
    if jar is None:
        pytest.skip("OPSIN 2.9.0 jar not found")
    names = [entry.opsin_name for entry in DERIVED_PARENTS.values()]
    out = subprocess.run(["java", "-jar", jar, "-o", "extendedsmi"], input="\n".join(names) + "\n",
                         capture_output=True, text=True, timeout=300).stdout.split("\n")
    return dict(zip(names, out))


@pytest.mark.opsin_gate
def test_every_entry_is_opsins_own_structure_and_numbering(opsin_lines):
    bad = []
    for name, entry in DERIVED_PARENTS.items():
        m = _AV.match(opsin_lines.get(entry.opsin_name, "").strip())
        if not m:
            bad.append((name, "OPSIN does not read it"))
            continue
        ours, theirs = Chem.MolFromSmiles(entry.smiles), Chem.MolFromSmiles(m.group(1))
        key = Chem.MolFromSmiles(parents._key_of(ours))
        a, b = parents._iso(ours, key), parents._iso(theirs, key)
        if a is None or b is None:
            bad.append((name, "another skeleton"))
            continue
        first = {a[i]: parents.normalize_locant(loc) for i, loc in enumerate(entry.locants)}
        other = {b[i]: parents.normalize_locant(loc) for i, loc in enumerate(m.group(2).split(";"))}
        if not parents._same_orbit(key, first, other):
            bad.append((name, "another numbering"))
    assert not bad, bad


def test_every_entry_is_a_pin_parent_name_with_a_written_derivation():
    for name, entry in DERIVED_PARENTS.items():
        assert parents.is_pin_parent_name(name), name
        assert entry.derivation and all(isinstance(line, int) and 2000 < line < 58000
                                        for _, line in entry.derivation), name
        assert re.sub(r"^((?:\d+[a-z]?H,)*\d+[a-z]?H)-", "", entry.opsin_name) == name, name
        assert name not in BB_PARENTS and name not in parents.PIN_PARENT_NAMES, name


def test_no_entry_shares_a_skeleton_with_another_table():
    for name, entry in DERIVED_PARENTS.items():
        key = parents._key_of(Chem.MolFromSmiles(entry.smiles))
        sources = {(e[0], e[3]) for e in parents._table_index().get(key, ())}
        assert sources == {(name, "derived_table")}, (name, sources)


def _by_locant():
    entry = DERIVED_PARENTS[MORPHINE_PARENT]
    mol = Chem.MolFromSmiles(entry.smiles)
    return mol, {loc: i for i, loc in enumerate(entry.locants)}


def _ring(mol, at, locs):
    return frozenset(at[x] for x in locs) in {frozenset(r) for r in mol.GetRingInfo().AtomRings()}


def test_the_rings_and_heteroatoms_of_the_morphine_parent():
    mol, at = _by_locant()
    assert _ring(mol, at, ("1", "2", "3", "4", "4a", "12b"))           # D: the pyridine ring
    assert _ring(mol, at, ("4a", "5", "6", "7", "7a", "12b"))          # C
    assert _ring(mol, at, ("7a", "8", "8a", "12a", "12b"))             # E: the furan
    assert _ring(mol, at, ("8a", "9", "10", "11", "12", "12a"))        # A: the benzene ring
    assert {loc for loc, i in at.items() if mol.GetAtomWithIdx(i).GetAtomicNum() != 6} == {"3", "8"}
    assert mol.GetAtomWithIdx(at["3"]).GetSymbol() == "N"
    assert mol.GetAtomWithIdx(at["8"]).GetSymbol() == "O"
    # the fusion atom 12b (isoquinoline 4a) carries four ring bonds, as naphthalene 8a in
    # 'naphtho[1,8a-b]azirine (PIN)',:14007,:14011)
    assert mol.GetAtomWithIdx(at["12b"]).GetDegree() == 4


def test_the_parent_component_is_isoquinoline_and_the_descriptor_is_3_2_e():
    # (a), (b): the nitrogen component with more rings is C + D; N is bonded to
    # locant 4, which is bonded to the fusion atom 4a: isoquinoline N2 / C1 / C8a. Then
    # isoquinoline 4a = 12b and isoquinoline 5 = 7a, so side e (4a-5) is the bond 12b-7a.
    mol, at = _by_locant()
    n = mol.GetAtomWithIdx(at["3"])
    assert {nb.GetIdx() for nb in n.GetNeighbors()} == {at["2"], at["4"]}
    assert mol.GetBondBetweenAtoms(at["4"], at["4a"]) is not None
    assert mol.GetBondBetweenAtoms(at["4a"], at["12b"]) is not None     # isoquinoline 8a-4a
    side_e = (at["12b"], at["7a"])
    furan = {at[x] for x in ("7a", "8", "8a", "12a", "12b")}
    ring_c = {at[x] for x in ("4a", "5", "6", "7", "7a", "12b")}
    assert furan & ring_c == set(side_e)
    # 1-benzofuran: O1, C2 (bonded to O1), C3 (bonded to C3a, a benzene fusion atom). C3
    # lies on isoquinoline 4a (12b) and C2 on isoquinoline 5 (7a): in the direction of
    # lettering, 4a -> 5, the attached locants are '3,2',:11911)
    assert mol.GetBondBetweenAtoms(at["7a"], at["8"]) is not None       # C2-O1
    assert mol.GetBondBetweenAtoms(at["12b"], at["12a"]) is not None    # C3-C3a


def test_the_peripheral_numbering_follows_the_periphery():
    # (:12501): consecutive locants are bonded around the periphery, and each
    # fusion carbon atom takes the number of the preceding nonfusion atom with a letter
    mol, at = _by_locant()
    order = ["1", "2", "3", "4", "4a", "5", "6", "7", "7a", "8", "8a", "9", "10", "11", "12",
             "12a", "12b"]
    for x, y in zip(order, order[1:] + order[:1]):
        assert mol.GetBondBetweenAtoms(at[x], at[y]) is not None, (x, y)
    for loc in order:
        atom = mol.GetAtomWithIdx(at[loc])
        fusion = atom.GetDegree() >= 3
        assert loc[-1].isalpha() == (fusion and atom.GetAtomicNum() == 6), loc


MORPHINE = "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5"


def test_the_methano_reading_wins_on_the_morphine_ring_system():
    # (a) (:14261), (b) (:14271): the 17-atom four-ring parent with a methano
    # bridge, not phenanthro[4,5-bcd]furan (15 atoms) with an azanoethano bridge
    mol = Chem.MolFromSmiles(MORPHINE)
    splits = selection.best_splits(mol, selection.ring_system(mol))
    assert splits and {s.parent for s in splits} == {MORPHINE_PARENT}
    assert all(len(s.residual) == 17 and [len(b) for b in s.bridges] == [1] for s in splits)


@pytest.mark.parametrize("smiles,name", [
    ("C1C=NC2=C3C=CC=C4C13C1=C(O4)C=CC=C1C2", "1H-4,12-methano[1]benzofuro[3,2-e]isoquinoline"),
    (MORPHINE, "(4R,4aR,7S,7aR,12bS)-3-methyl-2,3,4,4a,7,7a-hexahydro-1H-4,12-methano"
               "[1]benzofuro[3,2-e]isoquinoline-7,9-diol"),
    ("CN1CC[C@]23c4c5ccc(OC)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5",
     "(4R,4aR,7S,7aR,12bS)-9-methoxy-3-methyl-2,3,4,4a,7,7a-hexahydro-1H-4,12-methano"
     "[1]benzofuro[3,2-e]isoquinolin-7-ol"),
    ("CN1CC[C@]23[C@@H]4C(=CC=C2[C@H]1CC5=C3C(=C(C=C5)OC)O4)OC",
     "(4R,7aR,12bS)-7,9-dimethoxy-3-methyl-2,3,4,7a-tetrahydro-1H-4,12-methano"
     "[1]benzofuro[3,2-e]isoquinoline"),
    ("CN1CCC23c4c5cccc4OC2CCCC3C1C5",
     "3-methyl-2,3,4,4a,5,6,7,7a-octahydro-1H-4,12-methano[1]benzofuro[3,2-e]isoquinoline"),
])
def test_the_builder_names_the_epoxymorphinans_on_the_derived_parent(smiles, name):
    # the spelling OPSIN does not police: '[1]' in the prefix (OPSIN also reads
    # '4,12-methanobenzofuro[3,2-e]isoquinoline'), indicated hydrogen 1H, no hydro locant on
    # the bridge atom 13:14245; OPSIN rejects '...-2,4,4a,7,7a,13-hexahydro...')
    got = bfp.build(Chem.MolFromSmiles(smiles))
    assert got is not None and got[0] == name, got


@pytest.mark.parametrize("smiles", [
    # (the ring ketones oxycodone and hydromorphone are named since slice S3:
    # test_bf_s3_morphinan.py)
    # a stereocentre outside the ring system: buprenorphine
    "C[C@]([C@H]1C[C@@]23CC[C@@]1([C@H]4[C@@]25CCN([C@@H]3CC6=C5C(=C(C=C6)O)O4)CC7CC7)OC)(C(C)(C)C)O",
    # an ester principal group: diamorphine
    "CN1CC[C@]23c4c5ccc(OC(C)=O)c4O[C@H]2[C@@H](OC(C)=O)C=C[C@H]3[C@H]1C5",
    # a morphinan without the 4,5-epoxy bridge: phenanthrene + azanoethano
    #:14155), which OPSIN 2.9.0 cannot read: dextromethorphan
    "CN1CC[C@@]23CCCC[C@@H]2[C@@H]1CC4=C3C=C(C=C4)OC",
])
def test_the_builder_declines_what_it_does_not_certify(smiles):
    assert bfp.build(Chem.MolFromSmiles(smiles)) is None
