"""The Blue Book parent table of the bridged fused builder (``bridged_fused_pin.parent_table``):
every entry is a name the book prints as a PIN (or as the fused ring system of a PIN), and its
structure and numbering are OPSIN 2.9.0's own reading of that name:11903,
 :12501,:12658)."""
import re
import subprocess

import pytest
from rdkit import Chem

from orthonym.rules.bridged_fused_pin import parents
from orthonym.rules.bridged_fused_pin.parent_table import BB_PARENTS
from orthonym.validation.opsin_roundtrip import _find_opsin_jar

_AV = re.compile(r"^(\S+)\s+\|\$_AV:(.*)\$\|\s*$")


def _interior_atoms(mol):
    ri = mol.GetRingInfo()
    return [a.GetIdx() for a in mol.GetAtoms()
            if all(ri.NumBondRings(b.GetIdx()) >= 2 for b in a.GetBonds() if b.IsInRing())]


@pytest.fixture(scope="module")
def opsin_lines():
    jar = _find_opsin_jar("2.9.0")
    if jar is None:
        pytest.skip("OPSIN 2.9.0 jar not found")
    names = [entry[1] for entry in BB_PARENTS.values()]
    out = subprocess.run(["java", "-jar", jar, "-o", "extendedsmi"], input="\n".join(names) + "\n",
                         capture_output=True, text=True, timeout=900).stdout.split("\n")
    return dict(zip(names, out))


@pytest.mark.opsin_gate
def test_every_entry_is_opsins_own_structure_and_numbering(opsin_lines):
    bad = []
    for name, (line, opsin_name, smiles, locants) in BB_PARENTS.items():
        m = _AV.match(opsin_lines.get(opsin_name, "").strip())
        if not m:
            bad.append((name, "OPSIN does not read it"))
            continue
        ours, theirs = Chem.MolFromSmiles(smiles), Chem.MolFromSmiles(m.group(1))
        key = Chem.MolFromSmiles(parents._key_of(ours))
        a, b = parents._iso(ours, key), parents._iso(theirs, key)
        if a is None or b is None:
            bad.append((name, "another skeleton"))
            continue
        first = {a[i]: parents.normalize_locant(loc) for i, loc in enumerate(locants)}
        other = {b[i]: parents.normalize_locant(loc) for i, loc in enumerate(m.group(2).split(";"))}
        if not parents._same_orbit(key, first, other):
            bad.append((name, "another numbering"))
    assert not bad, bad


def test_every_entry_is_a_pin_parent_name_with_a_book_line():
    for name, (line, opsin_name, smiles, locants) in BB_PARENTS.items():
        assert parents.is_pin_parent_name(name), name
        assert isinstance(line, int) and 2000 < line < 58000, (name, line)
        assert re.sub(r"^((?:\d+[a-z]?H,)*\d+[a-z]?H)-", "", opsin_name) == name, (name, opsin_name)


def test_no_entry_has_an_interior_atom():
    # (:12658): an interior atom takes "that of the peripheral atom with a
    # superscript number"; OPSIN 2.9.0 letters it after the highest peripheral atom
    # ('pyrene' 10b, 10c), the numbering:12658 replaces
    for name, (_, _, smiles, _) in BB_PARENTS.items():
        assert not _interior_atoms(Chem.MolFromSmiles(smiles)), name


def test_no_entry_renames_a_skeleton_the_older_tables_name():
    # the table holds no skeleton the older tables name with another name, and no skeleton
    # twice (the catalogue entries it shares a skeleton with carry its name: 1,10-phenanthroline,
    # phenanthridine,... became PIN parent names through it)
    seen = {}
    for name, (_, _, smiles, _) in BB_PARENTS.items():
        key = parents._key_of(Chem.MolFromSmiles(smiles))
        assert key not in seen, (name, seen.get(key))
        seen[key] = name
        others = {e[0] for e in parents._table_index().get(key, ()) if e[3] != "bb_table"}
        assert others <= {name}, (name, others)


@pytest.mark.parametrize("smiles,name", [
    ("C1=CC2=Cc3ccccc3C2=C1", "cyclopenta[a]indene"),                    #:19829
    ("C1=CC=c2ccccc2=CC=CC=CCC=C1", "benzo[13]annulene"),                 #:23875
    ("C1=CC2=CC=CC2=C1", "pentalene"),                                    #:11459
    ("c1ccc2c(c1)oc1ccccc12", "dibenzo[b,d]furan"),                       #:33050
    ("c1ccc2cc3c(ccc4ccccc43)cc2c1", "tetraphene"),                       #:25762
    (BB_PARENTS["dibenzo[b,g][1,6]diazecine"][2], "dibenzo[b,g][1,6]diazecine"),  #:14454
    # the tables' '1H-benzimidazole' is not a PIN spelling; the book's '[1,3]benzimidazole'
    # (:12600) is, and the rule (:11815) "for preferred IUPAC names locants must be
    # cited"
    ("c1ccc2[nH]cnc2c1", "1,3-benzimidazole"),
    # the catalogue's '1,2-benzisoxazole' is not a PIN spelling:11982); the
    # book's '[1,2]benzoxazolo[6,5-g]quinoline (PIN)' (:13449) spells the benzo name
    ("c1ccc2c(c1)cno2", "1,2-benzoxazole"),
])
def test_book_parents_the_older_tables_could_not_certify_are_named(smiles, name):
    mol = Chem.MolFromSmiles(smiles)
    fp = parents.fused_parent(mol, set(range(mol.GetNumAtoms())))
    assert fp is not None and fp.name == name and name in BB_PARENTS, fp


def test_the_fusion_numbering_agrees_with_the_table_on_cyclopenta_a_naphthalene():
    # decision 7 of S2 holds for the table too: each table map is cross-checked by the fusion
    # numbering. Since S2c-1 the fusion numbering draws a system with a five-membered ring on
    # the hexagon grid (``fusion_orientation.grid_orientations``) and gives OPSIN's
    # cyclopenta[a]naphthalene numbering (3a, 5a, 9a, 9b, the book's own locants in
    # '3a,9b-dihydro-1H-cyclopenta[a]naphthalene-3,5(2H,4H)-dione (PIN)':25456), so the two
    # maps are one orbit and the skeleton is named; before S2c-1 the fusion numbering gave
    # {2a, 5a, 5b, 9a} and the skeleton declined
    from orthonym.rules.fusion_numbering import compute_fused_numbering
    mol = Chem.MolFromSmiles("C1=Cc2c(ccc3ccccc23)C1")
    key = parents._key_of(mol)
    key_mol = Chem.MolFromSmiles(key)
    assert "cyclopenta[a]naphthalene" in BB_PARENTS
    entries = [e for e in parents._table_index().get(key, ()) if e[3] == "bb_table"]
    assert [e[0] for e in entries] == ["cyclopenta[a]naphthalene"]
    _, pm, fixed, _ = entries[0]
    iso = parents._iso(pm, key_mol)
    cfn = compute_fused_numbering(pm, set(range(pm.GetNumAtoms())))
    assert iso is not None and cfn and len(cfn) == pm.GetNumAtoms()
    table_map = {iso[a]: loc for a, loc in fixed.items()}
    fusion_map = {iso[a]: parents.normalize_locant(loc) for a, loc in cfn.items()}
    assert parents._same_orbit(key_mol, table_map, fusion_map)
    fp = parents.fused_parent(mol, set(range(mol.GetNumAtoms())))
    assert fp is not None and fp.name == "cyclopenta[a]naphthalene", fp
    assert sorted(str(loc) for loc in fp.numberings[0].values() if not str(loc).isdigit()) == \
        ["3a", "5a", "9a", "9b"]


#: the Table 2.8 isomer notes (the Blue Book,:11563): "1,7-isomer shown; the PIN is
#: 1,7-phenanthroline; other isomers are: 1,8-; 1,9-; 1,10-; 2,7-; 2,8-; 2,9-; 3,7-; 3,8-;
#: 4,7-" and "(1,5-isomer shown; the PIN is 1,5-naphthyridine; other isomers are 1,6-; 1,7-;
#: 1,8-; 2,6-; 2,7-)"
ISOMER_NOTES = {
    11523: ("phenanthroline", "1,7", ("1,8", "1,9", "1,10", "2,7", "2,8", "2,9", "3,7", "3,8", "4,7")),
    11563: ("naphthyridine", "1,5", ("1,6", "1,7", "1,8", "2,6", "2,7")),
}


def test_the_entries_citing_an_isomer_note_are_the_isomers_it_lists():
    # the two named PINs and the fourteen other isomers, each read as the (:11505)
    # retained name with the isomer's locants (the table docstring); no other name hides
    # behind these two lines
    cited = {name for name, (line, *_rest) in BB_PARENTS.items() if line in ISOMER_NOTES}
    listed = {f"{loc}-{stem}" for stem, pin, others in ISOMER_NOTES.values() for loc in (pin, *others)}
    assert cited == listed
    assert len(listed) == 16
