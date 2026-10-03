"""The catalogue numbering of every fused ring system whose name is a PIN parent name is OPSIN
2.9.0's numbering of that name (up to the symmetry of the skeleton), the source the catalogue
cites for its maps (``data/fused_heterocycles.py``). (the Blue Book): the
periphery is numbered "including fusion heteroatoms but not fusion carbon atoms. Each fusion
carbon atom is given the same number as the immediately preceding nonfusion skeletal atom,
modified by a Roman letter"; (:12493) names the traditional exceptions, and
phenanthridine and the phenanthrolines are not among them."""
import re
import subprocess

import pytest
from rdkit import Chem

from orthonym.data.fused_heterocycles import FUSED_HETEROCYCLE_DATA
from orthonym.rules.bridged_fused_pin import parents
from orthonym.validation.opsin_roundtrip import _find_opsin_jar

_AV = re.compile(r"^(\S+)\s+\|\$_AV:(.*)\$\|\s*$")

#: maps that are not OPSIN's numbering: none. The four the S2 TRIAGE recorded (decision 7)
#: -- the lettered fusion nitrogen of 4H-quinolizine and pyrrolizine numbers
#: fusion heteroatoms) and the thianthrene and phenoxathiine maps whose sources disagreed --
#: follow OPSIN's numbering of each name since the quick-wins catalogue maps (QW.8)
KNOWN = set()


def _pin_parent_entries():
    out = []
    for smiles, entry in FUSED_HETEROCYCLE_DATA.items():
        name, locs = entry.get("name") or "", entry.get("iupac_locants") or {}
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or not parents.is_pin_parent_name(parents._IH.sub("", name)):
            continue
        if sorted(k for k in locs if isinstance(k, int)) != list(range(mol.GetNumAtoms())):
            continue
        out.append((name, mol, locs))
    return out


@pytest.mark.opsin_gate
def test_pin_parent_maps_are_opsins_numbering():
    jar = _find_opsin_jar("2.9.0")
    if jar is None:
        pytest.skip("OPSIN 2.9.0 jar not found")
    entries = _pin_parent_entries()
    names = [name for name, _, _ in entries]
    lines = subprocess.run(["java", "-jar", jar, "-o", "extendedsmi"], input="\n".join(names) + "\n",
                           capture_output=True, text=True, timeout=900).stdout.split("\n")
    differ = set()
    for (name, mol, locs), line in zip(entries, lines):
        m = _AV.match(line.strip())
        assert m, (name, "OPSIN does not read it")
        theirs = Chem.MolFromSmiles(m.group(1))
        key = Chem.MolFromSmiles(parents._key_of(mol))
        a, b = parents._iso(mol, key), parents._iso(theirs, key)
        ours = {a[i]: parents.normalize_locant(locs[i]) for i in range(mol.GetNumAtoms())}
        other = {b[i]: parents.normalize_locant(loc) for i, loc in enumerate(m.group(2).split(";"))}
        if not parents._same_orbit(key, ours, other):
            differ.add(name)
    assert {"phenanthridine", "1,10-phenanthroline"} <= set(names)
    assert differ == KNOWN, sorted(differ ^ KNOWN)


def test_phenanthridine_fusion_carbons_follow_the_nitrogen():
    # 4a, N5, 6, 6a... 10, 10a, 10b: the fusion carbon after C6 is 6a, not phenanthrene's 4b
    data = FUSED_HETEROCYCLE_DATA["c1ccc2c(c1)cnc1ccccc12"]["iupac_locants"]
    assert {v for v in data.values() if isinstance(v, str)} == {"4a", "6a", "10a", "10b"}
    mol = Chem.MolFromSmiles("c1ccc2c(c1)cnc1ccccc12")
    n5 = next(a for a in mol.GetAtoms() if a.GetSymbol() == "N")
    c6 = next(nb for nb in n5.GetNeighbors() if nb.GetDegree() == 2)
    assert data[n5.GetIdx()] == 5 and data[c6.GetIdx()] == 6
    assert {data[nb.GetIdx()] for nb in c6.GetNeighbors()} == {5, "6a"}


def test_phenanthridine_is_a_bridged_fused_parent():
    # Table 2.8 (:11537) 'phenanthridine (PIN)': with the map corrected, the catalogue, the
    # Blue Book table and the fusion numbering agree, so the parent source names it
    mol = Chem.MolFromSmiles("c1ccc2c(c1)cnc1ccccc12")
    fp = parents.fused_parent(mol, set(range(mol.GetNumAtoms())))
    assert fp is not None and fp.name == "phenanthridine"
