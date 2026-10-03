"""Every fused-catalogue numbering map equals OPSIN 2.9.0's numbering of the entry's name.

 (the Blue Book): peripheral atoms are numbered "including fusion heteroatoms
but not fusion carbon atoms. Each fusion carbon atom is given the same number as the immediately
preceding nonfusion skeletal atom, modified by a Roman letter". Seven maps broke it (a fusion N
labelled with a letter: '4a' in 4H-quinolizine for N5, '3a' in pyrrolizine for N4, '5a' in
imidazo[2,1-b]thiazole for N4) or numbered the system from the wrong side (thianthrene and
phenoxathiine S/O labels 5/10 swapped; phenanthridine and 1,10-phenanthroline interior atoms
4b/8a/10a for 6a/10a/10b; phenanthridin-6(5H)-one), and the 7H-purine entry carried purine's
numbering under the fusion name '1H-imidazo[4,5-d]pyrimidine' ("the PIN is 7H-purine",:11595).

The reference is `fused_catalogue_opsin_numbering.json`, written by
`scripts/gen_fused_catalogue_numbering.py` from OPSIN's `$_AV:` locants (no JVM here). Only ring
atoms are compared: an exocyclic atom's label ('=O', 'O6') is a catalogue convention. A map may
differ from the table by an automorphism of the ring system (both are valid numberings).
"""
import json
from pathlib import Path

import pytest
from rdkit import Chem

from orthonym.data.fused_heterocycles import FUSED_HETEROCYCLE_DATA

_TABLE = json.loads((Path(__file__).with_name("fused_catalogue_opsin_numbering.json")).read_text())
_ENTRIES = sorted(k for k, v in FUSED_HETEROCYCLE_DATA.items() if v.get("iupac_locants"))


def _skeleton(mol):
    e = Chem.RWMol(mol)
    for b in e.GetBonds():
        b.SetBondType(Chem.BondType.SINGLE)
        b.SetIsAromatic(False)
    for a in e.GetAtoms():
        a.SetIsAromatic(False)
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(0)
    return e.GetMol()


def test_every_entry_has_a_reference_numbering():
    missing = [FUSED_HETEROCYCLE_DATA[k]["name"] for k in _ENTRIES
               if not (_TABLE.get(k) or {}).get("ring_locants")]
    assert not missing, (f"run scripts/gen_fused_catalogue_numbering.py; no OPSIN numbering "
                         f"stored for {missing}")


@pytest.mark.parametrize("key", _ENTRIES, ids=[FUSED_HETEROCYCLE_DATA[k]["name"] for k in _ENTRIES])
def test_map_equals_opsin_numbering_up_to_automorphism(key):
    entry = FUSED_HETEROCYCLE_DATA[key]
    ref = (_TABLE.get(key) or {}).get("ring_locants")
    if not ref:
        pytest.skip("no reference numbering (see test_every_entry_has_a_reference_numbering)")
    assert _TABLE[key]["name"] == entry["name"], "the table was written for another name: regenerate it"
    mol = Chem.MolFromSmiles(key)
    ring = [a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()]
    have = {i: str(entry["iupac_locants"][i]) for i in ring if i in entry["iupac_locants"]}
    sk = _skeleton(mol)
    for auto in sk.GetSubstructMatches(sk, uniquify=False, maxMatches=5000):
        if all(have[i] == ref.get(str(auto[i])) for i in have):
            return
    pytest.fail(f"{entry['name']}: map {have} is not OPSIN's numbering {ref} under any automorphism")


@pytest.mark.parametrize("name,smiles,atom_symbol,locant", [
    ("4H-quinolizine", "C1=CCN2C=CC=CC2=C1", "N", 5),
    ("pyrrolizine", "C1=Cn2cccc2C1", "N", 4),
    ("imidazo[2,1-b]thiazole", "c1cn2ccsc2n1", "N", 4),
])
def test_a_fusion_nitrogen_gets_a_number(name, smiles, atom_symbol, locant):
    key = Chem.CanonSmiles(smiles)
    entry = FUSED_HETEROCYCLE_DATA[key]
    assert entry["name"] == name
    mol = Chem.MolFromSmiles(key)
    fusion_n = [a.GetIdx() for a in mol.GetAtoms()
                if a.GetSymbol() == atom_symbol and a.GetDegree() == 3 and a.IsInRing()]
    assert [entry["iupac_locants"][i] for i in fusion_n] == [locant]
