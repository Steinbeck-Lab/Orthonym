"""``perception.substructure.ordered_matches``: a query in another molecule's atom order matches in its canonical order.

A molecule parsed from an OPSIN name comes in OPSIN's atom order. For a molecule with many symmetric leaf groups
(fifteen tert-butyl groups on a boron-carbazole emitter) RDKit's matcher, which extends a match in the query's atom
order, did not return in 130 s; the same query in its canonical order matches in a fraction of a second. The
isotope frame (``rules.isotope_scopes._Prober._frame_of``) and the OPSIN locant map
(``validation.opsin_roundtrip.opsin_atom_locant_map``) match such parses against the input.
"""
import subprocess
import sys

import pytest
from rdkit import Chem

from orthonym.perception.substructure import ordered_matches

# the label-free skeleton of PubChem CID 172503690 as OPSIN 2.9.0 writes it (the atom order of its name)
_OPSIN_ORDER = (
    "C(C)(C)(C)C1=CC(=C(C(=C1)C1=CC=C(C=C1)C(C)(C)C)N1C2=CC(=CC=3N(C=4C=C(C=CC4B(C=4C=CC(=CC14)C1=CC(=CC(=C1)"
    "C(C)(C)C)C(C)(C)C)C32)N3C2=CC=C(C=C2C=2C=C(C=CC32)C(C)(C)C)C(C)(C)C)C3=C(C=C(C=C3C3=CC=C(C=C3)C(C)(C)C)C(C)"
    "(C)C)C3=CC(=CC(=C3)C(C)(C)C)C(C)(C)C)C3=CC(=CC=C3)C(C)(C)C)C3=C(C=C(C=C3)C(C)(C)C)C(C)(C)C")


@pytest.mark.parametrize("target,query", [
    ("CC(C)(C)c1ccccc1", "c1ccccc1C(C)(C)C"),
    ("OC(=O)c1ccc(N)cc1", "Nc1ccc(cc1)C(O)=O"),
    ("C[C@H](N)C(=O)O", "OC(=O)[C@H](C)N"),
])
@pytest.mark.parametrize("uniquify", [True, False])
def test_the_same_matches_as_the_query_in_its_own_order(target, query, uniquify):
    t, q = Chem.MolFromSmiles(target), Chem.MolFromSmiles(query)
    plain = t.GetSubstructMatches(q, uniquify=uniquify, maxMatches=10_000)
    ordered = ordered_matches(t, q, uniquify=uniquify, max_matches=10_000)
    assert len(ordered) == len(plain)
    if uniquify:    # one representative per set of atoms, which one can differ
        assert {frozenset(m) for m in ordered} == {frozenset(m) for m in plain}
    else:
        assert set(ordered) == set(plain)


def test_chirality_is_kept():
    t = Chem.MolFromSmiles("C[C@H](N)C(=O)O")
    assert ordered_matches(t, Chem.MolFromSmiles("OC(=O)[C@H](C)N"), use_chirality=True)       # the same enantiomer
    assert not ordered_matches(t, Chem.MolFromSmiles("OC(=O)[C@@H](C)N"), use_chirality=True)


def test_a_query_in_opsin_order_on_a_molecule_of_fifteen_tert_butyl_groups_returns():
    """In a subprocess with a time limit: the plain search on the same pair does not return."""
    code = (
        "from rdkit import Chem\n"
        "from orthonym.perception.substructure import ordered_matches\n"
        f"q = Chem.MolFromSmiles({_OPSIN_ORDER!r})\n"
        "t = Chem.MolFromSmiles(Chem.MolToSmiles(q))\n"
        "print(len(ordered_matches(t, q, use_chirality=True, uniquify=False, max_matches=4096)))\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert int(out.stdout.strip()) == 4096
