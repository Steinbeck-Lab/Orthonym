"""Producers that dropped ring bonds or acid-side branches decline instead.

Each name below was a different molecule, built and then stopped only by the final read-back
(OPSIN 2.9.0 parse, full InChIKey):

* name_sulfone / name_sulfoxide named a RING sulfur as the centre of an acyclic functional-class
  name: 'diphenyl sulfone' for 2-chlorothianthrene 5,5-dioxide, 'diphenyl sulfoxide' for
  phenoxathiine 10-oxide. The sibling name_sulfide already declined a ring sulfur; the ring
  producers name the oxides, the Blue Book).
* the ester acid word dropped acid-chain branches the prefix pipeline could not name:
  'heptan-2-yl ethanoate' for cloquintocet-mexyl, an aryloxyacetate.
"""
from rdkit import Chem

from orthonym.rules.sulfur import name_sulfone, name_sulfoxide


def _mol(smiles):
    return Chem.MolFromSmiles(smiles)


def test_ring_sulfone_and_sulfoxide_are_not_functional_class_names():
    m = _mol("ClC1=CC=2SC3=CC=CC=C3S(C2C=C1)(=O)=O")
    s = next(a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == "S" and a.GetDegree() == 4)
    assert name_sulfone(m, (s,)) is None
    m = _mol("C1=CC=CC=2OC3=CC=CC=C3S(C12)=O")
    s = next(a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == "S")
    assert name_sulfoxide(m, (s,)) is None


def test_acyclic_sulfone_and_sulfoxide_keep_their_names():
    m = _mol("CS(=O)(=O)C")
    assert name_sulfone(m, (1,)) == "dimethyl sulfone"
    m = _mol("CS(=O)C")
    assert name_sulfoxide(m, (1,)) == "dimethyl sulfoxide"


def test_the_ester_acid_word_keeps_or_declines_its_branches():
    from orthonym.perception.functional_groups import detect_functional_groups
    from orthonym.rules.esters import name_ester
    m = _mol("CCCCCC(C)OC(=O)COc1ccc(Cl)c2cccnc12")
    for match in detect_functional_groups(m).get("ester", []):
        name = name_ester(m, match)
        assert name != "heptan-2-yl ethanoate", name
