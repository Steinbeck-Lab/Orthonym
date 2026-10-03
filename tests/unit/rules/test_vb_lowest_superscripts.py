"""Von Baeyer superscripts as low as possible on substituted and hetero cages.

 (the Blue Book): "The superscript locants for the secondary bridges must be as low
as possible when considered as a set in ascending numerical order"; (:9699): "... as low
as possible when considered in the sequence of their order of citation in the name". The von Baeyer
numbering is the fixed numbering of the ring system, so it comes before heteroatoms, suffixes and
prefixes: (:9765) "Numbering is determined first by the fixed numbering of the hydrocarbon
system", (:16623). Only the pure-cage path chose the lowest superscripts; a substituted or
hetero cage kept the superscripts its atom order gave ('tricyclo[2.2.1.0^3,5]heptane' at pin_verified,
'tricyclo[4.4.0.0^5,10]', '2-oxatricyclo[9.4.0.0^4,9]'). Every name below reads back to the input's
full InChIKey with OPSIN 2.9.0.
"""
import pytest
from rdkit import Chem

from orthonym.rules.polycyclic import VonBaeyerAnalyzer, _superscript_shape_and_rank
from tests.support.pin_tiers import assert_pin_at_both_tiers

PIN_ROWS = [
    # milestone1500; was '(2S)-2,3-dimethyl-2-(4-methylpent-3-en-1-yl)tricyclo[2.2.1.0^3,5]heptane'
    ("CC(=CCC[C@]1(C2CC3C1(C3C2)C)C)C",
     "(7S)-1,7-dimethyl-7-(4-methylpent-3-en-1-yl)tricyclo[2.2.1.0^2,6]heptane"),
    # the bare cage (pure-cage path, unchanged)
    ("C12C3CC(CC31)C2", "tricyclo[2.2.1.0^2,6]heptane"),
    # milestone1500 / dev2000: '[4.4.0.0^5,10]' was not a PIN (the PIN tier abstained)
    ("CC1=CC[C@H]2[C@H]3C1[C@@]2(C)CC[C@H]3C(C)C",
     "(1S,6S,7S,8S)-1,3-dimethyl-8-(propan-2-yl)tricyclo[4.4.0.0^2,7]dec-3-ene"),
    ("C/C(CO)=C1/CCC2(C)C3CCC(C)(O)C2C13",
     "(8E)-8-(1-hydroxypropan-2-ylidene)-1,3-dimethyltricyclo[4.4.0.0^2,7]decan-3-ol"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,pin", PIN_ROWS)
def test_lowest_superscripts_at_both_tiers(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


def _ring_atoms(mol):
    return {a for r in mol.GetRingInfo().AtomRings() for a in r}


@pytest.mark.parametrize("smiles,want", [
    # a dev split hetero cage: '2-oxatricyclo[9.4.0.0^4,9]' -> the oxygen follows the fixed numbering
    ("COc1c(O)ccc2c(=O)oc3cc(C)cc(O)c3c(=O)c12", "tricyclo[9.4.0.0^3,8]"),
    ("CC(=CCC[C@]1(C2CC3C1(C3C2)C)C)C", "tricyclo[2.2.1.0^2,6]"),
    ("OC1C2CC3CC1C3C2", "tricyclo[3.2.1.0^3,6]"),
])
def test_analyzer_takes_the_lowest_superscripts(smiles, want):
    mol = Chem.MolFromSmiles(smiles)
    desc = VonBaeyerAnalyzer().analyze(mol, _ring_atoms(mol))
    assert desc is not None and desc.legality is True
    assert desc.descriptor_string == want


def test_shape_and_rank():
    shape, rank = _superscript_shape_and_rank("tricyclo[2.2.1.0^3,5]")
    assert shape == ("tricyclo", ("2", "2", "1", "0"), "")
    assert rank == ((3, 5), (3, 5))
    # example (:9705): '2,6,8,12' is lower than '8,12,2,6' in citation order
    a = _superscript_shape_and_rank("pentacyclo[13.7.4.3^3,8.0^18,20.1^13,28]")[1]
    assert a[0] == tuple(sorted(a[1]))
    # a different main ring or bridge length is a different shape, never a renumbering
    assert (_superscript_shape_and_rank("tricyclo[2.2.1.0^2,6]")[0]
            != _superscript_shape_and_rank("tricyclo[3.1.1.0^2,6]")[0])
    assert _superscript_shape_and_rank("bicyclo[2.2.2]")[1] == ((), ())


@pytest.mark.opsin_gate
@pytest.mark.xfail(strict=True, reason="deferred: a stereocentre of a cage nested inside a substituent "
                   "gets a bare '(R)' (the cage numbering is not threaded into the substituent stereo "
                   "step), which reads back to the other bridgehead under the lowest superscripts")
def test_nested_cage_stereocentre_cites_its_locant():
    """ (the Blue Book): "In preferred IUPAC names, stereodescriptors, preceded by a
    locant, must be cited"; (:44643) cites a substituent's descriptor at the front of its prefix
    with the locant. The bare '[(R)-...]' form read back right under the old superscripts (the
    stereocentre was C3) only because the unlocanted descriptor landed on it; under '0^1,6' the
    stereocentre is C6 and the bare form reads back to C1, so the alcohol parent is lost to a
    general-engine name with the -OH as a prefix. The located form below reads back to the full
    InChIKey (OPSIN 2.9.0)."""
    from tests.support.rt_assert import name_best_effort
    row = name_best_effort("OCC1CCC(CC1)C3=C4CCCC5([C@@H]3CCCN5)N4")
    assert row["name"] == (
        "{4-[(6R)-2,12-diazatricyclo[6.3.1.0^1,6]dodec-7-en-7-yl]cyclohexyl}methanol")
