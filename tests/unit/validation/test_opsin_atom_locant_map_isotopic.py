"""The locants OPSIN gives a name with an isotope descriptor, and the stereo composed onto it.

``validation.opsin_roundtrip.opsin_atom_locant_map`` reads the per-atom locants of OPSIN's extended SMILES of
a name (``-o extendedsmi``) and maps them onto the input. The atom-value list of that output ends at the last
atom that has a label, and the explicit isotopic hydrogens OPSIN writes after the atom they stand on
('(2-2H)', '(2H3)methyl') have none, so the list of a name with an isotope descriptor is shorter than its
atom count. Read as a length mismatch it left every labelled name without a map, and with it without the
stereo that the best-effort tier composes onto the name of the stereo-free structure
(``assembly.stereo_composed_offer``, ``namer._try_compose_input_stereo``).

Cited rule (the Blue Book, with its section heading): 'STEREOISOMERIC ISOTOPICALLY SUBSTITUTED
COMPOUNDS' (:43849): "Two types of stereoisomeric isotopically substituted compounds are possible: (1) those
in which the stereoisomerism results from isotopic modification; (2) those whose analogous unmodified
compounds are themselves stereoisomers"; its example '(1R)-(1-2H1)ethan-1-ol (PIN)'. The composed names put
the descriptors at the front of the complete name, in the locants of the stereo-free name, so they are not
the PIN NAMING OF STEREOISOMERS,:44643: "stereodescriptors are placed immediately at the front of the
part of the name to which they relate"); the tier is systematic_verified.
"""
import pytest
from rdkit import Chem

from orthonym.validation.opsin_roundtrip import opsin_atom_locant_map
from tests.support.rt_assert import name_best_effort, name_is_rt_exact

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("name,smiles", [
    # the deuterium of the last atom of OPSIN's SMILES has no value in the list
    ("ethyl 2-amino-3-methyl(2-2H)butanoate", "CCOC(=O)[C@@]([2H])(N)C(C)C"),
    # '(2H3)methyl' puts three hydrogens in the middle of the atom list, the last atoms are hydrogens
    ("ethyl 2-amino-3-(2H3)methylbutanoate", "CCOC(=O)C(N)C(C)C([2H])([2H])[2H]"),
])
def test_a_name_with_an_isotope_descriptor_has_a_locant_map(name, smiles):
    mol = Chem.MolFromSmiles(smiles)
    mapping = opsin_atom_locant_map(name, mol)
    assert mapping, name
    alpha = next(a.GetIdx() for a in mol.GetAtoms()
                 if a.GetSymbol() == "C" and any(n.GetSymbol() == "N" for n in a.GetNeighbors()))
    assert mapping[alpha] == 2


def test_a_name_without_a_label_has_its_locant_map_as_before():
    mol = Chem.MolFromSmiles("CCOC(=O)C(N)C(C)C")
    mapping = opsin_atom_locant_map("ethyl 2-amino-3-methylbutanoate", mol)
    alpha = next(a.GetIdx() for a in mol.GetAtoms()
                 if a.GetSymbol() == "C" and any(n.GetSymbol() == "N" for n in a.GetNeighbors()))
    assert mapping and mapping[alpha] == 2


@pytest.mark.parametrize("smiles,expected", [
    # two centres a deuterium makes, composed on the stereo-free name
    ("[H][C@@]1(C[C@](CC(C1C2=CC=C(C=C2)OCC)([2H])[2H])([2H])CCC)[2H]",
     "(4S,6S)-1-ethoxy-4-[4-propyl(2,2,4,6-2H4)cyclohexyl]benzene"),
    ("[H][C@@]1(C2=C(C3=CC=CC=C3N2)C4(CC4)CN1C(=O)C)[2H]",
     "(1'S)-1-(spiro[cyclopropane-1,4'-(1-2H1)tryptoline]-2'-yl)ethan-1-one"),
])
def test_the_stereo_of_a_labelled_molecule_is_composed_on_its_stereo_free_name(smiles, expected):
    """The input names the stereo-free structure at the best-effort tier, and the composed name reads back to
    the full InChIKey of the input (the isotope layer included)."""
    row = name_best_effort(smiles)
    assert row["name"] == expected, row
    assert row["tier"] == "systematic_verified" and not row["is_pin"], row
    assert name_is_rt_exact(row["name"], smiles), row
