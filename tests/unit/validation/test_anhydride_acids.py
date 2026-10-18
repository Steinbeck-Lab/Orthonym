"""A di- or polyanhydride name is read back through its acids (``validation.anhydride_acids``).

OPSIN 2.9.0 has no grammar for the class words of a poly-anhydride, and until the check the name shipped with
no read-back (the ``dianhydride`` / ``chalcogen_dianhydride`` carve-outs): 116 of the 208 names of the audited
PubChem rows were the name of another molecule.

* Di- and polyanhydrides (the Blue Book): "Di- and polyanhydrides have two or more
  -CO-O-CO- or related groups... They are named using the class name 'dianhydride', 'trianhydride', etc.,
  preceded by the names of the acid groups cited as separate words."
* (:32380): "Dianhydrides are named by citing the acid groups in their order of occurrence in the
  structure beginning with the end acid group lower in alphabetical order followed by the class term
  'dianhydride'. The numerical prefix 'di-' is used to generate preferred IUPAC names."; 'diacetic butanedioic
  dianhydride (PIN)' (:32386), 'butanoic butanedioic propanoic dianhydride (PIN)' (:32392).
* (:32443): 'diacetic butanedioic bis(thioanhydride) (PIN)' (:32446).
* Anhydrides of polynuclear non-carbon oxoacids (:36925): 'tetraacetic hypodiphosphoric
  tetraanhydride (PIN)' (:36933), 'acetic dithionous propanoic dianhydride (PIN)' (:36935).
* General methodology (:7029) (c): "any component which is substituted automatically requires use of
  the multiplicative forms 'bis', 'tris', etc."
"""
import pytest

from orthonym.validation.anhydride_acids import ANHYDRIDE_ACIDS_RULE, anhydride_acids_round_trip

pytestmark = pytest.mark.opsin_gate

# names of the Blue Book, and names of the same class that spell the multiplier as the rules give it
READ_BACK = [
    ("diacetic butanedioic dianhydride", "CC(=O)OC(=O)CCC(=O)OC(=O)C"),                       #:32386
    ("butanoic butanedioic propanoic dianhydride", "CCCC(=O)OC(=O)CCC(=O)OC(=O)CC"),         #:32392
    ("diacetic butanedioic bis(thioanhydride)", "CC(=O)SC(=O)CCC(=O)SC(C)=O"),               #:32446
    ("tetraacetic hypodiphosphoric tetraanhydride", "CC(=O)OP(=O)(OC(=O)C)P(=O)(OC(=O)C)OC(=O)C"),  #:36933
    ("acetic dithionous propanoic dianhydride", "CC(=O)OS(=O)S(=O)OC(=O)CC"),                #:36935
    ("diacetic disulfuric dianhydride", "CC(=O)OS(=O)(=O)OS(=O)(=O)OC(=O)C"),
    ("tetraacetic diphosphoric tetraanhydride", "CC(=O)OP(=O)(OC(=O)C)OP(=O)(OC(=O)C)OC(=O)C"),
    ("bis(2-methylbenzoic) butanedioic dianhydride", "Cc1ccccc1C(=O)OC(=O)CCC(=O)OC(=O)c1ccccc1C"),
    ("di(hexadecanoic) butanedioic dianhydride",
     "CCCCCCCCCCCCCCCC(=O)OC(=O)CCC(=O)OC(=O)CCCCCCCCCCCCCCC"),
    ("diacetic 2,3-dihydroxybutanedioic dianhydride", "CC(=O)OC(=O)C(O)C(O)C(=O)OC(C)=O"),
    ("diacetic 2-methylbutanedioic dianhydride", "CC(=O)OC(=O)C(C)CC(=O)OC(=O)C"),            #:32392 (the acids' key)
    ("diacetic dibutanedioic trianhydride", "CC(=O)OC(=O)CCC(=O)OC(=O)CCC(=O)OC(=O)C"),
    # the old collapsed spelling is read by its acid: the check proves the structure, never the spelling
    ("di2-methylbenzoic butanedioic dianhydride", "Cc1ccccc1C(=O)OC(=O)CCC(=O)OC(=O)c1ccccc1C"),
    # configurations the free acids and the molecule have alike,:44641)
    ("bis[(2S)-2-hydroxypropanoic] butanedioic dianhydride", "C[C@H](O)C(=O)OC(=O)CCC(=O)OC(=O)[C@H](C)O"),
    ("bis[(2S)-2-aminopropanoic] hexanedioic dianhydride", "C[C@@H](C(=O)OC(=O)CCCCC(=O)OC(=O)[C@H](C)N)N"),
    ("di[(2E)-but-2-enoic] butanedioic dianhydride", "C/C=C/C(=O)OC(=O)CCC(=O)OC(=O)/C=C/C"),
]

# the names of another molecule (each was shipped by the carve-out), and names the acids do not determine
NOT_READ_BACK = [
    # a hydroxy group of the central acid lost
    ("diacetic butanedioic dianhydride", "CC(=O)OC(=O)C(C(C(=O)OC(=O)C)O)O"),
    # the double bond of the central acid lost
    ("diprop-2-enoic butanedioic dianhydride", "C=CC(=O)OC(=O)C=CC(=O)OC(=O)C=C"),
    # the stereo centres of the end acids lost
    ("bis(2-aminopropanoic) hexanedioic dianhydride", "C[C@@H](C(=O)OC(=O)CCCCC(=O)OC(=O)[C@H](C)N)N"),
    # a central diacid that holds an ester O (an atom is lost: three acids, not two)
    ("diacetic octanedioic bis(thioanhydride)", "CC(=O)SC(=O)COC(=O)CCC(=O)OCC(=O)SC(=O)C"),
    # the class word counts one link too many / the wrong bridge atom
    ("diacetic butanedioic trianhydride", "CC(=O)OC(=O)CCC(=O)OC(=O)C"),
    ("diacetic butanedioic bis(thioanhydride)", "CC(=O)OC(=O)CCC(=O)OC(=O)C"),
    ("diacetic butanedioic dianhydride", "CC(=O)SC(=O)CCC(=O)SC(C)=O"),
    # the structure is the input's but the descriptor of the word is the free acid's: the third acid is S in the free
    # acid and R in the molecule, whose alpha carbon bears two carboxyl-type groups, the Blue Book);
    # the name with the molecule's descriptor reads (as a free acid) to the other enantiomer: neither is verified
    ("acetic butanedioic (2S)-3-methoxy-2-methyl-3-oxopropanoic dianhydride",
     "CC(=O)OC(=O)CCC(=O)OC(=O)[C@H](C)C(=O)OC"),
    ("acetic butanedioic (2R)-3-methoxy-2-methyl-3-oxopropanoic dianhydride",
     "CC(=O)OC(=O)CCC(=O)OC(=O)[C@H](C)C(=O)OC"),
    # a word that is no acid
    ("diacetic butanedioicc dianhydride", "CC(=O)OC(=O)CCC(=O)OC(=O)C"),
    # an unsymmetrical diacid between two different acids, with no locants: two structures answer to the name
    ("butanoic 2-methylbutanedioic propanoic dianhydride", "CCCC(=O)OC(=O)C(C)CC(=O)OC(=O)CC"),
    # the two ways the acids can be joined with two diacids between the end acids
    ("diacetic dipropanedioic butanedioic tetraanhydride", "CC(=O)OC(=O)CC(=O)OC(=O)CCC(=O)OC(=O)CC(=O)OC(C)=O"),
]


@pytest.mark.parametrize("name,smiles", READ_BACK)
def test_the_name_reads_back_through_its_acids(name, smiles):
    assert anhydride_acids_round_trip(name, smiles) == ANHYDRIDE_ACIDS_RULE


@pytest.mark.parametrize("name,smiles", NOT_READ_BACK)
def test_a_name_of_another_molecule_or_an_undetermined_one_is_not_verified(name, smiles):
    assert anhydride_acids_round_trip(name, smiles) is None


@pytest.mark.parametrize("name", ["acetic anhydride", "acetic propanoic anhydride", "ethanol", "", "diacetic"])
def test_a_name_of_no_poly_anhydride_class_is_not_read(name):
    """The check reads the class words of a poly-anhydride only; OPSIN reads the others itself."""
    assert anhydride_acids_round_trip(name, "CC(=O)OC(=O)CCC(=O)OC(=O)C") is None


def test_no_structure_is_no_proof():
    assert anhydride_acids_round_trip("diacetic butanedioic dianhydride", None) is None
    assert anhydride_acids_round_trip("diacetic butanedioic dianhydride", "") is None


def test_the_check_names_no_input_it_cannot_split():
    """A cyclic anhydride link is no link of a tree of acids: the structure is not the name's."""
    assert anhydride_acids_round_trip("diacetic butanedioic dianhydride", "O=C1CCC(=O)O1") is None
