"""LEADS2 topic 3 (SILOX): organyl groups on the hub atoms of an a(ba)n chain ('hexamethyldisiloxane').

Blue Book, all in ``the Blue Book``:

* ``****`` under ```` Heterogeneous acyclic parent hydrides (:8014): the unbranched chain of
  alternating atoms is named by "a multiplying prefix denoting the number of atoms of the terminal element followed
  by the 'a' term for that element, then the 'a' term of the other element in the chain and the ending 'ane'", and
  (:8016) "These parent hydrides are preselected parent hydrides (see and have priority to receive preferred
  IUPAC names, as long as they are used to name compounds containing carbon."
* A group on such a parent is a prefix of the name: 'Cl-SiH2-O-SiH3 chlorodisiloxane (from disiloxane, a preselected
  name)' (:2945, ``****``); 'H3Si-O-SiH2-COOH disiloxanecarboxylic acid (PIN)' (:29896);
  ``****`` (:23535) 'SnH3-O-SnH2-O-SnH3 tristannoxane (preselected name) [not bis(stannyloxy)stannane]'.
* The chain is one heterounit, so skeletal replacement is not asked for: ``****`` (:23348) "Skeletal
  replacement ('a') nomenclature rather than substitutive or multiplicative names must be used to generate preferred
  IUPAC names for acyclic structures when four or more heterounits are present in an unbranched chain containing at
  least one carbon atom", and (:23350) "A heterounit is a set of heteroatoms having a name of its own such as...
  -SiH2-O-SiH2-, disiloxane-1,3-diyl".
* Locants: ``****`` (:3007) "All locants are omitted in compounds or substituent groups in which all
  substitutable positions are completely substituted or modified... in the same way"; ``****`` (:2938)
  "The locant is omitted in monosubstituted symmetrical parent hydrides or parent compounds where there is only one
  kind of substitutable hydrogen"; ``### **** NUMBERING`` (:3219) (f) "detachable alphabetized prefixes, all
  considered together in a series of increasing numerical order" and (g) "lowest locants for the substituent cited
  first as a prefix in the name"; ```` (:2869) cites every locant otherwise.

Every expected name is read back by a fresh OPSIN call (``assert_pin_at_both_tiers``). The book prints no
'hexamethyldisiloxane': the names are the rules above applied (reasoning-backed, not a worked example).
"""
import pytest
from rdkit import Chem

from orthonym.rules.catenated_hydrides import name_catenated_hydride
from tests.support.pin_tiers import assert_pin_at_both_tiers, name_default

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

ROWS = [
    ("C[Si](C)(C)O[Si](C)(C)C", "hexamethyldisiloxane"),                                   #
    ("C[Si](C)(C)O[Si](C)(C)O[Si](C)(C)C", "octamethyltrisiloxane"),                        #
    ("C[Si](C)(C)O[Si](C)(C)O[Si](C)(C)O[Si](C)(C)O[Si](C)(C)C", "dodecamethylpentasiloxane"),
    ("C[SiH2]O[SiH3]", "methyldisiloxane"),                                                #
    ("C[SiH2]O[SiH2]C", "1,3-dimethyldisiloxane"),
    ("C[SiH](C)O[SiH](C)C", "1,1,3,3-tetramethyldisiloxane"),
    ("C[SiH](C)O[Si](C)(C)C", "1,1,1,3,3-pentamethyldisiloxane"),
    ("C[SiH2]O[SiH2]O[SiH3]", "1-methyltrisiloxane"),                                      # two kinds of hydrogen
    ("C[Si](C)(C)O[SiH2]O[Si](C)(C)C", "1,1,1,5,5,5-hexamethyltrisiloxane"),
    ("CC[Si](C)(C)O[Si](C)(C)CC", "1,3-diethyl-1,1,3,3-tetramethyldisiloxane"),
    ("C=C[Si](C)(C)O[Si](C)(C)C=C", "1,3-diethenyl-1,1,3,3-tetramethyldisiloxane"),         # alphanumerical order
    ("c1ccccc1[Si](C)(C)O[Si](C)(C)c1ccccc1", "1,1,3,3-tetramethyl-1,3-diphenyldisiloxane"),
    ("C[Si](C)(c1ccccc1)O[Si](C)(C)C", "1,1,1,3,3-pentamethyl-3-phenyldisiloxane"),
    ("C=C[Si](C)(C)O[Si](C)(C)O[Si](C)(C)C", "1-ethenyl-1,1,3,3,5,5,5-heptamethyltrisiloxane"),   # (g)
    ("C[Ge](C)(C)O[Ge](C)(C)C", "hexamethyldigermoxane"),
    ("C[Si](C)(C)S[Si](C)(C)C", "hexamethyldisilathiane"),
]


@pytest.mark.parametrize("smiles,pin", ROWS)
def test_the_chain_with_groups_on_its_hub_atoms_is_named_on_the_a_ba_n_parent(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


def test_a_long_chain_is_named_at_once():
    """PDMS oligomers: 8 of the 49 closed-shell Si-O-Si rows of the PubChem census12k (all TIMEOUT there) are this
    class; the name needs no search (a walk of the chain), and OPSIN reads it back."""
    smiles = "C[Si](C)(C)" + "O[Si](C)(C)" * 19 + "C"
    assert_pin_at_both_tiers(smiles, "dotetracontamethylicosasiloxane")
    assert name_catenated_hydride(Chem.MolFromSmiles(smiles)) == "dotetracontamethylicosasiloxane"


@pytest.mark.parametrize("smiles", [
    "[SiH3]O[SiH3]",                                  # the bare chain keeps 'disiloxane'
    "C[Si](C)(C)C",                                   # a mononuclear hub: tetramethylsilane (PIN)
    "C[Si](C)(C)[Si](C)(C)C",                         # no bridge atom: hexamethyldisilane
    "C[Si]1(C)O[Si](C)(C)O[Si](C)(C)O[Si](C)(C)O1",   # a ring of hub and bridge atoms
    "C[Si](C)(C)O[Si](C)(C)O[Si](C)(C)O",             # a hydroxy group: a principal group, not read here
    "CO[Si](C)(C)O[Si](C)(C)C",                       # an alkoxy group on a hub
    "C[Si](C)(C)O[Si](C)(O[Si](C)(C)C)O[Si](C)(C)C",  # branched
    "C[Sn](C)(C)O[Sn](C)(C)C",                        # tin is a metal of data/metal_scope.py
    "C[Si](C)(C)N[Si](C)(C)C",                        # a nitrogen bridge is an amine
    "C[Si](C)(C)O[Si](C)(C)[C@H](C)Cl",               # a stereo element
])
def test_a_molecule_outside_the_class_is_not_named_by_the_chain_with_groups(smiles):
    from orthonym.rules.catenated_hydride_groups import name_hub_chain_with_groups  # noqa: F401
    mol = Chem.MolFromSmiles(smiles)
    name = name_catenated_hydride(mol)
    assert name is None or name == "disiloxane", name


def test_the_bare_names_do_not_move():
    for smiles, name in (("[SiH3]O[SiH3]", "disiloxane"), ("[SiH3]O[SiH2]O[SiH3]", "trisiloxane"),
                         ("C[Si](C)(C)C", "tetramethylsilane"),
                         ("C[Si](C)(C)[Si](C)(C)C", "hexamethyldisilane")):
        row = name_default(smiles)
        assert (row["name"], row["tier"]) == (name, "pin_verified"), row
