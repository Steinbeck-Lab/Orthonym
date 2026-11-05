"""A ring carboxamide of a heteromonocycle is named with its N-substituents; a multiplied unit with marks is enclosed by the next kind
(lane/l17 item 7, fix round).

 (the Blue Book) puts the principal characteristic group before the seniority of the rings, and
"Systems composed of rings and chains" (:19336) (1) makes a ring senior to a chain of one class: the Pro amide of a peptide
is 'pyrrolidine-2-carboxamide', its N-substituents cited with the locant N,:32774) beside the ring prefixes
. The lone-monocycle producer of the general engine (``assembly.monocycle_ring_amide``) names them; before it held
the atoms of the N-substituents out of nowhere and refused, so the molecule was named on its thiazole or with a 'carbamoyl'
prefix and no amide suffix.

 "Nesting of enclosing marks" ({[({})]}): a multiplied unit that holds marks of its own is enclosed by the next
kind, and gives a unit whose name opens with a stereodescriptor 'bis'/'tris' like one that opens with a locant.
"""
import pytest

from orthonym.cli import _tiered_namer
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


def _row(smiles, tier):
    return _tiered_namer("pin", tier).name_tiered(smiles)


# --- the ring carboxamide whose ring has no cycloalkane or benzene name is a parent too (best-effort general engine) -------

@pytest.mark.parametrize("smiles,name", [
    ("CC(=O)N1CCCC1C(=O)N(C)C(C)C(=O)NC", "1-acetyl-N-methyl-N-[1-(methylamino)-1-oxopropan-2-yl]pyrrolidine-2-carboxamide"),
    ("CC(=O)N1CCCC1C(=O)NC", "1-acetyl-N-methylpyrrolidine-2-carboxamide"),
    ("CC(=O)N1CCC[C@H]1C(=O)N(C)[C@@H](Cc1ccccc1)C(=O)NC",
     "(2S)-1-acetyl-N-methyl-N-[(2S)-1-(methylamino)-1-oxo-3-phenylpropan-2-yl]pyrrolidine-2-carboxamide"),
])
def test_a_ring_carboxamide_of_a_heteromonocycle_is_named_with_its_n_substituents(smiles, name):
    """ (the Blue Book): R-CO-NR'R'' cites R' and R'' with the locant N; the pyrrolidine of the Pro
    amide is the parent (1)). The monocycle producer of the general engine held the N-substituents out of
    nowhere and refused (the molecule was named on its thiazole, or with a 'carbamoyl' prefix and no amide suffix)."""
    row = _row(smiles, "best-effort")
    assert row["name"] == name, row
    assert row["tier"] != "pin_verified", row
    assert_full_rt(row["name"], smiles)


def test_the_n_methyl_peptide_is_named_on_its_proline_carboxamide():
    smiles = ("CCCC(=O)N(C)[C@H](C(=O)N1CCC[C@H]1C(=O)N(C)[C@@H](Cc1ccccc1)C(=O)N[C@@H](CC(C)C)C(=O)N(C)[C@H]"
              "(C(=O)N(C)Cc1nccs1)C(C)C)C(C)C")
    row = _row(smiles, "best-effort")
    assert row["name"].endswith("pyrrolidine-2-carboxamide"), row
    assert "thiazol-2-yl" in row["name"] and row["tier"] != "pin_verified", row
    assert_full_rt(row["name"], smiles)


def test_the_ring_that_bears_the_amide_is_the_parent_whatever_the_other_rings_are():
    """: the benzene ring that carries the carbonyl is the parent of the benzamide, not the thiolane dioxide ring
    that holds the first atom of an N-substituent (the sulfone ring is the senior ring by."""
    smiles = "C1CS(=O)(=O)CC1N(CC2=C(C=CC=C2Cl)F)C(=O)C3=CC=CC=C3Cl"
    row = _row(smiles, "best-effort")
    assert row["name"] == "2-chloro-N-[(2-chloro-6-fluorophenyl)methyl]-N-(1,1-dioxothiolan-3-yl)benzene-1-carboxamide", row
    assert row["tier"] != "pin_verified", row
    assert_full_rt(row["name"], smiles)


def test_a_senior_fused_ring_keeps_the_choice_of_the_cage_producers():
    """The lone-monocycle producer takes the ring of the principal group only where the senior ring is a lone ring too: a
    senior bicyclic ring system is the cage producers' parent, and the molecule keeps the name it had (the monocycle
    producer's name of its piperidine cited the ring stereodescriptors of the substituent in the wrong configuration)."""
    smiles = r"C\1C/C(=C\2/C=NC(=N/C2=C/C=C1)NCC3=CCC(C=C3)O)/N4CCC(CC4)C(=O)NC5=CN=CC=C5"
    row = _row(smiles, "best-effort")
    assert row["name"] and row["name"] != "unknown organic compound" and row["tier"] != "abstain", row
    assert_full_rt(row["name"], smiles)


# --- the multiplied unit: a unit that holds enclosing marks is enclosed by the next kind ---------------------

def test_a_multiplied_unit_with_a_stereodescriptor_front_takes_bis_and_the_next_enclosing_marks():
    """ 'bis'/'tris' for a substituted unit, which a name that opens with its stereodescriptor ('(2S)-3-amino-...')
    is as much as one that opens with a locant; "Nesting of enclosing marks": the unit holds  and , so it is
    enclosed in { }."""
    from orthonym.rules.multiplicative import _assemble_multiplicative_name
    unit = "(2S)-3-amino-2-[2-(1H-indol-3-yl)acetamido]-N-phenylpropanamide"
    assert _assemble_multiplicative_name(4, "methylidyne", unit, 3) == f"4,4',4''-methylidynetris{{{unit}}}"
    assert _assemble_multiplicative_name(4, "oxy", "(2S)-2-aminopropanoic acid", 2) == \
        "4,4'-oxybis[(2S)-2-aminopropanoic acid]"


# --- the descriptor of a ring substituent is that of the whole molecule: no count of chiral centres relabels the molecule ----

# a bicyclic ring with a double bond at its attachment atom ('C2=c3', an exocyclic double bond to an aromatic-flagged atom of
# the pyrimidine ring): ``centres`` labels it 'Z', rdCIPLabeler 'E'
_CIP_SMILES = "C\\1C/C(=C\\2/C=NC(=N/C2=C/C=C1)NCC3=CCC(C=C3)O)/N4CCC(CC4)C(=O)NC5=CN=CC=C5"


def _bond_labels(mol):
    return {(b.GetBeginAtomIdx(), b.GetEndAtomIdx()): b.GetProp("_CIPCode") for b in mol.GetBonds() if b.HasProp("_CIPCode")}


def test_a_count_of_chiral_centres_leaves_the_cip_labels_of_the_molecule():
    """``Chem.FindMolChiralCenters`` relabels the molecule it is given with rdCIPLabeler, over the labels of
    ``assign_stereochemistry`` (``centres``); the two differ at an exocyclic double bond to an aromatic-flagged ring atom.
    ``bridged_fused_pin._stereo_from_rules_4_and_5`` counted the centres of the shared molecule, so every name built after
    it cited 'E' for the 'Z' of the whole molecule. ``find_chiral_centres`` counts them on a copy."""
    from rdkit import Chem

    from orthonym.perception.stereo import assign_stereochemistry, find_chiral_centres
    from orthonym.rules.bridged_fused_pin import _stereo_from_rules_4_and_5, _stereo_outside

    mol = Chem.MolFromSmiles(_CIP_SMILES)
    assign_stereochemistry(mol)
    before = _bond_labels(mol)
    assert "Z" in before.values(), before
    find_chiral_centres(mol, includeUnassigned=False, useLegacyImplementation=False)
    _stereo_outside(mol, {})
    _stereo_from_rules_4_and_5(mol, {})
    assert _bond_labels(mol) == before
    # the hazard is real: the raw call relabels the molecule (when RDKit stops doing so, the helper is moot)
    raw = Chem.Mol(mol)
    Chem.FindMolChiralCenters(raw, includeUnassigned=False, useLegacyImplementation=False)
    if _bond_labels(raw) == before:
        pytest.skip("this RDKit does not relabel the molecule in FindMolChiralCenters")


def test_a_bicyclic_substituent_of_a_piperidine_carboxamide_cites_the_descriptor_of_the_whole_molecule(monkeypatch):
    """With the lone-ring guard of ``senior_ring_system`` off (the monocycle producer then takes the piperidine of this
    molecule whatever the other rings are), the name of the bicyclic substituent cited '(1E,5E,7E)': the substituent
    namer asked ``bridged_fused_pin`` first, whose count of chiral centres relabelled the shared molecule. The
    descriptor is that of the molecule, 'Z': cited at the front of the substituent prefix)."""
    from orthonym.assembly import monocycle_ring_amide

    monkeypatch.setattr(monocycle_ring_amide, "is_lone_ring", lambda mol, system: True)
    row = _row(_CIP_SMILES, "best-effort")
    assert row["name"] and "(1E,5E,7E)" not in row["name"], row
    assert row["name"].startswith("1-[(1Z,5E,7E)-"), row
    assert row["name"].endswith("-N-(pyridin-3-yl)piperidine-4-carboxamide"), row
    assert_full_rt(row["name"], _CIP_SMILES)
