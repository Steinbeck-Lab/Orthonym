"""LEADS2 topic 2: a two-ring assembly is the parent of a molecule whose other rings hang on it by a chain, and
the groups of an assembly that are named with the atom that bonds to the ring.

Blue Book, all in ``the Blue Book``:

* ``### **** Retained names`` (:27665),:27722: "4-methoxy-1,1'-biphenyl (PIN) (not 4-phenylanisole; no
  substitution on anisole for PINs) [not 1-methoxy-4-phenylbenzene; the biphenyl ring system is senior to a single
  benzene ring]".
* ``****`` under ``### **** Systematic names of ethers`` (:27764),:27766 "When method (1),
  substitutive nomenclature, is used, the senior ring or ring system must be chosen as the parent hydride (see
  ." and:27768 "(1) 2-phenoxy-1,1'-biphenyl (PIN)".
* ``****`` (:20571) "Seniority criteria for ring assemblies are based on the appropriate criteria for
  rings and ring systems given in ", whose (d) (:19415) is "has the greater number of rings".
* ``### **** Systems composed of rings and chains`` (:19336), (1):19340 "Within the same class, a ring or
  ring system has seniority over a chain."
* ``****`` (:18873) "The senior parent structure has the maximum number of substituents corresponding to the
  principal characteristic group (suffix) or senior parent hydride".
* ``## **** CONSTRUCTION OF LINEAR COMPOUND AND/OR COMPLEX SUBSTITUENT PREFIXES`` (:24583),:24591
  "C6H5-CH2-O- benzyloxy (preferred prefix)"; the sulfur prefixes are printed in PINs: '2-(methanesulfonyl)benzoic
  acid (PIN)' (:31380), '(methanesulfinyl)methane (PIN)' (:46154), '1-{[2-(ethylsulfanyl)-1-(propylsulfanyl)ethen-1-
  yl]sulfanyl}propane (PIN; substitutive name)' (:6415).
* ``### **** NUMBERING`` (:3219), (g):3303 "lowest locants for the substituent cited first as a prefix in the
  name", and:3193 "Primed locants are placed immediately after the corresponding unprimed locants in a set arranged
  in ascending order".

change-asserted-value (three artifacts, as the skill asks), for the names that moved from a single-ring name to the
assembly PIN ('1-(benzyloxy)-4-phenylbenzene' and the like in ``test_nq_labels2_biphenyl_parent_label.py``;
'1-cyclohexyl...' rows of ``test_ring_ylidene_parent.py``): (1) the quotes above; (2) every expected name is read back
by a FRESH OPSIN call to the input structure (``assert_pin_at_both_tiers`` -> ``name_is_rt_exact``), independent of the
engine's own gate; (3) mutation: ``test_without_the_detector_the_rows_fall_back`` switches the detector off and
checks that the rows lose the name. What would make these wrong: a Blue Book reading in which a third ring on a chain
changes the parent (none found; compares ring systems and a ring on a chain is one more single ring).
"""
import pytest

from orthonym.rules import assembly_ring_branches
from tests.support.pin_tiers import assert_pin_at_both_tiers, name_default

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

RING_ON_A_CHAIN = [
    ("c1ccc(cc1)COc1ccc(cc1)-c1ccccc1", "4-(benzyloxy)-1,1'-biphenyl"),
    ("C1CCC(CC1)COc1ccc(cc1)-c1ccccc1", "4-(cyclohexylmethoxy)-1,1'-biphenyl"),
    ("c1ccc(cc1)Cc1ccc(cc1)-c1ccccc1", "4-benzyl-1,1'-biphenyl"),
    ("c1ccc(cc1)Oc1ccc(cc1)-c1ccccc1", "4-phenoxy-1,1'-biphenyl"),
    ("c1ccc(Oc2ccccc2-c2ccccc2)cc1", "2-phenoxy-1,1'-biphenyl"),                        #:27768 verbatim
    ("c1ccc(cc1)CCc1ccc(cc1)-c1ccccc1", "4-(2-phenylethyl)-1,1'-biphenyl"),
    ("C(=C1CCCCC1)c1ccc(cc1)c1ccccc1", "4-(cyclohexylidenemethyl)-1,1'-biphenyl"),
    ("C(C1CCCCC1)c1ccc(cc1)c1ccccc1", "4-(cyclohexylmethyl)-1,1'-biphenyl"),
    ("c1ccc2cc(COc3ccc(cc3)-c3ccccc3)ccc2c1", "4-[(naphthalen-2-yl)methoxy]-1,1'-biphenyl"),  # 12 atoms > 10
    ("Clc1ccc(COc2ccc(cc2)-c2ccccc2)cc1", "4-[(4-chlorophenyl)methoxy]-1,1'-biphenyl"),
    ("c1ccc(Oc2ccc(cc2)-c2ccc(Oc3ccccc3)cc2)cc1", "4,4'-diphenoxy-1,1'-biphenyl"),
    ("COc1ccc(-c2ccccc2)cc1OCc1ccccc1", "3-(benzyloxy)-4-methoxy-1,1'-biphenyl"),
    # a principal characteristic group on the assembly: it is the parent, the suffix on it
    ("Oc1ccc(-c2ccccc2)cc1OCc1ccccc1", "3-(benzyloxy)[1,1'-biphenyl]-4-ol"),
    ("OC(=O)c1ccc(cc1)-c1ccc(OCc2ccccc2)cc1", "4'-(benzyloxy)[1,1'-biphenyl]-4-carboxylic acid"),
    ("OC(=O)c1ccc(cc1)-c1ccc(Cc2ccccc2)cc1", "4'-benzyl[1,1'-biphenyl]-4-carboxylic acid"),
    ("Nc1ccc(cc1)-c1ccc(OCc2ccccc2)cc1", "4'-(benzyloxy)[1,1'-biphenyl]-4-amine"),
]


@pytest.mark.parametrize("smiles,pin", RING_ON_A_CHAIN)
def test_the_assembly_is_the_parent_and_the_ring_on_the_chain_is_cited_in_a_prefix(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


SULFUR_GROUPS = [
    ("CCCCCCSc1ccc(cc1)-c1ccccc1", "4-(hexylsulfanyl)-1,1'-biphenyl"),
    ("CCSc1ccc(cc1)-c1ccccc1", "4-(ethylsulfanyl)-1,1'-biphenyl"),
    ("CC(C)Sc1ccc(cc1)-c1ccccc1", "4-[(propan-2-yl)sulfanyl]-1,1'-biphenyl"),
    ("c1ccc(cc1)Sc1ccc(cc1)-c1ccccc1", "4-(phenylsulfanyl)-1,1'-biphenyl"),
    ("CS(=O)(=O)c1ccc(cc1)-c1ccccc1", "4-(methanesulfonyl)-1,1'-biphenyl"),
    ("CS(=O)c1ccc(cc1)-c1ccccc1", "4-(methanesulfinyl)-1,1'-biphenyl"),
]


@pytest.mark.parametrize("smiles,pin", SULFUR_GROUPS)
def test_a_group_bonded_through_sulfur_is_named_with_the_atom_that_bonds_to_the_ring(smiles, pin):
    """The fallback of ``_name_substituent`` named the fragment alone: '-SC2H5' gave None, '-NHCH3' gave
    'aminomethyl', '-OC(=O)CH3' gave 'acetyl'. The substituent namer reads the attachment atom."""
    assert_pin_at_both_tiers(smiles, pin)


def test_the_substituent_namer_reads_the_attachment_atom():
    from rdkit import Chem

    from orthonym.rules import ring_assemblies as ra
    cases = {"CNc1ccc(cc1)-c1ccccc1": "methylamino", "CC(=O)Oc1ccc(cc1)-c1ccccc1": "acetyloxy",
             "CCSc1ccc(cc1)-c1ccccc1": "ethylsulfanyl", "CC(=O)Nc1ccc(cc1)-c1ccccc1": "acetamido"}
    for smiles, token in cases.items():
        mol = Chem.MolFromSmiles(smiles)
        ring = {a for r in mol.GetRingInfo().AtomRings() for a in r}
        start = next(nb.GetIdx() for a in ring for nb in mol.GetAtomWithIdx(a).GetNeighbors()
                     if nb.GetIdx() not in ring)
        sub = ra._walk_substituent(mol, start, ring)
        assert ra._name_substituent(mol, sub, start) == token, smiles


TIES = [
    # (g): the prefix cited first takes the unprimed locant at equal numbers; either spelling of the input
    ("Brc1ccc(cc1)-c1ccc(Cl)cc1", "4-bromo-4'-chloro-1,1'-biphenyl"),
    ("Clc1ccc(cc1)-c1ccc(Br)cc1", "4-bromo-4'-chloro-1,1'-biphenyl"),
    ("Clc1ccc(cc1)-c1ccc(OC)cc1", "4-chloro-4'-methoxy-1,1'-biphenyl"),
    ("COc1ccc(cc1)-c1ccc(Cl)cc1", "4-chloro-4'-methoxy-1,1'-biphenyl"),
    ("c1ccc(cc1)COc1ccc(cc1)-c1ccc(Cl)cc1", "4-(benzyloxy)-4'-chloro-1,1'-biphenyl"),
    ("Clc1ccc(cc1)-c1ccc(OCc2ccccc2)cc1", "4-(benzyloxy)-4'-chloro-1,1'-biphenyl"),
    ("CCCCCCSc1ccc(cc1)-c1ccc(Cl)cc1", "4-chloro-4'-(hexylsulfanyl)-1,1'-biphenyl"),
]


@pytest.mark.parametrize("smiles,pin", TIES)
def test_the_prefix_cited_first_takes_the_unprimed_locant(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


# A molecule the detector does not claim keeps the name and the label it had.
NOT_CLAIMED = [
    # a ring bonded to the assembly directly: / (:24153), not a ring on a chain
    "C1CCC(CC1)c1ccc(cc1)-c1ccccc1",
    # the other ring is senior (a): a heterocycle), not junior to the assembly
    "c1cnccc1COc1ccc(cc1)-c1ccccc1",
    # two assemblies tie (e) and the type order give no decision)
    "c1ccc(cc1)-c1ccc(COc2ccc(cc2)-c2ccccc2)cc1",
    # a stereocentre; a nitro group (a nitrogen atom outside the assembly is not read)
    "C[C@H](Oc1ccc(cc1)-c1ccccc1)c1ccccc1",
    "O=[N+]([O-])c1ccc(cc1)-c1ccc(OCc2ccccc2)cc1",
]


@pytest.mark.parametrize("smiles", NOT_CLAIMED)
def test_a_molecule_outside_the_class_is_not_named_on_the_assembly(smiles):
    assert name_default(smiles).get("producer") != "inner:ring_assembly"


PCG_AS_PREFIX = [
    ("NC(=O)c1ccc(cc1)-c1ccccc1", "4-carbamoyl-1,1'-biphenyl"),
    ("Sc1ccc(cc1)-c1ccccc1", "4-sulfanyl-1,1'-biphenyl"),
]


@pytest.mark.parametrize("smiles,old_name", PCG_AS_PREFIX)
def test_a_principal_group_is_never_cited_as_a_prefix_of_an_assembly_pin(smiles, old_name):
    """ (:18873): the PIN of 4-biphenylcarboxamide is '[1,1'-biphenyl]-4-carboxamide', the carboxamide being
    the suffix of the parent; '4-carbamoyl-1,1'-biphenyl' was labelled pin_verified. The name still describes the
    molecule, so it keeps shipping, labelled below the PIN (a fallback is demoted, never withdrawn)."""
    from tests.support.rt_assert import name_best_effort, name_is_rt_exact
    row = name_default(smiles)
    assert row.get("tier") != "pin_verified", row
    be = name_best_effort(smiles)
    assert be.get("tier") != "pin_verified", be
    assert name_is_rt_exact(be["name"], smiles), be


@pytest.mark.parametrize("smiles", ["CNc1ccc(cc1)-c1ccccc1", "CCCCCCNc1ccc(cc1)-c1ccccc1",
                                    "OC(=O)c1ccc(cc1)-c1ccc(SCCCCCC)cc1"])
def test_widening_the_substituent_namer_adds_no_false_pin(smiles):
    """The substituent namer names '-NHCH3' as 'methylamino' now; the amine is the principal group, so
    '4-(methylamino)-1,1'-biphenyl' must not ship as the PIN (the PIN is 'N-methyl[1,1'-biphenyl]-4-amine')."""
    from tests.support.rt_assert import name_best_effort, name_is_rt_exact
    assert name_default(smiles).get("tier") != "pin_verified"
    be = name_best_effort(smiles)
    assert be.get("tier") != "pin_verified", be
    assert name_is_rt_exact(be["name"], smiles), be


def test_without_the_detector_the_rows_fall_back(monkeypatch):
    """Mutation: with the detector off the benzyloxy row is declined at the default tier again."""
    monkeypatch.setattr(assembly_ring_branches, "detect_assembly_with_ring_branches", lambda *a, **k: None)
    row = name_default("c1ccc(cc1)COc1ccc(cc1)-c1ccccc1")
    assert row.get("tier") != "pin_verified", row
