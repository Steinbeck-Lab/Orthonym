"""A benzene parent whose ring belongs to a ring assembly is not certified as the PIN (lane W2, item 26).

 "Retained names" (the Blue Book),:27722-27724: '4-methoxy-1,1'-biphenyl (PIN)...
[not 1-methoxy-4-phenylbenzene; the biphenyl ring system is senior to a single benzene ring]';
 (:19461): "The senior ring or ring system has the greater number of rings";
(:15542): two or more cyclic systems directly joined are a ring assembly. The assembly handler names a
molecule only when EVERY ring system of it joins the assembly, so a molecule with one more ring
(benzyloxy, pyridinylmethyl, phenoxy) reached the benzene writer, and '2-(benzyloxy)-4-phenylphenol'
and 'N,N-dimethyl-4-phenyl-2-[(pyridin-4-yl)methyl]aniline' were labelled pin_verified.
The name stays at every tier below the PIN (best-effort never goes down); only the label and the
default tier's emission change. The screen is the one the other producers of the project consult
(``rules.ring_assembly_screen``); a parent ring that is NOT a member of the assembly (a pyridine
parent, a benzoic acid whose phenyl-pyridine assembly is elsewhere) keeps its label."""
import pytest
from rdkit import Chem

from orthonym.rules import ring_assembly_screen
from tests.support.default_tier import declined_pin_row, default_tier_row
from tests.support.rt_assert import assert_full_rt, name_best_effort

pytestmark = pytest.mark.opsin_gate

# (smiles, the single-ring name the engine builds and keeps below the PIN)
DEMOTED = [
    ("Oc1ccc(-c2ccccc2)cc1OCc1ccccc1", "2-(benzyloxy)-4-phenylphenol"),                  # the item
    ("CN(C)c1ccc(-c2ccccc2)cc1Cc1ccncc1", "N,N-dimethyl-4-phenyl-2-[(pyridin-4-yl)methyl]aniline"),  # the item
    ("OC(=O)c1ccc(-c2ccccc2)cc1Cc1ccncc1", "4-phenyl-2-[(pyridin-4-yl)methyl]benzoic acid"),
    ("COc1ccc(-c2ccccc2)cc1OCc1ccccc1", "2-(benzyloxy)-1-methoxy-4-phenylbenzene"),
    ("c1ccc(Oc2ccccc2-c2ccccc2)cc1", "1-phenoxy-2-phenylbenzene"),     # the Blue Book row 2-phenoxy-1,1'-biphenyl
]


@pytest.mark.parametrize("smiles,name", DEMOTED)
def test_single_ring_parent_of_an_assembly_member_is_not_a_pin(smiles, name):
    row = declined_pin_row(smiles)                  # default tier declines; best-effort gives the same name
    assert row["name"] == name, row
    assert row["tier"] != "pin_verified" and row["is_pin"] is False, row
    be = name_best_effort(smiles)
    assert be["name"] == name and be["tier"] != "pin_verified", be
    assert_full_rt(name, smiles)


def test_the_ester_and_the_salt_of_such_an_acid_follow_the_acid():
    for smiles in ("COC(=O)c1ccc(-c2ccccc2)cc1Cc1ccncc1",
                   "[Na+].[O-]C(=O)c1ccc(-c2ccccc2)cc1Cc1ccncc1"):
        row = default_tier_row(smiles)
        assert row["tier"] != "pin_verified", row
        be = name_best_effort(smiles)
        assert be["tier"] != "pin_verified", be
        assert_full_rt(be["name"], smiles)


KEEPS = [
    ("Oc1ccc(-c2ccccc2)cc1", "[1,1'-biphenyl]-4-ol"),                                  # the assembly is the parent
    ("Oc1ccc(-c2ccccc2)cc1OC", "3-methoxy[1,1'-biphenyl]-4-ol"),
    ("Clc1ccc(-c2ccccc2)cc1Cc1ccncc1", "4-[(4-chloro[1,1'-biphenyl]-3-yl)methyl]pyridine"),  # pyridine is senior
    ("OC(=O)c1ccncc1Cc1ccc(OC)cc1-c1ccccc1",
     "3-[(5-methoxy[1,1'-biphenyl]-2-yl)methyl]pyridine-4-carboxylic acid"),
    ("OC(=O)c1ccc(-c2ccccn2)cc1Cc1ccncc1", "4-(pyridin-2-yl)-2-[(pyridin-4-yl)methyl]benzoic acid"),  # no assembly
    ("Oc1ccc(-c2ccccn2)cc1OCc1ccccc1", "2-(benzyloxy)-4-(pyridin-2-yl)phenol"),
]


@pytest.mark.parametrize("smiles,name", KEEPS)
def test_parents_outside_an_assembly_keep_their_label(smiles, name):
    row = default_tier_row(smiles)
    assert row["name"] == name and row["tier"] == "pin_verified", row
    assert_full_rt(name, smiles)


SCREEN = [
    # (smiles, SMARTS whose atom map 1 is an atom of the ring asked about, answer)
    ("Oc1ccc(-c2ccccc2)cc1OCc1ccccc1", "[OH][c:1]", True),            # phenol ring: member of the biphenyl
    ("Oc1ccc(-c2ccccc2)cc1OCc1ccccc1", "[cH:1]1[cH][cH][cH][cH]c1-c", True),  # the phenyl of the biphenyl
    ("Oc1ccc(-c2ccccc2)cc1OCc1ccccc1", "[c:1]1ccccc1CO", False),      # the benzyl ring: not joined to a ring
    ("Oc1ccc(-c2ccccn2)cc1OCc1ccccc1", "[OH][c:1]", False),           # phenyl-pyridine: different skeletons
    ("OC(=O)c1ccc(-c2ccccc2)cc1Cc1ccncc1", "C(=O)(O)[c:1]", True),
    ("CCOC(=O)c1ccc(cc1)C1CCCCC1", "C(=O)(O)[c:1]", False),            # benzene + cyclohexane:,:24153
    ("OC(=O)c1ccc(-c2ccc(-c3ccccn3)cc2)cc1", "C(=O)(O)[c:1]", True),   # the benzene pair inside a longer chain
    ("OC(=O)c1ccc(C2=CCCCC2)cc1", "C(=O)(O)[c:1]", False),             # benzene + cyclohexene: hydro assembly, not this
]


@pytest.mark.parametrize("smiles,pick,answer", SCREEN)
def test_ring_is_in_biphenyl_assembly(smiles, pick, answer):
    from orthonym.rules.ring_assembly_screen import ring_is_in_biphenyl_assembly
    mol = Chem.MolFromSmiles(smiles)
    query = Chem.MolFromSmarts(pick)
    mapped = next(a.GetIdx() for a in query.GetAtoms() if a.GetAtomMapNum() == 1)
    atom = mol.GetSubstructMatch(query)[mapped]
    ring = next(r for r in mol.GetRingInfo().AtomRings() if atom in r)
    assert ring_is_in_biphenyl_assembly(mol, ring) is answer


def test_ring_screen_fails_closed(monkeypatch):
    from orthonym.rules.ring_assembly_screen import ring_is_in_biphenyl_assembly
    def boom(mol):
        raise RuntimeError("x")
    monkeypatch.setattr(ring_assembly_screen, "_assemblies", boom)
    mol = Chem.MolFromSmiles("Oc1ccc(-c2ccccn2)cc1")
    assert ring_is_in_biphenyl_assembly(mol, mol.GetRingInfo().AtomRings()[0]) is True


# Review F1: a speculative benzene-parent candidate (named while the engine picks a parent ring) must
# not demote an unrelated shipped name; the label is recorded only where the chosen name ships.
SPECULATIVE = [
    ("O=S(=O)(Nc1ccc(-c2ccccc2)cc1)c1ccccc1", "N-([1,1'-biphenyl]-4-yl)benzenesulfonamide"),
    ("O=C(Nc1ccccc1)Nc1ccc(-c2ccccc2)cc1", "N-([1,1'-biphenyl]-4-yl)-N'-phenylurea"),
]


@pytest.mark.parametrize("smiles,name", SPECULATIVE)
def test_a_speculative_candidate_does_not_demote_the_shipped_pin(smiles, name):
    row = default_tier_row(smiles)
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    be = name_best_effort(smiles)
    assert (be["name"], be["tier"]) == (name, "pin_verified"), be
    assert_full_rt(name, smiles)
