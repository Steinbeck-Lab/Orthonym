"""W2F-P8 Task 7: fail-closed boundaries.

Accuracy #1 / fail-closed: an input outside the built envelope must refuse, never
emit a structure-dropping or stereo-lossy name.

NOTE on the test environment: conftest disables the OPSIN validity gate in pytest
(SUB-03/), so the WHOLE-MOLECULE 'unknown organic compound' suppression (which
in production is enforced by the SELF-01 / OPSIN-parse gate) is NOT observable
here. These unit tests therefore assert the GATE-INDEPENDENT source refusals my
code owns; the production fail-closed result ('unknown organic compound') for the
whole molecules is verified by the the gold set PROTECT rows and by scripts/diagnose
(gate ON): both `Brc1ccccc1Nc1ccccn1` and
`C(C)(Cl)c1ccc(Oc2ccc(S(=O)(=O)C)cc2)cc1` -> 'unknown organic compound'.
"""
from rdkit import Chem

import orthonym
from orthonym.rules.ring_substituents import decorated_ring_substituent_name


def test_unnameable_arm_core_refuses():
    # the substituted-aryloxy core namer returns None for a ring bearing a
    # methylsulfonyl (outside the v1 table) -> the phenoxy builder fails closed
    # rather than dropping the sulfonyl to a bare 'phenoxy'. (ring 8..17 is the
    # sulfonyl-decorated phenoxy ring; attach = the ether-O-bonded carbon 8.)
    mol = Chem.MolFromSmiles("C(C)(Cl)c1ccc(Oc2ccc(S(=O)(=O)C)cc2)cc1")
    assert decorated_ring_substituent_name(
        mol, (9, 10, 11, 16, 17, 8), 8) is None


def test_heteroaryl_n_ring_promotion_declines():
    # the substituted-N-aryl promotion is scoped to a carbocyclic benzene ring;
    # a pyridine N-ring must NOT be forced to 'N-(pyridin-2-yl)' — the core
    # namer refuses the heteroaryl ring so the promotion branch declines.
    mol = Chem.MolFromSmiles("Brc1ccccc1Nc1ccccn1")
    # pyridine ring atoms 8..13 (the N-aryl); attach = carbon bonded to the amine N
    py_ring = next(r for r in mol.GetRingInfo().AtomRings()
                   if any(mol.GetAtomWithIdx(i).GetSymbol() == 'N' for i in r))
    attach = next(
        i for i in py_ring
        for nb in mol.GetAtomWithIdx(i).GetNeighbors()
        if nb.GetSymbol() == 'N' and nb.GetIdx() not in py_ring
    )
    assert decorated_ring_substituent_name(mol, py_ring, attach) is None


def test_stereo_lossy_ether_not_emitted():
    # the R/S diaryl ether carries BOTH descriptors (a stereo-lossy string parses
    # to a different, achiral molecule and must never be emitted).
    name = orthonym.name_compound(
        "C[C@H](Cl)c1ccc(Oc2ccc(cc2)[C@@H](C)Cl)cc1", style="pin")
    assert "(1R)" in name and "(1S)" in name
