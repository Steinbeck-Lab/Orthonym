"""v33 Phase 6 completion, Task 3 (C): ester ring-acid re-anchor.

Two spies converged on `name_ester`'s ring-acid branch in
`src/orthonym/rules/esters.py` (`_build_ester_acid_word`'s ring path):

1. nit-1: the ring's own suffix-attachment locant ('-1-') was DROPPED
   whenever a ring substituent was present -- 'methyl
   2-methylcyclohexanecarboxylate' instead of the P-14.3.3-required
   'methyl 2-methylcyclohexane-1-carboxylate'.
2. nit-2: the ring was oriented in ONE arbitrary `RingInfo.AtomRings()`
   direction, never the reverse -- a lowest-locant violation AND a
   DETERMINISM defect (the same molecule, spelled with a different SMILES
   atom order, could flip which direction wins).
3. `acid_is_ring_acid`'s BFS-scanned `acid_atoms` set could wander into a
   second ester's atoms on the same ring; re-anchored on the KNOWN
   carbonyl carbon (`_carbonyl_is_ring_bonded`), as Wave-2 already did at
   one other call site.

Root fix: reuse the free-acid path's own ring-orientation primitive
(`orient_cycloalkane` / `_orient_cycloalkane_with_pg` in
`rules/cycloalkanes.py`, which the plain carboxylic-acid namer already
gets right and stable across SMILES atom orderings) instead of a
single-direction rotation, then insert the P-14.3.3 locant via the
existing `_insert_ring_ester_locant` helper.
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi
from orthonym import Orthonym
from orthonym.validation.opsin_roundtrip import opsin_parse


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


def _rt(smiles: str, name: str) -> bool:
    o = opsin_parse(name)
    if not o:
        return False
    return inchi.MolToInchiKey(Chem.MolFromSmiles(smiles)) == \
        inchi.MolToInchiKey(Chem.MolFromSmiles(o))


@pytest.mark.opsin_gate
def test_ring_locant_present(namer):
    smi = "COC(=O)C1CCCCC1C"
    name = namer.name(smi)
    assert name == "methyl 2-methylcyclohexane-1-carboxylate", name
    assert _rt(smi, name), name


@pytest.mark.opsin_gate
def test_lowest_locant_and_deterministic(namer):
    want = "methyl 2-methylcyclopentane-1-carboxylate"
    smi = "COC(=O)C1CCCC1C"
    for s in {smi, Chem.MolToSmiles(Chem.MolFromSmiles(smi))}:
        n = namer.name(s)
        assert n == want, (s, n)
        assert _rt(s, n), n


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi", [
    "O(C(=O)C1C(CCC1)C)C",
    "C1(C)C(CCC1)C(=O)OC",
    "O(C(=O)C1C(C)CCC1)C",
    "O=C(C1CCCC1C)OC",
    "C(=O)(OC)C1CCCC1C",
    "C1C(C(CC1)C(OC)=O)C",
    "C1(CCCC1C(=O)OC)C",
    "COC(C1CCCC1C)=O",
])
def test_order_invariance_extended(namer, smi):
    # Extra random-atom-order re-spellings of the same molecule (RDKit
    # doRandom=True), beyond the two the brief mandates -- the determinism
    # lock should hold for ALL of them, not just the canonical form.
    want = "methyl 2-methylcyclopentane-1-carboxylate"
    n = namer.name(smi)
    assert n == want, (smi, n)
    assert _rt(smi, n), (smi, n)


@pytest.mark.opsin_gate
def test_regression_aromatic_retained_stem_unchanged(namer):
    smi = "COC(=O)c1ccc(C)cc1"
    name = namer.name(smi)
    assert name == "methyl 4-methylbenzoate", name
    assert _rt(smi, name), name


@pytest.mark.opsin_gate
def test_regression_unsubstituted_ring_acid_no_locant(namer):
    # No ring substituent -> the P-14.3.4 licensed omission still applies;
    # this must NOT gain a '-1-' it doesn't need.
    smi = "COC(=O)C1CCCCC1"
    name = namer.name(smi)
    assert name == "methyl cyclohexanecarboxylate", name
    assert _rt(smi, name), name


@pytest.mark.opsin_gate
def test_regression_wave2_acyloxymethyl_ring_unchanged(namer):
    smi = "COC(=O)C1CCCCC1COC(C)=O"
    name = namer.name(smi)
    assert name == "methyl 2-[(acetyloxy)methyl]cyclohexane-1-carboxylate", name
    assert _rt(smi, name), name


@pytest.mark.opsin_gate
def test_regression_acyloxy_on_ring_unchanged(namer):
    smi = "COC(=O)c1ccc(OC(C)=O)cc1"
    name = namer.name(smi)
    assert name == "methyl 4-(acetyloxy)benzoate", name
    assert _rt(smi, name), name


@pytest.mark.opsin_gate
def test_regression_diacylglycerol_unchanged(namer):
    smi = "CCCCC/C=C\\CCCCCCCC(=O)OC[C@H](CO)OC(=O)CCCCCCC/C=C\\CCCCCCCC"
    name = namer.name(smi)
    assert name == (
        "(2S)-1-hydroxy-3-[(9Z)-pentadec-9-enoyloxy]propan-2-yl "
        "(9Z)-octadec-9-enoate"
    ), name
    assert _rt(smi, name), name
