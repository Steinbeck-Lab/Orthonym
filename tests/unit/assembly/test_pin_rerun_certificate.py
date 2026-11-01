"""DT3 step C: a kept re-run name keeps the PIN label only when evidence decides every part of it.

``assembly.pin_rerun.rerun_name_certified`` certifies a re-run name that ``Orthonym._name_with_pin_promotion`` keeps:
no class of ``rerun_class_guard`` "INTRODUCTION", the Blue Book: "Preferred IUPAC names (PINs) are not
identified for the compounds in this Chapter"), no spelling check that fails or abstains, and a reader that models every
parent-bearing unit of the name. The guard reads two more fundamental parents here: 'prostane (fundamental parent
structure)' (:51473) and the carotenes, 'β,β-carotene (fundamental parent structure)' (:51459).
"""
import pytest
from rdkit import Chem

from orthonym.assembly.pin_rerun import (
    _carotenoid_like,
    _prostane_like,
    rerun_class_guard,
    rerun_name_certified,
)


def _mol(smiles):
    return Chem.MolFromSmiles(smiles)


PROSTANOIDS = [
    # the prostane skeleton: C1-C7 at C8 of the ring, C13-C20 at C12
    "CCCCCCCC1CCCC1CCCCCCC(=O)O",
    # dinoprost-like: the ring is a 5-oxocyclopentyl, the omega chain a 3-hydroxyoct-1-enyl
    "CCCCC[C@H](O)/C=C/[C@H]1[C@H](O)CC(=O)[C@@H]1C/C=C\\CCCC(=O)[O-]",
    # an analogue whose omega chain ends in a ring
    "CC1=C(CCC1)/C=C/[C@@](C)(C/C=C/[C@H]2[C@@H](CC(=O)[C@@H]2CC/C=C\\CCC(=O)OC)O)O",
]

CAROTENOIDS = [
    # retinol
    "CC1=C(C(C)(C)CCC1)/C=C/C(C)=C/C=C/C(C)=C/CO",
    # a hydroxy-oxo carotene
    "CC1=C(/C=C/C(C)=C/C=C/C(C)=C/C=C/C=C(C)/C=C/C=C(C)/C=C/C2=C(C)C(=O)CCC2(C)C)C(C)(C)C[C@H](O)C1",
]

NEITHER = [
    "CCO",
    "CC(=O)Nc1ccc(O)cc1",
    # a linear tetraene with no methyl branch is no carotenoid
    "C/C=C/C=C/C=C/C=C/C",
    # citral: three double bonds, two conjugated
    "CC(=CCC/C(C)=C/C=O)C",
    # a cyclopentane with short side chains is no prostane
    "CCC1CCCC1CC(=O)O",
    "CCCCCCCC1CCC(CC1)C(=C)C2CCC(CC2)/C=C/C#N",
]


@pytest.mark.parametrize("smiles", PROSTANOIDS)
def test_prostane_skeleton_is_guarded(smiles):
    mol = _mol(smiles)
    assert _prostane_like(mol)
    assert rerun_class_guard(mol, "x") == "prostane skeleton (P-101)"


@pytest.mark.parametrize("smiles", CAROTENOIDS)
def test_carotenoid_polyene_is_guarded(smiles):
    mol = _mol(smiles)
    assert _carotenoid_like(mol)
    assert not _prostane_like(mol)


@pytest.mark.parametrize("smiles", NEITHER)
def test_other_molecules_are_neither(smiles):
    mol = _mol(smiles)
    assert not _prostane_like(mol)
    assert not _carotenoid_like(mol)


def test_a_guarded_class_is_never_certified():
    smiles = PROSTANOIDS[1]
    name = ("(5Z)-7-{(1R,2R,3R)-3-hydroxy-2-[(1E,3S)-3-hydroxyoct-1-en-1-yl]-5-oxocyclopentyl}hept-5-enoate")
    assert rerun_name_certified(_mol(smiles), name) is False


def test_no_structure_or_no_name_is_never_certified():
    assert rerun_name_certified(None, "ethanol") is False
    assert rerun_name_certified(_mol("CCO"), "") is False


def test_an_ylidene_name_is_never_certified():
    """The blind panel of step C: '2-[(1Z,2R,3S,4S)-...cyclohexylidene]propanal' cites the E/Z descriptor of the
    exocyclic double bond inside the prefix; (the Blue Book-48281) puts it at the front with the
    parent locant, '(2Z)-2-[...cyclohexylidene]propanal'. The class of that row is not certified."""
    smiles = ("C[C@H]1CCC(=C)[C@H](C1(C)C)/C=C/C(=C/CC[C@]2([C@@H](/C(=C(/C)\\C=O)/CC[C@]2(C)O)CCCO)C)/C")
    name = ("2-[(1Z,2R,3S,4S)-4-hydroxy-2-(3-hydroxypropyl)-3,4-dimethyl-3-{(3E,5E)-4-methyl-6-[(1R,3S)-2,2,3-"
            "trimethyl-6-methylidenecyclohexyl]hexa-3,5-dien-1-yl}cyclohexylidene]propanal")
    assert rerun_name_certified(_mol(smiles), name) is False


def test_a_name_with_an_abstaining_check_is_never_certified():
    """Review fix: a spelling check that abstains decides nothing, so the name keeps pin_unverified (the quinone: the
    descriptor-locant check abstains on it)."""
    smiles = "COc1cc(C)c(CC/C(C)=C/CC2=CC(=O)C=CC2=O)c(C)c1C"
    name = "2-[(2E)-5-(4-methoxy-2,3,6-trimethylphenyl)-3-methylpent-2-en-1-yl]cyclohexa-2,5-diene-1,4-dione"
    from orthonym.validation.pin_spelling import ABSTAIN, pin_spelling_report
    assert pin_spelling_report(_mol(smiles), name).count(ABSTAIN) > 0
    assert rerun_name_certified(_mol(smiles), name) is False


def test_a_fully_decided_name_is_certified():
    """The SNAC thioester: no guarded class, no failing or abstaining check, full coverage (a blind Blue Book reading
    judged it the PIN,, the Blue Book)."""
    smiles = "CC(=O)NCCSC(=O)/C=C/CCC[C@H](O)/C=C/CC(C)O"
    name = "S-(2-acetamidoethyl) (2E,7S,8E)-7,11-dihydroxydodeca-2,8-dienethioate"
    assert rerun_name_certified(_mol(smiles), name) is True
