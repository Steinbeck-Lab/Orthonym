"""_charge_forms takes its drawings from one function and filters them in another.

the contributor guide core principle 4 "Determinism": the same input always yields the same output. A cation whose +1
sits on one of several nitrogen atoms is one species whatever SMILES draws it, and the name is built from
the hydron site: "General rule for systematically naming cationic centers in parent hydrides"
(the Blue Book),:41368: "A cation derived formally by adding one or more hydrons to any position
of a neutral parent hydride... is named by replacing the final letter 'e' of the parent hydride name, if
any, by the suffix 'ium'". ``_rdkit_charge_candidates`` is the one place RDKit enumerates the drawings;
``_accepted_charge_forms`` decides which drawings are forms of the species; ``_charge_forms`` keeps every
bound. The form sets pinned here are the forms RDKit's enumeration yields for each constructed molecule.
"""
import gc

import pytest
from rdkit import Chem

from orthonym import namer
from tests.support.resonance_spy import supplier_calls

pytestmark = pytest.mark.unit

# input -> the forms of its species, one SMILES each (compared as canonical SMILES)
FORMS = {
    "Cc1c[nH]c[nH+]1": ["Cc1c[nH+]c[nH]1", "Cc1c[nH]c[nH+]1"],
    "Cc1c[nH+]c[nH]1": ["Cc1c[nH+]c[nH]1", "Cc1c[nH]c[nH+]1"],
    "CCCCn1cc[n+](C)c1": ["CCCC[n+]1ccn(C)c1", "CCCCn1cc[n+](C)c1"],
    "NC(=[NH2+])NC": ["CNC(N)=[NH2+]", "C[NH+]=C(N)N"],
    "CC(=[NH+]C)N": ["CNC(C)=[NH2+]", "C[NH+]=C(C)N"],
    "CCCN(CCC)C(=[N+](CCC)CCC)N": ["CCCN(CCC)C(=[NH2+])N(CCC)CCC", "CCCN(CCC)C(N)=[N+](CCC)CCC"],
    "CNC(=[NH2+])NCc1ccccc1": ["CNC(=[NH2+])NCc1ccccc1", "CNC(N)=[NH+]Cc1ccccc1", "C[NH+]=C(N)NCc1ccccc1"],
    "NC(=[NH2+])NCC(=O)[O-]": ["NC(=[NH2+])NCC(=O)[O-]", "NC(N)=[NH+]CC(=O)[O-]"],
    "Clc1ccc2[nH]c[nH+]c2c1": ["Clc1ccc2[nH+]c[nH]c2c1", "Clc1ccc2[nH]c[nH+]c2c1"],
    "Cc1cc[nH][nH+]1": ["Cc1cc[nH+][nH]1", "Cc1cc[nH][nH+]1"],
    "C[n+]1c[nH]c2ccccc21": ["C[n+]1c[nH]c2ccccc21", "Cn1c[nH+]c2ccccc21"],
    "C[N+]1=CNc2ccccc21": ["C[n+]1c[nH]c2ccccc21", "Cn1c[nH+]c2ccccc21"],
    "Cc1ccnn2c(C)n[n+](C)c12": ["Cc1ccn[n+]2c(C)nn(C)c12", "Cc1ccnn2c(C)n[n+](C)c12"],
}

# pass the pre-checks and reach RDKit, but the drawing has no second charge-only form
NO_SECOND_FORM = [
    "c1cc[nH+]cc1",                 # one nitrogen
    "C[n+]1ccccc1",                 # one nitrogen
    "Nc1cc[nH+]cc1",                # the charge would move through the ring to the amino group
    "[NH2+]=c1cc[nH]cc1",           # the iminium drawing of the same class
    # an azide on N: its internal +1 is no mobile charge (perception/charge_shift.py); the drawing
    # C[N+](C)=NN=[N-] places the charges on other atoms and is no form of this input. "AZIDES"
    # (the Blue Book) names the group by the prefix 'azido', and (:43298) gives the
    # zwitterion parent only as method (3), no PIN.
    "CN(C)N=[N+]=[N-]",
]

# named as drawn before any enumeration: no charge, no mobile N+, a CXSMILES string, a lone-pair
# stereocentre, more than _FORMS_MAX_HEAVY heavy atoms
NOT_ASKED = [
    "CCO",
    "C[N+](=O)[O-]",
    "C[N+](C)(C)C",
    "C[NH3+]",
    "C[C@H](O)CC |&1:1|",
    "C[S@](=O)CC.CC(=[NH+]C)N",
    ".".join(["CC(=[NH+]C)N"] * 20),
]


@pytest.fixture(autouse=True)
def _fresh_cache():
    namer._charge_forms.cache_clear()
    yield
    namer._charge_forms.cache_clear()


def _canonical(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


@pytest.mark.parametrize("smiles", list(FORMS))
def test_charge_forms_is_the_filter_over_the_rdkit_candidates(smiles):
    mol = Chem.MolFromSmiles(smiles)
    found = namer._accepted_charge_forms(mol, smiles, namer._rdkit_charge_candidates(mol))
    forms = namer._charge_forms(smiles)
    assert forms == tuple(found[canonical] for canonical in sorted(found))
    assert {_canonical(f[0]) for f in forms} == {_canonical(s) for s in FORMS[smiles]}


@pytest.mark.parametrize("smiles", NO_SECOND_FORM)
def test_a_drawing_with_no_other_form_has_only_itself_after_the_filter(smiles):
    mol = Chem.MolFromSmiles(smiles)
    found = namer._accepted_charge_forms(mol, smiles, namer._rdkit_charge_candidates(mol))
    assert list(found) == [Chem.MolToSmiles(mol)] and found[Chem.MolToSmiles(mol)] == (smiles, None)
    assert namer._charge_forms(smiles) is None


@pytest.mark.parametrize("smiles", NOT_ASKED)
def test_an_input_outside_the_pre_checks_is_named_as_drawn_without_enumeration(smiles):
    with supplier_calls() as calls:
        assert namer._charge_forms(smiles) is None
    assert calls == []


@pytest.mark.parametrize("smiles", list(FORMS))
def test_one_supplier_per_call_and_it_is_asked_by_the_candidate_source(smiles):
    mol = Chem.MolFromSmiles(smiles)
    with supplier_calls() as calls:
        namer._rdkit_charge_candidates(mol)
    assert calls == [("_rdkit_charge_candidates", mol.GetNumHeavyAtoms())]
    with supplier_calls() as calls:
        namer._charge_forms(smiles)
    assert calls == [("_charge_forms", mol.GetNumHeavyAtoms())]


def test_the_candidates_outlive_the_supplier_that_made_them():
    mol = Chem.MolFromSmiles("Cc1c[nH]c[nH+]1")
    candidates = namer._rdkit_charge_candidates(mol)
    gc.collect()
    assert len(candidates) >= 2
    for candidate in candidates:
        assert candidate is None or (candidate.GetNumAtoms() == mol.GetNumAtoms() and Chem.MolToSmiles(candidate))


def test_the_filter_asks_rdkit_nothing():
    smiles = "Cc1c[nH]c[nH+]1"
    mol = Chem.MolFromSmiles(smiles)
    other = Chem.MolFromSmiles("Cc1c[nH+]c[nH]1")
    with supplier_calls() as calls:
        found = namer._accepted_charge_forms(mol, smiles, [mol, None, other])
    assert calls == []
    drawn, second = Chem.MolToSmiles(mol), Chem.MolToSmiles(other)
    assert list(found) == [drawn, second] and found[drawn] == (smiles, None)
    written, order = found[second]
    assert _canonical(written) == second and len(order) == mol.GetNumAtoms()


@pytest.mark.parametrize("candidate", [
    "Cc1c[nH]cn1",          # neutral: the hydrogen count and the charge differ
    "Cc1[nH]cc[nH+]1",      # another species: 2-methyl, not 4-methyl
    "CCO",                  # another molecule
])
def test_the_filter_rejects_a_drawing_of_another_species(candidate):
    smiles = "Cc1c[nH]c[nH+]1"
    mol = Chem.MolFromSmiles(smiles)
    found = namer._accepted_charge_forms(mol, smiles, [Chem.MolFromSmiles(candidate)])
    assert list(found) == [Chem.MolToSmiles(mol)]


def test_charge_forms_asks_only_the_candidate_source(monkeypatch):
    smiles = "Cc1c[nH]c[nH+]1"
    seen = []

    def source(mol):
        seen.append(Chem.MolToSmiles(mol))
        return [Chem.MolFromSmiles("Cc1c[nH+]c[nH]1")]

    monkeypatch.setattr(namer, "_rdkit_charge_candidates", source)
    with supplier_calls() as calls:
        forms = namer._charge_forms(smiles)
    assert calls == [] and seen == [_canonical(smiles)]
    assert {_canonical(f[0]) for f in forms} == {_canonical(s) for s in FORMS[smiles]}


@pytest.mark.parametrize("source", [lambda mol: [], lambda mol: [None], lambda mol: [mol]])
def test_a_source_with_no_other_drawing_gives_no_forms(source, monkeypatch):
    monkeypatch.setattr(namer, "_rdkit_charge_candidates", source)
    assert namer._charge_forms("Cc1c[nH]c[nH+]1") is None


def test_a_source_that_raises_gives_no_forms(monkeypatch):
    def source(mol):
        raise RuntimeError("enumeration failed")

    monkeypatch.setattr(namer, "_rdkit_charge_candidates", source)
    assert namer._charge_forms("Cc1c[nH]c[nH+]1") is None


@pytest.mark.parametrize("bound, value", [("_FORMS_MAX", 1), ("_FORMS_MAX_WORK", 1)])
def test_the_bounds_stay_in_charge_forms(bound, value, monkeypatch):
    smiles = "Cc1c[nH]c[nH+]1"
    assert len(namer._charge_forms(smiles)) == 2
    namer._charge_forms.cache_clear()
    monkeypatch.setattr(namer, bound, value)
    assert namer._charge_forms(smiles) is None


def test_identical_groups_give_one_form_per_number_of_moved_charges():
    # 4 amidinium groups have 16 drawings; the forms are keyed by canonical SMILES, so the 5
    # distinct placements of the +1 over identical groups are the forms
    smiles = ".".join(["CC(=[NH+]C)N"] * 4)
    forms = namer._charge_forms(smiles)
    assert len(forms) == 5 and len({_canonical(f[0]) for f in forms}) == 5
