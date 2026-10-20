"""charge_shift finds the charge-only resonance forms of a conjugated cation by matching its pi bonds.

the contributor guide core principle 4 "Determinism": a cation whose +1 sits on one of several nitrogen atoms is one species
whatever SMILES draws it. The hydron site that names it is the one of "General rule for systematically
naming cationic centers in parent hydrides" (the Blue Book),:41368: "A cation derived formally by adding one or
more hydrons to any position of a neutral parent hydride... is named by replacing the final letter 'e' of the parent
hydride name, if any, by the suffix 'ium'". The oracle for the forms is RDKit's own enumeration on constructed molecules
small enough to finish; the ladders, the user molecules and the fixture rows of the flags-0 family are the large cases.
Rows taken from the benchmark corpora are in tests_local/unit/perception/test_charge_shift_local.py.

A carbon with a negative charge in the pi system of the cation (an N-heterocyclic carbene drawn [N+]=[C-], an N-ylide,
an isocyanide or a dicyanomethanide on the ring) is a different case: RDKit writes eight of the nine groups of
CARBANION_SYSTEMS below as one structure; for the dicyanomethanide it writes five, one of them with the +1 on the other
nitrogen and the carbanion's charge on a nitrile nitrogen, which the filter of ``accepted_charge_forms`` rejects (only
nitrogen atoms change their charge). The forms RDKit gives are the drawn one alone, and the helper must not offer one
that RDKit's filtered enumeration does not.
"""
import inspect
import random

import pytest
from rdkit import Chem

from orthonym import namer
from orthonym.perception import charge_shift as cs
from orthonym.rules import ions
from orthonym.validation import protonation_identity as protonation
from tests.support.resonance_rows import azide_ladder, carbene_ladder, imidazolium_ladder, n_azide_ladder, rows

pytestmark = pytest.mark.unit


def _brute(n, edges):
    adj = {i: set() for i in range(n)}
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)

    def rec(free):
        if not free:
            return True
        v = min(free)
        return any(rec(free - {v, w}) for w in adj[v] & free)

    return rec(frozenset(range(n)))


def _canon(forms, mol):
    """The canonical SMILES of each form after it is written and read back, as _accepted_charge_forms keys them,
    without the drawn form's."""
    drawn = Chem.MolToSmiles(mol)
    back = (Chem.MolFromSmiles(Chem.MolToSmiles(f, canonical=False)) for f in forms)
    return {Chem.MolToSmiles(b) for b in back if b is not None} - {drawn}


def _helper(smiles, why=None, **kwargs):
    mol = Chem.MolFromSmiles(smiles)
    forms = cs.charge_shift_forms(mol, why=why, **kwargs)
    return None if forms is None else _canon(forms, mol)


def _rdkit(smiles):
    mol = Chem.MolFromSmiles(smiles)
    found = namer._accepted_charge_forms(mol, smiles, namer._rdkit_charge_candidates(mol))
    return set(found) - {Chem.MolToSmiles(mol)}


def _centre(mol):
    return next(a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge() == 1)


def _atom(mol, smarts):
    return mol.GetSubstructMatch(Chem.MolFromSmarts(smarts))[0]


# input -> the forms of its species (canonical SMILES), the drawn form excluded; each equals RDKit's enumeration
FORMS = {
    "Cc1c[nH]c[nH+]1": {"Cc1c[nH+]c[nH]1"},
    "CCCCn1cc[n+](C)c1": {"CCCC[n+]1ccn(C)c1"},
    "NC(=[NH2+])NC": {"C[NH+]=C(N)N"},
    "CC(=[NH+]C)N": {"CNC(C)=[NH2+]"},
    "CCCN(CCC)C(=[N+](CCC)CCC)N": {"CCCN(CCC)C(=[NH2+])N(CCC)CCC"},
    "CNC(=[NH2+])NCc1ccccc1": {"CNC(N)=[NH+]Cc1ccccc1", "C[NH+]=C(N)NCc1ccccc1"},
    "NC(=[NH2+])NCC(=O)[O-]": {"NC(N)=[NH+]CC(=O)[O-]"},
    "Clc1ccc2[nH]c[nH+]c2c1": {"Clc1ccc2[nH+]c[nH]c2c1"},
    "Cc1cc[nH][nH+]1": {"Cc1cc[nH+][nH]1"},
    "C[n+]1c[nH]c2ccccc21": {"Cn1c[nH+]c2ccccc21"},
    "C[N+]1=CNc2ccccc21": {"Cn1c[nH+]c2ccccc21"},
    "Cc1ccnn2c(C)n[n+](C)c12": {"Cc1ccn[n+]2c(C)nn(C)c12"},
    "C[n+]1cc[nH]c1C(=O)[O-]": {"Cn1cc[nH+]c1C(=O)[O-]"},
    "CNC(=[NH2+])c1ccc(cc1)C(N)=[NH+]C": {"CNC(=[NH2+])c1ccc(C(=[NH2+])NC)cc1", "C[NH+]=C(N)c1ccc(C(N)=[NH+]C)cc1"},
}
# cations that pass the pre-check of _charge_forms and have no second charge-only form
NO_FORM = [
    "CN(C)C=CC=[N+](C)C",                   # a symmetric cyanine: the other drawing is the same canonical SMILES
    "CN(C)C=CC=CC=[N+](C)C",
    "CC(=[NH2+])N",
    "C[n+]1ccn(C)c1",
    "c1cc[n+](C)cc1",
    "NC(=[NH2+])c1ccc(cc1)C(N)=[NH2+]",
    "Nc1cc[nH+]cc1",                        # the iminium drawing has another number of exocyclic double bonds
    "[NH2+]=c1cc[nH]cc1",
]


# a carbon with a negative charge in the pi system of the cation; each has two nitrogen atoms that carry different groups
# or a donor beside the +1, so that a shift would be another canonical SMILES, and RDKit's enumeration gives none
CARBANION_SYSTEMS = [
    "CN1C=C[N+](CC)=[C-]1",                          # imidazol-2-ylidene, drawn with a Kekule structure
    "CC[n+]1[c-]n(C)cc1",                            # the same species, drawn aromatic
    "C[N+]1=[C-]N(CC)c2ccccc21",                     # benzimidazol-2-ylidene
    "C[N+]1=[C-]N(CC)CC1",                           # imidazolidin-2-ylidene
    "C[N+]1=[C-]N(CC)N=C1",                          # 1,2,4-triazol-5-ylidene
    "[CH2-][n+]1ccn(C)c1",                           # an N-ylide on the +1
    "C[n+]1c[c-]n(C)c1",                             # imidazol-4-ylidene, the carbanion beside the other nitrogen
    "C[n+]1cn(CC)c2cc([C-](C#N)C#N)ccc12",           # a dicyanomethanide conjugated with the ring
    "C[n+]1ccc2c(c1)n(C)c1cc([N+]#[C-])ccc12",       # an isocyanide on a carbolinium
]
# the same cations with the carbanion in another pi system: the +1 still moves
CARBANION_ELSEWHERE = [
    "Cc1c[nH]c[nH+]1.[C-]#[N+]c1ccccc1",              # another fragment
    "C[n+]1cn(CC)c2cc(CC[C-](C#N)C#N)ccc12",         # two sp3 carbons between the ring and the carbanion
    "Cc1c[nH]c[nH+]1.C[N+]1=[C-]N(CC)C=C1",          # a carbene in another fragment: only the amidinium forms
]


@pytest.mark.parametrize("seed", range(6))
def test_blossom_agrees_with_brute_force(seed):
    rng = random.Random(seed)
    for _ in range(100):
        n = rng.randint(2, 12)
        edges = {(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.3}
        adj = {i: [] for i in range(n)}
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        got = cs.perfect_matching(set(range(n)), adj)
        assert (got is not None) == _brute(n, edges), (n, sorted(edges))
        if got is not None:
            assert sorted(got) == list(range(n)) and all(got[got[v]] == v and got[v] in adj[v] for v in got)


def test_blossom_matches_through_an_odd_cycle():
    # a five-cycle (the pyrrole ring) with one pendant atom: the only perfect matching goes through the odd cycle
    adj = {0: [1, 4, 5], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3, 0], 5: [0]}
    assert cs.perfect_matching(adj, adj) == {0: 5, 5: 0, 1: 2, 2: 1, 3: 4, 4: 3}


def test_blossom_edge_cases():
    assert cs.perfect_matching([], {}) == {}
    assert cs.perfect_matching([0, 1, 2], {0: [1], 1: [0, 2], 2: [1]}) is None        # an odd number of nodes
    assert cs.perfect_matching([0, 1, 2, 3], {0: [1], 1: [0], 2: [3], 3: [2, 9]}) == {0: 1, 1: 0, 2: 3, 3: 2}
    assert cs.perfect_matching([0, 1, 2, 3], {0: [1, 2, 3], 1: [0], 2: [0], 3: [0]}) is None     # a star


def test_blossom_does_not_depend_on_the_order_of_the_neighbour_lists():
    adj = {i: [(i + 1) % 8, (i - 1) % 8, (i + 4) % 8] for i in range(8)}
    shuffled = {i: random.Random(i).sample(neighbours, 3) for i, neighbours in adj.items()}
    assert cs.perfect_matching(range(8), adj) == cs.perfect_matching(range(8), shuffled)


@pytest.mark.parametrize("smiles", sorted(FORMS) + NO_FORM)
def test_helper_never_misses_a_form_rdkit_gives(smiles):
    mine = _helper(smiles)
    assert mine is not None
    assert _rdkit(smiles) <= mine


@pytest.mark.parametrize("smiles", sorted(FORMS))
def test_form_sets_are_pinned(smiles):
    assert _helper(smiles) == FORMS[smiles]


@pytest.mark.parametrize("smiles", ["CN(C)N=[N+]=[N-]", "C[N+](C)=NN=[N-]", "CCN(CC)N=[N+]=[N-]",
                                    "O=C1NCCCN1c1ccc(N=[N+]=[N-])cc1"])
def test_the_filter_takes_no_form_in_which_an_azido_group_loses_or_gains_the_plus(smiles):
    # The +1 of an azido group does not move (``azide_atoms``, the Blue Book). The helper and RDKit's
    # enumeration still offer the +1 on the neutral amine nitrogen ('C[N+](C)=NN=[N-]') as a candidate; the filter that
    # decides which candidates are forms takes it from neither drawing, whichever of the two is the input.
    mol = Chem.MolFromSmiles(smiles)
    for candidates in (cs.charge_shift_forms(mol), namer._rdkit_charge_candidates(mol)):
        assert set(namer._accepted_charge_forms(mol, smiles, candidates)) == {Chem.MolToSmiles(mol)}


@pytest.mark.parametrize("smiles", NO_FORM)
def test_a_cation_without_a_second_form_has_the_empty_answer(smiles):
    assert cs.charge_shift_forms(Chem.MolFromSmiles(smiles)) == []


@pytest.mark.parametrize("smiles", ["CCO", "c1ccccc1", "C[N+](C)(C)C", "O=[N+]([O-])c1ccccc1", "C[O-].[Na+]"])
def test_a_molecule_without_a_mobile_nitrogen_cation_has_no_form(smiles):
    why = []
    assert cs.charge_shift_forms(Chem.MolFromSmiles(smiles), why=why) == [] and why == []


@pytest.mark.parametrize("smiles", CARBANION_SYSTEMS)
def test_a_carbanion_in_the_pi_system_has_the_forms_rdkit_gives_and_they_are_none(smiles):
    # the oracle is RDKit's enumeration of the constructed molecule; the empty answer is a proof, not an undecided one
    assert _helper(smiles) == _rdkit(smiles) == set()
    why = []
    assert cs.charge_shift_forms(Chem.MolFromSmiles(smiles), why=why) == [] and why == []


@pytest.mark.parametrize("smiles", CARBANION_ELSEWHERE)
def test_a_carbanion_in_another_pi_system_does_not_stop_the_shift(smiles):
    got = _helper(smiles)
    assert got == _rdkit(smiles) and len(got) >= 1


@pytest.mark.parametrize("smiles", ["C[N+]1=[C-]N(CC)c2ccccc21", "[CH2-][n+]1ccn(C)c1", CARBANION_SYSTEMS[-1]])
def test_the_carbanion_rule_does_not_depend_on_atom_order(smiles):
    mol = Chem.MolFromSmiles(smiles)
    for seed in range(5):
        order = list(range(mol.GetNumAtoms()))
        random.Random(seed).shuffle(order)
        other = Chem.RenumberAtoms(mol, order)
        assert cs.charge_shift_forms(other) == []


@pytest.mark.parametrize("n", [3, 4, 6])
def test_a_carbene_ladder_has_no_form_on_either_side_of_the_bound(n):
    mol = Chem.MolFromSmiles(carbene_ladder(n))
    assert cs.charge_shift_forms(mol) == []


@pytest.mark.parametrize("smiles", sorted(FORMS))
def test_namers_filter_accepts_every_form_and_only_those(smiles):
    # the seam of namer._charge_forms: the forms are candidates of the filter that decides which drawings are forms
    mol = Chem.MolFromSmiles(smiles)
    found = namer._accepted_charge_forms(mol, smiles, cs.charge_shift_forms(mol))
    assert set(found) - {Chem.MolToSmiles(mol)} == FORMS[smiles]


@pytest.mark.parametrize("smiles", sorted(FORMS))
def test_a_form_keeps_the_atom_indices_and_the_hydrogens_of_the_drawing(smiles):
    mol = Chem.MolFromSmiles(smiles)
    for form in cs.charge_shift_forms(mol):
        Chem.SanitizeMol(Chem.Mol(form))
        assert [a.GetAtomicNum() for a in form.GetAtoms()] == [a.GetAtomicNum() for a in mol.GetAtoms()]
        assert [a.GetTotalNumHs() for a in form.GetAtoms()] == [a.GetTotalNumHs() for a in mol.GetAtoms()]
        assert not any(a.GetNumRadicalElectrons() for a in form.GetAtoms())


def test_the_forms_come_in_a_fixed_order():
    mol = Chem.MolFromSmiles("CNC(=[NH2+])NCc1ccccc1")
    first = [Chem.MolToSmiles(f, canonical=False) for f in cs.charge_shift_forms(mol)]
    assert first == [Chem.MolToSmiles(f, canonical=False) for f in cs.charge_shift_forms(mol)] and len(first) == 2


@pytest.mark.parametrize("row", rows("n_azides.tsv"), ids=lambda r: r["id"])
def test_n_azides_keep_their_form(row):
    assert len(_helper(row["smiles"])) == 1


@pytest.mark.parametrize("row", rows("user_molecules.tsv"), ids=lambda r: r["id"])
def test_user_molecules_regression_pin(row):
    # a pin of the helper's own answer: RDKit never finished on them.  is a proof; None would be undecided
    assert _helper(row["smiles"]) == set()


@pytest.mark.parametrize("n", [1, 3, 5, 9])
def test_azide_ladder_has_no_form(n):
    assert _helper(azide_ladder(n)) == set()


@pytest.mark.parametrize("n", [1, 3, 5, 9])
def test_imidazolium_ladder_has_one_form(n):
    assert len(_helper(imidazolium_ladder(n))) == 1


@pytest.mark.parametrize("n", [3, 5, 6, 9])
def test_n_azide_ladder_has_one_form(n):
    assert len(_helper(n_azide_ladder(n))) == 1


def test_forms_do_not_depend_on_atom_order():
    mol = Chem.MolFromSmiles(imidazolium_ladder(6))
    reference = _canon(cs.charge_shift_forms(mol), mol)
    assert len(reference) == 1
    for seed in range(5):
        order = list(range(mol.GetNumAtoms()))
        random.Random(seed).shuffle(order)
        other = Chem.RenumberAtoms(mol, order)
        assert _canon(cs.charge_shift_forms(other), other) == reference


def test_undecided_is_none_not_empty():
    why = []                                      # a metal bonded in the fragment that holds the N+
    assert cs.charge_shift_forms(Chem.MolFromSmiles("Cl[Zn]c1ccc2[n+](C)c[nH]c2c1"), why=why) is None
    assert why == ["metal"]
    why = []                                      # a count overflow
    assert cs.charge_shift_forms(Chem.MolFromSmiles(imidazolium_ladder(3)), max_configurations=0, why=why) is None
    assert why == ["configurations"]
    # a separate metal ion is not bonded: the helper decides
    assert len(_helper("C[n+]1c[nH]c2ccccc21.[Zn+2]")) == 1


def test_a_metal_is_a_bonded_metal_and_selenium_germanium_tellurium_are_not():
    for element in ("Se", "Ge", "Te", "As", "B", "Si"):
        z = Chem.GetPeriodicTable().GetAtomicNumber(element)
        assert z not in cs.METALS
    for element in ("Na", "Mg", "Al", "Fe", "Zn", "Pt", "Sn", "Bi", "Eu", "U"):
        assert Chem.GetPeriodicTable().GetAtomicNumber(element) in cs.METALS


def test_an_unbonded_metal_and_a_bonded_selenium_leave_the_answer_decided():
    why = []
    mol = Chem.MolFromSmiles("C[n+]1c[nH]c2cc(-c3ccc[se]3)ccc21")
    assert cs.charge_shift_forms(mol, why=why) is not None and why == []


def test_a_kekule_failure_is_undecided():
    # a five-ring of four c and one n+ that no bond order draws
    mol = Chem.MolFromSmiles("C[n+]1cccc1", sanitize=False)
    mol.UpdatePropertyCache(strict=False)
    why = []
    assert cs.charge_shift_forms(mol, why=why) is None and why == ["kekulize"]


def test_the_matching_size_is_counted(monkeypatch):
    monkeypatch.setattr(cs, "MAX_PI_NODES", 2)
    why = []
    assert cs.charge_shift_forms(Chem.MolFromSmiles("Cc1c[nH]c[nH+]1"), why=why) is None and why == ["nodes"]


def test_a_drawing_without_an_inchi_key_is_undecided(monkeypatch):
    monkeypatch.setattr(Chem, "MolToInchiKey", lambda mol: "")
    why = []
    assert cs.charge_shift_forms(Chem.MolFromSmiles("Cc1c[nH]c[nH+]1"), why=why) is None and why == ["inchi"]


def test_an_unpaired_electron_leaves_no_form_but_makes_the_structure_questions_undecided():
    mol = Chem.MolFromSmiles("C[n+]1c[nH]c2ccccc21")
    mol.GetAtomWithIdx(0).SetNumRadicalElectrons(1)       # every structure keeps the electron, and a form has none
    why = []
    assert cs.charge_shift_forms(mol, why=why) == [] and why == []
    assert cs.charge_placements(mol, why=why) is None and why == ["radical"]
    why = []
    assert cs.hydron_sites(mol, 1, {1, 2, 3, 4, 5, 6, 7, 8, 9}, why=why) is None and why == ["radical"]


def test_a_very_large_conjugated_system_returns():
    # 32 phenylenes, 197 heavy atoms: a list or None, never a long search
    forms = _helper(imidazolium_ladder(32))
    assert forms is None or len(forms) == 1


def test_pi_atoms_is_a_count():
    sizes = [cs.pi_atoms(Chem.MolFromSmiles(imidazolium_ladder(n))) for n in (1, 4, 5, 9)]
    assert sizes == [11, 29, 35, 59]
    assert isinstance(cs.RDKIT_RESONANCE_MAX_CONJUGATED, int) and isinstance(cs.RDKIT_RESONANCE0_MAX_WORK, int)


def test_the_pi_count_does_not_depend_on_atom_order():
    mol = Chem.MolFromSmiles(rows("user_molecules.tsv")[0]["smiles"])
    sizes = set()
    for seed in range(5):
        order = list(range(mol.GetNumAtoms()))
        random.Random(seed).shuffle(order)
        sizes.add(cs.pi_system(Chem.RenumberAtoms(mol, order)))
    assert sizes == {cs.pi_system(mol)} and cs.pi_atoms(mol) == 60


def test_no_clock_in_the_module():
    source = inspect.getsource(cs)
    for word in ("import time", "perf_counter", "process_time", "signal", "threading"):
        assert word not in source


def test_the_bounds_are_read_when_the_predicates_run(monkeypatch):
    mol = Chem.MolFromSmiles(imidazolium_ladder(2))
    assert cs.rdkit_flags8_affordable(mol) and cs.rdkit_flags0_affordable(mol)
    monkeypatch.setattr(cs, "RDKIT_RESONANCE_MAX_CONJUGATED", cs.pi_atoms(mol) - 1)
    monkeypatch.setattr(cs, "RDKIT_RESONANCE0_MAX_WORK", cs.pi_work(mol) - 1)
    assert not cs.rdkit_flags8_affordable(mol) and not cs.rdkit_flags0_affordable(mol)


@pytest.mark.parametrize("smiles,expected", [
    ("c1ccccc1", True), ("CC", False), ("C[N+](C)(C)C", False), ("CN(C)C", True), ("C=C", True), ("CC(=O)N", True),
    ("C[O-]", True), ("[C+](C)(C)C", True), ("CCl", False), ("[Zn+2]", True), ("C[S+](C)C", True),
    ("CCCC[N+](C)(C)C", False)])
def test_pi_capable_atoms(smiles, expected):
    mol = Chem.MolFromSmiles(smiles)
    capable = [cs.pi_capable(a) for a in mol.GetAtoms()]
    assert any(capable) == expected


def test_a_lone_pi_atom_is_no_system():
    assert cs.pi_system(Chem.MolFromSmiles("[Zn+2]")) == (0, 0)
    assert cs.pi_system(Chem.MolFromSmiles("CCO")) == (0, 0)
    assert cs.pi_work(Chem.MolFromSmiles("CCO")) == 0


@pytest.mark.parametrize("smiles", ["[C+](c1ccccc1)(c1ccccc1)c1ccccc1", "[C-](c1ccccc1)(c1ccccc1)c1ccccc1"])
def test_a_carbocation_or_carbanion_centre_is_pi_capable(smiles):
    # three phenyl rings and the charged centre are one pi system of 19 atoms, not three of 6
    assert cs.pi_atoms(Chem.MolFromSmiles(smiles)) == 19


def test_pi_work_counts_two_per_charged_pi_atom():
    mol = Chem.MolFromSmiles("C[n+]1c[nH]cc1-c1cc(N=[N+]=[N-])c(N=[N+]=[N-])cc1")      # 17 pi atoms, 5 charged
    assert (cs.pi_atoms(mol), cs.pi_work(mol)) == (17, 27)


def test_a_charge_dense_group_beside_a_larger_neutral_one_sets_w0():
    # the neutral quaterphenyl (24 pi atoms) is the largest group; the imidazolium-NH with five diazonium groups
    # (21 pi atoms, 6 charged) is the group RDKit's enumeration without a flag pays for
    row = next(r for r in rows("flags0_ladders.tsv") if r["id"] == "charge_dense_beside_neutral")
    mol = Chem.MolFromSmiles(row["smiles"])
    assert (cs.pi_atoms(mol), cs.pi_work(mol)) == (24, 33)


@pytest.mark.parametrize("row", rows("flags0_ladders.tsv"), ids=lambda r: r["id"])
def test_the_flags0_ladders_are_routed_by_w0(row):
    mol = Chem.MolFromSmiles(row["smiles"])
    assert cs.pi_atoms(mol) == int(row["pi_atoms"])
    assert cs.pi_work(mol) == int(row["W0"])
    assert cs.rdkit_flags0_affordable(mol) == (row["side"] == "rdkit")


@pytest.mark.parametrize("smiles", [s for s, forms in sorted(FORMS.items()) if len(forms) >= 1])
def test_the_pi_count_is_the_same_for_every_charge_drawing(smiles):
    # the count re-draws the charge: it does not only renumber atoms
    mol = Chem.MolFromSmiles(smiles)
    found = namer._accepted_charge_forms(mol, smiles, namer._rdkit_charge_candidates(mol))
    assert len(found) >= 2
    sizes = {cs.pi_system(Chem.MolFromSmiles(text)) for text, _order in found.values()}
    works = {cs.pi_work(Chem.MolFromSmiles(text)) for text, _order in found.values()}
    assert len(sizes) == 1 and len(works) == 1


def test_a_move_is_restricted_by_the_atoms_that_lose_and_gain_the_charge():
    smiles = "Cc1[nH]c[nH+]c1-c1ccc(cc1)C(=[NH2+])NC"          # two cations, each with a second place for the +1
    mol = Chem.MolFromSmiles(smiles)
    ring_cation, amidinium = _atom(mol, "[nH+]"), _atom(mol, "[NH2+]")
    ring_donor, amidine_donor = _atom(mol, "[nH+0]"), _atom(mol, "[NH1]([CH3])")
    assert len(_helper(smiles, max_moves=1)) == 2
    assert len(_helper(smiles, max_moves=2)) == 3 == len(_rdkit(smiles))
    assert len(_helper(smiles, max_moves=0)) == 0
    assert len(_helper(smiles, lost=[ring_cation])) == 1
    assert len(_helper(smiles, lost=[amidinium])) == 1
    assert len(_helper(smiles, gained=[amidine_donor])) == 1
    assert _helper(smiles, lost=[ring_cation], gained=[amidine_donor]) == set()
    assert _helper(smiles, lost=[ring_cation], gained=[ring_donor]) != set()
    assert cs.charge_shift_forms(mol, lost=[]) == []


def test_a_bond_shift_isomer_of_a_ring_that_is_not_aromatic_is_a_form():
    # the same charges, the double bonds of an eight-ring moved round: the canonical SMILES differs
    smiles = "C[N+]1=CC=CC=CC=C1C"
    assert _helper(smiles) == {"CC1=[N+](C)C=CC=CC=C1"}
    assert _helper(smiles, gained=[]) == set()               # a restricted question moves charges only
    why = []
    assert cs.charge_shift_forms(Chem.MolFromSmiles(smiles), max_configurations=1, why=why) is None
    assert why == ["configurations"]


# A charge placement whose pi bonds have more than one Kekule structure in a ring that is not aromatic. RDKit's
# enumeration gives one of them (which one depends on its search), so the matching gives every one: its forms contain
# every form RDKit gives.
KEKULE_OF_A_SHIFT = {
    "CC(C)C1=[N+]=C(c2ccccc2)N1C": {"CC(C)C1=NC(c2ccccc2)=[N+]1C", "CC(C)C1=[N+](C)C(c2ccccc2)=N1"},
    "CN1C=CC=CC=CC1=[N+](C)C": {"CN(C)C1=CC=CC=CC=[N+]1C", "CN(C)C1=[N+](C)C=CC=CC=C1"},
    "C1=NC2=C(C=C1)C3=[N+]=C(N3C=C2)O": {"OC1=Nc2c3cccnc3cc[n+]21"},
    "C1=CN2C(=C1)C3=[N+]=C(N3C=C2)O": {"OC1=Nc2c3ccc[n+]-3ccn21", "OC1=Nc2c3cccn3cc[n+]21"},
}


@pytest.mark.parametrize("smiles", sorted(KEKULE_OF_A_SHIFT))
def test_every_kekule_structure_of_a_shifted_charge_is_a_form(smiles):
    mine = _helper(smiles)
    assert mine is not None
    assert _rdkit(smiles) <= mine                      # the oracle: nothing RDKit gives is missing
    assert KEKULE_OF_A_SHIFT[smiles] <= mine          # a regression pin: the structures this molecule has


def test_the_kekule_structures_of_a_shifted_charge_are_found_even_when_the_shifted_structure_is_not_a_form():
    # the shifted structure alone draws an exocyclic double bond to an aromatic atom (the filter refuses it); the other
    # Kekule structure of the same charges does not (the filter accepts it)
    mol = Chem.MolFromSmiles("C1=NC2=C(C=C1)C3=[N+]=C(N3C=C2)O")
    centre, donor = _centre(mol), _atom(mol, "[NX3;+0;!$(N=*)]")
    assert cs.charge_shift_forms(mol, lost=[centre], gained=[donor]) == []
    assert _helper("C1=NC2=C(C=C1)C3=[N+]=C(N3C=C2)O") == {"OC1=Nc2c3cccnc3cc[n+]21"}


def test_a_restricted_question_moves_charges_only_and_finds_no_other_kekule_structure():
    smiles = "CC(C)C1=[N+]=C(c2ccccc2)N1C"
    mol = Chem.MolFromSmiles(smiles)
    centre, donor = _centre(mol), _atom(mol, "[NX3;+0;!$(N=*)]")
    assert len(cs.charge_shift_forms(mol, lost=[centre], gained=[donor])) == 1
    assert len(_helper(smiles)) == 2


def test_the_configurations_are_counted():
    mol = Chem.MolFromSmiles("Cc1[nH]c[nH+]c1-c1ccc(cc1)C(=[NH2+])NC")
    why = []
    assert cs.charge_shift_forms(mol, max_configurations=1, why=why) is None and why == ["configurations"]
    assert cs.charge_shift_forms(mol, max_configurations=cs.MAX_CONFIGURATIONS) is not None


def test_the_exocyclic_double_bond_count_counts_double_bonds_between_aromatic_and_other_atoms():
    expected = {"Nc1cc[nH+]cc1": 0, "[NH2+]=c1cc[nH]cc1": 1, "CC(=O)c1ccccc1": 0, "O=C1C=CC(=O)C=C1": 0, "c1ccccc1": 0,
                "O=c1cc[nH]cc1": 1}
    for smiles, count in expected.items():
        assert cs.exocyclic_double_bonds_to_aromatic(Chem.MolFromSmiles(smiles)) == count


def test_the_namer_uses_the_definitions_of_this_module():
    assert namer._CHARGE_MOBILE_N is cs.MOBILE_N_CATION
    assert namer._accepted_charge_forms is cs.accepted_charge_forms


def test_the_mobile_nitrogen_cation_is_the_nitrogen_that_hands_its_charge_on():
    mobile = list(FORMS) + ["C[n+]1ccccc1", "[NH2+]=c1cc[nH]cc1"]
    for smiles in mobile:
        assert Chem.MolFromSmiles(smiles).HasSubstructMatch(cs.MOBILE_N_CATION), smiles
    for smiles in ["O=[N+]([O-])c1ccccc1", "[O-][n+]1ccccc1", "C[N+](=O)[O-]", "CC#[N+]C", "C[N+](C)(C)C"]:
        assert not Chem.MolFromSmiles(smiles).HasSubstructMatch(cs.MOBILE_N_CATION), smiles


class _Source:
    """A stand-in for RDKit's enumeration that records the molecules it is asked about."""

    def __init__(self, drawings):
        self.drawings = drawings
        self.asked = []

    def __call__(self, mol):
        self.asked.append(mol)
        return self.drawings


def test_charge_drawings_asks_the_source_up_to_the_bound_and_never_above_it():
    below = Chem.MolFromSmiles(imidazolium_ladder(4))
    above = Chem.MolFromSmiles(imidazolium_ladder(5))
    assert cs.pi_atoms(below) <= cs.RDKIT_RESONANCE_MAX_CONJUGATED < cs.pi_atoms(above)
    source = _Source(["a drawing"])
    assert cs.charge_drawings(below, source) == ["a drawing"] and source.asked == [below]
    source.asked.clear()
    forms = cs.charge_drawings(above, source)
    assert source.asked == []
    assert [Chem.MolToSmiles(f) for f in forms] == [Chem.MolToSmiles(f) for f in cs.charge_shift_forms(above)]
    assert len(forms) == 1


def test_charge_drawings_reads_the_bound_when_it_runs(monkeypatch):
    mol = Chem.MolFromSmiles(imidazolium_ladder(2))
    count = cs.pi_atoms(mol)
    source = _Source([])
    monkeypatch.setattr(cs, "RDKIT_RESONANCE_MAX_CONJUGATED", count - 1)
    cs.charge_drawings(mol, source)
    assert source.asked == []
    monkeypatch.setattr(cs, "RDKIT_RESONANCE_MAX_CONJUGATED", count)
    cs.charge_drawings(mol, source)
    assert source.asked == [mol]


def test_charge_drawings_above_the_bound_proves_no_form_with_an_empty_answer():
    mol = Chem.MolFromSmiles(azide_ladder(5))
    source = _Source(["never used"])
    assert cs.pi_atoms(mol) > cs.RDKIT_RESONANCE_MAX_CONJUGATED
    assert cs.charge_drawings(mol, source) == [] and source.asked == []


def test_charge_drawings_above_the_bound_is_undecided_for_a_bonded_metal_and_never_asks_the_source():
    mol = Chem.MolFromSmiles("[nH+]1c[nH]cc1-" + "-".join(["c1ccc(cc1)"] * 4 + ["c1ccc([Sn](C)(C)C)cc1"]))
    source = _Source(["never used"])
    why = []
    assert cs.pi_atoms(mol) > cs.RDKIT_RESONANCE_MAX_CONJUGATED
    assert cs.charge_drawings(mol, source, why=why) is None
    assert why == ["metal"] and source.asked == []


@pytest.mark.parametrize("smiles", ["C[N+]1=CNc2ccccc21", "C[n+]1cc[nH]c1", "C[n+]1c[nH]c2ccccc21"])
def test_hydron_sites_match_rdkit_on_the_constructed_positives(smiles):
    mol = Chem.MolFromSmiles(smiles)
    centre = _centre(mol)
    ring = ions._ring_system_of(mol, centre)
    want_mol, want_site = ions._ring_cation_hydron_site(mol, ring, centre)
    got = cs.hydron_sites(mol, centre, ring)
    rank = list(Chem.CanonicalRankAtoms(mol, breakTies=True))
    assert min(got, key=lambda i: rank[i]) == want_site
    assert Chem.MolToSmiles(got[want_site]) == Chem.MolToSmiles(want_mol)           # the structure, not only the site
    structure = got[want_site]
    assert structure.GetAtomWithIdx(centre).GetFormalCharge() == 0
    site_atom = structure.GetAtomWithIdx(want_site)
    assert site_atom.GetFormalCharge() == 1 and site_atom.GetTotalNumHs() == 1


@pytest.mark.parametrize("smiles", ["C[n+]1ccccc1", "C[n+]1c(C=Cc2cccs2)sc2ccccc21", "C[n+]1cc[n](C)c1"])
def test_hydron_sites_has_the_empty_answer_where_rdkit_finds_no_site(smiles):
    mol = Chem.MolFromSmiles(smiles)
    centre = _centre(mol)
    ring = ions._ring_system_of(mol, centre)
    why = []
    assert ions._ring_cation_hydron_site(mol, ring, centre) is None
    assert cs.hydron_sites(mol, centre, ring, why=why) == {} and why == []


def test_hydron_sites_of_a_centre_in_a_carbanion_system_is_empty_as_rdkits_is():
    # RDKit's enumeration without a flag gives this carbene no structure beyond the drawing, so it finds no site
    mol = Chem.MolFromSmiles("C[N+]1=[C-][NH]C=C1")
    centre = _centre(mol)
    ring = ions._ring_system_of(mol, centre)
    why = []
    assert ions._ring_cation_hydron_site(mol, ring, centre) is None
    assert cs.hydron_sites(mol, centre, ring, why=why) == {} and why == []


def test_hydron_sites_for_a_centre_that_holds_no_pi_cation_is_empty():
    mol = Chem.MolFromSmiles("C[N+]1(C)CCc2cc[nH]c2C1")
    assert cs.hydron_sites(mol, _centre(mol), {0, 1, 2}) == {}               # a quaternary ammonium carries no pi bond
    mol = Chem.MolFromSmiles("C[n+]1cc[nH]c1")
    assert cs.hydron_sites(mol, 0, {1, 2, 3, 4, 5}) == {}                    # a neutral atom is no cation centre


def test_the_zwitterion_class_of_site_2_is_undecided():
    # RDKit finds a site here; the matching moves the +1 only, and this +1 reaches the NH only with the -1
    smiles = "C[n+]1[nH]n(C)c(=O)c1C([O-])=NC"
    mol = Chem.MolFromSmiles(smiles)
    centre = _centre(mol)
    ring = ions._ring_system_of(mol, centre)
    assert ions._ring_cation_hydron_site(mol, ring, centre) is not None
    why = []
    assert cs.hydron_sites(mol, centre, ring, why=why) is None and why == ["anion"]


def test_a_site_found_beside_an_anion_is_returned():
    mol = Chem.MolFromSmiles("C[n+]1[nH]ccc1C([O-])=O")
    centre = _centre(mol)
    ring = ions._ring_system_of(mol, centre)
    why = []
    assert sorted(cs.hydron_sites(mol, centre, ring, why=why)) == [ions._ring_cation_hydron_site(mol, ring, centre)[1]]
    assert why == []


def test_hydron_sites_of_a_bonded_metal_is_undecided():
    mol = Chem.MolFromSmiles("C[Zn]c1ccc2c(c1)[nH]c[n+]2C")
    centre = _centre(mol)
    why = []
    assert cs.hydron_sites(mol, centre, ions._ring_system_of(mol, centre), why=why) is None and why == ["metal"]


def test_hydron_sites_counts_its_donors():
    mol = Chem.MolFromSmiles("C[n+]1c[nH]c2ccccc21")
    why = []
    assert cs.hydron_sites(mol, _centre(mol), {1, 2, 3}, max_configurations=0, why=why) is None
    assert why == ["configurations"]


@pytest.mark.parametrize("smiles", ["C[n+]1c[nH]c2ccccc21", imidazolium_ladder(2), "NC(=[NH2+])NC",
                                    "CN(C)C=CC=[N+](C)C", "Cc1ccnn2c(C)n[n+](C)c12"])
def test_charge_placements_give_the_neutral_analogues_rdkit_gives(smiles):
    mol = Chem.MolFromSmiles(smiles)
    analogues = {protonation._neutral_analogue_inchi(mol)}
    analogues |= {protonation._neutral_analogue_inchi(m) for m in cs.charge_placements(mol)}
    assert analogues == protonation._analogue_inchis(mol)


@pytest.mark.parametrize("n", [2, 6, 9])
def test_a_large_cation_keeps_both_charge_placements(n):
    # the two N of the imidazolium give two neutral analogues, on either side of any bound
    mol = Chem.MolFromSmiles(imidazolium_ladder(n))
    analogues = {protonation._neutral_analogue_inchi(mol)}
    analogues |= {protonation._neutral_analogue_inchi(m) for m in cs.charge_placements(mol)}
    assert len(analogues) == 2


def test_charge_placements_of_a_dipole_are_a_subset_of_rdkits_analogues():
    # an azide moves a -1 in RDKit's enumeration without a flag and the matching does not: the set is narrower, so a
    # verdict that reads it can only turn stricter, never looser
    mol = Chem.MolFromSmiles("CN(C)N=[N+]=[N-]")
    analogues = {protonation._neutral_analogue_inchi(m) for m in cs.charge_placements(mol)}
    assert analogues and analogues <= protonation._analogue_inchis(mol)


@pytest.mark.parametrize("smiles", CARBANION_SYSTEMS[:6])
def test_charge_placements_of_a_carbanion_system_are_the_analogues_rdkit_gives(smiles):
    # RDKit's enumeration without a flag gives the drawing's own analogue only; a placement beyond it would be looser
    mol = Chem.MolFromSmiles(smiles)
    assert protonation._analogue_inchis(mol) == {protonation._neutral_analogue_inchi(mol)}
    assert cs.charge_placements(mol) == []


def test_charge_placements_are_sanitized_structures_of_the_drawing():
    mol = Chem.MolFromSmiles("Cc1c[nH]c[nH+]1")
    placements = cs.charge_placements(mol)
    assert len(placements) == 1
    for m in placements:
        assert [a.GetAtomicNum() for a in m.GetAtoms()] == [a.GetAtomicNum() for a in mol.GetAtoms()]
        assert sum(a.GetFormalCharge() for a in m.GetAtoms()) == 1
        Chem.SanitizeMol(Chem.Mol(m))


def test_charge_placements_are_empty_without_a_mobile_cation_and_undecided_with_a_bonded_metal():
    assert cs.charge_placements(Chem.MolFromSmiles("CCO")) == []
    why = []
    assert cs.charge_placements(Chem.MolFromSmiles("C[Zn]c1ccc2c(c1)[nH]c[n+]2C"), why=why) is None and why == ["metal"]
    why = []
    assert cs.charge_placements(Chem.MolFromSmiles("Cc1[nH]c[nH+]c1-c1ccc(cc1)C(=[NH2+])NC"), max_configurations=1,
                                why=why) is None and why == ["configurations"]
