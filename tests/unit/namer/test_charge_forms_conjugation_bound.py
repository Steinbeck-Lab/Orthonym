"""_charge_forms asks RDKit for resonance drawings only when the largest pi system has at most
RDKIT_RESONANCE_MAX_CONJUGATED atoms (perception.charge_shift.pi_atoms); above that size the forms come from a matching
of the pi bonds.

the contributor guide core principle 4 "Determinism": a cation whose +1 sits on one of several nitrogen atoms is one species whatever
SMILES draws it, and the name is built from the hydron site: "General rule for systematically naming cationic
centers in parent hydrides" (the Blue Book),:41368: "A cation derived formally by adding one or more hydrons to
any position of a neutral parent hydride... is named by replacing the final letter 'e' of the parent hydride name, if
any, by the suffix 'ium'". RDKit builds every bond-order arrangement of a conjugated group (about x3 to x8 for each
linked ring); no cap stops it and the C++ call holds the GIL, so a molecule above the bound never reaches it. A matching
that cannot decide names the input as drawn; it never asks RDKit. The oracle for the forms above the bound is RDKit's
own enumeration, run live on constructed molecules that it finishes in well under a second. The rows taken from the
benchmark corpora are in tests_local/unit/namer/test_charge_forms_conjugation_bound_local.py.
"""
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest
from rdkit import Chem

import orthonym
from orthonym import Orthonym, namer
from orthonym.perception import charge_shift
from orthonym.perception.charge_shift import RDKIT_RESONANCE_MAX_CONJUGATED as BOUND
from orthonym.perception.charge_shift import pi_atoms
from tests.support.jars import jar_or_skip
from tests.support.resonance_rows import azide_ladder, carbene_ladder, imidazolium_ladder, n_azide_ladder, rows
from tests.support.resonance_spy import supplier_calls

pytestmark = pytest.mark.unit

LADDER = [1, 4, 5, 6, 9]
HARD_TIMEOUT_S = 60
CPU_LIMIT_S = 5.0
USER_MOLECULES = rows("user_molecules.tsv")
RELATIVES = [r for r in rows("controls.tsv") if r["id"] in {"p_azido_tpp", "mol2_nozn", "mol3_cation"}]

#: an imidazolium ladder (n = 5, 35 pi atoms) with one substituent on the last ring, and the elements it carries
ELEMENT_GROUPS = {"Ge": "[Ge](C)(C)C", "Si": "[Si](C)(C)C", "As": "[As](C)C", "B": "B(C)C", "Se": "[Se]C", "Te": "[Te]C"}
METAL_GROUPS = {"Sn": "[Sn](C)(C)C", "Hg": "[Hg]C", "Zn": "[Zn]C", "Fe": "[Fe]C", "Pt": "[Pt]C"}


@pytest.fixture(autouse=True)
def _fresh_cache():
    namer._charge_forms.cache_clear()
    yield
    namer._charge_forms.cache_clear()


def _forms(smiles):
    namer._charge_forms.cache_clear()
    return namer._charge_forms(smiles)


def _canonical(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def _form_keys(forms):
    """The canonical SMILES of every entry of ``_charge_forms`` (the drawing included)."""
    return sorted(_canonical(smiles) for smiles, _order in forms)


def _rdkit_keys(smiles):
    """The canonical SMILES of the drawing and of every form of its species, from RDKit's enumeration asked directly."""
    mol = Chem.MolFromSmiles(smiles)
    return sorted(namer._accepted_charge_forms(mol, smiles, namer._rdkit_charge_candidates(mol)))


def _ring_ladder_with(group):
    """An imidazolium ladder of five rings whose last ring carries ``group`` para to the link."""
    return "[nH+]1c[nH]cc1-" + "-".join(["c1ccc(cc1)"] * 4 + ["c1ccc(%s)cc1" % group])


def _n_aryl_benzimidazolium(n):
    """1-methyl-3-aryl-benzimidazolium on n para-linked phenylenes: two drawings, the +1 on either nitrogen."""
    return "C[n+]1cn(-" + "-".join(["c2ccc(cc2)"] * (n - 1) + ["c2ccccc2"]) + ")c2ccccc21"


def _imidazolium_with_pi_atoms(count):
    """An imidazolium ladder whose largest pi system has ``count`` atoms: para-linked phenylenes, and for the rest
    dimethylamino groups on the last ring (each adds one pi atom; the methyl groups are no pi atoms)."""
    rings, extra = divmod(count - 5, 6)
    last = "c1" + "".join("c(N(C)C)" if i < extra else "c" for i in range(5)) + "1"
    return "[nH+]1c[nH]cc1-" + "-".join(["c1ccc(cc1)"] * (rings - 1) + [last])


_CHILD = """
import json, sys, time
from orthonym import namer
out = {}
for name, smiles in json.loads(sys.argv[1]).items():
    start = time.process_time()
    forms = namer._charge_forms(smiles)
    out[name] = {"none": forms is None, "cpu": time.process_time() - start}
print(json.dumps(out))
"""


def test_the_three_user_molecules_leave_charge_forms_within_seconds():
    # a child process: RDKit's C++ call holds the GIL and ignores signals, so only a kill from outside ends a hang
    source_root = str(Path(orthonym.__file__).resolve().parents[1])
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [source_root, os.environ.get("PYTHONPATH")])))
    payload = json.dumps({row["id"]: row["smiles"] for row in USER_MOLECULES})
    try:
        done = subprocess.run([sys.executable, "-c", _CHILD, payload], capture_output=True, text=True,
                              timeout=HARD_TIMEOUT_S, env=env)
    except subprocess.TimeoutExpired:
        pytest.fail("_charge_forms did not return for the user molecules in %d s" % HARD_TIMEOUT_S, pytrace=False)
    assert done.returncode == 0, done.stderr[-2000:]
    result = json.loads(done.stdout.strip().splitlines()[-1])
    assert sorted(result) == sorted(row["id"] for row in USER_MOLECULES)
    for name, entry in result.items():
        assert entry["none"], name              # the matching proves they have no other charge-only form
        assert entry["cpu"] < CPU_LIMIT_S, (name, entry["cpu"])


@pytest.mark.parametrize("row", USER_MOLECULES + RELATIVES, ids=lambda r: r["id"])
def test_the_user_molecules_and_their_relatives_build_no_resonance_supplier(row):
    assert pi_atoms(Chem.MolFromSmiles(row["smiles"])) > BOUND
    with supplier_calls(forbid=True) as calls:
        assert _forms(row["smiles"]) is None
    assert calls == []


@pytest.mark.parametrize("smiles", [imidazolium_ladder(5), imidazolium_ladder(6), _n_aryl_benzimidazolium(4),
                                    _n_aryl_benzimidazolium(5)])
def test_forms_above_the_bound_are_the_forms_rdkit_gives(smiles):
    assert pi_atoms(Chem.MolFromSmiles(smiles)) > BOUND
    with supplier_calls(forbid=True) as calls:
        forms = _forms(smiles)
    assert calls == []
    assert len(forms) == 2
    assert _form_keys(forms) == _rdkit_keys(smiles)
    assert [order is None for _smiles, order in forms].count(True) == 1     # the drawn form keeps its own string
    assert smiles in [text for text, order in forms if order is None]


@pytest.mark.parametrize("element", list(ELEMENT_GROUPS))
def test_selenium_tellurium_germanium_arsenic_boron_and_silicon_are_decided_above_the_bound(element):
    smiles = _ring_ladder_with(ELEMENT_GROUPS[element])
    assert pi_atoms(Chem.MolFromSmiles(smiles)) > BOUND
    with supplier_calls(forbid=True) as calls:
        forms = _forms(smiles)
    assert calls == []
    assert forms is not None and _form_keys(forms) == _rdkit_keys(smiles)


@pytest.mark.parametrize("metal", list(METAL_GROUPS))
def test_a_bonded_metal_above_the_bound_is_named_as_drawn(metal):
    # the matching does not draw the charge and bond moves of a bonded metal: undecided, and RDKit is not asked
    smiles = _ring_ladder_with(METAL_GROUPS[metal])
    assert pi_atoms(Chem.MolFromSmiles(smiles)) > BOUND
    with supplier_calls(forbid=True) as calls:
        assert _forms(smiles) is None
    assert calls == []


@pytest.mark.parametrize("metal", ["Sn", "Zn"])
def test_a_bonded_metal_below_the_bound_still_goes_to_rdkit(metal):
    smiles = "[nH+]1c[nH]cc1-c1ccc(%s)cc1" % METAL_GROUPS[metal]
    assert pi_atoms(Chem.MolFromSmiles(smiles)) <= BOUND
    with supplier_calls() as calls:
        forms = _forms(smiles)
    assert [c[0] for c in calls] == ["_charge_forms"]
    assert forms is not None and len(forms) == 2


def test_an_error_in_the_matching_above_the_bound_names_the_input_as_drawn(monkeypatch):
    # the answer to "undecided" is never "ask RDKit", whatever the reason
    def fail(*args, **kwargs):
        raise RuntimeError("the matching failed")

    monkeypatch.setattr(charge_shift, "charge_shift_forms", fail)
    smiles = imidazolium_ladder(5)
    with supplier_calls(forbid=True) as calls:
        assert _forms(smiles) is None
    assert calls == []


@pytest.mark.parametrize("count", [BOUND - 3, BOUND, BOUND + 1, BOUND + 4])
def test_rdkit_is_asked_up_to_the_bound_and_never_above_it(count):
    smiles = _imidazolium_with_pi_atoms(count)
    assert pi_atoms(Chem.MolFromSmiles(smiles)) == count
    if count <= BOUND:
        with supplier_calls() as calls:
            forms = _forms(smiles)
        assert [c[0] for c in calls] == ["_charge_forms"]
    else:
        with supplier_calls(forbid=True) as calls:
            forms = _forms(smiles)
        assert calls == []
    assert forms is not None and len(forms) == 2                 # one pattern of forms on both sides of the bound
    assert _form_keys(forms) == _rdkit_keys(smiles)


def test_the_bound_is_read_from_charge_shift_when_the_function_runs(monkeypatch):
    smiles = imidazolium_ladder(5)
    monkeypatch.setattr(charge_shift, "RDKIT_RESONANCE_MAX_CONJUGATED", 40)
    with supplier_calls() as calls:
        forms = _forms(smiles)
    assert [c[0] for c in calls] == ["_charge_forms"] and len(forms) == 2
    monkeypatch.setattr(charge_shift, "RDKIT_RESONANCE_MAX_CONJUGATED", 10)
    with supplier_calls(forbid=True) as calls:
        assert len(_forms(imidazolium_ladder(1))) == 2
    assert calls == []


@pytest.mark.parametrize("n", LADDER)
def test_azide_ladder_has_no_forms_on_either_side_of_the_bound(n):
    smiles = azide_ladder(n)
    with supplier_calls(forbid=pi_atoms(Chem.MolFromSmiles(smiles)) > BOUND):
        assert _forms(smiles) is None


@pytest.mark.parametrize("n", LADDER)
def test_imidazolium_ladder_has_one_form_on_either_side_of_the_bound(n):
    smiles = imidazolium_ladder(n)
    with supplier_calls(forbid=pi_atoms(Chem.MolFromSmiles(smiles)) > BOUND):
        forms = _forms(smiles)
    assert forms is not None and len(forms) == 2
    assert [order is None for _smiles, order in forms].count(True) == 1      # the drawn form keeps its own string


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_a_carbene_ladder_has_no_second_form_on_either_side_of_the_bound(n):
    # the +1 does not move across the carbanion centre of an N-heterocyclic carbene drawn [N+]=[C-]: RDKit gives it no
    # second drawing below the bound, and above it the matching offers none and builds no supplier
    smiles = carbene_ladder(n)
    above = pi_atoms(Chem.MolFromSmiles(smiles)) > BOUND
    with supplier_calls(forbid=above) as calls:
        assert _forms(smiles) is None
    assert (calls == []) == above


@pytest.mark.parametrize("n", [5, 6, 9])
def test_n_azide_ladder_above_the_bound_has_no_form(n):
    # the +1 of an azido group does not move (``charge_shift.azide_atoms``,: the matching above the bound
    # still builds its candidates and builds no supplier, and the filter takes none of them
    smiles = n_azide_ladder(n)
    assert pi_atoms(Chem.MolFromSmiles(smiles)) > BOUND
    with supplier_calls(forbid=True) as calls:
        assert _forms(smiles) is None
    assert calls == []


def test_n_azide_ladder_has_no_form_on_either_side_of_the_bound():
    # Before the azide rule the two engines disagreed here: RDKit's 64-structure truncation gave one form at n = 1 and
    # none at n = 3 and n = 4, the matching one form at every n above the bound. The +1 of the azide does not move, so
    # no engine offers a form at any n and there is no jump left to pin.
    assert pi_atoms(Chem.MolFromSmiles(n_azide_ladder(4))) <= BOUND < pi_atoms(Chem.MolFromSmiles(n_azide_ladder(5)))
    for n in (1, 3, 4, 5):
        assert _forms(n_azide_ladder(n)) is None


def test_the_ladders_cross_the_bound():
    for ladder in (azide_ladder, imidazolium_ladder, n_azide_ladder):
        sizes = [pi_atoms(Chem.MolFromSmiles(ladder(n))) for n in LADDER]
        assert sizes[0] <= BOUND < sizes[-1] and sizes == sorted(sizes)


def test_rows_at_or_below_the_bound_still_use_rdkit():
    with supplier_calls() as calls:
        forms = _forms("Cc1c[nH]c[nH+]1")
    assert [c[0] for c in calls] == ["_charge_forms"] and len(forms) == 2


@pytest.mark.opsin_gate
@pytest.mark.parametrize("row", rows("n_azides.tsv"), ids=lambda r: r["id"])
def test_n_azide_has_no_form_and_is_named_as_drawn(row):
    # The +1 of an azido group does not move (``charge_shift.azide_atoms``): the input is named as it is drawn, with the
    # prefix 'azido' "AZIDES", the Blue Book). It used to get a second form, 'C[N+](C)=NN=[N-]' (the +1 on
    # the amine nitrogen), and the pin_verified name of THAT drawing, '4,4-dimethyltetraaz-1,3-dien-4-ium-1-ide', which a
    # fresh OPSIN call reads to other atoms than the azide's (tests/unit/rules/test_n_azides.py). The name is the
    # preferred N-azido amine name, ``rules/n_hetero_amines.py``), at both tiers. ASSUMED, not verified
    # against the book: it prints no N-azido amine (the ``basis`` column of the fixture says so); the assertion
    # of the form (none) and of the 'azido' reading is the one this test is about.
    jar_or_skip()
    assert _forms(row["smiles"]) is None
    from tests.support.rt_assert import name_best_effort
    named = name_best_effort(row["smiles"])
    assert (named["tier"], named["name"]) == ("pin_verified", row["name"])
    default = Orthonym(style="pin").name_tiered(row["smiles"])
    assert (default["tier"], default["name"]) == ("pin_verified", row["name"])


def test_the_count_ignores_atom_order():
    mol = Chem.MolFromSmiles(USER_MOLECULES[0]["smiles"])
    sizes = set()
    for seed in range(5):
        order = list(range(mol.GetNumAtoms()))
        random.Random(seed).shuffle(order)
        sizes.add(pi_atoms(Chem.RenumberAtoms(mol, order)))
    assert sizes == {60}


def test_the_forms_above_the_bound_do_not_depend_on_the_atom_order():
    smiles = imidazolium_ladder(5)
    mol = Chem.MolFromSmiles(smiles)
    expected = _form_keys(_forms(smiles))
    for seed in range(3):
        order = list(range(mol.GetNumAtoms()))
        random.Random(seed).shuffle(order)
        respelled = Chem.MolToSmiles(Chem.RenumberAtoms(mol, order), canonical=False)
        with supplier_calls(forbid=True) as calls:
            assert _form_keys(_forms(respelled)) == expected
        assert calls == []
