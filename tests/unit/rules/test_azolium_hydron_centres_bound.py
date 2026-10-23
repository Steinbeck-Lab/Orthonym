"""azolium_hydron_centres asks RDKit only within the work bound; above the bound the centres come from a matching of
the pi bonds.

 "General rule for systematically naming cationic centers in parent hydrides"
(the Blue Book),:41368: "A cation derived formally by adding one or more hydrons
to any position of a neutral parent hydride... is named by replacing the final letter
'e' of the parent hydride name, if any, by the suffix 'ium'"; '1H-imidazol-3-ium (PIN)'
(:41396). The charge of 1,3-dimethylimidazolium sits on either nitrogen, the cation is one
species, and the name is chosen among the nitrogens that can carry it (``azolium_demote_name``).

RDKit's enumeration without a flag builds every bond-order arrangement of each group that carries charge or N=X
bonds and its C++ call cannot be interrupted: a polyphenylene on one nitrogen of an azolium costs x3 for each
ring (62 s at ten rings, a 66-atom azolium of the PubChem corpus 94 s) and no Python alarm stops it. So it is asked
only when ``charge_shift.pi_work(mol)`` is at most ``RDKIT_RESONANCE0_MAX_WORK``. Above that count the placements of
``charge_shift.charge_placements`` answer, and an undecided answer is "the drawn centre alone": RDKit is never asked
in its place. Every test on a molecule that RDKit does not finish in a second builds no supplier
(``supplier_calls(forbid=True)``), so a regression fails at once instead of hanging.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from rdkit import Chem

import orthonym
from orthonym.perception import charge_shift
from orthonym.perception.charge_shift import RDKIT_RESONANCE0_MAX_WORK as B0
from orthonym.perception.charge_shift import pi_work
from orthonym.rules import azolium_cations, ions
from tests.support.jars import jar_or_skip
from tests.support.resonance_spy import supplier_calls

pytestmark = pytest.mark.unit

#: every molecule takes the matching route / the RDKit route
MATCHING_ONLY = -1
RDKIT_ONLY = 10 ** 6


def _phenylenes(n):
    """n para-linked phenylenes: 'c3ccc(cc3)-' x (n - 1), then 'c3ccccc3' (ring labels 3: the azolium uses 1 and 2)."""
    return "-".join(["c3ccc(cc3)"] * (n - 1) + ["c3ccccc3"])


#: ring nitrogens that can all carry the charge (smiles, the other centres, in index order)
TWO_CENTRES = [
    ("C[n+]1ccn(C)c1", [4]),
    ("CCn1cc[n+](C)c1", [2]),
    ("C[n+]1cn(C)c2ccccc21", [3]),
    ("C[n+]1cn(-c2ccccc2)c2ccccc21", [3]),
    ("C[n+]1cnn(C)c1", [4]),
    ("C[n+]1ccn(-c2ccc(N(C)C)cc2)c1", [4]),       # the NMe2 nitrogen is no ring centre
    ("C[n+]1ccn(-c2ccc(-c3ccncc3)cc2)c1", [4]),
]
#: one substituted nitrogen: the drawn centre alone
ONE_CENTRE = ["C[n+]1ccccc1", "C[n+]1ccccc1-c1ccccc1", "C[n+]1ccsc1"]


def _centre(mol):
    """The aromatic nitrogen cation with three connections and no hydrogen."""
    (centre,) = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 7 and a.GetIsAromatic()
                 and a.GetFormalCharge() == 1 and a.GetTotalNumHs() == 0 and a.GetDegree() == 3]
    return centre


def _centres(mol):
    centre = _centre(mol)
    return azolium_cations.azolium_hydron_centres(mol, ions._ring_system_of(mol, centre), centre)


@pytest.mark.parametrize("smiles, others", TWO_CENTRES)
def test_the_centres_are_the_drawn_nitrogen_and_the_other_substituted_ones(smiles, others):
    mol = Chem.MolFromSmiles(smiles)
    centre = _centre(mol)
    with supplier_calls() as calls:
        got = _centres(mol)
    assert got == [centre] + others
    assert [c[0] for c in calls] == ["_charge_structures"]


@pytest.mark.parametrize("smiles", ONE_CENTRE)
def test_a_single_substituted_nitrogen_is_its_own_centre(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert _centres(mol) == [_centre(mol)]


@pytest.mark.parametrize("smiles", [s for s, _ in TWO_CENTRES] + ONE_CENTRE)
def test_the_matching_route_gives_the_centres_of_the_rdkit_route(smiles, monkeypatch):
    mol = Chem.MolFromSmiles(smiles)
    monkeypatch.setattr(charge_shift, "RDKIT_RESONANCE0_MAX_WORK", RDKIT_ONLY)
    with supplier_calls() as rdkit_calls:
        want = _centres(mol)
    monkeypatch.setattr(charge_shift, "RDKIT_RESONANCE0_MAX_WORK", MATCHING_ONLY)
    with supplier_calls(forbid=True) as matching_calls:
        got = _centres(mol)
    assert [c[0] for c in rdkit_calls] == ["_charge_structures"] and matching_calls == []
    assert got == want


@pytest.mark.parametrize("bound, helper_side", [(MATCHING_ONLY, True), (RDKIT_ONLY, False)])
def test_the_function_reads_the_bound_when_it_runs(bound, helper_side, monkeypatch):
    monkeypatch.setattr(charge_shift, "RDKIT_RESONANCE0_MAX_WORK", bound)
    mol = Chem.MolFromSmiles("C[n+]1ccn(C)c1")
    with supplier_calls(forbid=helper_side) as calls:
        got = _centres(mol)
    assert got == [1, 4] and (calls == []) == helper_side


def test_rdkit_is_asked_up_to_the_bound_and_never_above_it():
    # the benzimidazolium on two phenylenes is below the bound, on three it is at B0 + 1
    below = Chem.MolFromSmiles("C[n+]1cn(-%s)c2ccccc21" % _phenylenes(2))
    above = Chem.MolFromSmiles("C[n+]1cn(-%s)c2ccccc21" % _phenylenes(3))
    assert pi_work(below) <= B0 < pi_work(above)
    with supplier_calls() as calls:
        assert _centres(below) == [1, 3]
    assert [c[0] for c in calls] == ["_charge_structures"]
    with supplier_calls(forbid=True) as calls:
        assert _centres(above) == [1, 3]
    assert calls == []


@pytest.mark.parametrize("rings", [5, 8, 12])
def test_a_polyphenylene_azolium_that_rdkit_does_not_finish_builds_no_supplier(rings):
    # RDKit needs a minute at ten rings and x3 for each ring more; the matching answers at once
    mol = Chem.MolFromSmiles("C[n+]1cn(-%s)c2ccccc21" % _phenylenes(rings))
    assert pi_work(mol) > B0
    with supplier_calls(forbid=True) as calls:
        assert _centres(mol) == [1, 3]
    assert calls == []


def test_the_centres_do_not_depend_on_where_the_charge_is_drawn():
    for smiles in ("C[n+]1cn(-%s)c2ccccc21" % _phenylenes(8), "Cn1c[n+](-%s)c2ccccc21" % _phenylenes(8)):
        mol = Chem.MolFromSmiles(smiles)
        assert pi_work(mol) > B0
        centre = _centre(mol)
        with supplier_calls(forbid=True):
            got = _centres(mol)
        assert got[0] == centre and len(got) == 2 and got[1] != centre
        assert {mol.GetAtomWithIdx(i).GetSymbol() for i in got} == {"N"}


def test_a_bonded_metal_above_the_bound_is_the_drawn_centre_alone_and_never_a_supplier():
    # the matching does not draw the charge and bond moves of a bonded metal: undecided, and RDKit is not asked
    mol = Chem.MolFromSmiles("C[n+]1cn(-%s-[Sn](C)(C)C)c2ccccc21" % _phenylenes(7))
    assert pi_work(mol) > B0
    why = []
    assert charge_shift.charge_placements(mol, why=why) is None and why == ["metal"]
    with supplier_calls(forbid=True) as calls:
        assert _centres(mol) == [_centre(mol)]
    assert calls == []


def test_an_error_in_the_matching_is_the_drawn_centre_alone_and_never_a_supplier(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("the matching failed")

    monkeypatch.setattr(charge_shift, "charge_placements", fail)
    mol = Chem.MolFromSmiles("C[n+]1cn(-%s)c2ccccc21" % _phenylenes(6))
    assert pi_work(mol) > B0
    with supplier_calls(forbid=True) as calls:
        assert _centres(mol) == [_centre(mol)]
    assert calls == []


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", ["C[n+]1ccn(C)c1", "CCn1cc[n+](C)c1", "C[n+]1cn(-c2ccccc2)c2ccccc21"])
def test_the_matching_route_gives_the_name_of_the_rdkit_route(smiles, monkeypatch):
    jar_or_skip()
    mol = Chem.MolFromSmiles(smiles)
    centre = _centre(mol)
    ring_system = ions._ring_system_of(mol, centre)
    monkeypatch.setattr(charge_shift, "RDKIT_RESONANCE0_MAX_WORK", RDKIT_ONLY)
    want = azolium_cations.azolium_demote_name(mol, ring_system, centre)
    monkeypatch.setattr(charge_shift, "RDKIT_RESONANCE0_MAX_WORK", MATCHING_ONLY)
    with supplier_calls(forbid=True) as calls:
        got = azolium_cations.azolium_demote_name(mol, ring_system, centre)
    assert "_charge_structures" not in [c[0] for c in calls]
    assert got == want != ""


_CHILD = """
import json, sys, time
from rdkit import Chem
from rdkit.Chem import rdchem
from orthonym.rules import azolium_cations, ions
built = []
def refuse(mol, *args, **kwargs):
    built.append(mol.GetNumHeavyAtoms())
    raise RuntimeError("a resonance supplier must not be built here")
Chem.ResonanceMolSupplier = refuse
rdchem.ResonanceMolSupplier = refuse
out = {}
for name, smiles in json.loads(sys.argv[1]).items():
    mol = Chem.MolFromSmiles(smiles)
    centre = [a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge() == 1 and a.GetIsAromatic()][0]
    start = time.process_time()
    got = azolium_cations.azolium_hydron_centres(mol, ions._ring_system_of(mol, centre), centre)
    out[name] = {"centres": got, "cpu": time.process_time() - start}
print(json.dumps({"built": built, "rows": out}))
"""

HARD_TIMEOUT_S = 60
CPU_LIMIT_S = 5.0


def test_azoliums_that_rdkit_does_not_finish_leave_the_centres_within_seconds():
    # a child process: RDKit's C++ call holds the GIL and ignores signals, so only a kill from outside ends a hang
    source_root = str(Path(orthonym.__file__).resolve().parents[1])
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [source_root, os.environ.get("PYTHONPATH")])))
    rows = {"ladder%d" % n: "C[n+]1cn(-%s)c2ccccc21" % _phenylenes(n) for n in (9, 12, 16)}
    rows["imidazolium12"] = "C[n+]1ccn(-%s)c1" % _phenylenes(12)
    try:
        done = subprocess.run([sys.executable, "-c", _CHILD, json.dumps(rows)], capture_output=True, text=True,
                              timeout=HARD_TIMEOUT_S, env=env)
    except subprocess.TimeoutExpired:
        pytest.fail("azolium_hydron_centres did not return in %d s" % HARD_TIMEOUT_S, pytrace=False)
    assert done.returncode == 0, done.stderr[-2000:]
    result = json.loads(done.stdout.strip().splitlines()[-1])
    assert result["built"] == []
    for name, entry in result["rows"].items():
        assert len(entry["centres"]) == 2, name
        assert entry["cpu"] < CPU_LIMIT_S, (name, entry["cpu"])
