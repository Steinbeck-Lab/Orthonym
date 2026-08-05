"""v30 vB-engine Piece 1: the general ring engine must pass the P-44.1
principal-group hint into ring-system selection under the best-effort tier ONLY,
so PIN stays byte-identical by construction.

Tests ``name_general_ring`` directly with real perceived features -- NOT through
``name_tiered`` -- so the assertion depends on the wiring, not on OPSIN-gate
routing (which is nondeterministic under pytest in this repo).
"""
from rdkit import Chem

from orthonym.assembly import general_engine as ge
from orthonym.namer import Orthonym
from orthonym.rules import ring_selection

# A ring-assembly whose principal group (carboxamide) sits on a LESS senior ring
# (a benzene) than the more-senior N-heterocycle (quinoline). Selection must
# honour P-44.1 under best-effort.
SMILES = "COc1ccc(C(=O)Nc2nc3ccccc3c(NCc3ccccc3)c2C#N)cc1"


def _features():
    nm = Orthonym(style="pin")
    mol = Chem.MolFromSmiles(SMILES)
    features = nm._perceive(mol, SMILES, Chem.MolToSmiles(mol))
    nm._classify(features)
    return mol, features


def _hints_seen(allow_aromatic_general):
    mol, features = _features()
    orig = ring_selection.select_principal_ring_system
    seen = []

    def spy(m, ring_systems, principal_group_atoms=None):
        seen.append(principal_group_atoms)
        return orig(m, ring_systems, principal_group_atoms=principal_group_atoms)

    ring_selection.select_principal_ring_system = spy
    try:
        ge.name_general_ring(mol, features,
                             allow_aromatic_general=allow_aromatic_general)
    finally:
        ring_selection.select_principal_ring_system = orig
    return seen


def test_best_effort_passes_the_principal_group_hint():
    seen = _hints_seen(allow_aromatic_general=True)
    assert any(h for h in seen), \
        "best-effort must pass a non-empty principal_group_atoms hint"


def test_pin_passes_no_hint_byte_identical():
    seen = _hints_seen(allow_aromatic_general=False)
    assert all(h is None for h in seen), \
        "PIN must pass no hint so ring selection is byte-identical"


def test_bridging_group_does_not_trigger_the_override():
    """Regression-safety: the carboxamide here BRIDGES two rings (C to the
    benzene, N to the quinoline), so BOTH ring systems 'bear' it. The override
    fires only on EXACTLY ONE bearing ring, so this falls through to the
    unchanged P-44.2 scoring (the quinoline) rather than guessing -- exactly the
    positive-evidence discipline that keeps the change from regressing.
    """
    mol, features = _features()
    with_hint = ring_selection.select_principal_ring_system(
        mol, features.ring_systems,
        principal_group_atoms=list(features.principal_group_atoms or []))
    without = ring_selection.select_principal_ring_system(
        mol, features.ring_systems)
    assert set(with_hint) == set(without), \
        "a group bridging two rings must not trigger the single-ring override"
