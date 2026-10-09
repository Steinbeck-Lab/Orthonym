"""Four large molecules the decomposition search once lost, named at every tier that named them.

Each has an ester (or a carbamate) whose alcohol holds the rest of the molecule: 126 atoms of a
cyclic dinucleotide with a polyethylene-glycol arm (two stereoisomers), 66 atoms of a vinca-type
dimer salt. The ester assembler (``fragment_assembly._assemble_ester``) proves its group word by a
round trip and cannot build a word for a group that size, so it returns no name. The decomposition
search then took that for an unsuccessful split and opened every other bond of the molecule,
naming each remainder in full: the 500-call analysis budget ran out and the molecule was abstained
(valid and complete tiers: STRUCTURE_TOO_LARGE for the two dinucleotides, UNNAMEABLE for the
dimer) although the general pipeline names each in 10-30 s. The assembler now reports the name it
could not prove (``outcome['unproven']``; the name itself is never returned), the search
remembers that split for the naming scope and does not name it again, and once it has met one it
may spend only one eighth of the analysis budget on the other bonds
(tests/unit/decomposition/test_unproven_split_search.py).

The fourth molecule (a spiro macrocycle, best-effort tier) does not run the assemblers at all: it
names in about 30 s through the rescue at the budget boundary. It is kept here so that the
tier outcome of all four is pinned.

The OPSIN validity gate is ON (``opsin_gate``): the tiers are the production configuration the
losses were measured in (the suite default, gate off, names the spiro macrocycle at best-effort
without its stereodescriptors).

The name is checked against the input's full InChIKey by a FRESH OPSIN parse outside the engine
(tests/support/rt_assert.py::assert_full_rt). A tier that abstains, or a name that describes
another molecule, fails.
"""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.errors import is_failure_name
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.opsin_gate]

CYCLIC_DINUCLEOTIDE_A = (
    "CC(=O)O[C@@H]1[C@H](C[C@H]([C@@H]([C@H]1OC(=O)C)OC(=O)C)OC2=C(C=C(C=C2)COC(=O)N(CCOCCOCCOCCOCCOCCOCCOCCOC)CC3=CC=CC=C3C(=O)NC4=NC5=C(C(=O)N4)N=CN5[C@H]6[C@H]7[C@@H]([C@H](O6)CO[P+](=O)O[C@H]8C[C@@H](C[C@@H]8COPO7)OC9=NC=NC=C9)O)NC(=O)CCNC(=O)OCC1C2=CC=CC=C2C2=CC=CC=C12)C(=O)OC"
)
VINCA_TYPE_DIMER_SALT = (
    "CCC1=C[N+]2(CC(C1)C[C@@](C3=C(C2)C4=CC=CC=C4N3)(C5=C(C=C6C(=C5)[C@]78CCN9[C@H]7[C@@](C=CC9)([C@H]([C@@](C8N6C)(C(=O)OC)O)OC(=O)C)CC)OC)C(=O)OC)CC(=O)N1CCC[C@H]1C(=O)OC.[Br-]"
)
CYCLIC_DINUCLEOTIDE_B = (
    "CC(=O)O[C@@H]1[C@H](C[C@H]([C@@H]([C@H]1OC(=O)C)OC(=O)C)OC2=C(C=C(C=C2)COC(=O)N(CCOCCOCCOCCOCCOCCOCCOCCOC)CC3=CC=CC=C3C(=O)NC4=NC5=C(C(=O)N4)N=CN5[C@@H]6C[C@@H]7CO[P+](=O)O[C@H]8C[C@@H](C[C@@H]8COPO[C@@H]6[C@@H]7O)OC9=NC=NC=C9)NC(=O)CCNC(=O)OCC1C2=CC=CC=C2C2=CC=CC=C12)C(=O)OC"
)
SPIRO_MACROCYCLE = (
    r"C[C@@H]1CC[C@@]2([C@H]([C@H]3[C@@H](O2)C[C@@H]4[C@@]3(CC[C@H]5[C@H]4/C/6=N\OCC(=O)N(CC(=O)NC7=CC=C(C=C7)OC8=CC=C(C=C8)NC(=O)CN(C(=O)CO/N=C/9\CC[C@]5([C@@H](C9)C6)C)[C@@H](C)C(=O)OC)[C@@H](C)C(=O)OC)C)C)OC1"
)

# (smiles, tier, the molecule must be named without a hang budget running out). The spiro
# macrocycle is named by the rescue at the budget boundary BY DESIGN (the same trip on every
# tree, measured), so only its name is pinned.
ROWS = [
    (CYCLIC_DINUCLEOTIDE_A, "valid", True),
    (CYCLIC_DINUCLEOTIDE_A, "complete", True),
    (CYCLIC_DINUCLEOTIDE_A, "best-effort", True),
    (VINCA_TYPE_DIMER_SALT, "valid", True),
    (VINCA_TYPE_DIMER_SALT, "complete", True),
    (CYCLIC_DINUCLEOTIDE_B, "valid", True),
    (CYCLIC_DINUCLEOTIDE_B, "complete", True),
    (CYCLIC_DINUCLEOTIDE_B, "best-effort", True),
    (SPIRO_MACROCYCLE, "best-effort", False),
]
IDS = ["dinucleotide-A-valid", "dinucleotide-A-complete", "dinucleotide-A-best-effort",
       "vinca-dimer-valid", "vinca-dimer-complete", "dinucleotide-B-valid",
       "dinucleotide-B-complete", "dinucleotide-B-best-effort", "spiro-macrocycle-best-effort"]


@pytest.mark.parametrize("smiles,tier,no_trip", ROWS, ids=IDS)
def test_large_molecule_is_named_and_round_trips(smiles, tier, no_trip):
    from orthonym.assembly.fragment_naming import hang_budget_trips

    trips = hang_budget_trips()
    row = Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)
    name = row.get("name")
    assert name and not is_failure_name(name) and row.get("tier") != "abstain", row
    assert_full_rt(name, smiles, what=f"{tier} tier: ")
    if no_trip:
        # The loss was the 500-call analysis budget running out (the molecule abstained at the
        # valid and complete tiers, and ran 64 s to the same name at best-effort, the rescue
        # at the budget boundary): a molecule the general pipeline names in 10-30 s trips
        # nothing. A counter of operations, not a clock, so it holds under load.
        assert hang_budget_trips() == trips, "a hang budget ran out while naming it"
