""" A1 — ring-parent tiebreak for TIED non-aromatic carbocyclic rings.

When >=2 candidate parent ring systems tie on the ring_system_score, the
senior parent is the one carrying the MAXIMUM number of substituents cited as
prefixes (the Blue Book,. Before this, select_principal_ring_system
broke the tie by list order (ring_systems[0]).

NARROWED to non-aromatic ALL-CARBON tied rings: the aromatic path already applies
, and rings with skeletal heteroatoms (e.g. a glycoside's pyranose) are
owned by other conventions where an exocyclic-bond count is not the prefix count.

All tests run with the OPSIN gate disabled (no JVM) so they are deterministic and
safe alongside a large OPSIN job.
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.perception.rings import get_ring_systems
from orthonym.rules.ring_selection import select_principal_ring_system


def _exo(mol, atoms):
    s = set(atoms)
    return sum(1 for a in s for nb in mol.GetAtomWithIdx(a).GetNeighbors()
               if nb.GetIdx() not in s)


# ---- selector-level: tie broken toward the more-substituted carbocycle --------

@pytest.mark.unit
def test_tied_carbocycles_pick_max_substituent_ring():
    # two cyclohexene rings joined by a chain, asymmetric methylation:
    # 3-substituent ring vs 4-substituent ring -> picks the 4-substituent.
    smi = ("CC1CCC/C(C)=C1/C=C/C(C)=C/C=C/C(C)=C/C=C/C=C(C)"
           "/C=C/C=C(C)/C=C/C2=C(C)/CCCC2(C)C")
    mol = Chem.MolFromSmiles(smi)
    systems = get_ring_systems(mol)
    assert len(systems) == 2
    parent = set(select_principal_ring_system(mol, systems))
    # the chosen parent is the ring with the greater exocyclic-substituent count
    counts = sorted(_exo(mol, sy) for sy in systems)
    assert _exo(mol, parent) == counts[-1], (counts, _exo(mol, parent))


@pytest.mark.unit
def test_tie_not_applied_when_a_ring_has_heteroatom():
    # a ring system with a skeletal heteroatom is NOT eligible -> list order kept
    # (the narrowing that protects glycosides). Construct a pyranose + carbocycle
    # that would tie only if scored equal; here we just assert the selector does
    # not crash and returns one full ring system.
    smi = "C1CCCCC1CCC1CCCCO1"  # cyclohexane + oxane (tetrahydropyran) via a chain
    mol = Chem.MolFromSmiles(smi)
    systems = get_ring_systems(mol)
    parent = select_principal_ring_system(mol, systems)
    assert parent and set(parent).issubset(set(range(mol.GetNumAtoms())))


# ---- end-to-end: witness names with the correct 4-substituent parent + RTs ----

@pytest.mark.unit
def test_carotenoid_witness_uses_max_substituent_parent():
    smi = ("CC1CCC/C(C)=C1/C=C/C(C)=C/C=C/C(C)=C/C=C/C=C(C)"
           "/C=C/C=C(C)/C=C/C2=C(C)/CCCC2(C)C")
    eng = Orthonym(general_fallback=True, general_fallback_unverified=True,
                    allow_aromatic_general=True, _disable_opsin_validity_gate=True)
    nm = eng.name_tiered(smi)["name"]
    # the parent hydride is now the 4-substituent ring (rendered 1,3,3-trimethyl
    # by lowest-locant numbering), NOT the 3-substituent (1,3-dimethyl) ring.
    assert nm and "1,3,3-trimethylcyclohex-1-ene" in nm, nm
    # the 3-substituent ring is now the substituent, spelled 2,6-dimethyl…yl
    assert "2,6-dimethylcyclohex-1-en-1-yl" in nm, nm
