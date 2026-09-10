""" CT.5 -- sub-lever D (true 3-way: spiro + fused + bridged) cage naming.

 a trace OUTCOME (a project rule/10/17): DOCUMENTED NO-OP + NAMED-BLOCKER.

The task charter proposed an offer-not-return retry in
``name_bridged_fused_system`` (``bridged_fused.py:617``): the premise was that the
handler "IS invoked for true spiro+fused+bridged witnesses but fragment-collapses
(returns a wrong fragment name -> rejects), fix = try alternative
parent/bridge-assignment candidates and keep the first that RT-verifies".

A fresh trace on current HEAD REFUTES that premise:

  1. The residual is REAL -- both grounding witnesses still ABSTAIN at best-effort
     (0 breadth), and 0-wrong holds (/ OPSIN gate reject every fragment).

  2. ``name_bridged_fused_system`` returns **None** for these -- it does NOT
     fragment-collapse. The fragment the pipeline finally rejects
     (``methyl 2-hydroxy-4-methylbenzoate`` for notoamide) comes from a DOWNSTREAM
     benzene/general handler, not this function.

  3. Class-wide (the full 1,049 in-scope ring-bearing best-effort-abstainer pool):
     only 18 molecules are composer-classified 'bridged-fused', and
     ``name_bridged_fused_system`` returns None for **all 18** -- 12 have an
     unnamed complex core + zero bridges found, 6 have a named core + zero bridges
     (several are salts/charged = out-of-construction). **Zero** reach the
     candidate-assembly code, so ``identify_bridges`` returns empty and there are
     NO bridge assignments to offer alternatives for. The offer-not-return lever
     has an EMPTY input set -> building it is dead code (a project rule).

  4. Not even every 3-way witness routes through this site: the spiro-epoxide
     macrolactam classifies 'polycyclic-bridged' and routes to
     ``name_polycyclic_complete`` (von Baeyer), which raises UNSUPPORTED_RING_SYSTEM
     -- so the brief's named site is OFF-PATH for that witness. The von-Baeyer
     universal net (``analyze_cage_universal``) also returns None for both
     witnesses AND their bare Murcko scaffolds -> no producer can build a candidate.

VERDICT: CT.5 is a genuine SKELETON-CONSTRUCTION gap (aromatic-fused rings +
spiro + bridge composite that needs a dedicated fused+bridged+spiro constructor,
or an aromatic-fused-capable von-Baeyer engine). It is a NAMED-BLOCKER; 0-wrong
already holds via abstain. No ``src/`` change is warranted.

These tests LOCK that finding as a deterministic canary. If a future
ring-construction follow-on teaches a producer to build these, a test here will
fail -- that is the intended signal to update the assertion (change-asserted-value)
and re-scope the blocker, NOT a regression.
"""
from rdkit import Chem

from orthonym.assembly.composer import _classify_complex_ring
from orthonym.rules.bridged_fused import name_bridged_fused_system
from orthonym.rules.vonbaeyer_universal import analyze_cage_universal

# The two grounding CT.5 witnesses (V37-a trace-CONSTRUCTION.md CT.0 addendum).
NOTOAMIDE = (
    "C=CC(C)(C)c1[nH]c2ccccc2c1C=C1NC(=O)[C@]23C[C@H]"
    "(c4c(c(C)cc(O)c4C(=O)OC)O2)[C@]2(CC(=O)C=C(OC)C2=O)N3C1=O"
)
SPIRO_EPOXIDE_MACROLACTAM = (
    "CC(=O)O[C@@H]1/C=C/[C@@](C)(O)C(=O)[C@@H](C)C/C=C/[C@H]2[C@H](O)"
    "[C@]3(CO3)[C@@H](C)[C@H]3[C@H](Cc4ccccc4)NC(=O)[C@@]312"
)


def test_notoamide_routes_to_bridged_fused_but_builds_no_candidate():
    """notoamide: composer routes to the brief's named site, which declines.

    Confirms the site IS on-path for this witness (ring_type 'bridged-fused')
    but ``name_bridged_fused_system`` builds NO candidate (returns None) -- there
    is nothing for an offer-not-return retry to re-assign.
    """
    mol = Chem.MolFromSmiles(NOTOAMIDE)
    assert mol is not None
    assert _classify_complex_ring(mol) == "bridged-fused"
    assert name_bridged_fused_system(mol) is None


def test_spiro_epoxide_macrolactam_is_off_path_for_bridged_fused():
    """spiro-epoxide macrolactam: the brief's named site is OFF-PATH.

    It classifies 'polycyclic-bridged' (routes to von Baeyer, not
    name_bridged_fused_system), and the bridged-fused handler returns None anyway.
    """
    mol = Chem.MolFromSmiles(SPIRO_EPOXIDE_MACROLACTAM)
    assert mol is not None
    assert _classify_complex_ring(mol) == "polycyclic-bridged"
    assert name_bridged_fused_system(mol) is None


def test_von_baeyer_universal_net_offers_no_candidate_for_either_witness():
    """No producer can build a candidate: the von-Baeyer universal net returns
    None for both witnesses -- so there is nothing to offer-and-RT-gate either."""
    for smi in (NOTOAMIDE, SPIRO_EPOXIDE_MACROLACTAM):
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None
        assert analyze_cage_universal(mol) is None
