""" M0 — recover fused-polycyclic abstains blocked by two coupled defects.

Root cause (traced 2026-09-08, internal notes):

1. Fused-ring `iupac_locants` catalog tables store fusion/prime locants as BARE
   STRINGS (`'4a'`, `'8a'`, `'10b'`). They reach ``compare_locant_sets`` through the
   ring parent-selection pool (``p44_scorer._candidate_locants``) UNCOERCED, and the
   fast-path ``sorted`` then raised
   ``TypeError: '<' not supported between 'str' and 'int'`` on a str/int-mixed set —
   which crashed ``_classify`` and abstained N fused polycyclics. Fixed by coercing
   bare-string locants to the declared ``int | (int, str)`` contract at the top of
   ``compare_locant_sets`` -correct: ``4 < 4a < 5``).

2. With the crash fixed, a genuinely combinatorial fused system runs the full main
   path and exhausts the macrocycle-hang budget (``PerfBudgetExceeded``), which used
   to abstain unconditionally. The outermost boundary now offers ONE bounded,
   strictly-OPSIN-RT-gated whole-molecule name before abstaining, so a molecule the
   coverage-by-construction producer can name cheaply and exactly is recovered — while
   a real hang witness (whose also explodes / does not round-trip) still abstains
   cleanly (that half is pinned in ``test_m25_workbudget.py``).

Both halves preserve 0-wrong: every recovered name OPSIN-round-trips to the input.
"""
import pytest

from orthonym.rules.locants import compare_locant_sets


# ── Part 1: comparator coercion (fast, no JVM) ──────────────────────────────

class TestCompareLocantSetsBareString:
    """`compare_locant_sets` must be total AND -correct on bare-string
    fusion locants, without changing the ordering of any pure-int / pure-tuple set."""

    def test_the_exact_crash_case_no_longer_raises(self):
        # measured live: parent-candidate comparison of a fused system's locant sets.
        # set_b sorts to [1,2,3,3a,4,5,6,7,7a]; first point of difference at index 3
        # is set_a's 4 vs set_b's 3a -> 4 > 3a -> set_b preferred (+1).
        assert compare_locant_sets([1, 2, 3, 4, 5, 6, 7, 8, 9],
                                   [1, 2, 3, 4, 5, 6, 7, '3a', '7a']) == 1

    def test_bare_string_orders_between_its_integer_neighbours(self):
        #: locant 4 is lower than 4a, and 4a is lower than 5.
        assert compare_locant_sets(['4a'], [4]) == 1     # 4 < 4a -> set_b preferred
        assert compare_locant_sets(['4a'], [5]) == -1    # 4a < 5 -> set_a preferred

    def test_multidigit_bare_strings_sort_numerically_not_lexically(self):
        # '10a' must sort AFTER '4a' (numeric base), not before it (lexical '1' < '4').
        assert compare_locant_sets(['10a'], ['4a']) == 1
        assert compare_locant_sets(['4a', '10a'], ['4a', '4b']) == 1  # 4b < 10a

    def test_pure_int_ordering_is_unchanged(self):
        assert compare_locant_sets([2, 3, 5], [3, 4, 6]) == -1
        assert compare_locant_sets([2, 4, 5], [2, 3, 5]) == 1
        assert compare_locant_sets([2, 3], [2, 3]) == 0

    def test_tuple_contract_path_is_unchanged(self):
        assert compare_locant_sets([(4, ''), (5, '')], [(4, 'a'), (5, '')]) == -1
        assert compare_locant_sets([(4, 'a'), (5, '')], [(4, 'b'), (5, '')]) == -1

    def test_prime_locant_string_is_accepted(self):
        # primed fusion locants ("2'") also appear bare in the catalog tables.
        assert compare_locant_sets(["2'"], [3]) == -1    # 2 < 3


# ── Part 2: the perf-budget rescue recovers a real giant, 0-wrong ─────────

# A 100-heavy-atom / 10-ring Si-free tetra-fused-heteroaromatic peptide-tail molecule
# (a temp dir/r1.smi). Its main path exhausts the op budget; the rescue names it via
# name_t4_complete and it round-trips full-InChIKey. Recovered from a clean abstain.
_GIANT = ("C1[C@@H](C2=C(N1C(=O)C3=CC4=C(N3)C=CC(=C4)NC(=O)C5=CC6=CC=CC=C6N5)"
          "C=C(C7=CC=CC=C72)OC(=O)OCCSSC[C@H](C(=O)O)NCCC[C@H](C(=O)O)NC(=O)"
          "CC[C@H](C(=O)O)NC(=O)CC[C@H](C(=O)O)NC(=O)C8=CC=C(C=C8)NCC9=CN=C1"
          "C(=N9)C(=O)NC(=N1)N)CCl")


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.roundtrip
@pytest.mark.opsin_gate
def test_perf_budget_giant_recovers_at_best_effort_and_round_trips():
    """The giant must (a) emit at the best-effort tier instead of abstaining, and
    (b) the emitted name must OPSIN-round-trip to the input's FULL InChIKey (0-wrong).
    Runs with the OPSIN validity gate on (the shipped default; the suite's autouse
    fixture turns it off otherwise): with the gate on the general engine names the
    giant; with it off a different, unparseable candidate is returned.
    Requires a JVM (OPSIN); skipped otherwise."""
    from orthonym.validation.opsin_roundtrip import _java_available
    if not _java_available():
        pytest.skip("OPSIN/JVM required: the perf-budget rescue is OPSIN-RT-gated")

    from rdkit import Chem
    from orthonym import Orthonym

    namer = Orthonym(general_fallback=True, general_fallback_unverified=True,
                      allow_aromatic_general=True)
    res = namer.name_tiered(_GIANT)
    name, tier = res.get("name"), res.get("tier")
    assert name and tier != "abstain", (
        f"giant abstained (perf-budget rescue did not fire): tier={tier}")

    # 0-wrong: the recovered name must denote the SAME molecule (full InChIKey).
    from orthonym.namer import _validity_gate_name_to_smiles
    opsmi = _validity_gate_name_to_smiles(name)
    assert opsmi is not None, f"recovered name does not OPSIN-parse: {name[:80]!r}"
    ik_in = Chem.MolToInchiKey(Chem.MolFromSmiles(_GIANT))
    ik_out = Chem.MolToInchiKey(Chem.MolFromSmiles(opsmi))
    assert ik_in == ik_out, (
        f"recovered name is a DIFFERENT molecule (0-wrong breach)\n"
        f" in={ik_in}\nout={ik_out}\nname={name[:120]!r}")
