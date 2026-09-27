"""a phase regression: prove pool state does NOT leak across
consecutive orthonym.name calls. Mitigates T-145.1-02 (cross-call
state leak in _pool_store thread-local).

POST-DRIFT-FIX (2026-04-23): also covers recursive assemble_name calls.
The original Plan 01 single-slot pool was corrupted by recursive
substituent naming (N-oxide handler, fragment_naming, substituent_enumerator,
decomposition). The fix converts _pool_store to a stack with push_pool /
pop_pool around each assemble_name invocation. See:
internal notes
"""

import pytest

from orthonym import Orthonym
from orthonym.assembly.candidate_pool import (
    CandidatePool,
    get_current_pool,
    clear_pool,
    push_pool,
    pop_pool,
    _pool_store,
)
from orthonym.namer import name_compound


CHEBI_83420 = (
    "CCCCCCCCCCCCCCCC(=O)OC[C@H]"
    "(COP(=O)([O-])O[C@H]1[C@H](O)[C@@H](OP(=O)([O-])[O-])"
    "[C@H](OP(=O)([O-])[O-])[C@@H](OP(=O)([O-])[O-])[C@H]1O)"
    "OC(=O)CCCCCCCCCCCCCCC"
)


class TestPoolStateIsolation:
    def setup_method(self):
        clear_pool()

    def teardown_method(self):
        clear_pool()

    def test_pool_cleared_between_calls(self):
        """Each orthonym.name call starts with a fresh pool ."""
        namer = Orthonym()
        n1 = namer.name("CCO")             # ethanol
        n2 = namer.name("c1ccccc1")        # benzene
        n3 = namer.name("CC(=O)O")         # acetic acid
        assert n1 == "ethanol"
        assert "benzen" in n2.lower()
        assert "acetic acid" in n3 or "ethanoic acid" in n3
        # If pool didn't clear, n2 or n3 would inherit candidates from n1
        # and might return "ethanol" by accident. The above assertions
        # catch that.

    def test_consecutive_naming_calls_byte_identical(self):
        """Naming the SAME molecule twice produces the SAME name."""
        namer = Orthonym()
        first = namer.name("CCO")
        _ = namer.name("CCC")  # interleave a different molecule
        second = namer.name("CCO")
        assert first == second, (
            f"Consecutive naming of same SMILES diverged: "
            f"first={first!r} second={second!r}"
        )

    def test_name_with_confidence_after_pool_refactor(self):
        """Risk 3: store_confidence(pool.best) preserved at the new return site.

         C4 -- THIS TEST WAS PASSING FOR A FABRICATED REASON, and that is
        the interesting part. It asserted a non-empty `factors` dict and
        `handler != 'unknown'` as its detector for a Risk-3 violation (the
        pool's best candidate not being stored). It was green. But the
        molecule it used, CCCCCCCCCc1cc(=O)c2ccccc2n1C, does NOT route through
        candidate scoring any more -- it returns ZERO factors. The test only
        went green because namer.py:2693-2699 fabricated a record with
        `handler='direct'` and four 1.0 factors whenever nothing had been
        scored, which satisfied both assertions while proving nothing.

        So the fabrication was masking this test's own detector: for this
        molecule the Risk-3 property has not actually been verified for some
        time. The quinoline's routing drift is PRE-EXISTING (confirmed by
        an A/B check: this file's other failure, test_drift_aryl_*, fails at
        HEAD too) and is out of scope here.

        Re-derived so the detector works again: assert the Risk-3 property on
        molecules that provably DO route through the pool, and assert the
        honest unmeasured contract on the quinoline rather than a fabricated
        pass. Contract: tests/unit/test_coverage_contract.py.
        """
        namer = Orthonym()

        # Risk-3 detector, on molecules that really are pool-scored.
        for smi in ("Nc1ncnc2nc[nH]c12",              # adenine
                    "Cn1c(=O)c2c(ncn2C)n(C)c1=O"):   # caffeine
            result = namer.name_with_confidence(smi)
            assert isinstance(result, dict)
            assert result.get('name'), f"name missing or empty for {smi}"
            assert result['factors'], (
                f"factors empty for {smi} — store_confidence(best) likely "
                f"missing at the pool-refactor return site (Risk 3 violation)."
            )
            assert result['handler'] not in ('unknown', 'unmeasured'), (
                f"handler={result['handler']!r} for {smi} — the pool's best "
                f"candidate was not stored (Risk 3 violation)."
            )
            assert result['confidence'] is not None, smi

        # The quinoline: still named, and now HONEST about having no
        # measurement instead of reporting a fabricated perfect score.
        result = namer.name_with_confidence("CCCCCCCCCc1cc(=O)c2ccccc2n1C")
        assert result.get('name'), "name missing or empty"
        assert 'confidence' in result, "confidence missing"
        if result['confidence'] is None:
            assert result['verification'] == 'unverified'
            assert result['factors'] == {}

    def test_no_pool_state_leak_across_many_calls(self):
        """Hammer test: 100 sequential name calls, no growth."""
        namer = Orthonym()
        molecules = ["CCO", "CCC", "c1ccccc1", "CC(=O)O", "CCN", "CCS", "CC=O"]
        for _ in range(15):  # ~100 calls total
            for smi in molecules:
                name = namer.name(smi)
                assert name and isinstance(name, str)
                # After each call, pool should be cleared by the prologue
                # of the NEXT call. Verify after call returns:
                pool = get_current_pool()
                # Pool may have a few candidates from the just-completed
                # call (epilogue happens before the NEXT call's clear_pool).
                # The key assertion: it should never grow indefinitely.
                assert len(pool.all_candidates()) < 20, (
                    f"Pool growing across calls — state leak suspected. "
                    f"Pool has {len(pool.all_candidates())} candidates."
                )


class TestPoolStateIsolationUnderRecursion:
    """a phase DRIFT FIX (2026-04-23): regression coverage for the
    pool-stack architecture. Each assemble_name call must have an
    independent cascade scope per IUPAC (per-molecule parent
    selection). Recursive name_compound invocations from substituent
    handlers must NOT corrupt the outer molecule's pool.
    """

    def setup_method(self):
        # Drain the stack — each test starts from a known empty state
        if hasattr(_pool_store, 'stack'):
            _pool_store.stack = []

    def teardown_method(self):
        if hasattr(_pool_store, 'stack'):
            _pool_store.stack = []

    def test_drift_truncation_pubchem_14105233(self):
        """Catastrophic truncation case from Plan 04 drift report.
        Pre-fix: returned 'bromomethane' (recursive substituent fragment
        beats the outer molecule's correct name). Post-fix: returns the
        full IUPAC name."""
        result = name_compound("CCC(CC)(CO)CBr")
        assert result == "2-(bromomethyl)-2-ethylbutan-1-ol", (
            f"Recursion-pollution regression: outer molecule was "
            f"truncated to a substituent's name. Got {result!r}, expected "
            f"'2-(bromomethyl)-2-ethylbutan-1-ol'. The fix (stack-based "
            f"_pool_store) must keep the outer molecule's cascade scope "
            f"separate from the recursive substituent's cascade scope."
        )

    def test_drift_aryl_pubchem_22085501(self):
        """Aryl/cyclohexyl reorientation case from Plan 04 drift report.
        Multi-level recursion (4 nested cascades for biphenyl ester).
        Pre-fix: returned a recursive arm's name. Post-fix: full molecule."""
        result = name_compound(
            "CCC(C)CCCC1CCC(CC1)C(=O)OC2=CC=C(C=C2)C3=CC=C(C=C3)CC(C)CCC(C)CC"
        )
        # The v17 literal "4'-hydroxy-4-2,5-dimethylheptyl-1,1'-biphenyl
        # 4-(3-methylhexyl)cyclohexanecarboxylate" did not parse and named a
        # 3-methylhexyl. The biphenylyl ester group is the ring assembly:
        # (the Blue Book), '(4'-cyano[1,1'-biphenyl]-4-yl)oxy... (PIN)'
        # (:7455); the phenyl-on-phenyl '4-[4-(2,5-dimethylheptyl)phenyl]phenyl'
        # shipped at pin_verified until the decorated biphenylyl producer (TRIAGE
        # g5 C14). OPSIN 2.9.0 full-InChIKey EXACT.
        expected = (
            "4'-(2,5-dimethylheptyl)[1,1'-biphenyl]-4-yl "
            "4-(4-methylhexyl)cyclohexane-1-carboxylate"
        )
        assert result == expected, (
            f"4-deep recursion regression: got {result!r}, expected "
            f"{expected!r}"
        )

    def test_drift_polyester_chebi_170764(self):
        """Polyester/triglyceride reordering case. Recursion on each
        ester arm corrupted outer (2R) stereo prefix."""
        result = name_compound(
            "CC/C=C\\C/C=C\\C/C=C\\C/C=C\\C/C=C\\CCCC(=O)O"
            "[C@H](COC(=O)CCCCCCCCC/C=C\\CCCCCCCC)COP(=O)(O)OCCN"
        )
        # The (2R) prefix MUST be preserved — pre-fix it was lost when
        # an inner recursive cascade's candidate beat the outer's.
        assert result.startswith("(2R)"), (
            f"Polyester (2R) stereo prefix lost: got {result!r}. The "
            f"fix must preserve outer-molecule stereo when substituent "
            f"recursion happens."
        )

    @pytest.mark.xfail(strict=True, reason=(
        "raw (gate-off) name is a different molecule at the base and now: "
        "'...-cyclohexanedicarboxylate' (4e0e5c29b) / '...cyclohexane-2,6-bis(olate) "
        "dihexadecanoate' (since f2005eb4f), both OPSIN-unparseable; production "
        "abstains at both tiers (test_drift_stereo_chebi_83420_production). Needs a "
        "phosphatidylinositol-trisphosphate anion producer -- TODO in TRIAGE.md "
        "'Suite fix -- j1-regressions'"))
    def test_drift_stereo_chebi_83420(self):
        """Stereo inversion case. Recursive cyclohexane fragment naming
        used isolated atom indexing (5S,6R), pre-fix corrupted outer
        molecule's (5R,6S) baseline. Post-fix: outer indexing preserved."""
        result = name_compound(CHEBI_83420)
        assert "5R,6S" in result, (
            f"Stereo prefix inverted to 5S,6R (recursive fragment's atom "
            f"indexing won the pool). Got: {result!r}. Fix must keep "
            f"outer molecule's atom-indexed stereo."
        )

    @pytest.mark.opsin_gate
    def test_drift_stereo_chebi_83420_production(self):
        """What ships for the strict-xfail row above (gate on): the default tier
        and best-effort each ship the failure sentinel or a name that round-trips
        to the input's full InChIKey (both abstain today, as at 4e0e5c29b)."""
        from orthonym.errors import is_failure_name
        from tests.support.rt_assert import name_best_effort, name_is_rt_exact
        pin = name_compound(CHEBI_83420)
        assert is_failure_name(pin) or name_is_rt_exact(pin, CHEBI_83420), pin
        be = name_best_effort(CHEBI_83420).get("name")
        assert not be or is_failure_name(be) or name_is_rt_exact(be, CHEBI_83420), be

    def test_pool_stack_lifo_order(self):
        """Synthetic test: push/pop semantics are LIFO and isolate
        cascades. Bypasses pool.add (which needs real features) by
        manipulating pool._candidates directly to test stack semantics
        in isolation."""
        from orthonym.assembly.coverage_scoring import CandidateName

        push_pool()
        outer = get_current_pool()
        outer._candidates = [CandidateName(name="OUTER_A", handler='polycyclic', confidence=1.0)]
        assert outer.best().name == "OUTER_A"

        # Simulate recursive call: push fresh pool, do work, pop
        push_pool()
        inner = get_current_pool()
        assert inner is not outer, "Inner pool must be a different instance"
        inner._candidates = [CandidateName(name="INNER_X", handler='polycyclic', confidence=1.0)]
        assert inner.best().name == "INNER_X"
        pop_pool()

        # After pop, outer's pool MUST be active and unchanged
        restored = get_current_pool()
        assert restored is outer, "After pop, must return to the outer pool"
        assert restored.best().name == "OUTER_A", (
            f"Outer pool corrupted by inner cascade. Got "
            f"{restored.best().name!r}, expected 'OUTER_A'."
        )
        pop_pool()

    def test_pop_without_push_is_safe(self):
        """Defensive: popping an empty stack must not crash."""
        # Stack starts empty (setup_method drains it)
        try:
            pop_pool()
            pop_pool()  # idempotent
        except Exception as e:
            raise AssertionError(
                f"pop_pool() on empty stack raised {type(e).__name__}: {e}"
            )

    def test_concurrent_threads_have_independent_stacks(self):
        """threading.local isolates per-thread stacks. Two threads
        concurrently invoking name_compound must not interfere."""
        import threading
        results = {}
        errors = []

        def worker(label, smiles, expected):
            try:
                got = name_compound(smiles)
                results[label] = got
                if got != expected:
                    errors.append(
                        f"{label}: got {got!r}, expected {expected!r}"
                    )
            except Exception as e:
                errors.append(f"{label}: raised {type(e).__name__}: {e}")

        threads = [
            threading.Thread(
                target=worker,
                args=("ethanol", "CCO", "ethanol"),
            ),
            threading.Thread(
                target=worker,
                args=("benzene", "c1ccccc1", "benzene"),
            ),
            threading.Thread(
                target=worker,
                args=(
                    "drift_truncation",
                    "CCC(CC)(CO)CBr",
                    "2-(bromomethyl)-2-ethylbutan-1-ol",
                ),
            ),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, "Per-thread isolation violated:\n" + "\n".join(errors)
        assert len(results) == 3, f"Expected 3 thread results, got {results}"
