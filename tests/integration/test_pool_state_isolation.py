"""Phase 145.1 SC-4 regression: prove pool state does NOT leak across
consecutive orthonym.name() calls. Mitigates T-145.1-02 (cross-call
state leak in _pool_store thread-local).
"""

from orthonym import Orthonym
from orthonym.assembly.candidate_pool import get_current_pool, clear_pool


class TestPoolStateIsolation:
    def setup_method(self):
        clear_pool()

    def teardown_method(self):
        clear_pool()

    def test_pool_cleared_between_calls(self):
        """Each orthonym.name() call starts with a fresh pool (D-09)."""
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
        """Risk 3: store_confidence(pool.best()) preserved at the new
        return site. name_with_confidence() must still return all 4 keys
        with non-empty values for a non-trivial molecule."""
        namer = Orthonym()
        result = namer.name_with_confidence("CCCCCCCCCc1cc(=O)c2ccccc2n1C")
        assert isinstance(result, dict)
        assert 'name' in result and result['name'], "name missing or empty"
        assert 'confidence' in result, "confidence missing"
        assert 'factors' in result and result['factors'], "factors empty"
        assert 'handler' in result and result['handler'] != 'unknown', (
            "handler defaulted to 'unknown' — store_confidence(best) likely "
            "missing at the pool-refactor return site (Risk 3 violation)."
        )

    def test_no_pool_state_leak_across_many_calls(self):
        """Hammer test: 100 sequential name() calls, no growth."""
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
