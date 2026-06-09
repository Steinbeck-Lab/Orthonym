"""WSD-04 (RING-06) — Group-14/13 (Si/Sn/Ge/B) heterocycle names as a ring,
not an acyclic substituted hydride.

Wave-0 scaffold (Phase 175). TARGET assertion is xfail until the WSD-04 code plan
(175-03) excludes a single ring-member Group-14/13 atom from `detect_metal_complex`
(perception/metals.py), so the existing (already-correct) Hantzsch-Widman ring path
emits the ring name instead of the organometallic CFR handler opening the ring.

Blue Book P-22.2 / P-21: sila/germa/stanna/bora are valid HW 'a'-prefixes; a ring
must be named as a ring, never as a substituted acyclic hydride.
"""

import pytest

from orthonym.namer import name_compound


@pytest.mark.unit
class TestSilacycleRing:
    @pytest.mark.xfail(reason="WSD-04 fix lands in Plan 175-03", strict=False)
    def test_silolane_named_as_ring(self):
        # Currently 'butylsilane' (ring opened to a Si-anchored chain).
        name = name_compound("C1CCC[SiH2]1").strip().lower()
        assert name in {"silolane", "1-silacyclopentane"}, f"Si ring opened: {name!r}"
