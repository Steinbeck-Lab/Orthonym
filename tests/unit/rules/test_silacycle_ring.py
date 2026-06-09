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
    def test_silolane_named_as_ring(self):
        # WSD-04 fixed (Plan 175-03): was 'butylsilane' (ring opened).
        name = name_compound("C1CCC[SiH2]1").strip().lower()
        assert name in {"silolane", "1-silacyclopentane"}, f"Si ring opened: {name!r}"

    def test_silole_named_as_ring(self):
        assert name_compound("C1=CC=C[SiH2]1").strip().lower() in {"silole", "1h-silole"}

    def test_phospholane_untouched(self):
        # OUT OF SCOPE: P is not Group-14/13; the WSD-04 guard must not perturb it.
        name = name_compound("C1CCC[PH]1").strip().lower()
        assert "silolane" not in name  # whatever it is, it's not mis-routed by WSD-04
