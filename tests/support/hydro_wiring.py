"""Is ``namer.py`` calling ``rules.ring_assemblies.detect_hydro_ring_assembly`` (leads R2A)?

The producer of the hydro-modified ring assemblies, is reached from the namer by one
call in ``Orthonym._classify`` (``proposals/R2A-namer-hydro-assembly.patch``). A test of the engine on a
molecule the producer names is an expected failure until that call is in the tree: ``needs_wiring`` says so
(strict, so it is a failure again once the call is there and the test does not pass).

The same patch runs the isotope decorator inside ``ring_assemblies.hydro_assembly_withheld`` (an isotope label
cannot be spliced into a name of this producer). With the first call in the tree and that one missing, an
isotope-labelled molecule of the class is an abstention: ``needs_label_wiring`` marks the tests that say it is
not as expected failures for exactly that state (strict)."""
import inspect

import pytest

import orthonym.namer as _namer_module

_SOURCE = inspect.getsource(_namer_module)
WIRED = "detect_hydro_ring_assembly" in _SOURCE
LABEL_WIRED = "hydro_assembly_withheld" in _SOURCE

needs_wiring = pytest.mark.xfail(
    not WIRED, strict=True,
    reason="needs proposals/R2A-namer-hydro-assembly.patch: the call of detect_hydro_ring_assembly in "
           "namer.py (leads R2A)")

needs_label_wiring = pytest.mark.xfail(
    WIRED and not LABEL_WIRED, strict=True,
    reason="needs proposals/R2A-namer-hydro-assembly.patch: the isotope decorator inside "
           "hydro_assembly_withheld in namer.py (leads R2A); without it a labelled hydro assembly is an "
           "abstention")
