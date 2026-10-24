"""Marks for tests whose expectation depends on what the active OPSIN reads (jars.active_opsin, opsin_capabilities)."""
import pytest

from orthonym import opsin_capabilities as caps

#: skipped unless the active OPSIN reads phane names (ORTHONYM_OPSIN=main)
needs_phane_opsin = pytest.mark.skipif(
    not caps.phane_enabled(), reason="needs an OPSIN that reads phane names: ORTHONYM_OPSIN=main")
#: a test of OPSIN 2.9.0 behaviour (no phane grammar): skipped when the active OPSIN reads phane names
release_opsin_only = pytest.mark.skipif(
    caps.phane_enabled(), reason="describes OPSIN 2.9.0, which reads no phane name")
#: a test of the twin route of a halogen -uide name: skipped when the active OPSIN reads such a name
release_uide_only = pytest.mark.skipif(
    caps.opsin_capabilities()["halogen_uide"],
    reason="describes OPSIN 2.9.0, which reads no substituted halogen -uide name")
