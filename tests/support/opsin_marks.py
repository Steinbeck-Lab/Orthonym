"""Marks for tests whose expectation depends on what the active OPSIN reads (jars.active_opsin, opsin_capabilities)."""
import pytest

from orthonym import jars
from orthonym import opsin_capabilities as caps


def release_opsin_in_use() -> bool:
    """True when the OPSIN in use is the pinned 2.9.0 release jar (not a build of OPSIN's main branch, not a jar
    given by ORTHONYM_OPSIN_JAR): the OPSIN the rules of ``rules.isotope_search`` were drawn from."""
    try:
        active = jars.active_opsin()
    except jars.JarUnavailable:
        return False
    return active["label"] == "2.9.0" and active["path"] is not None


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
#: a test of what OPSIN 2.9.0 refuses (the pruning of the isotope search is drawn from it and is off for any other OPSIN)
release_refusals_only = pytest.mark.skipif(
    not release_opsin_in_use(), reason="describes the refusals of OPSIN 2.9.0; the search prunes nothing for another OPSIN")


def release_capabilities(monkeypatch) -> None:
    """The engine as it is with OPSIN 2.9.0, whichever OPSIN is in use: no phane name is built (the producers read
    ``opsin_capabilities["phane"]``; the ``halogen_uide`` entry is set too, so the capability set is the release's,
    though no producer reads it). For a test of a naming path that a phane name outranks when the OPSIN in use reads
    phane names (a von Baeyer budget, an amido prefix): the path stays under test with OPSIN main, which reads every
    other name as 2.9.0 does."""
    monkeypatch.setattr(caps, "opsin_capabilities", lambda: {"phane": False, "halogen_uide": False})
