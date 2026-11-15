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
#: skipped unless the active OPSIN reads ring assembly names with composite locants (ORTHONYM_OPSIN=main)
needs_composite_opsin = pytest.mark.skipif(
    not caps.composite_assembly_enabled(),
    reason="needs an OPSIN that reads composite locants (1^1,2^1:2^4,3^1-terphenyl): ORTHONYM_OPSIN=main")
#: skipped unless the active OPSIN reads an aryl-aryl axis descriptor '(M)-' (ORTHONYM_OPSIN=main)
needs_axial_opsin = pytest.mark.skipif(
    not caps.opsin_capabilities().get("axial", False),
    reason="needs an OPSIN that reads axial descriptors ('(M)-[1,1'-binaphthalene]-2,2'-diol'): ORTHONYM_OPSIN=main")
#: a test of OPSIN 2.9.0, which reads no axial descriptor: skipped when the active OPSIN reads one
release_axial_only = pytest.mark.skipif(
    caps.opsin_capabilities().get("axial", False), reason="describes OPSIN 2.9.0, which reads no axial descriptor")
#: skipped unless the active OPSIN reads an '-aminiumyl' prefix ('(N,N-dimethylmethanaminiumyl)acetate')
needs_aminiumyl_opsin = pytest.mark.skipif(
    not caps.aminiumyl_enabled(), reason="needs an OPSIN that reads '-aminiumyl' prefixes: ORTHONYM_OPSIN=main")
#: a test of OPSIN 2.9.0, which reads no '-aminiumyl' prefix: skipped when the active OPSIN reads one
release_aminiumyl_only = pytest.mark.skipif(
    caps.aminiumyl_enabled(), reason="describes OPSIN 2.9.0, which reads no '-aminiumyl' prefix")


#: skipped unless the active OPSIN reads the name of a polyol ester of different acids
#: ('propane-1,2,3-triyl 1,2-diacetate 3-propanoate')
needs_polyester_opsin = pytest.mark.skipif(
    not caps.polyester_enabled(), reason="needs an OPSIN that reads multi-anion polyol ester names: ORTHONYM_OPSIN=main")
#: a test of OPSIN 2.9.0, which reads no multi-anion polyol ester name: skipped when the active OPSIN reads one
release_polyester_only = pytest.mark.skipif(
    caps.polyester_enabled(), reason="describes OPSIN 2.9.0, which reads no multi-anion polyol ester name")


def by_aminiumyl(release: str, reads_aminiumyl: str) -> str:
    """The expected wider-tier name: ``reads_aminiumyl`` where the active OPSIN reads an '-aminiumyl' prefix (the
    wider tiers then spell the PIN prefix), else ``release`` (OPSIN 2.9.0)."""
    return reads_aminiumyl if caps.aminiumyl_enabled() else release
#: a test of the twin route of a halogen -uide name: skipped when the active OPSIN reads such a name
release_uide_only = pytest.mark.skipif(
    caps.opsin_capabilities()["halogen_uide"],
    reason="describes OPSIN 2.9.0, which reads no substituted halogen -uide name")
#: a test of what OPSIN 2.9.0 refuses (the pruning of the isotope search is drawn from it and is off for any other OPSIN)
release_refusals_only = pytest.mark.skipif(
    not release_opsin_in_use(), reason="describes the refusals of OPSIN 2.9.0; the search prunes nothing for another OPSIN")


def release_capabilities(monkeypatch) -> None:
    """The engine as it is with OPSIN 2.9.0, whichever OPSIN is in use: no phane name is built (the producers read
    ``opsin_capabilities["phane"]``; no ring assembly name has composite locants (``composite_assembly``); the
    ``halogen_uide`` entry is set too, so the capability set is the release's, though no producer reads it). For a
    test of a naming path that a phane name outranks when the OPSIN in use reads phane names (a von Baeyer budget, an
    amido prefix), or of the primed names of a ring assembly of three or more rings: the path stays under test with
    OPSIN main, which reads every other name as 2.9.0 does."""
    monkeypatch.setattr(caps, "opsin_capabilities",
                        lambda: {"phane": False, "halogen_uide": False, "composite_assembly": False})


def composite_or_primed(primed: str, composite: str) -> str:
    """The name a test expects for an unbranched assembly of three or more identical rings.
    (the Blue Book): "The preferred numbering for ring assemblies composed of three or more identical cyclic
    systems uses composite locants rather than primed locants". With an OPSIN that reads composite locants the engine
    ships the composite PIN of ``rules.composite_assembly``; with OPSIN 2.9.0 it ships the primed name of general
    nomenclature that ``rules.ring_assemblies`` spells, labelled below the PIN."""
    return composite if caps.composite_assembly_enabled() else primed


def by_polyester(release: str, reads_polyester: str) -> str:
    """The expected wider-tier name: ``reads_polyester`` where the active OPSIN reads a multi-anion polyol ester name,
    else ``release`` (OPSIN 2.9.0)."""
    return reads_polyester if caps.polyester_enabled() else release
