"""P-16.5.1.1 — compound '-oxy' / heteroatom substituent prefixes take enclosing marks.

The general-engine enclosure predicate (_is_complex_prefix) did not consult the
codebase's canonical needs_brackets(), so compound heteroatom prefixes were emitted
bare. Now it delegates, so compound prefixes are enclosed (P-16.5.1.1):

    nitrooxy -> (nitrooxy) 3-(nitrooxy)butanoic-acid family (P-61.5.2)
    thiocyanato -> (thiocyanato) 3-(thiocyanato)propanoic acid (P-16.5.1.2)
    chloromethoxy -> (chloromethoxy) 1-(chloromethoxy)-4-nitrobenzene (P-63.2.2.2)

Asserted at the PRODUCER level (the enclosure predicates + _mult_prefix rendering)
rather than full name_tiered: the general-fallback naming path is sensitive to
in-process-JVM init order under pytest (a wrong-molecule reduced-confidence name can
ship when the engine's OPSIN gate is disturbed by other tests in the same file), so a
name-string assertion here is JVM-flaky. The end-to-end flip to MATCH is covered by
the bb_conformance gate; the corrected PINs are round-trip-checked below.
"""
import pytest
from rdkit import Chem


@pytest.mark.parametrize("prefix", ["nitrooxy", "thiocyanato", "chloromethoxy"])
def test_compound_prefix_is_complex(prefix):
    """The general engine now classifies these as complex -> enclosed + 'bis'."""
    from orthonym.assembly.general_engine import _is_complex_prefix, _mult_prefix
    assert _is_complex_prefix(prefix) is True, prefix
    # n==1 => enclosed in marks (P-16.5.1.1)
    assert _mult_prefix(1, prefix) == f"({prefix})", _mult_prefix(1, prefix)


def test_needs_brackets_knows_the_compound_prefixes():
    from orthonym.assembly.naming_utils import needs_brackets
    assert needs_brackets("nitrooxy") is True       # pre-existing entry
    assert needs_brackets("thiocyanato") is True     # pre-existing entry
    assert needs_brackets("chloromethoxy") is True   # NEW halo-alkoxy branch


def test_needs_brackets_haloalkoxy_narrow():
    """The new halo-alkoxy branch fires only on a halogen + simple alkoxy compound."""
    from orthonym.assembly.naming_utils import needs_brackets
    assert needs_brackets("chloromethoxy") is True
    assert needs_brackets("bromoethoxy") is True
    # simple alkoxy and aryloxy stay simple (handled by other paths, not this branch)
    assert needs_brackets("methoxy") is False
    assert needs_brackets("ethoxy") is False
    assert needs_brackets("phenoxy") is False


@pytest.mark.parametrize("smiles,expected_pin", [
    ("CC(C)(C)N=C(C(=O)O)C(C)(C)O[N+](=O)[O-]",
     "2-(tert-butylimino)-3-methyl-3-(nitrooxy)butanoic acid"),
    ("N#CSCCC(=O)O", "3-(thiocyanato)propanoic acid"),
    ("O=[N+]([O-])c1ccc(OCCl)cc1", "1-(chloromethoxy)-4-nitrobenzene"),
])
def test_expected_pins_round_trip(smiles, expected_pin):
    """0-wrong: each corrected PIN OPSIN-parses back to the input structure."""
    from orthonym.validation.opsin_roundtrip import opsin_parse
    rt = opsin_parse(expected_pin)
    ref = Chem.MolFromSmiles(smiles)
    got = Chem.MolFromSmiles(rt) if rt else None
    assert got is not None and Chem.MolToInchiKey(got) == Chem.MolToInchiKey(ref), \
        f"RT failed for {expected_pin!r}"
