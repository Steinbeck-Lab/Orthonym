"""v29 Phase 3B (Job 2) — a refusal sentinel is never consumed as a name component.

    CCS[Zn]SCC  ->  'zinc compound (not supported)ylethane'

A refusal STRING was taken for a substituent name and given a ``-yl`` ending. Any
caller checking "did I get a non-empty string?" reads that as success, which
defeats the fail-closed contract outright.

The root cause is one function: ``fragment_naming.name_fragment_recursively`` is
THE chokepoint where a whole-molecule naming becomes a name COMPONENT, and
``namer.name_compound`` is always-emit — when it cannot name the input it returns
a sentinel STRING, not ``None``. All ~40 call sites consume the result as a
component, so the test belongs there once rather than at each of them.

Closing it exposed a trap that these tests exist to hold shut: THREE carbon-count
fallbacks name a fragment by counting its carbons and discarding everything else,
so the refusal turned into ``'ethylethane'`` and then ``'ethane'`` — names for a
DIFFERENT molecule and, unlike the sentinel-bearing string they replaced, ones no
failure predicate can flag. A visible failure must not become a silent one.

Every namer assertion runs with the OPSIN validity gate explicitly DISABLED —
the no-JRE mode where the gate fails OPEN and this was leaking.
"""

import pytest

from orthonym.errors import is_failure_name, is_refusal_sentinel
from orthonym.namer import Orthonym


@pytest.fixture
def ungated_namer(monkeypatch):
    """A namer with the SELF-01 OPSIN validity gate explicitly DISABLED.

    The gate fails OPEN when no JRE is present, so every safety property here is
    asserted in the mode where nothing downstream can suppress a wrong producer
    output.
    """
    import orthonym.namer as _namer
    monkeypatch.setattr(_namer, "_DISABLE_VALIDITY_GATE", True, raising=False)
    return Orthonym()


_SENTINEL_SUBSTRINGS = ("not supported", "unknown", "substituent")


@pytest.mark.parametrize("name", [
    "",
    None,
    "unknown organic compound",
    "zinc compound (not supported)",
    "mercury compound (not supported)",
    "inorganic compound (not supported)",
    "compound with wildcard atoms (not supported)",
    "substituent",
    "  Substituent  ",
])
def test_is_refusal_sentinel_covers_every_family(name):
    assert is_refusal_sentinel(name) is True


@pytest.mark.parametrize("name", [
    "ethane", "tert-butylbenzene", "acetic acid", "N-tert-butoxymethanamine",
    "di-tert-butylzinc", "tetraethylplumbane", "sodium acetate",
])
def test_is_refusal_sentinel_does_not_fire_on_a_real_name(name):
    assert is_refusal_sentinel(name) is False


def test_is_refusal_sentinel_reuses_is_failure_name():
    """The predicate is built ON is_failure_name, not beside it — that function
    already recognises three of the four families exactly, and duplicating them
    is how this class of bug reached six copies. The one thing it must NOT
    recognise is the bare cascade placeholder: a caller asking "did naming fail?"
    of a FINISHED name must not be told yes merely because the word appears.
    """
    for name in ["", "unknown organic compound", "zinc compound (not supported)"]:
        assert is_failure_name(name) is True
        assert is_refusal_sentinel(name) is True
    assert is_failure_name("substituent") is False       # correctly NOT a failure
    assert is_refusal_sentinel("substituent") is True    # but IS unusable in a slot


@pytest.mark.parametrize("smiles", [
    "CCS[Zn]SCC",     # the reported case: 'zinc compound (not supported)ylethane'
    "CCS[Hg]SCC",
    "CCS[Cd]SCC",
    "CCS[Zn]SC",
])
def test_a_refusal_sentinel_is_never_welded_into_a_name(ungated_namer, smiles):
    """The reported defect: a refusal string given a '-yl' ending and consumed as
    a substituent prefix. Every "did I get a non-empty string?" caller read that
    as success.
    """
    out = ungated_namer.name(smiles)
    assert is_refusal_sentinel(out), f"{smiles} must refuse, got {out!r}"
    for token in _SENTINEL_SUBSTRINGS:
        assert not out.lower().endswith(f"{token}ylethane")
    assert "(not supported)yl" not in out


def test_the_refusal_is_not_replaced_by_a_structure_dropping_name(ungated_namer):
    """The trap this fix had to avoid.

    Refusing the sentinel at the recursion chokepoint exposed THREE carbon-count
    fallbacks that name a fragment by counting its carbons and discarding
    everything else, so CCS[Zn]SCC went from the visibly-broken
    'zinc compound (not supported)ylethane' to 'ethane' — a name for a DIFFERENT
    molecule that no failure predicate can flag, i.e. strictly worse. The
    refusal must survive all the way out.
    """
    out = ungated_namer.name("CCS[Zn]SCC")
    assert out == "zinc compound (not supported)"
    assert out not in ("ethane", "ethylethane", "diethyl sulfide")


def test_carbon_count_fallback_refuses_a_heteroatom_bearing_fragment():
    """The guard on those fallbacks: get_alkyl_name(n) can only honestly spell an
    unbranched saturated acyclic chain attached at a terminus, so it must not be
    reached for anything else."""
    from rdkit import Chem
    from orthonym.assembly.substituent_naming import (
        fragment_is_linear_terminal_alkyl as ok,
    )
    mol = Chem.MolFromSmiles("CCS[Zn]SCC")
    frag = [0, 1, 2, 3]            # CC-S-Zn : carries S and Zn
    assert ok(mol, frag, 0) is False
    mol2 = Chem.MolFromSmiles("CCCC")
    assert ok(mol2, [0, 1, 2, 3], 0) is True      # linear, attached at a terminus
    assert ok(mol2, [0, 1, 2, 3], 1) is False     # attached mid-chain
    mol3 = Chem.MolFromSmiles("C1CCCCC1C")
    assert ok(mol3, [0, 1, 2, 3, 4, 5], 0) is False   # a RING is not 'hexyl'
    mol4 = Chem.MolFromSmiles("C=CCC")
    assert ok(mol4, [0, 1, 2, 3], 0) is False         # unsaturated is not 'butyl'


@pytest.mark.parametrize("smiles", [
    # broad invariant corpus: refusal-prone inputs mixed with ordinary ones
    "CCS[Zn]SCC", "CCS[Hg]SCC", "CCO[Zn]OCC", "CCS[Cd]SCC", "CC[Ti](C)(C)C",
    "CCO", "CC(=O)O", "CCCCC", "c1ccccc1", "c1ccccc1CC", "CCOCC",
    "CC(=O)OCC", "CC(C)(C)c1ccccc1", "CC(C)(C)[Zn]C(C)(C)C", "C[Zn]C",
    "CC[Pb](CC)(CC)CC", "CC(C)(C)ONC", "CNOC", "[Na+].CC(=O)[O-]",
    "CC(C)(C)[Si](C)(C)OCC1CO1", "CC(C)(C)B(O)O", "CC(C)(C)P(=O)(O)O",
])
def test_no_emitted_name_contains_a_refusal_sentinel(ungated_namer, smiles):
    """THE broad invariant: a sentinel may be the WHOLE answer (an honest refusal)
    but must never appear as a SUBSTRING of a composed name."""
    out = ungated_namer.name(smiles)
    assert out, smiles
    if is_refusal_sentinel(out):
        return                     # an honest, whole-string refusal is fine
    low = out.lower()
    for token in _SENTINEL_SUBSTRINGS:
        assert token not in low, (
            f"{smiles} -> {out!r} embeds the refusal sentinel {token!r}"
        )
