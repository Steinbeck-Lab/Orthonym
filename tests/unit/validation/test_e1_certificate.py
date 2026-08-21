# tests/unit/validation/test_e1_certificate.py
"""v25 G1: E1 certificate — every heavy atom binds exactly once."""
import pytest
from rdkit import Chem

from orthonym.assembly.general_engine import GeneralEngineResult, TokenBinding
from orthonym.validation.e1_certificate import E1Verdict, verify_certificate

pytestmark = pytest.mark.unit

ETHANOL = Chem.MolFromSmiles("CCO")  # atoms: 0 C, 1 C, 2 O


def _res(bindings, name="ethan-1-ol"):
    return GeneralEngineResult(name=name, bindings=tuple(bindings))


def test_complete_partition_ok():
    v = verify_certificate(ETHANOL, _res([
        TokenBinding((0, 1), "eth", "parent"),
        TokenBinding((2,), "ol", "suffix"),
    ]))
    assert v.ok


def test_unbound_atom_fails():
    v = verify_certificate(ETHANOL, _res([
        TokenBinding((0, 1), "eth", "parent"),
    ]))
    assert not v.ok and "unbound" in v.reason


def test_double_bound_atom_fails():
    v = verify_certificate(ETHANOL, _res([
        TokenBinding((0, 1), "eth", "parent"),
        TokenBinding((1, 2), "ol", "suffix"),
    ]))
    assert not v.ok and "twice" in v.reason


def test_token_missing_from_name_fails():
    v = verify_certificate(ETHANOL, _res([
        TokenBinding((0, 1), "propan", "parent"),
        TokenBinding((2,), "ol", "suffix"),
    ]))
    assert not v.ok and "token" in v.reason


def test_charged_mol_fails():
    mol = Chem.MolFromSmiles("CC[O-]")
    v = verify_certificate(mol, _res([
        TokenBinding((0, 1), "eth", "parent"),
        TokenBinding((2,), "olate", "suffix"),
    ], name="ethanolate"))
    assert not v.ok and "charge" in v.reason


# --------------------------------------------------------------------------
# F-E1: per-token element soundness (sound-by-refusal).
# --------------------------------------------------------------------------
from orthonym.validation.e1_certificate import _token_is_confidently_all_carbon


def test_all_carbon_token_bound_to_heteroatom_is_rejected():
    """The synthetic gap: `ethyl`/`eth` tokens fabricated onto N and O atoms of
    CCN(CC)N=O used to certify as `1-ethylethane`. Now rejected."""
    mol = Chem.MolFromSmiles("CCN(CC)N=O")  # 0C 1C 2N 3C 4C 5N 6O
    v = verify_certificate(mol, _res([
        TokenBinding((2, 3, 4, 5, 6), "ethyl", "parent"),
        TokenBinding((0, 1), "eth", "parent"),
    ], name="1-ethylethane"), allow_charged=True)
    assert not v.ok and "ethyl" in v.reason


def test_real_general_result_with_heteroatom_token_still_certifies():
    """A composite token that legitimately declares heteroatoms (`...3-oxa-1,2-
    diaza...`) is NOT confidently all-carbon, so it is skipped -- the real
    general-engine result for CCN(CC)N=O still certifies."""
    mol = Chem.MolFromSmiles("CCN(CC)N=O")
    v = verify_certificate(mol, _res([
        TokenBinding((2, 3, 4, 5, 6), "1-ethyl-3-oxa-1,2-diazaprop-2-en-1-yl", "parent"),
        TokenBinding((0, 1), "eth", "parent"),
    ], name="1-(1-ethyl-3-oxa-1,2-diazaprop-2-en-1-yl)ethane"), allow_charged=True)
    assert v.ok


def test_heteroatom_suffix_token_on_heteroatom_not_rejected():
    """`ol` on the O of ethanol is a heteroatom-declaring token -> skipped, not
    rejected (the existing complete-partition test also covers this)."""
    v = verify_certificate(ETHANOL, _res([
        TokenBinding((0, 1), "eth", "parent"),
        TokenBinding((2,), "ol", "suffix"),
    ]))
    assert v.ok


@pytest.mark.parametrize("token,expected", [
    ("ethyl", True), ("methyl", True), ("propyl", True), ("ethane", True),
    ("cyclohexyl", True), ("phenyl", True), ("naphthyl", True), ("benzyl", True),
    # heteroatom / functional tokens must be skipped (never confidently all-carbon)
    ("methoxy", False), ("ethanol", False), ("methanamine", False),
    ("oxan-2-yl", False), ("pyridin-3-yl", False), ("furyl", False),
    ("thienyl", False), ("acetyl", False), ("formyl", False), ("chloro", False),
    ("hydroxy", False), ("sulfanyl", False), ("carbamoyl", False), ("ol", False),
])
def test_confidently_all_carbon_classifier(token, expected):
    assert _token_is_confidently_all_carbon(token) is expected


# ---------------------------------------------------------------------------
# Phase E (no-abstain universal namer): the shared shape-agnostic core
# ``_verify_partition`` -- ``verify_certificate`` is now a thin wrapper over it,
# and the universal recursive namer certifies its output through the SAME core
# (invariant 12: extend, don't duplicate). These tests exercise the core on
# plain ``(token, atom_ids)`` tuples (the ``UniversalResult.bindings`` shape),
# the branch-restricted ``atoms=`` reference set, and ``allow_charged``.
# ---------------------------------------------------------------------------
from orthonym.validation.e1_certificate import _verify_partition


def test_verify_partition_complete_ok():
    v = _verify_partition(ETHANOL, "ethan-1-ol",
                          [("ethan", frozenset({0, 1})), ("ol", frozenset({2}))])
    assert v.ok


def test_verify_partition_accepts_a_oneshot_generator():
    # the wrapper feeds a generator; the core must materialise it (iterated 3x)
    gen = (p for p in [("ethan", frozenset({0, 1})), ("ol", frozenset({2}))])
    v = _verify_partition(ETHANOL, "ethan-1-ol", gen)
    assert v.ok


def test_verify_partition_atom_drop_fails():
    v = _verify_partition(ETHANOL, "ethan-1-ol", [("ethan", frozenset({0, 1}))])
    assert not v.ok and "unbound" in v.reason


def test_verify_partition_double_count_fails():
    v = _verify_partition(ETHANOL, "x",
                          [("a", frozenset({0, 1})), ("b", frozenset({1, 2}))])
    assert not v.ok and "twice" in v.reason


def test_verify_partition_phantom_atom_fails():
    v = _verify_partition(ETHANOL, "x",
                          [("a", frozenset({0, 1})), ("b", frozenset({2, 99}))])
    assert not v.ok and ("non-heavy" in v.reason or "missing" in v.reason)


def test_verify_partition_token_in_name_fails():
    v = _verify_partition(ETHANOL, "ethan-1-ol",
                          [("propan", frozenset({0, 1})), ("ol", frozenset({2}))])
    assert not v.ok and "not in name" in v.reason


def test_verify_partition_token_in_name_tolerates_terminal_e_elision():
    # P-16.7.1(a)/P-74.1.1: a parent-hydride token's terminal 'e' is elided
    # before a vowel-initial ionic suffix, so the FULL token ('ethane') is
    # absent but its stem ('ethan') is present -> accepted on token-in-name.
    v = _verify_partition(ETHANOL, "ethan-1-ol",
                          [("ethane", frozenset({0, 1})), ("ol", frozenset({2}))])
    assert v.ok
    # but the elision tolerance strips ONLY a single trailing 'e' -- a genuinely
    # absent token (no shared stem) still fails.
    v2 = _verify_partition(ETHANOL, "ethan-1-ol",
                           [("propane", frozenset({0, 1})), ("ol", frozenset({2}))])
    assert not v2.ok and "not in name" in v2.reason


def test_verify_partition_f_e1_all_carbon_on_hetero_fails():
    v = _verify_partition(ETHANOL, "ethyl", [("ethyl", frozenset({0, 1, 2}))])
    assert not v.ok and "all-carbon token" in v.reason


def test_verify_partition_allow_charged_lifts_g1():
    mol = Chem.MolFromSmiles("CC[O-]")  # net -1
    pairs = [("ethan", frozenset({0, 1})), ("olate", frozenset({2}))]
    assert not _verify_partition(mol, "ethan-1-olate", pairs).ok            # G1 fires
    assert _verify_partition(mol, "ethan-1-olate", pairs, allow_charged=True).ok


def test_verify_partition_branch_scope_covers_frag():
    # atoms=frag restricts the partition to a branch subgraph; the two C's
    # are the whole 'branch', O (idx 2) is outside it and must NOT be bound.
    v = _verify_partition(ETHANOL, "ethyl", [("ethyl", frozenset({0, 1}))],
                          atoms=frozenset({0, 1}))
    assert v.ok


def test_verify_partition_branch_scope_rejects_out_of_frag_atom():
    # a branch binding that references an atom outside frag is a phantom
    v = _verify_partition(ETHANOL, "ethyl", [("ethyl", frozenset({0, 1, 2}))],
                          atoms=frozenset({0, 1}))
    assert not v.ok and ("non-heavy" in v.reason or "missing" in v.reason)


def test_wrapper_delegates_identically_to_core():
    # verify_certificate(mol, result) must equal _verify_partition over the
    # unpacked (token, atom_ids) tuples -- byte-identical logic for the 2
    # existing GeneralEngineResult callers.
    bindings = [TokenBinding((0, 1), "eth", "parent"), TokenBinding((2,), "ol", "suffix")]
    res = _res(bindings)
    a = verify_certificate(ETHANOL, res)
    b = _verify_partition(ETHANOL, res.name,
                          ((tb.token, tb.atom_ids) for tb in bindings))
    assert (a.ok, a.reason) == (b.ok, b.reason)
