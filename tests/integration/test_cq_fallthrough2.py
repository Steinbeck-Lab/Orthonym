"""CQ5 Task 1 — complete the best-effort RT-failure fall-through (RISK 4).

Task A wired ``_try_besteffort_clean_general_fallthrough`` at the ship-a-failure
path (namer.py ~3388) and reset the propagation contextvars + session depth, but
MISSED the third piece of state a fresh top-level ``name`` gets: a fresh
whole-molecule fragment MEMO cache. ``isolated_naming_session`` deliberately keeps
that cache live (the giant-hang fix), so the primary pass's
``recursion_depth_fallback`` SKIP entries for the deep substituents it could not
name (poisoned by the elevated session-depth floor) were still in the cache the
clean fall-through reused -- and the good name was never produced. The a review
witness ``COP(=O)(C=C(F)F)C=C(F)F`` reached namer.py:3388, the fall-through fired,
and it STILL returned None, so the molecule abstained even though a fresh top-level
call names it and OPSIN-round-trips it.

Fix (trace-confirmed, a project rule — `internal notes`):
``isolated_naming_session(reset_cache=True)`` installs a fresh empty memo cache for
the isolated body and restores the original on exit; the clean fall-through opts in.
This is the SAME site Task A opened (no new decline site exists -- the recoverable
class routes entirely through 3388; the NamingLimit/Exception early-return sites were
trace-checked and drop 0 convertible molecules), completed. Offer-not-return,
best-effort-gated, RT-gated: 0-wrong holds via the recovery's own OPSIN round-trip
gate (a project rule), and PIN/complete output is byte-identical.
"""
import random
from unittest.mock import patch

import pytest
from rdkit import Chem

from orthonym.namer import Orthonym
from orthonym.errors import is_failure_name
from orthonym.validation.opsin_roundtrip import opsin_parse
import orthonym.assembly.fragment_naming as fn

# The RT/validity gate is the whole point (it voids the primary + RT-gates the
# fall-through), so it MUST be enabled (conftest disables it suite-wide).
pytestmark = [pytest.mark.integration, pytest.mark.roundtrip,
              pytest.mark.opsin_gate]


# (smiles, expected_besteffort_name_or_None). None => assert RT-match only.
# All four are REAL abstainers from internal notes that abstained
# at f6a07bd6 and convert only with the fresh-cache isolation. Each has a deep
# heteroatom/replacement-nomenclature substituent whose primary-pass naming poisoned
# the memo cache with a depth-limited SKIP.
WITNESSES = [
    # The a review witness.
    ("COP(=O)(C=C(F)F)C=C(F)F",
     "1-[1-(2,2-difluoroeth-1-en-1-yl)-1-oxo-2-oxa-1-phosphapropyl]-"
     "2,2-difluoroeth-1-ene"),
    # A stereocentre-carrying phosphapentyl (full-InChIKey RT, stereo retained).
    ("CCOC(OCC)P(=O)(C[C@@H](O)CCl)OCC", None),
    # A diaza-phosphapropyl (phosphoramide).
    ("NP(=O)(N(Cl)Cl)N(CCCl)CCCl", None),
    # A dioxa-thia-phospha chain onto a benzene parent.
    ("CCP(=O)(COS(=O)(=O)c1c(C)cccc1SC(F)(F)F)OC(C)C", None),
]

# PIN gold controls (byte-identical current PIN output; must not shift).
PIN_GOLD = [
    ("CCCCCCCc1ccccc1", "heptylbenzene"),
    ("CCCCCCCCC1CCCCC1", "octylcyclohexane"),
    ("ClCCCCC", "1-chloropentane"),
    ("O=C1CCNCC1", "piperidin-4-one"),
    ("Cc1ccc(O)nc1", "5-methylpyridin-2-ol"),
]


def _besteffort():
    return Orthonym(general_fallback=True, general_fallback_unverified=True,
                     allow_aromatic_general=True)


def _inchikey(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchiKey(mol) if mol is not None else None


def _rt_inchikey_match(name, smiles):
    assert not is_failure_name(name), f"expected a real name, got {name!r}"
    osmi = opsin_parse(name)
    assert osmi is not None, f"OPSIN could not parse {name!r}"
    return _inchikey(osmi) == _inchikey(smiles)


@pytest.mark.parametrize("smiles,expected", WITNESSES)
def test_witness_converts_and_roundtrips(smiles, expected):
    """Each witness converts at best-effort AND OPSIN-round-trips to its input."""
    out = _besteffort().name(smiles)
    assert not is_failure_name(out), (
        f"best-effort abstained on {smiles!r} (got {out!r}); the fresh-cache "
        f"clean fall-through candidate was never adopted")
    if expected is not None:
        assert out == expected
    assert _rt_inchikey_match(out, smiles), (
        f"shipped name {out!r} does not OPSIN-round-trip to {smiles!r}")


def test_pin_tier_unchanged_on_witnesses():
    """The best-effort fall-through never leaks into the PIN tier: at the PIN
    tier (gfu=False) a witness either abstains or is named by the PIN path
    itself -- never the general engine -- and that name round-trips to its
    input on the full InChIKey.

    Suite fix j4 (TRIAGE g3 C10e): the premise "every witness abstains at the
    PIN tier" went stale when the pnictogen -inate ester handler,
    cf. the Blue Book 'methyl dimethylphosphinate (PIN)') began naming
    witnesses 1 and 2 as functional-class esters -- a real PIN-path gain, not
    a leak. Witness 2 then also carried a second, front-of-name '(2R)-'
    (OPSIN-unparseable), see test_pnictogen_ester_cites_each_descriptor_once.
    """
    pin = Orthonym()
    for smiles, _ in WITNESSES:
        res = pin.name_tiered(smiles)
        out = res["name"]
        if is_failure_name(out):
            continue
        assert res.get("source") == "pin_path", (
            f"PIN tier named {smiles!r} as {out!r} from {res.get('source')!r}: "
            f"the best-effort fall-through leaked into the PIN tier")
        assert _rt_inchikey_match(out, smiles), (
            f"PIN tier name {out!r} does not OPSIN-round-trip to {smiles!r}")


# Suite fix j4 (TRIAGE g3 C10e). A functional-class -inate ester has no numbered
# parent (the phosphinate is a mononuclear retained parent whose substituents take
# no locants), so every stereodescriptor stays with the organyl group it belongs
# to: (the Blue Book) descriptors "are placed at the front of the
# complete name when related to the parent structure [...] When they relate to
# substituent groups, they are cited at the front of the corresponding prefix",
# and (:2869) scopes locants per enclosing-mark unit. The handler used to prepend a second '(2R)-'
# read from a P-substituent's chain. All three OPSIN 2.9.0 full-InChIKey and
# canonical-SMILES exact (independent batch); the doubled form does not parse.
PNICTOGEN_ESTER_STEREO = [
    ("CCOC(OCC)P(=O)(C[C@@H](O)CCl)OCC",
     "ethyl [(2R)-3-chloro-2-hydroxypropyl](diethoxymethyl)phosphinate"),
    ("CCOP(=O)(CC)C[C@@H](C)O",
     "ethyl ethyl[(2R)-2-hydroxypropyl]phosphinate"),
    # boundary: a stereocentre in the ESTER organyl group keeps its descriptor
    ("CC[C@@H](C)OP(=O)(CC)CC", "(2R)-butan-2-yl diethylphosphinate"),
]


@pytest.mark.parametrize("smiles,expected", PNICTOGEN_ESTER_STEREO)
def test_pnictogen_ester_cites_each_descriptor_once(smiles, expected):
    res = Orthonym().name_tiered(smiles)
    assert res["name"] == expected
    assert res["tier"] == "pin_verified" and res["opsin"] == "verified", res


@pytest.mark.xfail(strict=True, reason=(
    "PIN spelling of the ethenyl prefix: P-14.3.4.4 example BlueBookV2.md:3003 "
    "'2-chloroethen-1-yl (preferred prefix)' ('eth-1-en-1-yl': 0 BB hits; "
    "'ethen-1-yl' :3003/:6415/:17366/:42468). The substituent namer cites the "
    "'en' locant; the name is RT-exact but not the PIN spelling. TODO: "
    ".planning/preexisting-triage/TRIAGE.md, 'Suite fix -- j4-pin-labels-a', "
    "new findings."))
def test_witness1_pin_spelling():
    assert Orthonym().name(WITNESSES[0][0]) == (
        "methyl bis(2,2-difluoroethen-1-yl)phosphinate")


@pytest.mark.parametrize("smiles,expected", PIN_GOLD)
def test_pin_gold_byte_identical(smiles, expected):
    assert Orthonym().name(smiles) == expected


def test_negative_rt_mismatch_stays_abstained():
    """0-wrong (a project rule): if the RT gate reports a mismatch, the pipeline
    abstains rather than shipping the fall-through candidate. Forcing
    ``_rt_match`` to False on a witness that otherwise converts proves the
    fresh-cache fall-through never ships an RT-failing name (immune to going
    stale, the same mechanism-level guard Task A's file uses)."""
    be = _besteffort()
    smi = WITNESSES[0][0]
    assert not is_failure_name(be.name(smi)), (
        "sanity: this witness must normally convert (so the mock below is the "
        "only thing causing the abstain)")
    with patch.object(Orthonym, "_rt_match", staticmethod(lambda *a, **kw: False)):
        out = be.name(smi)
    assert is_failure_name(out), (
        f"shipped {out!r} for {smi!r} even though _rt_match reported no match "
        f"-- invariant 9 violated")


def test_determinism_two_smiles_orders():
    """Two atom orderings of the COP witness yield the identical name."""
    smi = WITNESSES[0][0]
    mol = Chem.MolFromSmiles(smi)
    n = mol.GetNumAtoms()
    order_a, order_b = list(range(n)), list(range(n))
    random.Random(1).shuffle(order_a)
    random.Random(2).shuffle(order_b)
    smi_a = Chem.MolToSmiles(Chem.RenumberAtoms(mol, order_a), canonical=False)
    smi_b = Chem.MolToSmiles(Chem.RenumberAtoms(mol, order_b), canonical=False)
    assert smi_a != smi_b
    be = _besteffort()
    name_a, name_b = be.name(smi_a), be.name(smi_b)
    assert not is_failure_name(name_a)
    assert name_a == name_b, f"order-dependent name: {name_a!r} != {name_b!r}"


def test_reset_cache_isolates_and_restores():
    """Unit-level proof of the mechanism: ``isolated_naming_session(reset_cache=
    True)`` installs a FRESH empty memo cache inside the body (so a poisoned SKIP
    entry from the enclosing pass is not read back) and RESTORES the original
    cache on exit (so the enclosing molecule's memo survives). The default
    (reset_cache=False) keeps the live cache -- the T4-producer / giant-hang
    behaviour -- unchanged."""
    guard = fn._fragment_guard
    saved = getattr(guard, 'cache', None)
    try:
        poisoned = {"POISON_SMILES": None}
        guard.cache = poisoned
        # reset_cache=True: fresh empty cache inside, original restored after.
        with fn.isolated_naming_session(reset_cache=True):
            assert guard.cache == {}, "reset_cache did not install a fresh cache"
            assert guard.cache is not poisoned
            guard.cache["INNER"] = "x"          # inner writes are discarded
        assert guard.cache is poisoned, "reset_cache did not restore the original"
        assert "INNER" not in guard.cache
        # default (reset_cache=False): the live cache is kept (giant-hang fix).
        with fn.isolated_naming_session():
            assert guard.cache is poisoned, "default must keep the live cache"
    finally:
        guard.cache = saved


def test_no_hang_on_large_abstainer_bounded():
    """The fresh-cache re-explore stays bounded by the name-scope work budget:
    a large molecule that abstains does so with a real abstain sentinel (0-wrong)
    rather than hanging or shipping a wrong name."""
    smi = ("C[C@@H]1[C@H]2CC[C@@H](C2)[C@@H]1C(=O)N[C@@H]1[C@H]2C[C@@H]3"
           "[C@H]1[C@@H]3[C@@H]2C(=O)N[C@H]1CC12CCCC2")
    out = _besteffort().name(smi)
    # It abstains today; the point is it TERMINATES with a clean abstain, never a
    # wrong molecule. If a future engine names it, the RT gate still guarantees
    # 0-wrong, so accept either a clean abstain or an RT-verifying name.
    assert is_failure_name(out) or _rt_inchikey_match(out, smi)
