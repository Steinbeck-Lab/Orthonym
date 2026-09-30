"""The 11 single-blocked rows that reach the substituent cascade's last resort.

`assembly/substituent_enumerator.py`'s last-resort fallback returns the bare word
'substituent', which `errors.py:227` names CASCADE_PLACEHOLDER and
`is_refusal_sentinel` treats as a REFUSAL, not a name. These 11 rows are the
whole `enumerator_last_resort` terminal class on the 500-row best-effort census
(seed 42, `benchmarks/pubchem_2000.csv`) -- measured at, matching the
census figure of 11 recorded in internal notes.

These tests are RED ON PURPOSE. They pin the oracle for the follow-up build and
deliberately contain no fix: the RING sibling of this gap turned out to already
exist and merely be unreachable (-T1), so the build must a trace the site
before assuming a namer is missing.

xfail(strict=True): when a fragment starts naming, the xfail becomes an XPASS and
the suite FAILS, forcing the expectation to be updated deliberately rather than
drifting silently.

Observed while enumerating these (recorded, not asserted): the class is not one
chemical family. It contains a dithiocarbamate-molybdenum complex, an
organoaluminium cation, three long-chain sulfonate/sulfate surfactants, two
quaternary-ammonium esters and an acylhydrazone -- so a single "acyclic
fragment namer" is unlikely to clear it, and the follow-up must split it.
"""
import pytest

from orthonym.errors import is_refusal_sentinel
from orthonym.namer import Orthonym
from tests.support.default_tier import (  # noqa: E402
    declined_pin_row,
    default_tier_rule_applies,
)

# Default tier: the paper, Methods, "Tiers" (L73): "The default configuration emits a
# name only when the pipeline can build the preferred IUPAC name (PIN); otherwise, it
# declines." User decision 2026-09-30 ("Ship it in 1.0.2"): a name the code records
# as not the PIN is declined at the default tier with NO_VERIFIED_PIN; for the
# molecules below the test asserts that decline, the strict path's name and label,
# and the same name at the best-effort tier (tests/support/default_tier.py).
DEFAULT_TIER_DECLINES = frozenset({
    "CCCCCCCCCCCCCC(=O)NCCOS(=O)(=O)[O-]",
    "CCCCCCCCCCOCCCN(CCS(=O)(=O)[O-])C(=O)C(=C)C",
})
#... whose best-effort name is another one (it reads back exactly)
BEST_EFFORT_NAMES_IT_OTHERWISE = frozenset({
    "CCCCCCCCCCCCCC(=O)NCCOS(=O)(=O)[O-]",
})


def _declined_pin_row(smiles):
    return declined_pin_row(
        smiles, best_effort_same=smiles not in BEST_EFFORT_NAMES_IT_OTHERWISE)


def _dt_name_compound(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)["name"]
    return name_compound(smiles)


def _dt_name(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)["name"]
    return Orthonym(style="pin").name(smiles)


def _dt_row(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)
    return Orthonym(style="pin").name_tiered(smiles)


# REQUIRED, and it is what makes the assertion below mean anything.
# `tests/conftest.py:302-317` disables the OPSIN validity gate suite-wide and
# documents the trap: with the gate OFF these fragments emit a plausible name for
# a DIFFERENT molecule instead of a sentinel. Measured here -- without this
# marker 7 of the 11 XPASS while ` substituent_skip` fires, i.e. the
# "success" is a silent atom drop; in a fresh gated process all 11 refuse.
#
# With the gate ON the oracle is sound in both directions: escaping the sentinel
# requires a name accepts, and suppresses on a verified
# constitutional mismatch (`namer.py:920-924`). So an XPASS here means the
# fragment named AND kept its atoms -- it cannot be won by dropping them.
pytestmark = pytest.mark.opsin_gate

FRAGMENTS = [
    # acylurea on a tetrahydroisoquinolinium
    "C[C@H]([C@H]1C2=CC(=C(C=C2CC[NH+]1CC3=CC(=CC=C3)F)OC)OC)NC(=O)NC(C)C",
    # hexakis(diethyldithiocarbamate) + Mo -- organometallic, territory
    "CCN(CC)C(=S)[S-].CCN(CC)C(=S)[S-].CCN(CC)C(=S)[S-].CCN(CC)C(=S)[S-]."
    "CCN(CC)C(=S)[S-].CCN(CC)C(=S)[S-].[Mo]",
    # hydroperoxide on a 2H-pyran-2-one
    "CC(C1=CC(=O)OC(O1)(C)C)OO",
    # acrylamido sulfonate
    "CC(C(C)S(=O)(=O)[O-])NC(=O)C=C",
    # quaternary ammonium diester
    "CCCCCCCCCCOC[N+](C)(C)CCOC(=O)CCCCCCCCC",
    # organoaluminium dication
    "CCC(C)CC[Al+2]",
    # acylhydrazone
    "CC1=CC=C(C=C1)NCC(=O)NN=CC2=C(C=C(C=C2)C)O",
    # quaternary dicarboxylate with a thioether arm
    "CCCCCCCCCCC(CCCCCCCCCC)(CSCCC(=O)[O-])C(=O)[O-]",
    # allyl dithiocarbamate ammonium
    "C=CCNC(=S)SCC[NH3+]",
    # N-acyl aminoethyl sulfate
    "CCCCCCCCCCCCCC(=O)NCCOS(=O)(=O)[O-]",
    # methacrylamido sulfonate with an ether tail
    "CCCCCCCCCCOCCCN(CCS(=O)(=O)[O-])C(=O)C(=C)C",
]


def test_fragment_list_is_the_measured_set():
    assert len(FRAGMENTS) == 11, (
        "re-measure: the census recorded 11 rows with "
        "terminal_detail == 'enumerator_last_resort'"
    )


def test_fragments_are_distinct():
    """A duplicated SMILES would make the count agree while covering fewer rows."""
    assert len(set(FRAGMENTS)) == 11


# Three rows of the set name atom-complete now (TRIAGE g6 C15, 'Suite fix --
# j5-pin-labels-b'); each is asserted on its spelling and its tier instead of on
# the sentinel. The PIN spellings are OPSIN 2.9.0 full-InChIKey EXACT.
# - N-acyl on an amine inside a sulfate/sulfonate ester: the amide is cited as
# the '-amido' prefix, (the Blue Book) "Method (1)
# generates preferred IUPAC names." (:32998), '2-(N-methylpropanamido)benzene-
# 1-sulfonic acid (PIN)'. No producer builds it on the anion-ester path yet, so
# the shipped N-acyl float stays below pin_verified (decision A part 1).
# - the compound alkyloxy prefix inside 'methyl' takes its own marks,
# (:7232), '(benzyloxy)carbonyl (preferred prefix)' (:18116).
_PIN_BUILT = {
    "CCCCCCCCCCOC[N+](C)(C)CCOC(=O)CCCCCCCCC":
        "2-(decanoyloxy)-N-[(decyloxy)methyl]-N,N-dimethylethan-1-aminium",
}
_PIN_NOT_BUILT = {
    "CCCCCCCCCCCCCC(=O)NCCOS(=O)(=O)[O-]": "2-tetradecanamidoethyl sulfate",
    "CCCCCCCCCCOCCCN(CCS(=O)(=O)[O-])C(=O)C(=C)C":
        "2-{N-[3-(decyloxy)propyl]-2-methylprop-2-enamido}ethane-1-sulfonate",
}
_STILL_LAST_RESORT = [s for s in FRAGMENTS
                      if s not in _PIN_BUILT and s not in _PIN_NOT_BUILT]


@pytest.mark.parametrize("smiles", _STILL_LAST_RESORT)
@pytest.mark.xfail(strict=True, reason="acyclic last-resort gap, v30 phase PB")
def test_acyclic_fragment_names_without_a_sentinel(smiles):
    name = Orthonym(general_fallback=True).name(smiles)
    assert not is_refusal_sentinel(name), name


@pytest.mark.parametrize("smiles,expected", sorted(_PIN_BUILT.items()))
def test_fragment_named_at_the_pin_tier(smiles, expected):
    from tests.support.rt_assert import name_is_rt_exact
    r = _dt_row(smiles)
    assert r["name"] == expected, r
    assert r["tier"] == "pin_verified", r
    assert name_is_rt_exact(expected, smiles), r


@pytest.mark.parametrize("smiles", sorted(_PIN_NOT_BUILT))
def test_fragment_ships_below_the_pin_tier(smiles):
    from tests.support.rt_assert import name_is_rt_exact
    r = _dt_row(smiles)
    assert not is_refusal_sentinel(r["name"]), r
    assert r["tier"] != "pin_verified", r
    assert name_is_rt_exact(r["name"], smiles), r


@pytest.mark.parametrize("smiles,expected", sorted(_PIN_NOT_BUILT.items()))
@pytest.mark.xfail(strict=True, reason=(
    "PIN cites the N-acyl amine as an '-amido' prefix (P-66.1.1.4.3, "
    "BlueBookV2.md:32998) on the sulfate/sulfonate ester; the anion-ester path "
    "has no amido producer (it ships the N-acyl float below pin_verified) -- TODO "
    "in TRIAGE.md 'Suite fix -- j5-pin-labels-b'"))
def test_fragment_pin_spelling(smiles, expected):
    assert _dt_name(smiles) == expected
