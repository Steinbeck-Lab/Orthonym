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

# REQUIRED, and it is what makes the assertion below mean anything.
# `tests/conftest.py:302-317` disables the OPSIN validity gate suite-wide and
# documents the trap: with the gate OFF these fragments emit a plausible name for
# a DIFFERENT molecule instead of a sentinel. Measured here -- without this
# marker 7 of the 11 XPASS while `DROP-09 substituent_skip` fires, i.e. the
# "success" is a silent atom drop; in a fresh gated process all 11 refuse.
#
# With the gate ON the oracle is sound in both directions: escaping the sentinel
# requires a name SELF-01 accepts, and SELF-01 suppresses on a verified
# constitutional mismatch (`namer.py:920-924`). So an XPASS here means the
# fragment named AND kept its atoms -- it cannot be won by dropping them.
pytestmark = pytest.mark.opsin_gate

FRAGMENTS = [
    # acylurea on a tetrahydroisoquinolinium
    "C[C@H]([C@H]1C2=CC(=C(C=C2CC[NH+]1CC3=CC(=CC=C3)F)OC)OC)NC(=O)NC(C)C",
    # hexakis(diethyldithiocarbamate) + Mo -- organometallic, P-69 territory
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


@pytest.mark.parametrize("smiles", FRAGMENTS)
@pytest.mark.xfail(strict=True, reason="acyclic last-resort gap, v30 phase PB")
def test_acyclic_fragment_names_without_a_sentinel(smiles):
    name = Orthonym(general_fallback=True).name(smiles)
    assert not is_refusal_sentinel(name), name
