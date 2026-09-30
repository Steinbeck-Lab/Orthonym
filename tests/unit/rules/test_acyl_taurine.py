"""Simple N-acyl-taurine / acyl-amino-sulfonate anion naming (a phase).

a trace finding (see internal notes
acyltaurine-report.md for the full trace): the primary target,
N-acetyltaurine anion (``CC(=O)NCCS(=O)(=O)[O-]``), was measured as an
abstention in an earlier session but is ALREADY NAMED CORRECTLY at this
commit -- no production code change was required. The single-anion
sulfonate path (``ions.py::name_anion`` -> ``charged_router.route_charged``)
neutralizes to the acid, names it via the general/polyfunctional engine
(which already builds the method-(1) amido prefix --
``formamido``/``acetamido``/``{stem}anamido`` -- via
``composer.py::_check_for_acylamino`` -> ``substituent_naming.py::
linear_acyl_amido_prefix``), then converts the ``-ic acid`` suffix to
``-ate`` (the pre-existing, byte-identical-since-173.6 oxoacid-anion
suffix conversion). These tests LOCK that behaviour down as a regression
guard and add end-to-end coverage that did not exist before.

Also verified at the time this file was written (2026-08-17, not fixed here,
out of scope for this slice): a LONGER unbranched acyl chain (propanoyl, 3
carbons) on the SAME taurine skeleton abstained, because the general
engine's principal-chain selection picked the longer all-carbon acyl chain
as the parent instead of the shorter chain carrying the sulfonic acid
(principal characteristic group) -- a real but SEPARATE, broad
general-engine defect (chain selection ignoring principal-group location),
well outside "simple acyl-amino-sulfonate anion naming" scope. It failed
CLOSED (abstained to the sentinel) rather than emitting a wrong name, so
0-wrong held; not fixed in this Phase-3 slice.

UPDATE (2026-08-18, a phase lead a): that general-engine principal-chain
defect is now FIXED (`perception/chains.py::find_principal_chain` -- the
heteroatom-only-suffix acid classes, sulfonic/sulfinic/phosphonic/phosphinic
+ imidic/peroxoic/thioic S variants, now register their S/P-bearing carbon
into `fg_atoms` so criterion 1 -- "chain contains the PCG" -- picks the
correct, acid-bearing chain instead of tying at 0 and falling through to
"longest chain"). The propanoyl-taurine anion below now names correctly as
``2-propanamidoethane-1-sulfonate`` (RT-verified,
an InChIKey on both sides) --
`test_integration_longer_acyl_chain_failclosed_not_wrong` below was updated
to assert the new correct name in place of the stale abstain assertion. The
carbamoyl-direction shape (carboxylate senior parent, taurine as a
(2-sulfoethyl)carbamoyl prefix) is a different molecule/scope and was not
re-checked here; it may still abstain.
"""
import pytest
from orthonym import Orthonym
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
    "C[N+](C)(C)CCOS(=O)(=O)[O-]",
})
#... whose best-effort name is another one (it reads back exactly)
BEST_EFFORT_NAMES_IT_OTHERWISE = frozenset()


def _declined_pin_row(smiles):
    return declined_pin_row(
        smiles, best_effort_same=smiles not in BEST_EFFORT_NAMES_IT_OTHERWISE)


def _dt_obj_name(namer_obj, smiles):
    if smiles in DEFAULT_TIER_DECLINES:
        return _declined_pin_row(smiles)["name"]
    return namer_obj.name(smiles)


def _dt_obj_row(namer_obj, smiles):
    if smiles in DEFAULT_TIER_DECLINES:
        return _declined_pin_row(smiles)
    return namer_obj.name_tiered(smiles)


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



@pytest.fixture(scope="module")
def namer():
    return Orthonym()


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi,expected", [
    # N-acetyltaurine anion -- the primary target.
    ("CC(=O)NCCS(=O)(=O)[O-]", "2-acetamidoethane-1-sulfonate"),
    # N-formyltaurine anion -- a second simple acyl-taurine target.
    ("O=CNCCS(=O)(=O)[O-]", "2-formamidoethane-1-sulfonate"),
])
def test_integration_acyl_taurine_anion_names(namer, smi, expected):
    assert _dt_obj_name(namer, smi) == expected


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi,expected", [
    # neutral acyl-taurine acids (the intermediate the anion path
    # neutralizes through) unchanged/working.
    ("CC(=O)NCCS(=O)(=O)O", "2-acetamidoethane-1-sulfonic acid"),
    ("O=CNCCS(=O)(=O)O", "2-formamidoethane-1-sulfonic acid"),
    # regressions named in the task -- must be unchanged by this slice
    # (no production code was touched, but lock them anyway).
    ("CS(=O)(=O)[O-]", "methanesulfonate"),
    ("C1=CC=CC=C1S(=O)(=O)[O-]", "benzenesulfonate"),
    ("OC(=O)CCS(=O)(=O)[O-]", "2-carboxyethane-1-sulfonate"),
    ("O=C([O-])c1ccc(S(=O)(=O)[O-])cc1", "4-sulfonatobenzoate"),
    ("[O-]C(=O)CCC(=O)[O-]", "butanedioate"),
    ("CCCCCCCCCCCCOS(=O)(=O)[O-]", "dodecyl sulfate"),
    ("C[N+](C)(C)CCOS(=O)(=O)[O-]", "2-(trimethylazaniumyl)ethyl sulfate"),
])
def test_integration_regressions_unchanged(namer, smi, expected):
    assert _dt_obj_name(namer, smi) == expected


@pytest.mark.opsin_gate
def test_integration_bile_acid_taurine_conjugate_failclosed(namer):
    # A steroid/bile-acid taurine conjugate (cholan-24-amide of taurine,
    # anion form) needs the steroid ring parent to name correctly -- out
    # of scope here (SCOPE: only the simple acyl-amino-sulfonate case).
    # Must fail CLOSED to the sentinel abstention, never emit a wrong
    # (e.g. atom-dropping) name.
    smi = "C[C@H](CCC(=O)NCCS(=O)(=O)[O-])[C@H]1CCC2C1(C)CCC3C2CCC4CC(O)CCC34C"
    assert _dt_obj_name(namer, smi) == "unknown organic compound"


@pytest.mark.opsin_gate
def test_integration_longer_acyl_chain_now_named(namer):
    # A longer unbranched acyl (propanoyl) on the same taurine skeleton used
    # to abstain (a separate, out-of-scope general-engine principal-chain
    # selection defect -- see module docstring). a phase lead a fixed
    # that defect (perception/chains.py::find_principal_chain now registers
    # the sulfonic acid's bearing carbon), so this now names correctly.
    # RT-verified: an InChIKey on both sides.
    smi = "CCC(=O)NCCS(=O)(=O)[O-]"
    assert _dt_obj_name(namer, smi) == "2-propanamidoethane-1-sulfonate"
