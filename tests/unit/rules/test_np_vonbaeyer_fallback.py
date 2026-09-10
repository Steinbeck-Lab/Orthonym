""" giants Engine 3: best-effort von-Baeyer downgrade of a retained
natural-product parent hydride.

The stereoparents (`ursane`, `hopane`, `cevane`,...) ARE
the PIN, and `namer._final_opsin_validity_gate` whitelists them (its
`np_stereoparent` carve-out) because OPSIN 2.9.0 cannot parse a single one of
them. Correct for the PIN tiers -- but on the best-effort path it costs a
round-trip that the general engine wins, so `_try_np_systematic_downgrade`
offers the von-Baeyer systematic name there instead.

The round-trip gate on that swap is LOAD-BEARING, not defensive. Measured over
all 53 skeletons whose `NATURAL_PRODUCT_DERIVATIVES` value is in
`NAME_EXACT_NP_PARENTS`: 51 convert to a full-InChI-round-tripping systematic
name, and 2 do NOT --

  * `germacrane`, whose systematic form is a NON-round-tripping
    `...-1,7-dimethyl-4-(propan-2-yl)cyclodecane`, and
  * `corynoxan`, for which the general engine produces nothing at all.

An unconditional downgrade therefore REGRESSES those two (to a wrong name and
to the abstention sentinel respectively), which is what the two keep-retained
tests below pin.
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.data.natural_products import (
    NATURAL_PRODUCT_DERIVATIVES,
    NAME_EXACT_NP_PARENTS,
)

pytestmark = pytest.mark.opsin_gate

# SMILES are read from the shipped table, keyed by the retained name, so a test
# can never drift from the data the producer actually matches on.
_BY_NAME = {n: s for s, n in NATURAL_PRODUCT_DERIVATIVES.items()
            if n in NAME_EXACT_NP_PARENTS}

# Representative converters: two triterpenes, a pentacyclic triterpene pair, a
# steroidal alkaloid (N in the cage), an indole alkaloid parent and a diterpene.
CONVERTS = ["ursane", "hopane", "lupane", "oleanane", "cevane", "yohimban",
            "gibbane", "corynoxan"]

# The skeleton with no round-tripping systematic form. (: `corynoxan` now
# converts — its stereo-composed systematic spiro name round-trips at 0-wrong, so
# it moved to CONVERTS; `germacrane`'s systematic cyclodecane form still does not
# round-trip, so best-effort keeps the retained PIN. The BE-STRICT-unparseable
# gate used to DROP `germacrane` to the abstention sentinel at best-effort — a T4
# defect fixed by exempting the construction-verified NAME_EXACT_NP_PARENTS from
# best-effort suppression, since they are 0-wrong by exact table membership.)
KEEPS = ["germacrane"]


def _pin_namer():
    return Orthonym(style="pin")


def _best_effort_namer():
    return Orthonym(style="pin", general_fallback=True,
                     general_fallback_unverified=True,
                     allow_aromatic_general=True)


@pytest.mark.parametrize("retained", CONVERTS + KEEPS)
def test_default_path_still_emits_the_retained_pin(retained):
    """The PIN/default path is byte-identical -- Engine 3 is best-effort-only."""
    smiles = _BY_NAME[retained]
    assert _pin_namer().name(smiles) == retained


@pytest.mark.parametrize("retained", CONVERTS)
def test_best_effort_emits_a_round_tripping_systematic_name(retained):
    """Best-effort swaps in a systematic name that survives a FULL InChIKey
    round-trip through OPSIN (constitution AND stereo)."""
    from orthonym.validation import opsin_roundtrip_check

    smiles = _BY_NAME[retained]
    name = _best_effort_namer().name(smiles)
    assert name != retained, f"{retained} was not downgraded"

    rt = opsin_roundtrip_check(smiles, name)
    assert rt["passed"], f"{retained} -> {name!r}: {rt.get('error')}"

    # Full InChIKey identity, asserted independently of the helper's own verdict.
    opsin_mol = Chem.MolFromSmiles(rt["opsin_smiles"])
    assert opsin_mol is not None
    assert (Chem.MolToInchiKey(opsin_mol)
            == Chem.MolToInchiKey(Chem.MolFromSmiles(smiles)))


@pytest.mark.parametrize("retained", KEEPS)
def test_best_effort_keeps_the_retained_name_when_systematic_does_not_rt(retained):
    """The RT gate is load-bearing: these two have no round-tripping systematic
    form, so the retained PIN must survive best-effort untouched."""
    assert _best_effort_namer().name(_BY_NAME[retained]) == retained


@pytest.mark.parametrize("retained", KEEPS)
def test_kept_row_provenance_is_not_mislabelled_general_engine(retained):
    """FABLE regression: the RT-probe runs the general-engine recovery,
    which stamps `source="general_engine"` before the RT gate can decline it. A
    KEPT retained name must NOT inherit that label (it would skew the cohort /
    refusal-census attribution the project ranks levers by).
    `_try_np_systematic_downgrade` restores the pre-probe provenance on the
    reject path. Uses `name_tiered`, the API that clears provenance per call and
    is what the cohort/census machinery actually reads."""
    r = _best_effort_namer().name_tiered(_BY_NAME[retained])
    assert r["name"] == retained
    assert r["source"] != "general_engine"  # -> "pin_path", tier T1


def test_converted_row_provenance_is_general_engine(retained="ursane"):
    """The adopted (converted) row DOES carry the general-engine source -- the
    restore fires only on reject, never on adopt."""
    r = _best_effort_namer().name_tiered(_BY_NAME[retained])
    assert r["name"] != retained
    assert r["source"] == "general_engine"


def test_downgrade_helper_declines_off_the_best_effort_path():
    """`_try_np_systematic_downgrade` is a no-op without the best-effort opt-in,
    which is what makes the PIN default byte-identical by construction."""
    pin = _pin_namer()
    assert pin._try_np_systematic_downgrade(_BY_NAME["ursane"], "ursane") is None


def test_downgrade_helper_declines_a_name_outside_the_trigger_set():
    """Only a retained NP parent hydride is a downgrade candidate."""
    be = _best_effort_namer()
    assert be._try_np_systematic_downgrade("CCO", "ethanol") is None
