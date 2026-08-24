"""v36 Milestone B1+B2 -- salt organic-cation producer + mixed/amine salts.

Grounding spies: `` (§6) and ``V36-SPY-B2.md``.

The 0-wrong core (Task 1) is an atom-coverage guard between ``route_charged``'s
neutralize->re-enter step and the ionic-suffix step: the gate-DISABLED re-entry
(``_reenter``) can let an ATOM-DROPPING retained/natural-product name through
(benzatropine cation -> ``tropane`` -> ``tropanium``, dropping the C3
diphenylmethoxy) before the ``-ium`` suffix is glued on. The guard re-parses the
neutral name through OPSIN and, on an atom shortfall, retries the re-entry with
the gate ON (which rejects the atom-dropping retained name and derives the
systematic von Baeyer parent).

Every emitted cation-salt name below is OPSIN-round-trip-verified (full InChI /
InChIKey match) or the row abstains -- never a wrong / atom-dropped molecule, and
never an unverified stereo descriptor (OPSIN 2.9.0 cannot parse a stereo
descriptor cited at ring position 3 of an azabicyclo -- SPY-B1 §3).
"""
import pytest

from orthonym import Orthonym
from orthonym.errors import is_failure_name


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


def _rt_full(smi, name):
    """OPSIN round-trip: name -> OPSIN -> InChI == input InChI (full, incl. stereo)."""
    from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check
    return bool(opsin_roundtrip_check(smi, name).get("passed"))


# ---------------------------------------------------------------------------
# Positive controls -- must remain byte-identical (route_charged is SHARED)
# ---------------------------------------------------------------------------

@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi,expected", [
    # carbachol / bethanechol: acyclic QUATERNARY aminium -- already correct
    ("C[N+](C)(C)CCOC(N)=O.[Cl-]",
     "2-(carbamoyloxy)-N,N,N-trimethylethanaminium chloride"),
    ("CC(C[N+](C)(C)C)OC(N)=O.[Cl-]",
     "2-(carbamoyloxy)-N,N,N-trimethylpropan-1-aminium chloride"),
    ("CCO", "ethanol"),
    ("c1ccccc1", "benzene"),
])
def test_positive_controls_unchanged(namer, smi, expected):
    assert namer.name(smi) == expected


# ---------------------------------------------------------------------------
# Task 1 (B1 core): atom-coverage guard -- benzatropine cation salt (Category A)
# ---------------------------------------------------------------------------

# Stereo-flattened benzatropine mesylate: constitution only (the SPY-verified
# green case). Target: 3-(diphenylmethoxy)-8-methyl-8-azabicyclo[3.2.1]octan-8-ium
# methanesulfonate -- OPSIN-RT exact to this flat input.
BENZATROPINE_SALT_FLAT = "CS(=O)(=O)[O-].C[NH+]1C2CCC1CC(OC(c1ccccc1)c1ccccc1)C2"
BENZATROPINE_SALT_STEREO = (
    "CS(=O)(=O)[O-].C[NH+]1[C@@H]2CC[C@H]1C[C@@H](OC(c1ccccc1)c1ccccc1)C2")


@pytest.mark.opsin_gate
def test_benzatropine_flat_no_atom_drop_and_round_trips(namer):
    out = namer.name(BENZATROPINE_SALT_FLAT)
    # No atom-drop: the C3 diphenylmethoxy must be present, never the bare scaffold.
    assert not is_failure_name(out), f"abstained: {out!r}"
    assert "tropan" not in out.lower(), f"atom-dropped retained scaffold: {out!r}"
    assert "diphenylmethoxy" in out or "benzhydryloxy" in out, out
    # 0-wrong: full OPSIN round-trip to the (flat) input.
    assert _rt_full(BENZATROPINE_SALT_FLAT, out), f"round-trip failed: {out!r}"


@pytest.mark.opsin_gate
def test_benzatropine_full_stereo_never_ships_unverified(namer):
    """The FULL-stereo salt: OPSIN 2.9.0 cannot parse a stereo descriptor at ring
    position 3 of the azabicyclooctane (SPY-B1 §3), so a full-stereo name cannot
    round-trip. It must abstain -- NEVER ship an unverified-stereo (or atom-dropped
    ``tropanium``) name."""
    out = namer.name(BENZATROPINE_SALT_STEREO)
    if not is_failure_name(out):
        # If anything is emitted it MUST round-trip to the full-stereo input and
        # MUST NOT be the atom-dropped scaffold.
        assert "tropan" not in out.lower(), f"atom-dropped: {out!r}"
        assert _rt_full(BENZATROPINE_SALT_STEREO, out), (
            f"shipped an unverified name: {out!r}")


# ---------------------------------------------------------------------------
# Task 2 (B1): force charged N to win parent selection -- Category D (wrong parent)
# ---------------------------------------------------------------------------

# idx10: currently names the aromatic diester ring as parent, dropping the whole
# tert-butylamino-alcohol side chain that carries the charge.
CATD_SALT = ("CS(=O)(=O)[O-].Cc1ccc(C(=O)Oc2ccc(C(O)C[NH2+]C(C)(C)C)cc2OC(=O)"
             "c2ccc(C)cc2)cc1")


@pytest.mark.opsin_gate
def test_catD_charged_n_wins_parent(namer):
    out = namer.name(CATD_SALT)
    assert not is_failure_name(out), f"abstained: {out!r}"
    # The aminium centre must survive -- not the bare benzenium diester.
    assert "amin" in out.lower() or "azaniumyl" in out.lower(), out
    assert _rt_full(CATD_SALT, out), f"round-trip failed: {out!r}"


# ---------------------------------------------------------------------------
# Task 3 (B1): bridged/fused-ring quaternary-N producer -- Category B
# ---------------------------------------------------------------------------

# idx2: bridgehead quaternary N (N,N-dimethyl-8-azabicyclo[3.2.1]octanium valerate
# ester). route_charged declines today; _try_neutralize_and_name crashes on the
# over-valent N. Target: an azonia/-ium ring name that round-trips.
CATB_SALT = "CCCC(CCC)C(=O)OC1CC2CCC(C1)[N+]2(C)C.[Br-]"


@pytest.mark.opsin_gate
def test_catB_bridged_quaternary_ring_n(namer):
    out = namer.name(CATB_SALT)
    assert not is_failure_name(out), f"abstained: {out!r}"
    assert _rt_full(CATB_SALT, out), f"round-trip failed: {out!r}"


# ---------------------------------------------------------------------------
# Task 4 (B2): aminium-suffix correctness (trometamol) + [H+]-diamine hemisalt
# ---------------------------------------------------------------------------

TROMETAMOL_CATION = "[NH3+]C(CO)(CO)CO"


@pytest.mark.opsin_gate
def test_trometamol_cation_is_opsin_valid(namer):
    """The retained neutral name 'trometamol' + blind '-ium' -> 'trometamolium',
    which OPSIN cannot parse. The aminium transform must derive an OPSIN-valid
    systematic form (e.g. 2-hydroxy-1,1-bis(hydroxymethyl)ethan-1-aminium)."""
    out = namer.name(TROMETAMOL_CATION)
    assert not is_failure_name(out), f"abstained: {out!r}"
    assert out != "trometamolium"
    assert _rt_full(TROMETAMOL_CATION, out), f"round-trip failed: {out!r}"
