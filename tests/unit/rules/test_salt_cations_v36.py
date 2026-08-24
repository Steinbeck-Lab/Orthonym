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
# Task 2 (B1): Category D (wrong parent) -- MEASURED C3-breadth-blocked
# ---------------------------------------------------------------------------
# idx10 names the aromatic diester ring as parent, dropping the tert-butylamino-
# alcohol side chain. The plan's premise (forcing the charged N as principal
# fixes it) is REFUTED by measurement: even with the amine forced as principal,
# find_principal_chain LINEARIZES the aryl-diester side chain into a 24-carbon
# 'tetracosyl' substituent (the general ring-aware substituent-naming gap,
# Milestone C3) -- so no correct name is derivable here yet. What Task 1's
# atom-coverage guard DOES guarantee is 0-wrong: idx10 ABSTAINS cleanly instead
# of shipping the wrong-parent / atom-dropped benzenium diester.
CATD_SALT = ("CS(=O)(=O)[O-].Cc1ccc(C(=O)Oc2ccc(C(O)C[NH2+]C(C)(C)C)cc2OC(=O)"
             "c2ccc(C)cc2)cc1")


@pytest.mark.opsin_gate
def test_catD_abstains_cleanly_never_wrong_parent(namer):
    """0-wrong: idx10 must NOT ship the atom-dropped/wrong-parent benzenium
    diester. A correct name is blocked on C3 substituent breadth, so a clean
    abstention is the correct behaviour today."""
    out = namer.name(CATD_SALT)
    if not is_failure_name(out):
        # If ever named, it must be the right molecule (never the bare diester).
        assert "benzenium" not in out.lower(), f"wrong parent shipped: {out!r}"
        assert _rt_full(CATD_SALT, out), f"shipped non-round-tripping name: {out!r}"
    else:
        assert is_failure_name(out)  # clean abstain (0-wrong preserved)


# ---------------------------------------------------------------------------
# Task 3 (B1): bridged-ring quaternary N -- MEASURED C3-breadth-blocked
# ---------------------------------------------------------------------------
# idx2: bridgehead quaternary N (a 2-propylpentanoate ester of an N,N-dimethyl-
# 8-azabicyclo[3.2.1]octan-3-ol cation). The quaternary N cannot be neutralized,
# AND -- measured -- even the des-N-substituent NEUTRAL skeleton fails to name:
# the ester wins parent selection and LINEARIZES the bridged ring ('heptyl
# 2-propylpentanoate'), so the von-Baeyer ring parent is never built. That is
# the same C3 substituent/ring breadth gap as Task 2, not a charged-path fix.
# Task 1 guarantees a clean 0-wrong abstention.
CATB_SALT = "CCCC(CCC)C(=O)OC1CC2CCC(C1)[N+]2(C)C.[Br-]"


@pytest.mark.opsin_gate
def test_catB_abstains_cleanly_never_wrong(namer):
    out = namer.name(CATB_SALT)
    if not is_failure_name(out):
        assert _rt_full(CATB_SALT, out), f"shipped non-round-tripping name: {out!r}"
    else:
        assert is_failure_name(out)  # clean abstain (0-wrong preserved)


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


# --- Task 4b: [H+]-diamine / base-hemisalt protonation transform (salts.py) ---
# A base + organic-acid salt is often written with the proton(s) floating as bare
# [H+] and the base(s) NEUTRAL, the acid fully deprotonated. Protonate the amine
# site(s) (one [H+] per site) and re-enter the salt namer.
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi,expected", [
    # 2 ethylamine bases + malonate dianion + 2 [H+]
    ("CCN.CCN.O=C([O-])CC(=O)[O-].[H+].[H+]", "diethanaminium propanedioate"),
    # 2 ethylamine bases + oxalate dianion + 2 [H+]
    ("CCN.CCN.[O-]C(=O)C(=O)[O-].[H+].[H+]", "diethanaminium oxalate"),
    # ethylenediamine (a diamine) + malonate dianion + 2 [H+] -> bis(aminium)
    ("NCCN.O=C([O-])CC(=O)[O-].[H+].[H+]",
     "ethane-1,2-bis(aminium) propanedioate"),
])
def test_h_plus_base_hemisalt(namer, smi, expected):
    out = namer.name(smi)
    assert out == expected, f"{smi} -> {out!r}"
    assert _rt_full(smi, out), f"round-trip failed: {out!r}"


@pytest.mark.opsin_gate
def test_h_plus_controls_unchanged(namer):
    # halide hydroacid path still works (organic base . [H+] . [Cl-])
    assert namer.name("CCN.[H+].[Cl-]") == "ethanamine hydrochloride"
    # a bare proton + halide with NO base must still fail closed (0-wrong)
    assert is_failure_name(namer.name("[H+].[Cl-]"))


# ---------------------------------------------------------------------------
# FABLE #1 (determinism BLOCKER): _protonate_amines_with_h_plus must choose
# the protonation site(s) by a canonical-rank order (not raw atom index /
# SMARTS-match order), and backtrack over the alternatives, so that every
# spelling of the SAME molecule gives the SAME result. RED before the fix:
# '[H+].NCCCNC.CC([O-])=O' named while '[H+].CNCCCN.CC([O-])=O' (identical
# canonical SMILES) abstained, because the secondary N was protonated first
# for one atom ordering and that cation failed to name.
# ---------------------------------------------------------------------------

_HPLUS_DETERMINISM_SPELLINGS = [
    "[H+].NCCCNC.CC([O-])=O",
    "[H+].CNCCCN.CC([O-])=O",
    "[H+].CC([O-])=O.NCCCNC",
]
_HPLUS_DETERMINISM_EXPECTED = "3-(methylamino)propan-1-aminium acetate"


@pytest.mark.opsin_gate
def test_h_plus_protonation_site_choice_is_spelling_deterministic(namer):
    """Same canonical molecule, three different atom orderings in the input
    SMILES -> the SAME emitted name (and it must be the correct, round-
    tripping name -- not a shared abstention)."""
    from rdkit import Chem

    canon = {Chem.CanonSmiles(s) for s in _HPLUS_DETERMINISM_SPELLINGS}
    assert len(canon) == 1, "test fixture spellings are not the same molecule"

    outputs = {namer.name(s) for s in _HPLUS_DETERMINISM_SPELLINGS}
    assert len(outputs) == 1, (
        f"non-deterministic across spellings of the same canonical SMILES: "
        f"{[(s, namer.name(s)) for s in _HPLUS_DETERMINISM_SPELLINGS]}")

    out = outputs.pop()
    assert out == _HPLUS_DETERMINISM_EXPECTED, out
    assert not is_failure_name(out), f"abstained: {out!r}"
    for smi in _HPLUS_DETERMINISM_SPELLINGS:
        assert _rt_full(smi, out), f"round-trip failed for {smi!r}: {out!r}"


# ---------------------------------------------------------------------------
# FABLE #3 (hardening nit): dispatch_table._handle_poly_anion's name_anion
# fallback is now wrapped in the same _full_inchikey_rt_ok check the
# MIXED_SIGN_ZWITTERION door already applies to its identical fallback.
# Regression only -- confirm the RT gate does not break the working
# bare-oxoanion path it wraps.
# ---------------------------------------------------------------------------

@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi,expected", [
    ("O=S([O-])[O-]", "sulfite"),
    ("[O-][N+](=O)[O-]", "nitrate"),
    ("[O-][Cl+][O-]", "chlorite"),
])
def test_poly_anion_name_anion_fallback_rt_gate_regression(namer, smi, expected):
    out = namer.name(smi)
    assert out == expected, f"{smi} -> {out!r}"
    assert _rt_full(smi, out), f"round-trip failed: {out!r}"
