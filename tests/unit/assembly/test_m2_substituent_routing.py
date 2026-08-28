"""M2 Task 1 — tier propagation for ``name_substituent`` (best-effort scope only).

Measured (,
`.superpowers/sdd/M1-PLAN/m2-task-1-report.md`): 75/166 pubchem10k rows hit the
substituent cascade's ``substituent``/``unknown`` placeholder internally under the
best-effort tier. Of the 71 unique declined root fragments, 37 (mapping to 38
molecule rows) name ONLY at best-effort (never at PIN default) when tested
standalone -- the "tier-propagation" bucket. ``name_substituent``
(``substituent_enumerator.py:873``) is called by ~25 sites with the default
``allow_mancude=False``, so even inside a best-effort whole-molecule run the
cascade was still gated at the DEFAULT tier and declined those fragments.

An empirical fresh-process SPY over all 38 candidate rows (before vs. after the
fix, via ``) found the TRUE conversion rate at the
best-effort tier (``general_fallback=True, general_fallback_unverified=True,
allow_aromatic_general=True``) -- i.e. rows where the FINAL emitted name flips
from the abstention sentinel to a real, OPSIN-round-trip-verified name -- is
7/38: pubchem10k rows 23, 28, 31, 121, 126, 132, 141. The other 31 either
already named correctly via a parallel candidate path (3 rows: 20, 104, 154 --
not real abstentions) or still abstain and need Task 2's routing-to-the-general-
engine lever (28 rows). All 7 witnesses below are drawn from that measured set,
not invented.

⚠ Every test below is marked ``opsin_gate``. ``tests/conftest.py`` disables the
OPSIN self-consistency/validity gate (SELF-01) suite-wide by DEFAULT so most
tests can assert raw generator output cheaply -- but that makes "PIN default
still abstains" and "best-effort now names it" claims meaningless without the
gate, because gate-OFF ships whatever a recovery-lane candidate happens to
build, unguarded. Measured directly: WITHOUT the marker,
``test_default_tier_unchanged`` spuriously FAILED for 3/6 witnesses (a
gate-disabled recovery-lane artifact of the test harness, not a real
default-tier widening -- confirmed absent via `` against a
bare, unpatched ``orthonym`` import and reproduced/explained with
``tests/conftest.py``'s own ``_opsin_validity_gate_state`` fixture). WITH the
marker (gate ON, matching production), all 6 pass byte-identically.
"""
import pytest

from orthonym import Orthonym
from orthonym.errors import is_refusal_sentinel

# The 7 measured TRUE conversions (best-effort: abstain -> real OPSIN-valid name).
# pubchem10k.jsonl row indices kept in the comment for traceability back to the SPY.
TIER_PROPAGATION_WITNESSES = [
    "CCC[C@H](N)C(=O)N(C)[C@H]1CC[C@@H]2CN(Cc3ccc(C(F)(F)F)cc3)C[C@@H]21",  # row 23
    "CCC[C@@H](C)NC(=O)C[C@H]1Sc2ccc(C(F)(F)F)cc2NC1=O",                    # row 28
    "O=C(c1c2ccccc2cc2ccccc12)N1CCN(c2ccc(C(F)(F)F)cn2)CC1",                # row 31
    "O=C(Nc1ccc(Cl)c(C(=O)N(Cl)/N=C/CC(F)(F)F)c1)C1C(c2cc(Cl)cc(Cl)c2)C1(Cl)Cl",  # row 121
    "COc1ccc(C2Oc3cccc(OC(F)F)c3-c3ccc(NC(=O)N(C)C)cc32)cc1OC",             # row 126
    "CN=C(NCC1(O)CCSC1)N1CCN(C(C)C(F)(F)F)CC1",                            # row 132
    "CNC(=O)c1cc(COC(=O)NC2CC3(CCN(c4ccc5cc(F)ccc5n4)CC3)C2)[nH]n1",       # row 141
]

# A subset where PIN default was VERIFIED (head_ab.sh, HEAD vs. patched) to
# abstain byte-identically both before and after the fix -- the default-tier
# non-widening witnesses.
DEFAULT_STILL_ABSTAINS = [
    "CCC[C@H](N)C(=O)N(C)[C@H]1CC[C@@H]2CN(Cc3ccc(C(F)(F)F)cc3)C[C@@H]21",  # row 23
    "O=C(c1c2ccccc2cc2ccccc12)N1CCN(c2ccc(C(F)(F)F)cn2)CC1",                # row 31
    "O=C(Nc1ccc(Cl)c(C(=O)N(Cl)/N=C/CC(F)(F)F)c1)C1C(c2cc(Cl)cc(Cl)c2)C1(Cl)Cl",  # row 121
    "COc1ccc(C2Oc3cccc(OC(F)F)c3-c3ccc(NC(=O)N(C)C)cc32)cc1OC",             # row 126
    "CN=C(NCC1(O)CCSC1)N1CCN(C(C)C(F)(F)F)CC1",                            # row 132
    "CNC(=O)c1cc(COC(=O)NC2CC3(CCN(c4ccc5cc(F)ccc5n4)CC3)C2)[nH]n1",       # row 141
]

# M4-fold nested-branch prefixes (
# "M4-fold prefixes"): azido / sulfamoyl / methanesulfonyl on a substituent
# branch off a diol backbone. NOTE (measured via head_ab.sh): these three are
# UNCHANGED by the Task-1 edit -- best-effort already named them via a
# different path before the fix, so they are regression/documentation
# witnesses for the tier contrast, not conversions this fix produced.
M4_FOLD_CASES = {
    "azido": "OCC(CN=[N+]=[N-])CO",
    "sulfamoyl": "OCC(CS(N)(=O)=O)CO",
    "methanesulfonyl": "OCC(CS(C)(=O)=O)CO",
}


@pytest.fixture(scope="module")
def eng():
    return Orthonym(general_fallback=True, general_fallback_unverified=True,
                      allow_aromatic_general=True)


@pytest.fixture(scope="module")
def pin_eng():
    return Orthonym()


# --- Tier propagation: the splice is gone and a real name comes back --------

@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi", TIER_PROPAGATION_WITNESSES)
def test_tier_propagation_no_placeholder(eng, smi):
    name = eng.name(smi)
    assert name is not None
    assert not is_refusal_sentinel(name), name
    assert "substituent" not in name
    assert "unknown" not in name.lower()


@pytest.mark.roundtrip
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi", TIER_PROPAGATION_WITNESSES)
def test_tier_propagation_roundtrips(eng, smi):
    from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check
    name = eng.name(smi)
    assert opsin_roundtrip_check(smi, name)["passed"], name


# --- Negative: PIN default is byte-identical (no default-tier widening) -----

@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi", DEFAULT_STILL_ABSTAINS)
def test_default_tier_unchanged(pin_eng, smi):
    # These 6 rows abstain at PIN default both BEFORE and AFTER the fix
    # (verified via  on src/orthonym/assembly/
    # substituent_enumerator.py) -- best_effort_ctx is unset at default tier,
    # so name_substituent's effective allow_mancude is unchanged there. Needs
    # the ``opsin_gate`` marker (see module docstring) -- without it the
    # suite's gate-disabled default lets an unguarded recovery-lane candidate
    # leak through, which is a test-harness artifact, not a real regression.
    name = pin_eng.name(smi)
    assert is_refusal_sentinel(name), (
        f"PIN default must still abstain (byte-identical to pre-fix HEAD); "
        f"got a NEW default-tier emission: {name!r}")


# --- M4-fold nested-branch prefixes: best-effort names, default abstains ----

@pytest.mark.opsin_gate
@pytest.mark.parametrize("tag,smi", list(M4_FOLD_CASES.items()))
def test_m4_fold_besteffort_names_nested_prefix(eng, tag, smi):
    name = eng.name(smi)
    assert name is not None
    assert not is_refusal_sentinel(name), f"{tag}: {name!r}"
    assert "substituent" not in name
    assert "unknown" not in name.lower()


@pytest.mark.roundtrip
@pytest.mark.opsin_gate
@pytest.mark.parametrize("tag,smi", list(M4_FOLD_CASES.items()))
def test_m4_fold_besteffort_roundtrips(eng, tag, smi):
    from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check
    name = eng.name(smi)
    assert opsin_roundtrip_check(smi, name)["passed"], f"{tag}: {name}"
