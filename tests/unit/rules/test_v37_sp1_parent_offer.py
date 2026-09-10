""" — parent-selection OFFER-not-return (a project rule).

VERIFIED on HEAD 030cc1dd (fresh process, trace re-confirm 2026-08-25):
`p44_scorer.select_parent_unified` builds a real >=2 parent pool for these
molecules (`[('ring',6),('chain',4)]`, size 2) but historically hard-committed
`ranked[0]` and discarded `ranked[1:]` (p44_scorer.py:543-545); the gate-reject
retry (`namer._retry_cascade_on_gate_rejection`,:2740) only swaps the DISPATCH
CLASS, so it re-picks the SAME parent every pass. When the senior parent
(`ranked[0]`, the benzene ring) leads to an un-nameable remainder, the molecule
abstained even though the junior parent (`ranked[1]`, the chloro-butynyl chain)
names + round-trips.

The fix OFFERS the ranked pool: on the ship-a-failure BEST-EFFORT path only,
`_try_alternate_parent_rescue` re-names on a fresh instance with
`_forced_parent_rank=k` (k=1..pool-1) and adopts the FIRST junior candidate
whose full name round-trips. `ranked[0]` and the ranking are NEVER changed;
the RT gate keeps it 0-wrong.

TIER NOTE (measured): the rescued names are best-effort systematic/mancude forms
(e.g. `...cyclohexa-1,3,5-trien-1-yl...`), so the rescue is gated to the
best-effort tier — exactly like the sibling `_try_demote_senior_group_rescue` /
`_try_np_systematic_downgrade` levers. At the PIN/default tier these witnesses
still fail closed (abstain), which is correct: breadth lives at best-effort, PIN
is fail-closed. The brief's `name_compound(smi)` snippet was illustrative; the
verified rescue tier is best-effort.
"""
import pytest

from orthonym import errors
from orthonym.namer import name_compound, Orthonym
from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check

# W1 (ChEBI abstain census) / W2 (minimal) + W2's alternate spelling. All abstain
# on HEAD; each has a size-2 pool [ring6, chain4]; the junior chain names + RT.
W1 = "O=C(Nc1cccc(Cl)c1)OCC#CCCl"
W2 = "O=C(Nc1ccccc1)OCC#CCCl"
W2_ALT = "ClCC#CCOC(=O)Nc1ccccc1"
WITNESSES = [W1, W2, W2_ALT]


def _rt_full(smi: str, name) -> bool:
    """Full-InChIKey round-trip: name -> OPSIN -> InChI == input's InChI."""
    if not name or errors.is_failure_name(name):
        return False
    r = opsin_roundtrip_check(smi, name)
    return bool(r.get("passed") and r.get("inchi_match"))


def _best_effort() -> Orthonym:
    return Orthonym(general_fallback=True, general_fallback_unverified=True,
                     allow_aromatic_general=True)


# --- 0-wrong ABSOLUTE: the rescue never emits a wrong molecule ---------------
class TestNeverWrong:
    @pytest.mark.opsin_gate
    @pytest.mark.parametrize("smi", WITNESSES)
    def test_witness_names_right_molecule_or_abstains(self, smi):
        """Best-effort either abstains or names the RIGHT molecule (RT-exact).
        Holds in BOTH the pre-fix (abstain) and post-fix (rescue) states."""
        name = _best_effort().name(smi)
        assert name is None or errors.is_failure_name(name) or _rt_full(smi, name), (
            f"0-wrong violation: {smi} -> {name!r} does not round-trip")


# --- THE DELIVERABLE: the junior parent rescues the senior dead-end ----------
class TestJuniorParentRescue:
    @pytest.mark.opsin_gate
    @pytest.mark.parametrize("smi", WITNESSES)
    def test_junior_parent_rescues_when_senior_dead_ends(self, smi):
        name = _best_effort().name(smi)
        assert name and not errors.is_failure_name(name), (
            f"expected an offer-retry rescue name for {smi}, got {name!r}")
        assert _rt_full(smi, name), f"rescued name does not RT: {name!r}"

    @pytest.mark.opsin_gate
    def test_spelling_independent(self):
        """W2 and its alternate SMILES spelling must produce identical output
        (a structure function, not a SMILES-order artifact)."""
        assert _best_effort().name(W2) == _best_effort().name(W2_ALT)

    @pytest.mark.opsin_gate
    @pytest.mark.parametrize("smi", WITNESSES)
    def test_deterministic_two_fresh_instances(self, smi):
        assert _best_effort().name(smi) == _best_effort().name(smi)


# --- PIN never regresses -----------------------------------------------------
class TestPinNoRegress:
    @pytest.mark.opsin_gate  # production gate ON — else the pre-gate wrong
    # carbamate candidate ships unsuppressed (a test-env artifact, not a real
    # deployment: in a fresh process with the gate on abstains).
    @pytest.mark.parametrize("smi", WITNESSES)
    def test_pin_tier_still_fails_closed(self, smi):
        """At the PIN/default tier the offer-retry never fires (it is
        best-effort-gated), so these still abstain — fail-closed, not a wrong
        or non-PIN name."""
        name = name_compound(smi)
        assert name is None or errors.is_failure_name(name), (
            f"PIN tier must fail closed on {smi}, got {name!r}")

    @pytest.mark.parametrize("smi,expected", [
        ("CCO", "ethanol"),
        ("c1ccccc1", "benzene"),
        ("CC(C)(C)CC(=O)O", "3,3-dimethylbutanoic acid"),
        ("OCC1CCCCC1", "cyclohexylmethanol"),   # HAS a ring+chain pool; names at ranked[0]
        ("c1ccc2ccccc2c1", "naphthalene"),
        ("CCOC(=O)C", "ethyl acetate"),
        ("Cc1ccccc1", "toluene"),
        ("CC(=O)Nc1ccccc1", "N-phenylacetamide"),  # amide near the witnesses' class
    ])
    def test_default_tier_controls_byte_identical(self, smi, expected):
        """A molecule that already names at ranked[0] must be byte-identical —
        the ranking is never reordered and the offer only fires on abstention."""
        assert name_compound(smi) == expected

    @pytest.mark.opsin_gate
    @pytest.mark.parametrize("smi,expected", [
        ("OCC1CCCCC1", "cyclohexylmethanol"),
        ("CC(=O)Nc1ccccc1", "N-phenylacetamide"),
    ])
    def test_best_effort_controls_byte_identical(self, smi, expected):
        """Best-effort must not perturb a molecule that names at ranked[0]
        (its pool is offered but never consulted because ranked[0] succeeds)."""
        assert _best_effort().name(smi) == expected
