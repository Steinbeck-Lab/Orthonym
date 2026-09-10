""" a phase Task 1.1 — RT-gate the best-effort glycan path fail-CLOSED.

Census (PHASE0-GLYCAN-DECLINE-CENSUS.md, `scratchpad/glycan_census.py` pass B over
the 413-row glycan backlog): the best-effort tier shipped 44 WRONG + 46
opsin_unparseable names because `_sugar_name_rt_ok`'s "cannot determine" branches
(missing jar, or ANY exception during the OPSIN check) returned ``True`` -- i.e.
shipped the candidate name UNVERIFIED -- regardless of tier. The PIN/default tier
already measured 0 wrong on this same 413-row set because its own downstream
`_final_opsin_validity_gate` / gate is a second backstop that catches a
fail-open miss; the best-effort tier has NO such backstop (emissions bypass it
by design), so `_sugar_name_rt_ok`'s fail-open was the ONLY check standing between
an unverified glycan name and best-effort output.

⚠ a trace FINDING (a project rule, "choke point off path" -- recorded here so the next
session does not re-open this): none of the 44 wrong / 46 unparseable census
witnesses actually reach this function. Structural signal
(`scratchpad/pass_a_signals.json`, `oligo_cascade_name`) shows `name_disaccharide`
returns ``None`` for ALL 413 backlog rows (`_classify_units` fails first, the
documented root symptom) -- composing a name is a precondition for calling
`_sugar_name_rt_ok`, so it is never invoked with a candidate for this corpus at
all. The 44/46 wrong/unparseable emissions originate from an UNRELATED mechanism
(`namer.py`'s general-engine late-recovery ladder, ~line 3680-3745, which labels a
CONSTITUTION-ONLY round-trip "verified" when a stereocentre is flagged
`stereo_unexpressed` -- a different hole, out of this task's scope; see
task-1.1-report.md). This module's tests therefore exercise the ACTUAL changed
code path directly (a controlled RT-mismatch/exception under `best_effort_ctx`)
rather than replaying a census witness that never reaches it.
"""
from unittest.mock import patch

from rdkit import Chem

from orthonym.metrics.provenance import best_effort_ctx
from orthonym.rules import oligosaccharides as O
from orthonym.rules.oligosaccharides import _sugar_name_rt_ok

# beta-Maltose (same molecule as tests/unit/rules/test_disaccharide.py) -- a real
# disaccharide the assembler composes + OPSIN-round-trips correctly today.
MALTOSE_SMILES = (
    "OC[C@H]1O[C@H](O[C@H]2[C@H](O)[C@@H](O)[C@H](O)O[C@@H]2CO)"
    "[C@H](O)[C@@H](O)[C@@H]1O"
)
MALTOSE_EXPECTED = "α-D-glucopyranosyl-(1->4)-β-D-glucopyranose"

_MOD = "orthonym.validation.opsin_roundtrip"


def _with_be_ctx(value, fn, *args, **kwargs):
    tok = best_effort_ctx.set(value)
    try:
        return fn(*args, **kwargs)
    finally:
        best_effort_ctx.reset(tok)


class TestRtGateMissingJar:
    """The 'cannot check at all' branch (no OPSIN jar / no Java)."""

    def test_best_effort_fails_closed_when_jar_missing(self):
        mol = Chem.MolFromSmiles(MALTOSE_SMILES)
        with patch(f"{_MOD}._find_opsin_jar", return_value=None):
            ok = _with_be_ctx(True, _sugar_name_rt_ok, mol, "not a real name")
        assert ok is False, "best-effort must ABSTAIN when it cannot verify at all"

    def test_pin_tier_stays_fail_open_when_jar_missing(self):
        # Byte-identical regression: PIN/default tier behaviour is UNCHANGED.
        mol = Chem.MolFromSmiles(MALTOSE_SMILES)
        with patch(f"{_MOD}._find_opsin_jar", return_value=None):
            ok = _with_be_ctx(False, _sugar_name_rt_ok, mol, "not a real name")
        assert ok is True, "PIN/default tier must stay fail-OPEN (no regression)"


class TestRtGateException:
    """The 'check raised' branch -- previously fail-open unconditionally."""

    def test_best_effort_fails_closed_on_exception(self):
        mol = Chem.MolFromSmiles(MALTOSE_SMILES)
        with patch(f"{_MOD}._find_opsin_jar", return_value="/fake/opsin.jar"), \
             patch(f"{_MOD}._java_available", return_value=True), \
             patch(f"{_MOD}.opsin_roundtrip_check", side_effect=RuntimeError("boom")):
            ok = _with_be_ctx(True, _sugar_name_rt_ok, mol, "irrelevant name")
        assert ok is False, "best-effort must ABSTAIN when the RT check itself raises"

    def test_pin_tier_stays_fail_open_on_exception(self):
        mol = Chem.MolFromSmiles(MALTOSE_SMILES)
        with patch(f"{_MOD}._find_opsin_jar", return_value="/fake/opsin.jar"), \
             patch(f"{_MOD}._java_available", return_value=True), \
             patch(f"{_MOD}.opsin_roundtrip_check", side_effect=RuntimeError("boom")):
            ok = _with_be_ctx(False, _sugar_name_rt_ok, mol, "irrelevant name")
        assert ok is True, "PIN/default tier must stay fail-OPEN on exception (no regression)"


class TestRtGateCleanMismatch:
    """A CLEAN mismatch (OPSIN parses fine, InChI differs) already failed closed in
    BOTH tiers before this change -- must stay that way (only the 'cannot check'
    branches were fail-open)."""

    def test_clean_mismatch_abstains_in_both_tiers(self):
        mol = Chem.MolFromSmiles(MALTOSE_SMILES)
        with patch(f"{_MOD}._find_opsin_jar", return_value="/fake/opsin.jar"), \
             patch(f"{_MOD}._java_available", return_value=True), \
             patch(f"{_MOD}.opsin_roundtrip_check",
                   return_value={"passed": False, "error": "inchi_mismatch"}):
            ok_be = _with_be_ctx(True, _sugar_name_rt_ok, mol, "a wrong name")
            ok_pin = _with_be_ctx(False, _sugar_name_rt_ok, mol, "a wrong name")
        assert ok_be is False
        assert ok_pin is False


class TestRegressionRealDisaccharide:
    """A plain disaccharide OST already names correctly must be UNCHANGED on both
    tiers -- the RT-gate tightening must never abstain a name that genuinely
    round-trips. Uses the REAL OPSIN jar (no mocking) via the existing production
    verification path, matching tests/unit/rules/test_disaccharide.py."""

    def test_maltose_pin_tier_unchanged(self):
        mol = Chem.MolFromSmiles(MALTOSE_SMILES)
        assert mol is not None
        name = _with_be_ctx(False, O.name_disaccharide, mol)
        assert name == MALTOSE_EXPECTED, name

    def test_maltose_best_effort_unchanged(self):
        mol = Chem.MolFromSmiles(MALTOSE_SMILES)
        assert mol is not None
        name = _with_be_ctx(True, O.name_disaccharide, mol)
        assert name == MALTOSE_EXPECTED, name
