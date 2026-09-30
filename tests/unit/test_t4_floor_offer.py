""" a phase Task L3-1: the systematic floor as a surviving, full-RT-gated
offer -- THE breadth lever (measured ~36/86 discard-gap conversions).

The floor (`assembly.t4_coverage.name_t4_complete`) already builds a
correct, full-round-tripping name for these molecules; before this task the
pipeline shipped a WRONG name (stereo fabricated/omitted on an achiral or
partially-specified input -- the dominant discard-gap mechanism) or
abstained, reaching emission for only 1/37. `Orthonym._maybe_append_t4_floor_offer`
appends the floor as a 2nd `Offer`, and the L3-1 `_offer_rt_ok` full-InChIKey
change (`test_rt_over_pool.py`) is what lets it out-rank a constitution-only-
verified but stereo-wrong primary.

Best-effort ONLY throughout -- every test here constructs
`Orthonym(general_fallback=True, general_fallback_unverified=True,
allow_aromatic_general=True)`; PIN-default byte-identity is covered
separately (`test_pin_byte_identity` below + the existing gold-set suite).
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

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
    "CC(O)C(N)C(=O)NC(CS)C(=O)O",
})
#... whose best-effort name is another one (it reads back exactly)
BEST_EFFORT_NAMES_IT_OTHERWISE = frozenset()


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


# The L3-1 mechanism (full-InChIKey `_offer_rt_ok` + the floor offer) is
# fundamentally an OPSIN-gate behaviour -- it cannot do anything with the
# gate disabled (the suite-wide default, `tests/conftest.py`), since
# `_offer_rt_ok` fails OPEN (True) without a live jar and the floor would
# never be needed. `opsin_gate` skips cleanly if the jar is absent.
pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


def _full_inchikey(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"unparseable SMILES in test fixture: {smiles}"
    return inchi.MolToInchiKey(mol)


def _best_effort_namer() -> Orthonym:
    return Orthonym(general_fallback=True, general_fallback_unverified=True,
                      allow_aromatic_general=True)


class TestFloorWinsOverStereoWrongPrimary:
    """L3 CHARACTERIZATION rows: EMIT_RT_FAIL primaries (constitution-verified,
    full-InChIKey-wrong) where the floor is FLOOR_COVERS_RT_EXACT."""

    @pytest.mark.parametrize("smiles", [
        "CC(O)C(N)C(=O)NC(CS)C(=O)O",              # threonylcysteine
        "CC(C(=O)NC(CCSC)C(=O)NCC(=O)NC(C)C(=O)O)N",  # alanylmethionylglycylalanine
    ])
    def test_emit_wrong_primary_converted_to_floor(self, smiles):
        nm = _best_effort_namer()
        name = nm.name(smiles)
        assert name and not name.startswith("unknown"), (
            f"expected a real emitted name, got {name!r}")
        opsin_smi = None
        from orthonym.validation.opsin_roundtrip import opsin_parse
        opsin_smi = opsin_parse(name)
        if opsin_smi is None:
            pytest.skip("OPSIN jar unavailable in this environment")
        assert _full_inchikey(smiles) == _full_inchikey(opsin_smi), (
            f"emitted name {name!r} does not full-RT-match the input")
        # re-baseline: the invariant these rows prove is that a
        # full-InChIKey-correct name SHIPS for a formerly stereo-wrong primary
        # -- via whichever offer wins. When these tests were written that was
        # the t4_floor offer; the primary producers have since improved to emit
        # a constitution- and stereo-correct name DIRECTLY for these molecules
        # (source pin_path/general_engine), so the floor is correctly not
        # needed. The floor-preference mechanism itself is exercised by
        # test_t4_stereo_verify_before_commit_v35.py. What must hold here is the
        # 0-wrong outcome asserted above: the shipped name is the right molecule.


class TestFloorWinsOverAbstainedPrimary:
    """L3 CHARACTERIZATION: the 2 ABSTAIN discard rows -- the primary path
    never produces a real name at all; `_finish`'s narrower
    `is_failure_name(name)` branch must seed the floor offer and let it win."""

    def test_abstained_primary_converted_to_floor(self):
        smiles = "CC(=O)N[C@@H](CCCN=C(N)N)C(=O)CCCCCCCCN=C(N)N"
        nm = _best_effort_namer()
        name = nm.name(smiles)
        assert name and not name.startswith("unknown"), (
            f"expected a real emitted name (floor rescue), got {name!r}")
        from orthonym.validation.opsin_roundtrip import opsin_parse
        opsin_smi = opsin_parse(name)
        if opsin_smi is None:
            pytest.skip("OPSIN jar unavailable in this environment")
        assert _full_inchikey(smiles) == _full_inchikey(opsin_smi), (
            f"emitted name {name!r} does not full-RT-match the input")
        # re-baseline: see TestFloorWinsOverStereoWrongPrimary -- the
        # formerly-abstaining primary now emits a full-RT-correct name directly,
        # so `_t4_floor_candidate` is None (the floor was not needed). The
        # 0-wrong outcome asserted above is the invariant that must hold.


class TestStereoOnlyFloorCompletenessL3_2a:
    """ a phase L3-2a: the STEREO_ONLY 7-row bucket from the L3-2/L3-3 a trace
    (`FLOOR_COVERS_RT_FAIL` -- both the primary AND the floor omitted real
    defined stereo on a decorated acyclic side-chain substituent, e.g.
    '1,2,3-trihydroxypropyl'/'5-(propan-2-yl)heptan-2-yl' with NO leading
    descriptor block, even though the constitution was already right).

    Root cause: `substituent_naming._add_substituent_stereo`'s multi-centre
    branch unconditionally declined ('missing beats wrong') because it
    had no way to thread the substituent's OWN chain numbering -- fixed by
    reusing `_acyclic_alkyl_located_stereo_name` (the same deriver the
    single-centre branch already trusts). Measured: 5 of the 7 bucket rows
    converted to full-InChIKey RT-exact (the other 2 are different root
    causes -- a spiro/fused-ring CIP-numbering mismatch and an oxime E/Z
    omission -- explicitly OUT of this task's scope, see the L3-2a report).
    """

    @pytest.mark.parametrize("smiles", [
        # row 4 of the bucket: triacetate pyranose with a
        # '(2,3,4-trihydroxybutoxy)' alkoxy side chain (2 stereocentres).
        "CCCCCCCCCCCCCCCC(=O)O[C@@H]1[C@H](OC(C)=O)[C@H](OC[C@@H](O)[C@@H](O)CO)"
        "O[C@H](COC(C)=O)[C@H]1OC(C)=O",
        # row 7 of the bucket: tetracyclic diterpene with a
        # '5-(propan-2-yl)heptan-2-yl' branched side chain (2 stereocentres,
        # free valence NOT at locant 1).
        "C=C1[C@@H](O)CC[C@]2(C)C3=C(CC[C@@H]12)[C@]1(O)[C@@H](O)C[C@H]"
        "([C@H](C)CC[C@H](CC)C(C)C)[C@@]1(C)C[C@H]3O",
    ])
    def test_stereo_only_bucket_row_now_rt_exact(self, smiles):
        nm = _best_effort_namer()
        name = nm.name(smiles)
        assert name and not name.startswith("unknown"), (
            f"expected a real emitted name, got {name!r}")
        from orthonym.validation.opsin_roundtrip import opsin_parse
        opsin_smi = opsin_parse(name)
        if opsin_smi is None:
            pytest.skip("OPSIN jar unavailable in this environment")
        assert _full_inchikey(smiles) == _full_inchikey(opsin_smi), (
            f"emitted name {name!r} does not full-RT-match the input "
            "(stereo still omitted/mis-assigned)")


class TestByteIdentityUnderPin:
    """ a phase L3-1 guard: best-effort ONLY -- structurally impossible to
    reach `name_t4_complete` from `_finish` under `--emit-tier pin` (the
    default `Orthonym` constructor)."""

    def test_simple_alcohol_unchanged(self):
        nm = Orthonym()
        assert nm.name("CCO") == "ethanol"
        assert len(nm._offers) == 1

    def test_ibuprofen_unchanged(self):
        nm = Orthonym()
        assert nm.name("CC(C)Cc1ccc(cc1)C(C)C(=O)O") == (
            "2-[4-(2-methylpropyl)phenyl]propanoic acid")
        assert len(nm._offers) == 1

    def test_discard_gap_molecule_uses_single_offer_under_pin_default(self):
        """Under PIN-default only ONE offer ever exists (the floor is never
        appended -- that is `_general_fallback_unverified`-gated), so
        best-effort can never alter the PIN-default output.

         re-baseline: this row's input is stereo-UNDEFINED
        (`CC(O)C(N)...`, no `@`), so the old expected `threonylcysteine` was
        itself a 0-wrong DEFECT -- that retained name implies L-stereo the input
        does not carry, and full-InChIKey RT-mismatches. The primary now emits
        the constitution-correct, stereo-honest systematic name, which full-RT
        MATCHES. Assert the single-offer structural invariant plus that the one
        PIN-default name is the right molecule (not a specific string)."""
        from orthonym.validation.opsin_roundtrip import opsin_parse
        nm_pin = Orthonym()
        smiles = "CC(O)C(N)C(=O)NC(CS)C(=O)O"
        if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
            # the default tier declines it (NO_VERIFIED_PIN); the offer pool is the
            # strict path's, read with the emission rule switched off
            _declined_pin_row(smiles)
            import orthonym.namer as _nm
            _tok = _nm._DEFAULT_TIER_POLICY_OFF.set(True)
            try:
                name = nm_pin.name(smiles)
            finally:
                _nm._DEFAULT_TIER_POLICY_OFF.reset(_tok)
        else:
            name = nm_pin.name(smiles)
        assert name and not name.startswith("unknown"), f"no PIN name: {name!r}"
        assert len(nm_pin._offers) == 1  # floor never appended under PIN-default
        opsin_smi = opsin_parse(name)
        if opsin_smi is None:
            pytest.skip("OPSIN jar unavailable in this environment")
        assert _full_inchikey(smiles) == _full_inchikey(opsin_smi), (
            f"PIN-default name {name!r} does not full-RT-match the input")

    def test_ibuprofen_unchanged_under_best_effort_too(self):
        """A PIN primary that DOES full-RT-pass always wins regardless of
        tier -- best-effort must not touch an already-correct PIN."""
        nm = _best_effort_namer()
        assert nm.name("CC(C)Cc1ccc(cc1)C(C)C(=O)O") == (
            "2-[4-(2-methylpropyl)phenyl]propanoic acid")
        assert len(nm._offers) == 1  # floor never even computed (rt_ok short-circuit)
