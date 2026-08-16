"""v33 Phase 0 Task L3-1: the systematic floor as a surviving, full-RT-gated
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
        # The winning offer must be the T4 floor, not the stereo-fabricating
        # primary (proves the RANKING actually picked the floor, not a
        # coincidental match).
        winners = [o for o in nm._offers if o.name == name]
        assert winners and winners[0].source == "t4_floor", (
            f"expected the t4_floor offer to win; offers={nm._offers}")


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
        assert nm._t4_floor_candidate == name, (
            "the winning name should be exactly the stashed floor candidate")


class TestStereoOnlyFloorCompletenessL3_2a:
    """v33 Phase 0 L3-2a: the STEREO_ONLY 7-row bucket from the L3-2/L3-3 SPY
    (`FLOOR_COVERS_RT_FAIL` -- both the primary AND the floor omitted real
    defined stereo on a decorated acyclic side-chain substituent, e.g.
    '1,2,3-trihydroxypropyl'/'5-(propan-2-yl)heptan-2-yl' with NO leading
    descriptor block, even though the constitution was already right).

    Root cause: `substituent_naming._add_substituent_stereo`'s multi-centre
    branch unconditionally declined (D-09 'missing beats wrong') because it
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
    """v33 Phase 0 L3-1 guard: best-effort ONLY -- structurally impossible to
    reach `name_t4_complete` from `_finish` under `--emit-tier pin` (the
    default `Orthonym()` constructor)."""

    def test_simple_alcohol_unchanged(self):
        nm = Orthonym()
        assert nm.name("CCO") == "ethanol"
        assert len(nm._offers) == 1

    def test_ibuprofen_unchanged(self):
        nm = Orthonym()
        assert nm.name("CC(C)Cc1ccc(cc1)C(C)C(=O)O") == (
            "2-[4-(2-methylpropyl)phenyl]propanoic acid")
        assert len(nm._offers) == 1

    def test_discard_gap_molecule_unchanged_under_pin_default(self):
        """The SAME molecule that flips under best-effort must NOT flip
        under PIN-default -- only 1 offer ever exists there, so
        `_select_rt_passing_offer_name` falls back to the current (wrong)
        name rather than newly abstain -- L4-core's byte-identity guarantee."""
        nm_pin = Orthonym()
        smiles = "CC(O)C(N)C(=O)NC(CS)C(=O)O"
        name = nm_pin.name(smiles)
        assert name == "threonylcysteine"
        assert len(nm_pin._offers) == 1
        assert nm_pin._offers[0].name == "threonylcysteine"

    def test_ibuprofen_unchanged_under_best_effort_too(self):
        """A PIN primary that DOES full-RT-pass always wins regardless of
        tier -- best-effort must not touch an already-correct PIN."""
        nm = _best_effort_namer()
        assert nm.name("CC(C)Cc1ccc(cc1)C(C)C(=O)O") == (
            "2-[4-(2-methylpropyl)phenyl]propanoic acid")
        assert len(nm._offers) == 1  # floor never even computed (rt_ok short-circuit)
