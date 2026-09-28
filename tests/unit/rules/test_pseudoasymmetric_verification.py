"""Pseudoasymmetric (lowercase r/s) names: the stripped-form verifier, the tiers, and
numbering by (j) (pre-existing-failures plan, Task 4 continuation;
controller ruling on germacrane, 2026-09-25).

OPSIN 2.9.0 parses NO name with a lowercase pseudoasymmetric CIP descriptor, so
such a name can never round-trip. The ruling:
- the PIN tier keeps 'germacrane', the gold-validated np_stereoparent name it
  ships by design (gold row P14C-, "(c)");
- the best-effort tier does not keep 'germacrane'; its systematic name
  '(1R,4s,7S)-1,7-dimethyl-4-(propan-2-yl)cyclodecane' cannot pass the OPSIN
  round trip, so best-effort abstains (claims conformance, 2026-09-27: a name
  OPSIN rejects is never emitted at best-effort). The stripped-form verifier
  below (stereo-stripped OPSIN round trip plus the centres labeller) is kept for
  the adduct components (rules/adducts._name_component).

Numbering: (j) (the Blue Book): "the lower locant is assigned to
CIP stereodescriptors Z, R, M, and r (pseudoasymmetry) that are preferred to E, S,
P, and s". The Blue Book's own analogue is "13-norgermacrane (1R,4s,7S)-4-ethyl-
1,7-dimethylcyclodecane" (:51471). The cycloalkane orientation used to leave this
tie to the input atom order, giving '(1S,4s,7R)'.
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym, name_compound
from orthonym.cli import _emit_tier_flags
from orthonym.namer import (
    _drop_pseudoasymmetric_descriptors,
    _pseudoasymmetric_name_verified,
)

pytestmark = pytest.mark.unit

GERMACRANE = "CC(C)[C@@H]1CC[C@H](C)CCC[C@H](C)CC1"
GERMACRANE_SYSTEMATIC = "(1R,4s,7S)-1,7-dimethyl-4-(propan-2-yl)cyclodecane"


@pytest.mark.parametrize("name,reduced,codes", [
    (GERMACRANE_SYSTEMATIC, "(1R,7S)-1,7-dimethyl-4-(propan-2-yl)cyclodecane", ["s"]),
    ("(1s,4s)-cyclohexane-1,4-diol", "cyclohexane-1,4-diol", ["s", "s"]),
    ("2-(propan-2-yl)butane", "2-(propan-2-yl)butane", []),
    ("(2R,3S)-butane-2,3-diol", "(2R,3S)-butane-2,3-diol", []),
])
def test_drop_pseudoasymmetric_descriptors(name, reduced, codes):
    assert _drop_pseudoasymmetric_descriptors(name) == (reduced, codes)


def test_verifier_accepts_the_correct_pseudoasymmetric_name():
    assert _pseudoasymmetric_name_verified(GERMACRANE_SYSTEMATIC, GERMACRANE)


@pytest.mark.parametrize("name", [
    # the wrong pseudoasymmetric code
    "(1R,4r,7S)-1,7-dimethyl-4-(propan-2-yl)cyclodecane",
    # a wrong true stereocentre (OPSIN part of the check)
    "(1R,4s,7R)-1,7-dimethyl-4-(propan-2-yl)cyclodecane",
    # no pseudoasymmetric descriptor at all: not this check's business
    "1,7-dimethyl-4-(propan-2-yl)cyclodecane",
])
def test_verifier_rejects(name):
    assert not _pseudoasymmetric_name_verified(name, GERMACRANE)


# fix a performance pass (wp2): the verifier compared only the MULTISET of r/s codes, so a
# descriptor at the wrong locant (a CH2, out of range, or a second descriptor on
# a chiral centre) passed. Each removed (locant, code) must now land, in the
# parsed name's own numbering (OPSIN's per-atom locants for the reduced name), on
# a distinct pseudoasymmetric centre of the input with that code.
# (d) (the Blue Book): "Lower case stereodescriptors are used to
# describe pseudoasymmetric stereogenic units"; (:48107).
NORGERMACRANE = "CC[C@@H]1CC[C@H](C)CCC[C@H](C)CC1"   # BB:51471 (1R,4s,7S) isomer
CIS_14_DIOL = "O[C@H]1CC[C@@H](O)CC1"                 # (1s,4s)-cyclohexane-1,4-diol
TRIMETHYL_135 = "C[C@H]1C[C@H](C)C[C@@H](C)C1"        # (1r)-1,3,5-trimethylcyclohexane


@pytest.mark.parametrize("name,smiles", [
    ("(1R,2s,7S)-1,7-dimethyl-4-(propan-2-yl)cyclodecane", GERMACRANE),
    ("(1R,7S,9s)-1,7-dimethyl-4-(propan-2-yl)cyclodecane", GERMACRANE),
    ("(1R,2s,7S)-4-ethyl-1,7-dimethylcyclodecane", NORGERMACRANE),
    ("(1R,7S,9s)-4-ethyl-1,7-dimethylcyclodecane", NORGERMACRANE),
    ("(1R,1s,7S)-4-ethyl-1,7-dimethylcyclodecane", NORGERMACRANE),
    ("(1s,3s)-cyclohexane-1,4-diol", CIS_14_DIOL),
    ("(2r)-1,3,5-trimethylcyclohexane", TRIMETHYL_135),
])
def test_verifier_rejects_a_misplaced_pseudoasymmetric_locant(name, smiles):
    assert not _pseudoasymmetric_name_verified(name, smiles)


@pytest.mark.parametrize("name,smiles", [
    ("(1R,4s,7S)-4-ethyl-1,7-dimethylcyclodecane", NORGERMACRANE),
    ("(1s,4s)-cyclohexane-1,4-diol", CIS_14_DIOL),
    ("(1r)-1,3,5-trimethylcyclohexane", TRIMETHYL_135),
    # a legitimate alternative numbering of the same molecule still verifies
    ("(3r)-1,3,5-trimethylcyclohexane", TRIMETHYL_135),
])
def test_verifier_accepts_a_correctly_placed_pseudoasymmetric_locant(name, smiles):
    assert _pseudoasymmetric_name_verified(name, smiles)


def test_verifier_fails_closed_on_a_descriptor_without_locant():
    assert not _pseudoasymmetric_name_verified(
        "(s)-2,4,6-trimethyl-1,3,5-trioxane", "C[C@H]1O[C@@H](C)O[C@H](C)O1")


def test_verifier_fails_closed_on_mixed_codes():
    # 4-methylcyclohexan-1-ol has pseudoasymmetric centres of ONE code per
    # isomer; a name that claims both codes cannot be mapped without locants.
    assert not _pseudoasymmetric_name_verified(
        "(1r,4s)-4-methylcyclohexan-1-ol", "C[C@H]1CC[C@@H](O)CC1")


@pytest.mark.opsin_gate
def test_pin_tier_keeps_the_gold_np_stereoparent_name():
    assert name_compound(GERMACRANE) == "germacrane"


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", [
    GERMACRANE,
    # the same molecule written from other atoms (RDKit rootedAtAtom, same InChIKey)
    "[C@H]1(C)CC[C@@H](C(C)C)CC[C@@H](C)CCC1",
    "C[C@H]1CCC[C@@H](C)CC[C@@H](C(C)C)CC1",
])
def test_best_effort_abstains_on_the_name_opsin_cannot_read(smiles):
    # Claims conformance (2026-09-27): at best-effort a name OPSIN rejects is not
    # emitted (exact-match list names excepted). OPSIN 2.9.0 returns no structure
    # for GERMACRANE_SYSTEMATIC, so every spelling of the input abstains, with its
    # limit code, as in the code the paper measured.
    be = Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(smiles)
    assert be["name"] is None and be["tier"] == "abstain" and be["limit_code"]
    assert _pseudoasymmetric_name_verified(GERMACRANE_SYSTEMATIC, smiles)
