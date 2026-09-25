"""Pseudoasymmetric (lowercase r/s) names: verified at the general tiers, and
numbered by (j) (pre-existing-failures plan, Task 4 continuation;
controller ruling on germacrane, 2026-09-25).

OPSIN 2.9.0 parses NO name with a lowercase pseudoasymmetric CIP descriptor, so
such a name can never round-trip. The ruling:
- the PIN tier keeps 'germacrane', the gold-validated np_stereoparent name it
  ships by design (gold row P14C-, "(c)");
- the best-effort tier gives the VERIFIED systematic name
  '(1R,4s,7S)-1,7-dimethyl-4-(propan-2-yl)cyclodecane': the stereo-stripped form
  by OPSIN round trip (full InChIKey), the descriptors by the centres labeller.

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
def test_best_effort_gives_the_verified_systematic_name(smiles):
    be = Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(smiles)
    assert be["name"] == GERMACRANE_SYSTEMATIC
    assert _pseudoasymmetric_name_verified(be["name"], smiles)
