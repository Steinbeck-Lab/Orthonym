"""`_ORGANIC_ELEMENTS` is NOT the gate that blocks the elements.

 Phase A Task 1 planned to widen `errors._ORGANIC_ELEMENTS` by
{As, Sb, Bi, Ge, Sn, Pb, Te}, on the recorded premise that it is "the single
scope gate consulted by `classify_scope_limit`" and that "9 of 9 target
molecules name correctly the moment the guard is widened".

Both halves were refuted by measurement on 2026-08-04:

1. `classify_scope_limit` (`errors.py:266`) consults only ATOMIC NUMBER 0
   (wildcards). It never reads `_ORGANIC_ELEMENTS`. The set has exactly two
   consumers: `classify_failure_limit` (`errors.py:310`), which runs only AFTER
   naming has already failed and merely chooses the wording of the diagnostic,
   and `assembly/composer.py:8264`, a FAIL-CLOSED guard that refuses a
   substituent branch carrying an unnameable non-organic element.

2. A paired two-arm run over the 90 rows of the Blue Book conformance
   corpus (`benchmarks/bb_conformance/`), one fresh process per arm so no cache
   could serve one arm's answer to the other: **90 of 90 rows changed, and 0
   gained a name.** Every row moved from a specific refusal
   ("arsenic compound (not supported)") to the generic one
   ("unknown organic compound"). Emit was unchanged; the only effect was a less
   informative diagnostic plus the loss of the composer guard, whose own comment
   records that dropping such a branch shipped 'ethane' for CCS[Zn]SCC -- a
   plausible wrong molecule that no failure predicate can flag (a project rule).

   The nine molecules cited as the trace's positives (`[AsH3]`.. `phenylarsonic
   acid`) already name correctly at HEAD with the guard untouched, which is why
   the trace looked green: they were never blocked.

The real blocker for the largest nameable slice of those rows was the
Hantzsch-Widman six-membered-ring stem -- see
`tests/unit/data/test_hw_six_ring_least_senior_stem.py`.

This test exists so the refuted change cannot be re-attempted silently. It is
not a claim that these elements are permanently out of scope: it is a claim that
widening this particular set is not how they get named, and that anything which
does widen it must first show a molecule that gains a CORRECT name by doing so.
"""
import pytest

from orthonym.errors import _ORGANIC_ELEMENTS, classify_scope_limit
from orthonym.namer import Orthonym

from rdkit import Chem

P68 = {"As", "Sb", "Bi", "Ge", "Sn", "Pb", "Te"}


def test_scope_limit_does_not_consult_the_organic_element_set():
    """The documented "single scope gate" is off the path for these elements."""
    for smiles in ("[AsH3]", "c1ccccc1[As](O)(O)=O", "CC[Pb](CC)(CC)CC"):
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, smiles
        assert classify_scope_limit(mol) is None, (
            f"{smiles}: classify_scope_limit refused, so _ORGANIC_ELEMENTS may "
            "have been wired into it -- re-measure before widening the set"
        )


def test_p68_elements_stay_out_of_the_organic_element_set():
    """Widening this set moved 90/90 rows to a WORSE diagnostic and named 0."""
    assert not (P68 & _ORGANIC_ELEMENTS), (
        "Widening _ORGANIC_ELEMENTS by the P-68 elements was measured to name "
        "0 of 90 Blue Book rows while replacing a specific refusal with "
        "'unknown organic compound' and disarming the composer.py:8264 "
        "fail-closed guard. If you are widening it deliberately, first exhibit a "
        "molecule that gains a CORRECT name, and update this test with it."
    )


@pytest.mark.parametrize("smiles,expected", [
    ("[AsH3]", "arsane"),
    ("[SbH3]", "stibane"),
    ("[BiH3]", "bismuthane"),
    ("[GeH4]", "germane"),
    ("[SnH4]", "stannane"),
    ("[PbH4]", "plumbane"),
    ("[TeH2]", "tellane"),
    ("CC[Pb](CC)(CC)CC", "tetraethylplumbane"),
    ("c1ccccc1[As](O)(O)=O", "phenylarsonic acid"),
])
def test_p68_parent_hydrides_already_name_with_the_guard_untouched(
        smiles, expected):
    """These are the nine "trace positives". They pass WITHOUT the widening.

    `rules/mononuclear_hydrides.py` implements the group-14/15/16 parent hydride
    family, so a green result here says nothing about the guard -- which is
    exactly how the refuted premise came to look verified.
    """
    assert Orthonym().name(smiles) == expected


def test_element_refusal_stays_specific_not_generic():
    """The specific wording is the informative half that widening would destroy."""
    name = Orthonym().name("C1CC[Sn]CC1")
    assert name == "tin compound (not supported)", name
