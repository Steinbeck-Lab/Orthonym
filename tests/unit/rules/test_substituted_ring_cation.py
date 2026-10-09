""" / — SUBSTITUTED aromatic-N+ ring cations (a phase).

`rules/ions.py::_emit_ring_cumulative_suffix`'s DEMOTE branch (a 0-H ring
cation whose centre cannot be neutralized in place, e.g. an N-substituted
aromatic `C[n+]1ccccc1`) used to fail closed the instant the ring carried ANY
OTHER substituent besides the centre's own -- `_ring_frag_has_substituent`
rejected the severed fragment outright, so
``C[n+]1ccc(-c2ccccc2)cc1`` (target PIN ``1-methyl-4-phenylpyridin-1-ium``)
abstained (``unknown organic compound``) even though the bare-ring case
(``C[n+]1ccccc1`` -> ``1-methylpyridin-1-ium``) and the fused bare case
(``C[n+]1cccc2ccccc21`` -> ``1-methylquinolin-1-ium``) both worked.

Fix: reuse the ALREADY-SHIPPED charged-ring locant/prefix/stem trio
built for ``emit_zwitterion_ring_carboxylate`` (``ions.py:3058``) --
``_charged_ring_locants`` + ``_p74_ring_substituent_prefix`` +
``_p74_bare_ring_stem`` -- instead of the old severed-fragment naming. ONE
numbering, taken over the untouched original mol/ring_system, now covers
BOTH the centre's own exocyclic substituent(s) and any OTHER ring
substituent (phenyl / halo /...), each cited as an ordinary ring-substituent
prefix at its own locant. No new locant assembler was written.

Scope: substituents ``name_substituent``/``_p74_ring_substituent_prefix`` CAN
spell (phenyl, halo, alkyl). An oxime-type ring substituent is
OUT of scope and must abstain rather than emit a wrong name -- verified below
via the real, gate-enforced abstention path (`errors.is_failure_name`), not a
producer-internal guard. (An N-glycoside pyridinium carrying a phosphate group,
the NMN shape, was the second out-of-scope witness; it is now named with its
phosphate kept, see test_nmn_shape_is_named_with_its_phosphate_kept.)
"""

import pytest

from orthonym.errors import is_failure_name
from orthonym.namer import Orthonym

# The whole module needs the REAL OPSIN validity / gate switched on:
# the test suite disables it by default (most tests assert raw generator
# output), but every assertion here -- the RT-verified rows, the regressions,
# and the fail-closed abstentions -- depends on the production gate actually
# running. Without it, the fail-closed rows would see the raw (wrong)
# candidate instead of the abstention.
pytestmark = pytest.mark.opsin_gate


@pytest.fixture(scope="module")
def namer():
    return Orthonym(style="pin")


# ---------------------------------------------------------------------------
# Integration: RT-verified SUBSTITUTED aromatic-N+ ring cations (the fix)
# ---------------------------------------------------------------------------
# Each row was independently verified against OPSIN 2.9.0
# (`orthonym.validation.opsin_roundtrip.opsin_roundtrip_check`): the emitted
# name parses back to the identical structure (InChI match), not merely to
# *a* valid molecule.
@pytest.mark.parametrize("smiles,expected", [
    # BB-shape target: monocyclic pyridinium + an unrelated ring substituent
    # (phenyl) elsewhere on the ring -- the exact defect this task fixes.
    ("C[n+]1ccc(-c2ccccc2)cc1", "1-methyl-4-phenylpyridin-1-ium"),
    # A halogen ring substituent on a monocyclic pyridinium.
    ("C[n+]1ccc(Cl)cc1", "4-chloro-1-methylpyridin-1-ium"),
    # A halogen ring substituent on a FUSED bicyclic (quinolinium) ring
    # system -- exercises the `_ring_iupac_locants` fallback path (a fusion
    # atom has 3 ring-neighbours, so the single-cycle `_charged_ring_locants`
    # walk declines and the fallback numbering must ALSO see the substituent).
    ("C[n+]1cccc2ccc(Br)cc21", "7-bromo-1-methylquinolin-1-ium"),
    # A second fused-ring cation with an orientation-sensitive substituent
    # (methyl at position 6 in the isoquinolinium system) -- raises confidence
    # in the fused-ring fallback path where numbering direction matters.
    ("C[n+]1ccc2cc(C)ccc2c1", "2,6-dimethylisoquinolin-2-ium"),
])
def test_substituted_aromatic_ring_cation_rt_verified(namer, smiles, expected):
    assert namer.name(smiles) == expected


# ---------------------------------------------------------------------------
# Regressions: pre-existing DEMOTE-branch shapes that must NOT change
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,expected", [
    # The bare N-substituted monocycle (no other ring substituent).
    ("C[n+]1ccccc1", "1-methylpyridin-1-ium"),
    # A quaternary ring N+ carrying TWO of its own substituents (no other
    # ring substituent) --, gold-oracle row.
    ("C[N+]1(C)CCCCC1", "1,1-dimethylpiperidin-1-ium"),
    # The bare fused-ring cation (quinolinium) -- proves the
    # `_ring_iupac_locants` fallback path is unchanged for the ALREADY-working
    # unsubstituted-besides-centre fused shape.
    ("C[n+]1cccc2ccccc21", "1-methylquinolin-1-ium"),
    # A zwitterion-ring-carboxylate (a SEPARATE emitter,
    # emit_zwitterion_ring_carboxylate, reached before this DEMOTE branch for
    # any mol carrying BOTH a cation and an anion) -- confirms this fix did
    # not disturb that dispatch or its own substituent handling.
    ("C[n+]1ccc(Br)cc1C(=O)[O-]", "4-bromo-1-methylpyridin-1-ium-2-carboxylate"),
])
def test_regressions_unchanged(namer, smiles, expected):
    assert namer.name(smiles) == expected


# ---------------------------------------------------------------------------
# Fail-closed: shapes OUT of this fix's scope must abstain, never emit wrong
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("smiles", [
    # An oxime-type ring substituent (-CH=N-OH). `name_substituent` cannot
    # spell it as a correct prefix morpheme; the raw candidate
    # ('4-(N-hydroxymethaniminyl)-1-methylpyridin-1-ium') does not parse back
    # through OPSIN, so the gate suppresses it.
    "C[n+]1ccc(C=NO)cc1",
])
def test_out_of_scope_shapes_fail_closed(namer, smiles):
    name = namer.name(smiles)
    assert is_failure_name(name), (
        f"{smiles!r} must abstain (out of scope), not emit a name: {name!r}"
    )


def test_nmn_shape_is_named_with_its_phosphate_kept(namer):
    """The N-glycoside pyridinium carrying a phosphate group (the NMN nucleotide shape)
    used to be the second fail-closed witness above: its raw candidate silently dropped
    the phosphate group, so caught the wrong-molecule mismatch and suppressed
    it. That premise is gone: the name now keeps the phosphate
    ('(phosphonooxy)methyl') and OPSIN 2.9.0 reads it back to the input's full InChIKey,
    so the row moved from the fail-closed list to this positive test (a breadth gain, not
    a loosened check: a name that dropped the phosphate would fail the round trip).
    The molecule is named by substitutive nomenclature over the whole graph; the
    specialised nucleotide retained-name system / is not used."""
    from tests.support.rt_assert import name_is_rt_exact
    smiles = "NC(=O)c1ccc[n+](c1)C1OC(COP(=O)(O)O)C(O)C1O"
    name = namer.name(smiles)
    assert name == "3-carbamoyl-1-{3,4-dihydroxy-5-[(phosphonooxy)methyl]oxolan-2-yl}pyridin-1-ium"
    assert name_is_rt_exact(name, smiles)
