"""The bare ring amidinium cation with an exocyclic amino group: no PIN is claimed where the Blue Book names none (LO5).

'2-amino-4,5-dihydro-1,3-thiazol-3-ium' (``NC1=[NH+]CCS1``, or the drawing ``[NH2+]=C1NCCS1``) is the cation of
4,5-dihydro-1,3-thiazol-2-amine with the hydron on the ring nitrogen. Its other name is
'1,3-thiazolidin-2-iminium' (the hydron on the imine of the tautomer 1,3-thiazolidin-2-imine). The decision, with every
line of ``the Blue Book Blue Book`` it rests on, is in `internal notes` ("Bare
cation"). In short, the Blue Book does not decide it:

* ``### **** General rule for systematically naming cationic centers in parent hydrides`` (:41366),
  :41368 ("A cation derived formally by adding one or more hydrons to any position of a neutral parent hydride...
  or whose degree of hydrogenation has been modified (see is named by replacing the final letter 'e' of the
  parent hydride name, if any, by the suffix 'ium'"; '1*H*-imidazol-3-ium (PIN)':41396) names the ring cation;
* ``## **** Cationic centers on characteristic groups`` (:41413),:41415 ("the largest neutral parent
  possible is used. It is applied particularly in the case of neutral compounds expressed by suffixes containing
  nitrogen", Table 7.4: imine -> iminium,:41425) and ``## **** CHOICE OF A PARENT STRUCTURE`` (:42330),
  criterion (d) (:42362, "giving priority to the corresponding suffixes (see Table 4.4) and by using the general
  seniority order of classes (see and parent structures (see ") name the iminium;
* (a) (:42334), the cationic centres of the parent, is a tie (1 and 1), (b) and (c) are ties; (d) reads two
  ways (imines are class 20 and N-heterocycles class 21 in Table 4.1,:18193 and:18197, but amines are class 19,
  :18192, and rank 51 against 52 for the imines in Table 4.4,:18845-:18846), and none of its examples (:42366,
  :42371) sets a skeletal 'ium' against a cationic suffix; the book's one case of resonance forms of an N-C-N cation
  (:41459-:41461, 'N,N-dimethylguanidinium (PIN)') is settled by locants, which these two names do not have;
* no (PIN) example of a cyclic amidinium with an exocyclic nitrogen exists, every iminium PIN is acyclic (:26697,
  :26700,:43157), and 'imidamide' is an acyclic suffix (:34200).

So the PIN of the bare cation is ASSUMED neither way, and the label is not claimed: the default tier (a verified PIN or a
decline) declines it, and the best-effort tier ships the ring-'ium' name as ``systematic_verified`` (a name that
OPSIN reads back to the full InChIKey). The zwitterion with a carboxylate on the ring is decided by the book
 (a): the iminium form has no cationic parent there) and is pinned in
``test_ring_zwitterion_hydro_numbering.py``.

These rows are pinned so that the rows are not promoted by accident. ``ions._ring_bears_principal_group`` reads the
neutral ring '4,5-dihydro-1,3-thiazol-2-amine' as carrying an 'amidine' principal group (its exemption asks all three
atoms of the match to be skeletal) and so the ring-cation producer declines it, as it declines '4-aminopyridin-1-ium'
and '4-hydroxypyridin-1-ium' (the ring cation with a neutral principal group, / territory, of
``_emit_ring_cumulative_suffix``). The oxazoline, imidazoline and 3-substituted analogues are not declined, because
the functional-group perception reads their rings as a carbamimidate, a guanidine or a 0-H cation: they ship
``pin_verified`` today on the same open question (measured and flagged in `internal notes`,
LO5); they are not pinned here. If the decision is ever taken for the ring 'ium', these expectations flip together,
by one condition in that guard (measured there).

Every name is read back by a FRESH OPSIN call and must give the input's full InChIKey and fixed-H InChI.
"""
import pytest

from tests.support.jars import jar_or_skip
from tests.support.read_back import same_hydron_sites

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

# (SMILES drawings of one species, the best-effort name)
ROWS = [
    (("NC1=[NH+]CCS1", "[NH2+]=C1NCCS1", "C1CSC(N)=[NH+]1"),
     "2-amino-4,5-dihydro-1,3-thiazol-3-ium"),
    (("NC1=[NH+]C(C)CS1", "[NH2+]=C1NC(C)CS1"),
     "2-amino-4-methyl-4,5-dihydro-1,3-thiazol-3-ium"),
    (("NC1=[NH+]CCCS1", "[NH2+]=C1NCCCS1"),
     "2-amino-5,6-dihydro-4H-1,3-thiazin-3-ium"),
    (("NC1=[NH+]CCCCS1",),
     "2-amino-4,5,6,7-tetrahydro-1,3-thiazepin-3-ium"),
    (("NC1=[NH+]CC[Se]1",),
     "2-amino-4,5-dihydro-1,3-selenazol-3-ium"),
    (("NC1=[NH+]CCC1", "[NH2+]=C1NCCC1"),
     "5-amino-3,4-dihydro-2H-pyrrol-1-ium"),
    (("NC1=[NH+]CCSC1",),
     "5-amino-3,6-dihydro-2H-1,4-thiazin-4-ium"),
    (("CNC1=[NH+]CCS1",),
     "2-(methylamino)-4,5-dihydro-1,3-thiazol-3-ium"),
]


@pytest.mark.parametrize("drawings,name", ROWS, ids=[n for _, n in ROWS])
def test_the_bare_ring_amidinium_is_not_claimed_as_a_pin(drawings, name):
    """Default tier: a decline (the Blue Book gives no PIN, LO5). Best-effort: the ring-'ium' name, round-trip verified."""
    jar_or_skip()
    from orthonym import Orthonym
    from orthonym.cli import _emit_tier_flags
    default = Orthonym(style="pin")
    best_effort = Orthonym(style="pin", **_emit_tier_flags("best-effort"))
    for smiles in drawings:
        r = default.name_tiered(smiles)
        assert r["tier"] == "abstain" and r["name"] != name, (smiles, r)
        r = best_effort.name_tiered(smiles)
        assert (r["name"], r["tier"]) == (name, "systematic_verified"), (smiles, r)
    assert same_hydron_sites(name, drawings[0]), name
