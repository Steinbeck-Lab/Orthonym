"""A ring cation whose neutral ring is a hydro derivative of a mancude heteromonocycle is named
in ONE numbering: the '-ium' at the lowest locant, then the hydro prefixes, then the substituents.

The six-membered cyclic amidinium (2-methyl-3,4,5,6-tetrahydropyrimidin-1-ium, the cation of a
2-methyl-1,4,5,6-tetrahydropyrimidine) had no name at the default tier. The producer named the
neutral ring in the numbering of the neutral ring ('2-methyl-1,4,5,6-tetrahydropyrimidine', hydro
locants lowest) and cited the '-ium' from the numbering of the charged ring ('-3-ium'): two
numberings in one name, with the '-ium' at 3 where 1 is available. The check of the default
tier (``validation/spelling``, "suffix/free valence [3] > [1]") withdrew it.

 "General rule for systematically naming cationic centers in parent hydrides"
(the Blue Book),:41368: "A cation derived formally by adding one or more hydrons to any
position of a neutral parent hydride... or whose degree of hydrogenation has been modified (see
 is named by replacing the final letter 'e' of the parent hydride name, if any, by the suffix
'ium'". The '-ium' is a suffix of the name, so "NUMBERING" (:3219) ranks it by (c)
"principal characteristic groups and free valences (suffixes)" (:3256), ahead of (e) "(i) low
locants are given to hydro/dehydro prefixes" (:3288-:3289); the example of (c) puts the suffix on
the lowest locant before the double bond ('cyclohex-2-en-1-amine (PIN)',:3266). Indicated
hydrogen (b) (:3246) is ranked before the suffix, which is why '4,5-dihydro-1H-imidazol-3-ium'
keeps its '-3-ium' ('1H-imidazol-3-ium (PIN)',:41396); a hydro prefix cited after the
alphabetized prefixes stands directly before the parent: '2,2-dimethyl-1,3-dioxo-2,3-dihydro-1H-
isoindol-2-ium (PIN)' (:41447).

Every name is read back by a FRESH OPSIN call and must give the input's key and fixed-H InChI
(the hydron stays on the atom the input draws it on).
"""
import pytest

from tests.support.jars import jar_or_skip

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


def _fixed_h(smi):
    from rdkit import Chem
    from rdkit.Chem import inchi
    return inchi.MolToInchi(Chem.MolFromSmiles(smi), options="/FixedH /SNon")


def _key(smi):
    from rdkit import Chem
    return Chem.MolToInchiKey(Chem.MolFromSmiles(smi))


def _exact(name, smiles):
    from tests.support.rt_assert import _independent_parse
    parsed = _independent_parse(name, smiles)
    return bool(parsed) and _key(parsed) == _key(smiles) and _fixed_h(parsed) == _fixed_h(smiles)


@pytest.mark.parametrize("smiles,name", [
    # the cyclic amidinium of a 1,4,5,6-tetrahydropyrimidine, every drawing of one species
    ("CC1=[NH+]CCCN1", "2-methyl-3,4,5,6-tetrahydropyrimidin-1-ium"),
    ("[NH+]1=C(C)NCCC1", "2-methyl-3,4,5,6-tetrahydropyrimidin-1-ium"),
    ("C1CNC(C)=[NH+]C1", "2-methyl-3,4,5,6-tetrahydropyrimidin-1-ium"),
    ("C1=[NH+]CCCN1", "3,4,5,6-tetrahydropyrimidin-1-ium"),
    ("CCCC1=[NH+]CCCN1", "2-propyl-3,4,5,6-tetrahydropyrimidin-1-ium"),
    ("c1ccccc1C1=[NH+]CCCN1", "2-phenyl-3,4,5,6-tetrahydropyrimidin-1-ium"),
    # a substituent on the neutral nitrogen is cited at the nitrogen's own locant, 3
    ("CN1CCC[NH+]=C1", "3-methyl-3,4,5,6-tetrahydropyrimidin-1-ium"),
    # the substituent beside the neutral nitrogen (4) or beside the protonated one (6): two species
    ("CC1CC[NH+]=CN1", "4-methyl-3,4,5,6-tetrahydropyrimidin-1-ium"),
    ("CC1CCN=C[NH2+]1", "6-methyl-1,4,5,6-tetrahydropyrimidin-1-ium"),
    # no indicated hydrogen in the parent: the '-ium' decides the numbering, then the hydro locants
    ("C1C[NH+]=CCN1", "2,3,4,5-tetrahydropyrazin-1-ium"),
    # the charge drawn on the substituted nitrogen: one cation, the hydron on the other nitrogen
    ("CC1=[N+](C)CCCN1", "2,3-dimethyl-3,4,5,6-tetrahydropyrimidin-1-ium"),
    ("CC1=[NH+]CCCN1C", "2,3-dimethyl-3,4,5,6-tetrahydropyrimidin-1-ium"),
    # "NAMING OF STEREOISOMERS" (:44639),:44643: the descriptor is cited in the locants of
    # the numbering the name carries
    ("C[C@H]1CC[NH+]=CN1", "(4S)-4-methyl-3,4,5,6-tetrahydropyrimidin-1-ium"),
    ("C[C@@H]1CC[NH+]=CN1", "(4R)-4-methyl-3,4,5,6-tetrahydropyrimidin-1-ium"),
])
def test_ring_cation_is_named_in_one_numbering_with_the_ium_lowest(smiles, name):
    jar_or_skip()
    from orthonym import Orthonym
    r = Orthonym(style="pin").name_tiered(smiles)
    assert r["name"] == name and r["tier"] == "pin_verified", r
    assert _exact(name, smiles)


@pytest.mark.parametrize("smiles,name", [
    # indicated hydrogen (b) is ranked before the suffix (c): the '-ium' stays where it was
    ("CC1=[NH+]CCN1", "2-methyl-4,5-dihydro-1H-imidazol-3-ium"),
    ("CC1=[NH+]CCCCN1", "2-methyl-4,5,6,7-tetrahydro-1H-1,3-diazepin-3-ium"),
    ("C1CC=[NH+]C1", "3,4-dihydro-2H-pyrrol-1-ium"),
    ("C1CCC=[NH+]C1", "2,3,4,5-tetrahydropyridin-1-ium"),
    ("C1=CCC[NH2+]C1", "1,2,3,6-tetrahydropyridin-1-ium"),
    ("C1=NCCC[NH+]1C", "1-methyl-1,4,5,6-tetrahydropyrimidin-1-ium"),
    ("C1C[NH+]=COC1", "5,6-dihydro-4H-1,3-oxazin-3-ium"),
    ("CC1=[NH+]CCCS1", "2-methyl-5,6-dihydro-4H-1,3-thiazin-3-ium"),
    # a saturated ring has no hydro prefix: the producer of the neutral ring stands
    ("C1CC[NH2+]CC1", "piperidin-1-ium"),
])
def test_names_that_were_right_do_not_change(smiles, name):
    jar_or_skip()
    from orthonym import Orthonym
    r = Orthonym(style="pin").name_tiered(smiles)
    assert r["name"] == name and r["tier"] == "pin_verified", r
    assert _exact(name, smiles)


# ---- one species, every drawing: the stereodescriptors take part in the choice of the numbering ----
#
# The charge forms of an amidinium (the hydron on the nitrogen near or far from a ring stereocentre)
# are one species; each is named and the form with the lower locants stands (``Orthonym.
# _name_with_form_choice``). "NUMBERING" (the Blue Book), (f) "detachable alphabetized
# prefixes, all considered together in a series of increasing numerical order" (:3301) makes
# '4-methyl' the name of the species whichever drawing is given, and (j) (:3346) "When there is a
# choice for lower locants related to the presence of stereogenic centers or stereoisomers, the
# lower locant is assigned to CIP stereodescriptors Z, R, M, and r (pseudoasymmetry) that are
# preferred to E, S, P, and s" makes (4R,6S) the name of the cis-2,4,6-trimethyl cation, not
# (4S,6R). The leading descriptor group used to stay in the skeleton the comparison matches, so
# '(4S)-4-methyl-' and '(6S)-6-methyl-' were "not one skeleton" and the drawn form was kept.
@pytest.mark.parametrize("smiles,name", [
    # (f): the lower locant of the methyl, whichever nitrogen the hydron is drawn on
    ("C[C@H]1CCNC=[NH+]1", "(4S)-4-methyl-3,4,5,6-tetrahydropyrimidin-1-ium"),
    ("C[C@@H]1CCNC(C)=[NH+]1", "(4R)-2,4-dimethyl-3,4,5,6-tetrahydropyrimidin-1-ium"),
    ("CC1CCNC=[NH+]1", "4-methyl-3,4,5,6-tetrahydropyrimidin-1-ium"),
    ("CC1CCNC(C)=[NH+]1", "2,4-dimethyl-3,4,5,6-tetrahydropyrimidin-1-ium"),
    # the five-membered ring, where the same choice was already made on the base
    ("C[C@H]1C[NH+]=CN1", "(4S)-4-methyl-4,5-dihydro-1H-imidazol-3-ium"),
    ("C[C@H]1CNC=[NH+]1", "(4S)-4-methyl-4,5-dihydro-1H-imidazol-3-ium"),
    # (j): the cis (meso) cation has one name, R on the lower locant; the trans one is chiral and
    # keeps its own descriptors
    ("C[C@H]1C[C@@H](C)[NH+]=C(C)N1", "(4R,6S)-2,4,6-trimethyl-3,4,5,6-tetrahydropyrimidin-1-ium"),
    ("C[C@H]1C[C@@H](C)NC(C)=[NH+]1", "(4R,6S)-2,4,6-trimethyl-3,4,5,6-tetrahydropyrimidin-1-ium"),
    ("C[C@H]1C[C@H](C)[NH+]=C(C)N1", "(4S,6S)-2,4,6-trimethyl-3,4,5,6-tetrahydropyrimidin-1-ium"),
    ("C[C@@H]1C[C@@H](C)NC(C)=[NH+]1", "(4R,6R)-2,4,6-trimethyl-3,4,5,6-tetrahydropyrimidin-1-ium"),
    # one centre drawn, the other left open: (j) takes the lower locant for the descriptor
    ("CC1C[C@@H](C)NC(C)=[NH+]1", "(4R)-2,4,6-trimethyl-3,4,5,6-tetrahydropyrimidin-1-ium"),
])
def test_a_stereo_drawing_gets_the_name_of_the_species_in_the_lower_numbering(smiles, name):
    jar_or_skip()
    from orthonym import Orthonym, namer
    drawings = [smiles] + [f[0] for f in namer._charge_forms(smiles) or ()]
    for drawing in drawings:
        r = Orthonym(style="pin").name_tiered(drawing)
        assert r["name"] == name and r["tier"] == "pin_verified", (drawing, r)
    assert _exact(name, smiles)
