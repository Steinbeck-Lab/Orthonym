"""A chalcone with ring substituents is named on the retained parent.

 KETONES, "Retained names", (the Blue Book): "The name 'chalcone' is the only
retained name as a preferred IUPAC name and is limited to ring substitution only by characteristic groups lower
than 'ketone'. Chalcone refers only to the trans- or (E)- stereoisomer." Its example (:28303):
"2',4'-dihydroxy-3,3'-dimethoxychalcone (PIN) (2E)-1-(2,4-dihydroxy-3-methoxyphenyl)-3-(3-methoxyphenyl)
prop-2-en-1-one", and a senior group keeps the systematic name (:28305): "4-[(1E)-3-(2,4-dihydroxyphenyl)-3-
oxoprop-1-en-1-yl]benzamide (PIN) (not 2',4'-dihydroxychalcone-4-carboxamide)". The benzoyl ring takes the
primed locants, as the example shows. Only the bare chalcone had a producer (the retained table); the check of
``validation.spelling.checks_chalcone`` already refused the systematic name for these molecules, so the PIN tier
declined them.

The producer names whole characteristic groups (glossary "Characteristic group",:1804): halo, nitro, hydroxy,
sulfanyl, amino, and alkoxy, alkylsulfanyl or alkylamino with a plain alkyl part. A hydrocarbyl group is no
characteristic group, and a group with a ring or a substituted chain (benzyloxy) is more than one: those molecules
keep the systematic name.

Changed asserted values: '4-chloro' and '4-nitro' chalcone move from "not covered" to the retained parent, and the
benzyloxy row moves to "not covered", per the reading of recorded for the maintainers (characteristic
groups by the glossary, not only the groups of the book's example); OPSIN reads each name back. OPSIN reads each name back to the full InChIKey of its
structure (the ``opsin_gate`` marker puts the engine's own gate on).
"""
import pytest
from rdkit import Chem

import orthonym.rules.chalcones as chalcones
from orthonym import Orthonym, name_compound
from orthonym.rules.chalcones import name_substituted_chalcone

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


def _tiered(smiles: str):
    row = Orthonym().name_tiered(smiles)
    return row["name"], row["tier"]


@pytest.mark.parametrize("smiles, expected", [
    # the book's own example (:28303)
    ("COc1cccc(/C=C/C(=O)c2ccc(O)c(OC)c2O)c1", "2',4'-dihydroxy-3,3'-dimethoxychalcone"),
    # the ring on the carbonyl is primed, the ring on C-3 is not
    ("O=C(/C=C/c1ccccc1)c1ccc(O)cc1O", "2',4'-dihydroxychalcone"),
    ("O=C(/C=C/c1ccc(O)cc1)c1ccccc1", "4-hydroxychalcone"),
    ("COc1ccc(/C=C/C(=O)c2c(O)cc(O)cc2O)cc1", "2',4',6'-trihydroxy-4-methoxychalcone"),
    # the lowest locants decide the direction round each ring
    ("O=C(/C=C/c1ccccc1)c1ccc(O)c(OC)c1", "4'-hydroxy-3'-methoxychalcone"),
    ("O=C(/C=C/c1ccccc1)c1c(O)cccc1O", "2',6'-dihydroxychalcone"),
    # a compound prefix takes its marks, and the prefixes are in alphanumerical order
    ("O=C(/C=C/c1ccc(N)cc1)c1ccc(N(C)C)cc1", "4-amino-4'-(dimethylamino)chalcone"),
    ("O=C(/C=C/c1ccccc1)c1ccc(SC)cc1", "4'-(methylsulfanyl)chalcone"),
    # halo and nitro are characteristic groups lower than 'ketone' (glossary:1804; Table 5.1:24980)
    ("O=C(/C=C/c1ccc(Cl)cc1)c1ccccc1", "4-chlorochalcone"),
    ("O=C(/C=C/c1ccc([N+](=O)[O-])cc1)c1ccccc1", "4-nitrochalcone"),
])
def test_the_substituted_chalcone_is_named_on_the_retained_parent(smiles, expected):
    assert _tiered(smiles) == (expected, "pin_verified")


def test_the_bare_chalcone_is_still_the_retained_tables():
    """'chalcone (PIN)' (:28299): the producer leaves it to the table."""
    assert name_substituted_chalcone(Chem.MolFromSmiles("O=C(/C=C/c1ccccc1)c1ccccc1")) is None
    assert Orthonym().name_tiered("O=C(/C=C/c1ccccc1)c1ccccc1")["producer"] == "outer:retained_name"


@pytest.mark.parametrize("smiles", [
    "O=C(/C=C\\c1ccccc1)c1ccc(O)cc1",     # Z: 'Chalcone refers only to the trans- or (E)- stereoisomer'
    "O=C(C=Cc1ccccc1)c1ccc(O)cc1",        # no configuration drawn
    "O=C(/C=C/c1ccc(C(N)=O)cc1)c1ccc(O)cc1O",   # a group senior to the ketone: the systematic name (:28305)
    "O=C(/C=C/c1ccc(C)cc1)c1ccccc1",      # a hydrocarbyl group is no characteristic group (glossary:1804)
    "COc1c(O)c(C(=O)/C=C/c2ccc(OCc3ccccc3)cc2)c(OC)c(OC)c1OC",   # benzyloxy: more than one characteristic group
    "O=C(/C=C/c1ccccc1)c1ccc(O)c2ccccc12",        # a ring that is not a benzene ring of its own
])
def test_what_the_name_does_not_cover_keeps_its_name(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert name_substituted_chalcone(mol) is None
    assert "chalcone" not in name_compound(smiles, on_failure="label")


def test_a_chalcone_with_a_stereocentre_is_declined():
    """The name 'chalcone' implies the one (E) double bond; another stereo element would need a descriptor."""
    mol = Chem.MolFromSmiles("O=C(/C=C/c1ccccc1)c1ccc(O[C@H](C)CC)cc1")
    assert name_substituted_chalcone(mol) is None


def test_the_tests_fail_without_the_class(monkeypatch):
    """The mutation check: with the producer switched off the molecule is declined at the PIN tier again."""
    monkeypatch.setattr(chalcones, "name_substituted_chalcone", lambda mol: None)
    assert _tiered("O=C(/C=C/c1ccccc1)c1ccc(O)cc1O")[1] == "abstain"
