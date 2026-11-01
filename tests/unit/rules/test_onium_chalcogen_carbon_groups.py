"""A chalcogen onium cation with a branched, unsaturated or substituted carbon group.

 "Cationic centers on characteristic groups" (the Blue Book), (:41429) method (2):
a mononuclear chalcogen cation is named on the cationic parent hydride, the groups on the centre being prefixes:
'ethylideneoxidanium (PIN)' (:41463), 'ethyl(propan-2-ylidene)oxidanium (PIN)' (:41465), 'acetyloxidanium (PIN)'
(:41474), '(cyclohexanecarbonyl)di(methyl)oxidanium (PIN)' (:41480). The prefix of a carbon group is its
substituent name, as 'propan-2-yl', '2-methylpropyl', 'tert-butyl', 'prop-2-en-1-yl'; the prefixes are in
the order and with the marks of 'tert-butyldi(methyl)phosphane (PIN)' (:18929).

The producer (``ions.emit_onium_hydride_parent``) named a carbon group by its carbon count ('propyl' for
propan-2-yl, another constitution: the read-back refused the name and the cation had none) and declined a
heteroatom in the group. A carbon group is now named by the shared substituent namer; the count-based name is the
fallback. The book prints no cation with such a group on oxygen; the names are what these rules construct, and OPSIN
reads each back to the full InChIKey (the ``opsin_gate`` marker puts the engine's own gate on).

Changed asserted values, none: every row is new.
"""
import pytest

from orthonym import Orthonym

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


def _tiered(smiles: str):
    row = Orthonym().name_tiered(smiles)
    return row["name"], row["tier"]


@pytest.mark.parametrize("smiles, expected", [
    ("CC(C)[OH+]C", "methyl(propan-2-yl)oxidanium"),
    ("CC(C)[OH+]CC", "ethyl(propan-2-yl)oxidanium"),
    ("CC(C)[OH+]C(C)C", "di(propan-2-yl)oxidanium"),
    ("CC(C)(C)[OH+]C", "tert-butyl(methyl)oxidanium"),
    ("C[OH+]CC(C)C", "methyl(2-methylpropyl)oxidanium"),
    ("C[OH+]C(C)CC", "(butan-2-yl)(methyl)oxidanium"),
    ("C[OH+]CC=C", "methyl(prop-2-en-1-yl)oxidanium"),
    ("C[OH+]CC#C", "methyl(prop-2-yn-1-yl)oxidanium"),
    # the group holds a heteroatom: named whole, with its marks
    ("C[OH+]CCCl", "(2-chloroethyl)(methyl)oxidanium"),
    ("C[OH+]CCO", "(2-hydroxyethyl)(methyl)oxidanium"),
    ("C[OH+]CC(=O)O", "(carboxymethyl)(methyl)oxidanium"),
    ("ClCC[OH+]CCCl", "bis(2-chloroethyl)oxidanium"),
    # the same producer for the heavier chalcogens
    ("CC(C)[SH+]C", "methyl(propan-2-yl)sulfanium"),
    ("CC(C)(C)[S+](C)C", "tert-butyldi(methyl)sulfanium"),
    ("CC(C)[Se+](C)C", "dimethyl(propan-2-yl)selanium"),
])
def test_a_branched_or_substituted_carbon_group_on_the_chalcogen_cation(smiles, expected):
    assert _tiered(smiles) == (expected, "pin_verified")


@pytest.mark.parametrize("smiles, expected", [
    ("C[OH+]C", "dimethyloxidanium"),
    ("C[OH+]CCC", "methyl(propyl)oxidanium"),
    ("C[OH+]C1CCCCC1", "cyclohexyl(methyl)oxidanium"),
    ("c1ccccc1[OH+]C", "methyl(phenyl)oxidanium"),
    ("C[OH+]Cc1ccccc1", "benzyl(methyl)oxidanium"),
    ("CC[O+]=C(C)C", "ethyl(propan-2-ylidene)oxidanium"),         #:41465
    ("C[O+](C)C(=O)C1CCCCC1", "(cyclohexanecarbonyl)di(methyl)oxidanium"),   #:41480
])
def test_the_controls_keep_their_names(smiles, expected):
    assert _tiered(smiles) == (expected, "pin_verified")
