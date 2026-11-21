"""Sulfur acid groups on a polycyclic hydrocarbon parent (rules.pah_sulfur_groups).

 'Substitutive nomenclature, suffix mode, for sulfonic, sulfinic, etc., acids' (the Blue Book) and
 sulfonamides (:32744) give the suffixes; under a senior group the prefixes 'sulfamoyl*' (Appendix 2,
:55489) and 'chlorosulfonyl (preselected prefix)',:36472) are used, and amides rank "in the order of
the corresponding acids",:18184). The polycyclic producer knew only '-sulfonic acid', so
'naphthalene-2-sulfonamide' fell to a von Baeyer name.
"""
import pytest

from tests.support.pin_tiers import tier_row
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles,name", [
    ("NS(=O)(=O)c1ccc2ccccc2c1", "naphthalene-2-sulfonamide"),
    ("NS(=O)(=O)c1ccc2cc(S(N)(=O)=O)ccc2c1", "naphthalene-2,6-disulfonamide"),
    ("NS(=O)(=O)c1cccc2c(S(N)(=O)=O)cccc12", "naphthalene-1,5-disulfonamide"),
    ("NS(=O)(=O)c1ccc2cc3ccccc3cc2c1", "anthracene-2-sulfonamide"),
    ("OS(=O)c1ccc2ccccc2c1", "naphthalene-2-sulfinic acid"),
    ("NS(=O)(=O)c1ccc2cc(C#N)ccc2c1", "6-cyanonaphthalene-2-sulfonamide"),
    ("NS(=O)(=O)c1ccc2ccccc2c1O", "1-hydroxynaphthalene-2-sulfonamide"),
    # N-substituted: the italic-N groups of the one sulfonamide suffix
    ("CNS(=O)(=O)c1ccc2ccccc2c1", "N-methylnaphthalene-2-sulfonamide"),
    ("CN(C)S(=O)(=O)c1ccc2ccccc2c1", "N,N-dimethylnaphthalene-2-sulfonamide"),
    ("O=S(=O)(Nc1ccccc1)c1ccc2ccccc2c1", "N-phenylnaphthalene-2-sulfonamide"),
    ("O=S(=O)(NCc1ccccc1)c1cccc2ccccc12", "N-benzylnaphthalene-1-sulfonamide"),
    ("CCN(CC)S(=O)(=O)c1ccc2ccccc2c1O", "N,N-diethyl-1-hydroxynaphthalene-2-sulfonamide"),
    ("CNS(=O)(=O)c1ccc2cc(C(=O)O)ccc2c1", "6-(methylsulfamoyl)naphthalene-2-carboxylic acid"),
    # a senior group takes the suffix and the sulfur group is a prefix
    ("NS(=O)(=O)c1ccc2ccccc2c1C(=O)O", "2-sulfamoylnaphthalene-1-carboxylic acid"),
    ("NC(=O)c1ccc2cc(S(N)(=O)=O)ccc2c1", "6-sulfamoylnaphthalene-2-carboxamide"),
    ("O=S(=O)(Cl)c1ccc2ccc(S(=O)(=O)Cl)cc2c1C(=O)O", "2,7-bis(chlorosulfonyl)naphthalene-1-carboxylic acid"),
    # unchanged
    ("OS(=O)(=O)c1ccc2cc(S(=O)(=O)O)ccc2c1", "naphthalene-2,6-disulfonic acid"),
    ("ClS(=O)(=O)c1ccc2ccccc2c1", "naphthalene-2-sulfonyl chloride"),
])
def test_polycyclic_sulfur_group_pins(smiles, name):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)
