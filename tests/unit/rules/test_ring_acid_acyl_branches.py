"""Acyl branches on a ring acid at the PIN tier: aroyl, (aryl)carbamoyl and (aryloxy)carbonyl groups.

 "Acyl groups" (the Blue Book-17768): "acetyl (preferred prefix) (full substitution allowed except
that the carbon chain cannot be extended...)", "benzoyl (preferred prefix) (full substitution, see ";
 (:32922) names -CO-NHR on a ring '(R)carbamoyl' (method (1)).

Three defects, one each per producer:

* a cycloalkane acid: the ring chokepoint names an aromatic ring only with its breadth switch, so the branch got no
  name at the PIN tier and a carbon-count fallback named a wrong molecule ('4-heptylcyclohexane-1-carboxylic acid'
  for the phenylcarbamoyl acid) the read-back withdrew. The strict producer (``strict_ring_prefix.
  strict_acyl_branch_name``) now names it in the promotion re-run;
* the ketone of an aroyl branch was cited twice ('2-benzoyl-2-oxocyclohexane-1-carboxylic acid');
* a benzene acid named only plain 'benzoyl' and alkanoyl branches ('4-[3-(trifluoromethyl)benzoyl]benzoic acid'
  declined); the book's acyl writer now names the others.

A branch that holds the principal characteristic group is not named this way: the multiplicative name is the PIN
('4,4'-carbonyldibenzoic acid (PIN; multiplicative name)', not '4-(4-carboxybenzoyl)benzoic acid',:29473).
"""
import pytest

from orthonym.cli import _tiered_namer
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


def _row(smiles):
    return _tiered_namer("pin", "pin").name_tiered(smiles)


@pytest.mark.parametrize("smiles,name", [
    ("OC(=O)C1CCC(CC1)C(=O)Nc1cccc(C(F)(F)F)c1",
     "4-{[3-(trifluoromethyl)phenyl]carbamoyl}cyclohexane-1-carboxylic acid"),
    ("OC(=O)C1CCC(CC1)C(=O)Nc1ccccn1", "4-[(pyridin-2-yl)carbamoyl]cyclohexane-1-carboxylic acid"),
    ("OC(=O)C1CCC(CC1)C(=O)Oc1ccccc1", "4-(phenoxycarbonyl)cyclohexane-1-carboxylic acid"),
    ("OC(=O)C1CCCCC1C(=O)c1ccccc1", "2-benzoylcyclohexane-1-carboxylic acid"),
    ("OC(=O)C1CCC(CC1)C(=O)c1cccc(C(F)(F)F)c1", "4-[3-(trifluoromethyl)benzoyl]cyclohexane-1-carboxylic acid"),
    ("OC(=O)C1CCC(CC1)C(=O)c1cccnc1", "4-(pyridine-3-carbonyl)cyclohexane-1-carboxylic acid"),
    ("OC(=O)C1CCCCC1C(=O)Cc1ccccc1", "2-(phenylacetyl)cyclohexane-1-carboxylic acid"),
    ("OC(=O)c1ccc(cc1)C(=O)c1cccc(C(F)(F)F)c1", "4-[3-(trifluoromethyl)benzoyl]benzoic acid"),
    ("OC(=O)c1ccc(cc1)C(=O)c1ccc2ccccc2c1", "4-(naphthalene-2-carbonyl)benzoic acid"),
    ("NC(=O)c1ccc(cc1)C(=O)c1ccc(Cl)cc1Cl", "4-(2,4-dichlorobenzoyl)benzamide"),
    ("N#Cc1ccc(cc1)C(=O)C(C)C", "4-(2-methylpropanoyl)benzonitrile"),
    # ring N: '<ring>-1-carbonyl' (was the false PIN '[(2,3-dihydro-1H-indol-1-yl)methanonyl]')
    ("OC(=O)C1CCC(CC1)C(=O)N1CCc2ccccc12", "4-(2,3-dihydro-1H-indole-1-carbonyl)cyclohexane-1-carboxylic acid"),
    ("OC(=O)C1CCC(CC1)C(=O)N1CCCCCC1", "4-(azepane-1-carbonyl)cyclohexane-1-carboxylic acid"),
    ("O=C1CCCCC1C(=O)c1ccccc1", "2-benzoylcyclohexan-1-one"),
    # (the Blue Book): '1-oxo-2-phenylpropyl' names are "for use in general nomenclature"
    ("OC(=O)C1CCC(CC1)C(=O)[C@@H](C)c1ccccc1", "4-[(2S)-2-phenylpropanoyl]cyclohexane-1-carboxylic acid"),
    ("OC(=O)C1CCC(CC1)C(=O)Cc1ccc(Cl)cc1", "4-[(4-chlorophenyl)acetyl]cyclohexane-1-carboxylic acid"),
])
def test_an_acyl_branch_on_a_ring_acid_takes_its_acyl_prefix(smiles, name):
    row = _row(smiles)
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles,name", [
    ("OC(=O)c1ccc(cc1)C(=O)c1ccc(cc1)C(=O)O", "4,4'-carbonyldibenzoic acid"),
    ("NC(=O)c1ccc(cc1)C(=O)c1ccc(cc1)C(N)=O", "4,4'-carbonyldibenzamide"),
    ("CC(=O)c1ccccc1", "1-phenylethan-1-one"),
    ("O=C(c1ccccc1)c1ccccc1", "diphenylmethanone"),
])
def test_names_the_acyl_branch_does_not_take(smiles, name):
    assert _row(smiles)["name"] == name


def test_a_branch_with_the_principal_group_gets_no_substitutive_name():
    """3,4'-carbonyldibenzoic acid takes a multiplicative name, which is not built here: no
    '...-(carboxybenzoyl)benzoic acid' at the PIN tier."""
    name = _row("OC(=O)c1cccc(c1)C(=O)c1ccc(cc1)C(=O)O")["name"]
    assert "carboxybenzoyl" not in name

