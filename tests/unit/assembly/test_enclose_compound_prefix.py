"""A compound substituent prefix is enclosed even when it carries no locant.

 (the Blue Book): "Parentheses are used around compound (see and complex
(see prefixes"; '(silylamino)silyl' (:26322), '4-(disilylamino)cyclohexane-1-
carbonitrile (PIN)' (:38198), '(methylcarbamoyl)' in the Blue Book. The enclosure predicates keyed
on a locant, a hyphen or an inner mark, so 'silylamino', 'cyanoamino' and 'chlorophenyl' were cited
bare while is_substituted_substituent (the multiplier predicate) already read them as a substituent
prefix in front of a parent prefix ('4-silylaminobenzoic acid'). The contracted amide prefixes
stay bare ('2-acetamido...', '4-benzamido...' in the Blue Book) although they take 'bis'.
"""
import pytest

from orthonym.assembly.naming_utils import enclose_if_compound, format_substituent_prefix
from tests.support.pin_tiers import name_breadth
from tests.support.rt_assert import name_is_rt_exact


@pytest.mark.parametrize("name", ["silylamino", "cyanoamino", "diazenylamino", "chlorophenyl",
                                  "methylcarbamoyl", "nitrosooxy"])
def test_compound_prefix_is_enclosed(name):
    assert enclose_if_compound(name) == f"({name})"
    assert format_substituent_prefix(name, [4], 1) == f"4-({name})"


@pytest.mark.parametrize("name", ["methyl", "methoxy", "phenyl", "benzyl", "acetyl", "carbamoyl",
                                  "acetamido", "benzamido", "oxo", "tert-butyl"])
def test_simple_prefix_stays_bare(name):
    assert enclose_if_compound(name) == name
    assert format_substituent_prefix(name, [4], 1) == f"4-{name}"


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", [
    ("OC(=O)c1ccc(N[SiH3])cc1", "4-(silylamino)benzoic acid"),
    ("Oc1ccc(N[SiH3])cc1", "4-(silylamino)phenol"),
])
def test_best_effort_encloses_silylamino(smiles, name):
    row = name_breadth(smiles)
    assert row.get("name") == name, row
    assert name_is_rt_exact(name, smiles)
