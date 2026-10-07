"""Names that are right stay right at the default tier (name-quality program, item Q7).

Each name below is the Blue Book PIN (or follows the rule the book states for it), is emitted at
the default tier labelled pin_verified, and passes every spelling check:

* 'ethene (PIN)' (the Blue Book, (d): no locant in unsubstituted ethene);
* 'acetic acid (PIN)' (:2010);
* 'hydrazinecarboxamide (PIN)' (:32675);
* '1H-arsole': (:3721) "in a preferred IUPAC name a locant and the symbol 'H' must be
  cited", as in '1H-phosphole (PIN)' (:16930);
* 'trichloroarsane': (a) no locant on a substituted mononuclear parent
  ('dichlorosilane':2899), (:3448);
* '3-[(pyridine-3-carbonyl)oxy]propanoic acid (PIN)' (:31723).

And the indicated hydrogen of an imidazol-1-yl prefix: (:24639) "in preferred IUPAC
names indicated hydrogen must always be cited when present in the corresponding structure";
 (:17307) "Indicated and added indicated hydrogen atoms must be cited in names" of
substituent groups; the book prints 'di(1H-imidazol-1-yl)methanethione (PIN)' (:29544) and
'(1H-indol-1-yl)acetic acid (PIN)' (:2039). The writer keeps the '1H-'.
Every name is read back by a fresh OPSIN call to the input's full InChIKey. (The spelling checks
pass on each of these names: the negatives of tests/unit/validation/test_pin_spelling_*.py.)
"""
import pytest

from orthonym import Orthonym
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.integration, pytest.mark.opsin_gate]

RIGHT = [
    ("C=C", "ethene"),
    ("CC(=O)O", "acetic acid"),
    ("NNC(=O)N", "hydrazinecarboxamide"),
    ("[AsH]1cccc1", "1H-arsole"),
    ("Cl[As](Cl)Cl", "trichloroarsane"),
    ("O=C(O)CCOC(=O)c1cccnc1", "3-[(pyridine-3-carbonyl)oxy]propanoic acid"),
    ("S=C(n1ccnc1)n1ccnc1", "di(1H-imidazol-1-yl)methanethione"),
    ("OC(=O)Cn1ccnc1", "(1H-imidazol-1-yl)acetic acid"),
]


@pytest.mark.parametrize("smiles,name", RIGHT)
def test_a_right_name_stays_pin_verified_at_the_default_tier(smiles, name):
    row = Orthonym().name_tiered(smiles)
    assert (row["name"], row["tier"], row["is_pin"]) == (name, "pin_verified", True), row
    assert not row.get("spelling_failures"), row
    assert Orthonym().name(smiles) == name
    assert_full_rt(name, smiles)
