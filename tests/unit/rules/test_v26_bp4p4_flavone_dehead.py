"""v26 BP-4 Phase 4 — flavonoid parents de-headlined to their systematic PIN.

flavone/flavanone/isoflavone are retained GENERAL-nomenclature trivial names, not
PINs: P-102.6.1.4 (BB L53955) prints '…flavone' in the general column while the PIN
column is '…-4H-1-benzopyran-4-one' (1-benzopyran is the PIN ring parent per
P-19(d) L1736; the ketone substitutes the 4H >CH2 per P-64.2.2.2.2 L28410).

Fix (established chromone/chromanone precedent): removed from
NATURAL_PRODUCT_DERIVATIVES + added pin:false to iupac_2013_pin_list.json, so the
NP/retained dispatch misses and the cyclic-oxo engine emits the systematic PIN and
places substituents (hydroxyflavones) via the 1-benzopyran numbering. Pure data
edit; no locant-map code (approach A rejected — it would perpetuate non-PIN
substituted names).
"""
import pytest
from orthonym.namer import name_compound


def test_flavone_denied_and_general_preserved():
    from rdkit import Chem
    from orthonym.data import get_retained_name, get_general_retained_name
    cs = Chem.CanonSmiles("O=c1cc(-c2ccccc2)oc2ccccc12")
    assert get_retained_name(cs) is None               # de-headlined (PIN default)
    assert get_general_retained_name(cs) == "flavone"  # general-nomenclature lookup


@pytest.mark.parametrize("smi,pin", [
    ("O=c1cc(-c2ccccc2)oc2ccccc12", "2-phenyl-4H-1-benzopyran-4-one"),
    ("O=C1CC(c2ccccc2)Oc2ccccc21", "2-phenyl-2,3-dihydro-4H-1-benzopyran-4-one"),
    ("O=c1c(-c2ccccc2)coc2ccccc12", "3-phenyl-4H-1-benzopyran-4-one"),
])
def test_flavonoid_parents_pin(smi, pin):
    assert name_compound(smi) == pin


@pytest.mark.parametrize("smi,pin", [
    ("O=c1cc(-c2ccccc2)oc2cc(O)ccc12", "7-hydroxy-2-phenyl-4H-1-benzopyran-4-one"),
    ("O=c1cc(-c2ccccc2)oc2cccc(O)c12", "5-hydroxy-2-phenyl-4H-1-benzopyran-4-one"),
    ("O=c1cc(-c2ccccc2)oc2ccc(O)cc12", "6-hydroxy-2-phenyl-4H-1-benzopyran-4-one"),
    ("O=c1cc(-c2ccc(O)cc2)oc2ccccc12", "2-(4-hydroxyphenyl)-4H-1-benzopyran-4-one"),
])
def test_substituted_flavones_pin(smi, pin):
    assert name_compound(smi) == pin


@pytest.mark.parametrize("smi,pin", [
    ("O=c1ccoc2ccccc12", "4H-1-benzopyran-4-one"),               # chromone
    ("O=c1ccc2ccccc2o1", "2H-1-benzopyran-2-one"),               # coumarin
    ("O=c1c2ccccc2oc2ccccc12", "9H-xanthen-9-one"),              # xanthone
    ("O=c1c(Br)coc2ccccc12", "3-bromo-4H-1-benzopyran-4-one"),   # 3-bromochromone
    ("O=c1ccoc2ccc(Cl)cc12", "6-chloro-4H-1-benzopyran-4-one"),  # 6-chlorochromone
    ("O=C1CCOc2ccccc21", "2,3-dihydro-4H-1-benzopyran-4-one"),   # chromanone
])
def test_benzopyranone_regression_guards(smi, pin):
    assert name_compound(smi) == pin
