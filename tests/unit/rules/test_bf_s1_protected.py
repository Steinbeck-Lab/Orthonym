"""Slice S1 protection: the bridged fused names the engine ships today stay byte-identical.

The 12 Blue Book bridged fused rows that MATCH at the S1 base (the Blue Book lines in the
comments) and the bare-system outputs of ``name_bridged_fused_pin``. The package
``rules/bridged_fused_pin`` must also reproduce every bare naphthalene/anthracene output of
the bare-system paths it stands behind (a differential check: the package is the general
builder, the bare paths are its special cases)."""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.jvm_budget import jvm_slots
from orthonym.rules.bridged_fused import name_bridged_fused_pin

BB_MATCH_ROWS = [
    ("c1ccc2c3c4ccccc4c(c2c1)CC3", "9,10-ethanoanthracene"),                      #:14183
    ("c1ccc2c3ccc(o3)c2c1", "1,4-epoxynaphthalene"),                              #:14227
    ("c1c2cc3cc4c5ccc(o5)c4cc3c1C2", "5,8-epoxy-1,3-methanoanthracene"),          #:14229
    ("C1=C2CC(=C1)c1cc3c4ccc(c3cc12)CC4", "1,4-ethano-5,8-methanoanthracene"),    #:14235
    ("C1=C2CC(=C1)c1c2c2ccc1o2", "1,4-epoxy-5,8-methanonaphthalene"),             #:14237
    ("c1ccc2c(c1)C1CCN2c2ccccc21", "9H-9,10-ethanoacridine"),                     #:14251
    ("C1=CC2CCC1c1cc3ccccc3cc12", "1,4-dihydro-1,4-ethanoanthracene"),            #:14399
    ("c1ccc2c3ccc(c2c1)CC3", "1,4-ethanonaphthalene"),                            #:14407
    ("c1ccc2c3cc(cc2c1)C3", "1,3-methanonaphthalene"),                            #:19863
    ("C1=C2CC(=C1)c1ccccc12", "1,4-methanonaphthalene"),                          #:19479
    ("c1cc2cc3ccc4cc5ccc6cc7ccc8cc9ccc%10cc1c1cc%10c9cc8c7cc6c5cc4c3cc21",
     "12,19:13,18-di(metheno)dinaphtho[2,3-a:2',3'-o]pentaphene"),               #
    ("c1cc2oc1-c1coc3occ-2c13", "2,3,9-trioxa-5,8-methanocyclopenta[cd]azulene"),  #
]

#: bare-system outputs of name_bridged_fused_pin at the S1 base (naphthalene / anthracene
#: residuals; each is also asserted by an older test file)
BARE_OUTPUTS = [
    ("C1C2C=CC1c1ccccc12", "1,4-dihydro-1,4-methanonaphthalene"),
    ("C1=CC2OC1c1ccccc12", "1,4-dihydro-1,4-epoxynaphthalene"),
    ("C1CC2CCC1c1ccccc12", "1,2,3,4-tetrahydro-1,4-ethanonaphthalene"),
    ("C1CC2c3ccccc3C1c1ccccc21", "9,10-dihydro-9,10-ethanoanthracene"),
    ("C1=CC2c3ccccc3C1c1ccccc12", "9,10-dihydro-9,10-ethenoanthracene"),
    ("C12=CC=C(C3=CC=CC=C13)C=C2", "1,4-ethenonaphthalene"),
    ("C12=CC=C(C=3C4=CC=C(C13)C4)C2", "1,4:5,8-dimethanonaphthalene"),
    ("C1CC2CC1c1ccccc12", "1,2,3,4-tetrahydro-1,4-methanonaphthalene"),
] + [row for row in BB_MATCH_ROWS[:5] + BB_MATCH_ROWS[6:10]]


def _pin_row(smiles):
    with jvm_slots(1, purpose="bf-s1"):
        return Orthonym().name_tiered(smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", BB_MATCH_ROWS)
def test_bb_match_rows_unchanged(smiles, name):
    row = _pin_row(smiles)
    assert (row.get("name"), row["tier"]) == (name, "pin_verified")


@pytest.mark.parametrize("smiles,name", BARE_OUTPUTS)
def test_bare_outputs_unchanged(smiles, name):
    res = name_bridged_fused_pin(Chem.MolFromSmiles(smiles))
    assert res is not None and res[0] == name, res


@pytest.mark.parametrize("smiles,name", BARE_OUTPUTS)
def test_package_reproduces_the_bare_outputs(smiles, name):
    from orthonym.rules.bridged_fused_pin import build
    res = build(Chem.MolFromSmiles(smiles))
    assert res is not None and res[0] == name, res
