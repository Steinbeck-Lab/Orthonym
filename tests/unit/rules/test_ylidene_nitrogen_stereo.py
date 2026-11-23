"""The E/Z of a C=N that joins an '-ylidene' substituent to a nitrogen of the parent: method (b), below the PIN.

User decision 2026-11-23 (item 407): the Blue Book gives no PIN form for it "Descriptors 'E' and 'Z' to
denote double bonds linked to heteroatoms", the Blue Book, makes the carbon-parent methanimine the PIN and lists
the nitrogen-parent '...methylidene]hydroxylamine' as another name), so the descriptor is cited at the front of the
'-ylidene' prefix (1)(b), the Blue Book) and the name ships below the PIN; no front '(1E)' that
cites the ylidene group's own carbon, and no '(N'E)'.
"""
import pytest

from tests.support.rt_assert import name_best_effort, name_is_rt_exact

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,name", [
    ("CCCCCCC(=O)N/N=C/Cc1ccccc1", "N'-[(1E)-2-phenylethylidene]heptanehydrazide"),
    ("CCCCCCCC(=O)N/N=C\\CCC1CCCCC1", "N'-[(1Z)-3-cyclohexylpropylidene]octanehydrazide"),
    ("CC/C=N/NC(N)=O", "2-[(1E)-propylidene]hydrazine-1-carboxamide"),
    ("C/C=N/N", "[(1E)-ethylidene]hydrazine"),
])
def test_the_descriptor_stands_in_the_ylidene_prefix_below_the_pin(smiles, name):
    r = name_best_effort(smiles)
    assert r["name"] == name and r["tier"] == "systematic_verified" and not r["is_pin"], r
    assert name_is_rt_exact(name, smiles)


@pytest.mark.parametrize("smiles", ["CCOC(=O)N/N=C/c1ccccc1", "C(/C=N/NC(=O)C)=C/c1ccccc1"])
def test_every_ylidene_on_a_parent_nitrogen_is_below_the_pin(smiles):
    r = name_best_effort(smiles)
    assert r["name"] and not r["is_pin"] and "(N'E)" not in r["name"], r


@pytest.mark.parametrize("smiles,name", [
    ("c1ccccc1/N=C/c1ccccc1", "(1E)-N,1-diphenylmethanimine"),
    ("C/C(=N/O)c1ccccc1", "(1Z)-N-hydroxy-1-phenylethan-1-imine"),
    ("CCCCCCC(=O)NN=CCc1ccccc1", "N'-(2-phenylethylidene)heptanehydrazide"),
    # the carbon of the C=N is in the parent, the nitrogen in the 'hydrazinylidene' prefix: method (a), the PIN
    ("C/C(=N/Nc1nncc2ccccc12)C(=O)O", "(2Z)-2-[(phthalazin-1-yl)hydrazinylidene]propanoic acid"),
])
def test_a_carbon_parent_or_a_name_without_stereo_keeps_its_pin(smiles, name):
    r = name_best_effort(smiles)
    assert r["name"] == name and r["tier"] == "pin_verified", r
