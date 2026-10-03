"""Benzene fused to a heteromonocycle is named as a benzoheterocycle with its heteroatom
locants, the Blue Book;:11815 "for preferred IUPAC names locants must be
cited"; '1-benzofuran (PIN) benzofuran',:11827; '2-benzofuran (PIN) isobenzofuran
benzo[c]furan',:11829; 'hexahydro-2-benzothiophene-1,3-dione (PIN)',:32546).

The catalogue lacked 2-benzothiophene, 1,2-benzothiazole, 1-benzoselenophene and
2-benzoselenophene, so the algorithmic path built the fusion-descriptor names
('benzo[c]thiophene', 'benzo[d]isothiazole',...), valid but never the PIN, and the default
tier declined. The hydro twins take the catalogue parent through the mancude twin. A ring atom
with a nonstandard bonding number keeps the catalogue match from firing; the descriptor path
then takes the catalogue benzo name for the same skeleton ('1H-1λ4-1-benzothiophene'), and its
one indicated hydrogen is the mancude count ('1H-1λ4-thiophene (PIN)',,:9167).
Every name read back by OPSIN 2.9.0 to the input's full InChIKey.
"""
import pytest

from tests.support.pin_tiers import assert_pin_at_both_tiers

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,pin", [
    ("c1ccc2cscc2c1", "2-benzothiophene"),
    ("c1ccc2sncc2c1", "1,2-benzothiazole"),
    ("c1ccc2[se]ccc2c1", "1-benzoselenophene"),
    ("c1ccc2c[se]cc2c1", "2-benzoselenophene"),
    ("Cc1scc2ccccc12", "1-methyl-2-benzothiophene"),
    ("Cc1nsc2ccccc12", "3-methyl-1,2-benzothiazole"),
    ("Cc1cc2ccccc2[se]1", "2-methyl-1-benzoselenophene"),
    ("OC(=O)c1scc2ccccc12", "2-benzothiophene-1-carboxylic acid"),
    ("c1scc2c1CCCC2", "4,5,6,7-tetrahydro-2-benzothiophene"),
    ("c1nsc2c1CCCC2", "4,5,6,7-tetrahydro-1,2-benzothiazole"),
    ("[SH2]1C=CC2=C1C=CC=C2", "1H-1λ4-1-benzothiophene"),
    ("c1ccc2sccc2c1", "1-benzothiophene"),
])
def test_benzo_parent_at_both_tiers(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)
