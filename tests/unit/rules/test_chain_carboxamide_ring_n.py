"""A carboxamide at the end of a chain, whose nitrogen carries a ring, is a PIN-tier substituent prefix.

The PIN tier names a chain carboxamide by method (1) of ("method (1) is preferred for chains",
the Blue Book), 'amino' and 'oxo' on the terminal atom (``assembly.chain_carboxamide.is_chain_carboxamide``), but
only when the amide nitrogen carried hydrogen or saturated carbon; with a ring on the nitrogen the strict path had
no prefix and only the breadth re-run named the molecule (pin_unverified). The Blue Book's own preferred prefix has
a ring there: 'C6H5-NH-CO-CH2– 2-anilino-2-oxoethyl (preferred prefix)',:24603). The amino group itself, an
'-NH-R' branch with a ring R, is named at the PIN tier by the same located namer
(``assembly.chain_carboxamide.is_ring_amino``), and the bare retained 'anilino' takes no enclosing marks.
"""
import pytest

from tests.support.pin_tiers import tier_row
from tests.support.rt_assert import name_is_rt_exact

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,expected", [
    ("OC(=O)c1ccc(CC(=O)Nc2ccccc2)cc1", "4-(2-anilino-2-oxoethyl)benzoic acid"),
    ("OC(=O)c1ccc(CC(=O)Nc2ccncc2)cc1", "4-{2-oxo-2-[(pyridin-4-yl)amino]ethyl}benzoic acid"),
    ("O=C(Nc1ccccc1)CCCc1ccccc1C(=O)O", "2-(4-anilino-4-oxobutyl)benzoic acid"),
    ("O=C(O)CSCC(=O)Nc1cccnc1", "({2-oxo-2-[(pyridin-3-yl)amino]ethyl}sulfanyl)acetic acid"),
    # the amino group with a ring on its nitrogen is named inside the prefix too ('anilino' bare,
    ("O=C(Nc1ccccc1)CNC(=O)OC(C)(C)C", "tert-butyl (2-anilino-2-oxoethyl)carbamate"),
    ("CC(C)C[C@@H](C(=O)NC1CCCCC1)NC(=O)OC(C)(C)C",
     "tert-butyl [(2S)-1-(cyclohexylamino)-4-methyl-1-oxopentan-2-yl]carbamate"),
    ("CC(C)C[C@@H](C(=O)Nc1ccc(Cl)cc1)NC(=O)OC(C)(C)C",
     "tert-butyl [(2S)-1-(4-chloroanilino)-4-methyl-1-oxopentan-2-yl]carbamate"),
    ("O=C(Nc1ccncc1)CNC(=O)OC(C)(C)C", "tert-butyl {2-oxo-2-[(pyridin-4-yl)amino]ethyl}carbamate"),
    # a saturated alkyl group on the amino nitrogen is named the same way ('ethylamino')
    ("CC(C)C[C@@H](C(=O)NCC)NC(=O)OC(C)(C)C", "tert-butyl [(2S)-1-(ethylamino)-4-methyl-1-oxopentan-2-yl]carbamate"),
    ("CC(C)C[C@@H](C(=O)NC(C)C)NC(=O)OC(C)(C)C",
     "tert-butyl {(2S)-4-methyl-1-oxo-1-[(propan-2-yl)amino]pentan-2-yl}carbamate"),
    # a saturated N-substituent and a ring nitrogen keep their names
    ("OC(=O)c1ccc(CCC(=O)N2CCCCC2)cc1", "4-[3-oxo-3-(piperidin-1-yl)propyl]benzoic acid"),
    ("CCNC(=O)CNC(=O)OC(C)(C)C", "tert-butyl [2-(ethylamino)-2-oxoethyl]carbamate"),
])
def test_pin(smiles, expected):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (expected, "pin_verified"), row
    assert name_is_rt_exact(expected, smiles)
