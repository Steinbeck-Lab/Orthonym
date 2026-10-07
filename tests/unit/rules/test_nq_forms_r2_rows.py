"""Lane L2 proper fix (R2), end to end: the Blue Book row 44.2.1.1 and the class around
it. Every name read back by OPSIN to the input's full key.

* '1,8-di(bicyclo[3.2.1]octan-3-yl)anthracene (PIN)' (the Blue Book);
* '3,5-di([1,1'-biphenyl]-3-yl)pyridine (PIN)' (:23913;:7469: the brackets
  of a ring assembly are ignored for nesting);
* (a) (:7104): a substituted prefix takes 'bis' on every writer path.
"""
import pytest

from orthonym import Orthonym
from orthonym.assembly.naming_utils import format_substituent_prefix
from tests.support.rt_assert import name_is_rt_exact

BB_44_2_1_1 = "c1cc(C2CC3CCC(C3)C2)c2cc3c(C4CC5CCC(C5)C4)cccc3cc2c1"
PYR_HPM = "OC(=O)c1cc(C(O)c2ccccc2)nc(C(O)c2ccccc2)c1"


@pytest.mark.parametrize("name,expected", [
    ("bicyclo[3.2.1]octan-3-yl", "1,8-di(bicyclo[3.2.1]octan-3-yl)"),   #:19430
    ("spiro[4.5]decan-8-yl", "1,8-di(spiro[4.5]decan-8-yl)"),            # (f)
    ("(13C)methyl", "1,8-di[(13C)methyl]"),                              #:7492
])
def test_brackets_alone_never_decide_the_multiplier(name, expected):
    assert format_substituent_prefix(name, [1, 8], 2) == expected


PIN_ROWS = [
    (BB_44_2_1_1, "1,8-di(bicyclo[3.2.1]octan-3-yl)anthracene"),
    ("OC(=O)c1cc(C2CC3CCC(C3)C2)cc(C2CC3CCC(C3)C2)c1",
     "3,5-di(bicyclo[3.2.1]octan-3-yl)benzoic acid"),
    ("c1cc(C2CCC3(CC2)CCCC3)c2cc3c(C4CCC5(CC4)CCCC5)cccc3cc2c1",
     "1,8-di(spiro[4.5]decan-8-yl)anthracene"),
    ("c1ccc(-c2cccc(-c3cncc(-c4cccc(-c5ccccc5)c4)c3)c2)cc1",          # BB 52.2.5.3
     "3,5-di([1,1'-biphenyl]-3-yl)pyridine"),
    (PYR_HPM, "2,6-bis[hydroxy(phenyl)methyl]pyridine-4-carboxylic acid"),
    ("OC(=O)c1cc(N(C)C2CCCCC2)cc(N(C)C2CCCCC2)c1",
     "3,5-bis[cyclohexyl(methyl)amino]benzoic acid"),
    ("OC(=O)c1cc([C@@H](O)c2ccccc2)cc([C@@H](O)c2ccccc2)c1",
     "3,5-bis[(S)-hydroxy(phenyl)methyl]benzoic acid"),
    # the ether-chain writer's record reaches the multiplier (Task 3 Step 6 trace)
    ("COCc1cc(COC)ncc1", "2,4-bis(methoxymethyl)pyridine"),
    # unchanged rows (the record must not move them)
    ("Cc1cccc(C)n1", "2,6-dimethylpyridine"),
    ("ClCc1ccc(CCl)nc1", "2,5-bis(chloromethyl)pyridine"),
    ("CC(C)C1CCC(CC1)C(C)C", "1,4-di(propan-2-yl)cyclohexane"),          #:25719
    # properfix a performance pass (review I1): a ring assembly is a simple component
    # (f)); the record must survive the merge path.
    ("OC(c1cccc(-c2ccccc2)c1)c1cccc(-c2ccccc2)c1",
     "di([1,1'-biphenyl]-3-yl)methanol"),                                 #:23913
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", PIN_ROWS, ids=[r[1] for r in PIN_ROWS])
def test_multiplied_prefixes_at_the_pin_tier(smiles, expected):
    row = Orthonym().name_tiered(smiles)
    assert (row.get("name"), row.get("tier")) == (expected, "pin_verified"), row
    assert name_is_rt_exact(expected, smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("mode", ["on", "off", "verify"])
def test_the_record_gives_the_same_name_in_every_memo_mode(monkeypatch, mode):
    from orthonym.assembly import memo
    monkeypatch.setattr(memo, "_MODE", mode)
    memo.clear_process_cache()
    memo.reset_verify_mismatches()
    for smiles, expected in ((BB_44_2_1_1, PIN_ROWS[0][1]), (PYR_HPM, PIN_ROWS[4][1])):
        assert Orthonym().name_tiered(smiles).get("name") == expected
    assert memo.verify_mismatch_count() == 0
