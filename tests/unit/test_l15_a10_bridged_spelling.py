"""A bridge prefix is set off from an italic parent designator ('1,4-methano-s-indacene').

 "Hyphens are used in substitutive names" (the Blue Book), (d) "to separate italic letters from Roman
letters" (:6960), example '*as*-indacene' (:6966); the book prints the same hyphen in front of the italic designator in
'...cyclopenta[*cd*]-*s*-indacene (PIN)' (:14505). The bridge prefix of (:14046) stands in front of the fused
parent's name, and 'methano' ends in a Roman letter while 's-indacene',:11404) and 'as-indacene' (:11408)
begin with an italic one, so the prefix and the parent are joined with a hyphen: '1,4-methano-s-indacene', not
'1,4-methanos-indacene'. No vowel is elided here lists none for a bridge prefix, and (g) keeps the
terminal vowel of an attached-component prefix); the spelling defect was a missing hyphen. OPSIN 2.9.0 reads both
spellings to one structure, so only a spelling test sees the difference.
"""
import pytest

from orthonym.cli import _tiered_namer
from orthonym.rules.bridged_fused_pin import BridgedParent
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

# druglike row 32: an s-indacene with a 1,4-methano bridge
_ROW_32 = "C[C@@H]1CC[C@@H]2[C@@H]1CC3(C4CC2([C@]3(C(=C4)C(C)C)C(=O)O)C=O)C(=O)C#CCC5CCCCC5"
_ROW_32_NAME = ("(3aS,4aR,7R,7aR)-8a-(4-cyclohexylbut-2-ynoyl)-4-formyl-7-methyl-3-(propan-2-yl)-"
                "4,4a,5,6,7,7a,8,8a-octahydro-1,4-methano-s-indacene-3a(1H)-carboxylic acid")
# the angular isomer: as-indacene with a 4,8a-methano bridge
_AS_INDACENE = "C1CC2C3CC4CCCC4(C3)C2C1"
_AS_INDACENE_NAME = "decahydro-1H-4,8a-methano-as-indacene"
# the controls: a bridge prefix before a parent that opens with a Roman letter, and the unbridged s-indacene
_NAPHTHALENE_BRIDGED = "C1=CC2CC1c1ccccc12"
_S_INDACENE_UNBRIDGED = "C1CC2=C(C1)C=C1CCCC1=C2"


def _pin(smiles):
    return _tiered_namer("pin", "pin").name_tiered(smiles)


def test_row_32_spells_the_bridge_and_the_italic_parent_with_a_hyphen():
    row = _pin(_ROW_32)
    assert (row["name"], row["tier"]) == (_ROW_32_NAME, "pin_verified"), row
    assert "methanos-indacene" not in row["name"]
    assert_full_rt(row["name"], _ROW_32)


def test_an_as_indacene_bridge_takes_the_same_hyphen():
    row = _pin(_AS_INDACENE)
    assert (row["name"], row["tier"]) == (_AS_INDACENE_NAME, "pin_verified"), row
    assert_full_rt(row["name"], _AS_INDACENE)


@pytest.mark.parametrize("hydro,ih,bridges,base,text", [
    ("octahydro", "", "1,4-methano", "s-indacene", "octahydro-1,4-methano-s-indacene"),
    ("", "1H", "4,8a-methano", "as-indacene", "1H-4,8a-methano-as-indacene"),
    ("", "", "1,4-methano", "s-indacene", "1,4-methano-s-indacene"),
    ("", "", "4a,8a-ethano", "as-indacene", "4a,8a-ethano-as-indacene"),
    # the controls: a parent that opens with a Roman letter, or a locant, is spelled as before
    ("1,4-dihydro", "", "1,4-methano", "naphthalene", "1,4-dihydro-1,4-methanonaphthalene"),
    ("", "", "1,4-methano", "naphthalene", "1,4-methanonaphthalene"),
    ("", "2H,7H", "4a,7-ethano", "1-benzopyran", "2H,7H-4a,7-ethano-1-benzopyran"),
    # no bridge: the hydro prefix keeps its own hyphen before the italic designator
    ("1,2,3,5,6,7-hexahydro", "", "", "s-indacene", "1,2,3,5,6,7-hexahydro-s-indacene"),
])
def test_the_parent_text_joins_the_bridge_prefix_per_p_16_2_4_1(hydro, ih, bridges, base, text):
    assert BridgedParent(hydro, ih, bridges, base, {}, True).text() == text


def test_a_bridge_before_a_roman_letter_parent_is_unchanged():
    row = _pin(_NAPHTHALENE_BRIDGED)
    assert (row["name"], row["tier"]) == ("1,4-dihydro-1,4-methanonaphthalene", "pin_verified"), row
    assert_full_rt(row["name"], _NAPHTHALENE_BRIDGED)


def test_the_unbridged_s_indacene_is_unchanged():
    row = _pin(_S_INDACENE_UNBRIDGED)
    assert (row["name"], row["tier"]) == ("1,2,3,5,6,7-hexahydro-s-indacene", "pin_verified"), row
    assert_full_rt(row["name"], _S_INDACENE_UNBRIDGED)
