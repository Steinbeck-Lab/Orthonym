"""SF lane E: the read-back policy keeps no state between naming calls, and its two vetoes still veto.

The suite runs with the OPSIN validity gate OFF (``tests/conftest.py``), so a check that lets a
name through when OPSIN gives no structure is the only thing between that name and the caller.
Two checks do that, on purpose, for a name OPSIN cannot read at all (house convention: a missing
JVM never hard-fails a name): ``ions._validate_cation_name`` (the cation keeps its net charge)
and ``organometallics._opsin_read_net_charge``. Under the closed-shell reading a radical-only name
('2-methylpropane-1,3-diylidene' for the dication ``[CH2+]C(C)[CH2+]``) gives no structure, so both
had to keep their own reading (``legacy_reading``): they only veto, a structure with the wrong
charge is what they look for. ``tests/unit/rules/test_bis_suffix_multiplication_8c.py`` is the
file that showed it.
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.assembly.fragment_naming import innermost_name_input
from orthonym.rules.ions import _validate_cation_name
from orthonym.rules.organometallics import _opsin_read_net_charge
from orthonym.validation.read_back_policy import _BASE, _FORCED, input_allows_radicals
from orthonym.wallclock import leaked_naming_state

pytestmark = pytest.mark.unit

DICATION = "[CH2+]C(C)[CH2+]"
RADICAL_ONLY = "2-methylpropane-1,3-diylidene"


def _abstains(name):
    low = (name or "").lower()
    return (not name) or "unknown" in low or "not supported" in low


def _clean():
    assert innermost_name_input() is None
    assert _FORCED.get() is None and _BASE.get() is None
    assert input_allows_radicals() is True
    assert leaked_naming_state() == {}, leaked_naming_state()


@pytest.mark.parametrize("order", [("C[CH]C", DICATION), (DICATION, "C[CH]C"),
                                   ("[CH2]C[CH2]", DICATION, "CCO", DICATION)])
def test_a_radical_input_and_the_dication_in_one_process(order):
    for smi in order:
        name = Orthonym().name(smi)
        if smi == DICATION:
            assert _abstains(name), name
        elif smi == "C[CH]C":
            assert name == "propan-2-yl", name
        elif smi == "CCO":
            assert name == "ethanol", name
        _clean()


def test_the_dication_is_not_named_from_a_radical_reading_of_its_name():
    name = Orthonym().name(DICATION)
    assert not name or "diylidene" not in name, name


def test_the_charge_veto_of_the_cation_namer_sees_a_radical_only_name():
    mol = Chem.MolFromSmiles(DICATION)
    from orthonym.assembly.fragment_naming import enter_name_scope, exit_name_scope
    enter_name_scope(DICATION)                     # a closed-shell input being named
    try:
        assert input_allows_radicals() is False
        assert _validate_cation_name(mol, RADICAL_ONLY) == ""
        assert _opsin_read_net_charge(RADICAL_ONLY) == 0      # read with -r: neutral, not the +2
    finally:
        exit_name_scope()
    _clean()
