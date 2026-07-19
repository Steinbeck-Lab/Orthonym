# tests/unit/test_general_fallback_wiring.py
"""v25 G1: general_fallback flag — default OFF byte-identity, ON recovery."""
from unittest import mock

import pytest

from orthonym.namer import Orthonym

pytestmark = pytest.mark.unit


def test_default_off_is_byte_identical_on_easy_molecule():
    assert (Orthonym(_disable_opsin_validity_gate=True).name("CCO")
            == Orthonym(_disable_opsin_validity_gate=True,
                         general_fallback=False).name("CCO"))


def test_flag_defaults_false():
    assert Orthonym()._general_fallback is False


def test_engine_fires_when_legacy_general_abstains():
    """Force the legacy GENERAL pipeline to abstain; engine must recover."""
    nm = Orthonym(_disable_opsin_validity_gate=True, general_fallback=True)
    with mock.patch("orthonym.namer.assemble_name",
                    return_value="unknown organic compound"):
        out = nm.name("CC(Cl)CC")
    assert out == "2-chlorobutane"


def test_engine_does_not_fire_when_flag_off():
    nm = Orthonym(_disable_opsin_validity_gate=True, general_fallback=False)
    with mock.patch("orthonym.namer.assemble_name",
                    return_value="unknown organic compound"):
        out = nm.name("CC(Cl)CC")
    assert out != "2-chlorobutane"
