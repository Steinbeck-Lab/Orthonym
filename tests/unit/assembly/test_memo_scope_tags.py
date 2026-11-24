"""The memo key of a run (``assembly.memo._ck``) tells the isotopic naming scope and the cage-ceiling scopes apart.

Inside the isotopic naming scope (``rules.isotopes`` names the skeleton of a labelled molecule with its labels
stripped) a producer may refuse or spell otherwise, and inside ``pin_ceilings_scope`` / ``lifted_ring_ceiling_scope``
every von Baeyer analysis takes other ring-system ceilings. A value computed in one of them is not the plain one:
measured 2026-11-24 with ``ORTHONYM_MEMO=verify``, a 'substituent' sentinel stored in the isotopic scope was served to
the plain naming of '2,3-bis(bromomethyl)quinoxalin-6-yl', and a spiro prefix named under the best-effort ceilings was
served inside the PIN-ceiling rescue, which declines it.
"""
import pytest

from orthonym.assembly import memo
from orthonym.assembly.locant_omission import ISOTOPIC_STEREO_LOCANTS, isotopic_naming_scope
from orthonym.rules.vonbaeyer_universal import lifted_ring_ceiling_scope, pin_ceilings_scope

pytestmark = pytest.mark.unit


def test_each_scope_has_its_own_key():
    plain = memo._ck("name_substituent", ("CCO", 1))
    with isotopic_naming_scope():
        isotopic = memo._ck("name_substituent", ("CCO", 1))
    with isotopic_naming_scope(ISOTOPIC_STEREO_LOCANTS):
        stereo_locants = memo._ck("name_substituent", ("CCO", 1))
    with pin_ceilings_scope():
        pin_only = memo._ck("name_substituent", ("CCO", 1))
    with lifted_ring_ceiling_scope():
        lifted = memo._ck("name_substituent", ("CCO", 1))
    keys = [plain, isotopic, stereo_locants, pin_only, lifted]
    assert len(set(keys)) == len(keys)
    assert memo._ck("name_substituent", ("CCO", 1)) == plain      # the scopes leave no trace


def test_a_value_stored_in_the_isotopic_scope_is_not_served_outside_it(monkeypatch):
    monkeypatch.setattr(memo, "_MODE", "on")
    token = memo.push_scope()
    try:
        with isotopic_naming_scope():
            memo.cache_or_compute("ns", "k", lambda: "substituent")
        assert memo.cache_or_compute("ns", "k", lambda: "real prefix") == "real prefix"
    finally:
        memo.pop_scope(token)
