"""Two replacement terms of a chain parent are separated by a hyphen (found 2026-11-09 by a scan of all eval names).

 / hyphen rule: a name part is separated from a following locant by a hyphen, as in
'3,12-dioxa-6-thia-9-azatetradecane' (the Blue Book) and '1-oxa-4,8,11-triazacyclotetradecane (PIN)' (:8842).
``_handler_shared._generate_chain_parent`` joined the terms bare: '4,5-dithia3,6-diazaoctane'.
"""
import pytest

from tests.support.pin_tiers import tier_row

pytestmark = pytest.mark.opsin_gate


def test_two_replacement_terms_are_hyphenated():
    row = tier_row("CCN(P1OCC(CO1)(C)C)SSN(CC)P2OCC(CO2)(C)C", "best-effort")
    assert row["name"] == ("3,6-bis(5,5-dimethyl-1,3,2-dioxaphosphinan-2-yl)-4,5-dithia-3,6-diazaoctane"), row
