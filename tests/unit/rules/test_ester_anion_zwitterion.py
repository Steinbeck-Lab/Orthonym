"""Choline-family acid-ester-anion zwitterions (a phase B1).

Wires the cation-bearing-owner-substituent capability  into the
GUARD-4 zwitterion path (`charged_router._route_zwitterion`) so a cation +
acid-ester anion zwitterion (a single P/S oxoacid mono-ester whose owner arm
carries the quaternary cation, e.g. choline sulfate/phosphate) gets a real
name instead of abstaining. RT-gated (`@pytest.mark.opsin_gate`); the
top-level /OPSIN gate is the 0-wrong backstop.
"""
import pytest
from orthonym import Orthonym


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


# --- the 3 goal molecules -----------------------------------------------
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi,expected", [
    ("C[N+](C)(C)CCOS(=O)(=O)[O-]", "2-(trimethylazaniumyl)ethyl sulfate"),
    ("C[N+](C)(C)CCOP(=O)([O-])[O-]", "2-(trimethylazaniumyl)ethyl phosphate"),
    ("C[N+](C)(C)CCOP(=O)([O-])O", "2-(trimethylazaniumyl)ethyl hydrogen phosphate"),
])
def test_choline_ester_anion_zwitterion(namer, smi, expected):
    assert namer.name(smi) == expected


# --- regressions: untouched paths ----------------------------------------
@pytest.mark.opsin_gate
def test_carboxylate_betaine_unchanged(namer):
    # GUARD 4's existing (…azaniumyl)-prefix betaine path must be
    # completely unaffected by the new ester-anion helper (its central-atom
    # check declines instantly for a carboxylate O, whose neighbour is C).
    assert namer.name("C[N+](C)(C)CCC(=O)[O-]") == "3-(trimethylazaniumyl)propanoate"


@pytest.mark.opsin_gate
def test_pure_anion_dodecyl_sulfate_unchanged(namer):
    # Slice A's pure-anion producer (no cation at all) is a completely
    # separate dispatch path (route_charged never calls _route_zwitterion
    # when there is no cation site) -- must stay byte-identical.
    assert namer.name("CCCCCCCCCCCCOS(=O)(=O)[O-]") == "dodecyl sulfate"


# --- fail-closed: a bad shape must never emit a sulfate/phosphate name ---
@pytest.mark.opsin_gate
def test_thiophosphate_ester_zwitterion_failclosed(namer):
    # P=S (thio) ester zwitterion -- not the clean mono-ester shape; the new
    # helper must decline (not fabricate a phosphate/sulfate name). Abstain
    # (or some non-phosphate/sulfate fallback name) is acceptable; a name
    # containing 'phosphate' or 'sulfate' is NOT.
    out = namer.name("C[N+](C)(C)CCOP(=S)([O-])[O-]")
    assert out is not None
    assert "phosphate" not in out
    assert "sulfate" not in out
