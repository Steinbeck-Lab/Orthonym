"""
Wave2 Tier 4 — fused-name catalog extensions.

Reproduce-first (see the Tier-4 research) reframed the master-plan Tier 4:
the "coverage-veto" was a no-op (SELF-01 + the v23-13B(a) guard already fail
these closed) and the general benzo-heterocycle CONSTRUCTOR + indicated-H
via the algorithmic path both need a deep fusion-numbering-engine fix
(heteroatom lowest-locant, O>N) that is deferred. The tractable, correct wins
are bare fused-name CATALOG entries (the shipped DATA-01/S2a pattern):

  * P-25.2.2.4 benzo-heterocycles — benzene ortho-fused to a 7-membered hetero
    ring (benzoxepines/benzothiepine, and the benzazepines whose trivalent N
    carries a mandatory indicated hydrogen: 1H-/2H-/3H-).
  * P-25.7.1.3.2 aromatic-NH indicated-H on small pyrrole-fused bicyclics
    (furo/thieno[3,2-b]pyrrole -> 4H-, furo[2,3-b]pyrrole -> 6H-). The correct
    indicated-H locant (baked into the stored name) follows heteroatom
    seniority: for furo[3,2-b]pyrrole O=1 so the pyrrole N is 4 -> 4H-.
  * P-25.1.2 higher hydrocarbon series members (pentaphene/hexaphene aphenes,
    hexacene, hexahelicene). NOTE the helicene series starts at SIX rings
    (P-25.1.2.6), so [5]helicene is dibenzo[c,g]phenanthrene (fusion-engine,
    deferred), NOT "pentahelicene".

All PINs OPSIN-round-trip verified. Numbering (for substituted forms, which
were fail-closed at HEAD) was extendedsmi-derived and methyl-isomer-confirmed.
"""

import pytest

from orthonym import name_compound


# ---------------------------------------------------------------------------
# Bare fused-name catalog wins — each was fail-closed 'unknown' at HEAD.
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,expected",
    [
        # P-25.2.2.4 benzo-heteropines (divalent O/S -> no indicated H)
        ("C1=COc2ccccc2C=C1", "1-benzoxepine"),
        ("C1=COC=c2ccccc2=C1", "2-benzoxepine"),
        ("C1=Cc2ccccc2C=CO1", "3-benzoxepine"),
        ("C1=Cc2ccccc2C=CS1", "3-benzothiepine"),
        # benzazepines — trivalent N bears the mandatory indicated hydrogen
        ("C1=CNc2ccccc2C=C1", "1H-1-benzazepine"),
        ("C1=CNC=c2ccccc2=C1", "2H-2-benzazepine"),
        ("C1=Cc2ccccc2C=CN1", "3H-3-benzazepine"),
        # P-25.7.1.3.2 aromatic-NH indicated-H (locant by heteroatom seniority)
        ("c1cc2occc2[nH]1", "4H-furo[3,2-b]pyrrole"),
        ("c1cc2sccc2[nH]1", "4H-thieno[3,2-b]pyrrole"),
        ("c1cc2ccoc2[nH]1", "6H-furo[2,3-b]pyrrole"),
        # P-25.1.2 higher hydrocarbon series
        ("c1ccc2cc3c(ccc4cc5ccccc5cc43)cc2c1", "pentaphene"),
        ("c1ccc2cc3cc4c(ccc5cc6ccccc6cc54)cc3cc2c1", "hexaphene"),
        ("c1ccc2cc3cc4cc5cc6ccccc6cc5cc4cc3cc2c1", "hexacene"),
        ("c1ccc2c(c1)ccc1ccc3ccc4ccc5ccccc5c4c3c12", "hexahelicene"),
    ],
)
def test_tier4_bare_catalog_wins(smiles, expected):
    assert name_compound(smiles) == expected


# ---------------------------------------------------------------------------
# Substituted forms also number correctly (bonus — these were fail-closed too).
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,expected",
    [
        # methyl on the furo[3,2-b]pyrrole 2-position (OPSIN-RT)
        ("CC1=CC=2NC=CC2O1", "2-methyl-4H-furo[3,2-b]pyrrole"),
        # methyl on pentaphene position 6 (OPSIN-RT)
        ("CC1=C2C=C3C=CC=CC3=CC2=C2C=C3C=CC=CC3=CC2=C1", "6-methylpentaphene"),
    ],
)
def test_tier4_substituted_numbering(smiles, expected):
    assert name_compound(smiles) == expected


# ---------------------------------------------------------------------------
# Fail-closed: [5]helicene's PIN is the fusion name dibenzo[c,g]phenanthrene
# (helicene series starts at 6 rings), which needs the deferred polycomponent-
# fusion engine — must stay 'unknown', never a wrong "pentahelicene". In raw
# (gate-off) mode the substructure matcher over-matches a benzo[c]phenanthrene
# core (a 4-ring name for the 5-ring system); the production validity gate
# suppresses that to 'unknown'. The suite autouse-DISABLES the gate, so
# re-enable it to assert the production behavior (test_tier3b pattern).
# ---------------------------------------------------------------------------

@pytest.fixture
def _validity_gate_on(monkeypatch):
    import orthonym.namer as _namer
    monkeypatch.setattr(_namer, "_DISABLE_VALIDITY_GATE", False, raising=False)
    yield


@pytest.mark.unit
def test_pentahelicene_stays_fail_closed(_validity_gate_on):
    assert "unknown" in name_compound("c1ccc2c(c1)ccc1ccc3ccc4ccccc4c3c12")


# ---------------------------------------------------------------------------
# Protection — the existing PAH / fused-heterocycle catalog is unchanged.
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,expected",
    [
        ("c1ccc2ccccc2c1", "naphthalene"),
        ("c1ccc2cc3ccccc3cc2c1", "anthracene"),
        ("c1ccc2ccc3ccccc3c2c1", "phenanthrene"),
        ("c1ccc2cc3cc4cc5ccccc5cc4cc3cc2c1", "pentacene"),
        ("C1=CC=C2C=CC3=CC=CC4=CC=C1C2=C34", "pyrene"),
        ("c1ccc2cocc2c1", "2-benzofuran"),
        ("c1ccc2occc2c1", "benzofuran"),
        ("c1ccc2[nH]ccc2c1", "1H-indole"),
        ("c1ccc2ncccc2c1", "quinoline"),
        ("Cc1ccc2ccccc2c1", "2-methylnaphthalene"),
        ("Cc1ccc2cc3ccccc3cc2c1", "2-methylanthracene"),
    ],
)
def test_tier4_catalog_protection(smiles, expected):
    assert name_compound(smiles) == expected
