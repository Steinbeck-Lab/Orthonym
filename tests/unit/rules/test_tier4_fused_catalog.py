"""
Wave2 Tier 4 — fused-name catalog extensions.

Reproduce-first (see the Tier-4 research) reframed the master-plan Tier 4:
the "coverage-veto" was a no-op (+ the -13B(a) guard already fail
these closed) and the general benzo-heterocycle CONSTRUCTOR + indicated-H
via the algorithmic path both need a deep fusion-numbering-engine fix
(heteroatom lowest-locant, O>N) that is deferred. The tractable, correct wins
are bare fused-name CATALOG entries (the shipped /S2a pattern):

  * benzo-heterocycles — benzene ortho-fused to a 7-membered hetero
    ring (benzoxepines/benzothiepine, and the benzazepines whose trivalent N
    carries a mandatory indicated hydrogen: 1H-/2H-/3H-).
  * aromatic-NH indicated-H on small pyrrole-fused bicyclics
    (furo/thieno[3,2-b]pyrrole -> 4H-, furo[2,3-b]pyrrole -> 6H-). The correct
    indicated-H locant (baked into the stored name) follows heteroatom
    seniority: for furo[3,2-b]pyrrole O=1 so the pyrrole N is 4 -> 4H-.
  * higher hydrocarbon series members (pentaphene/hexaphene aphenes,
    hexacene, hexahelicene). NOTE the helicene series starts at SIX rings
    , so [5]helicene is dibenzo[c,g]phenanthrene (fusion-engine,
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
        # benzo-heteropines (divalent O/S -> no indicated H)
        ("C1=COc2ccccc2C=C1", "1-benzoxepine"),
        ("C1=COC=c2ccccc2=C1", "2-benzoxepine"),
        ("C1=Cc2ccccc2C=CO1", "3-benzoxepine"),
        ("C1=Cc2ccccc2C=CS1", "3-benzothiepine"),
        # benzazepines — trivalent N bears the mandatory indicated hydrogen
        ("C1=CNc2ccccc2C=C1", "1H-1-benzazepine"),
        ("C1=CNC=c2ccccc2=C1", "2H-2-benzazepine"),
        ("C1=Cc2ccccc2C=CN1", "3H-3-benzazepine"),
        # aromatic-NH indicated-H (locant by heteroatom seniority)
        ("c1cc2occc2[nH]1", "4H-furo[3,2-b]pyrrole"),
        ("c1cc2sccc2[nH]1", "4H-thieno[3,2-b]pyrrole"),
        ("c1cc2ccoc2[nH]1", "6H-furo[2,3-b]pyrrole"),
        # higher hydrocarbon series
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
# fusion engine — [5]helicene (pentahelicene) is named by fusion nomenclature as
# dibenzo[c,g]phenanthrene, which is its PREFERRED IUPAC NAME. "Pentahelicene"/
# "[5]helicene" is NOT a retained name: "Polyhelicenes" begins the
# helicene series at SIX rings ("The series begins with six rings and not five
# rings...", the Blue Book), so a five-ring helix has no helicene
# name and degrades to the fusion PIN. dibenzo[c,g]phenanthrene is marked (PIN)
# verbatim at the Blue Book [ (c) — phenanthrene base, two benzo
# first-order attached components preferred to one naphtho]. The engine now
# BUILDS this correctly (earlier it over-matched a 4-ring benzo[c]phenanthrene
# core and had to fail closed); OPSIN round-trips the emitted name to the input
# InChIKey (0-wrong, machine-confirmed via scripts/diagnose.py). The suite
# autouse-DISABLES the validity gate, so re-enable it to assert production output.
# ---------------------------------------------------------------------------

@pytest.fixture
def _validity_gate_on(monkeypatch):
    import orthonym.namer as _namer
    monkeypatch.setattr(_namer, "_DISABLE_VALIDITY_GATE", False, raising=False)
    yield


@pytest.mark.unit
def test_pentahelicene_names_as_fusion_pin(_validity_gate_on):
    # PIN = dibenzo[c,g]phenanthrene (round-trip verified, 0-wrong); [5]helicene
    # has no retained helicene name (series starts at 6 rings,.
    assert name_compound("c1ccc2c(c1)ccc1ccc3ccc4ccccc4c3c12") == \
        "dibenzo[c,g]phenanthrene"


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
        #: 1-benzofuran is the PIN, the Blue Book)
        ("c1ccc2occc2c1", "1-benzofuran"),
        ("c1ccc2[nH]ccc2c1", "1H-indole"),
        ("c1ccc2ncccc2c1", "quinoline"),
        ("Cc1ccc2ccccc2c1", "2-methylnaphthalene"),
        ("Cc1ccc2cc3ccccc3cc2c1", "2-methylanthracene"),
    ],
)
def test_tier4_catalog_protection(smiles, expected):
    assert name_compound(smiles) == expected
