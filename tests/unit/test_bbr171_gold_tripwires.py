"""Phase 171 BBR-PARENT/BBR-ASM/BBR-CHG — RED gold-target tripwires (DEF-1/3/4/8).

The executable form of 171-CONTEXT D-15 / the PIN-strict gold oracle
(``). Each target is marked
``xfail(strict=True)`` so the moment the owning workstream lands the fix the test
XPASSES, which ``strict=True`` turns into a hard error — forcing the implementing
wave to REMOVE the marker and leave a permanent green regression test.

NO band-aids: exact-string equality against the Blue-Book-cited PIN. No string
post-processing — the fixes are upstream (parent selection / assembly / charged router).

Phase-171 gold targets (def_id / SMILES / PIN / Blue Book / owning workstream):
  DEF-1  CCCCCCCc1ccccc1        -> heptylbenzene                         (P-44.1.2.2)  WS-1
  DEF-1  CCCCCCCCC1CCCCC1       -> nonylcyclohexane                      (P-44.1.2.2)  WS-1
  DEF-4  ClCCCCC                -> 1-chloropentane                       (P-14.3.4)    WS-3
  DEF-4  ClCC(F)C               -> 1-chloro-2-fluoropropane              (P-14.3.4)    WS-3
  DEF-4  FCCCl                  -> 1-chloro-2-fluoroethane               (P-14.3.4)    WS-3
  DEF-3  O=C(O)CCS(=O)(=O)[O-]  -> 2-carboxyethanesulfonate              (P-72.7/P-41) WS-2
  DEF-8  O=C(CCl)N(CCCl)CCCl    -> 2-chloro-N,N-bis(2-chloroethyl)acetamide (P-16.3.5) WS-3
  DEF-8  ClCCCCc1ccccc1         -> (4-chlorobutyl)benzene                (P-46)        WS-3
  DEF-8  CCC[Se]C               -> 1-(methylselanyl)propane              (P-16.3.5)    WS-3 (selenide parens)

DEF-6 (piperidin-4-one, 5-methylpyridin-2-ol) + DEF-7 (bicyclo[2.2.1]heptan-2-one)
are Phase 170 (ring construction) and are NOT armed here.
"""

import pytest

from orthonym import Orthonym


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


# --- DEF-1 — parent spine: ring senior to chain regardless of size (P-44.1.2.2), WS-1 ---

@pytest.mark.unit
@pytest.mark.xfail(strict=True, reason="WS-1 BBR-PARENT (171-03): delete the len(ring)>=chain_len size gate. Flip + remove marker when it lands.")
def test_def1_heptylbenzene(namer):
    assert namer.name("CCCCCCCc1ccccc1") == "heptylbenzene"


@pytest.mark.unit
@pytest.mark.xfail(strict=True, reason="WS-1 BBR-PARENT (171-03): ring senior regardless of size (P-44.1.2.2).")
def test_def1_nonylcyclohexane(namer):
    assert namer.name("CCCCCCCCC1CCCCC1") == "nonylcyclohexane"


# --- DEF-4 — locant-1 elision uses molecule-wide count (P-14.3.4), WS-3 ---

@pytest.mark.unit  # WS-3 BBR-ASM (171-02) FIXED: molecule-wide locant-1 count + Rule 5 chain==2
def test_def4_1_chloropentane(namer):
    assert namer.name("ClCCCCC") == "1-chloropentane"


@pytest.mark.unit  # WS-3 BBR-ASM (171-02) FIXED
def test_def4_1_chloro_2_fluoropropane(namer):
    assert namer.name("ClCC(F)C") == "1-chloro-2-fluoropropane"


@pytest.mark.unit  # WS-3 BBR-ASM (171-02) FIXED
def test_def4_1_chloro_2_fluoroethane(namer):
    assert namer.name("FCCCl") == "1-chloro-2-fluoroethane"


# --- DEF-3 — charged class-before-neutralize (P-72.7/P-41), WS-2 ---

@pytest.mark.unit
@pytest.mark.xfail(strict=True, reason="WS-2 BBR-CHG (171-04): classify the ionic FG class WITH the charge present so sulfonate stays senior.")
def test_def3_carboxyethanesulfonate(namer):
    assert namer.name("O=C(O)CCS(=O)(=O)[O-]") == "2-carboxyethanesulfonate"


# --- DEF-8 — assembly: bis()/enclosing marks + P-46 compound-substituent locants, WS-3 ---

@pytest.mark.unit  # WS-3 BBR-ASM (171-02) FIXED: one paren predicate + P-46 + amide locant
def test_def8_bis_chloroethyl_acetamide(namer):
    assert namer.name("O=C(CCl)N(CCCl)CCCl") == "2-chloro-N,N-bis(2-chloroethyl)acetamide"


@pytest.mark.unit  # WS-3 BBR-ASM (171-02) FIXED: P-46 attachment=locant-1 + enclosing parens
def test_def8_4_chlorobutylbenzene(namer):
    assert namer.name("ClCCCCc1ccccc1") == "(4-chlorobutyl)benzene"


@pytest.mark.unit  # WS-3 BBR-ASM (171-02) FIXED: selanyl/tellanyl now complex -> enclosing parens
def test_def8_methylselanyl_propane(namer):
    assert namer.name("CCC[Se]C") == "1-(methylselanyl)propane"


# --- Protect guard: the 11 PIN-strict protect rows must stay correct (HARD, not xfail) ---

_PROTECT = [
    ("CCO", "ethanol"),
    ("CC(=O)O", "acetic acid"),
    ("OC(=O)CCO", "3-hydroxypropanoic acid"),
    ("OC(=O)CCC(=O)O", "butanedioic acid"),
    ("CCC(=O)[O-].[K+]", "potassium propanoate"),
    ("C[N+](C)(C)CC(=O)[O-]", "(trimethylazaniumyl)acetate"),
    ("c1ccccc1", "benzene"),
    ("c1ccc2ccccc2c1", "naphthalene"),
    ("ClCC", "chloroethane"),
    ("CC(C)C", "2-methylpropane"),
    ("CCCCCCCCCC", "decane"),
]


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", _PROTECT)
def test_protect_rows_hold(namer, smiles, expected):
    # The 11 PIN-strict protect rows: every Phase-171 wave MUST keep these correct.
    # gold_pins.json may carry accept_also variants; this in-process guard asserts
    # the canonical PIN. If a wave regresses one, that is a HARD STOP (root-cause it).
    assert namer.name(smiles) == expected
