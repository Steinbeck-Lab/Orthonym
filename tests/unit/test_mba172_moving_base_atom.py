"""Phase 172 — Moving-Base-Atom Migration Audit (MBA-01 / MBA-02).

Executable form of 172-CONTEXT + the PIN-strict gold MBA rows
(``). Exact-string equality against the
Blue-Book-cited PIN — NO string post-processing; the fixes are upstream
(parent_to_prefix aldehyde branch + benzene functionalized-chain emitter).

MBA pattern (HERITAGE §8 / BlueBookV2.md):
  * suffix side  — a terminal C-FG named as a suffix absorbs its central carbon
    INTO the chain (`-dioic`/`-dial`/`-dinitrile` chain-length math). Already
    correct (whole-graph perception); locked here against regression.
  * prefix side  — `-CHO` on a substituent chain is absorbed -> `oxo` at the
    terminal locant (P-66.6.1; `1-oxo...yl` acyl form is NON-PIN per the
    Table-28.1 note); `formyl` ONLY when on a ring (P-66.6.1.1.3, not absorbable).
"""

import pytest

from orthonym import Orthonym
from orthonym.assembly.substituent_naming import parent_to_prefix


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


# --- MBA-02 — absorbed-aldehyde substituent prefix carries the correct oxo locant ---

@pytest.mark.unit  # FIXED (172 WS-A): parent_to_prefix + benzene table emit {n}-oxo, not locant-less
def test_mba02_5_oxopentyl_benzoic_acid(namer):
    # -(CH2)4CHO substituent: CHO carbon absorbed -> 5-oxopentyl (oxo at C5).
    assert namer.name("OC(=O)c1ccc(CCCCC=O)cc1") == "4-(5-oxopentyl)benzoic acid"


@pytest.mark.unit
def test_mba02_4_oxobutyl_benzoic_acid(namer):
    assert namer.name("OC(=O)c1ccc(CCCC=O)cc1") == "4-(4-oxobutyl)benzoic acid"


@pytest.mark.unit
def test_mba02_2_oxoethyl_benzoic_acid(namer):
    # -CH2CHO: short chain still carries the locant (2-oxoethyl), per P-14.3.4.
    assert namer.name("OC(=O)c1ccc(CC=O)cc1") == "4-(2-oxoethyl)benzoic acid"


@pytest.mark.unit  # NEGATIVE test: -CHO on the ring carbon is NOT absorbable -> formyl (P-66.6.1.1.3)
def test_mba02_formyl_on_ring_preserved(namer):
    assert namer.name("OC(=O)c1ccc(C=O)cc1") == "4-formylbenzoic acid"


@pytest.mark.unit
@pytest.mark.parametrize("parent,clen,expected", [
    ("pentanal", 5, "5-oxopentyl"),   # was the non-PIN '1-oxopentyl'
    ("butanal", 4, "4-oxobutyl"),
    ("propanal", 3, "3-oxopropyl"),
    ("ethanal", 2, "2-oxoethyl"),
])
def test_mba02_parent_to_prefix_oxo_locant(parent, clen, expected):
    # The former -CHO carbon sits at the chain terminus opposite the attachment
    # (= chain_length), NOT at C1 (the disfavoured acyl position).
    assert parent_to_prefix(parent, chain_length=clen) == expected


# --- MBA-01 — suffix-side carbon-absorption chain-math (already correct; LOCK) ---

_MBA01_SUFFIX_PROTECT = [
    ("OC(=O)CCCC(=O)O", "pentanedioic acid"),     # 5-C diacid (both COOH absorbed)
    ("OC(=O)CCCCCC(=O)O", "heptanedioic acid"),   # 7-C diacid (pentanedioic<->heptanedioic fixture)
    ("O=CCCC=O", "butanedial"),                    # -dial: both CHO absorbed
    ("N#CCCC#N", "butanedinitrile"),               # -dinitrile: both nitrile C absorbed
    ("OC(=O)CCCCC=O", "6-oxohexanoic acid"),       # principal-chain CHO -> oxo, correct locant
]


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", _MBA01_SUFFIX_PROTECT)
def test_mba01_suffix_chain_math_locked(namer, smiles, expected):
    assert namer.name(smiles) == expected


# --- MBA-01 (SECONDARY) — ester carbon-absorption (WS-C). Armed/disarmed at ship. ---

@pytest.mark.unit
@pytest.mark.xfail(
    strict=True,
    reason="MBA-01 WS-C (172): ester carbonyl-carbon absorption -> 4-ethoxy-4-oxobutanoic acid. "
           "If WS-C lands, this XPASSES (strict) -> remove the marker; if deferred, it stays a tripwire.",
)
def test_mba01_ester_carbon_absorption(namer):
    # monoethyl succinate: ester C absorbed into the 4-C acid chain -> oxo + ethoxy,
    # NOT a mis-counted 'ethoxycarbonyl' prefix.
    assert namer.name("CCOC(=O)CCC(=O)O") == "4-ethoxy-4-oxobutanoic acid"
