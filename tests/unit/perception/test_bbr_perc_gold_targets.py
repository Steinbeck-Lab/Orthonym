"""Phase 169.7 BBR-PERC — RED gold-target tripwires (DEF-5).

These are the executable form of CONTEXT D-15: "flip exactly these gold target
rows to MATCH". Each is the 3 DEF-5 perception targets from the PIN-strict gold
oracle (``). They are marked
``xfail(strict=True)`` so that the moment the BBR-PERC fix lands (Plan 02 for
hydroxylamine/selenide, Plan 03 for the azide chain-exclusion) the test XPASSES,
which ``strict=True`` turns into a hard error — forcing the implementing plan to
REMOVE the marker and leave a permanent green tripwire.

NO band-aids: these assert exact-string equality against the Blue-Book-cited PIN.
There is no string post-processing here — the fix is upstream in perception.

Gold rows (def_id / SMILES / expected PIN / Blue Book rule):
  DEF-5  CCCNO          -> N-propylhydroxylamine      (P-68.3.1.2.1)
  DEF-5  CN=[N+]=[N-]   -> azidomethane               (P-66.4.1)
  DEF-5  CCC[Se]C       -> 1-(methylselanyl)propane   (P-63.6)
"""

import pytest

from orthonym import Orthonym


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


@pytest.mark.unit
def test_propylhydroxylamine_perceived(namer):
    # DEF-5: hydroxylamine (R-NH-OH) is a recognised class (P-68.3). FIXED in
    # 169.7 Plan-02 (hydroxylamine SMARTS + handler). Permanent green tripwire.
    assert namer.name("CCCNO") == "N-propylhydroxylamine"


@pytest.mark.unit
@pytest.mark.xfail(
    strict=True,
    reason="169.7 BBR-PERC Plan-03: azide N's are walked into a fictional aza "
    "chain (CN=[N+]=[N-] -> '2,3-diazabutane'). Flips to PASS when the "
    "structural chain-exclusion lands; then remove this marker.",
)
def test_azidomethane_not_diazabutane(namer):
    # DEF-5: azide is a prefix-only characteristic group (P-66.4.1 / P-65.5);
    # its N's must NOT be absorbed into the skeletal/parent chain.
    assert namer.name("CN=[N+]=[N-]") == "azidomethane"


@pytest.mark.unit
@pytest.mark.xfail(
    strict=True,
    reason="169.7 BBR-PERC Plan-02 FIXED perception+naming: CCC[Se]C now ships "
    "'1-methylselanylpropane' (RT-passes; was 'unknown'). The ONLY residual gap "
    "vs the gold PIN is the enclosing parens '(methylselanyl)' — a DEF-8 "
    "needs-parens divergence (the chain handler omits the marks the polyfunctional "
    "path adds). That consolidation is Phase 171 / BBR-ASM (CONTEXT D-14 scope "
    "guard). Flip + remove this marker when Phase 171 lands the parens.",
)
def test_methyl_propyl_selenide(namer):
    # DEF-5: selenide is the Se analogue of an ether (P-63.6). 169.7 delivers the
    # perception + (methylselanyl) naming + locant + shipping; the parens are P-171.
    assert namer.name("CCC[Se]C") == "1-(methylselanyl)propane"
