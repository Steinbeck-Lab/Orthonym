"""a phase.7 functional-group perception fix — RED gold-target tripwires (DEF-5).

These are the executable form of CONTEXT: "flip exactly these gold target
rows to MATCH". Each is the 3 DEF-5 perception targets from the PIN-strict gold
oracle (``benchmarks/the gold set/gold_pins.json``). They are marked
``xfail(strict=True)`` so that the moment the functional-group perception fix lands (Plan 02 for
hydroxylamine/selenide, Plan 03 for the azide chain-exclusion) the test XPASSES,
which ``strict=True`` turns into a hard error — forcing the implementing plan to
REMOVE the marker and leave a permanent green tripwire.

NO band-aids: these assert exact-string equality against the Blue-Book-cited PIN.
There is no string post-processing here — the fix is upstream in perception.

Gold rows (def_id / SMILES / expected PIN / Blue Book rule):
  DEF-5 CCCNO -> N-propylhydroxylamine (P-68.3.1.2.1)
  DEF-5 CN=[N+]=[N-] -> azidomethane (P-66.4.1)
  DEF-5 CCC[Se]C -> 1-(methylselanyl)propane (P-63.6)
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
def test_azidomethane_not_diazabutane(namer):
    # DEF-5: azide is a prefix-only characteristic group (P-66.4.1 / P-65.5);
    # its N's must NOT be absorbed into the skeletal/parent chain. FIXED in 169.7
    # Plan-03 (structural chain-exclusion + prefix-only get_principal_group skip).
    # Permanent green tripwire.
    assert namer.name("CN=[N+]=[N-]") == "azidomethane"


@pytest.mark.unit  # a phase assembly/parenthesisation fix FIXED the enclosing parens (selanyl now classed complex)
def test_methyl_propyl_selenide(namer):
    # DEF-5: selenide is the Se analogue of an ether (P-63.6). 169.7 delivered the
    # perception + (methylselanyl) naming + locant; a phase assembly/parenthesisation fix added the
    # enclosing parens by adding selanyl/tellanyl to the complex-substituent suffixes.
    assert namer.name("CCC[Se]C") == "1-(methylselanyl)propane"
