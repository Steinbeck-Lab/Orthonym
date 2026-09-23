"""a phase functional-group perception fix — RED gold-target tripwires .

These are the executable form of internal notes: "flip exactly these gold target
rows to MATCH". Each is the 3 perception targets from the PIN-strict gold
oracle (``benchmarks/the gold set/gold_pins.json``). They are marked
``xfail(strict=True)`` so that the moment the functional-group perception fix lands (Plan 02 for
hydroxylamine/selenide, Plan 03 for the azide chain-exclusion) the test XPASSES,
which ``strict=True`` turns into a hard error — forcing the implementing plan to
REMOVE the marker and leave a permanent green tripwire.

NO band-aids: these assert exact-string equality against the Blue-Book-cited PIN.
There is no string post-processing here — the fix is upstream in perception.

Gold rows (def_id / SMILES / expected PIN / Blue Book rule):
    CCCNO -> N-propylhydroxylamine
    CN=[N+]=[N-] -> azidomethane
    CCC[Se]C -> 1-(methylselanyl)propane
"""

import pytest

from orthonym import Orthonym


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


@pytest.mark.unit
def test_propylhydroxylamine_perceived(namer):
    #: hydroxylamine (R-NH-OH) is a recognised class. FIXED in
    # 169.7 Plan-02 (hydroxylamine SMARTS + handler). Permanent green tripwire
    # against the ORIGINAL defect (the N-O silently dropped -> 'propane',
    # a wrong molecule) -- structure-correctness only.
    #
    # v52 P2, BB:38308/:38314) supersedes this row's
    # original expected string: R-NH-OH is named as an N-derivative of the
    # SENIOR AMINE, not the functional-class 'N-propylhydroxylamine' the
    # gold row asserted under the more general citation --
    # BB verbatim for the CH3 case, 'N-hydroxymethanamine (PIN)
    # N-methylhydroxylamine'. benchmarks/the gold set/gold_pins.json's
    # row (def_id, bluebook_ref carries the SAME stale
    # value and needs the identical correction -- flagged to the team lead
    # rather than edited here (shared gate-controlling file, outside this
    # task's file list).
    assert namer.name("CCCNO") == "N-hydroxypropan-1-amine"


@pytest.mark.unit
def test_azidomethane_not_diazabutane(namer):
    #: azide is a prefix-only characteristic group /;
    # its N's must NOT be absorbed into the skeletal/parent chain. FIXED in 169.7
    # Plan-03 (structural chain-exclusion + prefix-only get_principal_group skip).
    # Permanent green tripwire.
    assert namer.name("CN=[N+]=[N-]") == "azidomethane"


@pytest.mark.unit  # a phase assembly/parenthesisation fix FIXED the enclosing parens (selanyl now classed complex)
def test_methyl_propyl_selenide(namer):
    #: selenide is the Se analogue of an ether. 169.7 delivered the
    # perception + (methylselanyl) naming + locant; a phase assembly/parenthesisation fix added the
    # enclosing parens by adding selanyl/tellanyl to the complex-substituent suffixes.
    assert namer.name("CCC[Se]C") == "1-(methylselanyl)propane"
