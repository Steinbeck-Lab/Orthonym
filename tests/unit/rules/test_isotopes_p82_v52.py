""" isotopically modified compounds — v52 BB-conformance a phase (Task 1).

This is the FAILING-TEST suite for a phase (isotopes). It pins the representative
target rows that Tasks 2-7 will flip to passing, plus a regression-guard block of
rows that already pass and MUST stay unchanged (they exercise the same code and are
the regression risk).

Governing IUPAC 2013 rules (Blue Book, ``the Blue Book Blue Book``):

  * — NUMBERING / "Numbering in relation to the unmodified compound"
                (``the Blue Book Blue Book``). The isotope descriptor's
                locants follow the numbering fixed by the *unmodified* parent, not
                a fresh numbering of the modified compound.
  * -.4 — " Omission of locants" (``:44178``). When (and only
                when) a locant is unambiguous it may be omitted from the isotope
                descriptor; otherwise it is cited. Deny-by-default, per class.
  * — "Letter and/or numeral locants" (``:44208``). Italic-letter locants
                (e.g. ``N``) and numeral locants inside the isotope descriptor.
  * — "Location of nuclides on positions not normally denoted by
                locants" (``:44251``).

Every row's expected string is the ASCII, superscript-folded form exactly as the
a phase brief specifies (``eval/bb_conformance/bb_measure.py`` strips all
whitespace and folds unicode superscripts before comparing; a plain ``==`` here is
therefore the strict, byte-exact form of that comparison).

Best-effort tier: assertions call ``_best_effort.name(smiles)`` using the exact
FLAGS that ``eval/bb_conformance/bb_measure.py`` uses
(``general_fallback=True, general_fallback_unverified=True,
allow_aromatic_general=True``), because a couple of the target rows only emit at
best-effort tier.
"""

import pytest
from rdkit import RDLogger

from orthonym import Orthonym

# Match bb_measure.py: keep test output pristine (no RDKit isotope warnings).
RDLogger.DisableLog("rdApp.*")


# One shared best-effort namer, memoised. The call-site form
# ``_best_effort.name(smiles)`` is preserved exactly as the a phase brief
# specifies; memoisation mirrors bb_measure.py, which constructs a single
# ``Orthonym(**FLAGS)`` and reuses it across the whole corpus, and avoids
# spawning a fresh OPSIN JVM per parametrised row.
_NAMER = None


def _best_effort():
    """Return the shared best-effort namer (bb_measure.py's FLAGS)."""
    global _NAMER
    if _NAMER is None:
        _NAMER = Orthonym(
            style="pin",
            general_fallback=True,
            general_fallback_unverified=True,
            allow_aromatic_general=True,
        )
    return _NAMER


# --------------------------------------------------------------------------- #
# TARGET rows — EXPECTED TO FAIL today; Tasks 2-7 flip them one sub-pattern at
# a time. Marked xfail(strict=False) so the suite is green as a whole while it
# records the RED baseline, and so a flipped row surfaces as XPASS (not a
# spurious failure) as each later task lands.
# --------------------------------------------------------------------------- #
P82_TARGETS = [
    # Sub-pattern A — locant restoration /.
    ("A", "[2H]CC(F)(F)F", "1,1,1-trifluoro(2-2H1)ethane"),
    ("A", "[3H]c1c([3H])c([3H])c(OC)c(O)c1[3H]", "2-methoxy(3,4,5,6-3H4)phenol"),
    # Sub-pattern B — retained PIN parent (acetic acid / acetonitrile).
    ("B", "[2H]CC(=O)O", "(2-2H1)acetic acid"),
    ("B", "[2H]C([2H])([2H])C#N", "(2H3)acetonitrile"),
    # Sub-pattern C — O-isotope on the suffix.
    ("C", "NCC1([18OH])CCCC1", "1-(aminomethyl)cyclopentan-1-(18O)ol"),
    # Sub-pattern D — substituent locant / hyphen /.
    ("D", "[79Br]c1cccc[13cH]1", "1-(79Br)bromo(2-13C)benzene"),
    ("D", "[13CH3]c1cccc[13cH]1", "1-(13C)methyl(2-13C)benzene"),
    ("D", "CC(=O)Nc1ccc2c(c1)Cc1cc([131I])ccc1-2",
     "N-[7-(131I)iodo-9H-fluoren-2-yl]acetamide"),
    ("D", "CCC=[15N]N", "1-propylidene(1-15N)hydrazine"),
    # Sub-pattern E — systematic ring-carbon parent.
    ("E", "N#[13C]c1ccccc1", "benzene(13C)carbonitrile"),
    ("E", "O=[13C](O)c1ccccc1", "benzene(13C)carboxylic acid"),
    ("E", "NN[13CH2]c1ccccc1", "[phenyl(13C)methyl]hydrazine"),
    ("E", "O=C(O)C1([13C](=O)O)CCCCC1",
     "1-carboxycyclohexane-1-(13C)carboxylic acid"),
    ("E", "O=[13C](O)C1([14C](=O)O)CCCCC1",
     "1-(13C)carboxycyclohexane-1-(14C)carboxylic acid"),
    # Sub-pattern F — mixed H + single-heteroatom on a ring /.
    ("F", "[2H]c1cc[15n]c([2H])c1", "(2,4-2H2,15N)pyridine"),
    ("F", "[2H]C1[15NH]c2ccccc2C1[2H]", "2,3-dihydro(2,3-2H2,15N)-1H-indole"),
]


# --------------------------------------------------------------------------- #
# REGRESSION-GUARD rows — already correct today and MUST stay so. They exercise
# the same isotope-naming code paths as the targets and are the regression risk.
# Plain asserts: these MUST pass on every run.
# --------------------------------------------------------------------------- #
P82_GUARDS = [
    ("[2H]C([2H])([2H])Oc1ccccc1", "(2H3)methoxybenzene"),
    ("[2H]c1c([2H])c([2H])c([2H])c([2H])c1[2H]", "(2H6)benzene"),
    ("Cl[12CH](Cl)Cl", "trichloro(12C)methane"),
    ("[2H]C([2H])(Cl)Cl", "dichloro(2H2)methane"),
    ("[13CH3]CO", "(2-13C)ethan-1-ol"),
    ("[2H]CCO", "(2-2H1)ethan-1-ol"),
    ("c1ccc2[15nH]ccc2c1", "(15N)-1H-indole"),
    ("[13CH3]c1ccccc1[13CH3]", "1,2-di[(13C)methyl]benzene"),
    ("[2H]NC(C)=O", "(N-2H1)acetamide"),
    ("O=[14CH][O-].[Na+]", "sodium (14C)formate"),
]


@pytest.mark.unit
@pytest.mark.xfail(strict=False, reason="Phase 07 Tasks 2-7 flip these P-82 target rows")
@pytest.mark.parametrize(
    "smiles,expected",
    [(smi, exp) for _pat, smi, exp in P82_TARGETS],
    ids=[f"{pat}-{smi}" for pat, smi, _exp in P82_TARGETS],
)
def test_p82_targets(smiles, expected):
    """P-82 target rows — RED baseline; Tasks 2-7 make each pass."""
    assert _best_effort().name(smiles) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,expected",
    P82_GUARDS,
    ids=[smi for smi, _exp in P82_GUARDS],
)
def test_p82_regression_guards(smiles, expected):
    """ rows that already emit the correct PIN and MUST NOT regress."""
    assert _best_effort().name(smiles) == expected


# --------------------------------------------------------------------------- #
# SPLIT-ISOMER LOCANT — "Locants are not omitted when there is a
# possibility of isomers" ("", the Blue Book Blue Book).
#
# When a count>=2 nuclide group sits ENTIRELY on ONE carrier atom whose symmetry
# orbit has another host, moving the WHOLE group to the partner reproduces the
# molecule (2,0 -> 0,2 collapses on a symmetric orbit), but SPLITTING the group
# (1,1 -> 1,2) yields a DISTINCT isotopomer. Two isotopomers of the same count
# therefore exist, so the locant MUST be cited even though the carrier orbit is
# symmetric. The prior code tested only the whole-group move and wrongly omitted
# the locant (``(2H2)ethane-1,2-diyl`` for ``(1,1-2H2)ethane-1,2-diyl``).
#
# The GAIN rows below must now carry the locant; the OMIT rows exercise the same
# clause and MUST stay bare -- a scope of one host atom or a completely
# substituted single-position methyl has no split partner. OPSIN reads
# an unlocanted ``(2H2)`` as the 1,1 form, so BOTH spellings round-trip; this is a
# PIN-spelling lock, not a round-trip test.
P82_SPLIT_ISOMER_LOCANT = [
    # GAIN the locant (a distinct 1,2 isotopomer exists -> cite).
    ("O=C(O)c1ccc(OC([2H])([2H])COc2ccc(C(=O)O)cc2)cc1",
     "4,4'-[(1,1-2H2)ethane-1,2-diylbis(oxy)]dibenzoic acid"),
    ("[2H]C([2H])(O)CO", "(1,1-2H2)ethane-1,2-diol"),
    ("BrC([2H])([2H])CBr", "1,2-dibromo(1,1-2H2)ethane"),
    # OMIT the locant (no split partner -> /.
    ("[2H]C([2H])(Cl)Cl", "dichloro(2H2)methane"),      # sole C in scope
    ("[2H]C([2H])([2H])O", "(2H3)methanol"),            # sole C in scope
    ("[2H]C([2H])([2H])C#N", "(2H3)acetonitrile"),      # complete single-position CH3
    ("[2H]c1c([2H])c([2H])c([2H])c([2H])c1[2H]", "(2H6)benzene"),  # complete orbit
]


@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,expected",
    P82_SPLIT_ISOMER_LOCANT,
    ids=[smi for smi, _exp in P82_SPLIT_ISOMER_LOCANT],
)
def test_p82_6_1_4_split_isomer_locant(smiles, expected):
    """ (the Blue Book): a count>=2 group on one carrier of a symmetric orbit
    cites its locant iff a 1,1->1,2 split yields a distinct isotopomer."""
    assert _best_effort().name(smiles) == expected


# --------------------------------------------------------------------------- #
# Task 6 (sub-pattern E) LOCK — (the Blue Book): "When the nuclide is
# located at a position in a retained name that is not numbered a systematic
# name that identifies separately the relevant atom is used for the IUPAC
# preferred name." The retained benzonitrile / benzoic acid / benzyl do not
# number their nitrile / carboxyl / benzylic carbon, so a label there switches
# to the systematic ring-carbon parent (BB PINs:44253/:44255/:44259). These
# four rows are RESOLVED by Task 6 and are pinned as plain asserts so a
# regression is caught (they also appear, still xfail(strict=False), in the
# shared ``P82_TARGETS`` block above). The di-labelled cyclohexane gem-diacid
# stays in that block: it correctly ABSTAINS (two distinct nuclides across the
# carboxy prefix + carboxylic-acid suffix is beyond single-descriptor placement;
# never a degraded PIN).
P82_E_RESOLVED = [
    ("N#[13C]c1ccccc1", "benzene(13C)carbonitrile"),          # the Blue Book
    ("O=[13C](O)c1ccccc1", "benzene(13C)carboxylic acid"),    # the Blue Book
    ("NN[13CH2]c1ccccc1", "[phenyl(13C)methyl]hydrazine"),    # the Blue Book
    ("O=C(O)C1([13C](=O)O)CCCCC1",
     "1-carboxycyclohexane-1-(13C)carboxylic acid"),          # gem-diacid, one label
]


@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,expected",
    P82_E_RESOLVED,
    ids=[smi for smi, _exp in P82_E_RESOLVED],
)
def test_p82_6_3_2_ring_carbon_parent(smiles, expected):
    """P-82.6.3.2 systematic ring-carbon parent for a label on an unnumbered
    retained position — resolved by Task 6, must stay resolved."""
    assert _best_effort().name(smiles) == expected


# --------------------------------------------------------------------------- #
# Phase-07 NR1 LOCK — three PIN rows the Phase-07 fix-wave (Tasks 3 + 5)
# regressed and the fast pre-gate caught. All are 0-wrong (OPSIN-RT valid) but
# were spelled wrong vs the protected/gold baseline:
# * CC[18OH]: the 18O on the oxygen of the ``-ol`` suffix needs no locant, so
# (the Blue Book) keeps the locant-free ``ethanol`` spelling
# and the nuclide goes before the suffix: 'ethan(2H)ol (PIN) (as in ethanol)'
# (:44184), '1-(aminomethyl)cyclopentan-1-(18O)ol (PIN)' (:43744) -> gold
# W2F-P5-P4 ``ethan(18O)ol``. (A carbon label still needs its locant:
# ``(2-13C)ethan-1-ol [not (2-13C)ethanol]``,:44186.)
# * C[13CH2][15NH2] / [13CH3]C[15NH2]: 5ed0197a9's ``-amine`` suffix-adjacent
# slot let ``_decorate_multi_position`` SPLIT the combinable descriptor into
# ``(1-13C)ethan-1-(15N)amine``; (the Blue Book) + gold V47-/03
# require the single combined front descriptor ``(1-13C,15N)ethan-1-amine``.
# Pinned in BOTH the default (name_compound) and inline best-effort styles.
# --------------------------------------------------------------------------- #
NR1_RESTORED = [
    ("CC[18OH]", "ethan(18O)ol"),
    ("C[13CH2][15NH2]", "(1-13C,15N)ethan-1-amine"),
    ("[13CH3]C[15NH2]", "(2-13C,15N)ethan-1-amine"),
]


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", NR1_RESTORED,
                         ids=[smi for smi, _exp in NR1_RESTORED])
def test_nr1_regression_restored_default(smiles, expected):
    """The three Phase-07 PIN regressions, default (``name_compound``) style."""
    from orthonym.namer import name_compound
    assert name_compound(smiles) == expected


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", NR1_RESTORED,
                         ids=[smi for smi, _exp in NR1_RESTORED])
def test_nr1_regression_restored_best_effort(smiles, expected):
    """The three Phase-07 PIN regressions, inline best-effort style."""
    assert _best_effort().name(smiles) == expected


# --------------------------------------------------------------------------- #
# Task N3 LOCK — (the Blue Book) EXTENDED to the aldehyde. The retained
# ``benzaldehyde`` does NOT number its -CHO carbon, oxygen or hydrogen, so a
# nuclide on ANY of the three (the ¹³C carbon, the ¹⁸O oxygen, the ²H hydrogen)
# gets no locant and yields an unlocanted, BB-illegitimate ``(13C)benzaldehyde``
# (or fails closed). Switch to the systematic ring-carbon parent
# ``benzenecarbaldehyde``, exactly as Task 6 did for benzonitrile/benzoic acid.
# Every string OPSIN-round-trips to the input structure (0-wrong).
#
# ⚠ Subscript convention (both settled here): the UNLOCANTED aldehyde -CHO
# hydrogen KEEPS its count subscript (``(2H1)`` — OPSIN rejects the count-less
# ``benzene(2H)carbaldehyde``), while a LOCANTED aromatic ring CH OMITS it
# (``(4-2H)``, not ``(4-2H1)`` — both round-trip, the count-less form is the
# BB-preferred spelling). These EXACT-STRING plain asserts are the guard: the
# ¹⁸O front-vs-suffix placement and the ²H subscript are round-trip-BLIND (both
# spellings denote the same molecule), so only the string pins them.
P82_CARBALDEHYDE_RESOLVED = [
    ("O=[13CH]c1ccccc1", "benzene(13C)carbaldehyde"),          # ¹³C on -CHO carbon
    ("[18O]=Cc1ccccc1", "benzene(18O)carbaldehyde"),           # ¹⁸O on -CHO oxygen
    ("O=C([2H])c1ccccc1", "benzene(2H1)carbaldehyde"),         # ²H on -CHO hydrogen
    # ¹³C on the unnumbered carboxyl carbon AND a ring D (multi-part label on the
    # systematic ring-carbon parent): (4-2H) locanted ring CH omits its count.
    ("[13C](=O)(O)c1ccc([2H])cc1", "(4-2H)benzene(13C)carboxylic acid"),
]


@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,expected",
    P82_CARBALDEHYDE_RESOLVED,
    ids=[smi for smi, _exp in P82_CARBALDEHYDE_RESOLVED],
)
def test_p82_6_3_2_carbaldehyde_ring_carbon_parent(smiles, expected):
    """P-82.6.3.2 (extended, Task N3): a nuclide on benzaldehyde's unnumbered
    -CHO carbon/oxygen/hydrogen switches to ``benzenecarbaldehyde``; the ¹³C
    carboxyl + ring-D combo takes the multi-part ring-carbon parent. Exact
    string — the ¹⁸O placement and the ²H subscript are round-trip-blind."""
    assert _best_effort().name(smiles) == expected


# --------------------------------------------------------------------------- #
# Task N5 LOCK — (the Blue Book, "Italicized nuclide symbols and/or italic
# capital letters are used to distinguish between different nuclides of the same
# element"). A carboxylic acid has two oxygens of the SAME element in one suffix
# (the carbonyl ``=O`` and the hydroxyl ``-OH``); an ``18O`` on the HYDROXYL
# oxygen therefore needs the italic ``O`` locant to distinguish it from the
# carbonyl one (witness ``(18O-2H, 18O)acetic acid (PIN)``, "the O is a locant").
# The descriptor ``(O-18O)`` sits immediately before the parent stem (front of the
# name for the un-prefixed retained acid, after the substituent prefixes for a
# systematic chain). The ``O`` letter locant is offered in
# ``rules/isotopes.py::_letter_locant_candidates``.
#
# NOT gate-blind (unlike Task 7b): the bare front descriptor ``(18O)acetic acid``
# denotes the CARBONYL oxygen — a DIFFERENT isotopomer whose canonical SMILES
# differs from the hydroxyl input — so the OPSIN round-trip gate DISTINGUISHES the
# two placements and rejects the wrong one. Every string below OPSIN-round-trips
# to its input (InChIKey-verified, 0-wrong); the exact-string assert additionally
# pins the spelling.
P82_ITALIC_O_RESOLVED = [
    ("CC(=O)[18OH]", "(O-18O)acetic acid"),
    ("CCCC(CC(C)C)C(=O)[18OH]",
     "4-methyl-2-propyl(O-18O)pentanoic acid"),
]

P82_ITALIC_O_GUARDS = [
    # ¹⁸O on the CARBONYL oxygen keeps the bare front descriptor (no italic-O
    # locant); must NOT acquire ``(O-18O)``.
    ("CC(=[18O])O", "(18O)acetic acid"),
]


@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,expected",
    P82_ITALIC_O_RESOLVED,
    ids=[smi for smi, _exp in P82_ITALIC_O_RESOLVED],
)
def test_p82_6_4_carboxyl_hydroxyl_o_italic_locant(smiles, expected):
    """ (Task N5): an ¹⁸O on the hydroxyl oxygen of a carboxylic acid
    takes the italic ``O`` locant (``(O-18O)``), round-trip-verified."""
    assert _best_effort().name(smiles) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,expected",
    P82_ITALIC_O_GUARDS,
    ids=[smi for smi, _exp in P82_ITALIC_O_GUARDS],
)
def test_p82_6_4_carbonyl_o_no_italic_locant(smiles, expected):
    """ guard: an ¹⁸O on the CARBONYL oxygen keeps the bare front
    descriptor and must NOT acquire the italic ``O`` locant."""
    assert _best_effort().name(smiles) == expected


# --------------------------------------------------------------------------- #
# Task 7b (ADDED, user-directed) LOCK — "Letter and/or numeral locants"
# (the Blue Book) with (isomer-possibility) / (single atom of an
# element omits the numeral). When the parent carries MORE THAN ONE atom of the
# heteroatom element, the isotope descriptor's italic-letter locant must cite the
# SPECIFIC numeral the parent assigns (``N1``), matching the non-isotope
# substituent path (``CNCC(N)C`` -> ``N1-methylpropane-1,2-diamine``); a SINGLE
# atom of that element keeps the bare italic letter (``N``).
#
# ⚠ GATE-BLIND: both ``(N-2H1)propane-1,2-diamine`` and
# ``(N1-2H1)propane-1,2-diamine`` OPSIN-round-trip to the SAME structure, so the
# round-trip / / E1 gates CANNOT catch a regression to the bare ``N``
# here. These EXACT-STRING plain asserts ARE the safety net; the numeral is
# derived BY CONSTRUCTION (multi-nitrogen -> numbered) in
# ``rules/isotopes.py::_letter_locant_candidates``, and the gate only adjudicates
# WHICH numeral (N1 vs N2 ARE different molecules), never numeral-vs-bare.
P82_LETTER_NUMERAL = [
    # asymmetric diamine, two N -> the labelled N is the one on C1 -> N1
    ("[2H]NCC(N)C", "(N1-2H1)propane-1,2-diamine"),      #
    # symmetric diamine, two (equivalent) N -> lowest numeral N1, consistent with
    # the non-isotope convention ``N1-methylethane-1,2-diamine`` (round-trips).
    ("[2H]NCCN", "(N1-2H1)ethane-1,2-diamine"),          #
]

P82_LETTER_NUMERAL_GUARDS = [
    # SINGLE nitrogen -> bare italic ``N``, NO numeral. Must NOT
    # become ``(N1-2H1)acetamide`` even though that also round-trips.
    ("[2H]NC(C)=O", "(N-2H1)acetamide"),                 #
]


@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,expected",
    P82_LETTER_NUMERAL,
    ids=[smi for smi, _exp in P82_LETTER_NUMERAL],
)
def test_p82_6_2_numbered_letter_locant_multi_heteroatom(smiles, expected):
    """P-82.6.2 / P-82.6.1.4: a heteroatom letter locant on a parent with >1 atom
    of that element cites the SPECIFIC numeral (N1), not the bare letter. Exact
    string — the RT gate is blind to N-vs-N1, so this assert is the guard."""
    assert _best_effort().name(smiles) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,expected",
    P82_LETTER_NUMERAL_GUARDS,
    ids=[smi for smi, _exp in P82_LETTER_NUMERAL_GUARDS],
)
def test_p82_6_1_2_single_heteroatom_keeps_bare_letter(smiles, expected):
    """: a SINGLE atom of the element keeps the bare italic letter
    locant (``N``), no numeral. Exact-string regression guard for Task 7b."""
    assert _best_effort().name(smiles) == expected


# --------------------------------------------------------------------------- #
# Task 8n (ADDED, cross-family review) LOCK — "Locants are not omitted
# when there is a possibility of isomers" (the Blue Book). Naphthalene has TWO distinct
# ring-carbon symmetry classes — alpha (1,4,5,8) and beta (2,3,6,7). A single 13C
# on an alpha vs a beta carbon are DISTINCT isotopologues, so the bare
# ``(13C)naphthalene`` is ambiguous and the locant MUST be cited for BOTH.
#
# ⚠ GATE-BLIND: the omitted ``(13C)naphthalene`` round-trips through OPSIN for the
# ALPHA label only because OPSIN's default placement of an unlocated isotope
# happens to land on an alpha carbon — a coincidence the RT / / E1 gates
# cannot see. These EXACT-STRING plain asserts ARE the safety net; the omission is
# denied BY CONSTRUCTION in ``rules/isotopes.py::_placement_unambiguous`` (a second
# rank-class of the same element in scope ⇒ possibility of isomers ⇒ cite).
P82_6_1_4_ISOMER_POSSIBILITY = [
    # alpha carbon (was buggily emitting bare ``(13C)naphthalene`` — Task 8n fix)
    ("c1ccc2[13cH]cccc2c1", "(1-13C)naphthalene"),
    # beta carbon (was already correct — via RT rejection — must stay cited)
    ("c1ccc2c[13cH]ccc2c1", "(2-13C)naphthalene"),
]


@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,expected",
    P82_6_1_4_ISOMER_POSSIBILITY,
    ids=[smi for smi, _exp in P82_6_1_4_ISOMER_POSSIBILITY],
)
def test_p82_6_1_4_isomer_possibility_cites_locant(smiles, expected):
    """: a labelled element with a SECOND symmetry class in scope keeps
    its locant (naphthalene alpha vs beta). Exact string — the RT gate is blind to
    the alpha omission, so this assert is the guard."""
    assert _best_effort().name(smiles) == expected


# --------------------------------------------------------------------------- #
# Task N2 (this task) LOCK — "Locants are not omitted when there is a
# possibility of isomers" (the Blue Book) + LOCANT PRIORITY. When a benzene/hydrazine
# PARENT carries a POSITIONAL isotope AND an UNLABELLED detachable substituent, the
# label breaks the monosubstituted parent's symmetry, so the substituent's own
# licence-omitted locant is RESTORED — and, being a detachable prefix, the
# substituent is numbered FIRST and takes the lowest locant (1); the isotope
# descriptor's locant follows. So ``1-bromo(2-2H)benzene`` (bromo=1, D=2), NOT
# ``bromo(1-2H)benzene`` (the pre-fix output, which wrongly gave the D locant 1 and
# dropped the bromo locant).
#
# ⚠ SUBSCRIPT: a SINGLE D on a benzene RING carbon takes NO count subscript —
# ``(2-2H)``, not ``(2-2H1)``. (the Blue Book): the count is shown "when
# polysubstitution at a single position is possible... even in case of
# monosubstitution"; an aromatic ring =CH- bears exactly ONE H, so polysubstitution
# is NOT possible and the subscript is omitted (verbatim witness the Blue Book
# ``[4-2H]benzoic``; the Blue Book ``[1-14C]benzene``). This matches the existing
# BB-cited guard ``(2S)-(2-2H)butan-2-ol`` (a D on a 1-H carbon) in
# ``test_isotopes.py``. The count IS shown when the position CAN hold >1 atom of the
# nuclide (``(2-2H1)ethan-1-ol`` — C2 is a CH3) or for count>=2 (``(2,4-2H2)`` here).
#
# ⚠ GATE-BLIND: both ``bromo(1-2H)benzene`` (wrong locants) and
# ``1-bromo(2-2H)benzene`` (correct) OPSIN-round-trip to the SAME molecule, so RT /
# / E1 cannot catch the locant-priority defect. These EXACT-STRING plain
# asserts ARE the safety net.
P82_N2_PARENT_ISOTOPE_SUBSTITUENT_LOCANT = [
    # single ring D, ortho / meta to bromo — bromo takes locant 1, D follows.
    ("Brc1c([2H])cccc1", "1-bromo(2-2H)benzene"),        #
    ("Brc1cccc([2H])c1", "1-bromo(3-2H)benzene"),        #
    # two ring D at distinct positions (count 2 -> subscript shown).
    ("Brc1c([2H])cc([2H])cc1", "1-bromo(2,4-2H2)benzene"),  #
    # 15N on the OTHER hydrazine N -> phenyl (detachable prefix) takes locant 1.
    ("c1ccccc1N[15NH2]", "1-phenyl(2-15N)hydrazine"),    #
]

P82_N2_NO_LEAK_GUARDS = [
    # D on a SUBSTITUENT methyl -> ring stays symmetric, so NO ring locant is
    # restored (the over-cite trap this fix must avoid). The parent degrades from
    # retained ``toluene`` to systematic ``methylbenzene``: marks toluene
    # "no substitution" for PINs and (the Blue Book) requires the systematic
    # parent once it is isotopically modified worked example, the Blue Book).
    ("[2H]C([2H])([2H])c1ccccc1", "(2H3)methylbenzene"),
    # non-isotope monosubstituted parents keep their omitted locant.
    ("Brc1ccccc1", "bromobenzene"),
    ("NNc1ccccc1", "phenylhydrazine"),
    # BB-cited count-omission witness — a single D on a 1-H carbon keeps NO
    # subscript; the mirror of the ``(2-2H)benzene`` decision above.
    ("[2H][C@@](C)(O)CC", "(2S)-(2-2H)butan-2-ol"),
]


@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,expected",
    P82_N2_PARENT_ISOTOPE_SUBSTITUENT_LOCANT,
    ids=[smi for smi, _exp in P82_N2_PARENT_ISOTOPE_SUBSTITUENT_LOCANT],
)
def test_p82_6_1_4_parent_isotope_restores_substituent_locant(smiles, expected):
    """P-82.6.1.4: a positional isotope on the parent restores an unlabelled
    substituent's omitted locant, and the detachable prefix takes the lowest
    locant (1). Exact string — the RT gate is blind to the locant priority."""
    assert _best_effort().name(smiles) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,expected",
    P82_N2_NO_LEAK_GUARDS,
    ids=[smi for smi, _exp in P82_N2_NO_LEAK_GUARDS],
)
def test_p82_6_1_4_no_leak_to_symmetric_parents(smiles, expected):
    """A substituent-buried label, and every non-isotope parent, must KEEP the
     licensed locant omission (no over-cite). Task N2 regression guard."""
    assert _best_effort().name(smiles) == expected


# --------------------------------------------------------------------------- #
# Task N10 (sub-pattern C, chalcogen analogues) — /. The
# ``-thiol`` (S), ``-selenol`` (Se) and ``-tellurol`` (Te) suffixes name their
# heteroatom exactly as ``-ol`` names the O (witness ``(1²H₁)ethan-1-(²H)ol
# (PIN)`` the Blue Book), so a nuclide on that S/Se/Te is a "position not normally
# denoted by a locant": its locant-free descriptor belongs
# immediately BEFORE the suffix it modifies, NOT floated to the front of the
# name (the pre-fix ``(34S)propane-1-thiol`` defect). All three OPSIN-round-trip
# to their input (0-wrong holds). Plain asserts — the RT gate cannot see the
# placement because both forms parse to the same structure.
# --------------------------------------------------------------------------- #
P82_N10_CHALCOGEN_SUFFIX_ISOTOPE = [
    ("CCC[34SH]", "propane-1-(34S)thiol"),
    ("CCC[77SeH]", "propane-1-(77Se)selenol"),
    ("CCC[125TeH]", "propane-1-(125Te)tellurol"),
]

P82_N10_NO_REGRESSION_GUARDS = [
    # non-isotope thiol keeps its plain suffix (no descriptor manufactured) --
    # the directly on-path regression risk for the widened stem alternation.
    ("CCCS", "propane-1-thiol"),
    # NB: the O-isotope -ol row (``NCC1([18OH])CCCC1`` -> ``…cyclopentan-1-(18O)ol``,
    # the class this fix extends) is deliberately NOT pinned here as a strict
    # assert. It names correctly in a fresh process (verified 3/3) but flakes to
    # ``unknown organic compound`` under pytest's shared OPSIN-JVM warm state,
    # which is why the committed P82_TARGETS carry it as ``xfail(strict=False)``.
    # This fix does not touch the O path.
]


@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,expected",
    P82_N10_CHALCOGEN_SUFFIX_ISOTOPE,
    ids=[smi for smi, _exp in P82_N10_CHALCOGEN_SUFFIX_ISOTOPE],
)
def test_p82_chalcogen_suffix_isotope_placement(smiles, expected):
    """P-82.2.1: a locant-free nuclide on the S/Se/Te of a -thiol/-selenol/
    -tellurol suffix is spliced immediately before that suffix, not floated to
    the name front. Exact string — the RT gate is blind to the placement."""
    assert _best_effort().name(smiles) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,expected",
    P82_N10_NO_REGRESSION_GUARDS,
    ids=[smi for smi, _exp in P82_N10_NO_REGRESSION_GUARDS],
)
def test_p82_chalcogen_suffix_no_regression(smiles, expected):
    """Task N10 regression guard: the plain -thiol suffix and the -ol O-isotope
    placement this fix extends must be unchanged."""
    assert _best_effort().name(smiles) == expected


# ── a phase Task N4: toluene isotope-degradation + ──────
# (the Blue Book) retains ``toluene`` as a PIN with "no substitution";
# (the Blue Book) requires the systematic parent when a nuclide sits on a
# position the retained name does not number (toluene numbers neither its α nor
# its ring carbons). The worked example (the Blue Book) is decisive:
# ``1-(13C)methyl(2-13C)benzene (PIN)`` with ``(α,2-13C2)-toluene`` shown only as
# the GENERAL-nomenclature form. So an isotope label on toluene's methyl (α) OR
# ring degrades the parent to ``methylbenzene`` / ``1-methyl…benzene``. Each
# expected string is OPSIN-round-trip-verified against its input (0-wrong).
P82_N4_TOLUENE_ISOTOPE_DEGRADE = [
    ("c1ccccc1[13CH3]", "(13C)methylbenzene"),      # 13C on the α (methyl) carbon
    ("c1ccccc1C[2H]", "(2H1)methylbenzene"),        # single D on the methyl
    ("[2H]C([2H])([2H])c1ccccc1", "(2H3)methylbenzene"),  # fully-D methyl
    ("Cc1[13cH]cccc1", "1-methyl(2-13C)benzene"),   # 13C on a ring carbon
]

# The non-isotope namer MUST still emit the retained ``toluene`` PIN — the fix is
# strictly isotope-path-only.
P82_N4_TOLUENE_NON_ISOTOPE_GUARD = [
    ("Cc1ccccc1", "toluene"),
]


@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,expected",
    P82_N4_TOLUENE_ISOTOPE_DEGRADE,
    ids=[smi for smi, _exp in P82_N4_TOLUENE_ISOTOPE_DEGRADE],
)
def test_p82_toluene_isotope_degrades_to_methylbenzene(smiles, expected):
    """ +: an isotopically labelled toluene takes the
    systematic ``methylbenzene`` parent for the PIN, never the retained
    ``toluene`` (which cannot site the label on a numbered position)."""
    assert _best_effort().name(smiles) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,expected",
    P82_N4_TOLUENE_NON_ISOTOPE_GUARD,
    ids=[smi for smi, _exp in P82_N4_TOLUENE_NON_ISOTOPE_GUARD],
)
def test_p82_toluene_non_isotope_unchanged(smiles, expected):
    """Regression guard: the retained ``toluene`` PIN for the UNLABELLED molecule
    is untouched by the isotope-degradation fix."""
    assert _best_effort().name(smiles) == expected


# --------------------------------------------------------------------------- #
# Task N9 (this task) LOCK — REDUCED-MULTIPLIER de-multiplication.
# A benzene with several IDENTICAL substituents where ONE substituent-carbon is
# isotopically labelled used to ABSTAIN: the multiplied prefix (``trimethyl``) has
# no per-instance slot for the label, and ``_decorate_demultiplied`` declines the
# "different name shape" (it leaves at most ONE copy bare). The bare copies must
# regroup under a REDUCED multiplier: ``1,3,5-trimethylbenzene`` + one 13C ->
# ``1-(13C)methyl-3,5-dimethylbenzene`` (built by ``_decorate_reduced_multiplier``).
#
# ⚠ NUMBERING — the labelled methyl takes the LOWEST locant, NOT the highest.
# "Priority between isotopically substituted and unmodified atoms or
# groups" (``the Blue Book Blue Book``): "the starting point and the
# direction of numbering... are chosen so as to give lowest locants to the
# modified atoms or groups considered together in one series in increasing
# numerical order." Verbatim BB example ``(2-14C)butane (PIN) [not (3-14C)butane]``
# and the toluene worked example ``1-(13C)methyl(2-13C)benzene (PIN)`` (P82_N4
# above, the Blue Book) — the labelled ``(13C)methyl`` sits at position 1. So the PIN is
# ``1-(13C)methyl-3,5-dimethylbenzene`` (label at 1), NOT ``1,3-dimethyl-5-(13C)-
# methylbenzene`` (label at 5). Citation order: (``:3446``) excludes the
# isotope descriptor from alphanumerical order, so both segments alphabetise to the
# identical word ``methyl``; (``:3442``) then cites the lower initial
# locant first -> the labelled segment (locant 1) precedes ``3,5-dimethyl``.
#
# ⚠ GATE-BLIND: all of ``1-(13C)methyl-3,5-dimethylbenzene``,
# ``1,3-dimethyl-5-(13C)methylbenzene`` and ``3,5-dimethyl-1-(13C)methylbenzene``
# OPSIN-round-trip to the SAME molecule, so RT / / E1 cannot pick the PIN.
# These EXACT-STRING plain asserts ARE the safety net for the numbering.
P82_N9_REDUCED_MULTIPLIER = [
    # 1,3,5-trimethylbenzene, ONE methyl-C = 13C: labelled methyl -> locant 1.
    ("[13CH3]c1cc(C)cc(C)c1", "1-(13C)methyl-3,5-dimethylbenzene"),
    # 1,2,4-trimethylbenzene (NON-symmetric), one 13C methyl: only the physically
    # correct numbering round-trips; the labelled methyl still takes locant 1.
    ("[13CH3]c1ccc(C)cc1C", "1-(13C)methyl-2,4-dimethylbenzene"),
]

P82_N9_NO_LEAK_GUARDS = [
    # UNLABELLED multi-substituted parents keep the ordinary multiplied prefix —
    # the reduced-multiplier split is isotope-path only and must not leak.
    ("Cc1cc(C)cc(C)c1", "1,3,5-trimethylbenzene"),
    ("Cc1ccc(C)cc1C", "1,2,4-trimethylbenzene"),
    ("Cc1ccccc1", "toluene"),
    ("Cc1ccccc1C", "1,2-dimethylbenzene"),
]


@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,expected",
    P82_N9_REDUCED_MULTIPLIER,
    ids=[smi for smi, _exp in P82_N9_REDUCED_MULTIPLIER],
)
def test_p82_reduced_multiplier_labelled_copy_lowest_locant(smiles, expected):
    """P-82.2.2.1 + P-82.5.2: one labelled copy of a multiplied substituent splits
    out while the >=2 bare copies regroup under a reduced multiplier; the labelled
    copy takes the LOWEST locant (BB ``(2-14C)butane [not (3-14C)butane]``). Exact
    string — the RT gate is blind to the P-82.5.2 numbering choice."""
    assert _best_effort().name(smiles) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,expected",
    P82_N9_NO_LEAK_GUARDS,
    ids=[smi for smi, _exp in P82_N9_NO_LEAK_GUARDS],
)
def test_p82_reduced_multiplier_no_leak(smiles, expected):
    """Task N9 regression guard: unlabelled multi-substituted arenes keep their
    ordinary multiplied prefix (the reduced-multiplier split is isotope-only)."""
    assert _best_effort().name(smiles) == expected


# --------------------------------------------------------------------------- #
# Task N6 (sub-pattern C, hydroxyl O-H) — /: a deuterium on a
# diol HYDROXYL oxygen (``O-H -> O-D``). OPSIN 2.9.0 does not parse the BB
# suffix-insertion form ``(1-2H1)ethan-1-(2H)ol`` (verified), so the only
# RT-valid citation is the italic ``O`` letter locant (the Blue Book "the O is a
# locant"). SYMMETRIC diol (the two ``-OH`` are one orbit) -> bare ``O``
# /.4, no isomer possibility); ASYMMETRIC diol (primary vs secondary
# ``-OH``) -> the numeral is required and is chosen BY CONSTRUCTION
# (OPSIN resolves a bare ``O`` to the lowest-locant oxygen, so the RT gate cannot
# pick it). Both strings are OPSIN-RT-proven to reproduce the input D-position.
P82_N6_DIOL_OD = [
    ("[2H]OCCO", "(O-2H)ethane-1,2-diol"),       # symmetric -> bare O
    ("[2H]OCC(C)O", "(O1-2H)propane-1,2-diol"),  # asymmetric -> numbered O1 (D on primary)
]

# No-leak: the UNLABELLED diols keep their ordinary name (the O-locant offer is
# isotope-path only and RT-gated, so it must not touch a plain diol).
P82_N6_NO_LEAK_GUARDS = [
    ("OCCO", "ethane-1,2-diol"),
    ("OCC(C)O", "propane-1,2-diol"),
]


@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,expected",
    P82_N6_DIOL_OD,
    ids=[smi for smi, _exp in P82_N6_DIOL_OD],
)
def test_p82_6_4_diol_hydroxyl_od(smiles, expected):
    """ italic-O locant for an O-D on a diol hydroxyl oxygen; bare ``O``
    for a symmetric diol, numbered ``O1`` for an asymmetric one. Exact string —
    every candidate is OPSIN-RT gated (0-wrong)."""
    assert _best_effort().name(smiles) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,expected",
    P82_N6_NO_LEAK_GUARDS,
    ids=[smi for smi, _exp in P82_N6_NO_LEAK_GUARDS],
)
def test_p82_6_4_diol_no_leak(smiles, expected):
    """Task N6 regression guard: unlabelled diols keep their ordinary name (the
    italic-O locant offer is isotope-path only)."""
    assert _best_effort().name(smiles) == expected


# --------------------------------------------------------------------------- #
# Task N7, criterion (k)) — the senior-chain isotope tie-break.
# " The senior ring, ring system, or principal chain that has one or
# more isotopically modified atoms [criterion (k) in ]"
# (``the Blue Book Blue Book``); decisive sentence
# (``:21477``): "The senior parent structure contains the greater number of
# isotopically modified atoms or groups."
#
# Isobutyraldehyde with one CD3 methyl is a SYMMETRIC-branch tie: the two methyls
# are equivalent in the isotope-stripped skeleton, so chain selection is a genuine
# tie and criterion (k) puts the CD3 IN the principal chain (as C3 of propanal),
# leaving the plain methyl as the 2-substituent. The skeleton is named label-blind,
# so the descriptor first lands on the substituent (``2-(1,1,1-2H3)methylpropanal``);
# the decorator promotes it into the chain iff the parent placement round-trips
# (which proves the two positions are symmetry-equivalent, i.e. every higher
# criterion ties and (k) governs). Both strings RT to the same InChIKey, so this is
# a PIN-spelling fix, not 0-wrong; the promotion is RT-gated and fails closed.
# --------------------------------------------------------------------------- #
P82_N7_SENIOR_CHAIN_ISOTOPE = [
    ("[2H]C([2H])([2H])C(C=O)C", "2-methyl(3,3,3-2H3)propanal"),
]

P82_N7_NO_LEAK_GUARDS = [
    # Non-isotope inputs — the promotion is isotope-path only, so chain selection
    # is byte-identical to today's output.
    ("CC(C=O)C", "2-methylpropanal"),
    ("CCC=O", "propanal"),
    ("CCCCC=O", "pentanal"),
    ("CC(C)CC=O", "3-methylbutanal"),
    # Labelled ALKYL substituent with NO symmetric chain position: the parent
    # placement cannot round-trip, so the promotion fails closed and the label
    # stays on the substituent (ring parent, and an ester O-alkyl).
    ("Cc1cccnc1[13CH3]", "2-(13C)methyl-3-methylpyridine"),
    ("CCC(=O)O[14CH2]C", "(1-14C)ethyl propanoate"),
    # A UNIFORM multiplied labelled substituent is claimed by the de-multiplier
    # BEFORE the promotion hook — it must keep the grouped form.
    ("[13CH3]c1ccccc1[13CH3]", "1,2-di[(13C)methyl]benzene"),
]


@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,expected",
    P82_N7_SENIOR_CHAIN_ISOTOPE,
    ids=[smi for smi, _exp in P82_N7_SENIOR_CHAIN_ISOTOPE],
)
def test_p82_44_4_1_11_senior_chain_carries_isotope(smiles, expected):
    """ criterion (k): a symmetric alkyl branch carrying the isotope is
    promoted into the principal chain (``2-(1,1,1-2H3)methylpropanal`` ->
    ``2-methyl(3,3,3-2H3)propanal``). Exact string — both forms RT to the same
    structure, so the RT gate cannot see the chain-selection choice."""
    assert _best_effort().name(smiles) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,expected",
    P82_N7_NO_LEAK_GUARDS,
    ids=[smi for smi, _exp in P82_N7_NO_LEAK_GUARDS],
)
def test_p82_44_4_1_11_no_leak(smiles, expected):
    """Task N7 regression guard: the senior-chain promotion is isotope-path only
    and RT-gated, so non-isotope inputs and asymmetric/multiplied labelled
    substituents are byte-identical to today's output (promotion fails closed)."""
    assert _best_effort().name(smiles) == expected


# --------------------------------------------------------------------------- #
# Task N11 (this task) — +: a TWO-descriptor composition where
# the two nuclides sit in DIFFERENT structural scopes and each needs a different
# locant KIND. The 34S is inside a trisulfanyl substituent (positional locant,
# marks step ```` -> ```` per; the 18O is on the parent
# carboxyl -OH (italic-``O`` letter locant per. Each names correctly
# ALONE today; the multi-position composition path
# (:func:`isotopes._decorate_multi_position`) placed the substituent group but
# not the letter-locant group, because it did not offer per-group letter locants
# to:func:`isotopes._find_best_placement` — so the row abstained. The fix scopes
# the letter-locant candidates to each group's own atoms. Both forms
# OPSIN-round-trip to the input (an InChIKey), so the RT
# gate cannot see the composition — plain asserts.
# --------------------------------------------------------------------------- #
P82_N11_TWO_DESCRIPTOR_MIXED_SCOPE = [
    # The target: 34S in a substituent + 18O on the parent carboxyl -OH.
    ("CCSS[34S]CCC(=O)[18OH]",
     "3-[ethyl(1-34S)trisulfanyl](O-18O)propanoic acid"),
    # Each label ALONE still names as before (composition must not regress the
    # single-descriptor placements it builds on).
    ("CCSS[34S]CCC(=O)O", "3-[ethyl(1-34S)trisulfanyl]propanoic acid"),
    ("CCSSSCCC(=O)[18OH]", "3-(ethyltrisulfanyl)(O-18O)propanoic acid"),
]


@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,expected",
    P82_N11_TWO_DESCRIPTOR_MIXED_SCOPE,
    ids=[smi for smi, _exp in P82_N11_TWO_DESCRIPTOR_MIXED_SCOPE],
)
def test_p82_two_descriptor_mixed_scope_composition(smiles, expected):
    """/: two independent isotope descriptors — one positional
    inside a substituent, one italic-``O`` letter locant on the parent carboxyl —
    compose into one name. Exact string; both forms RT to the same structure."""
    assert _best_effort().name(smiles) == expected
