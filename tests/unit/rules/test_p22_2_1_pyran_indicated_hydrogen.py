"""P-22.2.1: the pyran family keeps its indicated hydrogen on every chalcogen.

Task AA3.  ``BlueBookV2.md:8141``, section **P-22.2.1 "Retained names of
heteromonocycles"**, adjudicates all four chalcogens in one sentence and prints
the bare stem as *not* the PIN each time::

    pyran (2H-isomer shown; the PIN is 2H-pyran)
    thiopyran (S instead of O) (2H-isomer shown; the PIN is 2H-thiopyran)
    selenopyran (Se instead of O) (2H-isomer shown; the PIN is 2H-selenopyran)
    telluropyran (Te instead of O) (2H-isomer shown; the PIN is 2H-telluropyran)

Before the fix, S/Se/Te emitted the bare stem.  The cause was **not** a missing
rule: ``rules/heterocycles.py`` derives every one of these unaided (measured).
The bare stem came from ``data/opsin_imports/aryl_groups.py``, whose rows were
promoted into ``ALL_RETAINED_NAMES`` and short-circuited the dispatch before the
generic path ever ran.  Oxygen escaped only because a hand-curated ``2H-pyran``
row happened to overwrite the OPSIN row in the merge -- i.e. the one member that
worked was the one that never reached the rule.  The fix is three deny rows in
``data/iupac_2013_pin_list.json``, each carrying the citation above and a
verified replacement.

⚠ No round-trip oracle can protect this.  OPSIN resolves the bare ``thiopyran``
to the 2H isomer, so name -> structure -> InChIKey agrees for a name the Blue
Book prints as not-the-PIN.  These assertions are spelling assertions, checked
against the Blue Book by eye; that is the point of the file.
"""
import pytest

from orthonym.data import ALL_RETAINED_NAMES
from orthonym.namer import name_compound


# --------------------------------------------------------------------------
# 1. BlueBookV2.md:8141 -- the four chalcogen 2H-isomers ("the PIN is 2H-...")
# --------------------------------------------------------------------------

# Tellurium parses in RDKit only in bracket form; ``C1=CCTeC=C1`` is not a
# valid SMILES (``Te`` unbracketed is read as two atoms), so the Te member is
# cited here as ``C1=CC[Te]C=C1``.
PYRAN_2H = [
    ("C1=CCOC=C1", "2H-pyran"),
    ("C1=CCSC=C1", "2H-thiopyran"),
    ("C1=CC[Se]C=C1", "2H-selenopyran"),
    ("C1=CC[Te]C=C1", "2H-telluropyran"),
]


@pytest.mark.parametrize("smiles,expected", PYRAN_2H)
def test_2h_isomer_keeps_indicated_hydrogen(smiles, expected):
    assert name_compound(smiles) == expected


# The 4H tautomers were already correct and must not move.  They are the
# control: they prove the generic indicated-hydrogen path was live and working
# on sulfur all along, which is why adding table rows would have been the wrong
# fix -- there was no rule missing to add.
PYRAN_4H = [
    ("C1=COC=CC1", "4H-pyran"),
    ("C1=CSC=CC1", "4H-thiopyran"),
    ("C1=C[Se]C=CC1", "4H-selenopyran"),
]


@pytest.mark.parametrize("smiles,expected", PYRAN_4H)
def test_4h_tautomer_unchanged(smiles, expected):
    assert name_compound(smiles) == expected


# --------------------------------------------------------------------------
# 2. The root cause, guarded structurally
# --------------------------------------------------------------------------

BARE_PYRAN_STEMS = {"pyran", "thiopyran", "selenopyran", "telluropyran"}


def test_no_bare_pyran_stem_is_a_headline_retained_name():
    """No bare chalcogen pyran stem may be reachable as a whole-molecule PIN.

    This is the invariant, not the three molecules above.  ``aryl_groups.py`` is
    a generated OPSIN import: every row in it carries ``is_pin: False`` as a
    hard-coded generator default, so a re-import can silently re-add these rows.
    Only the adjudicated deny list in ``iupac_2013_pin_list.json`` keeps them
    out, and this test fails if that ever stops holding -- including for
    ``pyran`` itself, which is denied de facto by the hand-curated ``2H-pyran``
    row winning the merge.
    """
    offenders = {
        smi: name for smi, name in ALL_RETAINED_NAMES.items()
        if name.lower().strip() in BARE_PYRAN_STEMS
    }
    assert offenders == {}, (
        "BlueBookV2.md:8141 (P-22.2.1 'Retained names of heteromonocycles') "
        f"prints each of these as NOT the PIN: {offenders}"
    )


# --------------------------------------------------------------------------
# 3. The load-bearing hand-curated rows next door
# --------------------------------------------------------------------------

# Measured while scoping the fix: withdrawing the hand-curated rows for these
# three does NOT fall through to the retained PIN, it falls through to the
# Hantzsch-Widman systematic name -- ``1H-1,3-diazole`` for imidazole and
# ``1H-1,2-diazole`` for pyrazole.  So those rows are load-bearing and the
# "remove the redundant mask" cleanup is only safe for pyran.  Recorded as a
# test so the next person measures instead of assuming.
TABLE_2_2_INDICATED_H = [
    ("c1cc[nH]c1", "1H-pyrrole"),
    ("c1c[nH]cn1", "1H-imidazole"),
    ("c1cn[nH]c1", "1H-pyrazole"),
]


@pytest.mark.parametrize("smiles,expected", TABLE_2_2_INDICATED_H)
def test_other_table_2_2_indicated_hydrogen_rows(smiles, expected):
    assert name_compound(smiles) == expected
