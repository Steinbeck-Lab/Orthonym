"""P-16.7.1(a) -- vowel elision on the von Baeyer / cage parent path.

``polycyclic._build_parent_with_unsaturation`` decided elision from the BARE
suffix, and its unsaturated branch did not decide at all::

    if has_unsaturation:
        result = stem + ''.join(unsat_parts) + suffix_part   # no elision check
    else:
        first_char_of_suffix = suffix_text[0]                # the BARE suffix

``unsat_parts`` hardcodes ``-en``/``-yn``, so the terminal ``e`` of the
``ene``/``yne`` ending was dropped unconditionally, and the saturated branch
tested ``one`` when the *complete* suffix was ``dione``.

Blue Book, **P-16.7 ELISION OF VOWELS**, **P-16.7.1(a)**
(``BlueBookV2/BlueBookV2.md:7595``):

    "the terminal letter 'e' in names of parent hydrides or endings **'ene' and
    'yne'** when followed by a suffix or 'en' ending beginning with 'a', 'e',
    'i', 'o', 'u', or 'y'"

-- the rule names the ``ene``/``yne`` endings explicitly, so the unsaturated
branch is squarely in scope.  Restated at ``:25013``:

    "**If, and only if, the complete suffix** (that is, the suffix plus its
    multiplying prefixes, if any) begins with a vowel, a terminal letter 'e'
    (if any) of the parent hydride name is elided."

-- decisive that the test is on the multiplied suffix, not the bare one.
``dione``/``diol``/``thiol``/``tetrol`` all begin with a consonant, so the
``e`` is RETAINED.

The fix delegates both branches to ``naming_utils.format_suffix_with_locants``
(which already routes through ``_join_multiplied_suffix`` and the shared
``_ELISION_VOWELS``) rather than maintaining a second elision implementation.
"""

import pytest

from orthonym.rules.polycyclic import _build_parent_with_unsaturation


def _build(atoms, doubles=(), triples=(), suffix=None, locants=(), stype="inline"):
    unsat = {"double_bonds": list(doubles), "triple_bonds": list(triples)}
    fg = None
    if suffix is not None:
        fg = {"suffix": suffix, "locants": list(locants), "type": stype}
    return _build_parent_with_unsaturation(atoms, unsat, fg_suffix=fg)


# ---------------------------------------------------------------------------
# Consonant-initial COMPLETE suffix -> the terminal 'e' is RETAINED
# ---------------------------------------------------------------------------

class TestTerminalEIsRetainedBeforeAConsonantSuffix:

    @pytest.mark.unit
    def test_ene_before_dione(self):
        """The reported case: 'dione' starts with 'd', so 'ene' keeps its 'e'."""
        assert _build(8, doubles=[2], suffix="one", locants=[4, 8]) == \
            "oct-2-ene-4,8-dione"

    @pytest.mark.unit
    def test_ene_before_diol(self):
        assert _build(8, doubles=[2], suffix="ol", locants=[4, 8]) == \
            "oct-2-ene-4,8-diol"

    @pytest.mark.unit
    def test_ene_before_thiol(self):
        """Consonant-initial even unmultiplied."""
        assert _build(8, doubles=[2], suffix="thiol", locants=[4]) == \
            "oct-2-ene-4-thiol"

    @pytest.mark.unit
    def test_yne_before_dione(self):
        assert _build(8, triples=[2], suffix="one", locants=[4, 8]) == \
            "oct-2-yne-4,8-dione"

    @pytest.mark.unit
    def test_saturated_before_dione_tests_the_complete_suffix(self):
        """The ':25013' clause: the bare suffix is 'one' (a vowel) but the
        COMPLETE suffix is 'dione' (a consonant), so the 'e' is retained."""
        assert _build(5, suffix="one", locants=[2, 4]) == "pentane-2,4-dione"

    @pytest.mark.unit
    def test_saturated_before_tetrol(self):
        """tetra+ol elides to 'tetrol' (P-63.1.2), which begins with 't'."""
        assert _build(6, suffix="ol", locants=[1, 2, 3, 4]) == \
            "hexane-1,2,3,4-tetrol"


# ---------------------------------------------------------------------------
# Vowel-initial COMPLETE suffix -> the terminal 'e' is ELIDED (must not change)
# ---------------------------------------------------------------------------

class TestTerminalEIsElidedBeforeAVowelSuffix:

    @pytest.mark.unit
    @pytest.mark.parametrize("kwargs,expected", [
        (dict(atoms=8, doubles=[2], suffix="one", locants=[4]), "oct-2-en-4-one"),
        (dict(atoms=8, doubles=[2], suffix="ol", locants=[4]), "oct-2-en-4-ol"),
        (dict(atoms=8, triples=[2], suffix="one", locants=[4]), "oct-2-yn-4-one"),
        (dict(atoms=5, suffix="one", locants=[2]), "pentan-2-one"),
        (dict(atoms=5, suffix="ol", locants=[1]), "pentan-1-ol"),
        (dict(atoms=5, suffix="amine", locants=[1]), "pentan-1-amine"),
        # Consonant-initial, already correct in the saturated branch.
        (dict(atoms=5, suffix="thiol", locants=[1]), "pentane-1-thiol"),
    ])
    def test_unchanged(self, kwargs, expected):
        assert _build(**kwargs) == expected


# ---------------------------------------------------------------------------
# Untouched shapes: no FG suffix at all, and the 'appended' suffix branch
# ---------------------------------------------------------------------------

class TestOtherBranchesUnaffected:

    @pytest.mark.unit
    @pytest.mark.parametrize("kwargs,expected", [
        (dict(atoms=6), "hexane"),
        (dict(atoms=6, doubles=[2]), "hex-2-ene"),
        (dict(atoms=6, doubles=[2, 4]), "hexa-2,4-diene"),
        (dict(atoms=6, triples=[2]), "hex-2-yne"),
        (dict(atoms=10, suffix="carboxylic acid", locants=[1], stype="appended"),
         "decane-1-carboxylic acid"),
        (dict(atoms=10, doubles=[2], suffix="carboxylic acid", locants=[1],
              stype="appended"), "dec-2-ene-1-carboxylic acid"),
    ])
    def test_unchanged(self, kwargs, expected):
        assert _build(**kwargs) == expected


# Whole-molecule regression evidence for this fix is collected through the CLI
# path instead of here: tests/conftest.py disables the OPSIN validity gate by
# default, so a name asserted via ``Orthonym.name_tiered`` under pytest can
# come from a different producer than the one the CLI selects (measured: the
# cage probe CC12CCCCC1CCCC2=O yields '1-methylbicyclo[4.4.0]decan-10-one' from
# the CLI but '2-(butan-1-yl)-2-methylcyclohexan-1-one' under pytest).  Encoding
# the gate-off artefact as an expectation here would guard the wrong pipeline.
