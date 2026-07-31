"""P-14.5 alphanumerical order: what may enter the primary key, and what decides ties.

Four defects, one interlocking fix. Each was independently masking the next, so any
one of them landing alone made a name WORSE -- the reason this shipped as a unit.

1. **Stereochemical / isotopic / Greek descriptors were in the primary key.**
   `### **P-14.5** ALPHANUMERICAL ORDER` closes its preamble (``BlueBookV2.md:3446``):

       In these subsections the principles of alphanumerical order do not include
       Greek letters (except in conjunctive names) or isotopic or stereochemical
       descriptors.

   and ``:44601`` from the other side -- capitalized CIP descriptors "are written in
   italics to indicate that they are not involved in the primary stage of
   alphanumerical order". So `[(E)-2-phenylethenyl]` alphabetizes at `p`, not `e`.

2. **Nothing implemented the tie that removing them creates.** ``**P-14.4**`` clause
   (j) (``:3346``) assigns the lower locant to *Z*, *R*, *M*, *r* over *E*, *S*, *P*,
   *s*, over the non-CIP *cis*/*trans*; ``**P-45.6.3**`` (``:22606``) is the
   citation-order half. ``:22589`` gives the tier order outright: "*since the
   alphabetic characters and locants (ignoring the configuration symbols) are
   identical the configurational symbols are compared and 'R' precedes 'S'*".

3. **Enclosing marks leaked into the key.** `4-[(1R)-1-chloroethyl]phenoxy` keyed as
   `[1-chloroethyl]phenoxy`; `[` is ASCII 91, below every lowercase letter, so it was
   cited ahead of `chloroethyl` and -- through P-14.4(g) -- took locant 1.

4. **Letters and locants were compared as one string.** P-14.5's preamble (``:3442``)
   compares "*Nonitalic Roman letters ... first*" and only "*When all the Roman letters
   are identical*" the locants. A single key string conflates the stages, so a leading
   digit decided comparisons that letters should have.

The interlock, concretely: with (1) alone, gold row `W2F-P8-01` regressed, because the
stereodescriptor's `(` had been masking (3) and (4). The gate caught it at 1649/1651.
"""

import pytest

from orthonym import name_compound
from orthonym.assembly.naming_utils import (
    alpha_sort_key,
    cip_descriptor_rank_key,
    prefix_citation_sort_key,
    strip_alphanumerical_noise,
)


class TestPrimaryKeyExcludesDescriptors:
    """P-14.5 :3446 -- the descriptor is not part of the primary key."""

    @pytest.mark.parametrize("raw,expected", [
        # only the DESCRIPTOR goes here; the locant is stripped later, by
        # alpha_sort_key's own P-14.5.2 leading-locant step
        ("(E)-3-phenylprop-2-en-1-yl", "3-phenylprop-2-en-1-yl"),
        ("(1E,3E,5E)-hepta-1,3,5-trien-1-yl", "hepta-1,3,5-trien-1-yl"),
        ("beta-D-glucopyranosyloxy", "glucopyranosyloxy"),   # Greek + configurational
        ("[4-2H]benzoyl", "benzoyl"),                        # isotopic, :7104
        ("rel-(1R,2S)-2-chlorocyclohexyl", "2-chlorocyclohexyl"),
        ("trans-4-methylcyclohexyl", "4-methylcyclohexyl"),
    ])
    def test_descriptor_removed(self, raw, expected):
        assert strip_alphanumerical_noise(raw.lower()) == expected.lower()

    @pytest.mark.parametrize("raw", [
        "decyl", "dec-1-yl", "methyl", "dimethyl", "cyclohexyl", "tert-butyl",
        "sec-butyl", "deca-1,3-dien-1-yl", "(2-chloroethyl)", "(propan-2-yl)",
        "bromomethyl", "chloro", "oxo",
    ])
    def test_ordinary_names_untouched(self, raw):
        """A prefix with no descriptor must pass through byte-identical -- the
        regex must never corrupt an ordinary name (a bare leading `d-` is far more
        likely to be a name than a configurational prefix)."""
        assert strip_alphanumerical_noise(raw) == raw

    def test_descriptor_does_not_decide_the_primary_key(self):
        """`methoxy` (m) must precede `(E)-2-phenylethenyl` (p, not e)."""
        assert sorted(["(E)-2-phenylethenyl", "methoxy"], key=alpha_sort_key) == \
            ["methoxy", "(E)-2-phenylethenyl"]


class TestEnclosingMarksAreNotAlphanumerical:
    """Defect 3: marks are P-16.5 typography, never sort-key content."""

    @pytest.mark.parametrize("raw,expected", [
        ("4-[(1R)-1-chloroethyl]phenoxy", "1-chloroethylphenoxy"),
        # fully enclosed: the P-14.5.2 branch also drops the INNER locant, since
        # a locant is not a letter -- so this reduces past the marks to 'chloroethyl'
        ("(2-chloroethyl)", "chloroethyl"),
        ("methyl", "methyl"),
    ])
    def test_no_mark_survives(self, raw, expected):
        assert alpha_sort_key(raw) == expected
        assert not set("()[]{}") & set(alpha_sort_key(raw))


class TestCipTieBreak:
    """P-14.4(j) :3346 / P-45.6.3 :22606."""

    @pytest.mark.parametrize("raw,rank", [
        ("(1R)-1-chloroethyl", (0,)),
        ("(1S)-1-chloroethyl", (1,)),
        ("(2Z)-pent-2-en-1-yl", (0,)),
        ("(2E)-pent-2-en-1-yl", (1,)),
        ("(1r,4r)-4-methylcyclohexyl", (0, 0)),
        ("(1s,4s)-4-methylcyclohexyl", (1, 1)),
        ("cis-4-methylcyclohexyl", (2,)),      # non-CIP ranks after every CIP
        ("trans-4-methylcyclohexyl", (2,)),
        ("methyl", ()),                        # absence ties with absence
    ])
    def test_rank(self, raw, rank):
        assert cip_descriptor_rank_key(raw) == rank

    def test_r_precedes_s(self):
        """``:22609`` `1-[(1R)-1-bromoethyl]-1-[(1S)-1-bromoethyl]cyclopentane (PIN)`
        [not the reverse; 'R' precedes 'S']."""
        assert sorted(["(1S)-1-bromoethyl", "(1R)-1-bromoethyl"],
                      key=prefix_citation_sort_key) == \
            ["(1R)-1-bromoethyl", "(1S)-1-bromoethyl"]

    def test_z_precedes_e_which_a_string_sort_gets_backwards(self):
        """``:45363``: "'Z' is senior to 'E', **irrespective of the alphabetical
        order**". This is the case that forces an explicit rank table -- three of
        the four CIP pairs happen to be alphabetical and this one is not."""
        pair = ["(2E)-pent-2-en-1-yl", "(2Z)-pent-2-en-1-yl"]
        assert sorted(pair, key=prefix_citation_sort_key) == \
            ["(2Z)-pent-2-en-1-yl", "(2E)-pent-2-en-1-yl"]
        # the trap, asserted so it cannot silently return:
        assert sorted(pair) == ["(2E)-pent-2-en-1-yl", "(2Z)-pent-2-en-1-yl"]

    def test_configuration_is_a_later_tier_than_letters(self):
        """Configuration may only break a tie, never outrank different letters."""
        assert sorted(["(1S)-1-chloroethyl", "(1R)-1-bromoethyl"],
                      key=prefix_citation_sort_key)[0] == "(1R)-1-bromoethyl"
        assert sorted(["(1R)-1-chloroethyl", "(1S)-1-bromoethyl"],
                      key=prefix_citation_sort_key)[0] == "(1S)-1-bromoethyl"


class TestP1451VersusP1452:
    """Whether an internal multiplier counts is a property of the NAME, not of
    whether the caller happened to add enclosing marks."""

    @pytest.mark.parametrize("raw,expected", [
        # P-14.5.1 -- multiplies SEPARATE simple prefixes, ignored
        ("dimethyl", "methyl"),
        ("trioxo", "oxo"),
        ("dihydroxy", "hydroxy"),
        ("tetrachloro", "chloro"),
        # P-14.5.2 -- internal to ONE compound substituent, counts. :3477's
        # `7-(1,2-difluorobutyl)-5-ethyltridecane (PIN)` alphabetizes at 'd'.
        ("(R)-4,5-dihydroxypentyl", "dihydroxypentyl"),
        ("(2E,6E)-3,7,11-trimethyldodeca-2,6,10-trien-1-yl",
         "trimethyldodeca-2,6,10-trien-1-yl"),
    ])
    def test_discriminator(self, raw, expected):
        assert alpha_sort_key(raw) == expected

    def test_chain_stem_guards_still_hold(self):
        """A numeral that is part of the STEM was never a multiplier."""
        for name in ("tridecyl", "octadecyl", "pentacosyl", "triacontyl",
                     "decanoyloxy", "pentanamido", "pentan-2-yl"):
            assert alpha_sort_key(name) == name


class TestGoldRowsEndToEnd:
    """The two curated rows that a partial fix regressed to 1649/1651."""

    def test_w2f_p8_01_diaryl_ether_r_before_s(self):
        """``:22597`` prints this row's own class --
        `4-{3,4-bis[(1R)-1-chloroethyl]phenoxy}-1,2-bis[(1S)-1-chloroethyl]benzene
        (PIN)`, "'R' precedes 'S'"."""
        assert name_compound("C[C@H](Cl)c1ccc(Oc2ccc(cc2)[C@@H](C)Cl)cc1") == \
            "1-[(1R)-1-chloroethyl]-4-{4-[(1S)-1-chloroethyl]phenoxy}benzene"

    def test_w2e_p0cf_01_parent_chain_chosen_by_name(self):
        """Two 7-carbon chains describe this molecule -- through the nitro branch
        or the fluoro branch -- with the SAME locant set 4,5,6. P-45.6.1 picks the
        alphanumerically-first name: 'difluoropropyl' < 'dinitropropyl'."""
        assert name_compound(
            "OC(=O)CCC(C(F)C(F)C)C([N+](=O)[O-])C([N+](=O)[O-])C") == \
            "4-(1,2-difluoropropyl)-5,6-dinitroheptanoic acid"

    def test_anilino_leak_from_the_paren_side(self):
        """The same mark leak, recorded earlier from the `(` side in
        test_anilino_preferred_prefix.py's docstring."""
        assert name_compound("OC(=O)c1ccc(Nc2ccc(Cl)cc2)c(Br)c1") == \
            "3-bromo-4-(4-chloroanilino)benzoic acid"
