"""-CLEANUP Item 2 + MINORs 4/5/8 — enclosure produced correctly at SOURCE.

WHAT WAS LEFT AFTER P3-REGRESSION
---------------------------------
P3-REGRESSION made `format_fg_prefix`'s `_enclose` IDEMPOTENT, which was right:
callers legitimately hand over a pre-enclosed prefix. But an idempotent sink
cannot tell a legitimate pre-enclosure from a MALFORMED one, and one producer was
emitting a malformed one:

    _name_amidine_chain_side -> '((4-chlorophenyl)methylamino)' <- '(' inside '('

`### **P"16.5.4** Multiple!types!of!enclosing!marks` (the Blue Book), verbatim
at:7446: "*When multiple types of enclosing marks are required, the nesting order
is as follows: {[({})]}*". A directly on-point PIN exists at:26529 —
`4-{[(4-chlorophenyl)methylidene]amino}aniline (PIN)`, under
`## **** Substitutive names for imines` (:26508): same aryl, same
N-attached prefix, spelled brace / bracket / paren from the outside in.

The fix is at the PRODUCER, not the sink: the sink stays idempotent (a repair
layer there would mask the next caller bug), while the producer stops emitting
the outer marks at all and starts emitting the INNER level it was missing.

THIS WAS ALSO A COVERAGE LOSS, NOT ONLY A SPELLING ONE
------------------------------------------------------
With two aryl branches the run-together prefix parsed as a DIFFERENT molecule (a
diarylmethyl), caught it, and the compound abstained. Measured on the real
path with the gates ON: `unknown organic compound` before, a correct OPSIN-clean
name after.
"""
import pytest

from orthonym.assembly.composition_primitives import (
    _join_prefixes as _join_sibling,
)
from orthonym.namer import name_compound
from orthonym.rules.polyfunctional import (
    _join_prefixes as _join_polyfn,
    format_fg_prefix,
)


# ---------------------------------------------------------------------------
# Item 2 — the producer emits a well-formed prefix
# ---------------------------------------------------------------------------

class TestAmidineNSubstituentEnclosure:

    def test_bb_p62_3_1_nesting_on_the_witness(self):
        """The review's witness, spelled brace / bracket / paren as BB:26529."""
        assert name_compound("COC(=O)CCC(=NCC)NCc1ccc(Cl)cc1") == (
            "methyl 4-{[(4-chlorophenyl)methyl]amino}-4-(ethylimino)butanoate"
        )

    def test_no_paren_directly_inside_a_paren(self):
        """The violation itself, stated as a property.

        Pinning only the literal above would pass for a name that fixed this
        witness and reintroduced `((` elsewhere."""
        n = name_compound("COC(=O)CCC(=NCC)NCc1ccc(Cl)cc1")
        assert "((" not in n and "[[" not in n and "{{" not in n, n

    def test_two_compound_n_substituents_each_get_their_own_marks(self):
        """The coverage-recovery case: this abstained to `unknown organic
        compound` before, because the run-together prefix named a diarylmethyl."""
        n = name_compound("COC(=O)CCC(=NCC)N(Cc1ccc(Cl)cc1)Cc1ccc(F)cc1")
        assert n == (
            "methyl 4-{[(4-chlorophenyl)methyl][(4-fluorophenyl)methyl]amino}"
            "-4-(ethylimino)butanoate"
        )
        assert "unknown" not in n

    @pytest.mark.parametrize("smiles,expected", [
        # A COMPOUND substituent with no marks of its own still needs enclosing
        # / — it just starts one level lower.
        ("COC(=O)CCC(=NCC)NCC(C)C",
         "methyl 4-(ethylimino)-4-[(2-methylpropyl)amino]butanoate"),
        ("COC(=O)CCC(=NCC)NCCc1ccccc1",
         "methyl 4-(ethylimino)-4-[(2-phenylethyl)amino]butanoate"),
    ])
    def test_compound_substituent_without_own_marks(self, smiles, expected):
        assert name_compound(smiles) == expected

    @pytest.mark.parametrize("smiles,expected", [
        ("COC(=O)CCC(=NCC)N(C)C",
         "methyl 4-(dimethylamino)-4-(ethylimino)butanoate"),
        ("COC(=O)CCC(=NCC)NC",
         "methyl 4-(ethylimino)-4-(methylamino)butanoate"),
        ("COC(=O)CCC(=NCC)N",
         "methyl 4-amino-4-(ethylimino)butanoate"),
        ("COC(=O)CCC(=NC)N(C)C",
         "methyl 4-(dimethylamino)-4-(methylimino)butanoate"),
        ("COC(=O)CCC(=NCc1ccccc1)NC",
         "methyl 4-(benzylimino)-4-(methylamino)butanoate"),
        # (the Blue Book, under the heading "ALPHANUMERICAL ORDER"):
        # "Simple prefixes (i.e., those describing atoms and unsubstituted substituents) are
        # arranged alphabetically; multiplicative prefixes, if necessary, are then inserted and
        # do not alter the alphabetical order already established": 'ethyl' precedes 'methyl',
        # whatever the primes. (The old value cited the N,N pair first.)
        ("CC(=NCC)N(C)C", "N'-ethyl-N,N-dimethylethanimidamide"),
    ])
    def test_simple_n_substituents_are_byte_identical(self, smiles, expected):
        """The regression half. A SIMPLE substituent (methyl/ethyl/benzyl) must
        stay bare, so the four gold rows P3-REGRESSION repaired do not move."""
        assert name_compound(smiles) == expected

    def test_producer_returns_a_bare_prefix(self):
        """The contract change itself: the outer marks belong to
        `format_fg_prefix`, which picks their TYPE per. A producer that
        hard-codes `(` is what produced `(` inside `(`."""
        from rdkit import Chem
        from orthonym.rules.polyfunctional import _name_amidine_chain_side
        mol = Chem.MolFromSmiles("COC(=O)CCC(=NCC)NCc1ccc(Cl)cc1")
        patt = Chem.MolFromSmarts("[CX3](=[NX2])[NX3]")
        match = mol.GetSubstructMatch(patt)
        assert match, "fixture drifted: no amidine matched"
        amidine_c, _imine_n, amino_n = match
        out = _name_amidine_chain_side(mol, amino_n, amidine_c, "amino")
        assert out == "[(4-chlorophenyl)methyl]amino"
        assert not out.startswith("("), out


# ---------------------------------------------------------------------------
# MINOR 4 — the fourth branch, now routed through the shared `_enclose`
# ---------------------------------------------------------------------------

class TestNoLocantDerivedMultiplierBranch:
    """`format_fg_prefix(prefix, , count>1)` with a derived multiplier was the
    one branch still hand-rolling its enclosure. Its guard omitted `{`, so a
    brace-enclosed prefix took a REDUNDANT extra level. Dead today (0 executions
    over 1934 gold + 300 corpus rows) — tested because a dead branch that bypasses
    the shared sink is a trap for the next change, which is the whole reason the
    sink exists."""

    def test_brace_enclosed_prefix_keeps_exactly_one_level(self):
        got = format_fg_prefix("{[(methylcarbamoyl)amino]methyl}", [], 2)
        assert got == "bis{[(methylcarbamoyl)amino]methyl}", got

    def test_agrees_with_the_locant_path(self):
        """The property that makes this a real fix: both paths must spell the
        same prefix the same way, modulo the locants."""
        p = "{[(methylcarbamoyl)amino]methyl}"
        no_locant = format_fg_prefix(p, [], 2)
        with_locants = format_fg_prefix(p, [2, 4], 2)
        assert with_locants == "2,4-" + no_locant, (no_locant, with_locants)

    def test_paren_enclosed_prefix_keeps_exactly_one_level(self):
        assert format_fg_prefix("(dimethylamino)", [], 2) == "bis(dimethylamino)"

    def test_bare_simple_prefix_still_gets_marks_from_the_derived_multiplier(self):
        """ — the rule the branch exists for must survive the
        refactor: `bis(sulfanyl)`, never `bissulfanyl`."""
        assert format_fg_prefix("sulfanyl", [], 2) == "bis(sulfanyl)"


# ---------------------------------------------------------------------------
# MINOR 5 — the surviving mutant: `_is_fully_enclosed` vs naive startswith/endswith
# ---------------------------------------------------------------------------

def test_opening_mark_is_matched_to_its_own_close():
    """`_enclose` uses `_is_fully_enclosed`, which pairs an opening mark with its
    OWN close, so a prefix that merely BEGINS with `(` and ENDS with `)` without
    being wrapped by them still escalates. Swapping in the naive
    `startswith`/`endswith` pair passed the whole suite before this test."""
    assert format_fg_prefix("(2-chloroethyl)oxy(methyl)", [4], 1) == (
        "4-[(2-chloroethyl)oxy(methyl)]"
    )


def test_a_genuinely_wrapped_prefix_is_left_alone():
    """Positive control — without it the test above is satisfied by a predicate
    that simply always escalates."""
    assert format_fg_prefix("(dimethylamino)", [4], 1) == "4-(dimethylamino)"


# ---------------------------------------------------------------------------
# MINOR 8 — the two `_join_prefixes` copies must agree about `}`
# ---------------------------------------------------------------------------

BRACE_CASES = [
    (["2-{[(methylcarbamoyl)amino]methyl}", "4-methyl"],
     "2-{[(methylcarbamoyl)amino]methyl}-4-methyl"),
    (["2-{[(methylcarbamoyl)amino]methyl}", "N-methyl"],
     "2-{[(methylcarbamoyl)amino]methyl}-N-methyl"),
]


@pytest.mark.parametrize("parts,expected", BRACE_CASES)
def test_sibling_copy_hyphenates_after_a_brace(parts, expected):
    """`composition_primitives._join_prefixes` omitted `}` from both tuples and
    dropped the separator. 24 `}`-bearing gold rows route through this copy."""
    assert _join_sibling(parts) == expected


@pytest.mark.parametrize("parts,expected", BRACE_CASES)
def test_both_copies_agree(parts, expected):
    """The real invariant: two copies of one rule must not disagree."""
    assert _join_sibling(parts) == _join_polyfn(parts) == expected


@pytest.mark.parametrize("parts,expected", [
    (["2-[(methylcarbamoyl)amino]", "4-methyl"],
     "2-[(methylcarbamoyl)amino]-4-methyl"),
    (["2-(dimethylamino)", "4-methyl"], "2-(dimethylamino)-4-methyl"),
    (["2,2-dimethyl", "4-methyl"], "2,2-dimethyl-4-methyl"),
])
def test_the_other_marks_are_unchanged(parts, expected):
    assert _join_sibling(parts) == _join_polyfn(parts) == expected
