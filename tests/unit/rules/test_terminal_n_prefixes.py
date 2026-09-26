"""Wave2 — claimed-atom mask + terminal N-heteroatom prefixes.

Two fixes in one batch / / /:

1. CLAIMED-ATOM MASK: FG prefixes that fully name their substituent branch
   (isocyanato, isothiocyanato, isocyano, guanidino) are added to the
   polyfunctional _POLY_GUARD_FG_TYPES skip so the generic branch walkers do
   not re-name the same atoms — killing the phantom co-substituent
   double-count ('8-formamido-8-isocyanatooctanoic acid',
   '4-guanidino-4-(methylamino)butanoic acid').

2. TERMINAL N-HETEROATOM PREFIXES: new perception SMARTS + PREFIX_FORMS for
   the preselected prefixes aminooxy (-O-NH2), diazenyl (HN=N-), the
   N-haloamines fluoro/chloro/bromo/iodoamino (-NH-X), and a restricted
   dispatcher row emitting hydroxyamino for the UNSUBSTITUTED -NH-OH.
   Blue Book: 'aminooxy (preselected prefix) (no elision of the final
   letter o of amino)', '2-(aminooxy)ethan-1-amine (PIN)',
   '4-(hydroxyamino)phenol (PIN)', 'diazenyl (preselected prefix)',
   '-NH-Cl chloroamino (preselected prefix)'.
"""

import pytest

from orthonym import name_compound
from orthonym.assembly.naming_utils import alpha_sort_key, needs_brackets


@pytest.mark.unit
class TestClaimedAtomMask:
    """Phantom co-substituent double-count is gone (mask over the FG branch)."""

    @pytest.mark.parametrize("smiles,expected", [
        ("O=C=NCCCCCCCC(=O)O", "8-isocyanatooctanoic acid"),
        ("S=C=NCCCCCCCC(=O)O", "8-isothiocyanatooctanoic acid"),
        ("[C-]#[N+]CCCCCCCC(=O)O", "8-isocyanooctanoic acid"),
        # (the Blue Book-34268): 'carbamimidoylamino (preferred prefix)'; 7. Prefixes (g) (:1700) 'guanidino' is no longer acceptable in PINs
        ("NC(=N)NCCCC(=O)O", "4-(carbamimidoylamino)butanoic acid"),
    ])
    def test_fg_prefix_alone_no_phantom(self, smiles, expected):
        assert name_compound(smiles) == expected

    @pytest.mark.parametrize("smiles,phantom", [
        # the branch walkers used to re-read the heterocumulene/guanidine
        # branch as a second substituent on the same locant
        ("O=C=NCCCCCCCC(=O)O", "formamido"),
        ("S=C=NCCCCCCCC(=O)O", "methylamino"),
        ("[C-]#[N+]CCCCCCCC(=O)O", "methylamino"),
        ("NC(=N)NCCCC(=O)O", "methylamino"),
    ])
    def test_no_phantom_co_substituent(self, smiles, phantom):
        assert phantom not in name_compound(smiles)


@pytest.mark.unit
class TestTerminalNHeteroatomPrefixes:
    """The new preselected prefixes emit the BB-verbatim PIN forms."""

    @pytest.mark.parametrize("smiles,expected", [
        ("N=NCCCCCCCC(=O)O", "8-diazenyloctanoic acid"),
        ("N=NCCC(=O)O", "3-diazenylpropanoic acid"),
        ("NOCCC(=O)O", "3-(aminooxy)propanoic acid"),
        ("ONCCC(=O)O", "3-(hydroxyamino)propanoic acid"),
        ("FNCCCCCCCC(=O)O", "8-(fluoroamino)octanoic acid"),
        ("ClNCCCCCCCC(=O)O", "8-(chloroamino)octanoic acid"),
        ("BrNCCCCCCCC(=O)O", "8-(bromoamino)octanoic acid"),
    ])
    def test_terminal_prefix_pins(self, smiles, expected):
        assert name_compound(smiles) == expected

    def test_diazenyl_alphabetizes_before_methyl(self):
        # 'di' in diazenyl is structural, not a multiplier — sorts at 'd'
        assert name_compound("CC(C)(N=N)CC(=O)O") == \
            "3-diazenyl-3-methylbutanoic acid"

    def test_n_substituted_hydroxylamine_fails_closed(self):
        # R-N(CH3)-OH needs a composed [hydroxy(methyl)amino] builder (not
        # built); the restricted dispatcher row must NOT fire for it
        assert "hydroxyamino" not in name_compound("ON(C)CCCC(=O)O")


@pytest.mark.unit
class TestProtectedNeighbors:
    """Paths adjacent to the mask/prefix changes are unchanged."""

    @pytest.mark.parametrize("smiles,expected", [
        # secondary amine substituent path (walker, not FG loop)
        ("CNCCCCCCCC(=O)O", "8-(methylamino)octanoic acid"),
        # ureido: the SEPARATE carbamoylamino PIN
        ("NC(=O)NCCC[C@H](N)C(=O)O",
         "(2S)-2-amino-5-(carbamoylamino)pentanoic acid"),
        #, BB 34338): a chain-terminal amidine carbon stays
        # IN the chain -> amino+imino (was '4-carbamimidoylbutanoic acid'; the
        # new form is OPSIN-RT canonical-equal to the same SMILES).
        ("N=C(N)CCCC(=O)O", "5-amino-5-iminopentanoic acid"),
        # hydroxylamine as PARENT (senior path untouched by the prefix row).
        # v52 P2, BB:38308/:38314): R-NH-OH is named as
        # an N-derivative of the senior amine ('N-hydroxymethanamine (PIN)
        # N-methylhydroxylamine' for the CH3 case), not the old
        # functional-class 'N-ethylhydroxylamine' direction.
        ("CCNO", "N-hydroxyethanamine"),
        # Wave2 supersedes the original 'butyl isocyanate' control here:
        # substitutive isocyanato is the PIN (BB VERBATIM
        # 'isocyanatocyclohexane (PIN) cyclohexyl isocyanate'); the
        # functional-class form remains under --trivial.
        ("O=C=NCCCC", "1-isocyanatobutane"),
        # azide is not stolen by the diazenyl SMARTS
        ("[N-]=[N+]=NCCCC(=O)O", "4-azidobutanoic acid"),
        # oxime is not stolen by hydroxylamine/diazenyl.
        # Expectation corrected (was unparenthesised): the compound prefix takes
        # enclosing marks, printed verbatim in three PINs --
        # the Blue Book `3-(hydroxyimino)butan-2-one (PIN)`,:38478
        # `4-(hydroxyimino)-1-methylcyclohexa-2,5-diene-1-carboxylic acid (PIN)`,
        #:30140 `5-hydroxy-5-(hydroxyimino)pentanoic acid (PIN)`.
        ("ON=CCCC(=O)O", "4-(hydroxyimino)butanoic acid"),
        # terminal hydrazine stays hydrazinyl sibling, existing)
        ("NNCCCC(=O)O", "4-hydrazinylbutanoic acid"),
    ])
    def test_protected(self, smiles, expected):
        assert name_compound(smiles) == expected


@pytest.mark.unit
class TestSortAndBrackets:
    """alpha_sort_key + needs_brackets rules for the new prefix class."""

    @pytest.mark.parametrize("prefix,key", [
        ("diazenyl", "diazenyl"),   # structural 'di' — NOT stripped
        ("diazo", "diazo"),         # structural 'di' — NOT stripped
        ("diazido", "azido"),       # genuine 2x azido — still strips
        #: 'dimethylamino' is ONE compound prefix, so its internal 'di'
        # is part of the complete name and alphabetizes at 'd'. Expectation
        # corrected from 'methylamino' against the Blue Book,
        # `1,5-bis(dimethylamino)-N,N-dimethylpentan-3-amine N-oxide (PIN)`:
        # bis(dimethylamino) is cited BEFORE N,N-dimethyl, which is only possible
        # if it keys at 'd' -- keying at 'm' gives methyl < methylamino and
        # inverts the PIN. (a) (:7104) independently lists
        # `bis(dimethylamino)` as a preferred prefix, i.e. a COMPOUND prefix.
        ("dimethylamino", "dimethylamino"),
        ("hydroxyamino", "hydroxyamino"),
        ("aminooxy", "aminooxy"),
        ("chloroamino", "chloroamino"),
    ])
    def test_alpha_sort_key(self, prefix, key):
        assert alpha_sort_key(prefix) == key

    @pytest.mark.parametrize("prefix,bracketed", [
        # compound heteroatom-substituted amino/oxy prefixes: marks
        ("hydroxyamino", True),
        ("aminooxy", True),
        ("fluoroamino", True),
        ("chloroamino", True),
        ("bromoamino", True),
        ("iodoamino", True),
        # simple preselected prefixes stay bare
        ("diazenyl", False),
        ("isocyanato", False),
        ("guanidino", False),
        ("amino", False),
    ])
    def test_needs_brackets(self, prefix, bracketed):
        assert needs_brackets(prefix) is bracketed
