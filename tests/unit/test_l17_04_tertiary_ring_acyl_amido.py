"""The N-substituted amido prefix of a tertiary acyl nitrogen whose acyl carries a ring (lane/l17 item 4).

 Substituents of the types -NH-CO-R and -NH-SO2-R (the Blue Book): "When a group having preference
for citation as a principal characteristic group is present, the group R-CO-NH-, or R-SO2-NH-... of an N-substituted
amide is named in two ways: (1) substitutively, by using a prefix formed by changing the final letter 'e' in the
complete name of the amide to 'o'... Method (1) generates preferred IUPAC names." The book's own example is
'2-(N-methylpropanamido)benzene-1-sulfonic acid (PIN)' (:33040): the acyl nitrogen takes the name of the whole amide,
the ring on the acyl among its prefixes ('[N-methyl-2-(4-methylpiperazin-1-yl)acetamido]acetic acid').

Two defects kept the prefix from such a molecule:

* the PIN tier names a prefix it has no producer for ('4-methylpiperazin-1-yl') only in the re-run that admits the
  promoted producers, but the fragment cache of the first run held the amide of the group as 'unnameable' (a refusal
  of the PIN vocabulary) and served it to the re-run: ``fragment_naming`` now serves a refusal only to the vocabulary
  that made it;
* the peptide handler's sub-namer is the PIN tier's at every tier, so at a wider tier it cited the acyl as a bare
  'N-<acyl>' float (a locant N on '(methylamino)acetic acid', which has no nitrogen suffix: Position of locants,
  the Blue Book, "Locants (numerals and/or letters) are placed immediately before that part of the name to which
  they relate"; the locant N of an N-substituted amide is that of its amide suffix, N-Substitution,:32774)
  before the tier's own engine could build the amido prefix.
"""
import re

import pytest

from tests.support.rt_assert import assert_full_rt
from tests.support.pin_tiers import tier_row

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

#: an 'N-' locant opening the name, behind a stereodescriptor block '(2S)-' or an ester word ('methyl N-...')
_FLOATS_N = re.compile(r"^(\([^)]*\)-)?(N-|[a-z]+ N-)")


# (smiles, the PIN) -- every name read back whole by OPSIN to the input's full InChIKey, spelled as the book spells it
_TARGETS = [
    # the tertiary acyl nitrogen on an acetic acid, a benzoic acid and a propanoic acid
    ("OC(=O)CN(C)C(=O)CN1CCN(C)CC1", "[N-methyl-2-(4-methylpiperazin-1-yl)acetamido]acetic acid"),
    ("OC(=O)c1ccc(N(C)C(=O)CN2CCN(C)CC2)cc1", "4-[N-methyl-2-(4-methylpiperazin-1-yl)acetamido]benzoic acid"),
    ("OC(=O)CCN(C)C(=O)CN1CCN(C)CC1", "3-[N-methyl-2-(4-methylpiperazin-1-yl)acetamido]propanoic acid"),
    # the other rings of the class: a ring the PIN vocabulary has no producer for beside a substituent on it
    ("OC(=O)c1ccc(N(C)C(=O)CN2CCN(CC)CC2)cc1", "4-[2-(4-ethylpiperazin-1-yl)-N-methylacetamido]benzoic acid"),
    ("OC(=O)CN(C)C(=O)CN1CCN(CC)CC1", "[2-(4-ethylpiperazin-1-yl)-N-methylacetamido]acetic acid"),
    ("OC(=O)CN(C)C(=O)CN1CCCN(C)CC1", "[N-methyl-2-(4-methyl-1,4-diazepan-1-yl)acetamido]acetic acid"),
    ("OC(=O)CN(C)C(=O)CN1CCC(C)CC1", "[N-methyl-2-(4-methylpiperidin-1-yl)acetamido]acetic acid"),
    ("OC(=O)CN(C)C(=O)CC1CCN(C)CC1", "[N-methyl-2-(1-methylpiperidin-4-yl)acetamido]acetic acid"),
    ("OC(=O)c1ccc(N(C)C(=O)CC2CCN(C)CC2)cc1", "4-[N-methyl-2-(1-methylpiperidin-4-yl)acetamido]benzoic acid"),
    # the N-substituent is an ethyl: the locant N cites it, the acyl chain keeps C-2
    ("OC(=O)CN(CC)C(=O)CN1CCN(C)CC1", "[N-ethyl-2-(4-methylpiperazin-1-yl)acetamido]acetic acid"),
    ("OC(=O)CCN(CC)C(=O)CN1CCN(C)CC1", "3-[N-ethyl-2-(4-methylpiperazin-1-yl)acetamido]propanoic acid"),
]


@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles,name", _TARGETS)
def test_a_tertiary_acyl_nitrogen_with_a_ring_on_the_acyl_is_the_amido_prefix(smiles, name, tier):
    """ (the Blue Book), method (1): the prefix is the complete name of the amide with the final 'e'
    changed to 'o'; '2-(N-methylpropanamido)benzene-1-sulfonic acid (PIN)' (:33040)."""
    row = tier_row(smiles, tier)
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)


# the NH analogs (a secondary acyl nitrogen) of the same ring: named by the same re-run
_NH_ANALOGS = [
    ("OC(=O)c1ccc(NC(=O)CN2CCN(C)CC2)cc1", "4-[2-(4-methylpiperazin-1-yl)acetamido]benzoic acid"),
    ("OC(=O)c1ccc(NC(=O)CN2CCN(CC)CC2)cc1", "4-[2-(4-ethylpiperazin-1-yl)acetamido]benzoic acid"),
    ("OC(=O)CCNC(=O)CN1CCN(C)CC1", "3-[2-(4-methylpiperazin-1-yl)acetamido]propanoic acid"),
]


# the NH analogs on an amino acid: a ring that has no producer in the first run of the PIN tier, on the acetyl group of an
# N-acylglycine / -leucine / -methionine. The molecule holds no dipeptide of amino acid residues (the piperazine nitrogen
# is no amino group of a residue, ``rules.peptides.peptide_of_residues``), so the name of the re-run is kept, as for
# '[2-(pyrrolidin-1-yl)acetamido]acetic acid' "Method (1) generates preferred IUPAC names",:32998)
_NH_AMINO_ACIDS = [
    ("OC(=O)CNC(=O)CN1CCN(C)CC1", "[2-(4-methylpiperazin-1-yl)acetamido]acetic acid"),
    ("CC(C)C[C@H](NC(=O)CN1CCN(C)CC1)C(O)=O", "(2S)-4-methyl-2-[2-(4-methylpiperazin-1-yl)acetamido]pentanoic acid"),
    ("CSCC[C@H](NC(=O)CN1CCN(C)CC1)C(O)=O",
     "(2S)-2-[2-(4-methylpiperazin-1-yl)acetamido]-4-(methylsulfanyl)butanoic acid"),
]


@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles,name", _NH_ANALOGS + _NH_AMINO_ACIDS)
def test_the_secondary_acyl_nitrogen_of_the_same_ring_is_named_as_well(smiles, name, tier):
    row = tier_row(smiles, tier)
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)


# the molecules that were named before the change, and keep their names
_POSITIVES = [
    ("OC(=O)CN(C)C(=O)CC", "(N-methylpropanamido)acetic acid"),
    ("OC(=O)CNC(=O)CN1CCCC1", "[2-(pyrrolidin-1-yl)acetamido]acetic acid"),
    ("OC(=O)CN(C)C(=O)CN1CCCC1", "[N-methyl-2-(pyrrolidin-1-yl)acetamido]acetic acid"),
    ("OC(=O)CN(C)C(=O)CN1CCOCC1", "[N-methyl-2-(morpholin-4-yl)acetamido]acetic acid"),
    ("OC(=O)CN(C)C(=O)Cc1ccccc1", "(N-methyl-2-phenylacetamido)acetic acid"),
    ("OC(=O)c1ccc(N(C)C(=O)CN2CCOCC2)cc1", "4-[N-methyl-2-(morpholin-4-yl)acetamido]benzoic acid"),
]


@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles,name", _POSITIVES)
def test_the_acyls_named_before_keep_their_names(smiles, name, tier):
    row = tier_row(smiles, tier)
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("tier", ["valid", "complete"])
def test_the_wider_tiers_below_best_effort_name_the_glycine_target_the_same(tier):
    smiles, name = _TARGETS[0]
    row = tier_row(smiles, tier)
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row


@pytest.mark.parametrize("smiles,name", _TARGETS + _NH_ANALOGS)
def test_no_wider_tier_name_cites_a_bare_n_acyl_on_a_parent_without_a_nitrogen_suffix(smiles, name):
    """ (the Blue Book): a locant is placed before the part of the name it relates to, and the locant N
    of an N-substituted amide is that of its amide suffix,:32774); the acid, ester or ring parent
    of the molecule has no nitrogen suffix: the best-effort name is no 'N-[(...)acetyl](methylamino)acetic acid'."""
    for tier in ("valid", "complete", "best-effort"):
        row = tier_row(smiles, tier)
        assert not _FLOATS_N.match(row["name"]), (tier, row)


def test_a_float_on_a_retained_residue_is_not_displaced_by_a_name_that_is_not_in_pin_form():
    """ Derivatives formed by substitution (the Blue Book): "Retained names are used to indicate carbon,
    nitrogen, oxygen and sulfur substitution", so the locant N of 'N-acylglutamyl-D-proline' cites a nitrogen of the
    name. The wider tier's own name of that peptide is the last-resort floor ('...-1,5-dioxopentyl]-2-carboxypyrrolidine'),
    in no PIN form, and the float stays."""
    smiles = "CC[C@H](C)[C@H](N)C(=O)N[C@@H](CCC(=O)O)C(=O)N1CCC[C@@H]1C(=O)O"
    row = tier_row(smiles, "best-effort")
    assert (row["name"], row["tier"]) == (
        "N-[(2S,3S)-2-amino-3-methylpentanoyl]glutamyl-D-proline", "systematic_verified"), row
    assert_full_rt(row["name"], smiles)


def test_the_float_pattern_catches_the_defect():
    assert _FLOATS_N.match("N-[(4-methylpiperazin-1-yl)acetyl](methylamino)acetic acid")
    assert not _FLOATS_N.match("[N-methyl-2-(4-methylpiperazin-1-yl)acetamido]acetic acid")


# ---- the fragment cache: a refusal is the refusal of one vocabulary ------------------------------------------------

def test_a_refusal_is_served_only_to_the_vocabulary_that_made_it(monkeypatch):
    """The re-run of the PIN tier admits producers the first run lacks (``assembly.memo.pin_promotion_var``): the
    amide 'CNC(=O)CN1CCN(C)CC1' that the first run could not name ('4-methylpiperazin-1-yl' has no PIN-vocabulary
    producer) is named by the re-run. A refusal the first run cached must not be served to it; within one vocabulary
    the cached refusal stands (no second attempt)."""
    import orthonym.namer as namer
    from orthonym.assembly import fragment_naming as fn
    from orthonym.assembly.memo import pin_promotion_var
    smiles = "CNC(=O)CN1CCN(C)CC1"
    promoted_name = "N-methyl-2-(4-methylpiperazin-1-yl)acetamide"
    seen = []

    def fake_name_compound(canonical, style="pin"):
        seen.append(pin_promotion_var.get())
        return promoted_name if pin_promotion_var.get() else "unknown organic compound"

    monkeypatch.setattr(namer, "name_compound", fake_name_compound)
    fn.start_naming_session()
    try:
        assert fn.name_fragment_recursively(smiles) is None
        assert fn.name_fragment_recursively(smiles) is None     # the refusal of the same vocabulary stands
        assert seen == [False]
        token = pin_promotion_var.set(True)
        try:
            assert fn.name_fragment_recursively(smiles) == promoted_name
            assert fn.name_fragment_recursively(smiles) == promoted_name
        finally:
            pin_promotion_var.reset(token)
        assert seen == [False, True]
    finally:
        fn.end_naming_session()


# ---- the rows the vocabulary-keyed refusals rename, and the tiers that agree on a float-demoted peptide ------------

# the peptide-like rows of a 846-row panel (pin and best-effort) that the fragment cache renamed beyond the targets:
# each is the name of the molecule at best-effort and none loses its name or its read-back
_RENAMED_AT_BEST_EFFORT = [
    # a cyclohexylidene dipeptide: the propanoic acid is the parent, the amido prefix of the N-terminal residue the
    # method (1) prefix of (the Blue Book); the name is a breadth producer's, not certified
    ("CC(=O)N[C@@H](/C=C1/C=C[C@H](N)CC1)C(=O)N[C@@H](/C=C1\\C=C[C@H](NC(C)=O)CC1)C(=O)O",
     "(2S,3Z)-2-{(2S,3E)-2-acetamido-3-[(4R)-4-aminocyclohex-2-en-1-ylidene]propanamido}-3-[(4R)-4-acetamido"
     "cyclohex-2-en-1-ylidene]propanoic acid", "pin_unverified"),
    # a sulfo-substituted acyl on the tertiary nitrogen of an aspartic acid: the float 'N-(4-amino-4-oxo-2-sulfobutanoyl)-
    # 2-(octadecylamino)butanedioic acid' cited N on a parent with no nitrogen suffix; the acylamino prefix of method (2)
    #,:32991) is a valid name below the PIN
    ("CCCCCCCCCCCCCCCCCCN(C(CC(=O)O)C(=O)O)C(=O)C(CC(=O)N)S(=O)(=O)O",
     "2-[(4-amino-4-oxo-2-sulfobutanoyl)(octadecyl)amino]butanedioic acid", "systematic_verified"),
    # an N-acylglutamine: the amido prefix on the pentanoic acid parent, not the 'pentanoyl]amino' acylamino form
    ("C=C1CC23C=CC(=O)C(C)(CCCC(C)C(=O)NC(CCC(N)=O)C(=O)O)C2CC1C(O)C3",
     "5-amino-2-[5-(12-hydroxy-5-methyl-9-methylidene-4-oxotricyclo[6.2.2.0^1,6]dodec-2-en-5-yl)-2-methylpentan"
     "amido]-5-oxopentanoic acid", "systematic_verified"),
]


@pytest.mark.parametrize("smiles,name,label", _RENAMED_AT_BEST_EFFORT)
def test_the_peptide_rows_the_refusal_cache_renames_keep_their_read_back_at_best_effort(smiles, name, label):
    row = tier_row(smiles, "best-effort")
    assert (row["name"], row["tier"]) == (name, label), row
    assert_full_rt(row["name"], smiles)


_DEPSIPEPTIDE = ("CCC(C)[C@H](C(=O)O[C@@H](C(=O)N(C)[C@H](C(=O)O)C(C)C)C(C)C)N(C)C(=O)[C@H](O)C(C)C",
                 "(2S)-2-{[(2R)-2-({(2R)-2-[(2R)-2-hydroxy-N,3-dimethylbutanamido]-3-methylpentanoyl}oxy)-3-"
                 "methylbutanoyl](methyl)amino}-3-methylbutanoic acid")


@pytest.mark.parametrize("tier", ["complete", "best-effort"])
def test_a_depsipeptide_cites_no_n_on_a_parent_without_a_nitrogen_suffix_at_the_tiers_that_build_the_name(tier):
    """ (the Blue Book): the locant N relates to the amide or amine suffix of a parent, and the acid has
    none. The best-effort engine, with the unverified opt-in, names the sub-molecule with the last-resort floor
    ('1-oxopentane'): the engine of the tier below (complete) builds the name, and the float is withdrawn at both."""
    smiles, name = _DEPSIPEPTIDE
    row = tier_row(smiles, tier)
    assert (row["name"], row["tier"]) == (name, "systematic_verified"), row
    assert not _FLOATS_N.match(row["name"]), row
    assert_full_rt(row["name"], smiles)


# the 15 peptide-like rows of a panel of 846 rows that the peptide handler's float is withdrawn on or kept for at the tiers
# valid and complete (lane/l17 item 4): (smiles, the name at valid, the name at complete or _SAME). Five names changed
# against the tree before the fragment cache kept its refusals per vocabulary, none was lost: the two piperazinyl and
# depsipeptide rows at both tiers, and the sulfo-substituted aspartic acid at complete.
# The two indole rows after lane/l17 item 7: the amide parent is chosen among the amide groups class 11,
# "the maximum number of substituents corresponding to the principal characteristic group (suffix)",,
# so the name ends in 'propanamide' and no longer in the suffix-less 'indole' parent (read back whole by OPSIN to the
# input's full InChIKey: the stereo labels are machine-confirmed).
_SAME = object()
_PASS_3_ROWS = [
    ("OC(=O)CN(C)C(=O)CN1CCN(C)CC1",
     "[N-methyl-2-(4-methylpiperazin-1-yl)acetamido]acetic acid",
     _SAME),
    ("CC(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N(C)[C@@H](Cc1ccccc1)C(=O)N/C=C\\c1c[nH]c2ccccc12",
     "(2S)-2-acetamido-N-[(1S)-1-benzyl-2-{[(1Z)-2-(1H-indol-3-yl)ethen-1-yl]amino}-2-oxoethyl]-3-(4-hydroxy"
     "phenyl)-N-methylpropanamide",
     _SAME),
    ("CC(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N(C)[C@@H](Cc1ccccc1)C(=O)N/C=C/c1c[nH]c2ccccc12",
     "(2S)-2-acetamido-N-[(1S)-1-benzyl-2-{[(1E)-2-(1H-indol-3-yl)ethen-1-yl]amino}-2-oxoethyl]-3-(4-hydroxy"
     "phenyl)-N-methylpropanamide",
     _SAME),
    ("CCCCCCCCC(C)(C)NC(=O)C[NH+](CC)CC",
     "N,N-diethyl-2-[(2-methyldecan-2-yl)amino]-2-oxoethan-1-aminium",
     _SAME),
    ("NC(=[NH2+])NCCCC(N)=O",
     "1-amino-4-(2-amino-1,3-diazaprop-1-en-1-ium-1-yl)-1-oxobutane",
     _SAME),
    ("C[C@]12CC[C@@H]3[C@@H]([C@@H]1CC[C@@]2(C)O)CCC4=CC(=CC[C@@]34C)NOCC(=O)NCC(=O)O",
     "N-[({[(1R,2S,10S,11S,14R,15S)-14-hydroxy-2,14,15-trimethyltetracyclo[8.7.0.0^2,7.0^11,15]heptade"
     "ca-4,6-dien-5-yl]amino}oxy)acetyl]glycine",
     _SAME),
    (_DEPSIPEPTIDE[0], _DEPSIPEPTIDE[1], _DEPSIPEPTIDE[1]),
    ("COc1c(C(SCC(NC(=O)CCC(N)C(=O)O)C(=O)NCC(=O)O)C(O)C=O)cc2c(c1OC)OCO2",
     "4-[1-({2-(4-amino-4-carboxybutanamido)-3-[(carboxymethyl)amino]-3-oxopropyl}sulfanyl)-2-h"
     "ydroxy-3-oxopropyl]-2,3-dimethoxy-7,9-dioxabicyclo[4.3.0]nona-1,3,5-triene",
     _SAME),
    ("CCCCCCCCCCCCCCCCCCN(C(CC(=O)O)C(=O)O)C(=O)C(CC(=O)N)S(=O)(=O)O",
     "N-(4-amino-4-oxo-2-sulfobutanoyl)-2-(octadecylamino)butanedioic acid",
     "2-[(4-amino-4-oxo-2-sulfobutanoyl)(octadecyl)amino]butanedioic acid"),
    ("CCC[C@@H](C(=O)OCC)N[C@H](C)C(=O)N1[C@H]2CCCC[C@H]2C[C@H]1C(=O)O",
     "(2S,3aS,7aS)-N-[(2R)-2-{[(2S)-1-ethoxy-1-oxopentan-2-yl]amino}propanoyl]octahydro-1H-indole-2-ca"
     "rboxylic acid",
     _SAME),
    ("NC(=O)[C@@H]1CCC[NH2+]C1",
     "(3R)-3-carbamoylpiperidin-1-ium",
     _SAME),
    ("C[C@@]12C3=CC=CC=C3CC[NH+]1CC(=O)N2",
     "(2S)-2-methyl-4-oxo-3,6-diazatricyclo[7.4.0.0^2,6]trideca-1(13),9,11-trien-6-ium",
     _SAME),
    ("C=CCNC(=S)SC[C@H](NC(C)=O)C(=O)O",
     "(2R)-2-acetamido-3-({[(prop-2-en-1-yl)amino](sulfanylidene)methyl}sulfanyl)propanoic acid",
     _SAME),
    ("CC[C@H](C)[C@H](N)C(=O)N[C@@H](CCC(=O)O)C(=O)N1CCC[C@@H]1C(=O)O",
     "N-[(2S,3S)-2-amino-3-methylpentanoyl]glutamyl-D-proline",
     _SAME),
    ("C=CC(C)(C=O)SC[C@H](NC(=O)CC[C@H]([NH3+])C(=O)[O-])C(=O)NCC(=O)[O-]",
     "(2S)-2-azaniumyl-5-({(2R)-3-[(2-formylbut-3-en-2-yl)sulfanyl]-1-[(2-oxido-2-oxoethyl)amino]-1-ox"
     "opropan-2-yl}amino)-1-oxido-1,5-dioxopentane",
     _SAME),
]


@pytest.mark.parametrize("tier", ["valid", "complete"])
@pytest.mark.parametrize("smiles,valid_name,complete_name", _PASS_3_ROWS)
def test_the_pass_3_rows_keep_their_names_at_valid_and_complete(smiles, valid_name, complete_name, tier):
    expected = valid_name if tier == "valid" or complete_name is _SAME else complete_name
    row = tier_row(smiles, tier)
    assert row["name"] == expected, row
    assert row["tier"] != "abstain", row


@pytest.mark.parametrize("smiles,name", [
    (_PASS_3_ROWS[0][0], _PASS_3_ROWS[0][1]),
    (_DEPSIPEPTIDE[0], _DEPSIPEPTIDE[1]),
    (_PASS_3_ROWS[8][0], _PASS_3_ROWS[8][2]),
])
def test_the_pass_3_names_that_changed_are_read_back_whole(smiles, name):
    assert_full_rt(name, smiles)


def test_a_refusal_is_keyed_on_the_whole_renaming_state_not_the_tier_flags_alone(monkeypatch):
    """The peptide handler names a molecule in up to three passes with the same tier flags: pass 1 withholds the N-acyl
    float (``decomposition.engine._NACYL_FLOAT_REFUSALS``), pass 2 ships it, and the handler's own flags tell the passes
    from a plain naming (``dispatch_table._PEPTIDE_SUBST_ACTIVE`` / ``_PEPTIDE_NAME_STANDS``). A fragment refused in
    one of these states is named again in the next, so pass 3 is never served the refusal of pass 2."""
    import orthonym.namer as namer
    from orthonym.assembly import fragment_naming as fn
    from orthonym.decomposition.engine import _NACYL_FLOAT_REFUSALS
    from orthonym.routing import dispatch_table
    smiles = "CNC(=O)CN1CCN(C)CC1"
    seen = []

    def fake_name_compound(canonical, style="pin"):
        seen.append((_NACYL_FLOAT_REFUSALS.get() is not None, dispatch_table._PEPTIDE_SUBST_ACTIVE,
                     dispatch_table._PEPTIDE_NAME_STANDS))
        return "unknown organic compound"

    monkeypatch.setattr(namer, "name_compound", fake_name_compound)
    fn.start_naming_session()
    try:
        assert fn.name_fragment_recursively(smiles) is None
        assert fn.name_fragment_recursively(smiles) is None              # the same state: the refusal stands
        assert seen == [(False, False, False)]
        monkeypatch.setattr(dispatch_table, "_PEPTIDE_SUBST_ACTIVE", True)
        assert fn.name_fragment_recursively(smiles) is None              # the peptide handler's attempt: named again
        token = _NACYL_FLOAT_REFUSALS.set([])
        try:
            assert fn.name_fragment_recursively(smiles) is None          # the float withheld: named again
        finally:
            _NACYL_FLOAT_REFUSALS.reset(token)
        monkeypatch.setattr(dispatch_table, "_PEPTIDE_NAME_STANDS", True)
        assert fn.name_fragment_recursively(smiles) is None              # a peptide name stands: named again
        assert seen == [(False, False, False), (False, True, False), (True, True, False), (False, True, True)]
        assert fn.name_fragment_recursively(smiles) is None              # each of them is served its own refusal
        assert len(seen) == 4
    finally:
        fn.end_naming_session()


# ---- the amido prefix of an acyl that holds a second COOH -----------------------------------------------------------

_ASP_LEU_MET_LEU = "CC(C)CC(C(=O)NC(CCSC)C(=O)NC(CC(C)C)C(=O)O)NC(=O)C(CC(=O)O)N"


def test_the_acid_name_of_a_second_cooh_is_not_turned_into_the_amido_prefix_of_the_acyl():
    """With the refusals of the fragment cache kept per vocabulary, the best-effort vocabulary names the tripeptide acid
    HO-CO-CH(R)-NH-CO-CH(R')-NH-CO-CH(NH2)-CH2-COOH as '3-amino-4-[...]-4-oxobutanoic acid', the '...oic acid' of which is
    the side-chain COOH of the Asp residue, not the carboxyl of the acyl. '...-4-oxobutanamido' method (1),
    the Blue Book, "changing the final letter 'e' in the complete name of the amide to 'o'") is the prefix of
    another molecule, the round trip voided that candidate and the molecule fell to the last-resort chain
    '...-1-oxa-6,9,12-triazahexadec-1-ene'. The prefix is the name of the amide of THIS acyl only where OPSIN reads it so."""
    row = tier_row(_ASP_LEU_MET_LEU, "best-effort")
    assert row["tier"] == "systematic_verified", row
    assert "triaza" not in row["name"], row
    assert_full_rt(row["name"], _ASP_LEU_MET_LEU)


def test_ordinary_amido_is_the_acyl_reads_the_amide_back():
    from rdkit import Chem
    from orthonym.assembly.acyl_amido_polyacid import ordinary_amido_is_the_acyl
    mol = Chem.MolFromSmiles("CNC(=O)C(N)CC(=O)O")     # the Asp acyl on the nitrogen of an N-methyl amide
    n_idx, carbonyl_c = 1, 2
    group = [i for i in range(mol.GetNumAtoms()) if i != 0]
    assert ordinary_amido_is_the_acyl(mol, n_idx, carbonyl_c, group, "2-amino-3-carboxypropanamido")
    assert not ordinary_amido_is_the_acyl(mol, n_idx, carbonyl_c, group, "2-aminobutanamido")
    # an acyl with no second acid is the ordinary route's, whatever the prefix
    plain = Chem.MolFromSmiles("CNC(=O)C(N)C")
    assert ordinary_amido_is_the_acyl(plain, 1, 2, [i for i in range(plain.GetNumAtoms()) if i != 0], "2-aminopropanamido")
