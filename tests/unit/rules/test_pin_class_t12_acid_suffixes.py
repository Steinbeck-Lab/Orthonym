"""PIN class program, Task 12: functional-replacement acid suffixes and retained acid stems
(IMIDIC_SUFFIX, THIOACID, ACETO_STEM).

(A) Hyphens in a substituted fused-ring prefix. (the Blue Book) "Hyphens
    are used in substitutive names:" (a) to separate locants from words, (b) after a
    closing mark followed by a locant, (c) between locants and an opening mark, (d)
    between italic and Roman letters. No hyphen goes between a prefix and a stem that
    opens with a Roman letter or a bracket: '3,4-dihydro[1,3]dioxolo[...]' (:52586),
    'N,N,N,1-tetramethylquinolin-1-ium-3-aminium (PIN)' (:42213).
    ``data/fused_heterocycles.py:get_substituted_fused_het_prefix``.
(B) Retained two-carbon stems. (:34750) "The following names are preferred
    IUPAC names, with unlimited substitution... CH3-CN acetonitrile (PIN) ethanenitrile"
    (:34754); its one substitutable position is C-2, so its locants are omitted
    ,:3031): '(3-bromophenyl)acetonitrile (PIN)' (:34827),
    'hydrazinylacetonitrile (PIN)' (:38545), 'cyclopropyl(fluoro)acetonitrile (PIN)'
    (:7314). (:33938) "Only the following five names are preferred IUPAC
    names and can be substituted in the same way as corresponding amides" -- 'acetohydrazide
    (PIN)' (:33944); (:34046) 'N-methylacetohydrazide (PIN) (not
    1-acetyl-1-methylhydrazine)' (:34048). Its chalcogen, imidic and hydrazonic analogues
    are named systematically (:33938,:33946): 'ethanimidohydrazide (PIN) (not
    acetimidohydrazide)' (:33950). ``assembly/handlers/_handler_shared.py:
    _assemble_fragments`` and ``rules/polyfunctional.py:name_polyfunctional``.
(C) Imidic and hydrazonic acids modified by S, Se, Te, and their esters.
    (:30208): 'CH3-CH2-C(=NH)-SH propanimidothioic acid (PIN)' (:30267),
    'CH3-CH2-CH2-C(=NNH2)-SeH butanehydrazonoselenoic acid (PIN)' (:30269),
    'N-hydroxycyclohexanecarboximidoselenoic acid (PIN)' (:30273);
    (:18537-:18538) ranks '-C(=NH)-SH carboximidothioic acid' and '-C(=NH)-SeH
    carboximidoselenoic acid' after the imidic acid. (:31983) "names of
    esters are all derived from acids modified by functional replacement whose
    substitutive names are systematically formed": 'ethyl propanehydrazonate (PIN)'
    (:31995), 'methyl benzenecarboximidothioate (PIN)' (:31997). (:30814)
    "Functional replacement of one of the -OH groups of carbonic acid by various atoms
    or groups is expressed by the following infixes: fluorido, -F; chlorido, -Cl;...":
    'Cl-C(=NH)-OH carbonochloridimidic acid (PIN)' (:30881); infixes in alphabetical
    order (:6537), the 'o' of an infix elided before a vowel (e),:7634).
    ``perception/functional_groups.py`` (imidothioic_acid... hydrazonotelluroic_acid;
    hydroximic_acid with an S/Se/Te acid), ``assembly/handlers/imidic_ester.py`` (the esters),
    ``rules/inorganic_acids.py:name_carbonic_frn_family``.
(D) Thioacids. (:30277): 'benzene-1,2-dicarbodithioic acid (PIN) (not
    tetrathiophthalic acid)' (:30313); 'HS-CS- dithiocarboxy (preferred prefix)' (:30907).
     (:31983): 'CH3-S-CO-CO-S-CH3 S,S-dimethyl ethanebis(thioate) (PIN)'
    (:32005); 'thioic acid' takes 'bis' (b),:7106). (:30608) "Names
    of acyl groups derived from carboxylic acids modified by functional replacement are
    all denoted by the ending 'oyl'", enclosed by (:7356) "Parentheses are used
    to clarify a modification made by functional replacement prefixes": '4-(ethanethioyl)-
    benzoic acid (PIN)' (:30246,:7358), '2-[amino(sulfanylidene)ethanethioyl]benzoic acid
    (PIN)',:33215). The plan's '(thiocarboxy)formic acid' for O=C(O)C(O)=S is
    the name of H{S/O}C-COOH, the acid whose chalcogen position is undetermined (:30301,
    :30700); with the position known, "combinations of prefixes such as 'hydroxy- and
    sulfanylidene-'... are used in acyclic compounds",:30219): 'HO-CS-COOH
    hydroxy(sulfanylidene)acetic acid (PIN)' (:30303-:30305), which the engine names
    already (a protection row). ``rules/benzene.py``, ``assembly/handlers/
    chalcogen_ester.py``, ``assembly/substituent_naming.py:acid_name_to_acyl_prefix``.
"""
import pytest

from tests.support.pin_tiers import (
    assert_not_pin_labelled,
    assert_pin_at_both_tiers,
)

pytestmark = pytest.mark.opsin_gate

HYPHEN_ROWS = [
    ("CC1=CC=CN2C1=NC=C2CO", "(8-methylimidazo[1,2-a]pyridin-3-yl)methanol"),
    ("CC1=CC=CN2C1=NC=C2CC(=O)O", "(8-methylimidazo[1,2-a]pyridin-3-yl)acetic acid"),
    ("CC1=CC=CN2C1=NC=C2CCC#N", "3-(8-methylimidazo[1,2-a]pyridin-3-yl)propanenitrile"),
    ("FC(F)(F)C1=CC=CN2C1=NC=C2CO",
     "[8-(trifluoromethyl)imidazo[1,2-a]pyridin-3-yl]methanol"),
    ("Cc1cc2ccsc2nc1CO", "(5-methylthieno[2,3-b]pyridin-6-yl)methanol"),
]
HYPHEN_NOT_PIN_ROWS = [
    ("CC1=CC=CN2C1=NC=C2CO", "(8-methyl-imidazo[1,2-a]pyridin-3-yl)methanol"),
    ("Cc1cc2ccsc2nc1CO", "(5-methyl-thieno[2,3-b]pyridin-6-yl)methanol"),
]


@pytest.mark.parametrize("smiles,pin", HYPHEN_ROWS)
def test_fused_ring_prefix_hyphen_pin(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles,non_pin", HYPHEN_NOT_PIN_ROWS)
def test_fused_ring_prefix_hyphen_not_pin(smiles, non_pin):
    assert_not_pin_labelled(smiles, non_pin)


def test_fused_ring_prefix_keeps_the_locant_hyphen():
    # (a): a stem that opens with a locant keeps its hyphen
    from tests.support.pin_tiers import name_default
    d = name_default("Cc1ccc2[nH]ccc2c1CCO")
    assert "(5-methyl-1H-indol-4-yl)" in (d.get("name") or ""), d


ACETO_ROWS = [
    ("CC(=O)N(C)N", "N-methylacetohydrazide"),                                 #:34048
    ("CC(=O)NNC", "N'-methylacetohydrazide"),
    ("NNC(=O)Cc1ccccc1", "2-phenylacetohydrazide"),
    ("NNC(=O)CCl", "2-chloroacetohydrazide"),
    ("CCCCC[C@@H]1[C@H](CC[C@@H]1O)CC(=O)NN",
     "2-[(1R,2R,3S)-3-hydroxy-2-pentylcyclopentyl]acetohydrazide"),           # milestone1500
    ("N#CCc1cccc(Br)c1", "(3-bromophenyl)acetonitrile"),                       #:34827
    ("N#CCNN", "hydrazinylacetonitrile"),                                      #:38545
    ("N#CC(F)C1CC1", "cyclopropyl(fluoro)acetonitrile"),                       #:7314
    ("CC1=CC=CN2C1=NC=C2CC#N", "(8-methylimidazo[1,2-a]pyridin-3-yl)acetonitrile"),
    ("N#CCc1ccccc1", "phenylacetonitrile"),
    ("N#CCN1CCOCC1", "(morpholin-4-yl)acetonitrile"),
    ("N#CC(F)(F)c1ccccc1", "difluoro(phenyl)acetonitrile"),
]
ACETO_NOT_PIN_ROWS = [
    ("CC(=O)N(C)N", "N-methylethanehydrazide"),
    ("N#CCc1cccc(Br)c1", "(3-bromophenyl)ethanenitrile"),
    ("N#CCNN", "hydrazinylethanenitrile"),
    ("CC1=CC=CN2C1=NC=C2CC#N", "(8-methyl-imidazo[1,2-a]pyridin-3-yl)ethanenitrile"),
]
# the boundary: the retained stems that stay, and the analogues that are systematic
ACETO_CONTROL_ROWS = [
    ("CC#N", "acetonitrile"),
    ("CC(=O)NN", "acetohydrazide"),
    ("CC(=S)NN", "ethanethiohydrazide"),
    ("CC(=N)NN", "ethanimidohydrazide"),                                      #:33950
    ("NC(=O)Cc1ccccc1", "2-phenylacetamide"),
    ("OC(=O)Cc1ccccc1", "phenylacetic acid"),
    ("N#CCC#N", "propanedinitrile"),
]


@pytest.mark.parametrize("smiles,pin", ACETO_ROWS + ACETO_CONTROL_ROWS)
def test_retained_two_carbon_stem_pin(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles,non_pin", ACETO_NOT_PIN_ROWS)
def test_retained_two_carbon_stem_not_pin(smiles, non_pin):
    assert_not_pin_labelled(smiles, non_pin)


IMIDIC_ROWS = [
    ("CCC(=N)S", "propanimidothioic acid"),                                   #:30267
    ("CCCC([SeH])=NN", "butanehydrazonoselenoic acid"),                       #:30269
    ("ON=C([SeH])C1CCCCC1", "N-hydroxycyclohexanecarboximidoselenoic acid"),  #:30273
    ("N=C(O)Cl", "carbonochloridimidic acid"),                                #:30881
    ("CCOC(CC)=NN", "ethyl propanehydrazonate"),                              #:31995
    ("CSC(=N)c1ccccc1", "methyl benzenecarboximidothioate"),                  #:31997
    ("N=C(S)C1CCCCC1", "cyclohexanecarboximidothioic acid"),
    ("SC(=N)c1ccccc1", "benzenecarboximidothioic acid"),
    ("SC(=N)c1ccc(Cl)cc1", "4-chlorobenzene-1-carboximidothioic acid"),
    ("ON=C(S)C", "N-hydroxyethanimidothioic acid"),
    ("N=C(O)Br", "carbonobromidimidic acid"),
    ("N=C(S)Cl", "carbonochloridimidothioic acid"),
    ("CSC(=N)C", "methyl ethanimidothioate"),
    ("CCOC(C)=NN", "ethyl ethanehydrazonate"),
    ("CCSC(=N)c1ccc(Cl)cc1", "ethyl 4-chlorobenzene-1-carboximidothioate"),
    ("CSC(=N)C1CCCCC1", "methyl cyclohexanecarboximidothioate"),
    ("CSC(=N)Cc1ccccc1", "methyl 2-phenylethanimidothioate"),
    # demoted at a chain end: the simple prefix pair (2),:30021)
    ("N=C(S)CCC(=O)O", "4-imino-4-sulfanylbutanoic acid"),
    ("NN=C(S)CCC(=O)O", "4-hydrazinylidene-4-sulfanylbutanoic acid"),
    ("CSC(=N)CCC(=O)O", "4-imino-4-(methylsulfanyl)butanoic acid"),
]
IMIDIC_NOT_PIN_ROWS = [
    ("CCC(=N)S", "1-iminopropane-1-thiol"),
    ("CCCC([SeH])=NN", "1-hydrazinylidenebutane-1-selenol"),
    ("ON=C([SeH])C1CCCCC1", "cyclohexyl(hydroxyimino)methaneselenol"),
    ("N=C(O)Cl", "1-chloromethanimidic acid"),
    ("CCOC(CC)=NN", "(1-ethoxypropylidene)hydrazine"),
    ("CSC(=N)c1ccccc1", "1-(methylsulfanyl)-1-phenylmethanimine"),
    # (:18499) ranks the carbothioic O-acid (:18520) above the carboximidothioic acid (:18537), and the
    # esters follow their acids,:31983): the thionoester is the parent, 'O-methyl (ethylsulfanyl)
    # (imino)ethanethioate (PIN)' (:32017); perception has no group for it, so the imidothioate is not named
    ("CCSC(=N)C(=S)OC", "ethyl 2-methoxy-2-sulfanylideneethanimidothioate"),
]
# the boundary: the classes around these that keep their names
IMIDIC_CONTROL_ROWS = [
    ("CC(=N)O", "ethanimidic acid"),
    ("CC(=S)O", "ethanethioic O-acid"),
    ("ON=C(O)C", "N-hydroxyethanimidic acid"),
    ("NC(=N)S", "carbamimidothioic acid"),                                    #:30808
    ("N=C(O)S", "carbonimidothioic acid"),                                    #:30806
    ("ClC(=O)O", "carbonochloridic acid"),
    ("CCOC(C)=N", "ethyl ethanimidate"),
    ("N=C(O)CCC(=O)O", "4-hydroxy-4-iminobutanoic acid"),
]


@pytest.mark.parametrize("smiles,pin", IMIDIC_ROWS + IMIDIC_CONTROL_ROWS)
def test_chalcogen_imidic_acid_pin(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles,non_pin", IMIDIC_NOT_PIN_ROWS)
def test_chalcogen_imidic_acid_not_pin(smiles, non_pin):
    assert_not_pin_labelled(smiles, non_pin)


# (:30273) 'N-hydroxycyclohexanecarboximidoselenoic acid (PIN)': with an acyl group that
# has a prefix of its own the 'N-hydroxy' is cited among the prefixes. That is built now:
# ``tests/unit/rules/test_pin_hydroximic_acids.py`` (the indole row, which this file used to pin as
# declined at the default tier, is its ``test_the_indole_row_is_the_pin_not_a_thiol``).


THIOACID_ROWS = [
    ("S=C(S)c1ccccc1C(=S)S", "benzene-1,2-dicarbodithioic acid"),             #:30313
    ("O=C(O)C(O)=S", "hydroxy(sulfanylidene)acetic acid"),                    #:30305
    ("CSC(=O)C(=O)SC", "S,S-dimethyl ethanebis(thioate)"),                    #:32005
    ("NC(=S)C(=S)c1ccccc1C(=O)O", "2-[amino(sulfanylidene)ethanethioyl]benzoic acid"),
    ("CC(=S)c1ccc(C(=O)O)cc1", "4-(ethanethioyl)benzoic acid"),               #:30246
    ("S=C(S)c1ccccc1", "benzenecarbodithioic acid"),
    ("S=C(S)c1ccc(Cl)cc1", "4-chlorobenzene-1-carbodithioic acid"),
    ("S=C(S)c1ccc(C(=O)O)cc1", "4-(dithiocarboxy)benzoic acid"),
    ("CSC(=O)CC(=O)SC", "S,S-dimethyl propanebis(thioate)"),
    ("C[Se]C(=O)C(=O)[Se]C", "Se,Se-dimethyl ethanebis(selenoate)"),
    ("CCC(=S)c1ccc(C(=O)O)cc1", "4-(propanethioyl)benzoic acid"),
]
THIOACID_NOT_PIN_ROWS = [
    ("S=C(S)c1ccccc1C(=S)S", "1,2-di[sulfanyl(sulfanylidene)methyl]benzene"),
    ("O=C(O)C(O)=S", "(thiocarboxy)formic acid"),
    ("CSC(=O)C(=O)SC", "S-methyl (methylsulfanyl)(oxo)ethanethioate"),
    ("NC(=S)C(=S)c1ccccc1C(=O)O", "2-(2-amino-1,2-disulfanylideneethyl)benzoic acid"),
]
THIOACID_CONTROL_ROWS = [
    ("O=C(S)c1ccccc1", "benzenecarbothioic S-acid"),
    ("S=C(S)C1CCCCC1", "cyclohexanecarbodithioic acid"),
    ("CSC(=O)C", "S-methyl ethanethioate"),
    ("CC(=S)c1ccccc1", "1-phenylethane-1-thione"),
    ("CC(=O)c1ccc(C(=O)O)cc1", "4-acetylbenzoic acid"),
]


@pytest.mark.parametrize("smiles,pin", THIOACID_ROWS + THIOACID_CONTROL_ROWS)
def test_thioacid_pin(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles,non_pin", THIOACID_NOT_PIN_ROWS)
def test_thioacid_not_pin(smiles, non_pin):
    assert_not_pin_labelled(smiles, non_pin)
