""" alphanumerical order: the Roman LETTERS decide first (fix a performance pass, wp7).

``### **** ALPHANUMERICAL ORDER`` (the Blue Book): "Nonitalic Roman
letters are considered first, unless used as locants... When all the Roman letters
are identical, the set of locants... are compared"; (:3448) multiplicative
prefixes "do not alter the alphabetical order"; (:3477) "The name of a
prefix for a substituent is considered to begin with the first letter of its
complete name"; (:3495) italic letters are considered only when the Roman
letters tie.

The shared key (``naming_utils.alpha_sort_key``, 93 call sites; it also decides ring
numbering through the (g) tiers,:3307) used to compare its key TEXT as a
plain string, so a hyphen or a digit, both below every lowercase letter in ASCII,
outranked a letter: 'prop-1-en-2-yl' < 'propan-2-ylidene' ('-' < 'a'),
'1-hydroxy-4-methylpentyl' < 'hydroxymethyl', a primed rendered prefix
"3',4,4'-trihydroxy" keyed at its digit, 'di-tert-butyl' at '-tert-butyl',
'bis(methoxycarbonyl)' at 'b', and 'butyl' / 'tert-butyl' tied (input order decided).

Every witness is named in six SMILES spellings (the input plus five RDKit random
spellings with the same canonical SMILES); each must give the one expected name. Every
expected name was checked by an independent OPSIN 2.9.0 full-InChIKey round trip
(``java -jar... -r -osmi`` outside the engine). The last three are Blue Book (PIN)
rows: ':3507' 1-(butan-2-yl)-3-tert-butylbenzene, ':3465' 4-butyl-4-tert-butyl-
cyclohexan-1-ol, ':3461' 5-(butan-2-yl)-5-butylhentriacontane.
"""
import pytest

from orthonym import name_compound
from orthonym.assembly.naming_utils import alpha_sort_key, prefix_citation_sort_key

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("first,second", [
    # letters first: propanylidene < propenyl:3442)
    ("propan-2-ylidene", "prop-1-en-2-yl"),
    # butanyl < butenyl; butanyl < butyl (:3461 '5-(butan-2-yl)-5-butylhentriacontane (PIN)')
    ("butan-2-yl", "but-2-en-1-yl"),
    ("butan-2-yl", "butyl"),
    # an initial segment comes first (:21663 '...-4-methyl-3-methylidenehexanoic acid (PIN)')
    ("hydroxymethyl", "1-hydroxy-4-methylpentyl"),
    ("methyl", "methylidene"),
    # primed locants are locants (:3442)
    ("2',4',5',7'-tetrabromo", "3',6'-dihydroxy"),
    ("3',4,4'-trihydroxy", "5'-pentyl"),
    # multipliers are ignored:3448), 'di-tert-' and 'bis(' included
    ("2,6-di-tert-butyl", "4-(propan-2-ylidene)"),
    ("4,5-dimethoxy", "3,6-bis(4-methoxyphenyl)"),
    ("22,23-bis(methoxycarbonyl)", "5-(3-methoxy-3-oxopropyl)"),
    # an italic prefix inside a compound prefix is not a letter (:25963
    # '2-(tert-butylimino)-3-methyl-3-(nitrooxy)butanoic acid (PIN)')
    ("[(tert-butoxycarbonyl)amino]", "3,3-dimethyl"),
    # an italic locant letter is not a letter (':3442' "unless used as locants")
    ("6,14-dihydroxy", "18-[(1H-indol-3-yl)methyl]"),
    # italic letters decide only a Roman-letter tie, absence first:3495,:3465)
    ("butyl", "tert-butyl"),
    ("sec-butyl", "tert-butyl"),
])
def test_letters_decide_first(first, second):
    assert alpha_sort_key(first) < alpha_sort_key(second)
    assert sorted([second, first], key=alpha_sort_key) == [first, second]
    assert sorted([first, second], key=alpha_sort_key) == [first, second]
    # the citation key agrees on the letters tier
    assert prefix_citation_sort_key(first)[0] <= prefix_citation_sort_key(second)[0]


def test_key_text_is_unchanged():
    """The key's TEXT (what callers and tests inspect) is the same as before; only its
    order is 's."""
    assert alpha_sort_key("dimethyl") == "methyl"
    assert alpha_sort_key("tert-butyl") == "butyl"
    assert alpha_sort_key("pentan-2-yl") == "pentan-2-yl"
    assert "zzzzz" > alpha_sort_key("methyl") and alpha_sort_key("methyl") < "zzzzz"


WITNESSES = [
    ('CC(C)=C1CCCC(C1)C(C)=C', '1-(propan-2-ylidene)-3-(prop-1-en-2-yl)cyclohexane'),
    ('C(=C1CC(C(=C)C)CCC1)(C)C', '1-(propan-2-ylidene)-3-(prop-1-en-2-yl)cyclohexane'),
    ('C1C(C(=C)C)CCCC1=C(C)C', '1-(propan-2-ylidene)-3-(prop-1-en-2-yl)cyclohexane'),
    ('C1(=C(C)C)CC(C(=C)C)CCC1', '1-(propan-2-ylidene)-3-(prop-1-en-2-yl)cyclohexane'),
    ('CC(C1CCCC(=C(C)C)C1)=C', '1-(propan-2-ylidene)-3-(prop-1-en-2-yl)cyclohexane'),
    ('C1CC(CC(C1)=C(C)C)C(C)=C', '1-(propan-2-ylidene)-3-(prop-1-en-2-yl)cyclohexane'),
    ('CCC(C)C1CCCC(C1)C/C=C/C', '1-(butan-2-yl)-3-[(2E)-but-2-en-1-yl]cyclohexane'),
    ('C1C(C/C=C/C)CC(CC1)C(C)CC', '1-(butan-2-yl)-3-[(2E)-but-2-en-1-yl]cyclohexane'),
    ('C1CC(CC(C/C=C/C)C1)C(CC)C', '1-(butan-2-yl)-3-[(2E)-but-2-en-1-yl]cyclohexane'),
    ('C1CCC(C/C=C/C)CC1C(CC)C', '1-(butan-2-yl)-3-[(2E)-but-2-en-1-yl]cyclohexane'),
    ('C1(C(CC)C)CCCC(C1)C/C=C/C', '1-(butan-2-yl)-3-[(2E)-but-2-en-1-yl]cyclohexane'),
    ('CC(CC)C1CCCC(C/C=C/C)C1', '1-(butan-2-yl)-3-[(2E)-but-2-en-1-yl]cyclohexane'),
    ('C=C(C)C1CCC(C(C)=C)C(=C(C)C)C1', '2-(propan-2-ylidene)-1,4-di(prop-1-en-2-yl)cyclohexane'),
    ('C(C)(=C)C1C(CC(C(C)=C)CC1)=C(C)C', '2-(propan-2-ylidene)-1,4-di(prop-1-en-2-yl)cyclohexane'),
    ('CC(=C)C1C(CC(C(=C)C)CC1)=C(C)C', '2-(propan-2-ylidene)-1,4-di(prop-1-en-2-yl)cyclohexane'),
    ('C=C(C1C(CC(C(=C)C)CC1)=C(C)C)C', '2-(propan-2-ylidene)-1,4-di(prop-1-en-2-yl)cyclohexane'),
    ('C1(=C(C)C)C(CCC(C(C)=C)C1)C(=C)C', '2-(propan-2-ylidene)-1,4-di(prop-1-en-2-yl)cyclohexane'),
    ('C(C)(=C1CC(CCC1C(C)=C)C(=C)C)C', '2-(propan-2-ylidene)-1,4-di(prop-1-en-2-yl)cyclohexane'),
    ('CC(C)CCC(O)C1C(=O)OCC1CO', '4-(hydroxymethyl)-3-(1-hydroxy-4-methylpentyl)oxolan-2-one'),
    ('C1(C(OCC1CO)=O)C(O)CCC(C)C', '4-(hydroxymethyl)-3-(1-hydroxy-4-methylpentyl)oxolan-2-one'),
    ('C1(OCC(C1C(CCC(C)C)O)CO)=O', '4-(hydroxymethyl)-3-(1-hydroxy-4-methylpentyl)oxolan-2-one'),
    ('O=C1OCC(C1C(CCC(C)C)O)CO', '4-(hydroxymethyl)-3-(1-hydroxy-4-methylpentyl)oxolan-2-one'),
    ('O1CC(CO)C(C1=O)C(CCC(C)C)O', '4-(hydroxymethyl)-3-(1-hydroxy-4-methylpentyl)oxolan-2-one'),
    ('C1C(C(C(O)CCC(C)C)C(O1)=O)CO', '4-(hydroxymethyl)-3-(1-hydroxy-4-methylpentyl)oxolan-2-one'),
    ('C=C1CCC(=C(C)C)CC1C(=C)C', '1-methylidene-4-(propan-2-ylidene)-2-(prop-1-en-2-yl)cyclohexane'),
    ('C1(CCC(CC1C(C)=C)=C(C)C)=C', '1-methylidene-4-(propan-2-ylidene)-2-(prop-1-en-2-yl)cyclohexane'),
    ('C1CC(C(C(C)=C)CC1=C(C)C)=C', '1-methylidene-4-(propan-2-ylidene)-2-(prop-1-en-2-yl)cyclohexane'),
    ('C(=C1CC(C(=C)C)C(=C)CC1)(C)C', '1-methylidene-4-(propan-2-ylidene)-2-(prop-1-en-2-yl)cyclohexane'),
    ('CC(=C1CCC(=C)C(C(=C)C)C1)C', '1-methylidene-4-(propan-2-ylidene)-2-(prop-1-en-2-yl)cyclohexane'),
    ('C1(C(=C)C)C(CCC(=C(C)C)C1)=C', '1-methylidene-4-(propan-2-ylidene)-2-(prop-1-en-2-yl)cyclohexane'),
    ('CC(=C1C=C(C(=O)C(=C1)C(C)(C)C)C(C)(C)C)C', '2,6-di-tert-butyl-4-(propan-2-ylidene)cyclohexa-2,5-dien-1-one'),
    ('C(C)(C)=C1C=C(C(C)(C)C)C(C(=C1)C(C)(C)C)=O', '2,6-di-tert-butyl-4-(propan-2-ylidene)cyclohexa-2,5-dien-1-one'),
    ('CC(C)(C)C1C(C(C(C)(C)C)=CC(=C(C)C)C=1)=O', '2,6-di-tert-butyl-4-(propan-2-ylidene)cyclohexa-2,5-dien-1-one'),
    ('C(C)(C1=CC(=C(C)C)C=C(C1=O)C(C)(C)C)(C)C', '2,6-di-tert-butyl-4-(propan-2-ylidene)cyclohexa-2,5-dien-1-one'),
    ('C1(C(C)(C)C)C(=O)C(C(C)(C)C)=CC(=C(C)C)C=1', '2,6-di-tert-butyl-4-(propan-2-ylidene)cyclohexa-2,5-dien-1-one'),
    ('CC(C)=C1C=C(C(C(=C1)C(C)(C)C)=O)C(C)(C)C', '2,6-di-tert-butyl-4-(propan-2-ylidene)cyclohexa-2,5-dien-1-one'),
    ('COC1=C(c2ccc(OC)cc2)C(=O)C(=O)C(c2ccc(OC)cc2)=C1OC', '4,5-dimethoxy-3,6-bis(4-methoxyphenyl)cyclohexa-3,5-diene-1,2-dione'),
    ('O(C)c1ccc(cc1)C1C(=O)C(C(=C(OC)C=1OC)c1ccc(OC)cc1)=O', '4,5-dimethoxy-3,6-bis(4-methoxyphenyl)cyclohexa-3,5-diene-1,2-dione'),
    ('c1(OC)ccc(C2C(=O)C(=O)C(c3ccc(OC)cc3)=C(C=2OC)OC)cc1', '4,5-dimethoxy-3,6-bis(4-methoxyphenyl)cyclohexa-3,5-diene-1,2-dione'),
    ('c1c(OC)ccc(C2=C(C(=C(C(=O)C2=O)c2ccc(OC)cc2)OC)OC)c1', '4,5-dimethoxy-3,6-bis(4-methoxyphenyl)cyclohexa-3,5-diene-1,2-dione'),
    ('c1cc(ccc1C1C(=O)C(=O)C(=C(OC)C=1OC)c1ccc(cc1)OC)OC', '4,5-dimethoxy-3,6-bis(4-methoxyphenyl)cyclohexa-3,5-diene-1,2-dione'),
    ('c1(C2C(=O)C(C(c3ccc(cc3)OC)=C(OC)C=2OC)=O)ccc(OC)cc1', '4,5-dimethoxy-3,6-bis(4-methoxyphenyl)cyclohexa-3,5-diene-1,2-dione'),
    ('CC(C)(C)[C@H](C(=O)[O-])NC(=O)OC(C)(C)C', '(2R)-2-[(tert-butoxycarbonyl)amino]-3,3-dimethylbutanoate'),
    ('C(OC(C)(C)C)(N[C@H](C(C)(C)C)C(=O)[O-])=O', '(2R)-2-[(tert-butoxycarbonyl)amino]-3,3-dimethylbutanoate'),
    ('N([C@@H](C(=O)[O-])C(C)(C)C)C(OC(C)(C)C)=O', '(2R)-2-[(tert-butoxycarbonyl)amino]-3,3-dimethylbutanoate'),
    ('[O-]C(=O)[C@H](NC(=O)OC(C)(C)C)C(C)(C)C', '(2R)-2-[(tert-butoxycarbonyl)amino]-3,3-dimethylbutanoate'),
    ('O=C([O-])[C@@H](C(C)(C)C)NC(=O)OC(C)(C)C', '(2R)-2-[(tert-butoxycarbonyl)amino]-3,3-dimethylbutanoate'),
    ('C([C@H](NC(OC(C)(C)C)=O)C(C)(C)C)(=O)[O-]', '(2R)-2-[(tert-butoxycarbonyl)amino]-3,3-dimethylbutanoate'),
    ('CCC(C)c1cccc(C(C)(C)C)c1', '1-(butan-2-yl)-3-tert-butylbenzene'),
    ('c1c(cc(C(CC)C)cc1)C(C)(C)C', '1-(butan-2-yl)-3-tert-butylbenzene'),
    ('c1cc(cc(C(C)(C)C)c1)C(CC)C', '1-(butan-2-yl)-3-tert-butylbenzene'),
    ('c1ccc(cc1C(C)CC)C(C)(C)C', '1-(butan-2-yl)-3-tert-butylbenzene'),
    ('c1(C(CC)C)cccc(C(C)(C)C)c1', '1-(butan-2-yl)-3-tert-butylbenzene'),
    ('CC(CC)c1cccc(c1)C(C)(C)C', '1-(butan-2-yl)-3-tert-butylbenzene'),
    ('CCCCC1(C(C)(C)C)CCC(O)CC1', '4-butyl-4-tert-butylcyclohexan-1-ol'),
    ('CC(C)(C1(CCC(CC1)O)CCCC)C', '4-butyl-4-tert-butylcyclohexan-1-ol'),
    ('CC(C)(C)C1(CCC(CC1)O)CCCC', '4-butyl-4-tert-butylcyclohexan-1-ol'),
    ('C1CC(CCC1(C(C)(C)C)CCCC)O', '4-butyl-4-tert-butylcyclohexan-1-ol'),
    ('C1C(O)CCC(C1)(C(C)(C)C)CCCC', '4-butyl-4-tert-butylcyclohexan-1-ol'),
    ('C1(O)CCC(CC1)(CCCC)C(C)(C)C', '4-butyl-4-tert-butylcyclohexan-1-ol'),
    ('CCCCCCCCCCCCCCCCCCCCCCCCCCC(CCCC)(CCCC)C(C)CC', '5-(butan-2-yl)-5-butylhentriacontane'),
    ('C(CCCCCCCCCCCCCCCCCCCCCCCCC(C(C)CC)(CCCC)CCCC)C', '5-(butan-2-yl)-5-butylhentriacontane'),
    ('C(CCC(C(CC)C)(CCCCCCCCCCCCCCCCCCCCCCCCCC)CCCC)C', '5-(butan-2-yl)-5-butylhentriacontane'),
    ('C(CCCCCCCC(CCCC)(CCCC)C(CC)C)CCCCCCCCCCCCCCCCCC', '5-(butan-2-yl)-5-butylhentriacontane'),
    ('C(CCCCCCCCCCCCCCCCCCC(C(CC)C)(CCCC)CCCC)CCCCCCC', '5-(butan-2-yl)-5-butylhentriacontane'),
    ('C(C(CCCC)(CCCC)CCCCCCCCCCCCCCCCCCCCCCCCCC)(C)CC', '5-(butan-2-yl)-5-butylhentriacontane'),
]


@pytest.mark.parametrize("smiles,expected", WITNESSES)
def test_letters_first_order_and_numbering(smiles, expected):
    assert name_compound(smiles) == expected
