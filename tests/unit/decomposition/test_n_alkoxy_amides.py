"""The N-alkoxy amides (hydroxamic acid ethers) cite the group R-O- as the prefix 'R-oxy'.

The decomposition engine names an amide by cutting its C(=O)-N bond and joining the two names.
The amine fragment of an N-alkoxy amide is an O-substituted hydroxylamine ('O-methylhydroxylamine'),
and the prefix of the amide nitrogen was read from that name by taking off 'amine':
'N-(O-methylhydroxyl)acetamide', 'N-(O-benzylhydroxyl)...acetamide'. 'O-methylhydroxyl' is no
prefix of the Blue Book. The amine fragment of a Weinreb amide is named 'N-methoxymethanamine',
whose N-substituent 'methoxy' was not read as one (only '-yl' words were), so the acyl was floated
onto the amine ('N-acetyl-N-methoxymethanamine') and the default tier declined.

Rules:
- "N-SUBSTITUTION OF AMINES AND IMINES BY HETEROATOMS" (the Blue Book): "Substitution
  of amines is permitted by -OR, -SR, -SeR, and -TeR groups" (:26596), 'N-methoxyethanamine (PIN)'
  (:26600).
- "N,O-Disubstitution of hydroxylamine" (:38372): 'N-methoxymethanamine (PIN)'
  (:38380).
- "Hydroxamic acids" (:32821): "named as N-hydroxy amides", 'N-hydroxypropanamide (PIN)'
  (:32825).
- "Retained names" (:27665): 'methoxy', 'ethoxy', 'propoxy', 'butoxy', 'phenoxy',
  'tert-butoxy' are simple prefixes; (:27633): the others are concatenated
  ('benzyloxy', '(propan-2-yl)oxy').

INFERRED, not verbatim: the Blue Book gives no N-alkoxy amide PIN. The amide names below follow
from the three rules above (the N-substituent of an amide is cited as a prefix with the locant N,
,:32774) and from the verbatim 'N'-ethoxy-1H-imidazole-2-carboximidamide (PIN)'
 "Amidoximes (amide oximes)",:34654, example:34671).
"""
import pytest

from orthonym.decomposition.amide_n_substituents import split_secondary_amine_name
from orthonym.decomposition.fragment_assembly import (
    _amine_to_prefix,
    _assemble_amide,
    _assemble_carbamate,
    _assemble_sulfonamide,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("amine,prefix", [
    ("O-methylhydroxylamine", "methoxy"),
    ("O-ethylhydroxylamine", "ethoxy"),
    ("O-propylhydroxylamine", "propoxy"),
    ("O-phenylhydroxylamine", "phenoxy"),
    ("O-tert-butylhydroxylamine", "tert-butoxy"),
    ("O-benzylhydroxylamine", "benzyloxy"),
    ("O-pentylhydroxylamine", "pentyloxy"),
    ("O-(propan-2-yl)hydroxylamine", "(propan-2-yl)oxy"),
    ("O-(butan-2-yl)hydroxylamine", "(butan-2-yl)oxy"),
    ("O-(2-methylpropyl)hydroxylamine", "2-methylpropoxy"),
    ("O-(oxan-2-yl)hydroxylamine", "(oxan-2-yl)oxy"),
    ("O-[(2S)-butan-2-yl]hydroxylamine", "[(2S)-butan-2-yl]oxy"),
])
def test_o_substituted_hydroxylamine_gives_its_alkoxy_prefix(amine, prefix):
    assert _amine_to_prefix(amine) == prefix


@pytest.mark.parametrize("amine,prefix", [
    ("hydroxylamine", "hydroxy"),            #: N-hydroxy amide, unchanged
    ("methylamine", "methyl"),
    ("aniline", "phenyl"),
])
def test_other_amine_names_are_unchanged(amine, prefix):
    assert _amine_to_prefix(amine) == prefix


@pytest.mark.parametrize("amine", [
    # an O-name that also carries an N-substituent, or no group at all, is no single O-group
    "O-methyl-N-phenylhydroxylamine",
    "O-hydroxylamine",
    "O-prop-2-en-1-ylhydroxylamine",
])
def test_unconvertible_o_name_has_no_prefix(amine):
    # 'O-...hydroxyl' is no prefix of the Blue Book, the Blue Book); the base
    # returned one from the rule that takes 'amine' off a name, and a PIN was labelled with it
    assert _amine_to_prefix(amine) is None


@pytest.mark.parametrize("amine,split", [
    ("N-methoxymethanamine", ("methoxy", "methanamine")),
    ("N-ethoxyethanamine", ("ethoxy", "ethanamine")),
    ("N-methoxyaniline", ("methoxy", "aniline")),
    ("N-tert-butoxymethanamine", ("tert-butoxy", "methanamine")),
    ("N-(benzyloxy)methanamine", ("(benzyloxy)", "methanamine")),
    # the '-yl' words are read as before
    ("N-methylethanamine", ("methyl", "ethanamine")),
    ("N-tert-butyl-2-methylpropan-2-amine", ("tert-butyl", "2-methylpropan-2-amine")),
])
def test_alkoxy_n_substituent_is_read_as_one_word(amine, split):
    assert split_secondary_amine_name(amine) == split


@pytest.mark.parametrize("names,smiles,expected", [
    # the amine is the O-substituted hydroxylamine: one N-substituent R-oxy
    ({"acid": "acetic acid", "amine": "O-methylhydroxylamine"},
     {"acid": "CC(=O)O", "amine": "CON"}, "N-methoxyacetamide"),
    ({"acid": "benzoic acid", "amine": "O-benzylhydroxylamine"},
     {"acid": "OC(=O)c1ccccc1", "amine": "NOCc1ccccc1"}, "N-(benzyloxy)benzamide"),
    ({"acid": "acetic acid", "amine": "O-(propan-2-yl)hydroxylamine"},
     {"acid": "CC(=O)O", "amine": "CC(C)ON"}, "N-[(propan-2-yl)oxy]acetamide"),
    ({"acid": "acetic acid", "amine": "O-tert-butylhydroxylamine"},
     {"acid": "CC(=O)O", "amine": "CC(C)(C)ON"}, "N-tert-butoxyacetamide"),
    # the amine is N,O-disubstituted (a Weinreb amide for N-methoxy-N-methyl): two N-substituents,
    # in alphanumerical order
    ({"acid": "acetic acid", "amine": "N-methoxymethanamine"},
     {"acid": "CC(=O)O", "amine": "CNOC"}, "N-methoxy-N-methylacetamide"),
    ({"acid": "benzoic acid", "amine": "N-ethoxyethanamine"},
     {"acid": "OC(=O)c1ccccc1", "amine": "CCNOCC"}, "N-ethoxy-N-ethylbenzamide"),
])
def test_n_alkoxy_amide_assembly(names, smiles, expected):
    assert _assemble_amide(names, "pin", smiles) == expected


@pytest.mark.xfail(strict=True, reason=(
    "_amine_to_prefix('N-methoxyaniline') is None (as for every N-alkyl aniline name), so "
    "_assemble_amide floats the acyl: 'N-benzoyl-N-methoxyaniline'"))
def test_n_methoxy_n_phenyl_amide_assembly():
    got = _assemble_amide(
        {"acid": "benzoic acid", "amine": "N-methoxyaniline"}, "pin",
        {"acid": "OC(=O)c1ccccc1", "amine": "CONc1ccccc1"})
    assert got == "N-methoxy-N-phenylbenzamide"


@pytest.mark.parametrize("amine,expected", [
    # a compound N-substituent is enclosed, the Blue Book, "Parentheses are used
    # around compound... prefixes";,:7416, "Brackets enclose substituent prefixes in
    # which parentheses have already been used"), a simple one is cited bare
    ("O-benzylhydroxylamine", "N-(benzyloxy)methanesulfonamide"),
    ("O-pentylhydroxylamine", "N-(pentyloxy)methanesulfonamide"),
    ("O-(propan-2-yl)hydroxylamine", "N-[(propan-2-yl)oxy]methanesulfonamide"),
    ("O-methylhydroxylamine", "N-methoxymethanesulfonamide"),
    ("ethylamine", "N-ethylmethanesulfonamide"),
])
def test_n_alkoxy_sulfonamide_assembly(amine, expected):
    assert _assemble_sulfonamide({"acid": "methanesulfonic acid", "amine": amine}, "pin") == expected


@pytest.mark.parametrize("amine,expected", [
    ("O-benzylhydroxylamine", "methyl (benzyloxy)carbamate"),
    ("O-methylhydroxylamine", "methyl methoxycarbamate"),
])
def test_n_alkoxy_carbamate_assembly_encloses_a_compound_prefix(amine, expected):
    assert _assemble_carbamate({"alkyl": "methanol", "amine": amine}, "pin") == expected
