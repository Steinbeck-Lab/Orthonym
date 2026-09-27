"""Suite fix j4 (TRIAGE g3 C10c): a substituent name its producer already wrapped
in plain parentheses gets its outer mark from its content.

 (the Blue Book): "When multiple types of enclosing marks are
required, the nesting order is as follows: {[({})]}".: the
parentheses of a stereodescriptor count. The benzene amido branch returned
'((2R)-2-chloropropanamido)', which format_substituent_prefix cited unchanged,
so '(' sat directly around '(' -- '4-((2S,3R)-...-4-oxobutanamido)benzamido'
at pin_verified. Every name below: OPSIN 2.9.0 full-InChIKey and canonical-SMILES
exact (independent batch, suite fix j4).
"""
import pytest

from orthonym import name_compound
from orthonym.assembly.naming_utils import format_substituent_prefix


@pytest.mark.parametrize("name,locants,count,expected", [
    ("((2R)-2-chloropropanamido)", [4], 1, "4-[(2R)-2-chloropropanamido]"),
    ("((2R)-2-chloropropanamido)", [3, 5], 2, "3,5-bis[(2R)-2-chloropropanamido]"),
    ("(4-amino-3-methoxy-2-[4-(4-nitrobenzamido)benzamido]-4-oxobutanamido)",
     [4], 1,
     "4-{4-amino-3-methoxy-2-[4-(4-nitrobenzamido)benzamido]-4-oxobutanamido}"),
    # content at nesting depth 0 keeps its parentheses (fusion brackets and
    # indicated hydrogen do not count, /.1.2)
    ("(2-methylpropyl)", [3], 1, "3-(2-methylpropyl)"),
    ("(furo[3,2-b]pyridin-2-yl)", [2], 1, "2-(furo[3,2-b]pyridin-2-yl)"),
    ("(1H-indol-3-yl)", [2], 1, "2-(1H-indol-3-yl)"),
])
def test_prewrapped_name_is_remarked_by_content(name, locants, count, expected):
    assert format_substituent_prefix(name, locants, count) == expected


@pytest.mark.parametrize("smiles,expected", [
    ("OC(=O)c1ccc(NC(=O)[C@@H](C)Cl)cc1",
     "4-[(2R)-2-chloropropanamido]benzoic acid"),
    ("OC(=O)c1ccc(NC(=O)[C@@H](NC(=O)c3ccccc3)[C@@H](OC)C(N)=O)cc1",
     "4-[(2S,3R)-4-amino-2-benzamido-3-methoxy-4-oxobutanamido]benzoic acid"),
    ("OC(=O)c1ccc(NC(=O)C(NC(=O)c3ccc(NC(=O)c4ccc([N+](=O)[O-])cc4)cc3)C(OC)C(N)=O)cc1",
     "4-{4-amino-3-methoxy-2-[4-(4-nitrobenzamido)benzamido]-4-oxobutanamido}"
     "benzoic acid"),
    ("CCOc1cc(C(=O)O)ccc1NC(=O)c1ccc(NC(=O)c2ccc(NC(=O)[C@@H](NC(=O)c3ccc(NC(=O)"
     "c4ccc([N+](=O)[O-])cc4)cc3)[C@@H](OC)C(N)=O)cc2)c(OC(C)C)c1O",
     "4-[4-(4-{(2S,3R)-4-amino-3-methoxy-2-[4-(4-nitrobenzamido)benzamido]-4-"
     "oxobutanamido}benzamido)-2-hydroxy-3-[(propan-2-yl)oxy]benzamido]-3-"
     "ethoxybenzoic acid"),
])
def test_amido_prefix_nesting_end_to_end(smiles, expected):
    assert name_compound(smiles) == expected, "P-16.5.4"


# The composer's N-attached fallback wrapped a compound carbon branch in '(' and
# then the whole '...amino' in '(' again: '1-(((4-fluorophenyl)methyl)amino)',
# re-marked outside only to '1-{((4-fluorophenyl)methyl)amino}' (pin_verified).
@pytest.mark.parametrize("smiles,expected", [
    ("CC(CNCC1=CC=C(C=C1)F)O", "1-{[(4-fluorophenyl)methyl]amino}propan-2-ol"),
    ("OCC(O)CNCc1ccc(Cl)cc1", "3-{[(4-chlorophenyl)methyl]amino}propane-1,2-diol"),
    ("CC(O)CNCc1ccncc1", "1-{[(pyridin-4-yl)methyl]amino}propan-2-ol"),
])
def test_n_branch_amino_prefix_nesting_end_to_end(smiles, expected):
    assert name_compound(smiles) == expected, "P-16.5.4"
