"""An amine / aminium name cites its N- and C-prefixes in ONE alphanumerical order, and a
substituted ethane parent keeps its locant '1' (fix a performance pass, wp7; verification panel
RISK on test_n_cation_non_pin_labels' choline control).

* (the Blue Book): "the omission of the locant '1' in 2-chloroethanol...
  is not allowed in preferred IUPAC names, thus the name 2-chloroethan-1-ol is the PIN";
  '2-aminoethan-1-aminium chloride (PIN)' (:43572). The suffix-locant tightening was
  scoped to NEUTRAL parents, so a charged one shipped '2-hydroxy-N,N,N-trimethyl-
  ethanaminium' at pin_verified (handlers/_handler_shared).
* (:3442; the N-substituent sorts among the C-prefixes in '4-(2-methylbutyl)-N-
  (3-methylbutyl)aniline (PIN)',:3523): the amine assembler always cited the C block
  first ('2-fluoro-N,N,N-triethylethan-1-aminium'). A locant-less C-prefix (a methanamine)
  cited second takes parentheses:7304, 'bromo(chloro)acetic acid (PIN)').

Each witness is named in six SMILES spellings; each must give the one name, and each name
was checked by an independent OPSIN 2.9.0 full-InChIKey round trip.
"""
import pytest

from orthonym import name_compound

pytestmark = pytest.mark.unit

WITNESSES = [
    ('OCC[N+](C)(C)C', '2-hydroxy-N,N,N-trimethylethan-1-aminium'),
    ('C[N+](CCO)(C)C', '2-hydroxy-N,N,N-trimethylethan-1-aminium'),
    ('C[N+](C)(CCO)C', '2-hydroxy-N,N,N-trimethylethan-1-aminium'),
    ('[N+](C)(CCO)(C)C', '2-hydroxy-N,N,N-trimethylethan-1-aminium'),
    ('C([N+](C)(C)C)CO', '2-hydroxy-N,N,N-trimethylethan-1-aminium'),
    ('C(O)C[N+](C)(C)C', '2-hydroxy-N,N,N-trimethylethan-1-aminium'),
    ('ClCC[N+](C)(C)C', '2-chloro-N,N,N-trimethylethan-1-aminium'),
    ('C[N+](CCCl)(C)C', '2-chloro-N,N,N-trimethylethan-1-aminium'),
    ('C[N+](C)(CCCl)C', '2-chloro-N,N,N-trimethylethan-1-aminium'),
    ('[N+](C)(CCCl)(C)C', '2-chloro-N,N,N-trimethylethan-1-aminium'),
    ('C([N+](C)(C)C)CCl', '2-chloro-N,N,N-trimethylethan-1-aminium'),
    ('C(Cl)C[N+](C)(C)C', '2-chloro-N,N,N-trimethylethan-1-aminium'),
    ('C[N+](C)(C)CCOC(N)=O.[Cl-]', '2-(carbamoyloxy)-N,N,N-trimethylethan-1-aminium chloride'),
    ('C(=O)(OCC[N+](C)(C)C)N.[Cl-]', '2-(carbamoyloxy)-N,N,N-trimethylethan-1-aminium chloride'),
    ('NC(OCC[N+](C)(C)C)=O.[Cl-]', '2-(carbamoyloxy)-N,N,N-trimethylethan-1-aminium chloride'),
    ('O=C(OCC[N+](C)(C)C)N.[Cl-]', '2-(carbamoyloxy)-N,N,N-trimethylethan-1-aminium chloride'),
    ('C[N+](CCOC(=O)N)(C)C.[Cl-]', '2-(carbamoyloxy)-N,N,N-trimethylethan-1-aminium chloride'),
    ('[N+](C)(C)(CCOC(N)=O)C.[Cl-]', '2-(carbamoyloxy)-N,N,N-trimethylethan-1-aminium chloride'),
    ('CC[N+](CC)(CC)CCF', 'N,N,N-triethyl-2-fluoroethan-1-aminium'),
    ('C([N+](CC)(CC)CC)CF', 'N,N,N-triethyl-2-fluoroethan-1-aminium'),
    ('C(F)C[N+](CC)(CC)CC', 'N,N,N-triethyl-2-fluoroethan-1-aminium'),
    ('FCC[N+](CC)(CC)CC', 'N,N,N-triethyl-2-fluoroethan-1-aminium'),
    ('C([N+](CC)(CCF)CC)C', 'N,N,N-triethyl-2-fluoroethan-1-aminium'),
    ('[N+](CC)(CCF)(CC)CC', 'N,N,N-triethyl-2-fluoroethan-1-aminium'),
    ('CC[N+](C)(CC)CCOC(=O)C(=C)C', 'N,N-diethyl-N-methyl-2-[(2-methylprop-2-enoyl)oxy]ethan-1-aminium'),
    ('C(C[N+](C)(CC)CC)OC(C(C)=C)=O', 'N,N-diethyl-N-methyl-2-[(2-methylprop-2-enoyl)oxy]ethan-1-aminium'),
    ('C(COC(C(=C)C)=O)[N+](CC)(CC)C', 'N,N-diethyl-N-methyl-2-[(2-methylprop-2-enoyl)oxy]ethan-1-aminium'),
    ('CC[N+](CCOC(=O)C(=C)C)(CC)C', 'N,N-diethyl-N-methyl-2-[(2-methylprop-2-enoyl)oxy]ethan-1-aminium'),
    ('C(C)[N+](CCOC(C(=C)C)=O)(CC)C', 'N,N-diethyl-N-methyl-2-[(2-methylprop-2-enoyl)oxy]ethan-1-aminium'),
    ('C[N+](CC)(CC)CCOC(=O)C(=C)C', 'N,N-diethyl-N-methyl-2-[(2-methylprop-2-enoyl)oxy]ethan-1-aminium'),
    ('C[N+](C)([O-])CCc1ccccc1', 'N,N-dimethyl-2-phenylethan-1-amine N-oxide'),
    ('[N+]([O-])(C)(C)CCc1ccccc1', 'N,N-dimethyl-2-phenylethan-1-amine N-oxide'),
    ('c1cc(CC[N+]([O-])(C)C)ccc1', 'N,N-dimethyl-2-phenylethan-1-amine N-oxide'),
    ('[O-][N+](C)(C)CCc1ccccc1', 'N,N-dimethyl-2-phenylethan-1-amine N-oxide'),
    ('c1cc(CC[N+](C)(C)[O-])ccc1', 'N,N-dimethyl-2-phenylethan-1-amine N-oxide'),
    ('C(C[N+]([O-])(C)C)c1ccccc1', 'N,N-dimethyl-2-phenylethan-1-amine N-oxide'),
    ('ClCN(C)C', 'chloro-N,N-dimethylmethanamine'),
    ('N(CCl)(C)C', 'chloro-N,N-dimethylmethanamine'),
    ('CN(CCl)C', 'chloro-N,N-dimethylmethanamine'),
    ('CN(C)CCl', 'chloro-N,N-dimethylmethanamine'),
    ('C(Cl)N(C)C', 'chloro-N,N-dimethylmethanamine'),
    ('N(C)(CCl)C', 'chloro-N,N-dimethylmethanamine'),
    ('ClCCN(C)C', '2-chloro-N,N-dimethylethan-1-amine'),
    ('C(Cl)CN(C)C', '2-chloro-N,N-dimethylethan-1-amine'),
    ('C(CCl)N(C)C', '2-chloro-N,N-dimethylethan-1-amine'),
    ('N(C)(C)CCCl', '2-chloro-N,N-dimethylethan-1-amine'),
    ('CN(C)CCCl', '2-chloro-N,N-dimethylethan-1-amine'),
    ('CN(CCCl)C', '2-chloro-N,N-dimethylethan-1-amine'),
    ('CC[N+](C)(C)C', 'N,N,N-trimethylethanaminium'),
    ('C(C)[N+](C)(C)C', 'N,N,N-trimethylethanaminium'),
    ('[N+](C)(CC)(C)C', 'N,N,N-trimethylethanaminium'),
    ('C[N+](C)(CC)C', 'N,N,N-trimethylethanaminium'),
    ('C[N+](C)(C)CC', 'N,N,N-trimethylethanaminium'),
    ('C([N+](C)(C)C)C', 'N,N,N-trimethylethanaminium'),
]


@pytest.mark.parametrize("smiles,expected", WITNESSES)
def test_amine_prefix_order_and_ethane_locant(smiles, expected):
    assert name_compound(smiles) == expected
