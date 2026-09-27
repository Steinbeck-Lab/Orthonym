"""Suite fix j4 (TRIAGE g3 C07): the algorithmic 2-component fusion path cites
the indicated hydrogen of a mancude system whose saturated atom RDKit perceives
as aromatic (a pyrrole-type N-H or N-R).

 (the Blue Book): "noncumulative double bonds are introduced
into the completed fused system. Hydrogen atoms not attached to atoms connected
by double bonds are denoted as indicated hydrogen atom(s)."
 (:14607): "In preferred IUPAC names, all indicated hydrogen atoms
must be cited when the names are constructed in accordance with the principles
of fusion nomenclature".

The legacy counter skipped aromatic atoms, so the path shipped
'thieno[2,3-c]pyrrole' and -- for two tautomers of one skeleton -- ONE bare name
('imidazo[4,5-c]pyridine' for both the 1H and the 3H tautomer). The standard
InChIKey cannot tell tautomers apart (mobile H), so the round-trip gate passed
the wrong one; these tests compare the OPSIN parse on canonical SMILES, which
does. Every value below: OPSIN 2.9.0 exact on the full InChIKey AND on canonical
SMILES (independent batch, suite fix j4).
"""
import pytest
from rdkit import Chem

from orthonym import name_compound
from tests.support.rt_assert import _independent_parse

ROWS = [
    # bare systems
    ("c1cc2c[nH]cc2s1", "5H-thieno[2,3-c]pyrrole"),
    ("c1cc2c[nH]cc2o1", "5H-furo[2,3-c]pyrrole"),
    ("c1cc2[nH]ncc2s1", "1H-thieno[3,2-c]pyrazole"),
    ("c1cc2n[nH]cc2s1", "2H-thieno[3,2-c]pyrazole"),
    ("c1cc2c[nH]nc2s1", "2H-thieno[2,3-c]pyrazole"),
    ("c1cc2[nH]cnc2cn1", "1H-imidazo[4,5-c]pyridine"),
    ("c1cc2nc[nH]c2cn1", "3H-imidazo[4,5-c]pyridine"),
    ("c1cnc2[nH]nnc2c1", "3H-[1,2,3]triazolo[4,5-b]pyridine"),
    # substituted systems (the indicated hydrogen stays with the N; an
    # N-substituent replaces its hydrogen: 5-methyl-5H-...)
    ("Cc1cc2c[nH]cc2s1", "2-methyl-5H-thieno[2,3-c]pyrrole"),
    ("Cn1cc2sccc2c1", "5-methyl-5H-thieno[2,3-c]pyrrole"),
    ("c1cc2cn(C)cc2o1", "5-methyl-5H-furo[2,3-c]pyrrole"),
    ("c1cc2c(s1)c[nH]c2C(=O)O", "5H-thieno[2,3-c]pyrrole-4-carboxylic acid"),
    ("O=C(O)c1cc2c[nH]cc2s1", "5H-thieno[2,3-c]pyrrole-2-carboxylic acid"),
    ("Cc1cc2nc[nH]c2cn1", "6-methyl-3H-imidazo[4,5-c]pyridine"),
    # a saturated ring carbon with no hydrogen is the indicated position too
    ("CC1(C)C=Cc2ncccc21", "5,5-dimethyl-5H-cyclopenta[b]pyridine"),
]

# Boundaries: no indicated hydrogen in the mancude parent, or the
# 'added indicated hydrogen' form,:3725), which this fix leaves alone.
UNCHANGED = [
    ("c1ncc2ccoc2n1", "furo[2,3-d]pyrimidine"),
    ("c1cc2ccoc2cn1", "furo[2,3-c]pyridine"),
    ("O=c1[nH]ccc2sccc12", "thieno[3,2-c]pyridin-4(5H)-one"),
]


def _canon(smi):
    mol = Chem.MolFromSmiles(smi) if smi else None
    return Chem.MolToSmiles(mol) if mol is not None else None


@pytest.mark.parametrize("smiles,expected", ROWS + UNCHANGED)
def test_fused_name_cites_indicated_hydrogen(smiles, expected):
    assert name_compound(smiles) == expected, "P-25.7.1.3.1"


@pytest.mark.roundtrip
@pytest.mark.parametrize("smiles,expected", ROWS)
def test_fused_name_denotes_the_input_tautomer(smiles, expected):
    """The emitted name parses back to the input on canonical SMILES, which
    distinguishes tautomers that share one standard InChIKey."""
    name = name_compound(smiles)
    parsed = _independent_parse(name)
    assert parsed is not None, f"OPSIN could not parse {name!r}"
    assert _canon(parsed) == _canon(smiles), (
        f"{name!r} parses to {parsed!r}, not the input tautomer {smiles!r}")
