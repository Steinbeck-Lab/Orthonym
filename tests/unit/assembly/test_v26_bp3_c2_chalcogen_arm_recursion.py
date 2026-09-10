""" cluster C2 — chalcogen-ether / sulfinyl-sulfonyl ARM recursion.

The `-S-R` / `-Se-R` / `-Te-R` / `-S(=O)-R` / `-S(=O)(=O)-R` substituent arm was
named by CARBON-COUNT, collapsing unsaturation / branching / hetero / aryl to a
saturated linear alkyl (allyl -> propyl, isobutyl -> butyl, 2-hydroxyethyl ->
ethyl, phenyl -> hexyl) — a constitutionally different molecule that then
suppressed to `unknown`.

C2-A recurses into the arm via ``name_substituent_fragment`` and encloses a
complex arm per (BB 27836 ``[(penta-1,4-dien-3-yl)sulfanyl]cyclobutane``;
BB 25713 ``(prop-2-en-1-yl)cyclohexane``). C2-B builds the acid-stem sulfinyl /
sulfonyl PIN for an arm ``_classify_oxide_side`` declines.

These assert the exact builder-return strings (no OPSIN); the enclosing-mark and
fail-closed contract are what this lever changes. Full-name round-trips are
verified separately via scripts/diagnose.py.
"""
import pytest
from rdkit import Chem

from orthonym.assembly.substituent_prefix_forms import (
    get_sulfanyl_prefix,
    get_sulfinyl_prefix,
    get_sulfonyl_prefix,
)
from orthonym.perception.functional_groups import FUNCTIONAL_GROUP_SMARTS


def _match(mol, fg):
    pat = Chem.MolFromSmarts(FUNCTIONAL_GROUP_SMARTS[fg])
    m = mol.GetSubstructMatches(pat)
    return m[0] if m else None


# ---------------------------------------------------------------------------
# C2-A — get_sulfanyl_prefix: complex arm enclosed, simple arm bare
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,expected", [
    ("C=CCSCCC(=O)O", "(prop-2-en-1-yl)sulfanyl"),   # unsaturated (allyl)
    ("CC(C)CSCCC(=O)O", "(2-methylpropyl)sulfanyl"),  # branched (isobutyl)
    ("OCCSCCC(=O)O", "(2-hydroxyethyl)sulfanyl"),     # hetero-bearing
])
def test_c2a_complex_arm_enclosed(smiles, expected):
    m = Chem.MolFromSmiles(smiles)
    assert get_sulfanyl_prefix(m, _match(m, "thioether"), None) == expected


@pytest.mark.parametrize("smiles,expected", [
    ("CSCCC(=O)O", "methylsulfanyl"),   # regression: simple methyl stays bare
    ("CCSCCC(=O)O", "ethylsulfanyl"),   # regression: simple ethyl stays bare
])
def test_c2a_simple_arm_bare_regression(smiles, expected):
    m = Chem.MolFromSmiles(smiles)
    assert get_sulfanyl_prefix(m, _match(m, "thioether"), None) == expected


def test_c2a_selanyl_complex_and_simple():
    # Se via suffix: complex allyl enclosed
    m = Chem.MolFromSmiles("C=CC[Se]CCC(=O)O")
    assert get_sulfanyl_prefix(
        m, _match(m, "selenoether"), None, suffix="selanyl"
    ) == "(prop-2-en-1-yl)selanyl"
    # regression: simple methyl stays bare
    m = Chem.MolFromSmiles("C[Se]CCC(=O)O")
    assert get_sulfanyl_prefix(
        m, _match(m, "selenoether"), None, suffix="selanyl"
    ) == "methylselanyl"


def test_c2a_acyl_on_chalcogen_guard_untouched():
    # The acyl-on-chalcogen guard precedes the new tail: S-acetyl stays acyl,
    # never collapsed to an alkyl by the recursion.
    m = Chem.MolFromSmiles("CC(=O)SCCC(=O)O")
    assert get_sulfanyl_prefix(m, _match(m, "thioether"), None) == "acetylsulfanyl"


# ---------------------------------------------------------------------------
# C2-B — get_sulfinyl_prefix / get_sulfonyl_prefix: acid-stem for declined arm
# ---------------------------------------------------------------------------

def test_c2b_sulfinyl_unsaturated_acid_stem():
    m = Chem.MolFromSmiles("C=CCS(=O)CCC(=O)O")
    assert get_sulfinyl_prefix(
        m, _match(m, "sulfoxide"), None
    ) == "prop-2-ene-1-sulfinyl"


def test_c2b_sulfinyl_saturated_regression():
    # saturated linear arm still handled by _classify_oxide_side (byte-identical)
    m = Chem.MolFromSmiles("CS(=O)CCC(=O)O")
    assert get_sulfinyl_prefix(
        m, _match(m, "sulfoxide"), None
    ) == "methanesulfinyl"


def test_c2b_sulfonyl_unsaturated_acid_stem():
    m = Chem.MolFromSmiles("C=CCS(=O)(=O)CCC(=O)O")
    assert get_sulfonyl_prefix(
        m, _match(m, "sulfone"), None
    ) == "prop-2-ene-1-sulfonyl"


def test_c2b_sulfonyl_saturated_regression():
    m = Chem.MolFromSmiles("CS(=O)(=O)CCC(=O)O")
    assert get_sulfonyl_prefix(
        m, _match(m, "sulfone"), None
    ) == "methanesulfonyl"
