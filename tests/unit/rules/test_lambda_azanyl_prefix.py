"""P-62.5(2): oxo-λ5-azanyl prefixes for an N-oxide on a substituent N.

The full-molecule expected PIN is the Blue Book P-62.5 example verbatim;
OPSIN-2.9.0 parses the Greek-λ spelling back to the input structure
(verified 2026-08-16; ASCII 'lambda' also parses, byte-identical structure).
"""
import pytest
from rdkit import Chem
from orthonym.namer import name_compound
from orthonym.assembly.substituent_naming import (
    _find_amine_oxide_n,
    _lambda5_azanyl_prefix,
)


@pytest.mark.unit
def test_dimethyl_oxo_azanyl_methyl_fragment():
    # single-oxide benzylic analog (the two-oxide parent tie-break is a
    # separate, nondeterministic concern — see test_two_oxide_bluebook_pin).
    mol = Chem.MolFromSmiles("C[N+]([O-])(C)Cc1ccccc1")
    patt = Chem.MolFromSmarts("[cH0]-[CH2]-[N+]([CH3])([CH3])[O-]")
    ring_c, ch2, n, me1, me2, o = mol.GetSubstructMatch(patt)
    assert _lambda5_azanyl_prefix(mol, [ch2, n, me1, me2, o], ch2) == \
        "[dimethyl(oxo)-λ5-azanyl]methyl"


@pytest.mark.unit
def test_bare_oxo_azanyl_ethyl_fragment():
    # BB P-62.5 substituent example: -CH2-CH2-NH2(O) -> 2-(oxo-λ5-azanyl)ethyl
    mol = Chem.MolFromSmiles("[O-][NH2+]CCOC(=O)c1ccccc1")
    patt = Chem.MolFromSmarts("[O-][NH2+][CH2][CH2]O")
    o, n, c2, c1, o_ester = mol.GetSubstructMatch(patt)
    assert _lambda5_azanyl_prefix(mol, [c1, c2, n, o], c1) == \
        "2-(oxo-λ5-azanyl)ethyl"


@pytest.mark.unit
def test_nitro_fragment_not_matched():
    # nitro N carries a DOUBLE bond -> detector must decline (no regression
    # for nitro prefixes)
    mol = Chem.MolFromSmiles("O=[N+]([O-])Cc1ccccc1")
    frag = [a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic() is False]
    assert _find_amine_oxide_n(mol, frag) is None


@pytest.mark.xfail(
    reason="P-62.5 two-oxide PIN needs 3 coupled upstream fixes beyond the "
    "lambda5-azanyl builder: (1) a deterministic amine-oxide parent tie-break "
    "(longer/senior amine chain wins) among equal-seniority oxide centres; "
    "(2) a composer substituent-walk BYPASS that emits the lambda vocabulary "
    "outside the guarded name_substituent_fragment chokepoint; (3) the validity "
    "gate not suppressing the grammar-invalid emitted name. FOLLOW-UP: build "
    "the amine-oxide parent tie-break, then this yields the BB PIN "
    "'2-(3-{[dimethyl(oxo)-λ5-azanyl]methyl}phenyl)-N,N-dimethylethan-1-amine "
    "N-oxide'. The lambda5-azanyl builder itself is correct (fragment tests).",
    strict=False,
)
@pytest.mark.unit
def test_two_oxide_bluebook_pin():
    assert name_compound("C[N+]([O-])(C)Cc1cccc(CC[N+](C)(C)[O-])c1") == (
        "2-(3-{[dimethyl(oxo)-λ5-azanyl]methyl}phenyl)-"
        "N,N-dimethylethan-1-amine N-oxide"
    )


@pytest.mark.unit
def test_single_oxide_protect():
    # method (1) path unchanged (recorded at HEAD, OPSIN-RT clean)
    assert name_compound("C[N+](C)([O-])CCc1ccccc1") == \
        "2-phenyl-N,N-dimethylethan-1-amine N-oxide"


@pytest.mark.unit
def test_nitrobenzene_protect():
    assert name_compound("O=[N+]([O-])c1ccccc1") == "nitrobenzene"
