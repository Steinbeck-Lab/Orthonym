"""Aromatic [n]annulene benzo/dibenzo-fused parents, fusion nomenclature).

New FUSED_HETEROCYCLE_DATA entries for the mancude 7-/8-membered ortho-fused
carbocyclic parents that were fail-closed ('unknown organic compound') at HEAD:

  * benzo[8]annulene (6-8 fused, C12H10, no indicated H)
  * 1H/5H/7H-benzo[7]annulene (6-7 fused, C11H10, one indicated H each)
  * 5H-dibenzo[a,d][7]annulene (6-7-6 fused, C15H12, indicated H at 5)

These join the SAME lookup that already names `heptalene` (7-7) and
`1H-cyclopenta[8]annulene` (5-8) — a data-only addition, no rule change.

WHY the brief's example SMILES were replaced: the brief cited
`C1=CC=CC2=CC=CC=CC=C12` as "benzo[7]annulene" and `C1=CC2=CC=CC=CC=C2C=C1` as
"heptalene". Both actually parse (RDKit) to the SAME molecule, InChIKey
`LSNYJLGMVHJXPD`, C12H10 — which is **benzo[8]annulene**, not either named
parent (real benzo[7]annulene is C11H10 with an indicated H; real heptalene is a
distinct C12H10, `DDTGNKBZWQHIEH`, and ALREADY names at HEAD). Every SMILES below
is the OPSIN-authoritative structure for its parent name, RT-verified.

Oracle: a REAL OPSIN round-trip — name the input, parse the emitted name back
through OPSIN, compare full InChIKeys. Mirrors tests/unit/rules/test_heptacene.py.
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym import Orthonym
from orthonym.validation.opsin_roundtrip import opsin_parse

pytestmark = pytest.mark.opsin_gate


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


def _full_rt(smi, name):
    """True iff `name` round-trips through OPSIN to the exact input InChIKey."""
    if not name or name == "unknown organic compound":
        return False
    opsin_smi = opsin_parse(name)
    if not opsin_smi:
        return False
    m_in = Chem.MolFromSmiles(smi)
    m_out = Chem.MolFromSmiles(opsin_smi)
    if m_in is None or m_out is None:
        return False
    return inchi.MolToInchiKey(m_in) == inchi.MolToInchiKey(m_out)


# (input SMILES, expected bare-parent name). Each SMILES is OPSIN's own
# structure for the name (verified: name -> OPSIN -> InChIKey == this SMILES).
BARE_PARENTS = [
    # benzo[8]annulene — 6-8 ortho-fused, fully mancude (no indicated H).
    # This is the molecule the B1 brief mislabelled twice; it is also a
    # bb_conformance gold row (expected 'benzo[8]annulene').
    ("C1=CC=Cc2ccccc2C=C1", "benzo[8]annulene"),
    # benzo[7]annulene — 6-7 ortho-fused, one indicated H. Three distinct
    # tautomers (1H, 5H(=9H), 7H), each its own molecule.
    ("C1=CC=C2CC=CC=C2C=C1", "1H-benzo[7]annulene"),
    ("C1=CCc2ccccc2C=C1", "5H-benzo[7]annulene"),
    ("C1=Cc2ccccc2C=CC1", "7H-benzo[7]annulene"),
    # 5H-dibenzo[a,d][7]annulene — the dibenzo core of the TCA / antihistamine
    # scaffolds (imipramine, amitriptyline, cyproheptadine, loratadine).
    ("C1=Cc2ccccc2Cc2ccccc21", "5H-dibenzo[a,d][7]annulene"),
]


@pytest.mark.parametrize("smiles,expected", BARE_PARENTS)
def test_bare_annulene_parent_names_and_rt(namer, smiles, expected):
    name = namer.name(smiles)
    assert name == expected, (smiles, name)
    assert _full_rt(smiles, name), (smiles, name)


def test_regression_azulene_heptalene_still_name(namer):
    """The pre-existing 5-7 (azulene) and 7-7 (heptalene) parents that share the
    same lookup must be unaffected."""
    assert namer.name("c1ccc2cccc-2cc1") == "azulene"
    assert namer.name("C1=CC=C2C=CC=CC=C2C=C1") == "heptalene"
