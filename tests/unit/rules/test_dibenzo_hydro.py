"""Hydro (part-saturated) forms of the intrinsic-indicated-H mancude annulene
parents that B1 added — the tricyclic-antidepressant / antihistamine drug cores.

B1 (commit 2a68b44e1) added the aromatic mancude parents
``5H-dibenzo[a,d][7]annulene`` (6-7-6) and ``1H/5H/7H-benzo[7]annulene`` (6-7)
to FUSED_HETEROCYCLE_DATA. The drug scaffolds are the HYDRO forms of those
parents and still abstained, because no path composed a ``x,y-dihydro`` prefix
onto a mancude parent that ALREADY carries an intrinsic indicated hydrogen.

B1b closes that: the suffix-free sibling of ``name_cyclic_oxo_compound``
(``name_hydro_mancude_fused_carbocycle``) reuses the proven mancude-parent +
added-indicated-H machinery with NO ring ketone.

Rule chain (the Blue Book):
  * (:8069) — an odd-n [n]annulene carries its extra H as INDICATED
    hydrogen; those annulene names are the PINs for fusion parent components
    . So the parent is spelled ``5H-dibenzo[a,d][7]annulene``.
  * (:3721) — "in a preferred IUPAC name a locant and the symbol 'H'
    must be cited"; the indicated hydrogen precedes the parent stem.
  * / (:1682) — hydro prefixes are detachable, nonalphabetized,
    placed just before the parent, and numbered by lowest locants AFTER priority
    has been given to indicated hydrogen. Order: [hydro]-[indicated H]-[parent]
    -> ``10,11-dihydro-5H-dibenzo[a,d][7]annulene``.

Every input SMILES is OPSIN 2.9.0's own structure for the target PIN (derived
name -> structure), so the asserted string is not hand-written. Oracle: a REAL
OPSIN round-trip (name -> OPSIN -> full InChIKey == input), mirroring
tests/unit/rules/test_benzo_annulene.py.
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


# (input SMILES, expected exact PIN). Each SMILES is OPSIN's own structure for
# the name (name -> OPSIN -> this structure), so the string is authoritative.
HYDRO_TARGETS = [
    # dibenzosuberane — the saturated (10,11-dihydro) dibenzo[a,d][7]annulene
    # core of amitriptyline / nortriptyline / protriptyline. C15H14.
    ("c1ccc2c(c1)CCc1ccccc1C2", "10,11-dihydro-5H-dibenzo[a,d][7]annulene"),
    # benzosuberane, dihydro form (one residual ring C=C). C11H12.
    ("C1=Cc2ccccc2CCC1", "6,7-dihydro-5H-benzo[7]annulene"),
    # benzsuberane — fully saturated 7-ring (benzsuberone precursor). C11H14.
    ("c1ccc2c(c1)CCCCC2", "6,7,8,9-tetrahydro-5H-benzo[7]annulene"),
    # NON-ADJACENT ("1,4-type") dihydro: the two hydro positions are separated
    # by the residual ring C=C, so they are NOT a reduced-C=C adjacent pair
    # (7=8 stays). PIN is the lowest-locant set {6,9}, NOT {8,9} (a review-2 fix).
    # C11H12, InChIKey an InChIKey.
    ("C1CC=CCc2ccccc21", "6,9-dihydro-5H-benzo[7]annulene"),
]


@pytest.mark.parametrize("smiles,expected", HYDRO_TARGETS)
def test_hydro_annulene_core_names_and_rt(namer, smiles, expected):
    name = namer.name(smiles)
    assert name == expected, (smiles, name)
    assert _full_rt(smiles, name), (smiles, name)


def test_nonadjacent_dihydro_takes_lowest_locants(namer):
    """a review-2 regression: for the non-adjacent ('1,4-type') dihydro isomer the
    hydro positions are chosen by LOWEST locants, not realised as an adjacent
    reduced-C=C pair.

     NUMBERING (the Blue Book): low locants are assigned in
    decreasing seniority to (b) indicated hydrogen (:3246) then (e)(i) 'hydro'
    prefixes (:3288). With the indicated H fixed at 5, {6,9} < {8,9} by,
    so the PIN is ``6,9-dihydro-5H-benzo[7]annulene``. Non-adjacent 'added
    hydrogen' is standard — cf. 1,4-dihydronaphthalene (PIN), the Blue Book.
    Both spellings round-trip to the same structure, so the old ``8,9-`` form was
    RIGHT_MOL / NON-PIN (0-wrong held), but not the lowest-locant name.
    """
    smi = "C1CC=CCc2ccccc21"
    name = namer.name(smi)
    assert name == "6,9-dihydro-5H-benzo[7]annulene", name
    assert name != "8,9-dihydro-5H-benzo[7]annulene", "non-lowest-locant name"
    assert _full_rt(smi, name), (smi, name)


def test_bare_dibenzo_parent_still_names(namer):
    """B1 positive control: the bare mancude parent (cyproheptadine core) must
    still name after B1b (0 regression to B1)."""
    name = namer.name("C1=Cc2ccccc2Cc2ccccc21")
    assert name == "5H-dibenzo[a,d][7]annulene", name
    assert _full_rt("C1=Cc2ccccc2Cc2ccccc21", name)


def test_tetralin_path_unregressed(namer):
    """The pre-existing naphthalene-family partial-saturation path (no intrinsic
    indicated H) must be untouched — B1b is scoped to intrinsic-iH parents only."""
    assert namer.name("c1ccc2c(c1)CCCC2") == "1,2,3,4-tetrahydronaphthalene"
    assert namer.name("C1CCC2CCCCC2C1") == "decahydronaphthalene"


def test_substituted_whole_drug_never_wrong(namer):
    """A substituted whole drug (amitriptyline) is out of the bare-core scope;
    it must either name correctly or abstain — NEVER a wrong structure (0-wrong).
    """
    ami = "CN(C)CCC=C1c2ccccc2CCc2ccccc21"
    name = namer.name(ami)
    if name and name != "unknown organic compound":
        assert _full_rt(ami, name), ("amitriptyline shipped a WRONG name", name)
