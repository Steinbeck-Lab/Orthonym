"""Boron-oxygen (and boron-carbon) fused-ring parents fusion +
Hantzsch-Widman skeletal-replacement 'bora').

New FUSED_HETEROCYCLE_DATA entries for the mancude boron-heterocyclic parents
that were fail-closed ('unknown organic compound') at HEAD (there were 0 boron
entries in the catalog):

  * 2H-1,3,2-benzodioxaborole (6-5 ortho-fused, O-B-O 5-ring on benzene)
  * 6H-dibenzo[d,f][1,3,2]dioxaborepine (6-7-6, O-B-O bridge across a biphenyl)
  * 2H-naphtho[1,8-de][1,3,2]dioxaborinine (peri-fused O-B-O on naphthalene)
  * 5H-phenanthro[4,5-def][1,3,2]dioxaborepine (O-B-O bridge across a phenanthrene bay)
  * 5H-dibenzo[b,d]borole (dibenzoborole; the fluorene-boron skeleton)
  * 10H-dibenzo[b,e][1,4]oxaborinine (6-6-6 dibenzo-1,4-oxaborine; the largest
                                          fused-boron class in the 1M corpus, 1,196)

These join the SAME lookup that already names the carbocyclic mancude parents
(heptalene, benzo[8]annulene) and every retained heterocycle — a data-only
addition, no rule change (the fused-core matcher accepts a boron ring atom
unchanged; verified by these tests).

Indicated hydrogen, the Blue Book; heading "Citation of
indicated hydrogen", the Blue Book): a preferred IUPAC name built by fusion
nomenclature MUST cite every indicated H. The boron is the single sp3 (saturated)
atom the mancude O-B-O / C-B / O-B ring needs, so its locant carries the cited
nH- (2H/5H/6H/10H). The H-LESS spelling ('1,3,2-benzodioxaborole') is GENERAL
nomenclature only (the Blue Book permits omission when unambiguous, e.g. '1,3-benzo-
dioxole rather than 2H-1,3-benzodioxole'). The indicated H is retained when the
boron is substituted (2-methyl-2H-1,3,2-benzodioxaborole), exactly as the
verbatim BB PIN '2-phenyl-2H,4H-[1,3,2]dioxaborolo[4,5-d]imidazole' (the Blue Book)
keeps its 2H under a 2-phenyl.

Ring names: the O,O,B five-/six-/seven-membered rings are Hantzsch-Widman
'dioxaborole/-inine/-epine' HW; 'oxa' O + 'bora' B, the 'a'-prefix
for boron per Table 2.8; heteroatom order O before B by seniority
O >... > B), fused onto benzo/naphtho/phenanthro components by fusion
nomenclature. The '1,3,2' set = O(1),O(3),B(2). 5H-dibenzo[b,d]borole is a
benzo-fused borole with a second benzo on the [b] and [d] edges (the Blue Book
PIN for the fluorene-boron skeleton, preferred over the 'bora'-replacement
synonym 9-borafluorene). 10H-dibenzo[b,e][1,4]oxaborinine is the O,B analogue of
dibenzofuran/acridine; the BB spells this ring 'phenoxaborinine' (the Blue Book) but
OPSIN 2.9.0 rejects that spelling, so the systematic 'dibenzo[b,e][1,4]-
oxaborinine' fusion name (which OPSIN parses and RTs) is emitted. Every
canonical-SMILES key and iupac_locants map was derived from OPSIN 2.9.0's own
`<name> -o extendedsmi` $_AV locants and RT-verified.

Oracle: a REAL OPSIN round-trip — name the input, parse the emitted name back
through OPSIN, compare full InChIKeys. Mirrors test_benzo_annulene.py.
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
    ("B1Oc2ccccc2O1", "2H-1,3,2-benzodioxaborole"),
    ("B1Oc2ccccc2-c2ccccc2O1", "6H-dibenzo[d,f][1,3,2]dioxaborepine"),
    ("B1Oc2cccc3cccc(O1)c23", "2H-naphtho[1,8-de][1,3,2]dioxaborinine"),
    ("B1Oc2cccc3ccc4cccc(O1)c4c23", "5H-phenanthro[4,5-def][1,3,2]dioxaborepine"),
    ("B1c2ccccc2-c2ccccc21", "5H-dibenzo[b,d]borole"),
    ("c1ccc2c(c1)Oc1ccccc1B2", "10H-dibenzo[b,e][1,4]oxaborinine"),
]

# (input SMILES, expected name). >= 1 substituted form per parent, to prove the
# CLASS names (the locant map is exercised), not just the bare skeleton. The
# indicated H is INHERITED by the substituted forms: substituting
# the boron does not drop the cited nH-, mirroring the Blue Book's '2-phenyl-2H,...'.
SUBSTITUTED = [
    # 2H-1,3,2-benzodioxaborole: substituent on the boron (position 2)
    ("CB1Oc2ccccc2O1", "2-methyl-2H-1,3,2-benzodioxaborole"),
    ("c1ccc(cc1)B1Oc2ccccc2O1", "2-phenyl-2H-1,3,2-benzodioxaborole"),
    ("FB1Oc2ccccc2O1", "2-fluoro-2H-1,3,2-benzodioxaborole"),
    # 6H-dibenzo[d,f][1,3,2]dioxaborepine: substituent on the boron (position 6)
    ("c1ccc(cc1)B1Oc2ccccc2-c2ccccc2O1", "6-phenyl-6H-dibenzo[d,f][1,3,2]dioxaborepine"),
    ("FB1Oc2ccccc2-c2ccccc2O1", "6-fluoro-6H-dibenzo[d,f][1,3,2]dioxaborepine"),
    # 2H-naphtho[1,8-de][1,3,2]dioxaborinine: substituent on the boron (position 2)
    ("CB1Oc2cccc3cccc(O1)c23", "2-methyl-2H-naphtho[1,8-de][1,3,2]dioxaborinine"),
    # 5H-phenanthro[4,5-def][1,3,2]dioxaborepine: substituent on the boron (position 5)
    ("CB1Oc2cccc3ccc4cccc(O1)c4c23", "5-methyl-5H-phenanthro[4,5-def][1,3,2]dioxaborepine"),
    # 5H-dibenzo[b,d]borole: a ring substituent on a benzo ring, and on the boron
    ("Cc1ccc2c(c1)-c1ccccc1B2", "2-methyl-5H-dibenzo[b,d]borole"),
    ("CB1c2ccccc2-c2ccccc21", "5-methyl-5H-dibenzo[b,d]borole"),
    # 10H-dibenzo[b,e][1,4]oxaborinine: substituent on the boron (position 10)
    ("c1ccc2c(c1)Oc1ccccc1B2-c1ccccc1", "10-phenyl-10H-dibenzo[b,e][1,4]oxaborinine"),
]


@pytest.mark.parametrize("smiles,expected", BARE_PARENTS)
def test_bare_boron_parent_names_and_rt(namer, smiles, expected):
    name = namer.name(smiles)
    assert name == expected, (smiles, name)
    assert _full_rt(smiles, name), (smiles, name)


@pytest.mark.parametrize("smiles,expected", SUBSTITUTED)
def test_substituted_boron_parent_names_and_rt(namer, smiles, expected):
    name = namer.name(smiles)
    assert name == expected, (smiles, name)
    assert _full_rt(smiles, name), (smiles, name)


def test_regression_saturated_and_boronic_acid_still_name(namer):
    """The pre-existing (non-fused) boron forms must be unaffected: the saturated
    1,3,2-dioxaborolane ring and phenylboronic acid still name as before."""
    assert namer.name("CB1OC(C)(C)C(C)(C)O1") == "2,4,4,5,5-pentamethyl-1,3,2-dioxaborolane"
    assert namer.name("OB(O)c1ccccc1") == "phenylboronic acid"
