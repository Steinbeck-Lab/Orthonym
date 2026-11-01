"""Anions of hydroxy-type groups beyond the 'ol' suffix: the enolate, the imidate and the aminoxide
(``rules.olate_anions``, ``charged_router._OLATE_GROUPS``).

* enolate: "Anions derived from hydroxy compounds" (the Blue Book),:41013: the anion of "a hydroxy
  characteristic group... that can be expressed by a suffix such as 'ol'" takes 'olate'; an enol is such a group
  ('ethenol'), the anion is the parent, class 4 'Anions',:18158) and a ketone of the molecule is the
  prefix 'oxo'. The metal salts of a beta-diketonate of Groups 3-12 are named in the same spelling and are not PINs:
  (:39753) puts them into coordination nomenclature (ruling of 2026-11-01, ``metal_labels.
  is_beta_diketonate_salt_of_group_3_12_metal``); the salts of Groups 1 and 2 keep the label of a salt of a hydroxy compound
  ,:28178).
* imidate: (:40955) replaces "the 'ic acid'... ending of the acid name by 'ate'" and
   "Carboximidic acids" (:29978) names the imidic acids ('ethanimidic acid (PIN)',:29994,
  'benzenecarboximidic acid (PIN)',:30000, 'N-hydroxyethanimidic acid (PIN)',:30106).
* aminoxide: (:41011): "(CH3)2N-O- dimethylaminoxide (PIN)" (:41031), 'aminoxide'
  retained and substituted "in the same way as the corresponding alcohols",:28180).

Every name is read back by an independent OPSIN call (the full standard InChIKey).
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.perception.ions import get_ion_sites
from orthonym.rules import charged_router, olate_anions
from tests.support.rt_assert import name_is_rt_exact

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

ENOLATE_ROWS = [
    # the hydroxyquinone anion of the dev2000 population: the ketones are 'oxo' prefixes of the olate
    ("C=CC/C=C\\C/C=C\\CCCCCCCC1=C([O-])C(=O)C=C(OC)C1=O",
     "4-methoxy-3,6-dioxo-2-[(8Z,11Z)-pentadeca-8,11,14-trien-1-yl]cyclohexa-1,4-dien-1-olate"),
    ("O=C1C=CC(=O)C=C1[O-]", "3,6-dioxocyclohexa-1,4-dien-1-olate"),
    ("C=C[O-]", "ethenolate"),
    ("CC(C)=C[O-]", "2-methylprop-1-en-1-olate"),
    ("CC(=O)C=C([O-])C", "4-oxopent-2-en-2-olate"),
]

IMIDATE_ROWS = [
    ("C(CC(=N)[O-])CS", "4-sulfanylbutanimidate"),
    ("CC(=N)[O-]", "ethanimidate"),
    ("N=C([O-])c1ccccc1", "benzenecarboximidate"),
    ("CC(=NC)[O-]", "N-methylethanimidate"),
    ("CC(=NO)[O-]", "N-hydroxyethanimidate"),
    # the anion is the parent: the carboxylic acid and the amide of the molecule are prefixes
    ("N=C([O-])CCC(=O)O", "3-carboxypropanimidate"),
    ("N=C([O-])CCC(=O)N", "4-amino-4-oxobutanimidate"),
]

AMINOXIDE_ROWS = [
    ("CN(C)[O-]", "dimethylaminoxide"),  # the book's own example,:41031
    ("CCN(CC)[O-]", "diethylaminoxide"),
    ("CC(C1=CC=CC=C1)N(CC2=CC=CC=C2)[O-]", "benzyl(1-phenylethyl)aminoxide"),
    ("c1ccccc1N(C)[O-]", "methyl(phenyl)aminoxide"),
    ("c1ccccc1N(c1ccccc1)[O-]", "diphenylaminoxide"),
    ("C1CC1N(C)[O-]", "cyclopropyl(methyl)aminoxide"),
    ("CN(CC=C)[O-]", "methyl(prop-2-en-1-yl)aminoxide"),
    # the suffix seam named the neutral '2-(methylamino)ethan-1-ol' (its producer drops the N-hydroxy group of
    # CN(O)CCO) and gave '2-(methylamino)ethan-1-olate', a name of another molecule: an N-O(-) anion's seam name must read back
    ("CN(CCO)[O-]", "(2-hydroxyethyl)(methyl)aminoxide"),
    # (:28180): 'aminoxide' "may be substituted in the same way as the corresponding alcohols": a PIN (ruling of 2026-11-01)
    ("CN[O-]", "methylaminoxide"),
    ("CCN[O-]", "ethylaminoxide"),
    ("c1ccccc1N[O-]", "phenylaminoxide"),
]

PIN_ROWS = ENOLATE_ROWS + IMIDATE_ROWS + AMINOXIDE_ROWS

TIERS = ["pin", "best-effort"]


def _row(smiles, tier):
    engine = Orthonym(style="pin", **(_emit_tier_flags(tier) if tier != "pin" else {}))
    return engine.name_tiered(smiles)


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("smiles, pin", PIN_ROWS)
def test_the_anion_is_named_on_its_group(smiles, pin, tier):
    row = _row(smiles, tier)
    assert (row["name"], row["tier"]) == (pin, "pin_verified"), row
    assert name_is_rt_exact(row["name"], smiles), row


@pytest.mark.parametrize("smiles, pin", PIN_ROWS)
def test_the_name_is_the_one_the_route_builds(smiles, pin):
    """The charged route names these (no later producer does)."""
    assert charged_router.route_charged(Chem.MolFromSmiles(smiles), "pin") == pin


#: what the two builders leave to other routes: a hydroxamate (an acyl group on the nitrogen), a nitrogen in a
#: ring (named on the ring, 'piperidin-1-olate'), a cyclic imidic acid, and the unsubstituted H2N-O(-) (the retained
#: 'aminoxide')
DECLINED = [
    "N[O-]",
    "CC(=O)N([O-])C",
    "[O-]N1CCCCC1",
    "[O-]C1=NCCC1",
]


@pytest.mark.parametrize("smiles", DECLINED)
def test_the_builders_decline_what_the_book_example_does_not_cover(smiles):
    mol = Chem.MolFromSmiles(smiles)
    sites = get_ion_sites(mol)
    assert olate_anions.aminoxide_anion_name(mol, sites, "pin") == ""
    assert olate_anions.imidate_anion_name(mol, sites, Chem.MolToSmiles(mol), "pin") == ""


@pytest.mark.parametrize("smiles", ["[O-]CC[NH3+]", "[O-]CC[CH2]", "[O-]c1ccc([O-])cc1", "CC(=N)[O-].[Na+]"])
def test_a_cation_a_radical_a_second_anion_or_a_salt_is_not_this_route(smiles):
    mol = Chem.MolFromSmiles(smiles)
    sites = get_ion_sites(mol)
    neutral = Chem.MolToSmiles(mol)
    assert olate_anions.imidate_anion_name(mol, sites, neutral, "pin") == ""
    assert olate_anions.aminoxide_anion_name(mol, sites, "pin") == ""


#: the salts of the acetylacetonate anion (ruling of 2026-11-01): the label is the metal's, the spelling the olate's
ACETYLACETONATE_SALTS = [
    # Groups 1 and 2: a salt of a hydroxy compound keeps pin_verified,:28178)
    ("CC(=O)C=C([O-])C.[Na+]", "sodium 4-oxopent-2-en-2-olate", "pin_verified"),
    ("CC(=O)C=C([O-])C.CC(=O)C=C([O-])C.[Mg+2]", "magnesium bis(4-oxopent-2-en-2-olate)", "pin_verified"),
    # Groups 3-12: a coordination compound,:39753), no PIN: the name was 'copper bis(2-oxido-4-oxopent-2-ene)'
    ("CC(=O)C=C([O-])C.CC(=O)C=C([O-])C.[Cu+2]", "copper bis(4-oxopent-2-en-2-olate)", "systematic_verified"),
    ("CC(=O)C=C([O-])C.CC(=O)C=C([O-])C.[Zn+2]", "zinc bis(4-oxopent-2-en-2-olate)", "systematic_verified"),
]


@pytest.mark.parametrize("smiles, name, tier", ACETYLACETONATE_SALTS)
def test_the_acetylacetonate_salts_take_the_olate_and_the_label_of_their_metal(smiles, name, tier):
    row = _row(smiles, "best-effort")
    assert (row["name"], row["tier"]) == (name, tier), row
    assert row["is_pin"] is (tier == "pin_verified"), row
    assert name_is_rt_exact(row["name"], smiles), row


def test_a_neutral_amine_oxide_and_the_neutral_imidic_acid_keep_their_names():
    # the oxide drawn N+-O- is the amine oxide of, never the aminoxide anion
    for smiles, name in [("C[N+](C)(C)[O-]", "N,N-dimethylmethanamine N-oxide"),
                         ("CC(=N)O", "ethanimidic acid"), ("CN(C)O", "N-hydroxy-N-methylmethanamine")]:
        row = _row(smiles, "pin")
        assert (row["name"], row["tier"]) == (name, "pin_verified"), row
