"""Suite fix j5-pin-labels-b (TRIAGE.md 'Suite fix -- j5-pin-labels-b'): the names the
job's producer fixes changed beyond the triage rows, each asserted exactly (gate on,
PIN tier) with an independent full-InChIKey OPSIN round trip.

Blue Book basis per group:
- azido prefix on a substituent chain: AZIDES (the Blue Book), '(2-azido
  ethyl)benzene (PIN)' (:25995); 'azidomethyl' is compound, (:7232).
- one level of enclosing marks per fragment, escalated: (:7444).
- decorated biphenylyl substituents are the ring assembly: (:15560),
  '(4'-cyano[1,1'-biphenyl]-4-yl)oxy... (PIN)' (:7455).
- concatenated alkyloxy prefixes are compound: (:27667),,
  '(benzyloxy)carbonyl (preferred prefix)' (:18116).
- N-substituted cycloalkanamines omit the ring locant: (c) (:2913),
  '*N*-butylcyclopropanamine (PIN)' (:26292).
- carbamic acid substituents take no N locant: (:30758,:30762).
- BF4-/PF6-: method (1) (:41097); the sodium salt is a wholly inorganic metal
  compound, labelled systematic_verified (branch review fixes;:2062), and the
  carbon-free anion itself too (texts, labels and spelling;:2056/:2058).
"""
import pytest

from orthonym.namer import Orthonym
from tests.support.rt_assert import name_is_rt_exact

pytestmark = pytest.mark.opsin_gate

PIN_NAMES = [
    # azido on a substituent chain (Pass 1c3) + compound 'azidomethyl'
    ("[N-]=[N+]=NCCCc1ccccc1", "(3-azidopropyl)benzene"),
    ("[N-]=[N+]=NC(C)CCc1ccccc1", "(3-azidobutyl)benzene"),
    ("[N-]=[N+]=NCc1ccccc1", "(azidomethyl)benzene"),
    ("OC(=O)c1ccc(CN=[N+]=[N-])cc1", "4-(azidomethyl)benzoic acid"),
    ("OC(=O)CC(CN=[N+]=[N-])CC", "3-(azidomethyl)pentanoic acid"),
    # principal substituent chain = the most substituents, criterion (k)
    # (the Blue Book, '2-hydroxy-1-methylethyl [not 1-(hydroxymethyl)ethyl]');
    # was '[2-(azidomethyl)propyl]benzene' / '[2-(nitromethyl)propyl]benzene'
    # (TRIAGE j12 finding 3)
    ("[N-]=[N+]=NCC(C)Cc1ccccc1", "(3-azido-2-methylpropyl)benzene"),
    ("O=[N+]([O-])CC(C)Cc1ccccc1", "(2-methyl-3-nitropropyl)benzene"),
    # monosubstituted benzene escalates its outer mark (the longest chain,,
    # comes before criterion (k): the butyl chain carries the nitromethyl branch)
    ("O=[N+]([O-])CC(CC)Cc1ccccc1", "[2-(nitromethyl)butyl]benzene"),
    # decorated biphenylyl
    ("CC(=O)Oc1ccc(-c2ccc(C)cc2)cc1", "4'-methyl[1,1'-biphenyl]-4-yl acetate"),
    ("CC(=O)Oc1ccc(-c2ccc(Cl)cc2Cl)cc1",
     "2',4'-dichloro[1,1'-biphenyl]-4-yl acetate"),
    ("CC(=O)Oc1ccc(-c2ccccc2)c(C)c1", "2-methyl[1,1'-biphenyl]-4-yl acetate"),
    ("CC(=O)Oc1cccc(-c2ccc(C)cc2)c1", "4'-methyl[1,1'-biphenyl]-3-yl acetate"),
    # concatenated alkyloxy prefixes
    ("CCCCCOCCc1ccccc1", "[2-(pentyloxy)ethyl]benzene"),
    ("Oc1ccc(COC)cc1", "4-(methoxymethyl)phenol"),
    ("Oc1ccc(COc2ccccc2)cc1", "4-(phenoxymethyl)phenol"),
    # N-substituted ring amines
    ("CCCCNC1CC1", "N-butylcyclopropanamine"),
    ("CN(C)C1CCCCC1", "N,N-dimethylcyclohexanamine"),
    ("CCNC1CCC(C)CC1", "N-ethyl-4-methylcyclohexan-1-amine"),
    ("CCNC1CCCC=C1", "N-ethylcyclohex-2-en-1-amine"),
    # carbamic acid family
    ("OC(=O)N(C)CC", "ethyl(methyl)carbamic acid"),
    ("OC(=O)N(CCO)CCO", "bis(2-hydroxyethyl)carbamic acid"),
    ("COC(=O)N(C)C", "methyl dimethylcarbamate"),
    # anions: 'hexafluoro-λ5-phosphanuide' is carbon-free, so it is
    # systematic_verified (test_the_p72_3_anion_itself_is_not_labelled_a_pin below)
]


@pytest.mark.parametrize("smiles,expected", PIN_NAMES)
def test_pin_name(smiles, expected):
    r = Orthonym().name_tiered(smiles)
    assert r["name"] == expected, r
    assert r["tier"] == "pin_verified", r
    assert name_is_rt_exact(expected, smiles), expected


def test_the_sodium_salt_of_a_p72_3_anion_is_not_labelled_a_pin():
    # Branch review fixes: a wholly inorganic metal compound (a metal atom, no carbon)
    # is systematic_verified, is_pin False; the name is unchanged. PRESELECTED
    # NAMES (the Blue Book): "Preselected names are names for structures or
    # structural components chosen among two or more names for noncarboncontaining
    # (inorganic) parents to be used as the basis for preferred IUPAC names for
    # organic derivatives"; (:4667) "preferred IUPAC names have not yet been
    # determined for inorganic components". (The PIN examples of carry
    # carbon: 'sodium trimethylboranuide (PIN)',:41116.)
    smiles = "[Na+].[B-](F)(F)(F)F"
    r = Orthonym().name_tiered(smiles)
    assert r["name"] == "sodium tetrafluoroboranuide", r
    assert r["tier"] == "systematic_verified" and r["is_pin"] is False, r
    assert name_is_rt_exact(r["name"], smiles), r


def test_the_p72_3_anion_itself_is_not_labelled_a_pin():
    # Texts, labels and spelling (2026-09-29): a carbon-free compound has a
    # preselected name at most, never a PIN. PREFERRED IUPAC NAMES
    # (the Blue Book) adds the label 'PIN' to compounds "that also contain at
    # least one carbon atom in their structure";:2058 the PIN rules for compounds
    # "that do not contain carbon... will be discussed in a further publication";
    # 'F6I- hexafluoro-λ5-iodanuide (preselected name)' (:41112). Name unchanged.
    smiles = "F[P-](F)(F)(F)(F)F"
    r = Orthonym().name_tiered(smiles)
    assert r["name"] == "hexafluoro-λ5-phosphanuide", r
    assert r["tier"] == "systematic_verified" and r["is_pin"] is False, r
    assert name_is_rt_exact(r["name"], smiles), r
