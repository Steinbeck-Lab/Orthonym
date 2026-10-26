"""PIN class program, Task 21: ring chalcogens with exocyclic =O or =S named as heterones (LAMBDA_ONE).

 "Thioketone and thioaldehyde oxides" (the Blue Book): "When a group -SO- or -SO2- is part of a
ring system, oxygen atom(s) are expressed substitutively by the suffix '-one' added to the name of the
heterocycle in which the sulfur atoms are designated as lambda4 or lambda6 atoms (see. This method
generates preferred IUPAC names rather than those based on functional class nomenclature, in which the class
name 'oxide' follows the name of the heterocycle". The same suffix names the sultam, the sultone and the sultine:
 "Sultams, sultims, and intramolecular amides of sulfinic acids" (:33248), '1λ6,2-thiazinane-1,1-dione
(PIN)' (:33260, 'method (1) leads to preferred IUPAC names'); (:33282), '1λ4,2-thiazinan-1-one
(PIN)' against '1,2-thiazinane 1-oxide' (:33290); (:32150), '3-methyl-1,2λ6-oxathiane-2,2-dione
(PIN)' and '1,2λ4-oxathiolane-2-thione (PIN)' (:32162 ff.). The parent is the Hantzsch-Widman name of the saturated
heteromonocycle with the lambda of the chalcogen cited with its locant; 'one', 'dione' and 'thione' stand at that
locant ('dione' and 'thione' keep the final 'e' of the parent, 'one' elides it, (a),:7595).

The additive name ('thiolane 1,1-dioxide') is recorded as not the PIN (rule: the PIN tier declines it
where no heterone is built (fused rings, unsaturated rings) and the default tier keeps it.

Not built here (internal notes): the mancude and fused rings ('1H-1λ4-thiophen-1-one',
'5H-λ6-thianthrene-5,5-dione',:29444: the indicated hydrogen of the lambda parent), rings with a stereocentre,
and a ring beside a group of the class of the ketone or a senior one.
"""
import pytest

from tests.support.pin_tiers import assert_declined_at_default, assert_pin_at_both_tiers

pytestmark = pytest.mark.opsin_gate

PIN_ROWS = [
    ("S=S1CCCO1", "1,2λ4-oxathiolane-2-thione"),          #,:32162 ff.
    ("O=S1CCCCN1", "1λ4,2-thiazinan-1-one"),              #,:33290
    ("O=S1(=O)CCCCN1", "1λ6,2-thiazinane-1,1-dione"),     #,:33260
    ("O=S1(=O)CCCO1", "1,2λ6-oxathiolane-2,2-dione"),     # the sultone that held before
]

# Other members of the class, read back by assert_pin_at_both_tiers.
PIN_ROWS_MORE = [
    ("O=S1CCCC1", "1λ4-thiolan-1-one"),
    ("O=S1(=O)CCCC1", "1λ6-thiolane-1,1-dione"),
    ("O=S1(=O)CCC1", "1λ6-thietane-1,1-dione"),
    ("CC1CCCS(=O)(=O)O1", "6-methyl-1,2λ6-oxathiane-2,2-dione"),
    ("CC1CCS(=O)(=O)N1", "3-methyl-1λ6,2-thiazolidine-1,1-dione"),
    ("CN1CCCCS1(=O)=O", "2-methyl-1λ6,2-thiazinane-1,1-dione"),
    # "Retained names of heteromonocycles", Table 2.3 (the Blue Book): 'thiomorpholine (S instead of O)
    # (PIN)': the heterone is named on the retained parent (chalcogen 1, nitrogen 4), not on '1,4-thiazinane'
    ("O=S1(=O)CCNCC1", "1λ6-thiomorpholine-1,1-dione"),
    ("O=S1CCNCC1", "1λ4-thiomorpholin-1-one"),
    ("O=S1(=O)CCN(C)CC1", "4-methyl-1λ6-thiomorpholine-1,1-dione"),
    ("CN1CCS(=O)(=O)CC1", "4-methyl-1λ6-thiomorpholine-1,1-dione"),
    ("OCCN1CCS(=O)(=O)CC1", "4-(2-hydroxyethyl)-1λ6-thiomorpholine-1,1-dione"),
    ("O=S1(=O)CCN(c2ccccc2)CC1", "4-phenyl-1λ6-thiomorpholine-1,1-dione"),
    ("CC1CS(=O)(=O)CC(C)N1", "3,5-dimethyl-1λ6-thiomorpholine-1,1-dione"),
    # the prefix cited first ('d' of 'dimethylthiomorpholinyl',,:3477) takes the lower locant (g),:3307)
    ("CC1C(SCCN1C2CS(=O)(=O)CC2O)C", "3-(2,3-dimethylthiomorpholin-4-yl)-4-hydroxy-1λ6-thiolane-1,1-dione"),
    ("O=S1(=O)CCOCC1", "1,4λ6-oxathiane-4,4-dione"),
    ("O=S1(=O)CCSCC1", "1λ6,4-dithiane-1,1-dione"),
    ("O=S1CCSCC1", "1λ4,4-dithian-1-one"),
    ("OC1CCS(=O)(=O)C1", "3-hydroxy-1λ6-thiolane-1,1-dione"),
    ("O=S1(=O)CC(Cl)C(C)C1", "3-chloro-4-methyl-1λ6-thiolane-1,1-dione"),
    ("O=S1(=O)CCC(c2ccccc2)C1", "3-phenyl-1λ6-thiolane-1,1-dione"),
    ("O=S1(=O)CCCCN1c1ccc(Cl)cc1", "2-(4-chlorophenyl)-1λ6,2-thiazinane-1,1-dione"),
]

# Not built: the additive name is recorded as not the PIN, so the PIN tier declines it and best-effort keeps it.
DECLINED_ROWS = [
    "O=S1C=CC=C1",                       # thiophene 1-oxide: the BB heterone is '1H-1λ4-thiophen-1-one' (:29440)
    "O=S1(=O)C=Cc2ccccc12",              # 1-benzothiophene 1,1-dioxide
    "O=S1(=O)c2ccccc2-c2ccccc21",        # dibenzo[b,d]thiophene 5,5-dioxide
    "O=S1c2ccccc2-c2ccccc21",            # dibenzo[b,d]thiophene 5-oxide
    # selenium and tellurium: words the ring heterone for -SO- and -SO2- (:29436,:29444); the heterones of Se and
    # Te are named by analogy gives them for acyclic heterones), so the name is no PIN
    "O=[Se]1CCNCC1",                     # 1λ4-selenomorpholin-1-one at best-effort
    "O=[Se]1(=O)CCNCC1",
    "O=[Te]1CCNCC1",
]

# A group of the class of the ketone beside the heterone, or a senior one, or an acyl on the ring nitrogen (a ketone of
# the PIN): the heterone is not the suffix, and no PIN is built for the ring here.
NOT_TAKEN_ROWS = [
    "OC(=O)C1CCS(=O)(=O)C1",
    "O=S1(=O)CCCN1C(=O)C",
]


@pytest.mark.parametrize("smiles,pin", PIN_ROWS + PIN_ROWS_MORE)
def test_pin_at_both_tiers(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles", DECLINED_ROWS)
def test_the_functional_class_name_is_declined_at_the_pin_tier(smiles):
    assert_declined_at_default(smiles)


@pytest.mark.parametrize("smiles", NOT_TAKEN_ROWS)
def test_a_senior_or_ketone_group_leaves_the_ring_to_the_other_producers(smiles):
    assert_declined_at_default(smiles)


# The amine of a ring heterone is cited as the prefix 'amino' of the heterone, the Blue Book: the ketone class
# is senior to the amines;,:29417): its N is no suffix nitrogen, so the 'N-<acyl>' float of the fragment
# assembler names an N that belongs to no parent ('N-(acetylcarbamoyl)-3-amino-1λ6-thiolane-1,1-dione'). The fragment
# leaves it to the producers that cite the acylated amine as a prefix. Mutation check: ``amine_is_on_ring_heterone``
# returning False brings the float back and fails the first row (VERIFIED, 2026-10-26).
FLOAT_NOT_BUILT_ROWS = [
    ("CC(=O)NC(=O)NC1CCS(=O)(=O)C1", "3-[(acetylcarbamoyl)amino]-1,1-dioxothiolane"),
]


@pytest.mark.parametrize("smiles,expected", FLOAT_NOT_BUILT_ROWS)
def test_no_acyl_float_on_the_amino_prefix_of_a_heterone(smiles, expected):
    from tests.support.rt_assert import assert_full_rt, name_best_effort
    row = name_best_effort(smiles)
    assert row["name"] == expected, row
    assert "-amino-" not in row["name"] and not row["name"].startswith("N-"), row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("name,smiles,expected", [
    ("3-amino-1λ6-thiolane-1,1-dione", "NC1CCS(=O)(=O)C1", True),     # amine as prefix on a heterone
    ("1,1-dioxothiolan-3-amine", "NC1CCS(=O)(=O)C1", False),           # amine suffix: the N is the parent's
    ("cyclohexanamine", "NC1CCCCC1", False),                           # no heterone
    ("2-aminoethane-1-sulfonic acid", "NCCS(=O)(=O)O", False),         # no ring: decision A's demotion stands
    ("3-aminocyclohexan-1-one", "NC1CCCC(=O)C1", False),               # a ring ketone: decision A's demotion stands
])
def test_amine_is_on_ring_heterone(name, smiles, expected):
    from orthonym.decomposition.amine_suffix import amine_is_on_ring_heterone
    assert amine_is_on_ring_heterone(name, smiles) is expected
