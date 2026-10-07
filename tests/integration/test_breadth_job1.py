"""Breadth Job 1 (2026-09-28): PIN-tier promotion of the gated substituent producers,
with a vocabulary guard (classes M01, M03, M04, M05 of the breadth plan).

* M01 -- the PIN tier names a molecule it could not name before by re-running it with
  the ring-substituent composition producers that only the best-effort tier used
  (``rules.pin_vocabulary.promote_at_pin_tier``). The re-run happens only when the
  first run found no name, so a name the PIN tier ships today is unchanged, and a
  re-run name is kept only as a verified PIN form. The systematic generators
  (``terminal_ring_name``, ``terminal_fragment_name``) stay best-effort-only.
* M03 -- names that carry vocabulary no PIN contains are labelled below the PIN
  (``rules.pin_vocabulary.non_pin_vocabulary``), and the spelling defects of the
  producers the re-run reaches are fixed at the producer: 'acetamide' and the
  locant-free 'acetate' / 'acetic acid',, 'ethen-1-yl'
   (d)), the gem-disubstituted ring locant set (f)), 'ene' before
  prefixes (e)), the indicated hydrogen before the suffix (b)), the
  hyphen before a locant (a)) and the enclosing-mark order.
* M04 -- descriptors inside substituent prefixes now reach the PIN tier.
* M05 -- fallbacks that described a different molecule decline instead, so the right
  producer names it ('4-chloronaphthalen-1-yl', 'pyridine-4-carboxamide').

Every expected name was checked against a FRESH OPSIN call that does not go through the
engine (``tests.support.rt_assert``) by full InChIKey. Dev-set witnesses only (m1500 /
a dev split / dev2000), never a holdout split rows.
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.errors import is_failure_name
from tests.support.default_tier import declined_at_default
from tests.support.rt_assert import assert_full_rt

# Default tier (paper, Methods, "Tiers", L73: "The default configuration emits a name
# only when the pipeline can build the preferred IUPAC name (PIN); otherwise, it
# declines"; user decision 2026-09-30): a name the code records as not the PIN is
# declined there with NO_VERIFIED_PIN, and each such asserted name is checked as the
# strict path's name and at the best-effort tier (tests/support/default_tier.py).

pytestmark = [pytest.mark.integration, pytest.mark.opsin_gate]


def _pin(smiles):
    return Orthonym(style="pin").name_tiered(smiles)


def _be(smiles):
    return Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(smiles)


# Branch review fixes (2026-09-28): a name only the PIN tier's promotion re-run
# builds is labelled pin_unverified, is_pin False -- the re-run admits the
# best-effort composition producers, so the name is not certified as the PIN (the
# paper: "The label pin_unverified means a name in PIN form that only a breadth
# producer built"; the re-run also built right-molecule non-PINs, e.g. 'methyl
#...-3-(methoxycarbonyl)pent-3-enoate' for the diester,
# the Blue Book, and an oxolan-2-one parent beside a senior
# 1-benzopyran-4-one,:29628). The rows below are the re-run's; every
# other row is named by the first (strict) run and stays pin_verified.
RERUN_BUILT = frozenset({
    "NCc1csc(-c2cccs2)n1",
    # (PIN class program Task 2) '2,5-bis[(4-methoxyphenyl)methyl]-4-methylpyrimidine'
    # is built by the strict first run now (the decorated ring-yl fold at every tier):
    # (b) (the Blue Book) the pyrimidine is the senior ring, the two
    # benzene units are its substituents, '7,7-bis[(2-butoxyethoxy)methyl]-...-dioic
    # acid (PIN)' (:18899) for 'bis[...]' on a senior parent; pin_verified.
    "C=C1CC1C(C)=O",
    "O=C(O)C(=O)C[C@H]1C=C[C@H](O)CC1",
    "CC1=C(CC/C(C)=C/C=C/C(C)=C/CO)C(C)(C)CCC1",
    "C1C2CC3CC1CC(C2)(C3)COC(=O)CCl",
    "CC1=C(C=CC(=C1)OCC(=O)NC2=CC=C(C=C2)N3CCCCC3)C(C)C",
    "C=C1C[C@@H](O)CC(C)(C)[C@@H]1/C=C/C1=CC(=O)OC1",
    "CC1=CCC(CC1)C(=C)CCO",
    "COc1c(C)cnc(CS(=O)c2nc3ccc(O)cc3[nH]2)c1C",
    "COc1ccc(C2=C(Cc3ccc4c(c3)CC(O)C(C)(C)O4)[C@H](OC)OC2=O)cc1",
    "Cc1nnc(C(C)C)n1C1CC2CCC(C1)N2CC[C@H](NC(=O)C1CCC(F)(F)CC1)c1ccc(O)cc1",
    "C1CC1C2=NC3=CC=CC=C3C(=C2)C(=O)O[C@H](C4=CC=CC=C4)C(=O)NC5CC5",
    "CO[C@H]1[C@H](O)[C@@H](O)[C@H](OCc2cc3cc(O)cc(O)c3c(=O)o2)O[C@@H]1CO",
    "CC/C=C\\C[C@H](O)/C=C/[C@@H]1[C@@H](C/C=C\\CCCC(=O)[O-])[C@@H](O)C[C@H]1O",
    "O=C(O)CCCCCC[C@H]1C(=O)C[C@@H](O)[C@@H]1/C=C/[C@@H](O)CCCCCO",
    "CC1=CC(=C(C=C1)C)OCC(CN2CCN(CC2)C3=CC(=CC=C3)Cl)O",
})


# A re-run name whose best-effort name is another one (the re-run is the PIN tier's).
BEST_EFFORT_NAMES_IT_OTHERWISE = frozenset({
    "O=C(O)CCCCCC[C@H]1C(=O)C[C@@H](O)[C@@H]1/C=C/[C@@H](O)CCCCCO",
})


def _assert_pin(smiles, expected):
    if smiles in RERUN_BUILT:
        # declined at the default tier (NO_VERIFIED_PIN); the strict path's re-run
        # still builds the name, and the best-effort tier names the molecule below
        # the PIN label
        be = declined_at_default(
            smiles, expected,
            best_effort_same=smiles not in BEST_EFFORT_NAMES_IT_OTHERWISE)
        assert be["tier"] != "pin_verified" and not be["is_pin"], be
        return be
    row = _pin(smiles)
    assert row["name"] == expected, row
    assert row["tier"] == "pin_verified" and row["is_pin"], row
    assert row["opsin"] == "verified", row
    assert_full_rt(row["name"], smiles)
    return row


# ---------------------------------------------------------------------------
# M01: the PIN tier's re-run with the promoted composition producers
# ---------------------------------------------------------------------------

M01 = [
    # (the Blue Book) ring-on-chain, '(thiophen-2-yl)methyl' (:16442)
    ("NCc1csc(-c2cccs2)n1", "1-[2-(thiophen-2-yl)-1,3-thiazol-4-yl]methanamine"),  #
    #:16280 '2-[(4-bromophenyl)methyl]pyridine (PIN)': the decorated ring folds in
    ("COc1ccc(Cc2ncc(Cc3ccc(OC)cc3)c(C)n2)cc1",
     "2,5-bis[(4-methoxyphenyl)methyl]-4-methylpyrimidine"),
    ("C=C1CC1C(C)=O", "1-(2-methylidenecyclopropyl)ethan-1-one"),
    ("O=C(O)C(=O)C[C@H]1C=C[C@H](O)CC1",
     "3-[(1S,4R)-4-hydroxycyclohex-2-en-1-yl]-2-oxopropanoic acid"),
    ("CC1=C(CC/C(C)=C/C=C/C(C)=C/CO)C(C)(C)CCC1",
     "(2E,4E,6E)-3,7-dimethyl-9-(2,6,6-trimethylcyclohex-1-en-1-yl)nona-2,4,6-trien-1-ol"),
]


@pytest.mark.parametrize("smiles,expected", M01)
def test_m01_promoted_substituent_names_at_the_pin_tier(smiles, expected):
    _assert_pin(smiles, expected)


def test_m01_best_effort_label_stays_below_the_pin_when_only_the_rerun_builds_it():
    # Branch review fixes: the strict twin is the strict path without the
    # promotion re-run, so a name only the re-run builds is not confirmed by it and
    # the best-effort label stays pin_unverified (as before the re-run existed)
    row = _be("NCc1csc(-c2cccs2)n1")
    assert row["name"] == "1-[2-(thiophen-2-yl)-1,3-thiazol-4-yl]methanamine", row
    assert row["tier"] == "pin_unverified" and not row["is_pin"], row
    assert_full_rt(row["name"], "NCc1csc(-c2cccs2)n1")


def test_m01_a_non_pin_rerun_name_is_not_shipped_at_the_pin_tier():
    # the only name the re-run can build comes from a best-effort ring writer (the
    # terminal ring namer is never a PIN-tier producer); it carried
    # '1-oxacyclopropan-2-yl' and, since roadmap N5d, the Hantzsch-Widman 'oxiran-2-yl'
    #, the Blue Book). The PIN tier stays as it was; best-effort
    # still names it, labelled below the PIN: pin_unverified now that its spelling
    # carries no part the code records as not the PIN, never is_pin (the label of a
    # name a breadth producer built, ``metrics.provenance``). Tier labels -- paper semantics (user decision
    # 2026-09-30; Methods, "Tiers": "The label systematic_verified means a correct
    # systematic name that is not the PIN"): the PIN path built the name, a round
    # trip verified it, and it carries a part the code records as not the PIN, so
    # it is systematic_verified, is_pin False.
    smi = "CC/C=C\\CC(O)/C=C/C=C\\C=C\\C=C\\C1OC1CCCCCC(=O)O"
    assert is_failure_name(_pin(smi)["name"])
    be = _be(smi)
    assert "oxiran-2-yl" in be["name"], be
    assert be["tier"] == "pin_unverified" and not be["is_pin"], be
    assert be["opsin"] == "verified", be
    assert_full_rt(be["name"], smi)


def test_m01_systematic_generators_are_not_promoted():
    # terminal_fragment numbers this von Baeyer core with the free valence at 4
    # ('...-7-oxabicyclo[4.1.0]hept-3-en-4-yl'); (c) before (e)
    # (the Blue Book) gives it 3. The PIN tier declines rather than ship it.
    smi = ("CC(C)=CCC/C(C)=C/CC/C(C)=C/C[C@@]12O[C@@H]1C(=O)C(COC(=O)CC(C)(O)CC(=O)O)"
           "=CC2=O")
    assert is_failure_name(_pin(smi)["name"])
    be = _be(smi)
    assert "hept-3-en-3-yl" in be["name"], be
    assert_full_rt(be["name"], smi)


# ---------------------------------------------------------------------------
# M03: spellings of the producers the PIN tier now reaches
# ---------------------------------------------------------------------------

M03 = [
    # (the Blue Book) 'acetic acid (PIN) (substitution allowed)';
    # 'ethyl diazoacetate (PIN)' (:25925): locants of the alpha carbon omitted
    ("C1C2CC3CC1CC(C2)(C3)COC(=O)CCl", "(adamantan-1-yl)methyl chloroacetate"),
    ("CCCCCCCCCCOC(=O)CN1CCCCC1", "decyl (piperidin-1-yl)acetate"),
    ("C/C=C/C(=O)NCC(=O)O", "[(2E)-but-2-enamido]acetic acid"),
    ("CCCCCCCC/C=C\\CCCCCCCC(=O)NCC(=O)[O-]", "[(9Z)-octadec-9-enamido]acetate"),
    # (:32691-32693) 'acetamide (PIN)', locants kept
    ("CC1=C(C=CC(=C1)OCC(=O)NC2=CC=C(C=C2)N3CCCCC3)C(C)C",
     "2-[3-methyl-4-(propan-2-yl)phenoxy]-N-[4-(piperidin-1-yl)phenyl]acetamide"),
    ("C1=CC(=C(C=C1Cl)Cl)CNC(=O)CC#N",
     "2-cyano-N-[(2,4-dichlorophenyl)methyl]acetamide"),
    ("CC(=O)Nc1n[nH]c(-c2ccccc2)c1C#Cc1ccccc1",
     "N-[5-phenyl-4-(2-phenylethynyl)-1H-pyrazol-3-yl]acetamide"),
    # (d) (:2891), '2-chloroethen-1-yl (preferred prefix)' (:3003)
    ("O=C(O)/C=C/c1ccc(C(=O)O)o1", "5-[(1E)-2-carboxyethen-1-yl]furan-2-carboxylic acid"),
    # (f) (:3301): the gem-dimethyl carrier counts twice -> {2,2,4,6}
    ("C=C1C[C@@H](O)CC(C)(C)[C@@H]1/C=C/C1=CC(=O)OC1",
     "4-[(1E)-2-[(1S,4S)-4-hydroxy-2,2-dimethyl-6-methylidenecyclohexyl]ethen-1-yl]"
     "furan-2(5H)-one"),
    # (e) (:3288-3290): 'ene' before the prefixes
    ("CC1=CCC(CC1)C(=C)CCO", "3-(4-methylcyclohex-3-en-1-yl)but-3-en-1-ol"),
    # (b) before (c) (:3246): the indicated hydrogen first
    ("Oc1ccc2nc[nH]c2c1", "1H-1,3-benzimidazol-6-ol"),
    ("Cc1nc2ccc(C(=O)O)cc2[nH]1", "2-methyl-1H-1,3-benzimidazole-6-carboxylic acid"),
    ("Clc1ccc2nc[nH]c2c1", "6-chloro-1H-1,3-benzimidazole"),
    ("COc1c(C)cnc(CS(=O)c2nc3ccc(O)cc3[nH]2)c1C",
     "2-{[(4-methoxy-3,5-dimethylpyridin-2-yl)methyl]sulfinyl}-1H-1,3-benzimidazol-6-ol"),
    # (a) (:6938): a hyphen separates a locant from a word
    ("COc1ccc(C2=C(Cc3ccc4c(c3)CC(O)C(C)(C)O4)[C@H](OC)OC2=O)cc1",
     "(5R)-4-[(3-hydroxy-2,2-dimethyl-3,4-dihydro-2H-1-benzopyran-6-yl)methyl]-5-methoxy-"
     "3-(4-methoxyphenyl)furan-2(5H)-one"),
    # (:2869) the ring suffix keeps its locant once a ring prefix is cited,
    # and N- and ring prefixes form one alphanumerical series,:3477)
    ("Cc1nnc(C(C)C)n1C1CC2CCC(C1)N2CC[C@H](NC(=O)C1CCC(F)(F)CC1)c1ccc(O)cc1",
     "4,4-difluoro-N-[(1S)-1-(4-hydroxyphenyl)-3-{3-[3-methyl-5-(propan-2-yl)-4H-1,2,4-"
     "triazol-4-yl]-8-azabicyclo[3.2.1]octan-8-yl}propyl]cyclohexane-1-carboxamide"),
    ("O=C(NC1CCCC1)C1CCC(F)(F)CC1", "N-cyclopentyl-4,4-difluorocyclohexane-1-carboxamide"),
    ("O=C(Nc1ccccc1)C1CCC(O)CC1", "4-hydroxy-N-phenylcyclohexane-1-carboxamide"),
    # (:7232) a compound prefix is enclosed: '2-(cyclopropylamino)-2-oxo'
    ("C1CC1C2=NC3=CC=CC=C3C(=C2)C(=O)O[C@H](C4=CC=CC=C4)C(=O)NC5CC5",
     "(1R)-2-(cyclopropylamino)-2-oxo-1-phenylethyl 2-cyclopropylquinoline-4-carboxylate"),
    # with '*S*-(2-cyanoethyl)... (PIN)' (:31765): a complex S-alkyl
    # word is enclosed; 'ethanethioic S-acid (PIN)' (:6662)
    ("CC(=O)SCCCO", "S-(3-hydroxypropyl) ethanethioate"),
    ("CC(=O)SC(C)C", "S-(propan-2-yl) ethanethioate"),
    # (:7444-7446) nesting order {}
    ("CO[C@H]1[C@H](O)[C@@H](O)[C@H](OCc2cc3cc(O)cc(O)c3c(=O)o2)O[C@@H]1CO",
     "3-({[(2R,3R,4R,5S,6R)-3,4-dihydroxy-6-(hydroxymethyl)-5-methoxyoxan-2-yl]oxy}"
     "methyl)-6,8-dihydroxy-1H-2-benzopyran-1-one"),
]


@pytest.mark.parametrize("smiles,expected", M03)
def test_m03_pin_spelling(smiles, expected):
    _assert_pin(smiles, expected)


def test_m03_a_stereogenic_alpha_carbon_keeps_the_located_spelling():
    # the locant-free acetate is used only without an alpha stereocentre; the
    # systematic form cites the locant and is labelled below the PIN
    smi = "COC(=O)[C@H](Cl)Br"
    row = declined_at_default(smi, "methyl (2R)-2-bromo-2-chloroethanoate")
    assert not row["is_pin"], row


def test_m03_alcohol_to_alkyl_keeps_the_free_valence_locant_where_it_is_needed():
    from orthonym.decomposition.fragment_assembly import _alcohol_to_alkyl
    # (the Blue Book): only a saturated chain drops '-1-'
    assert _alcohol_to_alkyl("prop-2-en-1-ol") == "prop-2-en-1-yl"
    assert _alcohol_to_alkyl("bicyclo[2.2.1]heptan-1-ol") == "bicyclo[2.2.1]heptan-1-yl"
    assert _alcohol_to_alkyl("naphthalen-1-ol") == "naphthalen-1-yl"
    # a closing enclosing mark of a prefix is not a von Baeyer descriptor
    assert _alcohol_to_alkyl("2-[(1S)-2-methylcyclohexyl]ethan-1-ol") == (
        "2-[(1S)-2-methylcyclohexyl]ethyl")
    assert _alcohol_to_alkyl("propan-1-ol") == "propyl"


# ---------------------------------------------------------------------------
# M04: descriptors inside substituent prefixes
# ---------------------------------------------------------------------------

M04 = [
    ("CC/C=C\\C[C@H](O)/C=C/[C@@H]1[C@@H](C/C=C\\CCCC(=O)[O-])[C@@H](O)C[C@H]1O",
     "(5Z)-7-{(1R,2R,3R,5S)-3,5-dihydroxy-2-[(1E,3S,5Z)-3-hydroxyocta-1,5-dien-1-yl]"
     "cyclopentyl}hept-5-enoate"),
    ("O=C(O)CCCCCC[C@H]1C(=O)C[C@@H](O)[C@@H]1/C=C/[C@@H](O)CCCCCO",
     "7-{(1R,2R,3R)-2-[(1E,3S)-3,8-dihydroxyoct-1-en-1-yl]-3-hydroxy-5-oxocyclopentyl}"
     "heptanoic acid"),
]


@pytest.mark.parametrize("smiles,expected", M04)
def test_m04_substituent_stereo_at_the_pin_tier(smiles, expected):
    _assert_pin(smiles, expected)


# ---------------------------------------------------------------------------
# M05: fallbacks that named a different molecule decline
# ---------------------------------------------------------------------------

def test_m05_fused_ring_is_not_read_as_a_decorated_monocycle():
    # was '2-chloro-5-[(3-fluorobenzoyl)oxy]bicyclo[4.4.0]deca-...' at best-effort and
    # nothing at the PIN tier; a monocycle reading of the fused ring gave
    # '2,3-dibutyl-4-chlorophenyl': the fused system is named as a whole)
    _assert_pin("C1=CC=C2C(=C1)C(=CC=C2Cl)OC(=O)C3=CC(=CC=C3)F",
                "4-chloronaphthalen-1-yl 3-fluorobenzoate")


def test_m05_name_amide_declines_a_ring_it_cannot_spell():
    from orthonym.rules.amides import name_amide
    pattern = Chem.MolFromSmarts("[CX3](=O)[NX3]")
    for smi in ("NC(=O)c1ccncc1", "NC(=O)C1=CCCCC1", "NC(=O)c1ccc2ccccc2c1"):
        mol = Chem.MolFromSmiles(smi)
        # (the Blue Book): the ring's own name takes
        # 'carboxamide'; 'cyclohexanecarboxamide' named a different molecule
        assert name_amide(mol, mol.GetSubstructMatches(pattern)[0]) is None, smi
    mol = Chem.MolFromSmiles("NC(=O)C1CCCCC1")
    assert name_amide(mol, mol.GetSubstructMatches(pattern)[0]) == (
        "cyclohexanecarboxamide")
    _assert_pin("NC(=O)c1ccncc1", "pyridine-4-carboxamide")


def test_m05_ring_on_ring_prefix_keeps_its_decorations():
    # the two-ring compound prefix named only the ring skeletons and dropped the
    # chloro ('1-(1-phenylpiperazin-4-yl)', a different molecule the round trip
    # rejected); it now declines a decorated ring and the ring-by-ring producers
    # name the group, the Blue Book)
    _assert_pin("CC1=CC(=C(C=C1)C)OCC(CN2CCN(CC2)C3=CC(=CC=C3)Cl)O",
                "1-[4-(3-chlorophenyl)piperazin-1-yl]-3-(2,5-dimethylphenoxy)propan-2-ol")
    from orthonym.rules.ring_assemblies import (
        _find_inter_system_bonds,
        name_mixed_ring_prefix,
    )
    mol = Chem.MolFromSmiles("CCN1CCN(CC1)c1cccc(Cl)c1")
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    attach = 2  # the piperazine N that carries the ethyl
    assert name_mixed_ring_prefix(mol, rings, _find_inter_system_bonds(mol, rings),
                                  attach) is None


# ---------------------------------------------------------------------------
# The guard: a re-run name carrying a non-PIN form never ships as the PIN
# ---------------------------------------------------------------------------

GUARDED = [
    # '{1-[(adamantan-1-yl)methyl]ethyl}', the Blue Book 'not
    # 1-(hydroxymethyl)ethyl'); the PIN prefix is 1-(adamantan-1-yl)propan-2-yl
    "CC(CC12CC3CC(C1)CC(C3)C2)NCCO",
    # '1-benzylethyl' (benzyl is phenylmethyl; the prefix chain is propan-2-yl)
    "CC(Cc1ccccc1)NCCO",
    # '1-benzylpropyl', which no guard token matches: the chain composer ends its
    # chain at the free valence, and the longest chain through it is butan-2-yl, so
    # it declines in the re-run
    "CCC(Cc1ccccc1)NCCO",
    # a ring assembly '3-(...-1H-indol-3-yl)-1H-indole-...' is not the
    # '[3,3'-bi-1H-indole]' PIN
    # ('N#Cc1ccccc1Cc1ccccc1C#N' left this list: the strict path now builds its PIN
    # '2,2'-methylenedibenzonitrile', the Blue Book),
    # test_the_guarded_multiplicative_is_now_the_pin)
    "Oc1cc2[nH]cc(-c3c[nH]c4cc(O)c(O)cc34)c2cc1O",
    # ('CNC(=O)C1CCC(C)CC1' left this list: the strict path now builds its PIN
    # 'N,4-dimethylcyclohexane-1-carboxamide', test_the_guarded_amide_is_now_the_pin)
]


def test_the_guarded_multiplicative_is_now_the_pin():
    # (the Blue Book); '2,2'-methylenedibenzonitrile (PIN)' (:2648),
    # built by the general multiplicative detector (PIN class program, Task 10)
    _assert_pin("N#Cc1ccccc1Cc1ccccc1C#N", "2,2'-methylenedibenzonitrile")


def test_the_guarded_amide_is_now_the_pin():
    # the N- and C-methyl are one multiplied prefix and the suffix keeps its locant
    # (b), the Blue Book;,:2869; ':32879
    # N,4-dimethyl-N-(3-methylphenyl)benzamide (PIN)'); the strict path's first run
    # builds it, so no re-run is involved
    smi = "CNC(=O)C1CCC(C)CC1"
    row = _pin(smi)
    assert row["name"] == "N,4-dimethylcyclohexane-1-carboxamide", row
    assert row["tier"] == "pin_verified" and row["is_pin"], row
    assert_full_rt(row["name"], smi)


@pytest.mark.parametrize("smiles", GUARDED)
def test_guarded_rerun_name_is_not_labelled_pin(smiles):
    row = _pin(smiles)
    assert not row["is_pin"] and row["tier"] != "pin_verified", row
    be = _be(smiles)
    assert not is_failure_name(be["name"]), be
    assert not be["is_pin"] and be["tier"] != "pin_verified", be
    assert_full_rt(be["name"], smiles)


@pytest.mark.parametrize("smiles,expected", [
    ("CC(CC12CC3CC(C1)CC(C3)C2)NCCO", "2-{[1-(adamantan-1-yl)propan-2-yl]amino}ethan-1-ol"),
    ("CC(Cc1ccccc1)NCCO", "2-[(1-phenylpropan-2-yl)amino]ethan-1-ol"),
])
def test_m03_located_prefix_numbers_its_substituent_low(smiles, expected):
    # the free valence sits at the centre of the three-carbon prefix chain either
    # way (c)); the substituent then takes the lower locant, (f)
    # (the Blue Book) -- never '3-(adamantan-1-yl)propan-2-yl'
    be = _be(smiles)
    assert be["name"] == expected, be
    assert_full_rt(be["name"], smiles)
