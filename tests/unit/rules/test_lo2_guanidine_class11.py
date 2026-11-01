"""LO2: an acyclic guanidine is a class-11 parent, senior to the groups from hydrazides down.

 SENIORITY ORDER FOR CLASSES, Table 4.1 "General compound classes listed in decreasing
order of seniority" (the Blue Book): ":18184 11 Amides [in the order of the
corresponding acids...]", ":18187 14 Nitriles", ":18188 15 Aldehydes", ":18189 16 Ketones",
":18190 17 Hydroxy compounds", ":18192 19 Amines". under "
Guanidine and its derivatives" (:34240): "In the presence of a characteristic group having
seniority over guanidine (see item 11 in, the following prefixes are used" (:34266), so
guanidine itself stands at item 11, as urea does:33320, the PIN class
program's task 13). A guanidine therefore outranks a nitrile, aldehyde, ketone, hydroxy
compound, thiol and amine, and its PIN is 'N-(2-hydroxyethyl)guanidine', not
'2-[(diaminomethylidene)amino]ethan-1-ol'; an acid, ester, carboxamide or amidine stays
senior ('4-[(diaminomethylidene)amino]butanoic acid (PIN)',:34276; 'N-carbamimidoylurea
(PIN)',:34292 puts urea above it).

Before this lane the molecule's principal group was read as the amine/alcohol/ketone (guanidine
is not in ``SENIORITY_ORDER``: listing it there loses names on every path that reads the list,
measured in `internal notes`), so the default tier shipped the amine-,
ol- and one-parent names as PINs and the best-effort tier labelled the ring-amine name of a ring
amidine beside a guanidine ``pin_unverified``. ASSUMED (not decided by the book): the place of
guanidine against the amides of the 7a acids and urea, kept as urea's
(``handlers.urea.urea_is_senior_to``), so those names are unchanged.
"""
import pytest
from rdkit import Chem

from tests.support.pin_tiers import (
    assert_declined_at_default,
    assert_not_pin_labelled,
    assert_pin_at_both_tiers,
    name_breadth,
    name_default,
)
from tests.support.rt_assert import name_is_rt_exact

pytestmark = pytest.mark.opsin_gate

# Guanidine over a junior group: the guanidine is the parent at both tiers.
PIN_ROWS = [
    ("OCCN=C(N)N", "N-(2-hydroxyethyl)guanidine"),                          # 17 hydroxy
    ("NCCCCN=C(N)N", "N-(4-aminobutyl)guanidine"),                          # 19 amine (agmatine)
    ("CC(=O)CCN=C(N)N", "N-(3-oxobutyl)guanidine"),                         # 16 ketone
    ("O=CCCN=C(N)N", "N-(3-oxopropyl)guanidine"),                           # 15 aldehyde
    ("SCCN=C(N)N", "N-(2-sulfanylethyl)guanidine"),                         # 17 thiol
    ("NCC(O)CN=C(N)N", "N-(3-amino-2-hydroxypropyl)guanidine"),
    ("Nc1ccc(CCN=C(N)N)cc1", "N-[2-(4-aminophenyl)ethyl]guanidine"),
    ("Oc1ccc(CCN=C(N)N)cc1", "N-[2-(4-hydroxyphenyl)ethyl]guanidine"),
    ("N=C(N)Nc1ccc(N)cc1", "N-(4-aminophenyl)guanidine"),
    ("Nc1ncc(CCN=C(N)N)cc1", "N-[2-(6-aminopyridin-3-yl)ethyl]guanidine"),
    ("CC(=O)c1ccc(CCN=C(N)N)cc1", "N-[2-(4-acetylphenyl)ethyl]guanidine"),
    # the enrichment of the complete guanidine name once spelled the arm a second time
    # ('ethylbenzenylN-{2-[4-(hydroxymethyl)phenyl]ethyl}guanidine', unparseable)
    ("OCc1ccc(CCN=C(N)N)cc1", "N-{2-[4-(hydroxymethyl)phenyl]ethyl}guanidine"),
]

# The names the guanidine as a PREFIX gave: right molecule, not the PIN.
NOT_PIN_ROWS = [
    ("OCCN=C(N)N", "2-[(diaminomethylidene)amino]ethan-1-ol"),
    ("NCCCCN=C(N)N", "4-[(diaminomethylidene)amino]butan-1-amine"),
    ("CC(=O)CCN=C(N)N", "4-[(diaminomethylidene)amino]butan-2-one"),
    ("SCCN=C(N)N", "2-[(diaminomethylidene)amino]ethane-1-thiol"),
    ("N=C(N)Nc1ccc(N)cc1", "4-(carbamimidoylamino)aniline"),
]

# Classes senior to guanidine (or urea's own place) keep the prefix and stay PINs.
CONTROL_ROWS = [
    ("NC(=O)CN=C(N)N", "2-[(diaminomethylidene)amino]acetamide"),           # 11 carboxamide
    ("OC(=O)CN=C(N)N", "[(diaminomethylidene)amino]acetic acid"),           # 7a acid
    ("NC(N)=NCCCC(=O)O", "4-[(diaminomethylidene)amino]butanoic acid"),     #:34276 (PIN)
    ("N=C(N)NO", "N-hydroxyguanidine"),                                     # hydroxylamine handler
    ("N=C(N)NCc1ccccc1", "N-benzylguanidine"),
    ("COc1ccc(CCN=C(N)N)cc1", "N-[2-(4-methoxyphenyl)ethyl]guanidine"),     # an ether is no group
]


@pytest.mark.parametrize("smiles,pin", PIN_ROWS + CONTROL_ROWS)
def test_guanidine_class_pin(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles,non_pin", NOT_PIN_ROWS)
def test_guanidine_prefix_name_is_not_the_pin(smiles, non_pin):
    assert_not_pin_labelled(smiles, non_pin)


# The two rows of the item: a ring amidine -NH2 (an amine, class 19) beside an acyclic guanidine.
# Primary amines (the Blue Book; 'quinolin-4-amine (PIN)',:26199) makes the ring
# -NH2 an amine, junior to the guanidine; 'N-[2-(2-aminoquinolin-6-yl)ethyl]guanidine' is the
# guanidine-parent name, verified by OPSIN but not certified (the ring prefix is a breadth
# producer's), so the best-effort tier ships it as pin_unverified and the default tier declines.
TARGET_ROWS = [
    ("Nc1ccc2cc(CCN=C(N)N)ccc2n1",
     "N-[2-(2-aminoquinolin-6-yl)ethyl]guanidine",
     "6-[2-(diaminomethylideneamino)ethyl]quinolin-2-amine"),
]


@pytest.mark.parametrize("smiles,parent_name,ring_name", TARGET_ROWS)
def test_ring_amidine_amine_beside_guanidine_names_the_guanidine(smiles, parent_name, ring_name):
    assert name_is_rt_exact(parent_name, smiles), parent_name
    b = name_breadth(smiles)
    assert (b.get("name"), b.get("tier")) == (parent_name, "pin_unverified"), b
    assert b.get("producer") == "inner:guanidine", b
    assert_not_pin_labelled(smiles, ring_name)
    assert_declined_at_default(smiles)


def test_a_dihydroquinolinyl_arm_is_built_by_the_fusion_builder_and_the_guanidine_name_is_the_pin():
    # the partly hydrogenated ring prefix is now built by the PIN tier's fused '-yl' builder (DT4 F2b, NPL 2026-11-01;
    #,, so the guanidine-parent name is certified at both tiers
    smiles = "NC1=Nc2cc(CCN=C(N)N)ccc2CC1"
    pin = "N-[2-(2-amino-3,4-dihydroquinolin-7-yl)ethyl]guanidine"
    assert name_is_rt_exact(pin, smiles), pin
    assert_pin_at_both_tiers(smiles, pin)
    assert_not_pin_labelled(smiles, "7-[2-(diaminomethylideneamino)ethyl]-3,4-dihydroquinolin-2-amine")


def test_ring_name_of_the_guanidine_arm_the_builder_cannot_spell_is_labelled_below_the_pin():
    """The guanidine builder cannot spell the 2-oxo-tetrahydroquinolinyl arm, so the ring parent
    name ships; it cites the guanidine as a prefix beside the junior ketone suffix, so it is
    labelled systematic_verified, not pin_unverified (the inner dispatch label of a name that is
    not the guanidine parent, as for urea)."""
    smiles = "O=C1CCc2ccc(CCN=C(N)N)cc2N1"
    # the prefix is '(diaminomethylidene)amino (preferred prefix)', (the Blue Book)
    ring_name = "7-{2-[(diaminomethylidene)amino]ethyl}-1,2,3,4-tetrahydroquinolin-2-one"
    b = name_breadth(smiles)
    assert (b.get("name"), b.get("tier")) == (ring_name, "systematic_verified"), b
    assert name_is_rt_exact(ring_name, smiles)
    d = name_default(smiles)
    assert d.get("tier") != "pin_verified", d


def test_nitrile_junior_to_guanidine_default_tier_declines():
    """Parity with urea ('N-(2-cyanoethyl)-N'-methylurea', test_pin_class_t13_carbonic.py):
    the PIN-tier substituent namer does not certify the 2-cyanoethyl arm, so the guanidine name
    ships at the best-effort tier only and the default tier declines the nitrile-parent name."""
    smiles = "N#CCCN=C(N)N"
    assert_not_pin_labelled(smiles, "3-[(diaminomethylidene)amino]propanenitrile")
    b = name_breadth(smiles)
    assert (b.get("name"), b.get("tier")) == ("N-(2-cyanoethyl)guanidine", "pin_unverified"), b
    assert name_default(smiles).get("tier") != "pin_verified"


# The independent spelling check (validation/spelling/checks_parent.py) sees a guanidine too.
@pytest.mark.parametrize("smiles,name,flagged", [
    ("OCCN=C(N)N", "2-[(diaminomethylidene)amino]ethan-1-ol", "alcohol"),
    ("N#CCCN=C(N)N", "3-[(diaminomethylidene)amino]propanenitrile", "nitrile"),
    ("NC(=N)NCCC1CCCC1=O", "2-[2-(carbamimidoylamino)ethyl]cyclopentan-1-one", "ketone"),
    ("Nc1ccc2cc(CCN=C(N)N)ccc2n1", "6-[2-(diaminomethylideneamino)ethyl]quinolin-2-amine", "amine"),
    ("NC(=O)CN=C(N)N", "2-[(diaminomethylidene)amino]acetamide", None),
    ("OC(=O)CN=C(N)N", "[(diaminomethylidene)amino]acetic acid", None),
    ("NC(N)=NS(=O)(=O)c1ccc(N)cc1", "4-amino-N-(diaminomethylidene)benzene-1-sulfonamide", None),
    ("NCCCCN=C(N)N", "N-(4-aminobutyl)guanidine", None),
])
def test_p41_spelling_check_reads_guanidine(smiles, name, flagged):
    from orthonym.validation.spelling.checks_parent import principal_class_check
    res = principal_class_check(Chem.MolFromSmiles(smiles), name)
    if flagged is None:
        assert res is None, res
    else:
        assert res is not None and res.rule == "P-41", res
        assert f"expresses '{flagged}'" in res.detail and "'guanidine'" in res.detail, res


# ---------------------------------------------------------------------------------------------
# Fix a performance pass: an acyclic imide is an amide, a hydrazide arm is not spelled by the builder.
#
# IMIDES, (the Blue Book): "Acyclic imides are N-acyl derivatives of primary
# amides and are named as such (see. Cyclic imides are preferably named as heterocyclic
# pseudoketones." So an acyclic -CO-NH-CO- (an N-acylurea too, 'N-carbamoyl-2-phenylacetamide
# (PIN)',:33364) is an amide, class 11 of (:18184), and stays senior to guanidine
# (ASSUMED as the other 7a amides are); only the cyclic imide ranks below it (:18186 "13 Imides").
# The guanidine builder named such a molecule 'N-[3-(carbamoylamino)-3-oxopropyl]guanidine'
# pin_verified at both tiers (the perceived principal group 'imide' was read as class 13).
#
# HYDRAZIDES, (:34014): "When the -CO-NHNH2 group is at the end of a chain the use
# of the prefixes 'hydrazinyl' and 'oxo' generates preferred IUPAC names" ('3-hydrazinyl-3-oxo
# propanoic acid (PIN)',:34018); (:34020) gives the substituted hydrazide arms their own
# prefixes ('acetohydrazido', 'acylhydrazinyl'). The substituent namer the guanidine builder cites
# its arms through spells -CO-NH-NH2 '(aminoamino)...oxo', so a hydrazide principal group, junior
# to guanidine (class 12), is not named on the guanidine: the default tier declines and the name
# the hydrazide handler builds is labelled below the PIN, as urea's is.
# ---------------------------------------------------------------------------------------------
# smiles, the name the guanidine builder gave in a performance pass (2a56b4ed8; pin_verified at both tiers, a non-PIN),
# the name and tier the best-effort tier ships now
ACYCLIC_IMIDE_ROWS = [
    ("N=C(N)NCCC(=O)NC(N)=O", "N-[3-(carbamoylamino)-3-oxopropyl]guanidine",
     "3-(carbamimidoylamino)-1-(carbamoylamino)-1-oxopropane"),
    ("N=C(N)NCCC(=O)NC(=O)C", "N-[3-(carbamimidoylamino)propanoyl]acetamide",
     "1-acetamido-3-(carbamimidoylamino)-1-oxopropane"),
    ("N=C(N)NCCC(=O)N(C)C(=O)C", "N-[3-(carbamimidoylamino)propanoyl]-N-methylacetamide",
     "3-(carbamimidoylamino)-1-(N-methylacetamido)-1-oxopropane"),
    ("N=C(N)NCCC(=O)NC(=O)NC", "N-{3-[(methylcarbamoyl)amino]-3-oxopropyl}guanidine",
     "3-(carbamimidoylamino)-1-[(methylcarbamoyl)amino]-1-oxopropane"),
]

HYDRAZIDE_ROWS = [
    # 'acetohydrazide', not 'ethanehydrazide': (the Blue Book-33944); still below the PIN
    # (the guanidine, class 11, outranks the hydrazide, class 12), read back by name_is_rt_exact below.
    ("N=C(N)NCC(=O)NN", "N-[2-(aminoamino)-2-oxoethyl]guanidine",
     "2-(carbamimidoylamino)acetohydrazide"),
    ("N=C(N)NCCC(=O)NN", "N-[3-(aminoamino)-3-oxopropyl]guanidine",
     "3-(carbamimidoylamino)propanehydrazide"),
    ("N=C(N)NCCCC(=O)NNC", "N-{4-[(methylamino)amino]-4-oxobutyl}guanidine",
     "4-(carbamimidoylamino)-1-[(methylamino)amino]-1-oxobutane"),
    # the best-effort name was '1-(acetamidoamino)-3-(carbamimidoylamino)-1-oxopropane' (a replacement-chain
    # name); the diacylhydrazine now has its hydrazide name, 'N'-acetyl-...propanehydrazide',
    # the Blue Book, "Acyl, diacyl, and triacyl derivatives of hydrazides are named by substituting the
    # senior hydrazide"). change-asserted-value: independent check = the read-back assert below
    # (name_is_rt_exact); mutation = tests/unit/rules/test_diacylhydrazides.py. It stays below the PIN and
    # the default tier still declines: the guanidine (class 11) outranks the hydrazide (class 12,:18184).
    ("N=C(N)NCCC(=O)NNC(C)=O", "N-[3-(acetamidoamino)-3-oxopropyl]guanidine",
     "N'-acetyl-3-(carbamimidoylamino)propanehydrazide"),
    ("N=C(N)NCCCC(=O)NN", "N-[4-(aminoamino)-4-oxobutyl]guanidine",
     "4-(carbamimidoylamino)butanehydrazide"),
]


@pytest.mark.parametrize("smiles,wrong,breadth_name", ACYCLIC_IMIDE_ROWS + HYDRAZIDE_ROWS)
def test_acyclic_imide_and_hydrazide_are_not_named_on_the_guanidine(smiles, wrong, breadth_name):
    """The guanidine-parent name was a non-PIN here (the acyclic imide is an amide, the hydrazide
    arm is misspelled): it is not shipped as a PIN at either tier, the default tier declines and
    the best-effort tier ships the exact name below the PIN, read back to the input."""
    assert_not_pin_labelled(smiles, wrong)
    d = name_default(smiles)
    assert d.get("tier") == "abstain", d
    b = name_breadth(smiles)
    assert (b.get("name"), b.get("tier")) == (breadth_name, "systematic_verified"), b
    assert name_is_rt_exact(breadth_name, smiles), breadth_name


def test_acylurea_aryl_row_keeps_its_amide_pin():
    """An N-carbamoyl benzamide beside a guanidine: the amide is the parent, the guanidine the
    'carbamimidoylamino' prefix (:34290 'N-carbamimidoylformamide (PIN)'). START named it so at
    both tiers; the first version of this class rank read the acyl urea as class 13 and the default
    tier declined it."""
    assert_pin_at_both_tiers("N=C(N)Nc1ccc(cc1)C(=O)NC(N)=O",
                             "4-(carbamimidoylamino)-N-carbamoylbenzamide")


# A CYCLIC imide is a pseudoketone,:33580), class 16 or 13 of Table 4.1: junior to guanidine,
# whose builder spells these ring arms ('2,5-dioxopyrrolidin-1-yl', 'pyrrolidine-2,5-dione (PIN)').
@pytest.mark.parametrize("smiles,pin", [
    ("N=C(N)NCCN1C(=O)CCC1=O", "N-[2-(2,5-dioxopyrrolidin-1-yl)ethyl]guanidine"),
    ("N=C(N)NCCN1C(=O)CCCC1=O", "N-[2-(2,6-dioxopiperidin-1-yl)ethyl]guanidine"),
    ("N=C(N)NCCN1C(=O)c2ccccc2C1=O",
     "N-[2-(1,3-dioxo-1,3-dihydro-2H-isoindol-2-yl)ethyl]guanidine"),
])
def test_cyclic_imide_stays_junior_to_the_guanidine(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


def test_acyclic_imide_present_reads_the_imide_nitrogen():
    """``handlers.urea.acyclic_imide_present``, the one home of the reading that the urea
    and the guanidine rank both use: an imide nitrogen outside every ring makes the group an amide."""
    from orthonym.assembly.handlers.urea import acyclic_imide_present
    patt = Chem.MolFromSmarts("[CX3](=O)[NX3][CX3](=O)")
    for smiles, acyclic in [("CC(=O)NC(=O)C", True), ("CC(=O)N(C)C(=O)C", True),
                            ("NC(=O)NC(=O)C", True), ("O=C1CCC(=O)N1", False),
                            ("O=C1C=CC(=O)N1C", False)]:
        mol = Chem.MolFromSmiles(smiles)
        assert acyclic_imide_present(mol, mol.GetSubstructMatches(patt)) is acyclic, smiles
    assert acyclic_imide_present(Chem.MolFromSmiles("CC"), None) is False


def test_urea_beside_an_acyclic_imide_is_not_labelled_below_the_pin():
    """The same reading in the urea rank: ``urea_parent_expected`` took the acyclic imide of
    'N-acetyl-4-[(methylcarbamoyl)amino]benzamide' for a class 13 group (junior to urea) and
    labelled the amide name below the PIN; the amide is senior to urea ('N-carbamoyl-2-phenylacetamide
    (PIN)',:33364), so that name keeps the label the producer earned. A cyclic imide stays junior."""
    b = name_breadth("CNC(=O)Nc1ccc(cc1)C(=O)NC(=O)C")
    assert (b.get("name"), b.get("tier")) == (
        "N-acetyl-4-[(methylcarbamoyl)amino]benzamide", "pin_unverified"), b
    c = name_breadth("NC(=O)NCCN1C(=O)CCC1=O")
    assert (c.get("name"), c.get("tier")) == (
        "1-[2-(carbamoylamino)ethyl]pyrrolidine-2,5-dione", "systematic_verified"), c
