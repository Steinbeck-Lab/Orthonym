"""Two rings joined through a ring-YLIDENE carbon: the senior RING is the parent.

 "Unsymmetrical substituent groups having terminal atoms with different types of
free valencies" (the Blue Book) gives the example verbatim:

    (cyclohexylidenemethyl)benzene (PIN)
    [not (phenylmethylidene)cyclohexane; benzene is senior to cyclohexane (see ] (:6390-6392)

The linker (=CH-) never overrules the ring seniority of /: the parent is the
senior ring system and the other ring is cited as a ring-ylidene prefix on the carrier carbon.
``ring_selection.select_principal_ring_system`` used to force the ring on the double-bond side
(W2E-D2) because that prefix had no producer at the default tier; it now has one
(``rules.ring_ylidene_carrier``), and the linker decides only between two rings that tie on.

The expected names are the Blue Book's own worked example for the first row, and the
application of the same rules to the others heterocycle > carbocycle, more
multiple bonds, more skeletal atoms; ylidene + alkyl-yl compound prefix;
enclosing marks). Each expected name is also read back by OPSIN to the input structure. Only the first
row is a Blue Book worked example; every other expected name is that rule applied (reasoning-backed,
not confirmed by a worked example of the book).

The ring-ylidene prefix of an unsaturated monocycle ('cyclohex-2-en-1-ylidene', '2H-pyrrol-2-ylidene')
is built by ``rules.ring_ylidene_prefix``, "mancude rings... method (2)",
lowest locants: free valence, then 'ene'; the indicated-hydrogen form is the Blue Book's own
'4H-thiopyran-4-ylidene (PIN)',, the Blue Book).
"""
import pytest
from rdkit import Chem

from tests.support.pin_tiers import (
    assert_pin_at_both_tiers,
    name_breadth,
    name_default,
)
from tests.support.rt_assert import name_is_rt_exact

pytestmark = pytest.mark.opsin_gate

# Ring B (single-bonded to the carrier) is senior to ring A (double-bonded): B is the parent.
SENIOR_SINGLE_BONDED_RING = [
    # (:6390): benzene > cyclohexane multiple bonds)
    ("C(=C1CCCCC1)c1ccccc1", "(cyclohexylidenemethyl)benzene"),
    ("C(=C1CCCC1)c1ccccc1", "(cyclopentylidenemethyl)benzene"),
    ("C(=C1CC1)c1ccccc1", "(cyclopropylidenemethyl)benzene"),
    # heteroatom ring > carbocycle
    ("C(=C1CCCCC1)c1ccccn1", "2-(cyclohexylidenemethyl)pyridine"),
    ("C(=C1CCCCC1)c1ccncc1", "4-(cyclohexylidenemethyl)pyridine"),
    ("C(=C1CCCCC1)c1cccs1", "2-(cyclohexylidenemethyl)thiophene"),
    # more rings; the fused parent names its own substituent locants
    ("C(=C1CCCCC1)c1ccc2ccccc2c1", "2-(cyclohexylidenemethyl)naphthalene"),
    # a fused heterocycle parent: the compound prefix keeps its enclosing marks
    ("C(=C1CCCCC1)c1ccc2[nH]ccc2c1", "5-(cyclohexylidenemethyl)-1H-indole"),
    ("C(=C1CCCCC1)c1ccc2ncccc2c1", "6-(cyclohexylidenemethyl)quinoline"),
    # more ring multiple bonds: cyclohexene > cyclohexane
    ("C(=C1CCCCC1)C1=CCCCC1", "1-(cyclohexylidenemethyl)cyclohex-1-ene"),
    # more skeletal atoms: cyclohexane > cyclopentane
    ("C(=C1CCCC1)C1CCCCC1", "(cyclopentylidenemethyl)cyclohexane"),
    # other substituents on the parent ring and on the ylidene ring
    ("Cc1ccc(C=C2CCCCC2)cc1", "1-(cyclohexylidenemethyl)-4-methylbenzene"),
    ("C(=C1CCCCC1)c1ccccc1Cl", "1-chloro-2-(cyclohexylidenemethyl)benzene"),
    ("COc1ccc(C=C2CCC(C)CC2)cc1", "1-methoxy-4-[(4-methylcyclohexylidene)methyl]benzene"),
    ("C(=C1CCC(Cl)CC1)c1ccccc1", "[(4-chlorocyclohexylidene)methyl]benzene"),
    # a principal characteristic group on the parent ring
    ("OC(=O)c1ccc(C=C2CCCCC2)cc1", "4-(cyclohexylidenemethyl)benzoic acid"),
    ("Oc1ccccc1C=C1CCCCC1", "2-(cyclohexylidenemethyl)phenol"),
    # an unsaturated ring-ylidene ring: free valence 1, then the lowest 'ene' locants
    ("C(=C1C=CCCC1)c1ccccc1", "[(cyclohex-2-en-1-ylidene)methyl]benzene"),
    ("C(=C1CC=CCC1)c1ccccc1", "[(cyclohex-3-en-1-ylidene)methyl]benzene"),
    ("C(=C1C=CC=CC1)c1ccccc1", "[(cyclohexa-2,4-dien-1-ylidene)methyl]benzene"),
    ("C(=C1C=CCC1)c1ccccc1", "[(cyclopent-2-en-1-ylidene)methyl]benzene"),
    ("C(=C1C=CC=C1)c1ccccc1", "[(cyclopenta-2,4-dien-1-ylidene)methyl]benzene"),
    ("C(=C1C=CCCC1)c1ccccn1", "2-[(cyclohex-2-en-1-ylidene)methyl]pyridine"),
    ("C(=C1C=CCCC1)c1ccc2ccccc2c1", "2-[(cyclohex-2-en-1-ylidene)methyl]naphthalene"),
    ("CC(=C1C=CCCC1)c1ccccc1", "[1-(cyclohex-2-en-1-ylidene)ethyl]benzene"),
    # one heteroatom in the ylidene ring: the ylidene atom is the indicated-hydrogen position;
    # the other ring is senior, nitrogen) so it is the parent
    ("C(=C1C=CC=CO1)c1ccccn1", "2-[(2H-pyran-2-ylidene)methyl]pyridine"),
    ("C(=C1C=COC=C1)c1ccncc1", "4-[(4H-pyran-4-ylidene)methyl]pyridine"),
    ("C(=C1C=CSC=C1)c1ccncc1", "4-[(4H-thiopyran-4-ylidene)methyl]pyridine"),
    #: lowest locants for indicated hydrogen, 1H-pyrrole is senior to 2H-pyrrole
    ("C1=CC(=Cc2ccc[nH]2)N=C1", "2-[(2H-pyrrol-2-ylidene)methyl]-1H-pyrrole"),
]

# A chain parent with the unsaturated ring-ylidene prefix, acetic acid keeps its retained name).
CHAIN_PARENT = [
    ("OC(=O)C=C1C=CCCC1", "(cyclohex-2-en-1-ylidene)acetic acid"),
    ("OC(=O)CC(C)=C1C=CCCC1", "3-(cyclohex-2-en-1-ylidene)butanoic acid"),
]

# The carrier is a chain: the ring-ylidene sits on carbon n of the alkyl-yl group (free valence = 1).
CHAIN_CARRIER = [
    ("CC(=C1CCCCC1)c1ccccc1", "(1-cyclohexylideneethyl)benzene"),
    ("CCC(=C1CCCCC1)c1ccccc1", "(1-cyclohexylidenepropyl)benzene"),
    ("C(=C1CCCCC1)Cc1ccccc1", "(2-cyclohexylideneethyl)benzene"),
    ("C(=C1CCCCC1)CCc1ccccc1", "(3-cyclohexylidenepropyl)benzene"),
    ("CC(=C1CCCCC1)c1ccccn1", "2-(1-cyclohexylideneethyl)pyridine"),
]


@pytest.mark.parametrize("smiles,pin", SENIOR_SINGLE_BONDED_RING + CHAIN_CARRIER + CHAIN_PARENT)
def test_senior_ring_is_the_parent(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


def test_blue_book_example_is_the_pin_and_the_old_name_is_not():
    """ (:6390,:6392): the PIN, and the form the Blue Book marks 'not'."""
    smiles = "C(=C1CCCCC1)c1ccccc1"
    assert name_is_rt_exact("(cyclohexylidenemethyl)benzene", smiles)
    # the old name describes the same structure, so only the label can tell them apart
    assert name_is_rt_exact("(phenylmethylidene)cyclohexane", smiles)
    for res in (name_default(smiles), name_breadth(smiles)):
        assert res["name"] == "(cyclohexylidenemethyl)benzene"
        assert "benzylidenecyclohexane" not in res["name"]


# Ring A (double-bonded) is the senior ring: it stays the parent, as before.
SENIOR_DOUBLE_BONDED_RING = [
    ("C(=C1CCCCCCC1)c1ccccc1", "benzylidenecyclooctane"),             # 8 ring atoms > 6
    ("C(=C1CCOCC1)c1ccccc1", "4-benzylideneoxane"),                     # heterocycle
    ("C(=C1CCNCC1)c1ccccc1", "4-benzylidenepiperidine"),
    ("C(=C1CC2CCC1C2)c1ccccc1", "2-benzylidenebicyclo[2.2.1]heptane"),  # more rings
    ("O=C1CCC(=Cc2ccccc2)CC1", "4-benzylidenecyclohexan-1-one"),        #: the group
]


@pytest.mark.parametrize("smiles,pin", SENIOR_DOUBLE_BONDED_RING)
def test_senior_double_bonded_ring_stays_the_parent(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


def test_tie_on_p44_2_with_the_same_group_on_both_sides_is_decided_by_alphanumerical_order():
    """Two rings that tie on (cyclohexane, cyclohexane) and carry the same group: "Criteria
    related to alphanumerical order of names" (the Blue Book), "The preferred IUPAC name is the name
    that is earlier in alphanumerical order": '-ylidenemethyl' is earlier than '-ylmethylidene', as in
     example (2) (:6398). The single-bonded ring is the parent.
    change-asserted-value: above; read back by assert_pin_at_both_tiers (OPSIN, fresh call);
    mutation = tests/unit/rules/test_pin_class_t14_parent_seniority.py (RING_ROWS fails on the old source)."""
    assert_pin_at_both_tiers("C(=C1CCCCC1)C1CCCCC1", "(cyclohexylidenemethyl)cyclohexane")


# Shapes with no strict ring-ylidene prefix decline at the default tier: the junior ring used to be the
# parent ('3-benzylidene-1-methylcyclohex-1-ene', a false PIN). The best-effort tier names the senior parent
# and does not label it a PIN.
@pytest.mark.parametrize("smiles,best_effort", [
    # a decorated unsaturated carbocycle: the decorated-ring producers have no strict ylidene form
    ("C(=C1C=C(C)CCC1)c1ccccc1", "[(3-methylcyclohex-2-en-1-ylidene)methyl]benzene"),
    ("C(=C1C=CCC(C)C1)c1ccccc1", "[(5-methylcyclohex-2-en-1-ylidene)methyl]benzene"),
    # the ylidene atom of a hydro position needs the added-hydrogen form, not built
    ("C(=C1C=CC=NC1)c1ccccn1", "2-[(2,3-dihydropyridin-3-ylidene)methyl]pyridine"),
])
def test_unbuilt_ylidene_ring_declines_at_default(smiles, best_effort):
    d = name_default(smiles)
    assert d.get("tier") != "pin_verified", d
    b = name_breadth(smiles)
    assert b["name"] == best_effort and b["tier"] != "pin_verified", b
    assert name_is_rt_exact(best_effort, smiles)


@pytest.mark.parametrize("smiles", [
    # the second ring is a ring assembly (biphenyl): its parent has no default-tier producer for a
    # substituent that is a ring, in the saturated case either ('1-cyclohexyl-4-phenylbenzene')
    "C(=C1CCCCC1)c1ccc(cc1)c1ccccc1",
    "C(C1CCCCC1)c1ccc(cc1)c1ccccc1",
    # two benzene rings tie and (alphanumerical order) decides; neither prefix is built at the
    # default tier, so neither name is claimed ('1-benzylidene-2-phenylcyclopentane' was a false PIN)
    "c1ccccc1C=C1CCCC1c1ccccc1",
])
def test_unbuilt_parent_comparison_does_not_claim_a_pin(smiles):
    d = name_default(smiles)
    assert d.get("tier") != "pin_verified", d
    assert "benzylidene" not in (d.get("name") or ""), d


@pytest.mark.parametrize("smiles", [
    "C1=C/C(=C/c2ccc[nH]2)N=C1",
    r"C1=C/C(=C\c2ccc[nH]2)N=C1",
    r"CC1=N/C(=C\C2=CC=C(N2)C)/C=C1",
])
def test_stereo_flagged_ylidene_bond_between_two_prefix_parts_is_not_a_pin(smiles):
    """The Blue Book gives no placement for the descriptor of a double bond that joins the two
    components of one prefix ('(5-methyl-2H-pyrrol-2-ylidene)methyl'): (the Blue Book)
    "When they relate to substituent groups, they are cited at the front of the corresponding prefix"
    does not say which one. The default tier declines; the best-effort name is not a PIN."""
    d = name_default(smiles)
    assert d.get("tier") != "pin_verified", d
    b = name_breadth(smiles)
    assert b.get("name") and b.get("tier") != "pin_verified" and name_is_rt_exact(b["name"], smiles), b


def test_ring_ylidene_bond_with_a_stereo_flag_is_not_named_here():
    """The E/Z descriptor of the ylidene bond is not cited by the carrier producer, so a bond
    that carries one is left to the other producers (never a name that drops it)."""
    smiles = r"c1ccccc1/C=C1\CCCCC1C"
    for res in (name_default(smiles), name_breadth(smiles)):
        if res.get("tier") == "pin_verified":
            assert name_is_rt_exact(res["name"], smiles), res


@pytest.mark.parametrize("token,complex_", [
    ("cyclohexylidenemethyl", True),
    ("1-cyclohexylideneethyl", True),
    ("cyclohexylidenemethoxy", True),
    ("cyclohexylidene", False),
    ("cyclohexylmethyl", True),
])
def test_ring_ylidene_alkyl_prefix_is_a_compound_prefix(token, complex_):
    """ (the Blue Book): a compound prefix takes enclosing marks."""
    from orthonym.assembly.naming_utils import is_complex_substituent
    assert is_complex_substituent(token) is complex_


def test_two_drawings_of_the_guanidine_prefix_are_two_prefixes():
    """ "Guanidine and its derivatives", (the Blue Book-34272;
    the prefix tables of the appendix agree,:55473 and:55818): the prefix for H2N-C(=NH)-NH- is
    'carbamimidoylamino (preferred prefix)', the prefix for (H2N)2C=N- is
    '(diaminomethylidene)amino (preferred prefix)', and the Blue Book's own (PIN) example
    '4-[(diaminomethylidene)amino]butanoic acid' (:34282) is drawn (H2N)2C=N-CH2-CH2-CH2-COOH.
    Arginine, (:54193), is drawn H2N-C(=NH)-NH-[CH2]3-CH(NH2)-COOH and named
    '2-amino-5-(carbamimidoylamino)pentanoic acid'. The name follows the DRAWN tautomer."""
    amidino_drawing = "NC(=N)NCCCC(=O)O"      # H2N-C(=NH)-NH-R
    ylidene_drawing = "NC(N)=NCCCC(=O)O"      # (H2N)2C=N-R, the Blue Book's drawing
    assert_pin_at_both_tiers(amidino_drawing, "4-(carbamimidoylamino)butanoic acid")
    assert_pin_at_both_tiers(ylidene_drawing, "4-[(diaminomethylidene)amino]butanoic acid")


# A ring-ylidene double bond with an E/Z flag: the senior ring is the parent and the ring-ylidene prefix of the
# other ring is built without the descriptor, which the whole name takes at the front of that prefix with the
# locant of its ylidene atom ('[(4E)-...-4-ylidene]methyl'). Specification by stereodescriptors 'E'
# and 'Z' (the Blue Book): method (b), "the stereodescriptor is cited at the front of the name of the
# substituent group of the 'ylidene' type; the name of the corresponding prefix is then placed in enclosing
# marks", is the general form and "Method (a) generates preferred IUPAC names" (the double bond as part of the
# parent, which a bond inside a prefix is not): such a name is never labelled pin_verified. The wide tiers had no
# producer for that prefix, so a molecule with such a bond and a p-block metal (no strict-path name) abstained.
STEREO_RING_YLIDENE = [
    ("C\\1COC2=C(/C1=C/C3=CN(C=N3)[Tl])C=C(C=C2)Br",
     "4-{[(4E)-6-bromo-3,4-dihydro-2H-1-benzopyran-4-ylidene]methyl}-1-(1λ1-thallamethyl)-1H-imidazole"),
    ("C1=CC=C2C(=C1)C=CC(=N2)/C=C\\3/C4=CC=CC=C4S(=O)(=O)N3[Al]",
     "2-{[(9Z)-8-(1λ1-aluminamethyl)-7,7-dioxo-7λ6-thia-8-azabicyclo[4.3.0]nona-1,3,5-trien-9-ylidene]methyl}quinoline"),
]


@pytest.mark.parametrize("smiles,best_effort", STEREO_RING_YLIDENE)
def test_stereo_flagged_ring_ylidene_branch_is_named_at_best_effort(smiles, best_effort):
    b = name_breadth(smiles)
    assert b["name"] == best_effort and b["tier"] != "pin_verified", b
    assert name_is_rt_exact(best_effort, smiles)


# method (b) is valid and is not the PIN: the label is below pin_unverified, the descriptor cited in the prefix
LABELLED_BELOW_THE_PIN = [
    ("CC1=N/C(=C\\C2=CC=C(N2)C)/C=C1", "2-methyl-5-{[(2Z)-5-methyl-2H-pyrrol-2-ylidene]methyl}-1H-pyrrole"),
    ("C\\1COC2=C(/C1=C/C3=CN(C=N3)C)C=C(C=C2)Br",
     "4-{[(4E)-6-bromo-3,4-dihydro-2H-1-benzopyran-4-ylidene]methyl}-1-methyl-1H-imidazole"),
]


@pytest.mark.parametrize("smiles,expected", LABELLED_BELOW_THE_PIN)
def test_the_descriptor_in_the_ylidene_prefix_is_labelled_below_the_pin(smiles, expected):
    b = name_breadth(smiles)
    assert b["name"] == expected and b["tier"] == "systematic_verified", b
    assert name_is_rt_exact(expected, smiles)


# A cation drawn with the charge on one ring: that ring is the cation parent, the other a ylidene prefix that keeps
# the configuration of its bond. The names of its two charge forms are both read back to the input's key, and the
# other form (the charge on the other nitrogen) drops the (2Z): it is the name of a less specific compound.
# (the Blue Book): the suffix 'ium' sits on the atom that carries the charge.
CYANINE = "CN\\1C2=CC=CC=C2S/C1=C\\C3=CC=[N+](C)C4=CC=CC=C34"
CYANINE_NAME = "1-methyl-4-{[(2Z)-3-methyl-2,3-dihydro-1,3-benzothiazol-2-ylidene]methyl}quinolin-1-ium"


def test_the_cation_parent_is_the_ring_that_carries_the_drawn_charge():
    from tests.support.rt_assert import _independent_parse
    b = name_breadth(CYANINE)
    assert b["name"] == CYANINE_NAME and b["tier"] == "systematic_verified", b
    assert name_is_rt_exact(CYANINE_NAME, CYANINE)
    mol = Chem.MolFromSmiles(_independent_parse(CYANINE_NAME, CYANINE))
    nitrogen = next(a for a in mol.GetAtoms() if a.GetSymbol() == "N" and a.GetFormalCharge() == 1)
    ring = min((r for r in mol.GetRingInfo().AtomRings() if nitrogen.GetIdx() in r), key=len)
    assert all(mol.GetAtomWithIdx(i).GetSymbol() != "S" for i in ring)    # the quinoline ring, as drawn

