"""Unnamed-breadth lane D (plan 2026-10-15-unnamed-breadth): PubChem rows that main left unnamed
because a ring iodine in the lambda state (a benziodoxolone, a benziodoxole, a benziodole...) had no
fusion component. The unnamed panel holds the signature in its ``hal_ring`` rows (95 of 2,793). Best-effort
must now name each row whose fused system is a benzene ring on a five- to seven-membered
heteromonocycle with a ring iodine, with a full read-back (full InChIKey + radical identity +
protonation site, a fresh OPSIN call).

Governing rules (the Blue Book Blue Book):
* 'Nomenclature based on halogen parent hydrides' (:39660): "For cyclic and acyclic parent
  hydrides the lambda convention is used to indicate nonstandard bonding numbers." Its examples print
  '1-methoxy-1λ3,2-benziodoxol-3(1H)-one (PIN)' (:39674) and '1H-1λ3-benziodole (PIN)' (:39692; the OCR
  of that line is math markup, so the QMUL copy https://iupac.qmul.ac.uk/BlueBook/P6a.html was read:
  it prints '1H-1λ3-benziodole (PIN)').
* 'Heteromonocyclic components fused to a benzene ring' (:11813): the locants of the
  heteroatoms go "at the front of the name consisting of the fusion prefix 'benzo' followed by... a
  Hantzsch-Widman systematic name... The locants cited correspond to the full bicyclic structure....
  locants are placed in the order corresponding to the order of citation of the heteroatoms in the
  heterocyclic component. The locant '1' is always assigned to the atom of the heterocyclic component
  next to a fusion atom... The letter 'o' of the 'benzo' prefix is elided when followed by a vowel.
  Indicated hydrogen is placed at the front of the name, when required."
* 'Fused ring systems with skeletal atoms with nonstandard bonding numbers' (:14531) and
   (:12452): the symbol "follows the locant of the atom with the nonstandard bonding number
  in accordance with the numbering of the fused system"; the examples '1λ4,5-benzodithiepine (PIN)'
  (:14549) and '3H-3λ3,2,4-benziodadioxepine (PIN)' (:14557) put it inside the list of heteroatom
  locants of a benzo name.
* 'added hydrogen' (:24689) for the '-3(1H)-one' of the benziodoxolone: the ketone at C3 and the
  hydrogen the mancude parent gives the iodine (a ring atom with bonding number three and two ring
  bonds takes one ring double bond,:9158: the lambda atom joins the mancude matching).

PIN track: a name is a PIN on the ring systems whose benzo name the book prints as a PIN
(``data/halogen_ring_parents.py``): 1λ3,2-benziodoxole, 1λ3-benziodole, 3λ3,2,4-benziodadioxepine. The
PIN_ROWS below are the rows built on those parents, each with the printed example it follows. Every other
parent (a six-membered ring, the isomeric 2,1-benziodoxole, a benziodazole, the λ5 states) is a valid name
the module labels below the PIN: the default tier declines it.
"""
import pytest
from rdkit import Chem, RDLogger

from tests.support.default_tier import assert_default_tier_declines, default_tier_row
from tests.support.pin_tiers import assert_pin_at_both_tiers
from tests.support.rt_assert import assert_full_rt, name_best_effort

RDLogger.DisableLog("rdApp.*")

# A test of what SHIPS runs with the OPSIN validity gate on (tests/conftest.py, the `opsin_gate` marker).
pytestmark = pytest.mark.opsin_gate

# ---------------------------------------------------------------------------------------------
# 1. The rows: the lead's RED rows and the real benziodoxole rows of the unnamed panel.
# ---------------------------------------------------------------------------------------------
ROWS = [
    # (id, smiles)
    ("178256366", "COC1=CC2=C(C=C1)I(OC2=O)C(F)(F)F"),
    ("plain-methyl", "O=C1OI(C)c2ccccc12"),
    ("book-methoxy", "COI1OC(=O)c2ccccc12"),
    ("book-benziodole", "[IH]1C=Cc2ccccc12"),
]

PANEL_ROWS = [
    # unnamed-panel rows (benchmarks/unnamed_panel/unnamed_panel.tsv, hal_ring signature) that main did not
    # name: Togni-type 3,3-disubstituted benziodoxoles, a benziodoxolone with an alkenyl on the iodine, an
    # iodine with an acyl or an acyloxy, a benziodazole.
    ("178256419", "CC1(C2=C(C=CC(=C2)OC)I(O1)C(F)(F)F)C"),
    ("177855636", "C1=CC=C(C=C1)I2C3=CC=CC=C3C(O2)(C(F)(F)F)C(F)(F)F"),
    ("177855566", "COC1=CC=C(C=C1)/C=C/I2C3=CC=CC=C3C(=O)O2"),
    ("101998644", "CC1(C2=CC=CC=C2I(O1)OC(=O)C(F)(F)F)C3=CC=CC=C3"),
    ("178524709", "CC(C)(C)C(=O)I1C2=CC=CC=C2C(O1)(F)F"),
    ("143422320", "CCI1C2=C(C=C(C=C2)OC3=CC(=NC=C3)C4=NC=C(N4)C5=CC=CC=C5)N=C1NC6=CC=C(C=C6)SC(F)(F)F"),
]

REAGENT_ROWS = [
    # the benziodoxole reagents of the literature: the lambda5 periodinane, an iodine with a halogen,
    # a hydroxy, an acyloxy, an aryl; a stereocentre at C3; the five- and the six-membered ketones
    ("dess-martin", "CC(=O)OI1(OC(C)=O)(OC(C)=O)OC(=O)c2ccccc12"),
    ("1-chloro-3-one", "ClI1OC(=O)c2ccccc12"),
    ("1-hydroxy-3-one", "OI1OC(=O)c2ccccc12"),
    ("1-acetoxy-3-one", "CC(=O)OI1OC(=O)c2ccccc12"),
    ("1-phenyl-3-one", "O=C1OI(c2ccccc2)c2ccccc12"),
    ("5-chloro", "ClI1OC(=O)c2cc(Cl)ccc12"),
    ("3-stereocentre", "C[C@@]1(CC)OI(Cl)c2ccccc12"),
    ("benziodoxole-dihydro", "[IH]1OCc2ccccc12"),
    ("2,1-benziodoxole", "C1=CC=C2C(=C1)OI(C)C2"),
    ("benziodoxine", "CI1OC=Cc2ccccc12"),
    ("benziodazole-amine", "CI1N=C(N)c2ccccc12"),
    ("benziodazol-3-one", "O=C1NI(C)c2ccccc12"),
    ("benziodole-3-one", "O=C1CI(Cl)c2ccccc12"),
    ("benziodoxin-3-one", "O=C1OI(C)c2ccccc2C1"),
]

ALL_ROWS = ROWS + PANEL_ROWS + REAGENT_ROWS
IDS = [c for c, _ in ALL_ROWS]


@pytest.mark.parametrize("cid,smiles", ALL_ROWS, ids=IDS)
def test_best_effort_names_row_with_full_read_back(cid, smiles):
    res = name_best_effort(smiles)
    name = assert_full_rt(res.get("name"), smiles, what=f"{cid}: ")
    assert "λ" in name and "lambda" not in name, name


# The rows built on a ring system whose benzo name the book prints as a PIN (module docstring "PIN track").
PIN_ROWS = {
    "178256366",       # 1λ3,2-benziodoxole::39674, '-3(1H)-one', 1-(trifluoromethyl) for the printed 1-methoxy
    "plain-methyl",    # the printed example with methyl for methoxy
    "book-methoxy",    #:39674 verbatim
    "book-benziodole", #:39692 verbatim
    "178256419", "177855636", "177855566",                           # 1λ3,2-benziodoxole parents
    "1-chloro-3-one", "1-hydroxy-3-one", "1-phenyl-3-one", "5-chloro", "3-stereocentre",
    "benziodoxole-dihydro",
    "benziodole-3-one",   # 1λ3-benziodole parent (:39692) with the ketone of the indole class: '1,2-dihydro-3H-...-3-one'
}


@pytest.mark.parametrize("cid,smiles", ALL_ROWS, ids=IDS)
def test_default_tier_never_claims_a_pin_without_a_citation(cid, smiles):
    res = default_tier_row(smiles)
    assert not res.get("is_pin") or cid in PIN_ROWS, f"{cid}: PIN claimed without a printed ring system"


# Each PIN case: the book's printed example (verbatim) or the lead's example, with the line it follows.
PIN_CASES = [
    # 'Nomenclature based on halogen parent hydrides', the Blue Book: "1-methoxy-1λ3,2-benziodoxol-3(1H)-one (PIN)"
    ("COI1OC(=O)c2ccccc12", "1-methoxy-1λ3,2-benziodoxol-3(1H)-one"),
    #,:39692 (QMUL copy checked): "1H-1λ3-benziodole (PIN)"
    ("[IH]1C=Cc2ccccc12", "1H-1λ3-benziodole"),
    #, the Blue Book, example for indicated hydrogen: "3H-3λ3,2,4-benziodadioxepine (PIN)"
    ("[IH]1OC=C2C=CC=CC2=CO1", "3H-3λ3,2,4-benziodadioxepine"),
    # CID 178256366: the construction of:39674 with the substituents of the row; the alphanumerical order of
    # the prefixes is methoxy < trifluoromethyl and the locant 5 is the benzene carbon of the methoxy
    ("COC1=CC2=C(C=C1)I(OC2=O)C(F)(F)F", "5-methoxy-1-(trifluoromethyl)-1λ3,2-benziodoxol-3(1H)-one"),
]


@pytest.mark.parametrize("smiles,pin", PIN_CASES, ids=[p for _, p in PIN_CASES])
def test_pin_tier_gives_the_cited_pin(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


# ---------------------------------------------------------------------------------------------
# 2. Where the book prints no PIN the name is a name, not a PIN: the default tier declines it.
# ---------------------------------------------------------------------------------------------
NOT_PRINTED = [
    ("CI1OC=Cc2ccccc12", "1-methyl-1H-1λ3,2-benziodoxine"),                       # six-membered ring
    ("C1=CC=C2C(=C1)OI(C)C2", "2-methyl-2,3-dihydro-2λ3,1-benziodoxole"),          # the isomeric ring
    ("CI1N=C(N)c2ccccc12", "1-methyl-1H-1λ3,2-benziodazol-3-amine"),               # a benziodazole
    ("CC(=O)OI1(OC(C)=O)(OC(C)=O)OC(=O)c2ccccc12",
     "1,1,1-tris(acetyloxy)-1λ5,2-benziodoxol-3(1H)-one"),                          # the lambda5 state
]


@pytest.mark.parametrize("smiles,name", NOT_PRINTED, ids=[n for _, n in NOT_PRINTED])
def test_a_parent_the_book_does_not_print_is_best_effort_only(smiles, name):
    assert_default_tier_declines(smiles)
    assert name_best_effort(smiles).get("name") == name


# A listed parent whose principal characteristic group sits elsewhere in the molecule: the ring system is
# then a prefix of the principal group's parent, not the parent, so the benzo name is no PIN of the molecule.
# 'Seniority order for classes' (the Blue Book), Table 4.1 (:18162): acids are class 7, amines
# class 19, both senior to the ketone (16) and to the heterocycle (21); (:18875): the senior parent
# structure carries "the principal characteristic group (suffix)... in accord with the seniority of classes
# ". The PIN is '4-(3-oxo-...-1(3H)-yl)benzoic acid' and '2-(...-1-yl)ethan-1-amine'; the engine
# spells the acid as 'carboxy' and the amine as 'aminoethyl' on the ring system, a valid name, never the PIN.
#
# Lane L3 changed both best-effort names (ring iodine was admitted to the ring 'a' table, so the floor
# and the substituent writers now name the ring system, and their names come before the senior-group
# rescue that gave the old ones):
# acid: '1-(4-carboxyphenyl)-1λ3,2-benziodoxol-3(1H)-one'
# -> '1-(4-carboxyphenyl)-3-oxo-1,3-dihydro-1λ3,2-benziodoxole'
# UNVERIFIED as an improvement: both are valid names of the same structure, neither the PIN
# (the acid is a prefix in both); machine-confirmed only that the new one reads back to the full
# InChIKey (a fresh OPSIN 2.9.0 call). Open item: the PIN '4-(3-oxo-...-1-yl)benzoic acid' wants a
# benzoic-acid parent for a benzoyl-bearing floor, which no producer builds yet.
# amine: '1-(2-aminoethyl)-3,3-dimethyl-1,3-dihydro-1λ3,2-benziodoxole'
# -> '2-(3,3-dimethyl-1λ3,2-benziodoxol-1(3H)-yl)ethan-1-amine'
# machine-confirmed: the new name has the parent the book requires (Table 4.1 'General compound
# classes listed in decreasing order of seniority' (the Blue Book): "19 Amines" (:18192) are
# senior to the heterocycle, so the amine is the suffix of ethane) and the ring as a substituent with
# the free valence on the ring heteroatom in the added-hydrogen form the book prints for
# '1-(3,4-dihydroquinolin-1(2H)-yl)ethan-1-one (PIN)' (:7467); it reads back to the full InChIKey; and
# the base tree gives the old name, so this assertion fails without the lane.
PRINCIPAL_GROUP_ELSEWHERE = [
    ("OC(=O)c1ccc(cc1)I1OC(=O)c2ccccc12", "1-(4-carboxyphenyl)-3-oxo-1,3-dihydro-1λ3,2-benziodoxole"),
    ("NCCI1OC(C)(C)c2ccccc12", "2-(3,3-dimethyl-1λ3,2-benziodoxol-1(3H)-yl)ethan-1-amine"),
]


@pytest.mark.parametrize("smiles,name", PRINCIPAL_GROUP_ELSEWHERE, ids=["acid", "amine"])
def test_default_tier_declines_when_the_principal_group_is_not_on_the_ring_system(smiles, name):
    row = default_tier_row(smiles)
    assert row["tier"] != "pin_verified" and not row["is_pin"], row
    res = name_best_effort(smiles)
    assert res.get("name") == name and not res.get("is_pin"), res
    assert_full_rt(res["name"], smiles)


def test_the_lambda5_state_is_not_claimed_as_a_pin():
    dmp = "CC(=O)OI1(OC(C)=O)(OC(C)=O)OC(=O)c2ccccc12"
    assert not default_tier_row(dmp).get("is_pin")


# ---------------------------------------------------------------------------------------------
# 3. The pieces, on the molecule alone (no OPSIN).
# ---------------------------------------------------------------------------------------------

def _benzo_name(smiles):
    from orthonym.rules.halogen_ring_fusion import name_halogen_benzo_system
    return name_halogen_benzo_system(Chem.MolFromSmiles(smiles))


@pytest.mark.parametrize("smiles,name", [
    ("[IH]1OC=C2C=CC=CC2=CO1", "3H-3λ3,2,4-benziodadioxepine"),
    ("[IH]1C=Cc2ccccc12", "1H-1λ3-benziodole"),
    ("[IH]1OCc2ccccc12", "1,3-dihydro-1λ3,2-benziodoxole"),
    ("CC1(C)OI(C(F)(F)F)c2ccccc12", "3,3-dimethyl-1-(trifluoromethyl)-1,3-dihydro-1λ3,2-benziodoxole"),
], ids=["dioxepine", "benziodole", "dihydro", "togni"])
def test_the_benzo_name_cites_the_lambda_inside_the_heteroatom_locants(smiles, name):
    assert _benzo_name(smiles) == name


def test_a_ring_system_outside_the_scope_is_declined():
    # a naphthalene, a fused ring of three rings, a charged iodine, a benzene ring on no halogen ring
    for smiles in ("c1ccc2ccccc2c1", "C[Si](C)(C)OC1=CC2=C(C=C1)[I+]C3=CC=CC=C32", "c1ccc2occc2c1",
                   "[SH2]1C=CC2=C1C=CC=C2"):
        assert _benzo_name(smiles) is None, smiles


def test_the_ring_valence_of_a_halogen_is_its_bonding_number():
    from orthonym.rules.lambda_convention import ring_lambda_valence
    m = Chem.MolFromSmiles("COI1OC(=O)c2ccccc12")
    iodine = next(a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == "I")
    assert ring_lambda_valence(m, iodine) == 3
    # a sulfur of bonding number four is not admitted: it keeps the old behaviour
    s = Chem.MolFromSmiles("[SH2]1C=CC2=C1C=CC=C2")
    assert ring_lambda_valence(s, next(a.GetIdx() for a in s.GetAtoms() if a.GetSymbol() == "S")) is None


def test_an_iodine_outside_a_ring_is_not_a_ring_halogen():
    from orthonym.rules.lambda_convention import ring_lambda_valence
    m = Chem.MolFromSmiles("c1ccccc1I(Cl)Cl")
    iodine = next(a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == "I")
    assert ring_lambda_valence(m, iodine) is None
    assert _benzo_name("c1ccccc1I(Cl)Cl") is None
