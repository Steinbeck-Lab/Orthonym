"""Unnamed-breadth lane L3: PubChem rows that main left unnamed because a ring iodine (a lambda3 or lambda5
iodine with two or more ring bonds, or an iodonium or iodanuide centre) sat in a ring system no producer had
an 'ioda' spelling for: mixed-heteroatom and fused rings, cages, rings with a cumulated bond, rings with two
iodine atoms, rings that are substituents, and ring iodine ions.

The Hantzsch-Widman namers already named the monocycle with one lambda iodine, and lane D (``test_unnamed_d_
iodine_fused``) a benzene ring on a five- to seven-membered ring with an iodine. What was missing was the
ring 'a' ('ioda') spelling that the von Baeyer, spiro and monocycle replacement namers share
(``rules/ring_replacement``), so every polycycle with an iodine void; and, for the rest, small gates.

Governing rules (the Blue Book, section headings with the line):
* "Heterogeneous heterocyclic von Baeyer systems" (:9763,:9765): the replacement ('a') prefixes
  "are placed in front of the name of the corresponding hydrocarbon..., cited in the order: F > Cl > Br > I > O
  > S >... > Tl" -- iodine has its citation position. (:9789) ranks eighteen elements for a numbering
  tie and no halogen: an admitted halogen takes the rank of the citation order (ahead of oxygen), and a name
  that carries it is recorded as not the PIN.
* "Heterocyclic polyalicyclic parent hydrides having heteroatoms with nonstandard bonding numbers"
  (:9867): "The symbol is placed before the appropriate 'a' prefix"; (:6448): "placed immediately after
  the locant". A ring iodine has two or more ring bonds and the standard bonding number is 1, so the symbol is
  always cited.
* "Heteromonocyclic hydrides named by skeletal replacement ('a') nomenclature" (:8478): rings of ten or
  fewer members take the Hantzsch-Widman name,:9158, the lambda form), so the 'a' monocycle name
  of ``terminal_ring`` stands aside where those namers name the ring and is the name of the rings they cannot
  name (a cumulated ring, two atoms of one halogen). The ring-closure double bond of a replacement monocycle is
  cited with the compound locant '1(n)' criterion (1),:16635, OPSIN reads it).
* "Anions formed by addition of hydride ions" (:41089), method (2): "by adding the suffix 'ide' to a
  parent hydride in which the bonding number at the anionic site is higher than the standard bonding number and is
  expressed by the lambda-convention" ('diphenyl-lambda3-iodanide',:41110): the ring iodanuide centre is the
  'ide' of the lambda3 hydride ('1lambda3-iodinan-1-ide'), with two fluorines the lambda5 one.
* "Application of the lambda-convention with the suffix 'ylium'" (:41694): "Cationic heterocycle having a
  cationic center on a heteroatom that has one more skeletal bonds than it has in the corresponding neutral
  heterocycle is named by adding the suffix 'ylium' to the name of the neutral parent hydride for which the
  lambda-convention has been used to describe a nonstandard bonding state of the heteroatom"
  ('1lambda3-benziodol-1-ylium (PIN)',:41708): the ring iodonium centre is the 'ylium' of the lambda3 hydride,
  '1lambda3-iodinan-1-ylium'. It is not the 'ium' of (:41366), which adds a hydron to a hydride
  ('diphenyliodanium (PIN)',:41362): 'iodinan-1-ium' is the [IH2+] of one hydrogen pair more, which the
  standard InChIKey cannot tell from the cation.
* (:11813) and (2) (:15814) for the benzoheterocycle as a substituent
  ('2-(1H-1lambda3-benziodol-1-yl)ethan-1-amine'); (:18162): amines (class 19,:18192) are senior
  to the heterocycle, so the amine is the suffix of the chain and the ring a substituent.

Every name below is a best-effort name read back through a FRESH OPSIN 2.9.0 call to the full InChIKey of the
input (``assert_full_rt``); none is a PIN, and the default tier declines each (NO_VERIFIED_PIN), except the
lane-D parents whose ring system the book prints (``data/halogen_ring_parents``). The hand-checked name
facts, each with its locant rule:
* single-heteroatom ring systems take the lowest locant for the heteroatom,:9777), so
  '2lambda3-iodabicyclo[3.3.0]octane' and not '3lambda3-';
* a ring iodine cited with its lambda before 'ioda' ('2lambda3-ioda-3,5,7-triazabicyclo[4.3.0]nona-...'), iodine
  first in the citation order.
"""
import pytest
from rdkit import Chem, RDLogger

from tests.support.default_tier import default_tier_row
from tests.support.rt_assert import assert_full_rt, name_best_effort

RDLogger.DisableLog("rdApp.*")

# A test of what SHIPS runs with the OPSIN validity gate on (tests/conftest.py, the `opsin_gate` marker).
pytestmark = pytest.mark.opsin_gate

L = "λ"  # the lambda of the book's spelling

# ---------------------------------------------------------------------------------------------
# 1. The names, by class. (smiles, the best-effort name). A PubChem row where the class has one.
# ---------------------------------------------------------------------------------------------
POLYCYCLES = [
    # CID163418152: a pyrrole ring on a six-membered ring of one iodine and two nitrogens: 'ioda' before
    # 'aza'; the heteroatom locants {2,3,5,7} are the lowest set,
    ("C1=CNC2=C1I=NC=N2", f"2{L}3-ioda-3,5,7-triazabicyclo[4.3.0]nona-1(6),2,4,8-tetraene"),
    # CID164151012: an imidazo-fused six-membered ring with the iodine at the position next to a ring fusion
    ("CC1=CN2C=IC=CC2=N1", f"8-methyl-3{L}3-ioda-1,7-diazabicyclo[4.3.0]nona-2,4,6,8-tetraene"),
    # a lambda5 iodine in a ring: cites the bonding number of the atom it has
    ("C1=CC2=C(C=[IH]=C1)NCCCN2", f"9{L}5-ioda-2,6-diazabicyclo[5.5.0]dodeca-1(7),8,9,11-tetraene"),
    # CID163750784: the iodine is a bridge of a cage
    ("C1CC2CC1C1C3CCC(C3)I21", f"2{L}3-iodatetracyclo[6.2.1.1^3,6.0^2,7]dodecane"),
    # a spiro-fused tricycle with iodine and nitrogen
    ("C1CI2CC23CNCC13", f"3{L}3-ioda-8-azatricyclo[4.3.0.0^1,3]nonane"),
    ("C1CC2CC[IH]C2C1", f"2{L}3-iodabicyclo[3.3.0]octane"),
    ("C1CCC2(CC1)CC[IH]CC2", f"3{L}3-iodaspiro[5.5]undecane"),
    # the iodine of a benzo ring with a three-membered ring
    ("C1=CC2CI2C=C1", f"1{L}3-iodabicyclo[4.1.0]hepta-2,4-diene"),
]

MONOCYCLES = [
    # CID154529533: the iodine is numbered 1 and one of its double bonds closes the ring: the compound locant
    # '1(7)' criterion (1),:16635), a cumulated I=C=C pair no mancude parent holds
    ("C1=CC=NI=C=C1", f"1{L}3-ioda-2-azacyclohepta-1(7),2,4,6-tetraene"),
    # CID163919246: the same cumulated pair in a five-membered ring, with two substituents
    ("CC1=C(I=C=C1)C", f"4,5-dimethyl-1{L}3-iodacyclopenta-1,2,4-triene"),
    # CID142695052 shape: two iodine atoms in one ring: the Hantzsch-Widman lambda namer refuses
    # two atoms of one element, so the 'a' name is the one there is)
    ("C1=CI=CC=I1", f"1{L}3,4{L}3-diiodacyclohexa-1,3,5-triene"),
]

IONS = [
    # the iodanuide centre of a ring: 'ide' of the lambda3 hydride method (2),:41089)
    ("C1CC[I-]CC1", f"1{L}3-iodinan-1-ide"),
    ("C1=CC=C[I-]C=C1", f"1H-1{L}3-iodepin-1-ide"),                          # CID163685007 shape
    ("C1C[I-]1", f"1{L}3-iodiran-1-ide"),
    ("C1CC2CC1[I-]2", f"5{L}3-iodabicyclo[2.1.1]hexan-5-ide"),
    # with two fluorines on the iodine: the lambda5 hydride, the fluorines as prefixes of its locant
    ("F[I-]1(F)CCCCC1", f"1,1-difluoro-1{L}5-iodinan-1-ide"),
    ("c1ccc2c(c1)[I-]2(F)F", f"7,7-difluoro-7{L}5-iodabicyclo[4.1.0]hepta-1,3,5-trien-7-ide"),   # CID178177482
    # the iodonium centre of a ring: 'ylium' of the lambda3 hydride,:41694); the indicated hydrogen
    # that stands on the centre goes with the hydride ion ('1lambda3-benziodol-1-ylium (PIN)',:41708)
    ("C1CC[I+]CC1", f"1{L}3-iodinan-1-ylium"),
    ("C1=CC=C[I+]C=C1", f"1{L}3-iodepin-1-ylium"),
    ("C1CC2CC1[I+]2", f"5{L}3-iodabicyclo[2.1.1]hexan-5-ylium"),
    ("c1ccc2c(c1)[I+]C=C2", f"1{L}3-benziodol-1-ylium"),     # the book's own example,:41708
    ("c1ccc2c(c1)[I-]C=C2", f"1H-1{L}3-benziodol-1-ide"),    # the fusion name of the hydride, 'ide' of it
    ("c1ccc2c(c1)[I+]CC2", f"2,3-dihydro-1{L}3-benziodol-1-ylium"),
    # diphenyleneiodonium, a real cation: the von Baeyer name of its three rings, 'ylium' at the iodine
    ("c1ccc2c(c1)[I+]c1ccccc1-2", f"8{L}3-iodatricyclo[7.4.0.0^2,7]trideca-1(13),2,4,6,9,11-hexaen-8-ylium"),   # CID89022182
    # the ring ion is the parent whatever the principal group of the chain on it: the ionic centre)
    ("CCCC(C1C[I-]1)N", f"2-(1-aminobutyl)-1{L}3-iodacyclopropan-1-ide"),   # CID148631374
    ("CCCC(C1C[I+]1)N", f"2-(1-aminobutyl)-1{L}3-iodacyclopropan-1-ylium"),
]

SUBSTITUENTS = [
    # a carbon of the lambda ring is the attachment: the Hantzsch-Widman substituent the general engine builds,
    # kept (main named these before the lane)
    ("OC(=O)CC1CC[IH]CC1", f"(1{L}3-iodinan-4-yl)acetic acid"),
    # the free valence on the iodine: no Hantzsch-Widman substituent builder, the 'a' name of the ring
    ("NCCI1CCCCC1", f"2-(1{L}3-iodacyclohexan-1-yl)ethan-1-amine"),
    # a benzene ring on the iodine ring: the benzoheterocycle name of the substituent, (2))
    ("NCCI1C=Cc2ccccc12", f"2-(1H-1{L}3-benziodol-1-yl)ethan-1-amine"),
    ("NCCI1OC(C)(C)c2ccccc12", f"2-(3,3-dimethyl-1{L}3,2-benziodoxol-1(3H)-yl)ethan-1-amine"),
    # (:16558): "In rings modified by skeletal replacement ('a') nomenclature, low locants are assigned
    # first to heteroatoms and then to unsaturated sites": the iodine of the ring is 1 and the free valence has the
    # lowest locant that leaves (the attachment is 4, not the iodine 4 of a ring numbered from the attachment)
    ("c1cc(ncc1)C1CC[I-]CC1", f"2-(1{L}3-iodacyclohexan-1-id-4-yl)pyridine"),
    ("c1cc(ncc1)C1CC[I+]CC1", f"2-(1{L}3-iodacyclohexan-1-ylium-4-yl)pyridine"),
    ("C1=NC=CC=C1C1CCCC[I-]1", f"3-(1{L}3-iodacyclohexan-1-id-2-yl)pyridine"),
    ("c1cc(ncc1)C1C[I-]CCC1", f"2-(1{L}3-iodacyclohexan-1-id-3-yl)pyridine"),
]

ALL_ROWS = POLYCYCLES + MONOCYCLES + IONS + SUBSTITUENTS
IDS = [s for s, _ in ALL_ROWS]


def _assert_same_structure_strictly(name, smiles):
    """The canonical SMILES of OPSIN's structure of ``name`` is the input's. Stricter than the InChIKey of
    ``assert_full_rt``, which drops the two hydrogens of an ``[IH2+]`` (the standard InChI of ``C[IH2+]C`` and
    of ``C[I+]C`` is one string), so the 'ium' of the hydride 'iodinane' would pass it."""
    from tests.support.rt_assert import _independent_parse
    back = _independent_parse(name, smiles)
    assert back is not None, name
    assert Chem.MolToSmiles(Chem.MolFromSmiles(back)) == Chem.MolToSmiles(Chem.MolFromSmiles(smiles)), (
        name, back, smiles)


@pytest.mark.parametrize("smiles,name", ALL_ROWS, ids=IDS)
def test_best_effort_names_the_row_and_it_reads_back(smiles, name):
    res = name_best_effort(smiles)
    assert res.get("name") == name, res
    assert_full_rt(name, smiles)
    _assert_same_structure_strictly(name, smiles)
    assert not res.get("is_pin"), res


@pytest.mark.parametrize("smiles,name", ALL_ROWS, ids=IDS)
def test_default_tier_declines_what_the_book_prints_no_pin_for(smiles, name):
    """No row of the lane is a PIN: the numbering rank of a halogen is not a rule of and the book
    prints no von Baeyer, spiro, monocycle replacement or ion name with a ring halogen. The default tier is the
    PIN-only one: it gives a name only when the strict path built the PIN, so each row is an abstention there
    (the reason is UNNAMEABLE where no strict producer has the ring, NO_VERIFIED_PIN where one has it and
    the name is recorded as not the PIN: either is a decline, and none is a claim)."""
    row = default_tier_row(smiles)
    assert row["tier"] == "abstain" and not row["is_pin"], row


# The Hantzsch-Widman names of the parents and substituents that main named stay (the lane changed no name
# of a ring the lambda namers name): a pin_verified parent keeps its label, a substituent attached at a carbon keeps
# its Hantzsch-Widman spelling.
HW_NAMES_KEPT = [
    ("C1CC[IH]CC1", f"1{L}3-iodinane"),
    ("C1=CC=IC=C1", f"1{L}3-iodinine"),
    ("C1=CC=CIC=C1", f"1H-1{L}3-iodepine"),
    ("C1C[IH]1", f"1{L}3-iodirane"),
    ("C1=NNC[IH]1", f"2,3-dihydro-1H-1{L}3,3,4-iodadiazole"),
]


@pytest.mark.parametrize("smiles,name", HW_NAMES_KEPT, ids=[s for s, _ in HW_NAMES_KEPT])
def test_the_hantzsch_widman_parent_keeps_its_name_and_its_label(smiles, name):
    res = name_best_effort(smiles)
    assert res.get("name") == name, res
    assert_full_rt(name, smiles)
    # the default tier is the PIN-only one: it gives exactly the names the strict path built before
    assert default_tier_row(smiles).get("name") == name


# ---------------------------------------------------------------------------------------------
# 2. The two things the lane must NOT do.
# ---------------------------------------------------------------------------------------------

def test_a_hypervalent_iodine_in_a_chain_is_still_a_lambda_iodane_not_a_chain_replacement_atom():
    """ (:39660): the iodine of '-I(OH)2' is the parent hydride 'lambda3-iodane'. The ring table
    admits the halogen as a RING atom only: before the guard, the terminal-fragment chain writer took 'ioda'
    for the iodine of a chain and the name was '(1-hydroxy-1lambda3-ioda-2-oxaethyl)benzene'."""
    smiles = "OI(O)c1ccccc1"
    res = name_best_effort(smiles)
    assert res.get("name") == f"dihydroxy(phenyl)-{L}3-iodane", res
    assert_full_rt(res["name"], smiles)


def test_no_row_of_the_panel_loses_the_hantzsch_widman_name_of_a_ring_attached_at_a_carbon():
    """The floor's 'a' spelling is placed AFTER the general engine's route for a fragment with a ring halogen
    (``substituent_enumerator._ring_halogen_fragment``): '(1lambda3-iodacyclohexan-4-yl)acetic acid' is not
    what main said for this molecule."""
    res = name_best_effort("OC(=O)CC1CC[IH]CC1")
    assert "iodinan-4-yl" in res["name"] and "iodacyclohexan" not in res["name"], res


@pytest.mark.parametrize("smiles", ["C1CCC2(CC1)CC[I+]CC2", "C1CCC2(C1)C[I+]C2", "C1CCC2(CC1)CC[I-](F)(F)CC2"])
def test_a_spiro_iodine_ion_is_never_named_with_the_hydrogens_of_its_neutral_view(smiles):
    """The spiro analyzer reads the ring through a neutral view (``vonbaeyer_universal._extract_spiro_submol``),
    in which an iodonium is the lambda3 hydride; its name '3lambda3-iodaspiro[5.5]undecan-3-ium' is the [IH2+] of
    one hydrogen pair more, and the standard InChI key of that and of the cation is the same. The cation is the
    'ylium' of that hydride; whatever name there is must be the structure's own, strictly, and never
    the 'ium' of the hydride."""
    res = name_best_effort(smiles)
    if res.get("name") and res.get("tier") != "abstain":
        _assert_same_structure_strictly(res["name"], smiles)
        assert "-ium" not in res["name"].replace("ylium", ""), res


def test_the_spiro_iodonium_is_the_ylium_of_the_lambda3_hydride():
    smiles = "C1CCC2(CC1)CC[I+]CC2"
    res = name_best_effort(smiles)
    assert res.get("name") == f"3{L}3-iodaspiro[5.5]undecan-3-ylium", res
    _assert_same_structure_strictly(res["name"], smiles)


def test_the_spiro_iodanuide_of_two_ring_bonds_is_still_named():
    """The neutral view of an iodanuide with only ring bonds IS the lambda3 hydride its name is built on."""
    smiles = "C1CCC2(CC1)CC[I-]CC2"
    res = name_best_effort(smiles)
    assert res.get("name") == f"3{L}3-iodaspiro[5.5]undecan-3-ide", res
    _assert_same_structure_strictly(res["name"], smiles)


# ---------------------------------------------------------------------------------------------
# 3. The pieces, on the molecule alone (no OPSIN).
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,kind,lam", [
    ("C1CC[I-]CC1", "anion", 3),
    ("F[I-]1(F)CCCCC1", "anion", 5),
    ("C1CC[I+]CC1", "cation", 3),
    ("C[I+]1(C)CCCCC1", "cation", 5),
    ("C1CC[IH]CC1", None, None),           # neutral
    ("C[I-]C", None, None),                # not a ring atom
    ("C1=CC=C[I+](C)C=C1", None, None),    # three bonds: no hydride one hydrogen above it
    ("C1CC[I-]=CC1", None, None),          # a double bond on the centre
])
def test_ring_halogen_ion_and_its_lambda(smiles, kind, lam):
    from orthonym.rules.lambda_convention import ring_halogen_ion, ring_halogen_ion_lambda
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        pytest.skip("RDKit does not take the valence")
    idx = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "I")
    assert ring_halogen_ion(mol, idx) == kind
    assert ring_halogen_ion_lambda(mol, idx) == lam


def test_ring_table_spells_a_ring_iodine_only_in_a_ring():
    from orthonym.rules.ring_replacement import build_replacement_prefix
    ring = Chem.MolFromSmiles("C1CC[IH]CC1")
    got = build_replacement_prefix(ring, {a.GetIdx(): a.GetIdx() + 1 for a in ring.GetAtoms()},
                                   {a.GetIdx() for a in ring.GetAtoms()})
    assert got.unexpressed == () and got.prefix == f"4{L}3-ioda", got
    chain = Chem.MolFromSmiles("CC[IH]CC")
    got = build_replacement_prefix(chain, {a.GetIdx(): a.GetIdx() + 1 for a in chain.GetAtoms()},
                                   {a.GetIdx() for a in chain.GetAtoms()})
    assert len(got.unexpressed) == 1 and "ioda" not in got.prefix, got


def test_a_ring_halogen_ion_that_is_not_the_simple_centre_is_unexpressed():
    from orthonym.rules.ring_replacement import build_replacement_prefix
    # three bonds on a cation: no hydride has the bonding number one more than that of its atom
    mol = Chem.MolFromSmiles("C1CC[I+](C)CC1")
    got = build_replacement_prefix(mol, {a.GetIdx(): a.GetIdx() + 1 for a in mol.GetAtoms()},
                                   {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()})
    assert len(got.unexpressed) == 1, got
    # four bonds are the lambda5 hydride, the simple centre of 'ylium': '1,1-dimethyl-1lambda5-iodinan-1-ylium'
    mol = Chem.MolFromSmiles("C1CC[I+](C)(C)CC1")
    got = build_replacement_prefix(mol, {a.GetIdx(): a.GetIdx() + 1 for a in mol.GetAtoms()},
                                   {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()})
    assert got.unexpressed == () and got.prefix == f"4{L}5-ioda", got


def test_the_macrocycle_builder_does_not_take_the_ring_iodine():
    """A ring of more than ten members is named by ``heterocycles._build_replacement_name`` with no lambda
    citation: an iodine there would have lost its bonding number. The 'ioda' name of such a ring is the
    one of ``terminal_ring`` and the von Baeyer namers."""
    from orthonym.rules.heterocycles import _table_1_5_prefix
    assert _table_1_5_prefix("I") is None
    assert _table_1_5_prefix("O") == "oxa"


@pytest.mark.parametrize("smiles,name", [
    ("C1=CC=C[I+]C=C1", f"1{L}3-iodepin-1-ylium"),
    ("C1=CC=C[I-]C=C1", f"1H-1{L}3-iodepin-1-ide"),
    ("C1=CC=C[I+]C=C1C", f"3-methyl-1{L}3-iodepin-1-ylium"),
])
def test_the_ylium_takes_the_indicated_hydrogen_that_stands_on_its_own_atom(smiles, name):
    """ (:41694), the example '1lambda3-benziodol-1-ylium (PIN)' (:41708): the hydride '1H-1lambda3-
    benziodole' loses its hydride ion from the very atom that carries the indicated hydrogen, and the cation
    has none to cite. The 'ide' of the same hydride is not an example of the book and keeps the symbol."""
    res = name_best_effort(smiles)
    assert res.get("name") == name, res
    assert_full_rt(name, smiles)
    _assert_same_structure_strictly(name, smiles)


def test_an_indicated_hydrogen_on_another_atom_stays_with_the_ylium():
    """'3H-1lambda4-thiophen-1-ylium (PIN)' (:41692): the hydrogen that is not on the cationic centre is cited."""
    from orthonym.assembly.halogen_ring_ions import _drop_ylium_indicated_hydrogen
    mol = Chem.MolFromSmiles("C1CC[I+]CC1")
    idx = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "I")
    ring = {a.GetIdx() for a in mol.GetAtoms()}
    a2l = {a.GetIdx(): a.GetIdx() + 1 for a in mol.GetAtoms()}
    on_centre = f"{a2l[idx]}H-{a2l[idx]}{L}3-iodinine"
    elsewhere = f"3H-{a2l[idx]}{L}3-iodinine"
    assert _drop_ylium_indicated_hydrogen(on_centre, mol, ring, a2l) == f"{a2l[idx]}{L}3-iodinine"
    assert _drop_ylium_indicated_hydrogen(elsewhere, mol, ring, a2l) == elsewhere
    assert (_drop_ylium_indicated_hydrogen(f"2,3-dihydro-{on_centre}", mol, ring, a2l)
            == f"2,3-dihydro-{a2l[idx]}{L}3-iodinine")


class TestClosureBondLocant:
    """The compound locant '1(n)' of a replacement monocycle's ring-closure double bond, written and audited."""

    NAME = f"1{L}3-ioda-2-azacyclohepta-1(7),2,4,6-tetraene"

    def test_the_grammar_reads_it_as_the_bond_between_locant_one_and_n(self):
        from orthonym.rules.terminal_ring import parse_monocycle_replacement_name
        size, elements, doubles, triples, fv = parse_monocycle_replacement_name(self.NAME)
        assert size == 7 and elements == {1: "I", 2: "N"} and fv is None and not triples
        assert doubles == {(1, 7), (2, 3), (4, 5), (6, 7)}

    @pytest.mark.parametrize("name", [
        "cyclohexa-2(5),3-diene",        # a compound locant that is not the closure bond
        "cyclohexa-1(5),3-diene",        # '1(m)' with m not the ring size
        "cyclohexa-3(6),1-diene",        # the first locant of the bond is not 1
    ])
    def test_any_other_compound_locant_is_refused(self, name):
        from orthonym.rules.terminal_ring import parse_monocycle_replacement_name
        assert parse_monocycle_replacement_name(name) is None

    def test_the_audit_holds_the_name_to_the_molecule(self):
        from orthonym.rules.terminal_ring import (
            audit_monocycle_replacement_name, build_monocycle_replacement_name, monocycle_numbering)
        mol = Chem.MolFromSmiles("C1=CC=NI=C=C1")
        Chem.Kekulize(mol, clearAromaticFlags=True)
        ring = sorted(mol.GetRingInfo().AtomRings()[0])
        numbering = monocycle_numbering(mol, ring)
        name = build_monocycle_replacement_name(mol, ring, numbering)
        assert name == self.NAME
        assert audit_monocycle_replacement_name(mol, ring, numbering, name)
        # the closure bond moved to a single bond in the name: not the molecule
        assert not audit_monocycle_replacement_name(
            mol, ring, numbering, f"1{L}3-ioda-2-azacyclohepta-2,4,6-triene")


def test_the_monocycle_builder_stands_aside_for_a_ring_the_hantzsch_widman_namers_name():
    from orthonym.rules.terminal_ring import build_monocycle_replacement_name, monocycle_numbering
    for smiles, name in (("C1CC[IH]CC1", None), ("C1=CI=CC=I1", f"1{L}3,4{L}3-diiodacyclohexa-1,3,5-triene")):
        mol = Chem.MolFromSmiles(smiles)
        Chem.Kekulize(mol, clearAromaticFlags=True)
        ring = sorted(mol.GetRingInfo().AtomRings()[0])
        got = build_monocycle_replacement_name(mol, ring, monocycle_numbering(mol, ring))
        assert got == name, (smiles, got)


def test_the_benzoheterocycle_candidates_of_a_halogen_system_are_the_lane_d_parents():
    from orthonym.rules.halogen_ring_fusion import halogen_benzo_candidates
    mol = Chem.MolFromSmiles("NCCI1OC(C)(C)c2ccccc12")
    ring = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    got = halogen_benzo_candidates(mol, ring)
    assert got and {parent for parent, _pairs, _ih in got} == {f"1{L}3,2-benziodoxole"}
    # not a benzoheterocycle: a lone ring
    mono = Chem.MolFromSmiles("C1CC[IH]CC1")
    assert halogen_benzo_candidates(mono, {a.GetIdx() for a in mono.GetAtoms()}) is None
