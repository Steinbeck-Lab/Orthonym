"""Charged molecules of the PubChem left-out pool that the lanes ION and ION2 left without a name and now get a verified
one (lane ION3).

Every case is named at the best-effort tier and read back through OPSIN independently of the engine's own gate
(``tests.support.rt_assert``): the name must read back to the full InChIKey of the input and to the canonical SMILES
of it, which carries the charges. Each class is pinned with the rule that governs it:

* the chalcogen anion of a group on a heteroatom, "Anionic centers on other characteristic groups"
  (the Blue Book): "named on the basis of the corresponding anionic parent hydrides" (:41071), 'tert-butyl-
  trioxidanide (PIN)' (:41075), 'acetylazanide (PIN)' (:41079): the oxygen of a phosphinite, a sulfenate or a
  hydroxylamine anion is the oxidanide of its group, the sulfur or selenium the sulfanide or selanide;
* the centre that holds two or three charges, "Anions derived from parent hydrides and their derivatives"
  (:40900): "Numerical prefixes 'di', 'tri', etc. are used to denote multiplicity" ('diphenylmethanediide (PIN)',:40910),
  and "Cationic centers in parent hydrides" (:41551): "Di- and polycations... are named by using the
  suffix 'ylium' and the multiplying prefixes 'bis', 'tris', etc." (the paragraph at:41559); the book's example of the
  form, 'ethylazanebis(ylium) (PIN)' (:41672), stands under "Cationic centers on other characteristic groups"
  (:41656);
* the boron of a mancude ring that adds a hydride ion, "ANIONS FORMED BY ADDITION OF HYDRIDE IONS" (:41089), the
  suffix 'uide' on the ring parent ('1,1-dimethylborinan-1-uide (PIN)',:41120), with the cumulative 'ium' of
   (:42415) for the ring that holds a cation as well;
* the silicon anion R4Si- of a ring, method (2) (:41094): the 'ide' of the lambda5 hydride, as the acyclic
  'tetramethyl-lambda5-silanide' (the book prints 'tetramethyl-λ5-phosphanide' for the phosphorus analogue,:41106, and
  'diphenyl-λ3-iodanide',:41110); R4Si- is a radical anion by its electron count, whose method (1) name is the 'uidyl'
  of (:43424, 'trimethylboranuidyl (PIN)',:43430), so the 'ide' is recorded as no PIN;
* the zwitterion whose anion is a nitrogen centre, "Anionic and cationic centers on different parent
  structures" (:42462): the anion is the parent and the cation a prefix, '2-(trimethylazaniumyl)ethan-1-aminide' (the
  locant of the amine cited once a group stands on the chain,,:2869), or the azanide of its groups,
   (:41079).

The names of the classes the Blue Book gives no example of are not PINs: the tier of each is below the PIN's.
"""
import pytest
from rdkit import Chem

from tests.support.rt_assert import _full_inchikey, _independent_parse, name_best_effort

# The suite runs with the OPSIN validity gate off. A ring ion whose parent hydride is offered first without its charges
# ('1-(triethylazaniumyl)borinine') has that candidate withdrawn by the gate in a run of the tool, so these cases run
# with the gate on (the marker of tests/conftest.py), as the names are shipped.
pytestmark = [pytest.mark.unit, pytest.mark.roundtrip, pytest.mark.opsin_gate]


def _reads_back(smiles: str, name: str) -> bool:
    parsed = _independent_parse(name, smiles)
    if not parsed:
        return False
    read, want = Chem.MolFromSmiles(parsed), Chem.MolFromSmiles(smiles)
    return (read is not None and _full_inchikey(parsed) == _full_inchikey(smiles)
            and Chem.MolToSmiles(read) == Chem.MolToSmiles(want))


CASES = [
    # (smiles, name, rule)
    # -- the chalcogen anion of a group on a heteroatom,:41071)
    ("CCCCSP([O-])Cl", "[(butylsulfanyl)(chloro)phosphanyl]oxidanide", "P-72.2.2.2.4"),
    ("COP(CCCCC1=CC=CC=C1)[O-]", "[methoxy(4-phenylbutyl)phosphanyl]oxidanide", "P-72.2.2.2.4"),
    ("CCCCCCCCCCCCCCP[O-]", "(tetradecylphosphanyl)oxidanide", "P-72.2.2.2.4"),
    ("CC(C)N(C(C)C)P([O-])OCCC#N", "{(2-cyanoethoxy)[di(propan-2-yl)amino]phosphanyl}oxidanide", "P-72.2.2.2.4"),
    ("C1=CC=C(C=C1)P[Se-]", "(phenylphosphanyl)selanide", "P-72.2.2.2.4"),
    ("C1CCC(CC1)NOP[O-]", "{[(cyclohexylamino)oxy]phosphanyl}oxidanide", "P-72.2.2.2.4"),
    # -- the anion of a group that is an oxoacid group, when no producer named it (the producers of the acid anions come first)
    ("CCN1CC[C@H](C1)CC2=CC=CC=C2N(C3=C(C4=C(C=C3)[C@H]5C[C@H]5CO4)C(=O)O)S(=O)[O-]",
     "(2S,3R,4R)-[1-(1-[8-carboxy-6-oxatricyclo[5.4.0.0^2,4]undeca-1(7),8,10-trien-9-yl]-1-{2-[1-(1-ethylpyrrolidin-3-yl)methan-1-yl]phenyl}"
     "-1-azamethan-1-yl)-2-oxa-1-thiaeth-1-en-1-yl]oxidanide", "P-72.2.2.2.4"),
    # -- the centre of two or three charges,:40900;,:41559;,:41656)
    ("CC(C)[P-2]", "(propan-2-yl)phosphanediide", "P-72.2.2.1"),
    ("CC1=C(C(=CC=C1)C)[P-2]", "(2,6-dimethylphenyl)phosphanediide", "P-72.2.2.1"),
    ("C#C[Si+3]", "ethynylsilanetris(ylium)", "P-73.2.2.1"),
    ("C([Si+3])(Cl)Cl", "(dichloromethyl)silanetris(ylium)", "P-73.2.2.1"),
    # -- the boron of a mancude ring that adds a hydride ion,:41089;,:42415)
    ("[B-]1(=CC=CC=C1)OCC2=CC=CC=C2", "1-(benzyloxy)borinin-1-uide", "P-72.3"),
    ("[B-]1(=CC=CC=C1)[N+](CC)(CC)CC", "1-(triethylazaniumyl)borinin-1-uide", "P-72.3"),
    ("[B-]1(=CC=C[N+](=C1)C)F", "3-fluoro-1-methyl-1,3-azaborinin-1-ium-3-uide", "P-74.1.1"),
    ("[B-]1(=CC=C[N+](=C1)C)OC(=O)C", "3-(acetyloxy)-1-methyl-1,3-azaborinin-1-ium-3-uide", "P-74.1.1"),
    # -- the boranuide with a ligand held by a double bond: the groups are prefixes of the parent hydride,:41098)
    ("[B-](=NC#C)(C)C", "(ethynylimino)di(methyl)boranuide", "P-72.3"),
    ("[B-](=S)(C)I", "iodo(methyl)(sulfanylidene)boranuide", "P-72.3"),
    ("[B-](=COCCCO)(F)F", "difluoro[(3-hydroxypropoxy)methylidene]boranuide", "P-72.3"),
    # a methylidene ligand is a ligand (only the cap of a radical's stand-in is not, see the bonding-number cases below)
    ("C=[B-](C)C", "dimethyl(methylidene)boranuide", "P-72.3"),
    ("C=[B-](F)F", "difluoro(methylidene)boranuide", "P-72.3"),
    # -- the silicon anion of a ring: the 'ide' of the lambda5 hydride method (2),:41094)
    ("C[Si-]1(C)CCCC1", "1,1-dimethyl-1\u03bb5-silolan-1-ide", "P-72.3"),
    ("CN(C)C1=CC2=C(C=C1)C(=O)C3=C([Si-]2(C)C)C=C(C=C3)N(C)C",
     "5,13-bis(dimethylamino)-2,2-dimethyl-9-oxo-2\u03bb5-silatricyclo[8.4.0.0^3,8]tetradeca-1(10),3(8),4,6,11,13-hexaen-2-ide",
     "P-72.3"),
    ("CC1=CC=CC=C1[Si-]2(CC3=CC=CC=C3C2)C4=CC=CC=C4C", "2,2-bis(2-methylphenyl)-2,3-dihydro-1H-2\u03bb5-benzosilol-2-ide",
     "P-72.3"),
    # -- the zwitterion of a nitrogen anion: the anion is the parent, the cation a prefix,:42462)
    ("C[N+](C)(C)CC[NH-]", "2-(trimethylazaniumyl)ethan-1-aminide", "P-74.1.3"),
    ("CC[N+](CC)(CC)CC[NH-]", "2-(triethylazaniumyl)ethan-1-aminide", "P-74.1.3"),
    ("C[N+](C)(C)CCC[NH-]", "3-(trimethylazaniumyl)propan-1-aminide", "P-74.1.3"),
    ("C=CC[N+](C)(CC=C)CC[N-]C", "methyl{2-[methyldi(prop-2-en-1-yl)azaniumyl]ethyl}azanide", "P-72.2.2.2.4"),
    # -- the azanide of a carbon group that has no amine name,:41079)
    ("N#C[NH-]", "cyanoazanide", "P-72.2.2.2.4"),
]


@pytest.mark.parametrize("smiles,expected,rule", [pytest.param(*c, id=c[0]) for c in CASES])
def test_name_pinned_and_read_back(smiles, expected, rule):
    row = name_best_effort(smiles)
    assert row["name"] == expected, (rule, row["name"], row["tier"])
    assert _reads_back(smiles, row["name"])


@pytest.mark.parametrize("smiles", [c[0] for c in CASES])
def test_class_without_a_blue_book_example_is_no_pin(smiles):
    """The Blue Book gives no example of these ions: the name is valid, not preferred."""
    assert name_best_effort(smiles)["tier"] != "pin_verified"


# The ester anions of a phosphorothioic acid are drawn with the semipolar pair (P+-O-): OPSIN reads the name as P=O, the
# same anion with the same full InChIKey and another SMILES, so these cases compare the key only. Each ester word is cited
# with the locant O of its oxygen, the Blue Book: 'sodium (R)-(O-methyl O-phenyl phosphorothioate) (PIN)'):
# the ester of a thio acid has two constitutions, on an oxygen or on a sulfur, and the bare 'phenyl phosphorodithioate'
# reads back only by OPSIN's default of the oxygen.
THIOPHOSPHATE_CASES = [
    ("c1ccccc1O[P+]([O-])([S-])[S-]", "O-phenyl phosphorodithioate"),
    ("CCCCCCCCC1=CC(=C(C=C1)CCCCCCCC)O[P+]([O-])([S-])[S-]", "O-(2,5-dioctylphenyl) phosphorodithioate"),
    ("C1CCC(C1)CO[P+]([O-])(OC[C@@H]2[C@H](C[C@@H](O2)N3C=NC4=C(N=CN=C43)N)O)[S-]",
     "O-{[(2R,3S,5R)-5-(6-amino-9H-purin-9-yl)-3-hydroxyoxolan-2-yl]methyl} O-(cyclopentylmethyl) phosphorothioate"),
    # the words in order: 'methyl' before '[(trimethylstannyl)methyl]' (a sort of the raw strings puts the bracket first)
    ("C[Sn](C)(C)COP(=S)([O-])OC", "O-methyl O-[(trimethylstannyl)methyl] phosphorothioate"),
]


@pytest.mark.parametrize("smiles,expected", [pytest.param(*c, id=c[0]) for c in THIOPHOSPHATE_CASES])
def test_thiophosphate_ester_anion_name_reads_back_to_the_same_anion(smiles, expected):
    """ "Acid esters of organic acids" (the Blue Book) with the functional replacement of
    (:35592): the ester anion of a thio acid is the ester words, each with its locant O,:46016;,
    :35954), and 'phosphorothioate'. No PIN: the book's PIN is the name of a SALT of the diester anion."""
    row = name_best_effort(smiles)
    assert row["name"] == expected, (row["name"], row["tier"])
    assert row["tier"] != "pin_verified"
    parsed = _independent_parse(row["name"], smiles)
    assert parsed and _full_inchikey(parsed) == _full_inchikey(smiles)


def test_the_rescue_asks_the_specific_class_before_the_oxidanide_of_the_group():
    """``CCOP(=S)([O-])OCC`` is named by a candidate before the rescue is reached; the rescue itself, asked for it, gives the
    ester anion ('O,O-diethyl phosphorothioate', two identical words) and not the oxidanide of the group of its oxygen."""
    from orthonym.metrics.provenance import best_effort_ctx
    from orthonym.rules.charged_fallback import late_charged_name
    smiles = "CCOP(=S)([O-])OCC"
    token = best_effort_ctx.set(True)       # the rescue runs in the best-effort cascade: there the group is named
    try:
        assert late_charged_name(smiles) == "O,O-diethyl phosphorothioate"     # not '(diethoxyphosphorothioyl)oxidanide'
    finally:
        best_effort_ctx.reset(token)
    parsed = _independent_parse("O,O-diethyl phosphorothioate", smiles)
    assert parsed and _full_inchikey(parsed) == _full_inchikey(smiles)


def test_the_locant_of_a_thiophosphate_ester_is_essential():
    """'S-phenyl phosphorodithioate' is another constitution (PhS-P) than 'O-phenyl phosphorodithioate' (PhO-P), so the
    locant of the ester word is cited, not left to the default of a reader, the Blue Book)."""
    on_oxygen = "c1ccccc1O[P+]([O-])([S-])[S-]"
    on_sulfur = _independent_parse("S-phenyl phosphorodithioate", on_oxygen)
    assert on_sulfur and _full_inchikey(on_sulfur) != _full_inchikey(on_oxygen)


# "Anionic and cationic centers on different parent structures" (the Blue Book): "Zwitterionic compounds with
# anionic and cationic centers on different parent structures may be named by prefixing the name of the cationic center... to
# the name of the anionic parent structure" (:42464); the book's own example has a phosphaniumyl prefix on an anionic parent,
# 'trimethyl{2-[methyldi(phenyl)phosphaniumyl]ethen-1-yl}boranuide (PIN)' (:42468), and the preferred name of a primary amine
# anion is the 'aminide',:41047). The router names these zwitterions at the PIN tier where the base tree named
# nothing (the lane creates names at the PIN tier through the router, not only below it): a phosphonium, a sulfonium and a
# ring N+, 'pyridin-1-ium-1-yl (preferred prefix)') on the amine anion. The azaniumyl of a carbon-substituted N+ is the
# one cation prefix labelled below the PIN (``record_amine_cation_prefix``: the PIN prefix of an amine cation is built on 'aminium').
@pytest.mark.parametrize("smiles,expected", [
    ("C[P+](C)(C)CC[NH-]", "2-(trimethylphosphaniumyl)ethan-1-aminide"),
    ("C[S+](C)CC[NH-]", "2-(dimethylsulfaniumyl)ethan-1-aminide"),
    ("C1=CC=[N+](CC[NH-])C=C1", "2-(pyridin-1-ium-1-yl)ethan-1-aminide"),
])
def test_amine_anion_zwitterion_of_an_onium_prefix_is_named_at_the_pin_tier(smiles, expected):
    from orthonym import Orthonym
    from orthonym.cli import _emit_tier_flags
    row = name_best_effort(smiles)
    assert row["name"] == expected, (row["name"], row["tier"])
    assert row["tier"] == "pin_verified"
    assert _reads_back(smiles, row["name"])
    default = Orthonym(style="pin", **_emit_tier_flags("pin")).name_tiered(smiles)
    assert (default["name"], default["tier"]) == (expected, "pin_verified")


def test_amine_anion_zwitterion_of_an_azaniumyl_prefix_is_below_the_pin():
    """A carbon group on the N+ makes an amine cation, whose PIN prefix is built on 'aminium' method (1),:42296)."""
    row = name_best_effort("C[N+](C)(C)CC[NH-]")
    assert row["name"] == "2-(trimethylazaniumyl)ethan-1-aminide" and row["tier"] == "systematic_verified"


def test_the_cap_of_a_radical_stand_in_is_no_boron_ligand():
    """``universal_substituent._cap_attachment`` closes the free valences of ``[B-](F)CN(C)C`` with a tagged carbon; the bond to it
    is not counted (the radical is named as a radical ion,, while the same bond to a carbon of a molecule is."""
    from orthonym.rules.uide_lambda import ATTACHMENT_CAP_PROP, uide_bonding_number
    mol = Chem.RWMol(Chem.MolFromSmiles("C=[B-](F)CN(C)C"))
    boron = next(a for a in mol.GetAtoms() if a.GetSymbol() == "B")
    assert uide_bonding_number(boron) == 4
    next(a for a in mol.GetAtoms() if a.GetSymbol() == "C" and a.GetDegree() == 1 and a.GetTotalNumHs() == 2
         and any(n.GetSymbol() == "B" for n in a.GetNeighbors())).SetBoolProp(ATTACHMENT_CAP_PROP, True)
    assert uide_bonding_number(boron) == 3


@pytest.mark.parametrize("parent,expected", [
    ("CCN", True), ("CCCN", True), ("CCCCCN", True),     # an amine on the end of an unbranched saturated chain: C1 is the amine carbon
    ("CN", False),                                        # one carbon: no chain locant
    ("CC(C)CN", False), ("CC(C)N", False),                # branched, or the amine inside the chain
    ("C=CCN", False), ("C#CCN", False),                   # unsaturated: the name cites the unsaturation too
    ("NCCO", False), ("NCCC(=O)O", False),                # another group on the chain
    ("NC1CCCC1", False), ("CCNC", False), ("CC[NH3+]", False),
    ("", False),
])
def test_amine_on_unbranched_chain_is_read_from_the_structure(parent, expected):
    from orthonym.rules.charged_router import _amine_on_unbranched_chain
    assert _amine_on_unbranched_chain(parent) is expected


# -- the names that were given before keep their spelling
CONTROLS = [
    ("C[B-]1(C)CCCCC1", "1,1-dimethylborinan-1-uide"),
    ("CC[P-]CC", "diethylphosphanide"),
    ("C[P-]C", "dimethylphosphanide"),
    ("C[Si-](C)(C)C", "tetramethyl-λ5-silanide"),
    ("C[Si+](C)C", "trimethylsilylium"),
    ("C[SiH2][Si-](C)C", "1,1,2-trimethyldisilanide"),
    ("CS[O-]", "methane(SO-thioperoxolate)"),
    ("CN(C)[O-]", "[methyl(oxido)amino]methane"),
    ("C(=O)([NH-])N", "carbamoylazanide"),
    ("C[Si]1(C)CCCC1", "1,1-dimethylsilolane"),
    ("CCOS(=O)(=O)[O-]", "ethyl sulfate"),
    ("CN(C)S(=O)[O-]", "[methyl(sulfinato)amino]methane"),
    ("COP(=O)([S-])OC", "[(1-methoxy-1-sulfido-2-oxa-1-phosphaeth-1-en-1-yl)oxy]methane"),
    ("CC[O-]", "ethoxide"),
    ("C[N+](C)(C)CC(=O)[O-]", "(trimethylazaniumyl)acetate"),
    ("CC(=O)[NH-]", "acetylazanide"),
    ("C[N+](C)(C)CCCC(=O)[O-]", "4-(trimethylazaniumyl)butanoate"),
]


@pytest.mark.parametrize("smiles,expected", CONTROLS, ids=[c[0] for c in CONTROLS])
def test_controls_keep_their_names(smiles, expected):
    assert name_best_effort(smiles)["name"] == expected


# -- the bonding number of the boron of a mancude ring, without a JVM,:41089)
@pytest.mark.parametrize("smiles,expected", [
    ("[B-]1(=CC=CC=C1)OC", 4),         # the Kekule drawing of the ring holds its double bond: four bonds, one above three
    ("C[n+]1ccc[b-](F)c1", 4),        # the same atom drawn aromatic
    ("C[B-]1(C)CCCCC1", 4),           # a saturated ring: four neighbours
    ("C[B-](C)(C)C", 4),
    ("CB(C)C", 3),
    ("[B-](=NC#C)(C)C", 4),           # a double bond to a neutral ligand counts twice
    ("C[B-](C)=[N+](C)C", 3),         # the double bond of B-=N+ is the aminoborane's, not a ligand's
    ("C=[B-](F)CN(C)C", 4),           # a methylidene is a ligand...
    ("C=[B-](C)C", 4),
    ("[B-](=S)(C)I", 4),              # the sulfanylidene group counts
])
def test_uide_bonding_number_of_boron(smiles, expected):
    from orthonym.rules.uide_lambda import uide_bonding_number
    mol = Chem.MolFromSmiles(smiles)
    boron = next(a for a in mol.GetAtoms() if a.GetSymbol().upper() == "B")
    assert uide_bonding_number(boron) == expected


# -- the ring ion centre of silicon, without a JVM method (2), the Blue Book)
@pytest.mark.parametrize("smiles,expected", [
    ("C[Si-]1(C)CCCC1", "anion"),         # four single bonds, no hydrogen, in a ring: the 'ide' of the lambda5 hydride
    ("C[Si-](C)(C)C", None),              # not in a ring: ``group14_centres.name_lambda_ide_anion`` names it
    ("C[SiH-]1(C)CCC1", None),            # a hydrogen on the centre
    ("C[Si+]1CCCC1", None),               # the cation has three bonds: an odd number
    ("C[Si]1(C)CCCC1", None),             # neutral
])
def test_ring_ion_centre_of_silicon(smiles, expected):
    from orthonym.rules.lambda_convention import ring_halogen_ion
    mol = Chem.MolFromSmiles(smiles)
    silicon = next(a for a in mol.GetAtoms() if a.GetSymbol() == "Si")
    assert ring_halogen_ion(mol, silicon.GetIdx()) == expected


def test_silicon_is_no_ring_halogen():
    """The halogens' share of the admission (the non-PIN record of a replacement name, the benzo-heterocycle producer) stays theirs."""
    from orthonym.rules.lambda_convention import FUSED_RING_LAMBDA_HALOGENS, RING_ION_LAMBDA_ELEMENTS
    assert "Si" not in FUSED_RING_LAMBDA_HALOGENS and "Si" in RING_ION_LAMBDA_ELEMENTS


# -- the late rescue asks the emitters only of a charged molecule that no producer named
def test_late_rescue_declines_a_neutral_molecule_and_a_class_it_has_not():
    from orthonym.rules.charged_fallback import late_charged_name
    assert late_charged_name("CCO") is None
    assert late_charged_name("not a smiles") is None
    assert late_charged_name("C[N+](C)(C)C") is None
