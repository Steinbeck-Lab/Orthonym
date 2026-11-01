"""The metals of PubChem's bonded drawings admitted to every mechanism zinc has (lanes B1 and B2, 2026-10-30).

``data.metal_scope.BONDED_METAL_VALENCES`` admits Al, Hg, Pd, Pt, Ru, Ti, V and W, each with the charges its cations
may carry. Their bonded drawings are named through their ionic drawing, as a complex by their ligands ("Compounds are
named by citing the names of ligands, including hydrogen atom(s), in alphanumerical order, followed by the name of the
metal", Compounds with at least one metal-carbon single bond), or as a adduct of their components. The
Blue Book gives such a name no PIN ("Coordination nomenclature is the primary nomenclature method used to name
organometallic compounds containing elements of Groups 3 through 12",; Mixed organic - inorganic
adducts), so every name here is systematic_verified at the best-effort tier and ships only when OPSIN reads it back to
the input's full standard InChIKey. The element is spelled as SPELLING asks (the Blue Book, "aluminium not
aluminum").

Lane B2 admits 38 more metals (``MORE_BONDED_METAL_VALENCES``) to the same mechanisms, tin and lead to
``PARENT_HYDRIDE_IONS`` ('tributylstannylium', Cationic centers in parent hydrides, Specific
method, the Blue Book:
"stannane, SnH4, and plumbane, PbH4... by replacing the 'ane' ending... by the suffix 'ylium'"), and the metals of
``BONDED_NAMES`` to the organic ligands joined by a double or a triple bond ('methylidene', 'ethylidyne', 'phenylimino':
 Definitions of terms, "names for substituent groups as defined in Section can also be used as ligand
names") and to the halides beside a multiply bonded ligand ('dichloro(methylidene)titanium').
"""
import pytest

from orthonym.cli import _tiered_namer
from orthonym.data.metal_scope import (ALL_MECHANISMS, BONDED_METAL_VALENCES, METAL_SCOPE,
                                       MORE_BONDED_METAL_VALENCES, has_valence, is_complete_metal)
from tests.support.rt_assert import name_best_effort, name_is_rt_exact, smiles_without_dative_bonds

pytestmark = pytest.mark.opsin_gate

ROWS = [
    ("[H-].CCC[Al+]C1=CC=CC=C1", "phenyl(propyl)aluminium(1+)—hydride (1/1)"),
    ("CCCC[Hg+]", "butylmercury(1+)"),
    ("N[Hg]Cl", "mercury(II) azanide chloride"),
    ("Cl[Pd-2](Cl)(Br)Br", "bromide—chloride—palladium(2+) (2/2/1)"),
    ("C1CNCCC1=O.N.Cl[Pt]Cl", "piperidin-4-one—ammonia—chloride—platinum(2+) (1/1/2/1)"),
    ("C1=CN[C-]=C1.F[Ru]F", "1H-pyrrol-2-ide—fluoride—ruthenium(2+) (1/2/1)"),
    ("N.N.N.N.[NH2-].[NH2-].[Ru+2]", "ammonia—azanide—ruthenium(2+) (4/2/1)"),
    ("Cl[Ti+](Cl)Br", "bromide—chloride—titanium(4+) (1/2/1)"),
    ("C(C(Cl)Cl)O.O=[V+2]", "2,2-dichloroethan-1-ol—oxovanadium(2+) (1/1)"),
    ("O.O.O[W](=O)(=O)O", "dihydroxydioxotungsten—water (1/2)"),
    # uranium, admitted on 2026-10-31 (MORE_BONDED_METAL_VALENCES)
    ("O=[U](=O)=O", "trioxouranium"),
    ("F[U](F)(F)(F)(F)F", "uranium(VI) fluoride"),
    ("F[U](F)(F)(F)F", "uranium(V) fluoride"),
    ("C[U](C)(C)C", "tetramethyluranium"),
    ("C=[U+2]=O", "methylidene(oxo)uranium(2+)"),
    ("[N+](=O)([O-])[O-].[N+](=O)([O-])[O-].O.O.O.O.O=[U+2]=O", "dioxouranium(2+) dinitrate tetrahydrate"),
]


def _opsin_reads(name) -> bool:
    """OPSIN main reads every name below; the release 2.9.0 does not read a hydride component of a mixture, and
    there such a name fails closed (the read-back cannot confirm it)."""
    from orthonym.namer import _validity_gate_name_to_smiles
    return bool(_validity_gate_name_to_smiles(name))


@pytest.mark.parametrize("smiles,expected", ROWS)
def test_best_effort_name(smiles, expected):
    res = name_best_effort(smiles)
    if not _opsin_reads(expected):
        assert res["name"] is None, res          # no name, never a wrong one
        return
    assert res["name"] == expected and res["tier"] == "systematic_verified" and not res["is_pin"], res
    assert name_is_rt_exact(res["name"], smiles), res


@pytest.mark.parametrize("smiles,_expected", ROWS)
def test_pin_tier_ships_no_pin(smiles, _expected):
    assert _tiered_namer("pin", "pin").name_tiered(smiles)["tier"] != "pin_verified"


def test_trimethylaluminium_is_spelled_as_the_book_asks():
    """ SPELLING (the Blue Book): "aluminium not aluminum"; OPSIN reads both spellings to one structure."""
    res = name_best_effort("C[Al](C)C")
    assert res["name"] == "trimethylaluminium", res
    assert name_is_rt_exact(res["name"], "C[Al](C)C")


def test_the_salt_of_an_acid_takes_the_label_zinc_takes():
    """A salt of an acid keeps pin_verified by decision of the maintainers (the contributor guide "Scope"): mercury as zinc."""
    for smiles, expected in (("[H-].CC(=O)[O-].[Zn+2]", "zinc acetate hydride"),
                             ("[H-].CC(=O)[O-].[Hg+2]", "mercury acetate hydride")):
        res = name_best_effort(smiles)
        assert (res["name"], res["tier"]) == (expected, "pin_verified"), res
        assert name_is_rt_exact(res["name"], smiles), res


def test_the_table_admits_the_eight_metals_to_every_mechanism():
    assert sorted(BONDED_METAL_VALENCES) == ["Al", "Hg", "Pd", "Pt", "Ru", "Ti", "V", "W"]
    assert METAL_SCOPE["Zn"].mechanisms == ALL_MECHANISMS
    for symbol, valences in BONDED_METAL_VALENCES.items():
        assert METAL_SCOPE[symbol].mechanisms == ALL_MECHANISMS, symbol
        assert METAL_SCOPE[symbol].valences == frozenset(valences), symbol
        assert is_complete_metal(symbol) is False, symbol      # eta and multiply bonded ligands have no producer yet
    assert has_valence("Pt", 4) is True and has_valence("Pt", 3) is False
    # the metals of lane B2 are in the table of their own and take every mechanism but ELEMENT where an ion has a
    # retained name; tin and lead have their own entry (``PARENT_HYDRIDE_IONS``): see ``test_metal_scope.py``
    assert "Zr" not in BONDED_METAL_VALENCES and "Sn" not in BONDED_METAL_VALENCES
    assert "Zr" in MORE_BONDED_METAL_VALENCES and "Sn" not in MORE_BONDED_METAL_VALENCES


#: rows that the guards of the start commit listed as "not admitted" and that the admission moves: the name now
MOVED_ROWS = [
    ("[Hg]", "mercury"),
    ("[Hg+2]", "mercury(2+)"),
    ("[Pt]", "platinum"),
    ("CCS[Hg]SCC", "mercury diethanethiolate"),                          # was the refusal 'mercury compound...'
    # the drawings of test_group12_inorganic_ligands.OTHER_METALS that the admission moves (BONDED_NAMES)
    ("CC[Hg]O", "ethyl(hydroxy)mercury"),
    ("O=[Hg]", "oxomercury"),
    ("[Pt]=O", "oxoplatinum"),
    ("[Ti]=O", "oxotitanium"),
    ("[O-][n+]1ccccc1[S-].[O-][n+]1ccccc1[S-].[Pt+2]", "platinum bis(1-oxido-2-sulfidopyridin-1-ium)"),
    ("CC(=CC(C)=O)[O-].CC(=CC(C)=O)[O-].[Pt+2]", "platinum bis(4-oxopent-2-en-2-olate)"),
    # the ionic drawing wins a tie of ranks (``assembly.metal_offer.name_with_metal_offer``), as for zinc:
    # 'ethanol—chloride—zinc(2+) (1/2/1)' for Cl[Zn]Cl.CCO
    ("C1CC=CCCC=C1.Cl[Pd]Cl", "cycloocta-1,5-diene—chloride—palladium(2+) (1/2/1)"),
]


@pytest.mark.parametrize("smiles,expected", MOVED_ROWS)
def test_the_rows_the_admission_moves(smiles, expected):
    res = name_best_effort(smiles)
    assert (res["name"], res["tier"]) == (expected, "systematic_verified"), res
    assert name_is_rt_exact(res["name"], smiles), res


def test_a_mercury_cation_beside_its_anion_is_connected():
    from orthonym.assembly import metal_offer
    assert metal_offer.metal_connected_spelling_and_order("C[Hg+].[Cl-]") == ("[CH3][Hg][Cl]", (0, 1, 2))


def test_a_dative_platinum_ammine_is_named_through_its_split_drawing():
    from tests.support.rt_assert import smiles_without_dative_bonds
    dative = "Cl[Pt](Cl)(<-N)<-N"
    res = name_best_effort(dative)
    assert (res["name"], res["tier"]) == ("ammonia—chloride—platinum(2+) (2/2/1)", "systematic_verified"), res
    assert name_is_rt_exact(res["name"], smiles_without_dative_bonds(dative)), res


@pytest.mark.parametrize("smiles", ["[Hg]", "[Hg+2]", "[Pt]"])
def test_the_atoms_and_ions_decline_at_the_default_tier(smiles):
    """No PIN for an atom or an ion named by its element name (ELEMENT,: the default tier declines, as for
    zinc."""
    res = _tiered_namer("pin", "pin").name_tiered(smiles)
    assert (res["tier"], res["limit_code"], res["is_pin"]) == ("abstain", "NO_VERIFIED_PIN", False), res


def test_every_declared_move_holds():
    """Each row of ``declared_moves_bonded_b1.tsv`` and ``declared_moves_bonded_b2.tsv`` (the other-metals baseline rows
    that lanes B1 and B2 move) is named as declared, and a name reads back to the full key: the guards that read the
    files see only some of their rows."""
    from orthonym import Orthonym
    from orthonym.cli import _emit_tier_flags
    from tests.support.metal_baselines import (DECLARED_BONDED_MOVE_FILES, declared_bonded_metal_moves, identity_view)
    namers = {tier: Orthonym(style="pin", **_emit_tier_flags(tier)) for tier in ("pin", "best-effort")}
    moves = declared_bonded_metal_moves()
    assert len(DECLARED_BONDED_MOVE_FILES) == 2
    assert len(moves) == 16 + 24                      # the rows of lane B1 and of lane B2 (three of the acetylacetonates' PIN rows: olate)
    for (smiles, tier_req), view in sorted(moves.items()):
        now = namers[tier_req].name_tiered(smiles)
        assert identity_view(now) == view, (smiles, tier_req)
        if now["name"] and now["tier"] != "abstain":
            assert name_is_rt_exact(now["name"], smiles), (smiles, tier_req)


def test_no_declared_move_lowers_a_pin_label():
    """A PIN label is never lowered by an admission: a declared move never turns a ``pin_verified`` row of the baseline
    into a lower one (the baseline rows with that label keep it; the respelled ones are 'iron diacetate' and the like,
    ``declared_respellings_pinc.tsv``, decided on main before these lanes)."""
    from tests.support.metal_baselines import baseline_path, declared_bonded_metal_moves, identity_view, read_rows
    moves = declared_bonded_metal_moves()
    for row in read_rows(baseline_path("other")):
        key = (row["smiles"], row["tier_req"])
        if key in moves and row["tier"] == "pin_verified":
            assert moves[key][1] == "pin_verified" and moves[key][2] is True, (key, moves[key])


# ---- lane B2: the 38 metals, the cations of tin and lead, the multiply bonded ligands (2026-10-30)

def _best_effort_row_checks(smiles, expected):
    """The best-effort name of ``smiles`` is ``expected``, below the PIN, read back to the full key of the drawing
    (a dative drawing against the drawing without its dative bonds); the default tier ships no PIN for it."""
    from orthonym.validation.halido_twin import halido_twin
    res = name_best_effort(smiles)
    read_as = halido_twin(expected)     # a halide ligand word no OPSIN reads is read through its 'chloro' twin (R2)
    if not _opsin_reads(read_as):
        assert res["name"] is None, res          # no name, never a wrong one (OPSIN 2.9.0 cannot read it)
        return
    assert res["name"] == expected and res["tier"] == "systematic_verified" and not res["is_pin"], res
    assert name_is_rt_exact(read_as, smiles_without_dative_bonds(smiles) or smiles), res
    assert _tiered_namer("pin", "pin").name_tiered(smiles)["tier"] != "pin_verified"


#: organic groups joined to the metal by a double or a triple bond, and halides beside a multiply bonded ligand: the
#: ligand prefixes are those of ('methylidene', 'ethylidyne'; 'imino',, cited as ligands in
#: alphanumerical order before the metal Compounds with at least one metal-carbon single bond); the halide is
#: 'chlorido', the Blue Book: an anion name ending in 'ide' ends in 'ido' as a ligand; the active OPSIN
#: reads the name through its 'chloro' twin; LEADS2 fix round R2, advisor ruling 2026-11-01, changed asserted values: the
#: rows were 'dichloro(...)') and the terminal =O is 'oxo'. The name states the bond order: the
#: single-bonded name of the same atoms ('methyl') would read back as another structure
MULTIPLY_BONDED_ROWS = [
    ("C=[Ti](Cl)Cl", "dichlorido(methylidene)titanium"),
    ("CC#[W](Cl)(Cl)Cl", "trichlorido(ethylidyne)tungsten"),
    ("Cl[Ti](Cl)=Nc1ccccc1", "dichlorido(phenylimino)titanium"),
    ("O=[V](Cl)Cl", "dichlorido(oxo)vanadium"),
    ("C=[Pd]", "methylidenepalladium"),
    ("C=[Zn]", "methylidenezinc"),
]


@pytest.mark.parametrize("smiles,expected", MULTIPLY_BONDED_ROWS)
def test_a_multiply_bonded_ligand_is_named_with_the_order_of_its_bond(smiles, expected):
    _best_effort_row_checks(smiles, expected)
    assert expected.count("ylidene") + expected.count("ylidyne") + expected.count("imino") + expected.count("oxo") == 1
    pin = _tiered_namer("pin", "pin").name_tiered(smiles)
    assert (pin["tier"], pin["is_pin"]) == ("abstain", False) and pin["limit_code"] == "NO_VERIFIED_PIN", pin


def test_a_halide_is_cited_as_chlorido_beside_a_multiply_bonded_ligand_as_in_the_books_example():
    """The ionic drawing of such a complex breaks only the single bonds, so the charge it gives the metal is not the
    metal's: the halide is cited as a ligand beside a multiply bonded one, with the ligand word of the book's own example,
    'trichlorido(methyl)titanium' Compounds with at least one metal-carbon single bond), and not as 'chloro'.
    LEADS2 fix round R2 (advisor ruling 2026-11-01): 'chlorido' is the only book form on a Group 3-12 metal,
    the Blue Book); no coordination name is a PIN,:39753): the label ceiling is systematic_verified, and
    the PIN tier declines. The names are read through their 'chloro' twin."""
    plain = name_best_effort("C[Ti](Cl)(Cl)Cl")
    assert (plain["name"], plain["tier"], plain["is_pin"]) == ("trichlorido(methyl)titanium", "systematic_verified",
                                                              False), plain
    multiple = name_best_effort("C=[Ti](Cl)Cl")
    assert (multiple["name"], multiple["tier"], multiple["is_pin"]) == (
        "dichlorido(methylidene)titanium", "systematic_verified", False), multiple
    assert name_is_rt_exact("dichloro(methylidene)titanium", "C=[Ti](Cl)Cl")           # the twin the name is read through
    assert _tiered_namer("pin", "pin").name_tiered("C=[Ti](Cl)Cl")["tier"] == "abstain"


def test_a_multiplied_simple_ligand_of_a_metal_takes_no_enclosing_marks():
    """ (:39789) 'trichlorido(methanido)titanium'; a metal is no mononuclear parent hydride, so
    (:7272, 'tert-butyldi(methyl)phosphane') does not reach it: 'benzylidenedichloridoruthenium', not
    'benzylidenedi(chlorido)ruthenium' (advisor ruling 2026-11-01, R4); bis/tris only for a compound ligand
    ,:7232). An organyl ligand keeps its marks where the bare name would read as another structure
    ('(chloromethyl)silane [PIN; the name chloro(methyl)silane would describe ClSiH2CH3]',; OPSIN reads
    'hydroxydimethylzinc(1-)' as hydroxymethyl(methyl)zinc), and 'bis(sulfanyl)' keeps its own (:7140; the rows of ``test_group12_inorganic_ligands``)."""
    assert name_best_effort("Cl[Ru](Cl)=Cc1ccccc1")["name"] == "benzylidenedichloridoruthenium"
    assert name_best_effort("Cl[Mo](Cl)(=O)=O")["name"] == "dichloridodioxomolybdenum"
    assert name_is_rt_exact("benzylidenedichlororuthenium", "Cl[Ru](Cl)=Cc1ccccc1")
    assert name_best_effort("[Zn-](O)(C)C")["name"] == "hydroxydi(methyl)zinc(1-)"


def test_one_component_twice_has_no_pin_and_the_pin_tier_declines_it():
    """'oxozinc oxozinc' (R3, advisor ruling 2026-11-01): the book has no form for one component twice,:4667;
    ,:3765 uses distinct components with ratios): systematic_verified at best, the PIN tier declines."""
    for smiles in ("O=[Zn].O=[Zn]", "CCO.CCO"):
        row = name_best_effort(smiles)
        assert row["tier"] == "systematic_verified" and row["is_pin"] is False, row
        assert _tiered_namer("pin", "pin").name_tiered(smiles)["tier"] == "abstain"


#: the cations of tin and lead on their parent hydrides: PINs ('triphenylsilylium (PIN)',:41557, is the Group 14 example
#: of the paragraph that names 'stannane' and 'plumbane' too), at every tier
@pytest.mark.parametrize("smiles,expected", [
    ("CCCC[Sn+](CCCC)CCCC", "tributylstannylium"),
    ("CC[Pb+](CC)CC", "triethylplumbylium"),
])
def test_a_tin_or_lead_cation_is_a_ylium_with_a_pin_label(smiles, expected):
    res = name_best_effort(smiles)
    assert (res["name"], res["tier"], res["is_pin"]) == (expected, "pin_verified", True), res
    assert name_is_rt_exact(expected, smiles)
    pin = _tiered_namer("pin", "pin").name_tiered(smiles)
    assert (pin["name"], pin["tier"], pin["is_pin"]) == (expected, "pin_verified", True), pin


def test_the_organotin_cation_beside_its_anion_is_named_through_its_bonded_drawing():
    """``CONNECT_ANION``: the bonded drawing of tributyltin chloride is a substituted stannane; the Blue Book
    gives the compound no PIN of this name here, so the label is systematic_verified and the default tier declines."""
    smiles = "CCCC[Sn+](CCCC)CCCC.[Cl-]"
    res = name_best_effort(smiles)
    assert (res["name"], res["tier"], res["is_pin"]) == ("tributyl(chloro)stannane", "systematic_verified", False), res
    assert name_is_rt_exact(res["name"], smiles)
    pin = _tiered_namer("pin", "pin").name_tiered(smiles)
    assert pin["tier"] == "abstain" and pin["limit_code"] == "NO_VERIFIED_PIN", pin


def test_a_dibutyltin_dication_with_two_acetates_is_named_and_read_back():
    smiles = "CCCC[Sn+2]CCCC.CC(=O)[O-].CC(=O)[O-]"
    res = name_best_effort(smiles)
    assert (res["name"], res["tier"], res["is_pin"]) == (
        "1-{[(acetyloxy)di(butyl)stannyl]oxy}-1-oxoethane", "systematic_verified", False), res
    assert name_is_rt_exact(res["name"], smiles)
    assert _tiered_namer("pin", "pin").name_tiered(smiles)["tier"] != "pin_verified"


@pytest.mark.parametrize("smiles", ["C[Sn+]C", "C[Sn+2]C", "[Sn+]", "C1CC[Sn+]C1", "[Pb+]"])
def test_a_tin_or_lead_cation_that_is_no_ylium_is_not_named_as_one(smiles):
    for res in (name_best_effort(smiles), _tiered_namer("pin", "pin").name_tiered(smiles)):
        assert not (res["name"] or "").endswith("ylium"), res
        assert res["tier"] != "pin_verified", res


#: one or more rows of the metals of lane B2, with the names of their mechanisms: the atom and the ion by the element
#: name (ELEMENT), a drawing with M-X bonds through its ionic drawing (DISCONNECT), a thiolate, a metal alkyl
#: (BONDED_NAMES), an adduct with an ion of an organic ligand, an ammine and the like (the other files of the lane)
MORE_METAL_ROWS = [
    ("[Zr+4]", "zirconium(4+)"),
    ("[Hf+4]", "hafnium(4+)"),
    ("[Zr]", "zirconium"),
    ("[La+3]", "lanthanum(3+)"),
    ("Cl[Cd]Cl", "cadmium dichloride"),
    ("CCS[Cd]SCC", "cadmium diethanethiolate"),
    ("CCS[Ni]SCC", "nickel diethanethiolate"),
    ("C[Cd]C", "dimethylcadmium"),
    ("C[Ga](C)C", "trimethylgallane"),
    ("C[In](C)C", "trimethylindigane"),
    ("CC(=O)[O-].[Tl+]", "thallium acetate"),
    ("CC[Tl+2]CC", "diethylthallium(2+)"),
    ("C1CC=CCCC=C1.Cl[Ni]Cl", "cycloocta-1,5-diene\u2014chloride\u2014nickel(2+) (1/2/1)"),
    ("CC(=CC(C)=O)[O-].CC(=CC(C)=O)[O-].[Cu+2]", "copper bis(4-oxopent-2-en-2-olate)"),
    ("[O-][n+]1ccccc1[S-].[O-][n+]1ccccc1[S-].[Cu+2]", "copper bis(1-oxido-2-sulfidopyridin-1-ium)"),
    ("C[Cu+].[Cl-]", "methylcopper(1+)\u2014chloride (1/1)"),             # OPSIN 2.9.0 cannot read it: no name there
    # a dative drawing is named through the drawing without its dative bonds (DATIVE)
    ("Cl[Ni](Cl)(<-N)<-N", "ammonia\u2014chloride\u2014nickel(2+) (2/2/1)"),
    ("N->[Ni+2](<-N)(<-N)<-N", "ammonia\u2014nickel(2+) (4/1)"),
]


@pytest.mark.parametrize("smiles,expected", MORE_METAL_ROWS)
def test_the_metals_of_lane_b2_are_named_below_the_pin_and_read_back(smiles, expected):
    _best_effort_row_checks(smiles, expected)


#: the PIN-labelled rows that the admission must keep: a salt of an acid, an alcohol or a thiol and a carbon-free salt
#: keep ``pin_verified`` by decision of the maintainers (the contributor guide "Scope"), and the ion of a metal whose ions have a
#: retained name keeps that name (``RETAINED_ION_METALS`` are not admitted to ELEMENT): none of them is lowered
PIN_KEPT_ROWS = [
    ("CC(=O)[O-].CC(=O)[O-].CC(=O)[O-].[La+3]", "lanthanum triacetate"),
    ("CC(=O)[O-].CC(=O)[O-].[Cd+2]", "cadmium diacetate"),
    ("CC(=O)[O-].CC(=O)[O-].CC(=O)[O-].[Gd+3]", "gadolinium triacetate"),
    ("CC(=O)[O-].CC(=O)[O-].[Co+2]", "cobalt diacetate"),
    ("CC(=O)[O-].[Ag+]", "silver acetate"),
    ("[Cl-].[Cl-].[Cd+2]", "cadmium dichloride"),
    ("[Cl-].[Ag+]", "silver chloride"),
    ("[Mo+5].[Cl-].[Cl-].[Cl-].[Cl-].[Cl-]", "molybdenum(V) chloride"),
    ("[Cu+2]", "copper(II)"),
    ("[Fe+3]", "iron(III)"),
    ("[Au+3]", "gold(III)"),
]


@pytest.mark.parametrize("smiles,expected", PIN_KEPT_ROWS)
def test_the_admission_lowers_no_pin_label(smiles, expected):
    for res in (name_best_effort(smiles), _tiered_namer("pin", "pin").name_tiered(smiles)):
        assert (res["name"], res["tier"], res["is_pin"]) == (expected, "pin_verified", True), res
    assert name_is_rt_exact(expected, smiles)


def test_a_carbonyl_drawn_with_double_bonds_is_carbonyl():
    """ Definitions of terms (the Blue Book): "Exceptionally, the molecules H2O, NH3, CO, and NO are
    named 'aqua', 'ammine', 'carbonyl', and 'nitrosyl' when used as ligands" -- never the prefix 'formylidene'."""
    smiles = "O=C=[Fe](=C=O)(=C=O)(=C=O)=C=O"
    res = name_best_effort(smiles)
    assert (res["name"], res["tier"]) == ("pentacarbonyliron", "systematic_verified"), res
    assert name_is_rt_exact(res["name"], smiles), res


def test_caesium_is_spelled_as_the_book_asks():
    """ SPELLING (the Blue Book): "caesium not cesium"; a carbon-free salt keeps its label."""
    res = name_best_effort("[Cs+].[Cl-]")
    assert (res["name"], res["tier"]) == ("caesium chloride", "pin_verified"), res
    assert name_is_rt_exact(res["name"], "[Cs+].[Cl-]"), res


@pytest.mark.parametrize("smiles", [
    "C[Sn+](C)C.[O-]c1ccccn1",                 # pyridin-2-olate: the charge also sits on the ring nitrogen
    "C[Sn+](C)C.O=c1cccc[n-]1",                # the same anion drawn on the nitrogen
    "CCCC[Sn+](CCCC)CCCC.c1ccc2[n-]nnc2c1",    # benzotriazol-1-ide and -2-ide are one anion
    "C[Sn+](C)C.[S-]C#N",                      # thiocyanate: S or N
])
def test_an_ambident_anion_is_not_connected_to_a_metal(smiles):
    """The drawn position of the charge of an ambident anion is no structure: a bond to one or the other donor gives
    two constitutional isomers with one standard InChIKey, so the read-back cannot tell them apart and the drawing is
    not connected (``perception.metals._charge_stays_on_the_donor``)."""
    from rdkit import Chem
    from orthonym.perception.metals import connect_anion_to_organometal_cation_sites
    assert connect_anion_to_organometal_cation_sites(Chem.MolFromSmiles(smiles)) is None
    name = name_best_effort(smiles)["name"]
    assert name is None or not any(word in name for word in ("stannane", "stannyl)")), name


@pytest.mark.parametrize("smiles", ["CC(=O)[O-].[Zn+]C", "C[Sn+](C)C.[O-]c1ccccc1", "C[Sn+](C)C.C[O-]"])
def test_an_anion_whose_charge_has_one_place_is_connected(smiles):
    """A carboxylate's two oxygens are one donor (one canonical structure); a phenolate's charge on ring carbons is no
    other donor."""
    from rdkit import Chem
    from orthonym.perception.metals import connect_anion_to_organometal_cation_sites
    assert connect_anion_to_organometal_cation_sites(Chem.MolFromSmiles(smiles)) is not None
