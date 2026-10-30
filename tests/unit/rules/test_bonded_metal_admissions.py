"""The metals of PubChem's bonded drawings admitted to every mechanism zinc has (lane B1, 2026-10-30).

``data.metal_scope.BONDED_METAL_VALENCES`` admits Al, Hg, Pd, Pt, Ru, Ti, V and W, each with the charges its cations
may carry. Their bonded drawings are named through their ionic drawing, as a complex by their ligands ("Compounds are
named by citing the names of ligands, including hydrogen atom(s), in alphanumerical order, followed by the name of the
metal", Compounds with at least one metal-carbon single bond), or as a adduct of their components. The
Blue Book gives such a name no PIN ("Coordination nomenclature is the primary nomenclature method used to name
organometallic compounds containing elements of Groups 3 through 12",; Mixed organic - inorganic
adducts), so every name here is systematic_verified at the best-effort tier and ships only when OPSIN reads it back to
the input's full standard InChIKey. The element is spelled as SPELLING asks (the Blue Book, "aluminium not
aluminum").
"""
import pytest

from orthonym.cli import _tiered_namer
from orthonym.data.metal_scope import (ALL_MECHANISMS, BONDED_METAL_VALENCES, METAL_SCOPE, has_valence,
                                       is_complete_metal)
from tests.support.rt_assert import name_best_effort, name_is_rt_exact

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
    ("O.O.O[W](=O)(=O)O", "dihydroxydi(oxo)tungsten—water (1/2)"),
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
    assert "Zr" not in BONDED_METAL_VALENCES and "Sn" not in BONDED_METAL_VALENCES   # the probe named none of them


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
    ("CC(=CC(C)=O)[O-].CC(=CC(C)=O)[O-].[Pt+2]", "platinum bis(2-oxido-4-oxopent-2-ene)"),
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
    """Each row of ``declared_moves_bonded_b1.tsv`` (the other-metals baseline rows the admission moves) is named
    as declared, and a name reads back to the full key: the guards that read the file see only some of its rows."""
    from orthonym import Orthonym
    from orthonym.cli import _emit_tier_flags
    from tests.support.metal_baselines import declared_bonded_metal_moves, identity_view
    namers = {tier: Orthonym(style="pin", **_emit_tier_flags(tier)) for tier in ("pin", "best-effort")}
    moves = declared_bonded_metal_moves()
    assert len(moves) == 15
    for (smiles, tier_req), view in sorted(moves.items()):
        now = namers[tier_req].name_tiered(smiles)
        assert identity_view(now) == view, (smiles, tier_req)
        if now["name"] and now["tier"] != "abstain":
            assert name_is_rt_exact(now["name"], smiles), (smiles, tier_req)
