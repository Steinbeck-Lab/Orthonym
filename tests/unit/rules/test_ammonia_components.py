"""Ammonia beside the ions of Groups 1 and 2 as a mixture component (2026-11-18, PubChem rerun 5 triage item 2b).

A mixture names its components as compounds of their own: "Mixed organic - inorganic adducts"
(the Blue Book), "names are formed by citing the names of individual compounds, connected by long (em) dashes".
'ammonia' is a retained name whose "use as preferred IUPAC names is deferred... thus, no PIN label will be assigned in
names including them" "Retained names",:7954). The adducts cited ammonia only beside the metals of the
AMMINE mechanism (zinc and the bonded metals, ``data/metal_scope.py``); Groups 1 and 2 are admitted to it now, so a
drawing with ammonia beside a sodium or a calcium ion is named. Every name still reads back to the drawing.
"""
import pytest

from orthonym import Orthonym
from orthonym.data import metal_scope
from orthonym.tiers import emit_tier_flags

pytestmark = pytest.mark.opsin_gate


def _tier(name):
    return Orthonym(style="pin", **emit_tier_flags(name))


_NAMED = [
    ("C=C.C(=O)([O-])[O-].N.N.[Na+].[Na+]", "carbonate—ethene—ammonia—sodium(1+) (1/1/2/2)"),     # the last producer
    ("[Na+].[Cl-].N", "ammonia—chloride—sodium(1+) (1/1/1)"),
    ("[Mg+2].N.N.N.N.N.N.[Cl-].[Cl-]", "ammonia—chloride—magnesium(2+) (6/2/1)"),
    ("N.[Na]", "ammonia—sodium (1/1)"),                                                         # the element-set name
    ("CC(=O)O.N.[Na]", "acetic acid—ammonia—sodium (1/1/1)"),
    ("[Zn+2].[Na+].N", "ammonia—sodium(1+)—zinc(2+) (1/1/1)"),                                  # two admitted metals
    ("[K+].CC(=O)[O-].N", "acetate—ammonia—potassium(1+) (1/1/1)"),
    ("[Rb+].[Cl-].N", "ammonia—chloride—rubidium(1+) (1/1/1)"),
    ("[Ca+2].[Cl-].[Cl-].N", "ammonia—calcium(2+)—chloride (1/1/2)"),
    ("[Be+2].[Cl-].[Cl-].N.N", "ammonia—beryllium(2+)—chloride (2/1/2)"),
    ("N.[Li]CCCC", "butyllithium—ammonia (1/1)"),                                               # a bonded drawing
    ("N.C[Mg]Br", "methylmagnesium bromide—ammonia (1/1)"),
    ("N.O[Na]", "ammonia—sodium(1+)—hydroxide (1/1/1)"),       # through the ionic drawing of a covalent one
    ("N.[K].[K]", "ammonia—potassium (1/2)"),
]


@pytest.mark.parametrize("smiles, name", _NAMED)
def test_ammonia_beside_a_group_1_or_2_ion_is_a_component(smiles, name):
    row = _tier("best-effort").name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "systematic_verified")


@pytest.mark.parametrize("tier", ["pin", "valid", "complete"])
def test_the_narrower_tiers_decline_this_drawing(tier):
    namer = Orthonym() if tier == "pin" else _tier(tier)
    assert namer.name_tiered("[Na+].[Cl-].N")["tier"] == "abstain"


def test_an_adduct_of_ions_is_named_at_the_complete_tier_too():
    """The complete tier builds some adducts of ions, and cites ammonia there as it does beside zinc
    ('benzoate—ammonia—zinc(2+) (2/1/1)'); still no PIN."""
    smiles = "N.[Ca+2].[O-]C(=O)c1ccccc1.[O-]C(=O)c1ccccc1"
    assert _tier("complete").name_tiered(smiles)["name"] == "benzoate—ammonia—calcium(2+) (2/1/1)"
    assert Orthonym().name_tiered(smiles)["tier"] == "abstain"


@pytest.mark.parametrize("smiles, name", [
    ("[H+].[Na+].[Cl-]", "chloride—sodium(1+)—hydron (1/1/1)"),     # the hydron words keep their gate
    ("[Th].N", None),                                               # thorium is not admitted
])
def test_names_shipped_before_keep_their_spelling(smiles, name):
    assert _tier("best-effort").name_tiered(smiles)["name"] == name


def test_without_the_mechanism_the_ammonia_declines(monkeypatch):
    """Mutation: sodium taken out of AMMINE again, the mixture declines."""
    scope = metal_scope.METAL_SCOPE["Na"]
    monkeypatch.setitem(metal_scope.METAL_SCOPE, "Na", metal_scope.MetalScope(
        scope.mechanisms - {metal_scope.AMMINE}, scope.valences, scope.complete))
    assert _tier("best-effort").name_tiered("[Na+].[Cl-].N")["tier"] == "abstain"
