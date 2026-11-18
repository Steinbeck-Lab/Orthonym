"""A one-atom hydride anion of a metalloid as a mixture component (2026-11-18, PubChem rerun 5 triage item 2b).

A mixture names its components as compounds of their own: "Mixed organic - inorganic adducts"
(the Blue Book), "names are formed by citing the names of individual compounds, connected by long (em) dashes".
'[BH4-]' is an anion of the parent hydride borane: "ANIONS FORMED BY ADDITION OF HYDRIDE IONS"
(:41089-41097), "the suffix 'uide' describes an anion formally derived by adding a hydride ion" ('boranuide');
'[SiH3-]' loses a hydron from silane, (:40902), '-ide' ('silanide'). The adduct producer took a one-atom
metalloid fragment for a metal atom and asked for its element name, which a hydride anion has none of, so the whole
mixture was declined. Only such an anion of B, Si, Ge, As, Sb or Te goes to the component pipeline now; a metal
hydride ('[ZnH2]') and the Al/Ga hydride anions keep today's path. A carbon-free component name is no PIN.
"""
import pytest

from orthonym import Orthonym
from orthonym.rules import adducts
from orthonym.tiers import emit_tier_flags

pytestmark = pytest.mark.opsin_gate


def _tier(name):
    return Orthonym(style="pin", **emit_tier_flags(name))


_NAMED = [
    ("[BH4-].CCO", "ethanol—boranuide (1/1)"),
    ("[2H][B-]([2H])([2H])[2H].CCO", "ethanol—(2H4)boranuide (1/1)"),
    ("[10BH4-].CCO", "ethanol—(10B)boranuide (1/1)"),
    ("[BH4-].[SiH4]", "boranuide—silane (1/1)"),
    ("[SiH3-].c1ccccc1", "benzene—silanide (1/1)"),
]


@pytest.mark.parametrize("smiles, name", _NAMED)
def test_a_metalloid_hydride_anion_is_a_component(smiles, name):
    row = _tier("best-effort").name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "systematic_verified")


@pytest.mark.parametrize("tier", ["pin", "valid", "complete"])
def test_the_narrower_tiers_decline_this_drawing(tier):
    namer = Orthonym() if tier == "pin" else _tier(tier)
    assert namer.name_tiered("[BH4-].CCO")["tier"] == "abstain"


def test_an_adduct_of_ions_is_named_at_the_complete_tier_too():
    """The complete tier builds an adduct of ions as well (as it names 'benzoate—ammonia—zinc(2+) (2/1/1)'), so a
    hydride anion beside ions is named there too; still no PIN."""
    row = _tier("complete").name_tiered("[BH4-].[Na+].NC(Cc1ccccc1)C(=O)O")
    assert (row["name"], row["tier"]) == ("2-amino-3-phenylpropanoic acid—boranuide—sodium(1+) (1/1/1)",
                                          "systematic_verified")


def test_a_multiplied_uide_anion_takes_bis():
    """ (d) (the Blue Book): a component name "ambiguous if multiplied by... 'di', 'tri'" takes
    'bis': 'diboranuide' reads as an anion of diborane, so the salt is 'magnesium bis(boranuide)', as
    'trimagnesium bis(phosphanetriide)' (``salts._ion_needs_enclosing_multiplier``)."""
    from orthonym.rules.salts import _ion_needs_enclosing_multiplier
    assert _ion_needs_enclosing_multiplier("boranuide") is True
    row = _tier("best-effort").name_tiered("[BH4-].[BH4-].[Mg+2]")
    assert (row["name"], row["tier"]) == ("magnesium bis(boranuide)", "systematic_verified")


@pytest.mark.parametrize("smiles, expected", [
    ("[BH4-]", True), ("[2H][B-]([2H])([2H])[2H]", True), ("[SiH3-]", True),
    ("[BH3]", False), ("[B-]", False), ("[ZnH3-]", False),
    ("[AlH4-]", True), ("[SnH3-]", True),          # admitted to PARENT_HYDRIDE_IONS (user decision 2026-11-18)
])
def test_the_hydride_anion_predicate(smiles, expected):
    """Explicit deuterium neighbours count as hydrogens (``GetTotalNumHs(includeNeighbors=True)``)."""
    from rdkit import Chem
    assert adducts._is_metalloid_hydride_anion(Chem.MolFromSmiles(smiles)) is expected


@pytest.mark.parametrize("smiles, name", [
    ("[BH4-]", "boranuide"),
    ("[BH4-].[Na+]", "sodium boranuide"),                # the salt producer
    ("[BH3].CCO", "ethanol—borane (1/1)"),               # no anion
    ("[ZnH2].CCO", None),                                # a metal hydride stays unnamed
    ("[MgH2].CCO", None),
])
def test_names_shipped_before_keep_their_spelling(smiles, name):
    assert _tier("best-effort").name_tiered(smiles)["name"] == name


@pytest.mark.parametrize("smiles", ["[BH4-].CCO", "[SiH3-].c1ccccc1"])
def test_the_metal_atom_name_is_what_declined_it(monkeypatch, smiles):
    """Mutation: with the hydride anion taken for a metal atom again, the mixture declines."""
    monkeypatch.setattr(adducts, "_is_metalloid_hydride_anion", lambda frag_mol: False)
    assert _tier("best-effort").name_tiered(smiles)["tier"] == "abstain"


def test_a_multiplied_uide_anion_beside_an_organic_cation_keeps_the_salt_label():
    """The 2:1 salt takes the label its 1:1 sibling ships with ('N,N,N-trimethylmethanaminium boranuide',
    pin_verified on the base code): a salt of an organic cation, as 'bis(methanaminium) sulfate (PIN)',
    the Blue Book)."""
    row = Orthonym().name_tiered("[BH4-].[BH4-].C[N+](C)(C)CC[N+](C)(C)C")
    assert (row["name"], row["tier"]) == (
        "N1,N1,N1,N2,N2,N2-hexamethylethane-1,2-bis(aminium) bis(boranuide)", "pin_verified")


def test_a_multiplied_uide_anion_of_another_hydride_takes_bis_too():
    """'calcium(2+)—phosphanuide (1/2)' (the adduct form shipped before) is now the salt 'calcium bis(phosphanuide)'."""
    row = _tier("best-effort").name_tiered("[PH4-].[PH4-].[Ca+2]")
    assert (row["name"], row["tier"]) == ("calcium bis(phosphanuide)", "systematic_verified")


# The hydride anions of the metals ``data/metal_scope.py`` admits to PARENT_HYDRIDE_IONS (Al, Ga, In, Tl, Sn, Pb) are
# named on the parent hydride as well: "ANIONS FORMED BY ADDITION OF HYDRIDE IONS" (the Blue Book-41097,
# 'lithium tert-butylbis(2-methylpropyl)alumanuide',:41118). User decision 2026-11-18: 'alumanuide' replaces the
# 'aluminohydride' of the trivial table (not an IUPAC name). A compound of aluminium has no PIN lists no Al).
@pytest.mark.parametrize("smiles, name", [
    ("[AlH4-]", "alumanuide"),
    ("[Li+].[AlH4-]", "lithium alumanuide"),
    ("[Na+].[AlH4-].[Na+].[AlH4-]", "disodium bis(alumanuide)"),     # was 'disodium dialuminohydride'
    ("[Mg+2].[AlH4-].[AlH4-]", "magnesium bis(alumanuide)"),
    ("[GaH4-]", "gallanuide"),
    ("C[AlH3-]", "methylalumanuide"),
    ("[AlH4-].CCO", "ethanol—alumanuide (1/1)"),
    ("[GaH4-].c1ccccc1", "benzene—gallanuide (1/1)"),
    ("[Li+].CC(C)(C)[AlH-](CC(C)C)CC(C)C", "lithium tert-butylbis(2-methylpropyl)alumanuide"),   # the book,:41118
    ("[Mg+2].C[AlH3-].C[AlH3-]", "magnesium bis(methylalumanuide)"),
])
def test_a_metal_hydride_anion_is_named_on_its_parent_hydride(smiles, name):
    from tests.support.rt_assert import name_is_rt_exact
    row = _tier("best-effort").name_tiered(smiles)
    assert (row["name"], row["tier"], row["is_pin"]) == (name, "systematic_verified", False), row
    assert name_is_rt_exact(name, smiles)
    assert Orthonym().name_tiered(smiles)["tier"] == "abstain"


def test_a_metal_not_admitted_keeps_no_hydride_anion_name(monkeypatch):
    """Mutation: aluminium taken out of PARENT_HYDRIDE_IONS, its hydride anion has no -uide name again."""
    from orthonym.data import metal_scope
    scope = metal_scope.METAL_SCOPE["Al"]
    monkeypatch.setitem(metal_scope.METAL_SCOPE, "Al", metal_scope.MetalScope(
        scope.mechanisms - {metal_scope.PARENT_HYDRIDE_IONS}, scope.valences, scope.complete))
    assert _tier("best-effort").name_tiered("[AlH4-]")["name"] == "aluminohydride"     # the trivial table again


@pytest.mark.parametrize("smiles, name", [
    ("[Al-](Cl)(Cl)(Cl)Cl", "tetrachloroaluminate"),     # a halide complex keeps its additive name
    ("[Ga-](Cl)(Cl)(Cl)Cl", "tetrachlorogallate"),
    ("[AlH2-]", "dihydridoaluminium(1-)"),               # not one bond above the standard bonding number
])
def test_other_anions_of_the_metals_keep_their_names(smiles, name):
    assert _tier("best-effort").name_tiered(smiles)["name"] == name


def test_an_organotin_uide_carries_the_label_its_silicon_sibling_does():
    """ (:41097): "Method (1) leads to preferred IUPAC names"; tin is an element of, and the engine labels
    'dichlorotri(methyl)silanuide' the same way."""
    row = Orthonym().name_tiered("C[Sn-](C)(C)(C)C")
    assert (row["name"], row["tier"]) == ("pentamethylstannanuide", "pin_verified")


def test_a_multiplied_substituted_uide_anion_takes_bis():
    """'dimethylsilanuide' would read as the anion with two methyl groups; (d) (the Blue Book) asks for
    'bis'. The salt of a silicon -uide anion carries the label its siblings carry ('methylsilanuide (PIN)',."""
    row = Orthonym().name_tiered("[Ca+2].C[SiH4-].C[SiH4-]")
    assert (row["name"], row["tier"]) == ("calcium bis(methylsilanuide)", "pin_verified")
