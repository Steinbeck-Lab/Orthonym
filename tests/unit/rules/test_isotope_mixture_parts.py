"""An isotope-labelled mixture named part by part (2026-11-17, PubChem rerun 5 triage item 2a).

The isotope decorator names the unlabelled mixture and places the nuclide descriptors in that name; when no
placement reads back, the input used to be declined. Each component is a compound of its own whose name the mixture
name cites "Mixed organic - inorganic adducts", the Blue Book: "names are formed by citing the names of
individual compounds, connected by long (em) dashes"), and a labelled component alone takes its descriptor before
"the part of the compound that is isotopically substituted" "Isotopically substituted compounds"). The
best-effort tier now names such a mixture from the names of its components alone, read back with every label; the
other tiers keep their names, and the name carries no PIN label.

Since 2026-11-17 (user decision, the slow PubChem rows) the best-effort tier names a labelled mixture of three or more
parts that is no salt part by part FIRST: the decorator's search over the whole joined name took the whole time limit
on them. A salt keeps the decorator first ('potassium (18F)fluoranide',, and so does a mixture of two; a
mixture whose labelled parts are PINs alone asks the PIN tier, so a PIN it names is kept.
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.rules import adducts, isotope_components, isotope_passes

pytestmark = pytest.mark.opsin_gate

_TRITIATED = "[H][C@@]1(CC([C@H](O1)CC)C(=O)F)[3H].C(=O)(C(Cl)(Cl)Cl)O"


def _best_effort():
    return Orthonym(general_fallback=True, general_fallback_unverified=True, allow_aromatic_general=True)


def test_labelled_mixture_is_named_from_its_parts_at_best_effort():
    row = _best_effort().name_tiered(_TRITIATED)
    assert row["name"] == "trichloroacetic acid—(2R,5S)-2-ethyl-3-[fluoro(oxo)methyl](5-3H1)oxolane (1/1)"
    assert row["tier"] not in ("pin_verified", "pin_unverified")


def test_pin_tier_still_declines_the_labelled_mixture():
    assert Orthonym().name_tiered(_TRITIATED)["tier"] == "abstain"


def test_a_join_that_drops_a_label_is_refused(monkeypatch):
    """Mutation: the labelled part gets the name of its unlabelled skeleton (a nested naming, where the decorator
    does not run, gives it); the read-back against the labelled input refuses the whole name."""
    monkeypatch.setattr(adducts, "_name_component_in_fresh_frame",
                        lambda once: "(2R)-2-ethyl-3-[fluoro(oxo)methyl]oxolane")
    namer = _best_effort()
    assert isotope_components.labelled_parts_adduct_name(Chem.MolFromSmiles(_TRITIATED), "pin", namer) is None


def test_a_mixture_the_decorator_names_keeps_its_name(monkeypatch):
    calls = []
    real = isotope_components.labelled_parts_adduct_name
    monkeypatch.setattr(isotope_components, "labelled_parts_adduct_name",
                        lambda *a, **k: calls.append(1) or real(*a, **k))
    row = _best_effort().name_tiered("CCO.[2H]c1ncc[nH]1")
    assert (row["name"], row["tier"]) == ("ethanol—(2-2H)-1H-imidazole (1/1)", "pin_verified")
    assert calls == []


def test_unlabelled_or_one_part_input_is_not_taken():
    namer = _best_effort()
    assert isotope_components.labelled_parts_adduct_name(Chem.MolFromSmiles("CCO.OC(=O)C"), "pin", namer) is None
    assert isotope_components.labelled_parts_adduct_name(Chem.MolFromSmiles("[2H]C(Cl)Cl"), "pin", namer) is None
    assert isotope_components.labelled_parts_adduct_name(Chem.MolFromSmiles(_TRITIATED), "pin", Orthonym()) is None


@pytest.mark.parametrize("smiles, name, tier", [
    ("[2H]C1=C(C(=C2C(=C1[2H])C(=O)C3=C(O2)C(=C(C(=C3[2H])[2H])C)C([2H])([2H])[2H])C([2H])([2H])C(=O)O)[2H].CC."
     "CC1=C(C2=C(C=C1)C(=O)C3=CC=CC(=C3O2)CC(=O)O)C",
     "(5,6-dimethyl-9-oxo-9H-xanthen-4-yl)acetic acid—[5-(2H3)methyl-6-methyl-9-oxo(2H5)-9H-xanthen-4-yl]"
     "(2H2)acetic acid—ethane (1/1/1)", "systematic_verified"),
    ("[2H]C1C(C(=O)N(C(=C)C1([2H])[2H])[2H])C(C)(C)C.[2H]C1C(=C)N(C(=O)C(C1([2H])[2H])C(C)(C)C)[2H]."
     "[2H]C1(C(C(=O)NC(=C)C1([2H])[2H])C(C)(C)C)[2H].[2H]C1(C(C(=O)N(C(=C)C1([2H])[2H])[2H])C(C)(C)C)[2H]."
     "[2H]C1(C(C(=O)N(C(=O)C1([2H])[2H])[2H])C(C)(C)C)[2H].C",
     "3-tert-butyl-6-methylidene(4,4,5,5-2H4)piperidin-2-one—3-tert-butyl-6-methylidene(1,4,4,5-2H4)piperidin-2-one—"
     "3-tert-butyl-6-methylidene(1,4,5,5-2H4)piperidin-2-one—3-tert-butyl-6-methylidene(1,4,4,5,5-2H5)piperidin-2-one—"
     "3-tert-butyl(1,4,4,5,5-2H5)piperidine-2,6-dione—methane (1/1/1/1/1/1)", "systematic_verified"),
])
def test_panel_mixtures_are_named_from_their_labelled_parts(smiles, name, tier):
    row = _best_effort().name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, tier)


def test_the_decorator_runs_once_per_labelled_part(monkeypatch):
    """Spy: inside the fallback each labelled part is a top-level naming of its own, where the isotope decorator
    runs; the unlabelled part (trichloroacetic acid) never reaches it."""
    from orthonym.rules import isotopes
    seen = []
    real = isotopes.decorate_isotopic_name
    monkeypatch.setattr(isotopes, "decorate_isotopic_name", lambda s, *a, **k: seen.append(s) or real(s, *a, **k))
    name = isotope_components.labelled_parts_adduct_name(Chem.MolFromSmiles(_TRITIATED), "pin", _best_effort())
    assert name and "(5-3H1)" in name
    parts = [s for s in seen if "." not in s]
    assert parts and all("[3H]" in s for s in parts)


# A labelled mixture of PubChem that ran past 120 s in the decorator's search over the whole name (2026-11-17).
_SLOW = ("[2H]C([2H])([2H])C1=CN=C(C=C1)C2=CC=CC=[C-]2.[2H]C([2H])([2H])C1=CN=C(C=C1)C2=CC=CC=[C-]2.C."
         "CC1=CN=C(C=C1)C2=CC=CC=[C-]2.CO.CO.C[Si](C)(C)C1=CN=C(C=C1C2=CC=CC=C2)C3=CC4=C(C=C3)C5=CC=CC=C5C6=CC=CC=C64."
         "C[Si](C)(C)C1=CN=C(C=C1C2=CC=CC=C2)C3=CC4=C(C=[C-]3)C5=CC=CC=C5C6=CC=CC=C64.C(F)(F)(F)S(=O)(=O)O.F.[Ir].[Ir]")


def _whole_mixture_passes(monkeypatch):
    """Spy: the SMILES of every call of the decorator's passes on a mixture (a part named alone has no dot)."""
    seen = []
    real = isotope_passes.decorate_passes
    monkeypatch.setattr(isotope_passes, "decorate_passes", lambda smiles, *a, **k: (
        seen.append(smiles) if "." in smiles else None) or real(smiles, *a, **k))
    return seen


def test_a_slow_labelled_mixture_is_named_part_by_part_first(monkeypatch):
    seen = _whole_mixture_passes(monkeypatch)
    row = _best_effort().name_tiered(_SLOW)
    assert (row["name"], row["tier"]) == (
        "trifluoromethanesulfonic acid—methanol—5-[4-phenyl-5-(trimethylsilyl)pyridin-2-yl]tetracyclo"
        "[12.4.0.0^2,7.0^8,13]octadeca-1,3,5,7,9,11,13,15,17-nonaen-4-ide—1-(5-methylpyridin-2-yl)cyclohexa-"
        "1,3,5-trien-6-ide—1-[5-(2H3)methylpyridin-2-yl]cyclohexa-1,3,5-trien-6-ide—4-phenyl-5-(trimethylsilyl)-"
        "2-(triphenylen-2-yl)"
        "pyridine—methane—hydrogen fluoride—iridium(0) (1/2/1/1/2/1/1/1/2)", "systematic_verified")
    assert seen == []


_THREE = "CCO.CC(=O)O.[2H]c1ncc[nH]1"


@pytest.mark.parametrize("first", [True, False])
def test_the_route_runs_first_on_a_mixture_of_three(monkeypatch, first):
    """Mutation: with the route taken out of first place it is not called (the decorator names the mixture); either
    way the PIN tier gives the same PIN (the components in the order that does not see a label,."""
    calls = []
    real = isotope_components.labelled_parts_adduct_name
    monkeypatch.setattr(isotope_components, "labelled_parts_adduct_name",
                        lambda *a, **k: calls.append(1) or real(*a, **k))
    if not first:
        monkeypatch.setattr(isotope_components, "takes_parts_first", lambda *a: False)
    row = _best_effort().name_tiered(_THREE)
    assert (row["name"], row["tier"]) == ("acetic acid—ethanol—(2-2H)-1H-imidazole (1/1/1)", "pin_verified")
    assert calls == ([1] if first else [])


def test_water_beside_a_labelled_part_is_no_pin():
    """: 'water' carries no PIN label (its use as a preferred name is deferred), so the PIN tier is not asked
    for a mixture whose name cites it (the decorator's naming of the whole mixture loses that record)."""
    row = _best_effort().name_tiered("CCO.O.[2H]C([2H])([2H])N")
    assert (row["name"], row["tier"]) == ("ethanol—(2H3)methanamine—water (1/1/1)", "systematic_verified")


def test_a_labelled_component_sorts_as_its_skeleton():
    """ "Alphanumerical order" (the Blue Book): the order "do[es] not include... isotopic or
    stereochemical descriptors"."""
    from orthonym.rules.adducts import component_sort_key
    assert component_sort_key("[2H]c1ccccc1C(=O)O")[:-1] == component_sort_key("c1ccccc1C(=O)O")[:-1]


@pytest.mark.parametrize("smiles, name, tier", [
    ("[K+].[18F-]", "potassium (18F)fluoranide", "systematic_verified"),
    ("[Na+].[2H-]", "sodium deuteride", "systematic_verified"),
    ("C[N+](C)(C)C.[36Cl-]", "N,N,N-trimethylmethanaminium (36Cl)chloranide", "systematic_verified"),
    ("[Na+].[13CH3]C([O-])=O", "sodium (2-13C)acetate", "pin_verified"),
])
def test_a_labelled_salt_keeps_the_decorator_first(smiles, name, tier):
    """A salt names its cation and anion, not ions of an adduct: the route does not run first."""
    assert not isotope_components.takes_parts_first(Chem.MolFromSmiles(smiles), _best_effort())
    row = _best_effort().name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, tier)


@pytest.mark.parametrize("smiles, name", [
    ("CC(=O)O.[2H]C([2H])([2H])N", "acetic acid—(2H3)methanamine (1/1)"),          # two parts: the decorator first
    ("CO.CC(=O)O.[2H]C([2H])([2H])N", "acetic acid—methanol—(2H3)methanamine (1/1/1)"),  # three: then the PIN tier
])
def test_a_mixture_of_pins_keeps_its_pin(smiles, name):
    row = _best_effort().name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "pin_verified")


def test_a_mixture_of_two_keeps_the_decorator_first():
    assert not isotope_components.takes_parts_first(Chem.MolFromSmiles(_TRITIATED), _best_effort())


def test_a_mixture_the_route_cannot_name_keeps_the_decorators_name():
    row = _best_effort().name_tiered("[2H]N.[2H]N.[2H]N.C.C.C.C.C.C.C.C")
    assert (row["name"], row["tier"]) == ("methane—(N-2H1)ammonia (8/3)", "systematic_verified")


def test_the_pin_tier_does_not_take_the_route():
    assert not isotope_components.takes_parts_first(Chem.MolFromSmiles(_SLOW), Orthonym())
    assert Orthonym().name_tiered(_SLOW)["tier"] == "abstain"
