"""An isotope-labelled mixture named part by part (2026-11-17, PubChem rerun 5 triage item 2a).

The isotope decorator names the unlabelled mixture and places the nuclide descriptors in that name; when no
placement reads back, the input used to be declined. Each component is a compound of its own whose name the mixture
name cites "Mixed organic - inorganic adducts", the Blue Book: "names are formed by citing the names of
individual compounds, connected by long (em) dashes"), and a labelled component alone takes its descriptor before
"the part of the compound that is isotopically substituted" "Isotopically substituted compounds"). The
best-effort tier now names such a mixture from the names of its components alone, read back with every label; the
other tiers keep their names, and the name carries no PIN label.
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.rules import adducts, isotope_components, isotope_slots

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
    monkeypatch.setattr(isotope_slots, "labelled_parts_adduct_name",
                        lambda *a, **k: calls.append(1) or real(*a, **k))
    row = _best_effort().name_tiered("CCO.[2H]c1ncc[nH]1")
    assert (row["name"], row["tier"]) == ("ethanol—(2-2H)-1H-imidazole (1/1)", "pin_verified")
    assert calls == []


def test_unlabelled_or_one_part_input_is_not_taken():
    namer = _best_effort()
    assert isotope_components.labelled_parts_adduct_name(Chem.MolFromSmiles("CCO.OC(=O)C"), "pin", namer) is None
    assert isotope_components.labelled_parts_adduct_name(Chem.MolFromSmiles("[2H]C(Cl)Cl"), "pin", namer) is None
    assert isotope_components.labelled_parts_adduct_name(Chem.MolFromSmiles(_TRITIATED), "pin", Orthonym()) is None


@pytest.mark.parametrize("smiles, name", [
    ("[2H]C1=C(C(=C2C(=C1[2H])C(=O)C3=C(O2)C(=C(C(=C3[2H])[2H])C)C([2H])([2H])[2H])C([2H])([2H])C(=O)O)[2H].CC."
     "CC1=C(C2=C(C=C1)C(=O)C3=CC=CC(=C3O2)CC(=O)O)C",
     "(5,6-dimethyl-9-oxo-9H-xanthen-4-yl)acetic acid—[5-(2H3)methyl-6-methyl-9-oxo(2H5)-9H-xanthen-4-yl]"
     "(2H2)acetic acid—ethane (1/1/1)"),
    ("[2H]C1C(C(=O)N(C(=C)C1([2H])[2H])[2H])C(C)(C)C.[2H]C1C(=C)N(C(=O)C(C1([2H])[2H])C(C)(C)C)[2H]."
     "[2H]C1(C(C(=O)NC(=C)C1([2H])[2H])C(C)(C)C)[2H].[2H]C1(C(C(=O)N(C(=C)C1([2H])[2H])[2H])C(C)(C)C)[2H]."
     "[2H]C1(C(C(=O)N(C(=O)C1([2H])[2H])[2H])C(C)(C)C)[2H].C",
     "3-tert-butyl-6-methylidene(4,4,5,5-2H4)piperidin-2-one—3-tert-butyl-6-methylidene(1,4,4,5-2H4)piperidin-2-one—"
     "3-tert-butyl-6-methylidene(1,4,5,5-2H4)piperidin-2-one—3-tert-butyl-6-methylidene(1,4,4,5,5-2H5)piperidin-2-one—"
     "3-tert-butyl(1,4,4,5,5-2H5)piperidine-2,6-dione—methane (1/1/1/1/1/1)"),
])
def test_panel_mixtures_are_named_from_their_labelled_parts(smiles, name):
    row = _best_effort().name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "systematic_verified")


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
