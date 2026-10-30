"""The atom, the ions and the isotopes of zinc, and the hydride ion beside zinc (``ELEMENT`` of
``data/metal_scope.py``, ``rules/metal_atoms.py``, the dispatch class ``METAL_ATOM``).

``[Zn]``, ``[Zn+2]`` and ``[65Zn]`` were named at no tier although OPSIN reads ``zinc``, ``zinc(2+)``
and ``(65Zn)zinc``. The retained table of cation words gave ``[Zn+2]`` the word 'zinc' (the cation word
of a salt, ``zinc dichloride``), which is no name of the free ion: OPSIN reads 'zinc' as the neutral
atom, so the namer's gate refused it. The class ``METAL_ATOM`` sits above ``CATION_RETAINED`` and names
the atom, the ion and the isotope by the element name, the charge number and the nuclide symbol.

Blue Book: Compounds with at least one metal-carbon single bond (the Blue Book)
cites 'pentaammine(ethyl)osmium(1+)', the charge number of a cation after its element name;
(:43718), "the nuclide symbol(s) enclosed in parentheses... before the part of the compound that is
isotopically substituted"; PREFERRED IUPAC NAMES (:2056): the label "PIN" belongs to compounds
of the listed elements "that also contain at least one carbon atom in their structure", so the names
of a zinc atom, ion or isotope are verified systematic names and no PIN. The hydride ion is cited as
'hydride' in the salt of zinc,:43590, "binary names formed by citing the name of the cation
followed by that of the anion"); a carbon-free salt keeps the label of its path (user decision
2026-09-30, answer 3 of 2026-10-13).

Every name is read back by a fresh OPSIN call that does not go through the engine, to the full
standard InChIKey of the input.
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.data import metal_scope
from orthonym.data.ion_retained_names import get_cation_name
from orthonym.data.metal_scope import CARBANION_SALTS, ELEMENT, MetalScope
from orthonym.perception.ions import parse_salt_fragments
from orthonym.routing.dispatch_table import DISPATCH_TABLE, StoutClass
from orthonym.routing.dispatcher import ClassFirstRouter
from orthonym.rules import adducts, metal_atoms, metal_labels
from orthonym.rules.metal_atoms import admitted_metal_atom_name, inorganic_anion_word, metal_atom_name
from tests.support.rt_assert import assert_full_rt, name_is_rt_exact

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


def _row(smiles, tier):
    return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


def _view(row):
    return (row["name"], row["tier"], row["is_pin"], row["limit_code"])


def _class_of(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return ClassFirstRouter().dispatch(mol, smiles, Chem.MolToSmiles(mol), _skip_decomposition=False,
                                       _style="pin").class_id


def _assert_named_without_a_pin(smiles, name, tiers=("best-effort", "valid")):
    for tier in tiers:
        row = _row(smiles, tier)
        assert row["name"] == name, (tier, row)
        assert row["tier"] == "systematic_verified" and row["is_pin"] is False, (tier, row)
        assert_full_rt(row["name"], smiles)
    declined = _row(smiles, "pin")                    # the default tier declines it, with a reason
    assert declined["tier"] == "abstain" and declined["limit_code"] == "NO_VERIFIED_PIN", declined
    assert declined["is_pin"] is False


# ---- the atom, the ions and the isotopes (panel B10; the C1 rows of the corpora)

ATOMS = [
    ("[Zn]", "zinc"),
    ("[Zn+2]", "zinc(2+)"),
    ("[Zn++]", "zinc(2+)"),                      # another spelling of the same drawing
    ("[Zn+]", "zinc(1+)"),
    ("[Zn+4]", "zinc(4+)"),                      # named as drawn: the read-back, not a valence list, decides
    ("[65Zn]", "(65Zn)zinc"),                    #: the nuclide symbol in front of the element name
    ("[66Zn]", "(66Zn)zinc"),
    ("[66Zn+2]", "(66Zn)zinc(2+)"),
    ("[65Zn+2]", "(65Zn)zinc(2+)"),
]


@pytest.mark.parametrize("smiles,name", ATOMS)
def test_the_atom_the_ion_and_the_isotope_of_zinc_are_named_and_read_back(smiles, name):
    """Systematic names, no PIN; the best-effort and the valid tier carry them, the default
    tier declines them with a reason (a name exists below the PIN)."""
    _assert_named_without_a_pin(smiles, name)


def test_the_free_ion_is_not_named_by_the_cation_word_of_a_salt():
    """The table of cation words keeps 'zinc' for the salt ('zinc dichloride'); as the name of the free
    ion [Zn+2] it is the neutral atom for OPSIN, so it is never shipped for it."""
    assert get_cation_name("[Zn+2]") == "zinc"
    assert name_is_rt_exact("zinc", "[Zn+2]") is False
    assert name_is_rt_exact("zinc(2+)", "[Zn+2]") is True
    assert _row("[Zn+2]", "best-effort")["name"] == "zinc(2+)"


def test_the_isotope_and_the_charge_are_in_the_name():
    """0-wrong: a name that drops or changes the nuclide or the charge reads back as another structure."""
    assert name_is_rt_exact("zinc", "[65Zn]") is False
    assert name_is_rt_exact("(66Zn)zinc", "[65Zn]") is False
    assert name_is_rt_exact("(65Zn)zinc(2+)", "[66Zn+2]") is False
    assert name_is_rt_exact("zinc(2+)", "[66Zn+2]") is False
    assert _row("[66Zn+2]", "best-effort")["name"] == "(66Zn)zinc(2+)"


def test_the_class_of_the_dispatch_table_names_zinc_before_the_retained_cation_word():
    assert _class_of("[Zn]") == StoutClass.METAL_ATOM
    assert _class_of("[Zn+2]") == StoutClass.METAL_ATOM         # not CATION_RETAINED, whose word is 'zinc'
    assert _class_of("[Cu+2]") == StoutClass.CATION_RETAINED    # a metal that is not admitted
    assert _class_of("[Na+]") == StoutClass.CATION_RETAINED
    assert _class_of("[Cd]") != StoutClass.METAL_ATOM          # a metal that is not admitted
    assert _class_of("[Hg]") == StoutClass.METAL_ATOM          # admitted to ELEMENT since 2026-10-30
    assert _class_of("[Zn-]") != StoutClass.METAL_ATOM
    assert _class_of("[ZnH]") != StoutClass.METAL_ATOM
    priorities = {c: e.priority for c, e in DISPATCH_TABLE.items()}
    assert priorities[StoutClass.CATION_QUATERNARY] < priorities[StoutClass.METAL_ATOM] < priorities[
        StoutClass.CATION_RETAINED]


# ---- the producer (pure: no JVM)

@pytest.mark.parametrize("smiles,expected", [
    ("[Zn]", "zinc"),
    ("[Zn+2]", "zinc(2+)"),
    ("[Zn+]", "zinc(1+)"),
    ("[65Zn]", "(65Zn)zinc"),
    ("[66Zn+2]", "(66Zn)zinc(2+)"),
    ("[Zn-]", None),                  # an '-ide' of a metal is not built
    ("[Zn-2]", None),
    ("[ZnH]", None),                  # a hydrogen on the metal: a hydride, named by its ligands (BONDED_NAMES)
    ("[ZnH+]", None),
    ("[Zn]=[Zn]", None),              # two atoms
    ("Cl[Zn]Cl", None),
    ("[Zn+2].[Cl-].[Cl-]", None),     # a salt, named by the salt route
    ("[Cu]", None),                   # a metal that the table does not admit to ELEMENT
    ("[Cu+2]", None),
    ("[Na+]", None),
    ("[Cd+2]", None),
    ("[Hg+2]", "mercury(2+)"),       # admitted to ELEMENT since 2026-10-30 (BONDED_METAL_VALENCES)
    ("[Cl-]", None),
    ("[Si]", None),                   # a metalloid is named substitutively, not by this mechanism
    ("C", None),
])
def test_the_producer_names_an_admitted_atom_and_nothing_else(smiles, expected):
    assert admitted_metal_atom_name(Chem.MolFromSmiles(smiles)) == expected


def test_the_producer_follows_the_scope_table(monkeypatch):
    """Admitting a metal to ELEMENT is an edit of the table; the element name is the one the adduct
    components use."""
    assert admitted_metal_atom_name(Chem.MolFromSmiles("[Cu+2]")) is None
    monkeypatch.setitem(metal_scope.METAL_SCOPE, "Cu", MetalScope(frozenset({ELEMENT}), frozenset({2})))
    assert admitted_metal_atom_name(Chem.MolFromSmiles("[Cu+2]")) == "copper(2+)"
    assert admitted_metal_atom_name(Chem.MolFromSmiles("[63Cu]")) == "(63Cu)copper"
    monkeypatch.setitem(metal_scope.METAL_SCOPE, "Zn", MetalScope(frozenset({CARBANION_SALTS}), frozenset({2})))
    assert admitted_metal_atom_name(Chem.MolFromSmiles("[Zn+2]")) is None


def test_the_element_names_have_one_home():
    """The adduct components of ``rules/adducts.py`` read the table and the function of
    ``rules/metal_atoms.py``."""
    assert adducts._METAL_ELEMENT_NAMES is metal_atoms.METAL_ELEMENT_NAMES
    assert adducts._metal_atom_component_name is metal_atoms.metal_atom_name
    assert metal_atom_name(Chem.MolFromSmiles("[Ni]")) == "nickel"
    assert metal_atom_name(Chem.MolFromSmiles("[Fe+2]")) == "iron(2+)"
    assert metal_atom_name(Chem.MolFromSmiles("[99Tc]")) == "(99Tc)technetium"
    assert metal_atom_name(Chem.MolFromSmiles("[Co-]")) is None


# ---- the label: no PIN, set by a predicate that the tier row reads

def test_the_label_predicate():
    f = metal_labels.is_metal_atom_or_ion_name_without_pin
    assert f("[Zn]", "zinc") and f("[Zn+2]", "zinc(2+)") and f("[65Zn]", "(65Zn)zinc")
    assert not f("[Cu+2]", "copper(II)")                                   # the retained name keeps its label
    assert not f("[Na+]", "sodium")
    assert not f("[Zn+2].[Cl-].[Cl-]", "zinc dichloride")                  # a carbon-free salt keeps its label
    assert not f("[H-].[H-].[Zn+2]", "zinc dihydride")
    assert not f("CC[Zn]CC", "diethylzinc")
    assert not f("[Zn+2]", "") and not f("", "zinc(2+)") and not f("[Zn+2]", None)
    assert metal_labels.is_metal_name_without_pin("[Zn+2]", "zinc(2+)") is True


def test_the_predicate_is_what_lowers_the_label(monkeypatch):
    """Switched off, the name of the ion would be a path label (pin_verified): the predicate is on
    the path of the tier row, and it is the only thing that sets the label."""
    monkeypatch.setattr(metal_labels, "is_metal_atom_or_ion_name_without_pin", lambda smiles, name: False)
    row = _row("[Zn+2]", "valid")
    assert row["name"] == "zinc(2+)" and row["tier"] == "pin_verified", row


# ---- the hydride ion beside zinc (panel B9 ionic)

def test_zinc_dihydride_is_named_and_read_back():
    """``[H-].[H-].[Zn+2]``: the salt route cites 'hydride' for the ion beside zinc. A carbon-free salt
    keeps the label of its path (the named exception of the label policy: 'zinc dichloride' alike)."""
    smiles = "[H-].[H-].[Zn+2]"
    for tier in ("pin", "valid", "best-effort"):
        row = _row(smiles, tier)
        assert row["name"] == "zinc dihydride", (tier, row)
        assert row["tier"] == "pin_verified" and row["is_pin"] is True, (tier, row)
        assert_full_rt(row["name"], smiles)
    assert _row("[Zn+2].[Cl-].[Cl-]", "pin")["tier"] == "pin_verified"   # the same label as zinc dichloride
    assert _row("[Zn+2].[Zn+2].[H-].[H-].[H-].[H-]", "best-effort")["name"] == "dizinc tetrahydride"
    assert_full_rt("dizinc tetrahydride", "[Zn+2].[Zn+2].[H-].[H-].[H-].[H-]")


def test_the_hydride_word_is_for_the_metals_the_table_admits():
    zinc = parse_salt_fragments(Chem.MolFromSmiles("[H-].[H-].[Zn+2]"))
    sodium = parse_salt_fragments(Chem.MolFromSmiles("[Na+].[H-]"))
    copper = parse_salt_fragments(Chem.MolFromSmiles("[Cu+2].[H-].[H-]"))
    both = parse_salt_fragments(Chem.MolFromSmiles("[Cu+2].[Zn+2].[H-].[H-].[H-].[H-]"))
    assert inorganic_anion_word("[H-]", zinc["cations"]) == "hydride"
    assert inorganic_anion_word("[H-]", sodium["cations"]) == "hydride"  # Groups 1-2: CARBANION_SALTS since 2026-10-30
    assert inorganic_anion_word("[H-]", copper["cations"]) is None
    assert inorganic_anion_word("[H-]", both["cations"]) is None          # one cation outside the table
    assert inorganic_anion_word("[H-]", []) is None
    assert inorganic_anion_word("[Cl-]", sodium["cations"]) == "chloride"  # the table of retained anions
    assert inorganic_anion_word("[O-2]", zinc["cations"]) == "oxide"
    assert inorganic_anion_word("[CH3-]", zinc["cations"]) is None


# ---- decision 2: a neutral atom beside other components is named only where a producer names it

def test_neutral_atom_with_neutral_components_is_unchanged():
    """Two neutral acids and a zinc atom: the adduct producer already names it (a component of a
     adduct is cited by its element name), byte for byte."""
    smiles = "CCCCCCCCCCCCCCCCCC(=O)O.CCCCCCCCCCCCCCCCCC(=O)O.[Zn]"
    row = _row(smiles, "best-effort")
    assert row["name"] == "octadecanoic acid—zinc (2/1)"
    assert row["tier"] == "systematic_verified" and row["is_pin"] is False
    assert_full_rt(row["name"], smiles)


def test_a_neutral_atom_beside_anions_is_cited_with_zero():
    """Decision 2 of the user (2026-10-13) declined a producer for a neutral metal atom beside anions;
    User decision 2026-10-30 (breadth lane, levers B and C) replaces it: the atom is cited with the oxidation number zero, at the best-effort tier only
    (``ZERO_VALENT`` of ``data/metal_scope.py``), and the name reads back to the neutral atom drawn."""
    row = _row("[Zn].[Cl-].[Cl-]", "best-effort")
    assert row["name"] == "chloride—zinc(0) (2/1)" and row["tier"] == "systematic_verified", row
    assert row["is_pin"] is False
    assert_full_rt(row["name"], "[Zn].[Cl-].[Cl-]")
    assert _row("[Zn].[Cl-].[Cl-]", "pin")["tier"] == "abstain"


# ---- the other metals and the other assemblies are not moved (START literals, computed on the start tree)

#: (default tier, best-effort tier) identity views at the start commit of the zinc work
START_ROWS = {
    "[Cu]": (
        ('copper compound (not supported)', 'abstain', False, 'UNSUPPORTED_ELEMENT'),
        (None, 'abstain', False, 'UNSUPPORTED_ELEMENT')),
    "[Cu+2]": (
        ('copper(II)', 'pin_verified', True, None),
        ('copper(II)', 'pin_verified', True, None)),
    "[Fe+3]": (
        ('iron(III)', 'pin_verified', True, None),
        ('iron(III)', 'pin_verified', True, None)),
    "[Fe+2]": (
        ('iron(II)', 'pin_verified', True, None),
        ('iron(II)', 'pin_verified', True, None)),
    "[Na+]": (
        ('sodium compound (not supported)', 'abstain', False, 'UNSUPPORTED_ELEMENT'),
        (None, 'abstain', False, 'UNSUPPORTED_ELEMENT')),
    "[Na]": (
        ('sodium compound (not supported)', 'abstain', False, 'UNSUPPORTED_ELEMENT'),
        (None, 'abstain', False, 'UNSUPPORTED_ELEMENT')),
    "[Li+]": (
        ('lithium compound (not supported)', 'abstain', False, 'UNSUPPORTED_ELEMENT'),
        (None, 'abstain', False, 'UNSUPPORTED_ELEMENT')),
    "[Mg+2]": (
        ('magnesium compound (not supported)', 'abstain', False, 'UNSUPPORTED_ELEMENT'),
        (None, 'abstain', False, 'UNSUPPORTED_ELEMENT')),
    "[Ca+2]": (
        ('calcium compound (not supported)', 'abstain', False, 'UNSUPPORTED_ELEMENT'),
        (None, 'abstain', False, 'UNSUPPORTED_ELEMENT')),
    "[Cd+2]": (
        ('cadmium compound (not supported)', 'abstain', False, 'UNSUPPORTED_ELEMENT'),
        (None, 'abstain', False, 'UNSUPPORTED_ELEMENT')),
    "[Sn+4]": (
        ('tin(IV)', 'pin_verified', True, None),
        ('tin(IV)', 'pin_verified', True, None)),
    "[Ag+]": (
        ('silver compound (not supported)', 'abstain', False, 'UNSUPPORTED_ELEMENT'),
        (None, 'abstain', False, 'UNSUPPORTED_ELEMENT')),
    "[Cu+2].[H-].[H-]": (
        ('copper compound (not supported)', 'abstain', False, 'UNSUPPORTED_ELEMENT'),
        (None, 'abstain', False, 'UNSUPPORTED_ELEMENT')),
}

#: zinc beside a cation that the table does not admit to ``CARBANION_SALTS`` (no hydride word), and a
#: charge-unbalanced assembly (no salt): no producer names them. The failure label is the engine's, UNNAMEABLE
#: (zinc is a complete metal and every metal of the assembly is admitted to the table); the text of the plain
#: call is the descriptive fallback of the metal of lowest atomic number
#: the hydride salts of Groups 1-2 and of zinc: User decision 2026-10-30 (breadth lane, levers B and C) admits Groups 1-2 to ``CARBANION_SALTS``,
#: so these are binary salts, carbon-free: kept pin_verified by the maintainers' decision)
ADMITTED_HYDRIDE_ROWS = {
    "[Na+].[H-]": "sodium hydride",
    "[Li+].[H-]": "lithium hydride",
    "[Mg+2].[H-].[H-]": "magnesium dihydride",
    "[Ca+2].[H-].[H-]": "calcium dihydride",
    "[Li+].[Zn+2].[H-].[H-].[H-]": "lithium zinc trihydride",
    "[Na+].[Zn+2].[H-].[H-].[H-]": "sodium zinc trihydride",
    "[H-].[H-].[H-].[Zn+2].[Li+]": "lithium zinc trihydride",
}


@pytest.mark.parametrize("smiles", sorted(ADMITTED_HYDRIDE_ROWS))
def test_admitted_hydride_salts(smiles):
    for tier in ("pin", "best-effort"):
        assert _view(_row(smiles, tier)) == (ADMITTED_HYDRIDE_ROWS[smiles], "pin_verified", True, None), (smiles, tier)


UNNAMED_ZINC_ASSEMBLIES = {
    "[Zn+2].[H-]": (
        ('zinc compound (not supported)', 'abstain', False, 'UNNAMEABLE'),
        # a charged assembly: the best-effort adduct of its ions (the hydride component, 2026-10-30)
        ('zinc(2+)—hydride (1/1)', 'systematic_verified', False, None)),
}


@pytest.mark.parametrize("smiles", sorted(UNNAMED_ZINC_ASSEMBLIES))
def test_zinc_assemblies_no_producer_names_are_unnameable(smiles):
    default_view, best_effort_view = UNNAMED_ZINC_ASSEMBLIES[smiles]
    assert _view(_row(smiles, "pin")) == default_view, smiles
    assert _view(_row(smiles, "best-effort")) == best_effort_view, smiles



@pytest.mark.parametrize("smiles", sorted(START_ROWS))
def test_other_metals_are_not_moved(smiles):
    """The atoms, ions and hydride salts of the metals that the table does not admit keep their rows of the
    start commit."""
    default_view, best_effort_view = START_ROWS[smiles]
    assert _view(_row(smiles, "pin")) == default_view, smiles
    assert _view(_row(smiles, "best-effort")) == best_effort_view, smiles


def test_the_other_metals_panel_has_no_atom_or_hydride_row_the_class_could_reach():
    """The 73-drawing panel of the other metals is untouched by construction: no row is one atom of
    an admitted metal, and no row holds a hydride ion beside zinc."""
    from tests.support.metal_baselines import baseline_path, read_rows
    for row in read_rows(baseline_path("other")):
        mol = Chem.MolFromSmiles(row["smiles"])
        assert admitted_metal_atom_name(mol) is None, row["smiles"]
        assert not ("[H-]" in row["smiles"] and "Zn" in row["smiles"]), row["smiles"]
