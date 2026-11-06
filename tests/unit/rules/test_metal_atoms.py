"""The atom, the ions and the isotopes of zinc and of every other element, and the hydride ion beside zinc
(``ELEMENT`` of ``data/metal_scope.py``, ``rules/metal_atoms.py``, the dispatch class ``ELEMENT_ATOM``, which was
``METAL_ATOM`` until lane LONE).

``[Zn]``, ``[Zn+2]`` and ``[65Zn]`` were named at no tier although OPSIN reads ``zinc``, ``zinc(2+)``
and ``(65Zn)zinc``. The retained table of cation words gave ``[Zn+2]`` the word 'zinc' (the cation word
of a salt, ``zinc dichloride``), which is no name of the free ion: OPSIN reads 'zinc' as the neutral
atom, so the namer's gate refused it. The class (now ``ELEMENT_ATOM``) sits above ``CATION_RETAINED`` and names
the atom, the ion and the isotope by the element name, the charge number and the nuclide symbol.

Blue Book: Compounds with at least one metal-carbon single bond (the Blue Book)
cites 'pentaammine(ethyl)osmium(1+)', the charge number of a cation after its element name;
(:43718), "the nuclide symbol(s) enclosed in parentheses... before the part of the compound that is
isotopically substituted"; PREFERRED IUPAC NAMES (:2056): the label "PIN" belongs to compounds
of the listed elements "that also contain at least one carbon atom in their structure", so the names
of a zinc atom, ion or isotope are verified systematic names and no PIN. The hydride ion is cited as
'hydride' in the salt of zinc,:43590, "binary names formed by citing the name of the cation
followed by that of the anion"). Since 2026-11-02 (user decision "decide what the Blue Book says", which
replaces the path labels of 2026-09-30) every carbon-free name is a preselected name at most and no PIN
 :2056,:2058 "... the label 'preselected name' is added to appropriate names"): the salts, the
retained ion names and the names on a parent hydride included; the PIN tier declines them.

Lane LONE (2026-11-01) names the lone atom of every element: the bare atom of a mononuclear parent hydride is its
radical, RADICALS DERIVED FROM PARENT HYDRIDES, (:40426) "named by adding the suffix 'yl' to the
name of the parent hydride" ('chloranyl'); an isotopic halide anion takes the preselected anion name of its parent
hydride, (2) (:40924) ('(36Cl)chloranide'); hydrogen takes Table 8.1 of NAMES FOR HYDROGEN ATOMS
AND IONS (:43659) ('deuteride').

Every name is read back by a fresh OPSIN call that does not go through the engine, to the full
standard InChIKey of the input.
"""
import pytest
from rdkit import Chem

from orthonym.data import metal_scope
from orthonym.data.ion_retained_names import get_cation_name
from orthonym.data.metal_scope import CARBANION_SALTS, ELEMENT, MetalScope
from orthonym.perception.ions import parse_salt_fragments
from orthonym.routing.dispatch_table import DISPATCH_TABLE, StoutClass
from orthonym.routing.dispatcher import ClassFirstRouter
from tests.support.opsin_marks import release_opsin_in_use
from orthonym.rules import adducts, metal_atoms, metal_labels
from orthonym.rules.metal_atoms import admitted_metal_atom_name, inorganic_anion_word, lone_atom_name, metal_atom_name
from tests.support.rt_assert import assert_full_rt, name_is_rt_exact
from tests.support.pin_tiers import tier_row

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


def _view(row):
    return (row["name"], row["tier"], row["is_pin"], row["limit_code"])


def _class_of(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return ClassFirstRouter().dispatch(mol, smiles, Chem.MolToSmiles(mol), _skip_decomposition=False,
                                       _style="pin").class_id


def _assert_named_without_a_pin(smiles, name, tiers=("best-effort", "valid")):
    for tier in tiers:
        row = tier_row(smiles, tier)
        assert row["name"] == name, (tier, row)
        assert row["tier"] == "systematic_verified" and row["is_pin"] is False, (tier, row)
        assert_full_rt(row["name"], smiles)
    declined = tier_row(smiles, "pin")                    # the PIN tier declines it, with a reason
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
    assert tier_row("[Zn+2]", "best-effort")["name"] == "zinc(2+)"


def test_the_isotope_and_the_charge_are_in_the_name():
    """0-wrong: a name that drops or changes the nuclide or the charge reads back as another structure."""
    assert name_is_rt_exact("zinc", "[65Zn]") is False
    assert name_is_rt_exact("(66Zn)zinc", "[65Zn]") is False
    assert name_is_rt_exact("(65Zn)zinc(2+)", "[66Zn+2]") is False
    assert name_is_rt_exact("zinc(2+)", "[66Zn+2]") is False
    assert tier_row("[66Zn+2]", "best-effort")["name"] == "(66Zn)zinc(2+)"


def test_the_class_of_the_dispatch_table_names_a_lone_atom_before_the_radical_and_retained_classes():
    assert _class_of("[Zn]") == StoutClass.ELEMENT_ATOM
    assert _class_of("[Zn+2]") == StoutClass.ELEMENT_ATOM         # not CATION_RETAINED, whose word is 'zinc'
    assert _class_of("[Cu+2]") == StoutClass.CATION_RETAINED      # the retained name with its oxidation number stands
    assert _class_of("[Tc+2]") == StoutClass.ELEMENT_ATOM         # admitted to ELEMENT alone (lane LONE)
    assert _class_of("[Na+]") == StoutClass.ELEMENT_ATOM          # 'sodium' is the word of a salt, no name of the ion
    assert _class_of("[Tl]") == StoutClass.ELEMENT_ATOM           # RDKit gives the bare atom a radical electron
    assert _class_of("[Cl]") == StoutClass.ELEMENT_ATOM
    assert _class_of("[Ne]") == StoutClass.ELEMENT_ATOM
    assert _class_of("[Hg]") == StoutClass.ELEMENT_ATOM
    assert _class_of("[Hf+4]") == StoutClass.ELEMENT_ATOM
    assert _class_of("[Cl-]") != StoutClass.ELEMENT_ATOM          # the retained anion name stands
    assert _class_of("[ZnH]") != StoutClass.ELEMENT_ATOM
    assert _class_of("[CH3]") != StoutClass.ELEMENT_ATOM
    priorities = {c: e.priority for c, e in DISPATCH_TABLE.items()}
    assert priorities[StoutClass.SALT] < priorities[StoutClass.ELEMENT_ATOM] < priorities[StoutClass.RADICAL]


# ---- the producer (pure: no JVM)

@pytest.mark.parametrize("smiles,expected", [
    ("[Zn]", "zinc"),
    ("[Zn+2]", "zinc(2+)"),
    ("[Zn+]", "zinc(1+)"),
    ("[65Zn]", "(65Zn)zinc"),
    ("[66Zn+2]", "(66Zn)zinc(2+)"),
    ("[Zn-]", "zinc(1-)"),            # any other anion: the element name with the charge number
    ("[ZnH]", None),                  # a hydrogen on the metal: a hydride, named by its ligands (BONDED_NAMES)
    ("[ZnH+]", None),
    ("[Zn]=[Zn]", None),              # two atoms
    ("Cl[Zn]Cl", None),
    ("[Zn+2].[Cl-].[Cl-]", None),     # a salt, named by the salt route
    ("[Cu]", "copper"),               # RETAINED_ION_METALS: the atom and the ions without a retained name
    ("[Fe+6]", "iron(6+)"),
    ("[Cu+2]", None),                 # 'copper(II)': CATION_RETAINED names it
    ("[Fe+2]", None),
    ("[Ni+2]", None),
    ("[Ag+]", "silver(1+)"),          # the table word 'silver' carries no oxidation number: a salt word only
    ("[Na+]", "sodium(1+)"),
    ("[Tc+2]", "technetium(2+)"),     # ELEMENT_ONLY_METALS
    ("[Pu]", "plutonium"),
    ("[Hg+2]", "mercury(2+)"),
    ("[U+4]", "uranium(4+)"),
    ("[Sn]", "tin"),
    ("[Sn+4]", None),                 # 'tin(IV)'
    ("[Tl]", "thallium"),
    ("[Si]", "silicon"),              # a metalloid outside RADICAL_PARENT_HYDRIDES: the element name
    ("[C]", "carbon"),
    ("[13C]", "(13C)carbon"),
    ("[Ne]", "neon"),
    ("[Kr+]", "krypton(1+)"),
    ("[222Rn]", "(222Rn)radon"),
    ("[At]", "astatine"),             # OPSIN reads no 'astanyl'
    #: the bare atom of a mononuclear parent hydride is its radical
    ("[Cl]", "chloranyl"),
    ("[36Cl]", "(36Cl)chloranyl"),
    ("[O]", "oxidanylidene"),
    ("[N]", "azanylidyne"),
    ("[B]", "boranylidyne"),
    ("[Bi]", "bismuthanylidyne"),
    ("[Po]", "polanylidene"),
    ("[Cl+]", "chloraniumylidene"),   # +: a hydron added, every hydrogen atom taken away
    ("[P-3]", None),                  # a closed-shell anion: the anion classes name it ('phosphanetriide')
    ("[N-]", "azanidylidene"),
    ("[O-]", "oxidanidyl"),
    ("[Si+]", "silyliumylidyne"),
    ("[Si+4]", "silanetetraylium"),
    ("[Se+6]", "selenium(6+)"),     # no hydride form accounts for the charge: the element name
    ("[18O-2]", "(18O)oxidanediide"),
    ("[59Fe+3]", None),               # the isotope decorator gives "(59Fe)iron(III)"
    # anions: the retained anion names stand; an isotope takes the parent-hydride anion of (2)
    ("[Cl-]", None),
    ("[S-2]", None),
    ("[36Cl-]", "(36Cl)chloranide"),
    ("[125I-]", "(125I)iodanide"),
    ("[B-]", "boron(1-)"),
    #
    ("[H]", "hydrogen"),
    ("[H-]", "hydride"),
    ("[2H-]", "deuteride"),
    ("[3H+]", "triton"),
    ("[2H]", "deuterium"),
    ("C", None),
    ("[NH4+]", None),
])
def test_the_producer_names_a_lone_atom_and_nothing_else(smiles, expected):
    assert lone_atom_name(Chem.MolFromSmiles(smiles)) == expected


def test_the_producer_follows_the_scope_table(monkeypatch):
    """Admitting a metal to ELEMENT is an edit of the table; the element name is the one the adduct
    components use."""
    assert lone_atom_name(Chem.MolFromSmiles("[Tc+2]")) == "technetium(2+)"
    monkeypatch.setitem(metal_scope.METAL_SCOPE, "Tc", MetalScope(frozenset({CARBANION_SALTS}), frozenset({2})))
    assert lone_atom_name(Chem.MolFromSmiles("[Tc+2]")) is None
    assert admitted_metal_atom_name(Chem.MolFromSmiles("[Tc+2]")) is None
    monkeypatch.setitem(metal_scope.METAL_SCOPE, "Zn", MetalScope(frozenset({CARBANION_SALTS}), frozenset({2})))
    assert lone_atom_name(Chem.MolFromSmiles("[Zn+2]")) is None


def test_every_true_metal_is_in_the_scope_table():
    """The table decides for every metal: a metal outside it would be named without an admission."""
    from orthonym.perception.metals import _TRUE_METAL_SYMBOLS_FOR_VETO
    assert not [s for s in _TRUE_METAL_SYMBOLS_FOR_VETO if s not in metal_scope.METAL_SCOPE]
    assert all(metal_scope.has_mechanism(s, ELEMENT) for s in _TRUE_METAL_SYMBOLS_FOR_VETO)


def test_an_element_mechanism_alone_does_not_name_complexes():
    """The failure label of an assembly reads ``is_named_metal``: ELEMENT names no complex or salt."""
    assert not metal_scope.is_named_metal("Tc") and not metal_scope.is_named_metal("Fm")
    assert metal_scope.is_named_metal("Na") and metal_scope.is_named_metal("Zn")


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
    f = metal_labels.is_lone_atom_name_without_pin
    assert f("[Zn]", "zinc") and f("[Zn+2]", "zinc(2+)") and f("[65Zn]", "(65Zn)zinc")
    assert f("[Ne]", "neon") and f("[Na+]", "sodium(1+)") and f("[Se+6]", "selenium(6+)") and f("[H-]", "hydride")
    # on a parent hydride: no PIN either, the drawing has no carbon (changed asserted value 2026-11-02, was False:
    #:2056 "... and that also contain at least one carbon atom in their structure";:2058 preselected name)
    assert f("[N]", "azanylidyne") and f("[36Cl-]", "(36Cl)chloranide")
    assert not f("[Cu+2]", "copper(II)")                                   # the retained name keeps its label
    assert not f("[Cl-]", "chloride")
    assert not f("[Zn+2].[Cl-].[Cl-]", "zinc dichloride")                  # no lone atom (the carbon-free rule of the namer)
    assert not f("[H-].[H-].[Zn+2]", "zinc dihydride")
    assert not f("CC[Zn]CC", "diethylzinc")
    assert not f("[Zn+2]", "") and not f("", "zinc(2+)") and not f("[Zn+2]", None)
    assert metal_labels.is_metal_name_without_pin("[Zn+2]", "zinc(2+)") is True


def test_the_predicate_is_what_lowers_the_label(monkeypatch):
    """Switched off, together with the namer's carbon-free demotion (which holds every lone-atom name since
    2026-11-02), the name of the ion would be a path label (pin_verified): the predicate is on the path of the
    tier row."""
    import orthonym.namer as namer
    monkeypatch.setattr(metal_labels, "is_lone_atom_name_without_pin", lambda smiles, name: False)
    monkeypatch.setattr(namer, "_is_carbon_free_compound", lambda smiles: False)
    row = tier_row("[Zn+2]", "valid")
    assert row["name"] == "zinc(2+)" and row["tier"] == "pin_verified", row


# ---- the hydride ion beside zinc (panel B9 ionic)

def test_zinc_dihydride_is_named_and_read_back():
    """``[H-].[H-].[Zn+2]``: the salt route cites 'hydride' for the ion beside zinc. Changed asserted labels
    (2026-11-02; was pin_verified at every tier): a carbon-free salt has a preselected name at most, never a PIN
     PREFERRED IUPAC NAMES, the Blue Book, "... and that also contain at least one carbon atom in
    their structure";:2058), so the PIN tier declines it, as 'zinc dichloride'."""
    smiles = "[H-].[H-].[Zn+2]"
    _assert_named_without_a_pin(smiles, "zinc dihydride")
    assert tier_row("[Zn+2].[Cl-].[Cl-]", "pin")["limit_code"] == "NO_VERIFIED_PIN"   # as zinc dichloride
    assert tier_row("[Zn+2].[Zn+2].[H-].[H-].[H-].[H-]", "best-effort")["name"] == "dizinc tetrahydride"
    assert_full_rt("dizinc tetrahydride", "[Zn+2].[Zn+2].[H-].[H-].[H-].[H-]")


def test_the_hydride_word_is_for_the_metals_the_table_admits():
    zinc = parse_salt_fragments(Chem.MolFromSmiles("[H-].[H-].[Zn+2]"))
    sodium = parse_salt_fragments(Chem.MolFromSmiles("[Na+].[H-]"))
    copper = parse_salt_fragments(Chem.MolFromSmiles("[Cu+2].[H-].[H-]"))
    technetium = parse_salt_fragments(Chem.MolFromSmiles("[Tc+2].[H-].[H-]"))
    both = parse_salt_fragments(Chem.MolFromSmiles("[Tc+2].[Zn+2].[H-].[H-].[H-].[H-]"))
    assert inorganic_anion_word("[H-]", zinc["cations"]) == "hydride"
    assert inorganic_anion_word("[H-]", sodium["cations"]) == "hydride"  # Groups 1-2: CARBANION_SALTS since 2026-10-30
    assert inorganic_anion_word("[H-]", copper["cations"]) == "hydride"   # CARBANION_SALTS since 2026-10-30 (lane B2)
    assert inorganic_anion_word("[H-]", technetium["cations"]) is None
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
    row = tier_row(smiles, "best-effort")
    assert row["name"] == "octadecanoic acid—zinc (2/1)"
    assert row["tier"] == "systematic_verified" and row["is_pin"] is False
    assert_full_rt(row["name"], smiles)


def test_a_neutral_atom_beside_anions_is_cited_with_zero():
    """Decision 2 of the user (2026-10-13) declined a producer for a neutral metal atom beside anions;
    User decision 2026-10-30 (breadth lane, levers B and C) replaces it: the atom is cited with the oxidation number zero, at the best-effort tier only
    (``ZERO_VALENT`` of ``data/metal_scope.py``), and the name reads back to the neutral atom drawn."""
    row = tier_row("[Zn].[Cl-].[Cl-]", "best-effort")
    assert row["name"] == "chloride—zinc(0) (2/1)" and row["tier"] == "systematic_verified", row
    assert row["is_pin"] is False
    assert_full_rt(row["name"], "[Zn].[Cl-].[Cl-]")
    assert tier_row("[Zn].[Cl-].[Cl-]", "pin")["tier"] == "abstain"


# ---- the other metals and the other assemblies are not moved (START literals, computed on the start tree)

#: (default tier, best-effort tier) identity views at the start commit of the zinc work; the retained ion names are
#: carbon-free, so no PIN since 2026-11-02 (changed asserted labels, were pin_verified at both tiers::2056
#: "... and that also contain at least one carbon atom in their structure";:2058 preselected name)
START_ROWS = {
    "[Cu+2]": (
        ('unknown organic compound', 'abstain', False, 'NO_VERIFIED_PIN'),
        ('copper(II)', 'systematic_verified', False, None)),
    "[Fe+3]": (
        ('unknown organic compound', 'abstain', False, 'NO_VERIFIED_PIN'),
        ('iron(III)', 'systematic_verified', False, None)),
    "[Fe+2]": (
        ('unknown organic compound', 'abstain', False, 'NO_VERIFIED_PIN'),
        ('iron(II)', 'systematic_verified', False, None)),
    "[Sn+4]": (
        ('unknown organic compound', 'abstain', False, 'NO_VERIFIED_PIN'),
        ('tin(IV)', 'systematic_verified', False, None)),
    "[Tc+2].[H-].[H-]": (
        ('inorganic compound (not supported)', 'abstain', False, 'UNSUPPORTED_ELEMENT'),
        (None, 'abstain', False, 'UNSUPPORTED_ELEMENT')),
}

#: the rows of the start commit that lane B2 moves (``MORE_BONDED_METAL_VALENCES``): the name now, with the label of its
#: tier; each name is read back to the input's full InChIKey. The ion of a metal of ``RETAINED_ION_METALS`` keeps its
#: retained name ('copper(II)': START_ROWS above); the hydride salt of copper is named from its ions at the best-effort
#: tier only (a carbanion or hydride beside a metal is organometallic for it, INTRODUCTION)
MOVED_ROWS = {
    "[Cd+2]": (('unknown organic compound', 'abstain', False, 'NO_VERIFIED_PIN'),
               ('cadmium(2+)', 'systematic_verified', False, None)),
    "[Cu+2].[H-].[H-]": (('copper compound (not supported)', 'abstain', False, 'UNSUPPORTED_ELEMENT'),
                         ('copper(2+)—hydride (1/2)', 'systematic_verified', False, None)),
}


def _opsin_reads(name) -> bool:
    """OPSIN main reads a hydride component of a mixture; the release 2.9.0 does not, and there the name fails closed
    (the read-back cannot confirm it)."""
    from orthonym.namer import _validity_gate_name_to_smiles
    return bool(_validity_gate_name_to_smiles(name))


@pytest.mark.parametrize("smiles", sorted(MOVED_ROWS))
def test_the_rows_that_lane_b2_moves(smiles):
    default_view, best_effort_view = MOVED_ROWS[smiles]
    assert _view(tier_row(smiles, "pin")) == default_view, smiles
    best_effort = tier_row(smiles, "best-effort")
    if not _opsin_reads(best_effort_view[0]):
        assert best_effort["name"] is None and best_effort["tier"] == "abstain", best_effort   # no name, never a wrong one
        return
    assert _view(best_effort) == best_effort_view, smiles
    assert_full_rt(best_effort["name"], smiles)

#: zinc beside a cation that the table does not admit to ``CARBANION_SALTS`` (no hydride word), and a
#: charge-unbalanced assembly (no salt): no producer names them. The failure label is the engine's, UNNAMEABLE
#: (zinc is a complete metal and every metal of the assembly is admitted to the table); the text of the plain
#: call is the descriptive fallback of the metal of lowest atomic number
#: the hydride salts of Groups 1-2 and of zinc: User decision 2026-10-30 (breadth lane, levers B and C) admits Groups 1-2 to ``CARBANION_SALTS``,
#: so these are binary salts; carbon-free, so no PIN (changed asserted labels 2026-11-02, were pin_verified at both
#: tiers::2056 "... and that also contain at least one carbon atom in their structure";:2058 preselected name)
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
    _assert_named_without_a_pin(smiles, ADMITTED_HYDRIDE_ROWS[smiles], tiers=("best-effort",))


UNNAMED_ZINC_ASSEMBLIES = {
    "[Zn+2].[H-]": (
        ('zinc compound (not supported)', 'abstain', False, 'UNNAMEABLE'),
        # a charged assembly: the best-effort adduct of its ions (the hydride component, 2026-10-30)
        ('zinc(2+)—hydride (1/1)', 'systematic_verified', False, None)),
}


@pytest.mark.parametrize("smiles", sorted(UNNAMED_ZINC_ASSEMBLIES))
def test_zinc_assemblies_no_producer_names_are_unnameable(smiles):
    default_view, best_effort_view = UNNAMED_ZINC_ASSEMBLIES[smiles]
    assert _view(tier_row(smiles, "pin")) == default_view, smiles
    assert _view(tier_row(smiles, "best-effort")) == best_effort_view, smiles


#: the rows of the start commit that lane LONE (2026-11-01) moves: every metal is admitted to ``ELEMENT``, and a bare
#: atom that RDKit gives radical electrons reaches the class ELEMENT_ATOM before RADICAL. The PIN tier declines
#: each name (no PIN,: no carbon); the best-effort tier gives it, read back to the full key
LONE_ROWS = {
    "[Cu]": "copper", "[Na+]": "sodium(1+)", "[Na]": "sodium", "[Li+]": "lithium(1+)", "[Mg+2]": "magnesium(2+)",
    "[Ca+2]": "calcium(2+)", "[Tc+2]": "technetium(2+)", "[Ag+]": "silver(1+)", "[Tl]": "thallium", "[Ne]": "neon",
    "[Kr+]": "krypton(1+)", "[199Tl]": "(199Tl)thallium", "[Cl]": "chloranyl", "[O]": "oxidanylidene",
    "[Bi]": "bismuthanylidyne", "[36Cl-]": "(36Cl)chloranide", "[125I-]": "(125I)iodanide", "[2H-]": "deuteride",
    "[H-]": "hydride", "[C]": "carbon", "[At]": "astatine", "[Sn+]": "tin(1+)",
}


@pytest.mark.parametrize("smiles", sorted(LONE_ROWS))
def test_the_rows_that_lane_lone_moves(smiles):
    # the names built on a parent hydride ('chloranyl', '(36Cl)chloranide') are carbon-free too: no PIN since
    # 2026-11-02 (user decision;:2056,:2058), so every row is asserted the same way
    _assert_named_without_a_pin(smiles, LONE_ROWS[smiles], tiers=("best-effort",))


def test_a_hydride_stays_out_of_an_assembly_of_a_metal_the_table_does_not_admit():
    """The hydride ion has a name of its own now; beside technetium (no CARBANION_SALTS) the assembly is still
    organometallic for the metal INTRODUCTION) and gets no adduct name."""
    assert tier_row("[Tc+2].[H-].[H-]", "best-effort")["name"] is None


@pytest.mark.parametrize("smiles,name", [
    ("[B+3].[H-].[H-].[H-]", "boron(3+)—hydride (1/3)"),      # not 'boranetriylium trihydride' through the salt route
    ("[Si+4].[O-2].[O-2]", "oxidanediide—silicon(4+) (2/1)"),
])
def test_the_salt_route_does_not_cite_a_one_atom_non_metal_ion_by_its_free_ion_name(smiles, name):
    """``rules.salts._name_organic_ion_best_effort`` names organic ions; a one-atom ion of hydrogen, carbon or a
    p-block element keeps the adduct name of its drawing (systematic_verified,."""
    row = tier_row(smiles, "best-effort")
    if smiles.startswith("[B+3]") and release_opsin_in_use():
        # OPSIN 2.9.0 reads no 'boron(3+)—hydride (1/3)' (OPSIN's main branch does): the row then has no name, and
        # never the free-ion name of its hydride
        assert (row["name"], row["tier"]) == (None, "abstain"), row
        return
    assert (row["name"], row["tier"]) == (name, "systematic_verified"), row
    assert_full_rt(name, smiles)


@pytest.mark.parametrize("smiles", ["CCO.[H]", "[Na+].[H]"])
def test_a_stray_hydrogen_atom_is_no_adduct_component(smiles):
    assert tier_row(smiles, "best-effort")["name"] is None


@pytest.mark.parametrize("smiles,name", [
    ("[C-4]", "methanetetraide"), ("[14C-4]", "(14C)methanetetraide"), ("[C+4]", "methane-1,1,1,1-tetrakis(ylium)"),
])
def test_carbon_ions_keep_their_names_on_methane(smiles, name):
    """'H3C– methanide (PIN)', the Blue Book): a carbon ion is named on methane and keeps the
    PIN label of (it contains carbon); ``lone_atom_name`` leaves it to the ion classes."""
    assert lone_atom_name(Chem.MolFromSmiles(smiles)) is None
    row = tier_row(smiles, "best-effort")
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row


@pytest.mark.parametrize("plain,labelled", [
    ("[P-3]", "[32P-3]"), ("[O-2]", "[18O-2]"), ("[Fe+2]", "[59Fe+2]"), ("[Cl-]", "[36Cl-]"), ("[Zn+2]", "[65Zn+2]"),
])
def test_an_isotopologue_takes_the_label_of_its_parent(plain, labelled):
    """The label of a lone atom is a function of the structure: a carbon-free name has no PIN, an element
    name or a name on a parent hydride alike; the nuclide changes neither ('(13C)methane (PIN)' and 'methane (PIN)',
    :7354)."""
    a, b = tier_row(plain, "best-effort"), tier_row(labelled, "best-effort")
    assert a["tier"] == b["tier"] and a["is_pin"] == b["is_pin"], (a, b)


@pytest.mark.parametrize("smiles", sorted(START_ROWS))
def test_other_metals_are_not_moved(smiles):
    """The atoms, ions and hydride salts of the metals that the table does not admit keep their rows of the
    start commit."""
    default_view, best_effort_view = START_ROWS[smiles]
    assert _view(tier_row(smiles, "pin")) == default_view, smiles
    assert _view(tier_row(smiles, "best-effort")) == best_effort_view, smiles


def test_the_other_metals_panel_has_no_atom_or_hydride_row_the_class_could_reach():
    """The 73-drawing panel of the other metals is untouched by construction: no row is one atom of
    an admitted metal, and no row holds a hydride ion beside zinc."""
    from tests.support.metal_baselines import baseline_path, read_rows
    for row in read_rows(baseline_path("other")):
        mol = Chem.MolFromSmiles(row["smiles"])
        assert admitted_metal_atom_name(mol) is None, row["smiles"]
        assert not ("[H-]" in row["smiles"] and "Zn" in row["smiles"]), row["smiles"]
