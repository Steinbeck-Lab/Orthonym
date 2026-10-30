"""Zincate and complex-anion drawings, and the ammonia of an ammine, for zinc (``AMMINE`` of
``data/metal_scope.py``, ``rules/metal_atoms.ammine_component_names``, ``rules/adducts.name_adduct``).

The zincates of the plan are named by the producers of the earlier tasks: a bonded zincate is
named through its ionic drawing (``DISCONNECT``), and the ions of a salt are composed
(``ION_SALTS``). These tests pin that state, so that no later task loses a row of the class.

An ammine complex was named at no tier: ``[Zn+2].N.N.N.N`` and its bonded drawing
``[NH3+][Zn-2]([NH3+])([NH3+])[NH3+]`` (which the rewrite of ``DISCONNECT`` turns into the ionic
drawing) are adducts whose component ``N`` the table of single-atom components does not hold, so the
whole adduct was refused ("more often a perception artefact than a genuine ammoniate"). The
component is cited as 'ammonia' when every true metal of the assembly is admitted to ``AMMINE``.

Blue Book: Mixed organic - inorganic adducts (the Blue Book), "preferred IUPAC names
cannot be assigned to mixed adducts because preferred IUPAC names have not yet been determined for
inorganic components", so no such name is a PIN; Retained names (:7954), "The common
names water, ammonia,... are used in these recommendations but their use as preferred IUPAC names
is deferred... thus, no PIN label will be assigned in names including them (see ";
 Definitions of terms (:39781), 'ammine' is the ligand name of NH3 in a coordination name
(OPSIN 2.9.0 does not read it, so the component of an adduct is cited by its own name).

Every name is read back by a fresh OPSIN call that does not go through the engine, to the full
standard InChIKey of the input.
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.data import metal_scope
from orthonym.data.metal_scope import AMMINE, BONDED_METAL_VALENCES, ELEMENT, METAL_SCOPE, MetalScope
from orthonym.rules import adducts, metal_atoms
from orthonym.rules.adducts import SINGLE_ATOM_COMPONENT_NAMES
from orthonym.rules.metal_atoms import ammine_component_names
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


def _row(smiles, tier):
    return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


def _view(row):
    return (row["name"], row["tier"], row["is_pin"], row["limit_code"])


def _assert_named_without_a_pin(smiles, name):
    """Best-effort: the name, a verified systematic label, no PIN, read back. Default tier: no name."""
    row = _row(smiles, "best-effort")
    assert row["name"] == name, row
    assert row["tier"] == "systematic_verified" and row["is_pin"] is False, row
    assert_full_rt(row["name"], smiles)
    declined = _row(smiles, "pin")
    assert declined["tier"] == "abstain" and declined["is_pin"] is False, declined


# ---- the ammines of zinc (panel D1, both drawings)

AMMINES = [
    # ionic drawing, the panel's D1 ionic
    ("[Zn+2].N.N.N.N", "ammonia—zinc(2+) (4/1)"),
    # bonded drawing: the Zn-N bonds are heterolysed (DISCONNECT) and the ionic drawing is named
    ("[NH3+][Zn-2]([NH3+])([NH3+])[NH3+]", "ammonia—zinc(2+) (4/1)"),
    ("[Zn+2].N", "ammonia—zinc(2+) (1/1)"),
    ("[Zn+2].N.N.N.N.N.N", "ammonia—zinc(2+) (6/1)"),
    # an ammine with other ions: the salt route declines a neutral co-former, the adduct names all
    ("[Zn+2].[Cl-].[Cl-].N.N", "ammonia—chloride—zinc(2+) (2/2/1)"),
    ("N.N.[Zn+2].CC(=O)[O-].CC(=O)[O-]", "acetate—ammonia—zinc(2+) (2/2/1)"),
    ("N.N.N.N.[Zn+2].[O-]S(=O)(=O)[O-]", "sulfate—ammonia—zinc(2+) (1/4/1)"),
    # a covalent drawing of zinc dichloride beside ammonia: the ionic drawing is named
    ("N.N.Cl[Zn]Cl", "ammonia—chloride—zinc(2+) (2/2/1)"),
    # neutral fragments and a zinc atom: the producer of the adduct names them with the new component
    ("CC(=O)O.N.[Zn]", "acetic acid—ammonia—zinc (1/1/1)"),
    ("CCO.N.Cl.[Zn]", "ethanol—hydrogen chloride—ammonia—zinc (1/1/1/1)"),
]


@pytest.mark.parametrize("smiles,name", AMMINES)
def test_an_ammine_of_zinc_is_named_and_read_back(smiles, name):
    _assert_named_without_a_pin(smiles, name)


def test_the_two_drawings_of_the_ammine_have_one_name():
    """Both drawings of D1 share the standard InChIKey; the bonded one is named through the ionic one."""
    a = _row("[Zn+2].N.N.N.N", "best-effort")
    b = _row("[NH3+][Zn-2]([NH3+])([NH3+])[NH3+]", "best-effort")
    assert a["name"] == b["name"] and a["tier"] == b["tier"] == "systematic_verified"


def test_the_ammonia_component_is_cited_as_ammonia():
    """ (:7954): the common name ammonia is used in these recommendations; the table of
    the other single-atom components cites water and hydrogen chloride alike."""
    assert metal_atoms.AMMINE_COMPONENT_NAMES == {"N": "ammonia"}
    assert "N" not in SINGLE_ATOM_COMPONENT_NAMES        # the table of the other metals is unchanged


# ---- the guard: AMMINE of every true metal of the assembly

@pytest.mark.parametrize("smiles,expected", [
    ("[Zn+2].N.N.N.N", {"N": "ammonia"}),
    ("N.N.Cl[Zn]Cl", {"N": "ammonia"}),
    ("[Zn]", {"N": "ammonia"}),                         # the guard reads the metals, not the fragments
    ("[NH3+][Zn-2]([NH3+])([NH3+])[NH3+]", {"N": "ammonia"}),
    ("N.N.O", {}),                                      # no metal
    ("[NH4+].[Cl-]", {}),
    ("[Na+].[Cl-].N", {}),                              # a metal that the table does not admit to AMMINE
    ("N.N.Cl[Cd]Cl", {}),
    ("N.N.Cl[Pt]Cl", {"N": "ammonia"}),                 # platinum is admitted since 2026-10-30
    ("N.N.N.N.[Cu+2]", {}),
    ("[Zn+2].[Cu+2].N", {}),                            # one metal outside: the whole assembly is outside
    ("[Zn+2].[Na+].N", {}),
    ("[Si].N", {}),                                     # a metalloid is no true metal
])
def test_the_guard_reads_the_table_for_every_metal_of_the_assembly(smiles, expected):
    assert ammine_component_names(Chem.MolFromSmiles(smiles)) == expected


def test_the_guard_follows_the_scope_table(monkeypatch):
    """Admitting a metal to AMMINE is an edit of the table: copper alone, then copper with zinc."""
    mol = Chem.MolFromSmiles("N.N.N.N.[Cu+2]")
    both = Chem.MolFromSmiles("N.N.[Cu+2].[Zn+2]")
    assert ammine_component_names(mol) == {} and ammine_component_names(both) == {}
    monkeypatch.setitem(metal_scope.METAL_SCOPE, "Cu", MetalScope(frozenset({ELEMENT, AMMINE}), frozenset({2})))
    assert ammine_component_names(mol) == {"N": "ammonia"}
    assert ammine_component_names(both) == {"N": "ammonia"}
    monkeypatch.setitem(metal_scope.METAL_SCOPE, "Zn", MetalScope(frozenset({ELEMENT}), frozenset({2})))
    assert ammine_component_names(Chem.MolFromSmiles("[Zn+2].N")) == {}


def test_the_adduct_reads_the_guard_in_the_unverified_call_only():
    """The widening of the adduct (a one-atom component beside a metal) is made in a call with
    ``general_fallback_unverified`` (the best-effort tier): a plain call of ``name_adduct`` (the PIN
    tier) names no ammine."""
    mol = Chem.MolFromSmiles("[Zn+2].N.N.N.N")
    assert adducts.name_adduct(mol, Chem.MolToSmiles(mol)) is None
    assert adducts.name_adduct(
        mol, Chem.MolToSmiles(mol), general_fallback=True, general_fallback_unverified=True,
        allow_aromatic_general=True) == "ammonia—zinc(2+) (4/1)"


# ---- the labels: no PIN, systematic_verified, by the adduct predicate

def test_the_ammine_names_carry_no_pin_label():
    for smiles, _name in AMMINES:
        row = _row(smiles, "best-effort")
        assert row["tier"] == "systematic_verified" and row["is_pin"] is False, (smiles, row)


# ---- the zincates and the complex anions are named by the producers of the earlier tasks

ZINCATES = [
    # a bare anion, drawn bonded: the ionic drawing is named, as an adduct
    ("Cl[Zn-2](Cl)(Cl)Cl", "chloride—zinc(2+) (4/1)"),
    ("Cl[Zn-](Cl)Cl", "chloride—zinc(2+) (3/1)"),
    ("Br[Zn-](Br)Br", "bromide—zinc(2+) (3/1)"),
    # a zincate beside cations: the salt of the ionic drawing
    ("[Na+].[Na+].Cl[Zn-2](Cl)(Cl)Cl", "disodium zinc tetrachloride"),
    ("[K+].Cl[Zn-](Cl)Cl", "potassium zinc trichloride"),
    ("[Na+].[Na+].O[Zn-2](O)(O)O", "disodium zinc tetrahydroxide"),
    ("[NH4+].[NH4+].Cl[Zn-2](Cl)(Cl)Cl", "diammonium zinc tetrachloride"),
    # an organic cation beside the zincate (the class C10 of the census)
    ("C[N+](C)(C)C.C[N+](C)(C)C.Cl[Zn-2](Cl)(Cl)Cl", "bis(N,N,N-trimethylmethanaminium) zinc tetrachloride"),
    ("C[N+](C)(C)C.Cl[Zn-](Cl)Cl", "N,N,N-trimethylmethanaminium zinc trichloride"),
    ("c1cc[nH+]cc1.c1cc[nH+]cc1.Cl[Zn-2](Cl)(Cl)Cl", "bis(pyridin-1-ium) zinc tetrachloride"),
]


@pytest.mark.parametrize("smiles,name", ZINCATES)
def test_a_zincate_is_named_and_read_back(smiles, name):
    """A drawing as written is named through its ionic drawing: the drawing as written is
    named too, and the better name ships), so the label is that of a rewritten drawing."""
    row = _row(smiles, "best-effort")
    assert row["name"] == name, row
    assert row["tier"] == "systematic_verified" and row["is_pin"] is False, row
    assert_full_rt(row["name"], smiles)
    assert _row(smiles, "pin")["tier"] == "abstain"


def test_an_ionic_zincate_keeps_the_label_of_a_carbon_free_salt():
    """The named exception of the label policy (user answer 3 of 2026-10-13): a carbon-free salt
    keeps the label of its path, as 'sodium chloride' does."""
    smiles = "[Na+].[Na+].[Cl-].[Cl-].[Cl-].[Cl-].[Zn+2]"
    for tier in ("pin", "best-effort"):
        row = _row(smiles, tier)
        assert row["name"] == "disodium zinc tetrachloride", (tier, row)
        assert row["tier"] == "pin_verified" and row["is_pin"] is True, (tier, row)
    assert_full_rt("disodium zinc tetrachloride", smiles)


def test_no_name_of_a_bare_zincate_is_built_by_a_second_producer():
    """The as-written name of the bare anion (``tetrachlorozinc(2-)``, which OPSIN reads) is not built:
    the ionic drawing ties with it at ``systematic_verified`` and a tie ships the rewritten name, so a
    second producer would change no name. The class has no unnamed row among the fixtures."""
    smiles = "Cl[Zn-2](Cl)(Cl)Cl"
    assert_full_rt("tetrachlorozinc(2-)", smiles)               # a name that exists
    assert _row(smiles, "best-effort")["name"] == "chloride—zinc(2+) (4/1)"


# ---- the other metals and the other assemblies are not moved (START literals, computed on the start tree)

START_ROWS = {
    "N.N.O": (
        ('inorganic compound (not supported)', 'abstain', False, 'UNSUPPORTED_ELEMENT'),
        (None, 'abstain', False, 'UNSUPPORTED_ELEMENT')),
    "[NH4+].[Cl-]": (
        ('ammonium chloride', 'pin_verified', True, None),
        ('ammonium chloride', 'pin_verified', True, None)),
    "[Na+].[Cl-].N": (
        ('sodium compound (not supported)', 'abstain', False, 'UNSUPPORTED_ELEMENT'),
        (None, 'abstain', False, 'UNSUPPORTED_ELEMENT')),
    # the platinum rows of the start commit are admitted since 2026-10-30 (``ADMITTED_PLATINUM_AMMINES``): cadmium
    "N.N.Cl[Cd]Cl": (
        ('cadmium compound (not supported)', 'abstain', False, 'UNSUPPORTED_ELEMENT'),
        (None, 'abstain', False, 'UNSUPPORTED_ELEMENT')),
    "N.N.[Cl-].[Cl-].[Cd+2]": (
        ('cadmium compound (not supported)', 'abstain', False, 'UNSUPPORTED_ELEMENT'),
        (None, 'abstain', False, 'UNSUPPORTED_ELEMENT')),
    "N.N.N.N.[Cu+2]": (
        ('copper compound (not supported)', 'abstain', False, 'UNSUPPORTED_ELEMENT'),
        (None, 'abstain', False, 'UNSUPPORTED_ELEMENT')),
    "N.N.N.N.[Ni+2]": (
        ('nickel compound (not supported)', 'abstain', False, 'UNSUPPORTED_ELEMENT'),
        (None, 'abstain', False, 'UNSUPPORTED_ELEMENT')),
    "N.N.N.N.N.N.[Co+3]": (
        ('cobalt compound (not supported)', 'abstain', False, 'UNSUPPORTED_ELEMENT'),
        (None, 'abstain', False, 'UNSUPPORTED_ELEMENT')),
    "[Mg+2].N.N.N.N.N.N.[Cl-].[Cl-]": (
        ('magnesium compound (not supported)', 'abstain', False, 'UNSUPPORTED_ELEMENT'),
        (None, 'abstain', False, 'UNSUPPORTED_ELEMENT')),
    "N.N.N.N.[Cu+2].[O-]S(=O)(=O)[O-]": (
        ('copper compound (not supported)', 'abstain', False, 'UNSUPPORTED_ELEMENT'),
        (None, 'abstain', False, 'UNSUPPORTED_ELEMENT')),
    "CC(=O)O.N.[Cu]": (
        ('copper compound (not supported)', 'abstain', False, 'UNSUPPORTED_ELEMENT'),
        (None, 'abstain', False, 'UNSUPPORTED_ELEMENT')),
    "CC(=O)O.N.[Na]": (
        ('sodium compound (not supported)', 'abstain', False, 'UNSUPPORTED_ELEMENT'),
        (None, 'abstain', False, 'UNSUPPORTED_ELEMENT')),
    # zinc beside a metal that the table does not admit to AMMINE: the ammonia is not cited
    "[Zn+2].[Cu+2].N.N.N.N.N.N.N.N": (
        ('copper compound (not supported)', 'abstain', False, 'UNSUPPORTED_ELEMENT'),
        (None, 'abstain', False, 'UNSUPPORTED_ELEMENT')),
    "[Zn+2].[Cd+2].N.N": (
        ('zinc compound (not supported)', 'abstain', False, 'UNSUPPORTED_ELEMENT'),
        (None, 'abstain', False, 'UNSUPPORTED_ELEMENT')),
}

#: the platinum rows of ``START_ROWS`` at the start commit, admitted to AMMINE since 2026-10-30
#: (``data.metal_scope.BONDED_METAL_VALENCES``): the best-effort adduct of their components, and at the default
#: tier the label of an assembly whose every metal is admitted and one complete (zinc) is UNNAMEABLE
ADMITTED_PLATINUM_AMMINES = {
    "N.N.Cl[Pt]Cl": (
        ('platinum compound (not supported)', 'abstain', False, 'UNSUPPORTED_ELEMENT'),
        ('ammonia—chloride—platinum(2+) (2/2/1)', 'systematic_verified', False, None)),
    "N.N.[Cl-].[Cl-].[Pt+2]": (
        ('platinum compound (not supported)', 'abstain', False, 'UNSUPPORTED_ELEMENT'),
        ('ammonia—chloride—platinum(2+) (2/2/1)', 'systematic_verified', False, None)),
    "[Zn+2].[Pt+2].N.N": (
        ('zinc compound (not supported)', 'abstain', False, 'UNNAMEABLE'),
        ('ammonia—platinum(2+)—zinc(2+) (2/1/1)', 'systematic_verified', False, None)),
}


@pytest.mark.parametrize("smiles", sorted(ADMITTED_PLATINUM_AMMINES))
def test_the_platinum_ammines_are_named(smiles):
    default_view, best_effort_view = ADMITTED_PLATINUM_AMMINES[smiles]
    assert _view(_row(smiles, "pin")) == default_view, smiles
    assert _view(_row(smiles, "best-effort")) == best_effort_view, smiles
    assert_full_rt(best_effort_view[0], smiles)


@pytest.mark.parametrize("smiles", sorted(START_ROWS))
def test_other_metals_are_not_moved(smiles):
    """The ammines of the metals that the table does not admit, and the assemblies without a metal, keep
    their rows of the start commit."""
    default_view, best_effort_view = START_ROWS[smiles]
    assert _view(_row(smiles, "pin")) == default_view, smiles
    assert _view(_row(smiles, "best-effort")) == best_effort_view, smiles


def test_the_other_metals_panel_holds_no_row_the_guard_could_reach():
    """The 73-drawing panel of the other metals is untouched by construction: no assembly of it is
    admitted to AMMINE (every one holds a metal other than zinc and the metals admitted since 2026-10-30, or
    none)."""
    from tests.support.metal_baselines import baseline_path, read_rows
    for row in read_rows(baseline_path("other")):
        mol = Chem.MolFromSmiles(row["smiles"])
        if any(a.GetSymbol() in BONDED_METAL_VALENCES for a in mol.GetAtoms()):
            continue
        assert ammine_component_names(mol) == {}, row["smiles"]


def test_zinc_and_the_bonded_metals_alone_are_admitted_to_ammine():
    assert sorted(s for s, sc in METAL_SCOPE.items() if AMMINE in sc.mechanisms) == sorted(["Zn", *BONDED_METAL_VALENCES])
