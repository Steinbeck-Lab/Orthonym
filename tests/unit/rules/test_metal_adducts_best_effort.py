"""Breadth Job 2: best-effort names for disconnected metal depictions.

A metal atom or ion drawn as its own fragment beside organic molecules or ions
('CC(N)C(=O)O.CC(N)C(=O)O.[Ni]', six cyanides and [Fe+2]) is named as a
adduct of its components, at the best-effort tier only (the user's decision,
2026-09-28): the metal atom takes its element name, a metal cation adds its charge
number ('iron(2+)', the notation of 'pentaammine(ethanido)osmium(1+) chloride',
the Blue Book), and a neutral metal halide or oxide with no name of its own
takes an additive name ('dichloropalladium').

 "Mixed organic - inorganic adducts" (the Blue Book): "preferred IUPAC
names cannot be assigned to mixed adducts because preferred IUPAC names have not yet
been determined for inorganic components". So the PIN tier keeps abstaining and the
best-effort label is below PIN.

Such a name ships only when OPSIN reads it back to exactly the drawn structure
(``adducts._parse_reproduces_depiction``: equal canonical SMILES, which sees charges
and hydrogens that an equal standard InChIKey does not), then after the full-key round
trip. Every expected name below is read back again here by a FRESH OPSIN call that
does not go through the engine (``tests.support.rt_assert``). The witnesses are
milestone1500 best-effort gap rows (breadth plan classes M17, M17b, M16).
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.errors import is_failure_name
from orthonym.jvm_budget import jvm_slots
from tests.support.rt_assert import _independent_parse, name_is_rt_exact

pytestmark = [pytest.mark.opsin_gate]


def _best_effort_row(smiles):
    with jvm_slots(1, purpose="breadth-job2-test"):
        return Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(smiles)


def _pin_row(smiles):
    with jvm_slots(1, purpose="breadth-job2-test"):
        return Orthonym().name_tiered(smiles)


def _canon(smiles):
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return Chem.MolToSmiles(mol) if mol is not None else None


def _assert_reads_back_exactly(name, smiles):
    parsed = _independent_parse(name)
    assert parsed and _canon(parsed) == _canon(smiles), (
        f"{name!r} reads back as {parsed!r}, not as the drawn {smiles}")
    assert name_is_rt_exact(name, smiles), f"{name!r}: no full-key round trip"


# M17: a metal atom or ion beside organic molecules or ions.
M17 = [
    ("CC(C(=O)O)N.CC(C(=O)O)N.[Ni]", "2-aminopropanoic acid—nickel (2/1)"),
    ("[C-]#N.[C-]#N.[C-]#N.[C-]#N.[C-]#N.[C-]#N.[Fe+2]",
     "cyanide—iron(2+) (6/1)"),
    ("CCCCCCCCCCCCCCO.CCCCCCCCCCCCCCO.CCCCCCCCCCCCCCO.CC(C)O.[Ti]",
     "tetradecan-1-ol—propan-2-ol—titanium (3/1/1)"),
    ("[NH2-].[NH2-].[NH2-].[NH2-].[NH2-].[Ru+2]", "azanide—ruthenium(2+) (5/1)"),
    ("C([C@H]([C@H]([C@@H]([C@@H](CO)O)O)O)O)O.[99Tc]",
     "(2R,3R,4R,5R)-hexane-1,2,3,4,5,6-hexol—(99Tc)technetium (1/1)"),
]
# M16: salts the salt namer cannot finish (a multiplied ester anion OPSIN cannot
# read, a charge-imbalanced drawing) name the same way.
M16 = [
    ("CCCCCCCCCCCCCCCCOP(=O)([O-])OCCCCCCCCCCCCCCCC."
     "CCCCCCCCCCCCCCCCOP(=O)([O-])OCCCCCCCCCCCCCCCC."
     "CCCCCCCCCCCCCCCCOP(=O)([O-])OCCCCCCCCCCCCCCCC.[Al+3]",
     # (c) (the Blue Book): a simple prefix "beginning with a
     # multiplicative prefix" takes parentheses when multiplied, 'di(dodecyl)
     # (preferred prefix)'; 'hexadecyl' begins with 'hexa'.
     "di(hexadecyl) phosphate—aluminium(3+) (3/1)"),
    ("C(C(=O)[O-])S.[Ca+2]", "sulfanylacetate—calcium(2+) (1/1)"),
]
# M17b: a metal halide component (the Hg row: test_group12_organometallic_ligands.py).
M17B = [
    ("C1CC=CCCC=C1.Cl[Pd]Cl", "cycloocta-1,5-diene—dichloropalladium (1/1)"),
]


@pytest.mark.parametrize("smiles,expected", M17 + M16 + M17B)
def test_best_effort_names_the_drawn_metal_structure(smiles, expected):
    row = _best_effort_row(smiles)
    name = row.get("name")
    assert name == expected, f"{smiles}: best-effort name {name!r}"
    _assert_reads_back_exactly(name, smiles)
    # Below PIN, the Blue Book;,:39735).
    assert row["tier"] in ("systematic_verified", "best_effort"), (name, row["tier"])
    assert row["is_pin"] is False


@pytest.mark.parametrize("smiles", [s for s, _ in M17 + M16 + M17B])
def test_pin_tier_still_abstains_on_the_metal_structure(smiles):
    row = _pin_row(smiles)
    assert row["tier"] == "abstain" and is_failure_name(row.get("name") or "unknown"), (
        f"{smiles}: PIN tier shipped {row.get('name')!r} ({row['tier']})")


def test_a_charge_form_the_name_does_not_draw_is_refused():
    """The depiction check sees what an equal standard InChIKey does not.

    OPSIN reads 'glycine' as the neutral amino acid; the zwitterion drawn here has the
    same standard InChIKey (mobile H), so only the canonical-SMILES comparison tells
    them apart. Such a name must never ship for the zwitterion drawing."""
    from rdkit.Chem import inchi
    from orthonym.rules.adducts import _parse_reproduces_depiction
    zwitterion = Chem.MolFromSmiles("[NH3+]CC(=O)[O-].[Na]")
    neutral = Chem.MolFromSmiles("NCC(=O)O.[Na]")
    assert inchi.MolToInchiKey(zwitterion) == inchi.MolToInchiKey(neutral)
    assert _parse_reproduces_depiction("glycine—sodium (1/1)", neutral)
    assert not _parse_reproduces_depiction("glycine—sodium (1/1)", zwitterion)


def test_opsin_charge_balancing_is_not_shipped():
    """OPSIN balances an adduct's charges itself: next to [Fe+2] it reads
    'oxovanadium' as [V+3]=O. The engine's name for this drawing would read back as a
    different structure, so the row abstains rather than ship it."""
    smiles = "OCCN(CCO)CCO.O=[V].[Fe+2]"
    row = _best_effort_row(smiles)
    assert row["tier"] == "abstain", (row.get("name"), row["tier"])


class TestComponentTables:
    def test_metal_atom_and_ion_names(self):
        from orthonym.rules.adducts import _metal_atom_component_name as f
        assert f(Chem.MolFromSmiles("[Ni]")) == "nickel"
        assert f(Chem.MolFromSmiles("[Fe+2]")) == "iron(2+)"
        assert f(Chem.MolFromSmiles("[Cs+]")) == "caesium(1+)"
        assert f(Chem.MolFromSmiles("[99Tc]")) == "(99Tc)technetium"
        assert f(Chem.MolFromSmiles("[NaH]")) is None      # a hydride, not an atom
        assert f(Chem.MolFromSmiles("[Co-]")) is None      # no metal '-ide' here
        assert f(Chem.MolFromSmiles("[Cl-]")) is None      # not a metal

    def test_the_table_covers_the_metal_element_set(self):
        from orthonym.perception.metals import METAL_ELEMENT_SYMBOLS
        from orthonym.rules.adducts import _METAL_ELEMENT_NAMES
        assert set(_METAL_ELEMENT_NAMES) == set(METAL_ELEMENT_SYMBOLS)

    def test_metal_halide_oxide_names(self):
        from orthonym.rules.adducts import _metal_halide_oxide_component_name as f
        assert f(Chem.MolFromSmiles("Cl[Pd]Cl")) == "dichloropalladium"
        assert f(Chem.MolFromSmiles("O=[V]")) == "oxovanadium"
        assert f(Chem.MolFromSmiles("[Cu]I")) == "iodocopper"
        assert f(Chem.MolFromSmiles("O=[V](Cl)(Cl)Cl")) == "trichlorooxovanadium"
        # Group 13-16 elements have parent hydrides: not named here.
        assert f(Chem.MolFromSmiles("Cl[Si](Cl)(Cl)Cl")) is None
        assert f(Chem.MolFromSmiles("ClB(Cl)Cl")) is None
        # anything but terminal halogen / oxo ligands, a charge, a hydrogen: None.
        assert f(Chem.MolFromSmiles("O[Cu]")) is None
        assert f(Chem.MolFromSmiles("Cl[Pd-]Cl")) is None
        assert f(Chem.MolFromSmiles("C[Pd]Cl")) is None

    def test_default_component_call_keeps_the_old_scope(self):
        """Without the best-effort switch a bare metal is still no component."""
        from orthonym.rules.adducts import _name_component
        assert _name_component("[Fe]", "pin") is None
        assert _name_component("[Fe+2]", "pin", charged_ok=True) is None


# A one-atom ion ('[Cl-]', '[I-]', '[Br-]', '[OH-]') is an inorganic component too:
# "Mixed organic - inorganic adducts" (the Blue Book;:4665 "organic
# components in order as described in, inorganic components in order as
# described in Ref 12";:4667 no PIN). A metal-free depiction with such an ion is
# named at the best-effort tier the same way as one with a metal atom, and only when
# the name reproduces the drawn charges.
ONE_ATOM_ION = [
    ("CCO.C[N+](C)(C)C.[Cl-]", "ethanol—N,N,N-trimethylmethanaminium—chloride (1/1/1)"),
    ("CC(=O)O.C[n+]1ccccc1.[I-]", "acetic acid—1-methylpyridin-1-ium—iodide (1/1/1)"),
    ("c1ccccc1.CCCC[N+](CCCC)(CCCC)CCCC.[Br-]",
     "N,N,N-tributylbutan-1-aminium—benzene—bromide (1/1/1)"),
    ("CC(=O)OCC.C[N+](C)(C)C.[Cl-].[Cl-]",
     "ethyl acetate—N,N,N-trimethylmethanaminium—chloride (1/1/2)"),
]


@pytest.mark.parametrize("smiles,expected", ONE_ATOM_ION)
def test_best_effort_names_a_one_atom_ion_without_a_metal(smiles, expected):
    row = _best_effort_row(smiles)
    name = row.get("name")
    assert name == expected, f"{smiles}: best-effort name {name!r}"
    _assert_reads_back_exactly(name, smiles)
    assert row["tier"] in ("systematic_verified", "best_effort"), (name, row["tier"])
    assert row["is_pin"] is False


@pytest.mark.parametrize("smiles", [s for s, _ in ONE_ATOM_ION])
def test_pin_tier_still_abstains_on_a_one_atom_ion_adduct(smiles):
    row = _pin_row(smiles)
    assert row["tier"] == "abstain" and is_failure_name(row.get("name") or "unknown"), (
        f"{smiles}: PIN tier shipped {row.get('name')!r} ({row['tier']})")


def test_one_atom_ion_component_stays_out_of_the_default_scope():
    """Without the best-effort switch a one-atom ion is still no adduct component."""
    from orthonym.rules.adducts import name_adduct
    mol = Chem.MolFromSmiles("CCO.C[N+](C)(C)C.[Cl-]")
    assert name_adduct(mol, style="pin", general_fallback=True,
                       allow_aromatic_general=True) is None
