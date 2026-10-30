"""Zinc complexes named by their ligands: hydrogen on the metal, terminal ligands (hydroxy, oxo,
sulfanylidene,...) and a charge number (``BONDED_NAMES`` of ``data/metal_scope.py``).

A drawing with ``Zn=O``, ``Zn=S``, a hydrogen on the zinc, a hydroxy ligand beside a Zn-C bond or a
charged zinc has no ionic twin with its standard InChIKey (the key is blind to bonds from a metal, but
not to the hydrogens and the charges), so the rewrite of ``assembly/metal_offer.py`` does not apply
and the producer of organometallic names names the drawing as drawn.

Blue Book: Compounds with at least one metal-carbon single bond (the Blue Book):
"Compounds are named by citing the names of ligands, including hydrogen atom(s), in alphanumerical
order, followed by the name of the metal. The presence of hydrogen attached to a metal atom must
always be indicated by the prefix 'hydrido'"; Definitions of terms (:39767): "names for
substituent groups as defined in Section can also be used as ligand names"; the examples of
 ('trichlorido(methyl)titanium', 'pentaammine(ethyl)osmium(1+)') cite the first ligand
without enclosing marks, each further one in parentheses, and end a charged complex in its charge
number. No PIN: (:39753), "Coordination nomenclature is the primary nomenclature method
used to name organometallic compounds containing elements of Groups 3 through 12", and
PREFERRED IUPAC NAMES (:2056), which gives the label "PIN" to compounds of the listed elements with
a carbon atom, zinc not among them. Every name is read back by a fresh OPSIN call that does not go
through the engine, to the full standard InChIKey of the drawing as written.
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.data import metal_scope
from orthonym.data.metal_scope import BONDED_NAMES, METAL_SCOPE, MetalScope
from orthonym.data.organometallics import TERMINAL_LIGAND_PREFIXES
from orthonym.perception.metals import detect_metal_complex
from orthonym.rules import metal_labels
from orthonym.rules.organometallics import _assemble_bonded_metal_name, assemble_organometallic_name
from tests.support.metal_baselines import baseline_path, identity_view, read_rows
from tests.support.rt_assert import assert_full_rt, name_is_rt_exact

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

P_69_2_1 = "P-69.2.1"   # coordination nomenclature is the primary method for Groups 3-12: no PIN


def _row(smiles, tier):
    return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


def _assembled(smiles):
    """The name the producer of organometallic names gives the drawing, or ``None`` (pure: no JVM)."""
    mol = Chem.MolFromSmiles(smiles)
    complex_ = detect_metal_complex(mol)
    return None if complex_ is None else _assemble_bonded_metal_name(complex_, mol)


def _assert_named_without_a_pin(smiles, name, tiers=("best-effort", "valid")):
    for tier in tiers:
        row = _row(smiles, tier)
        assert row["name"] == name, (tier, row)
        assert row["tier"] == "systematic_verified" and row["is_pin"] is False, (tier, row)
        assert_full_rt(row["name"], smiles)
    declined = _row(smiles, "pin")                    # the default tier declines it, with a reason
    assert declined["tier"] == "abstain" and declined["limit_code"] == "NO_VERIFIED_PIN", declined
    assert declined["is_pin"] is False


# ---- the panel rows of the class (Zn=O, Zn=S, Zn-H, Zn-C with a hydroxy ligand)

PANEL = [
    ("B2", "O=[Zn]", "oxozinc"),
    ("B2", "[Zn]=O", "oxozinc"),
    ("B2", "[O]=[Zn]", "oxozinc"),
    ("B4", "S=[Zn]", "sulfanylidenezinc"),
    ("B9", "[ZnH2]", "dihydridozinc"),
    ("A10", "CC[Zn]O", "ethyl(hydroxy)zinc"),
    ("A10", "CC[Zn+].[OH-]", "ethyl(hydroxy)zinc"),          # the connection of ZN5 meets the hydroxy ligand
]


@pytest.mark.parametrize("panel_id,smiles,name", PANEL)
def test_the_drawings_with_terminal_ligands_and_hydrogen_are_named_as_ligands_of_the_metal(panel_id, smiles, name):
    _assert_named_without_a_pin(smiles, name)


def test_the_lithium_zincate_is_named_and_read_back():
    """Panel A9: ``[Li+].C[Zn-](C)C``. The anion is ``trimethylzinc(1-)`` (the charge number, never
    '-ate': OPSIN reads no '-ate' name of a transition metal). The salt name
    'lithium trimethylzinc(1-)' that the salt route builds is withdrawn by the structure-loss veto of the
    namer for a compound with a metal-carbon bond that no organometallic handler won, so the adduct
    route gives the name; it reads back to the full key."""
    smiles = "[Li+].C[Zn-](C)C"
    row = _row(smiles, "best-effort")
    assert row["name"] == "trimethylzinc(1-)—lithium(1+) (1/1)"
    assert row["tier"] == "systematic_verified" and row["is_pin"] is False
    assert_full_rt(row["name"], smiles)
    assert _row(smiles, "pin")["tier"] == "abstain"                 # no name at the default tier
    sodium = _row("[Na+].C[Zn-](C)C", "best-effort")
    assert sodium["name"] == "trimethylzinc(1-)—sodium(1+) (1/1)" and sodium["tier"] == "systematic_verified"
    assert_full_rt(sodium["name"], "[Na+].C[Zn-](C)C")


# ---- the class: one rule for every ligand of the table, for hydrogen and for the charge

CLASS = [
    ("C[ZnH]", "hydrido(methyl)zinc"),                  #: 'hydrido' for the hydrogen of the metal
    ("[ZnH]O", "hydrido(hydroxy)zinc"),
    ("CC[Zn]S", "ethyl(sulfanyl)zinc"),
    ("CC[Zn]N", "amino(ethyl)zinc"),
    ("O=[Zn]=O", "dioxozinc"),
    ("C[Zn-](C)C", "trimethylzinc(1-)"),                # the charge number of the complex
    ("C[Zn-2](C)(C)C", "tetramethylzinc(2-)"),
    ("CC[Zn+]", "ethylzinc(1+)"),
    ("[ZnH+]", "hydridozinc(1+)"),
    ("[Zn-](O)(C)C", "hydroxydi(methyl)zinc(1-)"),     # first cited ligand bare, the next in parentheses
    ("C[Zn-](C)c1ccccc1", "dimethyl(phenyl)zinc(1-)"),  # alphanumerical order, the multiplier outside the marks
]


@pytest.mark.parametrize("smiles,name", CLASS)
def test_every_drawing_of_the_class_is_named_by_its_ligands(smiles, name):
    _assert_named_without_a_pin(smiles, name)


def test_every_ligand_of_the_table_is_exercised_and_named():
    """A row of ``TERMINAL_LIGAND_PREFIXES`` without a drawing here is untested: the table and the
    drawings cannot drift apart."""
    drawings = {
        ("H", 1, 0): "C[ZnH]",
        ("O", 1, 1): "CC[Zn]O",
        ("S", 1, 1): "CC[Zn]S",
        ("N", 1, 2): "CC[Zn]N",
        ("O", 2, 0): "O=[Zn]",
        ("S", 2, 0): "S=[Zn]",
    }
    assert set(drawings) == set(TERMINAL_LIGAND_PREFIXES)
    for key, smiles in drawings.items():
        assembled = _assembled(smiles)
        assert assembled is not None and TERMINAL_LIGAND_PREFIXES[key] in assembled[0], (key, smiles)
        mol = Chem.MolFromSmiles(smiles)
        assert assemble_organometallic_name(detect_metal_complex(mol), mol)[0] == assembled[0]   # the hook is on the path


# ---- spelling and order (pure: no JVM)

SPELLING = [
    ("O=[Zn]", "oxozinc"),
    ("[ZnH2]", "dihydridozinc"),                         # the multiplier of one ligand
    ("O[Zn]O", "dihydroxyzinc"),                         # 'hydroxy' is a simple prefix, not a compound -oxy
    ("S[Zn]S", "bis(sulfanyl)zinc"),                     # the Blue Book: 'bis(sulfanyl)' means two -SH groups
    ("CC[Zn]O", "ethyl(hydroxy)zinc"),                   # alphanumerical: ethyl before hydroxy
    ("[Zn-](O)(C)C", "hydroxydi(methyl)zinc(1-)"),       # alphanumerical: hydroxy before methyl, multiplier ignored
    ("C[Zn-](C)c1ccccc1", "dimethyl(phenyl)zinc(1-)"),   # the multiplier is no part of the sorted name
    ("[Zn-]([H])(C)C", "hydridodi(methyl)zinc(1-)"),
]


@pytest.mark.parametrize("smiles,name", SPELLING)
def test_the_ligands_are_cited_in_alphanumerical_order_with_the_marks_of_p_16_5_1_3_1(smiles, name):
    assembled = _assembled(smiles)
    assert assembled is not None and assembled[0] == name
    assert assembled[1] == name[name.index("zinc"):]       # the metal name part carries the charge number


# ---- what the producer leaves to the other branches (fail closed)

DECLINED = [
    "CC[Zn]CC",                # an all-organyl neutral complex: the organyl branch names it
    "C[Zn]C#N",                # a functional organyl ligand: best-effort only, by the organyl branch
    "CC[Zn]Cl",                # a halide: the Grignard branch
    "[Zn]([H])Cl",
    "[CH2-][Zn]C",             # a charge that is not the metal's
    "[Zn-]([CH2-])C",
    "CC[Zn]OC",                # an alkoxide ligand: no producer
    "[2H]O[Zn]C",              # an isotope in a ligand
    "[Zn]1CCCC1",              # a ring: the metallacycle branch
    "O=[Zn]=[Zn]=O",           # two metals
    "[Zn]",                    # no ligand, no hydrogen
]


@pytest.mark.parametrize("smiles", DECLINED)
def test_a_drawing_the_producer_cannot_name_is_left_to_the_other_branches(smiles):
    assert _assembled(smiles) is None


UNPERCEIVED = [
    "[O-]=[Zn]",               # a charged atom at the double bond is no ligand of the table
    "[2H]O[Zn]=O",             # an isotope in a ligand
    "[Zn]=[Zn]",
    "CC[Zn]O[Zn]CC",
]


@pytest.mark.parametrize("smiles", UNPERCEIVED)
def test_a_double_bond_to_an_atom_outside_the_table_is_no_ligand(smiles):
    complex_ = detect_metal_complex(Chem.MolFromSmiles(smiles))
    assert complex_ is None or _assemble_bonded_metal_name(complex_, Chem.MolFromSmiles(smiles)) is None
    if smiles in ("[O-]=[Zn]",):
        assert complex_ is None                           # as before: the perception declines the drawing


#: an organic group joined to the metal by one double bond from a carbon or a nitrogen atom is a ligand since 2026-10-30:
#: "names for substituent groups as defined in Section can also be used as ligand names" Definitions of
#: terms, the Blue Book); the prefix of a group with a double bond at its free valence ends in 'ylidene',
#: that of =NH is 'imino'
MULTIPLY_BONDED = [("C=[Zn]", "methylidenezinc"), ("[Zn]=N", "iminozinc"), ("[Zn]=[NH]", "iminozinc")]


@pytest.mark.parametrize("smiles,name", MULTIPLY_BONDED)
def test_an_organic_group_on_a_double_bond_is_a_ligand_named_as_drawn(smiles, name):
    assert _assembled(smiles)[0] == name
    _assert_named_without_a_pin(smiles, name)


def test_no_name_that_drops_a_hydrogen_or_a_bond_order_is_shipped():
    """The double bond is stated in the name, and the drawing is never named as the single-bonded one: never
    'methylzinc' for ``C=[Zn]`` (a methylidene named as a methyl), never 'aminozinc' for ``[Zn]=N``. Each name is read
    back to the full InChIKey of its drawing, which keeps every hydrogen and charge (the key is blind to the bond
    from the metal only)."""
    for smiles, name in MULTIPLY_BONDED:
        for tier in ("best-effort", "valid"):
            row = _row(smiles, tier)
            assert row["name"] == name and row["tier"] == "systematic_verified", (smiles, tier, row)
            assert row["name"] not in ("methylzinc", "aminozinc", "zinc"), (smiles, tier, row)
            assert name_is_rt_exact(row["name"], smiles), (smiles, tier, row)


# ---- labels: no PIN (the predicate beside the demotion of the tier row)

def test_the_label_predicate_covers_bonded_zinc_names_with_or_without_a_metal_carbon_bond():
    f = metal_labels.is_bonded_metal_name_without_pin
    assert f("O=[Zn]", "oxozinc") and f("[ZnH2]", "dihydridozinc")     # no carbon:
    assert f("CC[Zn]O", "ethyl(hydroxy)zinc") and f("C[Zn-](C)C", "trimethylzinc(1-)")
    assert f("[O-2].[Zn+2]", "zinc oxide") is False                    # an ionic drawing: no bond, no hydrogen
    assert f("[Cl-].[Cl-].[Zn+2]", "zinc dichloride") is False         # the carbon-free salt keeps its label
    assert f("O=[Cd]", "oxocadmium") is True and f("O=[Cu]", "oxocopper") is True     # admitted since 2026-10-30
    assert f("O=[Tc]", "oxotechnetium") is False and f("O=[Pu]", "oxoplutonium") is False   # not admitted
    assert f("", "oxozinc") is False and f("O=[Zn]", None) is False and f("not a smiles", "x") is False


def test_a_name_of_the_exact_structure_list_keeps_the_label_of_its_path(monkeypatch):
    import orthonym.namer as namer
    monkeypatch.setattr(namer, "_coordination_list_name", lambda smiles, name: True)
    assert metal_labels.is_bonded_metal_name_without_pin("O=[Zn]", "oxozinc") is False


def test_the_combined_predicate_keeps_the_carbanion_salts_and_adds_the_bonded_names():
    f = metal_labels.is_metal_name_without_pin
    assert f("[CH2-]C.[CH2-]C.[Zn+2]", "zinc bis(ethan-1-ide)") is True    # decision 3 of ZN4
    assert f("O=[Zn]", "oxozinc") is True
    assert f("[Cl-].[Cl-].[Zn+2]", "zinc dichloride") is False
    assert f("CC(=O)[O-].CC(=O)[O-].[Zn+2]", "zinc diacetate") is False


def test_the_predicate_is_what_lowers_the_label(monkeypatch):
    """Without the predicate the strict path's label would stand: 'oxozinc' is a name that the strict
    path builds and verifies, so the demotion is the only thing that keeps it from the label of a PIN."""
    monkeypatch.setattr(metal_labels, "is_bonded_metal_name_without_pin", lambda smiles, name: False)
    row = _row("O=[Zn]", "valid")
    assert row["name"] == "oxozinc" and row["tier"] == "pin_verified", row


# ---- the scope table decides, and no other metal moves

def test_the_producer_reads_the_scope_table(monkeypatch):
    assert _assembled("O=[Zn]") is not None
    reduced = MetalScope(METAL_SCOPE["Zn"].mechanisms - {BONDED_NAMES}, METAL_SCOPE["Zn"].valences)
    monkeypatch.setitem(METAL_SCOPE, "Zn", reduced)
    assert metal_scope.has_mechanism("Zn", BONDED_NAMES) is False
    assert detect_metal_complex(Chem.MolFromSmiles("O=[Zn]")) is None       # the perception declines as it did
    assert detect_metal_complex(Chem.MolFromSmiles("[ZnH2]")) is None
    assert _assembled("CC[Zn]O") is None and _assembled("C[Zn-](C)C") is None


#: Hg, Pt, Ti and (lane B2) Cd, Ni, Cu, Fe, Co, Mn, Cr are admitted to BONDED_NAMES since 2026-10-30
#: (``BONDED_METAL_VALENCES``, ``MORE_BONDED_METAL_VALENCES``): technetium, thorium and plutonium, which no mechanism
#: admits, take their places
OTHER_METALS = ["O=[Tc]", "[TcH2]", "CC[Tc]O", "C[Tc]O", "CC[Th]O", "O=[Th]", "O=[Pu]", "[PuH]", "[Th]=O", "C[Mg-](C)C",
                "CC[Mg]O", "[Pu]=O", "CC[Sn](C)(C)O", "O=[Tc]=O", "[Th]=S"]


@pytest.mark.parametrize("smiles", OTHER_METALS)
def test_other_metals_are_not_moved_by_the_ligand_producer(smiles):
    """The perception and the producer answer as they did for every metal the table does not admit to
    ``BONDED_NAMES``: the producer names none of these, and the perception of the drawings with a
    doubly bonded atom or a hydrogen on the metal is the one it was."""
    mol = Chem.MolFromSmiles(smiles)
    complex_ = detect_metal_complex(mol)
    assert complex_ is None or _assemble_bonded_metal_name(complex_, mol) is None
    if any(a.GetSymbol() in ("Tc", "Th", "Pu") and
           (a.GetTotalNumHs() > 0 or any(b.GetBondTypeAsDouble() == 2.0 for b in a.GetBonds()))
           for a in mol.GetAtoms()):
        assert complex_ is None


#: the drawings that the guard held for Cd, Hg, Ni, Cu, Fe, Co, Mn and Cr, named by their ligands since 2026-10-30: the
#: ligands in alphanumerical order, then the metal Compounds with at least one metal-carbon single bond)
NOW_ADMITTED = [
    ("O=[Cd]", "oxocadmium"), ("[CdH2]", "dihydridocadmium"), ("CC[Cd]O", "ethyl(hydroxy)cadmium"),
    ("C[Cd]O", "hydroxy(methyl)cadmium"), ("CC[Ni]O", "ethyl(hydroxy)nickel"), ("O=[Ni]", "oxonickel"),
    ("O=[Cu]", "oxocopper"), ("[CuH]", "hydridocopper"), ("O=[Fe]", "oxoiron"), ("[Co]=O", "oxocobalt"),
    ("O=[Mn]=O", "dioxomanganese"), ("[Cr]=O", "oxochromium"),
]


@pytest.mark.parametrize("smiles,name", NOW_ADMITTED)
def test_the_drawings_of_the_metals_admitted_since_2026_10_30_are_named_by_their_ligands(smiles, name):
    assert _assembled(smiles)[0] == name
    _assert_named_without_a_pin(smiles, name)
    assert metal_labels.is_bonded_metal_name_without_pin(smiles, name) is True


def test_other_metals_are_named_as_they_are_without_the_mechanism(monkeypatch):
    """A drawing of a metal that the table does not admit gets the same row with the mechanism of zinc
    removed from the table: no row of the other-metals baseline with a metal that the table does not admit to
    ``BONDED_NAMES`` is named by the producer, and the constructed drawings of Tc, Th and Pu give one answer
    either way."""
    rows = read_rows(baseline_path("other"))
    assert len(rows) >= 100, len(rows)
    named_by_the_producer = []
    for row in rows:
        mol = Chem.MolFromSmiles(row["smiles"])
        complex_ = detect_metal_complex(mol)
        if complex_ is not None and _assemble_bonded_metal_name(complex_, mol) is not None:
            named_by_the_producer.append(row["smiles"])
            # only a metal that the table admits to BONDED_NAMES reaches the producer
            assert any(metal_scope.has_mechanism(a.GetSymbol(), BONDED_NAMES) for a in mol.GetAtoms()), row["smiles"]
    assert "O=C=[Fe](=C=O)(=C=O)(=C=O)=C=O" in named_by_the_producer      # iron pentacarbonyl, drawn with double bonds
    namer = Orthonym(style="pin", **_emit_tier_flags("best-effort"))
    constructed = ["O=[Tc]", "CC[Tc]O", "CC[Th]O", "O=[Pu]", "C[Mg-](C)C", "[Na+].C[Mg-](C)C"]
    with_scope = [identity_view(namer.name_tiered(s)) for s in constructed]
    reduced = MetalScope(METAL_SCOPE["Zn"].mechanisms - {BONDED_NAMES}, METAL_SCOPE["Zn"].valences)
    monkeypatch.setitem(METAL_SCOPE, "Zn", reduced)
    without_scope = [identity_view(namer.name_tiered(s)) for s in constructed]
    assert with_scope == without_scope


def test_the_names_that_zinc_had_are_the_names_it_has():
    """Nothing named before changes: the zinc drawings of the panel that the class does not touch."""
    for smiles, name in (("CC[Zn]CC", "diethylzinc"), ("CC[Zn]Cl", "ethylzinc chloride"),
                         ("O[Zn]O", "zinc dihydroxide"),           # the rewrite to the ionic drawing still wins the tie
                         ("N#C[Zn]C#N", "dicyanozinc")):
        row = _row(smiles, "best-effort")
        assert row["name"] == name and row["tier"] == "systematic_verified" and row["is_pin"] is False, row
        assert_full_rt(name, smiles)
