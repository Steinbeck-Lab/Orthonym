"""Ionic organozinc halides: a cation with a zinc-carbon bond drawn beside its anion (M3a) is named
through its bonded drawing, and a carbanion drawn beside a halozinc cation (M3b) is named through
the ionic drawing that M1 gives.

Two families, told apart by the standard InChIKey:

* ``R[Zn+]`` + ``X-`` (``[Cl-].[Zn+]c1ccccc1``): the key equals the key of the bonded drawing
  ``R-Zn-X`` (``Cl[Zn]c1ccccc1``), because InChI ignores the bond from the metal to the anion and
  keeps the bond from the metal to carbon. The bonded drawing is the one the organometallic
  producers name, so the drawing as written is named through it
  (``assembly.metal_offer.metal_connected_spelling_and_order``, mechanism ``CONNECT_ANION`` of
  ``data/metal_scope.py``). The drawing as written is named as well and the better name ships.
* ``R-`` + ``[Zn+]X`` (``[CH2-]C.[Zn+]Br``): the key differs from the key of the bonded drawing
  (``CC[Zn]Br``) and equals the key of the ionic drawing ``R-`` + ``Zn2+`` + ``X-``; the metal
  charge of the ionic drawing is one of zinc's valences, so M1 offers it and the salt of zinc with
  the two anions is named. A connection of the carbanion to the zinc would name another structure.

Blue Book: INTRODUCTION, "Organometallic compounds are compounds having at least one bond
between one metal atom and one carbon atom"; Structures for Lewis 'adducts', "One of the
fundamental principles of organic chemical nomenclature is to name the structure that is drawn": a
name reached through another drawing is not the PIN of the drawing as drawn, so it is labelled
systematic_verified and the default tier declines it. Every name is read back by a fresh OPSIN call
that does not go through the engine, to the full standard InChIKey of the drawing as written.
"""
import csv
import os
import subprocess
import sys

import pytest
from rdkit import Chem

from orthonym import Orthonym, OrthonymLimitError
from orthonym.assembly import metal_offer
from orthonym.cli import _emit_tier_flags
from orthonym.perception.metals import atom_and_charge_ledger_ok
from tests.support.metal_baselines import baseline_path, identity_view, read_rows
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

RULE = "P-68.1.6.1"


def _row(smiles, tier):
    return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


def _panel_smiles(panel_id, drawing):
    with open(baseline_path("zinc").parent / "panel_pairs.tsv", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["id"] == panel_id and r["drawing"] == drawing:
                return r["smiles"]
    raise AssertionError((panel_id, drawing))


def _key(smiles):
    return Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))


PAIRS = [   # (ionic drawing, bonded drawing of the same molecule)
    ("[Cl-].[Zn+]c1ccccc1", "Cl[Zn]c1ccccc1"),                    # panel A3: aryl
    ("CC[Zn+].[Cl-]", "CC[Zn]Cl"),                                # alkyl
    ("[Br-].[Zn+]Cc1ccccc1", "Br[Zn]Cc1ccccc1"),                  # benzyl
]


@pytest.mark.parametrize("ionic,bonded", PAIRS)
def test_the_key_of_an_organozinc_cation_with_its_anion_is_the_key_of_the_bonded_drawing(ionic, bonded):
    assert _key(ionic) == _key(bonded)


@pytest.mark.parametrize("ionic,bonded", PAIRS)
def test_an_ionic_organozinc_halide_gets_the_name_of_its_bonded_drawing(ionic, bonded):
    a, b = _row(ionic, "best-effort"), _row(bonded, "best-effort")
    assert a["name"] and a["name"] == b["name"]                       # one molecule, one name
    assert a["tier"] == "systematic_verified" and a["is_pin"] is False   #,
    assert_full_rt(a["name"], ionic)                                  # the key of the ionic input
    for drawing in (ionic, bonded):                                   # the default tier declines, with a reason
        declined = _row(drawing, "pin")
        assert declined["tier"] == "abstain" and declined["limit_code"] == "NO_VERIFIED_PIN"


def test_the_ionic_organozinc_halide_of_the_panel_is_named():
    ionic = _panel_smiles("A3", "ionic")
    assert ionic == "[Cl-].[Zn+]c1ccccc1"
    assert _row(ionic, "best-effort")["name"] == _row(_panel_smiles("A3", "bonded"), "best-effort")["name"]


def test_the_ionic_organozinc_hydroxide_is_named_through_its_bonded_drawing():
    """``CC[Zn+].[OH-]`` is connected to ``CC[Zn]O``, whose hydroxy ligand the producer of
    organometallic names has (``TERMINAL_LIGAND_PREFIXES``): both drawings get one name."""
    ionic, bonded = _panel_smiles("A10", "ionic"), _panel_smiles("A10", "bonded")
    assert ionic == "CC[Zn+].[OH-]" and bonded == "CC[Zn]O"
    assert metal_offer.metal_connected_spelling_and_order(ionic) is not None          # the offer is made
    for drawing in (ionic, bonded):
        row = _row(drawing, "best-effort")
        assert row["name"] == "ethyl(hydroxy)zinc"
        assert row["tier"] == "systematic_verified" and row["is_pin"] is False
        assert_full_rt(row["name"], drawing)


def test_the_connection_is_the_bonded_drawing_with_every_atom_kept():
    connect = metal_offer.metal_connected_spelling_and_order
    written, order = connect("[Cl-].[Zn+]c1ccccc1")
    assert Chem.MolToSmiles(Chem.MolFromSmiles(written)) == Chem.MolToSmiles(Chem.MolFromSmiles("Cl[Zn]c1ccccc1"))
    assert sorted(order) == list(range(8))
    written, order = connect("CC[Zn+].[OH-]")
    assert Chem.MolToSmiles(Chem.MolFromSmiles(written)) == Chem.MolToSmiles(Chem.MolFromSmiles("CC[Zn]O"))
    assert sorted(order) == [0, 1, 2, 3]
    written, _order = connect("CC(=O)[O-].C[Zn+]")                                     # an ambident anion
    assert Chem.MolToSmiles(Chem.MolFromSmiles(written)) == Chem.MolToSmiles(Chem.MolFromSmiles("CC(=O)O[Zn]C"))


def test_the_connection_needs_an_admitted_metal_and_an_allowed_valence_and_no_choice():
    connect = metal_offer.metal_connected_spelling_and_order
    assert connect("C[Cd+].[Cl-]") is None and connect("C[Ni+].[Cl-]") is None       # not admitted
    assert connect("C[Hg+].[Cl-]") == ("[CH3][Hg][Cl]", (0, 1, 2))                    # admitted since 2026-10-30
    assert connect("C[Mg+].[Br-]") is None                                             # Group 2: ION_SALTS only
    assert connect("C[Zn+2].[Cl-].[Cl-]") is None                                      # [Zn+3] is no valence of zinc
    assert connect("C[Zn+].[Cl-].[Br-]") is None                                       # which anion: a choice
    assert connect("C[Zn+].[CH3-]") is None                                            # a carbanion is no donor
    assert connect("CCO") is None and connect("") is None and connect(None) is None
    assert connect("C[Zn+].[Cl-] |&1:0|") is None                                      # a CXSMILES extension


def test_the_connection_is_idempotent():
    written, _order = metal_offer.metal_connected_spelling_and_order("[Cl-].[Zn+]c1ccccc1")
    assert metal_offer.metal_connected_spelling_and_order(written) is None            # a bonded drawing has no free anion
    assert metal_offer.metal_rewrite_spelling_and_order(written) is None
    assert metal_offer.metal_rewrite_spelling_and_order("[Cl-].[Zn+]c1ccccc1") is not None


def test_a_connection_that_attaches_the_anion_to_another_atom_is_rejected_by_the_count():
    # atom order C0 C1 O2 O3 Zn4 C5 in all the molecules; the ionic drawing is the molecule read
    ionic = Chem.MolFromSmiles("CC(=O)[O-].[Zn+]C")
    right = Chem.MolFromSmiles("CC(=O)O[Zn]C")                       # the bond from Zn to the charged oxygen
    wrong = Chem.RWMol(right)                                        # the bond from Zn to the carbonyl oxygen
    wrong.RemoveBond(4, 3)
    wrong.AddBond(4, 2, Chem.BondType.SINGLE)
    wrong.GetBondBetweenAtoms(1, 2).SetBondType(Chem.BondType.SINGLE)
    wrong.GetBondBetweenAtoms(1, 3).SetBondType(Chem.BondType.DOUBLE)
    wrong = wrong.GetMol()
    Chem.SanitizeMol(wrong)
    lost = Chem.MolFromSmiles("CC(=O)O.[Zn+]C")                      # a hydrogen added, the net charge changed
    assert Chem.MolToInchiKey(wrong) == Chem.MolToInchiKey(ionic)    # the key alone cannot tell
    assert atom_and_charge_ledger_ok(right, ionic, [(4, 3)])         # the bonded drawing with its new bond broken is the molecule
    assert not atom_and_charge_ledger_ok(wrong, ionic, [(4, 2)])     # the donor carries no charge in the molecule
    assert not atom_and_charge_ledger_ok(wrong, ionic, [(4, 3)])     # a bond the bonded drawing does not have
    assert not atom_and_charge_ledger_ok(right, ionic, [(4, 2)])     # the declared bond is not the bond made
    assert not atom_and_charge_ledger_ok(lost, ionic, [(4, 3)])
    assert not atom_and_charge_ledger_ok(right, ionic, [])           # a bond left undeclared
    assert not atom_and_charge_ledger_ok(right, ionic, [(4, 3), (4, 3)])             # a bond declared twice


def test_every_public_entry_names_the_bonded_drawing():
    namer = Orthonym(style="pin", **_emit_tier_flags("best-effort"))
    ionic = "[Cl-].[Zn+]c1ccccc1"
    expected = namer.name("Cl[Zn]c1ccccc1")
    assert expected and expected != "unknown organic compound"
    assert namer.name(ionic) == expected
    assert namer.name(Chem.MolFromSmiles(ionic)) == expected                           # an RDKit Mol
    assert namer.name_with_tree(ionic).name == expected
    assert namer.name_with_confidence(ionic)["name"] == expected
    default = Orthonym(style="pin")
    assert default.name(ionic, on_failure="label") == "unknown organic compound"      # declined, never UNSUPPORTED_ELEMENT
    assert default.name(ionic) == ""                                                  # the caller's own call: no name, no label
    with pytest.raises(OrthonymLimitError) as raised:
        default.name(ionic, raise_on_limit=True)
    assert raised.value.code == "NO_VERIFIED_PIN"


def test_the_offer_is_made_once_per_call_by_the_caller_only(monkeypatch):
    calls = []
    original = metal_offer.metal_connected_spelling_and_order

    def counting(smiles):
        calls.append(smiles)
        return original(smiles)

    monkeypatch.setattr(metal_offer, "metal_connected_spelling_and_order", counting)
    assert _row("[Cl-].[Zn+]c1ccccc1", "best-effort")["tier"] == "systematic_verified"
    assert calls == ["[Cl-].[Zn+]c1ccccc1"]


def test_a_failing_connection_ships_nothing_and_the_drawing_as_written_keeps_its_name(monkeypatch):
    """Offer, not return: a rewrite that names nothing leaves the drawing as written as it was. The
    drawing as written is named by its own producers: the cation by its ligands and its charge
    number (``phenylzinc(1+)``) and the pair as an adduct, which loses the tie to the name of the
    connected drawing when the connection names something."""
    smiles = "[Cl-].[Zn+]c1ccccc1"
    monkeypatch.setattr(metal_offer, "metal_connected_spelling_and_order", lambda s: None)
    as_written = _row(smiles, "best-effort")
    assert as_written["name"] == "phenylzinc(1+)—chloride (1/1)" and as_written["tier"] == "systematic_verified"
    assert_full_rt(as_written["name"], smiles)
    monkeypatch.setattr(metal_offer, "metal_connected_spelling_and_order",
                        lambda s: ("[Hf+4]", (0,)) if s == smiles else None)
    row = _row(smiles, "best-effort")
    assert row["name"] == as_written["name"] and row["tier"] == as_written["tier"] and row["is_pin"] is False


# ---- the carbanion beside a halozinc cation (M3b): named through the ionic drawing of M1

CARBANION_HALIDES = [
    "[CH2-]C.[Zn+]Br",                                                  # ethanide
    "CCOC(=O)Cc1ccc(cc1)[CH2-].[Zn+]Br",                                # the Reformatsky-type drawing of the corpus
    "[c-]1ccccc1.Cl[Zn+]",                                              # benzenide
]


@pytest.mark.parametrize("smiles", CARBANION_HALIDES)
def test_a_carbanion_beside_a_halozinc_cation_is_named_as_a_salt_of_zinc(smiles):
    row = _row(smiles, "best-effort")
    assert row["name"].startswith("zinc ") and "—" not in row["name"]
    assert row["tier"] == "systematic_verified" and row["is_pin"] is False   #: a compound of the transition metals
    assert_full_rt(row["name"], smiles)
    assert _row(smiles, "pin")["tier"] == "abstain"                          # the default tier declines it


@pytest.mark.parametrize("smiles", CARBANION_HALIDES)
def test_the_two_drawings_of_a_carbanion_halide_get_one_name(smiles):
    """The halozinc drawing and the ionic drawing have one standard InChIKey and one name."""
    mol = Chem.MolFromSmiles(smiles)
    zn = next(a for a in mol.GetAtoms() if a.GetSymbol() == "Zn")
    halide = next(n for n in zn.GetNeighbors())
    rw = Chem.RWMol(mol)
    rw.RemoveBond(zn.GetIdx(), halide.GetIdx())
    zn_a, halide_a = rw.GetAtomWithIdx(zn.GetIdx()), rw.GetAtomWithIdx(halide.GetIdx())
    zn_a.SetFormalCharge(2)
    halide_a.SetFormalCharge(-1)
    ionic = Chem.MolToSmiles(rw.GetMol())
    assert _key(ionic) == _key(smiles)
    assert _row(ionic, "best-effort")["name"] == _row(smiles, "best-effort")["name"]


def test_the_connect_step_is_not_applied_to_a_carbanion_halide_pair():
    """The keys decide: the drawing equals the ionic drawing and differs from the bonded drawing, so a
    connection of the carbanion to the zinc would name another structure."""
    assert metal_offer.metal_connected_spelling_and_order("[CH2-]C.[Zn+]Br") is None
    assert _key("[CH2-]C.[Zn+]Br") != _key("CC[Zn]Br")
    assert _key("[CH2-]C.[Zn+]Br") == _key("[CH2-]C.[Zn+2].[Br-]")
    assert _key("CCOC(=O)[CH2-].[Zn+]Br") != _key("CCOC(=O)C[Zn]Br")
    assert _key("CCOC(=O)[CH2-].[Zn+]Br") == _key("CCOC(=O)[CH2-].[Zn+2].[Br-]")
    written, _order = metal_offer.metal_disconnected_spelling_and_order("[CH2-]C.[Zn+]Br")   # M1 is the rewrite that applies
    assert Chem.MolToSmiles(Chem.MolFromSmiles(written)) == Chem.MolToSmiles(Chem.MolFromSmiles("[CH2-]C.[Zn+2].[Br-]"))


def test_no_rootedAtAtom_text_on_stderr():
    """A molecule with more than one fragment reaches the substituent namer when the floor of the
    best-effort tier is offered for the drawing as written. RDKit refuses a rooted fragment SMILES
    there and logs "rootedAtAtom can only be used with molecules that have a single fragment"; the
    memo key of the substituent namer does not put the question to RDKit."""
    code = (
        "from orthonym import Orthonym\n"
        "from orthonym.cli import _emit_tier_flags\n"
        "from orthonym.jvm_budget import jvm_slots\n"
        "with jvm_slots(1, purpose='zn5-stderr-test'):\n"
        "    r = Orthonym(style='pin', **_emit_tier_flags('best-effort')).name_tiered("
        "'CCOC(=O)Cc1ccc(cc1)[CH2-].[Zn+]Br')\n"
        "print(r['name'])\n")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=dict(os.environ),
                          timeout=600)
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip().startswith("zinc ")
    assert "rootedAtAtom" not in done.stderr and "Pre-condition Violation" not in done.stderr, done.stderr[-2000:]


def _constructed_other_metal_rows():
    """Cations of metals that the table does not admit to ``CONNECT_ANION``, each with an anion: the
    drawings the connection could reach if it were not bound to the table; and a zinc cation whose
    charge is no valence after the connection."""
    return ["C[Cd+].[Cl-]", "C[Ni+].[Cl-]", "C[Mg+].[Br-]", "C[Cu+].[Cl-]", "CC[Zn+2].[Cl-].[Cl-]",
            "C[Sn+3].[Cl-].[Cl-].[Cl-]"]


def test_other_metals_are_not_moved(monkeypatch):
    """No row of the other-metals baseline holds a positively charged metal that the connection
    could reach; the cations of other metals, with an anion, are named as they are named without
    the connection."""
    rows = read_rows(baseline_path("other"))
    assert len(rows) >= 100, len(rows)
    for row in rows:
        assert metal_offer.metal_connected_spelling_and_order(row["smiles"]) is None, row["smiles"]
    namer = Orthonym(style="pin", **_emit_tier_flags("best-effort"))
    with_offer = [identity_view(namer.name_tiered(s)) for s in _constructed_other_metal_rows()]
    monkeypatch.setattr(metal_offer, "metal_connected_spelling_and_order", lambda smiles: None)
    without_offer = [identity_view(namer.name_tiered(s)) for s in _constructed_other_metal_rows()]
    assert with_offer == without_offer
