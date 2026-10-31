"""A zinc compound drawn with covalent bonds is named through its ionic drawing (M1), and the
drawing as written is named as well: the better of the two names ships.

Standard InChI ignores metal-ligand bonds, so ``Cl[Zn]Cl`` and ``[Cl-].[Cl-].[Zn+2]`` have one
standard InChIKey. ``assembly.metal_offer.metal_disconnected_spelling_and_order`` gives the ionic drawing of a
drawing in which a metal admitted to ``DISCONNECT`` (``data/metal_scope.py``) is joined to
non-carbon atoms by single bonds. The key does not bound the metal's charge and is blind to the
position of a charge on an ambident anion, so the rewrite also needs an allowed metal valence and
an independent count of atoms and charges (``perception.metals.atom_and_charge_ledger_ok``).

Blue Book: Structures for Lewis 'adducts', "One of the fundamental principles of
organic chemical nomenclature is to name the structure that is drawn". A name reached through
another drawing is therefore not the PIN of the drawing as drawn: it is labelled
systematic_verified, the default tier declines it (NO_VERIFIED_PIN), and the wider tiers keep it.
Every name is read back by a fresh OPSIN call that does not go through the engine, to the full
standard InChIKey of the covalent input.
"""
import csv

import pytest
from rdkit import Chem

from orthonym import Orthonym, OrthonymLimitError
from orthonym.assembly import metal_offer
from orthonym.cli import _emit_tier_flags
from tests.support.metal_baselines import baseline_path, declared_bonded_metal_moves, identity_view, read_rows
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

RULE = "P-68.1.6.1"


def _row(smiles, tier):
    return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


PAIRS = [   # (covalent drawing, ionic drawing of the same molecule)
    ("Cl[Zn]Cl", "[Cl-].[Cl-].[Zn+2]"),
    ("CC(=O)O[Zn]OC(C)=O", "CC(=O)[O-].CC(=O)[O-].[Zn+2]"),
    ("CN(C)C(=S)S[Zn]SC(=S)N(C)C", "CN(C)C(=S)[S-].CN(C)C(=S)[S-].[Zn+2]"),     # ziram: its label is audited in ZN11
    ("CCS[Zn]SCC", "CC[S-].CC[S-].[Zn+2]"),
    ("CCO[Zn]OCC", "CC[O-].CC[O-].[Zn+2]"),
    ("O[Zn]O", "[OH-].[OH-].[Zn+2]"),
    ("[O-][N+](=O)O[Zn]O[N+]([O-])=O", "O=[N+]([O-])[O-].O=[N+]([O-])[O-].[Zn+2]"),
    ("C[C@H](O)C(=O)O[Zn]OC(=O)[C@@H](C)O", "C[C@H](O)C(=O)[O-].C[C@@H](O)C(=O)[O-].[Zn+2]"),   # stereo is kept
]


@pytest.mark.parametrize("covalent,ionic", PAIRS)
def test_a_covalent_drawing_gets_the_name_of_its_ionic_drawing(covalent, ionic):
    a, b = _row(covalent, "best-effort"), _row(ionic, "best-effort")
    assert a["name"] == b["name"]                                   # one molecule, one name
    assert a["tier"] == "systematic_verified" and a["is_pin"] is False   #: the drawing is named as drawn
    assert_full_rt(a["name"], covalent)                             # the key of the covalent input
    declined = _row(covalent, "pin")                                # the default tier declines it, with a reason
    assert declined["tier"] == "abstain" and declined["limit_code"] == "NO_VERIFIED_PIN"
    assert declined["spelling_failures"] and declined["spelling_failures"][0]["rule"] == RULE


def test_the_ionic_drawing_keeps_its_own_label_at_both_tiers():
    """The rewrite changes nothing for the drawing that has no bond to break."""
    ionic = "CC(=O)[O-].CC(=O)[O-].[Zn+2]"
    for tier in ("pin", "best-effort"):
        row = _row(ionic, tier)
        assert row["name"] == "zinc diacetate" and row["tier"] == "pin_verified" and row["is_pin"] is True
        assert row["spelling_failures"] == []


BEST_EFFORT_ONLY = [
    "CC(=CC(C)=O)O[Zn]OC(C)=CC(C)=O",                  # Zn(acac)2, monodentate
    "[Zn-2]12(Sc3cccc[n+]3O1)Sc1cccc[n+]1O2",          # Zn pyrithione, chelate
    "[NH2+]1CC(=O)O[Zn-2]12OC(=O)C[NH2+]2",            # Zn glycinate, N,O chelate
    "C[Si](C)(C)N([Si](C)(C)C)[Zn]N([Si](C)(C)C)[Si](C)(C)C",    # the bonded silylamide
]


def _panel_smiles(panel_id, drawing):
    with open(baseline_path("zinc").parent / "panel_pairs.tsv", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["id"] == panel_id and r["drawing"] == drawing:
                return r["smiles"]
    raise AssertionError((panel_id, drawing))


@pytest.mark.parametrize("smiles", BEST_EFFORT_ONLY + [
    _panel_smiles("E1", "bonded 2 Zn-N"),              # the bonded Zn-TPP: its aromatic bonds change when the metal bonds go
    _panel_smiles("E2", "bonded 2 Zn-N"),
])
def test_chelate_and_macrocycle_drawings_are_named_at_best_effort_only(smiles):
    row = _row(smiles, "best-effort")
    assert row["tier"] == "systematic_verified" and row["is_pin"] is False
    assert_full_rt(row["name"], smiles)
    assert _row(smiles, "pin")["tier"] == "abstain"


def test_rewrite_needs_equal_key_and_an_admitted_metal():
    rewrite = metal_offer.metal_disconnected_spelling_and_order
    assert rewrite("[Zn]=O") is None                      # the key changes
    assert rewrite("CC[Zn]CC") is None                    # a Zn-C bond
    assert rewrite("CC(=O)O[Cu]OC(C)=O") is None          # copper is not admitted
    assert rewrite("N->[Zn](Cl)Cl<-N") is None            # no key (ZN10)
    assert rewrite("[Cl-].[Cl-].[Zn+2]") is None          # no bond to break
    assert rewrite("CCO") is None and rewrite("") is None and rewrite(None) is None
    written, order = rewrite("Cl[Zn]Cl")
    assert Chem.MolToSmiles(Chem.MolFromSmiles(written)) == "[Cl-].[Cl-].[Zn+2]"
    assert sorted(order) == [0, 1, 2]


def test_the_metal_valence_bound():
    rewrite = metal_offer.metal_disconnected_spelling_and_order
    assert rewrite("Cl[Zn](Cl)N") is None                 # [Zn+3] keeps the key and is not a valence of zinc
    assert rewrite("Cl[Zn-2](Cl)(Cl)Cl") is not None      # -2 + 4 bonds = +2
    assert rewrite("Cl[Zn+]") is not None                 # +1 + 1 bond = +2
    assert rewrite("Cl[Zn]") is None                      # +1 is no valence of zinc


def test_a_drawing_that_states_a_configuration_at_the_broken_bond_is_not_rewritten():
    rewrite = metal_offer.metal_disconnected_spelling_and_order
    assert rewrite("C[C@H](O)C(=O)O[Zn]OC(=O)[C@@H](C)O") is not None   # the stereo is on the ligands
    assert rewrite("C/C=N/[Zn]Cl") is None                # the bond to zinc carries the C=N configuration
    assert rewrite("C[S@](=O)CC(=O)O[Zn]OC(=O)C[S@](C)=O") is None       # a lone-pair stereocentre: left as it is
    assert rewrite("Cl[Zn]Cl |&1:0|") is None             # a CXSMILES extension: named as the stereo-group step reads it


def test_the_look_for_a_rewrite_logs_nothing_for_an_invalid_string(capfd):
    """The engine reads the string and logs what is wrong with it; the reading of the rewrite is
    silent, so an invalid string logs its parse error once, not twice."""
    from rdkit import RDLogger, rdBase
    status = rdBase.LogStatus()
    RDLogger.EnableLog("rdApp.*")
    try:
        Chem.MolFromSmiles("C(C")                                        # the control: this reading is logged
        assert "SMILES Parse Error" in capfd.readouterr().err
        for invalid in ("C(C", "CC(C)(C)(C)(C)C", "Cl[Zn]Cl.C(C"):
            assert metal_offer.metal_disconnected_spelling_and_order(invalid) is None
        assert capfd.readouterr().err == ""
    finally:
        for line in status.splitlines():
            channel, _, state = line.partition(":")
            (RDLogger.EnableLog if state == "enabled" else RDLogger.DisableLog)(channel)


CHARGE_SEPARATED = [                                       # RDKit's MetalDisconnector changes the key of these three
    ("Cl[Zn-2](Cl)(Cl)Cl", "[Cl-].[Cl-].[Cl-].[Cl-].[Zn+2]"),
    ("[NH3+][Zn-2]([NH3+])([NH3+])[NH3+]", "N.N.N.N.[Zn+2]"),
    ("[Zn-2]12(Sc3cccc[n+]3O1)Sc1cccc[n+]1O2", "[O-][n+]1ccccc1[S-].[O-][n+]1ccccc1[S-].[Zn+2]"),
]


@pytest.mark.parametrize("bonded,ionic", CHARGE_SEPARATED)
def test_charge_separated_drawings_are_disconnected_without_changing_the_net_charge(bonded, ionic):
    written, _order = metal_offer.metal_disconnected_spelling_and_order(bonded)
    assert Chem.MolToSmiles(Chem.MolFromSmiles(written)) == Chem.MolToSmiles(Chem.MolFromSmiles(ionic))


def test_a_rewrite_that_moves_a_charge_is_rejected_by_the_count():
    from orthonym.perception.metals import atom_and_charge_ledger_ok
    # atom order: C0 C1 O2 O3 Zn4 O5 C6 C7 O8 in all four molecules
    before = Chem.MolFromSmiles("CC(=O)O[Zn]OC(C)=O")
    changed = [(4, 3), (4, 5)]                                           # the two Zn-O bonds
    right = Chem.MolFromSmiles("CC(=O)[O-].[Zn+2].[O-]C(C)=O")           # -1 on the bonded oxygens
    wrong = Chem.MolFromSmiles("CC([O-])=O.[Zn+2].[O-]C(C)=O")           # -1 on a carbonyl oxygen: same standard InChIKey, other structure
    lost = Chem.MolFromSmiles("CC(=O)O.[Zn+2].[O-]C(C)=O")               # a hydrogen added, the net charge changed
    assert Chem.MolToInchiKey(wrong) == Chem.MolToInchiKey(before)       # the key alone cannot tell
    assert atom_and_charge_ledger_ok(before, right, changed)
    assert not atom_and_charge_ledger_ok(before, wrong, changed)
    assert not atom_and_charge_ledger_ok(before, lost, changed)
    assert not atom_and_charge_ledger_ok(before, right, [(4, 3)])        # a bond left undeclared
    assert not atom_and_charge_ledger_ok(before, right, [(4, 3), (4, 5), (4, 5)])   # a bond declared twice
    assert not atom_and_charge_ledger_ok(before, right, [(0, 3), (4, 5)])           # a bond the molecule does not have


def test_the_rewrite_is_idempotent_and_safe_in_a_nested_call():
    rewrite = metal_offer.metal_disconnected_spelling_and_order
    written, _ = rewrite("Cl[Zn]Cl")
    assert rewrite(written) is None                                      # an ionic drawing has no Zn-X bond to split
    # the nested entries of one call change nothing: both drawings get one name
    assert _row("Cl[Zn]Cl", "best-effort")["name"] == _row("[Cl-].[Cl-].[Zn+2]", "best-effort")["name"]


def test_the_offer_is_made_once_per_call_by_the_caller_only(monkeypatch):
    """Nested calls, the calls of the offer itself and the strict PIN twin name the string they are
    given: the rewrite is looked for once per public call."""
    calls = []
    original = metal_offer.metal_disconnected_spelling_and_order

    def counting(smiles):
        calls.append(smiles)
        return original(smiles)

    monkeypatch.setattr(metal_offer, "metal_disconnected_spelling_and_order", counting)
    row = _row("CCS[Zn]SCC", "best-effort")
    assert row["name"] == "zinc diethanethiolate"
    assert calls == ["CCS[Zn]SCC"]
    calls.clear()
    assert _row("[Cl-].[Cl-].[Zn+2]", "best-effort")["tier"] == "pin_verified"   # the twin runs for it
    assert calls == ["[Cl-].[Cl-].[Zn+2]"]


def test_every_public_entry_names_the_ionic_drawing():
    namer = Orthonym(style="pin", **_emit_tier_flags("best-effort"))
    assert namer.name("Cl[Zn]Cl") == "zinc dichloride"
    assert namer.name(Chem.MolFromSmiles("Cl[Zn]Cl")) == "zinc dichloride"           # an RDKit Mol
    assert namer.name_with_tree("Cl[Zn]Cl").name == "zinc dichloride"
    assert namer.name_with_confidence("Cl[Zn]Cl")["name"] == "zinc dichloride"
    default = Orthonym(style="pin")
    assert default.name("Cl[Zn]Cl", on_failure="label") == "unknown organic compound"  # declined, never UNSUPPORTED_ELEMENT
    assert default.name("Cl[Zn]Cl") == ""                                              # the caller's own call: no name, no label
    with pytest.raises(OrthonymLimitError) as raised:
        default.name("Cl[Zn]Cl", raise_on_limit=True)
    assert raised.value.code == "NO_VERIFIED_PIN"
    assert default.name("[Cl-].[Cl-].[Zn+2]") == "zinc dichloride"


AS_WRITTEN = "Cl[Zn]Cl.c1ccc(nc1)-c1ccccn1"      # panel D3, two fragments: named at START, as written, as an adduct of ZnCl2 and bipyridine
IONIC = "[Cl-].[Cl-].[Zn+2].c1ccc(nc1)-c1ccccn1"


def _start_row(smiles, tier_req):
    for row in read_rows(baseline_path("zinc")):
        if row["smiles"] == smiles and row["tier_req"] == tier_req:
            return row
    raise AssertionError(smiles)


def test_the_as_written_name_ships_when_the_rewrite_fails(monkeypatch):
    base = _start_row(AS_WRITTEN, "best-effort")["name"]                # the name at the start commit
    # the call on the rewritten string names nothing: the rewrite hands the engine a string it cannot name
    monkeypatch.setattr(metal_offer, "metal_disconnected_spelling_and_order",
                        lambda smiles: ("[Th+4]", (0,)) if smiles == AS_WRITTEN else None)
    row = _row(AS_WRITTEN, "best-effort")
    assert row["name"] == base and row["tier"] == "systematic_verified"
    assert_full_rt(row["name"], AS_WRITTEN)


def test_the_as_written_name_ships_when_the_rewrite_lands_lower(monkeypatch):
    base = _start_row(AS_WRITTEN, "best-effort")["name"]
    written, _order = metal_offer.metal_disconnected_spelling_and_order(AS_WRITTEN)
    original = metal_offer.form_result_rank

    def lowered(namer, string, result):                                 # the rewritten candidate comes back one tier below
        rank = original(namer, string, result)
        return (1, 0) if string == written and rank[0] > 1 else rank

    monkeypatch.setattr(metal_offer, "form_result_rank", lowered)
    row = _row(AS_WRITTEN, "best-effort")
    assert row["name"] == base and row["tier"] == "systematic_verified"
    assert_full_rt(row["name"], AS_WRITTEN)


def test_equal_tiers_ship_the_rewritten_name():
    a, b = _row(AS_WRITTEN, "best-effort"), _row(IONIC, "best-effort")
    assert a["name"] == b["name"]                 # one molecule, one name, whatever the drawing
    assert a["tier"] == b["tier"] == "systematic_verified"
    assert_full_rt(a["name"], AS_WRITTEN)


def test_a_failing_rewrite_leaves_the_default_tier_label_of_the_drawing_as_written(monkeypatch):
    """Two failures: the drawing as written ships its own label (UNNAMEABLE: zinc is a complete metal)."""
    monkeypatch.setattr(metal_offer, "metal_disconnected_spelling_and_order",
                        lambda smiles: ("[Th+4]", (0,)) if smiles == "Cl[Zn]Cl" else None)
    row = _row("Cl[Zn]Cl", "pin")
    assert row["tier"] == "abstain" and row["limit_code"] == "UNNAMEABLE"


def _bonded_non_zinc_rows():
    """The best-effort rows of the other-metals baseline whose drawing has a single bond from a metal
    other than zinc to a non-carbon atom: the rows a rewrite could reach if it were not bound to zinc."""
    from orthonym.perception.metals import _TRUE_METAL_SYMBOLS_FOR_VETO
    rows = []
    for row in read_rows(baseline_path("other")):
        if row["tier_req"] != "best-effort":
            continue
        mol = Chem.MolFromSmiles(row["smiles"])
        for bond in mol.GetBonds():
            a, b = bond.GetBeginAtom(), bond.GetEndAtom()
            for metal, other in ((a, b), (b, a)):
                if (metal.GetSymbol() in _TRUE_METAL_SYMBOLS_FOR_VETO and metal.GetSymbol() != "Zn"
                        and other.GetSymbol() != "C" and other.GetSymbol() not in _TRUE_METAL_SYMBOLS_FOR_VETO
                        and bond.GetBondType() == Chem.BondType.SINGLE):
                    rows.append(row)
                    break
            else:
                continue
            break
    return rows


def test_other_metals_are_not_moved():            # reads the baseline fixture (constructed molecules)
    rows = _bonded_non_zinc_rows()
    assert len(rows) >= 10, len(rows)
    namer = Orthonym(style="pin", **_emit_tier_flags("best-effort"))
    declared = declared_bonded_metal_moves()      # the metals admitted since 2026-10-30 move as declared there
    moved = []
    for row in rows:
        now = namer.name_tiered(row["smiles"])
        key = (row["smiles"], row["tier_req"])
        if key in declared:
            assert identity_view(now) == declared[key], key
            if now["name"] and now["tier"] != "abstain":
                assert_full_rt(now["name"], row["smiles"])
            continue
        if identity_view(now) != identity_view(row):
            moved.append((row["smiles"][:60], identity_view(row), identity_view(now)))
    assert moved == []
