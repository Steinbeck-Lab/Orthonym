"""A dative-bond drawing of zinc is named through the same drawing without its dative bonds.

RDKit gives a drawing with dative bonds (``->`` and ``<-``) no standard InChIKey, so no name can be
read back against it and the engine abstained at every tier. ``assembly.metal_offer`` writes the
split drawing (mechanism ``DATIVE`` of ``data/metal_scope.py``): the donor keeps its lone pair, the
metal its charge, and nothing else changes. The key of the split drawing is the reference key of the
read-back of a name, and an independent count of atoms and charges
(``perception.metals.atom_and_charge_ledger_ok``, no charge moved) ties the split drawing to the
drawing given. The split drawing is then treated as any drawing: the ionic drawing of
``DISCONNECT`` is offered beside it.

Blue Book: Structures for Lewis 'adducts' draws the 'dative' bond as structure (I) and
says "One of the fundamental principles of organic chemical nomenclature is to name the structure
that is drawn": a name reached through the split drawing is not the name of the structure drawn. It
is labelled systematic_verified, ``is_pin`` is False, the default tier declines it, and the row says
``verified`` is ``opsin_split``: OPSIN read the name back to the split drawing, not to the drawing as
written. Every name is read back by a fresh OPSIN call to the full standard InChIKey of the split
drawing.
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym import namer as _namer
from orthonym.assembly import metal_offer
from orthonym.cli import _emit_tier_flags
from orthonym.metrics import provenance as _provenance
from orthonym.perception.metals import atom_and_charge_ledger_ok, dative_bond_split_sites
from tests.support.default_tier import assert_default_tier_declines
from tests.support.rt_assert import (assert_full_rt, name_is_rt_exact, smiles_without_dative_bonds)

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

RULE = "P-68.1.6.1"

DATIVE_ACETATE = "CC(=O)[O-]->[Zn+2]<-[O-]C(C)=O"
IONIC_ACETATE = "CC(=O)[O-].CC(=O)[O-].[Zn+2]"
DATIVE_WATER = "O->[Zn+2]<-O"
IONIC_WATER = "O.O.[Zn+2]"
DATIVE_AMMINE_CHLORIDE = "Cl[Zn](Cl)(<-N)<-N"
SPLIT_AMMINE_CHLORIDE = "Cl[Zn](Cl).N.N"
HALF_DATIVE_ACETATE = "CC(=O)O[Zn+]<-[O-]C(C)=O"            # one covalent bond, one dative bond
DATIVE_BIPYRIDINE = "Cl[Zn]1(Cl)<-[n]2ccccc2-c2cccc[n]->12"
SPLIT_BIPYRIDINE = "Cl[Zn]Cl.c1ccc(nc1)-c1ccccn1"
DATIVE_PYRITHIONE = "c1cc[n+]2[O-]->[Zn]3(<-[O-][n+]4ccccc4[S]3)[S]c2c1"
IONIC_PYRITHIONE = "[O-][n+]1ccccc1[S-].[O-][n+]1ccccc1[S-].[Zn+2]"
DATIVE_CU = "N->[Cu+2](<-N)(<-N)<-N"
DATIVE_NI = "Cl[Ni](Cl)(<-N)<-N"                   # (platinum is admitted since 2026-10-30)


def _row(smiles, tier):
    return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


def _key(smiles):
    return Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))


# ----------------------------------------------------------------------------- a key-less input

@pytest.mark.parametrize("dative", [DATIVE_ACETATE, DATIVE_WATER, DATIVE_AMMINE_CHLORIDE, DATIVE_PYRITHIONE])
def test_the_premise_a_dative_drawing_has_no_standard_inchikey(dative):
    assert _key(dative) == ""
    assert _key(smiles_without_dative_bonds(dative)) != ""


@pytest.mark.parametrize("dative,name", [
    (DATIVE_ACETATE, "zinc diacetate"),
    (DATIVE_WATER, "zinc(2+)—water (1/2)"),
    (DATIVE_AMMINE_CHLORIDE, "ammonia—chloride—zinc(2+) (2/2/1)"),
])
@pytest.mark.parametrize("flags", [
    {},                                                          # the default tier
    {"besteffort_unverified": True, "general_fallback_tier": True},   # the best-effort tier
])
def test_keyless_input_is_never_shipped_unverified(dative, name, flags):
    """The gate declines a name that reads back to the split drawing when the input it is given is
    the dative drawing: the input has no key, so nothing can be compared, and the name is withdrawn."""
    assert name_is_rt_exact(name, dative)                      # the name is right for the split drawing
    shipped = _namer._final_opsin_validity_gate(name, dative, None, **flags)
    assert shipped != name


@pytest.mark.parametrize("dative", [DATIVE_ACETATE, DATIVE_WATER, DATIVE_AMMINE_CHLORIDE, DATIVE_PYRITHIONE])
def test_the_drawing_as_written_gives_no_name_of_its_own(dative, monkeypatch):
    """Without the split the dative drawing abstains at both tiers with the label UNNAMEABLE (zinc is a
    complete metal): no producer names the drawing as written, and a name it gave would be withdrawn by the
    gate."""
    monkeypatch.setattr(metal_offer, "metal_dative_split_spelling_and_order", lambda smiles: None)
    for tier in ("best-effort", "pin"):
        row = _row(dative, tier)
        assert row["tier"] == "abstain" and row["limit_code"] == "UNNAMEABLE", row
        assert row["verified"] == "unverified" and row["is_pin"] is False


def _count_ranks(monkeypatch):
    calls = []
    original = metal_offer.form_result_rank

    def counting(namer, string, result):
        calls.append(string)
        return original(namer, string, result)

    monkeypatch.setattr(metal_offer, "form_result_rank", counting)
    return calls


def test_the_drawing_as_written_is_run_only_when_no_other_drawing_names(monkeypatch):
    """One run per drawing of the offer; the dative drawing as written, which has no key to read a
    name back against, is run only when the others name nothing and decline nothing with
    NO_VERIFIED_PIN."""
    calls = _count_ranks(monkeypatch)
    assert _row(DATIVE_ACETATE, "best-effort")["name"] == "zinc diacetate"
    assert len(calls) == 1 and DATIVE_ACETATE not in calls                 # the split drawing only
    calls.clear()
    assert _row(DATIVE_AMMINE_CHLORIDE, "best-effort")["name"]
    assert len(calls) == 2 and DATIVE_AMMINE_CHLORIDE not in calls         # the split and its ionic drawing
    calls.clear()
    assert _row(DATIVE_ACETATE, "pin")["limit_code"] == "NO_VERIFIED_PIN"   # a decline that says a name exists
    assert len(calls) == 1 and DATIVE_ACETATE not in calls
    calls.clear()
    row = _row(DATIVE_WATER, "pin")                                         # nothing to say: the drawing as written runs
    assert row["limit_code"] == "UNNAMEABLE" and DATIVE_WATER in calls


@pytest.mark.parametrize("dative", [DATIVE_CU, DATIVE_NI])
def test_a_dative_drawing_of_a_metal_not_admitted_is_declined(dative):
    """Copper and nickel are not admitted to ``DATIVE``: no split, no name, at either tier, as before."""
    assert metal_offer.metal_dative_split_spelling_and_order(dative) is None
    assert metal_offer.metal_offer_for(dative) is None
    for tier in ("best-effort", "pin"):
        row = _row(dative, tier)
        assert row["tier"] == "abstain" and row["limit_code"] == "UNSUPPORTED_ELEMENT", row
        assert row["verified"] == "unverified" and row["is_pin"] is False


# ----------------------------------------------------------------------------- the split is verified

@pytest.mark.parametrize("dative,split", [
    (DATIVE_AMMINE_CHLORIDE, SPLIT_AMMINE_CHLORIDE),
    (DATIVE_BIPYRIDINE, SPLIT_BIPYRIDINE),
    (DATIVE_PYRITHIONE, IONIC_PYRITHIONE),
    (HALF_DATIVE_ACETATE, IONIC_ACETATE),
])
def test_dative_split_is_verified_by_key_and_count(dative, split):
    """The name reads back to the key of the split drawing (a fresh OPSIN call), and the count of
    atoms and charges between the dative drawing and its split drawing holds."""
    row = _row(dative, "best-effort")
    assert row["name"] and row["tier"] == "systematic_verified" and row["is_pin"] is False, row
    assert row["verified"] == "opsin_split", row                 # no claim of a comparison with the drawing as written
    assert_full_rt(row["name"], split)
    assert name_is_rt_exact(row["name"], dative)                 # the tool reads a dative input against its split
    written, order = metal_offer.metal_dative_split_spelling_and_order(dative)
    before, after = Chem.MolFromSmiles(dative), Chem.MolFromSmiles(written)
    sites = dative_bond_split_sites(before)
    assert atom_and_charge_ledger_ok(before, Chem.RenumberAtoms(after, _inverse(order)), sites, 0)
    assert _key(written) == _key(split)                          # the reference key is the split drawing's
    declined = _row(dative, "pin")
    assert declined["tier"] == "abstain" and declined["is_pin"] is False
    assert declined["verified"] == "unverified"


def _inverse(order):
    inverse = [0] * len(order)
    for position, atom_idx in enumerate(order):
        inverse[atom_idx] = position
    return inverse


def test_the_name_of_a_dative_drawing_is_the_name_of_its_ionic_drawing():
    """One compound, one name: the pyrithione chelate drawn with dative bonds, with covalent bonds
    and as ions (the chelate with covalent bonds: tests of ``test_metal_disconnect_rewrite``)."""
    ionic = _row(IONIC_PYRITHIONE, "best-effort")["name"]
    assert _row(DATIVE_PYRITHIONE, "best-effort")["name"] == ionic
    assert _row(DATIVE_ACETATE, "best-effort")["name"] == _row(IONIC_ACETATE, "best-effort")["name"]


# ----------------------------------------------------------------------------- the label

@pytest.mark.parametrize("dative,ionic,expected", [
    (DATIVE_ACETATE, IONIC_ACETATE, "zinc diacetate"),
    (DATIVE_WATER, IONIC_WATER, "zinc(2+)—water (1/2)"),
])
def test_a_dative_only_drawing_takes_its_label_from_the_split_not_from_m1(dative, ionic, expected):
    """The split of a drawing with dative bonds only has no Zn-X single bond, so M1 (DISCONNECT)
    finds nothing to break and sets nothing; the label is the split's own. The ionic twin of the
    acetate keeps pin_verified at the default tier (a salt of an acid,, so the decline of
    the dative drawing comes from the record, not from the name."""
    assert metal_offer.metal_disconnected_spelling_and_order(
        metal_offer.metal_dative_split_spelling_and_order(dative)[0]) is None
    row = _row(dative, "best-effort")
    assert row["name"] == expected and row["tier"] == "systematic_verified" and row["is_pin"] is False, row
    assert row["verified"] == "opsin_split"
    assert_full_rt(row["name"], ionic)
    declined = _row(dative, "pin")
    assert declined["tier"] == "abstain" and declined["is_pin"] is False, declined
    if dative == DATIVE_ACETATE:
        assert_default_tier_declines(dative)                      # NO_VERIFIED_PIN
        assert declined["spelling_failures"][0]["rule"] == RULE
        twin = _row(ionic, "pin")
        assert twin["tier"] == "pin_verified" and twin["is_pin"] is True and twin["name"] == expected


@pytest.mark.parametrize("dative", [DATIVE_ACETATE, DATIVE_AMMINE_CHLORIDE])
def test_the_flag_is_set_by_the_split_itself(dative, monkeypatch):
    """M1 switched off: the split drawing is still named, still labelled below the PIN and still
    declined at the default tier; the label does not depend on M1."""
    with_m1 = _row(dative, "best-effort")
    monkeypatch.setattr(metal_offer, "metal_disconnected_spelling_and_order", lambda smiles: None)
    row = _row(dative, "best-effort")
    assert row["name"] and row["tier"] == "systematic_verified" and row["is_pin"] is False, row
    assert row["verified"] == "opsin_split"
    assert_full_rt(row["name"], smiles_without_dative_bonds(dative))
    if dative == DATIVE_ACETATE:
        assert row["name"] == with_m1["name"] == "zinc diacetate"
        assert row["spelling_failures"][0]["rule"] == RULE
    default = _row(dative, "pin")
    assert default["tier"] == "abstain" and default["is_pin"] is False


# ----------------------------------------------------------------------------- the rewrite itself

def test_the_offer_lists_the_split_drawing_then_its_ionic_drawing():
    offer = metal_offer.metal_offer_for(DATIVE_AMMINE_CHLORIDE)
    assert offer.keyless is True and len(offer.drawings) == 2
    (split, split_order), (ionic, ionic_order) = offer.drawings
    assert Chem.MolToSmiles(Chem.MolFromSmiles(split)) == Chem.MolToSmiles(Chem.MolFromSmiles(SPLIT_AMMINE_CHLORIDE))
    assert Chem.MolToSmiles(Chem.MolFromSmiles(ionic)) == Chem.MolToSmiles(Chem.MolFromSmiles("[Cl-].[Cl-].[Zn+2].N.N"))
    count = Chem.MolFromSmiles(DATIVE_AMMINE_CHLORIDE).GetNumAtoms()
    assert sorted(split_order) == sorted(ionic_order) == list(range(count))   # atoms of the caller's reading
    # the order of the ionic drawing is composed: its atoms are the caller's atoms
    caller = Chem.MolFromSmiles(DATIVE_AMMINE_CHLORIDE)
    ionic_mol = Chem.MolFromSmiles(ionic)
    assert [ionic_mol.GetAtomWithIdx(k).GetSymbol() for k in range(count)] == [
        caller.GetAtomWithIdx(ionic_order[k]).GetSymbol() for k in range(count)]
    # a dative-only drawing has no second drawing; an ordinary drawing has no keyless offer
    assert len(metal_offer.metal_offer_for(DATIVE_ACETATE).drawings) == 1
    plain = metal_offer.metal_offer_for("Cl[Zn]Cl")
    assert plain.keyless is False and len(plain.drawings) == 1
    assert metal_offer.metal_offer_for(IONIC_ACETATE) is None


def test_the_split_is_idempotent_and_an_ionic_drawing_has_none():
    written, _order = metal_offer.metal_dative_split_spelling_and_order(DATIVE_ACETATE)
    assert metal_offer.metal_dative_split_spelling_and_order(written) is None
    assert metal_offer.metal_dative_split_spelling_and_order(IONIC_ACETATE) is None
    assert metal_offer.metal_dative_split_spelling_and_order("Cl[Zn]Cl") is None
    assert metal_offer.metal_dative_split_spelling_and_order("CC[Zn]CC") is None
    assert metal_offer.metal_dative_split_spelling_and_order("CC|") is None


@pytest.mark.parametrize("smiles", [
    "[Zn]->N",                                                  # the metal is the donor
    DATIVE_CU, DATIVE_NI,                                       # metals that are not admitted
    "C[P@](CC)(c1ccccc1)->[Zn+2]<-[O-]C(C)=O",                  # a stereocentre at the donor
    "N->[Zn+2].N->[Fe+2]",                                      # one metal admitted, one not
    "C/C=N(/C)->[Zn+2]",                                        # a stereo double bond at the donor
])
def test_the_sites_are_left_alone_when_the_split_is_not_safe(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    assert dative_bond_split_sites(mol) is None
    assert metal_offer.metal_dative_split_spelling_and_order(smiles) is None


def test_a_dative_split_that_changes_a_charge_is_rejected_by_the_count(monkeypatch):
    """A split that moves the charge from one oxygen of the acetate to the other keeps the standard
    InChIKey (the two oxygens are equivalent for it) and is another structure: the count rejects it,
    and the input stays unnamed."""
    before = Chem.MolFromSmiles(DATIVE_ACETATE)                 # atoms C0 C1 O2 O3 Zn4 O5 C6 C7 O8
    sites = dative_bond_split_sites(before)
    right = Chem.MolFromSmiles("CC(=O)[O-].[Zn+2].[O-]C(C)=O")
    wrong = Chem.MolFromSmiles("CC([O-])=O.[Zn+2].[O-]C(C)=O")   # -1 on the carbonyl oxygen
    lost = Chem.MolFromSmiles("CC(=O)O.[Zn+2].[O-]C(C)=O")      # a hydrogen added, the charge gone
    assert _key(Chem.MolToSmiles(right)) == _key(Chem.MolToSmiles(wrong))
    assert atom_and_charge_ledger_ok(before, right, sites, 0)
    assert not atom_and_charge_ledger_ok(before, wrong, sites, 0)
    assert not atom_and_charge_ledger_ok(before, lost, sites, 0)
    assert not atom_and_charge_ledger_ok(before, right, sites)   # the default moves +1 and -1: the count differs

    def moved(mol, found):
        rw = Chem.RWMol(mol)
        for metal_idx, donor_idx in found:
            rw.RemoveBond(metal_idx, donor_idx)
        out = rw.GetMol()
        out.GetAtomWithIdx(3).SetFormalCharge(0)
        out.GetAtomWithIdx(2).SetFormalCharge(-1)
        Chem.SanitizeMol(out)
        return out

    monkeypatch.setattr(metal_offer, "_DATIVE_KIND", metal_offer._DATIVE_KIND._replace(rewrite=moved))
    assert metal_offer.metal_dative_split_spelling_and_order(DATIVE_ACETATE) is None
    row = _row(DATIVE_ACETATE, "best-effort")
    assert row["tier"] == "abstain" and row["limit_code"] == "UNNAMEABLE"


# ----------------------------------------------------------------------------- the provenance field

def test_verified_field_reports_split_only_for_a_recorded_name():
    prov = {"split_verified_names": ("zinc diacetate",)}
    assert _provenance.verified_field("opsin", prov, "zinc diacetate") == "opsin_split"
    assert _provenance.verified_field("opsin", prov, "zinc dichloride") == "opsin"
    assert _provenance.verified_field("opsin", {}, "zinc diacetate") == "opsin"
    assert _provenance.verified_field("identity", prov, "zinc diacetate") == "identity"
    assert _provenance.verified_field("unverified", prov, "zinc diacetate") == "unverified"


def test_the_split_record_has_the_lifetime_of_the_other_name_records():
    _provenance.clear_provenance()
    _provenance.record_verified_through_split("zinc diacetate")
    snapshot = _provenance.get_provenance()
    assert snapshot["split_verified_names"] == ("zinc diacetate",)
    _provenance.clear_provenance()
    assert _provenance.get_provenance()["split_verified_names"] == ()
    _provenance.restore_provenance(snapshot)
    assert _provenance.get_provenance()["split_verified_names"] == ("zinc diacetate",)
    _provenance.clear_provenance()


def test_the_nested_memo_replays_every_provenance_key():
    from orthonym.assembly import nested_memo
    assert set(nested_memo._VARS) == set(_provenance.get_provenance())
    assert "split_verified_names" in nested_memo._ACCUMULATORS


def test_the_independent_reference_of_the_read_back():
    assert smiles_without_dative_bonds("CCO") is None
    assert smiles_without_dative_bonds("Cl[Zn]Cl") is None
    assert (Chem.MolToSmiles(Chem.MolFromSmiles(smiles_without_dative_bonds(DATIVE_ACETATE)))
            == Chem.MolToSmiles(Chem.MolFromSmiles("CC(=O)[O-].[Zn+2].[O-]C(C)=O")))
    assert name_is_rt_exact("zinc diacetate", DATIVE_ACETATE)
    assert not name_is_rt_exact("zinc dichloride", DATIVE_ACETATE)
    assert not name_is_rt_exact("zinc diacetate", "O->[Zn+2]<-O")
