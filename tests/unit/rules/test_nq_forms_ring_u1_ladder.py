"""Item 12a (task 3): one book spelling that no round trip confirms does not take the ring
names of the molecule with it. The retry rungs keep the single-ring names,
the Blue Book;,:23682) before they give up the book forms."""
import sys
from pathlib import Path

import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.assembly.book_prefixes import mechanical_forms, ring_forms_enabled
from orthonym.assembly.universal_substituent import name_universal_substitutive
from orthonym.cli import _emit_tier_flags
from orthonym.rules.monocycle_forms import monocycle_form

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "eval"))
from tests.support.rt_assert import name_is_rt_exact  # noqa: E402


def _ring_and_fv(smiles):
    mol = Chem.MolFromSmiles(smiles)
    ring = next(r for r in mol.GetRingInfo().AtomRings()
                if any(mol.GetBondBetweenAtoms(1, a) for a in r))
    fv = next(a for a in ring if mol.GetBondBetweenAtoms(1, a))
    return mol, list(ring), fv


def _form(smiles, **kw):
    mol, ring, fv = _ring_and_fv(smiles)
    branches = [a for a in ring for nb in mol.GetAtomWithIdx(a).GetNeighbors()
                if nb.GetIdx() not in ring and nb.GetIdx() != 1]
    return fv, monocycle_form(mol, ring, fv, branches, **kw)


def _row(smiles, tier):
    return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


# --- one book spelling that no round trip confirms must not take the ring names with it ---

def test_the_ring_rung_keeps_the_ring_names_when_the_other_book_forms_are_off():
    smiles = "C[C@H]1O[C@@H](O[C@]23C(=Cc4ccc(CC#N)c(Cl)c42)C=C[C@@H]3O)[C@H](O)C(=O)[C@@]1(C)O"
    mol = Chem.MolFromSmiles(smiles)
    ring = name_universal_substitutive(mol, book_forms=False, keep_forms=frozenset({"ring"}))
    assert ring is not None and "oxan-2-yl" in ring.name and "oxacyclohexan" not in ring.name
    assert name_is_rt_exact(ring.name, smiles)
    mech = name_universal_substitutive(mol, book_forms=False)
    assert "oxacyclohexan" in mech.name


def test_the_retry_scope_with_keep_rings_leaves_the_ring_writers_on():
    assert ring_forms_enabled()
    with mechanical_forms():
        assert not ring_forms_enabled()
        with mechanical_forms(keep_forms={"ring"}):
            assert ring_forms_enabled()
        assert not ring_forms_enabled()
    assert ring_forms_enabled()


@pytest.mark.opsin_gate
def test_a_failing_fused_spelling_does_not_cost_the_molecule_its_ring_names(monkeypatch):
    """The first run's name carries a fused-ring spelling; rejecting it at the round trip
    must leave 'oxan-2-yl', not the 'a' replacement ring, in the name that ships."""
    from orthonym.validation import reconstruct
    real = reconstruct.verify_or_none

    def reject_annulene(name, smi, *a, **k):
        return None if "annulene" in name else real(name, smi, *a, **k)

    monkeypatch.setattr(reconstruct, "verify_or_none", reject_annulene)
    smiles = "C[C@H]1O[C@@H](O[C@]23C(=Cc4ccc(CC#N)c(Cl)c42)C=C[C@@H]3O)[C@H](O)C(=O)[C@@]1(C)O"
    name = _row(smiles, "best-effort")["name"]
    assert "oxan-2-yl" in name and "oxacyclohexan" not in name
    assert name_is_rt_exact(name, smiles)


def _abstain_row():
    return {"name": "unknown organic compound", "tier": "abstain", "limit_code": "UNNAMEABLE"}


@pytest.mark.opsin_gate
@pytest.mark.parametrize("kinds,expected", [
    (("other",), [(True, True), (False, False)]),
    (("ring",), [(True, True), (False, False)]),     # only ring spellings: the rung would repeat the run
    (("ring", "other"), [(True, True), (False, True), (False, False)]),
])
def test_the_keep_rings_retry_runs_only_when_a_ring_and_another_spelling_were_given(
        monkeypatch, kinds, expected):
    from orthonym.assembly import book_prefixes as bp
    calls = []

    def spy(self, smiles):
        calls.append((bp.book_forms_enabled(), bp.ring_forms_enabled()))
        if bp.book_forms_enabled():
            for k in kinds:
                bp.note_book_form(k)
        return _abstain_row()
    monkeypatch.setattr(Orthonym, "_name_tiered_scoped", spy)
    Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered("CCO")
    assert calls == expected


@pytest.mark.opsin_gate
def test_the_retry_never_returns_a_lower_label_than_the_run_it_replaces(monkeypatch):
    """Book run: abstain. Keep-rings run: best_effort. Mechanical run: systematic_verified.
    The keep-rings row does not end the ladder: the mechanical row is the one base returns."""
    from orthonym.assembly import book_prefixes as bp
    seq = iter([("abstain", True), ("best_effort", False), ("systematic_verified", False)])

    def spy(self, smiles):
        tier, fires = next(seq)
        if fires:
            bp.note_book_form("ring")
            bp.note_book_form("other")
        return {"name": "x" if tier != "abstain" else "unknown organic compound", "tier": tier}
    monkeypatch.setattr(Orthonym, "_name_tiered_scoped", spy)
    row = Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered("CCO")
    assert row["tier"] == "systematic_verified"


def test_the_ladder_order_is_named_in_one_place():
    from orthonym.assembly.book_prefixes import RETRY_KEEP_ORDER, retry_keeps
    assert RETRY_KEEP_ORDER == (frozenset({"ring"}), frozenset({"chain"}))  # item 12a adds 'chain'
    assert retry_keeps({"ring", "other"}) == [frozenset({"ring"})]
    assert retry_keeps({"ring", "chain", "other"}) == [frozenset({"ring"}), frozenset({"chain"})]
    assert retry_keeps({"ring"}) == [] and retry_keeps({"other"}) == [] and retry_keeps(set()) == []
