"""Roadmap item 12a (task t1): the chain kind of the spelling ladder.

``book_prefixes.RETRY_KEEP_ORDER`` is the ladder book -> keep-forms rungs -> mechanical (lane
U1): one failing spelling of a kind that is not kept does not take the kept kinds' names with it.
This lane adds the kind 'chain' (the N/O group prefixes and the 'cyano' branch,,
the Blue Book, section General rules, "The chain must be terminated by a C atom or
one of the following heteroatoms: P, As, Sb, Bi, Si, Ge, Sn, Pb, B, Al, Ga, In, or Tl") after
'ring'. ``name_universal_substitutive`` never returns None where an earlier spelling names the
molecule: when the book build voids or raises it retries without the chain kind (the spelling
before the item), then mechanically.
"""
import pytest
from rdkit import Chem

from orthonym.assembly.book_prefixes import RETRY_KEEP_ORDER, retry_keeps
from orthonym.assembly.universal_substituent import name_universal_substitutive


def test_the_chain_kind_follows_the_ring_kind():
    assert RETRY_KEEP_ORDER[:2] == (frozenset({"ring"}), frozenset({"chain"}))


def test_a_rung_runs_only_when_its_kind_was_given_with_another():
    assert retry_keeps({"chain", "other"}) == [frozenset({"chain"})]
    assert retry_keeps({"chain"}) == []
    assert retry_keeps({"ring", "chain"}) == [frozenset({"ring"}), frozenset({"chain"})]


@pytest.mark.parametrize("smiles", ["N#C[S-]", "N#C[O-]", "COC#N", "N#CC#N", "C[N+](C)(C)[O-]"])
def test_a_book_spelling_never_voids_a_molecule_the_mechanical_one_names(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert name_universal_substitutive(mol, book_forms=False) is not None
    assert name_universal_substitutive(mol) is not None


@pytest.mark.opsin_gate
def test_a_failing_chain_prefix_costs_only_the_chain_kind(monkeypatch):
    """A chain prefix that fails the round trip sends the floor to the rung 'every book
    spelling except the chain kind' (``book_prefixes.RETRY_DROP_ORDER``), not to the keep rungs
    or the mechanical spelling: the methoxy group and the benzene rings keep their book names."""
    from orthonym.assembly import hetero_group_prefixes as H
    from orthonym.assembly import t4_coverage
    real = H.compose_group
    monkeypatch.setattr(H, "compose_group",
                        lambda shape, names: "zzz" if shape.kind == "diazenyl" else real(shape, names))
    smiles = "COc1ccccc1N=Nc1ccccc1"
    mol = Chem.MolFromSmiles(smiles)
    name = t4_coverage._verified_universal_floor(mol, Chem.MolToSmiles(mol))
    assert name is not None and "zzz" not in name
    assert "methoxy" in name and "benzene" in name and "diaza" in name


def test_the_rungs_are_drop_then_keep():
    from orthonym.assembly.book_prefixes import retry_rungs
    assert retry_rungs({"ring", "chain", "other"}) == [
        ("drop", frozenset({"chain"})), ("keep", frozenset({"ring"})),
        ("keep", frozenset({"chain"}))]
    assert retry_rungs({"chain"}) == []
