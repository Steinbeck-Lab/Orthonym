"""Lane L2 proper fix: the writers decide a form from facts they hold, never from the text
of a part they built the Blue Book,:3007, (a):2893,
 (1):15813). A test re-spells a part and checks that the decision does not move."""
import pytest
from rdkit import Chem

from orthonym.assembly.book_prefixes import positions_alike_prefix
from orthonym.assembly.universal_substituent import name_universal_substituent_prefix
from orthonym.rules.monocycle_forms import monocycle_form
from orthonym.rules.terminal_fragment import terminal_fragment_name


def test_positions_alike_takes_the_writers_fact():
    mol = Chem.MolFromSmiles("FC(F)(F)C(F)(F)F")
    assert positions_alike_prefix(mol, [1, 4], ["fluoro"] * 5,
                                  unit_cites_locant=False) == "pentafluoro"
    assert positions_alike_prefix(mol, [1, 4], ["fluoro"] * 5,
                                  unit_cites_locant=True) is None


@pytest.mark.parametrize("smiles,fv,expected", [
    ("c1ccncc1", None, False), ("c1ccncc1", 1, True), ("c1ccccc1", None, False),
    ("C1CCCCC1", None, False), ("c1c[nH]cn1", None, True), ("c1cocn1", None, True),
])
def test_the_monocycle_form_knows_whether_it_cites_a_locant(smiles, fv, expected):
    m = Chem.MolFromSmiles(smiles)
    ring = list(m.GetRingInfo().AtomRings()[0])
    form = monocycle_form(m, ring, ring[0] if fv else None)
    loc = form.numbering[ring[0]] if fv else None
    assert form.cites_locant(loc) is expected


def test_the_methyl_group_rule_does_not_read_the_chain_name(monkeypatch):
    import orthonym.rules.terminal_fragment as tf
    orig = tf.get_alkyl_name
    monkeypatch.setattr(tf, "get_alkyl_name", lambda n: "respelled" if n == 1 else orig(n))
    mol = Chem.MolFromSmiles("FC(F)(F)c1ccccc1")
    assert terminal_fragment_name(mol, {0, 1, 2, 3}, 1).name == "trifluoromethyl"


def test_the_floor_methyl_group_rule_does_not_read_the_spine_name(monkeypatch):
    import orthonym.assembly.universal_substituent as us
    orig = us._build_parent_with_unsaturation
    monkeypatch.setattr(us, "_build_parent_with_unsaturation",
                        lambda n, u, fg_suffix=None: "respelled" if n == 1 else orig(n, u, fg_suffix))
    mol = Chem.MolFromSmiles("BrC(Cl)c1ccccc1")
    assert name_universal_substituent_prefix(mol, [0, 1, 2], 1) == "bromo(chloro)methyl"


def test_the_floor_alkyl_rule_does_not_read_the_chain_name(monkeypatch):
    import orthonym.data.chain_names as cn
    orig = cn.get_chain_name
    monkeypatch.setattr(cn, "get_chain_name", lambda n: "respelled" if n == 2 else orig(n))
    mol = Chem.MolFromSmiles("ClCCc1ccccc1")
    assert name_universal_substituent_prefix(mol, [0, 1, 2], 2) == "2-chloroethyl"
