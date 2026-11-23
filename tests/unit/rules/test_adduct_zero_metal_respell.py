"""A neutral metal atom beside ions: the oxidation number zero and the component spellings OPSIN reads beside it.

A reader balances the charges of the components of an adduct name itself: OPSIN reads 'acetate—cobalt (2/1)' as
cobalt(2+) and 'acetate—cobalt(0) (2/1)' as acetic acid beside the neutral atom. The drawing (two acetate ions and a
neutral cobalt atom) has the name '1-oxido-1-oxoethane—cobalt(0) (2/1)', whose 'oxido' prefix OPSIN does not
neutralise (``rules.adducts._zero_metal_drawn_name``). An adduct name is no PIN, the Blue Book), so
these names ship at the best-effort tier only, each read back by OPSIN to the drawn structure.

The f-block metals, technetium and the superheavy metals take the zero as well ('curium(0)': ``data/metal_scope.py``
admits ``ELEMENT_ONLY_METALS`` to ``ZERO_VALENT``).
"""
import pytest

from tests.support.rt_assert import name_best_effort, name_is_rt_exact

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,name", [
    ("CC(=O)[O-].CC(=O)[O-].[Co]", "1-oxido-1-oxoethane—cobalt(0) (2/1)"),
    ("O=S(=O)([O-])C(F)(F)F.[Pt]", "trifluoro(sulfonato)methane—platinum(0) (1/1)"),
    ("CC(=O)[O-].[Cm]", "1-oxido-1-oxoethane—curium(0) (1/1)"),
    ("[Cl-].[Cl-].[Tc]", "chloride—technetium(0) (2/1)"),
])
def test_a_neutral_metal_beside_anions_ships_with_spellings_opsin_reads_as_drawn(smiles, name):
    r = name_best_effort(smiles)
    assert r["name"] == name and r["verified"] == "opsin", r
    assert r["tier"] == "systematic_verified" and not r["is_pin"], r
    assert name_is_rt_exact(name, smiles)


def test_the_zero_metal_step_respells_only_at_the_tiers_that_respell():
    """Without the respelling flags (the PIN tier) only the plain zero-metal name is tried."""
    from rdkit import Chem
    from orthonym.rules import adducts as A
    mol = Chem.MolFromSmiles("CC(=O)[O-].CC(=O)[O-].[Co]")
    ordered = [("CC(=O)[O-]", 2), ("[Co]", 1)]
    frags = {smi: Chem.MolFromSmiles(smi) for smi, _ in ordered}
    named = [("acetate", 2), ("cobalt", 1)]
    assert A._zero_metal_drawn_name(ordered, named, frags, mol, "pin", None) is None
    flags = dict(general_fallback=True, allow_aromatic_general=True, general_fallback_unverified=True)
    assert A._zero_metal_drawn_name(ordered, named, frags, mol, "pin", flags) == "1-oxido-1-oxoethane—cobalt(0) (2/1)"


def test_the_metal_word_is_never_respelled_and_a_moved_charge_is_refused(monkeypatch):
    """A one-atom component has no other spelling, and a reading with a charge on another component than drawn
    fails the exact depiction compare, so no zero-metal name of another structure ships."""
    from rdkit import Chem
    from orthonym.rules import adducts as A
    assert A._component_spellings_once("[Co]", "cobalt(0)", "pin", {}) == []
    mol = Chem.MolFromSmiles("CC(=O)[O-].CC(=O)[O-].[Co]")
    monkeypatch.setattr(A, "_adduct_opsin_read", lambda name: "CC(=O)O.CC(=O)[O-].[Co+]")
    assert not A._parse_reproduces_depiction("1-oxido-1-oxoethane—cobalt(0) (2/1)", mol)


def test_the_respelling_read_back_takes_dihydrogen_with_every_hydrogen_an_atom(monkeypatch):
    """OPSIN writes dihydrogen '[H][H]' where RDKit writes '[HH]'; a moved hydrogen stays refused."""
    from orthonym.rules import adducts as A
    monkeypatch.setattr(A, "_adduct_opsin_read", lambda name: "CC(=O)[O-].[H][H]")
    assert A._reads_back_as("x", "CC(=O)[O-].[HH]")
    monkeypatch.setattr(A, "_adduct_opsin_read", lambda name: "CC(=O)O.[H][H]")
    assert not A._reads_back_as("x", "CC(=O)[O-].[HH]")


@pytest.mark.parametrize("smiles", ["[HH].[Li].CC[CH2]"])
def test_a_radical_beside_lithium_and_dihydrogen_takes_the_bonded_name(smiles):
    """The radical identity check now reads the drawn '[HH]' as OPSIN's '[H][H]', so the lithium radical drawing
    takes the name of its bonded drawing, as it does without the dihydrogen ('propyllithium', mechanism
    ``RECONNECT_RADICAL`` of ``data/metal_scope.py``); OPSIN reads it back to the same full InChIKey."""
    r = name_best_effort(smiles)
    assert r["name"] in ("propyllithium—dihydrogen (1/1)", "dihydrogen—propyllithium (1/1)"), r
    assert not r["is_pin"], r
