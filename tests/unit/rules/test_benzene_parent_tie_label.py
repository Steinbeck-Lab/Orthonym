"""Of benzene rings that tie on the number of prefixes, the parent is decided by the lower locant set, the locants in
order of citation, then the earlier name in alphanumerical order ('1-bromo-3-[(3-chlorophenyl)methyl]benzene (PIN) [not
1-[(3-bromophenyl)methyl]-3-chlorobenzene]', the Blue Book). Where the engine's choice skipped a tied ring (the
'carbon-linked' pre-filter) and that ring gives the preferred name, the shipped name is kept below the PIN label
(rules/benzene_parent_tie.py); the name itself does not change.
"""
import pytest

from orthonym.cli import _tiered_namer

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles", [
    "C1=CC=C(C=C1)COC2=CC=CC=C2",                      # (phenoxymethyl)benzene: '(benzyloxy)benzene' is earlier
    "Fc1cccc(Oc2cccc(C)c2)c1",                         # 1-(3-fluorophenoxy)-3-methylbenzene: 'fluoromethylphenoxy...'
])
def test_a_choice_that_loses_the_tie_is_no_pin(smiles):
    row = _tiered_namer("pin", "best-effort").name_tiered(smiles)
    assert row["tier"] != "pin_verified" and not row["is_pin"], row


@pytest.mark.parametrize("smiles", [
    "BrCc1cccc(Oc2ccc(F)cc2)c1",                       # the shipped ring wins on the locant set {1,3} < {1,4}
    "FC1=CC=C(C=C1)C#CC2=CC=C(C=C2)C(F)(F)F",          # both rings compared by the selector already
])
def test_a_choice_the_criteria_confirm_keeps_the_pin(smiles):
    row = _tiered_namer("pin", "best-effort").name_tiered(smiles)
    assert row["tier"] == "pin_verified", row


@pytest.mark.parametrize("smiles,name", [
    ("Cc1ccc(COc2ccccc2)cc1", "1-methyl-4-(phenoxymethyl)benzene"),   # more prefixes decide (:21618)
    ("Clc1ccc(Oc2ccc(Cl)cc2)cc1", "1,1'-oxybis(4-chlorobenzene)"),     # identical rings: multiplicative
])
def test_decided_choices_keep_their_name(smiles, name):
    row = _tiered_namer("pin", "best-effort").name_tiered(smiles)
    assert row["name"] == name, row


def test_a_rival_name_that_drops_atoms_decides_nothing(monkeypatch):
    """A rival name that does not read back to the whole molecule ('sulfanylbenzene' for the other ring of
    '[2-(phenylsulfanyl)propyl]benzene') gives no answer, so the label is not changed on its strength."""
    from rdkit import Chem
    import orthonym.rules.benzene as bz
    import orthonym.rules.benzene_parent_tie as tie
    mol = Chem.MolFromSmiles("CC(Cc1ccccc1)Sc1ccccc1")
    rings = [r for r in mol.GetRingInfo().AtomRings() if bz.is_benzene_ring(mol, r)]
    shipped = (rings[1], "[2-(phenylsulfanyl)propyl]benzene")
    monkeypatch.setattr(tie, "_ring_name", lambda m, r, p: "(acetylsulfanyl)benzene")   # would win if accepted
    assert tie.chosen_loses_the_tie(mol, [shipped], shipped, bz.is_benzene_ring, bz._alphanumerical_name_key) is None
