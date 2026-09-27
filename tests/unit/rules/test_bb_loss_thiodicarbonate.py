"""Suite fix 'BB loss thiodicarbonate' (TRIAGE.md): the carbonic / polycarbonic
acid dianions keep their PIN after 44575ba9a made the junior anionic prefix
step fail closed.

The poly-anion path names O=C([O-])OC([O-])=S by converting the neutral
'1-thiodicarbonic acid' to '1-thiodicarbonate' and then asking
``_apply_anionic_substituent_prefixes`` for any junior site. Its [O-] on the
=S carbon is classified 'alkoxide' and the '-oate'-only suffix count did not
see that '-dicarbonate' already holds both acid sites, so the site was counted
as a junior 'oxido' the name cannot hold. Before 44575ba9a that branch passed
the name through unchanged; after it, the branch returns '' and the PIN tier
fell to '2,4-dioxido-1,3-dioxa-5-thiapenta-1,4-diene'.

Blue Book: (the Blue Book, "Anions derived from acids"):
"The preferred IUPAC name of anions formed by the removal of a hydron from the
chalcogen atom (O, S, Se, and Te) of an acid or peroxyacid characteristic group
or functional parent compound is formed by replacing the 'ic acid' or 'ous
acid' ending of the acid name by 'ate' or 'ite', respectively."
 (:32025, "Esters of carbonic acid, cyanic acid, and
polycarbonic acids modified by functional replacement"): "3-ethyl 1-S-methyl
1-thiodicarbonate (PIN)" (:32029), the ester of the dianion
'1-thiodicarbonate'. The acid PINs are (:31021-31023 'dicarbonic
acid (PIN)', 'tricarbonic acid (PIN)') and (:31047-31053
'2-thiodicarbonic acid (PIN)', '2-imidodicarbonic acid (PIN)',
'2-peroxydicarbonic acid (PIN)', '1-imidodicarbonic acid (PIN)').

Each name is asserted exactly at the PIN tier with the gate on and checked by
an independent full-InChIKey OPSIN round trip (tests/support/rt_assert.py).
"""
import pytest
from rdkit import Chem

from orthonym.namer import Orthonym
from orthonym.perception.ions import get_ion_sites
from orthonym.rules.ions import _apply_anionic_substituent_prefixes
from tests.support.rt_assert import name_is_rt_exact


CARBONIC_DIANIONS = [
    ("O=C([O-])OC([O-])=S", "1-thiodicarbonate"),     # the BB row,
    ("O=C([O-])OC([O-])=O", "dicarbonate"),
    ("O=C([O-])SC([O-])=O", "2-thiodicarbonate"),
    ("O=C([O-])NC([O-])=O", "2-imidodicarbonate"),
    ("O=C([O-])OOC([O-])=O", "2-peroxydicarbonate"),
    ("N=C([O-])OC([O-])=O", "1-imidodicarbonate"),
    ("O=C([O-])OC(=O)OC([O-])=O", "tricarbonate"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", CARBONIC_DIANIONS,
                         ids=[e for _, e in CARBONIC_DIANIONS])
def test_carbonic_dianion_pin(smiles, expected):
    r = Orthonym().name_tiered(smiles)
    assert r["name"] == expected
    assert r["tier"] == "pin_verified"
    assert name_is_rt_exact(expected, smiles)


@pytest.mark.parametrize("smiles,name", [
    ("O=C([O-])OC([O-])=S", "1-thiodicarbonate"),
    ("N=C([O-])OC([O-])=O", "1-imidodicarbonate"),
    ("O=C([O-])OC([O-])=O", "dicarbonate"),
])
def test_carbonic_parent_holds_both_sites(smiles, name):
    """Producer contract: the '-carbonate' suffix claims both acid sites, so
    there is no junior site and the name passes through, not ''."""
    mol = Chem.MolFromSmiles(smiles)
    anions = get_ion_sites(mol)["anions"]
    assert _apply_anionic_substituent_prefixes(mol, anions, name) == name


def test_site_left_after_the_carbonic_claim_is_junior():
    """Every carboxylate site the '-carbonate' parent did not claim is junior
     'carboxylato'), not discounted by the '-oate' multiplicity: the
    unconverted 'carboxymethyl' name denotes the dianion (OPSIN key differs)."""
    smi = "O=C([O-])N(CC(=O)[O-])C([O-])=O"
    mol = Chem.MolFromSmiles(smi)
    anions = get_ion_sites(mol)["anions"]
    out = _apply_anionic_substituent_prefixes(
        mol, anions, "2-(carboxymethyl)-2-imidodicarbonate")
    assert out == "2-(carboxylatomethyl)-2-imidodicarbonate"
    assert name_is_rt_exact(out, smi)
    assert not name_is_rt_exact("2-(carboxymethyl)-2-imidodicarbonate", smi)


def test_unexpressible_junior_site_still_fails_closed():
    """The intended fix of 44575ba9a holds: a junior -O-P(=O)(OH)O- site that
    no prefix expresses returns '' (the claim is scoped to '-carbonate')."""
    smi = "O=C([O-])COP(=O)(O)[O-]"
    mol = Chem.MolFromSmiles(smi)
    anions = get_ion_sites(mol)["anions"]
    assert _apply_anionic_substituent_prefixes(mol, anions, "(phosphonooxy)acetate") == ""
