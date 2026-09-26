"""The static fragment-name cache holds the names the top-level namer gives (fix
a performance pass, wp7; whole-branch verification panel NIT on 'disodium 2-oxidoethanoate').

``fragment_naming.FRAGMENT_NAME_CACHE`` short-circuits naming for common fragments. Its
header says every entry was verified against name_compound; ten had drifted to
non-PIN names, and they reached shipped names: the carboxylate namer neutralises the
anion and names the acid through this cache, so OCC(=O)[O-].[Na+] shipped 'sodium
2-hydroxyethanoate' and the dianion 'disodium 2-oxidoethanoate' at pin_verified while
the neutral acid was 'hydroxyacetic acid' and the thio analogue 'disodium
sulfidoacetate'. the Blue Book 'HO-CH2-COOH hydroxyacetic acid (PIN)';:2004
acetic acid is the retained PIN; the other nine are cited at their entries.

The amino acids are the one exception: user decision D-a keeps the semisystematic
amino-acid names at the PIN tier (the Blue Book identifies no PIN for names,
 :50943), so a stereo-free 'alanine' entry is not re-derived here.
"""
import pytest

from orthonym import name_compound
from orthonym.assembly import fragment_naming

pytestmark = pytest.mark.unit

_D_A_AMINO_ACIDS = {
    "CC(N)C(=O)O", "NC(CO)C(=O)O", "NC(CS)C(=O)O", "NC(Cc1ccc(O)cc1)C(=O)O",
    "NC(Cc1c[nH]cn1)C(=O)O", "NC(Cc1c[nH]c2ccccc12)C(=O)O", "NCCCCC(N)C(=O)O",
    "NC(CCC(=O)O)C(=O)O", "NC(CC(=O)O)C(=O)O", "NC(=O)CC(N)C(=O)O", "CSCCC(N)C(=O)O",
}

_ENTRIES = sorted(k for k in fragment_naming.FRAGMENT_NAME_CACHE if k not in _D_A_AMINO_ACIDS)


@pytest.mark.parametrize("smiles", _ENTRIES)
def test_cache_entry_is_the_top_level_name(smiles, monkeypatch):
    cached = fragment_naming.FRAGMENT_NAME_CACHE[smiles]
    # name the fragment WITHOUT its own cache entry, so the check is not circular
    monkeypatch.delitem(fragment_naming.FRAGMENT_NAME_CACHE, smiles)
    assert name_compound(smiles) == cached


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    ("OCC(=O)[O-].[Na+]", "sodium hydroxyacetate"),
    ("[Na+].OCC([O-])=O", "sodium hydroxyacetate"),
    ("OCC(=O)[O-]", "hydroxyacetate"),
    ("[O-]CC(=O)[O-].[Na+].[Na+]", "disodium oxidoacetate"),
    ("[Na+].[Na+].[O-]C(=O)C[O-]", "disodium oxidoacetate"),
    ("[S-]CC(=O)[O-].[Na+].[Na+]", "disodium sulfidoacetate"),
])
def test_hydroxyacetate_anions_take_the_acetic_acid_pin(smiles, expected):
    from orthonym import Orthonym
    from tests.support.rt_assert import name_is_rt_exact
    r = Orthonym(style="pin").name_tiered(smiles)
    assert r["name"] == expected and r["tier"] == "pin_verified", r
    assert name_is_rt_exact(expected, smiles)
