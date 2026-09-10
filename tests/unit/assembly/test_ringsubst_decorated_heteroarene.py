""" ring-substituent tranche — decorated monocyclic heteroaromatic
substituents name via the recursive composer (unified stem source, resolves
Composer #1 I1 duplication).

The recursion's `_monocycle_core_tail` sourced heteroarene stems from the small
`_PIN_HETEROARYL_STEMS` table, which (via identify_ring_system -> None) does NOT
cover the systematic azoles (isoxazole/oxazole/thiazole/triazole...) that the
authoritative dispatcher `get_ring_substituent_name` DOES name. So a DECORATED
azolyl substituent (which must go through the recursion because the narrow
decorated producer shares the incomplete table) declined. makes the gated
core-tail borrow the stem from the dispatcher.

Producers are called DIRECTLY (unit tests disable the OPSIN gate); every new
path is reached only under allow_mancude=True (best-effort/complete tier) so the
PIN default is byte-identical.
"""
from rdkit import Chem
from orthonym.assembly.substituent_enumerator import (
    name_substituent, _recursive_fragment_substituent_name)


def _frag(smiles, attach=0):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    return mol, list(range(mol.GetNumAtoms())), attach


def test_bare_isoxazolyl_via_recursion():
    # The recursion must name a bare isoxazol-3-yl (so decorated versions can
    # carry decorations on it). The dispatcher already names it '1,2-oxazol-3-yl'.
    mol, frag, attach = _frag("c1ccon1")  # isoxazole, ring C at idx 0
    got = _recursive_fragment_substituent_name(
        mol, frag, attach, allow_mancude=True)
    assert got is not None, "recursion still declines bare isoxazolyl"
    assert 'oxazol' in got and got.endswith('-yl'), got


def test_bare_isoxazolyl_pin_default_unaffected():
    # allow_mancude=False (PIN default) MUST still decline (byte-identical).
    mol, frag, attach = _frag("c1ccon1")
    got = _recursive_fragment_substituent_name(
        mol, frag, attach, allow_mancude=False)
    assert got is None


def test_decorated_methylisoxazolyl():
    # A methyl-decorated isoxazolyl must compose via name_substituent under
    # allow_mancude (the narrow decorated producer declines it -> recursion).
    mol, frag, attach = _frag("Cc1ccon1")  # methyl on the isoxazole ring
    got = name_substituent(mol, frag, attach, allow_mancude=True)
    assert got is not None and got != 'substituent', got
    assert 'oxazol' in got and 'methyl' in got, got
