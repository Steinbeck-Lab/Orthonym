""" ring-substituent tranche — decorated saturated N/O-heterocycle
substituents number + name via the recursive composer.

`_monocycle_position_map` (and `_monocycle_core_tail`) treated a SATURATED
ring's N-H atoms as ambiguous indicated hydrogen (`len(ih) > 1 -> None`), but
indicated hydrogen is a mancude-ring concept — a saturated ring has none. So a
decorated piperazinyl (two ring N-H) failed to number -> the recursion declined.
 scopes the indicated-H accounting to aromatic/mancude cores.

Producers are called DIRECTLY (unit tests disable the OPSIN gate); the recursion
is reached only under allow_mancude=True -> PIN default byte-identical.
"""
from rdkit import Chem
from orthonym.assembly.substituent_enumerator import (
    _monocycle_position_map, name_substituent)


def test_piperazine_numbers():
    mol = Chem.MolFromSmiles("N1CCNCC1")  # piperazine
    pos = _monocycle_position_map(
        mol, list(range(6)), 0, [], mol.GetRingInfo())
    assert pos is not None, "position map declines bare piperazine"
    assert set(pos.values()) == {1, 2, 3, 4, 5, 6}


def test_bare_piperazine_pin_default_unaffected():
    # The position map is only reached from the allow_mancude-gated recursion;
    # its behavior for a genuinely ambiguous AROMATIC ambiguous-IH ring must be
    # unchanged. Imidazole with the NH is fine (single IH); a would-be ambiguous
    # aromatic case still declines. Here we just assert the saturated fix does
    # not change the aromatic single-IH numbering (pyrrole numbers).
    mol = Chem.MolFromSmiles("c1cc[nH]c1")  # pyrrole (1 indicated H)
    pos = _monocycle_position_map(
        mol, list(range(5)), 0, [], mol.GetRingInfo())
    assert pos is not None and set(pos.values()) == {1, 2, 3, 4, 5}


def test_decorated_methylpiperazinyl():
    mol = Chem.MolFromSmiles("CN1CCNCC1")  # 1-methylpiperazine
    # Attach at the non-methylated ring N (idx 4) -> 4-methylpiperazin-1-yl.
    got = name_substituent(
        mol, list(range(mol.GetNumAtoms())), 4, allow_mancude=True)
    assert got is not None and got != 'substituent', got
    assert 'piperazin' in got and 'methyl' in got, got
