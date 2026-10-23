"""``substituent_enumerator._substituent_memo_key`` and a molecule that has more than one fragment.

The key roots the fragment SMILES at the attachment atom. RDKit refuses a rooted fragment SMILES of
a molecule with more than one fragment ("rootedAtAtom can only be used with molecules that have a
single fragment") and logs the refusal on every call. The key function asks the question itself, so
nothing reaches RDKit and nothing is logged: a rooted fragment of such a molecule (a mixture named as
one graph) is keyed by its unrooted SMILES and the place of the attachment atom in the canonical atom
order. The key used to be ``None`` there (no caching), so the strict twin's promotion re-run asked the
strict ring-prefix producer and ``name_substituent`` the same question tens of thousands of times for a
mixture (a 3.3 s row took more than 150 s). A fragment that is not rooted (the attachment atom is
outside it) and a molecule with one fragment are keyed as before.
"""
import pytest
from rdkit import Chem, RDLogger, rdBase

from orthonym.assembly.substituent_enumerator import _substituent_memo_key

pytestmark = pytest.mark.unit


def _key(smiles, frag_atoms, attach_idx):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    return _substituent_memo_key(mol, frag_atoms, attach_idx, False, False)


_MULTI = (("CCO.[Na+]", [0, 1, 2], 0),
          ("CCOC(=O)Cc1ccc(cc1)[CH2-].[Zn+]Br", [0, 1, 2, 3, 4], 2),
          ("[Cl-].[Zn+]c1ccccc1", [1, 2, 3, 4, 5, 6, 7], 1))


def test_a_rooted_fragment_of_a_multi_fragment_molecule_is_keyed_and_nothing_is_logged(capfd):
    status = rdBase.LogStatus()
    RDLogger.EnableLog("rdApp.*")
    try:
        # the control: RDKit's own refusal is logged
        mol = Chem.MolFromSmiles("CCO.[Na+]")
        with pytest.raises(RuntimeError):
            Chem.MolFragmentToSmiles(mol, atomsToUse=[0, 1, 2], rootedAtAtom=0)
        assert "rootedAtAtom" in capfd.readouterr().err
        for smiles, frags, attach in _MULTI:
            key = _key(smiles, frags, attach)
            assert key is not None and key[1] is True          # rooted: attach_in_frag
            assert key[0][0] == "multi-fragment"               # a key no single-fragment molecule shares
        assert capfd.readouterr().err == ""
    finally:
        for line in status.splitlines():
            channel, _, state = line.partition(":")
            (RDLogger.EnableLog if state == "enabled" else RDLogger.DisableLog)(channel)


def test_the_key_of_a_multi_fragment_molecule_tells_two_questions_apart_and_repeats_for_one():
    mol = Chem.MolFromSmiles("CCCCO.[Na+]")
    key = lambda frag, attach: _substituent_memo_key(mol, frag, attach, False, False)   # noqa: E731
    assert key([0, 1, 2, 3], 0) == key([0, 1, 2, 3], 0)
    assert key([0, 1, 2, 3], 0) != key([0, 1, 2, 3], 3)        # the other end: butyl vs butan-?-yl
    assert key([0, 1, 2, 3], 0) != key([0, 1, 2], 0)           # another fragment
    assert key([0, 1, 2, 3], 0) != key([1, 2, 3], 1)


def test_the_key_of_a_multi_fragment_molecule_holds_the_attachment_position():
    """The unrooted fragment SMILES is the same for either end of a chain; only the attachment atom's place
    in the canonical atom order tells the carbon end of the chain (``CCCO``, atom 0) from the oxygen end
    (``OCCC``, atom 0): without it the two keys are equal."""
    carbon_end = _key("CCCO.[Na+]", [0, 1, 2, 3], 0)
    oxygen_end = _key("OCCC.[Na+]", [0, 1, 2, 3], 0)
    assert carbon_end != oxygen_end                                 # the two questions are two keys
    assert carbon_end[0][1] == oxygen_end[0][1]                     # the unrooted fragment SMILES is one
    assert carbon_end[0][2] != oxygen_end[0][2]                     # the attachment position tells them apart


def test_the_multi_fragment_key_is_not_the_key_of_the_same_fragment_alone():
    mixed = _key("CCCCO.[Na+]", [0, 1, 2, 3], 0)
    alone = _key("CCCCO", [0, 1, 2, 3], 0)
    assert mixed is not None and alone is not None and mixed != alone


def test_a_strict_ring_prefix_of_a_multi_fragment_molecule_is_computed_once_per_question(monkeypatch):
    """The strict twin's promotion re-run asks ``_cached_strict_name`` for one (fragment, attachment) again
    and again; on a mixture the memo key was ``None`` and every ask recomputed (99.7% of 26,475 asks)."""
    from orthonym.assembly import memo, strict_ring_prefix as srp
    calls = []
    monkeypatch.setattr(srp, "strict_ring_substituent_name",
                        lambda mol, frag, attach: calls.append((tuple(frag), attach)) or "cyclohexyl")
    mol = Chem.MolFromSmiles("C1CCCCC1CO.[Na+]")
    scope = memo.push_scope()
    try:
        asked = [srp._cached_strict_name(mol, [0, 1, 2, 3, 4, 5], 0) for _ in range(5)]
        srp._cached_strict_name(mol, [0, 1, 2, 3, 4, 5], 1)
    finally:
        memo.pop_scope(scope)
    assert asked == ["cyclohexyl"] * 5
    assert calls == [((0, 1, 2, 3, 4, 5), 0), ((0, 1, 2, 3, 4, 5), 1)]


def test_a_single_fragment_molecule_and_an_unrooted_fragment_are_keyed():
    key = _key("CCO", [0, 1, 2], 0)
    assert key is not None and key[1] is True                          # rooted: attach_in_frag
    assert _key("CCOC(=O)C", [0, 1, 2], 0) is not None
    unrooted = _key("CCO.[Na+]", [1, 2], 0)                            # the attachment atom is outside the fragment
    assert unrooted is not None and unrooted[1] is False
