""" Piece 2 — decorated FUSED ring-substituents get a producer.

`_decorated_heteroaryl_substituent_name` names a decorated AROMATIC MONOCYCLE
ring-substituent (`5-hydroxy-1,3-dimethylpyrazol-4-yl`, `4-methoxyphenyl`) but
returns None the moment the ring system is FUSED (`ring_substituents.py:574`,
"fusion / spiro / not a simple monocycle"). So a decorated fused ring hanging
off a parent — e.g. a methoxynaphthalene on an acetic-acid chain — had no
producer and fell straight through to a silent atom drop /, even though
the BARE fused substituent (`naphthalen-2-yl`) already names.

Piece 2 reuses the fused-ring PARENT numbering (`compute_fused_numbering` +
`_ring_system_automorphisms`) and selects the symmetry-equivalent numbering that
gives the FREE VALENCE the lowest locant free-valence priority),
then reads each decoration's locant off that same one numbering.

Target verified by OPSIN round-trip (`a temp dir/piece2_spy_verify.py`):
`2-(6-methoxynaphthalen-2-yl)acetic acid` canonicalises to the InChIKey of
`COc1ccc2cc(CC(=O)O)ccc2c1` (`PHJFLPMVEFKEPL-UHFFFAOYSA-N`).
"""
import pytest
from rdkit import Chem

from orthonym.rules.ring_substituents import name_ring_system_substituent

pytestmark = pytest.mark.opsin_gate


def _names_at_every_ring_attachment(smiles, allow_mancude=True):
    """Every distinct substituent name this fragment yields with a carbon
    stand-in attached at each RING carbon in turn (fragment = every original
    atom; the attachment carbon carries the free valence).

    ``allow_mancude`` defaults to True: Piece 2 is BEST-EFFORT-scoped (PIN
    byte-identical by construction), so the fused producer is only reachable
    when the best-effort tier flag is threaded down."""
    base = Chem.MolFromSmiles(smiles)
    out = set()
    for at in range(base.GetNumAtoms()):
        a = base.GetAtomWithIdx(at)
        if a.GetSymbol() != 'C' or not a.IsInRing():
            continue
        rw = Chem.RWMol(base)
        marker = rw.AddAtom(Chem.Atom(6))
        rw.AddBond(at, marker, Chem.BondType.SINGLE)
        mol = rw.GetMol()
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            continue
        frag = [i for i in range(mol.GetNumAtoms()) if i != marker]
        try:
            name = name_ring_system_substituent(
                mol, frag, at, allow_mancude=allow_mancude)
        except Exception:
            name = None
        if name:
            out.add(name)
    return out


def test_decorated_fused_naphthalene_substituent_now_names():
    """6-methoxynaphthalene as a ring-substituent -> 6-methoxynaphthalen-2-yl
    (free valence at 2 by; the methoxy then takes 6)."""
    got = _names_at_every_ring_attachment("COc1ccc2ccccc2c1")
    assert "6-methoxynaphthalen-2-yl" in got, got


def test_fused_producer_is_best_effort_scoped():
    """PIN byte-identical by construction: with the best-effort flag OFF the
    fused decorated ring-substituent must still DECLINE (status quo), exactly
    as it did before Piece 2 — the new capability is opt-in via allow_mancude."""
    got_pin = _names_at_every_ring_attachment("COc1ccc2ccccc2c1",
                                              allow_mancude=False)
    assert not any("methoxynaphthalen" in n for n in got_pin), got_pin


def test_generalises_beyond_methoxy_and_to_two_decorations():
    """Not specific to methoxy: a halogen decoration and a two-decoration ring
    both name, the multiplied prefix taking the shared low-locant set."""
    got_cl = _names_at_every_ring_attachment("Clc1ccc2ccccc2c1")
    assert "6-chloronaphthalen-2-yl" in got_cl, got_cl
    got_2cl = _names_at_every_ring_attachment("Clc1ccc2cc(Cl)ccc2c1")
    assert "2,6-dichloronaphthalen-1-yl" in got_2cl, got_2cl


def test_bare_fused_substituent_unchanged():
    """Regression: a BARE fused ring still routes through the pre-existing
    bare-ring branch (`get_ring_substituent_name`), untouched by Piece 2."""
    got = _names_at_every_ring_attachment("c1ccc2ccccc2c1")
    assert got == {"naphthalen-1-yl", "naphthalen-2-yl"}, got


def test_numbering_is_smiles_order_independent():
    """Gate-critical determinism: the SAME molecule written many ways yields the
    SAME substituent name. The free-valence-first selection ties are broken by a
    key that fully fixes every locant->role assignment, so tied numberings
    assemble byte-identically."""
    import random

    def _name_the_naphthalene_substituent(mol):
        ringset = {a.GetIdx() for a in mol.GetAtoms()
                   if a.GetIsAromatic() and a.IsInRing()}
        frag, attach = set(ringset), None
        for ra in list(ringset):
            for nb in mol.GetAtomWithIdx(ra).GetNeighbors():
                ni = nb.GetIdx()
                if ni in ringset:
                    continue
                if nb.GetSymbol() == 'O':          # methoxy O + its methyl
                    frag.add(ni)
                    frag.update(n.GetIdx() for n in nb.GetNeighbors()
                                if n.GetIdx() != ra)
                elif nb.GetSymbol() == 'C':        # the parent (acetic-acid) bond
                    attach = ra
        return name_ring_system_substituent(mol, list(frag), attach,
                                            allow_mancude=True)

    base = Chem.MolFromSmiles("COc1ccc2cc(CC(=O)O)ccc2c1")
    random.seed(0)
    names = {
        _name_the_naphthalene_substituent(
            Chem.MolFromSmiles(Chem.MolToSmiles(base, doRandom=True)))
        for _ in range(8)
    }
    assert names == {"6-methoxynaphthalen-2-yl"}, names
