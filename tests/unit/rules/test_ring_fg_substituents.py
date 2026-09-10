""".1 — FG-aware ring-as-substituent emitter (oxo / cyano).

A ring demoted to a substituent of a chain parent must keep its
characteristic groups as prefixes inside the enclosing marks
 ketone->oxo; nitrile->cyano):

    O=C1CCCCC1CCCC(=O)O -> 4-(2-oxocyclohexyl)butanoic acid
    (HEAD before fix: "4-cyclohexylbutanoic acid" -- a structurally WRONG name)

Design per internal notes.1 (skeptic-
corrected): the chemistry primitive lives in rules.ring_substituents and is
consumed by composer._detect_ring_substituents; the widened (non-6-membered)
ring path fires ONLY when an FG prefix is present (no silent activation of
alkyl/halo emission on small rings -- that change set needs its own
enumerated canary budget); heteroaryl rings are out of scope (their PINs use
fixed heteroatom-lowest numbering,, not attachment-relative).

Every expected name below was OPSIN-verified (name -> structure -> InChI
match against the test SMILES) before being baked in.
"""
import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.rules.ring_substituents import ring_atom_fg_prefixes


def _ring_atom_with_exocyclic(mol, neighbor_pred):
    """Return (ring_atom_idx, ring_atom_set) for the first ring atom having
    an exocyclic neighbor satisfying neighbor_pred."""
    ri = mol.GetRingInfo()
    ring_atoms = set()
    for ring in ri.AtomRings():
        ring_atoms.update(ring)
    for idx in sorted(ring_atoms):
        atom = mol.GetAtomWithIdx(idx)
        for nbr in atom.GetNeighbors():
            if nbr.GetIdx() not in ring_atoms and neighbor_pred(mol, atom, nbr):
                return idx, ring_atoms
    raise AssertionError("test molecule lacks the expected exocyclic group")


class TestRingAtomFgPrefixes:
    """Chemistry primitive: per-ring-atom FG prefix detection."""

    def test_oxo_detected_on_ketone_ring_atom(self):
        mol = Chem.MolFromSmiles("O=C1CCCCC1C")
        idx, ring = _ring_atom_with_exocyclic(
            mol, lambda m, a, n: n.GetSymbol() == "O"
            and m.GetBondBetweenAtoms(a.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 2.0
        )
        assert ring_atom_fg_prefixes(mol, idx, ring) == ["oxo"]

    def test_cyano_detected(self):
        mol = Chem.MolFromSmiles("N#CC1CCCCC1C")
        idx, ring = _ring_atom_with_exocyclic(
            mol, lambda m, a, n: n.GetSymbol() == "C"
            and any(x.GetSymbol() == "N" for x in n.GetNeighbors())
        )
        assert ring_atom_fg_prefixes(mol, idx, ring) == ["cyano"]

    def test_plain_ring_atom_yields_nothing(self):
        mol = Chem.MolFromSmiles("C1CCCCC1C")
        ring = set(mol.GetRingInfo().AtomRings()[0])
        plain = next(i for i in ring
                     if all(n.GetIdx() in ring for n in
                            mol.GetAtomWithIdx(i).GetNeighbors()
                            if n.GetSymbol() != "H"))
        assert ring_atom_fg_prefixes(mol, plain, ring) == []

    def test_hydroxy_not_claimed_by_this_primitive(self):
        # hydroxy stays with the existing composer branch; the primitive
        # owns only the previously-dropped FG classes (oxo, cyano).
        mol = Chem.MolFromSmiles("OC1CCCCC1C")
        idx, ring = _ring_atom_with_exocyclic(
            mol, lambda m, a, n: n.GetSymbol() == "O")
        assert ring_atom_fg_prefixes(mol, idx, ring) == []

    def test_carboxy_emitted(self):
        # a phase (,: ring-COOH demotion is now reachable
        # (the S2 parent chokepoint landed), so the primitive emits 'carboxy'
        # for a ring atom bearing an exocyclic free carboxylic-acid carbon.
        mol = Chem.MolFromSmiles("OC(=O)C1CCCCC1C")
        idx, ring = _ring_atom_with_exocyclic(
            mol, lambda m, a, n: n.GetSymbol() == "C"
            and any(x.GetSymbol() == "O" for x in n.GetNeighbors()))
        assert ring_atom_fg_prefixes(mol, idx, ring) == ["carboxy"]

    def test_ester_carbonyl_not_cyano_not_oxo(self):
        mol = Chem.MolFromSmiles("COC(=O)C1CCCCC1C")
        idx, ring = _ring_atom_with_exocyclic(
            mol, lambda m, a, n: n.GetSymbol() == "C"
            and any(x.GetSymbol() == "O" for x in n.GetNeighbors()))
        assert ring_atom_fg_prefixes(mol, idx, ring) == []


class TestEmitterEndToEnd:
    """Full names. Targets were wrong on HEAD pre-fix (FG silently dropped)."""

    @pytest.mark.parametrize("smiles,expected", [
        # 6-ring oxo (the verified live defect, fix-plan section 1.1)
        ("O=C1CCCCC1CCCC(=O)O", "4-(2-oxocyclohexyl)butanoic acid"),
        # 6-ring cyano
        ("N#CC1CCCCC1CCCCC(=O)O", "5-(2-cyanocyclohexyl)pentanoic acid"),
        # 5-ring oxo: needs the len(ring_order)!=6 numbering fix + the
        # FG-gated widening of the 6-only ring-size gate
        ("O=C1CCCC1CCCC(=O)O", "4-(2-oxocyclopentyl)butanoic acid"),
        # 7-ring oxo (size 3-8 widening)
        ("O=C1CCCCCC1CCCC(=O)O", "4-(2-oxocycloheptyl)butanoic acid"),
        # aromatic all-carbon ring + cyano
        ("N#Cc1cccc(CCCC(=O)O)c1", "4-(3-cyanophenyl)butanoic acid"),
        # multiplier path: two oxo -> dioxo
        ("O=C1CCCC(=O)C1CCCC(=O)O", "4-(2,6-dioxocyclohexyl)butanoic acid"),
        # FG on the attachment atom itself (quaternary; the ring_pos==1
        # skip must not drop it)
        ("N#CC1(CCCC(=O)O)CCCCC1", "4-(1-cyanocyclohexyl)butanoic acid"),
    ])
    def test_target_names(self, smiles, expected):
        assert name_compound(smiles) == expected

    def test_protect_hydroxy_control(self):
        # Already correct on HEAD pre-fix; must stay byte-identical.
        assert name_compound("OC1CCCCC1CCCC(=O)O") == \
            "4-(2-hydroxycyclohexyl)butanoic acid"


class TestWideningGate:
    """Wave2 T3a: the old tripwire demanded that alkyl/halo emission on
    non-6 rings only land as a conscious, separately-enumerated change —
     is that change. The former gate returned the bare base_name,
    silently DROPPING the ring's substituents ('4-cyclopentylbutanoic acid'
    for the 2-methylcyclopentyl input: a different molecule, caught only by
    the OPSIN-armed). The widened path names every branch exactly
    via the recursive fragment namer or FAILS CLOSED (constitution-
    conservation guard in _detect_ring_substituents /
    _build_substituted_ring_name), so the skeptic's silent-activation
    hazard no longer exists."""

    def test_alkyl_on_5_ring_correct_name(self):
        # The former strict-xfail, flipped per the tripwire's instruction.
        assert name_compound("CC1CCCC1CCCC(=O)O") == \
            "4-(2-methylcyclopentyl)butanoic acid"
