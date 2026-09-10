""" a phase (P-67.1.4.4.2): sulfur-oxoacid acyl-oxy / -amino substituent prefixes.

When a sulfur oxoacid group is attached by oxygen or nitrogen to a compound that
also carries a group senior to the sulfur acid (here a carboxylic acid), the
sulfur group is cited as a substituent PREFIX, not the parent
(P-67.1.4.4.2, the Blue Book Blue Book). The Blue Book's verbatim (PIN)
examples are pinned below with their line numbers. The generic path names these
by skeletal ('a') replacement ('…-1,3-dioxa-2λ6-thiapropyl') — a valid,
round-tripping, but NON-PIN form (RIGHT_MOL_NONPIN). See
.superpowers/sdd/IMPLEMENTATION-PLAN-/task-11B1-report.md.
"""
import pytest
from rdkit import Chem

from orthonym.rules.sulfur_oxoacid import (
    _collect_subtree,
    _sulfur_oxoacid_acyl,
    name_sulfur_oxoacid_acyl_for_amino,
    name_sulfur_oxoacid_oxy_substituent,
)

pytestmark = pytest.mark.unit


def _c_o_s(smi):
    """Return (mol, ester_O_idx, parent_C_idx) for the ACID-side C-O-S linkage.

    In a molecule with two C-O-S oxygens (e.g. COS(=O)OCCC(=O)O has one on the
    methoxy side and one on the propanoate side), pick the ester O whose carbon
    reaches the carboxylic acid (the senior parent) WITHOUT crossing the S --
    that is the substituent attach point the engine uses."""
    m = Chem.MolFromSmiles(smi)
    acid_cs = {mt[0] for mt in
               m.GetSubstructMatches(Chem.MolFromSmarts('[CX3](=O)[OX2H1]'))}
    cands = []
    for a in m.GetAtoms():
        if a.GetSymbol() != 'O':
            continue
        nbrs = [n for n in a.GetNeighbors() if n.GetAtomicNum() > 1]
        if sorted(n.GetSymbol() for n in nbrs) == ['C', 'S']:
            c = [n for n in nbrs if n.GetSymbol() == 'C'][0]
            s = [n for n in nbrs if n.GetSymbol() == 'S'][0]
            cands.append((a.GetIdx(), c.GetIdx(), s.GetIdx()))

    def _reaches_acid(o_idx, c_idx, s_idx):
        seen = {o_idx, s_idx}
        stack = [c_idx]
        while stack:
            i = stack.pop()
            if i in acid_cs:
                return True
            if i in seen:
                continue
            seen.add(i)
            for n in m.GetAtomWithIdx(i).GetNeighbors():
                if n.GetIdx() not in seen:
                    stack.append(n.GetIdx())
        return False

    for o, c, s in cands:
        if _reaches_acid(o, c, s):
            return m, o, c
    if cands:
        return m, cands[0][0], cands[0][1]
    raise AssertionError("no C-O-S linkage found in %r" % smi)


def _s_frag_and_n(smi):
    """Return (mol, s_idx, frag_atoms) for a -NH-S(oxoacid) linkage.

    frag_atoms is the S subtree NOT crossing the linking N (the shape the
    amino-branch caller passes when it recurses into the S fragment)."""
    m = Chem.MolFromSmiles(smi)
    for a in m.GetAtoms():
        if a.GetSymbol() != 'S':
            continue
        n_nbrs = [n for n in a.GetNeighbors() if n.GetSymbol() == 'N']
        if len(n_nbrs) == 1:
            n_idx = n_nbrs[0].GetIdx()
            frag = set(_collect_subtree(m, a.GetIdx(), n_idx))
            return m, a.GetIdx(), frag
    raise AssertionError("no N-S linkage found in %r" % smi)


# ---- O-linked oxy substituent (5 of the 6 (PIN) rows) --------------------

def test_sulfooxy():
    # 3-(sulfooxy)propanoic acid (PIN):36488 (HO-SO2-O- contracts to sulfooxy)
    m, o, c = _c_o_s('O=C(O)CCOS(=O)(=O)O')
    assert name_sulfur_oxoacid_oxy_substituent(m, o, c) == 'sulfooxy'


def test_chlorosulfonyloxy():
    # 3-[(chlorosulfonyl)oxy]propanoic acid (PIN):36492
    m, o, c = _c_o_s('O=C(O)CCOS(=O)(=O)Cl')
    assert name_sulfur_oxoacid_oxy_substituent(m, o, c) == '(chlorosulfonyl)oxy'


def test_sulfamoyloxy():
    # 3-(sulfamoyloxy)propanoic acid (PIN):36494 (H2N-SO2-O- contracts to sulfamoyloxy)
    m, o, c = _c_o_s('NS(=O)(=O)OCCC(=O)O')
    assert name_sulfur_oxoacid_oxy_substituent(m, o, c) == 'sulfamoyloxy'


def test_aminosulfinyloxy_not_sulfinamoyloxy():
    # 3-[(aminosulfinyl)oxy]propanoic acid (PIN):36500 [not …sulfinamoyloxy…]
    m, o, c = _c_o_s('NS(=O)OCCC(=O)O')
    assert name_sulfur_oxoacid_oxy_substituent(m, o, c) == '(aminosulfinyl)oxy'


def test_methoxysulfinyloxy():
    # 3-[(methoxysulfinyl)oxy]propanoic acid (PIN):36490
    m, o, c = _c_o_s('COS(=O)OCCC(=O)O')
    assert name_sulfur_oxoacid_oxy_substituent(m, o, c) == '(methoxysulfinyl)oxy'


# ---- N-linked acyl (the 6th (PIN) row) -----------------------------------

def test_methoxysulfonyl_for_amino():
    # 3-[(methoxysulfonyl)amino]propanoic acid (PIN):36502
    # The acyl namer returns the bare acyl; the amino-branch caller wraps it.
    m, s, frag = _s_frag_and_n('COS(=O)(=O)NCCC(=O)O')
    assert name_sulfur_oxoacid_acyl_for_amino(m, s, frag) == 'methoxysulfonyl'


# ---- fail-closed / negatives (perception must not misfire) ---------------

def test_thioether_no_oxo_fails_closed():
    # -O-S- with 0 S=O is a plain sulfenate/thioether shape, not an oxoacid ->
    # None (COSC1CCCCC1 must stay '(methoxysulfanyl)cyclohexane').
    m, o, c = _c_o_s('COSC1CCCCC1')
    assert name_sulfur_oxoacid_oxy_substituent(m, o, c) is None


def test_carbon_R_sulfonyl_fails_closed():
    # -O-S(=O)(=O)-CH3: X is carbon -> a different nomenclature (methanesulfonyl)
    # -> fail closed here, never a garbage 'methylsulfonyloxy'.
    m, o, c = _c_o_s('O=C(O)CCOS(=O)(=O)C')
    assert name_sulfur_oxoacid_oxy_substituent(m, o, c) is None


def test_ring_sulfur_fails_closed():
    # An S inside a ring (here sulfolane's ring sulfone) is not an acyclic
    # S-oxoacid acyl -> the acyl builder fails closed on the IsInRing guard.
    m = Chem.MolFromSmiles('C1CCS(=O)(=O)C1')  # sulfolane: ring S(=O)(=O)
    s = [a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == 'S'][0]
    ring_c = [n.GetIdx() for n in m.GetAtomWithIdx(s).GetNeighbors()
              if n.GetSymbol() == 'C'][0]
    assert _sulfur_oxoacid_acyl(m, s, ring_c) is None


def test_carbon_R_amino_linker_fails_closed():
    # -NH-S(=O)(=O)-CH3 (methanesulfonamido): carbon-R -> None, so the row stays
    # '3-(methanesulfonamido)propanoic acid', NOT '(methoxysulfonyl)amino'.
    m, s, frag = _s_frag_and_n('CS(=O)(=O)NCCC(=O)O')
    assert name_sulfur_oxoacid_acyl_for_amino(m, s, frag) is None


def test_acyl_contractions_and_bases():
    # Direct acyl-builder table check (structure -> acyl), one per class.
    for smi, acyl in [
        ('O=C(O)CCOS(=O)(=O)O', 'sulfo'),          # n=2, X=OH (contract)
        ('O=C(O)CCOS(=O)(=O)Cl', 'chlorosulfonyl'),  # n=2, X=Cl
        ('NS(=O)(=O)OCCC(=O)O', 'sulfamoyl'),        # n=2, X=NH2 (contract)
        ('NS(=O)OCCC(=O)O', 'aminosulfinyl'),        # n=1, X=NH2 (NOT sulfinamoyl)
        ('COS(=O)OCCC(=O)O', 'methoxysulfinyl'),     # n=1, X=OMe
    ]:
        m, o, _c = _c_o_s(smi)
        s = [n.GetIdx() for n in m.GetAtomWithIdx(o).GetNeighbors()
             if n.GetSymbol() == 'S'][0]
        assert _sulfur_oxoacid_acyl(m, s, o) == acyl, smi


# ---- cascade integration: Tier 0.55 fires from name_substituent -----------
# Deterministic (no OPSIN): proves the enumerator cascade routes the S-oxoacid
# fragment to the new tier, returning the same token the unit tests assert.
# The authoritative whole-molecule byte-identity + round-trip proof is the
# bb_conformance measure (all 6 def_id 67.1.4.4.2 rows flip to MATCH,
# 0 losses; see task-11B1-report.md), not repeated here because the engine's
# internal OPSIN grammar gate is flaky under pytest (the documented OPSIN-pipe
# hazard in CLAUDE.md).

def test_cascade_routes_o_linked_fragment():
    from orthonym.assembly.substituent_enumerator import name_substituent
    from orthonym.rules.sulfur_oxoacid import _collect_subtree
    for smi, token in [
        ('O=C(O)CCOS(=O)(=O)O', 'sulfooxy'),
        ('O=C(O)CCOS(=O)(=O)Cl', '(chlorosulfonyl)oxy'),
        ('NS(=O)(=O)OCCC(=O)O', 'sulfamoyloxy'),
        ('NS(=O)OCCC(=O)O', '(aminosulfinyl)oxy'),
        ('COS(=O)OCCC(=O)O', '(methoxysulfinyl)oxy'),
    ]:
        m, o, c = _c_o_s(smi)
        frag = _collect_subtree(m, o, c)  # the -O-S(oxoacid) substituent
        assert name_substituent(m, frag, o, True) == token, smi


def test_cascade_routes_n_linked_fragment():
    from orthonym.assembly.substituent_enumerator import name_substituent
    m, s, frag = _s_frag_and_n('COS(=O)(=O)NCCC(=O)O')
    # The amino-branch caller recurses into the S fragment (attach=S) and wraps
    # the returned acyl as '(methoxysulfonyl)amino'.
    assert name_substituent(m, list(frag), s, True) == 'methoxysulfonyl'
