"""The free valence of an unsaturated von Baeyer substituent takes its low locant before the 'ene' and 'yne' endings.

``### ****...`` criteria of numbering: (indicated hydrogen, then principal characteristic groups
and free valences) come before (hydro/dehydro prefixes and 'ene' and 'yne' endings).
(the Blue Book): '2-[bicyclo[6.6.1]pentadeca-8(15)-en-1-yl]ethan-1-ol (PIN)', not '...pentadec-1(15)-en-8-yl...'.
The cage analyzer numbers the ring system for the parent (the enes low); ``ring_substituents._lowest_locant_cage_numbering``
then moved the free valence only through the automorphisms that keep every double bond on a double bond, so the enes kept
their locants and the free valence took the higher bridgehead. Here every automorphism of the ring SKELETON (elements
kept, bond orders left out) is a candidate numbering: it maps the skeleton onto itself, so the von Baeyer descriptor and
the heteroatom locants stay the same (each candidate is audited again), and the multiple bonds are cited on it anew.
"""
from __future__ import annotations

from typing import Optional, Tuple

from rdkit import Chem


def free_valence_first_enes(sub, cage, cage_atoms, attach_sub: int, stem: str,
                            allow_mancude: bool = False) -> Optional[Tuple[int, str]]:
    """``(free-valence locant, elided parent stem)`` for an unsaturated cage with no aromatic atom, or None (a saturated
    or aromatic cage, or no candidate that the audit and the unsaturation renderer accept: the caller keeps its numbering).
    The candidate with the lowest free-valence locant wins, then the lowest locants of the multiple bonds together,
    then of the double bonds."""
    from .polycyclic import _build_parent_with_unsaturation
    from .ring_unsaturation import render_ring_unsaturation
    from .vonbaeyer_universal import audit_von_baeyer_descriptor
    unsat = cage.unsaturation or {}
    if not (unsat.get('double_bonds') or unsat.get('triple_bonds')) or any(
            sub.GetAtomWithIdx(i).GetIsAromatic() for i in cage_atoms):
        return None     # saturated, or a mancude system named as a von Baeyer polyene (``allow_mancude``): left alone
    try:
        kek = Chem.RWMol(sub)
        Chem.Kekulize(kek, clearAromaticFlags=True)
        kek = kek.GetMol()
        skel = Chem.RWMol(sub)
        for bond in skel.GetBonds():
            bond.SetBondType(Chem.BondType.SINGLE)
            bond.SetIsAromatic(False)
            bond.SetStereo(Chem.BondStereo.STEREONONE)
        for atom in skel.GetAtoms():
            atom.SetIsAromatic(False)
            atom.SetNumRadicalElectrons(0)
            atom.SetNumExplicitHs(0)
            atom.SetNoImplicit(False)
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        skel.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(skel)
        autos = skel.GetSubstructMatches(skel, uniquify=False, maxMatches=512, useChirality=False)
    except Exception:  # noqa: BLE001 - a symmetry search must never break naming
        return None
    base = {i: cage.atom_to_locant[i] for i in cage_atoms}
    best = None
    for sigma in autos:
        try:
            cand = {i: base[sigma[i]] for i in cage_atoms}
        except (KeyError, IndexError):
            continue
        if sorted(cand.values()) != sorted(base.values()):
            continue
        ring_unsat = render_ring_unsaturation(kek, cand)
        if ring_unsat is None:
            continue
        lows = sorted(lo for lo, _ in ring_unsat.double_pairs + ring_unsat.triple_pairs)
        key = (cand[attach_sub], tuple(lows), tuple(sorted(lo for lo, _ in ring_unsat.double_pairs)))
        if best is not None and key >= best[0]:
            continue
        if not audit_von_baeyer_descriptor(kek, set(cage_atoms), cand, cage.descriptor):
            continue
        best = (key, ring_unsat)
    if best is None:
        return None
    (loc, _, _), ring_unsat = best
    block = _build_parent_with_unsaturation(cage.total_atoms, {
        'double_bonds': list(ring_unsat.double_locants), 'triple_bonds': list(ring_unsat.triple_locants)})
    if not block:
        return None
    return loc, (block[:-1] if block.endswith('e') else block)
