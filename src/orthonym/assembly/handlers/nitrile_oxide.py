"""P-66.5.4.1/2 neutral nitrile-oxide functional-class suffix handler (W2F p4).

A neutral nitrile oxide ``R-C#[N+]-[O-]`` is named by functional-class method (1)
of P-66.5.4.1 — the separate word ``oxide`` appended to the nitrile name
(``benzonitrile oxide`` BB:34876, ``acetonitrile oxide`` BB:43285, ``formonitrile
oxide`` BB:34883), which IS the preferred IUPAC name. Nitrile oxides are "classed
with zwitterions in the order of compound classes" (P-66.5.4.1), so they are senior
to acids/esters — a co-present ester/acid demotes to a prefix.

Fires ONLY for a NEUTRAL molecule (net formal charge 0). The deprotonated
salt/anion uses the ``(oxo-λ5-azanylidyne)methyl`` PREFIX path
(``benzene.py::_nitrile_oxide_prefix``, gated on ``GetFormalCharge < 0``) — the two
never double-fire.

Mechanism (research ``ring-suffix-trio`` §P-66.5.4.2): strip the ``[O-]`` +
neutralise the N -> plain nitrile SMILES; re-name it with the nitrile forced as the
principal group (the ``charged_router._reenter_forced`` pattern); append
``" oxide"``. Fail closed (return None) when the nitrile parent cannot be named as a
``...nitrile`` (a senior-group leak, e.g. the acid -> ``4-cyanobenzoic acid``) or
when the nitrile-bearing ring is AROMATIC but got mis-named as a saturated
carbocycle. The BB ester PIN ``4-(methoxycarbonyl)benzonitrile oxide`` stays a
buildable follow-on: it needs the aromatic-benzene forced-nitrile fix in
``_assemble_ring_nitrile_name`` AND enclosing marks for the ``(methoxycarbonyl)``
compound prefix (P-16.3.3, a general substituent-rendering feature).

Registered in inner_dispatch at priority 975 (specialty-intercept tier, before the
acid/ester handlers so the senior nitrile oxide wins) — predicate-pure +
direct-return + ``pool.add`` + ``_inject_stereo_if_missing`` (D-25 side_effect
inventory ()).
"""
from __future__ import annotations

from typing import Any, Optional

from rdkit import Chem

from ..name_tree import NameTreeNode, NamingResult

# R-C#[N+]-[O-] (the N-O bond is dative/single in the RDKit input); [C;+0] excludes
# the deprotonated carbanion form so only the NEUTRAL zwitterion matches.
_NITRILE_OXIDE_SMARTS = Chem.MolFromSmarts('[C;+0]#[N+]-[O-]')


def _nitrile_oxide_matches(mol: Any):
    if mol is None or _NITRILE_OXIDE_SMARTS is None:
        return ()
    return mol.GetSubstructMatches(_NITRILE_OXIDE_SMARTS)


def _nitrile_ring_is_aromatic(mol: Any, nitrile_c_idx: int) -> bool:
    """True if the ring carbon bearing the nitrile is aromatic (a benzene/arene
    ring). Used to decline the aromatic-ring-mis-named-as-cyclohexane case."""
    for nb in mol.GetAtomWithIdx(nitrile_c_idx).GetNeighbors():
        if nb.GetIsAromatic() and nb.IsInRing():
            return True
    return False


def _is_nitrile_oxide(features: Any) -> bool:
    """Predicate (D-07 pure, read-only): a NEUTRAL single-fragment molecule bearing
    EXACTLY ONE ``R-C#[N+]-[O-]`` nitrile-oxide group.

    Neutral-only (net charge 0) excludes the anion/salt (prefix path owns it).
    >=2 nitrile oxides -> False: the di-oxide has no reliable OPSIN oracle
    (D-probe) -> fail closed. NO mol/features mutation; NO module state."""
    mol = getattr(features, 'mol', None)
    if mol is None:
        return False
    if Chem.GetFormalCharge(mol) != 0:
        return False
    if len(Chem.GetMolFrags(mol)) != 1:
        return False
    return len(_nitrile_oxide_matches(mol)) == 1


def name_nitrile_oxide(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Emit ``<nitrile-name> oxide`` (P-66.5.4.1 method (1)) for a neutral nitrile
    oxide. Returns None (defer/fail closed) when the nitrile parent cannot be
    faithfully named."""
    from ..candidate_pool import get_current_pool
    from ..composer import _inject_stereo_if_missing

    if mol is None:
        mol = getattr(features, 'mol', None)
    if mol is None:
        return None
    matches = _nitrile_oxide_matches(mol)
    if len(matches) != 1:
        return None
    c_idx, n_idx, o_idx = matches[0][0], matches[0][1], matches[0][2]

    c_atom = mol.GetAtomWithIdx(c_idx)
    heavy_nbrs = [nb.GetIdx() for nb in c_atom.GetNeighbors() if nb.GetIdx() != n_idx]

    if not heavy_nbrs:
        # Bare H-C#[N+][O-]: the strip path would mis-name C#N as the retained
        # 'hydrogen cyanide'; use the retained PIN directly (BB:34883).
        nitrile_name: Optional[str] = 'formonitrile'
    else:
        # Strip the [O-]; neutralise the triple-N -> plain nitrile skeleton.
        rw = Chem.RWMol(mol)
        rw.GetAtomWithIdx(n_idx).SetFormalCharge(0)
        rw.RemoveAtom(o_idx)
        try:
            stripped = rw.GetMol()
            Chem.SanitizeMol(stripped)
            stripped_smi = Chem.MolToSmiles(stripped)
        except Exception:
            return None
        # Force the nitrile as the principal group; the [O-] is gone so the
        # nitrile-oxide predicate cannot re-fire (no recursion). The inner call's
        # validity gate is disabled (the outer top-level SELF-01 gate is the
        # accuracy backstop for the full '... oxide' name).
        from ...namer import Orthonym
        try:
            nitrile_name = Orthonym(
                style=style, _disable_opsin_validity_gate=True,
                _principal_group_override='nitrile').name(stripped_smi)
        except Exception:
            return None
        if not nitrile_name or 'unknown' in nitrile_name.lower():
            return None
        if not nitrile_name.rstrip().endswith('nitrile'):
            # A senior group leaked (e.g. the acid -> '4-cyanobenzoic acid') ->
            # the nitrile is not the parent here -> decline (fail closed).
            return None
        if 'cyclohex' in nitrile_name and _nitrile_ring_is_aromatic(mol, c_idx):
            # Aromatic benzene ring mis-named as cyclohexane by the forced-nitrile
            # ring path (the '4-(methoxycarbonyl)benzonitrile oxide' follow-on).
            return None

    name = f"{nitrile_name} oxide"
    pool = get_current_pool()
    cand = pool.add(name, "nitrile_oxide", features)
    if cand is None:
        return None

    final_name = _inject_stereo_if_missing(features, cand.name, atom_to_locant=None)
    return NamingResult(
        name=final_name,
        tree=NameTreeNode(
            parent_stem=final_name, class_id="nitrile_oxide",
            iupac_section_cite="P-66.5.4.1", fragment_legacy=final_name,
        ),
        atom_to_locant_hint=None,
    )


__all__ = ["name_nitrile_oxide", "_is_nitrile_oxide"]
