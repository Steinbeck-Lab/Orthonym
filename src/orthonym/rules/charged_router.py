"""The single mandatory parent-selection chokepoint for charged species.

Phase 169.6 Plan 03 (CHOKE-01, CHOKE-02). ROOT CAUSE of the flat round-trip
wall (V20 audit RC-2): every charged class (alkoxide / phenolate / carbanion /
thiolate / carbenium / onium / diazonium / aminium-fallback / radical) decided
its own parent through a *parallel carbon-counting stub* that ignores
connectivity, substituents and unsaturation — the literal ``heptanolate`` bug
(``CCCCCCC[O-]`` -> ``heptanolate``, dropping the locant and every substituent).
Those stubs BYPASS the sound ``select_parent`` cascade for 91% of wrong-parent
failures.

This module makes the chokepoint STRUCTURAL: ``route_charged(mol, style)`` is the
ONE funnel every charged dispatch handler delegates to. It GENERALIZES the proven
``_name_oxoacid_anion`` template (``ions.py:911``, the 169.5 SUB-01 fix that
round-trips):

    neutralize the chosen fragment -> re-enter the FULL pipeline
    ``Orthonym(style, _disable_opsin_validity_gate=True).name(neutral_smi)``
    -> re-apply the class-correct ionic suffix via ``apply_ion_suffix_to_name``.

A correct parent auto-corrects the locants (95% co-occurrence) on the same
molecule — this is the multi-defect-collapsing fix, not a per-class patch.

The four IUPAC-2013 guards (SYNTHESIS-authoritative-cascade.md §2; P-72.7 /
P-73.7 / P-74), applied IN ORDER:

  GUARD 1  FG-class-before-suffix (the ``heptanolate`` fix). A ``-S(=O)2-O-`` is
           an acid anion -> ``-sulfonate`` (P-72.2.2.2.1), NOT a hydroxy anion
           -> ``-olate`` (P-72.2.2.2.2). ``classify_anion`` picks the FG class;
           the per-class ``allowed_suffixes`` subset gates the textual seam so a
           sulfonate stem can NEVER mis-fire to ``-olate``.
  GUARD 2  ionic-center-count-first (P-72.7 a-c / P-73.7 a-b). On a multi-center
           ion the parent maximizes anionic / ``ide`` / ``uide`` center count
           BEFORE P-44 length is considered (dicarboxylate dianions). Realized
           by neutralizing ALL same-sign centers and letting the re-entered
           pipeline name the multi-suffix parent (``butanedioate``), exactly as
           the proven anion seam already does.
  GUARD 3  skeletal-charge element seniority (P-72.7 d / P-73.7 c):
           N > P > As > Sb > Bi > Si > Ge > Sn > Pb > B > Al > Ga > In > Tl > O
           > S > Se > Te > C. When the charge sits on a skeletal heteroatom the
           senior element bearing it is the parent, not the longest carbon chain.
  GUARD 4  zwitterion anion-is-parent override (P-74.0). The anion is FORCED as
           the parent; a separable cation (P-74.1.3, the betaine quaternary
           ammonium) is demoted to a structured ``(…azaniumyl)`` substituent
           prefix (``substituent_naming.cation_to_prefix``); a skeletal cation
           (P-74.1.2, a ring N+ of pyridinium-2-carboxylate) is deferred to the
           legacy path (the cumulative ium+ate suffix is out of scope this
           plan). 169.6-04 (was a Plan-03 detect-and-defer seam).

Returns '' on any failure / out-of-scope shape (metal complex, multi-fragment
salt, zwitterion, malformed re-entered parent) so the caller falls through to the
existing retained/legacy cascade — preserving the v18 byte-identical no-crash
contract.

NO carbon-counting. NO ``.replace``. NO molecule-specific branch. The router is
GENERAL (fix-methodology.md).
"""

from typing import Optional

from rdkit import Chem

from .ions import (
    _has_metal,
    classify_anion,
    classify_cation,
    apply_ion_suffix_to_name,
)
from ..perception.ions import get_ion_sites, _get_internal_charge_atoms


# =============================================================================
# GUARD 3 — element seniority for a skeletal (non-O, on-the-atom) charge.
# P-72.7(d) / P-73.7(c) (SYNTHESIS §"TIER -1", verbatim order). Lower index =
# more senior. Carbon is LAST: a charge on any heteroatom outranks a carbanion /
# carbenium for the parent-bearing-atom choice.
# =============================================================================
_ELEMENT_SENIORITY = {
    el: i for i, el in enumerate(
        ['N', 'P', 'As', 'Sb', 'Bi', 'Si', 'Ge', 'Sn', 'Pb', 'B',
         'Al', 'Ga', 'In', 'Tl', 'O', 'S', 'Se', 'Te', 'C']
    )
}


# =============================================================================
# GUARD 1 — per-class allowed_suffixes (the heptanolate fix).
# classify_anion / classify_cation -> the FG class -> the subset of neutral
# suffixes in the Plan-02 _ANION_SUFFIX_MAP / _CATION_SUFFIX_MAP that this class
# may take. The seam (apply_ion_suffix_to_name) is restricted to this subset so a
# sulfonate stem (allowed={'sulfonic acid'}) can never match the bare 'ol' key.
#
#   anion 'sulfonate'/'sulfinate'/'phosphonate' -> the matching oxoacid suffix
#       (P-72.2.2.2.1.1: acid anions -> -ate/-ite).
#   anion 'carboxylate'  -> {'oic acid','carboxylic acid'}  (P-72.2.2.2.1.1)
#   anion 'alkoxide'/'phenolate' -> {'ol'}  (P-72.2.2.2.2: hydroxy anions -> -olate)
#   anion 'thiolate'     -> {'thiol'}       (P-72.2.2.2.2)
#   anion 'carbanion'    -> None -> bare -> -ide  (P-72.1)
#   anion 'aminide'      -> {'amine'}       (P-72.2.2.2.3)
# =============================================================================
_ANION_ALLOWED_SUFFIXES = {
    'sulfonate': frozenset({'sulfonic acid'}),
    'sulfinate': frozenset({'sulfinic acid'}),
    'phosphonate': frozenset({'phosphonic acid', 'phosphinic acid',
                              'phosphoric acid'}),
    'carboxylate': frozenset({'oic acid', 'carboxylic acid'}),
    'alkoxide': frozenset({'ol'}),
    'phenolate': frozenset({'ol'}),
    'thiolate': frozenset({'thiol'}),
    'aminide': frozenset({'amine'}),
    # 'carbanion' -> None (bare -> -ide via the resolvers empty-suffix path).
}

# CATION class -> (cation_class arg for apply_ion_suffix_to_name, allowed_suffixes).
# The class-keyed cation transforms (ylium/acylium/diazonium) are NOT plain suffix
# swaps (Plan-02 _CLASS_KEYED_CATION_TRANSFORMS); they are passed via cation_class.
# 'aminium' is a plain map swap (amine->aminium, P-73.1.2.1) gated to {'amine'}.
_CATION_SPEC = {
    'ylium': ('ylium', None),       # P-73.2.2.1.1: ane->ylium (methane->methylium)
    'diazonium': ('diazonium', None),  # P-73.2.2.3: append diazonium to the hydride
    'aminium': (None, frozenset({'amine'})),  # P-73.1.2.1: amine->aminium (map swap)
    # 'onium' (oxonium/sulfonium/phosphonium on a heteroatom hydride) -> handled
    # via the generic map empty->'ium' default (no class key, no allowed subset).
}


# WR-01 / 169.5 SUB-01 (reused verbatim): a re-entered parent that lost its
# chain/ring stem (the deferred P-25 fused-ring limitation) leaves a bare
# unsaturation marker glued onto a suffix stem or a locant ('anesulfonic acid',
# 'ene-1-...'). A descriptive fallback ('unknown ...', 'not supported') is
# likewise not a valid parent. Refuse to propagate either; fall through instead.
import re as _re
_DEGENERATE_PARENT_RE = _re.compile(r'^(?:ane|ene|yne)(?:sulf|phosph|arso|boro|[0-9(-])')


def _is_malformed_parent(neutral_name: str) -> bool:
    """True if a re-entered neutral parent name must NOT be ionized (169.5 SUB-01)."""
    low = neutral_name.lstrip().lower()
    return bool(
        _DEGENERATE_PARENT_RE.match(low)
        or 'unknown' in low
        or 'not supported' in low
        or 'wildcard' in low
    )


def _is_zwitterion(sites) -> bool:
    """GUARD 4 (P-74.0): a single fragment carrying BOTH a (non-internal)
    cationic and a (non-internal) anionic center is a zwitterion. Routed by
    ``_route_zwitterion`` (anion-is-parent override + the structured cation
    prefix producer); 169.6-04 (was a Plan-03 detect-and-defer seam).
    """
    return bool(sites.get('cations')) and bool(sites.get('anions'))


def _cation_is_skeletal_to_anion_parent(mol, cation_idx: int, anion_idx: int) -> bool:
    """P-74.1.2 vs P-74.1.3 discriminator.

    P-74.1.2: the cationic atom is INSIDE the parent hydride that bears the
    anionic characteristic group — i.e. it is a ring atom of the SAME ring
    system the anion is attached to (the ring N+ of a pyridinium-2-carboxylate).
    Such a cation is kept on the parent as an ``-ium`` suffix, NOT demoted to a
    prefix.

    P-74.1.3: the cationic atom sits on a DIFFERENT parent (a quaternary
    ammonium hanging off the anion chain) -> the cation becomes a substituent
    prefix.

    Heuristic (structural, not per-molecule): the cation is skeletal iff it is a
    ring atom AND the anion's attachment carbon is in the SAME ring system
    (shares a ring with the cation). A non-ring (acyclic) quaternary ammonium is
    always P-74.1.3 (the betaine case).
    """
    cat = mol.GetAtomWithIdx(cation_idx)
    if not cat.IsInRing():
        return False  # acyclic cation -> separate parent (P-74.1.3 betaine)
    ri = mol.GetRingInfo()
    # The anion's parent-attachment atom = the heavy neighbour of the anion atom
    # (e.g. the carboxylate carbon). If that atom shares a ring with the cation,
    # the cation is skeletal to the anion's parent ring system (P-74.1.2).
    an = mol.GetAtomWithIdx(anion_idx)
    anchor_atoms = [n.GetIdx() for n in an.GetNeighbors() if n.GetSymbol() != 'H']
    for ar in ri.AtomRings():
        if cation_idx in ar and any(a in ar for a in anchor_atoms):
            return True
    return False


def _sever_cation_build_anion_parent(mol, cation_idx: int,
                                     parent_attach_idx: int) -> str:
    """Build the NEUTRAL anion-parent SMILES for P-74.1.3.

    Breaks the bond between the cationic atom and ``parent_attach_idx`` (the
    parent side of the attachment), discards the cation fragment, neutralizes
    the anion (re-protonate -> the neutral acid/alcohol form), and returns the
    canonical neutral SMILES of the parent fragment. Returns '' on failure.

    The cation atoms are NOT named here (they are re-expressed by the
    ``…azaniumyl`` prefix); only the anionic parent skeleton is named.
    """
    rw = Chem.RWMol(mol)
    bond = rw.GetBondBetweenAtoms(cation_idx, parent_attach_idx)
    if bond is None:
        return ''
    rw.RemoveBond(cation_idx, parent_attach_idx)
    # Cap the parent side with an explicit H (the severed valence becomes C-H).
    parent_atom = rw.GetAtomWithIdx(parent_attach_idx)
    parent_atom.SetNumExplicitHs(parent_atom.GetNumExplicitHs() + 1)
    # Neutralize every anion in place (re-protonate): the parent becomes the
    # neutral acid / alcohol that the pipeline can name.
    internal = _get_internal_charge_atoms(mol)
    for atom in rw.GetAtoms():
        ch = atom.GetFormalCharge()
        if ch < 0 and atom.GetIdx() not in internal:
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(atom.GetNumExplicitHs() + abs(ch))
    built = rw.GetMol()
    try:
        frags = Chem.GetMolFrags(built, asMols=True, sanitizeFrags=False)
        frag_idx_tuples = Chem.GetMolFrags(built, asMols=False, sanitizeFrags=False)
    except Exception:
        return ''
    # Keep the fragment that does NOT contain the cationic atom (the anion
    # parent). The cation fragment contains cation_idx.
    parent_mol = None
    for fr_mol, fr_idxs in zip(frags, frag_idx_tuples):
        if cation_idx not in fr_idxs:
            parent_mol = fr_mol
            break
    if parent_mol is None:
        return ''
    try:
        Chem.SanitizeMol(parent_mol)
    except Exception:
        return ''
    smi = Chem.MolToSmiles(parent_mol, canonical=True)
    return smi or ''


def _attachment_locant_on_anion_parent(mol, anion_idx: int,
                                       parent_attach_idx: int):
    """Locant of the cation-substituent attachment carbon on the anion parent.

    For a carboxylate anion, C1 is the carboxyl carbon (the anion O's heavy
    neighbour). The attachment locant = bond distance from that C1 to
    ``parent_attach_idx`` + 1. Returns None if no chain path exists (the
    substituent attaches off-chain — defer locant).
    """
    from rdkit.Chem import rdmolops
    an = mol.GetAtomWithIdx(anion_idx)
    anchors = [n.GetIdx() for n in an.GetNeighbors() if n.GetSymbol() != 'H']
    if not anchors:
        return None
    c1 = anchors[0]  # the carboxyl / characteristic-group carbon = locant 1
    path = rdmolops.GetShortestPath(mol, c1, parent_attach_idx)
    if not path:
        return None
    return len(path)  # distance(c1, attach) + 1 = the 1-indexed locant


def _parent_has_chain_locants(parent_anion_name: str) -> bool:
    """True if the anion-parent name is a systematic chain name that admits a
    substituent locant (``…anoate``). A retained 2-carbon ``acetate`` (and other
    locant-free retained anion names) carries no chain numbering, so the
    OPSIN-default position is used (no spurious locant)."""
    return parent_anion_name.endswith('anoate')


def _route_zwitterion(mol, sites, style: str) -> str:
    """GUARD 4: zwitterion anion-is-parent override (P-74.0).

    P-74.0 (verbatim): "anionic centers ... become the parent structure, into
    which the cationic part is substituted." So: FORCE the anion as the parent.

    - P-74.1.3 (cation on a DIFFERENT parent, e.g. the betaine quaternary
      ammonium): name the anion parent (neutralize ALL charges -> re-enter ->
      re-apply the anionic suffix) and PREFIX the structured cation-substituent
      ``(…azaniumyl)`` produced by ``cation_to_prefix``.
    - P-74.1.2 (cation INSIDE the anion's parent hydride, e.g. ring N+ of a
      pyridinium-2-carboxylate): the cation is kept on the parent as an ``-ium``
      suffix, NOT a prefix -> DEFER to the legacy path (returns '' here so the
      existing pipeline names it; route_charged does not own the skeletal-ium
      cumulative-suffix construction this plan — D-06 honest scope boundary).

    Sequencing (D-06): amino-acid zwitterions + betaines first. Ylides /
    amine-oxides / 1,n-dipolar (P-74.2) are out of scope -> '' (honest-fail).

    Returns the IUPAC name, or '' to fall through to the legacy path.
    """
    from rdkit.Chem import rdmolops

    cations = sites['cations']
    anions = sites['anions']

    # Scope (D-06): exactly one cationic and one anionic center (the amino-acid /
    # betaine majority). Multi-center dipolar zwitterions are deferred.
    if len(cations) != 1 or len(anions) != 1:
        return ''

    cation_idx = cations[0]['atom_idx']
    anion_idx = anions[0]['atom_idx']

    # P-74.1.2: cation skeletal to the anion's parent ring -> keep on parent as
    # an -ium suffix. route_charged does NOT build the cumulative ium+ate suffix
    # here; defer to the legacy path (honest scope boundary).
    if _cation_is_skeletal_to_anion_parent(mol, cation_idx, anion_idx):
        return ''

    # P-74.1.3: separable cation -> substituent prefix on the anion parent.
    # 1. The cation prefix (structured producer; '' on out-of-scope cation).
    path = rdmolops.GetShortestPath(mol, cation_idx, anion_idx)
    if len(path) < 2:
        return ''
    parent_attach_idx = path[1]  # the cation neighbour leading into the anion parent
    from ..assembly.substituent_naming import cation_to_prefix
    cat_prefix = cation_to_prefix(mol, cation_idx, parent_attach_idx)
    if not cat_prefix:
        return ''  # ylide / non-N onium / unnameable -> honest-fail

    # 2. The anion parent name (P-74.0: the anion is the parent). The cationic
    # part is a SEPARATE parent (P-74.1.3): SEVER the cation substituent from the
    # anion parent (break the cation-atom <-> parent_attach bond, cap the parent
    # side with H), keep ONLY the anion fragment, neutralize the anion, re-enter,
    # then re-apply the anionic suffix. (Neutralizing the quaternary cation in
    # place is impossible without breaking a bond — RDKit valence error — which
    # is exactly why P-74.1.3 demotes it to a prefix instead.)
    acls = classify_anion(mol, anions[0])

    parent_smi = _sever_cation_build_anion_parent(mol, cation_idx,
                                                  parent_attach_idx)
    if not parent_smi:
        return ''
    try:
        neutral_name = _reenter(parent_smi, style)
    except (RecursionError, ValueError, RuntimeError):
        return ''
    if not neutral_name or _is_malformed_parent(neutral_name):
        return ''

    if acls == 'carboxylate':
        # CARBOXYLATE parent: use the proven retained-name-aware transform
        # (name_carboxylate_anion handles 'ic acid'->'ate' as well as
        # 'oic acid'->'oate': acetic acid->acetate, propanoic acid->propanoate),
        # exactly as the deferred-carboxylate anion path does. The generic
        # apply_ion_suffix_to_name map keys only 'oic acid'/'carboxylic acid' and
        # cannot convert the retained 'acetic acid' (P-72.2.2.2.1.1).
        from .ions import name_carboxylate_anion
        parent_anion_name = name_carboxylate_anion(neutral_name)
    else:
        anion_total_charge = sum(a['charge'] for a in anions)
        parent_anion_name = apply_ion_suffix_to_name(
            neutral_name, anion_total_charge,
            allowed_suffixes=_ANION_ALLOWED_SUFFIXES.get(acls),
        )
    if not parent_anion_name:
        return ''

    # 3. Compute the attachment locant on the anion parent chain (PIN cites all
    # locants — the contributor guide pitfall 3). For a carboxylate parent C1 is the carboxyl
    # carbon; the locant of the cation-substituent carbon = its bond distance
    # from the carboxyl carbon + 1. Omit only when the parent is too short for an
    # ambiguity (a 1-carbon attach on a 2-carbon acetate, where OPSIN's default
    # is already correct and the retained 'acetate' carries no chain locant).
    locant_prefix = ''
    attach_locant = _attachment_locant_on_anion_parent(mol, anion_idx,
                                                        parent_attach_idx)
    if attach_locant is not None and attach_locant >= 2 \
            and _parent_has_chain_locants(parent_anion_name):
        locant_prefix = f'{attach_locant}-'

    # 4. Compose: {locant}-(cation-prefix)anion-parent (P-74.1.3 — prefix the
    # cation to the anionic parent). Enclosing marks per P-14.5.2 (complex prefix).
    return f'{locant_prefix}({cat_prefix}){parent_anion_name}'


def _neutralize_fragment(mol, *, add_h_for_cation: bool = False):
    """Strip ALL non-internal formal charges and radical electrons from a single
    organic fragment, returning the sanitized canonical neutral SMILES (or '').

    GUARD 2 is realized HERE: neutralizing ALL same-sign centers (not just one)
    lets the re-entered pipeline name the multi-center parent that bears every
    ionic suffix (e.g. butanedioic acid for a dicarboxylate dianion), which is
    exactly P-72.7(a) "maximum number of anionic centers". Mirrors the verified
    _name_oxoacid_anion neutralize loop (ions.py:929-939).

    Cation neutralization is class-dependent (CONTEXT D-02 — which neutral form):
      - PROTON-GAIN cations (protonated amine/onium): the cation has an EXTRA H,
        so removing |charge| H restores the neutral amine/hydride
        (``[NH4+]``->``ammonia``, ``C[NH3+]``->``methylamine``).  ``add_h_for_cation
        = False``.
      - HYDRIDE-LOSS cations (carbenium ``ylium``, acylium): the cation is the
        parent hydride MINUS a hydride (H-), so ADDING |charge| H restores the
        parent hydride (``[CH3+]``->``methane``, then ``ane``->``ylium`` =
        ``methylium``; P-73.2.2.1.1).  ``add_h_for_cation = True``.
    Internal (nitro/azide/N-oxide/diazo) charges are LEFT intact (P-59). Radical
    electrons are saturated with H (the P-71 suffix is re-applied by the caller).
    """
    internal = _get_internal_charge_atoms(mol)
    rw = Chem.RWMol(mol)
    changed = False
    for atom in rw.GetAtoms():
        idx = atom.GetIdx()
        n_rad = atom.GetNumRadicalElectrons()
        if n_rad:
            atom.SetNumRadicalElectrons(0)
            atom.SetNumExplicitHs(atom.GetNumExplicitHs() + n_rad)
            changed = True
        charge = atom.GetFormalCharge()
        if charge != 0 and idx not in internal:
            atom.SetFormalCharge(0)
            if charge < 0:
                # Anion: was deprotonated -> add H back.
                atom.SetNumExplicitHs(atom.GetNumExplicitHs() + abs(charge))
            elif add_h_for_cation:
                # Hydride-loss cation (ylium/acylium): restore the lost hydride.
                atom.SetNumExplicitHs(atom.GetNumExplicitHs() + charge)
            else:
                # Proton-gain cation: remove the extra proton(s).
                atom.SetNumExplicitHs(max(0, atom.GetNumExplicitHs() - charge))
            changed = True
    if not changed:
        return ''
    try:
        Chem.SanitizeMol(rw)
    except Exception:
        return ''
    smi = Chem.MolToSmiles(rw.GetMol(), canonical=True)
    return smi or ''


def _reenter(neutral_smi: str, style: str) -> str:
    """Re-enter the FULL pipeline on the neutral skeleton (the chokepoint core).

    SUB-03 (169.5): the neutral name is an INTERMEDIATE (ionized below), so the
    OPSIN validity gate is bypassed — a malformed intermediate must not be
    suppressed to a descriptive string before the ionize step. select_parent
    (cyclic) / find_principal_chain (acyclic) run INSIDE this call; this is why
    the chokepoint is "re-enter", NOT a direct select_parent call (the SMARTS
    that find the principal group are charge-sensitive and do not match a charged
    atom — IMPLEMENTATION-MAP §1).
    """
    from ..namer import Orthonym
    return Orthonym(style=style, _disable_opsin_validity_gate=True).name(neutral_smi)


def _classify_single_anion(mol, sites):
    """Return (cation_class, allowed_suffixes) for a single-/multi-anion fragment.

    GUARD 1: pick the FG class of the parent-bearing center, then its
    allowed_suffixes subset. For a multi-anion fragment the centers are
    homogeneous-by-construction here only when every center shares a class
    (the dicarboxylate case -> 'carboxylate'); a mixed multi-center fragment
    yields allowed=None (the seam runs unrestricted, matching the existing
    multi-anion neutralize-recurse behavior).
    """
    anions = sites['anions']
    classes = {classify_anion(mol, a) for a in anions}
    if len(classes) == 1:
        cls = next(iter(classes))
        return None, _ANION_ALLOWED_SUFFIXES.get(cls)  # None for 'carbanion'/'unknown'
    # Mixed classes (e.g. a carboxylate + an alkoxide): no single subset is safe;
    # leave the seam unrestricted (the re-entered multi-suffix parent name decides).
    return None, None


def _apply_guard3_reorder(mol, sites):
    """GUARD 3 (P-72.7d / P-73.7c) sanity check for a skeletal heteroatom charge.

    The neutralize->re-enter step already runs the SOUND P-44 cascade, which
    applies P-44.1.2 senior-skeletal-atom (the SAME element order) inside the
    re-entry — so for the single-center majority the senior-element parent is
    chosen automatically. This hook exists to make GUARD 3 EXPLICIT and auditable
    (and to refuse a pathological case where the only charged atom is a low-
    seniority element that the neutral cascade could mis-root); it does not
    re-implement P-44. Returns True if the chosen-fragment charge layout is
    acceptable for re-entry, False to bail ('' -> legacy fallthrough).
    """
    # Single ionic center is always fine (no competition).
    charged = sites['anions'] + sites['cations']
    if len(charged) <= 1:
        return True
    # Multi-center: acceptable when all charged atoms are the SAME sign (handled
    # by GUARD 2 neutralize-all) OR share the senior element. A mixed-sign multi-
    # center is a zwitterion (caught earlier). Same-element or same-sign -> OK.
    elems = {mol.GetAtomWithIdx(s['atom_idx']).GetSymbol() for s in charged}
    if len(elems) == 1:
        return True
    # Different elements on multiple same-sign centers: re-entry's P-44.1.2 picks
    # the senior one; accept (the cascade is sound). The element order is recorded
    # here for auditability (the senior element is the parent-bearing one).
    senior = min(elems, key=lambda e: _ELEMENT_SENIORITY.get(e, 999))
    return senior in _ELEMENT_SENIORITY


def route_charged(mol, style: str = 'pin') -> str:
    """THE single mandatory parent-selection chokepoint (CHOKE-01 / CHOKE-02).

    Generalizes _name_oxoacid_anion to every charged class. Returns the IUPAC
    name, or '' on any failure / out-of-scope shape (caller falls through to the
    legacy cascade — v18 byte-identical contract).

    Pipeline (IMPLEMENTATION-MAP §2):
      1. metal complex                       -> '' (salt composition is Plan 04)
      2. multi-fragment (salt / arbitrary)   -> '' (salt composition is Plan 04)
      3. classify ionic centers + GUARDS 1-4
      4. neutralize the fragment (+ sanitize)
      5. re-enter Orthonym(style).name(neutral_smi)
      6. apply_ion_suffix_to_name(..., allowed_suffixes=<class subset>, cation_class=<class>)
    """
    if mol is None:
        return ''

    # --- Step 1: metal complex -> Plan 04 (keep the anion-path regression clean).
    # P-65.6.2.1 simple-metal-salt composition (<cation word> <anion>) is Plan 04;
    # for now metals fall through so the byte-identical anion seam is untouched.
    if _has_metal(mol):
        return ''

    # --- Step 2: multi-fragment (dot-disconnected salt / arbitrary) -> Plan 04.
    if len(Chem.GetMolFrags(mol)) > 1:
        return ''

    # --- Step 3: enumerate + classify ionic / radical centers.
    sites = get_ion_sites(mol)  # excludes internal nitro/azide/N-oxide/diazo (P-59)
    n_anions = len(sites['anions'])
    n_cations = len(sites['cations'])
    from ..perception.ions import get_radical_sites
    radical_sites = get_radical_sites(mol)

    if n_anions == 0 and n_cations == 0 and not radical_sites:
        return ''  # nothing charged/radical here (internal-only charge -> neutral)

    # GUARD 4 (P-74.0): zwitterion anion-is-parent override. The anion is FORCED
    # as the parent; a separable cation (P-74.1.3) is demoted to an (…azaniumyl)
    # substituent prefix; a skeletal cation (P-74.1.2) is deferred to the legacy
    # path. 169.6-04 (was a Plan-03 detect-and-defer seam).
    if _is_zwitterion(sites):
        return _route_zwitterion(mol, sites, style)

    # A charged AND radical species (radical ion) is out of scope here -> bail.
    if radical_sites and (n_anions or n_cations):
        return ''

    # --- Determine the ionic-suffix obligation (GUARD 1 + class-keyed transforms).
    total_charge = sum(a['charge'] for a in sites['anions']) \
        + sum(c['charge'] for c in sites['cations'])
    cation_class: Optional[str] = None
    allowed_suffixes = None
    cation_kind: Optional[str] = None
    add_h_for_cation = False
    radical_suffix = None

    if radical_sites:
        # Radicals funnel through the chokepoint too (kill radicals.py alkyl
        # carbon counting): neutralize -> re-enter -> append the P-71 -yl/
        # -ylidene/-ylidyne suffix (applied AFTER re-entry below). SCOPE: ONLY a
        # carbon-centered ALKYL/-ylidene/-ylidyne radical. The acyl (R-C(=O).),
        # oxyl (R-O.), and aryl subtypes are NOT a plain parent-hydride hydrogen
        # loss (acyl -> -oyl on the acid name; oxyl -> -oxyl; aryl -> the ring
        # radical) and keep their structured radicals.py helpers -> bail here so
        # name_radical / _handle_radical fall through to them.
        if len(radical_sites) != 1:
            return ''  # di/poly-radicals out of scope -> legacy fallthrough
        from .radicals import classify_radical
        rinfo = classify_radical(mol, radical_sites[0])
        if rinfo['subtype'] in ('acyl', 'oxyl', 'aryl', 'aminyl', 'thiyl'):
            return ''  # structured helpers / out of scope -> legacy fallthrough
        n_e = radical_sites[0]['n_electrons']
        radical_suffix = {1: 'yl', 2: 'ylidene', 3: 'ylidyne'}.get(n_e)
        if radical_suffix is None:
            return ''
    elif n_cations and not n_anions:
        if not _apply_guard3_reorder(mol, sites):
            return ''
        cclasses = {classify_cation(mol, c) for c in sites['cations']}
        if len(cclasses) != 1:
            return ''  # heterogeneous multi-cation -> legacy fallthrough
        ccls = next(iter(cclasses))
        cation_class, allowed_suffixes = _CATION_SPEC.get(ccls, (None, None))
        # ylium / acylium are HYDRIDE-LOSS cations (P-73.2.2.1.1 / P-73.2.3.1):
        # neutralize by ADDING the lost hydride so the parent hydride is named.
        add_h_for_cation = ccls in ('ylium', 'acylium')
        cation_kind = ccls
    elif n_anions and not n_cations:
        if not _apply_guard3_reorder(mol, sites):
            return ''
        # CARBOXYLATE anions (mono + poly) are DEFERRED to the proven, retained-
        # name-aware carboxylate path (_name_carboxylate_systematic), which keeps
        # benzoate / 2-naphthoate / succinate / malonate byte-identical. The
        # textual chokepoint seam would flip those retained names to the
        # systematic -dioate form (a style change, NOT a fix), and the proven
        # path already neutralizes->re-enters->ionizes correctly. route_charged
        # OWNS only the deleted-stub classes (alkoxide/phenolate/carbanion/
        # thiolate/aminide) + the S/P oxoacid anions.
        if any(classify_anion(mol, a) == 'carboxylate' for a in sites['anions']):
            return ''
        cation_class, allowed_suffixes = _classify_single_anion(mol, sites)
    else:
        return ''  # defensive: mixed handled by zwitterion guard already

    # --- Step 4: neutralize the fragment (GUARD 2 = neutralize ALL same-sign).
    neutral_smi = _neutralize_fragment(mol, add_h_for_cation=add_h_for_cation)
    if not neutral_smi:
        return ''

    # --- Step 5: re-enter the FULL pipeline on the neutral skeleton.
    try:
        neutral_name = _reenter(neutral_smi, style)
    except (RecursionError, ValueError, RuntimeError):
        return ''
    if not neutral_name or _is_malformed_parent(neutral_name):
        return ''

    # --- Step 6: re-apply the class-correct ionic / radical suffix.
    if radical_suffix is not None:
        # P-71: alkane 'ane' -> 'yl'/'ylidene'/'ylidyne'; elide a trailing 'e'
        # otherwise. Structured (NOT carbon counting): operate on the parent name.
        return _apply_radical_suffix(neutral_name, radical_suffix)

    # (CARBOXYLATE anions are deferred to the proven path earlier; they never
    # reach this point — see the carboxylate guard in Step 3.)

    # AMINIUM (protonated amine, P-73.1.2.1) reuses the PROVEN name_aminium_cation
    # transform on the re-entered amine name (amine->aminium, ammonia->ammonium).
    # This is the existing _name_aminium_systematic primary path, generalized into
    # the funnel — so deleting the stub's carbon-counting FALLBACK (Task 2) keeps
    # the aminium output byte-identical (methylaminium / pyrrolidineium etc.).
    if cation_kind == 'aminium':
        from .ions import name_aminium_cation
        return name_aminium_cation(neutral_name) or ''

    return apply_ion_suffix_to_name(
        neutral_name, total_charge,
        allowed_suffixes=allowed_suffixes,
        cation_class=cation_class,
    )


def _apply_radical_suffix(neutral_name: str, radical_suffix: str) -> str:
    """Apply the P-71 radical suffix to a re-entered neutral parent name.

    Replaces the deleted radicals.py carbon-counting. Structured, general:
      - alkane 'ane' -> 'yl'/'ylidene'/'ylidyne' (methane->methyl; P-71.1.1).
      - otherwise elide a single trailing 'e' before the consonant-initial
        suffix (no double 'e'); append directly if there is none.
    Returns '' if the parent name is empty (caller falls through).
    """
    if not neutral_name:
        return ''
    name = neutral_name
    if name.endswith('ane'):
        return name[:-3] + radical_suffix       # P-71.1.1: methane->methyl
    if name.endswith('e'):
        return name[:-1] + radical_suffix       # elide trailing e (benzene->benzyl-shape)
    return name + radical_suffix
