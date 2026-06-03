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
  GUARD 4  zwitterion anion-is-parent override (P-74.0). DEFERRED to Plan 04 —
           a clean seam is left here (``_is_zwitterion`` -> return '' so the
           legacy path is unchanged this plan), NOT implemented.

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
    """GUARD 4 seam (P-74.0): a single fragment carrying BOTH a (non-internal)
    cationic and a (non-internal) anionic center is a zwitterion. DEFERRED to
    Plan 04 (the anion-is-parent override + the -aminiumyl prefix producer); here
    we only DETECT it so route_charged can return '' and leave the legacy path
    untouched this plan. NOT implemented — clean seam only.
    """
    return bool(sites.get('cations')) and bool(sites.get('anions'))


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

    # GUARD 4 (P-74.0): zwitterion -> DEFERRED to Plan 04. Clean seam: bail so the
    # legacy path is byte-identical this plan. (Do NOT implement anion-is-parent.)
    if _is_zwitterion(sites):
        return ''

    # A charged AND radical species (radical ion) is out of scope here -> bail.
    if radical_sites and (n_anions or n_cations):
        return ''

    # --- Determine the ionic-suffix obligation (GUARD 1 + class-keyed transforms).
    total_charge = sum(a['charge'] for a in sites['anions']) \
        + sum(c['charge'] for c in sites['cations'])
    cation_class: Optional[str] = None
    allowed_suffixes = None
    anion_class: Optional[str] = None
    cation_kind: Optional[str] = None
    add_h_for_cation = False
    radical_suffix = None

    if radical_sites:
        # Radicals funnel through the chokepoint too (kill radicals.py carbon
        # counting): neutralize -> re-enter -> append the P-71 -yl/-ylidene/
        # -ylidyne suffix (applied AFTER re-entry below).
        if len(radical_sites) != 1:
            return ''  # di/poly-radicals out of scope -> legacy fallthrough
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
        aclasses = {classify_anion(mol, a) for a in sites['anions']}
        anion_class = next(iter(aclasses)) if len(aclasses) == 1 else None
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

    # CARBOXYLATE / oxoacid anions reuse the PROVEN, byte-identical helpers
    # (_acid_name_to_carboxylate handles 'ic acid'/'oic acid'/'carboxylic acid'
    # AND the retained names acetic->acetate, formic->formate that the generic
    # SUFFIX_MAP seam does not carry). This keeps the 169.5 anion path
    # byte-identical instead of regressing acetate/formate to '' (Pitfall 4).
    if anion_class == 'carboxylate':
        from .ions import _acid_name_to_carboxylate
        carboxylate_count = sum(
            1 for a in sites['anions'] if classify_anion(mol, a) == 'carboxylate'
        )
        return _acid_name_to_carboxylate(neutral_name, carboxylate_count) or ''

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
