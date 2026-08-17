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
# Anti-hang complexity bound (169.6 follow-on). route_charged sends charged
# species through the FULL select_parent/assembly pipeline, whose candidate
# enumeration cost grows combinatorially with molecular size. On a pathological
# large / highly-symmetric charged molecule that pipeline blows up (RDKit valence
# churn) — the 169.5 carbon-counting stub used to absorb these instantly but
# WRONGLY; deleting the stub (Plan 03) exposed the blowup and hung the full-corpus
# benchmark for 14.5h with NO escape (the spin is signal-resistant, so the
# per-compound timeout cannot interrupt it). Bound it: above this heavy-atom count
# the router degrades GRACEFULLY (returns '' -> legacy fallback / honest
# descriptive) — it does NOT hang and does NOT emit a wrong name.
#
# Regression-safe threshold: the LARGEST charged compound that round-trips in the
# 169.5 baseline is 46 HA (p99=40); 50 leaves a margin above every RT-er while
# catching the 27 charged giants >= 80 HA (up to 209) that never round-trip.
# Independently corroborates HERITAGE's documented 44-atom hard limit. Raise it if
# the router is later shown to name larger charged molecules in bounded time.
# =============================================================================
_MAX_CHARGED_ROUTE_HEAVY_ATOMS = 50

# BBR-CHG-169.6-caveats (Phase 169.7): canonical SMILES of retained charged species
# that LACK a valid systematic PIN — their neutralize->re-name chokepoint path yields
# an OPSIN-unparseable form (the 169.6 'unknown'/wrong-retained regression). These use
# their sanctioned retained name (P-72/P-73/P-74). NARROW by design: alkoxides and
# carboxylate (poly)anions have valid systematic PINs (SUB-01 -olate / deferred -ate)
# and are NOT here, so GUARD 1/2 routing is preserved. Add a species here ONLY if its
# chokepoint systematic form is genuinely OPSIN-unparseable (verify before adding).
_RETAINED_FIRST_CHARGED = frozenset({
    '[SH3+]',                                     # sulfonium (P-73.1.1.1)
    'N[O-]',                                      # aminoxide (P-74)
    'O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F',     # bistriflimide (P-72)
    # Wave2 T2d (P-63.8.1): the retained alkoxide names ARE the PINs
    # ("sodium methoxide (PIN) sodium methanolate" — BB verbatim), so the
    # salt path must hit them BEFORE the systematic -olate chokepoint (the
    # standalone-anion path already resolves them via RETAINED_ANIONS).
    # Exact bare skeletons only; substituted alkoxides keep systematic
    # -olate. isopropoxide is general nomenclature (PIN propan-2-olate) and
    # is deliberately absent.
    'C[O-]',                                      # methoxide
    'CC[O-]',                                     # ethoxide
    'CCC[O-]',                                    # propoxide
    'CCCC[O-]',                                   # butoxide
    'CC(C)(C)[O-]',                               # tert-butoxide
    '[O-]c1ccccc1',                               # phenoxide
})


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
    # P-65.6.1: the carbo(di)thioate acid anion (R-C(=S)-S- / R-C(=O)-S-). Its
    # neutral form is a (di)thioic acid ending '-oic acid' (propanedithioic acid),
    # so it takes the -oate seam -> propanedithioate. Restricting to {'oic acid'}
    # keeps GUARD-1 tight (a stem must genuinely end in the acid suffix).
    'carbodithioate': frozenset({'oic acid'}),
    'alkoxide': frozenset({'ol'}),
    'phenolate': frozenset({'ol'}),
    'thiolate': frozenset({'thiol'}),
    'aminide': frozenset({'amine'}),
    # 'carbanion' -> None (bare -> -ide via the resolvers empty-suffix path).
}

# T3 (Phase 173.6): anion class -> the FG name forced as principal when a SENIOR
# neutral acid (carboxylic) would otherwise hijack the re-entry's principal slot.
# Per P-72/P-74 the CHARGED group is the principal characteristic group of an
# anion: O=C(O)CCS(=O)(=O)[O-] -> 2-carboxyethanesulfonate (carboxy PREFIX,
# sulfonate suffix), not the carboxylic-principal 'propanoate' the neutral
# seniority (P-41 carboxylic > sulfonic) yields for the di-acid skeleton.
_ANION_PRINCIPAL_FG = {
    'carboxylate': 'carboxylic_acid',   # WS-E.3 (P-72.2): carboxylate is the senior anion acid class
    'sulfonate': 'sulfonic_acid',       # 173.6 - byte-identical, do not change
    'sulfinate': 'sulfinic_acid',       # 173.6 - byte-identical, do not change
    'phosphonate': 'phosphonic_acid',   # 173.6 - byte-identical, do not change
}

# P-72.2 anion acid-class seniority (highest first) for the charge-first PCG pick.
# Declarative single source of truth (the "class-keyed transform table" pattern):
# the carboxylate > sulfonate > sulfinate > phosphonate order mirrors the P-72.2
# anion-acid-class seniority used when several ionized acid centres coexist.
_ANION_PCG_SENIORITY = ('carboxylate', 'sulfonate', 'sulfinate', 'phosphonate')


def classify_charged_pcg(mol, sites) -> Optional[str]:
    """Charge-first PCG classifier (WS-E.3, D-11/D-12). Runs on the ORIGINAL
    (un-neutralized) mol. Returns the FG-name to FORCE as the principal
    characteristic group (a detect_functional_groups KEY: 'carboxylic_acid',
    'sulfonic_acid', 'sulfinic_acid', 'phosphonic_acid'), or None when the
    charged class has NO neutral FG anchor (carbanion/alkoxide/thiolate/
    aminide/phenolate -> handled by the WS-E.2 emit_parent_hydride_cumulative_
    suffix primitive / the existing suffix seam, NOT this override) or when a
    different route owns the molecule.

    Ordering (D-12; P-72 anions / P-73 cations / P-74 zwitterion / P-33.3 radical):
      1. radical present -> None (P-33.3 radical>anion>cation; the -yl primitive owns it)
      2. mixed-sign (zwitterion) -> None (P-74 _route_zwitterion owns it)
      3. anion(s): senior ionized acid class per P-72 -> its acid-FG key
      4. cation(s) only -> None (the aminium / class-keyed cation transforms own it)
    The forced FG MUST be a features.functional_groups key (namer.py:1821) or it
    silently no-ops; for a class with no neutral FG anchor return None.
    BlueBookV2.md:17580 (radical>anion>cation seniority)."""
    # Rule 1 (P-33.3): a radical co-occurring outranks the ionic centre; the
    # P-71 -yl/-ylidene primitive owns it, not this override.
    from ..perception.ions import get_radical_sites
    if get_radical_sites(mol):
        return None
    # Rule 2 (P-74): a mixed-sign zwitterion is owned by _route_zwitterion
    # (anion-is-parent + azaniumyl prefix), never the charge-first override here.
    if sites.get('cations') and sites.get('anions'):
        return None
    # Rule 3 (P-72): pick the senior ionized acid class present among the anions.
    anions = sites.get('anions') or []
    if anions:
        acls = {classify_anion(mol, a) for a in anions}
        for senior in _ANION_PCG_SENIORITY:
            if senior in acls:
                return _ANION_PRINCIPAL_FG[senior]
        # No anion class has a neutral FG anchor (all carbanion/alkoxide/
        # phenolate/thiolate/aminide) -> handled by the suffix primitive, not here.
        return None
    # Rule 4: cation-only (or charge-free) fragments are not this classifier's job.
    return None

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
    substituent locant (``…anoate``, ``…enoate``, ``…ynoate``, ``…dioate``).
    A retained 2-carbon ``acetate`` (and other locant-free retained anion names)
    carries no chain numbering, so the OPSIN-default position is used (no spurious
    locant)."""
    return parent_anion_name.endswith(('anoate', 'enoate', 'ynoate', 'dioate'))


def _name_ester_anion_zwitterion(mol, cation_idx: int, anion_idx: int) -> str:
    """Name a choline-family acid-ester-anion ZWITTERION (v33 Phase 3 B1).

    P-74.0 forces the anion (a P/S oxoacid mono-ester anion, P-72.2.2.2.1.2)
    as the parent -- but here, unlike the betaine path in `_route_zwitterion`
    below (P-74.1.3, cation on a DIFFERENT parent, severed and re-attached as
    a `(...azaniumyl)` prefix), the cation sits INSIDE the ester-owner arm R
    itself (``R-O-SO3[-]``, ``R`` = ``2-(trimethylazaniumyl)ethyl``). So the
    owner substituent is named WITH the cation in place (the shipped
    cation-bearing-substituent capability, `e4936b8e`) via the same recursive
    ``_p_ester_owner_group`` helper the neutral/pure-anion acid-ester
    producers use (`phosphorus.py`, `acid_ester_anion.py`), and the acid word
    (``sulfate`` / ``hydrogen phosphate`` / ...) is derived IN PLACE from the
    surviving free ``-OH`` count via ``conjugate_controller`` (D-04 — never
    neutralize-then-rename): ``'{owner} {word}'``
    (``2-(trimethylazaniumyl)ethyl sulfate``).

    ``anion_idx`` is ANY one terminal acidic O of the acid centre -- the
    central P/S and every one of its OTHER terminal oxygens are discovered by
    walking ``central``'s own bonds, so a mono-ester DIANION (two independent
    ``[O-]`` "sites" on the same P, `get_ion_sites` being per-atom) is handled
    correctly regardless of which one the caller passes.

    Fail-closed (``''``) off anything but a clean single-centre, non-ring,
    neutral-P/S, single-ester-owner, single-acid-group shape: any non-O
    double bond (thio), any P-N/S-C substituent, a second ester owner
    (diester), a P-O-P bridge, a cation NOT on the owner arm, or an owner
    fragment `name_substituent` cannot spell all decline. The final
    atom-coverage check (owner_frag union {central, dbl-bonded O's, ester-O,
    terminal O's} must equal EVERY heavy atom in the molecule) is the 0-wrong
    guard -- it never lets a dropped atom through. The outer SELF-01/OPSIN
    gate RT-verifies the returned name as the ultimate backstop.
    """
    from .conjugate_controller import (PHOSPHATE_WORD, SULFATE_WORD,
                                       _terminal_acid_oxygens)
    from .phosphorus import _p_ester_owner_group

    anion_atom = mol.GetAtomWithIdx(anion_idx)
    if anion_atom.GetSymbol() != 'O' or anion_atom.GetFormalCharge() >= 0:
        return ''
    central_candidates = list(anion_atom.GetNeighbors())
    if len(central_candidates) != 1:
        return ''
    central = central_candidates[0]
    if central.GetSymbol() not in ('P', 'S') or central.GetFormalCharge() != 0 \
            or central.IsInRing():
        return ''
    central_idx = central.GetIdx()

    accounted = {central_idx}
    ester_oxygens = []
    dbl_oxo = 0
    for b in central.GetBonds():
        nb = b.GetOtherAtom(central)
        bt = b.GetBondType()
        sym = nb.GetSymbol()
        if bt == Chem.BondType.DOUBLE:
            if sym != 'O':
                return ''                          # P=S / P=C / S=C -> thio, defer
            dbl_oxo += 1
            accounted.add(nb.GetIdx())
            continue
        if bt != Chem.BondType.SINGLE:
            return ''
        if sym != 'O':
            return ''                              # P-N / P-C / S-C -> defer
        others = [x for x in nb.GetNeighbors() if x.GetIdx() != central_idx]
        if not others:
            # A terminal O: the counted [O-], a second [O-] (dianion), or a
            # free -OH. A bare terminal O with neither charge nor H is
            # ambiguous -> defer.
            if nb.GetFormalCharge() == 0 and nb.GetTotalNumHs() == 0:
                return ''
            accounted.add(nb.GetIdx())
        elif len(others) == 1 and others[0].GetSymbol() == 'C' \
                and nb.GetFormalCharge() == 0:
            ester_oxygens.append(nb.GetIdx())      # -O-C ester owner
            accounted.add(nb.GetIdx())
        else:
            return ''                              # P-O-P bridge / charged owner O -> defer

    if len(ester_oxygens) != 1:
        return ''                                  # need exactly one ester owner (no diester)
    ester_o_idx = ester_oxygens[0]

    required_dbl_oxo = 2 if central.GetSymbol() == 'S' else 1
    if dbl_oxo != required_dbl_oxo:
        return ''

    prot, _anion_n = _terminal_acid_oxygens(mol, central_idx, ester_o_idx)
    word_table = SULFATE_WORD if central.GetSymbol() == 'S' else PHOSPHATE_WORD
    word = word_table.get(prot)
    if not word:
        return ''

    got = _p_ester_owner_group(mol, ester_o_idx, central_idx)
    if not got:
        return ''
    owner, owner_frag = got
    if not owner or owner == 'substituent' or not isinstance(owner, str):
        return ''
    if cation_idx not in owner_frag:
        return ''                                  # cation not on the owner arm -> defer

    accounted |= owner_frag
    if accounted != set(range(mol.GetNumAtoms())):
        return ''                                  # atom-coverage guard -- never drop an atom

    return f'{owner} {word}'


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

    # v33 Phase 3 B1 (choline-family acid-ester-anion zwitterion, P-72.2.2.2.1.2):
    # a cation sitting INSIDE the ester-owner arm R of a P/S oxoacid mono-ester
    # anion (e.g. R-O-SO3[-], R = 2-(trimethylazaniumyl)ethyl -> choline
    # sulfate). Tried FIRST, before the single-cation/single-anion scope check
    # below, because `get_ion_sites` is per-ATOM: a mono-ester phosphate
    # DIANION (R-O-PO3^2-) surfaces as TWO independent anion "sites" (each
    # [O-] its own atom) even though it is ONE acid-ester anion functional
    # group -- GUARD 2's ionic-center-count-first philosophy already documented
    # above (a multi-center anion on one FG is one named entity, not per-atom).
    # `_name_ester_anion_zwitterion`'s own shape + atom-coverage validation
    # declines ('') anything that is not a clean single-cation, single-P/S-
    # centre mono-ester, so this is a pure ADD: every existing zwitterion class
    # (carboxylate betaine, amino-acid, ring carboxylate, ...) has an anion
    # whose sole neighbour is a carbon, so `central.GetSymbol() not in ('P',
    # 'S')` declines INSTANTLY and falls through unchanged to the scope check
    # and legacy paths below -- 0 behaviour change for anything but the new
    # shape.
    if len(cations) == 1 and len(anions) >= 1:
        est_name = _name_ester_anion_zwitterion(
            mol, cations[0]['atom_idx'], anions[0]['atom_idx'])
        if est_name:
            return est_name

    # Scope (D-06): exactly one cationic and one anionic center (the amino-acid /
    # betaine majority). Multi-center dipolar zwitterions are deferred.
    if len(cations) != 1 or len(anions) != 1:
        return ''

    cation_idx = cations[0]['atom_idx']
    anion_idx = anions[0]['atom_idx']
    cation_atom = mol.GetAtomWithIdx(cation_idx)

    # F-T6 (DD3, P-74.1.2): the cationic centre is a RING atom skeletal to the
    # anion's parent ring (the ring N+ of a pyridinium carboxylate). It is kept on
    # the parent as an -ium suffix and combined with the anion's -carboxylate into
    # the cumulative '<ring>-<N-locant>-ium-<carboxyl-locant>-carboxylate' (cation
    # cited before anion). Previously deferred (the line-413 skeletal check below
    # returned ''), which let the legacy path neutralize it to 'nicotinic acid'
    # (charge dropped). Attempt this BEFORE the protonated-amine defer (a
    # protonated ring N has totalH>0 and would otherwise be deferred). Fail-closed
    # ('' here) preserves every existing zwitterion (amino acid / betaine / ylide)
    # path: a non-ring cation, or a non-carboxylate anion, declines and falls
    # through to the established handling below. (The helper self-validates that
    # the cation is a ring atom AND the carboxylate hangs off that same ring;
    # _cation_is_skeletal_to_anion_parent is NOT a usable precondition here — it
    # inspects the anion's DIRECT neighbour, the exocyclic carboxyl carbon, which
    # is not a ring atom, so it returns False for exactly these molecules.)
    from .ions import emit_zwitterion_ring_carboxylate
    zwit = emit_zwitterion_ring_carboxylate(mol, cation_idx, anion_idx)
    if zwit:
        return zwit

    # SCOPE (D-06): GUARD 4's (azaniumyl) prefix is for a cation on a DIFFERENT
    # parent (P-74.1.3) — i.e. a QUATERNARY ammonium (0 H) that has NO neutral
    # free-amine form, the betaine class. A PROTONATED amine (NH3+/NH2+/NH+, >0
    # H) neutralizes to a free amino SUBSTITUENT on the parent, so an amino-acid
    # zwitterion / zwitterionic peptide is named by its established neutral /
    # retained / peptide form (P-74 neutral-form recommendation) — NOT the
    # azaniumyl prefix. Defer those to the legacy path (which sequences amino-acid
    # zwitterions + peptides correctly). Only the quaternary betaine class is
    # owned here. (Betaine N+ totalH=0; glycine/dipeptide N+ totalH=3.)
    if cation_atom.GetSymbol() == 'N' and cation_atom.GetTotalNumHs() > 0:
        return ''

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

    # SCOPE (D-06): GUARD 4 confidently handles the amino-acid / betaine majority
    # (a carboxylate anion) and clean hydroxy/thio anion parents. A phosphate /
    # sulfate ESTER oxygen ([O-] bonded to P or to an S that bears =O) is
    # mis-classified 'alkoxide' by classify_anion but is NOT a real alkoxide — a
    # large multifunctional phospholipid is out of scope -> decline so the
    # neutral-form path names it (honest-fail, not a malformed -olate).
    anion_atom = mol.GetAtomWithIdx(anion_idx)
    if anion_atom.GetSymbol() == 'O':
        for nb in anion_atom.GetNeighbors():
            if nb.GetSymbol() == 'P':
                return ''  # phosphate ester O- -> out of scope
            if nb.GetSymbol() == 'S' and any(
                    b.GetBondType() == Chem.BondType.DOUBLE
                    for b in nb.GetBonds()):
                return ''  # sulfate/sulfonate ester O- -> out of scope

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
    # cation to the anionic parent). Enclosing marks per P-16.5.1.1 (complex prefix).
    composed = f'{locant_prefix}({cat_prefix}){parent_anion_name}'

    # Defensive: never ship a malformed composition where the prefix glues
    # directly onto a parent locant with no separator ('(...)2-hydroxy...'). If
    # the parent name carries its own leading locant but we computed no
    # cation-locant prefix, the join is ambiguous -> decline (honest-fail, the
    # neutral-form path names it). A clean name has the parent starting with a
    # letter right after the ')'. Strip leading stereodescriptors (e.g. '(2E)')
    # before checking for a digit, so a future un-handled stereo parent with no
    # computed locant also fails closed.
    from ..assembly.naming_utils import _STEREO_PAREN_RE
    _after = _STEREO_PAREN_RE.sub('', parent_anion_name, count=1).lstrip('-')
    if not locant_prefix and _after[:1].isdigit():
        return ''
    return composed


def _name_diazonium(mol, cation_idx: int, style: str) -> str:
    """P-73.2.2.3: name a diazonium cation R-N2+ as ``<parent-hydride>diazonium``.

    The cationic N (``cation_idx``) sits somewhere in a terminal
    -N#N+/-N=N+ pair; WHICH of the two N atoms carries the formal charge is
    a resonance-drawing choice RDKit does not normalise (Phase 3B SPY,
    `` Q4). The CANONICAL
    drawing (``R-N+#N``) puts the charge on the PROXIMAL N -- directly
    bonded to the parent-attachment atom. The charge-shifted TWIN
    (``R-N=N+``) puts it on the TERMINAL N instead, whose only neighbour is
    the other N -- ``classify_cation`` already returns ``'diazonium'`` for
    both, but the parent-attachment atom is then two bonds away, not a
    direct neighbour of ``cation_idx``. Locate it by walking from whichever
    N is proximal to the parent, not by assuming ``cation_idx`` itself is.

    Sever the parent from the -N#N+/-N=N+ pair, cap the parent side with H,
    discard the diazo fragment, name the neutral parent hydride (benzene /
    methane), and append 'diazonium' (``benzenediazonium`` /
    ``methanediazonium``). Returns '' on any decline.
    """
    try:
        cat = mol.GetAtomWithIdx(cation_idx)
    except (RuntimeError, IndexError, OverflowError):
        return ''
    diazo_n = None
    parent_attach = None
    for nb in cat.GetNeighbors():
        b = mol.GetBondBetweenAtoms(cation_idx, nb.GetIdx())
        if (nb.GetSymbol() == 'N' and b is not None
                and b.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE)):
            diazo_n = nb.GetIdx()
        else:
            parent_attach = nb.GetIdx()
    if diazo_n is None:
        return ''

    # proximal_n is whichever N of the pair is directly bonded to the parent;
    # terminal_n is the OTHER one, and must have degree 1 (a genuine terminal
    # -N#N+/-N=N+, not an internal azo / ring N).
    proximal_n, terminal_n = cation_idx, diazo_n
    if parent_attach is None:
        # Resonance-shifted twin: cation_idx has no non-N neighbour (the
        # charge sits on the TERMINAL N), so the parent-attachment atom is
        # two bonds away -- reached via diazo_n's other neighbour, which
        # makes diazo_n the PROXIMAL N here (not terminal).
        dn = mol.GetAtomWithIdx(diazo_n)
        others = [nb.GetIdx() for nb in dn.GetNeighbors() if nb.GetIdx() != cation_idx]
        if len(others) != 1:
            return ''
        parent_attach = others[0]
        proximal_n, terminal_n = diazo_n, cation_idx

    if mol.GetAtomWithIdx(terminal_n).GetDegree() != 1:
        return ''
    rw = Chem.RWMol(mol)
    rw.RemoveBond(proximal_n, parent_attach)
    pa = rw.GetAtomWithIdx(parent_attach)
    pa.SetNumExplicitHs(pa.GetNumExplicitHs() + 1)
    built = rw.GetMol()
    try:
        frags = Chem.GetMolFrags(built, asMols=True, sanitizeFrags=False)
        fidx = Chem.GetMolFrags(built, asMols=False, sanitizeFrags=False)
    except Exception:
        return ''
    parent_mol = None
    for fm, fi in zip(frags, fidx):
        if cation_idx not in fi:
            parent_mol = fm
            break
    if parent_mol is None:
        return ''
    try:
        Chem.SanitizeMol(parent_mol)
        parent_smi = Chem.MolToSmiles(parent_mol, canonical=True)
    except Exception:
        return ''
    if not parent_smi:
        return ''
    try:
        neutral = _reenter(parent_smi, style)
    except (RecursionError, ValueError, RuntimeError):
        return ''
    if not neutral or _is_malformed_parent(neutral):
        return ''
    return apply_ion_suffix_to_name(neutral, 1, cation_class='diazonium') or ''


def emit_acylium(mol, cation_idx: int, style: str) -> str:
    """P-73.2.3.1 (BB 41623 PIN): name an acylium cation R-C(+)=O.

    Reconstructs the ACID (adds an -OH back onto the C+, restoring R-COOH),
    names it, then applies the acid->acylium class-keyed transform
    ('carboxylic acid'->'carbonylium', 'oic acid'->'oylium', 'ic acid'->
    'ylium'; ``apply_ion_suffix_to_name`` / ``_CLASS_KEYED_CATION_TRANSFORMS
    ['acylium']``). Deliberately NOT the generic ``add_h_for_cation`` path used
    for a plain carbenium ylium -- THAT restores an ALDEHYDE (R-CHO), not an
    acid, which would misname 'acetylium' as an aldehyde-ylium.

    Mirrors ``_name_diazonium`` (sever/reconstruct, name, class-keyed
    transform), but ADDS an atom (the reconstructed -OH) instead of severing
    one. Returns '' on any decline -- the caller's generic cascade then runs
    (``add_h_for_cation`` there cannot mis-fire either: an aldehyde name never
    ends in an acid suffix, so ``apply_ion_suffix_to_name`` also returns ''
    on that fallback path -- fail-closed, never a wrong name).
    """
    try:
        cat = mol.GetAtomWithIdx(cation_idx)
    except (RuntimeError, IndexError, OverflowError):
        return ''
    if cat.GetSymbol() != 'C' or cat.GetFormalCharge() != 1:
        return ''
    rw = Chem.RWMol(mol)
    rw.GetAtomWithIdx(cation_idx).SetFormalCharge(0)
    oh_idx = rw.AddAtom(Chem.Atom('O'))
    rw.AddBond(cation_idx, oh_idx, Chem.BondType.SINGLE)
    try:
        built = rw.GetMol()
        Chem.SanitizeMol(built)
        acid_smi = Chem.MolToSmiles(built, canonical=True)
    except Exception:
        return ''
    if not acid_smi:
        return ''
    try:
        acid_name = _reenter(acid_smi, style)
    except (RecursionError, ValueError, RuntimeError):
        return ''
    if not acid_name or _is_malformed_parent(acid_name):
        return ''
    return apply_ion_suffix_to_name(acid_name, 1, cation_class='acylium') or ''


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
    return Orthonym(style=style, _disable_opsin_validity_gate=True,
                     **_best_effort_reenter_kwargs()).name(neutral_smi)


def _best_effort_reenter_kwargs() -> dict:
    """v33 breadth: when the OUTER call is best-effort, name the neutral parent
    best-effort too, so a charged molecule whose neutral parent is nameable ONLY
    under the general/best-effort tier (a complex carboxylate/ammonium/phosphate,
    ~40% of the abstention census) converts instead of abstaining -- the charge is
    incidental, not the blocker.  Reads ``best_effort_ctx`` so the PIN/default tier
    (ctx False) re-enters PIN-only exactly as before (gate byte-identical); the
    best-effort name still faces the caller's E1/SELF-01 certification, so 0-wrong
    holds by construction."""
    try:
        from ..metrics.provenance import best_effort_ctx
        if best_effort_ctx.get():
            return dict(general_fallback=True, general_fallback_unverified=True,
                        allow_aromatic_general=True)
    except Exception:
        pass
    return {}


def _reenter_forced(neutral_smi: str, style: str, principal_fg: str) -> str:
    """T3 (Phase 173.6): re-enter the neutral skeleton with the anion's acid group
    FORCED as the principal characteristic group (P-72/P-74). Used only when the
    default re-entry let a senior neutral acid (carboxylic) take the principal slot,
    so the S/P-oxoacid suffix never appeared and the ionize step found no match."""
    from ..namer import Orthonym
    return Orthonym(style=style, _disable_opsin_validity_gate=True,
                     _principal_group_override=principal_fg,
                     **_best_effort_reenter_kwargs()).name(neutral_smi)


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


import threading as _threading

_route_reentry = _threading.local()
# Legitimate route_charged nesting (a zwitterion -> its anion parent -> the
# neutral re-entry) is <= 2 levels deep. Deeper recursion means the re-entered
# form re-triggers charged/radical routing WITHOUT converging -- e.g. a radical
# metal atom whose neutralization cannot remove the radical, so the re-entry is
# still a radical: route_charged -> name_radical -> _reenter -> _handle_radical
# -> name_radical -> route_charged -> ... forever (the `[99Tc]` 14.5h hang).
# Bound it: beyond this depth route_charged bails to '' (legacy fallthrough)
# instead of hanging. Thread-local so the benchmark's worker threads stay
# independent.
_MAX_ROUTE_DEPTH = 3


def _reentry_guarded(fn):
    """Bound route_charged recursion depth (anti-hang invariant: a species must
    never loop through the chokepoint)."""
    def _wrapped(mol, style: str = 'pin') -> str:
        depth = getattr(_route_reentry, 'depth', 0)
        if depth >= _MAX_ROUTE_DEPTH:
            return ''
        _route_reentry.depth = depth + 1
        try:
            return fn(mol, style)
        finally:
            _route_reentry.depth = depth
    return _wrapped


def _aminium_or_azaniumyl(neutral_name: str, n_cation_sites: int) -> str:
    """Protonated-amine cation: '-aminium' suffix vs 'azaniumyl' prefix.

    P-73.1.2.1: when the amine IS the principal characteristic group the parent
    name ends in '-amine' (or is a bare hydride / 'ammonia'); the cation is the
    '-aminium' suffix -- the proven ``name_aminium_cation`` transform
    (cysteamine -> cysteaminium).

    P-73.1.1.1 / P-74: when a SENIOR characteristic group owns the suffix
    (-oic acid / -ol / -one / -amide ...), the neutral pipeline already expresses
    the amine as an 'amino' substituent PREFIX, with its locant, N-substituents
    and enclosing marks placed correctly. The protonated nitrogen is then the
    cationic substituent prefix 'azaniumyl' (azanium = NH4+, P-73.1.1.1;
    'azaniumyl' = the N-attached cation, the OPSIN-parseable form of the
    -aminiumyl PIN). The correct cation name therefore UPGRADES that prefix in
    place, KEEPING the senior suffix: '2-aminooctanoic acid' ->
    '2-azaniumyloctanoic acid'.

    This REPLACES the broken ``name + 'ium'`` fallback (ions.name_aminium_cation)
    which appended 'ium' to the whole senior-group name ('...octanoic acidium').

    Guarded so the neutral name's locant / N-substituents carry over
    unambiguously: exactly ONE cationic site and exactly ONE 'amino' prefix
    token. Any other shape (multi-amine, ring-N protonation with no 'amino'
    prefix, ambiguous) falls back to the existing suffix transform --
    byte-identical, no regression.
    """
    from .ions import name_aminium_cation
    low = neutral_name.strip().lower()
    is_principal = (low == 'ammonia' or low.endswith('amine')
                    or low.endswith('amin') or low.endswith('ane'))
    if (not is_principal) and n_cation_sites == 1 and low.count('amino') == 1:
        return neutral_name.replace('amino', 'azaniumyl', 1)
    return name_aminium_cation(neutral_name) or ''


def _quaternary_rt_ok(name: str, mol) -> bool:
    """Mono-cation OPSIN round-trip backstop for the quaternary-aminium name.

    Phase 184 WS-E.1 (183 WR-01 precedent): parse the generated ``name`` back
    through OPSIN and confirm it reconstructs the SAME structure as ``mol`` (strict
    RDKit-canonical identity). A malformed quaternary name therefore fails CLOSED
    (the caller returns '') rather than shipping a structurally-wrong name.

    Fails OPEN (returns True) when the OPSIN jar is absent / the subprocess could
    NOT run (timeout / OSError), mirroring the existing fail-open-on-jar-missing
    convention so CI without OPSIN does not block. Uses the shared 10s-timeout
    ``OpsinOracle._invoke_opsin`` (NO ``-r`` flag — that radical-allowing change is
    Plan 04, not this gate). This is the targeted WR-01 backstop for the rare
    quaternary class, NOT a new gate for all rows.
    """
    if not name:
        return False
    try:
        from ..assembly.retained_substitution import OpsinOracle
        # Resolve the OPSIN jar the same way the namer does (Phase 168/169
        # precedent); if it cannot be found the oracle's _jar stays None and
        # _invoke_opsin raises -> the outer except fails OPEN (jar-missing).
        _jar = None
        try:
            import sys
            from pathlib import Path
            _scripts = str(Path(__file__).resolve().parent.parent.parent / "scripts")
            if _scripts not in sys.path:
                sys.path.insert(0, _scripts)
            from validate_retained_names import find_opsin_jar
            _jar = find_opsin_jar() or None
        except ImportError:
            _jar = None
        oracle = OpsinOracle(opsin_jar=_jar)
        if oracle._jar is None:
            return True  # jar missing -> fail OPEN (CI without OPSIN must not block)
        opsin_smi, ran = oracle._invoke_opsin(name)
    except Exception:
        # Oracle construction / invocation environment failure -> fail OPEN.
        return True
    if not ran:
        # Subprocess could not run (timeout / OSError / jar missing) -> fail OPEN.
        return True
    if not opsin_smi:
        # OPSIN ran and DEFINITIVELY rejected the name -> fail CLOSED.
        return False
    try:
        opsin_canon = Chem.CanonSmiles(opsin_smi)
        mol_canon = Chem.MolToSmiles(mol, canonical=True)
        return opsin_canon == mol_canon
    except Exception:
        # Could not canonicalize the OPSIN structure -> fail CLOSED (the name did
        # not parse back to a comparable structure).
        return False


@_reentry_guarded
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

    # --- Anti-hang complexity bound (169.6 follow-on; see
    # _MAX_CHARGED_ROUTE_HEAVY_ATOMS). The full-pipeline re-entry below can blow up
    # combinatorially on a pathological large charged molecule. Degrade GRACEFULLY
    # rather than hang. Regression-safe: largest RT-ing charged baseline cpd = 46 HA.
    if mol.GetNumHeavyAtoms() > _MAX_CHARGED_ROUTE_HEAVY_ATOMS:
        return ''

    # --- Step 2b (BBR-CHG-169.6-caveats, Phase 169.7): retained-name-first ONLY for
    # the CATEGORY of retained charged species that LACK a valid systematic PIN — i.e.
    # whose neutralize -> re-name -> re-apply-suffix path produces an OPSIN-unparseable
    # systematic form that the SUB-03 gate then suppresses to 'unknown' (the documented
    # 169.6 regression, audit Dim-08 §B Cause 1; sulfonium P-73.1.1.1, aminoxide/
    # bistriflimide P-72/P-74). This is DELIBERATELY NARROW: species WITH a valid
    # systematic PIN that the chokepoint already produces — alkoxides (-> -olate, the
    # SUB-01 systematic) and carboxylate (poly)anions (-> deferred -ate) — are NOT here
    # and keep their 169.6 chokepoint/defer routing (GUARD 1/2). RT-safe by construction.
    _canon = Chem.MolToSmiles(mol)
    if _canon in _RETAINED_FIRST_CHARGED:
        from ..data.ion_retained_names import get_cation_name, get_anion_name
        _retained = get_cation_name(_canon) or get_anion_name(_canon)
        if _retained:
            return _retained

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

    # W4-I3 (P-73.2.3.4): R-S+ / R-Se+ sulfanylium/selanylium. RDKit assigns a
    # 1-coordinate chalcogen cation SPURIOUS radical electrons (a valence-model
    # artifact — chemically it is a closed-shell cation, BB 41565 phenylsulfanylium
    # PIN), so it would trip the radical-ion bail below. Intercept it here as the
    # genuine cation it is. emit_chalcogen_ylium is tightly gated (S/Se, +1, 0 H,
    # one carbon substituent), so a REAL radical / other onium never matches (-> '').
    if n_cations == 1 and not n_anions:
        cc = classify_cation(mol, sites['cations'][0])
        if cc == 'onium':
            from .ions import emit_chalcogen_ylium
            cy = emit_chalcogen_ylium(mol, sites['cations'][0]['atom_idx'])
            if cy:
                return cy

    # Phase 3B (SPY  Q4): the SAME
    # RDKit valence-model artifact documented above for R-S+/R-Se+ also hits
    # the charge-shifted diazonium TWIN (R-N=N+). Its terminal N+ is degree-1
    # with only a DOUBLE bond (valence contribution 2) against the
    # +1-charged N's required valence of 4, so RDKit fills the shortfall
    # with 2 SPURIOUS radical electrons -- the canonical drawing's proximal
    # N+ (triple bond + single bond to the parent = valence 4, exact) gets 0.
    # Chemically both drawings are the SAME closed-shell diazonium cation
    # (BB P-73.2.2.3). Treat this one spurious site as not a radical at all,
    # so the ordinary single-cation path below (unchanged; still runs
    # _apply_guard3_reorder and every other guard) handles both drawings
    # exactly alike via the existing 'diazonium' -> _name_diazonium branch.
    #
    # Defense-in-depth (fable review, 2026-08-15): the condition MUST match the
    # exact artifact signature, not just "1 radical site == 1 cation site,
    # classified diazonium" -- that looser test also matches a GENUINE
    # open-shell monoradical cation such as ``c1ccccc1[N+]=N`` (1 radical
    # electron on a degree-2 N, classify_cation still says 'diazonium' since
    # that classifier is bond-order-only). Requiring exactly 2 radical
    # electrons on a DEGREE-1 atom whose SOLE bond is a DOUBLE bond pins this
    # to the valence-shortfall artifact only -- mirrors the rigor of the R-S+/
    # R-Se+ carve-out above (which gates on element/charge/H-count/substituent
    # count, not just "1 radical + 1 cation, same atom"). A genuine radical
    # cation now falls through to the ordinary "radical ion -> bail" guard
    # below instead of building a candidate SELF-01 has to catch.
    if (len(radical_sites) == 1 and n_cations == 1 and not n_anions
            and radical_sites[0]['atom_idx'] == sites['cations'][0]['atom_idx']
            and radical_sites[0]['n_electrons'] == 2
            and classify_cation(mol, sites['cations'][0]) == 'diazonium'):
        _rad_atom = mol.GetAtomWithIdx(radical_sites[0]['atom_idx'])
        _rad_nbrs = _rad_atom.GetNeighbors()
        if (_rad_atom.GetDegree() == 1 and len(_rad_nbrs) == 1
                and mol.GetBondBetweenAtoms(
                    _rad_atom.GetIdx(), _rad_nbrs[0].GetIdx()
                ).GetBondType() == Chem.BondType.DOUBLE):
            radical_sites = []

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
    anion_override_fg: Optional[str] = None  # T3: forced principal FG for the retry

    if radical_sites:
        # HETEROATOM-CENTRED radicals (P-71.2.1.2 / P-71.2.2.2 parent-hydride
        # radicals azanyl/sulfanyl/boranyl/azanylidene; P-71.3.2 amine/imine/amide
        # compound suffixes methanaminyl/propan-1-iminyl/formamidyl; P-71.3.3
        # multiplicative (ethane-1,2-diyl)bis(aminyl)) are named FIRST, with the
        # correct element-keyed contraction — the carbon chokepoint below drops the
        # whole 'ane' and would emit the wrong 'azyl'/'sulfyl'/'boryl'. The namer
        # fail-closes ('') for carbon-centred / out-of-scope radicals so the carbon
        # path runs unchanged (W4-I1).
        from .radicals import name_heteroatom_radical
        _het = name_heteroatom_radical(mol, radical_sites, style)
        if _het:
            return _het
        # Radicals funnel through the chokepoint too (kill radicals.py alkyl
        # carbon counting): neutralize -> re-enter -> append the P-71 -yl/
        # -ylidene/-ylidyne suffix (applied AFTER re-entry below). SCOPE: ONLY a
        # carbon-centered ALKYL/-ylidene/-ylidyne radical. The acyl (R-C(=O).),
        # oxyl (R-O.), and aryl subtypes are NOT a plain parent-hydride hydrogen
        # loss (acyl -> -oyl on the acid name; oxyl -> -oxyl; aryl -> the ring
        # radical) and keep their structured radicals.py helpers -> bail here so
        # name_radical / _handle_radical fall through to them.
        if len(radical_sites) >= 2:
            # P-71.2.3 multi-site free valences on ONE acyclic all-carbon parent.
            # Scope: every site element C, n_electrons<=3, none acyl/oxyl/aryl/
            # aminyl/thiyl (those are not plain parent-hydride H-loss). The namer
            # fail-closes ('') on anything else (hetero/ring/off-parent/mixed).
            from .radicals import classify_radical
            if any(s['element'] != 'C' or s['n_electrons'] > 3 for s in radical_sites):
                return ''
            if any(classify_radical(mol, s)['subtype'] in
                   ('acyl', 'oxyl', 'aryl', 'aminyl', 'thiyl') for s in radical_sites):
                return ''
            from .ions import emit_parent_hydride_polyvalent_suffixes
            centers = [(s['atom_idx'], s['n_electrons']) for s in radical_sites]
            return emit_parent_hydride_polyvalent_suffixes(mol, centers) or ''
        from .radicals import classify_radical
        rinfo = classify_radical(mol, radical_sites[0])
        if rinfo['subtype'] in ('acyl', 'oxyl', 'aryl', 'aminyl', 'thiyl', 'benzylic'):
            return ''  # structured helpers / out of scope -> legacy fallthrough
        # 'benzylic': bail so _handle_radical falls to name_radical, whose
        # (now-canonical) RETAINED_RADICALS lookup ships the retained PIN
        # 'benzyl' (P-57.1.2) instead of the non-PIN 'toluenyl' fallback.
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
        # W4-I3 (P-73.5): MULTI-cation handling. The ONLY in-scope multi-cation
        # construction here is a homogeneous poly-AMINIUM -> the bis/tris(aminium)
        # compound-suffix form (mirror of the W4-I2 poly-anion bis(aminide) path):
        # [NH3+]CC[NH3+] -> ethane-1,2-bis(aminium) (BB 42340). Neutralize ALL
        # centres -> re-enter -> map the neutral di/tri-amine suffix to
        # 'bis(aminium)' (NOT 'diaminium', P-70.3.2). EVERY other multi-cation
        # shape (poly-onium on a phenylene needs a MULTIPLICATIVE engine; poly-
        # quaternary; mixed) FAILS CLOSED here so the single-cation generic seam
        # below never emits a wrong '…diaminium' — the caller's neutralize-recurse
        # then names the neutral form (suppressed by the SELF gate) rather than
        # shipping a wrong charged name.
        if len(sites['cations']) >= 2:
            if ccls == 'aminium':
                from .ions import _poly_to_bis_cation_suffix
                _neutral = _neutralize_fragment(mol)
                _nn = ''
                if _neutral:
                    try:
                        _nn = _reenter(_neutral, style)
                    except (RecursionError, ValueError, RuntimeError):
                        _nn = ''
                bis = _poly_to_bis_cation_suffix(_nn) if _nn else ''
                if bis:
                    return bis
            # v33 Phase 3 (P-73.5.1.1/.2): a SYMMETRIC bis-quaternary-ammonium
            # dication joined by a straight, saturated, unbranched all-carbon
            # bridge -> the multiplicative '{bridge-diyl}bis({onium unit})'
            # assembly, e.g. hexamethonium
            # C[N+](C)(C)CCCCCC[N+](C)(C)C -> hexane-1,6-diylbis(trimethyl-
            # azanium). emit_bis_quaternary_ammonium fails closed ('') on
            # anything asymmetric / ring-borne / branched-bridge, falling
            # through unchanged to the legacy path below.
            if ccls == 'quaternary' and len(sites['cations']) == 2:
                from .ions import emit_bis_quaternary_ammonium
                biq = emit_bis_quaternary_ammonium(mol, sites['cations'])
                if biq:
                    return biq
            return ''
        # W4-I3 (P-73.2.2.3): a diazonium cation R-N2+ is named by appending
        # 'diazonium' to the parent hydride obtained by SEVERING the whole -N#N+
        # group and capping the attachment with H ([N+](#N)c1ccccc1 -> benzene ->
        # benzenediazonium). The generic neutralize (drop-charge) over-valences the
        # surviving N, so a dedicated sever-and-name emitter owns it. Decline ('')
        # -> fall through.
        if ccls == 'diazonium':
            dz = _name_diazonium(mol, sites['cations'][0]['atom_idx'], style)
            if dz:
                return dz
        # W8-P5 Task 1 (P-73.2.3.1): an acylium cation R-C(+)=O is named on the
        # RECONSTRUCTED ACID (add -OH), never the generic hydride-loss
        # ('add H' -> aldehyde) path below -- emit_acylium owns it. Mirrors the
        # diazonium interception immediately above.
        if ccls == 'acylium':
            acy = emit_acylium(mol, sites['cations'][0]['atom_idx'], style)
            if acy:
                return acy
        # (R-S+/R-Se+ sulfanylium is intercepted earlier, before the radical-ion
        # bail, because RDKit flags the 1-coordinate chalcogen cation as a radical.)
        # WS-E.1 (P-73.1.2.1 + Table 7.4): a QUATERNARY ammonium N (0 H, degree
        # >= 4) CANNOT take the _neutralize_fragment path — removing the lost
        # proton leaves an over-valent neutral N and SanitizeMol raises -> ''
        # (RESEARCH Pitfall 2) — and it is NOT an azaniumyl-prefix case (D-05;
        # azaniumyl is zwitterion-only, P-74.1.3). Name it directly via the
        # demote-N -> find_principal_chain -> '-aminium' emitter on the ORIGINAL
        # mol. C[N+](C)(C)C -> N,N,N-trimethylmethanaminium.
        if ccls == 'quaternary' and len(sites['cations']) == 1:
            cat_idx = sites['cations'][0]['atom_idx']
            # W4-I3 (P-73.1.1.2 / P-73.4): a quaternary RING N+ is named on the
            # ring parent with the -ium suffix and the exocyclic substituents at
            # the ring-N locant (C[N+]1(C)CCCCC1 -> 1,1-dimethylpiperidin-1-ium,
            # C[N+]1(C)CCOCC1 -> 4,4-dimethylmorpholin-4-ium), NOT demoted to an
            # acyclic amine chain (name_quaternary_aminium would linearize the
            # ring). The ring emitter's DEMOTE branch severs the substituents,
            # names the bare ring, and re-cites them at the centre locant. On
            # decline (fused/multi-ring) it returns '' -> fall through to the
            # acyclic quaternary-aminium emitter below.
            if mol.GetAtomWithIdx(cat_idx).IsInRing():
                from .ions import emit_parent_hydride_cumulative_suffix
                ring_ium = emit_parent_hydride_cumulative_suffix(mol, cat_idx, 'ium')
                if ring_ium:
                    return ring_ium
            from .ions import name_quaternary_aminium
            result = name_quaternary_aminium(mol, sites['cations'][0])
            if not result:
                return ''   # emitter declined (out of scope) -> legacy fallthrough
            # Mono-cation OPSIN RT-gate backstop (183 WR-01 precedent): a malformed
            # quaternary name fails CLOSED rather than shipping garbage. Fails OPEN
            # only when the OPSIN jar is absent (CI without OPSIN must not block).
            if not _quaternary_rt_ok(result, mol):
                return ''
            return result
        # F-T6 (DD3, P-73.1.1.2): a protonated / N-substituted RING-N cation is
        # named by the ring-aware cumulative-suffix emitter (ring numbering +
        # 'e' elision + cationic-centre locant), NOT the acyclic amine->aminium
        # textual transform (which produced 'pyridineium'/'morpholineium' and,
        # for the 0-H N-substituted aromatic case, an over-valent neutralize that
        # crashed to ''). The emitter owns both the in-place (protonated) and the
        # demote (N-substituted aromatic) sub-cases. On decline it returns ''
        # and we fall through to the existing acyclic-amine path (byte-identical
        # to HEAD for the ring shapes the emitter cannot number).
        if ccls == 'aminium' and len(sites['cations']) == 1:
            cat_idx = sites['cations'][0]['atom_idx']
            if mol.GetAtomWithIdx(cat_idx).IsInRing():
                from .ions import emit_parent_hydride_cumulative_suffix
                ring_ium = emit_parent_hydride_cumulative_suffix(
                    mol, cat_idx, 'ium')
                if ring_ium:
                    return ring_ium
            # W4-I5 (P-77.1.2 / P-73.1.2.1): a mono-protonated di-/polyAMINE — the
            # protonated N is the -aminium principal group, the OTHER neutral amine(s)
            # become `amino` PREFIXES (BB 43568 `2-aminoethan-1-aminium`), NOT both as
            # `diaminium`. The emitter self-declines ('') with no sibling amine (the
            # ordinary single-amine case) or an out-of-scope skeleton, preserving the
            # legacy `_aminium_or_azaniumyl` path below (including the senior-group
            # azaniumyl case, whose senior group trips the emitter's scope gate).
            else:
                from .ions import emit_mono_ionized_polyfunctional
                mono = emit_mono_ionized_polyfunctional(mol, cat_idx, 'aminium')
                if mono:
                    return mono
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
        # WS-E.2 (P-72.2.2.1 / Table 3.4): a CARBANION has no neutral FG anchor
        # (it neutralizes to a bare hydride 'hexane' — apply_ion_suffix_to_name finds
        # no carbanion key and returns ''), so it cannot use the
        # _principal_group_override seam. Name it directly via the index-preserving
        # emitter on the ORIGINAL mol: the -ide centre gets the lowest locant
        # (orient_chain), competing with unsaturation/substituents per P-31.1.4.
        # CCC[CH-]CC -> hexan-3-ide. The centre index comes straight off
        # sites['anions'][0]['atom_idx'] (same index space as mol) — do NOT
        # canonicalize/neutralize before reading it (Pitfall 1).
        _acls = {classify_anion(mol, a) for a in sites['anions']}
        if _acls == {'carbanion'} and len(sites['anions']) == 1:
            from .ions import emit_parent_hydride_cumulative_suffix
            center_idx = sites['anions'][0]['atom_idx']
            # The emitter now dispatches a RING carbanion through its ring branch
            # ([CH-]1CCCCC1 -> cyclohexan-1-ide), so ring carbanions are named
            # here too instead of dropping their charge (DD3 Defect C).
            carbanion_name = emit_parent_hydride_cumulative_suffix(mol, center_idx, 'ide')
            if carbanion_name:
                return carbanion_name
            return ''   # primitive declined (out of scope) -> legacy fallthrough
        # W4-I2 (P-72.2.2.1): a MULTI-carbanion on one acyclic all-carbon parent
        # hydride -> the '-di/tri-ide' PIN ([C-]#[C-] -> ethynediide, BB 40918).
        # HEAD dropped the charge (-> the retained general 'acetylide'/'acetylene').
        if _acls == {'carbanion'} and len(sites['anions']) >= 2:
            from .ions import emit_poly_carbanion_ide
            poly = emit_poly_carbanion_ide(
                mol, [a['atom_idx'] for a in sites['anions']])
            if poly:
                return poly
            return ''   # emitter declined (out of scope) -> legacy fallthrough
        # F-T6 (DD3, P-72.2.2.1): a skeletal Group-14/15 heteroatom anion
        # (P/As/Sb/Si/Ge) has no neutral FG anchor and no -ol/-thiol suffix the
        # generic seam could catch, so neutralize->re-enter->suffix-map currently
        # DROPS the charge (C[P-]C -> dimethylphosphane). Route it through the
        # same index-preserving emitter (heteroatom branch): the parent hydride is
        # named and the -anide ending added. C[P-]C -> dimethylphosphanide,
        # C[Si-](C)C -> trimethylsilanide.
        if _acls == {'heteroatom_hydride_anion'} and len(sites['anions']) == 1:
            from .ions import emit_parent_hydride_cumulative_suffix
            center_idx = sites['anions'][0]['atom_idx']
            het_name = emit_parent_hydride_cumulative_suffix(mol, center_idx, 'ide')
            if het_name:
                return het_name
            return ''   # primitive declined (un-nameable heterane) -> legacy
        # P-72.3 / P-72.8: a -uide (hydride-addition) anion — the ate-complex
        # (B(CH3)4- / CH3-SiH4- / (CH3)4P- / (C6H5)2I-). Named by the
        # substituted-'-uide'-parent emitter (cannot neutralize: the hypervalent
        # neutral hydride is invalid). W4-I2: generalized beyond Group 13.
        if _acls == {'uide_anion'} and len(sites['anions']) == 1:
            from .ions import _emit_group13_uide
            uide = _emit_group13_uide(mol, sites['anions'][0]['atom_idx'])
            if uide:
                return uide
            return ''   # emitter declined -> legacy
        # W4-I2 (P-72.2.2.2.2 / P-72.2.2.2.3): a HOMOGENEOUS poly compound-suffix
        # anion (a -1 on each of >=2 O/S/N atoms of the SAME class) uses the
        # olate/thiolate/aminide compound suffix multiplied by 'bis'/'tris' ('bis'
        # NOT 'di', P-70.3.2, "to avoid ambiguity"): [NH-]CC[NH-] -> ethane-1,2-
        # bis(aminide) (BB 41059), [O-]CC[O-] -> ethane-1,2-bis(olate) (BB 41267),
        # [S-]CC[S-] -> ethane-1,2-bis(thiolate), catechol dianion -> benzene-1,2-
        # bis(olate) (BB 28192). HEAD's generic seam produced the WRONG di- forms
        # (ethane-1,2-diaminide / -diolate / -dithiolate). Neutralize ALL centres ->
        # re-enter -> map the '<di/tri>{suffix}' poly name to bis(<suffix>ate/ide).
        # Fails through when the neutral is NOT a suffix-form poly name (a retained
        # 'hydroquinone', or the substituent-form 'bis(sulfanyl)benzene').
        if (_acls <= {'aminide', 'alkoxide', 'phenolate', 'thiolate'}
                and len(_acls) == 1 and len(sites['anions']) >= 2):
            from .ions import _poly_to_bis_compound_suffix
            _neutral = _neutralize_fragment(mol)
            if _neutral:
                try:
                    _nn = _reenter(_neutral, style)
                except (RecursionError, ValueError, RuntimeError):
                    _nn = ''
                bis = _poly_to_bis_compound_suffix(_nn) if _nn else ''
                if bis:
                    return bis
            return ''   # not a clean suffix-form poly -> legacy fallthrough
        # W4-I5 (P-77.2.2 / P-72.2.2.2.2): a SINGLE alkoxide/thiolate anion on a chain
        # that ALSO carries neutral sibling -OH / -SH groups is a mono-deprotonated
        # POLYOL / polythiol: the -O(-)/-S(-) is the -olate/-thiolate principal group,
        # the sibling(s) become hydroxy / sulfanyl PREFIXES (BB 43605
        # `2-hydroxyethan-1-olate`). The emitter self-declines ('') when there is no
        # sibling (the ordinary single-alkoxide case) or the skeleton is out of scope,
        # so the byte-identical legacy path below is preserved.
        _single = {classify_anion(mol, a) for a in sites['anions']}
        if _single == {'alkoxide'} and len(sites['anions']) == 1:
            from .ions import emit_mono_ionized_polyfunctional
            mono = emit_mono_ionized_polyfunctional(
                mol, sites['anions'][0]['atom_idx'], 'olate')
            if mono:
                return mono
        elif _single == {'thiolate'} and len(sites['anions']) == 1:
            from .ions import emit_mono_ionized_polyfunctional
            mono = emit_mono_ionized_polyfunctional(
                mol, sites['anions'][0]['atom_idx'], 'thiolate')
            if mono:
                return mono
        cation_class, allowed_suffixes = _classify_single_anion(mol, sites)
        # WS-E.3 (D-11/D-12): charge-first PCG on the ORIGINAL (un-neutralized) mol.
        # The actually-ionized senior acid class anchors the name (P-72); a neutral
        # group of higher P-41 seniority is demoted to a prefix. Subsumes the old
        # per-site _ANION_PRINCIPAL_FG.get(...) lookup (same dict + seniority order).
        # The carbanion short-circuit above already returned; classify_charged_pcg
        # returns None for carbanion/alkoxide-only anyway (== the old .get() == None),
        # so the 173.6 sulfonate/sulfinate/phosphonate behavior is byte-identical.
        anion_override_fg = classify_charged_pcg(mol, sites)
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
        _radical_center_idx = radical_sites[0]['atom_idx']
        # P-29.2 / Table 3.4: radicals are named as SUBSTITUENT GROUPS. A SIMPLE
        # unbranched terminal radical keeps the contracted retained form WITHOUT a
        # locant — the Blue Book lists 'CH3-CH2• ethyl (PIN)' (BlueBookV2.md:17597
        # Table 3.4 example), not 'ethan-1-yl'; likewise 'methyl'/'propyl'. So the
        # proven P-71.1.1 textual contraction (_apply_radical_suffix) IS the PIN for
        # those. It is ALSO the correct path for the single-carbon -ylidene/-ylidyne
        # (methylidene/methylidyne) the primitive would over-spell 'methanylidene'.
        if _is_simple_terminal_radical(mol, _radical_center_idx):
            return _apply_radical_suffix(neutral_name, radical_suffix)
        # NON-terminal / branched / unsaturated radical: the centre needs a
        # first-class locant (P-71 / P-31.1.4.3.4 lowest-locant for the free
        # valence, competing with unsaturation/substituents per P-31/P-14.4) —
        # exactly like the WS-E.2 carbanion -ide centre (BlueBookV2.md:17608
        # 'ethan-2-id-1-yl (PIN)'). Route through the 184-01 primitive so
        # CC[CH]CC -> 'pentan-3-yl', not the locant-less 'pentyl'.
        # CRITICAL (Pitfall 1): pass the ORIGINAL mol + the centre index in the
        # SAME index space — the primitive owns the atom-index-survival handling
        # (it H-saturates / re-numbers internally); do NOT canonicalize mol first.
        from .ions import emit_parent_hydride_cumulative_suffix
        emitted = emit_parent_hydride_cumulative_suffix(
            mol, _radical_center_idx, radical_suffix)
        if emitted:
            return emitted
        # Fail-closed fallback: the primitive declined (out of its parent-hydride
        # scope) -> keep the proven P-71.1.1 textual form rather than regress.
        return _apply_radical_suffix(neutral_name, radical_suffix)

    # (CARBOXYLATE anions are deferred to the proven path earlier; they never
    # reach this point — see the carboxylate guard in Step 3.)

    # AMINIUM (protonated amine, P-73.1.2.1) reuses the PROVEN name_aminium_cation
    # transform on the re-entered amine name (amine->aminium, ammonia->ammonium).
    # This is the existing _name_aminium_systematic primary path, generalized into
    # the funnel — so deleting the stub's carbon-counting FALLBACK (Task 2) keeps
    # the aminium output byte-identical (methylaminium / pyrrolidineium etc.).
    if cation_kind == 'aminium':
        # T1 (Phase 173.6): a senior-group protonated amine becomes an 'azaniumyl'
        # substituent prefix (P-73.1.1/P-74), not 'ium' appended to the parent;
        # a principal amine keeps the proven '-aminium' suffix.
        return _aminium_or_azaniumyl(neutral_name, len(sites['cations'])) or ''

    ionized = apply_ion_suffix_to_name(
        neutral_name, total_charge,
        allowed_suffixes=allowed_suffixes,
        cation_class=cation_class,
    )
    if ionized:
        return ionized

    # T3 (Phase 173.6): the first re-entry let a SENIOR neutral acid (carboxylic,
    # P-41) take the principal slot, so the anion's S/P-oxoacid suffix never appeared
    # and the ionize match failed. Per P-72/P-74 the CHARGED group IS the principal
    # characteristic group of an anion -> re-enter with the anion's acid FG forced as
    # principal (carboxylic acid demoted to a 'carboxy' prefix) and ionize that:
    # O=C(O)CCS(=O)(=O)[O-] -> '2-carboxyethanesulfonic acid' -> '2-carboxyethanesulfonate'.
    if anion_override_fg is not None:
        try:
            forced = _reenter_forced(neutral_smi, style, anion_override_fg)
        except (RecursionError, ValueError, RuntimeError):
            forced = ''
        if forced and not _is_malformed_parent(forced):
            return apply_ion_suffix_to_name(
                forced, total_charge,
                allowed_suffixes=allowed_suffixes,
                cation_class=cation_class,
            )
    return ''


def _is_simple_terminal_radical(mol, center_idx: int) -> bool:
    """P-29.2 / P-71: TRUE iff the radical is a SIMPLE substituent-group shape
    whose PIN is the contracted, locant-less ``-yl``/``-ylidene``/``-ylidyne`` form
    (``methyl``/``ethyl``/``propyl``/``methylidene``), so it must keep the proven
    ``_apply_radical_suffix`` textual contraction rather than the locant primitive.

    The contracted form is correct ONLY for an UNBRANCHED, ACYCLIC, SATURATED,
    ALL-CARBON chain whose free valence sits at a TERMINAL carbon (or a single
    carbon). In that case the free-valence locant is 1 on the unique longest chain
    and is omitted (Blue Book Table 3.4 lists ``CH3-CH2• ethyl (PIN)``).

    A BRANCHED terminal radical (e.g. isobutyl ``(CH3)2CH-CH2•`` -> the PIN
    ``2-methylpropan-1-yl`` carries the locant), a NON-terminal radical
    (``CC[CH]CC`` -> ``pentan-3-yl``), or any unsaturated/heteroatom/ring shape
    returns FALSE -> the locant primitive owns it.
    """
    try:
        center = mol.GetAtomWithIdx(center_idx)
    except (RuntimeError, IndexError, OverflowError):
        return False
    if center.GetSymbol() != 'C':
        return False
    # Whole-molecule must be an unbranched acyclic SATURATED all-carbon chain:
    #   - no rings, no heteroatoms, no multiple bonds;
    #   - every carbon has at most 2 carbon neighbours (a straight chain);
    #   - the radical centre is a terminus (<= 1 carbon neighbour) or a lone C.
    if mol.GetRingInfo().NumRings() > 0:
        return False
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != 'C':
            return False
        if sum(1 for nbr in atom.GetNeighbors() if nbr.GetSymbol() == 'C') > 2:
            return False  # a branch point -> needs the locant form
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE:
            return False  # unsaturation -> the centre competes for the locant
    n_carbon_nbrs = sum(1 for nbr in center.GetNeighbors() if nbr.GetSymbol() == 'C')
    return n_carbon_nbrs <= 1


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
