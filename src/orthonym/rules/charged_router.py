"""The single mandatory parent-selection chokepoint for charged species.

a phase Plan 03 (CHOKE-01, CHOKE-02). ROOT CAUSE of the flat round-trip
wall (V20 audit): every charged class (alkoxide / phenolate / carbanion /
thiolate / carbenium / onium / diazonium / aminium-fallback / radical) decided
its own parent through a *parallel carbon-counting stub* that ignores
connectivity, substituents and unsaturation — the literal ``heptanolate`` bug
(``CCCCCCC[O-]`` -> ``heptanolate``, dropping the locant and every substituent).
Those stubs BYPASS the sound ``select_parent`` cascade for 91% of wrong-parent
failures.

This module makes the chokepoint STRUCTURAL: ``route_charged(mol, style)`` is the
ONE funnel every charged dispatch handler delegates to. It GENERALIZES the proven
``_name_oxoacid_anion`` template (``ions.py:911``, the 169.5 fix that
round-trips):

    neutralize the chosen fragment -> re-enter the FULL pipeline
    ``Orthonym(style, _disable_opsin_validity_gate=True).name(neutral_smi)``
    -> re-apply the class-correct ionic suffix via ``apply_ion_suffix_to_name``.

A correct parent auto-corrects the locants (95% co-occurrence) on the same
molecule — this is the multi-defect-collapsing fix, not a per-class patch.

The four IUPAC-2013 guards (SYNTHESIS-authoritative-cascade.md; /
 /, applied IN ORDER:

  GUARD 1 FG-class-before-suffix (the ``heptanolate`` fix). A ``-S(=O)2-O-`` is
           an acid anion -> ``-sulfonate``, NOT a hydroxy anion
           -> ``-olate``. ``classify_anion`` picks the FG class;
           the per-class ``allowed_suffixes`` subset gates the textual seam so a
           sulfonate stem can NEVER mis-fire to ``-olate``.
  GUARD 2 ionic-center-count-first a-c / a-b). On a multi-center
           ion the parent maximizes anionic / ``ide`` / ``uide`` center count
           BEFORE length is considered (dicarboxylate dianions). Realized
           by neutralizing ALL same-sign centers and letting the re-entered
           pipeline name the multi-suffix parent (``butanedioate``), exactly as
           the proven anion seam already does.
  GUARD 3 skeletal-charge element seniority d / c):
           N > P > As > Sb > Bi > Si > Ge > Sn > Pb > B > Al > Ga > In > Tl > O
           > S > Se > Te > C. When the charge sits on a skeletal heteroatom the
           senior element bearing it is the parent, not the longest carbon chain.
  GUARD 4 zwitterion anion-is-parent override. The anion is FORCED as
           the parent; a separable cation, the betaine quaternary
           ammonium) is demoted to a structured ``(…azaniumyl)`` substituent
           prefix (``substituent_naming.cation_to_prefix``); a skeletal cation
           , a ring N+ of pyridinium-2-carboxylate) is deferred to the
           legacy path (the cumulative ium+ate suffix is out of scope this
           plan). 169.6-04 (was a Plan-03 detect-and-defer seam).

Returns '' on any failure / out-of-scope shape (metal complex, multi-fragment
salt, zwitterion, malformed re-entered parent) so the caller falls through to the
existing retained/legacy cascade — preserving the byte-identical no-crash
contract.

NO carbon-counting. NO ``.replace``. NO molecule-specific branch. The router is
GENERAL (fix-methodology.md).
"""

from typing import Optional

from rdkit import Chem

from ..perception.ions import _get_internal_charge_atoms, get_ion_sites
from .ions import (
    _has_metal,
    apply_ion_suffix_to_name,
    classify_anion,
    classify_cation,
)

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
# Independently corroborates AUTONOM's documented 44-atom hard limit. Raise it if
# the router is later shown to name larger charged molecules in bounded time.
# =============================================================================
_MAX_CHARGED_ROUTE_HEAVY_ATOMS = 50

# charged-species fix, 169.6 caveats (a phase): canonical SMILES of retained charged species
# that LACK a valid systematic PIN — their neutralize->re-name chokepoint path yields
# an OPSIN-unparseable form (the 169.6 'unknown'/wrong-retained regression). These use
# their sanctioned retained name //. NARROW by design: alkoxides and
# carboxylate (poly)anions have valid systematic PINs (-olate / deferred -ate)
# and are NOT here, so GUARD 1/2 routing is preserved. Add a species here ONLY if its
# chokepoint systematic form is genuinely OPSIN-unparseable (verify before adding).
_RETAINED_FIRST_CHARGED = frozenset({
    '[SH3+]',                                     # sulfonium
    'N[O-]',                                      # aminoxide
    '[NH-]O',                                     # hydroxyazanide, the Blue Book preselected)
    'O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F',     # bistriflimide
    # Wave2: the retained alkoxide names ARE the PINs
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
# (d) / (c) (SYNTHESIS §"TIER -1", verbatim order). Lower index =
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
# anion 'sulfonate'/'sulfinate'/'phosphonate' -> the matching oxoacid suffix
#: acid anions -> -ate/-ite).
# anion 'carboxylate' -> {'oic acid','carboxylic acid'}
# anion 'alkoxide'/'phenolate' -> {'ol'}: hydroxy anions -> -olate)
# anion 'thiolate' -> {'thiol'}
# anion 'carbanion' -> None -> bare -> -ide
# anion 'aminide' -> {'amine'}
# anion 'iminide' -> {'imine'}, the =N- imine anion)
# =============================================================================
_ANION_ALLOWED_SUFFIXES = {
    'sulfonate': frozenset({'sulfonic acid'}),
    'sulfinate': frozenset({'sulfinic acid'}),
    'phosphonate': frozenset({'phosphonic acid', 'phosphinic acid',
                              'phosphoric acid'}),
    'carboxylate': frozenset({'oic acid', 'carboxylic acid'}),
    #: the carbo(di)thioate acid anion (R-C(=S)-S- / R-C(=O)-S-). Its
    # neutral form is a (di)thioic acid ending '-oic acid' (propanedithioic acid),
    # so it takes the -oate seam -> propanedithioate. Restricting to {'oic acid'}
    # keeps GUARD-1 tight (a stem must genuinely end in the acid suffix).
    'carbodithioate': frozenset({'oic acid'}),
    'alkoxide': frozenset({'ol'}),
    'phenolate': frozenset({'ol'}),
    'thiolate': frozenset({'thiol'}),
    'aminide': frozenset({'amine'}),
    #: the imine anion (=N-) -> {'imine'} -> -iminide.
    'iminide': frozenset({'imine'}),
    # 'carbanion' -> None (bare -> -ide via the resolvers empty-suffix path).
}

# (a phase): anion class -> the FG name forced as principal when a SENIOR
# neutral acid (carboxylic) would otherwise hijack the re-entry's principal slot.
# Per / the CHARGED group is the principal characteristic group of an
# anion: O=C(O)CCS(=O)(=O)[O-] -> 2-carboxyethanesulfonate (carboxy PREFIX,
# sulfonate suffix), not the carboxylic-principal 'propanoate' the neutral
# seniority carboxylic > sulfonic) yields for the di-acid skeleton.
_ANION_PRINCIPAL_FG = {
    'carboxylate': 'carboxylic_acid',   #.3: carboxylate is the senior anion acid class
    'sulfonate': 'sulfonic_acid',       # 173.6 - byte-identical, do not change
    'sulfinate': 'sulfinic_acid',       # 173.6 - byte-identical, do not change
    'phosphonate': 'phosphonic_acid',   # 173.6 - byte-identical, do not change
}

# anion acid-class seniority (highest first) for the charge-first PCG pick.
# Declarative single source of truth (the "class-keyed transform table" pattern):
# the carboxylate > sulfonate > sulfinate > phosphonate order mirrors the
# anion-acid-class seniority used when several ionized acid centres coexist.
_ANION_PCG_SENIORITY = ('carboxylate', 'sulfonate', 'sulfinate', 'phosphonate')


def classify_charged_pcg(mol, sites) -> Optional[str]:
    """Charge-first PCG classifier (.3, /). Runs on the ORIGINAL
    (un-neutralized) mol. Returns the FG-name to FORCE as the principal
    characteristic group (a detect_functional_groups KEY: 'carboxylic_acid',
    'sulfonic_acid', 'sulfinic_acid', 'phosphonic_acid'), or None when the
    charged class has NO neutral FG anchor (carbanion/alkoxide/thiolate/
    aminide/phenolate -> handled by the.2 emit_parent_hydride_cumulative_
    suffix primitive / the existing suffix seam, NOT this override) or when a
    different route owns the molecule.

    Ordering (; anions / cations / zwitterion / radical):
      1. radical present -> None radical>anion>cation; the -yl primitive owns it)
      2. mixed-sign (zwitterion) -> None _route_zwitterion owns it)
      3. anion(s): senior ionized acid class per -> its acid-FG key
      4. cation(s) only -> None (the aminium / class-keyed cation transforms own it)
    The forced FG MUST be a features.functional_groups key (namer.py:1821) or it
    silently no-ops; for a class with no neutral FG anchor return None.
    the Blue Book (radical>anion>cation seniority)."""
    # Rule 1: a radical co-occurring outranks the ionic centre; the
    # -yl/-ylidene primitive owns it, not this override.
    from ..perception.ions import get_radical_sites
    if get_radical_sites(mol):
        return None
    # Rule 2: a mixed-sign zwitterion is owned by _route_zwitterion
    # (anion-is-parent + azaniumyl prefix), never the charge-first override here.
    if sites.get('cations') and sites.get('anions'):
        return None
    # Rule 3: pick the senior ionized acid class present among the anions.
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
# 'aminium' is a plain map swap (amine->aminium, gated to {'amine'}.
_CATION_SPEC = {
    'ylium': ('ylium', None),       #: ane->ylium (methane->methylium)
    'diazonium': ('diazonium', None),  #: append diazonium to the hydride
    'aminium': (None, frozenset({'amine'})),  #: amine->aminium (map swap)
    # 'onium' (oxonium/sulfonium/phosphonium on a heteroatom hydride) -> handled
    # via the generic map empty->'ium' default (no class key, no allowed subset).
}


# / 169.5 (reused verbatim): a re-entered parent that lost its
# chain/ring stem (the deferred fused-ring limitation) leaves a bare
# unsaturation marker glued onto a suffix stem or a locant ('anesulfonic acid',
# 'ene-1-...'). A descriptive fallback ('unknown...', 'not supported') is
# likewise not a valid parent. Refuse to propagate either; fall through instead.
import re as _re

_DEGENERATE_PARENT_RE = _re.compile(r'^(?:ane|ene|yne)(?:sulf|phosph|arso|boro|[0-9(-])')


def _is_malformed_parent(neutral_name: str) -> bool:
    """True if a re-entered neutral parent name must NOT be ionized (169.5)."""
    low = neutral_name.lstrip().lower()
    return bool(
        _DEGENERATE_PARENT_RE.match(low)
        or 'unknown' in low
        or 'not supported' in low
        or 'wildcard' in low
    )


def _is_zwitterion(sites) -> bool:
    """GUARD 4: a single fragment carrying BOTH a (non-internal)
    cationic and a (non-internal) anionic center is a zwitterion. Routed by
    ``_route_zwitterion`` (anion-is-parent override + the structured cation
    prefix producer); 169.6-04 (was a Plan-03 detect-and-defer seam).
    """
    return bool(sites.get('cations')) and bool(sites.get('anions'))


def _cation_is_skeletal_to_anion_parent(mol, cation_idx: int, anion_idx: int) -> bool:
    """ vs discriminator.

    : the cationic atom is INSIDE the parent hydride that bears the
    anionic characteristic group — i.e. it is a ring atom of the SAME ring
    system the anion is attached to (the ring N+ of a pyridinium-2-carboxylate).
    Such a cation is kept on the parent as an ``-ium`` suffix, NOT demoted to a
    prefix.

    : the cationic atom sits on a DIFFERENT parent (a quaternary
    ammonium hanging off the anion chain) -> the cation becomes a substituent
    prefix.

    Heuristic (structural, not per-molecule): the cation is skeletal iff it is a
    ring atom AND the anion's attachment carbon is in the SAME ring system
    (shares a ring with the cation). A non-ring (acyclic) quaternary ammonium is
    always (the betaine case).
    """
    cat = mol.GetAtomWithIdx(cation_idx)
    if not cat.IsInRing():
        return False  # acyclic cation -> separate parent betaine)
    ri = mol.GetRingInfo()
    # The anion's parent-attachment atom = the heavy neighbour of the anion atom
    # (e.g. the carboxylate carbon). If that atom shares a ring with the cation,
    # the cation is skeletal to the anion's parent ring system.
    an = mol.GetAtomWithIdx(anion_idx)
    anchor_atoms = [n.GetIdx() for n in an.GetNeighbors() if n.GetSymbol() != 'H']
    for ar in ri.AtomRings():
        if cation_idx in ar and any(a in ar for a in anchor_atoms):
            return True
    return False


def _sever_cation_build_anion_parent(mol, cation_idx: int,
                                     parent_attach_idx: int) -> str:
    """Build the NEUTRAL anion-parent SMILES for.

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


def _attachment_locant_on_polyacid_parent(mol, anion_idxs, parent_attach_idx):
    """Cation-substituent attachment locant on a MULTI-carboxylate anion parent
    (a phase, the ``_attachment_locant_on_anion_parent`` sibling for >= 2
    carboxylates).

    The cation was SEVERED before the parent was re-entered/renamed
    (``_sever_cation_build_anion_parent``), so the bare polyacid (e.g.
    pentanedioic acid) carries no other substituent and its two ends are
    numbering-equivalent — / lowest-locants then picks
    whichever end gives the SMALLER locant for the one substituent that will
    be cited (the azaniumyl prefix). Compute the candidate distance+1 from
    EVERY carboxyl-carbon anchor (every anion site) to ``parent_attach_idx``
    and return the minimum -- e.g. glutamate's alpha carbon is 4 bonds from
    one carboxyl and 2 from the other; the correct PIN locant is 2, matching
    the lower candidate. Returns None if no path exists to any anchor.
    """
    from rdkit.Chem import rdmolops
    best = None
    for anion_idx in anion_idxs:
        an = mol.GetAtomWithIdx(anion_idx)
        anchors = [n.GetIdx() for n in an.GetNeighbors() if n.GetSymbol() != 'H']
        for c1 in anchors:
            path = rdmolops.GetShortestPath(mol, c1, parent_attach_idx)
            if not path:
                continue
            d = len(path)  # distance(c1, attach) + 1 = the 1-indexed locant
            if best is None or d < best:
                best = d
    return best


def _name_ester_anion_zwitterion(mol, cation_idx: int, anion_idx: int) -> str:
    """Name a choline-family acid-ester-anion ZWITTERION (a phase B1).

     forces the anion (a P/S oxoacid mono-ester anion,
    as the parent -- but here, unlike the betaine path in `_route_zwitterion`
    below, cation on a DIFFERENT parent, severed and re-attached as
    a `(...azaniumyl)` prefix), the cation sits INSIDE the ester-owner arm R
    itself (``R-O-SO3[-]``, ``R`` = ``2-(trimethylazaniumyl)ethyl``). So the
    owner substituent is named WITH the cation in place (the shipped
    cation-bearing-substituent capability,) via the same recursive
    ``_p_ester_owner_group`` helper the neutral/pure-anion acid-ester
    producers use (`phosphorus.py`, `acid_ester_anion.py`), and the acid word
    (``sulfate`` / ``hydrogen phosphate`` /...) is derived IN PLACE from the
    surviving free ``-OH`` count via ``conjugate_controller`` (— never
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
    guard -- it never lets a dropped atom through. The outer /OPSIN
    gate RT-verifies the returned name as the ultimate backstop.
    """
    from .conjugate_controller import PHOSPHATE_WORD, SULFATE_WORD, _terminal_acid_oxygens
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


def _atom_coverage_ok_for_polyacid_zwitterion(mol, cation_idx: int,
                                              parent_attach_idx: int,
                                              anion_idxs: list) -> bool:
    """Finding A (a phase review) -- atom-coverage guard for
    ``_name_polyacid_zwitterion``.

    That function severs the cation and names the poly-acid parent, but its
    ``parent_attach_idx`` is computed from the shortest path to ``anions[0]``
    ONLY. Its own docstring scopes the cation to "a protonated amine OR
    quaternary onium" -- a quaternary onium can carry MULTIPLE branches, so if
    a second (or third) carboxylate sat on a DIFFERENT branch than
    ``anions[0]``, that atom would be discarded with the cation fragment by
    ``_sever_cation_build_anion_parent`` -- a silent atom/charge drop never
    asserted internally (only the outer /OPSIN gate would eventually
    catch the resulting wrong molecule).

    Mirror ``_name_ester_anion_zwitterion``'s ``accounted`` discipline
    (:518-519 above): split the molecule at the SAME bond
    ``_sever_cation_build_anion_parent`` is about to cut and require every
    anion atom to land in the fragment that does NOT contain the cation --
    the severed-parent fragment atoms union the cation-side fragment atoms is
    ALWAYS every atom of the molecule (``RemoveBond`` drops no atom, only
    splits connectivity), so "every anion is on the parent side" is the whole
    coverage guarantee. Returns False (fail closed) if any anion is stranded
    on the cation side, or if the cut does not even disconnect the graph
    (e.g. a ring bridges the two atoms some other way) so no cation-free
    fragment exists at all.
    """
    rw = Chem.RWMol(mol)
    bond = rw.GetBondBetweenAtoms(cation_idx, parent_attach_idx)
    if bond is None:
        return False
    rw.RemoveBond(cation_idx, parent_attach_idx)
    frag_idx_tuples = Chem.GetMolFrags(rw.GetMol(), asMols=False, sanitizeFrags=False)
    cation_side = None
    for fr_idxs in frag_idx_tuples:
        if cation_idx in fr_idxs:
            cation_side = set(fr_idxs)
            break
    if cation_side is None:
        return False  # unreachable in practice (cation_idx is always in SOME fragment); fail closed
    return not any(idx in cation_side for idx in anion_idxs)


def _name_polyacid_zwitterion(mol, cations: list, anions: list, style: str) -> str:
    """ generalized to a MULTI-carboxylate anion parent (a phase).

    A single cation (protonated amine or quaternary onium) riding on a
    POLY-carboxylate parent -- >= 2 ``-C(=O)[O-]`` groups on one acyclic
    skeleton, e.g. the glutamate zwitterion anion
    ``[NH3+]C(CCC(=O)[O-])C(=O)[O-]`` -> ``2-azaniumylpentanedioate`` (RT-
    verified 2026-08-17) -- mirrors the established single-anion
    branch below (sever the cation -> neutralize EVERY anion in place ->
    re-enter the full pipeline -> convert the resulting polyacid name via the
    SAME ``name_carboxylate_anion`` transform that already turns
    ``'pentanedioic acid'`` into ``'pentanedioate'``), generalized from
    exactly-one to >= 2 carboxylate anions.

    Deliberately WITHOUT the single-anion branch's protonated-amine defer
    (, ``_route_zwitterion`` below): that defer exists because a
    MONO-carboxylate protonated-amine zwitterion has an established
    retained/neutral-form name (glycine, GABA,...) reachable via the legacy
    amino-acid-zwitterion path. A >= 2-carboxylate one does not --
    ``_name_amino_acid_zwitterion``'s neutral-form dispensation is licensed
    only for the monoamino MONOcarboxylic Table-10.4 amino acids,
    so for a diacid amino acid the systematic azaniumyl-prefixed acid-anion
    form built here IS the name (and the neutral 'glutamic acid'/'aspartic
    acid' fallback the legacy path might otherwise reach is a DIFFERENT,
    lower-charge molecule the outer /net-charge check would reject
    anyway).

    SCOPE (fail-closed on anything else, per invariant "0-wrong ABSOLUTE"):
    exactly 1 cation, >= 2 anions, EVERY anion classified 'carboxylate' (a
    mixed carboxylate + sulfonate/alkoxide/etc. zwitterion declines), the
    cation NOT a ring atom (a ring-skeletal cation is, out of scope
    here), and a real severable cation-attach bond. Returns '' on any decline
    -- the caller falls through to the existing exactly-one-anion scope check
    and the legacy amino-acid-zwitterion path, unchanged. 0 behaviour change
    for anything but this new multi-carboxylate shape; the top-level
    /OPSIN gate is the final backstop as elsewhere in this module.
    """
    from rdkit.Chem import rdmolops

    if len(cations) != 1 or len(anions) < 2:
        return ''
    if any(classify_anion(mol, a) != 'carboxylate' for a in anions):
        return ''  # mixed anion types -> out of scope, decline

    cation_idx = cations[0]['atom_idx']
    cation_atom = mol.GetAtomWithIdx(cation_idx)
    if cation_atom.IsInRing():
        return ''  # skeletal cation -> out of scope here (defer)

    anion_idx0 = anions[0]['atom_idx']
    path = rdmolops.GetShortestPath(mol, cation_idx, anion_idx0)
    if len(path) < 2:
        return ''
    parent_attach_idx = path[1]  # the cation neighbour leading into the parent

    anion_atom_idxs = [a['atom_idx'] for a in anions]
    if not _atom_coverage_ok_for_polyacid_zwitterion(
            mol, cation_idx, parent_attach_idx, anion_atom_idxs):
        return ''  # Finding A: an anion is stranded on the cation side of the
                   # severed bond -> would be silently dropped -- fail closed

    from ..assembly.substituent_naming import cation_to_prefix
    cat_prefix = cation_to_prefix(mol, cation_idx, parent_attach_idx)
    if not cat_prefix:
        return ''  # ylide / non-N onium / unnameable -> honest-fail

    parent_smi = _sever_cation_build_anion_parent(mol, cation_idx, parent_attach_idx)
    if not parent_smi:
        return ''
    try:
        neutral_name = _reenter(parent_smi, style)
    except (RecursionError, ValueError, RuntimeError):
        return ''
    if not neutral_name or _is_malformed_parent(neutral_name):
        return ''

    from .ions import name_carboxylate_anion
    parent_anion_name = name_carboxylate_anion(neutral_name)
    if not parent_anion_name:
        return ''

    # Attachment locant: the MINIMUM candidate over every carboxyl anchor (the
    # severed-cation parent is numbering-symmetric with no other substituent to
    # break the tie -- lowest locants picks the smaller end).
    locant_prefix = ''
    attach_locant = _attachment_locant_on_polyacid_parent(
        mol, anion_atom_idxs, parent_attach_idx)
    if attach_locant is not None and attach_locant >= 2:
        if _parent_has_chain_locants(parent_anion_name):
            locant_prefix = f'{attach_locant}-'
        else:
            # Finding B (a phase review): the parent-anion name is not a
            # form `_parent_has_chain_locants` recognizes (e.g. a 3+-carboxylate
            # `...tricarboxylate` from `name_carboxylate_anion` -- --
            # which matches none of anoate/enoate/ynoate/dioate). A required
            # locant (>= 2, so omitting it is NOT one of the
            # licences) would then silently stay uncited, letting OPSIN default
            # the azaniumyl onto the wrong ring/chain position is
            # deny-by-default). Fail closed rather than ship an unlocanted
            # prefix on a parent form this producer cannot locant correctly.
            return ''

    #: a SIMPLE (unsubstituted) prefix like a bare protonated-amine
    # 'azaniumyl' is cited WITHOUT enclosing marks (BB '2-aminopentanedioic
    # acid (PIN)' pattern -- no parens around 'amino'); a COMPOUND/substituted
    # one (a quaternary 'trimethylazaniumyl') still takes them, matching the
    # established single-anion branch's ``({cat_prefix})`` convention below.
    # Reuse the general compound/complex test rather than a new ad hoc rule.
    from ..assembly.naming_utils import enclose_if_compound
    enclosed_cat_prefix = enclose_if_compound(cat_prefix)
    composed = f'{locant_prefix}{enclosed_cat_prefix}{parent_anion_name}'

    # Defensive: never ship a malformed composition where the prefix glues
    # directly onto a parent locant with no separator (mirrors the
    # single-anion branch's identical guard below).
    from ..assembly.naming_utils import _STEREO_PAREN_RE
    _after = _STEREO_PAREN_RE.sub('', parent_anion_name, count=1).lstrip('-')
    if not locant_prefix and _after[:1].isdigit():
        return ''
    return composed


# =============================================================================
# charged Slice B — primary protonated-amine azaniumyl PIN.
#
# (the Blue Book item (e)): a zwitterion whose ionic centres sit
# in ONE parent is NOT named as a neutral suffix-bearing compound; its PIN is the
# IONIC form -- the ANION is the parent (keeps its -oate/... suffix) and each
# protonated-amine CATION becomes an ``azaniumyl`` substituent prefix (two or
# more of the same kind -> ``bis(azaniumyl)`` / ``tris(azaniumyl)``,.
#
# The betaine SEVER path below cannot serve a PRIMARY amino-acid
# zwitterion: severing the -NH3+ and capping its carbon with H DESTROYS the alpha
# stereocentre, so it can never spell the ``(2R)``/``(2S)`` descriptor the ionic
# PIN needs (measured: S-methylcysteine's C2 is a stereocentre only WHILE the N
# is attached). This builder instead NEUTRALIZES the whole zwitterion in place
# (-NH3+ -> -NH2, -COO- -> -COOH), names the neutral amino acid through the full
# pipeline (which keeps every stereocentre + numbering + substituent order),
# converts the acid to its carboxylate anion via the proven
# ``name_carboxylate_anion`` transform, and re-expresses the grouped ``amino``
# prefix as the ionic ``azaniumyl`` prefix. Replacing -NH2 by -NH3+ does not
# change the alpha carbon's CIP priorities, so the descriptor is invariant
# between the two forms (verified: the neutral and ionic InChIKeys share the same
# stereo layer). Every emission is full-InChIKey RT-gated (charges + stereo), so
# a wrong transform / wrong descriptor / dropped charge fails CLOSED (0-wrong).
# =============================================================================

# Simple multiplying prefix (di/tri/...) that the neutral namer uses to group
# identical detachable ``amino`` prefixes, and the multiplicative prefix
# (bis/tris/...) that the azaniumyl ionic prefix takes instead:
# ``bis``/``tris`` are used before a substituted/parenthesised substituent group
# name; ``2,5-bis(azaniumyl)pentanoate`` is OPSIN-RT verified).
_SIMPLE_MULTIPLIER = {1: '', 2: 'di', 3: 'tri', 4: 'tetra', 5: 'penta', 6: 'hexa'}
_MULTIPLICATIVE_PREFIX = {2: 'bis', 3: 'tris', 4: 'tetrakis',
                          5: 'pentakis', 6: 'hexakis'}


def _full_inchikey_rt_ok(mol, name: str) -> bool:
    """0-wrong gate for the azaniumyl builder: OPSIN-parse ``name`` and require
    its FULL standard InChIKey (constitution + charge + stereo) to equal the
    input's. Fail CLOSED on a missing jar / parse failure / any mismatch. This is
    the producer's OWN backstop, independent of the namer's gate (which is
    constitution-only on the default path and would miss a wrong descriptor or a
    neutralised net charge) -- mirrors ``added_carbon_parent._whole_name_stereo_ok``.
    """
    from ..namer import _validity_gate_name_to_smiles
    smi = _validity_gate_name_to_smiles(name)
    if not smi:
        return False
    parsed = Chem.MolFromSmiles(smi)
    if parsed is None:
        return False
    try:
        if Chem.MolToInchiKey(parsed) != inchikey_of(mol):
            return False
    except Exception:
        return False
    # An equal full key is not enough for a radical (it encodes neither radical
    # electrons nor bond order): validation/radical_identity.py.
    from ..validation.radical_identity import radical_identity_verdict
    return radical_identity_verdict(Chem.MolToSmiles(mol), smi) != "mismatch"


def _iminide_omit_locants(name: str, mol) -> str:
    """ locant omission for an ``-iminide`` anion name -- RE-ANCHORED and
    full-InChIKey RT-AUDITED, so it can never change the structure (0-wrong).

    The ``=N-`` anion nitrogen carries NO substitutable hydrogen, which unlocks two
     licences the NEUTRAL imine name cannot claim (its ``=NH`` is a second
    kind of substitutable H):

      * the imine-position suffix locant drops -- ``butan-1-iminide`` ->
        ``butaniminide`` (the Blue Book, the terminal ``=N`` group, family);
      * the substituent locants on the sole heteroatom parent centre drop when that
        centre carries the only kind of substitutable H:
        ``P,P,P-trimethyl-λ5-phosphaniminide`` -> ``trimethyl-λ5-phosphaniminide``
        (the Blue Book). Contrast the NEUTRAL ``As,As,As-trimethyl-λ5-arsanimine``
        (the Blue Book), which KEEPS them because its ``N-H`` is a second kind of H.

    Only these two structurally-licensed patterns are stripped (never an arbitrary
    locant), and EVERY candidate is verified with:func:`_full_inchikey_rt_ok`, so a
    strip that would change the molecule -- an internal ``butan-2-iminide`` collapsing
    to the position-1 default ``butaniminide`` -- is REJECTED and the locanted form is
    kept. Returns the maximally-omitted RT-verified name, or ``name`` unchanged (which
    is itself RT-valid, so the row still ships as RIGHT-molecule).
    """
    import re
    if not name.endswith('iminide'):
        return name
    candidates = [name]
    # (a) drop the numeric imine-position suffix locant: '...-<n>-iminide'.
    a = re.sub(r'-\d+-iminide$', 'iminide', name)
    if a != name:
        candidates.append(a)
    # (b) drop a leading run of identical element locants on the parent centre
    # ('P,P,P-'/'As,As,As-'): a capitalised element token repeated and joined
    # by commas. Substituent prefixes are lowercase and descriptors start with
    # '(', so a leading capitalised token is an element locant.
    for base in list(candidates):
        b = re.sub(r'^([A-Z][a-z]?)(?:,\1)*-', '', base)
        if b != base and b not in candidates:
            candidates.append(b)
    # Prefer the maximally-omitted form that still RT-matches the input EXACTLY.
    for cand in sorted(candidates, key=len):
        if _full_inchikey_rt_ok(mol, cand):
            return cand
    return name


def _amino_prefix_to_azaniumyl(anion_name: str, n: int) -> str:
    """Re-express the grouped neutral ``amino`` prefix of a carboxylate-anion name
    as the ionic ``azaniumyl`` prefix. ``n`` = the number of primary
    protonated amines.

    Re-anchored, not blind string surgery: the builder's scope guarantees the
    molecule carries EXACTLY ``n`` nitrogen atoms, all primary -NH3+, so the
    neutral name contains exactly one detachable amino cluster -- ``amino``
    (n=1) or ``{di|tri|...}amino`` (n>=2) -- and no other occurrence of the
    substring. Swap that single locanted token for ``azaniumyl`` (n=1) or
    ``{bis|tris|...}(azaniumyl)`` (n>=2), preserving its locant set. Returns ''
    (fail-closed) if the expected token is absent or ambiguous; the caller then
    abstains. The whole result is full-InChIKey RT-gated by the caller.
    """
    simple = _SIMPLE_MULTIPLIER.get(n)
    if simple is None:
        return ''
    token = simple + 'amino'
    if n == 1:
        repl = 'azaniumyl'
    else:
        mult = _MULTIPLICATIVE_PREFIX.get(n)
        if not mult:
            return ''
        repl = mult + '(azaniumyl)'
    # The token must appear exactly once and be a STANDALONE detachable prefix,
    # not a substring inside a larger token (e.g. a compound '...ylamino'): reject
    # only when the char immediately before it is a LETTER. Both start-of-name
    # (idx == 0, an UNLOCANTED amino such as glycine's ``aminoacetate``) and a
    # preceding locant separator ('-', e.g. ``2,3-diamino...``) are valid prefix
    # boundaries. The whole result is full-InChIKey RT-gated by the caller.
    if anion_name.count(token) != 1:
        return ''
    idx = anion_name.find(token)
    if idx > 0 and anion_name[idx - 1].isalpha():
        return ''
    return anion_name[:idx] + repl + anion_name[idx + len(token):]


def _name_primary_amine_azaniumyl_zwitterion(mol, cations, anions, style) -> str:
    """ azaniumyl PIN for a PRIMARY protonated-amine zwitterion
    (net-zero amino-acid-shaped OR net-charged mixed-sign, e.g. a protonated
    diamino-acid). The anion is the parent, each -NH3+ an ``azaniumyl`` prefix.

    SCOPE (fail-closed '' on anything else -> caller falls through unchanged):
      * every cation is a PRIMARY protonated ammonium -NH3+ (N, +1, 3 H, bonded
        to exactly one heavy atom) -- quaternary betaines keep the SEVER path;
      * EVERY nitrogen in the molecule is one of those cations (so the neutral
        name has exactly one grouped ``amino`` prefix and no stray N token);
      * every anion is a carboxylate (the ``name_carboxylate_anion`` transform);
      * the cation/anion counts avoid the ``_name_polyacid_zwitterion`` region it
        already owns: EITHER exactly 1 cation + exactly 1 anion, OR >= 2 cations.

    All four RT-verified targets are carboxylate; a sulfinate/sulfonate amino
    acid is out of scope here (declines -> abstain) and is left for a later
    widening. Every emission is full-InChIKey RT-gated.
    """
    if not cations or not anions:
        return ''
    # Partition: the 1-cation/>=2-anion region is _name_polyacid_zwitterion's.
    if not ((len(cations) == 1 and len(anions) == 1) or len(cations) >= 2):
        return ''
    # Every cation a primary protonated ammonium -NH3+.
    for c in cations:
        a = mol.GetAtomWithIdx(c['atom_idx'])
        if (a.GetSymbol() != 'N' or a.GetFormalCharge() != 1
                or a.GetTotalNumHs() != 3):
            return ''
        heavy = [nb for nb in a.GetNeighbors() if nb.GetSymbol() != 'H']
        if len(heavy) != 1:
            return ''
    # No OTHER nitrogen -> the neutral name's only detachable N prefix is the
    # grouped ``amino`` cluster (guards the re-expression against a stray amino).
    n_nitrogen = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'N')
    if n_nitrogen != len(cations):
        return ''
    # Every anion a carboxylate.
    if any(classify_anion(mol, a) != 'carboxylate' for a in anions):
        return ''

    n = len(cations)
    from .salts import _neutralize_zwitterion
    neutral = _neutralize_zwitterion(mol)
    if neutral is None:
        return ''
    neutral_smi = Chem.MolToSmiles(neutral, canonical=True)
    if not neutral_smi:
        return ''
    try:
        neutral_name = _reenter(neutral_smi, style)
    except (RecursionError, ValueError, RuntimeError):
        return ''
    if not neutral_name or _is_malformed_parent(neutral_name):
        return ''

    from .ions import name_carboxylate_anion
    anion_name = name_carboxylate_anion(neutral_name)
    if not anion_name:
        return ''

    composed = _amino_prefix_to_azaniumyl(anion_name, n)
    if not composed:
        return ''

    # 0-wrong: full-InChIKey RT gate (charges + stereo). Never ship on a mismatch.
    if not _full_inchikey_rt_ok(mol, composed):
        return ''
    return composed


def _route_zwitterion(mol, sites, style: str) -> str:
    """GUARD 4: zwitterion anion-is-parent override.

     (verbatim): "anionic centers... become the parent structure, into
    which the cationic part is substituted." So: FORCE the anion as the parent.

    - (cation on a DIFFERENT parent, e.g. the betaine quaternary
      ammonium): name the anion parent (neutralize ALL charges -> re-enter ->
      re-apply the anionic suffix) and PREFIX the structured cation-substituent
      ``(…azaniumyl)`` produced by ``cation_to_prefix``.
    - (cation INSIDE the anion's parent hydride, e.g. ring N+ of a
      pyridinium-2-carboxylate): the cation is kept on the parent as an ``-ium``
      suffix, NOT a prefix -> DEFER to the legacy path (returns '' here so the
      existing pipeline names it; route_charged does not own the skeletal-ium
      cumulative-suffix construction this plan — honest scope boundary).

    Sequencing : amino-acid zwitterions + betaines first. Ylides /
    amine-oxides / 1,n-dipolar are out of scope -> '' (honest-fail).

    Returns the IUPAC name, or '' to fall through to the legacy path.
    """
    from rdkit.Chem import rdmolops

    cations = sites['cations']
    anions = sites['anions']

    # a phase B1 (choline-family acid-ester-anion zwitterion,:
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
    # (carboxylate betaine, amino-acid, ring carboxylate,...) has an anion
    # whose sole neighbour is a carbon, so `central.GetSymbol not in ('P',
    # 'S')` declines INSTANTLY and falls through unchanged to the scope check
    # and legacy paths below -- 0 behaviour change for anything but the new
    # shape.
    if len(cations) == 1 and len(anions) >= 1:
        est_name = _name_ester_anion_zwitterion(
            mol, cations[0]['atom_idx'], anions[0]['atom_idx'])
        if est_name:
            return est_name

    # a phase generalized to a MULTI-carboxylate anion parent):
    # a single non-ring cation (protonated amine or quaternary onium) riding
    # on a POLY-carboxylate anion parent -- e.g. the glutamate/aspartate
    # zwitterion anion [NH3+]C(CCC(=O)[O-])C(=O)[O-] -> 2-azaniumylpentane-
    # dioate. Tried BEFORE the exactly-one-anion scope check below (which
    # would otherwise decline this shape outright): `_name_polyacid_zwitterion`
    # self-validates the shape (single non-ring cation, EVERY anion a
    # carboxylate) and returns '' on anything else, so this is a pure ADD --
    # 0 behaviour change for the established single-anion amino-acid / betaine
    # majority.
    if len(cations) == 1 and len(anions) >= 2:
        poly_name = _name_polyacid_zwitterion(mol, cations, anions, style)
        if poly_name:
            return poly_name

    # charged Slice B: PRIMARY protonated-amine azaniumyl PIN.
    # Handles the two shapes the sever paths above cannot: (a) a single-cation
    # single-anion amino acid whose alpha stereocentre must survive into the
    # ``(2R)-2-azaniumyl...oate`` descriptor (severing destroys it), and (b) a
    # MULTI-cation / net-charged protonated diamino-acid -> ``bis(azaniumyl)``.
    # Self-validates its scope (primary -NH3+ only, no other N, carboxylate
    # anions, counts disjoint from the polyacid region above) and full-InChIKey
    # RT-gates every emission, so this is a pure ADD: a quaternary betaine
    # (0 H on N) declines instantly and falls through to the sever paths below,
    # 0 behaviour change for the established betaine/choline majority.
    azm = _name_primary_amine_azaniumyl_zwitterion(mol, cations, anions, style)
    if azm:
        return azm

    # Scope : exactly one cationic and one anionic center (the amino-acid /
    # betaine majority). Multi-center dipolar zwitterions are deferred.
    if len(cations) != 1 or len(anions) != 1:
        return ''

    cation_idx = cations[0]['atom_idx']
    anion_idx = anions[0]['atom_idx']
    cation_atom = mol.GetAtomWithIdx(cation_idx)

    # F- (DD3,: the cationic centre is a RING atom skeletal to the
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

    # SCOPE : this single-cation SEVER path (below) builds GUARD 4's
    # (azaniumyl) prefix for a QUATERNARY ammonium (0 H) betaine, whose severed
    # cation carbon is not a stereocentre. A PROTONATED amine (NH3+/NH2+/NH+, >0
    # H) whose azaniumyl PIN was buildable has ALREADY been named by
    # `_name_primary_amine_azaniumyl_zwitterion` above (charged Slice B,
    # — the stereo-preserving neutralize-in-place builder, tried
    # before the scope check). Reaching HERE with a protonated amine therefore
    # means that builder DECLINED (out of scope for it: a non-carboxylate anion,
    # a stray extra nitrogen, or a failed RT gate). The sever path cannot help
    # such a case (it would drop the alpha stereocentre), so defer to the legacy
    # path — which, on the DEFAULT tier, now abstains per (the neutral
    # form is not the PIN; salts.name_zwitterion best-effort-gates that fallback).
    if cation_atom.GetSymbol() == 'N' and cation_atom.GetTotalNumHs() > 0:
        return ''

    #: cation skeletal to the anion's parent ring -> keep on parent as
    # an -ium suffix. route_charged does NOT build the cumulative ium+ate suffix
    # here; defer to the legacy path (honest scope boundary).
    if _cation_is_skeletal_to_anion_parent(mol, cation_idx, anion_idx):
        return ''

    #: separable cation -> substituent prefix on the anion parent.
    # 1. The cation prefix (structured producer; '' on out-of-scope cation).
    path = rdmolops.GetShortestPath(mol, cation_idx, anion_idx)
    if len(path) < 2:
        return ''
    parent_attach_idx = path[1]  # the cation neighbour leading into the anion parent

    # SCOPE : GUARD 4 confidently handles the amino-acid / betaine majority
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

    # 2. The anion parent name: the anion is the parent). The cationic
    # part is a SEPARATE parent: SEVER the cation substituent from the
    # anion parent (break the cation-atom <-> parent_attach bond, cap the parent
    # side with H), keep ONLY the anion fragment, neutralize the anion, re-enter,
    # then re-apply the anionic suffix. (Neutralizing the quaternary cation in
    # place is impossible without breaking a bond — RDKit valence error — which
    # is exactly why demotes it to a prefix instead.)
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
        # cannot convert the retained 'acetic acid'.
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

    # 4. Compose: {locant}-(cation-prefix)anion-parent — prefix the
    # cation to the anionic parent). Enclosing marks per (complex prefix).
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


def _diazonium_sever_parts(mol, cation_idx: int):
    """Locate the three atoms of a terminal -N#N+/-N=N+ diazonium group given a
    cation N index: ``(proximal_n, terminal_n, parent_attach)`` — or ``None`` when
    the shape is not a genuine terminal diazonium.

    ``proximal_n`` is whichever N of the pair is directly bonded to the parent;
    ``terminal_n`` is the OTHER one (degree 1); ``parent_attach`` is the parent-
    hydride atom the group hangs off. Handles the resonance twin (charge drawn on
    the terminal N) exactly as ``_name_diazonium`` used to inline. Shared by
    ``_name_diazonium`` (single site) and ``_name_poly_diazonium`` (>= 2 sites)."""
    try:
        cat = mol.GetAtomWithIdx(cation_idx)
    except (RuntimeError, IndexError, OverflowError):
        return None
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
        return None

    proximal_n, terminal_n = cation_idx, diazo_n
    if parent_attach is None:
        # Resonance-shifted twin: cation_idx has no non-N neighbour (the charge
        # sits on the TERMINAL N), so the parent-attachment atom is two bonds
        # away -- reached via diazo_n's other neighbour, making diazo_n PROXIMAL.
        dn = mol.GetAtomWithIdx(diazo_n)
        others = [nb.GetIdx() for nb in dn.GetNeighbors() if nb.GetIdx() != cation_idx]
        if len(others) != 1:
            return None
        parent_attach = others[0]
        proximal_n, terminal_n = diazo_n, cation_idx

    if mol.GetAtomWithIdx(terminal_n).GetDegree() != 1:
        return None
    return proximal_n, terminal_n, parent_attach


def _name_poly_diazonium(mol, cation_indices, style: str) -> str:
    """ (the Blue Book): a poly-diazonium cation — >= 2 terminal -N#N+
    groups on ONE ring parent hydride — named ``<ring>-<locants>-bis(diazonium)``
    (``N#[N+]c1ccc([N+]#N)cc1`` -> ``benzene-1,4-bis(diazonium)``). Suffix
    multiplication with the COMPLEX multiplier (bis/tris), mirroring the single
    -diazonium append but citing an attachment locant per group.

    Sever every -N#N+ from the parent (cap each attachment carbon with H), keep
    ONE fragment carrying all attachment atoms, name that neutral ring parent,
    number the ring for lowest locants to the attachment SET
    (``_ring_locants_lowest_to_set``), and compose. SCOPE: a single monocyclic
    ring parent (the numbering helper fails closed on fused/multi rings). The
    composed name is then RT-gated (``_cation_name_rt_ok``) so any ring-numbering
    mismatch on an exotic heteroarene abstains rather than ships a wrong locant
    (0-wrong). Returns '' on any decline."""
    from ..assembly.naming_utils import COMPLEX_MULTIPLIERS
    from .ions import _ring_locants_lowest_to_set

    # 1. Sever parts for every diazonium; collect (proximal_n, parent_attach).
    cuts = []          # (proximal_n, parent_attach)
    attach_atoms = []  # parent-hydride atoms carrying a diazonium
    diazo_ns = set()   # every N of every -N#N+ group (to identify the parent frag)
    for cidx in cation_indices:
        parts = _diazonium_sever_parts(mol, cidx)
        if parts is None:
            return ''
        proximal_n, terminal_n, parent_attach = parts
        cuts.append((proximal_n, parent_attach))
        attach_atoms.append(parent_attach)
        diazo_ns.update((proximal_n, terminal_n))
    if len(set(attach_atoms)) != len(attach_atoms):
        return ''  # two groups on one atom -> out of scope

    # 2. Sever every group on ONE index-preserving RWMol, H-capping each attach.
    rw = Chem.RWMol(mol)
    for proximal_n, parent_attach in cuts:
        rw.RemoveBond(proximal_n, parent_attach)
        pa = rw.GetAtomWithIdx(parent_attach)
        pa.SetNumExplicitHs(pa.GetNumExplicitHs() + 1)
    built = rw.GetMol()
    try:
        frags = Chem.GetMolFrags(built, asMols=True, sanitizeFrags=False)
        fidx = Chem.GetMolFrags(built, asMols=False, sanitizeFrags=False)
    except Exception:
        return ''
    # 3. The parent fragment is the one carrying every attachment atom (and no
    # diazo N). Build an original-index -> fragment-index map for numbering.
    attach_set = set(attach_atoms)
    parent_mol = None
    orig_to_frag = None
    for fm, fi in zip(frags, fidx):
        fi_set = set(fi)
        if attach_set.issubset(fi_set):
            if fi_set & diazo_ns:
                return ''  # a diazo N landed in the parent frag -> not severed
            parent_mol = fm
            orig_to_frag = {orig: k for k, orig in enumerate(fi)}
            break
    if parent_mol is None or orig_to_frag is None:
        return ''
    try:
        Chem.SanitizeMol(parent_mol)
        parent_smi = Chem.MolToSmiles(parent_mol, canonical=True)
    except Exception:
        return ''
    if not parent_smi:
        return ''

    # 4. Name the neutral ring parent.
    try:
        parent = _reenter(parent_smi, style)
    except (RecursionError, ValueError, RuntimeError):
        return ''
    if not parent or _is_malformed_parent(parent):
        return ''
    # A composed '<ring>-<locs>-bis(diazonium)' is well-formed only for a BARE
    # ring parent (stem, no principal characteristic-group suffix and no existing
    # locants of its own); refuse anything else -> RT-gate would catch it, but
    # fail closed early.
    low = parent.lower()
    if any(tok in low for tok in ('-', 'ic acid', 'amine', 'nitrile', 'one', 'ol')):
        return ''

    # 5. Number the parent ring for lowest locants to the attachment SET.
    from ..perception.rings import get_ring_systems
    frag_attach = [orig_to_frag[a] for a in attach_atoms]
    ring_system = None
    try:
        for rs in get_ring_systems(parent_mol, include_spiro=True):
            if set(frag_attach).issubset(set(rs)):
                ring_system = rs
                break
    except (RuntimeError, ValueError):
        return ''
    if ring_system is None:
        return ''
    loc_map = _ring_locants_lowest_to_set(parent_mol, ring_system, frag_attach)
    if not loc_map:
        return ''
    try:
        locs = sorted(loc_map[a] for a in frag_attach)
    except KeyError:
        return ''

    mult = COMPLEX_MULTIPLIERS.get(len(locs))
    if not mult:
        return ''
    name = f"{parent}-{','.join(map(str, locs))}-{mult}(diazonium)"
    # 6. 0-wrong RT-gate (ring-numbering coupling can differ from the parent
    # name's own numbering on an exotic heteroarene): ship only if it parses
    # back to the input structure.
    if not _cation_name_rt_ok(name, mol):
        return ''
    return name


def _name_diazonium(mol, cation_idx: int, style: str) -> str:
    """: name a diazonium cation R-N2+ as ``<parent-hydride>diazonium``.

    The cationic N (``cation_idx``) sits somewhere in a terminal
    -N#N+/-N=N+ pair; WHICH of the two N atoms carries the formal charge is
    a resonance-drawing choice RDKit does not normalise (a phase a trace,
    `internal notes` Q4). The CANONICAL
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
    parts = _diazonium_sever_parts(mol, cation_idx)
    if parts is None:
        return ''
    proximal_n, terminal_n, parent_attach = parts
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


def _bare_parent_ion_unit(mol, ion_idx: int, kind: str):
    """Free-ion UNIT name + its linker-attachment atom for a BARE parent-ion
    centre / multiplicative-parent unit). A bare centre
    has EXACTLY ONE heavy neighbour — the atom of the shared linker it hangs off:

      ``kind='heteroatom_hydride_anion'`` -> ``<parent-hydride stem>ide``
          (P/As/Sb/Si/Ge/Sn/Pb: phosphanide, arsanide, silanide, …).
      ``kind='onium'`` -> ``cation_to_prefix(as_free_ion=True)``
          (the standalone onium-cation unit: phosphanium, sulfanium, oxidanium…).

    Returns ``(unit_name, attach_idx)`` or ``None`` (out of scope: a substituted
    centre with >1 heavy neighbour, or an element with no parent-hydride stem).
    A substituted centre is refused so the repeated UNIT is unambiguously bare
    and the linker-attachment atom is the sole heavy neighbour."""
    from .ions import _HETEROATOM_HYDRIDE_IDE_STEMS
    atom = mol.GetAtomWithIdx(ion_idx)
    heavy = [nb for nb in atom.GetNeighbors() if nb.GetSymbol() != 'H']
    if len(heavy) != 1:
        return None
    attach = heavy[0].GetIdx()
    if kind == 'heteroatom_hydride_anion':
        stem = _HETEROATOM_HYDRIDE_IDE_STEMS.get(atom.GetSymbol())
        if not stem:
            return None
        # parent hydride stem ('phosphane') -> '-ide' anion, eliding the final 'e'.
        unit = (stem[:-1] if stem.endswith('e') else stem) + 'ide'
        return unit, attach
    if kind == 'onium':
        from ..assembly.substituent_naming import cation_to_prefix
        unit = cation_to_prefix(mol, ion_idx, attach, as_free_ion=True)
        if not unit:
            return None
        return unit, attach
    return None


def _divalent_skeletal_linker(mol, ion_indices, attach_atoms):
    """Name the DIVALENT skeletal LINKER joining exactly two identical parent-ion
    units that are bonded DIRECTLY to it /. Returns
    ``(linker_name, needs_enclosing_marks)`` or ``None`` (fail closed). Two shapes:

      - a single ISOLATED benzene ring -> ``'<l1>,<l2>-phenylene'`` (the retained
        divalent arene,; the two ion carbons take the lowest ring
        locants). Enclosing marks REQUIRED — the reference PIN is
        ``(1,4-phenylene)bis(phosphanide)``.
      - a linear, unbranched, saturated, acyclic all-carbon bridge whose two
        termini are the ion-attachment carbons -> ``'alkane-1,n-diyl'`` (the
        ``_walk_linear_carbon_bridge`` primitive, exactly as
        ``emit_bis_quaternary_ammonium`` uses it). NO enclosing marks (cf.
        ``hexane-1,6-diylbis(trimethylazanium)``).

    Fails closed on fused / hetero / branched / substituted linkers -> the caller
    abstains rather than shipping a wrong multiplicative parent."""
    if len(set(attach_atoms)) != 2:
        return None
    a1, a2 = attach_atoms
    # --- benzene phenylene ------------------------------------------------
    from ..perception.rings import get_ring_systems
    from .ions import _ring_locants_lowest_to_set
    try:
        ring_systems = get_ring_systems(mol, include_spiro=True)
    except (RuntimeError, ValueError):
        ring_systems = []
    for rs in ring_systems:
        if a1 in rs and a2 in rs:
            # An isolated benzene: exactly 6 members, every one an aromatic
            # carbon. A fused / heteroaromatic ring is out of the phenylene
            # scope (would need naphthalenediyl / a heteroarenediyl name).
            if len(rs) != 6 or not all(
                    mol.GetAtomWithIdx(i).GetSymbol() == 'C'
                    and mol.GetAtomWithIdx(i).GetIsAromatic() for i in rs):
                return None
            loc_map = _ring_locants_lowest_to_set(mol, rs, [a1, a2])
            if not loc_map:
                return None
            try:
                l1, l2 = sorted((loc_map[a1], loc_map[a2]))
            except KeyError:
                return None
            return f"{l1},{l2}-phenylene", True
    # --- linear alkanediyl ------------------------------------------------
    from .ions import _walk_linear_carbon_bridge
    bridge = _walk_linear_carbon_bridge(mol, ion_indices[0], ion_indices[1])
    if bridge and {bridge[0], bridge[-1]} == set(attach_atoms):
        from ..data.chain_names import get_chain_prefix
        n = len(bridge)
        try:
            stem = get_chain_prefix(n)
        except ValueError:
            return None
        return f"{stem}ane-1,{n}-diyl", False
    return None


def _name_bis_parent_ion_linker(mol, ion_indices, kind: str, style: str) -> str:
    """ / Case B (multiplicative LINKER): >= 2 IDENTICAL bare
    parent-ion units hanging off ONE di/polyvalent skeletal linker ->
    ``(LINKER)bis(UNIT)``:

        [PH-]c1ccc([PH-])cc1 -> (1,4-phenylene)bis(phosphanide) (anion)
        [PH3+]c1ccc([PH3+])cc1 -> (1,4-phenylene)bis(phosphanium) (cation)

    Mirrors ``emit_bis_quaternary_ammonium`` (the two units must be BYTE-IDENTICAL
    free-ion strings, the linker is named as one divalent substituent group), but
    generalises the linker from a linear carbon bridge to a benzene phenylene, and
    covers the anion (heteroatom-hydride -ide) as well as the cation (onium) side.
    ``kind`` in ``{'heteroatom_hydride_anion', 'onium'}``. RT-gated (0-wrong):
    fails closed ('') on anything not this exact bare-unit-on-divalent-linker
    class, and on any composed name that does not OPSIN-round-trip to ``mol``."""
    from ..assembly.naming_utils import COMPLEX_MULTIPLIERS
    if len(ion_indices) != 2 or len(set(ion_indices)) != 2:
        return ''
    units, attach = [], []
    for ii in ion_indices:
        r = _bare_parent_ion_unit(mol, ii, kind)
        if r is None:
            return ''
        unit, at = r
        units.append(unit)
        attach.append(at)
    if units[0] != units[1]:
        return ''  # not identical units -> fail closed (asymmetric)
    linker = _divalent_skeletal_linker(mol, ion_indices, attach)
    if linker is None:
        return ''
    linker_name, needs_encl = linker
    mult = COMPLEX_MULTIPLIERS.get(len(ion_indices))
    if not mult:
        return ''
    body = f"({linker_name})" if needs_encl else linker_name
    name = f"{body}{mult}({units[0]})"
    if not _cation_name_rt_ok(name, mol):
        return ''
    return name


def _name_bis_acyl_azanide(mol, anion_indices, style: str) -> str:
    """ Case B (acyl LINKER): >= 2 identical acyl-azanide anions
    (R-CO-NH-) sharing ONE di/polyacyl linker -> ``<polyacyl>bis(azanide)``
    ([NH-]C(=O)CCC([NH-])=O -> ``butanedioylbis(azanide)``).

    The linker here carries the carbonyl carbons (an acyl-diyl, not a bare
    skeletal bridge), so it is named the same way the single ``_name_acyl_azanide``
    is: transmute EVERY anionic N -> an -OH oxygen, name the resulting neutral
    poly-ACID, convert it to the poly-ACYL group (``_acid_to_acyl``), and append
    the bis/tris(azanide) compound suffix. RT-gated (0-wrong). '' on any decline
    (a non-acyl N-, an acid whose acyl form does not round-trip -> abstain)."""
    from ..assembly.naming_utils import COMPLEX_MULTIPLIERS
    if len(anion_indices) < 2 or len(set(anion_indices)) != len(anion_indices):
        return ''
    rw = Chem.RWMol(mol)
    for ai in anion_indices:
        atom = mol.GetAtomWithIdx(ai)
        heavy = [nb for nb in atom.GetNeighbors() if nb.GetSymbol() != 'H']
        if len(heavy) != 1 or heavy[0].GetSymbol() != 'C':
            return ''
        acyl_c = heavy[0]
        if not any(b.GetBondType() == Chem.BondType.DOUBLE
                   and b.GetOtherAtom(acyl_c).GetSymbol() in ('O', 'S', 'Se')
                   for b in acyl_c.GetBonds()):
            return ''  # N- not on an acyl carbon -> not an acyl azanide
        a = rw.GetAtomWithIdx(ai)
        a.SetAtomicNum(8)         # N- -> O
        a.SetFormalCharge(0)
        a.SetNoImplicit(False)
        a.SetNumExplicitHs(0)     # let RDKit add the acid -OH hydrogen
    try:
        acid_mol = rw.GetMol()
        Chem.SanitizeMol(acid_mol)
        acid_smi = Chem.MolToSmiles(acid_mol)
    except (RuntimeError, ValueError):
        return ''
    from ..assembly.fragment_naming import name_fragment_recursively
    acid_name = name_fragment_recursively(acid_smi)
    if not acid_name:
        return ''
    from .lipids import _acid_to_acyl
    acyl = _acid_to_acyl(acid_name)
    if not acyl:
        return ''
    mult = COMPLEX_MULTIPLIERS.get(len(anion_indices))
    if not mult:
        return ''
    name = f"{acyl}{mult}(azanide)"
    if not _cation_name_rt_ok(name, mol):
        return ''
    return name


def emit_secondary_amine_azanide(mol, anion_idx: int, style: str) -> str:
    """ (the Blue Book, ``acetylazanide (PIN)``): a
    DISUBSTITUTED azanide R-[N-]-R' — a deprotonated secondary amide /
    sulfonamide / secondary amine named as azanide (NH2-) carrying two
    N-substituent prefixes. point (3) disallows the ``...aminide``
    method-(1) form for amides, so azanide IS the PIN for the acyl/sulfonyl
    arms; for a pure secondary-amine arm the PIN is technically ``...aminide``
    , and this azanide form is then a correct RT-verified
    best-effort — either way the emitted structure is exactly the input.

    Names each N-arm as a substituent prefix (``name_substituent_fragment``),
    alphabetises, escalates the enclosing marks via
    ``apply_enclosing_marks``), and appends ``azanide``. RT-gated on the FULL
    InChIKey (``_cation_name_rt_ok``) — fail closed to '' (never a wrong or
    label-dropping name). Scoped to the ACYCLIC secondary-N-anion shape: a
    non-aromatic N-, formal charge -1, no H, exactly two heavy neighbours that
    fall into DISJOINT fragments once the two N-bonds are severed (a ring
    bridging both arms back to N is an in-ring N-anion, a different class).
    A trisubstituted N- (azanediide) is out of scope here.
    """
    from ..assembly.substituent_naming import name_substituent_fragment
    from ..assembly.naming_utils import apply_enclosing_marks, alpha_sort_key
    a = mol.GetAtomWithIdx(anion_idx)
    if (a.GetSymbol() != 'N' or a.GetFormalCharge() != -1
            or a.GetTotalNumHs() != 0 or a.GetIsAromatic()):
        return ''
    neighbours = [nb.GetIdx() for nb in a.GetNeighbors()
                  if nb.GetSymbol() != 'H']
    if len(neighbours) != 2:
        return ''
    rw = Chem.RWMol(mol)
    for nb in neighbours:
        rw.RemoveBond(anion_idx, nb)
    frag = rw.GetMol()
    comps = Chem.GetMolFrags(frag, asMols=False, sanitizeFrags=False)
    idx_to_comp = {}
    for ci, atoms in enumerate(comps):
        for at in atoms:
            idx_to_comp[at] = ci
    if idx_to_comp[neighbours[0]] == idx_to_comp[neighbours[1]]:
        return ''  # arms share a ring back to N -> in-ring N-anion, not azanide
    prefixes = []
    for nb in neighbours:
        sub_atoms = list(comps[idx_to_comp[nb]])
        pref = name_substituent_fragment(mol, sub_atoms, nb, [anion_idx])
        if not pref:
            return ''
        prefixes.append(pref)
    prefixes.sort(key=alpha_sort_key)
    enclosed = ''.join(apply_enclosing_marks(p, -1) for p in prefixes)
    name = f"{enclosed}azanide"
    if not _cation_name_rt_ok(name, mol):
        return ''
    return name


def emit_primary_amine_anide(mol, anion_idx: int, style: str) -> str:
    """ (the Blue Book, ``benzenaminide (PIN)``): a PRIMARY
    amine anion R-[NH-] named on the SYSTEMATIC amine parent, converting the
    ``-amine`` suffix to ``-aminide``.

    The default ionize seam converts ``amine``->``aminide`` on the re-entered
    NEUTRAL name; that already covers a CHAIN amine (``methanaminide``,
    ``ethanaminide``, because the neutral is the systematic ``methanamine`` etc.).
    But an AROMATIC / ring primary amine re-enters as the RETAINED headline
    ``aniline`` -- which carries no ``-amine`` suffix token to convert, and the
    engine never emits the systematic ``benzenamine`` (``rules/benzene.py`` returns
    ``aniline``; ``rules/radicals.py`` fails closed on exactly this gap). So build
    the systematic amine parent here: delete the amine N, name the remaining parent
    hydride (`` -> ``benzene``), append ``amine`` (eliding the parent's
    trailing ``e``), then convert ``amine``->``aminide``.

    FULL-InChIKey RT-gated (0-wrong): a substituted / heteroatom parent whose
    bare (locant-omitted) amine spelling does NOT round-trip -- ``toluenamine``,
    ``pyridinamine`` -- is REJECTED and the caller abstains. Scope: a single
    ACYCLIC primary-amine N (charge -1, exactly one H, exactly one heavy
    neighbour, a carbon) whose parent fragment (N removed) is one connected
    component the pipeline can name. Fail-closed ('') otherwise. Placed AFTER the
    ionize seam so it only ever converts a would-be ABSTAIN (never renames an
    already-valid aminide)."""
    a = mol.GetAtomWithIdx(anion_idx)
    if (a.GetSymbol() != 'N' or a.GetFormalCharge() != -1
            or a.GetTotalNumHs() != 1 or a.IsInRing() or a.GetIsAromatic()):
        return ''
    heavy = [nb for nb in a.GetNeighbors() if nb.GetSymbol() != 'H']
    if len(heavy) != 1 or heavy[0].GetSymbol() != 'C':
        return ''
    rw = Chem.RWMol(mol)
    rw.RemoveAtom(anion_idx)
    frag = rw.GetMol()
    try:
        Chem.SanitizeMol(frag)
    except (RuntimeError, ValueError):
        return ''
    if len(Chem.GetMolFrags(frag)) != 1:
        return ''
    frag_smi = Chem.MolToSmiles(frag)
    if not frag_smi:
        return ''
    try:
        parent = _reenter(frag_smi, style)
    except (RecursionError, ValueError, RuntimeError):
        return ''
    if not parent or _is_malformed_parent(parent):
        return ''
    from .ions import apply_ion_suffix_to_name
    amine_name = (parent[:-1] if parent.endswith('e') else parent) + 'amine'
    aminide = apply_ion_suffix_to_name(amine_name, -1)
    if aminide and _full_inchikey_rt_ok(mol, aminide):
        return aminide
    return ''


def emit_acylium(mol, cation_idx: int, style: str) -> str:
    """ (BB 41623 PIN): name an acylium cation R-C(+)=O.

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
    exactly (a) "maximum number of anionic centers". Mirrors the verified
    _name_oxoacid_anion neutralize loop (ions.py:929-939).

    Cation neutralization is class-dependent (internal notes — which neutral form):
      - PROTON-GAIN cations (protonated amine/onium): the cation has an EXTRA H,
        so removing |charge| H restores the neutral amine/hydride
        (``[NH4+]``->``ammonia``, ``C[NH3+]``->``methylamine``). ``add_h_for_cation
        = False``.
      - HYDRIDE-LOSS cations (carbenium ``ylium``, acylium): the cation is the
        parent hydride MINUS a hydride (H-), so ADDING |charge| H restores the
        parent hydride (``[CH3+]``->``methane``, then ``ane``->``ylium`` =
        ``methylium``;. ``add_h_for_cation = True``.
    Internal (nitro/azide/N-oxide/diazo) charges are LEFT intact. Radical
    electrons are saturated with H (the suffix is re-applied by the caller).
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

     (169.5): the neutral name is an INTERMEDIATE (ionized below), so the
    OPSIN validity gate is bypassed — a malformed intermediate must not be
    suppressed to a descriptive string before the ionize step. select_parent
    (cyclic) / find_principal_chain (acyclic) run INSIDE this call; this is why
    the chokepoint is "re-enter", NOT a direct select_parent call (the SMARTS
    that find the principal group are charge-sensitive and do not match a charged
    atom — IMPLEMENTATION-MAP).
    """
    from ..namer import Orthonym
    return Orthonym(style=style, _disable_opsin_validity_gate=True,
                     **_best_effort_reenter_kwargs()).name(neutral_smi)


def _best_effort_reenter_kwargs() -> dict:
    """ breadth: when the OUTER call is best-effort, name the neutral parent
    best-effort too, so a charged molecule whose neutral parent is nameable ONLY
    under the general/best-effort tier (a complex carboxylate/ammonium/phosphate,
    ~40% of the abstention census) converts instead of abstaining -- the charge is
    incidental, not the blocker. Reads ``best_effort_ctx`` so the PIN/default tier
    (ctx False) re-enters PIN-only exactly as before (gate byte-identical); the
    best-effort name still faces the caller's E1/ certification, so 0-wrong
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
    """ (a phase): re-enter the neutral skeleton with the anion's acid group
    FORCED as the principal characteristic group /. Used only when the
    default re-entry let a senior neutral acid (carboxylic) take the principal slot,
    so the S/P-oxoacid suffix never appeared and the ionize step found no match."""
    from ..namer import Orthonym
    return Orthonym(style=style, _disable_opsin_validity_gate=True,
                     _principal_group_override=principal_fg,
                     **_best_effort_reenter_kwargs()).name(neutral_smi)


def _reenter_gated(neutral_smi: str, style: str) -> str:
    """ B1: re-enter the neutral skeleton with the OPSIN validity gate ON.

    The default ``_reenter`` deliberately DISABLES the gate (: a valid-but-
    unparseable INTERMEDIATE must not be suppressed before the ionize step). But
    the disabled gate also lets an ATOM-DROPPING retained / natural-product name
    through unchecked (benzatropine free base -> ``tropane``, dropping the C3
    diphenylmethoxy). This gated re-entry is the RETRY used by the atom-coverage
    guard: with the gate ON, rejects the atom-dropping retained name and
    the pipeline falls through to the systematic von Baeyer / substitutive parent
    (``3-(diphenylmethoxy)-8-methyl-8-azabicyclo[3.2.1]octane``). Returns
    ``'unknown organic compound'`` (a malformed-parent sentinel) when no covering
    systematic name exists -> the caller then abstains (0-wrong)."""
    from ..namer import Orthonym
    return Orthonym(style=style,
                     **_best_effort_reenter_kwargs()).name(neutral_smi)


def _reenter_amine_forced_no_retained(neutral_smi: str, style: str,
                                      amine_fg: str) -> str:
    """ B2: re-enter the neutral skeleton with the amine FORCED as the
    principal characteristic group AND the RETAINED_NAME dispatch handler
    excluded.

    A retained neutral name whose principal suffix is NOT the amine (trometamol
    is diol-principal) cannot take a valid ``-ium``/``-aminium`` suffix
    (``name_aminium_cation`` yields the OPSIN-unparseable ``trometamolium``).
    Excluding the retained headline lets the systematic amine-principal parent be
    derived instead (``1,3-dihydroxy-2-(hydroxymethyl)propan-2-amine`` ->
    ``...propan-2-aminium``, which round-trips). Returns the neutral systematic
    name, or ``'unknown organic compound'`` / '' if none exists."""
    from ..namer import Orthonym
    try:
        from ..routing.dispatch_table import StoutClass
        excl = frozenset({StoutClass.RETAINED_NAME})
    except Exception:
        excl = frozenset()
    return Orthonym(style=style, _disable_opsin_validity_gate=True,
                     _principal_group_override=amine_fg,
                     _seed_excluded_dispatch_classes=excl,
                     **_best_effort_reenter_kwargs()).name(neutral_smi)


def _heavy_atom_multiset(smiles: str):
    """Counter of heavy-atom element symbols for ``smiles`` (H excluded), or None
    if it cannot be parsed. Used by the atom-coverage guard to detect an
    atom-DROPPING re-entry name (a parseable retained/NP name missing atoms)."""
    from collections import Counter
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    return Counter(a.GetSymbol() for a in m.GetAtoms() if a.GetSymbol() != 'H')


def _reenter_atom_coverage(neutral_name: str, neutral_smi: str) -> Optional[bool]:
    """ B1 atom-coverage guard (the ``covered == all-atoms``
    precondition). Re-parse ``neutral_name`` through OPSIN (PLAIN, neutral-vs-
    neutral -- valid, unaffected by 's unparseable-INTERMEDIATE concern) and
    compare its heavy-atom multiset to ``neutral_smi``:

      * True -- the name's heavy-atom composition EQUALS the fragment's (covered).
      * False -- the name PARSES but its composition differs (the atom-DROP
                 signature: a parseable retained/NP name that silently omits
                 substituents, e.g. ``tropane`` for the benzatropine free base).
      * None -- the name is OPSIN-UNPARSEABLE. This is the legitimate
                 intermediate shape; return None so the caller LEAVES IT as-is
                 (byte-identical, no retry) -- the atom-drop bug always produces a
                 PARSEABLE name, so an unparseable one is never the bug.
    """
    from ..validation.opsin_roundtrip import opsin_parse
    try:
        parsed = opsin_parse(neutral_name)
    except Exception:
        return None
    if not parsed:
        return None
    got = _heavy_atom_multiset(parsed)
    want = _heavy_atom_multiset(neutral_smi)
    if got is None or want is None:
        return None
    return got == want


def _cation_name_rt_ok(name: str, mol) -> bool:
    """Strict 0-wrong RT gate for a hand-derived cation name, via the RELIABLE
    JPype ``opsin_parse`` (NOT the subprocess ``_quaternary_rt_ok``, which fails
    OPEN and would let an unverifiable full-stereo name -- one OPSIN cannot parse
    at ring position 3 -- ship). Parse ``name`` and require a FULL-InChIKey match
    to ``mol``. Fails CLOSED when OPSIN returns None (unparseable / jar absent):
    the caller then falls through / abstains -- never ships an unverified name."""
    if not name:
        return False
    from rdkit.Chem.inchi import MolToInchiKey

    from ..validation.opsin_roundtrip import opsin_parse
    try:
        parsed = opsin_parse(name)
        if not parsed:
            return False
        pm = Chem.MolFromSmiles(parsed)
        if pm is None:
            return False
        return MolToInchiKey(pm) == inchikey_of(mol)
    except Exception:
        return False


def _ring_aza_cation_name(neutral_name: str, mol) -> str:
    """ B1: derive the cationic name for a protonated RING nitrogen from the
    (atom-coverage-verified) neutral von-Baeyer / replacement parent name.

    A ring-N cation cannot use the acyclic ``amine -> aminium`` text transform
    (``name_aminium_cation`` would emit ``...octanium`` -- OPSIN puts the charge on
    the wrong atom). The neutral name for such a nitrogen is a skeletal ``aza``
    replacement (``...8-azabicyclo[3.2.1]octane``); the cation at that same N is
    named by EITHER the parent-hydride ``-ium`` suffix cited at the aza locant
    (``...octan-8-ium``, OR the ``azonia`` replacement (``...8-azonia-
    bicyclo[3.2.1]octane``,. Both spellings are OPSIN-valid (a trace-B1
    ). Build both candidates and RETURN THE FIRST that OPSIN round-trips against
    ``mol`` via ``_cation_name_rt_ok`` (the JPype ``opsin_parse`` path -- full-InChIKey
    identity, **fail-CLOSED** whether the jar is present OR absent; NOT the subprocess
    ``_quaternary_rt_ok``, which fails OPEN when the jar is absent and would ship an
    unverifiable full-stereo name). Any full-stereo name whose descriptor sits at ring
    position 3 fails to parse -> both candidates reject -> '' (the caller abstains;
    NEVER an unverified-stereo ship).

    Returns '' (fall through to the acyclic transform) unless exactly one skeletal
    ``aza`` locant is present and a candidate round-trips."""
    import re as _re2
    # Exactly one skeletal 'aza' replacement (a locant + 'aza' + a ring word);
    # a di-/tri-aza or 0-aza name is out of scope (the correspondence to the single
    # cation site would be ambiguous).
    aza = _re2.findall(r'(?<![0-9a-z])(\d+[a-z]?)-aza(?=[a-z])', neutral_name)
    if len(aza) != 1:
        return ''
    loc = aza[0]
    candidates = []
    # (1) parent-hydride '-ium' at the aza locant: '...octane' -> '...octan-8-ium'.
    if neutral_name.endswith('e'):
        candidates.append(f"{neutral_name[:-1]}-{loc}-ium")
    # (2) 'azonia' skeletal-replacement cation: '{loc}-aza' -> '{loc}-azonia'.
    candidates.append(
        _re2.sub(rf'(?<![0-9a-z]){_re2.escape(loc)}-aza(?=[a-z])',
                 f'{loc}-azonia', neutral_name, count=1))
    for cand in candidates:
        if cand and _cation_name_rt_ok(cand, mol):
            return cand
    return ''


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
    """GUARD 3 / sanity check for a skeletal heteroatom charge.

    The neutralize->re-enter step already runs the SOUND cascade, which
    applies senior-skeletal-atom (the SAME element order) inside the
    re-entry — so for the single-center majority the senior-element parent is
    chosen automatically. This hook exists to make GUARD 3 EXPLICIT and auditable
    (and to refuse a pathological case where the only charged atom is a low-
    seniority element that the neutral cascade could mis-root); it does not
    re-implement. Returns True if the chosen-fragment charge layout is
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
    # Different elements on multiple same-sign centers: re-entry's picks
    # the senior one; accept (the cascade is sound). The element order is recorded
    # here for auditability (the senior element is the parent-bearing one).
    senior = min(elems, key=lambda e: _ELEMENT_SENIORITY.get(e, 999))
    return senior in _ELEMENT_SENIORITY


import threading as _threading
from ..perception.molcache import inchikey_of

_route_reentry = _threading.local()
# Legitimate route_charged nesting (a zwitterion -> its anion parent -> the
# neutral re-entry) is <= 2 levels deep. Deeper recursion means the re-entered
# form re-triggers charged/radical routing WITHOUT converging -- e.g. a radical
# metal atom whose neutralization cannot remove the radical, so the re-entry is
# still a radical: route_charged -> name_radical -> _reenter -> _handle_radical
# -> name_radical -> route_charged ->... forever (the `[99Tc]` 14.5h hang).
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

    : when the amine IS the principal characteristic group the parent
    name ends in '-amine' (or is a bare hydride / 'ammonia'); the cation is the
    '-aminium' suffix -- the proven ``name_aminium_cation`` transform
    (cysteamine -> cysteaminium).

     /: when a SENIOR characteristic group owns the suffix
    (-oic acid / -ol / -one / -amide...), the neutral pipeline already expresses
    the amine as an 'amino' substituent PREFIX, with its locant, N-substituents
    and enclosing marks placed correctly. The protonated nitrogen is then the
    cationic substituent prefix 'azaniumyl' (azanium = NH4+,;
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

    a phase.1 (183 precedent): parse the generated ``name`` back
    through OPSIN and confirm it reconstructs the SAME structure as ``mol`` (strict
    RDKit-canonical identity). A malformed quaternary name therefore fails CLOSED
    (the caller returns '') rather than shipping a structurally-wrong name.

    Fails OPEN (returns True) when the OPSIN jar is absent / the subprocess could
    NOT run (timeout / OSError), mirroring the existing fail-open-on-jar-missing
    convention so CI without OPSIN does not block. Uses the shared 10s-timeout
    ``OpsinOracle._invoke_opsin`` (NO ``-r`` flag — that radical-allowing change is
    Plan 04, not this gate). This is the targeted backstop for the rare
    quaternary class, NOT a new gate for all rows.
    """
    if not name:
        return False
    try:
        from ..assembly.retained_substitution import OpsinOracle
        # Resolve the OPSIN jar the same way the namer does (a phase/169
        # precedent); if it cannot be found the oracle's _jar stays None and
        # _invoke_opsin raises -> the outer except fails OPEN (jar-missing).
        from ..validation.opsin_roundtrip import _find_opsin_jar
        _jar = _find_opsin_jar()
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


def _uronium_rt_ok(name: str, mol) -> bool:
    """InChIKey (resonance-tolerant) OPSIN round-trip backstop for the uronium class.

    Task 8B RC3: the parent cation ``uronium`` represents BOTH tautomeric
    structures of protonated (iso)urea, so a well-formed uronium name and its input can
    have DIFFERENT localised canonical SMILES while denoting the same delocalised cation
    (e.g. ``N,S-dimethyl-N'-phenylthiouronium`` <-> a charge-shifted resonance form).
    A strict canonical-SMILES comparison (``_quaternary_rt_ok``) would reject those
    correct names, so this compares the InChIKey — the project's headline correctness
    oracle, which normalises the resonance/tautomer difference — giving a true 0-wrong
    check for the class. Fails OPEN (True) when OPSIN cannot run (jar missing / timeout),
    mirroring ``_quaternary_rt_ok`` so CI without OPSIN does not block; fails CLOSED
    (False) on a definitive OPSIN rejection or an InChIKey mismatch.
    """
    if not name:
        return False
    try:
        from ..assembly.retained_substitution import OpsinOracle
        from ..validation.opsin_roundtrip import _find_opsin_jar
        _jar = _find_opsin_jar()
        oracle = OpsinOracle(opsin_jar=_jar)
        if oracle._jar is None:
            return True  # jar missing -> fail OPEN
        opsin_smi, ran = oracle._invoke_opsin(name)
    except Exception:
        return True
    if not ran:
        return True
    if not opsin_smi:
        return False
    try:
        from rdkit.Chem import inchi
        op_mol = Chem.MolFromSmiles(opsin_smi)
        if op_mol is None:
            return False
        op_ik = inchi.MolToInchiKey(op_mol)
        mol_ik = inchikey_of(mol)
        return bool(op_ik) and op_ik == mol_ik
    except Exception:
        return False


@_reentry_guarded
def route_charged(mol, style: str = 'pin') -> str:
    """THE single mandatory parent-selection chokepoint (CHOKE-01 / CHOKE-02).

    Generalizes _name_oxoacid_anion to every charged class. Returns the IUPAC
    name, or '' on any failure / out-of-scope shape (caller falls through to the
    legacy cascade — byte-identical contract).

    Pipeline (IMPLEMENTATION-MAP):
      1. metal complex -> '' (salt composition is Plan 04)
      2. multi-fragment (salt / arbitrary) -> '' (salt composition is Plan 04)
      3. classify ionic centers + GUARDS 1-4
      4. neutralize the fragment (+ sanitize)
      5. re-enter Orthonym(style).name(neutral_smi)
      6. apply_ion_suffix_to_name(..., allowed_suffixes=<class subset>, cation_class=<class>)
    """
    if mol is None:
        return ''

    # --- Step 1: metal complex -> Plan 04 (keep the anion-path regression clean).
    # simple-metal-salt composition (<cation word> <anion>) is Plan 04;
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

    # --- Step 2b (charged-species fix, 169.6 caveats, a phase): retained-name-first ONLY for
    # the CATEGORY of retained charged species that LACK a valid systematic PIN — i.e.
    # whose neutralize -> re-name -> re-apply-suffix path produces an OPSIN-unparseable
    # systematic form that the gate then suppresses to 'unknown' (the documented
    # 169.6 regression, audit Dim-08 §B Cause 1; sulfonium, aminoxide/
    # bistriflimide /. This is DELIBERATELY NARROW: species WITH a valid
    # systematic PIN that the chokepoint already produces — alkoxides (-> -olate, the
    # systematic) and carboxylate (poly)anions (-> deferred -ate) — are NOT here
    # and keep their 169.6 chokepoint/defer routing (GUARD 1/2). RT-safe by construction.
    _canon = Chem.MolToSmiles(mol)
    if _canon in _RETAINED_FIRST_CHARGED:
        from ..data.ion_retained_names import get_anion_name, get_cation_name
        _retained = get_cation_name(_canon) or get_anion_name(_canon)
        if _retained:
            return _retained

    # --- Step 3: enumerate + classify ionic / radical centers.
    sites = get_ion_sites(mol)  # excludes internal nitro/azide/N-oxide/diazo
    n_anions = len(sites['anions'])
    n_cations = len(sites['cations'])
    from ..perception.ions import get_radical_sites
    radical_sites = get_radical_sites(mol)
    #: a radical ion whose charge and radical sit on one atom of a
    # mononuclear ionic parent ('trimethylboranuidyl', 'propyloxidaniumyl').
    if n_anions or n_cations:
        from .radicals import _name_mononuclear_radical_ion
        _radical_ion = _name_mononuclear_radical_ion(mol)
        if _radical_ion:
            return _radical_ion

    # The common neutralize/ionize TAIL (Step 4+) is shared by the cation and anion
    # single-sign branches. One late anion-only tail fallback reads ``_single``, which
    # is assigned ONLY inside the `elif n_anions and not n_cations` branch. A
    # single-CATION shape that falls through to the tail with a falsy ``ionized`` hit
    # ``if _single == {'aminide'} …`` with ``_single`` UNBOUND -> an UnboundLocalError
    # (surfaced on a silicon cation; caught upstream as an abstain, so 0-wrong held,
    # but a caught crash is a latent robustness defect). Seed ``_single`` with a no-op
    # default so a cation reaches the final ``return ''`` cleanly. ``anion_is_iminide``
    # and ``anion_override_fg`` ALREADY have safe defaults further down before any tail
    # read; they are re-seeded here only for locality/symmetry (harmless no-ops).
    # Behaviour-preserving: the anion branch overwrites ``_single``, and the only tail
    # branch that EMITS is gated on ``anion_override_fg is not None`` -> a no-op for a
    # cation. No new emission.
    _single: set = set()
    anion_is_iminide = False
    anion_override_fg = None

    if n_anions == 0 and n_cations == 0 and not radical_sites:
        return ''  # nothing charged/radical here (internal-only charge -> neutral)

    # GUARD 4: zwitterion anion-is-parent override. The anion is FORCED
    # as the parent; a separable cation is demoted to an (…azaniumyl)
    # substituent prefix; a skeletal cation is deferred to the legacy
    # path. 169.6-04 (was a Plan-03 detect-and-defer seam).
    if _is_zwitterion(sites):
        return _route_zwitterion(mol, sites, style)

    # W4-I3: R-S+ / R-Se+ sulfanylium/selanylium. RDKit assigns a
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
            # 0-wrong RT gate (mirrors the emit_halogen_onium / emit_onium_hydride_parent
            # siblings below): emit_chalcogen_ylium names its carbon substituent via the
            # shared classify_substituent, so a mis-named ligand (e.g. the cyano fix
            # widening what that helper returns) must be full-InChIKey verified before
            # shipping. [S+]C#N -> cyanosulfanylium is now proven, not lucky.
            if cy and _cation_name_rt_ok(cy, mol):
                return cy

    # (the Blue Book): a disubstituted halogen(III) cation R2X+ (diaryl-
    # iodanium) is named on the halogen parent hydride + '-ium' — the SYSTEMATIC PIN
    # ((C6H5)2I+ -> diphenyliodanium), the cation mirror of the diphenyliodanuide
    # -uide anion. classify_cation returns 'unknown' for a halogen centre and the
    # generic neutralize path cannot build it (drop-charge -> invalid neutral halogen
    # radical; add-H -> the wrong λ-parent), so emit_halogen_onium owns it directly.
    # RT-gated (0-wrong): a mis-built name fails CLOSED. Tightly scoped (halogen, +1,
    # degree>=2) so no other cation class is disturbed.
    if n_cations == 1 and not n_anions:
        from .ions import emit_halogen_onium
        ho = emit_halogen_onium(mol, sites['cations'][0]['atom_idx'])
        if ho and _cation_name_rt_ok(ho, mol):
            return ho

    # Task 8B RC3, the Blue Book): a SUBSTITUTED uronium / thiouronium cation
    # (protonated iso/urea — a C bonded to two N + one O/S, net +1) is named on the
    # retained parent cation 'uronium'/'thiouronium' with N,N'/O/S substituent locants
    # -> N,N'-dimethyl-O-phenyluronium. The neutralize path cannot reach it (the neutral
    # isourea names as a carbamimidate ester). emit_uronium is tightly gated (2 N + one
    # O/S, >=1 substituent) so guanidinium (3 N) and the unsubstituted retained parents
    # never match; the RT gate then fails a mis-built name CLOSED (0-wrong).
    if n_cations == 1 and not n_anions:
        from .ions import emit_uronium
        ur = emit_uronium(mol, sites)
        if ur and _uronium_rt_ok(ur, mol):
            return ur

    # (the Blue Book, "ethylideneoxidanium (PIN)":41463, "acetyloxidanium
    # (PIN)":41474): a SUBSTITUTED mononuclear chalcogen cation R(n)X+ (X = O/S/Se/
    # Te, +1, not in a ring, not bonded to another chalcogen) is named on the parent
    # hydride oxidane/sulfane/selane/tellane + '-ium' with alkyl / acyl / alkylidene
    # substituent prefixes -- the systematic PIN, NOT the '-a'/'-onia'-replacement
    # spelling (CC=[OH+] -> ethylideneoxidanium, not 1-oxaprop-1-en-1-ium). The
    # neutralize/re-enter path cannot build it (the neutral chalcogen is a different
    # constitution), so emit_onium_hydride_parent owns it directly. Tightly scoped
    # (mononuclear chalcogen, +1) and RT-gated (0-wrong): a mis-built name fails
    # CLOSED. Runs after the ylidene/halogen/uronium onium carve-outs above so those
    # narrower classes win first.
    if n_cations == 1 and not n_anions:
        from .ions import emit_onium_hydride_parent
        oh = emit_onium_hydride_parent(mol, sites['cations'][0]['atom_idx'])
        if oh and _cation_name_rt_ok(oh, mol):
            return oh

    # (the Blue Book): a single carbon bearing a DOUBLE (or higher) positive
    # charge is the geminal, one-atom degenerate of the poly-carbenium — two -ylium
    # free valences on ONE skeletal carbon (C[C+2]C -> propane-2,2-bis(ylium)).
    # get_ion_sites reports it as a SINGLE +2 cation, so the >=2-CENTRE poly-ylium
    # branch (in the n_cations>=2 block below) never sees it and the generic
    # neutralize/re-enter seam drops a charge (-> 'propylium', a wrong +1 molecule).
    # emit_poly_cation_ylium (generalized to a per-centre charge multiplicity) builds
    # the bis(ylium) form. RT-gated (0-wrong): a mis-built name fails CLOSED; the
    # emitter itself fails closed on rings / substituted / non-carbon shapes, so
    # every existing single-cation class (onium/ylium carve-outs above, ordinary +1
    # ylium below) is untouched (a +1 carbon does not enter this >=2-charge branch).
    if n_cations == 1 and not n_anions:
        _c0_idx = sites['cations'][0]['atom_idx']
        _c0_atom = mol.GetAtomWithIdx(_c0_idx)
        if _c0_atom.GetSymbol() == 'C' and _c0_atom.GetFormalCharge() >= 2:
            from .ions import emit_poly_cation_ylium
            _py = emit_poly_cation_ylium(mol, [_c0_idx])
            if _py and _cation_name_rt_ok(_py, mol):
                return _py

    # a phase (a trace internal notes Q4): the SAME
    # RDKit valence-model artifact documented above for R-S+/R-Se+ also hits
    # the charge-shifted diazonium TWIN (R-N=N+). Its terminal N+ is degree-1
    # with only a DOUBLE bond (valence contribution 2) against the
    # +1-charged N's required valence of 4, so RDKit fills the shortfall
    # with 2 SPURIOUS radical electrons -- the canonical drawing's proximal
    # N+ (triple bond + single bond to the parent = valence 4, exact) gets 0.
    # Chemically both drawings are the SAME closed-shell diazonium cation
    # (BB. Treat this one spurious site as not a radical at all,
    # so the ordinary single-cation path below (unchanged; still runs
    # _apply_guard3_reorder and every other guard) handles both drawings
    # exactly alike via the existing 'diazonium' -> _name_diazonium branch.
    #
    # Defense-in-depth (a review review, 2026-08-15): the condition MUST match the
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
    # below instead of building a candidate has to catch.
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
    anion_is_iminide = False  #: enables the RT-audited iminide locant omission

    if radical_sites:
        # HETEROATOM-CENTRED radicals / parent-hydride
        # radicals azanyl/sulfanyl/boranyl/azanylidene; amine/imine/amide
        # compound suffixes methanaminyl/propan-1-iminyl/formamidyl;
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
        # carbon counting): neutralize -> re-enter -> append the -yl/
        # -ylidene/-ylidyne suffix (applied AFTER re-entry below). SCOPE: ONLY a
        # carbon-centered ALKYL/-ylidene/-ylidyne radical. The acyl (R-C(=O).),
        # oxyl (R-O.), and aryl subtypes are NOT a plain parent-hydride hydrogen
        # loss (acyl -> -oyl on the acid name; oxyl -> -oxyl; aryl -> the ring
        # radical) and keep their structured radicals.py helpers -> bail here so
        # name_radical / _handle_radical fall through to them.
        if len(radical_sites) >= 2:
            # multi-site free valences on ONE acyclic all-carbon parent.
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
            # 0-wrong RT gate: this emitter names substituents via the shared
            # classify_substituent (widened by the cyano fix) and has no atom-drop
            # veto of its own, so full-InChIKey verify before shipping; '' on
            # failure. Every in-scope diyl/triyl/diylidene name round-trips, so no
            # existing polyvalent radical regresses.
            poly = emit_parent_hydride_polyvalent_suffixes(mol, centers)
            if poly and _full_inchikey_rt_ok(mol, poly):
                return poly
            return ''
        from .radicals import classify_radical
        rinfo = classify_radical(mol, radical_sites[0])
        if rinfo['subtype'] in ('acyl', 'oxyl', 'aryl', 'aminyl', 'thiyl', 'benzylic'):
            return ''  # structured helpers / out of scope -> legacy fallthrough
        # 'benzylic': bail so _handle_radical falls to name_radical, whose
        # (now-canonical) RETAINED_RADICALS lookup ships the retained PIN
        # 'benzyl' instead of the non-PIN 'toluenyl' fallback.
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
        # W4-I3: MULTI-cation handling. The ONLY in-scope multi-cation
        # construction here is a homogeneous poly-AMINIUM -> the bis/tris(aminium)
        # compound-suffix form (mirror of the W4-I2 poly-anion bis(aminide) path):
        # [NH3+]CC[NH3+] -> ethane-1,2-bis(aminium) (BB 42340). Neutralize ALL
        # centres -> re-enter -> map the neutral di/tri-amine suffix to
        # 'bis(aminium)' (NOT 'diaminium',. EVERY other multi-cation
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
            # / the Blue Book): >= 2 carbenium -ylium centres
            # on ONE acyclic all-carbon parent -> suffix multiplication
            # 'parent-<locs>-bis(ylium)' ([CH2+]C[CH2+] -> propane-1,3-bis(ylium),
            # [CH2+][CH2+] -> ethane-1,2-bis(ylium)). emit_poly_cation_ylium fails
            # closed ('') on rings (a ring poly-ylium needs Item-4 ring numbering)
            # and substituted / off-parent shapes -> fall through.
            if ccls == 'ylium':
                from .ions import emit_poly_cation_ylium
                poly_yl = emit_poly_cation_ylium(
                    mol, [c['atom_idx'] for c in sites['cations']])
                if poly_yl:
                    return poly_yl
            #: >= 2 terminal -N#N+ diazonium groups on ONE
            # ring parent -> 'ring-<locs>-bis(diazonium)' (N#[N+]c1ccc([N+]#N)cc1 ->
            # benzene-1,4-bis(diazonium)). _name_poly_diazonium severs every group,
            # names the ring parent, numbers the attachment set, and RT-gates the
            # composed name (0-wrong). Fails closed ('') on non-ring / fused parents.
            if ccls == 'diazonium':
                pdz = _name_poly_diazonium(
                    mol, [c['atom_idx'] for c in sites['cations']], style)
                if pdz:
                    return pdz
            # a phase /.2): a SYMMETRIC bis-quaternary-ammonium
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
            # Case B, the Blue Book): exactly 2 IDENTICAL bare
            # ONIUM units (O/S/P/Se+) on ONE divalent skeletal linker ->
            # '(LINKER)bis(onium)' ([PH3+]c1ccc([PH3+])cc1 -> (1,4-phenylene)-
            # bis(phosphanium)). Mirrors the bis-quaternary linker assembly but
            # allows a benzene phenylene linker and the free onium unit. RT-gated;
            # fails closed ('') on substituted units / non-divalent-linker shapes.
            if ccls == 'onium' and len(sites['cations']) == 2:
                biq = _name_bis_parent_ion_linker(
                    mol, [c['atom_idx'] for c in sites['cations']], 'onium', style)
                if biq:
                    return biq
            return ''
        # W4-I3: a diazonium cation R-N2+ is named by appending
        # 'diazonium' to the parent hydride obtained by SEVERING the whole -N#N+
        # group and capping the attachment with H ([N+](#N)c1ccccc1 -> benzene ->
        # benzenediazonium). The generic neutralize (drop-charge) over-valences the
        # surviving N, so a dedicated sever-and-name emitter owns it. Decline ('')
        # -> fall through.
        if ccls == 'diazonium':
            dz = _name_diazonium(mol, sites['cations'][0]['atom_idx'], style)
            if dz:
                return dz
        # W8-P5 Task 1: an acylium cation R-C(+)=O is named on the
        # RECONSTRUCTED ACID (add -OH), never the generic hydride-loss
        # ('add H' -> aldehyde) path below -- emit_acylium owns it. Mirrors the
        # diazonium interception immediately above.
        if ccls == 'acylium':
            acy = emit_acylium(mol, sites['cations'][0]['atom_idx'], style)
            if acy:
                return acy
        # method (1): a single ONIUM (P+/S+/O+/...) at a SPIRO
        # JUNCTION -- a ring hetero-cation whose four bonds are all ring bonds --
        # cannot be neutralized (the neutral atom is over-valent, SanitizeMol
        # raises) so the generic neutralize -> re-enter seam below abstains. Name
        # it as the neutral heteroatom-spiro PARENT + '-ium' at the cation locant
        # (...phosphaspiro...-ium /...thiaspiro...-ium, the PIN method (1); NOT
        # the 'onia' cationic-replacement form). name_charged_spiro_system is
        # scoped to a spiro-ATOM cation, so it cannot override a working
        # ring-member onium; RT-gate and ship only on a full-InChIKey match.
        if ccls == 'onium' and len(sites['cations']) == 1:
            cat_idx = sites['cations'][0]['atom_idx']
            if mol.GetAtomWithIdx(cat_idx).IsInRing():
                from .spiro import name_charged_spiro_system
                spiro_ium = name_charged_spiro_system(mol, cat_idx)
                if spiro_ium and _cation_name_rt_ok(spiro_ium, mol):
                    return spiro_ium
            # (the Blue Book "General rule for systematically naming cationic
            # centers in parent hydrides";:41388 "1,2,3-trimethyltrisulfan-2-ium
            # (PIN)",:41390 "2,2-dichloro-1,1,1-trimethyldiphosphan-1-ium (PIN)",
            #:41429/:41474 dioxidane family): a CATENATED homonuclear chalcogen /
            # pnictogen cation (trisulfan-2-ium, diphosphan-1-ium, dioxidan-1-ium)
            # or a substituted MONONUCLEAR pnictogen cation (chlorotri(methyl)-
            # phosphanium) is named on the parent hydride + '-ium', NOT the
            # '-a'/'-onia'-replacement form. The emitter declines ('') for
            # a ring / chalcogen-mononuclear / out-of-scope centre, so the existing
            # onium path is byte-identical on decline. RT-gate: ship only on a
            # full-InChIKey match.
            from .ions import emit_parent_hydride_cumulative_suffix
            ph_ium = emit_parent_hydride_cumulative_suffix(mol, cat_idx, 'ium')
            if ph_ium and _cation_name_rt_ok(ph_ium, mol):
                return ph_ium
        # (R-S+/R-Se+ sulfanylium is intercepted earlier, before the radical-ion
        # bail, because RDKit flags the 1-coordinate chalcogen cation as a radical.)
        #.1 + Table 7.4): a QUATERNARY ammonium N (0 H, degree
        # >= 4) CANNOT take the _neutralize_fragment path — removing the lost
        # proton leaves an over-valent neutral N and SanitizeMol raises -> ''
        # (RESEARCH Pitfall 2) — and it is NOT an azaniumyl-prefix case (;
        # azaniumyl is zwitterion-only,. Name it directly via the
        # demote-N -> find_principal_chain -> '-aminium' emitter on the ORIGINAL
        # mol. C[N+](C)(C)C -> N,N,N-trimethylmethanaminium.
        if ccls == 'quaternary' and len(sites['cations']) == 1:
            cat_idx = sites['cations'][0]['atom_idx']
            # W4-I3 /: a quaternary RING N+ is named on the
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
                # method (1): a quaternary onium at a SPIRO JUNCTION
                # (a ring N+ whose four bonds are all ring bonds) has no exocyclic
                # substituent for emit_parent_hydride's DEMOTE branch to sever, and
                # it cannot be neutralized (the neutral 4-bonded N is over-valent).
                # Name it as the neutral aza-spiro PARENT + '-ium' at the cation
                # locant (...azaspiro...-ium, the PIN; NOT...azoniaspiro...), then
                # RT-gate: ship only on a full-InChIKey match, else abstain.
                from .spiro import name_charged_spiro_system
                spiro_ium = name_charged_spiro_system(mol, cat_idx)
                if spiro_ium and _cation_name_rt_ok(spiro_ium, mol):
                    return spiro_ium
            # (the Blue Book "pentamethylhydrazinium (PIN)"): a CATENATED
            # homonuclear N cation (a hydrazinium) is named on the retained N-N
            # parent hydride + '-ium', NOT LINEARIZED by name_quaternary_aminium
            # (CN(C)[N+](C)(C)C -> pentamethylhydrazinium, not the diazabutan-ium
            # replacement). The emitter declines ('') for a mononuclear / ring /
            # out-of-scope N, so the aminium path below is byte-identical on
            # decline. RT-gate: ship only on a full-InChIKey match.
            from .ions import emit_parent_hydride_cumulative_suffix
            q_ium = emit_parent_hydride_cumulative_suffix(mol, cat_idx, 'ium')
            if q_ium and _cation_name_rt_ok(q_ium, mol):
                return q_ium
            from .ions import name_quaternary_aminium
            result = name_quaternary_aminium(mol, sites['cations'][0])
            if not result:
                return ''   # emitter declined (out of scope) -> legacy fallthrough
            # Mono-cation OPSIN RT-gate backstop (183 precedent): a malformed
            # quaternary name fails CLOSED rather than shipping garbage. Fails OPEN
            # only when the OPSIN jar is absent (CI without OPSIN must not block).
            if not _quaternary_rt_ok(result, mol):
                return ''
            return result
        # F- (DD3,: a protonated / N-substituted RING-N cation is
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
            # W4-I5 /: a mono-protonated di-/polyAMINE — the
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
        # Task 8E / RC-A: a ring CARBENIUM cation (hydride loss
        # from a ring carbon) is named on the neutral ring parent with the
        # -ylium suffix at the charged-centre ring locant ([C+]1=CC=CO1 ->
        # furan-2-ylium, C1=C[CH+]C=C1 -> cyclopenta-2,4-dien-1-ylium), NOT the
        # acyclic textual 'ane'->'ylium' transform, which appended a locant-free
        # 'furanylium'/'cyclopentadienylium'. Mirrors the ring-N aminium/
        # quaternary interceptions above: the ring emitter reads the ring
        # numbering + 'e' elision and cites the -ylium at the centre locant. On
        # decline (ring bears an FG, fused shape it cannot number) it returns ''
        # and we fall through to the existing acyclic ylium path (byte-identical
        # to HEAD for those shapes).
        if ccls == 'ylium' and len(sites['cations']) == 1:
            cat_idx = sites['cations'][0]['atom_idx']
            if mol.GetAtomWithIdx(cat_idx).IsInRing():
                from .ions import emit_parent_hydride_cumulative_suffix
                ring_ylium = emit_parent_hydride_cumulative_suffix(
                    mol, cat_idx, 'ylium')
                if ring_ylium:
                    return ring_ylium
        cation_class, allowed_suffixes = _CATION_SPEC.get(ccls, (None, None))
        # ylium / acylium are HYDRIDE-LOSS cations /:
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
        #.2 / Table 3.4): a CARBANION has no neutral FG anchor
        # (it neutralizes to a bare hydride 'hexane' — apply_ion_suffix_to_name finds
        # no carbanion key and returns ''), so it cannot use the
        # _principal_group_override seam. Name it directly via the index-preserving
        # emitter on the ORIGINAL mol: the -ide centre gets the lowest locant
        # (orient_chain), competing with unsaturation/substituents per.
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
            # 0-wrong RT gate: the carbanion emitter names substituents on the
            # index-preserving parent, and a mis-named substituent would ship a
            # WRONG molecule (`N#C[C-](C#N)C#N` -> `1,1,1-trimethylmethanide`
            # before the cyano fix). Full-InChIKey verify every emission; on a
            # mismatch fall through to the systematic tier (never a wrong name).
            if carbanion_name and _full_inchikey_rt_ok(mol, carbanion_name):
                return carbanion_name
            return ''   # declined / RT-rejected -> legacy fallthrough
        # W4-I2: a MULTI-carbanion on one acyclic all-carbon parent
        # hydride -> the '-di/tri-ide' PIN ([C-]#[C-] -> ethynediide, BB 40918).
        # HEAD dropped the charge (-> the retained general 'acetylide'/'acetylene').
        if _acls == {'carbanion'} and len(sites['anions']) >= 2:
            from .ions import emit_poly_carbanion_ide
            poly = emit_poly_carbanion_ide(
                mol, [a['atom_idx'] for a in sites['anions']])
            if poly:
                return poly
            return ''   # emitter declined (out of scope) -> legacy fallthrough
        # F- (DD3,: a skeletal Group-14/15 heteroatom anion
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
        # Case B, the Blue Book): exactly 2 IDENTICAL bare
        # heteroatom-hydride -ide anions (P/As/Si/…) on ONE divalent skeletal
        # linker -> '(LINKER)bis(-ide)' ([PH-]c1ccc([PH-])cc1 -> (1,4-phenylene)-
        # bis(phosphanide)). Mirrors the onium/bis-quaternary linker assembly on
        # the anion side. RT-gated; on decline falls through to the general path
        # (byte-identical to HEAD, which abstains for these shapes anyway).
        if _acls == {'heteroatom_hydride_anion'} and len(sites['anions']) >= 2:
            biq = _name_bis_parent_ion_linker(
                mol, [a['atom_idx'] for a in sites['anions']],
                'heteroatom_hydride_anion', style)
            if biq:
                return biq
        # Case B): >= 2 identical acyl-azanide anions
        # (R-CO-NH-) on ONE di/polyacyl linker -> '<polyacyl>bis(azanide)'
        # ([NH-]C(=O)CCC([NH-])=O -> butanedioylbis(azanide)). The acyl linker
        # is named off the reconstructed neutral poly-acid (as the single
        # _name_acyl_azanide does). RT-gated; on decline falls through unchanged.
        if _acls == {'acyl_azanide'} and len(sites['anions']) >= 2:
            biq = _name_bis_acyl_azanide(
                mol, [a['atom_idx'] for a in sites['anions']], style)
            if biq:
                return biq
        # /: a -uide (hydride-addition) anion — the ate-complex
        # (B(CH3)4- / CH3-SiH4- / (CH3)4P- / (C6H5)2I-). Named by the
        # substituted-'-uide'-parent emitter (cannot neutralize: the hypervalent
        # neutral hydride is invalid). W4-I2: generalized beyond Group 13.
        if _acls == {'uide_anion'} and len(sites['anions']) == 1:
            from .ions import _emit_group13_uide
            uide = _emit_group13_uide(mol, sites['anions'][0]['atom_idx'])
            if uide:
                return uide
            return ''   # emitter declined -> legacy
        # W4-I2 /: a HOMOGENEOUS poly compound-suffix
        # anion (a -1 on each of >=2 O/S/N atoms of the SAME class) uses the
        # olate/thiolate/aminide compound suffix multiplied by 'bis'/'tris' ('bis'
        # NOT 'di',, "to avoid ambiguity"): [NH-]CC[NH-] -> ethane-1,2-
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
        # W4-I5 /: a SINGLE alkoxide/thiolate anion on a chain
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
        #: a single =N- imine anion enables the RT-audited iminide
        # locant omission after the suffix seam (butan-1-iminide -> butaniminide).
        anion_is_iminide = (_single == {'iminide'} and len(sites['anions']) == 1)
        #.3 (/): charge-first PCG on the ORIGINAL (un-neutralized) mol.
        # The actually-ionized senior acid class anchors the name; a neutral
        # group of higher seniority is demoted to a prefix. Subsumes the old
        # per-site _ANION_PRINCIPAL_FG.get(...) lookup (same dict + seniority order).
        # The carbanion short-circuit above already returned; classify_charged_pcg
        # returns None for carbanion/alkoxide-only anyway (== the old.get == None),
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

    # --- Step 5b (B1): ATOM-COVERAGE GUARD for the aminium re-entry.
    # ``_reenter`` runs with the OPSIN validity gate DISABLED , which also
    # lets an atom-DROPPING retained / natural-product name through unchecked
    # (benzatropine free base -> ``tropane``, dropping the C3 diphenylmethoxy;
    # idx10 -> ``1,2-bis(benzoyloxy)benzene``, dropping the whole amine side chain).
    # Before the ``-ium`` suffix is glued on, verify the neutral name covers ALL
    # heavy atoms of the fragment; on a PARSEABLE shortfall, retry the re-entry with
    # the gate ON (then rejects the atom-dropper and the pipeline derives
    # the systematic parent). Scoped to the aminium branch so no other charged class
    # changes. This closes the gate-disabled atom-drop door -- a 0-wrong FIX.
    if cation_kind == 'aminium':
        if _reenter_atom_coverage(neutral_name, neutral_smi) is False:
            try:
                retry = _reenter_gated(neutral_smi, style)
            except (RecursionError, ValueError, RuntimeError):
                retry = ''
            if (retry and not _is_malformed_parent(retry)
                    and _reenter_atom_coverage(retry, neutral_smi) is not False):
                neutral_name = retry
            else:
                return ''  # no covering systematic name -> abstain (0-wrong)

    # --- Step 6: re-apply the class-correct ionic / radical suffix.
    if radical_suffix is not None:
        _radical_center_idx = radical_sites[0]['atom_idx']
        #: an acyl radical (a radical centre with a double bond to a
        # chalcogen or N, e.g. (CH3)2P(=O). 'dimethylphosphinoyl (PIN)',
        # the Blue Book) is named from its acid, not by a suffix on the
        # neutral name ('dimethyl-lambda5-phosphanonyl'). Strict round trip inside.
        from .radicals import _name_acyl_radical_from_acid, _name_carbon_radical
        _acyl_name = _name_acyl_radical_from_acid(mol, _radical_center_idx)
        if _acyl_name:
            return _acyl_name
        #: a monovalent carbon radical is named like its substituent
        # group ('cyanomethyl', not 'cyanomethanyl'); strict round trip inside.
        if radical_suffix == 'yl':
            _c_name = _name_carbon_radical(mol, _radical_center_idx)
            if _c_name:
                return _c_name
        # / Table 3.4: radicals are named as SUBSTITUENT GROUPS. A SIMPLE
        # unbranched terminal radical keeps the contracted retained form WITHOUT a
        # locant — the Blue Book lists 'CH3-CH2• ethyl (PIN)' (the Blue Book
        # Table 3.4 example), not 'ethan-1-yl'; likewise 'methyl'/'propyl'. So the
        # proven textual contraction (_apply_radical_suffix) IS the PIN for
        # those. It is ALSO the correct path for the single-carbon -ylidene/-ylidyne
        # (methylidene/methylidyne) the primitive would over-spell 'methanylidene'.
        if _is_simple_terminal_radical(mol, _radical_center_idx):
            return _apply_radical_suffix(neutral_name, radical_suffix)
        # NON-terminal / branched / unsaturated radical: the centre needs a
        # first-class locant / lowest-locant for the free
        # valence, competing with unsaturation/substituents per / —
        # exactly like the.2 carbanion -ide centre (the Blue Book
        # 'ethan-2-id-1-yl (PIN)'). Route through the 184-01 primitive so
        # CC[CH]CC -> 'pentan-3-yl', not the locant-less 'pentyl'.
        # CRITICAL (Pitfall 1): pass the ORIGINAL mol + the centre index in the
        # SAME index space — the primitive owns the atom-index-survival handling
        # (it H-saturates / re-numbers internally); do NOT canonicalize mol first.
        from .ions import emit_parent_hydride_cumulative_suffix
        emitted = emit_parent_hydride_cumulative_suffix(
            mol, _radical_center_idx, radical_suffix)
        # The same round-trip gate as the carbanion call site: the emitter and
        # the textual fallback both build a name the gate has not seen, and a
        # wrong radical site shares the input's full InChIKey.
        if emitted and _full_inchikey_rt_ok(mol, emitted):
            return emitted
        # Fallback: the primitive declined or its name did not round-trip ->
        # the textual form, verified the same way.
        textual = _apply_radical_suffix(neutral_name, radical_suffix)
        return textual if textual and _full_inchikey_rt_ok(mol, textual) else ''

    # (CARBOXYLATE anions are deferred to the proven path earlier; they never
    # reach this point — see the carboxylate guard in Step 3.)

    # AMINIUM (protonated amine, reuses the PROVEN name_aminium_cation
    # transform on the re-entered amine name (amine->aminium, ammonia->ammonium).
    # This is the existing _name_aminium_systematic primary path, generalized into
    # the funnel — so deleting the stub's carbon-counting FALLBACK (Task 2) keeps
    # the aminium output byte-identical (methylaminium / pyrrolidineium etc.).
    if cation_kind == 'aminium':
        # B1: a protonated RING N (reached here when the ring-aware emitters at
        # Step 3 declined -- a bridged / von-Baeyer bicyclic they cannot number)
        # takes the '-ium'/'azonia' cationic form cited at the aza locant, derived
        # from the (coverage-verified) neutral replacement name and RT-verified.
        # The acyclic amine->aminium text transform would mis-place the charge
        # (benzatropine -> 'tropanium'/'...octanium', a different molecule).
        if len(sites['cations']) == 1:
            _cat_idx = sites['cations'][0]['atom_idx']
            if mol.GetAtomWithIdx(_cat_idx).IsInRing():
                ring_cat = _ring_aza_cation_name(neutral_name, mol)
                if ring_cat:
                    return ring_cat
        # (a phase): a senior-group protonated amine becomes an 'azaniumyl'
        # substituent prefix /, not 'ium' appended to the parent;
        # a principal amine keeps the proven '-aminium' suffix.
        naive = _aminium_or_azaniumyl(neutral_name, len(sites['cations'])) or ''
        # B2 (aminium-suffix correctness): the naive transform blindly appends
        # 'ium' to whatever neutral name it is handed. When that neutral name is a
        # RETAINED name whose principal suffix is NOT the amine (trometamol is
        # diol-principal), the result is OPSIN-unparseable ('trometamolium').
        # Verify the naive name round-trips; if it does NOT (and OPSIN is present),
        # re-derive from the SYSTEMATIC amine-principal parent (retained handler
        # excluded) and ship that only if IT round-trips. In a jar-absent config
        # BOTH checks fail -> the naive name is kept (byte-identical). This branch
        # never regresses an already-valid aminium (it fires only when the naive
        # name already failed RT). NOTE: on the fall-through (`return naive`) this
        # function CAN still return an unverified acyclic name in a jar-absent
        # config; the 0-wrong backstop for that path is the top-level gate,
        # NOT this branch (which only ever ships an RT-verified re-derivation).
        if (len(sites['cations']) == 1 and naive
                and not _cation_name_rt_ok(naive, mol)):
            cat = mol.GetAtomWithIdx(sites['cations'][0]['atom_idx'])
            n_c = sum(1 for nb in cat.GetNeighbors() if nb.GetSymbol() == 'C')
            amine_fg = {1: 'primary_amine', 2: 'secondary_amine',
                        3: 'tertiary_amine'}.get(n_c)
            if amine_fg is not None:
                try:
                    sys_neutral = _reenter_amine_forced_no_retained(
                        neutral_smi, style, amine_fg)
                except (RecursionError, ValueError, RuntimeError):
                    sys_neutral = ''
                if sys_neutral and not _is_malformed_parent(sys_neutral):
                    rederived = _aminium_or_azaniumyl(
                        sys_neutral, len(sites['cations'])) or ''
                    if rederived and _cation_name_rt_ok(rederived, mol):
                        return rederived
        return naive

    ionized = apply_ion_suffix_to_name(
        neutral_name, total_charge,
        allowed_suffixes=allowed_suffixes,
        cation_class=cation_class,
    )
    if ionized:
        #: an -iminide name inherits the neutral imine's locants
        # (butan-1-iminide, P,P,P-trimethyl-...); the =N- anion removes the N-H, so
        # licences the omission the neutral could not claim. RT-audited.
        if anion_is_iminide:
            ionized = _iminide_omit_locants(ionized, mol)
        return ionized

    # M4: the suffix-swap above has no ``-amine``/``-imine``
    # token to convert for a disubstituted secondary-N-anion (its N is emitted as
    # a prefix / ring-N, not an ``-amine`` suffix), so it returned ''. Before
    # abstaining, try the disubstituted-azanide construction ``(A)(B)azanide``.
    # Placed AFTER the ionize seam declines (never before), so no currently-named
    # aminide changes name -- this only converts a would-be abstain, RT-gated.
    if _single == {'aminide'} and len(sites['anions']) == 1:
        az = emit_secondary_amine_azanide(
            mol, sites['anions'][0]['atom_idx'], style)
        if az:
            return az
        #: the ionize seam also declines for a PRIMARY aromatic/ring
        # amine anion whose neutral is the RETAINED `aniline` (no `-amine` suffix
        # token to convert). Build the systematic amine parent (`benzenamine`) and
        # convert to `benzenaminide`, RT-gated. Only converts a would-be abstain.
        pa = emit_primary_amine_anide(
            mol, sites['anions'][0]['atom_idx'], style)
        if pa:
            return pa

    # (a phase): the first re-entry let a SENIOR neutral acid (carboxylic,
    # take the principal slot, so the anion's S/P-oxoacid suffix never appeared
    # and the ionize match failed. Per / the CHARGED group IS the principal
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
    """ /: TRUE iff the radical is a SIMPLE substituent-group shape
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
    # - no rings, no heteroatoms, no multiple bonds;
    # - every carbon has at most 2 carbon neighbours (a straight chain);
    # - the radical centre is a terminus (<= 1 carbon neighbour) or a lone C.
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
    """Apply the radical suffix to a re-entered neutral parent name.

    Replaces the deleted radicals.py carbon-counting. Structured, general:
      - alkane 'ane' -> 'yl'/'ylidene'/'ylidyne' (methane->methyl;.
      - otherwise elide a single trailing 'e' before the consonant-initial
        suffix (no double 'e'); append directly if there is none.
    Returns '' if the parent name is empty (caller falls through).
    """
    if not neutral_name:
        return ''
    name = neutral_name
    # (the Blue Book): other parent hydrides take 'yl' "eliding
    # the final letter 'e'", so 'methylborane' -> 'methylboranylidene'
    # ('boranylidene (preselected prefix) (not borylidene)',:15884). Only the
    # Group-14 hydrides and alkanes drop the whole 'ane' (silyl, methyl).
    from .radicals import _ELIDE_E_HYDRIDE_RADICAL
    if any(name.endswith(h) for h in _ELIDE_E_HYDRIDE_RADICAL.values()):
        return name[:-1] + radical_suffix
    if name.endswith('ane'):
        return name[:-3] + radical_suffix       #: methane->methyl
    if name.endswith('e'):
        return name[:-1] + radical_suffix       # elide trailing e (benzene->benzyl-shape)
    return name + radical_suffix
