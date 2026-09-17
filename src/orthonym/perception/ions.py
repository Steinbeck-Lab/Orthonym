"""
Ion and radical detection for IUPAC nomenclature.

This module provides functions to detect and classify charged and radical species.
Detection happens early in the naming pipeline to route molecules to the correct
naming path (neutral, ion, zwitterion, salt, or radical).

IUPAC 2013 rules:
- Cations: named with suffix -ium or -ylium
- Anions: named with suffix -ide, -ate, or -uide
- Radicals: named with suffix -yl
- Salts: named as "cation anion" (e.g., sodium acetate)
- Zwitterions: named as neutral compounds with +/- indicated
"""

import weakref
from typing import Any, Dict, List, Set

from rdkit import Chem

from .molcache import atoms_of, bonds_of  # audit 2026-09-03 (S2): per-call atom/bond tuples
from .molcache import inchikey_of

# IUPAC: prefix-only groups with internal formal charges.
# These are bonding features, NOT ionic charges.
_INTERNAL_CHARGE_SMARTS = [
    Chem.MolFromSmarts('[NX3+](=O)[O-]'),       # nitro
    Chem.MolFromSmarts('[n+][O-]'),               # aromatic N-oxide
    Chem.MolFromSmarts('[N+;!a][O-]'),            # aliphatic N-oxide (cyclic + acyclic)
    Chem.MolFromSmarts('[N;+0]=[N+]=[N-]'),        # organic azide (NOT azide anion [N-]=[N+]=[N-])
    Chem.MolFromSmarts('[#6]=[N+]=[N-]'),         # diazo
    Chem.MolFromSmarts('[C-]#[N+]'),              # isocyanide R-[N+]#[C-]
]
# Filter out any None from failed SMARTS compilation
_INTERNAL_CHARGE_SMARTS = [p for p in _INTERNAL_CHARGE_SMARTS if p is not None]


# O, S, Se, Te. This is not a hand-picked element list: it is the Blue Book's
# OWN enumeration of the class in "Phosphine oxides and chalcogen
# analogues" (":43041") -- "Chalcogen analogues are phosphine sulfides,
# phosphine selenides, and phosphine telluride (where O is replaced by S, Se,
# and Te, respectively)" -- and in (":43008") "chalcogen analogues
# are amine sulfides, imine selenides, etc. (where O is replaced by S, Se, or
# Te)". The ANION element is what the Blue Book uses to draw the boundary; see
# _semipolar_chalcogenide_atoms.
_CHALCOGEN_ATOMIC_NUMS = frozenset({8, 16, 34, 52})

# Raising a semipolar single bond to the multiple bond of the uncharged form.
_BOND_ORDER_UP = {
    Chem.BondType.SINGLE: Chem.BondType.DOUBLE,
    Chem.BondType.DOUBLE: Chem.BondType.TRIPLE,
}


def _semipolar_chalcogenide_atoms(mol) -> Set[int]:
    """Atoms of a semipolar (dative) ``X(+)-A(-)`` chalcogenide pair.

    ** "DIPOLAR COMPOUNDS"** (heading, ``the Blue Book``): *"Dipolar
    compounds are electrically neutral molecules carrying a negative and a
    positive charge in at least one of their major canonical resonance
    structures.... 1,2-Dipolar compounds have the opposite charges on adjacent
    atoms."*

    ** "'Ylides'"** (heading, ``:42509``) explains why the SMARTS list
    above was ever sufficient, and why it stopped being so -- the decisive
    clause is the last one: *"If 'X' is a saturated atom of an element from the
    second row of the periodic system, the 'ylide' is commonly represented by a
    charge-separated form; if 'X' is a third, fourth, etc. row element uncharged
    canonical forms are usually shown, RmX=YRn."* A second-row cation (N) has
    no uncharged depiction, so nitro / N-oxide / azide / diazo are ALWAYS drawn
    charge-separated and each earned an explicit SMARTS. A third/fourth-row
    cation (P, S, As, Se, Sb, Te, I...) is *usually* drawn uncharged -- which is
    why no SMARTS was ever written for it -- but nothing prevents an input from
    being drawn charge-separated, and when it is, the pair used to read as a
    genuine ionic centre.

    **** (heading ``:43041``) settles what the pair means: *"Phosphine
    oxides have the generic formula R3P+ -O- <-> R3P=O."* The Blue Book's own
    double-headed arrow says the two depictions are one compound, and *"Method
    (3) leads to preferred IUPAC names"* makes the NEUTRAL name (a
    ``l5``-phosphanone) the PIN. Likewise **** (``:43015``): *"Method
    (2) leads to preferred IUPAC names when one amine oxide is present....
    Hence, zwitterionic compounds are never PINs"*.

    THE BOUNDARY -- the Blue Book draws it by the ANION, and the other side of
    it must keep its charges VISIBLE:

    * anion on CARBON is an ylide. **** ``:42513``: *"Method (1) is
      applicable to all 'ylides' and leads to preferred IUPAC names"*, method
      (1) being *"as zwitterionic compounds"*. So an ylide's PIN IS the
      zwitterion name -- masking it would emit a valid but non-preferred
      ``l5`` name. (Measured: ``C[P+](C)(C)[CH2-]`` already emits the correct
      ``(trimethylphosphaniumyl)methanide``.)
    * anion on NITROGEN under a nitrogen cation is an amine imide.
      **** (heading ``:43022``): *"Method (1) leads to preferred IUPAC
      names"*, method (1) being *"as a zwitterion based on hydrazine"*.
      (Measured: already emits ``1,2,2,2-tetramethylhydrazin-2-ium-1-ide``.)
    * anion on a CHALCOGEN is the oxide / chalcogenide class -- neutral PIN, so
      the charges are internal. That is this function.

    Two guards keep it from swallowing a genuine oxoanion, and neither is a
    count standing in for a structure proof:

    1. **Local charge balance.** The cation's positive charge must be exactly
       cancelled by the terminal chalcogenide anions bonded to it, per 's
       "electrically neutral" above. ``[O-][I+]([O-])(O)(O)(O)O`` puts two
       ``O-`` on a ``+1`` iodine and ``[O-][Cl+3]([O-])([O-])[O-]`` four on a
       ``+3`` chlorine; both are genuine anions and both are refused here.
    2. **The uncharged form must denote the SAME SPECIES.** The pair is
       rewritten as the multiple bond of the uncharged depiction and the result
       must both sanitise and carry the same standard InChIKey. This is what
       replaces the cation element list a local workaround used to carry: an
       element list cannot tell a semipolar oxide from an oxoanion, whereas this
       proof is exactly the claim being relied on -- that naming the neutral
       form names the input molecule. ``CC1CO[PH+](C1)[O-]`` and
       ``CC1CP(OC1)=O`` share ``an InChIKey``, which is why
       ``4-methyl-2-oxo-1,2-oxaphospholane`` is the correct name for it.

    Fails closed: any parse/sanitise/InChI failure leaves the charges visible.
    """
    # Cheap pre-filter -- the overwhelming majority of molecules leave here.
    by_cation: Dict[int, List[tuple]] = {}
    for bond in bonds_of(mol):
        begin, end = bond.GetBeginAtom(), bond.GetEndAtom()
        for cation, anion in ((begin, end), (end, begin)):
            if cation.GetFormalCharge() < 1 or anion.GetFormalCharge() != -1:
                continue
            if anion.GetAtomicNum() not in _CHALCOGEN_ATOMIC_NUMS:
                continue
            # Terminal and hydrogen-free: an oxide, not a hydroxide or a bridge.
            if anion.GetDegree() != 1 or anion.GetTotalNumHs():
                continue
            by_cation.setdefault(cation.GetIdx(), []).append(
                (anion.GetIdx(), bond.GetIdx())
            )
    if not by_cation:
        return set()

    # Guard 1: local charge balance "electrically neutral").
    balanced = {
        idx: pairs
        for idx, pairs in by_cation.items()
        if mol.GetAtomWithIdx(idx).GetFormalCharge() == len(pairs)
    }
    if not balanced:
        return set()

    try:
        reference_key = inchikey_of(mol)
    except Exception:  # noqa: BLE001 -- perception must never raise
        return set()
    if not reference_key:
        return set()

    internal: Set[int] = set()
    for cation_idx, pairs in balanced.items():
        rw = Chem.RWMol(mol)
        rewritten = True
        for anion_idx, bond_idx in pairs:
            bond = rw.GetBondWithIdx(bond_idx)
            raised = _BOND_ORDER_UP.get(bond.GetBondType())
            if raised is None:
                rewritten = False
                break
            bond.SetBondType(raised)
            _neutralize_pinning_hydrogens(rw.GetAtomWithIdx(anion_idx))
        if not rewritten:
            continue
        _neutralize_pinning_hydrogens(rw.GetAtomWithIdx(cation_idx))
        candidate = rw.GetMol()
        try:
            Chem.SanitizeMol(candidate)
            # Guard 2: the uncharged depiction must be the SAME SPECIES.
            if Chem.MolToInchiKey(candidate) != reference_key:
                continue
        except Exception:  # noqa: BLE001 -- an invalid rewrite proves nothing
            continue
        internal.add(cation_idx)
        internal.update(anion_idx for anion_idx, _ in pairs)
    return internal


def _neutralize_pinning_hydrogens(atom) -> None:
    """Zero an atom's formal charge while holding its hydrogen count fixed.

    Raising the bond order changes the implicit-H count RDKit would compute, so
    the total H is pinned as explicit first. Without this the rewritten molecule
    silently loses the ``H`` of a ``[PH+]`` and the InChIKey comparison rejects a
    pair that is in fact semipolar."""
    atom.SetNumExplicitHs(atom.GetTotalNumHs())
    atom.SetNoImplicit(True)
    atom.SetFormalCharge(0)


_ICA_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _get_internal_charge_atoms(mol) -> Set[int]:
    """Per-mol memoized wrapper (perf 2026-08-28): ~104 calls/mol measured. Pure
    function of the molecule (SMARTS + structural proof, no context/env). Keyed by
    the RDKit Mol; returns a COPY of the cached set so the read-only callers can
    never corrupt the cache. Real work is in _get_internal_charge_atoms_impl."""
    cached = _ICA_CACHE.get(mol)
    if cached is None:
        cached = _get_internal_charge_atoms_impl(mol)
        try:
            _ICA_CACHE[mol] = cached
        except TypeError:
            return set(cached)
    return set(cached)


def _get_internal_charge_atoms_impl(mol) -> Set[int]:
    """Return atom indices whose formal charges are bonding features, not ionic.

    Two complementary mechanisms, because the Blue Book itself describes the
    class in two ways ``:42509``, quoted in
    ``_semipolar_chalcogenide_atoms``):

    1. Named prefix-only groups whose cation is a SECOND-ROW element and which
       therefore have no uncharged depiction at all -- nitro, N-oxide, azide,
       diazo, per IUPAC Matched by SMARTS.
    2. semipolar ``X(+)-A(-)`` chalcogenides, whose cation is a
       third/fourth-row element and which DO have an uncharged depiction. There
       is no closed list of these, so they are proven structurally rather than
       enumerated.

    Note: Nitroso (N=O) is excluded -- no formal charges in standard SMILES.
    Note: Only organic azides [N]=[N+]=[N-] are filtered, NOT the azide anion
    [N-]=[N+]=[N-] which is a genuine ion.

     B3 (root cause, mirrors errors.py's carbon-free structural honesty
    floor): every class above is, by its own / definition, a
    SUBSTITUENT GROUP hung off a carbon-bearing organic skeleton -- Table 5.1
    lists nitro/N-oxide/azide/diazo as prefix-only groups, never as a
    whole-molecule anion word on their own. Two of the five SMARTS above
    (aliphatic N-oxide ``[N+;!a][O-]``, nitro ``[NX3+](=O)[O-]``) carry no
    constraint on the group's OTHER substituent, so they also match a bare,
    carbon-free oxoanion drawn with the same formal-charge pattern (nitrate's
    ``O=[N+]([O-])[O-]`` matches nitro on one reading and aliphatic N-oxide on
    the other -- verified). That falsely marks nitrate's own three charge
    centres " internal", so ``get_ion_sites`` reports it as carrying NO
    ionic sites at all and ``route_charged`` bails before the correct
    ``_name_inorganic_oxoacid_anion`` ever runs (V36-a trace-B3 c). A carbon-
    free molecule is never a substituent on anything -- it IS the whole ion --
    so short-circuit to "nothing is internal" for it. This is deliberately
    NOT a per-SMARTS carbon constraint (e.g. requiring `[#6]` on the nitro
    pattern) because that would also exclude a genuine alkyl NITRATE ESTER
    substituent (``CCO[N+](=O)[O-]`` -> "nitrooxyethane", whose nitro-nitrogen
    substituent is an ester oxygen, not carbon) -- verified regression risk,
    caught empirically before this fix shipped.
    """
    if not any(atom.GetAtomicNum() == 6 for atom in atoms_of(mol)):
        return set()
    internal: Set[int] = set()
    for pat in _INTERNAL_CHARGE_SMARTS:
        for match in mol.GetSubstructMatches(pat):
            for idx in match:
                if mol.GetAtomWithIdx(idx).GetFormalCharge() != 0:
                    internal.add(idx)
    internal |= _semipolar_chalcogenide_atoms(mol)
    internal |= _resonance_twin_internal_atoms(mol)
    return internal


def _resonance_twin_internal_atoms(mol) -> Set[int]:
    """Resonance-shifted azide/diazo drawings the fixed SMARTS above miss.

    ``[N;+0]=[N+]=[N-]`` (azide) and ``[#6]=[N+]=[N-]`` (diazo) both assume
    ONE literal bond-order pattern; RDKit does not normalise resonance forms,
    so the charge-separated twin (``R-[N-]-[N+]#N``) is invisible to them and
    used to fall through to the zwitterion path (a phase a trace,
    `internal notes` Q1/Q2). ADDITIVE: this is
    consulted alongside the SMARTS above, not instead of them -- a molecule
    the SMARTS already handle just gets the same atoms added again to a set
    (a no-op).

    Diazonium is deliberately EXCLUDED here: it is a genuine external cation
    (net charge != 0), not a internal bonding feature, and both its
    drawings are already routed correctly by ``detect_species_type``'s
    non-zero-net-charge branch -- its remaining bug is downstream, in
    ``rules.charged_router._name_diazonium``'s attach-atom walk, not here.
    """
    from ..data.resonance_templates import find_resonance_chains
    internal: Set[int] = set()
    for cls, chain in find_resonance_chains(mol):
        if cls not in ('azido', 'diazo'):
            continue
        for idx in chain:
            if mol.GetAtomWithIdx(idx).GetFormalCharge() != 0:
                internal.add(idx)
    return internal


def detect_species_type(mol) -> str:
    """Memoising front of:func:`_detect_species_type_impl` (perf lever A7, 2026-09-13).

    The dispatch predicates (``_is_salt``, ``_is_zwitterion``, ``_is_organometallic``,
    ``_is_cation_quaternary``,...) and ``_perceive`` call this about 13 times per pipeline
    pass on the SAME Mol (30,038 calls per 300 molecules, 3.5 s). The value depends on the
    atoms' formal charges and radical electrons, which can be edited in place on a
    ``Chem.Mol``, so that live signature is part of the memo key; connectivity cannot change
    without an RWMol, which molcache never caches. Outside a naming scope this is a plain call.
    """
    try:
        from .molcache import cached_by_key
        sig = tuple((a.GetIdx(), a.GetFormalCharge(), a.GetNumRadicalElectrons())
                    for a in atoms_of(mol)
                    if a.GetFormalCharge() != 0 or a.GetNumRadicalElectrons() > 0)
    except Exception:
        return _detect_species_type_impl(mol)
    return cached_by_key(mol, "species_type", sig, lambda: _detect_species_type_impl(mol))


def _detect_species_type_impl(mol) -> str:
    """
    Detect the type of charged/radical species.

    Classification priority:
    1. Radical - any atom with unpaired electrons
    2. Salt - multiple fragments with opposite charges
    3. Ion - net non-zero charge (single fragment)
    4. Zwitterion - net zero charge but has both + and - atoms
    5. Neutral - no charges or radicals

    Args:
        mol: RDKit Mol object

    Returns:
        One of: 'radical', 'salt', 'ion', 'zwitterion', 'neutral'

    Example:
        >>> mol = Chem.MolFromSmiles('[NH4+]')
        >>> detect_species_type(mol)
        'ion'
        >>> mol = Chem.MolFromSmiles('[Na+].[O-]C(C)=O')
        >>> detect_species_type(mol)
        'salt'
    """
    if mol is None:
        return 'neutral'

    # Gather charge/radical info in one pass
    net_charge = Chem.GetFormalCharge(mol)
    has_any_charge = False
    has_radical = False
    for atom in atoms_of(mol):
        if atom.GetFormalCharge() != 0:
            has_any_charge = True
        if atom.GetNumRadicalElectrons() > 0:
            has_radical = True

    # If molecule has formal charges, prioritize ionic classification
    # over radical detection. RDKit may assign radical electrons to
    # certain charged heteroatoms (e.g., [SeH+] gets 2 radical electrons).
    if has_any_charge:
        # Get molecular fragments for salt detection
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        if len(frags) > 1:
            has_pos = any(Chem.GetFormalCharge(f) > 0 for f in frags)
            has_neg = any(Chem.GetFormalCharge(f) < 0 for f in frags)
            if has_pos and has_neg:
                return 'salt'

        if net_charge != 0:
            # 169.6-04 (Task 3): the >10-HA size-cutoff band-aid that
            # reclassified a large single-charge ion (protonated amine /
            # quaternary N) as ``neutral`` was REMOVED. It DROPPED the charge
            # (``CCCCCCCCCCCC[NH3+]`` -> ``dodecane``, discarding the amine).
            # route_charged now names large ions structurally (neutralize ->
            # re-enter -> re-apply the ionic suffix: dodecan-1-aminium), so the
            # charge is preserved. Gate: the large-NEUTRAL negative-test set
            # (tests/unit/test_size_cutoff_removal.py) stays byte-identical
            # because this cutoff only ever fired on a CHARGED species (measured,
            # RESEARCH A2). The OPSIN self-test 500 byte-identical gate confirms
            # no neutral regression.
            return 'ion'

        # Net charge is 0 but has charges -> possible zwitterion
        # Fall through to zwitterion check below
    elif has_radical:
        # No charges at all, just radical electrons -> radical
        return 'radical'

    # If net charge is 0 but all charged atoms are internal (nitro, N-oxide,
    # azide, diazo), molecule is neutral — not a zwitterion.
    if has_any_charge and net_charge == 0:
        internal_atoms = _get_internal_charge_atoms(mol)
        all_charged = {a.GetIdx() for a in atoms_of(mol) if a.GetFormalCharge() != 0}
        if all_charged and all_charged.issubset(internal_atoms):
            return 'neutral'

    # 169.6-04 (Task 3): the >20-HA quaternary-N zwitterion size-cutoff band-aid
    # (which reclassified large quat-N net-zero zwitterions — phosphatidyl-
    # cholines etc. — as ``neutral``) was REMOVED. name_zwitterion now delegates
    # to route_charged GUARD 4 anion-is-parent), and where GUARD 4
    # declines (the multifunctional phospholipid case) it falls through to the
    # neutral-form path — so the charge is no longer silently dropped at
    # perception time. Gate (RESEARCH A2, measured): the self-test-500 only
    # contains nitro compounds at >20 HA, and those are already returned
    # ``neutral`` by the all-internal-charge check above, so the
    # byte-identical gate is unaffected by this removal.

    # Check for zwitterion (net zero but has both + and - atoms)
    # EXCLUDE functional groups with internal charges (nitro, azide, etc.)
    # These are not true zwitterions in IUPAC nomenclature sense
    if _has_true_zwitterion_character(mol):
        return 'zwitterion'

    return 'neutral'


def _has_true_zwitterion_character(mol) -> bool:
    """
    Determine if a molecule has true zwitterionic character.

    A true zwitterion has separated positive and negative sites that are
    NOT part of a single functional group with internal charge distribution.

    Excludes:
    - Nitro groups: [N+](=O)[O-] - internal charge distribution
    - Azide groups: [N-]=[N+]=[N-] - internal charge distribution
    - N-oxides: [N+][O-] directly bonded - internal charge distribution
    - Sulfonyl groups with charge separation

    True zwitterions:
    - Amino acid zwitterions: [NH3+]... [COO-] separated by carbon(s)
    - Betaines: [N+](C)(C)(C)....[O-] separated by carbons

    Args:
        mol: RDKit Mol object

    Returns:
        True if molecule has true zwitterionic character
    """
    if mol is None:
        return False

    # Collect positive and negative atoms. Atoms belonging to an internal-
    # charge characteristic group (nitro / azide / diazo / N-oxide, the
    # Table 5.1 SMARTS set) are NOT zwitterion charges -- without this mask a
    # nitro O- cross-paired with an azide central N+ (4-azidonitrobenzene)
    # false-positived here, and the pre-dispatch neutralisation then corrupted
    # the molecule (valence error -> unknown). Wave-2 completion.
    internal = _get_internal_charge_atoms(mol)
    positive_atoms = []
    negative_atoms = []

    for atom in atoms_of(mol):
        if atom.GetIdx() in internal:
            continue
        charge = atom.GetFormalCharge()
        if charge > 0:
            positive_atoms.append(atom)
        elif charge < 0:
            negative_atoms.append(atom)

    if not positive_atoms or not negative_atoms:
        return False

    # Check if ALL charge pairs are functional group internal charges
    # If any charge pair is NOT directly bonded, it's a true zwitterion
    for pos_atom in positive_atoms:
        for neg_atom in negative_atoms:
            # Check if directly bonded (functional group internal charge)
            bond = mol.GetBondBetweenAtoms(pos_atom.GetIdx(), neg_atom.GetIdx())
            if bond is None:
                # Not directly bonded - could be true zwitterion
                # Check for nitro group pattern: N+ bonded to O= and O-
                if not _is_nitro_or_similar_group(mol, pos_atom, neg_atom):
                    return True

    return False


def _is_nitro_or_similar_group(mol, pos_atom, neg_atom) -> bool:
    """
    Check if the + and - atoms are part of a nitro or similar functional group.

    Nitro group: [N+](=O)[O-] where N+ is bonded to the O- through a shared C
    or directly through a resonance structure.

    Args:
        mol: RDKit Mol object
        pos_atom: Positively charged atom
        neg_atom: Negatively charged atom

    Returns:
        True if atoms are part of a functional group with internal charges
    """
    pos_element = pos_atom.GetSymbol()
    neg_element = neg_atom.GetSymbol()

    # Nitro group: N+ bonded to O= and O-
    # The N+ neighbors should include both the O- and an O=
    if pos_element == 'N' and neg_element == 'O':
        # Check if this N is bonded to both an O= and an O-
        n_neighbors = list(pos_atom.GetNeighbors())
        oxygen_neighbors = [n for n in n_neighbors if n.GetSymbol() == 'O']

        if len(oxygen_neighbors) >= 2:
            # N bonded to 2+ oxygens - likely nitro or nitroso
            has_double_o = False
            has_negative_o = False

            for o_neighbor in oxygen_neighbors:
                bond = mol.GetBondBetweenAtoms(pos_atom.GetIdx(), o_neighbor.GetIdx())
                if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
                    has_double_o = True
                if o_neighbor.GetFormalCharge() < 0:
                    has_negative_o = True

            if has_double_o and has_negative_o:
                return True

    # N-oxide: N+ directly bonded to O-
    if pos_element == 'N' and neg_element == 'O':
        bond = mol.GetBondBetweenAtoms(pos_atom.GetIdx(), neg_atom.GetIdx())
        if bond is not None:
            return True  # Directly bonded N+-O- is N-oxide or similar

    # Azide: [N-]=[N+]=[N-] pattern
    if pos_element == 'N' and neg_element == 'N':
        # Check if they're part of an azide chain
        n_neighbors = [n for n in pos_atom.GetNeighbors() if n.GetSymbol() == 'N']
        if len(n_neighbors) >= 2:
            return True  # Likely azide

    return False


def get_ion_sites(mol, exclude_internal=True) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get all charged atom sites in a molecule.

    Args:
        mol: RDKit Mol object
        exclude_internal: If True (default), exclude atoms whose formal charges
            are bonding features of prefix-only groups (nitro, N-oxide, azide,
            diazo) per IUPAC

    Returns:
        Dictionary with 'cations' and 'anions' lists.
        Each entry contains:
        - atom_idx: int - atom index in molecule
        - charge: int - formal charge (+1, -1, +2, etc.)
        - element: str - element symbol (N, O, C, etc.)
        - hybridization: str - hybridization state (SP3, SP2, etc.)
        - n_hydrogens: int - number of attached hydrogens

    Example:
        >>> mol = Chem.MolFromSmiles('[NH4+]')
        >>> sites = get_ion_sites(mol)
        >>> sites['cations'][0]['element']
        'N'
        >>> sites['cations'][0]['charge']
        1
    """
    result: Dict[str, List[Dict[str, Any]]] = {
        'cations': [],
        'anions': []
    }

    if mol is None:
        return result

    for atom in atoms_of(mol):
        charge = atom.GetFormalCharge()

        if charge == 0:
            continue

        site_info = {
            'atom_idx': atom.GetIdx(),
            'charge': charge,
            'element': atom.GetSymbol(),
            'hybridization': str(atom.GetHybridization()),
            'n_hydrogens': atom.GetTotalNumHs()
        }

        if charge > 0:
            result['cations'].append(site_info)
        else:
            result['anions'].append(site_info)

    if exclude_internal:
        internal = _get_internal_charge_atoms(mol)
        result['cations'] = [s for s in result['cations'] if s['atom_idx'] not in internal]
        result['anions'] = [s for s in result['anions'] if s['atom_idx'] not in internal]

    return result


def get_radical_sites(mol) -> List[Dict[str, Any]]:
    """
    Get all radical centers in a molecule.

    Args:
        mol: RDKit Mol object

    Returns:
        List of radical site dictionaries, each containing:
        - atom_idx: int - atom index in molecule
        - n_electrons: int - number of unpaired electrons (1, 2, 3)
        - element: str - element symbol
        - radical_type: str - 'monovalent' (1), 'divalent' (2), 'trivalent' (3)
        - hybridization: str - hybridization state

    Example:
        >>> mol = Chem.MolFromSmiles('[CH3]')
        >>> sites = get_radical_sites(mol)
        >>> sites[0]['element']
        'C'
        >>> sites[0]['radical_type']
        'monovalent'
    """
    result: List[Dict[str, Any]] = []

    if mol is None:
        return result

    radical_type_map = {
        1: 'monovalent',
        2: 'divalent',
        3: 'trivalent'
    }

    for atom in atoms_of(mol):
        n_radical = atom.GetNumRadicalElectrons()

        if n_radical == 0:
            continue

        site_info = {
            'atom_idx': atom.GetIdx(),
            'n_electrons': n_radical,
            'element': atom.GetSymbol(),
            'radical_type': radical_type_map.get(n_radical, f'{n_radical}-valent'),
            'hybridization': str(atom.GetHybridization())
        }

        result.append(site_info)

    return result


def parse_salt_fragments(mol) -> Dict[str, List[Dict[str, Any]]]:
    """
    Parse a salt into its cation and anion fragments.

    For multi-component salts (dot-separated SMILES), this separates
    the positively and negatively charged fragments.

    Args:
        mol: RDKit Mol object (may contain multiple fragments)

    Returns:
        Dictionary with 'cations', 'anions', and 'neutrals' lists.
        Each entry contains:
        - mol: RDKit Mol object for the fragment
        - charge: int - net charge of the fragment
        - smiles: str - canonical SMILES of the fragment

    Example:
        >>> mol = Chem.MolFromSmiles('[Na+].[O-]C(C)=O')
        >>> frags = parse_salt_fragments(mol)
        >>> len(frags['cations'])
        1
        >>> len(frags['anions'])
        1
        >>> frags['cations'][0]['smiles']
        '[Na+]'
    """
    result: Dict[str, List[Dict[str, Any]]] = {
        'cations': [],
        'anions': [],
        'neutrals': []
    }

    if mol is None:
        return result

    # Get molecular fragments as separate molecules
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)

    for frag in frags:
        charge = Chem.GetFormalCharge(frag)
        smiles = Chem.MolToSmiles(frag)

        frag_info = {
            'mol': frag,
            'charge': charge,
            'smiles': smiles
        }

        if charge > 0:
            result['cations'].append(frag_info)
        elif charge < 0:
            result['anions'].append(frag_info)
        else:
            result['neutrals'].append(frag_info)

    return result
