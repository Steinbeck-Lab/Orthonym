"""
Radical naming rules per IUPAC 2013.

Handles naming of:
- Monovalent radicals: -yl suffix (methyl, ethyl, phenyl)
- Divalent radicals: -ylidene suffix (methylidene, ethylidene)
- Trivalent radicals: -ylidyne suffix (methylidyne)
- Acyl radicals: -oyl suffix (acetyl, benzoyl)
- Oxyl radicals: -oxyl suffix (methoxyl, phenoxyl)

IUPAC 2013 References:
-: Radical nomenclature

Key naming patterns:
- Monovalent alkyl radicals: alkane - H -> alkyl (methyl, ethyl)
- Divalent alkylidene radicals: methane -> methylidene
- Trivalent alkylidyne radicals: methane -> methylidyne
- Acyl radicals: acid - OH -> -oyl (acetyl from acetic)
- Oxyl radicals: R-O. -> R-oxyl (methoxyl from methanol)
"""

from typing import Any, Dict, List, Optional

from rdkit import Chem

from ..perception.ions import get_radical_sites

# === SUFFIX MAPPINGS ===

RADICAL_SUFFIXES = {
    1: 'yl',        # Monovalent: methyl, ethyl
    2: 'ylidene',   # Divalent: methylidene, ethylidene
    3: 'ylidyne',   # Trivalent: methylidyne
}

# Radical type names
RADICAL_TYPE_NAMES = {
    1: 'monovalent',
    2: 'divalent',
    3: 'trivalent',
}


# === CHAIN PREFIXES ===
# Delegated to centralized chain_names module
from ..data.chain_names import get_chain_prefix as _get_chain_prefix

# Retained radical names (canonical SMILES -> name)
# These are looked up first before systematic naming
RETAINED_RADICALS = {
    # Monovalent alkyl radicals
    '[CH3]': 'methyl',
    '[CH2]C': 'ethyl',
    '[CH2]CC': 'propyl',
    '[CH2]CCC': 'butyl',

    # Divalent radicals
    '[CH2]': 'methylidene',
    '[CH]C': 'ethylidene',

    # Trivalent radicals
    '[CH]': 'methylidyne',

    # Aryl radicals. (1) (the Blue Book): "the preferred IUPAC name
    # for a radical may not be the same as the preferred prefix" -- its own
    # examples are '2-methylpropan-2-yl (PIN)' not tert-butyl (:40453),
    # 'benzene-1,4-diyl (PIN)' not 1,4-phenylene (:40540), and the cation
    # 'benzenylium (PIN)' not phenylium (:41537). The book prints no C6H5. or
    # C6H5CH2. radical; by that pattern they are benzenyl and phenylmethyl
    # (user decision Q1 = A, 2026-09-25).
    '[c]1ccccc1': 'benzenyl',
    '[CH2]c1ccccc1': 'phenylmethyl',

    # Acyl radicals (carbonyl radicals)
    '[CH]=O': 'formyl',
    'C[C]=O': 'acetyl',
    'CC[C]=O': 'propanoyl',
    'O=[C]c1ccccc1': 'benzoyl',

    # Oxyl radicals (oxygen-centered). (the Blue Book): "The
    # names methoxyl, ethoxyl, propoxyl, butoxyl, tert-butoxyl, phenoxyl, and
    # aminoxyl... are retained and are preferred IUPAC names" -- a closed list.
    'C[O]': 'methoxyl',
    'CC[O]': 'ethoxyl',
    'CCC[O]': 'propoxyl',
    'CCCC[O]': 'butoxyl',
    'CC(C)(C)[O]': 'tert-butoxyl',
    '[O]c1ccccc1': 'phenoxyl',
    # (the Blue Book): "the IUPAC preferred name for HO. is
    # 'hydroxyl'... and... for HOO. is 'hydroperoxyl'" (not to be substituted).
    '[OH]': 'hydroxyl',
    '[O]O': 'hydroperoxyl',
}


# === RADICAL CLASSIFICATION ===

def classify_radical(mol, radical_site: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classify radical type based on the radical center environment.

    Examines the local chemical environment of the radical atom to
    determine the appropriate naming approach.

    Args:
        mol: RDKit Mol object
        radical_site: Dictionary from get_radical_sites containing:
            - atom_idx: int
            - n_electrons: int (1, 2, or 3)
            - element: str
            - radical_type: str ('monovalent', 'divalent', 'trivalent')
            - hybridization: str

    Returns:
        Dictionary with:
        - 'n_electrons': 1, 2, or 3
        - 'radical_type': 'monovalent', 'divalent', 'trivalent'
        - 'subtype': 'alkyl', 'acyl', 'oxyl', 'aryl', or 'generic'

    Example:
        >>> mol = Chem.MolFromSmiles('[CH3]')
        >>> sites = get_radical_sites(mol)
        >>> classify_radical(mol, sites[0])
        {'n_electrons': 1, 'radical_type': 'monovalent', 'subtype': 'alkyl'}
    """
    n_electrons = radical_site['n_electrons']
    element = radical_site['element']
    atom_idx = radical_site['atom_idx']
    radical_type = radical_site['radical_type']

    atom = mol.GetAtomWithIdx(atom_idx)

    result = {
        'n_electrons': n_electrons,
        'radical_type': radical_type,
        'subtype': 'generic',
    }

    # Classify based on element
    if element == 'O':
        # Oxygen-centered radical
        # Check if attached to carbon (oxyl radical: R-O.)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetSymbol() == 'C':
                result['subtype'] = 'oxyl'
                return result
        result['subtype'] = 'oxyl'

    elif element == 'C':
        # Carbon-centered radical
        # Check for acyl radical (C. attached to C=O)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetSymbol() == 'O':
                bond = mol.GetBondBetweenAtoms(atom_idx, neighbor.GetIdx())
                if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
                    result['subtype'] = 'acyl'
                    return result

        # Check if aromatic (aryl radical)
        if atom.GetIsAromatic():
            result['subtype'] = 'aryl'
            return result

        # Check if attached to aromatic ring (benzylic)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetIsAromatic():
                result['subtype'] = 'benzylic'
                return result

        # Default to alkyl
        result['subtype'] = 'alkyl'

    elif element == 'N':
        # Nitrogen-centered radical (aminyl)
        result['subtype'] = 'aminyl'

    elif element == 'S':
        # Sulfur-centered radical (thiyl)
        result['subtype'] = 'thiyl'

    return result


# === SUFFIX HELPERS ===

def get_radical_suffix(n_electrons: int) -> str:
    """
    Return suffix based on radical electron count.

    Args:
        n_electrons: Number of unpaired electrons (1, 2, or 3)

    Returns:
        Suffix string: 'yl' for 1, 'ylidene' for 2, 'ylidyne' for 3

    Example:
        >>> get_radical_suffix(1)
        'yl'
        >>> get_radical_suffix(2)
        'ylidene'
        >>> get_radical_suffix(3)
        'ylidyne'
    """
    return RADICAL_SUFFIXES.get(n_electrons, 'yl')


def _count_chain_carbons(mol, start_idx: int, exclude: set) -> int:
    """Count carbon atoms in a chain via BFS from a starting point."""
    from collections import deque

    visited = set()
    queue = deque([start_idx])
    count = 0

    while queue:
        atom_idx = queue.popleft()
        if atom_idx in visited or atom_idx in exclude:
            continue
        visited.add(atom_idx)

        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetSymbol() == 'C':
            count += 1
            for neighbor in atom.GetNeighbors():
                nbr_idx = neighbor.GetIdx()
                if nbr_idx not in visited and nbr_idx not in exclude:
                    if neighbor.GetSymbol() in ('C', 'H'):
                        queue.append(nbr_idx)

    return count


def _is_carboxyl_carbon(mol, idx: int, from_idx: int) -> bool:
    """True if atom `idx` is a carboxylic-acid carbon -C(=O)OH reached from
    `from_idx` (a chain carbon) — i.e. C bonded to =O and -OH and nothing else
    heavy but `from_idx`."""
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != 'C' or atom.GetIsAromatic():
        return False
    dbl_o = single_oh = 0
    other_heavy = 0
    for nbr in atom.GetNeighbors():
        n_idx = nbr.GetIdx()
        if n_idx == from_idx:
            continue
        bond = mol.GetBondBetweenAtoms(idx, n_idx)
        if nbr.GetSymbol() == 'O':
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                dbl_o += 1
            elif bond.GetBondType() == Chem.BondType.SINGLE and nbr.GetTotalNumHs() >= 1:
                single_oh += 1
            else:
                other_heavy += 1
        else:
            other_heavy += 1
    return dbl_o == 1 and single_oh == 1 and other_heavy == 0


def _name_carboxy_alkyl_radical(mol, radical_idx: int) -> Optional[str]:
    """ cls 1: the radical (free valence) is the MOST senior
    class, senior to a carboxylic acid. A monovalent alkyl radical whose linear
    carbon chain (starting at the free valence = C-1) terminates in a
    carboxylic-acid carbon names the chain as the -yl parent and cites the acid
    as a 'carboxy' prefix, e.g. ``HOOC-CH2-CH2.`` -> ``2-carboxyethyl``.

    Fail-closed (returns None) for anything but a single unbranched all-carbon
    chain from the free valence bearing exactly one terminal -COOH and no other
    substituent/heteroatom/ring/unsaturation.
    """
    if mol.GetRingInfo().NumRings():
        return None
    # Walk the chain from the radical carbon. Each step: the current C must have
    # exactly one onward C neighbour (unbranched), until we reach a carboxyl C.
    chain: List[int] = [radical_idx]
    prev = -1
    cur = radical_idx
    acid_locant: Optional[int] = None
    while True:
        atom = mol.GetAtomWithIdx(cur)
        if atom.GetSymbol() != 'C' or atom.GetIsAromatic():
            return None
        # onward heavy neighbours (excluding where we came from)
        onward = [n.GetIdx() for n in atom.GetNeighbors()
                  if n.GetIdx() != prev]
        # any onward atom must be C (no ethers/amines etc. in scope)
        carboxyls = [i for i in onward if _is_carboxyl_carbon(mol, i, cur)]
        carbons = [i for i in onward
                   if mol.GetAtomWithIdx(i).GetSymbol() == 'C'
                   and i not in carboxyls]
        others = [i for i in onward
                  if i not in carboxyls and i not in carbons]
        if others:
            return None
        if carboxyls:
            if len(carboxyls) != 1 or carbons:
                return None  # branch or >1 acid: out of scope
            acid_locant = len(chain)  # current carbon's chain locant
            break
        if len(carbons) != 1:
            return None  # branch or chain terminus without an acid
        prev, cur = cur, carbons[0]
        chain.append(cur)
        if len(chain) > 30:
            return None
    if acid_locant is None:
        return None
    # any ring/unsaturation on the chain is out of scope (keep it simple)
    for a_idx in chain:
        a = mol.GetAtomWithIdx(a_idx)
        for b in a.GetBonds():
            if b.GetBondType() != Chem.BondType.SINGLE:
                return None
    stem = _get_chain_prefix(len(chain))
    # A single-carbon stem carries no locant ambiguity: 'carboxymethyl'
    # (not '1-carboxymethyl').
    if len(chain) == 1:
        candidate = f"carboxy{stem}yl"
    else:
        candidate = f"{acid_locant}-carboxy{stem}yl"
    # The shape walk ignores isotopes and hydrogen counts ([13CH2]CC(=O)O would
    # be '2-carboxyethyl', a different isotopologue): ship only what the strict
    # radical round trip confirms.
    return candidate if _radical_identity_round_trips(mol, candidate) else None


# 169.6-03 (CHOKE-01, kill-list): the carbon-counting chain-counter that
# fed the alkyl / -ylidene / -ylidyne radical naming was DELETED. The three
# helpers below are now thin shims over route_charged (the single chokepoint),
# which H-saturates the radical center, re-enters the FULL pipeline, and
# re-applies the suffix — so a SUBSTITUTED/branched/unsaturated alkyl
# radical is named correctly instead of by a bare carbon count.


def name_alkyl_radical(mol, radical_site: Dict[str, Any]) -> str:
    """
    Name an alkyl radical (R.) via the route_charged chokepoint.

    169.6-03: delegates to route_charged (neutralize the radical center ->
    re-enter the full pipeline -> re-apply the -yl/-ylidene/-ylidyne suffix),
    replacing the old carbon-counting body. Returns the radical name (e.g.
    'methyl', 'propyl') or '' on fall-through.

    Args:
        mol: RDKit Mol object
        radical_site: Dictionary from get_radical_sites (kept for API stability)

    Example:
        >>> mol = Chem.MolFromSmiles('[CH3]')
        >>> sites = get_radical_sites(mol)
        >>> name_alkyl_radical(mol, sites[0])
        'methyl'
    """
    from .charged_router import route_charged
    return route_charged(mol, 'pin')


def name_acyl_radical(mol, radical_site: Dict[str, Any]) -> str:
    """
    Name an acyl radical (RC.=O).

    Acyl radicals are derived from carboxylic acids by loss of -OH.
    Named with -oyl suffix (e.g., acetyl from acetic acid).

    Args:
        mol: RDKit Mol object
        radical_site: Dictionary from get_radical_sites

    Returns:
        Acyl radical name (e.g., 'acetyl', 'propanoyl', 'benzoyl')

    Example:
        >>> mol = Chem.MolFromSmiles('C[C]=O')
        >>> sites = get_radical_sites(mol)
        >>> name_acyl_radical(mol, sites[0])
        'acetyl'
    """
    radical_idx = radical_site['atom_idx']
    radical_atom = mol.GetAtomWithIdx(radical_idx)

    # Count carbons (excluding the carbonyl oxygen)
    carbon_count = 1  # Start with the carbonyl carbon

    for neighbor in radical_atom.GetNeighbors():
        if neighbor.GetSymbol() == 'C':
            carbon_count += _count_chain_carbons(mol, neighbor.GetIdx(), {radical_idx})

    # Map to acyl name
    ACYL_NAMES = {
        1: 'formyl',      # H-C(=O).
        2: 'acetyl',      # CH3-C(=O).
        3: 'propanoyl',   # C2H5-C(=O).
        4: 'butanoyl',
        5: 'pentanoyl',
        6: 'hexanoyl',
        7: 'heptanoyl',
        8: 'octanoyl',
    }

    # Check for aromatic acyl (benzoyl)
    if any(neighbor.GetIsAromatic() for neighbor in radical_atom.GetNeighbors()):
        candidate = 'benzoyl'
    elif carbon_count in ACYL_NAMES:
        candidate = ACYL_NAMES[carbon_count]
    else:
        candidate = _get_chain_prefix(carbon_count) + 'anoyl'
    # A carbon count sees neither substituents, heteroatoms nor isotopes
    # (O=[C]C(=O)O would be 'acetyl'; any aryl would be 'benzoyl'): ship only
    # what the strict radical round trip confirms.
    return candidate if _radical_identity_round_trips(mol, candidate) else ''


def _collect_fragment_excluding(mol, start_idx: int, exclude: set) -> List[int]:
    """BFS the atoms of an R group starting at ``start_idx``, never crossing
    into any atom index in ``exclude``. General shape shared by every
    R-substituent-fragment walk in this module (mirrors the R-fragment
    collection already used for the ester-alkyl side elsewhere in this
    codebase, e.g. ``acid_halides.py``'s ``r_atoms``/``r_root`` walk); an
    R-O. oxyl radical excludes only the radical oxygen, while an
    R-O-O. peroxyl radical must exclude BOTH oxygens of the bridge so the
    peroxide bridge itself is never swept into the R group."""
    from collections import deque
    seen = set(exclude)
    frag: List[int] = []
    queue = deque([start_idx])
    while queue:
        cur = queue.popleft()
        if cur in seen:
            continue
        seen.add(cur)
        frag.append(cur)
        for nbr in mol.GetAtomWithIdx(cur).GetNeighbors():
            if nbr.GetIdx() not in seen:
                queue.append(nbr.GetIdx())
    return frag


def _collect_oxyl_parent_fragment(mol, radical_idx: int, attach_idx: int) -> List[int]:
    """BFS the atoms of the R group hanging off an R-O. oxyl radical's
    oxygen, starting at ``attach_idx`` (the O's sole heavy neighbour) and
    never crossing back through ``radical_idx`` (the radical oxygen itself)."""
    return _collect_fragment_excluding(mol, attach_idx, {radical_idx})


def _is_plain_alkyl_radical_fragment(mol, radical_idx: int, attach_idx: int) -> bool:
    """True iff the R group of an aliphatic R-O. radical is a BARE, LINEAR,
    SATURATED hydrocarbon chain rooted AT ``attach_idx`` -- the exact shape
    the existing retained '-oxyl' contraction (methoxyl/ethoxyl/propoxyl/
    .../hexoxyl) was already built for (BB /: 'methoxyl,
    ethoxyl, propoxyl, butoxyl, *tert*-butoxyl, phenoxyl... are retained and
    are preferred IUPAC names' -- a FIXED short list of straight-chain/simple
    contractions, not a licence to count carbons on ANY hydrocarbon shape).

    a review finding (A2 hardening): the original guard checked only
    aromatic/ring/heteroatom and NOT branching or unsaturation, so a
    BRANCHED or UNSATURATED alkyl fragment was routed to the same carbon-
    COUNT contraction as a straight chain -- e.g. propan-2-yl (isopropyl)
    was misnamed 'propoxyl' (which OPSIN parses back to the straight-chain
    '[O]CCC', a structural MISMATCH). Tightened to require:
      (1) ``attach_idx`` itself has AT MOST ONE carbon neighbour inside the
          fragment (it is a chain TERMINUS, i.e. true C-1 -- excludes
          propan-2-yl, where the attachment carbon has two branches);
      (2) every atom in the fragment is a non-aromatic, non-ring carbon with
          at most two carbon neighbours inside the fragment (excludes any
          OTHER branch point, e.g. neopentyl/isobutyl);
      (3) every bond between two fragment atoms is a single bond (excludes
          any C=C/C#C unsaturation, e.g. ethenyl/allyl).
    Anything failing (1)-(3) falls through to the general substituent
    pipeline (``_name_oxyl_parent_group`` / ``(<parent>)oxyl`` via
    ``_compose_oxyl_name``) instead of being silently misnamed."""
    frag = _collect_oxyl_parent_fragment(mol, radical_idx, attach_idx)
    frag_set = set(frag)

    attach_atom = mol.GetAtomWithIdx(attach_idx)
    attach_heavy_in_frag = [n for n in attach_atom.GetNeighbors()
                             if n.GetIdx() in frag_set]
    if len(attach_heavy_in_frag) > 1:
        return False  # attachment carbon itself branches -> not a chain C-1

    for idx in frag:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != 'C' or atom.GetIsAromatic() or atom.IsInRing():
            return False
        heavy_in_frag = [n for n in atom.GetNeighbors() if n.GetIdx() in frag_set]
        if len(heavy_in_frag) > 2:
            return False  # branch point elsewhere in the chain
        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            if other.GetIdx() in frag_set and bond.GetBondType() != Chem.BondType.SINGLE:
                return False  # any C=C/C#C unsaturation
    return True


def _name_oxyl_parent_group(mol, radical_idx: int, attach_idx: int) -> str:
    """Name the R group of an R-O. oxyl radical as a substituent -yl
    prefix), via the project's existing recursive substituent-naming
    chokepoint ``name_substituent_fragment`` -- the SAME pipeline other rule
    modules use to name an R group hanging off an excluded heteroatom (e.g.
    the ester-alkyl side in ``acid_halides.py``/``esters.py``). '' on failure
    (caller falls through / abstains -- never a wrong name)."""
    frag = _collect_oxyl_parent_fragment(mol, radical_idx, attach_idx)
    if not frag:
        return ''
    from ..assembly.substituent_naming import name_substituent_fragment
    return name_substituent_fragment(mol, frag, attach_idx, [radical_idx]) or ''


def _radical_name_round_trips(mol, name: str) -> bool:
    """``-r`` (allowRadicals) OPSIN round-trip check for a radical-name
    CANDIDATE against ``mol``'s own structure. Two uses in this module: (a)
    belt-and-suspenders over the shape guards above (never trust a retained or
    contracted candidate on shape alone); (b) the tie-break between 's
    two composition methods in ``_compose_oxyl_name`` below.

    True only when OPSIN parses the name to the input's full InChIKey AND to the
    same radical graph (validation/radical_identity.py): the full key alone
    cannot separate a radical from a closed-shell or differently placed twin.
    Fails CLOSED -- no jar, no JVM, an unserved or rejected parse, or an InChI
    failure all return False, so a candidate this check cannot confirm is never
    shipped from here."""
    if not name:
        return False
    try:
        from ..jvm_bridge import opsin_stdout
        from ..validation.opsin_roundtrip import _find_opsin_jar
        from ..validation.radical_identity import radical_identity_verdict
        jar = _find_opsin_jar("2.9.0")
        if not jar:
            return False
        txt, served = opsin_stdout(name, allow_radicals=True, jar_path=jar)
    except Exception:
        return False
    if not served:
        return False
    smi = (txt or "").strip()
    parsed = Chem.MolFromSmiles(smi) if smi else None
    if parsed is None:
        return False
    try:
        if (Chem.InchiToInchiKey(Chem.MolToInchi(parsed))
                != Chem.InchiToInchiKey(Chem.MolToInchi(mol))):
            return False
    except Exception:
        return False
    return radical_identity_verdict(Chem.MolToSmiles(mol), smi) != "mismatch"


def _radical_identity_round_trips(mol, name: str) -> bool:
    """Strict ``-r`` OPSIN round trip for a radical name candidate: True only
    when OPSIN parses ``name`` to exactly ``mol``, radical electrons, charges
    and hydrogen counts included (canonical isomeric SMILES equality).

    The full InChIKey alone does not decide this: it encodes neither radical
    electrons nor bond order, so an N-oxyl radical and a closed-shell or
    charge-separated twin can share a key. Fails CLOSED when OPSIN cannot be
    reached -- a producer that uses this check never ships an unverified
    name."""
    if not name:
        return False
    try:
        from ..jvm_bridge import opsin_stdout
        from ..validation.opsin_roundtrip import _find_opsin_jar
        jar = _find_opsin_jar("2.9.0")
        if not jar:
            return False
        txt, served = opsin_stdout(name, allow_radicals=True, jar_path=jar)
    except Exception:
        return False
    if not served:
        return False
    parsed = Chem.MolFromSmiles((txt or "").strip())
    if parsed is None:
        return False
    try:
        return Chem.MolToSmiles(parsed) == Chem.MolToSmiles(mol)
    except Exception:
        return False


_ACYL_HETERO = {'O', 'S', 'Se', 'Te', 'N'}


def _aminoxyl_prefix_string(prefixes: List[str]) -> str:
    """The substituent prefixes of a substituted aminoxyl, as in the Blue Book's
    'bis(chloromethyl)aminoxyl (PIN)', the Blue Book):
    alphanumerical order, ``alpha_sort_key``), 'di'/'bis' multiplying
    prefixes /: 'bis' and enclosing marks for a compound
    prefix), a hyphen between a simple multiplier and an italicized
    structural prefix (b), decided by the shared
    naming_utils.multiplier_needs_hyphen), and enclosing marks around
    every prefix after the first so that 'methyl(phenyl)' cannot be read as
    one prefix."""
    from collections import Counter
    from ..assembly.naming_utils import (alpha_sort_key, enclose_if_compound,
                                         multiplier_needs_hyphen)
    counts = Counter(prefixes)
    simple = {2: 'di', 3: 'tri', 4: 'tetra'}
    compound_mult = {2: 'bis', 3: 'tris', 4: 'tetrakis'}
    out = []
    for i, p in enumerate(sorted(counts, key=alpha_sort_key)):
        compound = enclose_if_compound(p) != p
        n = counts[p]
        if n > 1:
            if n not in simple:
                return ''
            if compound:
                out.append(f"{compound_mult[n]}({p})")
            else:
                out.append(simple[n] + ("-" if multiplier_needs_hyphen(p) else "") + p)
        elif compound or i:
            out.append(f"({p})")
        else:
            out.append(p)
    return "".join(out)


def _n_substituent_prefix(mol, frag: List[int], attach_idx: int, general: bool):
    """(prefix, from_general_route) for one group on an N-oxyl nitrogen, or
    (None, False). The PIN route gets first refusal -- the ring-system
    substituent namer for a ring attachment, the substituent enumerator for a
    chain -- exactly as the composer does (assembly/composer.py, 'ADDITIVE-ONLY:
    the PIN route gets first refusal'). The wider route (allow_mancude, which
    can build skeletal-replacement forms such as '1-oxa-3-azacyclopentan-3-yl'
    for 1,3-oxazolidin-3-yl) is tried only when ``general`` is set, and its
    result is reported so the caller can demote the tier."""
    from ..assembly.substituent_enumerator import name_substituent
    from .ring_substituents import name_ring_system_substituent

    def _one(allow_mancude):
        if mol.GetAtomWithIdx(attach_idx).IsInRing():
            got = name_ring_system_substituent(mol, sorted(frag), attach_idx,
                                               allow_mancude=allow_mancude)
        else:
            got = name_substituent(mol, frag, attach_idx, allow_mancude=allow_mancude)
        return got if got and got.endswith('yl') else None

    prefix = _one(False)
    if prefix and _indicated_h_kept(mol, frag, attach_idx, prefix):
        return prefix, False
    if general:
        prefix = prefix or _one(True)
        if prefix:
            return prefix, True
    return None, False


def _fragment_parent_name(mol, frag: List[int], attach_idx: int) -> str:
    """The name of the fragment as a neutral parent: every bond from
    ``attach_idx`` to an atom outside ``frag`` becomes a hydrogen."""
    import re as _re  # noqa: F401 (kept local; this module imports lazily)
    work = Chem.RWMol(mol)
    keep = set(frag)
    try:
        a = work.GetAtomWithIdx(attach_idx)
        lost = sum(1 for n in a.GetNeighbors() if n.GetIdx() not in keep)
        a.SetNumRadicalElectrons(0)
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(a.GetTotalNumHs() + lost)
        for i in sorted((i for i in range(work.GetNumAtoms()) if i not in keep), reverse=True):
            work.RemoveAtom(i)
        parent = work.GetMol()
        Chem.SanitizeMol(parent)
    except (RuntimeError, ValueError):
        return ''
    return _reenter_neutral_name(parent)


def _indicated_h_kept(mol, frag: List[int], attach_idx: int, prefix: str) -> bool:
    """: a substituent prefix of a ring that needs indicated
    hydrogen cites it -- '(1H-indol-1-yl)acetic acid (PIN)' (the Blue Book.
    The shared prefix namer can drop it ('pyrrol-1-yl' for 1H-pyrrol-1-yl), so a
    prefix whose parent's own name carries indicated hydrogen must carry one
    too; otherwise it is not a PIN-route prefix."""
    import re
    parent = _fragment_parent_name(mol, frag, attach_idx)
    return not (re.search(r"\d+H-", parent) and not re.search(r"\d+H", prefix))


def _name_n_oxyl_radical(mol, radical_idx: int, n_atom) -> str:
    """: name an N-oxyl (aminoxyl) radical R2N-O. -- the free valence
    on an oxygen whose only neighbour is a neutral, saturated amine nitrogen.

    VERIFIED, (the Blue Book-40705): the radical is named
    '(1) additively, using the term oxyl', and "Method (1) generates preferred
    IUPAC names"; 'aminoxyl' (H2N-O.) is a retained contraction of aminooxyl
    (:40681,:40699), and its substituted form is the PIN
    '(ClCH2)2N-O. bis(chloromethyl)aminoxyl (PIN)' (:40703). So:
      - H2N-O.: 'aminoxyl';
      - an acyclic N with carbon substituents: '<prefixes>aminoxyl';
      - a ring N: method (1), the ring-N substituent prefix + 'oxyl', e.g.
        '(2,2,6,6-tetramethylpiperidin-1-yl)oxyl', enclosed as a compound
        prefix like '(chloroacetyl)oxyl (PIN)'.
    Ruling: the CATION example '(CH3)2N-O+ N-methylmethanaminoxylium
    (PIN)' (:41652) uses an amine parent instead; for radicals the radical
    section's own example (:40703) governs.

    Prefixes come from the PIN route first; a prefix only the wider route can
    build is used only at a general tier and demotes the name out of
    pin_verified (``record_general_ring_prefix``), so a non-PIN form never
    ships as a PIN.

    Out of this producer, returning '' (no name): a charged, radical,
    aromatic or multiply bonded N (iminoxyl, N-oxide), a non-carbon
    substituent on N, and an acyl substituent (a hydroxamic-acid radical,
    which needs the amide rules). Every candidate must pass the strict
    radical-identity round trip, so nothing unverified ships."""
    if (n_atom.GetFormalCharge() or n_atom.GetNumRadicalElectrons()
            or n_atom.GetIsAromatic()
            or any(b.GetBondType() != Chem.BondType.SINGLE for b in n_atom.GetBonds())):
        return ''
    from ..metrics.provenance import general_fallback_ctx, record_general_ring_prefix
    try:
        general = bool(general_fallback_ctx.get())
    except Exception:  # a tier read must never break naming
        general = False
    n_idx = n_atom.GetIdx()
    used_general = False
    if n_atom.IsInRing():
        frag = _collect_fragment_excluding(mol, n_idx, {radical_idx})
        parent, used_general = _n_substituent_prefix(mol, frag, n_idx, general)
        if not parent:
            return ''
        from ..assembly.naming_utils import enclose_if_compound
        candidate = f"{enclose_if_compound(parent)}oxyl"
    else:
        prefixes = []
        for sub in n_atom.GetNeighbors():
            if sub.GetIdx() == radical_idx:
                continue
            if sub.GetSymbol() != 'C' or sub.GetFormalCharge() or sub.GetNumRadicalElectrons():
                return ''
            if any(b.GetBondType() != Chem.BondType.SINGLE
                   and b.GetOtherAtom(sub).GetSymbol() in _ACYL_HETERO
                   for b in sub.GetBonds()):
                return ''  # acyl on N: hydroxamic-acid radical, not this producer
            frag = _collect_fragment_excluding(mol, sub.GetIdx(), {n_idx, radical_idx})
            prefix, wide = _n_substituent_prefix(mol, frag, sub.GetIdx(), general)
            if not prefix:
                return ''
            used_general = used_general or wide
            prefixes.append(prefix)
        candidate = _aminoxyl_prefix_string(prefixes) + 'aminoxyl'
    if not _radical_identity_round_trips(mol, candidate):
        return ''
    if used_general:
        record_general_ring_prefix()
    return candidate


def _acyl_from_acid_name(acid: str) -> str:
    """The acyl group name of an acid name,,: the
    shared converter (which also owns 'carbamic acid' -> 'carbamoyl' and
    'butanimidic acid' -> 'butanimidoyl (PIN)') plus the forms it leaves to
    this radical context -- 'dimethylphosphinic acid' -> 'dimethylphosphinoyl
    (PIN)', the Blue Book-40610), and a thioic acid ->
    '...thioyl' ('ethanethioyl (PIN)'). '' if the name is not an acid name."""
    import re
    if not acid:
        return ''
    for end, acyl in (("phosphinic acid", "phosphinoyl"),
                      ("phosphonic acid", "phosphonoyl"), ("arsinic acid", "arsinoyl"),
                      ("arsonic acid", "arsonoyl")):
        if acid.endswith(end):
            return acid[:-len(end)] + acyl
    m = re.match(r"^(.*thio)ic (?:[OS]-)?acid$", acid)
    if m:
        return m.group(1) + "yl"
    from ..decomposition.fragment_assembly import _acid_to_acyl
    return _acid_to_acyl(acid) or ''


def _acid_name_with_oh_at(mol, idx: int) -> str:
    """Name the acid made by putting OH on atom ``idx`` in place of its radical
    electron -- the formal parent of an acyl radical or of an
    acyl-oxyl R-CO-O. (then ``idx`` is the radical oxygen, which becomes the
    acid's OH). '' on failure."""
    work = Chem.RWMol(mol)
    try:
        a = work.GetAtomWithIdx(idx)
        if a.GetSymbol() == 'O':
            a.SetNumRadicalElectrons(0)
            a.SetNoImplicit(True)
            a.SetNumExplicitHs(1)
        else:
            a.SetNumRadicalElectrons(a.GetNumRadicalElectrons() - 1)
            o = work.AddAtom(Chem.Atom(8))
            work.AddBond(idx, o, Chem.BondType.SINGLE)
            a.SetNoImplicit(True)
        acid = work.GetMol()
        Chem.SanitizeMol(acid)
    except (RuntimeError, ValueError):
        return ''
    name = _reenter_neutral_name(acid)
    return name if name.endswith('acid') else ''


def _is_acyl_atom(mol, idx: int, exclude: int = -1) -> bool:
    """True when atom ``idx`` has a double bond to O/S/Se/Te/N (other than to
    ``exclude``): the radical centre of an acyl radical, or the
    carbonyl carbon of an acyl-oxyl."""
    atom = mol.GetAtomWithIdx(idx)
    return any(b.GetBondType() == Chem.BondType.DOUBLE
               and b.GetOtherAtom(atom).GetSymbol() in ('O', 'S', 'Se', 'Te', 'N')
               and b.GetOtherAtomIdx(idx) != exclude
               for b in atom.GetBonds())


def _name_acyl_radical_from_acid(mol, idx: int) -> str:
    """ "Acyl radicals": named from the acid (hexanoyl, benzoyl,
    dimethylphosphinoyl, ethanethioyl, butanimidoyl -- the Blue Book-
    40612). A monovalent radical on an atom with a double bond to a chalcogen or
    N; the name is kept only if it passes the strict radical round trip."""
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetNumRadicalElectrons() != 1 or atom.GetFormalCharge() or not _is_acyl_atom(mol, idx):
        return ''
    acyl = _acyl_from_acid_name(_acid_name_with_oh_at(mol, idx))
    return acyl if acyl and _radical_identity_round_trips(mol, acyl) else ''


def _name_acyl_oxyl_radical(mol, radical_idx: int, attach_idx: int) -> str:
    """ method (1) for R-CO-O.: the acyl group name + 'oxyl', e.g.
    '(chloroacetyl)oxyl (PIN)', 'butanoyloxyl (PIN)' (the Blue Book,
    :40711). The acyl name comes from the acid R-CO-OH; strict round trip."""
    if not _is_acyl_atom(mol, attach_idx, exclude=radical_idx):
        return ''
    acyl = _acyl_from_acid_name(_acid_name_with_oh_at(mol, radical_idx))
    if not acyl:
        return ''
    candidate = f"{_enclose_acyl(mol, acyl, attach_idx, {radical_idx})}oxyl"
    return candidate if _radical_identity_round_trips(mol, candidate) else ''


def _acyl_is_substituted(mol, acyl_idx: int, exclude: set) -> bool:
    """True when the acyl group R-C(=X)- rooted at ``acyl_idx`` (``exclude``: the
    atoms on the far side, e.g. the radical O) carries a substituent: any
    heteroatom other than the acyl's own =X, a branch in its chain, or a ring
    atom with a side group. A substituted acyl group is a compound prefix and
    takes enclosing marks: '(chloroacetyl)oxyl (PIN)'
    (the Blue Book; '(chloroacetyl)' 4 of 4 English occurrences) and
    '(chloroethanethioyl)sulfanyl', but 'butanoyloxyl (PIN)'."""
    frag = set(_collect_fragment_excluding(mol, acyl_idx, set(exclude)))
    acyl = mol.GetAtomWithIdx(acyl_idx)
    x_atoms = {b.GetOtherAtomIdx(acyl_idx) for b in acyl.GetBonds()
               if b.GetBondType() == Chem.BondType.DOUBLE
               and b.GetOtherAtom(acyl).GetSymbol() in ('O', 'S', 'Se', 'Te', 'N')}
    body = frag - x_atoms
    for i in body:
        a = mol.GetAtomWithIdx(i)
        if i != acyl_idx and a.GetSymbol() != 'C':
            return True
        nbrs = [n.GetIdx() for n in a.GetNeighbors() if n.GetIdx() in body]
        if not a.IsInRing() and len(nbrs) > 2:
            return True
        if a.IsInRing() and any(not mol.GetAtomWithIdx(n).IsInRing() and n != acyl_idx
                                for n in nbrs):
            return True
    return False


def _enclose_acyl(mol, acyl: str, acyl_idx: int, exclude: set) -> str:
    from ..assembly.naming_utils import enclose_if_compound
    enclosed = enclose_if_compound(acyl)
    if enclosed == acyl and _acyl_is_substituted(mol, acyl_idx, exclude):
        enclosed = f"({acyl})"
    return enclosed


def _placeholder_prefix(mol, idx: int, general: bool):
    """(prefix, from_general_route) for the whole molecule named as a
    substituent group attached at the radical atom ``idx`` -- a
    radical formed by removing one H is named like the substituent prefix of
    the same group ('piperidin-1-yl', 'naphthalen-2-yl'). The substituent
    namers expect a satisfied valence at the attachment, so a methyl
    placeholder takes the radical electron's place and is left out of the
    named fragment (the N-oxyl producer does the same with its O)."""
    work = Chem.RWMol(mol)
    try:
        a = work.GetAtomWithIdx(idx)
        a.SetNumRadicalElectrons(a.GetNumRadicalElectrons() - 1)
        c = work.AddAtom(Chem.Atom(6))
        work.AddBond(idx, c, Chem.BondType.SINGLE)
        a.SetNoImplicit(True)
        surrogate = work.GetMol()
        Chem.SanitizeMol(surrogate)
    except (RuntimeError, ValueError):
        return None, False
    frag = [i for i in range(mol.GetNumAtoms())]
    return _n_substituent_prefix(surrogate, frag, idx, general)


def _tier_is_general() -> bool:
    from ..metrics.provenance import general_fallback_ctx
    try:
        return bool(general_fallback_ctx.get())
    except Exception:  # a tier read must never break naming
        return False


def _ship(mol, candidate: str, used_general: bool) -> str:
    """Strict radical round trip; a name built by the wide route is demoted
    out of pin_verified (the composer's contract)."""
    if not candidate or not _radical_identity_round_trips(mol, candidate):
        return ''
    import re
    if re.search(r"\((?:\d+)?[RSEZrs](?:,|\))", candidate):
        # Stereodescriptors inside a radical's prefix: their PIN placement is
        # brief item B10, not yet derived, so such a name is never shipped as a
        # PIN -- no name at the PIN tier, a demoted name at a general tier.
        if not _tier_is_general():
            return ''
        used_general = True
    if used_general:
        from ..metrics.provenance import record_general_ring_prefix
        record_general_ring_prefix()
    return candidate


_CHALCOGEN_RADICAL = {'S': 'sulfanyl', 'Se': 'selanyl', 'Te': 'tellanyl'}


def _name_chalcogen_radical(mol, idx: int) -> str:
    """ chalcogen analogues (the Blue Book): "named on the basis of
    preselected parent radical names, such as 'sulfanyl', 'selanyl',
    'disulfanyl'": 'phenylsulfanyl (PIN)' (:40727), 'CH3-Se. methylselanyl (PIN)'
    (:40733), 'tert-butyldisulfanyl (PIN)' (:40737), and with an acyl group
    '(chloroethanethioyl)sulfanyl'. A monovalent, neutral S/Se/Te radical with
    one carbon (or one same-element chalcogen, then carbon) neighbour."""
    x = mol.GetAtomWithIdx(idx)
    parent = _CHALCOGEN_RADICAL.get(x.GetSymbol())
    if (parent is None or x.GetNumRadicalElectrons() != 1 or x.GetFormalCharge()
            or x.GetDegree() != 1 or x.GetTotalNumHs()):
        return ''
    y = x.GetNeighbors()[0]
    exclude = {idx}
    if y.GetSymbol() == x.GetSymbol():
        others = [n for n in y.GetNeighbors() if n.GetIdx() != idx]
        if (y.GetFormalCharge() or y.GetNumRadicalElectrons() or len(others) != 1
                or others[0].GetSymbol() != 'C'):
            return ''
        exclude.add(y.GetIdx())
        parent = 'di' + parent
        attach = others[0]
    elif y.GetSymbol() == 'C':
        attach = y
    else:
        return ''
    if attach.GetFormalCharge() or attach.GetNumRadicalElectrons():
        return ''
    general = _tier_is_general()
    if len(exclude) == 1 and _is_acyl_atom(mol, attach.GetIdx()):
        # R-C(=X)-S.: the acyl group comes from the acid R-C(=X)-OH.
        work = Chem.RWMol(mol)
        try:
            w = work.GetAtomWithIdx(idx)
            w.SetAtomicNum(8)
            w.SetNumRadicalElectrons(0)
            w.SetNoImplicit(True)
            w.SetNumExplicitHs(1)
            acid = work.GetMol()
            Chem.SanitizeMol(acid)
        except (RuntimeError, ValueError):
            return ''
        acid_name = _reenter_neutral_name(acid)
        acyl = _acyl_from_acid_name(acid_name) if acid_name.endswith('acid') else ''
        if not acyl:
            return ''
        return _ship(mol, _enclose_acyl(mol, acyl, attach.GetIdx(), exclude) + parent, False)
    frag = _collect_fragment_excluding(mol, attach.GetIdx(), exclude)
    prefix, wide = _n_substituent_prefix(mol, frag, attach.GetIdx(), general)
    if not prefix:
        return ''
    from ..assembly.naming_utils import enclose_if_compound
    return _ship(mol, enclose_if_compound(prefix) + parent, wide)


def _name_ring_n_radical(mol, idx: int) -> str:
    """A radical on a ring nitrogen that carries no H: named as the
    ring's N-yl substituent group, e.g. '2,5-dioxopyrrolidin-1-yl (PIN)'
    (the Blue Book), piperidin-1-yl, 9H-carbazol-9-yl."""
    n = mol.GetAtomWithIdx(idx)
    if (n.GetSymbol() != 'N' or n.GetNumRadicalElectrons() != 1 or n.GetFormalCharge()
            or not n.IsInRing() or n.GetTotalNumHs()):
        return ''
    prefix, wide = _placeholder_prefix(mol, idx, _tier_is_general())
    return _ship(mol, prefix or '', wide)


# Contracted preferred prefixes that are NOT '<parent hydride>yl'. (1):
# "the preferred IUPAC name for a radical may not be the same as the preferred
# prefix" (the Blue Book) -- e.g. (CH3)3C. is '2-methylpropan-2-yl (PIN)'
# (:40453), not 'tert-butyl'. A radical whose placeholder prefix ends in one of
# these is left to the other producers; the phenyl/benzyl choice is open
# question Q1 of the radical brief.
_CONTRACTED_PREFIX_ENDINGS = (
    'tert-butyl', 'sec-butyl', 'isopropyl', 'isobutyl', 'neopentyl', 'phenyl',
    'benzyl', 'tolyl', 'xylyl', 'mesityl', 'naphthyl', 'anthryl', 'phenanthryl',
    'vinyl', 'allyl', 'phenethyl', 'trityl', 'benzhydryl', 'cumyl', 'furyl',
    'thienyl', 'pyridyl')


def _name_carbon_radical(mol, idx: int) -> str:
    """: a monovalent carbon radical formed by removing one H is named
    like the substituent group of the whole molecule attached at the radical
    carbon, e.g. naphthalen-2-yl (PIN, the Blue Book), pyridin-4-yl,
    cyclohexa-2,4-dien-1-yl, 1-carboxyethyl and 2-hydroxyethyl: every
    characteristic group is cited as a prefix,:40526), cyanomethyl. Skips acyl
    radicals (named from their acid) and names that end in a contracted prefix."""
    c = mol.GetAtomWithIdx(idx)
    if (c.GetSymbol() != 'C' or c.GetNumRadicalElectrons() != 1 or c.GetFormalCharge()
            or _is_acyl_atom(mol, idx)):
        return ''
    prefix, wide = _placeholder_prefix(mol, idx, _tier_is_general())
    if not prefix:
        return ''
    if prefix == 'benzyl':
        prefix = 'phenylmethyl'
    elif prefix.endswith('phenyl') and c.GetIsAromatic() and '(' not in prefix:
        # A radical is named with the suffix 'yl' on the parent hydride name, not
        # with the substituent prefix: (the Blue Book) "named by
        # adding the suffix 'yl' to the name of the parent hydride, eliding the final
        # letter 'e'", so C6H5* is 'benzenyl'; the cation in the same table is
        # 'phenyl cation... benzenylium (PIN)' (:41537). 'phenyl' (preferred prefix,
        #:24450) is the substituent-prefix form. The numbering is phenyl's
        # (attachment at 1); with a substituent every locant is cited:
        # 4-methylbenzen-1-yl.
        base = prefix[:-len('phenyl')]
        prefix = f"{base}benzen-1-yl" if base else 'benzenyl'
    if prefix.rstrip(')]}').endswith(_CONTRACTED_PREFIX_ENDINGS):
        return ''
    import re
    m = re.search(r"([\da-z,]+)-(?:di|tetra|hexa|octa|deca)hydro.*-(\d+[a-z]?)-yl$", prefix)
    if m and m.group(2) in m.group(1).split(','):
        # The radical sits on a position the hydro prefix itself adds H to: the
        # Blue Book uses 'added hydrogen' there -- '(C60-Ih)[5,6]fulleren-1(9H)-yl
        # (PIN)' vs '1,9-dihydro...fulleren-1-yl' (the Blue Book), so the
        # hydro form is not shipped as a PIN.
        if not _tier_is_general():
            return ''
        wide = True
    return _ship(mol, prefix, wide)


# === MULTI-CENTRE RADICALS,, ===

_SIMPLE_MULT = {2: 'di', 3: 'tri', 4: 'tetra'}


def _ring_polyvalent_name(frag, centers: List[int]) -> str:
    """'<ring>-<locants>-diyl' for k monovalent centres on an UNSUBSTITUTED
    carbocyclic monocycle (cyclopropane... or benzene), locants as low as
    possible over every numbering of the ring, e.g. 'benzene-1,4-diyl
    (PIN)' (the Blue Book), 'cyclopropane-1,2-diyl'. '' otherwise
    (substituted, hetero, fused or partly unsaturated rings are not built here)."""
    ri = frag.GetRingInfo()
    if ri.NumRings() != 1:
        return ''
    ring = list(ri.AtomRings()[0])
    if len(ring) != frag.GetNumAtoms() or any(frag.GetAtomWithIdx(i).GetSymbol() != 'C' for i in ring):
        return ''
    n = len(ring)
    aromatic = all(frag.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
    if aromatic and n == 6:
        parent = 'benzene'
    elif not any(frag.GetAtomWithIdx(i).GetIsAromatic() for i in ring) and all(
            b.GetBondType() == Chem.BondType.SINGLE for b in frag.GetBonds()):
        parent = 'cyclo' + _get_chain_prefix(n) + 'ane'
    else:
        return ''
    # ring atoms in cyclic order
    order = [ring[0]]
    while len(order) < n:
        nxt = [nb.GetIdx() for nb in frag.GetAtomWithIdx(order[-1]).GetNeighbors()
               if nb.GetIdx() in ring and nb.GetIdx() not in order]
        if not nxt:
            return ''
        order.append(nxt[0])
    best = None
    for start in range(n):
        for step in (1, -1):
            pos = {order[(start + step * k) % n]: k + 1 for k in range(n)}
            locs = tuple(sorted(pos[c] for c in centers))
            if best is None or locs < best:
                best = locs
    k = len(centers)
    mult = _SIMPLE_MULT.get(k)
    if mult is None:
        return ''
    return f"{parent}-{','.join(map(str, best))}-{mult}yl"


def _polyvalent_central_name(frag, centers: List[int]) -> str:
    from .ions import emit_parent_hydride_polyvalent_suffixes
    try:
        name = emit_parent_hydride_polyvalent_suffixes(frag, [(c, 1) for c in centers])
    except Exception:
        name = ''
    return name or _ring_polyvalent_name(frag, centers)


def _terminal_radical_group(mol, idx: int):
    """(term, compound, terminal_atoms, attach_idx) for a radical centre that is
    the free end of a terminal group on a carbon: oxyl, peroxyl, sulfanyl,
    disulfanyl, methyl (an exocyclic CH2.). None otherwise."""
    a = mol.GetAtomWithIdx(idx)
    if a.GetNumRadicalElectrons() != 1 or a.GetFormalCharge() or a.GetDegree() != 1:
        return None
    y = a.GetNeighbors()[0]
    sym = a.GetSymbol()
    if sym == 'C' and a.GetTotalNumHs() == 2 and y.GetSymbol() == 'C':
        return ('methyl', False, {idx}, y.GetIdx())
    if sym in ('O', 'S') and not a.GetTotalNumHs():
        base = {'O': 'oxyl', 'S': 'sulfanyl'}[sym]
        if y.GetSymbol() == 'C' and not _is_acyl_atom(mol, y.GetIdx()):
            return (base, True, {idx}, y.GetIdx())
        if y.GetSymbol() == sym and y.GetDegree() == 2 and not y.GetFormalCharge():
            z = [nb for nb in y.GetNeighbors() if nb.GetIdx() != idx][0]
            if z.GetSymbol() == 'C':
                return ({'O': 'peroxyl', 'S': 'disulfanyl'}[sym], True, {idx, y.GetIdx()}, z.GetIdx())
    return None


def _name_multiplied_terminal_radicals(mol, sites) -> str:
    """: identical terminal radical groups on a multivalent central group
    are named multiplicatively: '(cyclopropane-1,2-diyl)dimethyl (PIN)',
    '(2,4-dimethylpentane-2,4-diyl)bis(oxyl) (PIN)', '(cyclobutane-1,3-diyl)
    bis(peroxyl) (PIN)', '(naphthalene-2,6-diyl)bis(disulfanyl) (PIN)'
    (the Blue Book-40751): 'di' before 'methyl', 'bis(...)' before the
    compound radical terms."""
    groups = [_terminal_radical_group(mol, s['atom_idx']) for s in sites]
    if any(g is None for g in groups) or len({g[0] for g in groups}) != 1:
        return ''
    attach = [g[3] for g in groups]
    terminal = set().union(*(g[2] for g in groups))
    if len(set(attach)) != len(attach) or terminal & set(attach):
        return ''
    work = Chem.RWMol(mol)
    for c in attach:
        work.GetAtomWithIdx(c).SetIntProp('_mc', 1)
    for i in sorted(terminal, reverse=True):
        work.RemoveAtom(i)
    frag = work.GetMol()
    centers = []
    for a in frag.GetAtoms():
        if a.HasProp('_mc'):
            a.SetNumRadicalElectrons(1)
            a.SetNoImplicit(True)
            a.SetNumExplicitHs(max(0, a.GetTotalNumHs()))
            a.ClearProp('_mc')
            centers.append(a.GetIdx())
    try:
        Chem.SanitizeMol(frag)
    except (RuntimeError, ValueError):
        return ''
    central = _polyvalent_central_name(frag, centers)
    if not central:
        return ''
    term, compound = groups[0][0], groups[0][1]
    k = len(sites)
    mult = (_COMPOUND_MULTIPLIER if compound else _SIMPLE_MULT).get(k)
    if mult is None:
        return ''
    tail = f"{mult}({term})" if compound else f"{mult}{term}"
    return _ship(mol, f"({central}){tail}", False)


def _name_multi_acyl_radical(mol, sites) -> str:
    """ with several acyl radical centres: named from the polyacid,
    'benzene-1,4-dicarbonyl (PIN)', 'benzene-1,4-disulfinyl (PIN)'
    (the Blue Book-40618)."""
    idxs = [s['atom_idx'] for s in sites]
    if any(mol.GetAtomWithIdx(i).GetNumRadicalElectrons() != 1 or mol.GetAtomWithIdx(i).GetFormalCharge()
           or not _is_acyl_atom(mol, i) for i in idxs):
        return ''
    work = Chem.RWMol(mol)
    try:
        for i in idxs:
            a = work.GetAtomWithIdx(i)
            a.SetNumRadicalElectrons(0)
            o = work.AddAtom(Chem.Atom(8))
            work.AddBond(i, o, Chem.BondType.SINGLE)
            a.SetNoImplicit(True)
        acid = work.GetMol()
        Chem.SanitizeMol(acid)
    except (RuntimeError, ValueError):
        return ''
    acid_name = _reenter_neutral_name(acid)
    acyl = _acyl_from_acid_name(acid_name) if acid_name.endswith('acid') else ''
    return _ship(mol, acyl, False)


def _name_multiplied_amidyl_iminyl(mol, sites) -> str:
    """: compound suffixes on a polyamide / polyimine parent take 'bis'
    and enclosing marks: 'methanebis(iminyl) (PIN)', 'benzene-1,2-bis(carbox-
    amidyl) (PIN)', 'butanebis(amidyl) (PIN)' (the Blue Book-40670). Each
    centre is a monovalent N whose parent is the N-H form of an amide or imine."""
    idxs = [s['atom_idx'] for s in sites]
    k = len(idxs)
    if any(mol.GetAtomWithIdx(i).GetSymbol() != 'N' or mol.GetAtomWithIdx(i).GetNumRadicalElectrons() != 1
           or mol.GetAtomWithIdx(i).GetFormalCharge() for i in idxs):
        return ''
    work = Chem.RWMol(mol)
    try:
        for i in idxs:
            a = work.GetAtomWithIdx(i)
            a.SetNumRadicalElectrons(0)
            a.SetNoImplicit(True)
            a.SetNumExplicitHs(a.GetTotalNumHs() + 1)
        neutral_mol = work.GetMol()
        Chem.SanitizeMol(neutral_mol)
    except (RuntimeError, ValueError):
        return ''
    neutral = _reenter_neutral_name(neutral_mol)
    mult = _SIMPLE_MULT.get(k)
    cmult = _COMPOUND_MULTIPLIER.get(k)
    if not neutral or not mult or not cmult:
        return ''
    for base, term in (('carboxamide', 'carboxamidyl'), ('amide', 'amidyl'), ('imine', 'iminyl')):
        end = mult + base
        if neutral.endswith(end):
            return _ship(mol, neutral[:-len(end)] + f"{cmult}({term})", False)
    return ''


def _name_multicentre_radical(mol, sites) -> str:
    for producer in (_name_multi_acyl_radical, _name_multiplied_amidyl_iminyl,
                     _name_multiplied_terminal_radicals):
        named = producer(mol, sites)
        if named:
            return named
    # several monovalent centres on one unsubstituted carbocycle: 'benzene-1,4-diyl'
    idxs = [s['atom_idx'] for s in sites]
    if all(mol.GetAtomWithIdx(i).GetNumRadicalElectrons() == 1 and mol.GetAtomWithIdx(i).GetSymbol() == 'C'
           for i in idxs):
        return _ship(mol, _ring_polyvalent_name(mol, idxs), False)
    return ''


def _name_nitrene(mol, idx: int) -> str:
    """Nitrenes R-N: (a neutral N with two radical electrons and one heavy
    neighbour), / /:
      - on a hydrazine N: '(CH3)2N-N: dimethylhydrazinylidene (PIN)'
        (the Blue Book) -- the parent hydride hydrazine + 'ylidene';
      - on an imidoyl carbon: '(N-methylethanimidoyl)azanylidene (PIN)' (:43362),
        the acyl group from its acid;
      - otherwise the PIN is an '-aminylidene' / '-amidylidene' name
        ('benzenaminylidene (PIN)':40570, 'acetamidylidene (PIN)':40649) that
        OPSIN cannot parse, so the PIN tier gives no name and a general tier ships
        the parseable '<R>azanylidene' (method (2)) with a non-PIN label."""
    n = mol.GetAtomWithIdx(idx)
    if (n.GetSymbol() != 'N' or n.GetNumRadicalElectrons() != 2 or n.GetFormalCharge()
            or n.GetDegree() != 1 or n.GetTotalNumHs()):
        return ''
    y = n.GetNeighbors()[0]
    general = _tier_is_general()
    from ..assembly.naming_utils import enclose_if_compound
    if y.GetSymbol() == 'N':
        if (y.GetFormalCharge() or y.GetNumRadicalElectrons() or y.GetIsAromatic()
                or any(b.GetBondType() != Chem.BondType.SINGLE for b in y.GetBonds())):
            return ''
        prefixes = []
        for sub in y.GetNeighbors():
            if sub.GetIdx() == idx:
                continue
            if (sub.GetSymbol() != 'C' or sub.GetFormalCharge() or sub.GetNumRadicalElectrons()
                    or _is_acyl_atom(mol, sub.GetIdx())):
                return ''
            frag = _collect_fragment_excluding(mol, sub.GetIdx(), {idx, y.GetIdx()})
            prefix, wide = _n_substituent_prefix(mol, frag, sub.GetIdx(), general)
            if not prefix or wide:
                return ''
            prefixes.append(prefix)
        return _ship(mol, _aminoxyl_prefix_string(prefixes) + 'hydrazinylidene', False)
    if y.GetSymbol() != 'C' or y.GetFormalCharge() or y.GetNumRadicalElectrons():
        return ''
    imidoyl = any(b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(y).GetSymbol() == 'N'
                  for b in y.GetBonds())
    if imidoyl:
        acyl = _acyl_from_acid_name(_acid_name_with_oh_at_nitrene(mol, idx))
        if acyl:
            return _ship(mol, f"{enclose_if_compound(acyl)}azanylidene", False)
        return ''
    if not general:
        return ''
    if _is_acyl_atom(mol, y.GetIdx()):
        acyl = _acyl_from_acid_name(_acid_name_with_oh_at_nitrene(mol, idx))
        prefix = acyl
    else:
        frag = _collect_fragment_excluding(mol, y.GetIdx(), {idx})
        prefix, _ = _n_substituent_prefix(mol, frag, y.GetIdx(), True)
    if not prefix:
        return ''
    return _ship(mol, f"{enclose_if_compound(prefix)}azanylidene", True)


def _acid_name_with_oh_at_nitrene(mol, idx: int) -> str:
    """The acid made by turning the nitrene N into OH (for the acyl group on it)."""
    work = Chem.RWMol(mol)
    try:
        a = work.GetAtomWithIdx(idx)
        a.SetAtomicNum(8)
        a.SetNumRadicalElectrons(0)
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(1)
        acid = work.GetMol()
        Chem.SanitizeMol(acid)
    except (RuntimeError, ValueError):
        return ''
    name = _reenter_neutral_name(acid)
    return name if name.endswith('acid') else ''


# ionic parent hydrides (mononuclear) that a radical centre can sit on.
_IONIC_PARENT = {
    ('N', -1): 'azanide', ('N', 1): 'azanium', ('O', 1): 'oxidanium',
    ('S', 1): 'sulfanium', ('Se', 1): 'selanium', ('B', -1): 'boranuide',
    ('P', 1): 'phosphanium', ('P', -1): 'phosphanide', ('As', 1): 'arsanium',
    ('C', 1): 'methylium', ('C', -1): 'methanide',
}


def _name_mononuclear_radical_ion(mol) -> str:
    """ "Radical ions derived from parent hydrides" (the Blue Book-
    43422): the ionic parent hydride's name + 'yl'/'ylidene', "with elision of
    the final letter 'e'", substituents as prefixes: trimethylboranuidyl (PIN,
    :43430), (methoxycarbonyl)methanidylidene (PIN,:43439), methyliumyl (PIN,
    :43443), propyloxidaniumyl (PIN,:43515), acetylazanidyl (PIN,:43517).
    Scope: one fragment, exactly one charged atom, which also carries the only
    radical electrons (1 or 2); a carbon centre only when no substituent is a
    plain carbon group (a carbon chain would be the parent instead)."""
    if len(Chem.GetMolFrags(mol)) != 1:
        return ''
    charged = [a for a in mol.GetAtoms() if a.GetFormalCharge()]
    radical = [a for a in mol.GetAtoms() if a.GetNumRadicalElectrons()]
    if len(charged) != 1 or len(radical) != 1 or charged[0].GetIdx() != radical[0].GetIdx():
        return ''
    x = charged[0]
    if x.GetFormalCharge() > 0 and x.GetNumRadicalElectrons() == 2:
        # RDKit counts the EMPTY orbital of an '-ylium' cation (C[O+], [NH2+],
        # c1ccccc1[S+]) as two radical electrons. Those are cations --
        # 'methoxylium (PIN)', 'azanylium', '1H-pyrrole-2-carboxamidylium (PIN)'
        # -- not radical ions, and the SMILES cannot tell the two readings
        # apart, so no round trip can either. They belong to the separate
        # build (user decision Q2 = A); this producer leaves them alone.
        return ''
    parent = _IONIC_PARENT.get((x.GetSymbol(), x.GetFormalCharge()))
    suffix = {1: 'yl', 2: 'ylidene'}.get(x.GetNumRadicalElectrons())
    if parent is None or suffix is None:
        return ''
    general = _tier_is_general()
    prefixes = []
    for sub in x.GetNeighbors():
        if sub.GetSymbol() != 'C':
            return ''
        if _is_acyl_atom(mol, sub.GetIdx()):
            acid = _acid_name_with_oh_replacing(mol, x.GetIdx())
            acyl = _acyl_from_acid_name(acid) if acid else ''
            if acyl:
                prefixes.append(acyl)
                continue
        if x.GetSymbol() == 'C' and not any(
                b.GetBondType() != Chem.BondType.SINGLE
                and b.GetOtherAtom(sub).GetSymbol() in ('O', 'S', 'N')
                for b in sub.GetBonds()):
            return ''  # a plain carbon group: the chain is the parent, not methane
        frag = _collect_fragment_excluding(mol, sub.GetIdx(), {x.GetIdx()})
        prefix, wide = _n_substituent_prefix(mol, frag, sub.GetIdx(), general)
        if not prefix or wide:
            return ''
        prefixes.append(prefix)
    name = _aminoxyl_prefix_string(prefixes) + parent[:-1] + suffix if parent.endswith('e') \
        else _aminoxyl_prefix_string(prefixes) + parent + suffix
    return _ship(mol, name, _n_radical_ion_has_suffix_parent(mol, x))


def _n_radical_ion_has_suffix_parent(mol, x) -> bool:
    """True when the PIN of this N radical ion is built on a SUFFIX-derived ion,
    so the azanium/azanide name above is valid but not the PIN.

     "Radical ions on ionic suffix groups" (the Blue Book,:43497):
    "When ions may be named by using modified suffixes (see and
    , the suffixes denoting radical centers are added to the name of
    the cationic or anionic parent hydride": 'benzenaminiumyl (PIN)' (:43501),
    'methanaminidyl (PIN)' (:43503), and 'methanaminiumyl (PIN)' (:17608).
    A carbon group on N+ makes an amine (or, for an acyl group, an amide: Table
    7.4:41421 'amidium') cation, and a non-acyl carbon group on N- an amine anion
     :41049 'aminide'). Those '-aminiumyl'/'-aminidyl' PINs cannot be
    verified (the round-trip parser reads none of them), so the verified azanium
    name ships at a general tier. An acyl group on N- stays the PIN: 'amidide' is
    not recommended:41071) and the Blue Book names CH3-CO-N(.-)
    'acetylazanidyl (PIN)':43517). H-only N ('azanidyl (preselected
    name)',:43426) and the other elements have no suffix-derived ion here."""
    if x.GetSymbol() != 'N' or not x.GetDegree():
        return False
    if x.GetFormalCharge() > 0:
        return True
    return not all(_is_acyl_atom(mol, sub.GetIdx()) for sub in x.GetNeighbors())


def _acid_name_with_oh_replacing(mol, idx: int) -> str:
    """The acid made by replacing atom ``idx`` (and its charge and radical) with
    OH -- the formal parent of the acyl group on a radical-ion centre."""
    work = Chem.RWMol(mol)
    try:
        a = work.GetAtomWithIdx(idx)
        a.SetAtomicNum(8)
        a.SetFormalCharge(0)
        a.SetNumRadicalElectrons(0)
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(1)
        acid = work.GetMol()
        Chem.SanitizeMol(acid)
    except (RuntimeError, ValueError):
        return ''
    name = _reenter_neutral_name(acid)
    return name if name.endswith('acid') else ''


_METHANE_RADICAL = {1: 'methyl', 2: 'methylidene', 3: 'methylidyne'}


def _name_methane_centred_radical(mol, idx: int) -> str:
    """A carbon radical centre all of whose neighbours are ring atoms: the
    parent is methane (no chain can extend through it), the ring groups are
    prefixes, the suffix is methyl / methylidene / methylidyne,
    : 'diphenylmethylidene (PIN)' (the Blue Book,
    triphenylmethyl, cyclohexyl(phenyl)methyl."""
    c = mol.GetAtomWithIdx(idx)
    suffix = _METHANE_RADICAL.get(c.GetNumRadicalElectrons())
    if (c.GetSymbol() != 'C' or suffix is None or c.GetFormalCharge() or c.IsInRing()
            or c.GetDegree() < 2 or any(not n.IsInRing() for n in c.GetNeighbors())):
        return ''
    general = _tier_is_general()
    prefixes = []
    for n in c.GetNeighbors():
        frag = _collect_fragment_excluding(mol, n.GetIdx(), {idx})
        prefix, wide = _n_substituent_prefix(mol, frag, n.GetIdx(), general)
        if not prefix or wide:
            return ''
        prefixes.append(prefix)
    return _ship(mol, _aminoxyl_prefix_string(prefixes) + suffix, False)


# Unbranched homogeneous heteroatom chains: 2 -> hydrazine/disilane...
_HOMOGENEOUS_STEM = {'Si': 'silane', 'Ge': 'germane', 'Sn': 'stannane', 'P': 'phosphane',
                     'S': 'sulfane', 'N': 'azane'}


def _name_homogeneous_chain_radical(mol, sites) -> str:
    """Radical centres (one electron each) on an UNSUBSTITUTED, unbranched chain
    of one heteroatom kind, named from the parent hydride with the lowest
    locants for the free valences,: 'trisilan-2-yl (PIN)',
    'hydrazine-1,2-diyl (PIN)' (the Blue Book,."""
    atoms = list(mol.GetAtoms())
    el = atoms[0].GetSymbol()
    if (len(atoms) < 2 or any(a.GetSymbol() != el for a in atoms) or el not in _HOMOGENEOUS_STEM
            or any(a.GetFormalCharge() for a in atoms) or mol.GetRingInfo().NumRings()
            or any(a.GetDegree() > 2 for a in atoms)
            or any(b.GetBondType() != Chem.BondType.SINGLE for b in mol.GetBonds())
            or any(s['n_electrons'] != 1 for s in sites)):
        return ''
    n = len(atoms)
    parent = 'hydrazine' if (el == 'N' and n == 2) else (
        ('di' if n == 2 else _get_chain_prefix(n)[:-1] + 'a' if n > 4 else {3: 'tri', 4: 'tetra'}[n])
        + _HOMOGENEOUS_STEM[el])
    end = [a.GetIdx() for a in atoms if a.GetDegree() <= 1][0]
    order = [end]
    while len(order) < n:
        order.append([x.GetIdx() for x in mol.GetAtomWithIdx(order[-1]).GetNeighbors()
                      if x.GetIdx() not in order][0])
    centres = [s['atom_idx'] for s in sites]
    if len(centres) == 1 and mol.GetAtomWithIdx(centres[0]).GetDegree() <= 1:
        # A single free valence at a chain end: the Blue Book writes such prefixes
        # without a locant ('hydrazinyl', 'disilanyl'), a spelling not derived
        # here -- left to the other producers.
        return ''
    best = min(tuple(sorted(o.index(c) + 1 for c in centres)) for o in (order, order[::-1]))
    k = len(centres)
    mult = {1: '', 2: 'di', 3: 'tri'}.get(k)
    if mult is None:
        return ''
    tail = f"-{','.join(map(str, best))}-{mult}yl"
    name = (parent[:-1] if not mult and parent.endswith('e') else parent) + tail
    return _ship(mol, name, False)


def _compose_oxyl_name(mol, parent: str, additive_suffix: str, systematic_suffix: str) -> str:
    """ (VERIFIED, the Blue Book-40709): a radical formed by
    removing the hydrogen of a hydroxy/peroxy characteristic group is named
    in TWO ways -- (1) additively, ``R`` + 'oxyl'/'peroxyl' (e.g.
    '(chloroacetyl)oxyl (PIN)', 'hexanoylperoxyl (PIN)', 'butanoyloxyl
    (PIN)'); (2) by substituting the preselected parent radical
    'oxidanyl'/'dioxidanyl' with R (e.g. '(chloroacetyl)oxidanyl',
    'hexanoyldioxidanyl') -- and the Blue Book states, in terms: **"Method
    (1) generates preferred IUPAC names."** So method (1) is tried FIRST and
    used whenever it verifies; method (2) is kept only as a round-trip-
    verified fallback for an R shape method (1) cannot express. '' if
    NEITHER verifies (fail closed -- never a wrong name).

    A compound/locanted R takes enclosing marks via the same
    ``enclose_if_compound`` gate every other compound substituent prefix in
    this codebase uses, for both methods alike."""
    from ..assembly.naming_utils import enclose_if_compound
    enclosed = enclose_if_compound(parent)
    additive = f"{enclosed}{additive_suffix}"
    if _radical_name_round_trips(mol, additive):
        return additive
    systematic = f"{enclosed}{systematic_suffix}"
    if _radical_name_round_trips(mol, systematic):
        return systematic
    return ''


def _name_peroxyl_radical(mol, radical_idx: int, bridge_o_idx: int) -> str:
    """: name an R-O-O. peroxyl radical. ``bridge_o_idx`` is the
    peroxide oxygen bonded to the radical oxygen; its remaining heavy
    neighbour (excluding the radical O) is the R group's attachment point,
    named through the SAME substituent-naming chokepoint the plain oxyl
    branch above uses (``name_substituent_fragment``) so a compound R
    (branched, substituted, aromatic,...) is never dropped. Composes via
    ``_compose_oxyl_name`` (method (1) 'R + peroxyl', PIN, tried first;
    method (2) '(R)dioxidanyl' as a verified fallback) -- VERIFIED -r
    round-trip witnesses: 'methylperoxyl' for ``[O]OC``, 'tert-butylperoxyl'
    for ``[O]OC(C)(C)C``. The constraint is on the BRIDGE, not on R: the
    bridge oxygen must have exactly one non-radical neighbour and that
    neighbour must be carbon (a single carbon ATTACHMENT point) -- the R
    group hanging off it is unconstrained and, per the chokepoint above, can
    be branched/substituted/aromatic. Anything else (a second substituent on
    the bridge oxygen, or a non-carbon attachment) fails closed to '' so the
    caller's plain 'oxyl' fallback -- never a wrong name -- takes over."""
    bridge_atom = mol.GetAtomWithIdx(bridge_o_idx)
    r_neighbors = [n for n in bridge_atom.GetNeighbors() if n.GetIdx() != radical_idx]
    if len(r_neighbors) != 1 or r_neighbors[0].GetSymbol() != 'C':
        return ''
    attach_idx = r_neighbors[0].GetIdx()
    frag = _collect_fragment_excluding(mol, attach_idx, {radical_idx, bridge_o_idx})
    if not frag:
        return ''
    from ..assembly.substituent_naming import name_substituent_fragment
    parent = name_substituent_fragment(mol, frag, attach_idx, [bridge_o_idx]) or ''
    if not parent:
        return ''
    return _compose_oxyl_name(mol, parent, 'peroxyl', 'dioxidanyl')


def name_oxyl_radical(mol, radical_site: Dict[str, Any]) -> str:
    """
    Name an oxyl radical (RO.) or a peroxyl radical (ROO.).

    Oxygen-centered radicals named as R-oxyl for the small retained set of
    UNSUBSTITUTED R groups (methoxyl/ethoxyl/phenoxyl/propoxyl/.../hexoxyl),
    or per method (1) as ``(R)oxyl``/``R-peroxyl`` -- the VERIFIED
    PIN form ("Method (1) generates preferred IUPAC names",
    the Blue Book) -- when R itself carries a further substituent, via
    ``_compose_oxyl_name``. A2 generalisation: the R group is named
    through the existing substituent-naming pipeline instead of being
    collapsed to the unsubstituted retained form (the pre- bug: any
    aromatic-O radical mapped unconditionally to 'phenoxyl', dropping every
    ring substituent) or misnamed by a bare carbon count that ignored
    branching/unsaturation (A2 a review hardening, see
    ``_is_plain_alkyl_radical_fragment``).

    Args:
        mol: RDKit Mol object
        radical_site: Dictionary from get_radical_sites

    Returns:
        Oxyl/peroxyl radical name (e.g., 'methoxyl', 'phenoxyl',
        '(4-hydroxyphenyl)oxyl', 'methylperoxyl'), or '' if the R group
        cannot be named (fail-closed -- never a wrong name).

    Example:
        >>> mol = Chem.MolFromSmiles('[O]C')
        >>> sites = get_radical_sites(mol)
        >>> name_oxyl_radical(mol, sites[0])
        'methoxyl'
    """
    radical_idx = radical_site['atom_idx']
    radical_atom = mol.GetAtomWithIdx(radical_idx)

    heavy_neighbors = list(radical_atom.GetNeighbors())
    # The bare 'oxyl' denotes HO. / O.. only; every fallback to it below is
    # shipped only when the strict round trip confirms it (C[Si](C)(C)[O] would
    # otherwise be named 'oxyl', which drops the whole R side).
    if len(heavy_neighbors) != 1:
        return _ship(mol, 'oxyl', False)

    neighbor = heavy_neighbors[0]

    if neighbor.GetSymbol() == 'O':
        # R-O-O. peroxyl radical: the free valence sits on the
        # OUTER oxygen of a peroxide bridge. '' (fail-closed) falls through
        # to the plain 'oxyl' default below rather than mis-cutting the
        # bridge -- never a wrong name.
        peroxyl_name = _name_peroxyl_radical(mol, radical_idx, neighbor.GetIdx())
        if peroxyl_name:
            return peroxyl_name
        return _ship(mol, 'oxyl', False)

    if neighbor.GetSymbol() == 'N':
        # R2N-O. aminoxyl radicals. When the producer declines, the
        # answer is '' (no name), never the bare 'oxyl': that string denotes
        # HO-less O. and drops the whole N side, and with the OPSIN gate off it
        # used to ship as a pin_verified name.
        return _name_n_oxyl_radical(mol, radical_idx, neighbor)

    if neighbor.GetSymbol() != 'C':
        return _ship(mol, 'oxyl', False)

    attach_idx = neighbor.GetIdx()

    if neighbor.GetIsAromatic():
        # name_radical's RETAINED_RADICALS canonical-SMILES lookup already
        # returns 'phenoxyl' for the bare unsubstituted ring ('[O]c1ccccc1')
        # BEFORE classify_radical/this function ever run (name_radical.py:
        # the `canonical in RETAINED_RADICALS` check precedes the
        # classify_radical dispatch) -- but ONLY when style != 'systematic'.
        # With style='systematic' that lookup is skipped entirely (see
        # name_radical's `if style != 'systematic':` guard), so the bare
        # unsubstituted ring DOES reach this function and this branch in
        # that mode -- the check below is real dispatch, not dead code.
        parent = _name_oxyl_parent_group(mol, radical_idx, attach_idx)
        if not parent:
            return ''
        if parent == 'phenyl':
            return _ship(mol, 'phenoxyl', False)
        return _compose_oxyl_name(mol, parent, 'oxyl', 'oxidanyl')

    # Aliphatic R: keep the existing retained '-oxyl' contraction for a PLAIN
    # LINEAR SATURATED hydrocarbon chain (see the tightened
    # _is_plain_alkyl_radical_fragment) -- unchanged PINs -- and generalise
    # via the same substituent pipeline for anything else (branching,
    # unsaturation, a ring, or any heteroatom substituent the old carbon
    # count could not see). The retained candidate is additionally verified
    # with a -r OPSIN round-trip before being trusted (belt-and-suspenders
    # over the shape guard): on a mismatch it falls through to the
    # systematic '(<parent>)oxyl'/'(<parent>)oxidanyl' composition below
    # instead of shipping an unverified name.
    acyl_oxyl = _name_acyl_oxyl_radical(mol, radical_idx, attach_idx)
    if acyl_oxyl:
        return acyl_oxyl

    if _is_plain_alkyl_radical_fragment(mol, radical_idx, attach_idx):
        carbon_count = _count_chain_carbons(mol, attach_idx, {radical_idx})

        # The retained contractions are a CLOSED list,:40681):
        # a longer chain takes method (1), '<R>oxyl' ('pentyloxyl'), below.
        ALKOXY_NAMES = {
            1: 'methoxyl',
            2: 'ethoxyl',
            3: 'propoxyl',
            4: 'butoxyl',
        }

        retained = ALKOXY_NAMES.get(carbon_count)
        if retained and _radical_name_round_trips(mol, retained):
            return retained
        # shape guard passed but the candidate still failed to verify --
        # fall through to the systematic branch below.

    parent = _name_oxyl_parent_group(mol, radical_idx, attach_idx)
    if not parent:
        return ''
    return _compose_oxyl_name(mol, parent, 'oxyl', 'oxidanyl')


def name_divalent_radical(mol, radical_site: Dict[str, Any]) -> str:
    """
    Name a divalent (carbene-like) radical -> -ylidene, via route_charged.

    169.6-03: delegates to the chokepoint (replacing the carbon-counting body).

    Example:
        >>> mol = Chem.MolFromSmiles('[CH2]')
        >>> sites = get_radical_sites(mol)
        >>> name_divalent_radical(mol, sites[0])
        'methylidene'
    """
    from .charged_router import route_charged
    return route_charged(mol, 'pin')


def name_trivalent_radical(mol, radical_site: Dict[str, Any]) -> str:
    """
    Name a trivalent (carbyne-like) radical -> -ylidyne, via route_charged.

    169.6-03: delegates to the chokepoint (replacing the carbon-counting body).

    Example:
        >>> mol = Chem.MolFromSmiles('[CH]')
        >>> sites = get_radical_sites(mol)
        >>> name_trivalent_radical(mol, sites[0])
        'methylidyne'
    """
    from .charged_router import route_charged
    return route_charged(mol, 'pin')


def name_aryl_radical(mol, radical_site: Dict[str, Any]) -> str:
    """
    Name an aryl radical (Ar.).

    Aromatic carbon-centered radicals.

    Args:
        mol: RDKit Mol object
        radical_site: Dictionary from get_radical_sites

    Returns:
        Aryl radical name (e.g., 'phenyl', 'naphthyl')

    Example:
        >>> mol = Chem.MolFromSmiles('[c]1ccccc1')
        >>> sites = get_radical_sites(mol)
        >>> name_aryl_radical(mol, sites[0])
        'phenyl'
    """
    # Only exact C6H5. is named here (1): 'benzenyl', see
    # RETAINED_RADICALS). Counting aromatic carbons named a tolyl radical
    # 'phenyl' and naphthalen-2-yl 'naphthyl' -- wrong molecules only the gate
    # stopped; ring radicals are named by _name_carbon_radical instead.
    if Chem.MolToSmiles(mol) == '[c]1ccccc1':
        return 'benzenyl'
    return ''


# === HETEROATOM-CENTRED RADICALS / / / ===
#
# The carbon chokepoint (charged_router.route_charged) names a parent hydride by
# neutralize -> re-enter -> _apply_radical_suffix, which drops the whole 'ane'
# ending (methane->methyl, silane->silyl). That contraction is the rule
# and is correct ONLY for Group-14 mononuclear hydrides / acyclic-hydrocarbon
# termini / monocyclic saturated hydrocarbon rings. For a NON-Group-14 heteroatom
# parent hydride (azane, sulfane, borane,...) the general method elides
# ONLY the final 'e' (azane->azanyl, sulfane->sulfanyl, borane->boranyl) — so the
# chokepoint produced the WRONG 'azyl'/'sulfyl'/'boryl'. These namers own the
# heteroatom cases with the correct, element-keyed contraction and fail closed
# ('' -> the caller's carbon path) for everything else.

# /: Group-14 mononuclear parent hydrides -> drop 'ane'.
_GROUP14_HYDRIDE_RADICAL = {
    'C': 'methane', 'Si': 'silane', 'Ge': 'germane', 'Sn': 'stannane', 'Pb': 'plumbane',
}
# /: non-Group-14 heteroatom parent hydrides -> elide 'e' only.
_ELIDE_E_HYDRIDE_RADICAL = {
    'N': 'azane', 'P': 'phosphane', 'As': 'arsane', 'Sb': 'stibane', 'Bi': 'bismuthane',
    'S': 'sulfane', 'Se': 'selane', 'Te': 'tellane',
    'B': 'borane', 'Al': 'alumane', 'Ga': 'gallane', 'In': 'indigane', 'Tl': 'thallane',
}
_RADICAL_SUFFIX_BY_NE = {1: 'yl', 2: 'ylidene', 3: 'ylidyne'}
#: multiplying prefixes 'bis'/'tris'/... precede compound suffixes.
_COMPOUND_MULTIPLIER = {2: 'bis', 3: 'tris', 4: 'tetrakis'}


def _parent_hydride_radical_name(element: str, n_electrons: int) -> str:
    """/.2 + /.2: name a MONONUCLEAR heteroatom radical from its
    element's parent-hydride name. Group-14 (silane/germane) drop the whole 'ane';
    non-Group-14 (azane/sulfane/borane) elide only the final 'e'.
    -> azanyl, azanylidene, sulfanyl, boranyl, silyl, germyl. '' if unmapped.

    BB (PIN/preselected): HS. sulfanyl (40430); H2N. azanyl (40434); H2B. boranyl,
    'not boryl' (40438); HN: azanylidene (40506); H2Si: silylidene (40484)."""
    suffix = _RADICAL_SUFFIX_BY_NE.get(n_electrons)
    if suffix is None:
        return ''
    if element in _GROUP14_HYDRIDE_RADICAL:
        return _GROUP14_HYDRIDE_RADICAL[element][:-3] + suffix
    if element in _ELIDE_E_HYDRIDE_RADICAL:
        return _ELIDE_E_HYDRIDE_RADICAL[element][:-1] + suffix
    return ''


def _reenter_neutral_name(mol) -> str:
    """Name the neutral parent by re-entering the full pipeline with the OPSIN
    validity gate disabled (radical intermediates are correct-by-construction; the
    outer gate re-checks the final radical name). '' on failure/unknown."""
    smi = Chem.MolToSmiles(mol)
    if not smi:
        return ''
    try:
        from ..namer import Orthonym
        name = Orthonym(style='pin', _disable_opsin_validity_gate=True).name(smi)
    except (RecursionError, ValueError, RuntimeError):
        return ''
    if not name or 'unknown' in name.lower():
        return ''
    return name


def _name_amine_family_radical(mol, radical_site: Dict[str, Any]) -> str:
    """P-71.3.2 (Table 7.2): a radical on a substituted amine / imine / amide N is
    named with the compound suffix -aminyl / -iminyl / -amidyl on the NEUTRAL parent
    name (the N as its principal characteristic group). The cumulative radical
    suffix is appended by eliding the parent name's final 'e' (P-70.3.3.2.1):
      methanamine -> methanaminyl, propan-1-imine -> propan-1-iminyl,
      formamide -> formamidyl.
    Fail-closed ('') when the neutral parent does NOT name the N as a suffix
    (e.g. the retained 'aniline', where the systematic 'benzenamine' would be
    needed for 'benzenaminyl') — never a wrong name."""
    idx = radical_site['atom_idx']
    n_e = radical_site['n_electrons']
    cum = {1: 'yl', 2: 'ylidene'}.get(n_e)
    if cum is None:
        return ''
    work = Chem.RWMol(mol)
    try:
        a = work.GetAtomWithIdx(idx)
        a.SetNumRadicalElectrons(0)
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(a.GetTotalNumHs() + n_e)
        wm = work.GetMol()
        Chem.SanitizeMol(wm)
    except (RuntimeError, ValueError):
        return ''
    neutral = _reenter_neutral_name(wm)
    if not neutral:
        return ''
    # (the Blue Book): 'benzenaminyl (PIN)... (not anilino)'. The
    # retained parent 'aniline' denotes exactly 'benzenamine', which carries the
    # -amine suffix the radical suffix needs; the result is still verified by
    # the strict round trip below / at the gate.
    if neutral.endswith('aniline'):
        neutral = neutral[:-len('aniline')] + 'benzenamine'
    # '-carboxamide'/'-amide'/'-imine'/'-amine' -> elide final 'e' + cumulative suffix.
    for base in ('carboxamide', 'amide', 'imine', 'amine'):
        if neutral.endswith(base):
            name = neutral[:-1] + cum
            #: on an =N. radical the N carries no substitutable H, so the
            # substituent locants on the sole heteroatom centre drop, as in
            # '(CH3)3P=N. trimethyl-lambda5-phosphaniminyl (PIN)' (the Blue Book)
            # and the iminide anion (charged_router._iminide_omit_locants). Kept
            # only when the shorter name still passes the strict round trip.
            import re
            short = re.sub(r'^([A-Z][a-z]?)(?:,\1)*-', '', name)
            if short != name and _radical_identity_round_trips(mol, short):
                return short
            # The neutral parent's principal group may sit on ANOTHER nitrogen
            # (NC(=O)CC[NH]: '3-aminopropanamide' -> '3-aminopropanamidyl', whose
            # radical is on the amide N, a different molecule). Ship only what
            # the strict round trip confirms.
            return name if _radical_identity_round_trips(mol, name) else ''
    return ''


def _name_multiplicative_amine_radical(mol, radical_sites: List[Dict[str, Any]]) -> str:
    """P-71.3.3 method (1): poly-amine radicals named as
    (central-multivalent-group)bis/tris(aminyl). Scope (tight, fail-closed): >=2
    monovalent radical centres, each a nitrogen bearing exactly one carbon neighbour
    (a primary-amine -NH.) on a shared acyclic all-carbon central skeleton, all
    identically derived. -> [NH]CC[NH] (ethane-1,2-diyl)bis(aminyl) (BB 40660, PIN).

    Per P-71.3.3 the parent radical in multiplicative nomenclature is 'azanyl'
    (method 2); method (1) — the PIN — uses the reserved suffix 'aminyl'."""
    centers_c: List[int] = []
    for s in radical_sites:
        a = mol.GetAtomWithIdx(s['atom_idx'])
        if a.GetSymbol() != 'N' or s['n_electrons'] != 1:
            return ''
        heavy = [nb for nb in a.GetNeighbors() if nb.GetSymbol() != 'H']
        if len(heavy) != 1 or heavy[0].GetSymbol() != 'C':
            return ''
        centers_c.append(heavy[0].GetIdx())
    if len(set(centers_c)) != len(centers_c):
        return ''  # two amine N on one carbon -> out of scope
    mult = _COMPOUND_MULTIPLIER.get(len(radical_sites))
    if mult is None:
        return ''
    # Build the central multivalent group: mark the attachment carbons, sever the
    # amine nitrogens, set the carbons as free-valence centres, name via the
    # polyvalent parent-hydride primitive (-> 'ethane-1,2-diyl').
    n_idxs = [s['atom_idx'] for s in radical_sites]
    work = Chem.RWMol(mol)
    for c_idx in centers_c:
        work.GetAtomWithIdx(c_idx).SetIntProp('_mult_center', 1)
    for nidx in sorted(n_idxs, reverse=True):
        work.RemoveAtom(nidx)
    frag = work.GetMol()
    centers: List[tuple] = []
    for a in frag.GetAtoms():
        if a.HasProp('_mult_center'):
            a.SetNumRadicalElectrons(1)
            a.SetNoImplicit(False)
            a.ClearProp('_mult_center')
            centers.append((a.GetIdx(), 1))
    if len(centers) != len(radical_sites):
        return ''
    try:
        Chem.SanitizeMol(frag)
    except (RuntimeError, ValueError):
        return ''
    from .ions import emit_parent_hydride_polyvalent_suffixes
    central = emit_parent_hydride_polyvalent_suffixes(frag, centers)
    if not central:
        return ''
    # The central group is built on a stripped copy, so isotopes on the removed
    # nitrogens and stereo on the skeleton are not in it ([NH]C[C@H](C)[NH] would
    # lose its stereo): the strict round trip decides.
    return _ship(mol, f"({central}){mult}(aminyl)", False)


def name_heteroatom_radical(mol, radical_sites: List[Dict[str, Any]],
                            style: str = 'pin') -> str:
    """Heteroatom-centred radical PINs (called from the route_charged chokepoint
    BEFORE its carbon-centric path so a heteroatom never gets a wrong Group-14
    contraction):

      * / mononuclear parent-hydride radicals — azanyl,
        azanylidene, sulfanyl, boranyl, silyl, germyl,...
      * amine / imine / amide compound-suffix radicals — methanaminyl,
        propan-1-iminyl, formamidyl.
      * multiplicative poly-amine radicals — (ethane-1,2-diyl)bis(aminyl).

    Returns '' (fail-closed) for any carbon-centred or out-of-scope radical so the
    caller's existing carbon path / name_radical helpers run unchanged."""
    if not radical_sites:
        return ''
    # Multi-centre: only the multiplicative poly-amine class is in scope
    # here; every other multi-radical is the carbon polyvalent primitive's job.
    if len(radical_sites) >= 2:
        return _name_multiplicative_amine_radical(mol, radical_sites)

    site = radical_sites[0]
    atom = mol.GetAtomWithIdx(site['atom_idx'])
    element = atom.GetSymbol()
    if element in ('C', 'O'):
        # Carbon -> caller's chokepoint; oxygen -> the oxyl/hydroxyl handler.
        return ''
    heavy_nbrs = [nb for nb in atom.GetNeighbors() if nb.GetSymbol() != 'H']
    # (A) Mononuclear heteroatom parent-hydride radical (only H neighbours).
    if not heavy_nbrs and mol.GetNumHeavyAtoms() == 1:
        # The element name ignores the hydrogen count and isotopes ([SH3] and
        # [15NH2] would be 'sulfanyl' / 'azanyl'): the strict round trip decides.
        return _ship(mol, _parent_hydride_radical_name(element, site['n_electrons']),
                     False)
    # (B) Substituted nitrogen -> amine / imine / amide compound suffix.
    if element == 'N':
        return _name_amine_family_radical(mol, site)
    # Substituted non-N heteroatom radical -> out of scope, fail closed.
    return ''


# === MAIN NAMING FUNCTION ===

def name_radical(mol, style: str = 'pin') -> str:
    """
    Generate IUPAC name for a radical species.

    Workflow:
    1. Get canonical SMILES
    2. Check retained names (unless systematic style)
    3. Detect radical sites
    4. Classify radical type
    5. Generate systematic name

    Args:
        mol: RDKit Mol object with radical center(s)
        style: Naming style ('pin', 'systematic', 'common')

    Returns:
        IUPAC name for the radical (e.g., 'methyl', 'methylidene')

    Example:
        >>> mol = Chem.MolFromSmiles('[CH3]')
        >>> name_radical(mol)
        'methyl'
        >>> mol = Chem.MolFromSmiles('[CH2]')
        >>> name_radical(mol)
        'methylidene'
    """
    if mol is None:
        return ''

    # Get canonical SMILES for lookup
    canonical = Chem.MolToSmiles(mol, canonical=True)

    # Check retained names first (unless systematic requested)
    if style != 'systematic':
        if canonical in RETAINED_RADICALS:
            return RETAINED_RADICALS[canonical]

    # Get radical sites
    sites = get_radical_sites(mol)

    if not sites:
        return ''

    # For single radical, classify and name
    if len(sites) == 1:
        site = sites[0]
        info = classify_radical(mol, site)
        subtype = info['subtype']
        #: an acyl radical (C, P, S... with a double bond to a
        # chalcogen or N) is named from its acid; the older producers below
        # remain the fallback.
        acyl = _name_acyl_radical_from_acid(mol, site['atom_idx'])
        if acyl:
            return acyl
        chain = _name_homogeneous_chain_radical(mol, sites)
        if chain:
            return chain
        # chalcogen analogues and ring-N radicals.
        for producer in (_name_chalcogen_radical, _name_ring_n_radical, _name_carbon_radical,
                         _name_nitrene, _name_methane_centred_radical):
            named = producer(mol, site['atom_idx'])
            if named:
                return named

        # Route to appropriate naming function. The acyl/oxyl/aryl heteroatom-
        # context subtypes keep their structured helpers (they are NOT plain
        # parent-hydride -yl/-ylidene/-ylidyne loss and are not route_charged's
        # scope).
        if subtype == 'oxyl':
            return name_oxyl_radical(mol, site)
        elif subtype == 'acyl':
            return name_acyl_radical(mol, site)
        elif subtype == 'aryl':
            return name_aryl_radical(mol, site)

        # cls 1: the free valence is the MOST senior class,
        # senior to a carboxylic acid — a chain-terminal -COOH on an alkyl
        # radical is demoted to a 'carboxy' prefix (2-carboxyethyl). Try this
        # before route_charged (which H-saturates the acid and mis-drops it).
        if info['radical_type'] == 'monovalent':
            carboxy_name = _name_carboxy_alkyl_radical(mol, site['atom_idx'])
            if carboxy_name:
                return carboxy_name

        # 169.6-03 (CHOKE-01, kill-list): the alkyl / -ylidene / -ylidyne
        # carbon-counting paths are DELETED. Delegate to route_charged, which
        # H-saturates the radical center, re-enters the FULL pipeline (so
        # substituents/unsaturation/branching are named correctly), and re-applies
        # the -yl/-ylidene/-ylidyne suffix. On '' fall through to '' (no
        # carbon-counted misname).
        from .charged_router import route_charged
        return route_charged(mol, style)

    # / / multi-centre producers first (strict round trip).
    multi = _name_multicentre_radical(mol, sites) or _name_homogeneous_chain_radical(mol, sites)
    if multi:
        return multi

    # Multiple radicals: delegate to route_charged multi-site free-valence
    # namer). No first-site oxyl/acyl shortcut — it produced structure-dropping names
    # (e.g. [O]CC[O] -> 'ethoxyl') that only suppressed. Fail closed on ''.
    from .charged_router import route_charged
    return route_charged(mol, style)
