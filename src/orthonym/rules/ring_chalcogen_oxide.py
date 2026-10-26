"""Ring-chalcogen oxide namer /, Wave-2 completion).

A NEUTRAL ring sulfur/selenium/tellurium bearing 1-2 exocyclic terminal =O
is named additively on the intact ring parent — the S-oxide sibling of the
established ``pyridine 1-oxide`` N-oxide path::

    O=S1c2ccccc2-c2ccccc21 -> dibenzo[b,d]thiophene 5-oxide
    O=S1(=O)c2ccccc2-c2ccccc21 -> dibenzo[b,d]thiophene 5,5-dioxide
    O=S1CCCC1 -> thiolane 1-oxide

SCOPE (fail-closed, accuracy-first): a single oxidised ring chalcogen; every
non-ring heavy atom of the molecule is one of its oxide oxygens (bare ring
system otherwise — substituted variants cascade onward); the de-oxidised base
ring must resolve BOTH a name and an authoritative IUPAC locant for the
chalcogen. Anything else returns None — never a wrong name.

Graph/atom classifier (no SMARTS broadening); pure — no mol mutation.
"""

from typing import List, Optional

from rdkit import Chem

from ..perception.molcache import (  # audit 2026-09-03 (S2): per-call atom/bond tuples
    atoms_of,
    bonds_of,
)

_CHALCOGENS = {'S', 'Se', 'Te'}

#: the prefix of the retained morpholine parent whose chalcogen replaces the oxygen, Table 2.3)
_MORPHOLINE_PREFIX = {'S': 'thio', 'Se': 'seleno', 'Te': 'telluro'}

_HW_RING_SIZES = frozenset(range(3, 11))


def _ring_heterone_site(mol, ring: List[int]):
    """``(site, exocyclic atoms, class)`` of a monocycle whose one ring chalcogen carries the terminal double-bonded
    O or S atoms of a heterone, else None. ``class`` is 'one' with =O ('dione' with two) or 'thione' with =S."""
    ring_set = set(ring)
    site, exo, kind = None, [], None
    for i in ring:
        atom = mol.GetAtomWithIdx(i)
        if atom.GetSymbol() not in _CHALCOGENS:
            continue
        here = []
        for nb in atom.GetNeighbors():
            if nb.GetIdx() in ring_set:
                continue
            bond = mol.GetBondBetweenAtoms(i, nb.GetIdx())
            if (nb.GetSymbol() in ('O', 'S') and nb.GetDegree() == 1 and nb.GetTotalNumHs() == 0
                    and bond.GetBondType() == Chem.BondType.DOUBLE):
                here.append(nb)
            else:
                return None          # another exocyclic group on a chalcogen: not this class
        if not here:
            continue
        if site is not None or len({nb.GetSymbol() for nb in here}) != 1:
            return None
        site, exo = i, [nb.GetIdx() for nb in here]
        kind = 'one' if here[0].GetSymbol() == 'O' else 'thione'
    if site is None or (kind == 'thione' and len(exo) != 1):
        return None
    return site, exo, kind


def has_ring_heterone(mol) -> bool:
    """True when a ring of ``mol`` is the ring of a heterone (:func:`_ring_heterone_site`): a ring chalcogen with =O or =S."""
    return mol is not None and any(_ring_heterone_site(mol, list(r)) is not None for r in mol.GetRingInfo().AtomRings())


def _ring_heterone_substituents(mol, ring_set, skip) -> Optional[dict]:
    """The substituents of the ring atoms as ``{ring atom: [name,...]}``; None when one cannot be named."""
    from ..assembly.substituent_enumerator import name_substituent
    from ..errors import is_refusal_sentinel
    out = {}
    for a in sorted(ring_set):
        for nb in mol.GetAtomWithIdx(a).GetNeighbors():
            j = nb.GetIdx()
            if j in ring_set or j in skip or nb.GetAtomicNum() <= 1:
                continue
            frag, seen, stack = [], set(ring_set) | set(skip), [j]
            while stack:
                x = stack.pop()
                if x in seen:
                    continue
                seen.add(x)
                frag.append(x)
                stack.extend(n2.GetIdx() for n2 in mol.GetAtomWithIdx(x).GetNeighbors() if n2.GetIdx() not in seen)
            nm = name_substituent(mol, frag, j)
            if not nm or is_refusal_sentinel(nm):
                return None          # a substituent the cascade refused is no prefix: the producers after this one name it
            out.setdefault(a, []).append(nm)
    return out


def _name_ring_heterone(mol) -> Optional[str]:
    """ "Thioketone and thioaldehyde oxides" (the Blue Book): "When a group -SO- or -SO2- is part of
    a ring system, oxygen atom(s) are expressed substitutively by the suffix '-one' added to the name of the
    heterocycle in which the sulfur atoms are designated as lambda4 or lambda6 atoms... This method generates
    preferred IUPAC names rather than those based on functional class nomenclature, in which the class name 'oxide'
    follows the name of the heterocycle". The same suffix names the sultam, sultone and sultine rings:
    '1lambda6,2-thiazinane-1,1-dione (PIN)',:33260), '1lambda4,2-thiazinan-1-one (PIN)'
    ,:33290), '3-methyl-1,2lambda6-oxathiane-2,2-dione (PIN)' and '1,2lambda4-oxathiolane-2-thione
    (PIN)',:32162 ff.): the Hantzsch-Widman name of the saturated heteromonocycle with the lambda
    of the chalcogen cited with its locant, and the suffix 'one', 'dione' or 'thione' at that locant.

    Scope (fail-closed): one saturated monocycle of 3-10 members made of carbon, oxygen, nitrogen and chalcogen atoms,
    one chalcogen of which carries one or two terminal =O, or one =S; no stereo; no group of the molecule senior to
    the ketone class (the heterone takes the suffix only then,, and no ring ketone beside it. The numbering is
    the Hantzsch-Widman one (the heteroatoms,, then the lowest locant of the suffix, then of all the
    prefixes together (c), (f)) and of the one cited first (g)). The prefixes are the substituent
    names of the ring atoms; an unnameable one leaves the molecule to the producers after this one."""
    from ..assembly.naming_utils import alpha_sort_key, format_substituent_prefix
    from .heterocycles import _monocycle_numberings, build_hw_name
    from .lambda_convention import LAMBDA, nonstandard_bonding_number

    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        return None
    for atom in atoms_of(mol):
        if (atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() != 0
                or atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED):
            return None
    for b in bonds_of(mol):
        if b.GetStereo() != Chem.BondStereo.STEREONONE:
            return None
    ri = mol.GetRingInfo()
    # the ring of the heterone is one of the rings of the molecule that is alone (no fusion, no bridge, no spiro atom);
    # the others are in its substituents
    candidates = [list(r) for r in ri.AtomRings()
                  if any(mol.GetAtomWithIdx(i).GetSymbol() in _CHALCOGENS
                         and any(nb.GetIdx() not in r and mol.GetBondBetweenAtoms(i, nb.GetIdx()).GetBondTypeAsDouble() == 2.0
                                 for nb in mol.GetAtomWithIdx(i).GetNeighbors()) for i in r)]
    if len(candidates) != 1:
        return None
    ring = candidates[0]
    n = len(ring)
    ring_set = set(ring)
    if n not in _HW_RING_SIZES or any(ri.NumAtomRings(i) != 1 for i in ring):
        return None
    for b in bonds_of(mol):
        if b.GetBeginAtomIdx() in ring_set and b.GetEndAtomIdx() in ring_set:
            if b.GetIsAromatic() or b.GetBondType() != Chem.BondType.SINGLE:
                return None      # a saturated ring: the unsaturated ones are not built here
    for i in ring:
        if mol.GetAtomWithIdx(i).GetSymbol() not in ('C', 'N', 'O') + tuple(sorted(_CHALCOGENS)):
            return None
    found = _ring_heterone_site(mol, ring)
    if found is None:
        return None
    site, exo, kind = found
    # a heterone is the suffix only when no group of a senior class is present, and a ring ketone competes with it;
    # a carbonyl, an imine or a nitrile elsewhere (an acyl on the ring nitrogen) is such a group or may become one
    owned = {site, *exo}
    for b in bonds_of(mol):
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        if (b.GetBondTypeAsDouble() >= 2.0 and not b.GetIsAromatic() and not owned & {a1.GetIdx(), a2.GetIdx()}
                and (a1.GetSymbol() != 'C' or a2.GetSymbol() != 'C')):
            return None
    from .benzene import molecule_principal_group_and_fgs
    from .seniority import SENIORITY_ORDER, get_principal_group
    _pg, fgs = molecule_principal_group_and_fgs(mol)
    if fgs is None:
        return None
    # the groups of the chalcogen itself (the sulfonamide of a sultam, the sulfonate of a sultone) are the heterone
    others = {key: [m for m in matches if site not in m] for key, matches in fgs.items()}
    pg, _matches = get_principal_group(mol, others)
    if pg is not None and (pg not in SENIORITY_ORDER or SENIORITY_ORDER.index(pg) <= SENIORITY_ORDER.index('ketone')):
        return None
    subs = _ring_heterone_substituents(mol, ring_set, exo)
    if subs is None:
        return None
    lam = nonstandard_bonding_number(mol, site)
    if lam is None:
        return None

    # the numbering: the heteroatom cascade, then the suffix, then the prefixes together, then the one cited first
    best = None
    for het_key, loc in _monocycle_numberings(mol, ring):
        locants = sorted((loc[ring.index(a)], nm) for a, names in subs.items() for nm in names)
        cited = tuple(sorted((alpha_sort_key(nm), lc) for lc, nm in locants))
        key = (het_key, loc[ring.index(site)], tuple(lc for lc, _nm in locants), cited)
        if best is None or key < best[0]:
            best = (key, loc)
    loc = best[1]
    site_loc = loc[ring.index(site)]
    heteroatoms = sorted((loc[p], mol.GetAtomWithIdx(a).GetSymbol())
                         for p, a in enumerate(ring) if mol.GetAtomWithIdx(a).GetSymbol() != 'C')
    site_symbol = mol.GetAtomWithIdx(site).GetSymbol()
    if n == 6 and heteroatoms == [(1, site_symbol), (4, 'N')]:
        # "Retained names of heteromonocycles", Table 2.3 (the Blue Book): 'thiomorpholine (S instead of O)
        # (PIN) selenomorpholine (Se instead of O) (PIN) telluromorpholine (Te instead of O) (PIN)': the heterone is
        # named on the retained parent, whose numbering (chalcogen 1, nitrogen 4) is the Hantzsch-Widman one
        parent = f"{site_loc}{LAMBDA}{lam}-{_MORPHOLINE_PREFIX[site_symbol]}morpholine"
    else:
        parent = build_hw_name(heteroatoms=heteroatoms, ring_size=n, is_saturated=True, is_aromatic=False,
                               lambda_by_locant={site_loc: lam})
    if not parent:
        return None
    count = len(exo)
    suffix = {1: {'one': 'one', 'thione': 'thione'}, 2: {'one': 'dione', 'thione': 'dithione'}}[count][kind]
    # 'one' begins with a vowel: the final 'e' of the parent is elided (a)); 'dione' and 'thione' do not
    stem = parent[:-1] if suffix == 'one' and parent.endswith('e') else parent
    core = f"{stem}-{','.join([str(site_loc)] * count)}-{suffix}"

    groups = {}
    for a, names in subs.items():
        for nm in names:
            groups.setdefault(nm, []).append(loc[ring.index(a)])
    prefix = "-".join(format_substituent_prefix(nm, sorted(groups[nm]), len(groups[nm]))
                      for nm in sorted(groups, key=alpha_sort_key))
    name = f"{prefix}-{core}" if prefix else core
    if site_symbol != 'S':
        # words the ring heterone for a ring -SO- or -SO2- (:29436,:29444); selenium and tellurium are
        # taken by analogy gives the heterone for S, Se and Te, but for the acyclic ones)
        from ..metrics.provenance import record_non_pin_label
        record_non_pin_label(name, rule="P-64.4.2",
                             detail="the ring heterone of P-64.4.2 is worded for a ring sulfur; the selenium and tellurium "
                                    "ring heterones are named by analogy")
    return name


def name_ring_chalcogen_oxide(mol) -> Optional[str]:
    """Return the additive ring-chalcogen oxide name, else ``None``."""
    if mol is None:
        return None

    # and: the heterone (the sultone and the sultam among them)
    # takes precedence over the additive '...dioxide' form.
    heterone = _name_ring_heterone(mol)
    if heterone is not None:
        return heterone

    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    for atom in atoms_of(mol):
        if atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() != 0:
            return None

    ri = mol.GetRingInfo()
    if ri.NumRings() == 0:
        return None
    ring_atoms = set()
    for r in ri.AtomRings():
        ring_atoms.update(r)

    # Locate the oxidised ring chalcogen and its terminal =O oxygens.
    site = None
    oxide_oxygens = []
    for atom in atoms_of(mol):
        if atom.GetSymbol() not in _CHALCOGENS:
            continue
        if atom.GetIdx() not in ring_atoms:
            continue
        oxos = []
        for nb in atom.GetNeighbors():
            bond = mol.GetBondBetweenAtoms(atom.GetIdx(), nb.GetIdx())
            if (nb.GetSymbol() == 'O' and nb.GetDegree() == 1
                    and nb.GetTotalNumHs() == 0
                    and bond.GetBondType() == Chem.BondType.DOUBLE):
                oxos.append(nb.GetIdx())
        if not oxos:
            continue
        if site is not None:
            return None  # two oxidised sites — not built, fail closed
        if len(oxos) > 2:
            return None
        site = atom.GetIdx()
        oxide_oxygens = oxos
    if site is None:
        return None

    # Bare ring system: every non-ring heavy atom must be an oxide oxygen.
    non_ring = {a.GetIdx() for a in atoms_of(mol)} - ring_atoms
    if non_ring != set(oxide_oxygens):
        return None

    # De-oxidise: remove the =O atoms; the base must sanitize and name.
    rw = Chem.RWMol(mol)
    for a in rw.GetAtoms():
        a.SetIntProp('__rco_orig', a.GetIdx())
    for idx in sorted(oxide_oxygens, reverse=True):
        rw.RemoveAtom(idx)
    base = rw.GetMol()
    try:
        Chem.SanitizeMol(base)
        base_smiles = Chem.MolToSmiles(base, canonical=True)
    except Exception:
        return None

    from ..assembly.fragment_naming import name_fragment_recursively
    try:
        base_name = name_fragment_recursively(base_smiles)
    except Exception:
        return None
    if not base_name or 'unknown' in base_name:
        return None

    # Authoritative locant of the chalcogen under the BASE ring numbering
    # (dibenzothiophene S = 5). Fail closed when the ring-info cascade offers
    # no full iupac_locants map — a guessed locant would be a wrong name.
    base_site = next(
        (i for i in range(base.GetNumAtoms())
         if base.GetAtomWithIdx(i).GetIntProp('__rco_orig') == site), None)
    if base_site is None:
        return None
    try:
        from ..namer import _build_ring_info_for_parent_selection, compute_features
        feats = compute_features(base)
        rinfo = _build_ring_info_for_parent_selection(feats)
    except Exception:
        return None
    iupac = (rinfo or {}).get('iupac_locants')
    if not iupac or base_site not in iupac:
        # Single-ring base: the chalcogen is locant 1 by HW convention
        # (thiolane 1-oxide) — accept ONLY the unambiguous one-heteroatom
        # monocycle; everything else fails closed.
        base_ri = base.GetRingInfo()
        if base_ri.NumRings() == 1:
            ring = base_ri.AtomRings()[0]
            hetero_in_ring = [i for i in ring
                              if base.GetAtomWithIdx(i).GetAtomicNum()
                              not in (1, 6)]
            if hetero_in_ring == [base_site]:
                loc = 1
            else:
                return None
        else:
            return None
    else:
        loc = _lowest_equivalent_locant(base, base_site, iupac)
        if loc is None:
            return None

    name = f"{base_name} {loc}-oxide" if len(oxide_oxygens) == 1 else f"{base_name} {loc},{loc}-dioxide"
    # (the Blue Book): the heterone ('1H-1lambda4-thiophen-1-one', '5H-lambda6-thianthrene-5,5-dione')
    # is the PIN; the class name 'oxide' after the name of the heterocycle is the name the book does not prefer
    from ..metrics.provenance import record_non_pin_label
    record_non_pin_label(name, rule="P-64.4.2",
                         detail="a ring S, Se or Te with exocyclic =O is named as a heterone, not by the class name 'oxide'")
    return name


def _locant_int(loc) -> Optional[int]:
    if isinstance(loc, tuple):
        loc = loc[0]
    return loc if isinstance(loc, int) else None


def _lowest_equivalent_locant(base, site: int, iupac) -> Optional[int]:
    """The chalcogen's locant, lowest over the automorphisms of the base ring system.

    An automorphism of the base renumbers it as validly as the map itself, so the
    chalcogen may take the locant of any atom it maps to; the lowest one is cited
    , low locants within the fixed numbering): 'thianthrene 5-oxide', not
    '10-oxide' (both sulfur atoms are equivalent).
    """
    loc = _locant_int(iupac.get(site))
    if loc is None:
        return None
    for match in base.GetSubstructMatches(base, uniquify=False, useChirality=False,
                                          maxMatches=1000):
        other = _locant_int(iupac.get(match[site]))
        if other is not None and other < loc:
            loc = other
    return loc


__all__ = ["name_ring_chalcogen_oxide"]
