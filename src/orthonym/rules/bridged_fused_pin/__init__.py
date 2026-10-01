"""Bridged fused ring system PINs on naphthalene and anthracene parents, in any
hydrogenation state, with substituent prefixes, the -ol / -carboxylic acid / -amine
suffixes, the -one suffix on a saturated bridge atom, and stereodescriptors (slice S1 of
the bridged fused PIN builder).

``build(mol)`` returns the complex-ring tuple ``(name, ring_atoms, atom_to_locant, True)``
or None. ``substituents_included`` is True: this package spells every prefix, suffix and
stereodescriptor itself on its own numbering, because the numbering choice depends on them
 (b)-(g), the Blue Book-:3307) and the composer's complex-ring enrichment adds
prefixes only.
"""
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from rdkit import Chem

from ..partial_saturation import _PARTIAL_SAT_PCG_SUFFIX
from . import hydro, numbering, prefixes, selection


@dataclass(frozen=True)
class BridgedParent:
    """The parent hydride of one bridged fused name: '1,2,3,4-tetrahydro-1,4-methano'
    + 'naphthalene'. ``atom_to_locant`` holds every ring-system atom, bridge atoms too."""
    hydro: str
    indicated_hydrogen: str
    bridges: str
    base: str
    atom_to_locant: Dict[int, Any]
    is_pin_form: bool

    def text(self, elide: bool = False) -> str:
        base = self.base[:-1] if elide and self.base.endswith("e") else self.base
        head = [p for p in (self.hydro, self.indicated_hydrogen) if p]
        return "-".join(head + [f"{self.bridges}{base}"])


# The suffixes slice S1 spells -ol, -carboxylic acid,
# -amine), keyed by the principal-group names of rules/seniority.py.
_SUFFIX_KIND = {
    "primary_alcohol": "ol", "secondary_alcohol": "ol", "tertiary_alcohol": "ol",
    "alcohol": "ol", "phenol": "ol",
    "carboxylic_acid": "carboxylic_acid",
    "primary_amine": "amine", "aromatic_amine": "amine",
    "ketone": "one",
}
_SUFFIX_SMARTS = {
    "ol": "[#6:1]-[OX2H1:2]",
    "carboxylic_acid": "[#6:1]-[CX3:2](=[OX1:3])-[OX2H1:4]",
    "amine": "[#6:1]-[NX3H2+0:2]",
    "one": "[#6:1]=[OX1:2]",
}
#: the suffix spellings: the carbocycle speller's three and 'one'
#: (a vowel-initial suffix: 'naphthalen-9-one', but 'naphthalene-9,10-dione')
_SUFFIX_STEM = {**_PARTIAL_SAT_PCG_SUFFIX, "one": ("one", True)}


def principal_suffix(mol, system: Set[int]):
    """('none', None, set, set) when the principal characteristic group has no suffix;
    ('suffix', kind, ring atoms, exocyclic suffix atoms) for the suffixes slice S1 spells;
    None (decline) for any other principal group, or when the group does not sit on the
    ring system, or when one substituent carries more of it than the ring system
    : the parent has the maximum number of principal characteristic groups;
    : a ring is senior to a chain on a tie)."""
    from ...perception.functional_groups import detect_functional_groups
    from ..seniority import get_principal_group
    pg, _ = get_principal_group(mol, detect_functional_groups(mol))
    if pg is None:
        return ("none", None, set(), set())
    kind = _SUFFIX_KIND.get(pg)
    if kind is None:
        return None
    patt = Chem.MolFromSmarts(_SUFFIX_SMARTS[kind])
    on_ring: Set[int] = set()
    exo: Set[int] = set()
    off_ring_hosts: List[int] = []
    for match in mol.GetSubstructMatches(patt):
        host, rest = match[0], match[1:]
        if host in system and not any(r in system for r in rest):
            if host in on_ring:
                return None            # two such groups on one ring atom
            on_ring.add(host)
            exo.update(rest)
        else:
            off_ring_hosts.append(host)
    if not on_ring:
        return None
    # the substituent fragments that carry the same group
    frag_of: Dict[int, int] = {}
    seen: Set[int] = set(system)
    for a in mol.GetAtoms():
        start = a.GetIdx()
        if start in seen:
            continue
        stack, comp = [start], set()
        while stack:
            c = stack.pop()
            if c in seen:
                continue
            seen.add(c)
            comp.add(c)
            stack.extend(nb.GetIdx() for nb in mol.GetAtomWithIdx(c).GetNeighbors())
        for c in comp:
            frag_of[c] = start
    counts: Dict[int, int] = {}
    for h in off_ring_hosts:
        if h in frag_of:
            counts[frag_of[h]] = counts.get(frag_of[h], 0) + 1
    if counts and max(counts.values()) > len(on_ring):
        return None
    return ("suffix", kind, on_ring, exo)


def _stereo_outside(mol, atom_to_locant: Dict[int, Any]) -> bool:
    """True when a stereocentre or a stereogenic double bond lies outside the ring
    system: slice S1 cites stereodescriptors on the ring-system locants only."""
    from rdkit.Chem import FindMolChiralCenters
    for idx, _ in FindMolChiralCenters(mol, includeUnassigned=False, useLegacyImplementation=False):
        if idx not in atom_to_locant:
            return True
    for b in mol.GetBonds():
        if b.GetStereo() != Chem.BondStereo.STEREONONE and not (
                b.GetBeginAtomIdx() in atom_to_locant and b.GetEndAtomIdx() in atom_to_locant):
            return True
    return False


def _stereo_from_rules_4_and_5(mol, atom_to_locant: Dict[int, Any]) -> bool:
    """True when a stereocentre of the ring system has two ligands of the same
    constitution (equal canonical ranks with chirality ignored): Sequence Rules 1-3 do
    not rank them, so its descriptor is decided by Rules 4 and 5,
    the Blue Book "When the use of Sequence Rules 1, 2, and 3 does not permit the
    determination of the ranking of all ligands of a stereogenic unit, Sequence Rule 4 is
    applied"), as for the 9,10 bridgeheads of an 11-substituted 9,10-ethanoanthracene or
    a pseudoasymmetric bridge atom. OPSIN 2.9.0 cannot place such a descriptor, so the
    name can never be read back to the full InChIKey; the builder declines it, so the
    default tier does not ship it as pin_unverified."""
    from rdkit.Chem import FindMolChiralCenters
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False, includeChirality=False))
    for idx, _ in FindMolChiralCenters(mol, includeUnassigned=False, useLegacyImplementation=False):
        if idx not in atom_to_locant:
            continue
        nbr = [ranks[n.GetIdx()] for n in mol.GetAtomWithIdx(idx).GetNeighbors()]
        if len(set(nbr)) < len(nbr):
            return True
    return False


def _prefix_groups(mol, atom_to_locant, exclude: Set[int]) -> List[Tuple[str, List[int]]]:
    """The substituent prefix names in citation order with the ring atoms that carry
    them -- the same naming call ``_partial_sat_substituent_prefix`` makes, used here
    only to rank numberings by (g) (:3307)."""
    from ...assembly.naming_utils import alpha_sort_key
    from ...assembly.substituent_naming import name_substituent_fragment
    ring_set = set(atom_to_locant)
    halo = {"F": "fluoro", "Cl": "chloro", "Br": "bromo", "I": "iodo"}
    groups: Dict[str, List[int]] = {}
    for idx in atom_to_locant:
        for n in mol.GetAtomWithIdx(idx).GetNeighbors():
            j = n.GetIdx()
            if j in ring_set or j in exclude:
                continue
            if n.GetSymbol() in halo and n.GetDegree() == 1:
                groups.setdefault(halo[n.GetSymbol()], []).append(idx)
                continue
            seen, stack, frag = set(ring_set) | exclude, [j], []
            while stack:
                c = stack.pop()
                if c in seen:
                    continue
                seen.add(c)
                frag.append(c)
                stack.extend(nb.GetIdx() for nb in mol.GetAtomWithIdx(c).GetNeighbors())
            name = name_substituent_fragment(mol, frag, j, list(ring_set)) or "?"
            groups.setdefault(name, []).append(idx)
    return sorted(groups.items(), key=lambda kv: alpha_sort_key(kv[0]))


def _assemble(mol, system: Set[int], parent: BridgedParent, suffix) -> Optional[str]:
    """The whole name on this parent: stereodescriptors, detachable prefixes, the
    parent hydride (hydro, indicated hydrogen, bridges, fused parent) and the suffix."""
    from ...assembly.composition_primitives import _join_prefix_to_name
    from ...perception.stereo import assign_stereochemistry
    from ..polycyclics import _offring_substituent_atoms, _partial_sat_substituent_prefix
    from ..stereochemistry import collect_stereodescriptors, format_stereodescriptor_string
    a2l = parent.atom_to_locant
    status, kind, ring_atoms, exo = suffix
    suffix_text, elide = "", False
    if status == "suffix":
        locs = sorted((a2l[a] for a in ring_atoms), key=numbering.loc_key)
        mult = {1: "", 2: "di", 3: "tri", 4: "tetra"}.get(len(locs))
        if mult is None:
            return None
        stem, elides = _SUFFIX_STEM[kind]
        suffix_text = f"-{','.join(str(x) for x in locs)}-{mult}{stem}"
        elide = elides and not mult
    sub_prefix = _partial_sat_substituent_prefix(mol, a2l, exclude_atoms=set(exo))
    if sub_prefix is None and (_offring_substituent_atoms(mol, a2l) - set(exo)):
        return None
    stem = f"{parent.text(elide=elide)}{suffix_text}"
    name = _join_prefix_to_name((sub_prefix or "").rstrip("-"), stem)
    assign_stereochemistry(mol)
    stereo = format_stereodescriptor_string(collect_stereodescriptors(mol, a2l))
    return f"{stereo}{name}" if stereo else name


def _other_ring_systems_are_junior(mol, system: Set[int]) -> bool:
    """ (the Blue Book-:19415): the senior ring system "(a) is a
    heterocycle;... (d) has the greater number of rings". Slice S1 names its ring system
    as the parent only when every other ring system of the molecule is a carbocycle with
    fewer rings; otherwise it declines (the parent choice is the composer's)."""
    own = selection.cycle_rank(mol, system)
    for other in selection.ring_systems(mol):
        if other == system:
            continue
        if any(mol.GetAtomWithIdx(a).GetAtomicNum() != 6 for a in other):
            return False
        if selection.cycle_rank(mol, other) >= own:
            return False
    return True


def _options(mol, system: Set[int]):
    splits = selection.best_splits(mol, system)
    if not splits:
        return None
    options = []
    for sp in splits:
        bps = [prefixes.bridge_prefix(mol, sp, i) for i in range(len(sp.bridges))]
        if any(bp is None for bp in bps):
            return None
        state = hydro.hydro_state(mol, sp)
        if state is None:
            return None
        nums = numbering.numberings(mol, sp, [bp.name for bp in bps])
        if not nums:
            return None
        options.append((sp, bps, state, nums))
    # (i) "have the lowest locants at the location of bridges"
    best_i = min(min(n.attachment_set for n in o[3]) for o in options)
    options = [o for o in options if min(n.attachment_set for n in o[3]) == best_i]
    # (j) (:14395) "have the maximum number of noncumulative double bonds in
    # the parent ring system" (ex.:14399 '1,4-dihydro-1,4-ethanoanthracene (PIN) (not
    # 1,2,3,4-tetrahydro-1,4-ethenoanthracene)')
    best_j = max(o[2].n_double for o in options)
    return [o for o in options if o[2].n_double == best_j]


def build(mol) -> Optional[Tuple[str, Set[int], Dict[int, Any], bool]]:
    """The bridged fused name of ``mol`` (whose largest ring system is a naphthalene or
    anthracene with one or two simple divalent bridges), or None."""
    if mol is None:
        return None
    system = selection.ring_system(mol)
    if not system or not _other_ring_systems_are_junior(mol, system):
        return None
    options = _options(mol, system)
    if not options:
        return None
    # Splits still tied after (a)-(j) with the same parent and the same bridges are one
    # parent hydride read in two ways (the two ethano links of 1,2,3,4-tetrahydro-1,4-
    # ethanonaphthalene): their numberings are pooled and chooses. Splits with
    # different bridges are a tie no rule breaks.
    if len({(o[0].parent, tuple(sorted(bp.name for bp in o[1]))) for o in options}) != 1:
        return None
    suffix = principal_suffix(mol, system)
    if suffix is None:
        return None
    status, kind, pcg_atoms, exo = suffix
    sub_atoms = {a for a in system for nb in mol.GetAtomWithIdx(a).GetNeighbors()
                 if nb.GetIdx() not in system and nb.GetIdx() not in exo}
    scored = []
    for sp, bps, state, nums in options:
        for nb in nums:
            a2l = nb.atom_to_locant
            ih, hydro_atoms = hydro.choose_indicated_hydrogen(state, a2l)
            key = (nb.attachment_set, nb.citation_order,                      # (a), (b)
                   numbering.locant_tuple(a2l[a] for a in ih),                 # (b)
                   numbering.locant_tuple(a2l[a] for a in pcg_atoms),          # (c)
                   numbering.locant_tuple(a2l[a] for a in hydro_atoms),        # (e)
                   numbering.locant_tuple(a2l[a] for a in sub_atoms))          # (f)
            scored.append((key, sp, bps, state, nb, ih, hydro_atoms))
    best = min(s[0] for s in scored)
    tied = [s for s in scored if s[0] == best]
    if len(tied) > 1:
        # (g) (:3307) "lowest locants for the substituent cited first as a prefix"
        def first_cited(s):
            groups = _prefix_groups(mol, s[4].atom_to_locant, set(exo))
            return (numbering.locant_tuple(s[4].atom_to_locant[a] for a in groups[0][1])
                    if groups else ())
        g = min(first_cited(s) for s in tied)
        tied = [s for s in tied if first_cited(s) == g]
    results = {}
    for key, sp, bps, state, nb, ih, hydro_atoms in tied:
        a2l = nb.atom_to_locant
        if _stereo_outside(mol, a2l) or _stereo_from_rules_4_and_5(mol, a2l):
            return None
        if kind == "one" and not pcg_atoms <= {a for chain in sp.bridges for a in chain}:
            return None               # a ring ketone needs added hydrogen (slice S3)
        htext = hydro.hydro_text(state, hydro_atoms, a2l)
        btext = prefixes.bridge_text(nb.bridge_locants)
        if htext is None or btext is None:
            return None
        parent = BridgedParent(htext, hydro.ih_text(ih, a2l), btext, sp.parent, a2l,
                               all(bp.is_pin_form for bp in bps))
        name = _assemble(mol, system, parent, suffix)
        if name is None:
            return None
        results.setdefault(name, (parent, bps))
    if len(results) != 1:
        return None                   # a tie no rule of / (a)-(g) breaks
    name, (parent, bps) = next(iter(results.items()))
    for bp in bps:
        if not bp.is_pin_form:
            from ..bridged_fused import _record_general_bridge
            _record_general_bridge(bp.name)
    return (name, set(system), dict(parent.atom_to_locant), True)
