"""Bridged fused ring system PINs on every fused parent the engine's parent tables
name with a PIN name (``parents.py``), in any hydrogenation state, with substituent
prefixes, the -ol / -carboxylic acid / -amine / -carbonitrile / -carbaldehyde /
-carboxamide suffixes, the -one suffix on a saturated bridge atom, stereodescriptors, and
as a '-yl' substituent prefix (slices S1 and S2 of the bridged fused PIN builder); a reading
that is a cyclic phane by is declined (slice S2b). Slice S3 (``hydro.accommodate``):
the -one suffix on a ring atom of the fused parent, by indicated hydrogen that accommodates
it or by 'added indicated hydrogen' ('1,2,3,7,8,8a-hexahydro-4H-3a,7-
methanoazulene-4,9-dione (PIN)', the Blue Book); cyclic anhydrides, esters and amides
of the ring system as pseudoketones, '3a,4,7,7a-tetrahydro-1H-4,7-methanoisoindole-
1,3(2H)-dione'); and every other suffix or free valence on an atom with no hydrogen in the
mancude parent ('3,4-dihydro-1,4-methanonaphthalen-1(2H)-ol').

``build(mol)`` returns the complex-ring tuple ``(name, ring_atoms, atom_to_locant, True)``
or None. ``substituents_included`` is True: this package spells every prefix, suffix and
stereodescriptor itself on its own numbering, because the numbering choice depends on them
 (b)-(g), the Blue Book-:3307) and the composer's complex-ring enrichment adds
prefixes only.
"""
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from rdkit import Chem

from ...data.multipliers import multiplier_table
from ..partial_saturation import _PARTIAL_SAT_PCG_SUFFIX
from ..stereochemistry import format_stereodescriptor_string
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
        """ / (:16872,:16880) and (:14642): hydro prefixes,
        indicated hydrogen, bridge prefixes, fused parent. (a) (:6938): a hyphen
        separates the bridge prefix from a parent name that begins with a locant
        ('2H,7H-4a,7-ethano-1-benzopyran',:14648)."""
        base = self.base[:-1] if elide and self.base.endswith("e") else self.base
        head = [p for p in (self.hydro, self.indicated_hydrogen) if p]
        joint = "-" if self.bridges and base[:1].isdigit() else ""
        stem = f"{self.bridges}{joint}{base}"
        if not self.bridges and not self.indicated_hydrogen and self.hydro:
            # (the Blue Book): hydro prefixes stand "immediately at the front of
            # the name of the parent hydride"; a stem that begins with a letter follows them
            # without a hyphen ('decahydronaphthalene'), and so does one that begins with an enclosing mark
            # ('octahydro[1,4]dioxocino[2,3-c][1,6]dioxecine-2,5,9,12-tetrone (PIN)',:32191; the book prints
            # 'hydro[' and never 'hydro-['); one that begins with a locant needs it, and so does the italic
            # designator of 's-indacene' ("to separate italic letters from Roman letters", (d),:6960):
            # '1,2,3,5,6,7-hexahydro-s-indacene'
            from ...assembly.naming_utils import begins_with_italic_designator
            return self.hydro + ("-" if stem[:1].isdigit() or begins_with_italic_designator(stem)
                                 else "") + stem
        return "-".join(head + [stem])


#: the most identical suffixes one name cites: the multiplying prefixes of (Table 1.4),
#: as in 'benzenehexol (PIN, ' (the Blue Book); a bridged fused polyol cites five
#: and more ('-pentol'), the tetrodotoxin skeleton among them
MAX_SUFFIX_COUNT = 12

# The suffixes the package spells -ol, -carboxylic acid,
# -amine,...), keyed by the principal-group names of rules/seniority.py.
_SUFFIX_KIND = {
    "primary_alcohol": "ol", "secondary_alcohol": "ol", "tertiary_alcohol": "ol",
    "alcohol": "ol", "phenol": "ol",
    "carboxylic_acid": "carboxylic_acid",
    "primary_amine": "amine", "aromatic_amine": "amine",
    "ketone": "one",
    # (the Blue Book) "The suffix 'carbonitrile' is always used to name
    # nitriles having the –CN group attached to a ring or ring system";
    # (:34941) 'carbaldehyde' for a –CHO group on a ring; (:32669) "The
    # suffix 'carboxamide' is always used to name amides with the –CO-NH2 group attached
    # to a ring, ring system"
    "nitrile": "carbonitrile",
    "aldehyde": "carbaldehyde",
    "primary_amide": "carboxamide",
}
_SUFFIX_SMARTS = {
    "ol": "[#6:1]-[OX2H1:2]",
    "carboxylic_acid": "[#6:1]-[CX3:2](=[OX1:3])-[OX2H1:4]",
    "amine": "[#6:1]-[NX3H2+0:2]",
    "one": "[#6:1]=[OX1:2]",
    "carbonitrile": "[#6:1]-[CX2:2]#[NX1:3]",
    "carbaldehyde": "[#6:1]-[CX3H1:2]=[OX1:3]",
    "carboxamide": "[#6:1]-[CX3:2](=[OX1:3])-[NX3H2+0:4]",
}
#: the suffix spellings: the carbocycle speller's three, 'one' and the free
#: valence 'yl' (vowel-initial suffixes: 'naphthalen-9-one', but 'naphthalene-9,10-dione'),
#: and the 'carbo-' suffixes, which keep the final 'e' ('naphthalene-2-carbonitrile')
_SUFFIX_STEM = {**_PARTIAL_SAT_PCG_SUFFIX, "one": ("one", True), "yl": ("yl", True),
                "carbonitrile": ("carbonitrile", False), "carbaldehyde": ("carbaldehyde", False),
                "carboxamide": ("carboxamide", False)}


#: (the Blue Book) "Cyclic anhydrides, esters and amides are named as
#: pseudoketones; the resulting names are preferred IUPAC names" (cyclic imides:,
#: 'hexahydro-1H-isoindole-1,3(2H)-dione (PIN)':33849); (:29628) "There is no
#: seniority order difference between ketones and pseudoketones"; (:32112) "A
#: lactone, as a pseudoketone, ranks lower in the seniority of classes than an acid or an
#: ester, but higher than an alcohol, amine, or imine."
_PSEUDOKETONE_CLASSES = ("anhydride", "ester", "secondary_amide", "tertiary_amide", "imide", "hydrazide")


def _cyclic_in(mol, system: Set[int], match) -> bool:
    """True when every carbonyl carbon of the group and every heteroatom bonded to one of
    them in the match lies in the ring system (the -CO-X- unit is part of the ring), and every
    nitrogen of the match does (the N-N of a ring hydrazide: (a), the Blue Book,
    "cyclic compounds in which a carbonyl group in a ring is bonded to one or two skeletal heteroatoms")."""
    atoms = set(match)
    carbonyl = {a for a in atoms if mol.GetAtomWithIdx(a).GetAtomicNum() == 6 and any(
        b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(mol.GetAtomWithIdx(a)).GetAtomicNum() == 8
        and b.GetOtherAtomIdx(a) in atoms for b in mol.GetAtomWithIdx(a).GetBonds())}
    if not carbonyl:
        return False
    hetero = {nb.GetIdx() for c in carbonyl for nb in mol.GetAtomWithIdx(c).GetNeighbors()
              if nb.GetIdx() in atoms and nb.GetAtomicNum() != 6
              and mol.GetBondBetweenAtoms(c, nb.GetIdx()).GetBondType() == Chem.BondType.SINGLE}
    return bool(hetero) and (carbonyl | hetero) <= set(system) and all(
        mol.GetAtomWithIdx(a).GetAtomicNum() != 7 or a in system for a in atoms)


def _pseudoketone_principal(mol, system: Set[int], groups) -> Optional[str]:
    """'ketone' when the principal class is a cyclic anhydride, ester or amide of this ring
    system (a pseudoketone) and nothing senior to a ketone is left; None otherwise."""
    from ..seniority import SENIORITY_ORDER, get_principal_group
    rest = dict(groups)
    for cls in _PSEUDOKETONE_CLASSES:
        for match in rest.pop(cls, None) or ():
            if not _cyclic_in(mol, system, match):
                return None
    other, _ = get_principal_group(mol, rest)
    if other is not None and other != "ketone" and (
            other not in SENIORITY_ORDER
            or SENIORITY_ORDER.index(other) < SENIORITY_ORDER.index("ketone")):
        return None
    return "ketone"


#: functional-group classes that are parent hydrides of their own (urea, thiourea, guanidine, carbamic
#: acid and its esters, cyanamide, carbamimidic esters) and that ``rules/seniority.SENIORITY_ORDER``
#: does not rank, so ``get_principal_group`` answers None for a molecule that carries no ranked group
#: besides them. (the Blue Book) "The preferred IUPAC name for the 'amidine'
#: related to carbonic acid... is the retained name 'guanidine'"; (:34266) "In the
#: presence of a characteristic group having seniority over guanidine (see item 11 in, the
#: following prefixes are used": 'carbamimidoylamino' (preferred), never the 'guanidinyl' the
#: substituent namer spells. Whether the ring system or such a parent is the parent of the molecule
#: is the engine's choice, not this builder's.
_UNRANKED_PARENT_CLASSES = ("urea", "thiourea", "guanidine", "carbamate", "carbamimidate", "cyanamide")

#: the prefix the fragment namer spells for a guanidine group on the ring system, which the Blue
#: Book does not print:34266 gives 'carbamimidoylamino', the spelling the enumerator's
#: namer gives an acyclic N-attached group, ``polycyclics.exocyclic_substituent_name``); a name that
#: carries it is valid but never a PIN, so it is recorded as a non-PIN fragment (name-scoped, demote
#: not delete)
_GUANIDINE_PREFIX_SPELLED = "guanidinyl"


def _unranked_groups_outside(mol, groups, system: Set[int]) -> Set[str]:
    """The classes of ``_UNRANKED_PARENT_CLASSES`` with a connecting atom (one with two or more
    heavy neighbours) outside the ring system. A ring-internal one, a cyclic carbamate or urea of
    the ring system, has only terminal atoms outside it (its =O, =S, -NH2): a pseudoketone
    (``_pseudoketone_principal``), not a parent of its own."""
    return {cls for cls in _UNRANKED_PARENT_CLASSES for match in groups.get(cls) or ()
            if any(mol.GetAtomWithIdx(a).GetDegree() > 1 for a in match if a not in system)}


#: suffixes that may stand twice on one ring atom: a geminal diol ('...-10,10-diol') names a ring atom
#: with two hydroxy groups, the hydrate of a ketone. "Citation of locants" (the Blue Book)
#: "if any locants are essential... then all locants must be cited": the locant is cited once per
#: group, as the book's 'cyclohexane-1,1-dicarboximidamide (PIN)' (:34236) shows for a suffix repeated
#: on one ring atom. The atom then needs two hydrogen atoms in the parent, as a ketone does (``_lacking``).
_GEMINAL_KINDS = ("ol",)


def _each(atoms):
    """The suffix atoms once per group (``principal_suffix`` returns a Counter of the ring
    atoms: a geminal diol atom counts twice); a plain set of atoms once each."""
    return atoms.elements() if isinstance(atoms, Counter) else atoms


def principal_suffix(mol, system: Set[int]):
    """('none', None, set, set) when the principal characteristic group has no suffix;
    ('suffix', kind, ring atoms, exocyclic suffix atoms) for the suffixes the package spells;
    None (decline) for any other principal group, or when the group does not sit on the
    ring system, or when one substituent carries more of it than the ring system
    : the parent has the maximum number of principal characteristic groups;
    : a ring is senior to a chain on a tie). A cyclic anhydride, ester, amide or
    imide of the ring system is a pseudoketone (``_pseudoketone_principal``): suffix 'one'. An
    amidine or guanidine whose carbon is a skeletal atom of the ring system is the ring itself,
    not a group, and its one exocyclic -NH2, if it has one, is an amine
    (``seniority.ring_skeleton_units_resolved``); one with any other exocyclic part declines
    unless a senior group carries the name."""
    from ...perception.functional_groups import detect_functional_groups
    from ..seniority import get_principal_group, ring_skeleton_units_resolved
    # an amidine or guanidine whose carbon is a skeletal atom of the ring system is the ring itself,
    # not a '-imidamide' or 'guanidine' group:34173,:34200;:34242): the
    # DBU / DBN / TBD skeleton; its one exocyclic -NH2 is an amine:26168)
    groups = ring_skeleton_units_resolved(mol, detect_functional_groups(mol), system)
    pg, _ = get_principal_group(mol, groups)
    outside = _unranked_groups_outside(mol, groups, system)
    if outside:
        # no ranked group: the urea, guanidine or carbamate is the parent, not the ring system.
        # A senior group (an acid, an ester,...) makes the ring the parent and the group a prefix
        #:34266). The substituent namer spells a cyanamide as 'carbamoyl' (another
        # molecule) and a guanidine as 'guanidinyl' (not the book's form, so never a PIN); the urea,
        # thiourea and carbamate it does not spell at all.
        if pg is None or "cyanamide" in outside:
            return None
        if "guanidine" in outside:
            from ...metrics.provenance import record_non_pin_fragment
            record_non_pin_fragment(_GUANIDINE_PREFIX_SPELLED)
    if pg is None and any(m[1] in system for m in groups.get("guanidine") or ()):
        # a guanidine unit of the ring that ``ring_skeleton_units_resolved`` kept has an exocyclic
        # nitrogen that is not a plain -NH2: with no senior group it is the 'imine' (or the
        # substituted amine) of the ring, expressed by a suffix this builder does not spell
        #:26510 "All imines are named substitutively using the suffix 'imine'"); the
        # prefix 'imino' is for "characteristic groups having seniority over imines" (:26540)
        return None
    if pg is None:
        return ("none", None, set(), set())
    if pg in _PSEUDOKETONE_CLASSES:
        pg = _pseudoketone_principal(mol, system, groups)
        if pg is None:
            return None
    kind = _SUFFIX_KIND.get(pg)
    if kind is None:
        return None
    patt = Chem.MolFromSmarts(_SUFFIX_SMARTS[kind])
    on_ring: Counter = Counter()
    exo: Set[int] = set()
    off_ring_hosts: List[int] = []
    for match in mol.GetSubstructMatches(patt):
        host, rest = match[0], match[1:]
        if host in system and not any(r in system for r in rest):
            if host in on_ring and kind not in _GEMINAL_KINDS:
                return None            # two such groups on one ring atom
            on_ring[host] += 1
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
    if counts and max(counts.values()) > sum(on_ring.values()):
        return None
    return ("suffix", kind, on_ring, exo)


def _stereo_outside(mol, atom_to_locant: Dict[int, Any]) -> bool:
    """True when a stereocentre or a stereogenic double bond lies outside the ring system. The
    package cites descriptors on the ring-system locants only; those of a substituent are cited
    by the substituent namer at the front of its prefix (``_outside_stereo_spelled``)."""
    from rdkit.Chem import FindMolChiralCenters
    for idx, _ in FindMolChiralCenters(mol, includeUnassigned=False, useLegacyImplementation=False):
        if idx not in atom_to_locant:
            return True
    for b in mol.GetBonds():
        if b.GetStereo() != Chem.BondStereo.STEREONONE and not (
                b.GetBeginAtomIdx() in atom_to_locant and b.GetEndAtomIdx() in atom_to_locant):
            return True
    return False


def _outside_stereo_spelled(mol, name: str) -> bool:
    """True when OPSIN reads ``name`` back to ``mol`` with every stereogenic unit outside the ring
    system: NAMING OF STEREOISOMERS (the Blue Book), "When they relate to substituent
    groups, they are cited at the front of the corresponding prefix" (:44643,
    '[(1R)-1-chloropropyl]benzene (PIN)'). The prefix speller cites a substituent's descriptors
    when it can; a prefix that leaves one out reads back to another stereoisomer, so the name is
    declined here and the molecule keeps the older producers' name (``fused_pin_offer._reads_back``
    gives False without a JVM, so a name is never shipped unproved)."""
    from ..fused_pin_offer import _reads_back
    return _reads_back(mol, name)


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
    from ..polycyclics import exocyclic_substituent_name, ring_ylidene_prefix
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
            ylidene = ring_ylidene_prefix(mol, j, idx)
            if ylidene is not None:
                groups.setdefault(ylidene, []).append(idx)
                continue
            seen, stack, frag = set(ring_set) | exclude, [j], []
            while stack:
                c = stack.pop()
                if c in seen:
                    continue
                seen.add(c)
                frag.append(c)
                stack.extend(nb.GetIdx() for nb in mol.GetAtomWithIdx(c).GetNeighbors())
            name = exocyclic_substituent_name(mol, frag, j, list(ring_set)) or "?"
            groups.setdefault(name, []).append(idx)
    return sorted(groups.items(), key=lambda kv: alpha_sort_key(kv[0]))


def _assemble(mol, system: Set[int], parent: BridgedParent, suffix, added=frozenset()):
    """(stereodescriptors, name without them) on this parent: detachable prefixes, the
    parent hydride (hydro, indicated hydrogen, bridges, fused parent) and the suffix.
     (the Blue Book): 'added indicated hydrogen' "is enclosed in parentheses
    and inserted into the name immediately following the locant(s) for... principal
    characteristic groups" ('pyrimidine-4,6(1H,5H)-dione (PIN)':24709)."""
    from ...assembly.composition_primitives import _join_prefix_to_name
    from ...perception.stereo import assign_stereochemistry
    from ..polycyclics import _offring_substituent_atoms, _partial_sat_substituent_prefix
    from ..stereochemistry import collect_stereodescriptors
    a2l = parent.atom_to_locant
    status, kind, ring_atoms, exo = suffix
    suffix_text, elide = "", False
    if status == "suffix":
        locs = sorted((a2l[a] for a in _each(ring_atoms)), key=numbering.loc_key)
        mult = multiplier_table(MAX_SUFFIX_COUNT, one="").get(len(locs))
        if mult is None:
            return None
        stem, elides = _SUFFIX_STEM[kind]
        # (c) (:7619) "the terminal letter 'a' in the names of numerical
        # multiplicative prefixes when followed by a suffix beginning with 'a' or 'o'"
        # ('tetrahydro-4,8-ethanopyrano[4,3-c]pyran-1,3,5,7-tetrone (PIN)',:32535); also
        # 'tetrol' (:7625 'benzenehexol (not benzenehexaol)') and 'tetramine' (:26348)
        if mult.endswith("a") and stem[:1] in ("a", "o"):
            mult = mult[:-1]
        extra = f"({hydro.ih_text(added, a2l)})" if added else ""
        suffix_text = f"-{','.join(str(x) for x in locs)}{extra}-{mult}{stem}"
        elide = elides and not mult
    assign_stereochemistry(mol)
    descriptors = collect_stereodescriptors(mol, a2l)
    omit = not descriptors and _single_prefix_locant_is_omitted(
        mol, system, parent, bool(suffix_text), set(exo))
    sub_prefix = _partial_sat_substituent_prefix(
        mol, a2l, exclude_atoms=set(exo), omit_locants=omit, ylidene_prefixes=True)
    if sub_prefix is None and (_offring_substituent_atoms(mol, a2l) - set(exo)):
        return None
    stem = f"{parent.text(elide=elide)}{suffix_text}"
    name = _join_prefix_to_name((sub_prefix or "").rstrip("-"), stem)
    return descriptors, name


def _single_prefix_locant_is_omitted(mol, system: Set[int], parent: BridgedParent,
                                     has_suffix: bool, exo: Set[int]) -> bool:
    """ (the Blue Book, 'chlorocoronene (PIN)':2947): the one locant of the one
    prefix of a parent hydride whose substitutable positions are all alike is not cited. The
    decision is ``assembly.locant_omission.l3_locant_omitted_for_parent_atoms``, which measures
    the symmetry classes of the parent's atoms; it is asked only for a plain mancude parent (no
    hydro prefix, indicated hydrogen or bridge), no suffix, exactly one substituent, and a
    scope that is the whole molecule named,:2869, denies unless all that holds)."""
    if has_suffix or parent.hydro or parent.indicated_hydrogen or parent.bridges:
        return False
    attached = [(a, nb.GetIdx()) for a in system for nb in mol.GetAtomWithIdx(a).GetNeighbors()
                if nb.GetIdx() not in system and nb.GetIdx() not in exo]
    if len(attached) != 1:
        return False
    from ...assembly.handlers._handler_shared import locant_scope_is_a_name_component
    if locant_scope_is_a_name_component():
        return False
    from ...assembly.locant_omission import l3_locant_omitted_for_parent_atoms
    return l3_locant_omitted_for_parent_atoms(
        mol, system, prefix_locants=[parent.atom_to_locant[attached[0][0]]], suffix_locants=[],
        parent_cites_locants=False, stereo_text="")


def _other_ring_systems_are_junior(mol, system: Set[int]) -> bool:
    """ (the Blue Book-:19418, ``selection.p44_2_1_key``): the bridged fused
    system is the parent only when it is senior to every other ring system of the molecule
    by criteria (a)-(g) (heterocycle, nitrogen, first heteroatom, rings, skeletal atoms,
    heteroatoms, heteroatoms in order). A tie, or a senior other ring system, declines (the
    parent choice is then the composer's)."""
    own = selection.p44_2_1_key(mol, system)
    for other in selection.ring_systems(mol):
        if other != system and selection.p44_2_1_key(mol, other) <= own:
            return False
    return True


def _lacking(state, kind, atoms) -> FrozenSet[int]:
    """The suffix atoms (or the free-valence atom) without enough hydrogen in the mancude
    parent (``hydro.lacks_hydrogen``): a ketone needs two, the Blue Book);
    every other suffix the package spells and a free valence need one, which a bridgehead, a
    fusion atom or a ring =N- does not have:24719 '1,3,4,5-tetrahydronaphthalene-
    4a(2H)-carboxylic acid (PIN)';:17340 '1,3,4,5-tetrahydronaphthalen-4a(2H)-yl
    (preferred prefix)';:7467 '1-(3,4-dihydroquinolin-1(2H)-yl)ethan-1-one')."""
    if kind is None:
        return frozenset()

    def needed(a):
        # a ketone needs two; any other suffix one per group, two for a geminal diol (``_GEMINAL_KINDS``)
        return 2 if kind == "one" else atoms[a] if isinstance(atoms, Counter) else 1
    return frozenset(a for a in atoms if hydro.lacks_hydrogen(state, a, needed(a)))


def _options(mol, system: Set[int]):
    splits = selection.best_splits(mol, system)
    if not splits:
        return None
    options = []
    for sp in splits:
        bps = [prefixes.bridge_prefix(mol, sp, i) for i in range(len(sp.bridges))]
        if any(bp is None for bp in bps):
            return None
        bps = prefixes.multiplied_in_general_form(bps)
        state = hydro.hydro_state(mol, sp)
        if state is None:
            return None
        nums = numbering.numberings(mol, sp, [bp.name for bp in bps], [bp.first for bp in bps])
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
    options = [o for o in options if o[2].n_double == best_j]
    # (:23843): a cyclic phane is senior to a bridged fused system; a reading
    # that cuts a mancude ring of a chain-linked compound into a bridge is the book's phane
    # case (:23895, ``selection.phane_reading``), so the builder declines it
    if any(selection.phane_reading(mol, o[0]) for o in options):
        return None
    return options


def has_bridged_fused_reading(mol) -> bool:
    """True when the largest ring system of ``mol`` cannot be named as a fused ring system
    and (the Blue Book-:14395) selects a reading of it as a bridged
    fused system whose fused parent the parent tables name (``selection.best_splits``),
    whether or not ``build`` completes the name (an ester, a bridge prefix the builder does
    not certify, or a suffix whose indicated or added hydrogen fails the tautomer check or
    exceeds ``hydro.MAX_INDICATED_H`` / ``hydro.MAX_ADDED_H`` can still stop it).
     (:14241): "When a
    polycyclic ring system cannot be named completely as a fused ring system, possible
    ways for naming it as a bridged fused system are considered"; (:23843):
    "fused ring systems > bridged fused systems > non-fused bridged systems". For such a
    molecule the PIN is a name, never a trivial or name,:50943).

    The natural-product route asks this for every molecule it names, so the selection runs
    under the builder's error guard (``bridged_fused.builder_declining_on_error``, as
    ``build`` does): an internal error answers False, logged with its traceback, and the
    route returns its name as it did before this check existed."""
    if mol is None:
        return False
    from ..bridged_fused import builder_declining_on_error
    return bool(builder_declining_on_error(_reading_exists, mol))


def _reading_exists(mol) -> bool:
    """``has_bridged_fused_reading`` without the error guard."""
    system = selection.ring_system(mol)
    return bool(system) and bool(selection.best_splits(mol, system))


def has_unchecked_parent_numbering(mol) -> bool:
    """True when a bridged fused reading of ``mol`` names a fused parent that OPSIN numbers
    (``parents`` source ending '+opsin', not a table) and whose skeleton has a tetravalent fusion
    atom (``fusion_components.tetravalent_fusion_atoms``, the 'naphtho[1,8a-b]azirine' topology of
    , the Blue Book). Nothing checks such a numbering: the engine's
    ``compute_fused_numbering`` has no such atom in its grid, so ``parents._parent_for_key`` has no
    second source to agree with OPSIN's, and ``fusion_components.periphery_numbering_ok`` only checks
    that the locants run round the periphery,:12501). A parent of the tables (the
    morphinan '[1]benzofuro[3,2-e]isoquinoline', source 'derived_table') carries its own numbering.
    Under the builder's error guard, as ``has_bridged_fused_reading``: an internal error answers False."""
    if mol is None:
        return False
    from ..bridged_fused import builder_declining_on_error
    return bool(builder_declining_on_error(_numbering_unchecked, mol))


def _numbering_unchecked(mol) -> bool:
    """``has_unchecked_parent_numbering`` without the error guard."""
    from . import fusion_components, parents
    system = selection.ring_system(mol)
    for split in (selection.best_splits(mol, system) if system else None) or ():
        residual = set(split.residual)
        skeleton = Chem.MolFromSmiles(Chem.MolToSmiles(parents._skeleton(mol, residual)[0]))
        rings = fusion_components.ortho_rings(skeleton)
        if rings is None or not fusion_components.tetravalent_fusion_atoms(skeleton, rings):
            continue
        parent = parents.fused_parent(mol, residual)
        if parent is not None and parent.source.endswith("+opsin"):
            return True
    return False


def build(mol) -> Optional[Tuple[str, Set[int], Dict[int, Any], bool]]:
    """The bridged fused name of ``mol`` (whose largest ring system is a nameable fused
    parent with one to three simple divalent bridges), or None."""
    if mol is None:
        return None
    system = selection.ring_system(mol)
    if not system or not _other_ring_systems_are_junior(mol, system):
        return None
    return _name_options(mol, system, _options(mol, system))


def _fused_options(mol, system: Set[int]):
    """The one option of a fused ring system (carbocyclic or heterocyclic) that is its own parent
    (``selection.fused_split``): the same ``(split, bridge prefixes, hydrogenation state,
    numberings)`` tuple ``_options`` returns, with no bridge."""
    split = selection.fused_split(mol, system)
    if split is None:
        return None
    state = hydro.hydro_state(mol, split)
    if state is None:
        return None
    nums = numbering.numberings(mol, split, [], [])
    return [(split, [], state, nums)] if nums else None


def _lone_pair_centre_in(mol, system: Set[int]) -> bool:
    """True when a ring atom of ``system`` is a lone-pair stereocentre (P(III), As(III), S(IV),
    Se(IV): three ligands and the lone pair). The descriptor the labeller gives such a centre on the
    working molecule can be the other configuration than the input's (RDKit's reading of the
    parity, ``namer._exit_lone_pair_check``), and the engine's exit check then withdraws the whole
    name, leaving no fallback: the builder does not name it, and the older ring producers, which
    carry the lone-pair handling, keep the molecule."""
    from ...namer import _lone_pair_compared_atom
    return any(mol.GetAtomWithIdx(a).GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
               and _lone_pair_compared_atom(mol.GetAtomWithIdx(a))
               and mol.GetAtomWithIdx(a).GetTotalDegree() == 3 for a in system)


def _isotope_labels_pending(mol, system: Set[int]) -> bool:
    """True when the molecule is named without its isotope labels (``rules/isotopes.
    decorate_isotopic_name`` names the stripped skeleton, then splices the descriptor in) and the
    ring system has a heteroatom. (the Blue Book) puts the descriptor in front of
    the part it modifies, and OPSIN 2.9.0 cannot read it on the heteroatom of an indicated-
    hydrogen fusion name ('(2-2H)octahydro-1H-isoindole'), so no decoration of a fusion name
    round-trips; the vonBaeyer path, whose '(8-2H)-8-azabicyclo[4.3.0]nonane' does, stays the name
    of such an input as before this builder named hetero systems."""
    from ...assembly.locant_omission import scope_has_isotopic_modification
    return scope_has_isotopic_modification() and any(
        mol.GetAtomWithIdx(a).GetAtomicNum() != 6 for a in system)


def build_fused(mol) -> Optional[Tuple[str, Set[int], Dict[int, Any], bool]]:
    """The fusion name of ``mol`` whose largest ring system is an ortho- or ortho-
    and peri-fused system with at least two rings of five or more members, in any hydrogenation
    state, with substituent prefixes and the suffixes ``principal_suffix`` spells; the same
    tuple as ``build``. (the Blue Book) gives such a system the fusion name as
    its preferred IUPAC name ("When fusion names are not allowed, unsaturated von Baeyer ring
    system names are preferred IUPAC names"); the hydrogenation is spelled as in ``build``
     :16872,:16880,:24766) and the numbering is chosen by
     (b)-(g) (:3246-:3307). None when the system is not that class or a step declines;
    the caller keeps its older path."""
    if mol is None:
        return None
    system = selection.ring_system(mol)
    if not system or not _other_ring_systems_are_junior(mol, system):
        return None
    if _isotope_labels_pending(mol, system):
        return None
    if _lone_pair_centre_in(mol, system):
        return None
    return _name_options(mol, system, _fused_options(mol, system))


def _name_options(mol, system: Set[int], options):
    """The name of ``mol`` from the options of its ring system (``_options`` for a bridged
    fused system, ``_fused_options`` for a fused one), or None."""
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
    # one entry per prefix bond: (f) (:3301) counts every prefix, so a gem pair cites its locant twice
    sub_atoms = [a for a in system for nb in mol.GetAtomWithIdx(a).GetNeighbors()
                 if nb.GetIdx() not in system and nb.GetIdx() not in exo]
    scored = []
    for sp, bps, state, nums in options:
        need = _lacking(state, kind, pcg_atoms)
        for nb in nums:
            a2l = nb.atom_to_locant
            got = hydro.accommodate(state, a2l, need)
            if got is None:
                continue
            ih, added, hydro_atoms = got
            key = (nb.attachment_set, nb.citation_order,                      # (a), (b)
                   numbering.locant_tuple(a2l[a] for a in ih),                 # (b)
                   numbering.locant_tuple(a2l[a] for a in _each(pcg_atoms)),   # (c)
                   numbering.locant_tuple(a2l[a] for a in added),              # (d)
                   numbering.locant_tuple(a2l[a] for a in hydro_atoms),        # (e)
                   numbering.locant_tuple(a2l[a] for a in sub_atoms))          # (f)
            scored.append((key, sp, bps, state, nb, ih, hydro_atoms, added, need))
    if not scored:
        return None
    picked = _choose(mol, system, scored, suffix, exo, kind, pcg_atoms)
    if picked is None:
        return None
    name, parent, bps = picked
    if any(_tautomer_sensitive(mol, set(s[5]) | set(s[7]), s[6]) for s in scored) \
            and not _same_tautomer(mol, name):
        return None
    if _stereo_outside(mol, parent.atom_to_locant) and not _outside_stereo_spelled(mol, name):
        return None
    for bp in bps:
        if not bp.is_pin_form:
            from ..bridged_fused import _record_general_bridge
            _record_general_bridge(bp.name)
    return (name, set(system), dict(parent.atom_to_locant), True)


#: (j) (:3346): "the lower locant is assigned to CIP stereodescriptors Z, R, M, and
#: r (pseudoasymmetry) that are preferred to E, S, P, and s, respectively"
_CIP_PREFERRED = {"Z": (0, 0), "E": (0, 1), "R": (1, 0), "S": (1, 1), "M": (2, 0), "P": (2, 1),
                  "r": (3, 0), "s": (3, 1)}


def _cip_key(descriptors):
    """The (j) comparison key of one numbering's descriptors (sorted by locant), or
    None for a descriptor (h)-(j) do not rank."""
    out = []
    for loc, code in descriptors:
        if code not in _CIP_PREFERRED:
            return None
        out.append((numbering.loc_key(loc), _CIP_PREFERRED[code]))
    return tuple(out)


def _isotope_then_cip(mol, tied, exo):
    """The numberings of ``tied`` that stand after (i) (:3336), the lowest locants to the isotopically
    modified atoms, and (j) (:3346), the CIP stereodescriptors of the parent and those cited inside its
    prefixes, together, at their locants (``### **** NUMBERING``,:3219). "Isotopic modifications have
    seniority for low locants over stereodescriptors described in (j) below" (:3346)."""
    from ...assembly.locant_omission import isotope_locant_key
    from ...assembly.naming_utils import numbering_cip_key
    from ...perception.stereo import assign_stereochemistry
    from ..stereochemistry import collect_stereodescriptors

    def i_key(s):
        return isotope_locant_key(mol, {a: numbering.loc_key(loc) for a, loc in s[4].atom_to_locant.items()})

    best = min(i_key(s) for s in tied)
    tied = [s for s in tied if i_key(s) == best]
    if len(tied) == 1:
        return tied

    def j_key(s):
        a2l = s[4].atom_to_locant
        assign_stereochemistry(mol)
        parent = [(numbering.loc_key(loc), code) for loc, code in collect_stereodescriptors(mol, a2l)]
        return numbering_cip_key(parent, [(numbering.loc_key(a2l[a]), name)
                                          for name, atoms in _prefix_groups(mol, a2l, set(exo)) for a in atoms])

    best = min(j_key(s) for s in tied)
    return [s for s in tied if j_key(s) == best]


def _choose(mol, system, scored, suffix, exo, kind, pcg_atoms):
    """(name, parent, bridge prefixes) of the numbering and (b)-(g), (j)
    choose, or None."""
    best = min(s[0] for s in scored)
    tied = [s for s in scored if s[0] == best]
    if len(tied) > 1:
        # (g) (:3307) "lowest locants for the substituent cited first as a prefix". leaves the
        # stereodescriptors out (:3446), so the prefixes that tie with the first-cited one on it (its R and S
        # twins) are that one prefix here; the choice between them is (j)'s, after (i) (:3336,:3346).
        from ...assembly.naming_utils import alpha_sort_key

        def first_cited(s):
            groups = _prefix_groups(mol, s[4].atom_to_locant, set(exo))
            if not groups:
                return ()
            k0 = alpha_sort_key(groups[0][0])
            return numbering.locant_tuple(s[4].atom_to_locant[a] for name, atoms in groups
                                          if alpha_sort_key(name) == k0 for a in atoms)
        g = min(first_cited(s) for s in tied)
        tied = [s for s in tied if first_cited(s) == g]
        tied = _isotope_then_cip(mol, tied, exo)
    results = {}
    for key, sp, bps, state, nb, ih, hydro_atoms, added, need in tied:
        a2l = nb.atom_to_locant
        if _stereo_from_rules_4_and_5(mol, a2l):
            return None
        htext = hydro.hydro_text(state, hydro_atoms, a2l, ih, need)
        btext = prefixes.bridge_text(nb.bridge_locants)
        if htext is None or btext is None:
            return None
        parent = BridgedParent(htext, hydro.ih_text(ih, a2l), btext, sp.parent, a2l,
                               all(bp.is_pin_form for bp in bps))
        got = _assemble(mol, system, parent, suffix, added)
        if got is None:
            return None
        descriptors, plain = got
        name = f"{format_stereodescriptor_string(descriptors)}{plain}"
        results.setdefault(name, (descriptors, plain, parent, bps))
    if len(results) == 1:
        name, (_, _, parent, bps) = next(iter(results.items()))
        return name, parent, bps
    # (h) nonstandard valence and (i) isotopes come before (j); the package ranks
    # neither, so a tie with either present declines.
    if len({v[1] for v in results.values()}) != 1 or any(
            mol.GetAtomWithIdx(a).GetIsotope() for a in system):
        return None                   # a tie no rule of / (a)-(g) breaks
    keyed = [(_cip_key(v[0]), name, v) for name, v in results.items()]
    if any(k is None for k, _, _ in keyed) or len({tuple(l for l, _ in k) for k, _, _ in keyed}) != 1:
        return None
    best = min(k for k, _, _ in keyed)
    top = [(name, v) for k, name, v in keyed if k == best]
    if len(top) != 1:
        return None
    name, (_, _, parent, bps) = top[0]
    return name, parent, bps


def _tautomer_sensitive(mol, ih, hydro_atoms) -> bool:
    """True when an indicated-hydrogen or hydro atom is a ring nitrogen or is bonded to one
    (spec section 8): the standard InChIKey does not tell the 1H- from the 2H- tautomer of
    an N-H / N= ring pair."""
    for a in set(ih) | set(hydro_atoms):
        atom = mol.GetAtomWithIdx(a)
        if atom.GetAtomicNum() == 7:
            return True
        if any(nb.GetAtomicNum() == 7 and nb.IsInRing() for nb in atom.GetNeighbors()):
            return True
    return False


def _same_tautomer(mol, name: str) -> bool:
    """OPSIN's reading of ``name`` is the input's tautomer: equal fixed-H InChIs
    (``validation.protonation_identity.tautomer_verdict``). False when OPSIN cannot read
    the name or the verdict is not 'ok'."""
    from ...namer import _validity_gate_name_to_smiles
    from ...validation.protonation_identity import tautomer_verdict
    from ...validation.twin_spellings import opsin_spelling
    parsed = _validity_gate_name_to_smiles(opsin_spelling(name))
    if not parsed:
        return False
    return tautomer_verdict(Chem.MolToSmiles(mol), parsed) == "ok"


def _same_tautomer_as_radical(mol, name: str) -> bool:
    """OPSIN's reading of the '-yl' prefix ``name`` (a radical), its free valence capped with a
    hydrogen atom, is the input's tautomer: equal fixed-H InChIs (``validation.protonation_identity.
    tautomer_verdict``), the check ``_same_tautomer`` makes for a whole name. ``mol`` is the ring
    system with the broken bond left as a hydrogen atom, so the capped parse is that hydride.
    False when OPSIN cannot read the name or the verdict is not 'ok'."""
    from ...jvm_bridge import opsin_stdout
    from ...validation.opsin_roundtrip import _find_opsin_jar
    from ...validation.protonation_identity import tautomer_verdict
    from ...validation.twin_spellings import opsin_spelling
    # A read-back by the in-process OPSIN, which serves only the jar its JVM started with: the OPSIN
    # in use (ORTHONYM_OPSIN), not a pinned version.
    jar = _find_opsin_jar()
    text, served = opsin_stdout(opsin_spelling(name, radical=True), True, jar) if jar else (None, False)
    smiles = (text or "").strip()
    parsed = Chem.MolFromSmiles(smiles) if served and smiles else None
    if parsed is None:
        return False
    rw = Chem.RWMol(parsed)
    for atom in rw.GetAtoms():
        if atom.GetNumRadicalElectrons():
            atom.SetNumRadicalElectrons(0)
            atom.SetNoImplicit(False)
    try:
        capped = rw.GetMol()
        Chem.SanitizeMol(capped)
    except Exception:
        return False
    return tautomer_verdict(Chem.MolToSmiles(mol), Chem.MolToSmiles(capped)) == "ok"


def build_substituent(mol, attach: int) -> Optional[str]:
    """The '-yl' prefix of a bare bridged fused ring system ``mol`` (a detached ring
    system whose free valence is on atom ``attach``, the broken bond left as a hydrogen),
    or None. (the Blue Book): free-valence locants are "as low as is
    consistent with any established numbering of the parent hydride"; (:17326):
    "low locants go first to the fixed numbering of the system, then indicated hydrogen,
    followed by free valence suffix, and finally 'hydro' prefixes" -- the free valence
    takes the place of the suffix in (c) (:3256). Declined: a decorated ring
    system, a stereocentre, an indicated-hydrogen or hydro atom on or next to a ring
    nitrogen whose tautomer OPSIN's reading of the prefix does not share
    (``_same_tautomer_as_radical``)."""
    if mol is None or attach is None:
        return None
    system = selection.ring_system(mol)
    if set(system) != {a.GetIdx() for a in mol.GetAtoms()} or attach not in system:
        return None
    if any(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in mol.GetAtoms()):
        return None
    return _substituent_name(mol, system, attach, _options(mol, system))


def build_fused_substituent(mol, attach: int) -> Optional[str]:
    """The '-yl' prefix of a bare fused ring system ``mol``, carbocyclic or heterocyclic
    (``selection.fused_split``), whose free valence is on atom ``attach``: the rules and declines of
    ``build_substituent``."""
    if mol is None or attach is None:
        return None
    system = selection.ring_system(mol)
    if set(system) != {a.GetIdx() for a in mol.GetAtoms()} or attach not in system:
        return None
    if any(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in mol.GetAtoms()):
        return None
    if _isotope_labels_pending(mol, system):
        return None
    return _substituent_name(mol, system, attach, _fused_options(mol, system))


def build_fused_decorated_substituent(mol, attach: int, outside: Set[int]) -> Optional[str]:
    """The '-yl' prefix of the fused ring system of ``mol`` that holds ``attach``, with its own prefixes and
    stereodescriptors, when ``outside`` (the parent side, bonded to ``attach`` only) is left out:
    '(3S,8S,9S,10R,13R,14S,17R)-3-hydroxy-10,13-dimethylhexadecahydro-1H-cyclopenta[a]phenanthren-17-yl'.
     (the Blue Book) the free valence; (:23710) the fusion name; (:44643, example:44668) "When they
    relate to substituent groups, they are cited at the front of the corresponding prefix". The descriptors are the
    whole molecule's CIP labels on the ring locants, so the attachment atom's is the one in context; a stereogenic unit
    of a prefix of the system is cited by that prefix's namer at the front of the prefix ('17-[(2R)-6-methylheptan-2-
    yl]'), and the read-back of the whole name checks it. None when a step declines (the caller keeps its older path):
    a ring system bonded to the parent side more than once, a tautomer-sensitive hydrogen placement, isotope labels
    pending, a lone-pair centre."""
    if mol is None or attach is None or not outside:
        return None
    system = next((s for s in selection.ring_systems(mol) if attach in s), None)
    if not system or system & set(outside):
        return None
    side, stack = set(), list(system)
    while stack:
        c = stack.pop()
        if c in side:
            continue
        side.add(c)
        stack.extend(n.GetIdx() for n in mol.GetAtomWithIdx(c).GetNeighbors() if n.GetIdx() not in outside)
    links = [(a, n.GetIdx()) for a in side for n in mol.GetAtomWithIdx(a).GetNeighbors() if n.GetIdx() in outside]
    if len(links) != 1 or links[0][0] != attach:
        return None
    # a double bond to the parent side is an '-ylidene' prefix, not '-yl'
    if mol.GetBondBetweenAtoms(*links[0]).GetBondType() != Chem.BondType.SINGLE:
        return None
    if _isotope_labels_pending(mol, system) or _lone_pair_centre_in(mol, system):
        return None
    name = _substituent_name(mol, system, attach, _fused_options(mol, system), frozenset({links[0][1]}))
    # (c) (the Blue Book): a substituted prefix is multiplied by 'bis', 'tris'
    from ...assembly.prefix_derivation import built
    return built(name, substituted=bool(side - system)) if name else None


def _substituent_name(mol, system: Set[int], attach: int, options, exo: FrozenSet[int] = frozenset()) -> Optional[str]:
    """The '-yl' prefix of ``system`` with the free valence on ``attach``; ``exo`` are the atoms of the parent side
    (none for a bare system), left out of the prefixes."""
    if not options:
        return None
    if len({(o[0].parent, tuple(sorted(bp.name for bp in o[1]))) for o in options}) != 1:
        return None
    suffix = ("suffix", "yl", {attach}, set(exo))
    # the ring atom of each prefix bond, for (f) (:3301; a gem pair twice); none for a bare system
    sub_atoms = [a for a in system for nb in mol.GetAtomWithIdx(a).GetNeighbors()
                 if nb.GetIdx() not in system and nb.GetIdx() not in exo]
    scored = []
    sensitive = False
    for sp, bps, state, nums in options:
        need = _lacking(state, "yl", {attach})
        for nb in nums:
            a2l = nb.atom_to_locant
            got = hydro.accommodate(state, a2l, need)
            if got is None:
                continue
            ih, added, hydro_atoms = got
            sensitive = sensitive or _tautomer_sensitive(mol, set(ih) | set(added), hydro_atoms)
            key = (nb.attachment_set, nb.citation_order,
                   numbering.locant_tuple(a2l[a] for a in ih),
                   numbering.locant_tuple([a2l[attach]]),
                   numbering.locant_tuple(a2l[a] for a in added),
                   numbering.locant_tuple(a2l[a] for a in hydro_atoms),
                   numbering.locant_tuple(a2l[a] for a in sub_atoms))
            scored.append((key, sp, bps, state, nb, ih, hydro_atoms, added, need))
    if not scored:
        return None
    picked = _choose(mol, system, scored, suffix, set(exo), "yl", {attach})
    if picked is None:
        return None
    name, parent, bps = picked
    # the radical read-back compares the whole of ``mol``; a decorated substituent has no such molecule
    if sensitive and (exo or not _same_tautomer_as_radical(mol, name)):
        return None
    if any(not bp.is_pin_form for bp in bps):
        from ..bridged_fused import _record_general_bridge
        for bp in bps:
            _record_general_bridge(bp.name)
    return name
