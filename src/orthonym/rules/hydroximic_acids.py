"""Hydroximic acids: the N-hydroxy derivatives of the imidic acids.

 "Substitutive nomenclature, suffix mode" (the Blue Book,:30096): "Acids in which the
carbonyl oxygen atom of a carboxylic acid group has been replaced by =N-OH are named as *N*-hydroxy
derivatives of imidic acids named as described in. This method is used to generate preferred
IUPAC names." ('N-hydroxyethanimidic acid (PIN)',:30106; 'N-hydroxybenzenecarboximidic acid (PIN)',:30108.)
The acid chalcogen may be S, Se or Te, (:30208): 'N-hydroxycyclohexanecarboximidoselenoic acid
(PIN)' (:30273), so the suffix is that of the imidic, imidothioic, imidoselenoic or imidotelluroic acid of
the same chalcogen (:func:`suffix_group`) and the N-hydroxy is a PREFIX, cited with the others in one
alphanumerical series 'ALPHANUMERICAL ORDER',:3436,:3440; 'N,4-dimethyl-N-(3-methylphenyl)benzamide (PIN)',:32879, puts the
italic N and the numbers in one list). Its italic locant is essential "Citation of locants",
:2869), so no licence withholds a locant beside it (``n_substituted_acids.n_italic_locant_cited``), and a
'hydroxy' the parent also carries is one multiplied prefix with it (b),:7067): 'N,2-dihydroxy-
propanimidic acid'. The group demoted by a senior one is the compound prefix 'C,N-dihydroxycarbonimidoyl'
on a ring (1),:30131; '4-(C,N-dihydroxycarbonimidoyl)benzoic acid (PIN)',:30138) and
'hydroxy' with 'hydroxyimino' at the end of a chain (:30132).

The group used to be built by one handler that put 'N-hydroxy' in front of the finished name of the imidic
acid, which is right only when that name has no prefix of its own: a substituted acyl group got
'N-hydroxy2-methylpropanimidic acid' or nothing. It is built now where the other groups are: the
fragment assembly of ``general_acyclic`` and ``polyfunctional`` for a chain or a saturated ring, the benzene
namer for a benzene ring. This module holds what those producers share. Not built, and left to the
producers that name every structure: two hydroximic groups on one parent (their N locants carry the
parent's numbers, 'N1,N4-dihydroxybutanediimidic acid',:30112), a ring that is neither benzene nor a
saturated carbocycle ('N2-hydroxy-1H-pyrrole-2-carboximidic acid (PIN)',:30121 numbers the N with its ring
position), and the S, Se and Te analogues demoted on a ring (the book gives them no prefix).

The standard InChIKey does not see the configuration of the C=N of -C(=N-OH)-XH (its mobile hydrogen):
'C/C(O)=N/O' and 'C/C(O)=N\\O' have one key. The descriptor ('(1E)-N-hydroxy...') is read back against OPSIN's
isomeric SMILES in the tests, not by the key.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from rdkit import Chem

# the N-substituent of the group, its suffix and its rank are the one rule of ``n_substituted_acids``; these
# names stay here for the callers that read the hydroximic acid through this module
from .n_substituted_acids import (  # noqa: E402,F401
    CN_GROUPS,
    IMIDIC_GROUP as _IMIDIC_GROUP,
    group_at_rank,
    suffix_group,
    suffix_group_atoms,
    with_n_prefix_fragments,
)


def junior_to(principal_group: Optional[str], mol, match) -> bool:
    """True when the hydroximic acid ``match`` is junior to ``principal_group``: the principal group is
    strictly senior to the imidic acid it is the N-hydroxy derivative of.

     (the Blue Book): the prefixes are used "When another group is present that has
    seniority for citation as principal group". A plain imidic acid has none over an N-hydroxy imidic acid
    (both are imidic-acid suffixes:,:18875, "the maximum number of substituents corresponding
    to the principal characteristic group", 'N1,N'4-dihydroxybenzene-1,4-dicarboximidic acid (PIN)',
    :30125), and a hydrazonic acid or an imidothioic acid is junior to it (class 7a,:18172)."""
    from .seniority import seniority_index
    try:
        return seniority_index(principal_group) < seniority_index("hydroximic_acid", mol, [match])
    except ValueError:                   # not a group of the order: nothing is known to be senior to it
        return False


# ---------------------------------------------------------------------------------------------------------
# The configuration of the C=N of a ring-parent group: a descriptor with no locant
# ---------------------------------------------------------------------------------------------------------

def cn_labels(mol) -> Dict[frozenset, str]:
    """``{frozenset({C, N}): 'E' | 'Z'}`` for the configured C=N bonds of the hydroximic and N-substituted imidic acids of ``mol``."""
    from ..perception.stereo import assign_stereochemistry, get_double_bond_stereo
    assign_stereochemistry(mol)
    groups = {frozenset(m[:2]) for m in mol.GetSubstructMatches(_CN_GROUP)}
    return {frozenset(d["atoms"]): d["stereo"] for d in get_double_bond_stereo(mol)
            if frozenset(d["atoms"]) in groups}


def with_n_group_fragments(features: Any, fragments: list) -> Optional[list]:
    """``fragments`` (parent, suffix, prefix and stereo fragments of a general-acyclic name) with the N-substituent
    of the principal group cited among the prefixes (:func:`with_n_prefix_fragments`: the N-hydroxy of a hydroximic,
    hydroxamic or sulfonimidic acid, the carbon group on the N of an imidic acid) and, for a hydroximic or
    N-substituted imidic acid on a ring parent, the configuration of its C=N in the stereo fragment; None where
    the name cannot be built.

    A descriptor that relates to the parent structure is "placed at the front of the complete name", "preceded by
    a numerical or letter locant... when such locants are present", the Blue Book). The carbon of
    the suffix of 'cyclohexanecarboximidic acid' is not an atom of the ring and has none, so the descriptor is
    bare, as for the one-carbon parent of '(Z)-N-hydroxy(4-chlorophenyl)(phenyl)methanimine (PIN)' (:47666):
    '(E)-N-hydroxycyclohexanecarboximidic acid'. (A chain parent numbers the carbon: ``_generate_stereodescriptors``
    cites '(1E)-'.) Another descriptor at the front would need one block with a bare member, which the book does
    not print: the name is not built. Unchanged for every other group."""
    from ..assembly.composer import NameFragment
    from .n_substituted_acids import N_PREFIX_GROUPS
    if features.principal_group not in N_PREFIX_GROUPS:
        return fragments
    prefixes = with_n_prefix_fragments(features, [f for f in fragments if f.fragment_type == "prefix"])
    if prefixes is None:
        return None
    stereo = [f for f in fragments if f.fragment_type == "stereo"]
    label = None if features.principal_chain or features.principal_group not in CN_GROUPS else cn_labels(
        features.mol).get(frozenset(features.principal_group_atoms[0][:2]))
    if label:
        if stereo:
            return None
        stereo = [NameFragment(text=f"({label})-", fragment_type="stereo")]
    return [f for f in fragments if f.fragment_type in ("parent", "suffix")] + prefixes + stereo


def ring_stereodescriptors(features: Any, descriptors: list) -> Optional[list]:
    """``descriptors`` (``(locant, label)`` of the parent's stereogenic units) with the bare descriptor of the C=N of
    a principal hydroximic acid on a ring parent (see:func:`with_n_group_fragments`), or None where another
    descriptor stands at the front too. Unchanged for every other group."""
    matches = features.principal_group_atoms or []
    if features.principal_group not in CN_GROUPS or len(matches) != 1:
        return descriptors
    label = cn_labels(features.mol).get(frozenset(matches[0][:2]))
    if not label:
        return descriptors
    return None if descriptors else [("", label)]


# ---------------------------------------------------------------------------------------------------------
# A ring-attached group demoted by a senior one (polyfunctional ring parents)
# ---------------------------------------------------------------------------------------------------------

#: (1) (the Blue Book) and (1) (:30131): the compound prefix of the group
#: bonded to a ring ('2-(C-hydroxycarbonimidoyl)cyclopentane-1-carboxylic acid (PIN)',:30029;
#: '2-(C,N-dihydroxycarbonimidoyl)cyclopentane-1-carboxylic acid (PIN)',:30134)
# Substituted substituent prefixes ('hydroxy' on 'carbonimidoyl'), so the multiplicative forms 'bis', 'tris'
# "General methodology" (c),:7035;,:2826): '3,5-bis(C-hydroxycarbonimidoyl)benzoic acid'.
# Each is built with its writer's record (``assembly.prefix_derivation``), which ``is_substituted_substituent``
# reads, and descriptor-bearing forms are built the same way.
def _compound_prefix(text: str) -> str:
    from ..assembly.prefix_derivation import built
    return built(text, substituted=True, enclosed=True)


_RING_ACYL_PREFIX = {"imidic_acid": _compound_prefix("C-hydroxycarbonimidoyl"),
                     "hydroximic_acid": _compound_prefix("C,N-dihydroxycarbonimidoyl")}


def ring_acyl_prefixes(features, fg_name: str, matches, ring_set, atom_to_locant,
                       principal_group: Optional[str]):
    """The formatted ring prefixes of the imidic or hydroximic acids ``matches`` bonded to a ring, or None where
    one of them cannot be cited (its carbon is a ring atom, it has no single ring neighbour, or it has no
    compound prefix::func:`ring_prefix_ok`).

    The compound prefix of the group bonded to the ring, 'C-hydroxycarbonimidoyl' (1), the Blue Book)
    or 'C,N-dihydroxycarbonimidoyl' (1),:30131). A descriptor of the C=N of the hydroximic acid
    relates to the substituent group, so it "is cited at the front of the corresponding prefix",:44639),
    within its enclosing marks, and without a locant, for the carbon of the acyl group has none ('1-[(Z)-ethylidene]-
    3-[(E)-ethylidene]cyclopentane',:48292; '(Z)-N-hydroxy(4-chlorophenyl)(phenyl)methanimine (PIN)',:47666):
    '4-[(E)-C,N-dihydroxycarbonimidoyl]benzoic acid'. Groups of one configuration are one multiplied prefix,
    'bis[(E)-C,N-dihydroxycarbonimidoyl]' (c),:7035); groups of two are two prefixes."""
    from .polyfunctional import format_fg_prefix
    mol = features.mol
    labels = cn_labels(mol) if fg_name == "hydroximic_acid" else {}
    groups: Dict[str, List[int]] = {}
    for match in matches:
        carbon = match[0]
        ring_neighbours = [nb.GetIdx() for nb in mol.GetAtomWithIdx(carbon).GetNeighbors()
                           if nb.GetIdx() in ring_set and nb.GetIdx() in atom_to_locant]
        if (carbon in ring_set or len(ring_neighbours) != 1
                or not ring_prefix_ok(fg_name, mol, match, principal_group)):
            return None
        label = labels.get(frozenset(match[:2]))
        form = _RING_ACYL_PREFIX[fg_name]
        groups.setdefault(_compound_prefix(f"({label})-{form}") if label else form, []).append(
            atom_to_locant[ring_neighbours[0]])
    return [format_fg_prefix(form, sorted(locants), len(locants)) for form, locants in groups.items()]


def ring_prefix_ok(fg_name: str, mol, match, principal_group: Optional[str]) -> bool:
    """False for a group that has no compound prefix on a ring: a hydroximic acid that is not junior to the
    principal group (:func:`junior_to`: two imidic acids of one rank are a dicarboximidic acid with an
    N-hydroxy,:30125, not built) or whose acid chalcogen is not O (the book gives the S, Se and Te
    analogues none)."""
    return (fg_name != "hydroximic_acid"
            or (len(match) >= 4 and mol.GetAtomWithIdx(match[3]).GetSymbol() == "O"
                and junior_to(principal_group, mol, match)))


def chain_end_prefixes(mol, matches, chain_set, atom_to_locant, principal_group: Optional[str]):
    """The prefixes of the hydroximic acids ``matches`` demoted at the end of a chain: ``(prefixes, atoms)``;
    None where the group is not at a chain end (the caller goes on as before).

     (2) (the Blue Book): "the combination of the prefixes 'hydroxy' and 'hydroxyimino'
    at the end of a carbon chain is used in preferred IUPAC names rather than the prefix
    'dihydroxycarbonimidoyl'" ('5-hydroxy-5-(hydroxyimino)pentanoic acid (PIN)',:30140); the S, Se and Te
    analogues give 'sulfanyl' and so on for the 'hydroxy',:30208). 'hydroxyimino' is a compound
    prefix, parenthesised. A match is (C, imino N, O on N, acid X).

    The prefixes are the PIN's only under a principal group that is strictly senior (:func:`junior_to`). Under an
    acid of the group's own rank the name ('4-hydroxy-4-(hydroxyimino)butanimidic acid') still describes the
    structure (a read-back holds) but is not the PIN, which is the dicarboximidic acid with an N-hydroxy
    (:30125, not built): it is recorded as a non-PIN part (``record_non_pin_label``), so the default tier
    declines it and the best-effort tier ships it below the PIN, never deletes it."""
    from .polyfunctional import _acid_chalcogen_prefix, format_fg_prefix
    units = {}
    for m in matches:
        if len(m) < 4 or m[0] not in chain_set:
            return None
        units[m[0]] = m
    prefixes, atoms = [], set()
    for c, m in units.items():
        loc = atom_to_locant.get(c)
        if loc is None:
            return None
        prefixes += [format_fg_prefix(_acid_chalcogen_prefix(mol, m[3]), [loc], 1),
                     format_fg_prefix('(hydroxyimino)', [loc], 1)]
        atoms.update(m[1:4])
        if not junior_to(principal_group, mol, m):
            from ..metrics.provenance import record_non_pin_label
            record_non_pin_label(f"{loc}-(hydroxyimino)", rule="P-44.1.1",
                                 detail="an imidic acid of the same rank is a second suffix, not a prefix")
    return (prefixes, atoms) if prefixes else None


# ---------------------------------------------------------------------------------------------------------
# Benzene (rules/benzene.py reads these through rules/benzene_acid_suffixes.py)
# ---------------------------------------------------------------------------------------------------------

#: ring suffix name of a hydroximic acid on benzene -> the imidic suffix it is the N-hydroxy derivative of.
#: The suffix name carries the N-hydroxy, so the group is its own class on the ring (it demotes to its own
#: prefix and cannot be mistaken for the plain imidic acid), and the configuration of its C=N ('(E)-', '(Z)-': a
#: name of its own for each, so that two groups of two configurations are two prefixes). O acids first: the
#: order is the order of seniority (``benzene._SUFFIX_PRIORITY``).
_CHALCOGEN_ACIDS = ("carboximidic acid", "carboximidothioic acid", "carboximidoselenoic acid",
                    "carboximidotelluroic acid")
#: The N-substituted imidic acids (``n_substituted_acids``) are the same classes with a carbon group on the N in
#: place of the OH: 'N-substituted' stands for the group in the suffix name, the name of the group is the
#: record's ``n_substituents``. They have no prefix on the ring (the book prints none for a carbon group on the N,
#: (1),:30020), so they are claimed only as the principal group (:func:`claims_benzene_group`).
BENZENE_SUFFIX_NAMES = {
    f"{descriptor}N-{kind}{imidic}": imidic
    for kind in ("hydroxy", "substituted ") for imidic in _CHALCOGEN_ACIDS for descriptor in ("", "(E)-", "(Z)-")
}
#: (group key, ring suffix name, SMARTS). The =N-OH (a bonded O with its H) keeps the oxime ether and the
#: amidoxime out, the -XH the imidate ester
_HYDROXIMIC_BENZENE_ROWS = (
    ("hydroximic_acid", "N-hydroxycarboximidic acid", "[CX3](=[NX2][OX2H1])[OX2H1]"),
    ("hydroximic_thioic_acid", "N-hydroxycarboximidothioic acid", "[CX3](=[NX2][OX2H1])[SX2H1]"),
    ("hydroximic_selenoic_acid", "N-hydroxycarboximidoselenoic acid", "[CX3](=[NX2][OX2H1])[SeX2H1]"),
    ("hydroximic_telluroic_acid", "N-hydroxycarboximidotelluroic acid", "[CX3](=[NX2][OX2H1])[TeX2H1]"),
)
_N_SUBSTITUTED_BENZENE_ROWS = tuple(
    (f"n_substituted_imidic_{x}", f"N-substituted {imidic}",
     f"[CX3](=[NX2;!R;$([NX2][#6,$([SX2H1,SeX2H1,TeX2H1])])])[{x}X2H1]")
    for imidic, x in zip(_CHALCOGEN_ACIDS, ("O", "S", "Se", "Te")))
_BENZENE_ROWS = _HYDROXIMIC_BENZENE_ROWS + _N_SUBSTITUTED_BENZENE_ROWS
#: (key, imidic suffix) in the order the groups are looked for, as ``benzene_acid_suffixes.ACID_SUFFIX_FGS``
ACID_SUFFIX_FGS = tuple((key, BENZENE_SUFFIX_NAMES[name]) for key, name, _ in _BENZENE_ROWS)
ACID_SUFFIX_SMARTS = {key: Chem.MolFromSmarts(smarts) for key, _, smarts in _BENZENE_ROWS}
_SUFFIX_OF_KEY = {key: name for key, name, _ in _BENZENE_ROWS}
_N_SUBSTITUTED_KEYS = frozenset(key for key, _, _ in _N_SUBSTITUTED_BENZENE_ROWS)
IMIDIC_DEMOTED_PREFIX = _RING_ACYL_PREFIX["imidic_acid"]
#: (1) (:30138); the S, Se and Te analogues have no prefix and are claimed only as the suffix. The prefix
#: of a configured group carries its descriptor at its front,:44639; ``ring_acyl_prefixes``).
DEMOTED_PREFIX = {f"{d}N-hydroxycarboximidic acid": _compound_prefix(f"{d}C,N-dihydroxycarbonimidoyl")
                  for d in ("", "(E)-", "(Z)-")}


def acid_suffix_extra(key: str, mol, match) -> Dict[str, Any]:
    """What the record of the group ``key`` carries beyond the plain acid record: its own suffix name (with the
    descriptor of its C=N where it is configured) and its N-substituent; nothing for the plain acids."""
    if key not in _SUFFIX_OF_KEY:
        return {}
    label = cn_labels(mol).get(frozenset(match[:2]))
    n_substituents = ["hydroxy"]
    if key in _N_SUBSTITUTED_KEYS:
        from ..perception.functional_groups import detect_functional_groups
        from .n_substituted_acids import n_substituents as _n_substituents
        found = _n_substituents(mol, "n_substituted_imidic_acid", match, detect_functional_groups(mol)) or []
        n_substituents = [name for name, _ in found]
    return {"suffix_name": f"({label})-{_SUFFIX_OF_KEY[key]}" if label else _SUFFIX_OF_KEY[key],
            "n_substituents": n_substituents}


def claims_benzene_group(key: str, mol) -> bool:
    """True when the benzene namer takes the group ``key`` it matched: every group but a hydroximic acid with
    an S, Se or Te acid group (no prefix form in the book: it is the suffix or it is not named here) and an
    N-substituted imidic acid (none either), that is not the principal group of the molecule."""
    if key not in ("hydroximic_thioic_acid", "hydroximic_selenoic_acid", "hydroximic_telluroic_acid",
                   *_N_SUBSTITUTED_KEYS):
        return True
    from ..perception.functional_groups import detect_functional_groups
    from .seniority import get_principal_group
    return get_principal_group(mol, detect_functional_groups(mol))[0] == (
        "n_substituted_imidic_acid" if key in _N_SUBSTITUTED_KEYS else "hydroximic_acid")


#: what ``name_imidic_on_benzene`` returns for a ring the general suffix assembly names
UNHANDLED = object()


def name_imidic_on_benzene(chosen_suffix: str, suffix_groups: Dict[str, List[int]],
                           n_substituents_map: Dict[str, List[str]],
                           remaining_prefix_groups: Dict[str, List[int]], locants: List[int],
                           stereo_descriptors=None) -> Any:
    """The name of a benzene ring whose suffix is a single imidic acid, plain or N-hydroxy; ``UNHANDLED`` for
    any other suffix and for several imidic acids (they fall through to the general assembler:
    'benzene-1,4-dicarboximidic acid'); None where the ring is not built.

    Suite fix j6 (TRIAGE g8 C23): (the Blue Book) 'benzenecarboximidic acid (PIN)' (bare,
    no locant); with other substituents the PIN cites '-1-' on the group: '2-(propanimidoylselanyl)benzene-1-
    carboximidic acid (PIN)' (:32013).

    A hydroximic acid is the N-hydroxy derivative,:30096): 'N-hydroxybenzenecarboximidic acid
    (PIN)' (:30108), and with other ring prefixes the suffix keeps its locant, 'N-hydroxy-4-methylbenzene-1-
    carboximidic acid', as '4-methylbenzene-1-carboximidic acid' does,:2869). Position 1 is the carbon
    of the group. The N-hydroxy and a ring 'hydroxy' are ONE multiplied prefix, (b) ('N,2-dihydroxy-
    benzene-1-carboximidic acid'), and the whole set is alphanumerically ordered,:3440), as
    ``_name_substituted_benzamide`` does for the italic N of an amide. The configuration of its C=N is the
    descriptor at the front of the name ('(E)-N-hydroxybenzenecarboximidic acid'; ``with_n_group_fragments``),
    in the one block with no other: ring ``stereo_descriptors`` (none, or not established) are not built beside it.
    One group only: two would need the N locants to carry the ring numbers of their groups ('N1,N4-dihydroxy-
    benzene-1,4-dicarboximidic acid',:30125), and so would a plain imidic acid or a hydroximic acid of another
    configuration beside it (one rank: a dicarboximidic acid with an N-hydroxy on one of its N, not an imidic acid
    with a 'C,N-dihydroxycarbonimidoyl' prefix). Every other group on the ring must be one the name cites as a
    prefix."""
    from .benzene import _SUFFIX_TO_PREFIX, _build_prefix_string_with_locants, _renumber_relative_to

    hydroximic = chosen_suffix in BENZENE_SUFFIX_NAMES
    if chosen_suffix != "carboximidic acid" and not hydroximic:
        return UNHANDLED
    front = ""
    if hydroximic:
        imidic = BENZENE_SUFFIX_NAMES[chosen_suffix]
        if (len(locants) != 1 or len(n_substituents_map.get(chosen_suffix) or []) != 1
                or "carboximidic acid" in suffix_groups
                or any(sfx not in _SUFFIX_TO_PREFIX or BENZENE_SUFFIX_NAMES.get(sfx) == imidic
                       for sfx in suffix_groups if sfx != chosen_suffix)):
            return None
        if chosen_suffix.startswith("("):
            if stereo_descriptors is None or stereo_descriptors:
                return None
            front = chosen_suffix[:chosen_suffix.index(")") + 2]
    else:
        if any(name in suffix_groups for name in BENZENE_SUFFIX_NAMES):
            return None
        if len(locants) != 1:
            return UNHANDLED
        imidic = chosen_suffix
        if not remaining_prefix_groups:
            return "benzenecarboximidic acid"
    renumbered_groups = _renumber_relative_to(remaining_prefix_groups, locants[0])
    merged: Dict[str, List] = {name: list(locs) for name, locs in renumbered_groups.items()}
    if hydroximic:
        merged.setdefault(n_substituents_map[chosen_suffix][0], []).append("N")
    parent = f"benzene-1-{imidic}" if renumbered_groups else f"benzene{imidic}"
    return f"{front}{_build_prefix_string_with_locants(merged, mono_needs_locant=True)}{parent}"


# ---------------------------------------------------------------------------------------------------------
# The configuration of the C=N, which the standard InChIKey does not see
# ---------------------------------------------------------------------------------------------------------

#: C(=N-OH)(XH) and C(=N-R)(XH), R a carbon: the groups whose C=N has a mobile hydrogen (the group is a tautomer of
#: the amide, thioamide... of the same atoms, so the key is one for both, and for both configurations)
_CN_GROUP = Chem.MolFromSmarts(
    "[CX3](=[NX2;!R;$([NX2][#6,$([OX2H1,SX2H1,SeX2H1,TeX2H1])])])[#8,#16,#34,#52;X2H1]")


def _cn_configurations(mol) -> Dict[Any, List[str]]:
    """``{place of the group: sorted configurations}`` of the hydroximic acids of ``mol``: 'STEREOE', 'STEREOZ'
    ... for a configured C=N and '-' for one that is not. The place is the canonical class of the group's
    C and N (constitution only), so two readings of one molecule key a group alike; groups of one class
    (equivalent groups) share a place and a list."""
    matches = mol.GetSubstructMatches(_CN_GROUP)
    if not matches:
        return {}
    ranks = (list(Chem.CanonicalRankAtoms(mol, breakTies=False, includeChirality=False))
             if len(matches) > 1 else None)
    found: Dict[Any, List[str]] = {}
    for m in matches:
        bond = mol.GetBondBetweenAtoms(m[0], m[1])
        label = "-" if bond is None or bond.GetStereo() == Chem.BondStereo.STEREONONE else str(bond.GetStereo())
        place = tuple(sorted((ranks[m[0]], ranks[m[1]]))) if ranks else ()
        found.setdefault(place, []).append(label)
    return {place: sorted(labels) for place, labels in found.items()}


def cn_configuration_verdict(input_smiles: Optional[str], opsin_smiles: Optional[str]) -> str:
    """``"mismatch"`` when OPSIN's reading of a name has not the configuration of the C=N of a hydroximic acid
    that the input has, else ``"ok"``.

    The mobile hydrogen of -C(=N-OH)-XH puts the group in one layer of the standard InChI, so the full key is
    the same for both configurations ('C/C(O)=N/O' and 'C/C(O)=N\\O' share an InChIKey) and
    no gate that compares keys sees a descriptor that is flipped, or one that is missing. The configurations are
    compared group by group, both ways: a name that states one the input does not carry, one that is not the
    input's, or none where the input configures one,:44718: "In preferred IUPAC names, all stereogenic
    units must be specified, unless an omission is allowed according to ", which allows it for the double
    bonds of three- to seven-membered alicyclic rings only,:44637). The reading, not the string of the name, says
    what the name states. Cheap: nothing is parsed unless a SMILES configures a double bond."""
    if not input_smiles or not opsin_smiles:
        return "ok"
    if not any(mark in smiles for smiles in (input_smiles, opsin_smiles) for mark in "/\\"):
        return "ok"
    mi, mo = Chem.MolFromSmiles(input_smiles), Chem.MolFromSmiles(opsin_smiles)
    if mi is None or mo is None:
        return "ok"
    return "ok" if _cn_configurations(mi) == _cn_configurations(mo) else "mismatch"
