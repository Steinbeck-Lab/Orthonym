"""Which name of one species' charge forms stands.

A cation whose +1 can sit on several nitrogen atoms is one species (``namer._charge_forms``), and
``Orthonym._name_with_form_choice`` names each of its forms and keeps one name. This module is the rule
that ranks two of those names. It is pure string logic (no RDKit, no handlers), and it compares only
the names of one species, so what it reads from a name is what the Blue Book orders by.

The order, each step a whole criterion before the next, the first that differs deciding:

0. **More of the input's configured stereo elements kept** (the ``*_stereo`` counts of:func:`beats`, from
   ``perception.charge_shift.kept_stereo``, counted on each form's own mol against the drawn input): a form is
   written from a Kekule structure whose double bonds are not the drawing's, and a bond that is single in it
   cannot carry the E/Z the input configures, so the name of such a form is the name of a less specific compound
   than the input. STERODESCRIPTORS (the Blue Book), "The configuration of an organic compound is
   systematically indicated by one or more affixes added to a name that does not itself prescribe
   configuration; such affixes are called 'stereodescriptors'" (:44570); Omission of stereodescriptors
   (:44635) allows omitting them for double bonds only in three- to seven-membered alicyclic rings and in the
   ring systems it lists, "From nine-membered rings onwards, stereodescriptors are required" (:44637), never for
   an acyclic polymethine. NAMING OF STEREOISOMERS (:44639) says the descriptors "do not change the name or
   the numbering of a compound established by the principles, rules and conventions", so the Blue Book gives no
   rank between this and the choices below: ASSUMED order, the one that never lets the choice drop a
   configuration the input carries (the contributor guide core principle 2). The standard InChIKey that admits a form does
   not see these bonds, so no other gate does either. Where the counts are equal or not given, this is a tie.
1. **Fewer constructs the engine records as not PIN** (:func:`non_pin_parts`, the fragments): a construction
   the Blue Book does not allow in a PIN, or allows only where no fusion or Hantzsch-Widman name exists
    "Five-membered ring requirement" (:23708), "When fusion names are not allowed, unsaturated von
   Baeyer ring system names are preferred IUPAC names" (:23710); (:23682), "Preferred IUPAC names
   for heteromonocyclic rings with no more than ten ring members are Hantzsch-Widman names"; 'azamethane' is
   no Blue Book name (0 occurrences in the Blue Book), the one-atom nitrogen parent being azane,
   "General rule for systematically naming cationic centers in parent hydrides",:41366). A name that is built from a construction
   of that kind is no candidate for the preferred name, whatever its parent, so this comes first; for the
   names of one species that use the same constructions it is a tie.
2. ** CHOICE OF A PARENT STRUCTURE** (the Blue Book): "A parent cationic structure is chosen
   by applying the following criteria in order until a definitive choice is achieved: (a) parent with
   the maximum number of cationic, including cationic suffixes" (:42334; its example,
   2-(piperidin-1-ium-3-yl)propane-1,2-bis(aminium) over 3-(1,2-diaminiumylpropan-2-yl)piperidin-1-ium,
   "bis(aminium) is senior to piperidinium"). A name whose parent holds the cation stands over one that
   moves the cation into a substituent prefix and leaves the parent neutral (:func:`cationic_parent_centres`).
3. ** NUMBERING** (:3219), for two names that are one skeleton with other locants
   (:func:`lower_locants`): (b) indicated hydrogen, (c) "principal characteristic groups and free
   valences (suffixes)" (the ionic suffix 'ium' and the free valence 'yl', of the parent, then of a prefix),
   (e)(i) "low locants are given to hydro/dehydro prefixes... and 'ene' and 'yne' endings", with, for the
   ene list of a von Baeyer name, **** (:16633, "If there is a choice of names and numbering, the
   following criteria are considered in order until a decision is reached"): "(1) a minimum number of
   compound locants" (:16635), "(2) when comparing double bond locants that also include compound locants,
   any number in parentheses is ignored" (:16655), "(3) if there is still a choice, low locants are
   selected considering all locants (including those in parentheses) as a set" (:16667); then (f)
   "detachable alphabetized prefixes, all considered together in a series of increasing numerical
   order" (:3301), (g) "lowest locants for the substituent cited first as a prefix in the name"
   (:3307), then (j) "the lower locant is assigned to CIP stereodescriptors Z, R, M, and r" (:3346)
   for the leading configuration block. Between two Kekule structures of one ring system says the same thing ("A benzene
   ring is shown and described as a cyclohexatriene corresponding to the Kekule structure. Other
   aromatic rings are treated similarly, when required",:16635). ASSUMED (the Blue Book orders the
   criteria for numbering one structure, not for choosing between two drawings of a delocalised cation):
   that the same order ranks the names of two drawings.
4. **Fewer parts the engine records as label-only non-PIN** (the second number of:func:`non_pin_parts`): a
   part that is never the PIN spelling but that the PIN tier's promotion re-run still ships.

The caller keeps the form that comes first by canonical SMILES where none of these decides, so one species
gets one name whichever drawing it came from (the contributor guide core principle 4).

**A form that no producer names is not ranked, and it does not lower the label of the name the species gets**
(decision LO9, `internal notes`): an unnamed form has the rank (0, 0),
below every label. ``pin_verified`` says that the shipped name is the PIN of the structure, which the Blue Book's
criteria decide; whether every drawing of the structure could be named is not one of them. INTRODUCTION
(the Blue Book),:42411: "in zwitterionic compounds anionic centers are preferred for lower locants and become
the parent structure, into which the cationic part is substituted"; CHOICE OF A PARENT STRUCTURE (:42330),
(a) "parent with the maximum number of cationic, including cationic suffixes" (:42334). So the form whose cation sits
in the parent (a ring 'ium', '2-amino-4,5-dihydro-1,3-thiazol-3-ium-4-carboxylate') stands over the form whose cation
can only be a prefix ('2-azaniumylidene-1,3-thiazolidine-4-carboxylate', CATIONIC PREFIX NAMES,:42290)
whether or not that prefix name could be built, and the label stays. An unnamed form of the same ring cation with
other locants is another matter: it can outrank the survivor on (b), (c), (f), which nobody computed. That
is closed by naming the form (``rules/azolium_cations.azolium_hydron_site_name`` for an azolium that carries a
principal group; the rule of this module then ranks it), never by withdrawing the label of a name for which the
Blue Book gives no proof that it is not the PIN."""
from __future__ import annotations

import re
from typing import Iterable, Optional, Tuple

from .naming_utils import bracket_depths, top_level_text, top_level_tokens

# ---------------------------------------------------------------------------------------------------------------------
# (a): the cationic centres of the parent
# ---------------------------------------------------------------------------------------------------------------------

_MULTIPLIER_VALUE = {'bis': 2, 'tris': 3, 'tetrakis': 4, 'di': 2, 'tri': 3, 'tetra': 4, 'penta': 5, 'hexa': 6}

#: a multiplied cationic suffix written in enclosing marks, 'propane-1,2-bis(aminium)': the marks belong to the
#: multiplier, not to a substituent prefix, so the suffix is part of the parent
_MULTIPLIED_SUFFIX = re.compile(r"(bis|tris|tetrakis)\(([a-z]*ium)\)")

#: the cationic endings of (ring 'ium' after a locant, 'ylium'), ('aminium',
#: 'iminium', 'amidium', 'imidium', 'nitrilium') and the 'anium' of a mononuclear parent hydride (azanium,
#: oxidanium, sulfanium, phosphanium...); a word such as 'sodium' or 'titanium' is none of them
_CATIONIC_ENDING = re.compile(
    r"(?:(?<=-)(?P<m1>di|tri|tetra|penta|hexa)?(?:ylium|ium)"
    r"|(?P<m2>bis|tris|tetrakis|di|tri|tetra|penta|hexa)?(?:amin|imin|amid|imid|nitril)ium"
    r"|(?P<m3>di|tri|tetra|penta|hexa)?(?:az|oxid|sulf|phosph|ars|stib|sel|tell|sil|bor)anium)"
    r"(?![a-z])")


def cationic_parent_centres(name: str) -> Tuple[int, ...]:
    """The number of cationic centres the parent of each component of ``name`` expresses by a suffix: the
    endings of:data:`_CATIONIC_ENDING` outside every enclosing mark, a multiplied ending ('-6,34-diium',
    'bis(aminium)') counted by its multiplier. (a), the Blue Book. The components of a salt or
    a solvate ('... chloride', '...-ium dibromide', '...-ium pentanedioate pentahydrate') are the words that
    stand outside every enclosing mark, so a cation of one species that one name multiplies ('bis[...-ium]')
    and the other writes twice is a different number of components, which:func:`beats` does not compare.
    Names compared here are the names of one species, so what differs is where the cation sits, in the
    parent or in a prefix."""
    counts = []
    for word in top_level_tokens(_MULTIPLIED_SUFFIX.sub(r"\1\2", name), ' ', '([{', clamp=True):
        for component in word.split('\u2014'):
            top = top_level_text(component, '([{', clamp=True)
            if not any(ch.isalpha() for ch in top):
                continue
            total = 0
            for m in _CATIONIC_ENDING.finditer(top):
                multiplier = m.group('m1') or m.group('m2') or m.group('m3')
                total += _MULTIPLIER_VALUE.get(multiplier, 1)
            counts.append(total)
    return tuple(counts)


# ---------------------------------------------------------------------------------------------------------------------
# /: the locant roles of a name
# ---------------------------------------------------------------------------------------------------------------------

_LOCANT_TOKEN = re.compile(r"(?<![A-Za-z\d\[\]^.])(\d+[a-z]?|[NOPS]'*)(?![A-Za-z\d])")

_INDICATED_H = re.compile(r"(?<![A-Za-z\d])(\d+[a-z]?(?:,\d+[a-z]?)*)H-")

#: the locants of the hydro prefixes ('2,3-dihydro-') and of the 'de'-hydro prefixes of: (e)(i)
_HYDRO_LOCANTS = re.compile(
    r"(?<![A-Za-z\d\[\]^.])((?:\d+[a-z]?|[NOPS]'*)(?:,(?:\d+[a-z]?|[NOPS]'*))*)-"
    r"(?:di|tri|tetra|penta|hexa|hepta|octa|nona|deca)?(?:de)?hydro(?![a-z])")

#: the ionic and free-valence endings that carry the locants of (c), "principal characteristic groups
#: and free valences (suffixes)": 'ium', 'ide', 'ylium', 'ylidene', 'ylidyne', and the 'yl' of a substituent
#: group that ends there ('-1-yl)'; 'ylmethyl' and the like carry their locant as a prefix locant), and the
#: 'carboxylate' of a zwitterion, the anionic suffix cited after the cationic one,:42447)
_ENDING = re.compile(
    r"-((?:\d+[a-z]?|[NOPS]'*)(?:,(?:\d+[a-z]?|[NOPS]'*))*)-(?:di|tri|tetra|penta|hexa)?"
    r"(?:ium|ide|uide|ylium|ylidene|ylidyne|yl|aminium|aminide|iminium|carboxylate)(?:\b|$)")

_ENE_MULTIPLIER = (r"(?:mono|di|tri|tetra|penta|hexa|hepta|octa|nona|deca|undeca|dodeca|trideca|tetradeca|"
                   r"pentadeca|hexadeca|heptadeca|octadeca|nonadeca|icosa|[a-z]{0,9}cosa|[a-z]{0,9}triaconta|"
                   r"[a-z]{0,9}tetraconta)")
_COMPOUND_ITEM = r"\d+[a-z]?(?:\(\d+[a-z]?\))?"
#: the locants of the 'ene' and 'yne' endings, a compound locant written 'n(m)' (1)): the list
#: that stands in front of '-undecaen', '-ene', '-trien', '-yn'; ``kind`` is 'en' or 'yn'
_ENE_LIST = re.compile(
    r"(?<=-)(?P<list>" + _COMPOUND_ITEM + r"(?:," + _COMPOUND_ITEM + r")*)"
    r"(?=-" + _ENE_MULTIPLIER + r"?(?P<kind>en|yn))")
_COMPOUND = re.compile(r"(\d+[a-z]?)(?:\((\d+[a-z]?)\))?")

#: the enclosing marks of two names of one skeleton differ where their nesting differs ('[2-(...)]' against
#: '(2-(...))' once a compound locant is dropped from the ene list), so the roles are read with one kind of mark;
#: a locant that opens a mark ('[3-(4-cyanophenyl)...') is then a locant like any other
_MARKS = str.maketrans('[]{}', '()()')


def _locant_sort_item(token: str):
    """A locant as a sortable item: an italic letter ('N', 'N'') before the numbers, unprimed
    before primed 'lowest locants')."""
    if token[0].isdigit():
        digits = re.match(r"\d+", token).group(0)
        return (1, int(digits), token[len(digits):])
    return (0, len(token) - 1, token[0])


def _ene_roles(lists: Iterable[Tuple[str, str]]):
    """The three roles of the ene and yne locants, each compared as a whole in turn, for the lists
    ((locants, 'en' or 'yn'),...) in the order the name cites them.

    Double bonds only,: the number of compound locants (1), the set of the locants with any number
    in parentheses ignored (2), the set of all the locants (3).

    Double and triple bonds, (the Blue Book), "If both double and triple bonds are present,
    the following criteria for numbering are considered, in order": "(1) lower locants are assigned to
    multiple bonds as a set" (:16677), "(2) lower locants are assigned to double bonds" (:16683), "(3)
    compound locants are kept to a minimum" (:16689): the set of the locants of both kinds, the set of the
    locants of the double bonds, the number of compound locants."""
    lists = list(lists)
    parsed = [([_COMPOUND.fullmatch(t) for t in text.split(',')], kind) for text, kind in lists]
    compound = tuple(sum(1 for m in items if m.group(2)) for items, _kind in parsed)
    if any(kind == 'yn' for _items, kind in parsed):
        every_bond = tuple(sorted(_locant_sort_item(m.group(1)) for items, _kind in parsed for m in items))
        double = tuple(sorted(_locant_sort_item(m.group(1))
                              for items, kind in parsed if kind == 'en' for m in items))
        return every_bond, double, compound
    main = tuple(tuple(sorted(_locant_sort_item(m.group(1)) for m in items)) for items, _kind in parsed)
    every = tuple(tuple(sorted(_locant_sort_item(t) for m in items for t in m.groups() if t))
                  for items, _kind in parsed)
    return compound, main, every


#: one item of a leading configuration block ('4S', '3aR', 'Z', 'r'): an optional locant and one CIP or
#: non-CIP symbol, then the whole block as the name opens with it: '(4S,6R)-', 'rel-(1R,2S)-'
_STEREO_ITEM = r"(?:\d+[a-z]?'*)?[RSEZMPrsmpct]\*?"
_LEADING_STEREO = re.compile(rf"^((?:rel-|rac-)?)\(({_STEREO_ITEM}(?:,{_STEREO_ITEM})*)\)-")
_STEREO_PAIR = re.compile(r"(\d+[a-z]?'*)?([RSEZMPrsmpct])")


def _leading_stereo(name: str):
    """(the name without its leading configuration block, the block as (locant item, symbol) pairs;
    no pairs when the name does not open with one). A locant-less symbol ('(E)-') sorts after
    every locant."""
    m = _LEADING_STEREO.match(name)
    if m is None:
        return name, ()
    pairs = tuple((_locant_sort_item(loc) if loc else (2, 0, ''), sym)
                  for loc, sym in _STEREO_PAIR.findall(m.group(2)))
    return m.group(1) + name[m.end():], pairs


def locant_roles(name: str):
    """(the name with its locants taken out, then the roles 'NUMBERING' (the Blue Book)
    ranks in turn: (b) indicated hydrogen, (c) the principal characteristic group or ionic suffix (those
    of the parent, outside every enclosing mark, then those of the prefixes), (e) the hydro prefixes, then
    the ene and yne locants by (compound locants, the set without and with the numbers in
    parentheses) or, where a triple bond is present, by (:func:`_ene_roles`), (f) the detachable alphabetized prefixes 'all considered together in a series of
    increasing numerical order' (:3301, the sorted set of their locants), then (g) 'lowest locants for the
    substituent cited first as a prefix in the name' (:3307, their locants in the order the name cites
    them), then (j) 'the lower locant is assigned to CIP stereodescriptors Z, R, M, and r... that are
    preferred to E, S, P, and s' (:3346, ``cip_locant_rank_key`` of the leading configuration block)).

    The leading configuration block is out of the skeleton, which keeps only its symbols as a set:
    (j) chooses between numberings of one set of descriptors, so names whose blocks hold other
    symbols (one atom R in one charge form and S in the other) are not one skeleton and are not
    compared."""
    from .naming_utils import cip_locant_rank_key
    name, stereo = _leading_stereo(name)
    indicated = tuple(_locant_sort_item(t) for m in _INDICATED_H.finditer(name)
                      for t in m.group(1).split(','))
    rest = _INDICATED_H.sub('H-', name).translate(_MARKS)
    ene_lists = []

    def _take(match):
        ene_lists.append((match.group('list'), match.group('kind')))
        return ''

    rest = _ENE_LIST.sub(_take, rest)
    depth_at = {i: before for i, _ch, before, _after in bracket_depths(rest, '([{', clamp=True)}
    top, nested = [], []
    spans = []
    for m in _ENDING.finditer(rest):
        group = tuple(sorted(_locant_sort_item(t) for t in m.group(1).split(',')))
        (top if depth_at.get(m.start(1), 0) == 0 else nested).append(group)
        spans.append((m.start(1), m.end(1)))
    hydro_spans = [(m.start(1), m.end(1)) for m in _HYDRO_LOCANTS.finditer(rest)]
    hydro, cited = [], []
    for m in _LOCANT_TOKEN.finditer(rest):
        item = _locant_sort_item(m.group(1))
        if any(lo <= m.start(1) < hi for lo, hi in spans):
            continue
        if any(lo <= m.start(1) < hi for lo, hi in hydro_spans):
            hydro.append(item)
        else:
            cited.append(item)
    block = '(' + ''.join(sorted(sym for _loc, sym in stereo)) + ')' if stereo else ''
    skeleton = block + _LOCANT_TOKEN.sub('', rest)
    return (skeleton, indicated, tuple(top), tuple(nested), tuple(sorted(hydro)), *_ene_roles(ene_lists),
            tuple(sorted(cited)), tuple(cited), cip_locant_rank_key(stereo))


def lower_locants(a: str, b: str) -> int:
    """-1 if the name ``a`` has the lower locants by (b), (c), (e), (f), (g), then (j) than
    ``b``, 1 if ``b`` does, 0 if they tie or the two names are not the same skeleton with other
    locants (the comparison is then left to the next criterion of:func:`beats`).

    The roles are compared in the order of the rule, each as a whole before the next: the set
    of all the prefix locants (f) is compared before the locants of the first-cited prefix (g)
    ('6-bromo-4-chloro-1H-1,3-benzimidazol-3-ium', the set 4,6, over '5-bromo-7-chloro-', 5,7,
    although the first cited, bromo, has the higher locant in the first), and the
    stereodescriptors (j) last: '(4S)-4-methyl-' over '(6S)-6-methyl-3,4,5,6-tetrahydro-
    pyrimidin-1-ium' by (f), '(4R,6S)-' over '(4S,6R)-2,4,6-trimethyl-' by (j)."""
    skeleton_a, *roles_a = locant_roles(a)
    skeleton_b, *roles_b = locant_roles(b)
    if skeleton_a != skeleton_b:
        return 0
    for ra, rb in zip(roles_a, roles_b):
        if ra != rb:
            return -1 if ra < rb else 1
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# the parts a name carries that are not PIN
# ---------------------------------------------------------------------------------------------------------------------

def non_pin_parts(name: Optional[str], provenance: Optional[dict]) -> Tuple[int, int]:
    """(the constructs, the label-only parts) the run recorded as not PIN (``metrics.provenance``) that
    ``name`` contains, each counted as often as the name writes it. The constructs are the name fragments of
    ``record_non_pin_fragment``; a fragment that holds another carried one is the enclosing substituent that
    the engine records again for every level of nesting ('keeps the status of its source'), so only the
    innermost are counted. The label-only parts are the parts of ``record_non_pin_label`` and the names of
    ``record_uncertified_pin_name``. A record is name-scoped (a fragment of a candidate that was discarded
    is not in the shipped name), so this counts what the name carries. Two names that write the same number
    of non-PIN rings are a tie, whichever rings they are: a ring written twice counts twice."""
    if not isinstance(name, str) or not provenance:
        return 0, 0
    fragments = {item for item in (provenance.get("non_pin_fragments") or ()) if item and item in name}
    soft = {item for item in (*(provenance.get("non_pin_labels") or ()),
                              *(provenance.get("uncertified_pin_names") or ())) if item and item in name}
    leaves = [item for item in fragments if not any(item != other and other in item for other in fragments)]
    return sum(name.count(item) for item in leaves), sum(name.count(item) for item in soft)


def beats(challenger: str, incumbent: str, challenger_provenance: Optional[dict] = None,
          incumbent_provenance: Optional[dict] = None, challenger_stereo: Optional[int] = None,
          incumbent_stereo: Optional[int] = None) -> bool:
    """True when ``challenger`` is the better name of one species than ``incumbent``, by the order of the
    module docstring; False for a tie (the form that comes first by canonical SMILES then stands).
    ``challenger_stereo`` and ``incumbent_stereo`` are the configured stereo elements of the input that the
    form of each name keeps (:func:`orthonym.perception.charge_shift.kept_stereo`); the most decides first, and
    the criterion is skipped when either is None."""
    if challenger_stereo is not None and incumbent_stereo is not None and challenger_stereo != incumbent_stereo:
        return challenger_stereo > incumbent_stereo
    mine = non_pin_parts(challenger, challenger_provenance)
    theirs = non_pin_parts(incumbent, incumbent_provenance)
    if mine[0] != theirs[0]:
        return mine[0] < theirs[0]
    centres_mine, centres_theirs = cationic_parent_centres(challenger), cationic_parent_centres(incumbent)
    if len(centres_mine) == len(centres_theirs) and sum(centres_mine) != sum(centres_theirs):
        return sum(centres_mine) > sum(centres_theirs)
    lower = lower_locants(challenger, incumbent)
    if lower:
        return lower < 0
    return mine[1] < theirs[1]


def stereo_decides_below_pin(label: Tuple[int, int], incumbent_label: Tuple[int, int], kept: Optional[int],
                             incumbent_kept: Optional[int], pin_rank: int) -> int:
    """+1 when a name below the PIN that keeps more of the input's configured stereo elements (criterion 0 of the
    module docstring) stands over an incumbent below the PIN that keeps fewer, -1 for the reverse, else 0 and the
    labels decide. Both labels are those of a verified name below the PIN (a label rank of 2 up to ``pin_rank`` - 1,
    not the last-resort floor's): a name of another drawing that has lost the configuration of the input is the name
    of a less specific compound than the input,:44568), and such a label does not say that it is the better
    name of the species, so it does not outrank the drawing's own name that keeps the configuration."""
    if kept is None or incumbent_kept is None or kept == incumbent_kept:
        return 0
    if not (1 < label[0] < pin_rank and 1 < incumbent_label[0] < pin_rank):
        return 0
    return 1 if kept > incumbent_kept else -1
