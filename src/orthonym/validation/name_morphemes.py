"""v29 Phase 1: the independent morpheme-arity oracle.

WHY this module exists
----------------------
``binding_spine.py`` proves that a name's bindings partition the graph (P1),
claim every bond (P2) and account for every charge (P3). All three take the
producer at its word about WHICH atoms a token covers: they check that the
claims are mutually consistent and total, never that a token's *text* actually
spells the atoms it claims. A producer that hands the token ``"methyl"`` a
six-atom subtree yields a perfectly consistent spine, and P1-P3 pass it.

This module is the independent falsifier for that hole. It answers one
question from the naming data tables alone -- **"how many heavy atoms does
this token string actually spell?"** -- with no reference to the binding, the
graph, or the producer that made either. A later proof (P6) compares this
answer against ``len(binding.atom_ids)``; a disagreement means the name does
not say what the spine claims it says. Because the two numbers come from
genuinely independent sources (one from the token's morphemes, one from the
producer's atom bookkeeping), their agreement is evidence rather than
tautology.

THE SOUNDNESS CONTRACT -- the whole value of the module
-------------------------------------------------------
**A confident answer must be right.** A false-confident answer would make P6
reject a CORRECT name, which is the one outcome this milestone cannot ship.
Everything unrecognised, ambiguous, or only partially parsed returns
``ArityEstimate(None, False, "<why>")``. Coverage is grown later by
measurement, never by guessing.

Three specific refusals follow from that contract, and each is deliberate:

* **Ambiguity is refusal, not a tie-break.** The scanner enumerates EVERY
  morpheme decomposition of a token rather than committing to the greedy
  longest match, then requires them all to agree on the total. ``"tridecyl"``
  reads as ``tridec|yl`` (13) or as ``tri|dec|yl`` (10); a greedy scan would
  confidently answer one of them, so this module answers neither.
* **Position-sensitive morphemes are segregated.** ``"ol"`` is an alcohol
  suffix (1 O) in suffix position and part of a ring stem elsewhere, so it
  lives in a suffix-only table consulted only for ``kind == SUFFIX``. The
  general table carries only position-independent morphemes.
* **Whole classes are refused rather than approximated.** The lambda
  (hypervalence) convention is out of scope for Phase 1.

WHAT IT REFUSES TO GUESS
------------------------
Unknown morphemes, lambda convention, von Baeyer/spiro tokens whose bracket
descriptor is missing or does not agree with the chain stem beside it, tokens
whose multiplied group cannot be delimited, and any token admitting two
decompositions with different totals.

COMPOSITE TOKENS ARE THE COMMON CASE
------------------------------------
Every production binding producer in ``general_engine.py`` binds a
substituent's whole branch under one composed string (``"2-chloroethyl"``,
``"4-(methoxymethyl)phenyl"``). A single-morpheme oracle would answer
"unverified" for nearly every real binding and P6 would have no teeth, so
``token_arity`` scores a whole token string by summing its morphemes.

A VIEW OVER SHIPPED DATA, NOT A NEW CATALOG
--------------------------------------------
Every count below is derived from a table this project already ships, so the
oracle cannot drift away from the tables the namer itself names from:

===========================  =========================================
Morpheme class               Source
===========================  =========================================
chain stems (meth-, dec-)    ``data.chain_names.get_chain_prefix``
                             (inverted over n=1..``_MAX_CHAIN``)
suffix particles (-ol, -ic)  ``data.opsin_imports.suffix_rules`` --
                             ``OPSIN_SUFFIX_APPLICABILITY`` bridges the
                             surface morpheme to a rule, whose
                             ``addgroup`` SMILES gives the atom count
substituent prefixes         ``data.opsin_imports`` group tables
(hydroxy-, nitro-, chloro-)  (SMILES per entry)
retained/trivial names       ``data.ALL_RETAINED_NAMES`` (inverted) and
                             ``data/iupac_2013_pin_list.json``
ring systems, aryl groups    ``data.opsin_imports`` aryl/cyclic tables
ring substituents (phenyl-)  ``rules.ring_substituents`` chained to the
                             parent ring's own count
fusion prefixes (benzo-)     ``data.fusion_components`` ``ring_size``
Hantzsch-Widman stems        ``data.hw_stems.HW_STEMS`` (inverted)
replacement prefixes (aza-)  ``rules.skeletal_replacement``
multipliers (di-, bis-)      ``assembly.naming_utils`` (inverted)
===========================  =========================================

Only morphemes that NO shipped table covers are written by hand, in
``_STRUCTURAL_AFFIXES`` below, and each carries a comment saying why it is not
derivable. They are all zero-atom skeletal markers, where "contributes no
atoms" is a definition rather than a measurement.

Scope and status (Phase 1): PURE and AUDIT-ONLY. Nothing imports this yet; it
gates nothing and changes no emitted name.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# --------------------------------------------------------------------------
# Morpheme categories. The category is not cosmetic: it decides how a
# multiplier's scope is delimited (see _evaluate).
# --------------------------------------------------------------------------
STEM = "stem"        # carries skeletal atoms; can be the token's head
SUBST = "subst"      # a COMPLETE simple substituent (chloro, hydroxy, nitro)
ATTACH = "attach"    # an affix that creates an attachment point (yl, oxy)
AFFIX = "affix"      # a non-terminating affix (ane, ene, hydro, cyclo)
REPL = "repl"        # skeletal replacement prefix (aza, oxa) -- 0 net atoms
OPEN = "open"        # '(' structural marker
CLOSE = "close"      # ')' structural marker
MULT = "mult"        # multiplying prefix; `atoms` field carries multiplicity

# SUBST and ATTACH both END a simple substituent group, which is what lets an
# unenclosed multiplier's scope be delimited without guessing.
_GROUP_TERMINATORS = (SUBST, ATTACH)

# Chain stems are inverted out of get_chain_prefix() up to this length. Beyond
# it a token simply goes unrecognised (safe) rather than wrong.
_MAX_CHAIN = 60

# Search caps. A token that needs more work than this returns unconfident
# rather than risking a partial exploration being mistaken for "unambiguous".
_MAX_PARSES = 256
_MAX_STEPS = 40000
_MAX_TOKEN_LEN = 120

# Characters that carry no atoms: locants, separators, primes, enclosing marks
# handled structurally elsewhere. NOTE: whitespace is deliberately absent --
# a space is only ever legal INSIDE a registered multi-word morpheme such as
# "oic acid", so a stray space makes a token unparseable rather than ignorable.
_SKIP_CHARS = frozenset("0123456789,.-′″'’?*")

# Locant debris that precedes a morpheme and dies with its trailing hyphen:
# "2-", "N,N-", "4a-", "1'-", "N-". Applied to the ORIGINAL token, before
# lower-casing, because the italic element locants are capitalised and would
# otherwise be indistinguishable from morpheme text.
_LOCANT_RUN = re.compile(
    r"(?:^|(?<=[-(\[{,]))"
    r"[0-9NOSPCH][0-9NOSPCH,'′″a-h]*"
    r"-"
)

# A stereo descriptor in parentheses: (R), (2S,3R), (E), (1R,2S), (cis).
# Deliberately NARROW. A permissive character class here would be unsound in
# the worst direction: falsely stripping a real substituent group ("(chloro)")
# removes its atoms and yields a CONFIDENT UNDERCOUNT. So the content must
# match the descriptor grammar exactly -- optionally-located R/S/E/Z letters,
# or a whole-word cis/trans -- and anything else is left in place to be parsed
# as morphemes (or to fail loudly as unparsed residue).
_STEREO_GROUP = re.compile(
    r"\((?:"
    r"[0-9]*[RSEZrsez]\*?(?:[,;][0-9]*[RSEZrsez]\*?)*"
    r"|cis|trans|syn|anti"
    r")\)-?"
)

# Hypervalence: out of scope for Phase 1 (see module docstring).
_LAMBDA = re.compile(r"lambda|λ")

# von Baeyer / spiro heads. 'cyclo' alone is monocyclic and needs no
# descriptor; these do.
_VB_HEAD = re.compile(r"(?:^|[^a-z])(spiro|(?:bi|tri|tetra|penta|hexa)cyclo)")

# Superscript digits used for secondary-bridge locants inside a descriptor.
_SUPERSCRIPTS = str.maketrans("", "", "⁰¹²³⁴"
                                      "⁵⁶⁷⁸⁹")

# --------------------------------------------------------------------------
# Hand-written morphemes. EVERY entry here is a zero-atom skeletal marker that
# no shipped table covers, with the reason it is not derivable. Nothing that
# contributes atoms may be added here -- an atom count must come from a table.
# --------------------------------------------------------------------------
_STRUCTURAL_AFFIXES: Dict[str, Tuple[int, str]] = {
    # Saturation endings. Not derivable: OPSIN applies saturation with its
    # unsaturator (a bond-order edit), so no suffix rule and no SMILES exists
    # for them. They state bond order, never atoms -- 0 by definition. The
    # elided forms (an/en/yn) appear before a following suffix ("propan-1-ol").
    "ane": (0, "saturation ending, states bond order not atoms"),
    "an": (0, "elided saturation ending before a suffix"),
    "ene": (0, "unsaturation ending, states bond order not atoms"),
    "en": (0, "elided unsaturation ending before a suffix"),
    "yne": (0, "unsaturation ending, states bond order not atoms"),
    "yn": (0, "elided unsaturation ending before a suffix"),
    # Ring-formation marker: says the stem's atoms close a ring, adds none.
    # Not derivable: it is a topology statement, so no table gives it a SMILES.
    "cyclo": (0, "ring-closure marker, adds no atoms"),
    # Added/indicated hydrogen. Hydrogen is not a heavy atom, so 0 by the
    # heavy-atom scoping this module shares with P1-P3.
    "hydro": (0, "added hydrogen is not a heavy atom"),
}

# Surface spellings whose ATOM COUNT is derived from an OPSIN suffix rule but
# whose written form the applicability table does not carry, because OPSIN
# tokenises the linking vowel and the functional word "acid" separately. Only
# the SPELLING is stated here; every number still comes from the rule.
_ACID_WORD_SURFACES: Dict[str, str] = {
    "oic acid": "ic",              # propanoic acid -> rule 'ic'
    "ic acid": "ic",
    "oate": "ate",                 # propanoate -> rule 'ate'
    "carboxylic acid": "carboxylic",
    "carboxylate": "carboxylate",
    "sulfonic acid": "sulfonic",
}

# Cyclic ("carb-") forms: the chain form plus the carbon that OPSIN's
# 'addSuffixPrefixIfNonePresentAndCyclic' transformation adds when the parent
# is a ring. Both halves of that sum are read from the rule.
_CARB_FORM_SURFACES: Dict[str, str] = {
    "carbaldehyde": "aldehyde",
    "carbonitrile": "nitrile",
}

# OPSIN group_type whose suffix semantics are the ordinary chain/ring ones.
# The other types (acidStem, aminoAcid, carbohydrate, ...) attach the SAME
# surface morpheme with a different atom accounting -- e.g. surface 'yl' is
# the 0-atom radical suffix on a standardGroup but the 1-atom 'oyl' rule on an
# acidStem. Restricting to standardGroup is what makes the surface->count map
# single-valued instead of ambiguous.
_STANDARD_GROUP = "standardGroup"


@dataclass(frozen=True)
class ArityEstimate:
    """How many heavy atoms a token spells, or an explicit refusal.

    ``confident`` is the only field a caller may branch on for a proof:
    ``heavy_atoms`` is meaningful ONLY when ``confident`` is True, and is
    always ``None`` otherwise. ``basis`` always explains the answer -- which
    morphemes were read, or why the token was refused -- so a measurement pass
    can group refusals by cause without re-deriving them.
    """

    heavy_atoms: Optional[int]
    confident: bool
    basis: str


@dataclass(frozen=True)
class _Morph:
    """One lexicon entry: what it contributes and how it behaves."""

    atoms: int
    category: str


@dataclass(frozen=True)
class _Seg:
    """One consumed span of a token during a parse."""

    text: str
    atoms: int
    category: str


# --------------------------------------------------------------------------
# Lexicon construction. Built once, lazily, and memoised.
# --------------------------------------------------------------------------

def _heavy_atoms(smiles: str) -> Optional[int]:
    """Heavy atoms in a SMILES, ignoring any ``[*]`` attachment wildcard.

    Returns None when RDKit cannot parse it, so an unreadable table entry
    drops out of the lexicon instead of contributing a wrong count.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles.replace("[*]", "*"))
    if mol is None:
        return None
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 0)


class _Lexicon:
    """Morpheme -> contribution, with conflicts resolved by REFUSAL.

    When two sources disagree about a morpheme's atom count, the morpheme is
    REMOVED rather than arbitrated. That is the soundness contract applied to
    table construction: a morpheme the data does not agree on must never
    produce a confident answer.
    """

    def __init__(self) -> None:
        self.table: Dict[str, _Morph] = {}
        self.dropped: Set[str] = set()

    def add(self, key: str, atoms: Optional[int], category: str) -> None:
        if atoms is None or atoms < 0:
            return
        key = key.strip().lower()
        # Two chars is too short to be a distinctive morpheme; such keys match
        # everywhere and manufacture spurious decompositions.
        if len(key) < 2 or key in self.dropped:
            return
        if not all(ch.isalpha() or ch == " " for ch in key):
            return
        existing = self.table.get(key)
        if existing is None:
            self.table[key] = _Morph(atoms, category)
        elif existing.atoms != atoms:
            # Genuine disagreement between sources -> refuse the morpheme.
            del self.table[key]
            self.dropped.add(key)
        # Same count, different category: keep the first registration. The
        # registration order below runs most-specific-first, so the structural
        # reading of a morpheme wins over an incidental catalog entry.


def _add_suffix_morphemes(general: _Lexicon, suffix_only: _Lexicon) -> None:
    """Suffix particles, bridged surface -> rule -> SMILES.

    ``OPSIN_SUFFIX_APPLICABILITY`` rows carry the surface morpheme in
    ``suffix_value`` and the rule name in ``suffix_text`` (the importer's
    field names follow the XML attribute/'text' split, not the semantics).
    A rule's ``addgroup`` SMILES is what the suffix ADDS; the separate
    ``addSuffixPrefixIfNonePresentAndCyclic`` carbon is added only for a ring
    parent, which is exactly the chain/ring split that makes ``-oic acid``
    spell 2 atoms while ``-carboxylic acid`` spells 3.

    An ATTACH suffix is one whose rule creates an attachment point
    (``setOutAtom`` or ``outIDs``) -- that is read off the rule, not assumed.
    Those go in the GENERAL table because they legitimately occur inside a
    substituent prefix token ("methoxymethyl"). The rest are principal
    characteristic group suffixes and are position-sensitive, so they go in
    the suffix-only table.
    """
    from orthonym.data.opsin_imports.suffix_rules import (
        OPSIN_SUFFIX_APPLICABILITY, OPSIN_SUFFIX_RULES)

    rules = {r["value"]: r.get("transformations", []) for r in OPSIN_SUFFIX_RULES}

    def score(rule: str) -> Optional[Tuple[int, int, bool]]:
        """(chain atoms, extra atoms when cyclic, creates attachment)."""
        transformations = rules.get(rule)
        if transformations is None:
            return None
        chain = 0
        cyclic_extra = 0
        attaches = False
        for step in transformations:
            tag = step.get("tag")
            if tag == "setOutAtom" or step.get("outIDs"):
                attaches = True
            smiles = step.get("SMILES")
            if not smiles:
                continue
            count = _heavy_atoms(smiles)
            if count is None:
                return None
            if tag == "addSuffixPrefixIfNonePresentAndCyclic":
                cyclic_extra += count
            elif tag == "addgroup":
                chain += count
        return chain, cyclic_extra, attaches

    # surface -> the standardGroup rules it can mean
    surfaces: Dict[str, Set[str]] = {}
    for row in OPSIN_SUFFIX_APPLICABILITY:
        if row.get("group_type") != _STANDARD_GROUP:
            continue
        surface = (row.get("suffix_value") or "").strip().lower()
        rule = (row.get("suffix_text") or "").strip()
        if surface and rule:
            surfaces.setdefault(surface, set()).add(rule)

    for surface, candidate_rules in surfaces.items():
        scored = [score(r) for r in sorted(candidate_rules)]
        if any(s is None for s in scored):
            continue
        chains = {s[0] for s in scored}
        if len(chains) != 1:
            # The same surface means different atom counts even within
            # standardGroup -> refuse it.
            continue
        chain_atoms = chains.pop()
        attaches = any(s[2] for s in scored)
        target = general if attaches else suffix_only
        target.add(surface, chain_atoms, ATTACH if attaches else AFFIX)

    # Surface spellings the applicability table does not carry, counts still
    # derived from the rules above.
    for surface, rule in _ACID_WORD_SURFACES.items():
        scored = score(rule)
        if scored is not None:
            suffix_only.add(surface, scored[0], AFFIX)
    for surface, rule in _CARB_FORM_SURFACES.items():
        scored = score(rule)
        if scored is not None:
            # chain form + the carbon the rule adds for a ring parent
            suffix_only.add(surface, scored[0] + scored[1], AFFIX)


def _add_group_tables(general: _Lexicon) -> None:
    """Substituent prefixes, aryl/cyclic ring names and simple groups.

    ``OPSIN_SUBSTITUENT_NAMES`` is registered FIRST and as SUBST, because its
    entries are complete simple substituents (hydroxy, nitro, chloro) and that
    is what lets an unenclosed multiplier's scope be delimited. The remaining
    tables are ring/skeleton names and register as STEM.
    """
    import importlib

    ordered = [
        ("substituent_names_opsin", "OPSIN_SUBSTITUENT_NAMES", SUBST),
        ("simple_groups", "OPSIN_SIMPLE_GROUPS", STEM),
        ("aryl_groups", "OPSIN_ARYL_GROUPS", STEM),
        ("cyclic_groups", "OPSIN_CYCLIC_GROUPS", STEM),
    ]
    for module_name, symbol, category in ordered:
        try:
            table = getattr(
                importlib.import_module(
                    f"orthonym.data.opsin_imports.{module_name}"), symbol)
        except (ImportError, AttributeError):
            continue
        for key, record in table.items():
            # '||' keys encode OPSIN parser-side transformations: the SMILES
            # before the marker does not describe the named molecule.
            if "||" in key:
                continue
            smiles = record.get("smiles", key)
            if "||" in smiles:
                continue
            count = _heavy_atoms(smiles)
            if count is None:
                continue
            for name in record.get("names", ()):
                general.add(name, count, category)


def _add_retained_names(general: _Lexicon) -> None:
    """Retained/trivial names and the shipped PIN list.

    ``ALL_RETAINED_NAMES`` is keyed by SMILES, so it is inverted here; the
    heavy-atom count comes from the key it was inverted out of. Only
    single-word alphabetic names are usable as morphemes -- a name carrying
    locants or stereo descriptors is not a morpheme, it is a whole name.
    """
    from orthonym.data import ALL_RETAINED_NAMES

    for smiles, name in ALL_RETAINED_NAMES.items():
        name = name.strip().lower()
        if not name.isalpha() or len(name) < 4:
            continue
        general.add(name, _heavy_atoms(smiles), STEM)

    # data/iupac_2013_pin_list.json ships name+smiles side by side, but the
    # data package loader keeps only the name allow/deny sets, so the SMILES
    # are read directly here. Read-only.
    try:
        import orthonym.data as data_pkg

        path = Path(data_pkg.__file__).parent / "iupac_2013_pin_list.json"
        with open(path) as handle:
            entries = json.load(handle).get("entries", [])
    except (OSError, ValueError, AttributeError):
        entries = []
    for entry in entries:
        name = (entry.get("name") or "").strip().lower()
        smiles = entry.get("smiles")
        if name and smiles and name.isalpha() and len(name) >= 4:
            general.add(name, _heavy_atoms(smiles), STEM)


def _add_ring_derived(general: _Lexicon) -> None:
    """Ring substituent names and fusion prefixes, chained to their ring.

    ``phenyl`` has no SMILES anywhere in the project, but
    ``RING_SUBSTITUENT_NAMES`` states that it is benzene's substituent form,
    and benzene's own count is already in the lexicon. A monovalent
    substituent prefix spells the same skeleton as its parent ring, so the
    count carries across unchanged -- derived, not asserted.
    """
    try:
        from orthonym.rules.ring_substituents import RING_SUBSTITUENT_NAMES
    except ImportError:
        RING_SUBSTITUENT_NAMES = {}
    for ring_name, substituent_name in RING_SUBSTITUENT_NAMES.items():
        parent = general.table.get(ring_name.strip().lower())
        if parent is not None:
            general.add(substituent_name, parent.atoms, STEM)

    # Fusion prefixes: a mancude monocyclic component's heavy-atom count is
    # its ring size (every ring position is one heavy atom).
    try:
        from orthonym.data.fusion_components import MONOCYCLIC_COMPONENTS
    except ImportError:
        MONOCYCLIC_COMPONENTS = {}
    for component in MONOCYCLIC_COMPONENTS.values():
        prefix = component.get("prefix")
        ring_size = component.get("ring_size")
        if prefix and isinstance(ring_size, int):
            general.add(prefix, ring_size, STEM)


def _add_chain_stems(general: _Lexicon) -> None:
    """Chain stems, inverted out of the generator that produces them.

    Hand-writing this map would miss the compositional rules above 20
    (henicos-, dotriacont-), so it is built by evaluating the shipped
    generator and inverting the result.
    """
    from orthonym.data.chain_names import get_chain_prefix

    for n in range(1, _MAX_CHAIN + 1):
        try:
            stem = get_chain_prefix(n)
        except Exception:
            continue
        if stem:
            general.add(stem, n, STEM)


def _add_replacement_prefixes(general: _Lexicon) -> None:
    """Skeletal replacement ('a') prefixes contribute ZERO net atoms.

    ``aza`` turns a skeletal carbon into a nitrogen; it substitutes an atom
    rather than adding one, so the stem's own count already includes it. The
    elided forms (az-, ox-, thi-) appear before a vowel.
    """
    try:
        from orthonym.rules.skeletal_replacement import REPLACEMENT_TERMS
    except ImportError:
        REPLACEMENT_TERMS = {}
    for prefix in REPLACEMENT_TERMS.values():
        general.add(prefix, 0, REPL)
        if prefix.endswith("a"):
            general.add(prefix[:-1], 0, REPL)


@lru_cache(maxsize=1)
def _hw_stems() -> Dict[str, int]:
    """Hantzsch-Widman stem -> ring size, inverted from HW_STEMS.

    Kept OUT of the general lexicon and consulted only immediately after a
    replacement prefix. Without that gate the HW reading of ``ane`` (a
    six-membered ring, 6 atoms) would collide with the saturation ending
    ``ane`` (0 atoms) and make every ``cyclohexane`` ambiguous. With it,
    ``oxane`` reads ox+ane = 6 and ``cyclohexane`` reads cyclo+hex+ane = 6.
    A stem claimed by two different ring sizes is dropped.
    """
    try:
        from orthonym.data.hw_stems import HW_STEMS
    except ImportError:
        return {}
    sizes: Dict[str, Set[int]] = {}
    for ring_size, variants in HW_STEMS.items():
        if not isinstance(ring_size, int):
            continue
        for stem in variants.values():
            if not isinstance(stem, str) or not stem:
                continue
            stem = stem.strip().lower()
            sizes.setdefault(stem, set()).add(ring_size)
            # ELIDED form. A HW stem drops its terminal 'e' before a following
            # suffix: "oxane" but "oxan-4-yl". Registering only the full form
            # made 'oxan' parse uniquely as ox+an (replacement + saturation
            # ending) = 0 atoms and answer CONFIDENTLY 0 for a 6-atom ring --
            # found by auditing real production tokens. With the elided form
            # present the token reads 0 or 6, the two disagree, and it is
            # refused instead of being confidently wrong.
            if stem.endswith("e") and len(stem) > 2:
                sizes.setdefault(stem[:-1], set()).add(ring_size)
    return {stem: next(iter(s)) for stem, s in sizes.items() if len(s) == 1}


@lru_cache(maxsize=1)
def _multipliers() -> Dict[str, int]:
    """Multiplying prefixes, inverted from the tables the namer emits with."""
    result: Dict[str, int] = {}
    try:
        from orthonym.assembly.naming_utils import (COMPLEX_MULTIPLIERS,
                                                     SIMPLE_MULTIPLIERS)
    except ImportError:
        return result
    for count, word in SIMPLE_MULTIPLIERS.items():
        result[word.lower()] = count
    for count, word in COMPLEX_MULTIPLIERS.items():
        result[word.lower()] = count
    return result


@lru_cache(maxsize=2)
def _lexicon(include_suffixes: bool) -> Dict[str, _Morph]:
    """The morpheme table for one binding kind.

    Registration order is most-specific-first so that a structural reading of
    a morpheme wins the CATEGORY over an incidental catalog entry with the
    same count. Disagreements on the COUNT drop the morpheme either way.
    """
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    try:
        general = _Lexicon()
        suffix_only = _Lexicon()

        for key, (atoms, _why) in _STRUCTURAL_AFFIXES.items():
            general.add(key, atoms, AFFIX)
        _add_suffix_morphemes(general, suffix_only)
        _add_replacement_prefixes(general)
        _add_chain_stems(general)
        _add_group_tables(general)
        _add_retained_names(general)
        _add_ring_derived(general)

        table = dict(general.table)
        if include_suffixes:
            for key, morph in suffix_only.table.items():
                # A suffix reading may not silently overrule a general
                # morpheme of the same spelling; conflicts are refused.
                existing = table.get(key)
                if existing is None:
                    table[key] = morph
                elif existing.atoms != morph.atoms:
                    del table[key]
        return table
    finally:
        RDLogger.EnableLog("rdApp.*")


@lru_cache(maxsize=2)
def _by_first_char(include_suffixes: bool) -> Dict[str, List[str]]:
    """Morpheme keys bucketed by first character, longest first."""
    buckets: Dict[str, List[str]] = {}
    for key in _lexicon(include_suffixes):
        buckets.setdefault(key[0], []).append(key)
    for keys in buckets.values():
        keys.sort(key=len, reverse=True)
    return buckets


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------

def _normalise(token: str) -> str:
    """Strip stereo descriptors, locant runs and enclosing marks; lowercase.

    Locant stripping runs BEFORE lower-casing because italic element locants
    (the N of "N-methyl") are capitalised, and after lower-casing they would
    be indistinguishable from morpheme text.
    """
    text = token.strip()
    while len(text) >= 2 and text[0] in "([{" and text[-1] in ")]}":
        inner = text[1:-1]
        if inner.count("(") == inner.count(")"):
            text = inner.strip()
        else:
            break
    text = _STEREO_GROUP.sub("", text)
    text = _LOCANT_RUN.sub("", text)
    return text.lower().strip()


def _von_baeyer(text: str) -> Optional[Tuple[int, str, str]]:
    """Resolve a von Baeyer/spiro head into (skeletal atoms, rest, basis).

    Returns None when the token has no such head. Raises nothing: an
    unusable descriptor comes back as a refusal through ``_VonBaeyerRefusal``.

    The descriptor and the chain stem beside it state the SAME ring-atom
    count two different ways -- ``bicyclo[2.2.1]heptane`` says 2+2+1+2 = 7 and
    then says "hept". Requiring them to agree is what makes this path safe:
    a confident answer needs two independent statements to coincide, and any
    disagreement is a refusal.
    """
    match = _VB_HEAD.search(text)
    if match is None:
        return None
    head = match.group(1)
    after_head = text[match.end(1):]
    if not after_head.startswith("["):
        return _VB_REFUSAL(f"{head} without a bracket descriptor")
    close = after_head.find("]")
    if close < 0:
        return _VB_REFUSAL(f"{head} descriptor bracket never closes")
    content = after_head[1:close].translate(_SUPERSCRIPTS)
    # Secondary bridges carry superscript locants; once those are gone the
    # descriptor must be pure numbers separated by dots. Anything else (a
    # ring-assembly spiro name like spiro[cyclohexane-1,1'-indene]) is not
    # descriptor arithmetic and is refused.
    if not re.fullmatch(r"\d+(?:\.\d+)+", content):
        return _VB_REFUSAL(f"{head} descriptor {content!r} is not numeric")
    numbers = [int(n) for n in content.split(".")]
    # One shared atom for a spiro union, two bridgeheads for a von Baeyer
    # polycycle (every secondary bridge hangs off the same two).
    skeletal = sum(numbers) + (1 if head == "spiro" else 2)

    rest = after_head[close + 1:]
    stem_match = _longest_stem(rest)
    if stem_match is None:
        return _VB_REFUSAL(f"{head} descriptor not followed by a chain stem")
    stem, stem_atoms = stem_match
    if stem_atoms != skeletal:
        return _VB_REFUSAL(
            f"{head} descriptor says {skeletal} skeletal atoms but stem "
            f"{stem!r} says {stem_atoms}")
    prefix = text[:match.start(1)]
    return skeletal, prefix + rest[len(stem):], (
        f"{head}{content} = {skeletal} skeletal atoms, confirmed by stem {stem!r}")


class _VonBaeyerRefusal(Exception):
    """Carries the reason a von Baeyer/spiro token cannot be scored."""


def _VB_REFUSAL(reason: str):
    raise _VonBaeyerRefusal(reason)


def _longest_stem(text: str) -> Optional[Tuple[str, int]]:
    """Longest chain stem at position 0 of ``text``, with its length."""
    table = _lexicon(False)
    for size in range(min(len(text), 20), 1, -1):
        candidate = text[:size]
        morph = table.get(candidate)
        if morph is not None and morph.category == STEM:
            return candidate, morph.atoms
    return None


# --------------------------------------------------------------------------
# Parsing: enumerate EVERY decomposition, then require agreement.
# --------------------------------------------------------------------------

def _parse_all(text: str, include_suffixes: bool
               ) -> Tuple[List[List[_Seg]], int, bool]:
    """All morpheme tilings of ``text``.

    Returns (parses, furthest position reached, hit_cap). Enumerating every
    tiling rather than taking the greedy longest match is what lets ambiguity
    be DETECTED instead of silently resolved.
    """
    table = _lexicon(include_suffixes)
    buckets = _by_first_char(include_suffixes)
    multipliers = _multipliers()
    hw = _hw_stems()

    parses: List[List[_Seg]] = []
    furthest = 0
    steps = 0
    hit_cap = False

    def walk(pos: int, acc: List[_Seg]) -> None:
        nonlocal furthest, steps, hit_cap
        if hit_cap:
            return
        steps += 1
        if steps > _MAX_STEPS or len(parses) >= _MAX_PARSES:
            hit_cap = True
            return
        if pos > furthest:
            furthest = pos
        if pos >= len(text):
            parses.append(list(acc))
            return

        char = text[pos]
        if char in _SKIP_CHARS:
            walk(pos + 1, acc)
            return
        if char in "([{":
            acc.append(_Seg(char, 0, OPEN))
            walk(pos + 1, acc)
            acc.pop()
            return
        if char in ")]}":
            acc.append(_Seg(char, 0, CLOSE))
            walk(pos + 1, acc)
            acc.pop()
            return

        # Hantzsch-Widman stems are legal ONLY straight after a replacement
        # prefix (see _hw_stems).
        previous = acc[-1] if acc else None
        if previous is not None and previous.category == REPL:
            for stem, ring_size in hw.items():
                if text.startswith(stem, pos):
                    acc.append(_Seg(stem, ring_size, STEM))
                    walk(pos + len(stem), acc)
                    acc.pop()

        for key in buckets.get(char, ()):
            if text.startswith(key, pos):
                morph = table[key]
                acc.append(_Seg(key, morph.atoms, morph.category))
                walk(pos + len(key), acc)
                acc.pop()

        for word, count in multipliers.items():
            if word[0] == char and text.startswith(word, pos):
                acc.append(_Seg(word, count, MULT))
                walk(pos + len(word), acc)
                acc.pop()

    walk(0, [])
    return parses, furthest, hit_cap


def _evaluate(segments: List[_Seg]) -> Optional[int]:
    """Total heavy atoms for one parse, or None if its multipliers are unclear.

    Multiplier scoping follows the IUPAC distinction the tokens themselves
    observe: ``di-``/``tri-`` multiply a SIMPLE substituent and are written
    unenclosed, while ``bis-``/``tris-`` multiply a COMPLEX one and are always
    followed by enclosing marks. So an enclosed group's extent is read from the
    brackets, and an unenclosed one ends at the first morpheme that completes a
    simple substituent (an attachment affix such as ``-yl``/``-oxy``, or a
    complete substituent prefix such as ``chloro``).

    A leading multiplier whose group is the WHOLE token is a statement about
    how many times this substituent occurs, not about how big it is: the token
    ``"dimethyl"`` spells one methyl. That multiplicity belongs to the
    multiplicity proof, not to arity, so it is ignored here. A multiplier with
    a skeletal head after it ("dimethylphenyl") genuinely multiplies an inner
    group and IS applied.
    """
    total = 0
    index = 0
    count = len(segments)

    while index < count:
        segment = segments[index]
        if segment.category != MULT:
            total += segment.atoms
            index += 1
            continue

        multiplicity = segment.atoms
        index += 1
        if index >= count:
            return None  # trailing multiplier governs nothing

        following = segments[index]

        # A zero-atom skeletal marker ("dihydro", "tetrahydro"): whatever the
        # multiplier's true scope, it multiplies nothing, so the reading is
        # unambiguous regardless.
        if following.category == AFFIX and following.atoms == 0:
            index += 1
            continue

        if following.category == OPEN:
            depth = 0
            group_atoms = 0
            end = index
            while end < count:
                current = segments[end]
                if current.category == OPEN:
                    depth += 1
                elif current.category == CLOSE:
                    depth -= 1
                    if depth == 0:
                        break
                elif current.category == MULT:
                    return None  # nested multiplier: scope not resolvable here
                else:
                    group_atoms += current.atoms
                end += 1
            if depth != 0:
                return None  # unbalanced enclosing marks
            index = end + 1
        else:
            group_atoms = 0
            end = index
            terminated = False
            while end < count:
                current = segments[end]
                if current.category in (MULT, OPEN, CLOSE):
                    break
                group_atoms += current.atoms
                end += 1
                if current.category in _GROUP_TERMINATORS:
                    terminated = True
                    break
            if not terminated:
                return None  # cannot delimit the multiplied group
            index = end

        # Does anything at all follow the multiplied group? If not, the group
        # IS the token and the multiplier counts how many times that whole
        # substituent occurs -- multiplicity, not arity. If something follows,
        # the multiplied group is an inner part of a larger token and the
        # multiplier genuinely applies. What follows need not be skeletal:
        # "dimethylamino" is 2 methyls on a nitrogen (3), so testing for a
        # STEM specifically would under-count every N,N-dialkyl token.
        tail = [s for s in segments[index:] if s.category not in (OPEN, CLOSE)]
        total += group_atoms * multiplicity if tail else group_atoms

    return total


@lru_cache(maxsize=4096)
def token_arity(token: str, kind: "object" = "prefix") -> ArityEstimate:
    """How many heavy atoms does ``token`` spell?

    ``kind`` may be a :class:`~orthonym.validation.binding_spine.BindingKind`
    or the equivalent plain string. It selects the lexicon: ``SUFFIX`` adds the
    position-sensitive suffix morphemes (``-ol`` is 1 oxygen in suffix position
    and nothing of the sort elsewhere), every other kind uses the general
    table only.

    A confident answer is guaranteed correct; an unconfident one carries a
    ``basis`` explaining the refusal. See the module docstring.
    """
    if not isinstance(token, str) or not token.strip():
        return ArityEstimate(None, False, "empty token")
    if len(token) > _MAX_TOKEN_LEN:
        return ArityEstimate(None, False,
                             f"token longer than {_MAX_TOKEN_LEN} characters")

    kind_name = str(getattr(kind, "value", kind)).strip().lower()
    include_suffixes = kind_name == "suffix"

    text = _normalise(token)
    if not text:
        return ArityEstimate(None, False, "token is only locants/punctuation")
    if _LAMBDA.search(text):
        return ArityEstimate(None, False, "lambda convention")

    base = 0
    basis_prefix = ""
    try:
        resolved = _von_baeyer(text)
    except _VonBaeyerRefusal as refusal:
        return ArityEstimate(None, False, str(refusal))
    if resolved is not None:
        base, text, basis_prefix = resolved
        basis_prefix += "; "
        if not text.strip():
            return ArityEstimate(base, True, basis_prefix.rstrip("; "))

    parses, furthest, hit_cap = _parse_all(text, include_suffixes)
    if hit_cap:
        return ArityEstimate(None, False,
                             "too many decompositions to decide")
    if not parses:
        return ArityEstimate(
            None, False,
            f"{basis_prefix}unparsed residue {text[furthest:]!r}")

    totals: Set[int] = set()
    for parse in parses:
        value = _evaluate(parse)
        if value is None:
            return ArityEstimate(
                None, False,
                f"{basis_prefix}multiplier scope not resolvable in "
                f"{'+'.join(s.text for s in parse)!r}")
        totals.add(value)

    if len(totals) != 1:
        readings = sorted(totals)
        return ArityEstimate(
            None, False,
            f"{basis_prefix}ambiguous: decompositions disagree {readings}")

    shortest = min(parses, key=len)
    return ArityEstimate(
        base + totals.pop(), True,
        basis_prefix + "+".join(s.text for s in shortest
                                if s.category not in (OPEN, CLOSE)))


def lexicon_size(include_suffixes: bool = False) -> int:
    """Number of morphemes the oracle knows. For measurement passes."""
    return len(_lexicon(include_suffixes))


# ---------------------------------------------------------------------------
# P-29.2 free-valence morphology (Phase 1b)
# ---------------------------------------------------------------------------
#
# A second, much smaller text oracle with the same contract as ``token_arity``:
# it reads a prefix token's ENDING and reports how many free valences that text
# asserts. IUPAC 2013 P-29.2:
#
#     -yl       one free valence
#     -ylidene  two on the same skeletal atom
#     -ylidyne  three on the same skeletal atom
#
# It lives here, next to ``token_arity``, because it is pure text analysis with
# no knowledge of any graph, and because it is SHARED: the substituent producer
# uses it to refuse emitting a single-valence token for a multi-order
# attachment, and the spine's P7 uses it to refuse certifying one. Deriving it
# once is what keeps those two from drifting apart.
#
# Refusals dominate by design. Most prefixes (``oxo``, ``hydroxy``, ``chloro``)
# spell no free-valence morpheme at all, and a multiplied ending (``-diyl``,
# ``-triyl``) distributes its valences over several atoms, where the
# single-attachment reading does not apply. Both are refused rather than
# guessed: a confident wrong answer would suppress a correct name on the
# producer side and certify an incorrect one on the proof side.

#: Endings that settle the count. They are mutually exclusive as suffixes --
#: ``-ylidene`` and ``-ylidyne`` do not themselves end in ``yl`` -- so the
#: order below is for reading, not for correctness.
_FREE_VALENCE_ENDINGS: Tuple[Tuple[str, int], ...] = (
    ("ylidyne", 3),
    ("ylidene", 2),
    ("yl", 1),
)

#: A multiplier immediately in front of the ending means several free valences
#: on several atoms (``ethane-1,2-diyl``), which this oracle does not judge.
_MULTIPLIED_FREE_VALENCE = re.compile(
    r"(?:di|tri|tetra|penta|hexa|hepta|octa|nona|deca|kis)"
    r"(?:ylidyne|ylidene|yl)$")


@dataclass(frozen=True)
class FreeValenceEstimate:
    """How many free valences a token's text asserts, or an explicit refusal.

    Mirrors :class:`ArityEstimate`: ``confident`` is the only field a caller
    may branch on, ``free_valences`` is meaningful ONLY when it is True and is
    ``None`` otherwise, and ``basis`` always explains the answer.
    """

    free_valences: Optional[int]
    confident: bool
    basis: str


def free_valence_morphology(token: object) -> FreeValenceEstimate:
    """Read the P-29.2 free-valence count that ``token``'s own text asserts."""
    if not isinstance(token, str):
        return FreeValenceEstimate(None, False, "not a string")
    text = token.strip().lower()
    if not text:
        return FreeValenceEstimate(None, False, "empty token")
    if _MULTIPLIED_FREE_VALENCE.search(text):
        return FreeValenceEstimate(
            None, False,
            "multiplied free-valence ending: several valences on several "
            "atoms, which a single attachment bond cannot decide")
    for ending, count in _FREE_VALENCE_ENDINGS:
        if text.endswith(ending):
            return FreeValenceEstimate(count, True, f"-{ending} (P-29.2)")
    return FreeValenceEstimate(
        None, False, "token spells no P-29.2 free-valence ending")
