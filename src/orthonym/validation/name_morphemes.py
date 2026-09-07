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

Four specific refusals follow from that contract, and each is deliberate:

* **Ambiguity is refusal, not a tie-break.** The scanner enumerates EVERY
  morpheme decomposition of a token rather than committing to the greedy
  longest match, then requires them all to agree on the total. ``"tridecyl"``
  reads as ``tridec|yl`` (13) or as ``tri|dec|yl`` (10); a greedy scan would
  confidently answer one of them, so this module answers neither.
* **A morphotactically impossible reading refuses the WHOLE token** -- see
  "WHY AGREEMENT IS NOT ENOUGH" below. This is the repair of the soundness
  argument itself, not an extra heuristic.
* **Position-sensitive morphemes are flagged, not hidden.** ``"ol"`` is an
  alcohol suffix (1 O) in suffix position and part of a ring stem elsewhere.
  Both readings are always enumerated and the suffix reading carries a
  ``suffix_only`` flag that the positional rules police; the *lexicon* is
  never narrowed by ``kind``, because narrowing it can only manufacture false
  confidence (again, see below).
* **Whole classes are refused rather than approximated.** The lambda
  (hypervalence) convention and fusion nomenclature are out of scope: fused
  components SHARE their fusion atoms, so a fusion prefix's atoms cannot be
  summed with the base component's at all.

WHY AGREEMENT IS NOT ENOUGH (the v29 confident-wrong defect)
------------------------------------------------------------
"Enumerate every decomposition and require agreement" is sound only if the
correct decomposition is among those enumerated. When a morpheme is MISSING
from the lexicon, a token can still tile -- uniquely -- into a reading that is
structurally impossible, and unanimity over a set of one then certifies it.
That is not a hypothetical: ``benzoyl`` had no ``benz`` acyl stem, so its only
tiling was the *fusion prefix* ``benzo`` plus ``yl``, and the oracle answered a
CONFIDENT 6 for an 8-atom acyl group. ``diazenyl`` had no ``diazen`` parent
hydride, so its only tiling was ``di|az|en|yl`` -- a multiplier, a skeletal
replacement prefix qualifying nothing, and two bond-order endings -- and the
oracle answered a CONFIDENT 0 for two nitrogens. Both would make P6 reject a
Blue Book PIN.

The repair is ``_well_formed``: a small set of morphotactic rules, each read
off an IUPAC construction rule, that say when a tiling cannot describe any
molecule (a replacement prefix that qualifies no skeleton, a suffix that
attaches to no parent hydride, two skeletons juxtaposed with no attachment
affix between them, a fusion prefix, a multiplier governing nothing). A
violation is evidence that the scanner matched across a boundary the lexicon
cannot see -- i.e. that the lexicon is INCOMPLETE FOR THIS TOKEN -- and once
that is known, "all readings agree" no longer implies "the reading is right".
So a violation refuses the whole token rather than pruning the offending
reading: pruning would leave exactly the surviving-but-wrong reading that
caused the defect.

WHAT IT REFUSES TO GUESS
------------------------
Unknown morphemes, lambda convention, fusion prefixes, von Baeyer/spiro tokens
whose bracket descriptor is missing or does not agree with the chain stem
beside it, tokens whose multiplied group cannot be delimited, any token
admitting two decompositions with different totals, and any token admitting a
morphotactically impossible decomposition.

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
elided stems (thiazol-)      terminal-'e' elision (P-16.7.1(a)) of the
                             skeletons the tables above agreed on
===========================  =========================================

Only morphemes that NO shipped table covers are written by hand, in
``_STRUCTURAL_AFFIXES`` below, and each carries a comment saying why it is not
derivable. They are all zero-atom skeletal markers, where "contributes no
atoms" is a definition rather than a measurement.

Being a view over shipped data also means a table entry whose SMILES does not
describe the fragment its name spells would propagate straight through. Four
derivation screens reject exactly that shape, each pointing at a contradiction
inside the entry rather than second-guessing either half of it: a disconnected
SMILES and an all-hydrogen SMILES (both in ``_heavy_atoms``), and a name
asserting an isotope or a Hantzsch-Widman ring the SMILES does not carry (both
in ``_entry_atoms``).

Scope and status: PURE and side-effect-free. ``token_arity`` gates nothing and
changes no emitted name -- P6 consumes it in audit mode only. (The separate
``free_valence_morphology`` at the foot of the module IS consulted by the
substituent producers; ``token_arity`` is not.)
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
FUSE = "fuse"        # attached-component fusion prefix (benzo, pyrido)
OPEN = "open"        # '(' structural marker
CLOSE = "close"      # ')' structural marker
MULT = "mult"        # multiplying prefix; `atoms` field carries multiplicity

# SUBST and ATTACH both END a simple substituent group, which is what lets an
# unenclosed multiplier's scope be delimited without guessing. A suffix
# morpheme ends one too: "hexanedioic acid" multiplies '-oic acid', and the
# suffix is the last thing in the token by construction (see _well_formed R3).
_GROUP_TERMINATORS = (SUBST, ATTACH)

# Roles inside _STRUCTURAL_AFFIXES. The split is load-bearing for _well_formed
# R4: a marker that PRECEDES the skeleton it qualifies ("cyclohexane",
# "dihydronaphthalene") may legally sit in front of a stem, whereas a bond-order
# ENDING closes a stem and nothing skeletal may follow it without an
# intervening attachment affix.
ENDING = "ending"
PREMARK = "premark"

# Chain stems are inverted out of get_chain_prefix() up to this length. Beyond
# it a token simply goes unrecognised (safe) rather than wrong.
_MAX_CHAIN = 60

# Minimum length of an ELIDED stem spelling. A shorter one is not distinctive:
# it matches inside unrelated words and manufactures readings the lexicon cannot
# contradict. This is the same discipline _Lexicon.add applies to every other
# morpheme, applied to the two elision paths that bypass it.
_MIN_ELIDED = 4

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
_STRUCTURAL_AFFIXES: Dict[str, Tuple[int, str, str]] = {
    # Saturation endings. Not derivable: OPSIN applies saturation with its
    # unsaturator (a bond-order edit), so no suffix rule and no SMILES exists
    # for them. They state bond order, never atoms -- 0 by definition. The
    # elided forms (an/en/yn) appear before a following suffix ("propan-1-ol").
    "ane": (0, ENDING, "saturation ending, states bond order not atoms"),
    "an": (0, ENDING, "elided saturation ending before a suffix"),
    "ene": (0, ENDING, "unsaturation ending, states bond order not atoms"),
    "en": (0, ENDING, "elided unsaturation ending before a suffix"),
    "yne": (0, ENDING, "unsaturation ending, states bond order not atoms"),
    "yn": (0, ENDING, "elided unsaturation ending before a suffix"),
    # Ring-formation marker: says the stem's atoms close a ring, adds none.
    # Not derivable: it is a topology statement, so no table gives it a SMILES.
    "cyclo": (0, PREMARK, "ring-closure marker, adds no atoms"),
    # Added/indicated hydrogen. Hydrogen is not a heavy atom, so 0 by the
    # heavy-atom scoping this module shares with P1-P3.
    "hydro": (0, PREMARK, "added hydrogen is not a heavy atom"),
}

#: The subset of the above that may legally stand in front of a skeleton.
_PREMARKS = frozenset(
    key for key, (_atoms, role, _why) in _STRUCTURAL_AFFIXES.items()
    if role == PREMARK)

#: Hydrogen-isotope morphemes. A SMILES with no isotope label cannot state the
#: atom count of a name that asserts one, because RDKit merges unlabelled
#: hydrogens into the implicit count while it KEEPS labelled ones as atoms --
#: so 'borodeuteride' paired with '[BH4-]' loses exactly the four atoms the
#: name spells. Named as a class (P-82 isotopic modification), not per token.
_ISOTOPE_MORPHEMES = re.compile(r"deuter|triti|proti")

#: HW stems distinctive enough to be a reliable ring assertion in a name's
#: ENDING. The short ones ('ane', 'ene', 'ine', 'ole', 'ete') double as chain
#: endings and name fragments ('glycine', 'butane'), so they are excluded: this
#: screen must fire on a contradiction, never on an ordinary acyclic name.
_HW_RING_ENDINGS = frozenset({
    "irane", "irene", "irine", "iridine", "etane", "etidine", "olane",
    "olidine", "inane", "inine", "epine", "epane", "ocine", "ocane",
    "onine", "onane", "ecine", "ecane",
})

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
    """One lexicon entry: what it contributes and how it behaves.

    ``suffix_only`` marks a principal-characteristic-group suffix particle
    (``-ol``, ``-oic acid``, ``-amide``). It is a FLAG rather than a separate
    table because a reading that is removed from the enumeration can no longer
    contradict a wrong one -- see "WHY AGREEMENT IS NOT ENOUGH".
    """

    atoms: int
    category: str
    suffix_only: bool = False
    hydride: bool = False
    chain: bool = False
    chain_only: bool = False


@dataclass(frozen=True)
class _Seg:
    """One consumed span of a token during a parse."""

    text: str
    atoms: int
    category: str
    suffix_only: bool = False
    hydride: bool = False
    chain: bool = False
    chain_only: bool = False


# --------------------------------------------------------------------------
# Lexicon construction. Built once, lazily, and memoised.
# --------------------------------------------------------------------------

def _heavy_atoms(smiles: str) -> Optional[int]:
    """Heavy atoms in a SMILES, ignoring any ``[*]`` attachment wildcard.

    The count is deliberately RDKit's own surviving-atom count, because that is
    what a producer's ``binding.atom_ids`` is counted against: RDKit merges
    unlabelled hydrogens into the implicit count and keeps everything else, so
    counting what it kept is what makes the two numbers comparable.

    Returns None -- dropping the entry out of the lexicon rather than letting it
    contribute a wrong count -- when RDKit cannot parse the SMILES, and for two
    shapes that cannot state ONE morpheme's atom count at all:

    * **more than one fragment.** A ``.`` means the entry describes several
      disconnected species; their total is not the arity of a single morpheme.
      (``inosinylyl`` shipped as ``O=PO.OC[C@H]1...`` and summed to 22 for a
      19-atom group.)
    * **nothing but hydrogen.** A purely hydrogen fragment has no heavy atoms;
      its apparent count of 1 is only RDKit declining to merge a lone ``[H]``.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles.replace("[*]", "*"))
    if mol is None:
        return None
    atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 0]
    if not atoms:
        return None
    if len(Chem.GetMolFrags(mol)) > 1:
        return None
    if all(atom.GetAtomicNum() == 1 for atom in atoms):
        return None
    return len(atoms)


def _hydride_flags(smiles: str) -> Tuple[bool, bool]:
    """``(is_parent_hydride, is_chain_hydride)`` for a table entry's SMILES.

    A **parent hydride** is a bare skeleton carrying no characteristic group.
    Every non-carbon atom being a RING atom is the whole test, and it is enough
    for the one question ``_well_formed`` R3 asks: a characteristic-group suffix
    attaches to a parent hydride (P-14.2), never to a molecule that already
    carries its own characteristic group. ``benzene``, ``cyclohexane``,
    ``pyridine`` and ``1,3-thiazole`` pass; ``phosphoramid`` (whose SMILES
    already holds the acid oxygens that ``-ic acid`` would add again, giving 7
    for a 5-atom acid) and ``nitroform`` (a whole molecule, not a stem) do not.

    A **chain hydride** is one with no ring, which is what R3c needs: the bare
    ``-oic acid``/``-al``/``-amide`` forms count the acid carbon as part of the
    chain, so they belong on a chain and a ring takes the carb- form instead.
    Deriving this from the SMILES rather than from the registration source is
    what keeps ``propane``/``hexane`` (retained names, not chain-stem entries)
    from being mistaken for rings, which refused every ``propanoic acid``.

    Chain stems and Hantzsch-Widman stems carry no SMILES and are hydrides by
    construction, so they are flagged at their own registration sites.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles.replace("[*]", "*"))
    if mol is None:
        return False, False
    hydride = all(atom.GetAtomicNum() == 6 or atom.IsInRing()
                  for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)
    return hydride, hydride and not mol.GetRingInfo().NumRings()


def _entry_atoms(name: str, smiles: str) -> Optional[int]:
    """``_heavy_atoms``, refused when the NAME contradicts its own SMILES.

    Both screens compare two statements the shipped entry makes about itself,
    so neither guesses at a structure:

    * an **isotope** the name asserts and the SMILES does not carry -- the atoms
      would be invisible in the implicit hydrogen count (``borodeuteride`` /
      ``[BH4-]``: the name spells B + 4 D, the SMILES counts 1);
    * a **ring** the name's Hantzsch-Widman ending asserts and the SMILES does
      not close (``aluminane`` / ``[AlH]``, ``iodinane`` / ``[IH3]``: the name
      spells a six-membered ring, the SMILES is one atom).

    Either way the entry is self-inconsistent, so no count can be read off it.
    """
    from rdkit import Chem

    count = _heavy_atoms(smiles)
    if count is None:
        return None
    mol = Chem.MolFromSmiles(smiles.replace("[*]", "*"))
    if mol is None:
        return None
    lowered = name.strip().lower()
    if _ISOTOPE_MORPHEMES.search(lowered) and not any(
            atom.GetIsotope() for atom in mol.GetAtoms()):
        return None
    if not mol.GetRingInfo().NumRings() and any(
            lowered.endswith(ending) for ending in _HW_RING_ENDINGS):
        return None
    return count


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

    def add(self, key: str, atoms: Optional[int], category: str, *,
            suffix_only: bool = False, override: bool = False,
            hydride: bool = False, chain: bool = False,
            chain_only: bool = False) -> None:
        """Register ``key``, or refuse it when two sources disagree.

        ``override`` re-categorises an already-registered key. It exists for one
        purpose: a fusion prefix must be recognisable AS a fusion prefix even
        when some other table already spelled the same letters, because the
        FUSE category is what makes ``_well_formed`` refuse the token. Losing
        that categorisation would silently restore the confident-wrong reading,
        so the refusing category has to win.
        """
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
            self.table[key] = _Morph(atoms, category, suffix_only, hydride,
                                     chain, chain_only)
        elif existing.atoms != atoms:
            # Genuine disagreement between sources -> refuse the morpheme.
            del self.table[key]
            self.dropped.add(key)
        elif override:
            self.table[key] = _Morph(atoms, category, suffix_only, hydride,
                                     chain, chain_only)
        # Same count, different category: keep the first registration. The
        # registration order below runs most-specific-first, so the structural
        # reading of a morpheme wins over an incidental catalog entry.


def _add_suffix_morphemes(general: _Lexicon) -> None:
    """Suffix particles, bridged surface -> rule -> SMILES.

    ``OPSIN_SUFFIX_APPLICABILITY`` rows carry the surface morpheme in
    ``suffix_value`` and the rule name in ``suffix_text`` (the importer's
    field names follow the XML attribute/'text' split, not the semantics).
    A rule's ``addgroup`` SMILES is what the suffix ADDS; the separate
    ``addSuffixPrefixIfNonePresentAndCyclic`` carbon is added only for a ring
    parent, which is exactly the chain/ring split that makes ``-oic acid``
    spell 2 atoms while ``-carboxylic acid`` spells 3.

    An ATTACH suffix is one whose rule creates an attachment point
    (``setOutAtom`` or ``outIDs``) -- that is read off the rule, not assumed --
    and legitimately occurs inside a substituent prefix token
    ("methoxymethyl"). The rest are principal characteristic group suffixes,
    which are position-sensitive: they are registered with ``suffix_only=True``
    so ``_well_formed`` can require them to be final and to attach to a parent
    hydride, rather than being hidden from the enumeration.
    """
    from orthonym.data.opsin_imports.suffix_rules import (
        OPSIN_SUFFIX_APPLICABILITY,
        OPSIN_SUFFIX_RULES,
    )

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
        # A rule with a cyclic-extra atom states TWO different accountings:
        # the bare surface's count is the CHAIN one, valid only when the acid
        # carbon belongs to the chain. On a ring the carb- form is required
        # (P-65.1.1, P-66.1), so the bare surface is chain-only.
        general.add(surface, chain_atoms, ATTACH if attaches else AFFIX,
                    suffix_only=not attaches,
                    chain_only=bool(any(s[1] for s in scored)) and not attaches)

    # Surface spellings the applicability table does not carry, counts still
    # derived from the rules above.
    for surface, rule in _ACID_WORD_SURFACES.items():
        scored = score(rule)
        if scored is not None:
            general.add(surface, scored[0], AFFIX, suffix_only=True,
                        chain_only=bool(scored[1]))
    for surface, rule in _CARB_FORM_SURFACES.items():
        scored = score(rule)
        if scored is not None:
            # chain form + the carbon the rule adds for a ring parent. This IS
            # the ring form, so it is deliberately NOT chain_only.
            general.add(surface, scored[0] + scored[1], AFFIX,
                        suffix_only=True)


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
            hydride, chain = _hydride_flags(smiles)
            for name in record.get("names", ()):
                general.add(name, _entry_atoms(name, smiles), category,
                            hydride=hydride, chain=chain)


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
        hydride, chain = _hydride_flags(smiles)
        general.add(name, _entry_atoms(name, smiles), STEM,
                    hydride=hydride, chain=chain)

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
            hydride, chain = _hydride_flags(smiles)
            general.add(name, _entry_atoms(name, smiles), STEM,
                        hydride=hydride, chain=chain)


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
            general.add(substituent_name, parent.atoms, STEM,
                        hydride=parent.hydride, chain=parent.chain)

    # Fusion prefixes. A mancude monocyclic component's heavy-atom count is its
    # ring size, but that count may NOT be summed with the base component's:
    # ortho-fusion SHARES the two atoms of the fusion bond, so
    # benzo(6)+pyran(6) spells 10 atoms, not 12. They are registered so the
    # morpheme is RECOGNISED -- and categorised FUSE so ``_well_formed`` refuses
    # any token containing one, which is the same "whole class out of scope"
    # refusal the lambda convention gets.
    #
    # Recognising them is not optional: this is where the benzoyl defect lived.
    # ``benzoyl`` has no ``benz`` acyl stem in any shipped table, so its ONLY
    # tiling was benzo+yl -- a fusion prefix in a position where nothing is
    # being fused -- and the oracle certified 6 atoms for an 8-atom group.
    # ``override=True`` because the refusing category has to win over any
    # coincidental same-count registration from another table.
    try:
        from orthonym.data.fusion_components import MONOCYCLIC_COMPONENTS
    except ImportError:
        MONOCYCLIC_COMPONENTS = {}
    for component in MONOCYCLIC_COMPONENTS.values():
        prefix = component.get("prefix")
        ring_size = component.get("ring_size")
        if prefix and isinstance(ring_size, int):
            general.add(prefix, ring_size, FUSE, override=True)


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
            general.add(stem, n, STEM, hydride=True, chain=True)


def _add_elided_stems(general: _Lexicon) -> None:
    """Terminal-'e' elision of the NAMED skeletons (P-16.7.1(a)).

    A parent hydride drops its final 'e' before a suffix or a locant:
    "1,3-thiazole" but "1,3-thiazol-5-yl", "pyridine" but "pyridin-2-yl". The
    elided spelling denotes the same skeleton, so the count carries across
    unchanged -- derived, not asserted.

    Runs LAST, so it only ever adds a spelling for a skeleton the tables have
    already agreed on. ``_MIN_ELIDED + 1`` keeps the elided form itself at least
    ``_MIN_ELIDED`` characters, i.e. distinctive: a short elision matches inside
    unrelated words and manufactures readings (see ``_hw_stems``).
    """
    for key, morph in list(general.table.items()):
        if morph.category != STEM or not key.endswith("e"):
            continue
        if len(key) < _MIN_ELIDED + 1 or " " in key:
            continue
        general.add(key[:-1], morph.atoms, STEM, hydride=morph.hydride,
                    chain=morph.chain)


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
    ``ane`` (0 atoms) and make every ``cyclohexane`` ambiguous.
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
            # suffix: "oxane" but "oxan-4-yl".
            #
            # ``_MIN_ELIDED``: this injection path bypasses ``_Lexicon.add``
            # entirely -- no length screen, no cross-source conflict refusal --
            # and an elided spelling is DERIVED rather than shipped, so it
            # carries the higher bar. It matters: the 2-character elision of
            # 'ine' gave a bare 'in' that matched inside four Blue Book acid
            # names and planted a spurious six-membered ring in each --
            # "benzeneseleninic acid" read as benzene+selen+IN+ic acid and
            # answered a CONFIDENT 14 for a 9-atom acid.
            #
            # HONEST SCOPE: with R3c (a chain-form suffix needs a chain parent)
            # in place, the swept corpus shows 0 confident-wrong at a floor of 2
            # as well, because R3c catches that whole family by a second route.
            # This is defence in depth against the demonstrated mechanism, not
            # the only barrier in front of it, and
            # ``test_elided_hw_stems_stay_distinctive`` pins it directly so it
            # cannot rot into an untested rule.
            #
            # The elided forms that real ring names need come from
            # ``_add_elided_stems`` instead, which elides the NAMED ring
            # morphemes ('thiazole' -> 'thiazol') where a distinctive stem
            # anchors the reading.
            if stem.endswith("e") and len(stem[:-1]) >= _MIN_ELIDED:
                sizes.setdefault(stem[:-1], set()).add(ring_size)
    return {stem: next(iter(s)) for stem, s in sizes.items() if len(s) == 1}


@lru_cache(maxsize=1)
def _multipliers() -> Dict[str, int]:
    """Multiplying prefixes, inverted from the tables the namer emits with."""
    result: Dict[str, int] = {}
    try:
        from orthonym.assembly.naming_utils import COMPLEX_MULTIPLIERS, SIMPLE_MULTIPLIERS
    except ImportError:
        return result
    for count, word in SIMPLE_MULTIPLIERS.items():
        result[word.lower()] = count
    for count, word in COMPLEX_MULTIPLIERS.items():
        result[word.lower()] = count
    return result


# ---------------------------------------------------------------------------
# On-disk cache of the built lexicon (audit 2026-09-03, item S3)
#
# The table below is a pure function of the package's own source (the data
# tables and the derivation steps in _LEXICON_SOURCES), yet every process
# rebuilt it: 530 ms per CLI call, per pytest worker, per batch shard. A pickle
# of the 2,546-entry table is 116 KB and loads in 3 ms.
#
# Invariants (each is load-bearing for byte-identity):
# * A cache HIT must equal a fresh build. The key is a SHA-256 over every source
#   file the build reads, the derivation-step tuple, the package version, the
#   Python major.minor and the RDKit version, so any change to any input yields
#   a new file name; stale files are simply never opened.
# * Anything unexpected -> build fresh and try to rewrite. A missing directory, an
#   unreadable or corrupt file, a wrong-shaped payload, a read-only filesystem:
#   none of these may change the returned table or raise.
# * Writes are atomic (tmp file + os.replace) because 40 shards start at once.
# * ORTHONYM_LEXICON_CACHE=off disables it; ORTHONYM_CACHE_DIR moves it
#   (default: $XDG_CACHE_HOME/orthonym or ~/.cache/orthonym).
# The file is a pickle from the user's own cache directory; the build reads no
# untrusted input, so this is the same trust boundary as the installed package.
# ---------------------------------------------------------------------------
import hashlib as _hashlib
import os as _os
import pickle as _pickle
import sys as _sys
import tempfile as _tempfile

_PKG_ROOT = Path(__file__).resolve().parents[1]
_LEXICON_SOURCE_FILES: Tuple[str, ...] = (
    "validation/name_morphemes.py",
    "assembly/naming_utils.py",
    "rules/ring_substituents.py",
    "rules/skeletal_replacement.py",
)


def lexicon_cache_key() -> str:
    """SHA-256 identifying the exact inputs of the lexicon build."""
    h = _hashlib.sha256()
    files = [_PKG_ROOT / rel for rel in _LEXICON_SOURCE_FILES]
    files += sorted((_PKG_ROOT / "data").rglob("*.py"))
    for f in files:
        try:
            h.update(f.read_bytes())
        except OSError:
            h.update(b"<unreadable>")
    try:
        from orthonym import __version__ as _ver
    except Exception:  # pragma: no cover - only during a broken partial import
        _ver = "?"
    try:
        import rdkit
        _rd = getattr(rdkit, "__version__", "?")
    except Exception:  # pragma: no cover
        _rd = "?"
    h.update(repr(_LEXICON_SOURCES).encode())
    h.update(f"{_ver}|py{_sys.version_info[0]}.{_sys.version_info[1]}|rdkit{_rd}".encode())
    return h.hexdigest()


def _lexicon_cache_enabled() -> bool:
    return _os.environ.get("ORTHONYM_LEXICON_CACHE", "on").strip().lower() not in ("off", "0", "no")


def _lexicon_cache_path() -> Path:
    base = _os.environ.get("ORTHONYM_CACHE_DIR")
    if not base:
        xdg = _os.environ.get("XDG_CACHE_HOME")
        base = str(Path(xdg) / "orthonym") if xdg else str(Path.home() / ".cache" / "orthonym")
    return Path(base) / f"lexicon-{lexicon_cache_key()}.pkl"


def _well_shaped_lexicon(obj) -> bool:
    return (isinstance(obj, dict) and len(obj) > 0
            and all(isinstance(k, str) and isinstance(v, _Morph) for k, v in obj.items()))


def _load_lexicon_cache(path: Path) -> Optional[Dict[str, _Morph]]:
    try:
        obj = _pickle.loads(path.read_bytes())
    except Exception:
        return None
    return obj if _well_shaped_lexicon(obj) else None


def _store_lexicon_cache(path: Path, table: Dict[str, _Morph]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = _tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
        try:
            with _os.fdopen(fd, "wb") as fh:
                _pickle.dump(table, fh, protocol=_pickle.HIGHEST_PROTOCOL)
            _os.replace(tmp, path)
        except Exception:
            try:
                _os.unlink(tmp)
            except OSError:
                pass
            raise
    except Exception:
        return  # a cache that cannot be written is simply not a cache


@lru_cache(maxsize=1)
def _lexicon() -> Dict[str, _Morph]:
    """The one morpheme table (see :func:`_build_lexicon`), served from the
    on-disk cache when a file for exactly this code exists."""
    if not _lexicon_cache_enabled():
        return _build_lexicon()
    try:
        path = _lexicon_cache_path()
    except Exception:
        return _build_lexicon()
    cached = _load_lexicon_cache(path)
    if cached is not None:
        return cached
    table = _build_lexicon()
    _store_lexicon_cache(path, table)
    return table


def _build_lexicon() -> Dict[str, _Morph]:
    """The one morpheme table, used for every binding kind.

    There is deliberately only ONE. An earlier revision consulted a narrowed
    table for non-suffix kinds, which made the prefix path LESS safe than the
    suffix path: removing the ``-ol`` suffix reading of ``ethaneselenol`` left
    only the spurious Hantzsch-Widman ``selen|ol`` ring reading, and unanimity
    over that single reading certified 7 atoms for a 3-atom group. Dropping
    candidate readings can only manufacture false confidence, so every reading
    is always enumerated and position is policed by ``_well_formed`` instead.

    Registration order is most-specific-first so that a structural reading of
    a morpheme wins the CATEGORY over an incidental catalog entry with the
    same count. Disagreements on the COUNT drop the morpheme either way.
    """
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    try:
        general = _Lexicon()

        for key, (atoms, _role, _why) in _STRUCTURAL_AFFIXES.items():
            general.add(key, atoms, AFFIX)
        _add_suffix_morphemes(general)
        _add_replacement_prefixes(general)
        _add_chain_stems(general)
        _add_group_tables(general)
        _add_retained_names(general)
        _add_ring_derived(general)
        _add_elided_stems(general)
        return dict(general.table)
    finally:
        RDLogger.EnableLog("rdApp.*")


@lru_cache(maxsize=1)
def _by_first_char() -> Dict[str, List[str]]:
    """Morpheme keys bucketed by first character, longest first."""
    buckets: Dict[str, List[str]] = {}
    for key in _lexicon():
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
    table = _lexicon()
    for size in range(min(len(text), 20), 1, -1):
        candidate = text[:size]
        morph = table.get(candidate)
        if morph is not None and morph.category == STEM:
            return candidate, morph.atoms
    return None


# --------------------------------------------------------------------------
# Parsing: enumerate EVERY decomposition, then require agreement.
# --------------------------------------------------------------------------

def _parse_all(text: str) -> Tuple[List[List[_Seg]], int, bool]:
    """All morpheme tilings of ``text``.

    Returns (parses, furthest position reached, hit_cap). Enumerating every
    tiling rather than taking the greedy longest match is what lets ambiguity
    be DETECTED instead of silently resolved.
    """
    table = _lexicon()
    buckets = _by_first_char()
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
                    # A HW stem is a parent hydride by construction.
                    acc.append(_Seg(stem, ring_size, STEM, hydride=True))
                    walk(pos + len(stem), acc)
                    acc.pop()

        for key in buckets.get(char, ()):
            if text.startswith(key, pos):
                morph = table[key]
                acc.append(_Seg(key, morph.atoms, morph.category,
                                morph.suffix_only, morph.hydride,
                                morph.chain, morph.chain_only))
                walk(pos + len(key), acc)
                acc.pop()

        for word, count in multipliers.items():
            if word[0] == char and text.startswith(word, pos):
                acc.append(_Seg(word, count, MULT))
                walk(pos + len(word), acc)
                acc.pop()

    walk(0, [])
    return parses, furthest, hit_cap


def _content(segments: List[_Seg]) -> List[_Seg]:
    """``segments`` without the enclosing marks, which carry no morphology."""
    return [s for s in segments if s.category not in (OPEN, CLOSE)]


def _ends_free_valence(text: str) -> bool:
    """Does ``text`` end in a P-29.2 free-valence affix?

    A morpheme that does is a SUBSTITUENT form ("phenyl", "cyclohexyl") and may
    legally be followed by the skeleton it is attached to, which a bare stem may
    not. Shares ``_FREE_VALENCE_ENDINGS`` with ``free_valence_morphology`` at the
    foot of this module so the two readings of ``-yl`` cannot drift apart.
    """
    return any(text.endswith(ending) for ending, _n in _FREE_VALENCE_ENDINGS)


def _well_formed(segments: List[_Seg], kind_name: str,
                 total: Optional[int]) -> Optional[str]:
    """Why this tiling cannot describe any molecule, or None if it can.

    Each rule is one IUPAC construction rule read in reverse. A violation means
    the scanner matched morphemes across a boundary the lexicon cannot see, so
    the caller refuses the WHOLE token rather than dropping this tiling -- see
    "WHY AGREEMENT IS NOT ENOUGH" in the module docstring.
    """
    content = _content(segments)
    for index, segment in enumerate(content):
        previous = content[index - 1] if index else None

        # R1. Fusion is not addition (P-25.3.1): the components share the atoms
        # of every fusion bond, so no sum over a fusion prefix is a count.
        if segment.category == FUSE:
            return (f"fusion prefix {segment.text!r}: fused components share "
                    f"their fusion atoms, so their arities cannot be summed")

        # R2. A skeletal replacement prefix REPLACES an atom of a skeleton
        # (P-15.4), so it contributes 0 only because some stem beside it already
        # counted that atom. With no stem to qualify, the 0 counts nothing --
        # this is the 'diazenyl' = di|az|en|yl = 0 defect.
        if segment.category == REPL:
            ahead = index + 1
            while ahead < len(content):
                nxt = content[ahead]
                if nxt.category in (REPL, MULT) or nxt.text in _PREMARKS:
                    ahead += 1
                    continue
                break
            if ahead >= len(content) or content[ahead].category != STEM:
                return (f"replacement prefix {segment.text!r} qualifies no "
                        f"skeleton, so its zero atoms replace nothing")

        # R3. A characteristic-group suffix attaches to a parent hydride and
        # closes the name (P-14.2, P-15.1). It may not float in the middle of a
        # token ('alanylalanine' read as al|an|yl|alanine) and it may not hang
        # off a substituent prefix ('imido|hydrazide' -- where the true
        # accounting is functional REPLACEMENT, P-25.3, not addition).
        if segment.suffix_only:
            if any(not s.suffix_only for s in content[index + 1:]):
                return (f"suffix morpheme {segment.text!r} is not final, so it "
                        f"is not the suffix of anything")
            if previous is None:
                # Legitimate for a SUFFIX binding, whose parent is a DIFFERENT
                # binding: the token really is just "-oic acid".
                if kind_name != "suffix":
                    return (f"suffix morpheme {segment.text!r} opens a "
                            f"{kind_name} token, attaching to no parent hydride")
            elif not ((previous.category == STEM and previous.hydride)
                      or previous.category == MULT
                      or (previous.category == AFFIX and not previous.suffix_only)
                      or previous.suffix_only):
                return (f"suffix morpheme {segment.text!r} follows "
                        f"{previous.text!r}, which is no parent hydride")

            # R3c. The bare chain form of an acid/amide/nitrile suffix counts the
            # acid carbon as part of the CHAIN. On a ring parent the carb- form
            # is required instead (P-65.1.1, P-66.1) and spells one atom more, so
            # the bare form beside a ring stem is a mis-tiling:
            # "ethylideneazinic acid" read as eth|ylidene|azin|IC ACID and
            # answered a CONFIDENT 10 for a 5-atom acid.
            if segment.chain_only:
                back = index - 1
                while back >= 0 and (
                        content[back].category == MULT
                        or (content[back].category == AFFIX
                            and not content[back].suffix_only)):
                    back -= 1
                host = content[back] if back >= 0 else None
                if host is None:
                    if kind_name != "suffix":
                        return (f"suffix morpheme {segment.text!r} names no "
                                f"chain for its acid carbon to belong to")
                elif not (host.category == STEM and host.chain):
                    return (f"chain-form suffix {segment.text!r} sits on "
                            f"{host.text!r}, which is no chain parent hydride "
                            f"(a ring parent takes the carb- form)")

        # R4. Two skeletons cannot be juxtaposed: a name joins them with an
        # attachment affix ("cyclohexylmethane") or a linking prefix. A stem
        # sitting straight on a bond-order ending or on another stem is the
        # signature of a missing morpheme ('cyclohexane|carbohydrazide',
        # 'eth|an|eth|io|amide').
        if segment.category == STEM and previous is not None:
            joined = (previous.category in (MULT, REPL, ATTACH, SUBST)
                      or previous.text in _PREMARKS
                      or _ends_free_valence(previous.text))
            if not joined:
                return (f"skeleton {segment.text!r} is juxtaposed on "
                        f"{previous.text!r} with no attachment affix between "
                        f"them")

    # R5. A multiplier that ends up multiplying nothing at all, in a token that
    # spells no atoms, is not a reading of anything: 'heptaene' tiled as
    # hepta|ene and totalled 0 for a seven-carbon chain.
    if total == 0 and any(s.category == MULT for s in content):
        return ("token spells no atoms yet carries a multiplier, so its "
                "multiplied group was never identified")
    return None


def _evaluate(segments: List[_Seg]) -> Optional[int]:
    """Total heavy atoms for one parse, or None if its multipliers are unclear.

    Multiplier scoping follows the IUPAC distinction the tokens themselves
    observe: ``di-``/``tri-`` multiply a SIMPLE substituent and are written
    unenclosed, while ``bis-``/``tris-`` multiply a COMPLEX one and are always
    followed by enclosing marks. So an enclosed group's extent is read from the
    brackets, and an unenclosed one ends at the first morpheme that completes a
    simple substituent (an attachment affix such as ``-yl``/``-oxy``, a complete
    substituent prefix such as ``chloro``, or a characteristic-group suffix,
    which R3 has already established is the end of the token).

    A multiplier at the very START of a token, governing the whole of it, is a
    statement about how many times this substituent occurs, not about how big it
    is: the token ``"dimethyl"`` spells one methyl. That multiplicity belongs to
    the multiplicity proof, not to arity, so it is ignored here. Anywhere else
    the multiplier is INTERNAL to the substituent's own name and genuinely
    multiplies an inner group -- whether something follows it
    ("dimethylphenyl") or something precedes it ("butanedioyl" = butane + TWO
    -oyl = 6, which was under-counted as 5 while only the following side was
    checked).
    """
    total = 0
    index = 0
    count = len(segments)
    # Content before the multiplier under consideration. A multiplier is a
    # token-level occurrence count only when it opens the token.
    seen_content = False

    while index < count:
        segment = segments[index]
        if segment.category != MULT:
            if segment.category not in (OPEN, CLOSE):
                seen_content = True
            total += segment.atoms
            index += 1
            continue

        multiplicity = segment.atoms
        index += 1
        if index >= count:
            return None  # trailing multiplier governs nothing

        following = segments[index]

        # A zero-atom skeletal marker ("dihydro", "tetrahydro") OR a zero-atom
        # skeletal REPLACEMENT prefix ("dioxa", "diaza", "triphospha"):
        # whatever the multiplier's true scope, it multiplies nothing, so the
        # reading is unambiguous regardless. REPL is 0 NET atoms BY
        # DEFINITION (it replaces an existing skeletal position the stem's own
        # count already includes -- "heptyl" is 7 atoms whether or not two of
        # them are relabelled O by "1,3-dioxa"), so a locant-count multiplier
        # in front of one can never add atoms, unlike a multiplier in front of
        # a genuine SUBST/ATTACH group ("dimethylamino"). Phase 0c Task 2b
        # regression: without this, a SUBST prefix earlier in the SAME
        # composite token (e.g. "2-hydroxy-2-oxo-1,3-dioxa-6-aza-2-phosphaheptyl")
        # set ``seen_content`` before the multiplier was reached, so the
        # general "internal multiplier" branch below fired and multiplied the
        # REPL chain's atom count as if ``di`` were multiplying a real
        # substituent -- a token claiming 9 real heavy atoms was confidently
        # answered 16. Isolated (nothing preceding), the old code reached the
        # same right answer by the ACCIDENT of the whole-token branch
        # (``internal=False``); this makes it right for the right reason and
        # for BOTH cases.
        if following.atoms == 0 and following.category in (AFFIX, REPL):
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
                # A whole-word substituent morpheme ends the group just as an
                # attachment affix does: 'phenyl' IS a complete simple
                # substituent, so "diphenylmethanone" multiplies the phenyl and
                # not everything after 'di'. Reading it as the whole tail is how
                # this under-counted 14 atoms as 8.
                if (current.category in _GROUP_TERMINATORS
                        or current.suffix_only
                        or _ends_free_valence(current.text)):
                    terminated = True
                    break
            if not terminated:
                return None  # cannot delimit the multiplied group
            index = end

        # Is the multiplied group the WHOLE token? If so the multiplier counts
        # how many times that whole substituent occurs -- multiplicity, not
        # arity. If anything precedes or follows it, the multiplied group is an
        # inner part of a larger token and the multiplier genuinely applies.
        # What follows need not be skeletal: "dimethylamino" is 2 methyls on a
        # nitrogen (3), so testing for a STEM specifically would under-count
        # every N,N-dialkyl token.
        tail = [s for s in segments[index:] if s.category not in (OPEN, CLOSE)]
        internal = seen_content or bool(tail)
        total += group_atoms * multiplicity if internal else group_atoms
        seen_content = True

    return total


@lru_cache(maxsize=4096)
def token_arity(token: str, kind: "object" = "prefix") -> ArityEstimate:
    """How many heavy atoms does ``token`` spell?

    ``kind`` may be a :class:`~orthonym.validation.binding_spine.BindingKind`
    or the equivalent plain string. It does NOT select a lexicon -- there is one
    lexicon and every reading is always enumerated (see ``_lexicon``). It settles
    one positional question only: whether a characteristic-group suffix may open
    the token. For a ``SUFFIX`` binding it may, because the parent hydride it
    attaches to is a different binding and the token really is just
    ``"-oic acid"``; for any other kind a leading suffix morpheme is a
    mis-tiling.

    A confident answer is guaranteed correct; an unconfident one carries a
    ``basis`` explaining the refusal. See the module docstring.
    """
    if not isinstance(token, str) or not token.strip():
        return ArityEstimate(None, False, "empty token")
    if len(token) > _MAX_TOKEN_LEN:
        return ArityEstimate(None, False,
                             f"token longer than {_MAX_TOKEN_LEN} characters")

    kind_name = str(getattr(kind, "value", kind)).strip().lower()

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

    parses, furthest, hit_cap = _parse_all(text)
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
        # A morphotactically impossible reading refuses the whole token: it is
        # evidence the lexicon is incomplete HERE, and once that is known,
        # unanimity among the readings that DID tile proves nothing. See "WHY
        # AGREEMENT IS NOT ENOUGH".
        violation = _well_formed(parse, kind_name, value)
        if violation is not None:
            return ArityEstimate(
                None, False,
                f"{basis_prefix}{'+'.join(s.text for s in _content(parse))!r} "
                f"is not constructible: {violation}")
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


def lexicon_size(include_suffixes: bool = True) -> int:
    """Number of morphemes the oracle knows. For measurement passes.

    ``include_suffixes`` is accepted and ignored: there is now one lexicon for
    every binding kind (see ``_lexicon``), so both answers are the same number.
    """
    return len(_lexicon())


def lexicon_entries() -> Dict[str, Tuple[int, str, bool]]:
    """``{morpheme: (atoms, category, suffix_only)}`` -- the whole lexicon.

    Exposed for the arity sweep and the tripwire test: a morpheme that reaches
    the table without arity data, or a NEW morpheme class nobody has swept
    against an independent structure, is exactly what this module must not
    answer confidently about.
    """
    return {key: (morph.atoms, morph.category, morph.suffix_only)
            for key, morph in _lexicon().items()}


#: Pinned by ``test_lexicon_sources_are_pinned``: see ``lexicon_sources``.
_LEXICON_SOURCES: Tuple[str, ...] = (
    "_STRUCTURAL_AFFIXES",
    "_add_suffix_morphemes",
    "_add_replacement_prefixes",
    "_add_chain_stems",
    "_add_group_tables",
    "_add_retained_names",
    "_add_ring_derived",
    "_add_elided_stems",
)


def lexicon_sources() -> Tuple[str, ...]:
    """Names of the derivation steps ``_lexicon`` runs, in order.

    Pinned by a tripwire test: a NEW morpheme source has not been swept against
    an independent structure oracle, and its entries have not been checked
    against ``_entry_atoms``' screens, so it must not slip in unnoticed.
    """
    return _LEXICON_SOURCES


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
