"""Twin spellings: a Blue Book PIN spelling that OPSIN 2.9.0 has no grammar for, checked through a
spelling of the same structure that OPSIN reads.

A twin is a token-level rewrite of the PIN spelling into a general-nomenclature spelling that
names the same structure. It proves the structure and never the spelling: the producer that
builds the PIN spelling owns its correctness, and the twin replaces only the OPSIN read-back. A
name ships through its twin when OPSIN rejects the name, the twin reads to the input's FULL
InChIKey (``namer._shipped_name_round_trip``: the radical and the protonation site are checked
too) and OPSIN gives no ``APPEARS_AMBIGUOUS`` warning for the twin. The twin is never shipped.

Why the ambiguity check: Citation of locants (the Blue Book) "if any locants are
essential for defining the structure of the parent structure..., then all locants must be
cited". A twin that OPSIN reads only by choosing among readings does not prove the structure of
the name it stands for.

The families (the rules of:data:`TWIN_RULES`, in table order):

* ``bridge_sulfano``, ``bridge_disulfano``, ``bridge_azano``: Divalent bridges,
   (the Blue Book-14110): "The following names are now recommended as
  preselected (see bridge prefixes: 'sulfano', –S–;'disulfano', –SS–;... 'azano', –NH–"
  and "The bridge prefixes 'epithio', 'epidithio', 'episeleno', 'epitelluro', and 'epimino'...
  may be used in general nomenclature" (:14097); the PIN example '1,4-dihydro-1,4-
  sulfanonaphthalene (PIN)' stands under CYCLIC ETHERS, SULFIDES, SELENIDES AND
  TELLURIDES (:28063). The twins are 'epithio', 'epidithio' and 'epimino'. 'selano' and
  'tellano' have no twin: OPSIN reads 'episeleno' and 'epitelluro' as a [SeH2] / [TeH2] bridge,
  another structure. A bridge cited more than once ('1,4:5,8-' before the prefix) has no twin
  rule: 'disulfano' is the name of the –S–S– bridge (:14105), so it cannot stand for two –S–
  bridges.
* ``interior_phenalene``: Numbering of interior carbon atoms (the Blue Book),
   (:12658): "The locant for the interior atom is that of the peripheral atom with a
  superscript number corresponding to the number of bonds between the two atoms. The previous rule
  ... recommended that interior atoms follow the highest numbered peripheral atom adding Roman letters
  in sequence to the appropriate peripheral number." The interior atom of phenalene is '3a^1' (the
  former '9b'), as in '1,3a^1,4,9-tetraazaphenalene (PIN)',:14470, example:14491). OPSIN
  reads the former locant: '1,9b,4,9-tetraazaphenalene'. The numbering of the peripheral atoms is the
  same in both, so the twin is the same structure; the rule applies to a name of the parent
  'phenalene' only.
* ``isotope_letter_once``: (heading NAMES,:43824, example:43830) "In a name
  consisting of one word, the isotopic descriptor is placed before the name, with an appropriate
  locant", '(N-2H2)aniline (PIN)': a letter locant is cited once for a nuclide that is multiplied.
  OPSIN reads the letter once per atom: '(N,N-2H2)aniline'.
* ``isotope_count_one``: (heading Omission of locants,:44184) "ethan(2H)ol
  (PIN) (as in ethanol)", (:44214) '(1-2H1)ethan-1-(2H)ol (PIN)', (heading
   ORDER OF NUCLIDE SYMBOLS,:43832; the rule:43834) 'methan(2H,18O)ol (PIN)' (:43840): a
  hydrogen nuclide that needs neither a count nor a locant is cited bare. OPSIN reads such a
  nuclide with a count: '(2H1)'. A hydrogen nuclide with a locant in its descriptor
  ('(O-2H)acetic acid') is read as written and is not rewritten, and the ``(2H)`` / ``(3H)`` that
  follows a locant, primed or not ('naphthalen-2(3H)-one', '[1,1'-biphenyl]-4'(2H)-one'), is the
  added-hydrogen mark.
* ``polyester_acyloxy`` (``validation.polyester_twin``): method (1), the functional-class
  polyol polyester name 'propane-1,2,3-triyl 1,3-diacetate 2-propanoate (PIN)' (the Blue Book)
  and its acyloxy spelling '1,3-bis[acetyloxy]-2-[propanoyloxy]propane', substitutive nomenclature of
  the same ester,:31968). OPSIN reads the second and not the first. A half ester of a
  polybasic acid has no acyloxy spelling OPSIN reads, and its name does not ship.
* ``halido_o`` (``validation.halido_twin``): (the Blue Book) 'trichlorido(methyl)titanium'
  (:39789) and the long-established ligand spelling 'trichloro(methyl)titanium',:39761,:39767),
  which OPSIN reads; the additive names of the transition-metal producer, without a hapto descriptor.
* ``uide_lambda`` (``validation.uide_twin``): ANIONS FORMED BY ADDITION OF HYDRIDE IONS
  (the Blue Book), method (1) 'diphenyliodanuide (PIN)' and its method (2) spelling
  'diphenyl-lambda3-iodanide' (:41110), which OPSIN reads. The family is structure-blind like the
  others but is not a rewrite of the name alone: the lambda number is the one method (2) pairs with
  an unmarked -uide name of the element, so it lives in its own module and is reached from
  :func:`twin_round_trip`; a general tier ships a name through its twin for this family and the next.
* ``thioperoxol_sulfenic``, ``thioperoxol_peroxol`` (``validation.thioperoxol_twin``): THE SUFFIXES
  'SO-THIOPEROXOL', AND CHALOGEN ANALOGUES (the Blue Book), 'CH3-S-OH methane-SO-thioperoxol (PIN)
  (not methanesulfenic acid)' (:24326); Basic functional suffixes (:17456), "-OS-thioperoxol for
  -OSH" is 'peroxol' modified by functional replacement (:17460). The twin of '-SO-thioperoxol' is
  '-sulfenic acid', that of '-OS-thioperoxol' is '-peroxol' read against the structure with the sulfur an
  oxygen; OPSIN has no grammar for either suffix at any tier. The default tier ships them through the twin; a
  general tier ships the name OPSIN reads itself, '(hydroxysulfanyl)methane', and the twin only for the anion of
  -S-OH, which has no such name, and as the safety net of a molecule that only the PIN path names
  (``validation.thioperoxol_twin``).
* ``anhydride_acids`` (``validation.anhydride_acids``): Di- and polyanhydrides (the Blue Book)
  'diacetic butanedioic dianhydride (PIN)' (:32386). No twin spelling of such a name exists (OPSIN reads a
  two-word anhydride only), so the check reads each acid word as ``<acid> acid`` and builds the poly-anhydride
  from those acids; it ships a name at the default tier only.
* ``long_count`` (``validation.long_count_read_back``): Basic multiplicative prefixes (the Blue Book),
   Derivation of basic numerical terms (:2811), Hydrocarbons (:7984) 'heptacontane (PIN)'. OPSIN
  has a numerical term up to 50 and none above, so a name with a count word of 51 and more has no OPSIN reading and
  no twin spelling. The reconstructor builds the structure from the text of the name (the parent chain, its multiple
  bonds, its prefixes) and the constitution must be the input's, at every tier that ships a name OPSIN cannot read.
"""
import re
from typing import Callable, Optional, Tuple

from ..metrics.provenance import wider_tier_ctx
from .anhydride_acids import ANHYDRIDE_ACIDS_RULE, anhydride_acids_round_trip
from .halido_twin import halido_twin
from .long_count_read_back import LONG_COUNT_RULE, long_count_round_trip
from .opsin_ambiguity import ambiguity_messages
from .polyester_twin import polyester_has_free_acid, polyester_twin
from .thioperoxol_twin import ships_at_general_tier, thioperoxol_twin_round_trip
from .uide_twin import uide_twin_round_trip

#: one locant pair, cited once before a bridge prefix ('1,4', '4a,8a'); not the second pair of a
#: colon-separated list, not part of a longer locant list
_PAIR_BEFORE = r"(?<![\w,:'^])(\d+[a-z]?,\d+[a-z]?)-"


def _bridge_rule(pin: str, twin: str, not_before: str = "") -> Callable[[str], str]:
    pattern = re.compile(_PAIR_BEFORE + pin + not_before)
    return lambda name: pattern.sub(r"\1-" + twin, name)


#: the interior atom of phenalene, as a locant in a name of the parent 'phenalene', 'phenalen-3-amine',
#: 'phenalen-3-yl' (not 'phenaleno', the attached component of a fusion name)
_INTERIOR = re.compile(r"(?<![\w^'])3a\^1(?![\w^'])")
_PHENALENE = re.compile(r"phenalen(?:e(?![a-z])|(?=[-(]))")


def _interior_phenalene(name: str) -> str:
    return _INTERIOR.sub("9b", name) if _PHENALENE.search(name) else name


#: a locant of an isotopic descriptor: a number with an optional letter and primes, or an
#: element letter
_LOCANT = r"(?:\d+[a-z]?'*|[A-Z][a-z]?'*)"
#: one nuclide item of a descriptor: optional locants and a hyphen, the mass number, the element
#: symbol, an optional count
_ITEM = re.compile(rf"(?:{_LOCANT}(?:,{_LOCANT})*-)?(\d+)([A-Z][a-z]?)(\d*)")
_LETTER_ONCE = re.compile(r"\(([A-Z])-(\d+[A-Z][a-z]?)(\d+)\)")
_GROUP = re.compile(r"\(([^()]*)\)")
_LOCANT_BEFORE = re.compile(r"\d[a-z]?'*$")


def _letter_once(name: str) -> str:
    def repeat(m):
        count = int(m.group(3))
        return m.group(0) if count < 2 else f"({','.join([m.group(1)] * count)}-{m.group(2)}{m.group(3)})"
    return _LETTER_ONCE.sub(repeat, name)


def _count_one_group(name: str, m) -> str:
    """The group ``m`` of ``name`` with every bare ``2H`` / ``3H`` item given its count 1, or the group as it
    stands when it is no descriptor of nuclides or follows a locant (that is the added-hydrogen mark)."""
    if _LOCANT_BEFORE.search(name[:m.start()]):
        return m.group(0)
    body, pos, out = m.group(1), 0, []
    while True:
        item = _ITEM.match(body, pos)
        if item is None:
            return m.group(0)
        mass, symbol, count = item.group(1, 2, 3)
        if (mass, symbol) == ("1", "H"):
            return m.group(0)
        bare = item.start(1) == pos and mass in ("2", "3") and symbol == "H" and not count
        out.append(item.group(0) + ("1" if bare else ""))
        pos = item.end()
        if pos == len(body):
            return "(" + ",".join(out) + ")"
        if body[pos] != ",":
            return m.group(0)
        pos += 1


def _count_one(name: str) -> str:
    """'ethan(2H)ol' -> 'ethan(2H1)ol': a ``2H`` or ``3H`` item without locants and count inside a
    parenthesised group of nuclide items that holds no ``1H`` item, when no locant stands in front
    of the group (that is the added-hydrogen mark)."""
    return _GROUP.sub(lambda m: _count_one_group(name, m), name)


def descriptor_spans(name: str) -> list:
    """``(start, end)`` of every isotopic descriptor of ``name``: a parenthesised group of nuclide items
    that no locant precedes."""
    spans = []
    for m in _GROUP.finditer(name):
        body, pos = m.group(1), 0
        while True:
            item = _ITEM.match(body, pos)
            if item is None:
                break
            pos = item.end()
            if pos == len(body):
                if not _LOCANT_BEFORE.search(name[:m.start()]):
                    spans.append(m.span())
                break
            if body[pos] != ",":
                break
            pos += 1
    return spans


def bare_hydrogen_groups(name: str) -> list:
    """``(start, end)`` of every descriptor of ``name`` that cites a hydrogen nuclide without count and
    locant, the groups of the rule ``isotope_count_one``."""
    return [m.span() for m in _GROUP.finditer(name) if _count_one_group(name, m) != m.group(0)]


#: (rule id, rewrite) in the order the rules apply; each rewrite is pure and deterministic
TWIN_RULES: Tuple[Tuple[str, Callable[[str], str]], ...] = (
    ("bridge_sulfano", _bridge_rule("sulfano", "epithio", "(?!ne)")),
    ("bridge_disulfano", _bridge_rule("disulfano", "epidithio")),
    ("bridge_azano", _bridge_rule("azano", "epimino")),
    ("interior_phenalene", _interior_phenalene),
    ("isotope_letter_once", _letter_once),
    ("isotope_count_one", _count_one),
    ("polyester_acyloxy", polyester_twin),
    ("halido_o", halido_twin),
)


def twin_of(name: str) -> Optional[Tuple[str, Tuple[str, ...]]]:
    """``(twin, rule ids in table order)`` of ``name``, or None when no rule applies."""
    twin, applied = name, []
    for rule_id, rewrite in TWIN_RULES:
        rewritten = rewrite(twin)
        if rewritten != twin:
            twin = rewritten
            applied.append(rule_id)
    return (twin, tuple(applied)) if applied else None


def twin_round_trip(name: str, smiles: Optional[str]) -> Optional[str]:
    """The applied rule ids joined by ``+`` when the twin of ``name`` reads to the FULL
    InChIKey of ``smiles`` and OPSIN reads it once; else None (no twin, a twin of another
    structure, an unavailable OPSIN, an ambiguous reading). A halogen -uide name and a thioperoxol name
    have a family of their own (``validation.uide_twin``, ``validation.thioperoxol_twin``), a poly-anhydride
    name is read through its acids (``validation.anhydride_acids``) and a name with a count word above 50 is
    read by the reconstructor (``validation.long_count_read_back``). This is the check of the default tier: a
    thioperoxol name ships through its twin there always, at a general tier only the anion does, and a neutral name
    as the safety net (:func:`general_tier_twin_round_trip`)."""
    found = twin_of(name)
    if found is None:
        return (uide_twin_round_trip(name, smiles) or thioperoxol_twin_round_trip(name, smiles)
                or anhydride_acids_round_trip(name, smiles) or long_count_round_trip(name, smiles))
    if not smiles:
        return None
    twin, rules = found
    if "polyester_acyloxy" in rules and polyester_has_free_acid(smiles):
        return None  # a half ester: the PIN is the acid (validation.polyester_twin)
    from ..namer import _shipped_name_round_trip
    if _shipped_name_round_trip(twin, smiles) != "verified" or ambiguity_messages(twin):
        return None
    return "+".join(rules)


def general_tier_twin_round_trip(name: str, smiles: Optional[str]) -> Optional[str]:
    """The twin families a general tier ships a name through: a name no OPSIN grammar reads at any tier and for
    which the engine has no other spelling (the halogen -uide names, ``validation.uide_twin``), and an additive
    organometallic name with 'chlorido' ligands, ``validation.halido_twin``, whose Blue Book ligand word no OPSIN
    grammar reads either); the other families spell the form OPSIN reads there. The anion of -S-OH, a thioperoxolate
    name, belongs here too: the engine has no other spelling for it (``validation.thioperoxol_twin``). A neutral
    thioperoxol name has one, the name by prefixes that OPSIN reads ('(hydroxysulfanyl)methane'): it ships through
    its twin only inside ``thioperoxol_twin.last_resort_scope``, for a molecule that has no other name. The rule
    id, or None. A name with a count word above 50 has no other spelling either, and OPSIN has no numerical term
    for it: the reconstructor reads it (``validation.long_count_read_back``)."""
    found = twin_of(name)
    if found is not None and found[1] == ("halido_o",):
        return twin_round_trip(name, smiles)
    return (uide_twin_round_trip(name, smiles) or long_count_round_trip(name, smiles)
            or (thioperoxol_twin_round_trip(name, smiles) if ships_at_general_tier(name) else None))


def _opsin_reads(name: str, radical: bool) -> bool:
    if not radical:
        from ..namer import _validity_gate_name_to_smiles
        return _validity_gate_name_to_smiles(name) is not None
    from ..jvm_bridge import opsin_stdout
    from .opsin_roundtrip import _find_opsin_jar
    # Like the plain branch above (the validity gate), this asks the OPSIN in use (ORTHONYM_OPSIN) whether it
    # reads ``name``: the twin is the spelling for an OPSIN that does not, and the in-process OPSIN serves
    # only the jar its JVM started with.
    jar = _find_opsin_jar()
    text, served = opsin_stdout(name, True, jar) if jar else (None, False)
    return bool(served and (text or "").strip())


def opsin_spelling(name: str, radical: bool = False) -> str:
    """The spelling to give OPSIN to read the structure of ``name``: ``name``, or its twin when
    OPSIN rejects ``name`` and reads the twin. For the checks of a builder that read the name it
    is about to return (the tautomer of the drawing, the stereo of a prefix); ``radical`` is True
    for a name with a free valence (a prefix), which OPSIN reads with its radical setting on. The
    validity gate still refuses a twin OPSIN reads as ambiguous (:func:`twin_round_trip`)."""
    found = twin_of(name)
    if found is None:
        return name
    twin = found[0]
    if not _opsin_reads(name, radical) and _opsin_reads(twin, radical):
        return twin
    return name


def twin_opsin_parse(name: str) -> Optional[str]:
    """OPSIN's SMILES for the twin of ``name`` at a default-tier call; None at a wider tier, when
    no rule applies, when OPSIN rejects the twin or reads it as ambiguous (:func:`twin_round_trip`).
    The isotope placement search reads its candidates through this."""
    found = twin_of(name) if pin_spellings_on() else None
    if found is None or ambiguity_messages(found[0]):
        return None
    from .opsin_roundtrip import opsin_parse
    return opsin_parse(found[0])


def twin_gate_ships(name: str, smiles: Optional[str], general_tier: bool = False) -> bool:
    """The twin branch of the validity gate: True, with the outcome ``twin_verified:<rules>``
    (``acid_split_verified:<rule>`` for a poly-anhydride, ``reconstructed:<rule>`` for a name the reconstructor
    reads) recorded for ``name``, when:func:`twin_round_trip` verifies the twin of ``name``. A general
    tier (``general_tier``) ships only a name of:func:`general_tier_twin_round_trip` through its twin."""
    rules = general_tier_twin_round_trip(name, smiles) if general_tier else twin_round_trip(name, smiles)
    if rules is None:
        return False
    from ..metrics import provenance as pv
    # the acids of a poly-anhydride are read one by one: no OPSIN reading of the name or of a spelling of it
    pv.record_gate_outcome(
        pv.acid_split_outcome(rules) if rules == ANHYDRIDE_ACIDS_RULE
        else pv.reconstructed_outcome(rules) if rules == LONG_COUNT_RULE else pv.twin_outcome(rules), name)
    return True


def twin_outcome_holds(name: str, smiles: Optional[str]) -> Optional[bool]:
    """None unless the gate recorded a twin outcome for exactly this string; else whether the
    twin still reads back to ``smiles`` (the OPSIN parse is memoized: no new JVM call)."""
    from ..metrics import provenance as pv
    prov = pv.get_provenance()
    outcome = pv.resolve_gate_outcome(prov["gate_outcome"], prov["gate_outcome_name"], name)
    if not outcome.startswith(pv.NAME_READ_BACK_PREFIXES):
        return None
    return True if not smiles else twin_round_trip(name, smiles) is not None


def twin_offer_holds(name: str, smiles: Optional[str]) -> bool:
    """Whether the twin of ``name``, an offer no gate has seen yet (an isotope-labelled name is gated at
    the exit of the call), reads back to ``smiles``; False at a wider tier."""
    return pin_spellings_on() and twin_round_trip(name, smiles) is not None


def publish_tier_flags(wider: bool) -> list:
    """Publish the tier of a naming call for the producers: ``ps_tier_ctx`` as the call's own flag
    and, for a wider tier, ``wider_tier_ctx`` (``metrics.provenance``). The (variable, token) pairs are returned for
    the caller to reset when its call ends."""
    from ..rules.oxoacid_group_prefix import ps_tier_ctx
    published = [(ps_tier_ctx, ps_tier_ctx.set(wider))]
    if wider:
        published.append((wider_tier_ctx, wider_tier_ctx.set(True)))
    return published


def pin_spellings_on() -> bool:
    """True where the producers spell the Blue Book PIN form: a default-tier call. A wider tier
    (``wider_tier_ctx``) spells the general form OPSIN reads."""
    return not wider_tier_ctx.get()
