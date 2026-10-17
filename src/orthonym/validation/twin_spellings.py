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
"""
import re
from typing import Callable, Optional, Tuple

from ..metrics.provenance import wider_tier_ctx
from .opsin_ambiguity import ambiguity_messages

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
    structure, an unavailable OPSIN, an ambiguous reading)."""
    found = twin_of(name)
    if found is None or not smiles:
        return None
    twin, rules = found
    from ..namer import _shipped_name_round_trip
    if _shipped_name_round_trip(twin, smiles) != "verified" or ambiguity_messages(twin):
        return None
    return "+".join(rules)


def _opsin_reads(name: str, radical: bool) -> bool:
    if not radical:
        from ..namer import _validity_gate_name_to_smiles
        return _validity_gate_name_to_smiles(name) is not None
    from ..jvm_bridge import opsin_stdout
    from .opsin_roundtrip import _find_opsin_jar
    jar = _find_opsin_jar("2.9.0")
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


def twin_gate_ships(name: str, smiles: Optional[str]) -> bool:
    """The twin branch of the validity gate: True, with the outcome ``twin_verified:<rules>``
    recorded for ``name``, when:func:`twin_round_trip` verifies the twin of ``name``."""
    rules = twin_round_trip(name, smiles)
    if rules is None:
        return False
    from ..metrics import provenance as pv
    pv.record_gate_outcome(pv.twin_outcome(rules), name)
    return True


def twin_outcome_holds(name: str, smiles: Optional[str]) -> Optional[bool]:
    """None unless the gate recorded a twin outcome for exactly this string; else whether the
    twin still reads back to ``smiles`` (the OPSIN parse is memoized: no new JVM call)."""
    from ..metrics import provenance as pv
    prov = pv.get_provenance()
    outcome = pv.resolve_gate_outcome(prov["gate_outcome"], prov["gate_outcome_name"], name)
    if not outcome.startswith(pv.GATE_OUTCOME_TWIN_PREFIX):
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
