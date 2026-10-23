"""A thioperoxol name read back through the spelling OPSIN reads of the same group.

OPSIN 2.9.0 has no grammar for the suffixes '-SO-thioperoxol' (R-S-OH) and '-OS-thioperoxol' (R-O-SH).
Until this module such a name shipped with no read-back at all (the ``thioperoxol`` grammar-gap
carve-out, which took any name that contained 'thioperoxol'), and 52 of the 308 names the CO audit read
back through a check that does not use the producer were the name of another molecule: a CH2 dropped
between a ring and the suffix (``C1CC1CSO`` named ``cyclopropane-1-SO-thioperoxol``), five and six bonds
on one carbon, a CF2-PH2 group named ``difluorophosphanylmethyl``, a second group the name does not cite,
a cation named as the neutral molecule.

The Blue Book gives the group two names and says which is which.

 THE SUFFIXES 'SO-THIOPEROXOL', AND CHALOGEN ANALOGUES (the Blue Book): "The suffixes
SO-thioperoxol and its chalcogen analogues are now introduced to replace the suffix 'sulfenic acid' and
its chalcogen analogues that were discarded in the 1993 recommendations." (:24322), 'CH3-S-OH
methane-SO-thioperoxol (PIN) (not methanesulfenic acid)' (:24326); (:27963), 'CH3-SOH... (2)
S-methyl thiohydroperoxide (no longer methanesulfenic acid)' (:27990). 'X-sulfenic acid' is a retired
spelling, not a PIN, and the book equates it with 'X-SO-thioperoxol': they name one structure. OPSIN reads
it ('propane-1-sulfenic acid' is CCCSO), and 'X-sulfenate' for the anion,:41035: 'CH3-CH2-S-O-
ethane(SO-thioperoxolate) (PIN) (ethylsulfanyl)oxidanide (not ethanesulfenate; sulfenic acids are no longer
recommended; see '). The twin of ``<parent>-SO-thioperoxol`` is ``<parent>-sulfenic acid``; the
parent, the locants, the prefixes and the stereodescriptors are the same string, so the read-back checks
them all, and the group -S-OH is the same in both spellings, so no CIP descriptor changes.

OPSIN reads no retired or functional-class spelling of -O-S-H ('O-propyl thiohydroperoxide' is rejected,
 (2)). Basic functional suffixes (:17456): "The suffix 'peroxol', for -OOH, has been
added to the list of basic suffixes. It is modified by functional replacement generating the suffixes
'-OS-thioperoxol' for -OSH, and '-SO-thioperoxol' for -SOH." (:17460), Table 6.1 (:27972): '-OS-thioperoxol'
is 'peroxol' with its second oxygen replaced by sulfur. The twin of ``<parent>-OS-thioperoxol`` is
``<parent>-peroxol`` (OPSIN reads it: 'propane-1-peroxol' is CCCOO), read against the structure with the
sulfur of that group made an oxygen. The peroxol group marks WHICH oxygen group carries the sulfur, which
the alcohol (-O-S-H made -O-H) cannot: a hydroxy prefix and the suffix exchanged read to one structure. The
sulfur edit changes the CIP ranks of a stereocentre that carries the group only where the sulfur decided
the rank (an -OP versus an -OS branch); the twin then does not read to the input and the name is
declined, a loss and never a wrong name.

The twin proves the structure and never the spelling: the producer owns the spelling. A name ships through
its twin when OPSIN reads the twin to the FULL InChIKey of the input
(``namer._shipped_name_round_trip``) and gives no ``APPEARS_AMBIGUOUS`` warning for it, as the other
twin families do (``validation.twin_spellings``). The twin is never shipped.

The default tier ships the PIN spelling through its twin. A general tier (``valid``, ``complete``,
``best-effort``) prefers a name OPSIN reads itself, as the other twin families do with
``twin_spellings.pin_spellings_on`` (a wider tier spells the form OPSIN reads). The general engine names the
neutral group by the prefixes the book gives it where it is not the principal characteristic group,
'(hydroxysulfanyl)methane' for -S-OH and '(sulfanyloxy)methane' for -O-S-H: Chalcogen analogues of
hydroperoxides (the Blue Book), "(1) by using prefixes such as 'hydroperoxy', –OOH; 'disulfanyl', –SSH, or by
combining simple prefixes, 'hydroxy' –OH; 'oxy-', –O–; 'sulfanyl', –SH; etc." (:28005), the PIN examples
'3-(sulfanyloxy)propanenitrile (PIN)' (:28027) and '4-[(hydroxysulfanyl)methyl]benzoic acid (PIN)' (:29962), the
glossary entries 'hydroxysulfanyl (HO-S–)' (:1810) and 'sulfanyloxy (HS-O–)' (:1812). OPSIN reads those names in
full, so such a name ships by the round trip of the name itself, and a neutral thioperoxol name that only its twin
reads is withdrawn at a general tier. The anion of -S-OH is the exception,:func:`ships_at_general_tier`: the engine
builds no name OPSIN reads for it (OPSIN reads '(ethylsulfanyl)oxidanide',,:41035, but the engine
builds no such name), so a general tier ships it through its twin, in the one pass, as the default tier does. The PIN spelling
of a neutral name is also the safety net of a general tier: for a molecule that only the PIN path names it ships
through its twin inside:func:`last_resort_scope` (``assembly.default_tier_offer.twin_last_resort_row``), so that a
wider tier never names less than the default tier does.

A -O-S-H name is not read back when the input also holds a -OOH group: the peroxol twin reads the name that puts the
suffix on the other group as well (``1-(hydroperoxy)propane-2-OS-thioperoxol`` for ``CC(OO)COS``), (:17460).
An anion twin needs an [S-O-] in the input for each group the name cites: 'ethanesulfenate' is read for the ester
``CCSOC`` too, and the book names that [(methylsulfanyl)oxy]ethane, 'not ethyl methanesulfenate' Mixed
chalcogen analogues of peroxides,:27900;:27912).

No twin: the anion of -O-S-H (OPSIN reads 'peroxolate' as the neutral peroxol, and no producer builds the
anion); the '-ium' names (no spelling of the cation is read, and the two audit rows were a protonation
site the name did not state); the dithio- family ('ethanedithioperoxol', which OPSIN reads itself, so a
name of it that OPSIN rejects is a name that is wrong); a multiplied suffix glued onto the italic
descriptor ('ethane-1,2-bisSO-thioperoxol', the formatting defect); a name with the suffix twice.
"""
import contextlib
import contextvars
import itertools
import re
from typing import Iterator, NamedTuple, Optional

from rdkit import Chem

from .opsin_ambiguity import ambiguity_messages

#: the rule ids of the two twin outcomes (``twin_verified:thioperoxol_sulfenic`` / ``..._peroxol``)
SULFENIC_RULE = "thioperoxol_sulfenic"
PEROXOL_RULE = "thioperoxol_peroxol"
THIOPEROXOL_TWIN_RULES = (SULFENIC_RULE, PEROXOL_RULE)

#: the suffix at the end of a word, led by the hyphen of the locants (or of the bare parent): a multiplied
#: form encloses the descriptor, 'bis(SO-thioperoxol)'; a form glued onto it ('bisSO-thioperoxol') is none.
#: The anion of a parent without a locant encloses its suffix, 'ethane(SO-thioperoxolate)',:41035)
_SUFFIX = re.compile(r"(?<=-)(?:(?P<mult>bis|tris|tetrakis)\((?P<kind>SO|OS)-thioperoxol(?P<ate>ate)?\)"
                     r"|(?P<kind2>SO|OS)-thioperoxol(?P<ate2>ate)?)(?=\s|$)"
                     r"|(?<=[a-z])\((?P<kind3>SO|OS)-thioperoxolate\)(?=\s|$)")
#: the twin suffix by (kind, anion): the anion of -O-S-H has none
_TWIN_SUFFIX = {("SO", False): "sulfenic acid", ("SO", True): "sulfenate", ("OS", False): "peroxol"}
#: the groups the -OS- suffix cites: a sulfur on an oxygen on a carbon
_O_S_H = Chem.MolFromSmarts("[#6][OX2+0][SX2H1+0]")
#: a -OOH group: the peroxol twin cannot tell it from a group whose sulfur was made an oxygen
_O_O_H = Chem.MolFromSmarts("[OX2][OX2H1]")
#: the anion of -S-OH: the group an anion twin cites
_S_O_ANION = Chem.MolFromSmarts("[SX2][OX1-]")
#: no input with more groups than this is checked (the subsets of the cited ones are tried)
_MAX_GROUPS = 8


#: True while a component of a mixture is named (``embedded_name``)
_EMBEDDED = contextvars.ContextVar("orthonym_thioperoxol_embedded", default=False)


@contextlib.contextmanager
def embedded_name():
    """Scope of naming a component or skeleton that is joined into a name OPSIN reads whole: a mixture component
    (``rules.adducts``, 'ethanol—(hydroxysulfanyl)methane (1/1)') and the isotope-stripped skeleton the
    isotope decorator labels (``rules.isotopes``,: every decorated candidate is read by OPSIN). The twin reads
    a whole name only and no OPSIN grammar reads the thioperoxol suffix, so a part named so would drop the whole name:
    inside this scope no twin ships and the part is named as before, by the general engine ('(hydroxysulfanyl)methane')."""
    token = _EMBEDDED.set(True)
    try:
        yield
    finally:
        _EMBEDDED.reset(token)


#: True while a general tier names a molecule it has left without a name, once more, with the twin of a neutral
#: thioperoxol name admitted (:func:`last_resort_scope`)
_LAST_RESORT = contextvars.ContextVar("orthonym_thioperoxol_last_resort", default=False)


@contextlib.contextmanager
def last_resort_scope():
    """Scope of the safety net of a general tier (``assembly.default_tier_offer.twin_last_resort_row``): the twin
    of a neutral thioperoxol name is admitted in the validity gate, as at the default tier, so a molecule for which
    every other path of the tier found no name keeps the PIN spelling that its twin reads back."""
    token = _LAST_RESORT.set(True)
    try:
        yield
    finally:
        _LAST_RESORT.reset(token)


def last_resort_active() -> bool:
    """Whether a call is inside:func:`last_resort_scope`."""
    return _LAST_RESORT.get()


def ships_at_general_tier(name: str) -> bool:
    """Whether a general tier ships the thioperoxol name ``name`` through its twin: the anion of -S-OH
    ('methane(SO-thioperoxolate)'), for which the engine has no other spelling, always, and a neutral name only
    inside:func:`last_resort_scope`: it gives way to the name OPSIN reads."""
    found = thioperoxol_twin(name)
    return found is not None and (found.anion or last_resort_active())


class ThioperoxolTwin(NamedTuple):
    """The twin of a thioperoxol name: its text, the rule, the groups the suffix cites, whether anion."""
    name: str
    rule: str
    count: int
    anion: bool


def thioperoxol_twin_outcome(outcome: Optional[str]) -> bool:
    """Whether ``outcome`` is the gate outcome of a thioperoxol name read back through its twin
    (``twin_verified:thioperoxol_sulfenic`` / ``twin_verified:thioperoxol_peroxol``)."""
    from ..metrics.provenance import twin_outcome
    return outcome in {twin_outcome(rule) for rule in THIOPEROXOL_TWIN_RULES}


def thioperoxol_twin(name: str) -> Optional[ThioperoxolTwin]:
    """The twin of the thioperoxol name ``name``, or None when ``name`` is no well-formed one that has a twin
    (see the module docstring): the suffix replaced and nothing else."""
    if not name or name.count("thioperoxol") != 1:
        return None
    found = _SUFFIX.search(name)
    if found is None:
        return None
    kind = found["kind"] or found["kind2"] or found["kind3"]
    anion = bool(found["ate"] or found["ate2"] or found["kind3"])
    suffix = _TWIN_SUFFIX.get((kind, anion))
    if suffix is None:
        return None
    multiplier = found["mult"]
    cited = {None: 1, "bis": 2, "tris": 3, "tetrakis": 4}[multiplier]
    twin = name[:found.start()] + (f"{multiplier}({suffix})" if multiplier else suffix) + name[found.end():]
    return ThioperoxolTwin(twin, SULFENIC_RULE if kind == "SO" else PEROXOL_RULE, cited, anion)


def _peroxol_references(smiles: str, count: int) -> Iterator[str]:
    """The input with the sulfur of ``count`` of its -O-S-H groups made an oxygen, for every choice of the
    groups the suffix cites (the others stay, as the prefixes 'sulfanyloxy' of the name)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.HasSubstructMatch(_O_O_H):
        return
    sulfurs = sorted({match[2] for match in mol.GetSubstructMatches(_O_S_H)})
    if count > len(sulfurs) or len(sulfurs) > _MAX_GROUPS:
        return
    for cited in itertools.combinations(sulfurs, count):
        edited = Chem.RWMol(mol)
        for sulfur in cited:
            edited.GetAtomWithIdx(sulfur).SetAtomicNum(8)
        try:
            peroxol = edited.GetMol()
            Chem.SanitizeMol(peroxol)
        except Exception:
            continue
        yield Chem.MolToSmiles(peroxol)


def thioperoxol_twin_round_trip(name: str, smiles: Optional[str]) -> Optional[str]:
    """The rule id of the twin of ``name`` when OPSIN reads that twin once to the FULL InChIKey of ``smiles``
    (the -S-OH rule) or of ``smiles`` with the cited -O-S-H sulfurs made oxygens (the -O-S-H rule); else None
    (no twin, a twin of another structure, an unavailable OPSIN, an ambiguous reading)."""
    twin = thioperoxol_twin(name)
    if twin is None or not smiles or _EMBEDDED.get():
        return None
    if twin.anion:   # 'sulfenate' reads an ester of the group too,:27912): an anion needs its [S-O-]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or len(mol.GetSubstructMatches(_S_O_ANION)) < twin.count:
            return None
    from ..namer import _shipped_name_round_trip
    references = [smiles] if twin.rule == SULFENIC_RULE else _peroxol_references(smiles, twin.count)
    if not any(_shipped_name_round_trip(twin.name, reference) == "verified" for reference in references):
        return None
    return None if ambiguity_messages(twin.name) else twin.rule
