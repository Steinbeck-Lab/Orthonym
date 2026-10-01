"""Name vocabulary that is never part of a Preferred IUPAC Name (PIN).

`non_pin_vocabulary(name)` returns the first token of ``name`` that a PIN never
contains, or ``None``. It is a CLOSED list of name forms the Blue Book names as
the non-preferred alternative of a PIN form, each cited below. It never rewrites
a name: callers use it to (a) keep a name that carries such a token off the PIN
label, and (b) let a producer that runs at the PIN tier decline such a name.

Each pattern is checked against the Blue Book's own PIN strings
(tests/unit/test_pin_vocabulary.py): a pattern that matches a BB PIN is wrong.
"""
from __future__ import annotations

import contextlib
import contextvars
import re
from typing import Callable, Optional

_HETERO_A = (r"(?:oxa|thia|selena|tellura|aza|phospha|arsa|stiba|bisma|sila|germa"
             r"|stanna|plumba|bora)")

# / (the Blue Book,:23682): a heteromonocycle with no
# more than ten ring members is named by the (extended) Hantzsch-Widman system in
# a PIN ('oxolane', not '1-oxacyclopentane'); skeletal replacement ('a') names are
# used only for rings of eleven or more members. Limited to O/S/Se/Te/N and P/As/
# Sb/Si/Ge/Sn/B rings (Hantzsch-Widman stems exist for all: 'phosphepane',
# 'borinane', 'silolane'; '3-propanoylphosphepan-2-one (PIN)', 'borinan-1-yl acetate
# (PIN)'), to rings without a triple bond (which the
# Hantzsch-Widman system cannot express: '1,2,5,6-tetrasilacycloocta-3,7-diyne
# (PIN)',:21091), and never a fusion prefix ('1,7-dioxacyclopenta[cd]indene').
_SMALL_RING_REPLACEMENT = re.compile(
    r"(?:oxa|thia|selena|tellura|aza|phospha|arsa|stiba|sila|germa|stanna|bora)"
    r"cyclo(?:prop|but|pent|hex|hept|oct|non|dec)"
    r"(?=a|-)(?!adec|acos|atriacont|atetracont)(?!a?\[)(?![^\s()\[\]{}]*yn)")

# (2) (the Blue Book, "The suffixes 'yl', 'ylidene', and 'ylidyne'
# are added to the name of the parent hydride... The locants for the atoms of
# free valences are as low as..."): a ring substituent prefix cites the locant of
# its free valence ('naphthalen-2-yl (preferred prefix) 2-naphthyl (contracted
# name)',:2865; '(furan-2-yl)... (2-furyl)',:7268). The contracted or
# locant-free spellings of the legacy ring table ('aziridinyl', 'pyridyl',
# 'furyl', 'naphthyl',...) occur in no PIN of the Blue Book. 'oxiranyl' is left
# out: oxirane has one kind of substitutable hydrogen, so its locant may be
# omitted,:2939), as in '*tert*-butyldi(methyl)(oxiranylmethoxy)
# silane (PIN)' (:18929).
_UNLOCANTED_RING_YL = re.compile(
    r"(?<![a-z])(?:pyridyl|furyl|thienyl|naphthyl|pyrrolyl|imidazolyl|pyrimidinyl|"
    r"pyrazinyl|pyridazinyl|oxanyl|piperidinyl|morpholinyl|piperazinyl|oxolanyl|"
    r"pyrrolidinyl|oxetanyl|azetidinyl|aziridinyl|indolyl|quinolinyl|"
    r"isoquinolinyl|benzofuranyl|benzothienyl|benzimidazolyl|purinyl|carbazolyl)")

# (the Blue Book, 'benzene (PIN) (not [6]annulene)') and
# ('phenyl'): the mancude six-membered carbocycle is benzene; 'cyclohexa-1,3,5-triene'
# is never the PIN spelling of it or of its substituent prefix.
_BENZENE_AS_TRIENE = re.compile(r"cyclohexa-\d+,\d+,\d+-trien")

# (the Blue Book): skeletal replacement ('a') nomenclature gives a
# PIN for an acyclic chain only when four or more heterounits are present. A block of
# 'a' prefixes on an ACYCLIC stem ('14-oxatetradecyl', '1,1-dioxo-1λ6-thia-2-azaethyl',
# '2-azaeth-1-yn-1-yl') with fewer than four heteroatom locants is not a PIN form.
# (A heterounit such as -SS- has two locants, so counting locants can only
# over-count heterounits: the check errs toward NOT flagging.)
_A_SEGMENT = (r"\d+(?:λ\d+)?(?:,\d+(?:λ\d+)?)*-"
              r"(?:di|tri|tetra|penta|hexa|hepta|octa|nona|deca)?" + _HETERO_A + r"-?")
_ACYCLIC_STEM = (r"(?:meth|eth|prop|but|pent|hex|hept|oct|non|dec|undec|dodec|tridec"
                 r"|tetradec|pentadec|hexadec|heptadec|octadec|nonadec|icos|henicos"
                 r"|docos|tricos|tetracos|pentacos|hexacos|heptacos|octacos|nonacos"
                 r"|triacont|hentriacont|dotriacont|tritriacont|tetratriacont"
                 r"|pentatriacont|hexatriacont|heptatriacont|octatriacont|nonatriacont"
                 r"|tetracont|hentetracont|dotetracont|tritetracont|tetratetracont"
                 r"|pentatetracont|hexatetracont|heptatetracont|octatetracont"
                 r"|nonatetracont|pentacont)")
_ACYCLIC_A_NAME = re.compile(
    r"(?<![A-Za-z])(?P<block>(?:" + _A_SEGMENT + r")+)" + _ACYCLIC_STEM
    + r"(?=a?n|a-|a\d|-\d|en|yn|yl|a,)")
_LOCANT = re.compile(r"\d+(?:λ\d+)?")

# / (the Blue Book 'formyl (preferred prefix)...
# oxomethyl';:30444 'formyl (preferred prefix) methanoyl oxomethyl'): 'oxomethyl'
# (and '-oxomethylene', for carbonyl,:29473) is never the PIN form. 'oxomethylidene'
# (=C=O, is left alone.
_OXOMETHYL = re.compile(r"oxomethyl(?!idene)")

# retained 'acetic acid' / 'formic acid' (PINs) and the names derived
# from them 'acetamide (PIN)', the Blue Book;:30444
# 'formyl... methanoyl'): 'ethanoic acid', 'ethanamide', 'ethanoyl', 'ethanoate'
# and the methane analogues are never the PIN spelling.
_UNRETAINED_C1C2_ACID = re.compile(
    r"(?<!m)ethan(?:amid|oat|oic acid|oyl)|methan(?:amid|oat|oic acid|oyl)")

# (the Blue Book-33007): -NH-CO-R cited as a prefix is named by
# the 'amido' method in a PIN ('4-formamidobenzoic acid (PIN)', not
# '4-(formylamino)benzoic acid'; 'acetamido', not 'acetylamino'). A carboxylic acyl
# ('...anoyl', '...enoyl', '...ynoyl', acetyl, formyl, benzoyl) on 'amino' is
# therefore not a PIN form. Acyls of carbonic, sulfur and phosphorus acids and of
# amidines ('carbamothioylamino (PIN)':33501, 'carbamimidoylamino':17832,
# 'oxamoylamino':17842) are not matched.
_ACYLAMINO = re.compile(
    r"(?:acetyl|formyl|benzoyl|[a-z]anoyl|[a-z]enoyl|[a-z]ynoyl)\)?\]?amino(?!carbonyl)")

# (the Blue Book "All preferred IUPAC names for esters are named
# by functional class nomenclature") and (:31698): an ester is cited as
# an 'acyloxy' prefix only when another group has priority for citation as the
# principal group. A name whose parent carries no suffix at all (it ends in the
# parent hydride's 'ane'/'ene'/'yne') has no such group, so an acyloxy prefix in it
# ('(2R)-3-(docosanoyloxy)-2-(tetradecanoyloxy)propane') is not the PIN form.
_ACYLOXY = re.compile(
    r"(?:acetyl|formyl|benzoyl|[a-z]anoyl|[a-z]enoyl|[a-z]ynoyl)\)?\]?\}?oxy")
_SUFFIX_FREE_PARENT = re.compile(r"(?:ane|ene|yne)$")
# The same rule, (:31698) with (:18160, esters are class
# 9; amides 11, nitriles 14, aldehydes 15, ketones 16, hydroxy compounds 17, amines
# 19, imines 20): a carboxylic or sulfur-acid ester (acyloxy, '...sulfonyloxy') or
# thioester (acylsulfanyl, '...sulfinylsulfanyl'; '*S*-(2-cyanoethyl) cyclohexane
# sulfinothioate (PIN)':31765, not '3-(cyclohexanesulfinylsulfanyl)propanenitrile') cited as
# a prefix while the principal group expressed by the suffix is JUNIOR to esters is
# not the PIN, which is the functional class ester name ('4-hydroxyphenyl acetate',
# not '4-(acetyloxy)phenol'; '2,5-dioxopyrrolidin-1-yl... ate' (:31663), not
# '1-(acyloxy)pyrrolidine-2,5-dione'). Only a name whose last word ends in such a
# suffix is read; ionic names (anions and cations are senior to esters) and names
# ending in an acid or an ester word are not.
_ACYL_ESTER_PREFIX = re.compile(
    r"(?:acetyl|formyl|benzoyl|[a-z]anoyl|[a-z]enoyl|[a-z]ynoyl|[a-z]sulfonyl"
    r"|[a-z]sulfinyl)\)?\]?\}?"
    r"(?:oxy|sulfanyl)(?!carbonyl)"
    # attached to carbon: not an acyl-X group on another heteroatom prefix
    # ('1-[(acetylsulfanyl)peroxy]propan-1-one (PIN)',:39499)
    r"(?![)\]}]*(?:peroxy|oxy|sulfanyl|disulfanyl|amino|imino|hydrazinyl|phosph"
    r"|sulfonyl|sulfinyl|silyl|boranyl|selanyl))")
_JUNIOR_TO_ESTER_END = re.compile(
    r"(?:amide|nitrile|carbaldehyde|al|one|ol|thiol|amine|imine)$")
# Branch review fixes: the same rule for the esters of the mononuclear noncarbon
# oxoacids of sulfur and phosphorus attached by oxygen -- 'sulfooxy' (-O-SO2-OH),
# 'phosphonooxy' (-O-PO(OH)2) and the phosphodiester link '(hydroxy)phosphoryl...
# oxy' (-O-PO(OH)-O-). (the Blue Book) and (:36327)
# cite such a group as a prefix only when "another substituent having priority over
# the... group for citation as principal group" is present ('3-(sulfooxy)propanoic
# acid (PIN)',:36488; '(phosphonooxy)acetic acid',:36333); otherwise the ester is
# named by functional class nomenclature,:35916-35918, 'methyl hydrogen
# sulfate (PIN)':35968; '(2S)-2,3-dihydroxypropyl dihydrogen phosphate':55148),
# and acids and esters are senior to amides, ketones, hydroxy compounds and amines
#,:18170-18192). Read with the same junior-suffix end as above.
# Branch review fixes: indicated hydrogen on a six-membered azine parent.
# / (the Blue Book): indicated hydrogen is cited only "consistent with
# the maximum number of noncumulative double bonds", and mancude pyridine, pyrimidine,
# pyrazine and pyridazine (an even ring of C and N) need none, so '1H-pyrimidine' is
# no parent; a free valence or ketone on such a ring takes ADDED hydrogen,
# (:24695) 'pyridin-1(2H)-yl (preferred prefix)' ('2,6-dioxo-1H-pyrimidin-
# 3-yl' for '2,4-dioxo-3,4-dihydropyrimidin-1(2H)-yl'). A fusion prefix ('6H-pyrazino
# [2,3-b]carbazole (PIN)') is not matched. Label-only, as the form below.
_AZINE_INDICATED_H = re.compile(
    r"(?<![A-Za-z0-9])\d+H-(?:pyridin|pyrimidin|pyrazin|pyridazin)(?=-|e(?![a-z]))")
# The ester prefixes of EVERY mononuclear noncarbon oxoacid, derived from one
# source: the acyl prefix the engine gives each noncarbon oxoacid group
# (``seniority.PREFIX_FORMS``: 'sulfo', 'sulfino', 'phosphono', 'arsono', 'stibono',
# 'borono', 'selenono',... and the acyl groups 'nitro' / 'nitroso' of nitric and
# nitrous acid) joined to 'oxy' or to a chalcogen analogue.
# (the Blue Book): substituent groups derived from the esters of nitric and
# nitrous acid "are named by concatenation by adding the acyl groups 'nitro' for
# -NO2 and 'nitroso' for -NO to the prefix 'oxy' or by substituting these acyl
# groups into substituent groups such as 'sulfanyl', 'selanyl', or 'tellanyl'"
# ('nitrooxy (preselected prefix)':36405); (:36327) for the B, N, P,
# As and Sb groups; 'pentyl nitrite (PIN)',:35922). The word list it
# replaces ('sulfooxy', 'phosphonooxy') left 'nitrooxy' at the PIN label under every
# junior class ('2-(nitrooxy)ethan-1-ol', PIN '2-hydroxyethyl nitrate').
_OXOACID_ESTER_PREFIX_RE: Optional["re.Pattern"] = None


def _oxoacid_ester_prefix_re() -> "re.Pattern":
    global _OXOACID_ESTER_PREFIX_RE
    if _OXOACID_ESTER_PREFIX_RE is None:
        from .seniority import PREFIX_FORMS
        # an acid group's acyl prefix is one lower-case word ending in 'o'
        # ('carboxy', 'carbamoyloxy', 'N-hydroxyamido' of the carbon acids are not)
        acyl = {v for k, v in PREFIX_FORMS.items()
                if k.endswith("_acid") and v and v.isalpha() and v.islower()
                and v.endswith("o")}
        acyl |= {PREFIX_FORMS["nitro"], PREFIX_FORMS["nitroso"]}
        alt = "|".join(sorted((re.escape(a) for a in acyl), key=len, reverse=True))
        _OXOACID_ESTER_PREFIX_RE = re.compile(
            rf"(?<![a-z])(?:{alt})(?:oxy|sulfanyl|selanyl|tellanyl)"
            r"|(?:\(hydroxy\)|hydroxy)phosphoryl[)\]}]*oxy")
    return _OXOACID_ESTER_PREFIX_RE

# Breadth job 3 review fixes: the class of the parent that carries an ester prefix,
# read from the END of the whole name as the complement of the classes senior to
# esters. (the Blue Book-18194) ranks radicals, radical ions,
# anions, zwitterions, cations (classes 1-6,:18164-18169), acids (7,:18170), and
# anhydrides (8) above esters (9,:18182); acid halides (10) and everything below
# them -- amides, hydrazides, imides, nitriles, aldehydes, ketones, hydroxy compounds,
# hydroperoxides, amines, imines (11-20) and the heterane classes 21-43 -- are
# junior. The _JUNIOR_TO_ESTER_END word list above names only some junior suffixes
# and missed retained names and functional modifiers ('4-(sulfooxy)benzaldehyde',
# '4-(sulfooxy)aniline', '3-(sulfooxy)propanehydrazide', '(sulfooxy)acetaldehyde
# oxime'); this reader recognises the senior heads instead, so a junior head it does
# not know is still junior. A name is headed by a class senior to esters when it
# ends in 'acid' (7) or '-anhydride' (8, 'dianhydride', 'thioanhydride'), in an anion or ester word '-ate', '-ite' or
# anionic '-ide' (4, 9: 'propanoate', 'ethyl... propanoate', 'methanide',
# 'boranuide'), or carries a cation word '-ium' (6, a salt or a cationic parent:
# 'N,N,N-trimethylethan-1-aminium'). '-amide', '-imide', '-hydrazide' and the
# functional-class words ('oxide', 'chloride', 'azide', 'cyanide',...) are not
# anions. A substituent-prefix string ('4-(sulfooxy)phenyl', '[(...)oxy]') has no
# head of its own and is not read.
_NAME_WORD_SPLIT = re.compile(r"[\s—]+")
_STOICHIOMETRY_WORD = re.compile(r"\(\d+(?:/\d+)+\)")
_PREFIX_STRING_END = re.compile(r"(?:yl|ylidene|ylidyne|ylene|y|o)$")
_NON_ANION_IDE_END = re.compile(
    r"(?:amide|imide|hydrazide|oxide|sulfide|selenide|telluride|fluoride|chloride"
    r"|bromide|iodide|azide|cyanide|hydride|nitride|carbide)$")


def _head_senior_to_esters(name: str) -> Optional[bool]:
    """True when ``name`` is headed by a class senior to esters (classes 1-9),
    False when by a junior class (10-43), None for a substituent-prefix string."""
    raw = [w for w in _NAME_WORD_SPLIT.split(name.strip()) if w]
    while raw and _STOICHIOMETRY_WORD.fullmatch(raw[-1]):
        raw.pop()
    words = [w.rstrip(")]}") for w in raw]
    words = [w for w in words if w]
    if not words:
        return None
    last = words[-1]
    if last == "acid" or last.endswith("anhydride"):
        return True
    if any(w.endswith("ium") for w in words):
        return True
    if last.endswith(("ate", "ite")) and not last.endswith("hydrate"):
        return True
    if last.endswith("ide") and not _NON_ANION_IDE_END.search(last):
        return True
    if len(words) == 1 and _PREFIX_STRING_END.search(last):
        return None
    return False

# (the Blue Book 'propan-1-yl propyl (preferred prefix)'): a
# saturated acyclic (or monocyclic, prefix with its free valence at
# position 1 drops the locant and 'an': 'propyl', 'cyclohexyl'. Polyvalent forms
# ('ethan-1-yl-2-ylidene',:15975), von Baeyer / spiro / retained cage prefixes
# ('bicyclo[2.2.2]octan-1-yl', 'adamantan-1-yl') and the fixed-numbering 'a' chains
# ('2,4,6,8-tetrasilanonan-1-yl',:22658) are not matched.
_ALKAN_1_YL = re.compile(
    r"(?<![\]a])(?<!un)(?<!do)(?<!tri)(?<!m)(?:cyclo)?"
    r"(?:meth|eth|prop|but|pent|hex|hept|oct|non|dec|undec|dodec)an-1-yl(?![a-z-])")

# (the Blue Book,:24233 'decahydronaphthalene (PIN)
# bicyclo[4.4.0]decane') and (:23710 "Fusion nomenclature gives
# preferred IUPAC names only to compounds having at least two rings of at least
# five or more members... When fusion names are not allowed, unsaturated von
# Baeyer ring system names are preferred IUPAC names"): a ring system whose von
# Baeyer descriptor has ONLY zero-atom bridges is ortho- (or ortho- and peri-)
# fused, so with two or more rings of five or more members it is named by fusion
# nomenclature (with hydro prefixes) in a PIN -- 'bicyclo[4.4.0]decane',
# 'tetracyclo[8.7.0.0^2,7.0^11,15]heptadecane' (the steroid skeleton) are not PIN
# forms; 'bicyclo[4.2.0]octa-1,3,5,7-tetraene (PIN)' (:23725) is.
_VON_BAEYER = re.compile(
    r"(?:bi|tri|tetra|penta|hexa|hepta|octa|nona|deca|undeca|dodeca)cyclo"
    r"\[(\d+)\.(\d+)\.(\d+)((?:\.\d+\^\d+,\d+)*)\]")
_VB_SECONDARY = re.compile(r"\.(\d+)\^(\d+),(\d+)")


def _ortho_fused_von_baeyer(m) -> bool:
    a, b, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if c != 0:
        return False
    secondary = [(int(n), int(x), int(y))
                 for n, x, y in _VB_SECONDARY.findall(m.group(4) or "")]
    if any(n != 0 for n, _, _ in secondary):
        return False
    n_atoms = a + b + 2
    if not secondary:
        return a >= 3 and b >= 3
    try:
        from rdkit import Chem
        rw = Chem.RWMol()
        for _ in range(n_atoms):
            rw.AddAtom(Chem.Atom(6))
        for i in range(n_atoms):
            rw.AddBond(i, (i + 1) % n_atoms, Chem.BondType.SINGLE)
        rw.AddBond(0, a + 1, Chem.BondType.SINGLE)
        for _, x, y in secondary:
            if not (1 <= x <= n_atoms and 1 <= y <= n_atoms):
                return False
            if rw.GetBondBetweenAtoms(x - 1, y - 1) is None:
                rw.AddBond(x - 1, y - 1, Chem.BondType.SINGLE)
        mol = rw.GetMol()
        mol.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(mol)
        rings = [set(r) for r in Chem.GetSymmSSSR(mol)]
        n_rings = mol.GetNumBonds() - mol.GetNumAtoms() + 1
    except Exception:  # noqa: BLE001 - an unparsable descriptor is not flagged
        return False
    # Ortho- (and ortho- and peri-) fused: any two rings share at most two atoms
    # (one bond), and the smallest set of rings is unique. A system in which two
    # rings share three or more atoms (tricyclo[3.2.0.0^2,6]heptane) is bridged,
    # whatever its bridge lengths, and keeps its von Baeyer name.
    if len(rings) != n_rings:
        return False
    if any(len(rings[i] & rings[j]) > 2
           for i in range(len(rings)) for j in range(i + 1, len(rings))):
        return False
    return sum(1 for r in rings if len(r) >= 5) >= 2

# / (the Blue Book, "cyclic phane systems > fused
# ring systems > bridged fused systems > non-fused bridged systems";:23883
# "the bridged fused ring name is preferred to the von Baeyer name",
# 'hexadecahydro-1H-8,12-methanobenzo[13]annulene (PIN)' for tricyclo[12.3.1.
# 0^5,10]octadecane): a bridged von Baeyer system in which a zero-atom bridge
# ortho-fuses two rings of five or more members (they share exactly one bond)
# has a bridged fused name, which is the PIN -- 'tricyclo[5.2.1.0^2,6]
# decane' (a 4,7-methanoindene). Systems with non-zero secondary bridges are not
# read (not flagged); a system with no such fused pair ('bicyclo[2.2.1]heptane',
# 'tricyclo[2.2.1.0^2,6]heptane') keeps its von Baeyer name. A fused pair that holds
# every atom of a system with more rings closes the other rings by bonds only, and
# (:14025) "An atom or group of atoms is named as a bridge": no bridged
# fused name from that pair ('tricyclo[4.4.0.0^5,10]decane', a decalin with a
# cyclobutane closed across it, keeps its von Baeyer name).


def _bridged_fused_von_baeyer(m) -> bool:
    a, b, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
    secondary = [(int(n), int(x), int(y))
                 for n, x, y in _VB_SECONDARY.findall(m.group(4) or "")]
    if any(n != 0 for n, _, _ in secondary):
        return False
    if c != 0 and not secondary:
        return False
    try:
        from rdkit import Chem
        main = a + b + 2
        n_atoms = main + c
        rw = Chem.RWMol()
        for _ in range(n_atoms):
            rw.AddAtom(Chem.Atom(6))
        for i in range(main):
            rw.AddBond(i, (i + 1) % main, Chem.BondType.SINGLE)
        # the main bridge, numbered from the atom nearer to main bridgehead 1
        prev = 0
        for k in range(c):
            rw.AddBond(prev, main + k, Chem.BondType.SINGLE)
            prev = main + k
        if rw.GetBondBetweenAtoms(prev, a + 1) is None:
            rw.AddBond(prev, a + 1, Chem.BondType.SINGLE)
        for _, x, y in secondary:
            if not (1 <= x <= n_atoms and 1 <= y <= n_atoms):
                return False
            if rw.GetBondBetweenAtoms(x - 1, y - 1) is None:
                rw.AddBond(x - 1, y - 1, Chem.BondType.SINGLE)
        mol = rw.GetMol()
        mol.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(mol)
        rings = [set(r) for r in Chem.GetSymmSSSR(mol)]
    except Exception:  # noqa: BLE001 - an unparsable descriptor is not flagged
        return False
    n_rings = mol.GetNumBonds() - mol.GetNumAtoms() + 1
    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            if len(rings[i]) < 5 or len(rings[j]) < 5:
                continue
            shared = rings[i] & rings[j]
            if len(shared) == 2:
                x, y = shared
                if mol.GetBondBetweenAtoms(x, y) is not None:
                    if len(rings[i] | rings[j]) == mol.GetNumAtoms() and n_rings > 2:
                        # the other rings are closed by bonds only: no bridged fused
                        # name from this pair, so the von Baeyer name can be the PIN,
                        # but only in the form of its descriptor
                        if not _vb_superscripts_lowest(m):
                            return True
                        continue
                    return True
    return False


def _vb_superscripts_lowest(m) -> bool:
    """ (the Blue Book): "The superscript locants for the secondary
    bridges must be as low as possible when considered as a set in ascending numerical
    order". True when the descriptor's zero-atom secondary bridges carry the lowest
    superscript set over every numbering of its own bicyclic system [a.b.c] (main ring
    and main bridge numbered as numbers them); False when a lower set exists or
    the descriptor has a secondary bridge with atoms (not judged here)."""
    a, b, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
    secondary = [(int(n), int(x), int(y))
                 for n, x, y in _VB_SECONDARY.findall(m.group(4) or "")]
    if not secondary or any(n != 0 for n, _, _ in secondary):
        return False
    try:
        from rdkit import Chem
        main = a + b + 2
        rw = Chem.RWMol()
        for _ in range(main + c):
            atom = Chem.Atom(6)
            atom.SetNoImplicit(True)
            rw.AddAtom(atom)
        for i in range(main):
            rw.AddBond(i, (i + 1) % main, Chem.BondType.SINGLE)
        prev = 0
        for k in range(c):
            rw.AddBond(prev, main + k, Chem.BondType.SINGLE)
            prev = main + k
        if rw.GetBondBetweenAtoms(prev, a + 1) is None:
            rw.AddBond(prev, a + 1, Chem.BondType.SINGLE)
        bicyclic = rw.GetMol()
        bicyclic.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(bicyclic)
        for _, x, y in secondary:
            if not (1 <= x <= main + c and 1 <= y <= main + c):
                return False
            if rw.GetBondBetweenAtoms(x - 1, y - 1) is None:
                rw.AddBond(x - 1, y - 1, Chem.BondType.SINGLE)
        system = rw.GetMol()
        system.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(system)
        main_bonds = {frozenset((bd.GetBeginAtomIdx(), bd.GetEndAtomIdx()))
                      for bd in bicyclic.GetBonds()}
        own = sorted(v for _, x, y in secondary for v in (x, y))
        for emb in system.GetSubstructMatches(bicyclic, uniquify=False, useChirality=False,
                                              maxMatches=10000):
            locant = {s: q + 1 for q, s in enumerate(emb)}
            sup = sorted(locant[e] for bd in system.GetBonds()
                         for e in (bd.GetBeginAtomIdx(), bd.GetEndAtomIdx())
                         if frozenset((locant[bd.GetBeginAtomIdx()] - 1,
                                       locant[bd.GetEndAtomIdx()] - 1)) not in main_bonds)
            if sup < own:
                return False
        return True
    except Exception:  # noqa: BLE001 - an unparsable descriptor is not judged
        return False

# (a) (the Blue Book): hyphens separate locants from words. A letter
# (or a closing bracket) followed directly by a digit ('2,2-dimethyl3,4-dihydro',
# '5-hydroxy1H-indol', '...yl]1-ethoxy') is a missing hyphen, never a PIN spelling.
_MISSING_HYPHEN = re.compile(r"[a-z\]][0-9]")

# (the Blue Book): "In preferred IUPAC names, stereodescriptors are
# placed immediately at the front of the part of the name to which they relate...
# When they relate to substituent groups, they are cited at the front of the
# corresponding prefix. They are preceded by a numerical or letter locant". Each
# descriptor of a set therefore sits on its own locant of ONE numbering; a set that
# repeats a locant ('(1S,1S,2S,4R)-N-[1-(bicyclo[2.2.1]heptan-2-yl)ethyl]...' hoists
# the centres of two components to the front) is never a PIN spelling.
_STEREO_SET = re.compile(r"\(([0-9A-Za-z',*]+)\)")
_STEREO_ITEM = re.compile(r"^(\d+[a-z]?'*)?([RSEZrsPM]\*?)$")


def _duplicated_stereo_locant(name: str) -> Optional[str]:
    for m in _STEREO_SET.finditer(name):
        items = m.group(1).split(",")
        parsed = [_STEREO_ITEM.match(it) for it in items]
        if not all(parsed):
            continue
        locs = [p.group(1) for p in parsed if p.group(1)]
        if len(locs) != len(set(locs)):
            return m.group(0)
    return None


# (the Blue Book 'phenoxy (substitution allowed...) phenyloxy';
#:27689 '2-methylpropoxy (preferred prefix)';:24591 'benzyloxy (preferred prefix)
# phenylmethoxy'): the ether prefixes of methyl, ethyl, propyl, butyl and phenyl are
# contracted ('methoxy'... 'phenoxy'), substituted or not, so an enclosed
# '(...phenyl)oxy', '(...methyl)oxy' is never the PIN spelling. '(butan-2-yl)oxy'
# (:27639, a locant-bearing prefix) is not matched.
_UNCONTRACTED_OXY = re.compile(r"(?:phen|meth|eth|prop|but)yl[)\]}]oxy")

# / (the Blue Book 'propan-2-yl (preferred prefix) 1-methylethyl';
#:22744 '1-hydroxypropan-2-yl (preferred prefix)... [not 1-(hydroxymethyl)ethyl]'):
# a substituent prefix is named on its longest chain. A (substituted) methyl on an
# ethyl prefix always extends that chain to three carbons, so '1-methylethyl',
# '{1-[(adamantan-1-yl)methyl]ethyl}', '2-(chloromethyl)ethyl' and '1-benzylethyl'
# (benzyl is phenylmethyl) are never PIN prefixes. (On a longer chain the methyl need not extend it: '2-bromo-2-
# (bromomethyl)butyl (preferred prefix)',:15787, is not matched.)
_METHYL_ON_ETHYL = re.compile(r"(?:methyl|benzyl)[)\]}]*eth(?:yl|enyl)")

# / (the Blue Book 'adamantane (PIN) tricyclo[3.3.1.1^3,7]
# decane';:16374 'adamantan-2-yl (preferred prefix) tricyclo[3.3.1.1^3,7]decan-2-yl';
#:11234 '...[1]phosphaadamantane] (PIN)'): the von Baeyer name of the adamantane
# skeleton is never the PIN.
_ADAMANTANE_VB = re.compile(r"tricyclo\[3\.3\.1\.1\^?\{?3,7\}?\]dec")

# (the Blue Book "The prefix 'hydroperoxy' is formed... Method (1)
# leads to preferred IUPAC names";:40426, 'dioxidanyl' is the systematic name of
# the HOO radical, whose PIN is 'hydroperoxyl'): 'dioxidanyl' is never PIN vocabulary.
_DIOXIDANYL = re.compile(r"dioxidanyl")
# (the Blue Book): "N,N'-methylenediethanamine (PIN)
# N,N'-diethylmethanediamine" -- two amines on ONE carbon, each nitrogen carrying
# the same substituent, are identical N-substituted parents on a symmetric linker,
# and (:23180) prefers the multiplicative name for "multiple occurrences
# of identical parent structures, other than alkanes". The substitutive geminal
# form ('N,N'-diethylmethanediamine', 'N1,N'1-diethylethane-1,1-diamine': one
# multiplied prefix over N<k> and N'<k> at the same position k of a CARBON
# chain's geminal diamine) is its non-preferred alternative. Scoped to carbon
# parents: the Blue Book keeps the substitutive name for the Group-13/14 hydrides
# ('1-methyl-N,N'-disilylsilanediamine (PIN)':38182, 'N,N'-bis(butylboranyl)
# boranediamine (PIN)':37425) and for ureas, guanidines and amides. Label-only:
# the name stays (it round-trips), it is not certified as the PIN.
_GEMINAL_N_N_PRIME_MULTIPLIED = re.compile(
    r"(?<![A-Za-z])N(\d*)(?:,N\1)*,N'\1(?:,N'\1)*-"
    r"(?:di|tri|tetra|bis|tris|tetrakis)\S*"
    r"(?:methanediamine|[a-z]ane-(\d+),\2-diamine)$")

# / (the Blue Book "Acyl prefixes formed from acyclic parent
# hydrocarbons and prefixes such as 'oxo'... for example, '1-oxopropyl' are not
# preferred prefixes";:30612 'propanoyl (preferred prefix) propionyl 1-oxopropyl').
# '1-oxopropan-2-yl' (an aldehyde-bearing prefix, free valence at 2) is not matched.
_OXO_ALKYL_ACYL = re.compile(
    r"(?<![\d,])1-oxo(?:eth|prop|but|pent|hex|hept|oct|non|dec|undec|dodec)yl")

# (the Blue Book, "Parentheses are used around compound... and
# complex... prefixes") and the ester example:31765 'S-(2-cyanoethyl)... (PIN)':
# a locant followed by a compound prefix standing as its own component
# ('2-cyclopropylamino-2-oxo...', '4-methylsulfanyl-'), or an element locant
# followed by a numbered (complex) prefix ('S-8-amino-8-oxo...octyl'), lacks the
# parentheses a PIN gives it.
_UNENCLOSED_COMPOUND = re.compile(
    r"(?<![\w'\]\)}])\d+[a-z]?-(?:cyclo)?[a-z]{3,}yl(?:amino|oxy|sulfanyl|sulfinyl"
    r"|sulfonyl)(?=[-)\]}]|$)")
_UNENCLOSED_ELEMENT_LOCANT = re.compile(
    r"(?<![A-Za-z])(?:S|O|Se|Te|N'*)-\d+[a-z]?(?:,\d+[a-z]?)*-")

# / (the Blue Book): a compound or complex prefix is
# multiplied by 'bis', 'tris'... ('3-[2,3-bis(carboxymethyl)naphthalen-1-yl]
# propanoic acid (PIN)'); 'di(carboxymethyl)' multiplies a compound prefix with
# 'di'. Only parenthesised contents without locants are read; 'di(propan-2-yl)'
# ('di(methyl)', simple prefixes the BB multiplies with 'di') is not matched.
_DI_COMPOUND = re.compile(r"(?<![a-z])(?:di|tri|tetra)\(([a-z]+)\)")


def _di_compound(name: str) -> Optional[str]:
    try:
        from ..assembly.naming_utils import is_complex_substituent
    except Exception:  # noqa: BLE001
        return None
    for m in _DI_COMPOUND.finditer(name):
        if is_complex_substituent(m.group(1)):
            return m.group(0)
    return None


# (multiplicative nomenclature) with: identical parent acids on
# one central unit are multiplied -- '2,2'-(naphthalene-2,3-diyl)diacetic acid (PIN)',
# '2,2'-azanediyldiacetic acid' -- so an acetic acid (acetate) parent carrying a
# 'carboxymethyl' (another acetic acid unit) is not the PIN spelling.
_CARBOXYMETHYL_ON_ACETIC = re.compile(r"carboxymethyl.*acet(?:ic acid|ate)$")

# (the Blue Book, a -CO-NH2 group that is not the suffix):
# "For generation of IUPAC preferred names, method (1) is preferred for chains"
# (:32928) -- 'amino' and 'oxo' on the terminal atom of the carbon chain: '5-(2-
# amino-2-oxoethyl)furan-2-carboxylic acid (PIN)' (:32940), not '(2) 5-(carbamoyl
# methyl)furan-2-carboxylic acid' (:32941). A (substituted) carbamoyl cited on a
# 'methyl' carrier is that carbon chain written by method (2).
_CARBAMOYL_ON_METHYL = re.compile(r"carbamoyl[)\]}]*methyl")

# (the Blue Book, "if any locants are essential for defining the
# structure of the parent structure... then all locants must be cited") with
# (c) (:2913, the locant '1' is omitted only "in monosubstituted
# homogeneous monocyclic rings"): '3-(2-iminopropyl)cyclohexane-1-carboxylic acid
# (PIN)' (:26549). A cycloalkane carrying a '-carbox...' / '-carbo...' suffix
# written without its locant while a numbered prefix sits on the same ring
# ('4,4-difluoro-N-[...]cyclohexanecarboxamide') is not the PIN spelling. Only the
# top level of the last word is read, where every numeric locant is a ring
# position of that parent ('N-hydroxycyclohexanecarboxamide (PIN)',:30150, has
# none).
_UNLOCANTED_RING_SUFFIX = re.compile(
    r"cyclo[a-z]+ane(?:carbox|carbo|carbal|carbon)[a-z]*$")


# / (the Blue Book seniority order of classes; "the
# principal chain... has the maximum number of skeletal atoms... bearing the
# principal characteristic group";:31698, an ester is a prefix only
# "in the presence of a group having priority for citation as the principal group"):
# a substitutive name whose parent carries no suffix at all -- it ends in the parent
# hydride ('...propyl)benzene', '...-2-oxopyrrolidine', '...-4,5-dihydro-1,3-
# thiazole') -- has no principal characteristic group, so none of its prefixes may
# denote a group that a suffix expresses (hydroxy -> -ol, oxo -> -one, amino /
# anilino -> -amine, imino -> -imine, amido / carbamoyl -> -amide, cyano ->
# -nitrile, formyl -> -al, carboxy -> -oic acid): '(2-acetamidopropyl)benzene' is not
# the PIN of N-(1-phenylpropan-2-yl)acetamide. 'isocyano' (a prefix-only class) and
# ring names that merely contain 'oxo' ('1,3-dioxolane', '1,5-dioxocane') are not
# matched; names with a functional-class word (a space) are not read.
_PARENT_HYDRIDE_END = re.compile(r"(?:ane|ene|yne|ole|ine|an)$")
_NOT_PARENT_HYDRIDE_END = re.compile(
    r"(?:amine|imine|aniline|guanidine|amidine|hydrazine|hydroxylamine"
    r"|ylidene|ylidyne|ylene)$")
_SUFFIX_GROUP_PREFIX = re.compile(
    r"hydroxy|(?<![a-z])(?:\d+(?:,\d+)*-)?(?:di|tri|tetra|penta|hexa)?oxo(?!l|c|n[ai]n)"
    r"|amino|anilino|imino|amido|carbamoyl|(?<!iso)cyano|formyl|carboxy")


def _suffix_group_on_parent_hydride(name: str) -> Optional[str]:
    if " " in name or not _PARENT_HYDRIDE_END.search(name):
        return None
    if _NOT_PARENT_HYDRIDE_END.search(name):
        return None
    m = _SUFFIX_GROUP_PREFIX.search(name)
    return m.group(0) if m else None


# (the Blue Book, indicated hydrogen; '2H-pyran-2-one',
# '1H-pyrrole-2,5-dione', '1,3-dihydro-2H-imidazol-2-one'): a mancude parent that
# itself needs indicated hydrogen (pyrrole, pyran, the azoles, azepine, indole,
# indene, purine,...) keeps it when a ring position becomes C=O / C=S / C=NH, so
# a ring ketone written on the bare stem ('...pyrrol-2-one') names no definite
# parent and is never the PIN spelling. Parents that need no indicated hydrogen
# (furan, pyridine, naphthalene: 'furan-2,5-dione (PIN)', 'naphthalene-1,4-dione')
# are not listed. An indicated hydrogen or 'hydro' anywhere in the ring's own word
# (fusion brackets included, '4H-pyrido[1,2-a]pyrimidin-4-one') is accepted.
_IH_STEM = (r"(?:pyrrol|pyrazol|imidazol|triazol|tetrazol|pyran|thiopyran|azepin"
            r"|oxazin|thiazin|indol|isoindol|indazol|benzimidazol|inden|purin)")
_BARE_IH_RING_ONE = re.compile(
    _IH_STEM + r"e?-\d+[a-z]?(?:,\d+[a-z]?)*-(?:di|tri)?(?:one|thione|imine)")
_FUSION_SEGMENT = r"(?:(?:[\d']+(?:,[\d']+)*-)?[a-z]{1,3}'*(?:,[a-z]{1,3}'*)*|[\d']+(?:,[\d']+)*)"
_FUSION_DESCRIPTOR = re.compile(
    r"^" + _FUSION_SEGMENT + r"(?::" + _FUSION_SEGMENT + r")*$")


def _ring_word_before(name: str, end: int) -> str:
    """The part of ``name`` before ``end`` that belongs to the same ring parent:
    back to the enclosing mark or space that opens it, skipping fusion
    descriptors ('[1,2-a]')."""
    i = end
    while i > 0:
        ch = name[i - 1]
        if ch == "]":
            j = name.rfind("[", 0, i - 1)
            if j >= 0 and _FUSION_DESCRIPTOR.match(name[j + 1:i - 1]):
                i = j
                continue
            break
        if ch in "()[{} ":
            break
        i -= 1
    return name[i:end]


def _bare_ih_ring_one(name: str) -> Optional[str]:
    for m in _BARE_IH_RING_ONE.finditer(name):
        word = _ring_word_before(name, m.start())
        if re.search(r"\d+[a-z]?H[-,]|hydro(?!x|peroxy)", word):
            continue
        return m.group(0)
    return None


def _top_level(word: str) -> str:
    depth, out = 0, []
    for ch in word:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif depth == 0:
            out.append(ch)
    return "".join(out)


def _unlocanted_ring_suffix(name: str) -> Optional[str]:
    words = name.split(" ")
    word = words[-1]
    if word in ("acid", "chloride", "bromide", "iodide", "fluoride") and len(words) > 1:
        word = words[-2]
    top = _top_level(word)
    if not _UNLOCANTED_RING_SUFFIX.search(top):
        return None
    if re.search(r"(?:^|[-,])\d+[a-z]?(?:,\d+[a-z]?)*-", top):
        return _UNLOCANTED_RING_SUFFIX.search(top).group(0)
    return None


def non_pin_vocabulary(name: Optional[str], *, label_forms: bool = True) -> Optional[str]:
    """The first token of ``name`` that no PIN contains, or ``None``.

    ``label_forms=False`` leaves out the forms added by the branch review fixes
    (the S/P oxoacid ester prefixes beside a junior suffix, indicated hydrogen on a
    six-membered azine parent) -- used by the PIN tier's
    promotion re-run to decide which of its names ship at all, a decision these
    label-only forms must not change: every re-run name is labelled below the PIN
    anyway (``metrics.provenance.record_pin_promotion_rerun``), so they would only
    remove a name the default tier shipped before."""
    if not name:
        return None
    for rx in (_SMALL_RING_REPLACEMENT, _BENZENE_AS_TRIENE, _OXOMETHYL,
               _UNRETAINED_C1C2_ACID, _ACYLAMINO, _ALKAN_1_YL, _MISSING_HYPHEN,
               _UNCONTRACTED_OXY, _METHYL_ON_ETHYL, _ADAMANTANE_VB, _DIOXIDANYL,
               _OXO_ALKYL_ACYL, _UNENCLOSED_COMPOUND, _UNENCLOSED_ELEMENT_LOCANT,
               _GEMINAL_N_N_PRIME_MULTIPLIED, _UNLOCANTED_RING_YL, _CARBAMOYL_ON_METHYL):
        m = rx.search(name)
        if m:
            return m.group(0)
    _uls = (_unlocanted_ring_suffix(name) or _suffix_group_on_parent_hydride(name)
            or _bare_ih_ring_one(name) or _di_compound(name))
    if not _uls:
        _m = _CARBOXYMETHYL_ON_ACETIC.search(name)
        _uls = "carboxymethyl" if _m else None
    if _uls:
        return _uls
    if _SUFFIX_FREE_PARENT.search(name):
        m = _ACYLOXY.search(name)
        if m:
            return m.group(0)
    if label_forms:
        # the label reads the head class from (``_head_senior_to_esters``)
        if _head_senior_to_esters(name) is False:
            m = (_ACYL_ESTER_PREFIX.search(name)
                 or _oxoacid_ester_prefix_re().search(name))
            if m:
                return m.group(0)
    elif " " not in name and _JUNIOR_TO_ESTER_END.search(name):
        # the promotion re-run's keep-or-drop decision keeps its own form
        m = _ACYL_ESTER_PREFIX.search(name)
        if m:
            return m.group(0)
    if label_forms:
        m = _AZINE_INDICATED_H.search(name)
        if m:
            return m.group(0)
    for m in _VON_BAEYER.finditer(name):
        if _ortho_fused_von_baeyer(m) or _bridged_fused_von_baeyer(m):
            return m.group(0)
    for m in _ACYCLIC_A_NAME.finditer(name):
        if len(_LOCANT.findall(m.group("block"))) < 4:
            return m.group(0)
    return _duplicated_stereo_locant(name)


def identical_parent_units(mol, sys_a, sys_b) -> bool:
    """False only when two ring systems of the same skeleton are shown NOT to be
    identical parent units of a multiplicative name; True otherwise (identical, or
    undecidable here: the caller keeps treating them as a multiplicative candidate).

     "Preferred IUPAC multiplicative names" (the Blue Book): a
    multiplicative PIN needs "(2) the multiplicative groups, other than the central
    multiplicative group, are symmetrically substituted; and (3) the locants of all
    substituent groups on the identical parent structures, including suffix groups,
    are identical" (:23183-23184); "all substituent groups, including the principal
    characteristic groups must be identical and have the same locant" (:23186). When
    they are not, "preferred IUPAC names are generated by substitutive nomenclature"
    ,:6231): '4-chloro-2-[(3-cyanophenyl)methyl]benzonitrile (PIN)'
    (:6235), '1-bromo-3-[(3-chlorophenyl)methyl]benzene (PIN)' (:6279); units that
    differ only in configuration are not identical either (:22587).

    A unit is its ring system with every atom still connected to it once the atoms
    of the shortest path to the other system (the linker) are removed; the linker
    atom next to the unit is written as a dummy atom, so the attachment position is
    part of the comparison. Units whose canonical isomeric SMILES differ are not
    identical. Undecidable (True): the systems are bonded directly (a ring
    assembly), the two units share an atom (a second path joins them), or any
    error."""
    return _unit_identity(mol, sys_a, sys_b) is not False


def _unit_identity(mol, sys_a, sys_b, out=None):
    """True / False when the two decorated units are / are not identical (see
    ``identical_parent_units``); None when undecidable here (a ring assembly, units
    joined by a second path, no path, an error). ``out`` (a dict), when given,
    receives ``linker`` (the atoms of the shortest path between the two systems,
    outside them) and ``units`` once they are known."""
    try:
        from rdkit import Chem

        sys_a, sys_b = set(sys_a), set(sys_b)
        if sys_a & sys_b:
            return None
        # breadth-first from every atom of A to the first atom of B
        prev = {a: None for a in sys_a}
        frontier = list(sys_a)
        hit = None
        while frontier and hit is None:
            nxt = []
            for cur in frontier:
                for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
                    ni = nb.GetIdx()
                    if ni in prev:
                        continue
                    prev[ni] = cur
                    if ni in sys_b:
                        hit = ni
                        break
                    nxt.append(ni)
                if hit is not None:
                    break
            frontier = nxt
        if hit is None:
            return None
        path = [hit]
        while prev[path[-1]] is not None:
            path.append(prev[path[-1]])
        path.reverse()                     # A-atom, linker..., B-atom
        linker = [i for i in path if i not in sys_a and i not in sys_b]
        if not linker:
            return None                    # directly bonded: a ring assembly
        if out is not None:
            out["linker"] = list(linker)
        cut = set(linker)

        def unit(seed_sys):
            seen, stack = set(seed_sys), list(seed_sys)
            while stack:
                cur = stack.pop()
                for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
                    ni = nb.GetIdx()
                    if ni not in seen and ni not in cut:
                        seen.add(ni)
                        stack.append(ni)
            return seen

        unit_a, unit_b = unit(sys_a), unit(sys_b)
        if out is not None:
            out["units"] = (unit_a, unit_b)
        if unit_a & unit_b:
            return None

        def key(unit_atoms, dummy_idx):
            rw = Chem.RWMol(mol)
            d = rw.GetAtomWithIdx(dummy_idx)
            d.SetAtomicNum(0)
            d.SetFormalCharge(0)
            d.SetIsotope(0)
            d.SetNoImplicit(True)
            d.SetNumExplicitHs(0)
            d.SetIsAromatic(False)
            rw.UpdatePropertyCache(strict=False)
            return Chem.MolFragmentToSmiles(rw, atomsToUse=sorted(unit_atoms | {dummy_idx}),
                                            canonical=True, isomericSmiles=True)

        return key(unit_a, linker[0]) == key(unit_b, linker[-1])
    except Exception:  # noqa: BLE001 - unknown
        return None


def multiplicative_candidate(mol) -> bool:
    """True when the molecule may take a multiplicative PIN, so a
    substitutive name built by the PIN tier's re-run is not known to be its PIN.

    The molecule-level form of the strict producers' ring-branch guard (suite fix
    j6, ``assembly.substituent_naming._ring_branch_may_need_other_parent``):
    two or more ring systems with the same skeleton, and either
    - the principal characteristic group sits on (or directly on) two of them
      ('2,2'-methylenedibenzonitrile', the Blue Book; '2,2'-[ethane-1,2-
      diylbis(oxymethylene)]di(cyclohexane-1-carboxylic acid)'), or
    - there is no principal characteristic group and that skeleton is the senior
      ring system of the molecule: N-heterocycle, heterocycle, more rings,
      more ring atoms) -- '1,1'-(1-chloroethane-1,2-diyl)dibenzene', not
      '(1-chloro-2-phenylethyl)benzene'.
    A skeleton that is only a substituent of a senior parent ('2,5-bis[(4-methoxy
    phenyl)methyl]-4-methylpyrimidine') or a principal group on the linking chain
    ('bis(4-chlorophenyl)methanol') does not count, nor do exactly two ring systems
    that ``identical_parent_units`` shows are not identically substituted units
     conditions (2) and (3)). Fails closed (True) on error.
    """
    try:
        from rdkit import Chem

        from ..perception.functional_groups import detect_functional_groups
        from .seniority import get_principal_group
        ri = mol.GetRingInfo()
        rings = [set(r) for r in ri.AtomRings()]
        if len(rings) < 2:
            return False
        systems = []
        for r in rings:
            merged = [s for s in systems if s & r]
            for s in merged:
                systems.remove(s)
                r = r | s
            systems.append(r)
        if len(systems) < 2:
            return False

        def skel(atoms):
            return Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(atoms),
                                            canonical=True, isomericSmiles=False,
                                            kekuleSmiles=False)
        keys = [skel(sy) for sy in systems]
        groups = {}
        for k, sy in zip(keys, systems):
            groups.setdefault(k, []).append(sy)
        repeated = {k: v for k, v in groups.items() if len(v) >= 2}
        if not repeated:
            return False
        fgs = detect_functional_groups(mol)
        pg, pg_matches = get_principal_group(mol, fgs)
        if pg:
            # a principal group belongs to ONE ring system when it lies in it or
            # is bonded to it and to no other ring system (the carbinol of
            # bis(4-chlorophenyl)methanol touches both rings: it is the linker)
            near = []
            for sy in systems:
                zone = set(sy)
                for a in sy:
                    zone.update(n.GetIdx() for n in mol.GetAtomWithIdx(a).GetNeighbors())
                near.append(zone)
            owner = []
            for m in (pg_matches or ()):
                touched = [i for i, z in enumerate(near) if set(m) & z]
                if len(touched) == 1:
                    owner.append(touched[0])
            for members in repeated.values():
                idx = {i for i, sy in enumerate(systems) if any(sy is x for x in members)}
                if len(idx & set(owner)) >= 2:
                    # (2)/(3): two units that are not identically
                    # substituted give no multiplicative PIN
                    # ('4-chloro-2-[(3-cyanophenyl)methyl]benzonitrile',:6235)
                    if (len(members) == 2
                            and not identical_parent_units(mol, members[0], members[1])):
                        continue
                    return True
            return False

        def seniority(sy):
            syms = {mol.GetAtomWithIdx(a).GetSymbol() for a in sy}
            n_rings = sum(1 for r in rings if r <= sy)
            return ('N' in syms, bool(syms - {'C'}), n_rings, len(sy))
        top = max(seniority(sy) for sy in systems)
        return any(seniority(v[0]) == top
                   and not (len(v) == 2 and not identical_parent_units(mol, v[0], v[1]))
                   for v in repeated.values())
    except Exception:  # noqa: BLE001 - unknown: do not claim the PIN
        return True



def _carbamate_ester_pattern():
    from rdkit import Chem
    return Chem.MolFromSmarts("[#7;!R]-[CX3;!R](=O)-[OX2;!R]-[#6]")


_CARBAMATE_ESTER = _carbamate_ester_pattern()


def _linker_sequence(mol, linker, start_system) -> tuple:
    """The linker read from one unit: (element, aromatic, bond order from the
    previous atom) for each atom, the first bond being the one to the unit."""
    seq = []
    prev = None
    for a in linker:
        atom = mol.GetAtomWithIdx(a)
        if prev is None:
            bond = next((mol.GetBondBetweenAtoms(a, x) for x in start_system
                         if mol.GetBondBetweenAtoms(a, x) is not None), None)
        else:
            bond = mol.GetBondBetweenAtoms(prev, a)
        seq.append((atom.GetSymbol(), atom.GetIsAromatic(),
                    bond.GetBondTypeAsDouble() if bond is not None else None))
        prev = a
    return tuple(seq)


def multiplicative_pin_expected(mol) -> bool:
    """True when the molecule's PIN is a multiplicative name, so no other name built
    for it is its PIN.

     "Preferred IUPAC multiplicative names" (the Blue Book):
    "Multiplicative nomenclature is preferred to substitutive nomenclature for
    generating preferred IUPAC names to express multiple occurrences of identical
    parent structures, other than alkanes when (1) the linking bonds... are
    identical and (2) the multiplicative groups... are symmetrically substituted;
    and (3) the locants of all substituent groups on the identical parent structures,
    including suffix groups, are identical" (:23180-23184); '1,1'-oxybis(4-bromo
    benzene) (PIN)',:6186); esters of such parents follow it
    ,:31790, 'dimethyl butanedioylbis(oxy-2,1-phenylene)
    dibutanedioate (PIN)',:31806).

    Narrower than ``multiplicative_candidate`` (which also answers True when it
    cannot decide): the molecule has a principal characteristic group (or, which the
    perception does not report, an N-substituted carbamic acid ester); two ring
    systems of one skeleton each own an instance of it (the group lies in the
    unit's ring system or is bonded to it and to no other ring system); their
    decorated units are identical (``_unit_identity``); the linker between them is
    symmetrical; and the two units hold more instances of the group than the rest
    of the molecule. A ring assembly, a principal group on the linker, no principal
    group, or anything undecidable gives False."""
    try:
        if not multiplicative_candidate(mol):
            return False
        from itertools import combinations

        from rdkit import Chem

        from ..perception.functional_groups import detect_functional_groups
        from .seniority import get_principal_group
        ri = mol.GetRingInfo()
        rings = [set(r) for r in ri.AtomRings()]
        systems = []
        for r in rings:
            merged = [x for x in systems if x & r]
            for x in merged:
                systems.remove(x)
                r = r | x
            systems.append(r)

        def skel(atoms):
            return Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(atoms),
                                            canonical=True, isomericSmiles=False,
                                            kekuleSmiles=False)
        groups = {}
        for i, sy in enumerate(systems):
            groups.setdefault(skel(sy), []).append(i)
        fgs = detect_functional_groups(mol)
        pg, pg_matches = get_principal_group(mol, fgs)
        if not pg:
            # An ester of an N-substituted carbamic acid is not reported by the
            # principal-group perception; esters rank above the classes it does
            # report for these skeletons, and a multiplied acid component
            # is named multiplicatively,:31790).
            pg_matches = tuple(mol.GetSubstructMatches(_CARBAMATE_ESTER))
            pg = "carbamate_ester" if pg_matches else None
        if not pg:
            # No claim without a principal characteristic group: the skeleton's
            # parent then depends on classes this check does not see (a ring
            # assembly, a ketene, a sulfoxide or sulfanediimine hub, an
            # unsymmetrical ether linker).
            return False
        near = []
        for sy in systems:
            zone = set(sy)
            for a in sy:
                zone.update(n.GetIdx() for n in mol.GetAtomWithIdx(a).GetNeighbors())
            near.append(zone)
        owners = set()
        for m in (pg_matches or ()):
            touched = [i for i, z in enumerate(near) if set(m) & z]
            if len(touched) == 1:
                owners.add(touched[0])
        for members in groups.values():
            if len(members) < 2:
                continue
            for i, j in combinations(members, 2):
                if not (i in owners and j in owners):
                    continue
                info = {}
                if _unit_identity(mol, systems[i], systems[j], info) is not True:
                    continue
                # (1)/(2): the central multiplicative group links the
                # units by identical bonds and is itself symmetrical -- the linker
                # read from either unit is the same sequence ('methylene',
                # 'ethane-1,2-diyl', '1,3-phenylenebis(methylene)'); an '-O-CH2-'
                # linker is not ('(phenoxymethyl)benzene' stays substitutive).
                linker = info.get("linker") or []
                if _linker_sequence(mol, linker, systems[i]) != _linker_sequence(
                        mol, list(reversed(linker)), systems[j]):
                    continue
                # (:18875): the senior parent has the maximum number of
                # principal characteristic groups; a multiplicative parent counts
                # those of all its identical units (:18907-18909). Claimed only when
                # the two units hold more instances than the rest of the molecule,
                # so no other structure can be the senior parent.
                both = set().union(*info.get("units", (set(), set())))
                inside = sum(1 for m in pg_matches if set(m) <= both)
                if inside <= len(pg_matches) - inside:
                    continue
                return True
        return False
    except Exception:  # noqa: BLE001 - unknown: no claim
        return False

# Branch review fixes (perf): set by ``Orthonym._name_with_pin_promotion`` around
# its FIRST run. The re-run differs from the first run only at the sites that read the
# promotion switch: ``pin_promotion_active`` (the systematic generators that decline
# in the re-run, the von Baeyer core numbering) and ``promote_at_pin_tier`` (the
# promoted producers). The probe records whether an ``active`` site ran, and every
# promotion call with its producer and the context it ran in, so that
# ``promotion_could_change`` can preview what the re-run's promoted producers would
# return before the whole molecule is named again.
_PROMOTION_SITES = contextvars.ContextVar("orthonym_pin_promotion_sites",
                                          default=None)
#: A first run with more promotion calls than this is re-run without a preview.
_MAX_PREVIEW_SITES = 64


class PromotionProbe:
    """What the first run did at the sites the promotion re-run changes."""

    __slots__ = ("active", "sites", "overflow")

    def __init__(self):
        self.active = False
        self.sites = []
        self.overflow = False

    def __bool__(self):  # "some site ran"
        return self.active or self.overflow or bool(self.sites)


def _in_first_run_probe():
    probe = _PROMOTION_SITES.get()
    if probe is None:
        return None
    from ..assembly.memo import pin_promotion_var
    return None if pin_promotion_var.get() else probe


def _record_promotion_site(produce) -> None:
    probe = _in_first_run_probe()
    if probe is None:
        return
    if len(probe.sites) >= _MAX_PREVIEW_SITES:
        probe.overflow = True
        return
    from ..assembly.fragment_naming import _fragment_guard as _g
    visited = getattr(_g, 'visited', None)
    probe.sites.append((produce, contextvars.copy_context(),
                        getattr(_g, 'session_depth', 0),
                        None if visited is None else set(visited)))


@contextlib.contextmanager
def promotion_site_probe():
    """Record what the enclosed (first) naming run did at the sites whose behaviour
    the PIN tier's promotion re-run changes; yields a ``PromotionProbe``."""
    probe = PromotionProbe()
    tok = _PROMOTION_SITES.set(probe)
    try:
        yield probe
    finally:
        _PROMOTION_SITES.reset(tok)


def _preview_site(site) -> bool:
    """True when the promoted producer of one recorded call returns a name that
    ``promote_at_pin_tier`` keeps (or the preview cannot tell). Runs in the context
    the call ran in (its context variables, fragment-recursion depth and visited
    set) with every per-molecule cache and budget sandboxed, so nothing it computes
    reaches the re-run or the caller."""
    produce, ctx, depth, visited = site

    def _run():
        from ..assembly.fragment_naming import (
            PerfBudgetExceeded,
            _fragment_guard,
            speculative_fragment_naming,
        )
        from ..assembly.memo import pin_promotion_var, pop_sandbox, push_sandbox
        tok = pin_promotion_var.set(True)
        box = push_sandbox()
        try:
            with speculative_fragment_naming():
                _fragment_guard.session_depth = depth
                _fragment_guard.visited = None if visited is None else set(visited)
                return promote_at_pin_tier(produce) is not None
        except PerfBudgetExceeded:
            return True
        except Exception:  # noqa: BLE001 -- unknown: let the re-run decide
            return True
        finally:
            pop_sandbox(box)
            pin_promotion_var.reset(tok)
    try:
        return bool(ctx.run(_run))
    except Exception:  # noqa: BLE001
        return True


def promotion_could_change(probe) -> bool:
    """False only when the PIN tier's promotion re-run is known to repeat the first
    run: no ``active`` site ran, the promotion calls fit the preview limit, and the
    promoted producer of every call returns no kept name. Then every promotion call
    of the re-run returns None, as in the first run, so the re-run takes the same
    path to the same (failed) result. Measured on the 452 dev2000 PIN-tier
    abstentions whose first run reached a promotion call: 275 of them."""
    if probe is None:
        return True
    if probe.active or probe.overflow:
        return True
    return any(_preview_site(site) for site in probe.sites)


def at_pin_tier() -> bool:
    """True when no breadth context is set: the default (PIN) tier, where a
    best-effort-gated producer runs only through `promote_at_pin_tier`. These are
    the four breadth contexts the substituent memo keys on, so a producer that
    reads this cannot return different names under one memo key."""
    try:
        from ..metrics.provenance import (
            allow_aromatic_general_ctx,
            best_effort_ctx,
            full_coverage_ctx,
            general_fallback_ctx,
        )
        return not (best_effort_ctx.get() or general_fallback_ctx.get()
                    or allow_aromatic_general_ctx.get() or full_coverage_ctx.get())
    except Exception:  # noqa: BLE001
        return False


def in_pin_promotion() -> bool:
    """True inside the PIN tier's promotion re-run (``assembly.memo.
    pin_promotion_var`` set by ``Orthonym.name``) with no breadth context set.

    The systematic generators (``terminal_ring_name``, ``terminal_fragment_name``,
    the acyclic chain composer) decline here: they build a deterministic,
    complete name, not the PIN ("The backbone choice is deterministic, NOT the
     seniority cascade", ``rules/terminal_fragment``), so their numbering can
    differ from the one the Blue Book prescribes -- '2,5-dioxo-1-[...]-7-oxa-
    bicyclo[4.1.0]hept-3-en-4-yl', where (c) before (e) (the Blue Book:
    3256, "principal characteristic groups and free valences (suffixes)", ahead of
    the 'ene' endings) gives the free valence the lower locant ('...-6-[...]-7-oxa-
    bicyclo[4.1.0]hept-3-en-3-yl'). Only the composition producers are promoted."""
    return pin_promotion_active()


def pin_promotion_active() -> bool:
    """True inside the PIN tier's promotion re-run with no breadth context set (the
    reading behind ``in_pin_promotion``). A site that reads it with no breadth
    context set is a site the re-run changes, so the first run's probe
    (``promotion_site_probe``) notes it."""
    try:
        if not at_pin_tier():
            return False
        _probe = _in_first_run_probe()
        if _probe is not None:
            _probe.active = True
        from ..assembly.memo import pin_promotion_var
        return bool(pin_promotion_var.get())
    except Exception:  # noqa: BLE001
        return False


def promote_at_pin_tier(produce: Callable[[], Optional[str]]) -> Optional[str]:
    """Run a best-effort-gated substituent producer at the PIN tier, keeping only
    a name whose vocabulary is all PIN vocabulary.

    ``produce`` is a zero-argument call of the producer with its breadth switch
    (``allow_mancude``) on. The name it returns is kept only when it is a single
    token, carries no `non_pin_vocabulary` token, did not set the whole-call
    ``general_ring_prefix`` flag (a prefix only the general tier builds), and
    contains no fragment a producer recorded as non-PIN during the call.
    Otherwise the provenance recorded during the call is rolled back and ``None``
    is returned, so the caller's PIN-tier behaviour is exactly as before -- apart
    from the name-scoped non-PIN records, which ``restore_provenance`` keeps: the
    call's cache entries survive the rejection, and a later hit on one of them
    must still meet the record its computation made (else the same string passes
    this guard with the memo on and fails it with the memo off). The round trip
     still judges the whole name afterwards.

    Active only in the PIN tier's re-run of a molecule its first run could not
    name (``assembly.memo.pin_promotion_var``, set by ``Orthonym.name``): a
    molecule the PIN tier names today keeps its name byte for byte, and a route
    that becomes reachable only through a promoted prefix can never pre-empt the
    route that named it.
    """
    from ..assembly.memo import pin_promotion_var
    if not pin_promotion_var.get():
        _record_promotion_site(produce)
        return None
    from ..metrics import provenance as _pv
    snap = _pv.get_provenance()
    try:
        name = produce()
    except Exception:  # noqa: BLE001 -- a producer bug must never crash naming
        name = None
    ok = bool(name) and name != "substituent" and " " not in name
    if ok and non_pin_vocabulary(name, label_forms=False) is not None:
        ok = False
    if ok:
        prov = _pv.get_provenance()
        if prov.get("general_ring_prefix") and not snap.get("general_ring_prefix"):
            ok = False
        elif any(f in name for f in (prov.get("non_pin_fragments") or ())):
            ok = False
    if not ok:
        _pv.restore_provenance(snap)
        return None
    return name
