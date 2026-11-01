"""Spelling checks of two substituent prefixes whose general-nomenclature spelling is not the PIN
(registered with:func:`orthonym.validation.pin_spelling.register`).

A round trip proves the molecule, never the spelling: OPSIN reads '2-[(2,6-dichlorophenyl)amino]-2-
oxoethyl' and '1-methyl-2-oxoethyl' back to the structures of the preferred '2-(2,6-dichloroanilino)-
2-oxoethyl' and '1-oxopropan-2-yl'. A producer that writes the general form must not ship it as a PIN.
Both checks read the RDKit structure and the name text only.

* "Retained names" (the Blue Book): "The prefix name 'anilino' is retained as
  the preferred prefix for C6H5-NH– with full substitution allowed. The name 'phenylamino' may be
  used in general nomenclature." Its examples (:26151-:26153) 'anilino (preferred prefix)
  phenylamino', '4-chloroanilino (preferred prefix) (4-chlorophenyl)amino'; with the nitrogen
  substituted too, (:26298) ':26308' '3-(N-methylanilino)phenol (PIN)' against
  '3-[methyl(phenyl)amino]phenol'. The structure side is a nitrogen outside any ring bonded to a
  benzene ring that is no part of a fused system.
* "Retained prefixes no longer recommended as approved prefixes" (:16398),:16416 'butan-2-yl
  (preferred prefix) 1-methylpropyl'; (:16334) "The prefixes isopropyl, isopropylidene,
  and trityl are retained for use in general nomenclature but no substitution of any kind is
  allowed", with:24402 'propan-2-yl (preferred prefix) 1-methylethyl isopropyl' under
  "PREFIXES DERIVED FROM PARENT HYDRIDES"; "Retained prefixes that are preferred
  prefixes" (:16270),:24412 '-C(CH3)3 tert-butyl (preferred prefix) 1,1-dimethylethyl';
  (:40549) "Method (1) generates preferred IUPAC names": 'propan-2-yl (PIN) 1-methylethyl'
  (:40558). The structure side is the group itself, a saturated carbon with its two or three
  carbon neighbours.
* (:22660) "The principal substituent chain has the greater number of skeletal atoms,
  i.e., the longest chain": a methyl cited at locant 1 of an alkyl chain ('1-methyl-2-oxoethyl',
  '1-methylbutyl') is a carbon that extends the chain, so the group is named on the longer chain
  ('1-oxopropan-2-yl', 'pentan-2-yl'). The check reads the prefix block of every unbranched
  alkyl stem in the name; the structure side is an acyclic methyl on a branched acyclic carbon.
* "Hydrazine and derivatives" (:38507): "For H2N-NH– 'hydrazinyl'"; an amino group
  cited on an amino group ('aminoamino', '(methylamino)amino') is the hydrazine unit and takes
  'hydrazinyl'. The structure side is a nitrogen-nitrogen single bond.
* (:40620), Table 7.2 "Suffixes for Radicals of Amines, Imines and Amides" (:40622): '-iminyl'
  is the suffix of the radical =N•. A closed-shell structure takes no '-iminyl' word: R-CH=N- joined
  through nitrogen is '(R-ylidene)amino' "Substitutive names for imines",:26529), and
  R-C(=NH)- joined through carbon is an acyl group,:26540). The structure side is a
  molecule without a radical electron or a free valence.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from rdkit import Chem

from ..pin_spelling import SpellingFailure, register
from .lexer import normalise

#: an acyclic neutral nitrogen bonded to a carbon of a benzene ring that is in no other ring
_ANILINE_N = Chem.MolFromSmarts("[NX3;!R;+0]-[c;R1]1:[c;R1]:[c;R1]:[c;R1]:[c;R1]:[c;R1]:1")

#: 'phenylamino', '(X-phenyl)amino', '[X-phenyl]amino' and the '(phenyl)amino' of 'methyl(phenyl)amino':
#: the amino group cited on a phenyl group, and the phenyl group cited first of two ('(4-chlorophenyl)(methyl)
#: amino'). 'phenylaminyl', 'phenylaminium' and the like are other words, and so is 'methyl(phenyl)aminoxide',
#: the anion R2N-O(-) of (the Blue Book), whose 'aminoxide' is no amino group.
_PHENYLAMINO = re.compile(r"phenyl[)\]}]amino(?!xide)|phenylamino(?!xide)"
                          r"|phenyl[)\]}][(\[{](?:[^()\[\]{}]|[(\[{][^()\[\]{}]*[)\]}])*[)\]}]amino(?!xide)")


@register("P-62.2.1.1.1")
def anilino_prefix_check(mol, name) -> Optional[SpellingFailure]:
    """ (:26139): C6H5-N< is cited as the prefix 'anilino', whatever else the ring and
    the nitrogen carry,:26308); '(phenyl)amino' and 'phenylamino' are the general
    spelling. A name that cites the general spelling for a molecule that holds such a nitrogen is
    not the PIN."""
    hit = _PHENYLAMINO.search(normalise(name))
    if hit is None or not mol.HasSubstructMatch(_ANILINE_N):
        return None
    return SpellingFailure(
        "P-62.2.1.1.1",
        f"'{hit.group(0)}' cites C6H5-N< in the general spelling; the preferred prefix is 'anilino'")


#: (name token, the SMARTS of the group it names, rule, preferred prefix); the token stands alone:
#: a digit, letter or prime in front of it, or a letter after it, makes it another word
#: ('11-methylethyl', '1-methylethylidene', '1-methylpropylidene')
_BRANCHED_ALKYL: Tuple[Tuple[str, str, str, str], ...] = (
    ("1-methylethyl", "[CX4;!R;H1]([CH3])[CH3]", "P-29.6.2.2", "propan-2-yl"),
    ("1-methylpropyl", "[CX4;!R;H1]([CH3])[CH2][CH3]", "P-29.6.3", "butan-2-yl"),
    ("1,1-dimethylethyl", "[CX4;!R;H0]([CH3])([CH3])[CH3]", "P-29.6.1", "tert-butyl"),
)
_ALKYL_PATTERNS: List[Tuple[re.Pattern, Chem.Mol, str, str, str]] = [
    (re.compile(r"(?<![\w'])" + re.escape(token) + r"(?![a-z])"), Chem.MolFromSmarts(smarts),
     rule, token, preferred)
    for token, smarts, rule, preferred in _BRANCHED_ALKYL
]


def _branched_alkyl(mol, name, rule_id: str) -> Optional[SpellingFailure]:
    s = normalise(name)
    for pattern, group, rule, token, preferred in _ALKYL_PATTERNS:
        if rule == rule_id and pattern.search(s) and mol.HasSubstructMatch(group):
            return SpellingFailure(
                rule, f"'{token}' is the general spelling; the preferred prefix is '{preferred}'")
    return None


@register("P-29.6.2.2")
def propan_2_yl_check(mol, name) -> Optional[SpellingFailure]:
    """ (:16334,:24402) and (:40549): the group CH3-CH(-)-CH3 is 'propan-2-yl',
    never '1-methylethyl'."""
    return _branched_alkyl(mol, name, "P-29.6.2.2")


@register("P-29.6.3")
def butan_2_yl_check(mol, name) -> Optional[SpellingFailure]:
    """ (:16398),:16416: the group CH3-CH2-CH(-)-CH3 is 'butan-2-yl', never
    '1-methylpropyl'."""
    return _branched_alkyl(mol, name, "P-29.6.3")


@register("P-29.6.1")
def tert_butyl_check(mol, name) -> Optional[SpellingFailure]:
    """ (:16270,:24412): the group -C(CH3)3 is 'tert-butyl', never '1,1-dimethylethyl'."""
    return _branched_alkyl(mol, name, "P-29.6.1")


# ------------------------------------------------------------------ the principal chain,
_CHAIN_STEM = re.compile(
    r"(?P<stem>eth|prop|but|pent|hex|hept|oct|non|dec)(?:yl|an-(?P<k>\d+)-yl)(?![a-z])")
_STEM_LENGTH = {"eth": 2, "prop": 3, "but": 4, "pent": 5, "hex": 6, "hept": 7, "oct": 8,
                "non": 9, "dec": 10}
#: a methyl cited at locant 1 as a prefix of its own (followed by the next locant or the end of the
#: prefix block): '1-methyl-2-oxo', '1-methyl', '1,1-dimethyl'
_METHYL_AT_1 = re.compile(r"(?:^|-)1(?:,1)*-(?:di|tri)?methyl(?=-|$)")
_NESTED = re.compile(r"[(\[{][^()\[\]{}]*[)\]}]")
#: the descriptor of a von Baeyer, spiro or fusion name that ends the text in front of a stem:
#: '[2.2.1]', '[4.5]', '[1,2-a]', '[3.3.1.1^3,7]' (a bracket of a nested substituent holds words)
_RING_DESCRIPTOR = re.compile(r"\[[\d.,:'^\-]*[a-z]?[\d.,:'^\-]*\]$")


def _belongs_to_a_ring_name(s: str, start: int) -> bool:
    """True when the stem at ``s[start]`` is the end of a ring name or of a replacement name, not
    an alkyl chain: 'cyclohexyl', 'bicyclo[2.2.1]heptan-2-yl', 'spiro[4.5]decan-8-yl', '2-oxabutyl'."""
    before = s[:start]
    return (before.endswith("cyclo") or before.endswith("a")
            or _RING_DESCRIPTOR.search(before) is not None)


def _unit_block(s: str, end: int) -> Optional[tuple]:
    """``(start, block)`` of the substituent unit that ends at ``s[end]``: the text back to the
    unmatched opening mark, the space or the start of the name, with every enclosed group
    replaced by 'X' in ``block``. ``None`` when the stem is cited bare after a locant ('N-ethyl')."""
    if end == 0 or s[end - 1] in "- ":
        return None
    depth, start = 0, 0
    for k in range(end - 1, -1, -1):
        c = s[k]
        if c in ")]}":
            depth += 1
        elif c in "([{":
            if depth == 0:
                start = k + 1
                break
            depth -= 1
        elif c == " " and depth == 0:
            start = k + 1
            break
    block = s[start:end]
    prev = None
    while prev != block:
        prev, block = block, _NESTED.sub("X", block)
    return start, block


def _holds_chain_group(mol, n: int, k: int) -> bool:
    """True when the structure holds the group the name cites: ``n`` acyclic carbons in a row by
    single bonds, a methyl outside the row on its first atom, and the free valence (a neighbour
    that is none of the row and that methyl) on its ``k``-th atom."""
    def acyclic_c(a):
        return a.GetAtomicNum() == 6 and not a.IsInRing()

    def grow(path):
        if len(path) == n:
            first, fv = mol.GetAtomWithIdx(path[0]), mol.GetAtomWithIdx(path[k - 1])
            for m in first.GetNeighbors():
                if (m.GetIdx() in path or not acyclic_c(m) or m.GetDegree() != 1):
                    continue
                if any(x.GetIdx() not in path and x.GetIdx() != m.GetIdx()
                       for x in fv.GetNeighbors()):
                    return True
            return False
        for nb in mol.GetAtomWithIdx(path[-1]).GetNeighbors():
            bond = mol.GetBondBetweenAtoms(path[-1], nb.GetIdx())
            if (nb.GetIdx() not in path and acyclic_c(nb)
                    and bond.GetBondType() == Chem.BondType.SINGLE and grow(path + [nb.GetIdx()])):
                return True
        return False

    return k <= n and any(acyclic_c(a) and grow([a.GetIdx()]) for a in mol.GetAtoms())


@register("P-46.1.2")
def chain_methyl_check(mol, name) -> Optional[SpellingFailure]:
    """ (:22660): the principal chain of a substituent group is its longest chain, so a
    methyl at locant 1 of an alkyl chain is a chain atom the name leaves out of the stem. Only
    an alkyl stem is judged (not the end of a ring or replacement name), and only when the
    structure holds that group: the chain of the stem's length with a methyl on its first atom."""
    s = normalise(name)
    for m in _CHAIN_STEM.finditer(s):
        if (m.start() > 0 and s[m.start() - 1] == "m" and m.group("stem") == "eth"
                or _belongs_to_a_ring_name(s, m.start())):
            continue                                           # 'methyl', 'cyclohexyl',...
        unit = _unit_block(s, m.start())
        if (unit is not None and _METHYL_AT_1.search(unit[1])
                and _holds_chain_group(mol, _STEM_LENGTH[m.group("stem")],
                                       int(m.group("k") or 1))):
            return SpellingFailure(
                "P-46.1.2", f"'{s[unit[0]:m.end()]}' cites a methyl at locant 1 of a chain: the "
                            f"principal substituent chain is the longer one")
    return None


# ------------------------------------------------------------------ hydrazinyl,
_N_N = Chem.MolFromSmarts("[NX3]-[NX3]")
_AMINO_ON_AMINO = re.compile(r"aminoamino|[a-z]amino\)amino|[a-z]amino\]amino|[a-z]amino\}amino")


@register("P-68.3.1.2.1")
def hydrazinyl_check(mol, name) -> Optional[SpellingFailure]:
    """ (:38507): H2N-NH- is 'hydrazinyl' and its substituted forms are cited from it
    ('2-methylhydrazin-1-yl',,:2869); 'aminoamino' and '(methylamino)amino' are not."""
    hit = _AMINO_ON_AMINO.search(normalise(name))
    if hit is None or not mol.HasSubstructMatch(_N_N):
        return None
    return SpellingFailure(
        "P-68.3.1.2.1", f"'{hit.group(0)}' cites an amino group on an amino group; the unit is "
                        f"'hydrazinyl'")


# ------------------------------------------------------------------ iminyl,
#: the suffix 'iminyl' ('methaniminyl', 'propan-2-iminyl'); 'iminylium' is a cation suffix
_IMINYL = re.compile(r"[a-z0-9-]*iminyl(?!ium)")


@register("P-71.3.2")
def iminyl_word_check(mol, name) -> Optional[SpellingFailure]:
    """ (:40620), Table 7.2 (:40622): '-iminyl' is the suffix of the radical =N•, so a
    name of a structure with no radical electron and no free valence holds no such word."""
    hit = _IMINYL.search(normalise(name))
    if hit is None:
        return None
    if any(a.GetNumRadicalElectrons() or a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
        return None
    return SpellingFailure(
        "P-71.3.2", f"'{hit.group(0)}' is the suffix of a radical; a closed-shell structure takes "
                    f"'(R-ylidene)amino' through nitrogen and an acyl group through carbon")
