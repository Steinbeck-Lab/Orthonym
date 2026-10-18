"""A phane name read back by building the structure it states, apart from the producer.

OPSIN 2.9.0 has no phane grammar at all (neither '1,4(1,4)-dibenzenacyclohexaphane' nor the bracket
form '[2.2]paracyclophane' parses), so a phane name has no read-back and used to ship on its
construction alone (the ``phane`` grammar-gap carve-out). 40 of the 66 names the default tier gave
a sample of PubChem cyclophanes were the name of another molecule: the producer named a bridge that
holds a double or a triple bond as if it were saturated (the CO audit, 2026-10-17).

This module reads a name of the class the producer builds, the monocycle of benzene amplificants and
bridge atoms of with the endings of, as the structure it states and compares that
structure with the input by the FULL InChIKey. It shares no code with ``rules.phane``: the name is
parsed here (its own numeral words, its own reading of the locants and of the endings), the atoms and
bonds are built here, and what is compared is the key of the built molecule against the key of the input.

* (the Blue Book): the simplified skeleton has one node per superatom (an amplificant)
  and one per bridge atom, named 'cyclo' + the multiplying affix of the number of nodes + 'phane'.
*, (:14935,:14939): the superatom locants come first, the attachment locants of each
  amplificant in parentheses, 'dibenzena' is two benzene amplificants; identical attachment locants are
  cited once,:14947). The first attachment locant is the one adjacent to the lower
  skeleton locant,:14957); a benzene ring is symmetric, so the structure does not depend
  on which of the two bridge neighbours it faces.
* (:16793): 'ene' and 'yne' replace the final 'e' of 'phane'; (:16489) the
  locant is the lower one of the two consecutive primary locants of the multiple bond (bridge atoms
  are the only ones with a multiple bond here), 'ene' before 'yne', the 'e' of 'ene' dropped before
  'yne'; the 'e' of 'phane' dropped before a vowel only, 'phan-2-ene', 'phane-2,5-diene'.
* (:49686): "for up to seven nodes, no descriptor is necessary. When more than seven nodes
  are present, all descriptors are must be cited": up to seven nodes an uncited double bond is the cis
  one the ring allows, so it is read as 'Z' or as open, and an input that defines it 'E' does not read.

The check proves the structure and never the spelling: the producer owns the spelling, and a name this
module cannot read (any other amplificant, a substituent, a heteroatom bridge) is not read back.
"""
import itertools
import re
from typing import Dict, List, Optional, Tuple

from rdkit import Chem

#: the rule id of the read-back outcome
PHANE_READ_BACK_RULE = "phane_build"

def _numeral_words() -> Dict[str, int]:
    """The multiplying affixes of / read as numbers: 'tetra' 4... 'nona' 9, 'deca' 10,
    'undeca' 11, 'dodeca' 12, 'trideca'... 'nonadeca' 19, 'icosa' 20, 'henicosa' 21, 'docosa' 22,
    'tricosa' 23... 'triaconta' 30, 'hentriaconta' 31, 'dotriaconta' 32, 'tritriaconta' 33... up to 99."""
    stems = {1: "hen", 2: "do", 3: "tri", 4: "tetra", 5: "penta", 6: "hexa", 7: "hepta", 8: "octa", 9: "nona"}
    tens = {3: "triaconta", 4: "tetraconta", 5: "pentaconta", 6: "hexaconta", 7: "heptaconta",
            8: "octaconta", 9: "nonaconta"}
    words = {"di": 2, "tri": 3, "tetra": 4, "penta": 5, "hexa": 6, "hepta": 7, "octa": 8, "nona": 9, "deca": 10,
             "undeca": 11, "icosa": 20, "henicosa": 21}
    for n in range(12, 20):
        words[stems[n - 10] + "deca"] = n
    for n in range(22, 30):
        words[stems[n - 20] + "cosa"] = n
    for ten, word in tens.items():
        words[word] = ten * 10
        for unit in range(1, 10):
            words[stems[unit] + word] = ten * 10 + unit
    return words


_WORDS = _numeral_words()
_COUNTS = dict(_WORDS, **{"": 1})

_SIZE = "|".join(sorted(_WORDS, key=len, reverse=True))
_MULT = _SIZE
#: one 'ene' / 'yne' segment of the ending: locants, the multiplying prefix, the stem
_SEGMENT = rf"-(?P<{{n}}l>\d+(?:,\d+)*)-(?P<{{n}}m>(?:{_MULT})?)"
_NAME = re.compile(
    r"^(?:\((?P<stereo>\d+[EZ](?:,\d+[EZ])*)\)-)?"
    r"(?P<sup>\d+(?:,\d+)*)\((?P<att>\d+,\d+)\)-"
    rf"(?P<k>(?:{_MULT})?)benzenacyclo(?P<size>{_SIZE})phan"
    r"(?P<tail>.*)$")
_ENE = re.compile(_SEGMENT.format(n="e") + r"en(?P<ene_end>e|(?=-))")
_YNE = re.compile(_SEGMENT.format(n="y") + r"yne")


def phane_name_shape(name: Optional[str]) -> bool:
    """Whether ``name`` has the shape of a simplified-skeletal phane name: superatom locants,
    attachment locants in parentheses, an amplification prefix, 'phane' or its 'ene'/'yne' ending, and
    optionally the stereodescriptors in front. It is the shape the validity gate sends to the read-back."""
    return bool(name) and bool(re.match(
        r"^(?:\(\d+[EZ](?:,\d+[EZ])*\)-)?\d+(?:,\d+)*\([\d,]+\)-[a-z]+(?:cyclo)?[a-z]*phan"
        r"(?:e|e?-\d+(?:,\d+)*-[a-z]*(?:ene?|yne)(?:-\d+(?:,\d+)*-[a-z]*yne)?)$", name))


def _parse(name: str):
    """``(nodes, superatoms, attachment, doubles, triples, cited)`` of a name of the class, or None.

    ``cited`` maps a double-bond locant to 'Z' or 'E'. The ending must follow the Blue Book spelling:
    one multiplying prefix for each count, 'ene' before 'yne', the 'e' of 'phane' present exactly when
    the first ending has a multiplying prefix."""
    found = _NAME.match(name)
    if found is None:
        return None
    superatoms = [int(x) for x in found["sup"].split(",")]
    attachment = tuple(int(x) for x in found["att"].split(","))
    amplificants = _COUNTS.get(found["k"])
    nodes = _WORDS.get(found["size"])
    if (nodes is None or amplificants is None or amplificants != len(superatoms)
            or superatoms != sorted(set(superatoms)) or not 1 <= superatoms[0] <= superatoms[-1] <= nodes
            or any(not 1 <= a <= 6 for a in attachment) or attachment[0] == attachment[1]):
        return None
    tail = found["tail"]
    doubles: List[int] = []
    triples: List[int] = []
    if tail != "e":
        e_here = tail.startswith("e")
        rest = tail[1:] if e_here else tail
        segments = []
        pos = 0
        ene = _ENE.match(rest, pos)
        if ene is not None:
            segments.append((ene["el"], ene["em"], "ene"))
            pos = ene.end()
        yne = _YNE.match(rest, pos)
        if yne is not None:
            segments.append((yne["yl"], yne["ym"], "yne"))
            pos = yne.end()
        if not segments or pos != len(rest):
            return None
        if ene is not None and yne is not None and ene["ene_end"] == "e":
            return None
        if ene is not None and yne is None and ene["ene_end"] != "e":
            return None
        if e_here != bool(segments[0][1]):
            return None
        for locants, multiplier, kind in segments:
            listed = [int(x) for x in locants.split(",")]
            if len(listed) != _COUNTS[multiplier] or listed != sorted(set(listed)):
                return None
            (doubles if kind == "ene" else triples).extend(listed)
    cited: Dict[int, str] = {}
    if found["stereo"]:
        for item in found["stereo"].split(","):
            cited[int(item[:-1])] = item[-1]
        if set(cited) - set(doubles):
            return None
    return nodes, superatoms, attachment, doubles, triples, cited


def _build(nodes: int, superatoms: List[int], attachment: Tuple[int, int], doubles: List[int],
           triples: List[int], stereo: Dict[int, str]) -> Optional[Chem.Mol]:
    """The molecule of the name: one benzene ring per superatom, attached at the two ring positions of
    ``attachment``; one carbon per other node; a bond between consecutive nodes, closing the cycle."""
    rw = Chem.RWMol()
    ends: Dict[int, Tuple[int, int]] = {}
    for node in range(1, nodes + 1):
        if node in superatoms:
            ring = [rw.AddAtom(Chem.Atom(6)) for _ in range(6)]
            for i in range(6):
                rw.AddBond(ring[i], ring[(i + 1) % 6], Chem.BondType.AROMATIC)
                rw.GetAtomWithIdx(ring[i]).SetIsAromatic(True)
            ends[node] = (ring[attachment[0] - 1], ring[attachment[1] - 1])
        else:
            atom = rw.AddAtom(Chem.Atom(6))
            ends[node] = (atom, atom)
    multiple = set(doubles) | set(triples)
    for node in range(1, nodes + 1):
        following = node % nodes + 1
        if node in multiple and (node in superatoms or following in superatoms or node == nodes):
            return None  # a multiple bond joins two bridge atoms
        order = (Chem.BondType.TRIPLE if node in triples
                 else Chem.BondType.DOUBLE if node in doubles else Chem.BondType.SINGLE)
        rw.AddBond(ends[node][1], ends[following][0], order)
    for locant, label in stereo.items():
        before = (locant - 2) % nodes + 1
        after = (locant + 1) % nodes + 1
        bond = rw.GetBondBetweenAtoms(ends[locant][1], ends[locant % nodes + 1][0])
        bond.SetStereoAtoms(ends[before][1], ends[after][0])
        bond.SetStereo(Chem.BondStereo.STEREOCIS if label == "Z" else Chem.BondStereo.STEREOTRANS)
    try:
        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol


def _key(mol: Optional[Chem.Mol]) -> Optional[str]:
    try:
        return Chem.MolToInchiKey(mol) or None if mol is not None else None
    except Exception:
        return None


def phane_read_back(name: Optional[str], smiles: Optional[str]) -> bool:
    """True when the structure ``name`` states has the FULL InChIKey of ``smiles``.

    A double bond that a name of up to seven nodes does not cite is read as open and as 'Z',
    each of the double bonds on its own: the name states the cis configuration the ring allows, and an
    input that leaves the bond open is no other molecule than the one the name has named. A name of more
    than seven nodes cites every double bond it states, and a double bond it does not cite is open."""
    if not name or not smiles:
        return False
    parsed = _parse(name)
    reference = Chem.MolFromSmiles(smiles)
    if parsed is None or reference is None:
        return False
    nodes, superatoms, attachment, doubles, triples, cited = parsed
    wanted = _key(reference)
    if wanted is None:
        return False
    open_bonds = [d for d in doubles if d not in cited]
    if nodes > 7:
        variants = [dict(cited)]
    else:
        if len(open_bonds) > 8:
            return False
        variants = []
        for labels in itertools.product(("Z", None), repeat=len(open_bonds)):
            stereo = dict(cited)
            stereo.update({d: label for d, label in zip(open_bonds, labels) if label})
            variants.append(stereo)
    for stereo in variants:
        built = _build(nodes, superatoms, attachment, doubles, triples, stereo)
        if built is not None and _key(built) == wanted:
            return True
    return False
