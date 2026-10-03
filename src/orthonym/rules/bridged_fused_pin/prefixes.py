"""Bridge prefixes and their citation: simple divalent bridges and the one composite
bridge OPSIN 2.9.0 reads, '(epoxymethano)'.

 (the Blue Book): "A divalent acyclic hydrocarbon bridge is named as a
prefix derived from the corresponding unbranched hydrocarbon name by changing the final
letter 'e' to 'o'." with the preferred prefixes '-CH2- methano', '-CH2-CH2- ethano',
'-CH2-CH2-CH2- propano', '-CH=CH- etheno'. (:14095-:14110): '-O-' is 'epoxy'
(a preselected prefix); 'epithio' and 'epimino' are general nomenclature only (the
preselected 'sulfano' / 'azano' cannot be read back by OPSIN 2.9.0), so a name that
carries one is recorded as not a PIN (``bridged_fused.GENERAL_ONLY_BRIDGE_PREFIXES``).
 (:14173): "If there is more than one bridge, they are cited in alphabetical
order... Two or more identical bridges are indicated by the numerical prefixes 'di',
'tri', etc. with simple bridges... The locant pairs of identical bridges are separated by
colons." (:14181): the locants of a divalent symmetric bridge "are cited in
numerical order".
"""
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

from ..bridged_fused import GENERAL_ONLY_BRIDGE_PREFIXES
from .numbering import loc_key

#: (:14069): the unbranched hydrocarbon name with final 'e' changed to 'o';
#: OPSIN 2.9.0 reads up to 'hexano' (a longer bridge is not spelled)
_ALKANO = {1: "methano", 2: "ethano", 3: "propano", 4: "butano", 5: "pentano", 6: "hexano"}
#: (:14071,:14073): the unsaturated bridges, keyed by (chain length, double
#: bond positions numbered from the end that gives them the lowest locants)
_ALKENO = {(2, (1,)): "etheno", (3, (1,)): "prop[1]eno", (4, (1,)): "but[1]eno",
           (4, (2,)): "but[2]eno", (4, (1, 3)): "buta[1,3]dieno"}
#: (:14095-:14118): one-atom heteroatom bridges (the preselected 'epoxy',
#: 'silano', 'stannano', 'borano'; the general-nomenclature 'epithio' / 'epimino',
#: recorded as not a PIN)
_ONE_ATOM = {"O": "epoxy", "S": "epithio", "N": "epimino", "Si": "silano", "Sn": "stannano",
             "B": "borano"}
_MULTIPLIER = {2: "di", 3: "tri", 4: "tetra"}
#: (the Blue Book) "The bonding number of a skeletal atom is standard when it
#: has the value given in Table 1.3" (:2750-:2754): B 3, C Si Sn 4, N 3, O S 2
_STANDARD_BONDING = {"B": 3, "C": 4, "Si": 4, "Sn": 4, "N": 3, "O": 2, "S": 2}


@dataclass(frozen=True)
class BridgePrefix:
    """``first`` is the bridgehead whose locant is cited first: None for a symmetric bridge
     :14181, "cited in numerical order"), else the bridgehead bonded to the
    first atom of the bridge name:14201, "in the order expressed or implied
    by the name of the bridge"): the O of '(epoxymethano)', C1 of 'prop[1]eno'."""
    name: str
    is_pin_form: bool
    first: Optional[int] = None


def _chain_double_bonds(chain: Sequence[int], unsat: Sequence[bool]) -> Optional[Tuple[int, ...]]:
    """Positions (1-based, from chain[0]) of the double bonds inside the chain: the unique
    perfect matching of its unsaturated atoms along the path, or None."""
    out, k = [], 0
    while k < len(chain):
        if not unsat[k]:
            k += 1
            continue
        if k + 1 >= len(chain) or not unsat[k + 1]:
            return None
        out.append(k + 1)
        k += 2
    return tuple(out)


def bridge_prefix(mol, split, i: int) -> Optional[BridgePrefix]:
    """The prefix of bridge ``i`` of ``split``, or None for a bridge this package does not
    spell (a ring bridge, a composite other than '(epoxymethano)', a charged or radical
    atom, a nonstandard valence, an unsaturated chain the Blue Book lists no prefix for).
    Unsaturation is read from ``split.unsaturated`` (the ring double bonds of the input,
    independent of the Kekule structure RDKit picks)."""
    chain = split.bridges[i]
    heads = split.bridgeheads[i]
    atoms = [mol.GetAtomWithIdx(a) for a in chain]
    if any(a.GetFormalCharge() != 0 or a.GetNumRadicalElectrons() != 0 for a in atoms):
        return None
    if any(a.GetTotalValence() != _STANDARD_BONDING.get(a.GetSymbol()) for a in atoms):
        return None
    unsat = [a in split.unsaturated for a in chain]
    symbols = [a.GetSymbol() for a in atoms]
    if len(chain) == 1 and symbols[0] in _ONE_ATOM and not unsat[0]:
        prefix = _ONE_ATOM[symbols[0]]
        return BridgePrefix(prefix, prefix not in GENERAL_ONLY_BRIDGE_PREFIXES)
    if symbols == ["O", "O"] and not any(unsat):
        return BridgePrefix("epidioxy", True)                   #:14101 (preselected)
    if symbols == ["O", "O", "O"] and not any(unsat):
        return BridgePrefix("epitrioxy", True)                  #:14102 (preselected)
    if symbols == ["N", "N"] and all(unsat):
        return BridgePrefix("diazeno", True)                    #:14112 (preselected)
    if sorted(symbols) == ["C", "O"] and not any(unsat):
        # (:14146): composite names cite the simple bridges "starting from the
        # senior prefix", O first (:14148); (:14153) parentheses;:14155
        # '–O-CH2– (epoxymethano) (preferred prefix)'
        o_end = chain[0] if symbols[0] == "O" else chain[-1]
        first = heads[0] if o_end == chain[0] else heads[1]
        return BridgePrefix("(epoxymethano)", True, first)
    if any(s != "C" for s in symbols):
        return None
    if not any(unsat):
        name = _ALKANO.get(len(chain))
        return BridgePrefix(name, True) if name else None
    forward = _chain_double_bonds(chain, unsat)
    backward = _chain_double_bonds(chain[::-1], unsat[::-1])
    if forward is None or backward is None:
        return None
    best = min(forward, backward)
    name = _ALKENO.get((len(chain), best))
    if name is None:
        return None
    if forward == backward:
        return BridgePrefix(name, True)
    first = heads[0] if forward == best else heads[1]
    return BridgePrefix(name, True, first)


def citation_key(prefix: str) -> str:
    """Alphanumerical order of bridge prefixes:14173), parentheses ignored."""
    return prefix.strip("()")


def bridge_text(cited: Sequence[Tuple[str, Tuple[Any, Any]]]) -> Optional[str]:
    """'1,4-epoxy-4a,8a-ethano', '1,4:5,8-dimethano': groups of identical prefixes in
    alphabetical order, each group's locant pairs ascending and joined by colons. Two
    identical composite bridges would need 'bis', which OPSIN 2.9.0 cannot
    read: None."""
    groups: List[Tuple[str, List[Tuple[Any, Any]]]] = []
    for prefix, pair in cited:
        if groups and groups[-1][0] == prefix:
            groups[-1][1].append(pair)
        else:
            groups.append((prefix, [pair]))
    parts = []
    for prefix, pairs in groups:
        pairs = sorted(pairs, key=lambda p: [loc_key(x) for x in p])
        if len(pairs) > 1 and prefix.startswith("("):
            return None
        mult = "" if len(pairs) == 1 else _MULTIPLIER.get(len(pairs))
        if mult is None:
            return None
        locs = ":".join(",".join(str(x) for x in p) for p in pairs)
        parts.append(f"{locs}-{mult}{prefix}")
    return "-".join(parts)
