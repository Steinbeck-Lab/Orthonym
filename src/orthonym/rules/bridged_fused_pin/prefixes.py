"""Bridge prefixes and their citation (slice S1: simple divalent acyclic bridges).

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

_ALKANO = {1: "methano", 2: "ethano", 3: "propano", 4: "butano"}
_MULTIPLIER = {2: "di", 3: "tri", 4: "tetra"}


@dataclass(frozen=True)
class BridgePrefix:
    name: str
    is_pin_form: bool


def bridge_prefix(mol, split, i: int) -> Optional[BridgePrefix]:
    """The prefix of bridge ``i`` of ``split``, or None for a bridge slice S1 does not
    spell (composite, cyclic, charged, an unsaturation other than the one '-CH=CH-' of
    etheno). Unsaturation is read from ``split.unsaturated`` (the ring double bonds of
    the input, independent of the Kekule structure RDKit picks)."""
    chain = split.bridges[i]
    atoms = [mol.GetAtomWithIdx(a) for a in chain]
    if any(a.GetFormalCharge() != 0 or a.GetNumRadicalElectrons() != 0 for a in atoms):
        return None
    unsat = [a in split.unsaturated for a in chain]
    symbols = [a.GetSymbol() for a in atoms]
    if len(chain) == 1 and symbols[0] in ("O", "S", "N") and not unsat[0]:
        prefix = {"O": "epoxy", "S": "epithio", "N": "epimino"}[symbols[0]]
        return BridgePrefix(prefix, prefix not in GENERAL_ONLY_BRIDGE_PREFIXES)
    if any(s != "C" for s in symbols):
        return None
    if not any(unsat):
        name = _ALKANO.get(len(chain))
        return BridgePrefix(name, True) if name else None
    if len(chain) == 2 and all(unsat):
        return BridgePrefix("etheno", True)
    return None


def bridge_text(cited: Sequence[Tuple[str, Tuple[Any, Any]]]) -> Optional[str]:
    """'1,4-epoxy-4a,8a-ethano', '1,4:5,8-dimethano': groups of identical prefixes in
    alphabetical order, each group's locant pairs ascending and joined by colons."""
    groups: List[Tuple[str, List[Tuple[Any, Any]]]] = []
    for prefix, pair in cited:
        if groups and groups[-1][0] == prefix:
            groups[-1][1].append(pair)
        else:
            groups.append((prefix, [pair]))
    parts = []
    for prefix, pairs in groups:
        pairs = sorted(pairs, key=lambda p: [loc_key(x) for x in p])
        mult = "" if len(pairs) == 1 else _MULTIPLIER.get(len(pairs))
        if mult is None:
            return None
        locs = ":".join(",".join(str(x) for x in p) for p in pairs)
        parts.append(f"{locs}-{mult}{prefix}")
    return "-".join(parts)
