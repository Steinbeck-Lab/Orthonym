"""The lambda convention in the skeletal-replacement prefix of a cage substituent core.

A skeletal heteroatom of a ring whose bonding number is not its standard one carries 'λn' after its locant: the
sulfur of '7,7-dioxo-7λ6-thia-8-azabicyclo[4.3.0]nona-1,3,5-trien-9-yl' "Nonstandard bonding numbers",
the Blue Book; the group -SO2- in a ring is expressed by 'oxo' prefixes or '-one' on the heterocycle with the sulfur
designated λ4 or λ6,,:29436). The cage core of ``ring_substituents._cage_core_numbering`` is the ring alone,
so its heteroatom prefix ('7-thia-8-aza') never saw the oxygens on the sulfur; the bonding number is read from the whole
molecule here and cited after the locant.
"""
import re
from typing import Dict, Optional

from .lambda_convention import format_lambda_token, nonstandard_bonding_number

_GROUP = re.compile(r"(\d+(?:,\d+)*)-((?:di|tri|tetra|penta|hexa)?[a-z]+a)")


def lambda_hetero_prefix(mol, hetero_prefix: str, atom_to_locant: Dict[int, int],
                         sub_to_orig: Dict[int, int]) -> Optional[str]:
    """``hetero_prefix`` ('7-thia-8-aza') with 'λn' after the locant of each cage atom whose bonding number in ``mol``
    is not standard; the prefix itself when there is none; None when it cannot be cited (a lambda on carbon, a prefix
    this reads no group of)."""
    lam = {}
    for sub_idx, locant in atom_to_locant.items():
        n = nonstandard_bonding_number(mol, sub_to_orig[sub_idx])
        if n is not None:
            if mol.GetAtomWithIdx(sub_to_orig[sub_idx]).GetSymbol() == 'C' or not isinstance(locant, int):
                return None
            lam[locant] = n
    if not lam or not hetero_prefix:
        return hetero_prefix
    groups = list(_GROUP.finditer(hetero_prefix))
    if "".join(m.group(0) + "-" for m in groups).rstrip("-") != hetero_prefix.rstrip("-") or not groups:
        return None
    cited = set()
    parts = []
    for m in groups:
        locants = [int(x) for x in m.group(1).split(",")]
        cited.update(locants)
        parts.append(",".join(format_lambda_token(loc, lam.get(loc)) for loc in locants) + "-" + m.group(2))
    if not set(lam) <= cited:
        return None  # a lambda atom the prefix does not cite is no heteroatom of it
    return "-".join(parts) + ("-" if hetero_prefix.endswith("-") else "")
