"""The principal characteristic group of an acyclic arm of a multiplied unit.

``rules.multiplicative._acyclic_arm_parent_and_locant`` names a linear acyclic arm (a suffix group at one end, the bridge
attachment at the other) as the unit of a multiplicative name: 'acetic acid', 'ethan-1-ol', 'ethan-1-amine' and, an acyl
halide of the acid, 'acetyl chloride'. "Acyl halides from suffix acids" (the Blue Book): the acyl group of
the acid "followed by the name(s) of the specific class(es)... each preceded by a multiplicative prefix, as needed"; the
acid of the arm is the multiplicative PIN '2,2'-oxydiacetic acid',:5848), so its acyl halide is
'2,2'-oxydiacetyl dichloride' (``multiplied_class_word_units``), as the ring acid '4,4'-oxydibenzoic acid' gives
'4,4'-oxydibenzoyl dichloride'.
"""
from typing import Optional, Set, Tuple

#: the halogens of an acyl halide and the class word of each, Table 6.3, the Blue Book)
ACYL_HALIDE_WORDS = {"F": "fluoride", "Cl": "chloride", "Br": "bromide", "I": "iodide"}


def find_arm_pcg(mol, arm: Set[int], arm_carbons, allow_amine: bool) -> Optional[Tuple[int, str, Set[int], Optional[str]]]:
    """``(pcg_carbon, pcg, pcg_atoms, halide_word)`` of the arm, or None when it has no single group of the kinds above.

    ``pcg`` is 'acid' (=O and -OH on one carbon), 'acyl_halide' (=O and one halogen; ``halide_word`` its class word),
    'ol' (an -OH on a carbon without a carbonyl) or, with ``allow_amine``, 'amine' (a terminal -NH2)."""
    halide_word = None
    for c in arm_carbons:
        has_carbonyl_o = None
        has_hydroxyl_o = None
        for nbr in mol.GetAtomWithIdx(c).GetNeighbors():
            if nbr.GetSymbol() != 'O' or nbr.GetIdx() not in arm:
                continue
            bond = mol.GetBondBetweenAtoms(c, nbr.GetIdx())
            if bond.GetBondTypeAsDouble() == 2.0:
                has_carbonyl_o = nbr.GetIdx()
            elif nbr.GetTotalNumHs() >= 1 and nbr.GetDegree() == 1:
                has_hydroxyl_o = nbr.GetIdx()
        if has_carbonyl_o is not None and has_hydroxyl_o is not None:
            return c, 'acid', {has_carbonyl_o, has_hydroxyl_o}, None
        if has_carbonyl_o is not None:
            halogens = [nbr.GetIdx() for nbr in mol.GetAtomWithIdx(c).GetNeighbors()
                        if nbr.GetSymbol() in ACYL_HALIDE_WORDS and nbr.GetIdx() in arm
                        and nbr.GetDegree() == 1
                        and mol.GetBondBetweenAtoms(c, nbr.GetIdx()).GetBondTypeAsDouble() == 1.0]
            if len(halogens) == 1:
                halide_word = ACYL_HALIDE_WORDS[mol.GetAtomWithIdx(halogens[0]).GetSymbol()]
                return c, 'acyl_halide', {has_carbonyl_o, halogens[0]}, halide_word
    for c in arm_carbons:
        if any(mol.GetBondBetweenAtoms(c, nbr.GetIdx()).GetBondTypeAsDouble() == 2.0
               for nbr in mol.GetAtomWithIdx(c).GetNeighbors()
               if nbr.GetSymbol() == 'O' and nbr.GetIdx() in arm):
            continue  # carbonyl present -> not a plain alcohol carbon
        for nbr in mol.GetAtomWithIdx(c).GetNeighbors():
            if (nbr.GetSymbol() == 'O' and nbr.GetIdx() in arm
                    and nbr.GetTotalNumHs() >= 1 and nbr.GetDegree() == 1):
                return c, 'ol', {nbr.GetIdx()}, None
    if allow_amine:
        # Terminal primary amine -NH2 (BB "2,2'-oxydi(ethan-1-amine)" (PIN)).
        for c in arm_carbons:
            for nbr in mol.GetAtomWithIdx(c).GetNeighbors():
                if (nbr.GetSymbol() == 'N' and nbr.GetIdx() in arm
                        and nbr.GetTotalNumHs() == 2 and nbr.GetDegree() == 1
                        and mol.GetBondBetweenAtoms(c, nbr.GetIdx()).GetBondTypeAsDouble() == 1.0):
                    return c, 'amine', {nbr.GetIdx()}, None
    return None
