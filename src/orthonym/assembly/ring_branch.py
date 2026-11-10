"""The group a ring prefix of a chain parent names: the ring system and everything that hangs from it outside the chain."""


def ring_branch_atoms(mol, ring_atoms, chain_set) -> set:
    """``ring_atoms`` and every heavy atom reachable from them without passing the parent chain ``chain_set``.

    It is the atom set of the group a ring prefix names (``substituent_atoms`` of ``terminal_ring_name``): the bond to the chain is
    its free valence, and the groups on the ring are its prefixes. NAMING OF STEREOISOMERS (the Blue Book), second
    paragraph (:44643): "When they relate to substituent groups, they are cited at the front of the corresponding prefix": the same
    branch gives ``ring_yl_attachment_descriptor`` its stereocentre."""
    branch = set(ring_atoms)
    stack = list(ring_atoms)
    while stack:
        cur = stack.pop()
        for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
            ni = nb.GetIdx()
            if ni in branch or ni in chain_set or nb.GetAtomicNum() <= 1:
                continue
            branch.add(ni)
            stack.append(ni)
    return branch
