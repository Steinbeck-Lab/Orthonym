"""The amine (aniline) suffix of a benzene ring when the amine is the forced principal group ('4-carboxyanilinium').

The amine is the principal group beside an acid only when the charged router forces it for a protonated aryl amine:
the cation is senior to the acids, the Blue Book-18170) and is named from the amine by
'aniline -> anilinium':41415, 'N,N,N-trimethylanilinium (PIN)':41455); a group the neutral molecule names
as a suffix is then its prefix ('carboxy',:29910). Both questions of ``benzene.
benzene_prefix_suffix_promotion`` (the suffix of the name, the ring atoms of the numbering) read ``amine_takes_suffix``.
"""


def amine_takes_suffix(own_suffix, has_amine_candidates: bool, suffix_names, suffix_to_prefix) -> bool:
    """True when the principal group is an amine and every suffix group of the ring has a prefix form."""
    return (own_suffix == 'amine' and has_amine_candidates
            and all(s in suffix_to_prefix for s in suffix_names))


def demote_suffixes_below_amine(suffix_groups, prefix_groups, n_substituents_map, demoted_prefix_overrides,
                                suffix_to_prefix) -> bool:
    """Move every suffix group but the amine to its prefix form, in place; False (decline) when one carries
    N-substituents or a stem override, which a plain prefix would drop."""
    seniors = [s for s in suffix_groups if s != 'amine']
    if any(s in n_substituents_map or s in demoted_prefix_overrides for s in seniors):
        return False
    for senior in seniors:
        prefix_groups[suffix_to_prefix[senior]].extend(suffix_groups.pop(senior))
        prefix_groups[suffix_to_prefix[senior]].sort()
    return True
