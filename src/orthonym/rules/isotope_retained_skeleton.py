"""Which retained skeleton name the isotope decorator gives up for the systematic one (``rules.isotopes``)."""


def retained_skeleton_dropped(pin_name, style, original, nonsubstitutable) -> bool:
    """Whether the retained skeleton name gives way to the systematic one: a retained arene that has no substitutable
    position for the label (``isotopes._ISOTOPE_NONSUBSTITUTABLE_RETAINED``, passed as ``nonsubstitutable``), and, at the PIN tier, a retained amino-acid name with
    a label at a position the retained name does not number. (inserted 1 December 2021): "When the nuclide is
    located at a position in a retained name that is not numbered, a systematic name that identifies separately the
    relevant atom is used for the IUPAC preferred name"; (:50943) identifies no PIN for 'glycine'. A completely
    labelled molecule names no position,:44196, '(2H6)benzene (PIN)'): it keeps the retained name
    ('(2H5)glycine', below the PIN)."""
    if pin_name in nonsubstitutable:
        return True
    from ..data.amino_acids import is_retained_amino_acid_name
    if style != "pin" or not is_retained_amino_acid_name(pin_name) or original is None:
        return False
    elements = {a.GetSymbol() for a in original.GetAtoms() if a.GetIsotope()}
    for element in elements:
        if element == "H":
            complete = (all(a.GetIsotope() for a in original.GetAtoms() if a.GetSymbol() == "H")
                        and not any(a.GetTotalNumHs() for a in original.GetAtoms() if a.GetSymbol() != "H"))
        else:
            complete = all(a.GetIsotope() for a in original.GetAtoms() if a.GetSymbol() == element)
        if not complete:
            return True
    return False
