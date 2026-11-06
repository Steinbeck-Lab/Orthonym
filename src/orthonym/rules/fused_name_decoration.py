"""The hydrogen, lambda and substituent part of a fusion name built on a computed numbering.

Shared by the two-component ortho-fusion producer (``fused_rings._try_algorithmic_fusion_name``) and the
peri-fusion producer (``peri_fusion``): given the parent name of the mancude ring system and its atom -> locant map
(and, for a substituted system, its substituents), it adds the indicated or added hydrogen,
, the lambda tokens and the substituents, through the assembler of the catalogue path.
"""


def decorate_fused_name(mol, ring_atom_set, name, atom_to_locant, substituents, has_substituents,
                        skeleton_matched=False):
    """The full name, or None when the hydrogen or a lambda atom cannot be expressed."""
    from .fused_rings import (
        _assemble_fused_heterocycle_name,
        _compute_general_indicated_h,
        _hydrogen_marked_substituents,
        _join_hydrogen_prefix,
        _mancude_indicated_h_count,
        _mancude_indicated_h_locants,
        _placed_ring_hydrogen,
        _record_non_pin_spelling,
        fused_lambda_name_and_prefix,
        render_locants,
    )

    # A ring atom with an exocyclic double bond (ketone suffix; ylidene, imino or oxo
    # prefix): the hydrogen of the name is the mancude parent's, with the groups
    # accommodated by / (``ring_hydrogen``); the counters below read
    # the substituted ring instead and drop it ('pyrido[1,2-b]pyridazin-6-one' for the PIN
    # '6H-pyrido[1,2-b]pyridazin-6-one', the Blue Book).
    ring_h = None
    hydro_txt = ''
    if has_substituents:
        ring_h = _placed_ring_hydrogen(mol, ring_atom_set, atom_to_locant, substituents)
        if ring_h is not None:
            from . import ring_hydrogen as _rh
            hydro_txt = _rh.hydro_text(ring_h, atom_to_locant)
            if hydro_txt is None:
                ring_h, hydro_txt = None, ''
    if ring_h is not None:
        from .bridged_fused_pin.numbering import loc_key as _loc_key
        indicated_h = sorted((atom_to_locant[a] for a in ring_h.indicated), key=_loc_key)
        substituents = _hydrogen_marked_substituents(substituents, ring_h, atom_to_locant)
    else:
        indicated_h = _compute_general_indicated_h(mol, ring_atom_set, atom_to_locant)
        if not indicated_h:
            indicated_h = _mancude_indicated_h_locants(
                mol, ring_atom_set, atom_to_locant, has_substituents)
    if skeleton_matched:
        # The descriptor was accepted on the skeleton alone: ship only with the
        # established indicated hydrogen cited,:14607), never bare.
        _ih = _mancude_indicated_h_locants(
            mol, ring_atom_set, atom_to_locant, has_substituents)
        if not _ih or list(indicated_h) != list(_ih):
            return None

    #: lambda (nonstandard bonding number) tokens follow the
    # atom's fused-system locant and are cited at the beginning of the name,
    # after any indicated hydrogen (monocyclic precedent 1H-1lambda4-thiophene,
    #. Fail-closed: a lambda atom without a determinate NUMERIC
    # fused locant (or on skeletal carbon) invalidates the whole name -- the
    # lambda-less name would denote a DIFFERENT molecule.
    name, lam_prefix = fused_lambda_name_and_prefix(name, mol, ring_atom_set, atom_to_locant)
    if lam_prefix is None:
        return None

    if indicated_h:
        h_parts = render_locants(str(loc) + 'H' for loc in indicated_h)
        name = f"{h_parts}-{lam_prefix}{name}"
        # (the Blue Book): indicated hydrogen tells apart the
        # isomers of the MANCUDE parent; (:17026): the degree of
        # hydrogenation is given by 'hydro' prefixes ('6,7-dihydro-5H-benzo[7]annulene
        # (PIN)',:17032). More indicated hydrogen than the mancude parent takes
        # ('5H,6H,7H,8H,9H-cyclohepta[b]pyridine', '1H,2H,3H-benzo[d]imidazole') is
        # a saturated region spelled as indicated hydrogen: valid, never the PIN.
        if len(indicated_h) > _mancude_indicated_h_count(mol, ring_atom_set):
            _record_non_pin_spelling(name[:-1] if name.endswith('e') else name)
    elif lam_prefix:
        name = f"{lam_prefix}{name}"
    if hydro_txt:
        name = _join_hydrogen_prefix(hydro_txt, name)

    # a phase: attach the substituents discovered above against the
    # canonical numbering, reusing the exact assembler the catalog path uses.
    # The bare-parent `name` (with any indicated-H/lambda prefix already
    # assembled) is the parent the assembler decorates.
    if has_substituents:
        name = _assemble_fused_heterocycle_name(
            mol, name, substituents, atom_to_locant
        )
    return name
