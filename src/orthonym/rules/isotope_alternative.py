"""The second skeleton the isotope decorator is given when the first one takes no descriptor.

The decorator names the label-free skeleton of a labelled molecule and splices the isotopic descriptor into that
name; every candidate is read back with the labels (``rules.isotopes._isotope_round_trips``). Two producers give
skeleton names that no candidate can pass, although another producer names the same skeleton in a way that can:

- a member of a hydrocarbon series whose rings OPSIN draws in the other Kekule set than the input
  (``tetraphenylene``: OPSIN puts the double bonds in the eight-membered ring), which the read-back refuses
  for a labelled name (``_kekule_forms_of_one_compound``, a 4n circuit is two compounds);
- a senior mononuclear hub, ``bis[...]di(phenyl)silane``) whose like substituents the labels make unlike,
  so that no multiplied spelling can carry the descriptor;
- a polyester of one alcoholic component named by its multivalent organyl group of identical arms
  ('1,2-phenylenebis(methylene) diacetate', ``rules.multiplicative_general.name_ester_organyl_group``), whose arms the
  labels make unlike ('{2-[(acetyloxy)(13C)methyl]phenyl}methyl acetate': names a polyester of unlike
  arms by substitutive nomenclature);
- an ester of one multivalent group and several identical acids ('dibutylstannanediyl diacetate', 'ethane-1,2-diyl
  diacetate',: one labelled anion among the others makes them unlike, and the Blue Book then cites the
  anions apart (method (1) of, 'ethane-1,2-diyl 1-acetate 2-propanoate'), a form OPSIN 2.9.0 does not
  read, so no candidate of it can be checked. The next producer names the groups as acyloxy prefixes (method (2),
  "acceptable in general nomenclature"), which the decorator labels.

The namer then runs the decorator once more inside ``alternative_skeleton``, where those producers make no
offer and the skeleton gets the name the next producer gives. Outside that block nothing changes.
"""
import contextlib
import contextvars

_ACTIVE: "contextvars.ContextVar[bool]" = contextvars.ContextVar("orthonym_isotope_alternative_skeleton",
                                                                 default=False)


@contextlib.contextmanager
def alternative_skeleton():
    """Name the skeleton without the series, senior-hub and multiplied-anion ester producers while this block runs."""
    token = _ACTIVE.set(True)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def has_alternative(mol) -> bool:
    """True when ``mol`` (the label-free skeleton) has a producer this scope withholds: a series member, a senior
    mononuclear hub, a polyester of identical arms or two identical acyloxy groups. Otherwise the second pass would give
    the skeleton the name it had."""
    from .esters import has_identical_acyloxy_groups
    from .hydrocarbon_series import identify_series
    from .mononuclear_hydrides import _BELOW_HUB, _scan_senior_hubs
    from .multiplicative_general import polyester_organyl_group_named
    try:
        return (identify_series(mol) is not None or bool(_scan_senior_hubs(mol, _BELOW_HUB))
                or polyester_organyl_group_named(mol) or has_identical_acyloxy_groups(mol))
    except Exception:  # noqa: BLE001 -- a skeleton these readers refuse has no alternative here
        return False


def alternative_skeleton_active() -> bool:
    """True inside:func:`alternative_skeleton` (the memo keys its caches apart on it)."""
    return _ACTIVE.get()


def decorate_alternative(stripped, decorate):
    """``decorate`` inside:func:`alternative_skeleton`: the name the decorator finds on the second skeleton of the
    label-free ``stripped`` molecule, or None. A polyester that has a multivalent organyl group of identical arms is named
    on its second skeleton by substitutive nomenclature, because the labels make its arms unlike,
    the Blue Book), and "One or more isotopically modified atoms" (:21473) makes the principal chain that
    has the modified atoms the senior parent, which the descriptor spliced into the skeleton name does not decide: the
    name is recorded as no PIN."""
    from .multiplicative_general import polyester_organyl_group_named
    with alternative_skeleton():
        name = decorate()
    if name and polyester_organyl_group_named(stripped):
        from ..metrics.provenance import record_non_pin_label
        record_non_pin_label(
            name, rule="P-44.4.1.11",
            detail="the labels make the arms of the polyester unlike: the senior parent chain is the one with the "
                   "modified atoms, which the descriptor spliced into the skeleton name does not decide")
    return name
