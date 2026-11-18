"""An isotope-labelled mixture named part by part (IUPAC 2013, and.

The isotope decorator (``rules.isotopes.decorate_isotopic_name``) names the unlabelled skeleton and then writes the
nuclide descriptors into that name. For a mixture the skeleton name is an adduct name, and the search for the
descriptor places runs over the whole joined name: on the labelled PubChem mixtures that ran out of time it was the
whole cost (2026-11-17 study, ``.planning``: a median 14,121 trial names at 120 s).

Each component of a mixture is a compound of its own whose name the mixture name cites: "Mixed organic -
inorganic adducts" (the Blue Book), "names are formed by citing the names of individual compounds, connected by
long (em) dashes". A labelled component alone is named by the decorator, which inserts "the nuclide symbol(s)...
before the part of the compound that is isotopically substituted" "Isotopically substituted compounds").
So this module names the mixture with the adduct producer, each labelled component named as a compound alone
(``rules.adducts.LABELLED_PARTS_ALONE``), and ships the name only when OPSIN reads it back to the labelled input.
At the best-effort tier it runs FIRST for a mixture of three or more parts that is no salt (:func:`takes_parts_first`,
user decision 2026-11-17), and the whole-mixture search only when it gives None; otherwise it runs after the decorator.
"""
from typing import Optional

from rdkit import Chem


def takes_parts_first(original: Optional[Chem.Mol], namer) -> bool:
    """True when the labelled mixture ``original`` is named part by part before the decorator's whole-mixture search:
    the best-effort tier, three or more fragments, a nuclide label, and not a salt (no fragment with a positive net
    charge beside one with a negative net charge). A salt keeps the decorator first: its name cites the cation and the
    anion ('potassium (18F)fluoranide', 'sodium deuteride',, where the adduct producer cites ions
    ('sodium(1+)—deuteride (1/1)'). A mixture of two keeps it first too: its whole-mixture search is no longer than a
    part's own, and naming the part alone cost more (2026-11-17 panel: every mixture named later was one of two, every
    one newly named one of three or more)."""
    if original is None or not getattr(namer, "_general_fallback_unverified", False):
        return False
    frags = Chem.GetMolFrags(original, asMols=True, sanitizeFrags=False)
    if len(frags) < 3 or not any(a.GetIsotope() for a in original.GetAtoms()):
        return False
    nets = [sum(a.GetFormalCharge() for a in f.GetAtoms()) for f in frags]
    return not (any(n > 0 for n in nets) and any(n < 0 for n in nets))


def parts_first(original: Optional[Chem.Mol], style: str, namer) -> Optional[str]:
    """The part-by-part name of ``original`` before the decorator searches the whole mixture, or None. Runs only when
    :func:`takes_parts_first` holds, in a memo scope of its own, so a mixture it does not name is named after it with
    the caches the decorator would have had."""
    if not takes_parts_first(original, namer):
        return None
    from ..assembly.memo import fresh_scope, pop_sandbox
    scope = fresh_scope()
    try:
        return labelled_parts_adduct_name(original, style, namer)
    finally:
        pop_sandbox(scope)


def parts_last(original: Optional[Chem.Mol], style: str, namer) -> Optional[str]:
    """The part-by-part name of ``original`` after the decorator gave None, for a mixture:func:`parts_first` did not
    take (a salt), or None."""
    return None if takes_parts_first(original, namer) else labelled_parts_adduct_name(original, style, namer)


def labelled_parts_adduct_name(original: Optional[Chem.Mol], style: str, namer) -> Optional[str]:
    """The name of the labelled mixture ``original`` from the names of its components alone, or None.

    Only at the best-effort tier (``namer._general_fallback_unverified``; the PIN, valid and complete tiers keep
    their names). The name ships when OPSIN reads it back with every label, every stereodescriptor and the full
    standard InChIKey (``rules.isotopes._isotope_round_trips_once``) and with every charge where it is drawn
    (``rules.adducts._parse_reproduces_depiction``). The labelled parts are named in an empty context, so their own
    non-PIN records do not reach this naming: the whole name is recorded as no PIN here. When every labelled part's
    name is free of a non-PIN record in its own naming (``rules.adducts.LABELLED_PARTS_ALONE`` collects that) and the
    PIN tier names each labelled part alone as a PIN, the PIN tier is asked for the whole name
    (``assembly.pin_first.note_pin_tier_asked``), so a mixture the PIN tier names keeps its PIN.
    """
    if original is None or not getattr(namer, "_general_fallback_unverified", False):
        return None
    if len(Chem.GetMolFrags(original)) < 2 or not any(a.GetIsotope() for a in original.GetAtoms()):
        return None
    from . import adducts
    from .isotopes import _isotope_round_trips_once
    clean_parts: list = [{}]     # the shareable memo entries of the parts named so far, then a flag per part
    token = adducts.LABELLED_PARTS_ALONE.set(clean_parts)
    try:
        name = adducts._name_adduct_body(
            original, style=style, general_fallback=namer._general_fallback,
            allow_aromatic_general=namer._allow_aromatic_general,
            general_fallback_unverified=namer._general_fallback_unverified)
    except Exception:
        return None
    finally:
        adducts.LABELLED_PARTS_ALONE.reset(token)
    if not name or not _isotope_round_trips_once(name, original) or not adducts._parse_reproduces_depiction(
            name, original):
        return None
    from ..metrics.provenance import get_provenance, name_carries_non_pin_part, record_non_pin_label
    flags = clean_parts[1:]
    others_clean = not name_carries_non_pin_part(get_provenance(), name)    # the unlabelled parts ('water':
    record_non_pin_label(name, rule="P-14.8.2",
                         detail="an isotope-labelled mixture named by its components has no preferred IUPAC name")
    if flags and all(flags) and others_clean and _labelled_parts_are_pins(original, namer):
        from ..assembly.pin_first import note_pin_tier_asked
        note_pin_tier_asked()
    return name


def _labelled_parts_are_pins(original: Chem.Mol, namer) -> bool:
    """True when ``original`` has one labelled component and the PIN tier names it alone with a PIN read back directly
    (``assembly.pin_first._pin_tier_row``): only then is the PIN tier asked for the whole mixture. Its search over the
    whole mixture took 75 s on a PubChem mixture whose labelled part it declines alone, and 206 s on one of seven
    labelled parts that are PINs alone (with no PIN at the end)."""
    from ..assembly.pin_first import _pin_tier_row
    parts = {Chem.MolToSmiles(f): f.GetNumHeavyAtoms() for f in Chem.GetMolFrags(original, asMols=True)
             if any(a.GetIsotope() for a in f.GetAtoms())}
    if len(parts) != 1:     # the PIN tier's search over a mixture of several labelled parts took 206 s (PubChem, 7 parts)
        return False
    return all(_pin_tier_row(namer, smiles) is not None for smiles in sorted(parts, key=lambda s: (parts[s], s)))
