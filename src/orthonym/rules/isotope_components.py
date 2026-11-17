"""An isotope-labelled mixture named part by part (IUPAC 2013, and.

The isotope decorator (``rules.isotopes.decorate_isotopic_name``) names the unlabelled skeleton and then writes the
nuclide descriptors into that name. For a mixture the skeleton name is an adduct name, and when no descriptor
placement in it reads back, the decorator gives None and ``Orthonym.name`` declines the input.

Each component of a mixture is a compound of its own whose name the mixture name cites: "Mixed organic -
inorganic adducts" (the Blue Book), "names are formed by citing the names of individual compounds, connected by
long (em) dashes". A labelled component alone is named by the decorator, which inserts "the nuclide symbol(s)...
before the part of the compound that is isotopically substituted" "Isotopically substituted compounds").
So this module names the mixture with the adduct producer, each labelled component named as a compound alone
(``rules.adducts.LABELLED_PARTS_ALONE``), and ships the name only when OPSIN reads it back to the labelled input.
"""
from typing import Optional

from rdkit import Chem


def labelled_parts_adduct_name(original: Optional[Chem.Mol], style: str, namer) -> Optional[str]:
    """The name of the labelled mixture ``original`` from the names of its components alone, or None.

    Only at the best-effort tier (``namer._general_fallback_unverified``; the PIN, valid and complete tiers keep
    their names) and only after every placement of the decorator failed (the caller). The name ships when OPSIN
    reads it back with every label, every stereodescriptor and the full standard InChIKey
    (``rules.isotopes._isotope_round_trips_once``) and with every charge where it is drawn
    (``rules.adducts._parse_reproduces_depiction``). The parts are named in an empty context, so their own
    non-PIN records do not reach this naming: the whole name is recorded as no PIN here.
    """
    if original is None or not getattr(namer, "_general_fallback_unverified", False):
        return None
    if len(Chem.GetMolFrags(original)) < 2 or not any(a.GetIsotope() for a in original.GetAtoms()):
        return None
    from . import adducts
    from .isotopes import _isotope_round_trips_once
    token = adducts.LABELLED_PARTS_ALONE.set(True)
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
    from ..metrics.provenance import record_non_pin_label
    record_non_pin_label(name, rule="P-14.8.2",
                         detail="an isotope-labelled mixture named by its components has no preferred IUPAC name")
    return name
