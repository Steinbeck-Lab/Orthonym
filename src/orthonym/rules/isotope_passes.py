"""The passes of the isotope decorator over one labelled molecule (IUPAC 2013,.

``rules.isotopes._decorate_labelled`` runs the strategies of the decorator once. Two kinds of input need a
second look when it declines, and each is made only then, so no name the first pass builds changes:

* a stereocentre a nuclide makes: the strategies run again on the molecule with those centres
  unspecified and the name they build is completed (``rules.isotope_induced``);
* a name that OPSIN reads as a tautomer of the input, where the first pass accepts only the drawing: the
  strategies run again with that reading accepted (``rules.isotope_compare.tautomer_passes``), when the
  first pass refused one.

Every name still reads back as the input by the full InChIKey with its isotope layer, so a pass can add
names and never a wrong one.
"""
from __future__ import annotations

from typing import Optional

from rdkit import Chem

from . import isotopes as _iso
from .isotope_compare import tautomer_passes
from .isotope_induced import decorate_induced, induced_centres


def decorate_passes(smiles: str, style: str, namer, original: Chem.Mol, stripped: Chem.Mol,
                    label_map) -> Optional[str]:
    """The name of ``original`` (``stripped`` and ``label_map`` as ``rules.isotopes.strip_isotopes``
    returns them for it), or None."""
    induced = induced_centres(original)
    with tautomer_passes(original) as state:
        for accept in (False, True):
            state["accept"] = accept
            # The scope composer is not run as drawn on a molecule with an induced centre: every name it
            # builds lacks that descriptor and fails the read-back; it runs on the molecule without the
            # centre (``decorate_induced``).
            result = _iso._decorate_labelled(smiles, style, namer, original, stripped, label_map,
                                             composer=not induced)
            if result is None and induced:
                result = decorate_induced(style, namer, original, induced)
            if result is not None or not state["refused"]:
                return result
    return None
