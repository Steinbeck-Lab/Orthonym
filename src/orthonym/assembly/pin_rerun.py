"""The PIN tier's promotion re-run for a first-run name the default tier will decline (DT3 step A).

``Orthonym._name_with_pin_promotion`` names the molecule once, and names it again with the promoted
ring-substituent producers admitted (``rules.pin_vocabulary.promote_at_pin_tier``) when the first run
named nothing. A first run that names the molecule with a name the default tier's emission rule then
declines (``namer._default_tier_emits``: a general-engine name, a strict-path name that carries a
recorded non-PIN part, a name that fails a spelling check) used to end the wrapper at once, so the
re-run, which can answer with a name every piece of which a strict producer built
(``assembly.strict_ring_prefix``), was never tried for it: 'N-[2-(5-methoxy-1H-indol-3-yl)ethyl]
acetamide' (melatonin) was declined at the default tier while the same molecule without the methoxy group
is a ``pin_verified`` name there.

This module holds the three pieces of that path that do not belong in ``namer.py``:

*:func:`first_run_declined` -- the default tier's own decision on the first run's name, the same row and
  the same rule the caller's ``name`` applies to the name the wrapper returns, without its side effects;
*:func:`first_run_state` -- the per-call state of the first run that ``Orthonym._tier_row_and_pin_form``
  reads (dispatch class, winning offer, prefix-order fallback), restored when the re-run's name is not
  shipped, so the decision on the first run's name is the one it had before the re-run ran;
*:func:`rerun_class_guard` -- the classes for which a re-run name is not shipped here. "INTRODUCTION"
  (the Blue Book, sentence:50943): preferred IUPAC names are not identified for the compounds of
  Chapter (carbohydrates, natural products, lipids, amino acids and peptides), so a sugar-ring molecule, a
  natural-product scaffold, a lipid backbone or a peptide-like molecule keeps the first run's result; and a name that breaks a spelling rule of the PIN
  (``validation.pin_spelling``) is no PIN.
"""
from __future__ import annotations

from typing import Callable, Optional

from rdkit import Chem

#: the attributes of the first run that ``Orthonym._tier_row_and_pin_form`` reads after the run
_STATE = ("_last_dispatch_class", "_last_selected_offer", "_prefix_order_fallback_name", "_offers")

def first_run_declined(namer, smiles: str, name: str) -> bool:
    """True when the default tier's emission rule declines ``name``, the first run's name of ``smiles``.

    The row is the one ``Orthonym._tier_row_and_pin_form`` derives from the provenance of the first run
    and the rule is ``namer._default_tier_emits``: the pair ``Orthonym._default_tier_decision`` applies to
    the name the wrapper returns. A row that cannot be derived is no decline (the first run's name is
    returned as before)."""
    from .. import namer as _namer
    from ..metrics.provenance import get_provenance
    try:
        row, pin_form = namer._tier_row_and_pin_form(smiles, name, get_provenance())
        return not _namer._default_tier_emits(row, pin_form, smiles, namer._trivial_fallback)
    except Exception:  # noqa: BLE001 - the decision is information; the first run's name stands
        return False


def first_run_state(namer, named: bool) -> Callable[[], None]:
    """A function that puts the per-call state of the first run back on ``namer``; a no-op when the first
    run named nothing (``named`` False), where nothing reads that state."""
    if not named:
        return lambda: None
    saved = {attr: getattr(namer, attr, None) for attr in _STATE}

    def restore() -> None:
        for attr, value in saved.items():
            setattr(namer, attr, value)
    return restore


def rerun_class_guard(mol, name: str) -> Optional[str]:
    """The reason the re-run name ``name`` of ``mol`` is not shipped over a declined first run, or None.

    * a pyranose / furanose pattern (``assembly.strict_ring_prefix.carbohydrate_core``):
      "Carbohydrates" (the Blue Book) names carbohydrates with their own names, and
      "INTRODUCTION" (:50943) identifies no PIN for the compounds of Chapter;
    * a natural-product class of the engine's own router (``namer.classify_compound_class``: steroid,
      alkaloid, terpene, carbohydrate scaffolds): "NOMENCLATURE FOR NATURAL PRODUCTS BASED ON PARENT
      HYDRIDES, (ALKALOIDS, STEROIDS, TERPENES,...)" (:50952) sits under the same sentence of;
    * a lipid backbone (``perception.lipids.detect_lipid_backbone``: a glyceride, a phospholipid or a
      sphingoid): the same sentence of, and a glyceride named by prefixes is the method (2) name
      of "When anions are different, two methods are used" (:31831), whose PIN is the
      method (1) name (:31846);
    * a dipeptide unit (``rules.peptides.peptide_like``): "AMINO ACIDS AND PEPTIDES" (:54150) is a chapter whose
      compounds (:50943) gives no PIN, and the substitutive name of a peptide-like molecule is not promoted
      (DT3 lead decision 1);
    * a spelling failure of the PIN (``validation.pin_spelling.check_pin_spelling``), for example the
      stereodescriptor of a substituent at the front of a heteroatom parent,:44643)."""
    if mol is None:
        return "no structure"
    from ..namer import classify_compound_class
    from ..perception.lipids import detect_lipid_backbone
    from ..validation.pin_spelling import check_pin_spelling
    from .strict_ring_prefix import carbohydrate_core
    try:
        if carbohydrate_core(mol, range(mol.GetNumAtoms())):
            return "sugar ring (P-102.1.1)"
        natural_product = classify_compound_class(mol, Chem.MolToSmiles(mol))
        if natural_product is not None:
            return f"natural-product class {natural_product} (P-101)"
        if detect_lipid_backbone(mol) is not None:
            return "lipid backbone (P-100)"
        from ..rules.peptides import peptide_like
        if peptide_like(mol):
            return "peptide-like (P-103)"
        failures = check_pin_spelling(mol, name)
    except Exception as exc:  # noqa: BLE001 - a guard that cannot read the molecule lets nothing through
        return f"the guard could not read the molecule ({type(exc).__name__})"
    if failures:
        return f"{failures[0].rule} {failures[0].detail}"
    return None
