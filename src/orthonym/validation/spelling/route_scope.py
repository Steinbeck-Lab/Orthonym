"""Which spelling checks read the name of a Chapter route before its PIN label is given
(``namer.Orthonym._tier_row_and_pin_form``)."""
from typing import Optional

from rdkit import Chem

#: Dispatch classes whose names follow Chapter nomenclature: "Preferred IUPAC names (PINs) are not
#: identified for the compounds in this Chapter" (the Blue Book). The natural-product route keeps the
#: labels of the natural-product decision (pin-class plan Task 32; user ruling 2026-09-30), so the spelling
#: checks do not read its names; the lipid route is read by the check only (its sugar parts,;
#: the carbohydrate, amino-acid and peptide routes are read by every check relabel 2026-10-31).
_SPELLING_CHECK_SKIPPED_CLASSES = ("NATURAL_PRODUCT",)
_SPELLING_CHECK_P100_ONLY_CLASSES = ("LIPID",)


def pin_spelling_rows(dispatch_class: Optional[str], smiles: str, name: str) -> list:
    """The spelling failures of ``name`` the PIN label reads, as rows, for the route ``dispatch_class``."""
    if dispatch_class in _SPELLING_CHECK_SKIPPED_CLASSES:
        return []
    from ..pin_spelling import check_pin_spelling
    failures = check_pin_spelling(Chem.MolFromSmiles(smiles), name)
    if dispatch_class in _SPELLING_CHECK_P100_ONLY_CLASSES:
        failures = [f for f in failures if f.rule == "P-100"]
    return [{"rule": f.rule, "detail": f.detail} for f in failures]

