"""Element -> cation word table for salt composition (IUPAC 2013 P-65.6.2.1).

Phase 169.6 Plan 04 (CHOKE-01, D-07/D-08).

P-65.6.2.1 (verbatim, BlueBookV2 line 31563): "Neutral salts of acids are named
by citing the name of the cation(s) followed by the name of the anion (see
P-72.2.2.2) as a separate word. Different cations are cited in alphabetical
order. Formation of salts is a functionalization and not a substitution."

A salt = cation word(s) (alphabetical) + the anion name as separate words
(``potassium propanoate``; ``potassium sodium butanedioate``). The cation word
is the element name for a metal cation, and ``ammonium`` for NH4+ (P-73.1.1).

This module is the single source of truth for the metal cation word. It REUSES
``namer._METAL_NAMES`` (the existing 50-element element->word lookup — do NOT
duplicate the table) and adds the NH4+ -> ``ammonium`` case, which is not a
metal. Resolves 169.6-RESEARCH open question 3 (the table EXISTS; import/extend
it rather than rebuild).
"""

from typing import Optional

from rdkit import Chem


def get_cation_word(frag_mol) -> Optional[str]:
    """Return the salt-composition cation word for a counter-cation fragment.

    P-65.6.2.1 (metals) / P-73.1.1 (ammonium). Handles:
      - a monatomic metal cation ([Na+], [K+], [Ca+2], ...) -> the element word
        from ``namer._METAL_NAMES`` (sodium / potassium / calcium / ...);
      - the ammonium cation [NH4+] -> ``ammonium``.

    Returns the cation word, or None for a cation that is NOT a simple
    metal/ammonium counter-cation (an organic cation, a coordination complex,
    etc.) -> the caller composes it via route_charged / honest-fails (D-08).
    """
    if frag_mol is None:
        return None

    atoms = [a for a in frag_mol.GetAtoms()]

    # Ammonium NH4+ (P-73.1.1): a single positively-charged N with 4 H and no
    # heavy neighbours (a bare ammonium counter-cation, NOT a substituted
    # organic ammonium — those route through route_charged / name_cation).
    if len(atoms) == 1 and atoms[0].GetSymbol() == 'N' \
            and atoms[0].GetFormalCharge() == 1 \
            and atoms[0].GetTotalNumHs() == 4:
        return 'ammonium'

    # Monatomic metal cation: a single positively-charged atom whose element is
    # in the metal word table (P-65.6.2.1).
    if len(atoms) == 1 and atoms[0].GetFormalCharge() > 0:
        from ..namer import _METAL_NAMES
        return _METAL_NAMES.get(atoms[0].GetSymbol())

    return None


def is_simple_cation(frag_mol) -> bool:
    """True iff ``frag_mol`` is a simple metal/ammonium counter-cation that
    ``get_cation_word`` can name (P-65.6.2.1 scope)."""
    return get_cation_word(frag_mol) is not None
