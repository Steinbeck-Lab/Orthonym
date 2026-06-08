"""Named limit/error catalog for out-of-scope inputs (HYG-02, Phase 173).

Orthonym's default posture is *always-emit*: it returns a name (or a
descriptive fallback string) for every input. That reaches inputs a
refuse-when-unsure system would decline, but it cannot tell a caller whether a
result is a confident name or a plausible-but-wrong guess for something
Orthonym genuinely cannot handle (the §5.1C failure mode — e.g. a bare atom or
a wildcard structure named as if it were a real molecule).

This module adds HERITAGE-style *named limit codes* so a caller can distinguish
**"can't handle"** from **"got it wrong"** WITHOUT changing the default
always-emit behaviour. The structured `OrthonymLimitError` is surfaced only via
opt-in paths (`Orthonym.name(..., raise_on_limit=True)`,
`name_with_confidence()['limit']`, and `classify_limit()`); the default string
path is byte-identical to before.

Provenance (each code cites the HERITAGE analog it mirrors):
 §10`
("Hard limits & error catalog"): 40 codes in 0x104–0x18f, limits table
(125 total atoms / 44 per chain·ring·assembly / 32 stem candidates /
2 components / 255 chars), representative refusals ERR-210 (out of organic
range), ERR-215 (radicals), ERR-263 (bare atom: `[H]`/`[Na+]`), ERR-266
(inorganic: `O`/`N`), ERR-270 (unidentified FG), ERR-274 (atoms not assignable).
"""

from typing import Dict, Optional

from rdkit import Chem


# HERITAGE total-atoms hard limit (code 212): 125 atoms / structure.
ATOM_LIMIT = 125

# Elements Orthonym names as organic skeletons / substituents. Single source
# of truth — re-exported from orthonym.namer for backward compatibility
# (ml_fallback.quality_gate and data.cation_words import _METAL_NAMES there).
_ORGANIC_ELEMENTS = {
    'C', 'H', 'N', 'O', 'S', 'P', 'Se', 'F', 'Cl', 'Br', 'I', 'B', 'Si',
}

# Metal element -> name mapping for descriptive messages.
_METAL_NAMES = {
    'Li': 'lithium', 'Na': 'sodium', 'K': 'potassium', 'Rb': 'rubidium',
    'Cs': 'cesium', 'Be': 'beryllium', 'Mg': 'magnesium', 'Ca': 'calcium',
    'Sr': 'strontium', 'Ba': 'barium', 'Al': 'aluminium', 'Ga': 'gallium',
    'In': 'indium', 'Tl': 'thallium', 'Sn': 'tin', 'Pb': 'lead',
    'Bi': 'bismuth', 'Ti': 'titanium', 'V': 'vanadium', 'Cr': 'chromium',
    'Mn': 'manganese', 'Fe': 'iron', 'Co': 'cobalt', 'Ni': 'nickel',
    'Cu': 'copper', 'Zn': 'zinc', 'Zr': 'zirconium', 'Mo': 'molybdenum',
    'Ru': 'ruthenium', 'Rh': 'rhodium', 'Pd': 'palladium', 'Ag': 'silver',
    'Cd': 'cadmium', 'W': 'tungsten', 'Re': 'rhenium', 'Os': 'osmium',
    'Ir': 'iridium', 'Pt': 'platinum', 'Au': 'gold', 'Hg': 'mercury',
    'Sb': 'antimony', 'Te': 'tellurium', 'Yb': 'ytterbium', 'La': 'lanthanum',
    'Ce': 'cerium', 'Nd': 'neodymium', 'Sm': 'samarium', 'Eu': 'europium',
    'Gd': 'gadolinium', 'Tb': 'terbium', 'Dy': 'dysprosium', 'Ho': 'holmium',
    'Er': 'erbium', 'Tm': 'thulium', 'Lu': 'lutetium', 'Sc': 'scandium',
    'Y': 'yttrium',
}


# Symbolic Orthonym codes -> (generic message, HERITAGE analog). The per-instance
# message may be more specific (e.g. the metal name); these are the defaults.
LIMIT_CATALOG: Dict[str, Dict[str, str]] = {
    'WILDCARD_ATOMS': {
        'message': 'compound with wildcard atoms (not supported)',
        'heritage_ref': 'ERR-311',
    },
    'UNSUPPORTED_ELEMENT': {
        'message': 'inorganic compound (not supported)',
        'heritage_ref': 'ERR-210/266',
    },
    'ISOLATED_ATOM': {
        'message': 'isolated atom or substructure too small to name',
        'heritage_ref': 'ERR-263/266',
    },
    'STRUCTURE_TOO_LARGE': {
        'message': f'structure exceeds the {ATOM_LIMIT}-atom limit',
        'heritage_ref': 'ERR-212',
    },
    'UNNAMEABLE': {
        'message': 'unknown organic compound',
        'heritage_ref': 'ERR-270/274',
    },
}


class OrthonymLimitError(Exception):
    """Raised (opt-in) when an input is provably out of Orthonym's scope.

    Carries a symbolic ``code`` (a key of ``LIMIT_CATALOG``), a human-readable
    ``message``, the mirrored HERITAGE ``heritage_ref``, and the offending
    ``smiles`` when available. A caller that catches this knows Orthonym
    *cannot handle* the input — as opposed to a returned name, which is a
    best-effort *attempt*.
    """

    def __init__(self, code: str, message: str,
                 heritage_ref: Optional[str] = None,
                 smiles: Optional[str] = None):
        self.code = code
        self.message = message
        self.heritage_ref = heritage_ref
        self.smiles = smiles
        super().__init__(f"[{code}] {message}")

    def as_dict(self) -> Dict[str, Optional[str]]:
        return {
            'code': self.code,
            'message': self.message,
            'heritage_ref': self.heritage_ref,
        }


def _make(code: str, message: Optional[str] = None,
          smiles: Optional[str] = None) -> OrthonymLimitError:
    entry = LIMIT_CATALOG[code]
    return OrthonymLimitError(
        code=code,
        message=message if message is not None else entry['message'],
        heritage_ref=entry['heritage_ref'],
        smiles=smiles,
    )


def is_failure_name(name: Optional[str]) -> bool:
    """True if ``name`` is Orthonym's failure signal (empty or contains 'unknown').

    Mirrors the predicate ``name_compound`` already uses to decide whether to
    fall back to a descriptive string, so the limit classifier fires on exactly
    the inputs that today produce a descriptive fallback (never on a real name).
    """
    if not name:
        return True
    return 'unknown' in name.lower()


def classify_scope_limit(mol: Chem.Mol) -> Optional[OrthonymLimitError]:
    """Pre-naming structural refusal — ONLY for zero-false-positive classes.

    The sole provably-safe pre-check is a wildcard (`*`, atomic number 0): no
    definite molecule contains one and the namer would silently drop it
    (`CC*`->"ethane"). Everything else (bare atoms, metals, oversize) is left to
    `classify_failure_limit`, which keys off an actual naming failure and so can
    never refuse an input Orthonym does in fact name (e.g. `[H][H]`->molecular
    hydrogen, sodium salts).
    """
    if mol is None:
        return None
    if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
        return _make('WILDCARD_ATOMS')
    return None


def classify_failure_limit(mol: Chem.Mol,
                           smiles: Optional[str] = None) -> OrthonymLimitError:
    """Map an already-failed input to a named limit code.

    Returns a code that explains *why* Orthonym could not produce a real name.
    The ``.message`` is byte-identical to the legacy ``_descriptive_fallback``
    string for that branch (so the default always-emit output never changes);
    the ``.code`` / ``.heritage_ref`` add the new "can't handle" signal. The
    richer detail for the size/isolated branches lives in the code, not the
    message, precisely to preserve the legacy strings.
    """
    if mol is None or mol.GetNumAtoms() == 0:
        return _make('UNNAMEABLE', message='unknown', smiles=smiles)

    # Wildcard atoms (atomic number 0) — definite scope violation.
    if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
        return _make('WILDCARD_ATOMS', smiles=smiles)

    # Non-organic elements (iterate in atom-index order for deterministic output;
    # this mirrors the legacy _descriptive_fallback's stable-order metal pick).
    non_organic_in_order = []
    seen = set()
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym in _ORGANIC_ELEMENTS or sym in seen:
            continue
        seen.add(sym)
        non_organic_in_order.append(sym)

    if non_organic_in_order:
        for elem in non_organic_in_order:
            if elem in _METAL_NAMES:
                return _make(
                    'UNSUPPORTED_ELEMENT',
                    message=f"{_METAL_NAMES[elem]} compound (not supported)",
                    smiles=smiles,
                )
        return _make(
            'UNSUPPORTED_ELEMENT',
            message='inorganic compound (not supported)',
            smiles=smiles,
        )

    # Organic but unnameable. Message stays the legacy "unknown organic compound"
    # for byte-identity; the CODE distinguishes oversize / isolated / generic.
    heavy = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)
    if heavy > ATOM_LIMIT:
        code = 'STRUCTURE_TOO_LARGE'
    elif heavy <= 1:
        code = 'ISOLATED_ATOM'
    else:
        code = 'UNNAMEABLE'
    return _make(code, message='unknown organic compound', smiles=smiles)
