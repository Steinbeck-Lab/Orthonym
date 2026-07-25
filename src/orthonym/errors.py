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
# (data.cation_words imports _METAL_NAMES there).
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
    # Phase B honesty fix: 'As' and 'Ge' were the two p-block metalloids missing
    # from this table, so an UNNAMEABLE arsenic input fell through to the generic
    # 'inorganic compound (not supported)' message -- factually wrong for a
    # carbon-bearing organoarsenic compound such as C7H9AsO3. Their row-mates
    # (Sn, Pb, Sb, Bi, Te) were already present, which is why antimony reported
    # correctly and arsenic did not. Affects FAILURE DIAGNOSTICS only: this table
    # is consulted after naming has already failed, so no successful name moves.
    'As': 'arsenic', 'Ge': 'germanium',
    'Sb': 'antimony', 'Te': 'tellurium', 'Yb': 'ytterbium', 'La': 'lanthanum',
    'Ce': 'cerium', 'Nd': 'neodymium', 'Sm': 'samarium', 'Eu': 'europium',
    'Gd': 'gadolinium', 'Tb': 'terbium', 'Dy': 'dysprosium', 'Ho': 'holmium',
    'Er': 'erbium', 'Tm': 'thulium', 'Lu': 'lutetium', 'Sc': 'scandium',
    'Y': 'yttrium',
}


def _build_descriptive_fallback_names() -> frozenset:
    """Build the descriptive-fallback name set (the strings Orthonym emits
    when an input is out of scope / unnameable).

    The OPSIN validity gate (`namer._final_opsin_validity_gate`) skips
    re-gating these — they are intentional descriptive fallbacks that do
    not OPSIN-parse, so re-suppressing them is wasted work and must not
    alter output. Relocated here (v21 ML retirement, ADR-21-01) from the
    deleted `ml_fallback.quality_gate`; behaviour is byte-identical.
    """
    base = {
        "unknown organic compound",
        "compound with wildcard atoms (not supported)",
        "inorganic compound (not supported)",
    }
    for metal_name in _METAL_NAMES.values():
        base.add(f"{metal_name} compound (not supported)")
    return frozenset(base)


# Descriptive-fallback names the OPSIN validity gate must not re-gate
# (byte-identity anchor for `namer._final_opsin_validity_gate`).
_DESCRIPTIVE_FALLBACK_NAMES: frozenset = _build_descriptive_fallback_names()


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
    # G0 fail-closed safety (DD7 S1): a ring system Orthonym recognises as
    # complex (polycomponent-fused, bridged-fused, or aromatic-in-a-von-Baeyer
    # cage) but cannot yet name CORRECTLY. Raised mid-assembly to refuse rather
    # than emit a structurally-wrong de-aromatised cage / phantom substituent.
    # The message is the generic unnameable string (so the default always-emit
    # path is unchanged and ``is_failure_name``/``is_unknown`` still fire); the
    # CODE carries the "coverage gap, not garbage" signal. The correct PINs are
    # Phase-G1+ bridged/polycomponent-fusion builds.
    'UNSUPPORTED_RING_SYSTEM': {
        'message': 'unknown organic compound',
        'heritage_ref': 'ERR-274',
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


def unsupported_ring_system(smiles: Optional[str] = None) -> OrthonymLimitError:
    """Build the G0 fail-closed refusal for a complex ring system Orthonym
    cannot yet name correctly (DD7 S1). Raised mid-assembly by the von-Baeyer /
    bicyclo / polycomponent-fusion paths; caught once at ``Orthonym.name``
    (default path returns ``.message`` = 'unknown organic compound';
    ``raise_on_limit=True`` re-raises this error)."""
    return _make('UNSUPPORTED_RING_SYSTEM', smiles=smiles)


def is_failure_name(name: Optional[str]) -> bool:
    """True if ``name`` is Orthonym's failure signal (empty or contains 'unknown').

    Mirrors the predicate ``name_compound`` already uses to decide whether to
    fall back to a descriptive string, so the limit classifier fires on exactly
    the inputs that today produce a descriptive fallback (never on a real name).
    """
    if not name:
        return True
    # A descriptive fallback ('<metal> compound (not supported)', 'inorganic
    # compound (not supported)', 'compound with wildcard atoms (not supported)')
    # is ALSO a failure signal — Orthonym emits it precisely when it cannot name
    # the input. The old check saw only ''/'unknown', so after name() started
    # normalizing an empty fail-closed result to the descriptive fallback
    # (determinism fix), a metal-bearing unnameable input returned e.g. 'antimony
    # compound (not supported)' and was wrongly treated as a REAL name (raise_on_limit
    # never fired; the trivial fallback never triggered).
    low = name.lower()
    return 'unknown' in low or '(not supported)' in low


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

    # Non-organic elements. Pick the reported metal DETERMINISTICALLY by lowest
    # atomic number — NOT atom-index order. Atom-index order varies with the
    # SMILES spelling, so a multi-metal molecule (e.g. an Hg+Sb organometallic)
    # emitted 'mercury compound (not supported)' on some spellings and 'antimony
    # compound (not supported)' on others (determinism defect flagged by the
    # w2f-p11 gate). Which metal a fail-closed diagnostic names is arbitrary;
    # only determinism matters. Single-metal molecules (the common case) are
    # byte-identical — the sole metal is trivially lowest.
    non_organic = {atom.GetSymbol() for atom in mol.GetAtoms()
                   if atom.GetSymbol() not in _ORGANIC_ELEMENTS}
    if non_organic:
        pt = Chem.GetPeriodicTable()
        metals = sorted((s for s in non_organic if s in _METAL_NAMES),
                        key=lambda s: pt.GetAtomicNumber(s))
        if metals:
            return _make(
                'UNSUPPORTED_ELEMENT',
                message=f"{_METAL_NAMES[metals[0]]} compound (not supported)",
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
