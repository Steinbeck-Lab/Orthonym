"""Named limit/error catalog for out-of-scope inputs (HYG-02, Phase 173).

Orthonym's default posture is *always-emit*: it returns a name (or a
descriptive fallback string) for every input. That reaches inputs a
refuse-when-unsure system would decline, but it cannot tell a caller whether a
result is a confident name or a plausible-but-wrong guess for something
Orthonym genuinely cannot handle (the §5.1C failure mode — e.g. a bare atom or
a wildcard structure named as if it were a real molecule).

This module adds AUTONOM-style *named limit codes* so a caller can distinguish
**"can't handle"** from **"got it wrong"** WITHOUT changing the default
always-emit behaviour. The structured `OrthonymLimitError` is surfaced only via
opt-in paths (`Orthonym.name(..., raise_on_limit=True)`,
`name_with_confidence()['limit']`, and `classify_limit()`); the default string
path is byte-identical to before.

Provenance (each code cites the AUTONOM analog it mirrors):
`.planning/research/autonom-comparison/AUTONOM-ARCHITECTURE.md §10`
("Hard limits & error catalog"): 40 codes in 0x104–0x18f, limits table
(125 total atoms / 44 per chain·ring·assembly / 32 stem candidates /
2 components / 255 chars), representative refusals ERR-210 (out of organic
range), ERR-215 (radicals), ERR-263 (bare atom: `[H]`/`[Na+]`), ERR-266
(inorganic: `O`/`N`), ERR-270 (unidentified FG), ERR-274 (atoms not assignable).
"""

from typing import Dict, Optional

from rdkit import Chem

# AUTONOM total-atoms hard limit (code 212): 125 atoms / structure.
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


# Symbolic Orthonym codes -> (generic message, AUTONOM analog). The per-instance
# message may be more specific (e.g. the metal name); these are the defaults.
LIMIT_CATALOG: Dict[str, Dict[str, str]] = {
    'WILDCARD_ATOMS': {
        'message': 'compound with wildcard atoms (not supported)',
        'design_note_ref': 'ERR-311',
    },
    'UNSUPPORTED_ELEMENT': {
        'message': 'inorganic compound (not supported)',
        'design_note_ref': 'ERR-210/266',
    },
    'ISOLATED_ATOM': {
        'message': 'isolated atom or substructure too small to name',
        'design_note_ref': 'ERR-263/266',
    },
    'STRUCTURE_TOO_LARGE': {
        'message': f'structure exceeds the {ATOM_LIMIT}-atom limit',
        'design_note_ref': 'ERR-212',
    },
    'UNNAMEABLE': {
        'message': 'unknown organic compound',
        'design_note_ref': 'ERR-270/274',
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
        'design_note_ref': 'ERR-274',
    },
}


class OrthonymLimitError(Exception):
    """Raised (opt-in) when an input is provably out of Orthonym's scope.

    Carries a symbolic ``code`` (a key of ``LIMIT_CATALOG``), a human-readable
    ``message``, the mirrored AUTONOM ``design_note_ref``, and the offending
    ``smiles`` when available. A caller that catches this knows Orthonym
    *cannot handle* the input — as opposed to a returned name, which is a
    best-effort *attempt*.
    """

    def __init__(self, code: str, message: str,
                 design_note_ref: Optional[str] = None,
                 smiles: Optional[str] = None):
        self.code = code
        self.message = message
        self.design_note_ref = design_note_ref
        self.smiles = smiles
        super().__init__(f"[{code}] {message}")

    def as_dict(self) -> Dict[str, Optional[str]]:
        return {
            'code': self.code,
            'message': self.message,
            'design_note_ref': self.design_note_ref,
        }


def _make(code: str, message: Optional[str] = None,
          smiles: Optional[str] = None) -> OrthonymLimitError:
    entry = LIMIT_CATALOG[code]
    return OrthonymLimitError(
        code=code,
        message=message if message is not None else entry['message'],
        design_note_ref=entry['design_note_ref'],
        smiles=smiles,
    )


def unsupported_ring_system(smiles: Optional[str] = None) -> OrthonymLimitError:
    """Build the G0 fail-closed refusal for a complex ring system Orthonym
    cannot yet name correctly (DD7 S1). Raised mid-assembly by the von-Baeyer /
    bicyclo / polycomponent-fusion paths; caught once at ``Orthonym.name``
    (default path returns ``.message`` = 'unknown organic compound';
    ``raise_on_limit=True`` re-raises this error)."""
    return _make('UNSUPPORTED_RING_SYSTEM', smiles=smiles)


def unsupported_element_branch(symbol: str,
                               smiles: Optional[str] = None
                               ) -> OrthonymLimitError:
    """Fail-closed refusal for a substituent branch that Orthonym cannot name
    and that carries an element outside ``_ORGANIC_ELEMENTS``.

    Raised mid-assembly and caught at the single ``Orthonym.name`` catch point,
    exactly like ``unsupported_ring_system``. It exists because SKIPPING such a
    branch is not a smaller error than mis-naming it — it ships a name for a
    DIFFERENT molecule: with the refusal sentinel no longer leaking through as a
    string, ``CCS[Zn]SCC`` went from the visibly-broken
    'zinc compound (not supported)ylethane' to the plausible and therefore far
    more dangerous 'ethane', which no failure predicate can flag.

    The message names the element when known, so the refusal stays as
    informative as the descriptive fallback it mirrors.
    """
    metal = _METAL_NAMES.get(symbol)
    message = (f'{metal} compound (not supported)' if metal
               else LIMIT_CATALOG['UNSUPPORTED_ELEMENT']['message'])
    return _make('UNSUPPORTED_ELEMENT', message=message, smiles=smiles)


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


# The substituent cascade's placeholder. `assembly.substituent_enumerator`'s
# tiers return this bare word as an absolute last resort, meaning "there is a
# substituent here and I could not name it". It is a REFUSAL, not a name, but it
# is not a descriptive fallback either, so `is_failure_name` does not see it.
CASCADE_PLACEHOLDER = 'substituent'


def is_refusal_sentinel(name: Optional[str]) -> bool:
    """True if ``name`` is any refusal sentinel and so must never be CONSUMED
    as a name component (a substituent prefix, a parent stem, an ester word...).

    This is the slot-level predicate. It is deliberately built ON TOP of
    ``is_failure_name`` rather than beside it: that function already recognises
    three of the four sentinel families exactly (empty, ``'unknown ...'``, and the
    ``'... (not supported)'`` descriptive fallbacks), and duplicating them is how
    this class of bug reached six copies in the first place. What it does NOT
    recognise is the substituent cascade's bare ``'substituent'`` placeholder,
    because that string is not a whole-molecule failure signal — a caller asking
    "did naming fail?" of a finished name must not be told yes merely because the
    word appears. Hence one extra leg here, and no re-implementation.

    The failure this closes: a sentinel accepted into a substituent slot is
    silently welded into a name — ``CCS[Zn]SCC`` produced
    ``'zinc compound (not supported)ylethane'``, which every "did I get a
    non-empty string?" caller reads as success.

    ⚠ **The placeholder leg is a SUBSTRING test, not equality.** Measured
    2026-08-04 over the 10,000-row head-to-head: exact equality missed every
    *decorated* occurrence, because the placeholder reaches a slot check with a
    locant or italic element prefix already attached —
    ``'N-substituentformamide'``, ``'N-substituenthydroxyphosphonooxytricos…'``,
    ``'…-3-amino-sulfanyl-N-substituentpropanamide'``. 310 rows shipped such a
    name. Widening is safe by measurement, not by assumption: **0 of the 3,616
    round-tripping names in that corpus contain the substring**, so the
    predicate cannot fire on a name known to be correct (the ``check-target``
    bar). The other three families were already substring-matched by
    :func:`is_failure_name`, which is why only this leg leaked.
    """
    if is_failure_name(name):
        return True
    return CASCADE_PLACEHOLDER in name.lower()


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
    the ``.code`` / ``.design_note_ref`` add the new "can't handle" signal. The
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
    # v36 B3 honesty floor: the ORIGINAL non_organic test above is
    # ELEMENT-SET-based, so a bare carbon-free ion built entirely from
    # `_ORGANIC_ELEMENTS` (nitrate O=[N+]([O-])[O-], sulfite, [S-2], [H+]) has
    # an EMPTY non_organic set and used to fall through to the "unknown
    # organic compound" sentinel below -- factually dishonest for a molecule
    # containing no carbon at all. Make the test STRUCTURAL as well: no atom
    # with atomic number 6 anywhere in the fragment means this is not an
    # organic compound, full stop, regardless of which specific elements it
    # is built from. A molecule that DOES contain carbon is completely
    # unaffected (falls through to the organic-unnameable branch exactly as
    # before -- never over-broadened).
    has_carbon = any(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms())
    if non_organic or not has_carbon:
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
